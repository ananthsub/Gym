# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Threaded variant of ``load_driver.py``.

Same CLI contract and same ``summary.json`` output shape as ``load_driver.py``,
but drives load via ``concurrent.futures.ThreadPoolExecutor`` + synchronous
``requests`` calls instead of one asyncio event loop with N coroutines.

This intentionally mirrors the production driver pattern in
``RL/nemo_rl/algorithms/async_utils.py::AsyncTrajectoryCollector`` — N worker
threads each making a synchronous rollout call. We use it to test whether the
thread-per-rollout pattern surfaces connection-lifecycle errors
(``ClientPayloadError`` / ``ConnectionResetError`` / similar) that the
asyncio driver in ``load_driver.py`` does NOT surface, even though the matrix
shows zero failures end-to-end on the asyncio side.

Key shape differences vs ``load_driver.py``:

- **Per-thread ``requests.Session``** by default (``--session-mode per_thread``).
  Each worker thread gets its own urllib3 connection pool. Maximizes per-thread
  isolation; closer to "K independent clients hammering one server" than to
  "one client multiplexing K coroutines on a shared pool".
- **No ``aiohttp``.** Synchronous ``requests`` POST per rollout. No event loop
  involved on the driver side.
- **No ``loop_lag_p99_ms``** in process_metrics.csv. There's no event loop to
  measure. ``ProcessMetricsSampler`` is started with ``loop=None`` and that
  column is recorded as 0.
- **Retry semantics mirror the asyncio version**: outer ``MAX_OUTER_RETRIES``
  loop on ``requests.exceptions.ChunkedEncodingError`` (the threaded analog
  of ``aiohttp.ClientPayloadError``), plus on any other exception up to the
  cap. Same ``RetryTracker`` instrumentation.

Run the same way as ``load_driver.py``::

    python thread_load_driver.py \
        --config configs/single_agent_base.yaml \
        --input-jsonl data/sweep_20k.jsonl \
        --output-dir results/<run_id>/

The cell summary is written to ``summary.json`` with the same field names so
existing analysis tooling works without modification. The summary additionally
records ``driver_kind: threaded`` and ``session_mode`` so downstream readers
can tell which driver produced the cell.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import requests
from omegaconf import DictConfig, OmegaConf
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.server_utils import ServerClient

# Make `instrumentation` importable when running this script directly from tools/scale_sim/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrumentation import KernelWatcher, ProcessMetricsSampler, RetryTracker  # noqa: E402


MAX_OUTER_RETRIES = 10  # mirror the cap from load_driver.py
DEFAULT_REQUEST_TIMEOUT_S = 600.0


class ThreadLoadDriver:
    """Threaded driver. See module docstring for design notes."""

    def __init__(
        self,
        config_path: Path,
        input_jsonl_path: Optional[Path],
        output_dir: Path,
        head_server_host: str,
        head_server_port: int,
        mode: str = "loaded",
        idle_window_s: float = 30.0,
        session_mode: str = "per_thread",
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        rollouts_per_thread: int = 1,
    ) -> None:
        if mode not in ("loaded", "spinup_only"):
            raise ValueError(f"mode must be 'loaded' or 'spinup_only', got {mode!r}")
        if session_mode not in ("per_thread", "shared"):
            raise ValueError(
                f"session_mode must be 'per_thread' or 'shared', got {session_mode!r}"
            )
        if rollouts_per_thread < 1:
            raise ValueError(
                f"rollouts_per_thread must be >= 1, got {rollouts_per_thread}"
            )
        self.mode = mode
        self.session_mode = session_mode
        self.idle_window_s = idle_window_s
        self.request_timeout_s = request_timeout_s
        # Production's AsyncTrajectoryCollector spawns one thread per
        # *prompt group* (typically 256 threads for 256 prompts × 16
        # generations = c=4 096). Each thread handles 16 rollouts. The
        # default rollouts_per_thread=1 is the original §3.1.D shape (one
        # thread per rollout); rollouts_per_thread > 1 batches rollouts
        # onto fewer threads to match production's per-thread connection
        # pool footprint.
        self.rollouts_per_thread = rollouts_per_thread

        self.config_path = config_path
        self.input_jsonl_path = input_jsonl_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg = OmegaConf.load(config_path)
        scale_sim_cfg = cfg.get("scale_sim")
        if scale_sim_cfg is None:
            raise ValueError(f"Config {config_path} has no top-level `scale_sim:` block.")
        self.scale_sim_cfg: DictConfig = scale_sim_cfg

        agent_names_cfg = scale_sim_cfg.get("agent_names")
        if agent_names_cfg:
            self.agent_names: List[str] = list(agent_names_cfg)
        else:
            single = scale_sim_cfg.get("agent_name")
            if single is None:
                raise ValueError(
                    f"Config {config_path}: `scale_sim` must define either `agent_name` "
                    f"or `agent_names: [list]`."
                )
            self.agent_names = [str(single)]
        self.agent_name: str = (
            self.agent_names[0]
            if len(self.agent_names) == 1
            else ",".join(self.agent_names)
        )

        self.concurrency: int = int(scale_sim_cfg.concurrency)
        self.total_requests: int = int(scale_sim_cfg.total_requests)
        self.early_stop_failure_rate: float = float(scale_sim_cfg.get("early_stop_failure_rate", 0.10))
        self.early_stop_retry_rate: float = float(scale_sim_cfg.get("early_stop_retry_rate", 0.30))
        self.early_stop_wall_clock_s: float = float(scale_sim_cfg.get("early_stop_wall_clock_s", 600))
        self.early_stop_window_s: float = float(scale_sim_cfg.get("early_stop_window_s", 30))
        # semaphore_enabled is read from config for parity with load_driver.py
        # but, for the threaded driver, the ThreadPoolExecutor's max_workers
        # already imposes a hard concurrency cap. We surface the value in
        # summary.json so cells from both drivers are comparable.
        self.semaphore_enabled: bool = bool(scale_sim_cfg.get("semaphore_enabled", True))

        self.head_server_config = BaseServerConfig(host=head_server_host, port=head_server_port)

        # Output artifacts (same shape as load_driver.py)
        self.retry_jsonl_path = self.output_dir / "per_rollout_retries.jsonl"
        self.summary_json_path = self.output_dir / "summary.json"
        self.latencies_csv_path = self.output_dir / "latencies.csv"
        self.process_metrics_csv_path = self.output_dir / "process_metrics.csv"
        self.kernel_metrics_csv_path = self.output_dir / "kernel_metrics.csv"
        self.console_log_path = self.output_dir / "driver.log"

        self.tracker = RetryTracker(self.retry_jsonl_path)
        self._latencies: List[float] = []
        self._latencies_lock = threading.Lock()
        self._stop_requested = False
        self._stop_reason: Optional[str] = None

        # Thread-local storage for per-thread sessions.
        self._tls = threading.local()
        # Shared session (only used when session_mode == "shared").
        self._shared_session: Optional[requests.Session] = None

        # Resolved direct URLs per agent (filled in run()).
        self._agent_urls: Dict[str, str] = {}

    # ---------------------------------------------------------------------
    # Input handling
    # ---------------------------------------------------------------------

    def _load_input_rows(self) -> List[Dict[str, Any]]:
        if self.input_jsonl_path is None:
            raise ValueError("Loaded mode requires --input-jsonl.")
        rows: List[Dict[str, Any]] = []
        with self.input_jsonl_path.open("rb") as f:
            for line in f:
                rows.append(orjson.loads(line))
        if not rows:
            raise ValueError(f"No rows in {self.input_jsonl_path}")
        if len(rows) < self.total_requests:
            print(
                f"Input has {len(rows)} rows, total_requests={self.total_requests}. Cycling input to fill."
            )
        cycled: List[Dict[str, Any]] = []
        n_agents = len(self.agent_names)
        for i in range(self.total_requests):
            row = dict(rows[i % len(rows)])
            row["task_index"] = i
            row["rollout_index"] = 0
            row["agent_ref"] = {"name": self.agent_names[i % n_agents]}
            cycled.append(row)
        return cycled

    # ---------------------------------------------------------------------
    # URL resolution + session management
    # ---------------------------------------------------------------------

    def _resolve_agent_urls(self, global_config_dict: DictConfig) -> Dict[str, str]:
        """Resolve each agent's direct ``http://host:port`` URL from the global
        config, mirroring how ``ServerClient.request`` does it. Bypasses the
        head server's forwarding the same way ``load_driver.py`` does.
        """
        urls: Dict[str, str] = {}
        for name in self.agent_names:
            cfg = get_first_server_config_dict(global_config_dict, name)
            urls[name] = f"http://{cfg.host}:{cfg.port}"
        return urls

    def _make_session(self) -> requests.Session:
        """Create a requests.Session sized for this driver's concurrency.

        Uses urllib3 ``HTTPAdapter`` with a per-host pool sized to one
        connection (per-thread mode) or to a fraction of total concurrency
        (shared mode) so we don't bottleneck on the urllib3 pool.
        """
        session = requests.Session()
        if self.session_mode == "per_thread":
            # One thread = one in-flight call, so a 1-slot pool per thread
            # is enough. We give it 2 to absorb keepalive jitter.
            pool_size = 2
        else:
            # Shared session: pool size large enough to admit all in-flight
            # requests without urllib3-side queueing.
            pool_size = max(2, self.concurrency * 2)
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0,  # we own retry semantics; let urllib3 surface errors
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_session(self) -> requests.Session:
        if self.session_mode == "shared":
            assert self._shared_session is not None
            return self._shared_session
        session = getattr(self._tls, "session", None)
        if session is None:
            session = self._make_session()
            self._tls.session = session
        return session

    # ---------------------------------------------------------------------
    # Per-rollout worker
    # ---------------------------------------------------------------------

    def _post_one(self, row: Dict[str, Any]) -> None:
        rollout_idx = row["task_index"]
        agent_name = row["agent_ref"]["name"]
        url = f"{self._agent_urls[agent_name]}/run"
        session = self._get_session()
        t0 = time.perf_counter()
        try:
            for attempt in range(1, MAX_OUTER_RETRIES + 1):
                if self._stop_requested:
                    self.tracker.record_cancellation(rollout_idx, agent_name=agent_name)
                    return
                self.tracker.record_attempt(rollout_idx)
                try:
                    resp = session.post(url, json=row, timeout=self.request_timeout_s)
                    resp.raise_for_status()
                    # Parse body so we exercise the full client-side cost path,
                    # matching what get_response_json does in load_driver.py.
                    _ = orjson.loads(resp.content)
                    with self._latencies_lock:
                        self._latencies.append(time.perf_counter() - t0)
                    self.tracker.record_completion(rollout_idx, succeeded=True, agent_name=agent_name)
                    return
                except requests.exceptions.ChunkedEncodingError as e:
                    # Threaded analog of aiohttp.ClientPayloadError: the
                    # connection went away mid-response. Always retry,
                    # consistent with load_driver.py's outer ClientPayloadError
                    # branch.
                    self.tracker.record_attempt(rollout_idx, error_class=type(e).__name__)
                    time.sleep(0.5)
                except Exception as e:
                    self.tracker.record_attempt(rollout_idx, error_class=type(e).__name__)
                    if attempt >= MAX_OUTER_RETRIES:
                        break
                    time.sleep(0.5)
            self.tracker.record_completion(rollout_idx, succeeded=False, agent_name=agent_name)
        except BaseException:
            # Defensive: if anything unexpected escapes, mark as cancelled
            # rather than letting the worker thread silently drop the rollout.
            self.tracker.record_cancellation(rollout_idx, agent_name=agent_name)
            raise

    def _post_chunk(self, rows: List[Dict[str, Any]]) -> None:
        """Process a chunk of rollouts sequentially on the same worker thread.

        Used when ``rollouts_per_thread > 1``. The thread's
        ``requests.Session`` (per-thread mode) or the shared one (shared
        mode) is reused across all rollouts in the chunk — so connection
        pool count is bounded by `n_threads`, not by `total_requests`. This
        matches production's prompt-group thread shape: one thread per
        prompt group, ``num_generations_per_prompt`` rollouts handled
        serially per thread.
        """
        for row in rows:
            if self._stop_requested:
                self.tracker.record_cancellation(
                    row["task_index"], agent_name=row["agent_ref"]["name"]
                )
                continue
            try:
                self._post_one(row)
            except Exception:
                # _post_one already records its own retries / failures;
                # we just don't want one bad row to abort the whole chunk
                # (and starve the kept-alive connection of pending work).
                pass

    # ---------------------------------------------------------------------
    # Top-level orchestration
    # ---------------------------------------------------------------------

    def run(self) -> None:
        # Bootstrap global config (sync GET) — this is what ServerClient does
        # under the asyncio driver too, so the bootstrap path is identical.
        client = ServerClient.load_from_global_config(self.head_server_config)
        global_config_dict = client.global_config_dict

        if self.mode == "spinup_only":
            self._run_spinup_only()
            return

        rows = self._load_input_rows()
        self._agent_urls = self._resolve_agent_urls(global_config_dict)
        if self.session_mode == "shared":
            self._shared_session = self._make_session()

        n_agents = len(self.agent_names)
        agent_label = (
            self.agent_names[0]
            if n_agents == 1
            else f"{n_agents} agents (round-robin: {self.agent_names[0]} … {self.agent_names[-1]})"
        )
        print(
            f"[thread_driver] driving {self.total_requests} requests at concurrency={self.concurrency} "
            f"against {agent_label} via head_server={self.head_server_config.host}:{self.head_server_config.port}",
            flush=True,
        )
        # Compute the actual thread shape. With rollouts_per_thread=1
        # (default) we keep the original §3.1.D shape: one thread per
        # rollout, max_workers=concurrency. With rollouts_per_thread>1 we
        # batch rollouts onto fewer threads to match production's
        # AsyncTrajectoryCollector layout, where each prompt-group thread
        # handles `num_generations_per_prompt` rollouts sequentially. Number
        # of threads is concurrency / rollouts_per_thread (rounded up); each
        # thread pulls a slice of `rollouts_per_thread` rows and runs them
        # one at a time on its own connection pool.
        if self.rollouts_per_thread > 1:
            n_threads = max(
                1, (self.concurrency + self.rollouts_per_thread - 1) // self.rollouts_per_thread
            )
            print(
                f"[thread_driver] rollouts_per_thread={self.rollouts_per_thread}  "
                f"n_threads={n_threads} (= concurrency / rollouts_per_thread, ceil)",
                flush=True,
            )
        else:
            n_threads = self.concurrency
        print(
            f"[thread_driver] session_mode={self.session_mode}  request_timeout_s={self.request_timeout_s}  "
            f"max_workers={n_threads}",
            flush=True,
        )

        # Instrumentation. ProcessMetricsSampler accepts loop=None — there's
        # no event loop to measure here, so loop_lag column will be 0.
        sampler = ProcessMetricsSampler(self.process_metrics_csv_path, loop=None)
        kernel_watcher = KernelWatcher(self.kernel_metrics_csv_path)
        sampler.start()
        kernel_watcher.start()

        # Background summary thread (drives early-stop). Uses same thresholds
        # as load_driver.py's _summary_and_early_stop_loop.
        summary_stop = threading.Event()
        summary_thread = threading.Thread(
            target=self._summary_and_early_stop_loop,
            args=(summary_stop,),
            daemon=True,
            name="summary",
        )
        summary_thread.start()

        executor = ThreadPoolExecutor(
            max_workers=n_threads,
            thread_name_prefix="rollout",
        )
        futures: List[Future[None]] = []
        try:
            # Submission loop. We submit eagerly (executor's queue absorbs
            # excess) so this thread doesn't gate dispatch — same shape as
            # load_driver.py's `asyncio.create_task(...) for row in rows`.
            if self.rollouts_per_thread > 1:
                # Batch rows into chunks of `rollouts_per_thread`; submit one
                # task per chunk. Each task handles its rows sequentially on
                # the same worker thread → same connection pool, no pool
                # multiplication. This matches production's per-prompt-group
                # thread shape.
                for start in range(0, len(rows), self.rollouts_per_thread):
                    if self._stop_requested:
                        print(
                            f"[thread_driver] early-stop fired during submission: "
                            f"{self._stop_reason}. Stopping submission.",
                            flush=True,
                        )
                        break
                    chunk = rows[start : start + self.rollouts_per_thread]
                    futures.append(executor.submit(self._post_chunk, chunk))
            else:
                for row in rows:
                    if self._stop_requested:
                        print(
                            f"[thread_driver] early-stop fired during submission: "
                            f"{self._stop_reason}. Stopping submission.",
                            flush=True,
                        )
                        break
                    futures.append(executor.submit(self._post_one, row))

            # Drain. tqdm shows progress; we propagate exceptions defensively
            # but the worker already records them as failed/cancelled rollouts.
            for fut in tqdm(_as_done(futures), total=len(futures), desc="rollouts"):
                if self._stop_requested:
                    # Tell workers to bail out. Their cancellation-record path
                    # marks any outstanding rollouts as cancelled.
                    print(
                        f"[thread_driver] early-stop fired: {self._stop_reason}. "
                        f"Cancelling outstanding rollouts.",
                        flush=True,
                    )
                    break
                try:
                    fut.result()
                except Exception:
                    # Worker already recorded the error; we just don't want
                    # the executor's exception to bubble up here.
                    pass
        finally:
            self._stop_requested = True  # signal in-flight workers to short-circuit
            summary_stop.set()
            # cancel_futures=True drops any tasks still queued without running
            # them. In-flight tasks see _stop_requested and short-circuit.
            executor.shutdown(wait=True, cancel_futures=True)
            summary_thread.join(timeout=5.0)
            sampler.stop()
            kernel_watcher.stop()
            self.tracker.flush()
            self.tracker.close()
            self._dump_final_summary()

    # ---------------------------------------------------------------------
    # Spinup-only mode (parity with load_driver.py)
    # ---------------------------------------------------------------------

    def _run_spinup_only(self) -> None:
        sampler = ProcessMetricsSampler(self.process_metrics_csv_path, loop=None)
        kernel_watcher = KernelWatcher(self.kernel_metrics_csv_path)
        sampler.start()
        kernel_watcher.start()
        print(
            f"[thread_driver] spinup_only: sampling idle resource state for {self.idle_window_s:.0f}s "
            f"(n_agents={len(self.agent_names)})",
            flush=True,
        )
        try:
            time.sleep(self.idle_window_s)
        finally:
            sampler.stop()
            kernel_watcher.stop()
            self.tracker.close()
            self._dump_spinup_summary()

    # ---------------------------------------------------------------------
    # Summary + early-stop
    # ---------------------------------------------------------------------

    def _summary_and_early_stop_loop(self, stop_event: threading.Event) -> None:
        """Synchronous variant of load_driver.py::_summary_and_early_stop_loop."""
        deadline = time.time() + self.early_stop_wall_clock_s
        interval_s = min(10.0, self.early_stop_window_s / 3)
        while not stop_event.is_set():
            stop_event.wait(interval_s)
            if stop_event.is_set():
                return
            if time.time() >= deadline:
                self._stop_requested = True
                self._stop_reason = f"wall_clock>{self.early_stop_wall_clock_s}s"
                return
            window_summary = self.tracker.summary_window(self.early_stop_window_s)
            if window_summary["n_rollouts"] >= 50:
                if window_summary["failure_rate"] > self.early_stop_failure_rate:
                    self._stop_requested = True
                    self._stop_reason = (
                        f"failure_rate={window_summary['failure_rate']:.2%} "
                        f">{self.early_stop_failure_rate:.0%} (window={self.early_stop_window_s}s)"
                    )
                    return
                if window_summary["retry_rate"] > self.early_stop_retry_rate:
                    self._stop_requested = True
                    self._stop_reason = (
                        f"retry_rate={window_summary['retry_rate']:.2%} "
                        f">{self.early_stop_retry_rate:.0%} (window={self.early_stop_window_s}s)"
                    )
                    return
            print(
                f"[thread_driver] window({self.early_stop_window_s}s): "
                f"completed={window_summary['n_rollouts']} "
                f"failure_rate={window_summary['failure_rate']:.2%} "
                f"retry_rate={window_summary['retry_rate']:.2%} "
                f"p_at_least_1_retry={window_summary['p_at_least_1_retry']:.2%}",
                flush=True,
            )

    def _dump_spinup_summary(self) -> None:
        idle: Dict[str, Any] = {}
        if self.kernel_metrics_csv_path.exists():
            try:
                lines = self.kernel_metrics_csv_path.read_text().splitlines()
                if len(lines) >= 2:
                    header = lines[0].split(",")
                    last = lines[-1].split(",")
                    idle = dict(zip(header, last))
            except Exception:
                idle = {}
        proc: Dict[str, Any] = {}
        if self.process_metrics_csv_path.exists():
            try:
                lines = self.process_metrics_csv_path.read_text().splitlines()
                if len(lines) >= 2:
                    header = lines[0].split(",")
                    last = lines[-1].split(",")
                    proc = dict(zip(header, last))
            except Exception:
                proc = {}

        summary = {
            "config_path": str(self.config_path),
            "mode": "spinup_only",
            "driver_kind": "threaded",
            "session_mode": self.session_mode,
            "agent_names": list(self.agent_names),
            "n_agents": len(self.agent_names),
            "idle_window_s": self.idle_window_s,
            "idle_kernel": idle,
            "idle_driver_process": proc,
        }
        self.summary_json_path.write_text(json.dumps(summary, indent=2))
        print("\n=== thread_driver spinup_only summary ===")
        print(json.dumps(summary, indent=2))

    def _dump_final_summary(self) -> None:
        all_summary = self.tracker.summary_all()
        latency_summary: Dict[str, Any]
        with self._latencies_lock:
            latencies = list(self._latencies)
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)

            def pct(p: float) -> float:
                return sorted_lat[min(n - 1, int(n * p))]

            latency_summary = {
                "n_completed_with_latency": n,
                "p50_s": pct(0.50),
                "p90_s": pct(0.90),
                "p99_s": pct(0.99),
                "p999_s": pct(0.999),
                "max_s": sorted_lat[-1],
            }
        else:
            latency_summary = {"n_completed_with_latency": 0}

        summary = {
            "config_path": str(self.config_path),
            "input_jsonl_path": str(self.input_jsonl_path) if self.input_jsonl_path else None,
            "agent_name": self.agent_name,
            "agent_names": list(self.agent_names),
            "n_agents": len(self.agent_names),
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "semaphore_enabled": self.semaphore_enabled,
            "stop_reason": self._stop_reason,
            "driver_kind": "threaded",
            "session_mode": self.session_mode,
            "request_timeout_s": self.request_timeout_s,
            "rollouts_per_thread": self.rollouts_per_thread,
            "retry_summary": all_summary,
            "latency_summary": latency_summary,
        }
        per_agent = self.tracker.summary_by_agent()
        if per_agent:
            summary["per_agent"] = per_agent
        self.summary_json_path.write_text(json.dumps(summary, indent=2))

        with self.latencies_csv_path.open("w") as f:
            f.write("latency_s\n")
            for lat in latencies:
                f.write(f"{lat:.6f}\n")

        print("\n=== thread_driver summary ===")
        print(json.dumps(summary, indent=2))


def _as_done(futures: List[Future[None]]):
    """Yield completed futures preserving submission order so tqdm progresses
    monotonically. ``concurrent.futures.as_completed`` is event-driven and
    fine but yields out of order; preserving order makes the progress bar
    tick predictably and matches ``asyncio.as_completed``'s behavior more
    intuitively for a sweep operator watching the log.
    """
    for fut in futures:
        try:
            fut.result()
        except Exception:
            # Worker handles its own logging; we just propagate the future
            # so the caller's loop sees it.
            pass
        yield fut


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threaded variant of load_driver.py — fires K threads instead of K coroutines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Required for --mode loaded; ignored for spinup_only.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-server-host", default="0.0.0.0")
    parser.add_argument("--head-server-port", type=int, default=5000)
    parser.add_argument("--mode", choices=("loaded", "spinup_only"), default="loaded")
    parser.add_argument(
        "--idle-window-s",
        type=float,
        default=30.0,
        help="Window for spinup_only mode.",
    )
    parser.add_argument(
        "--session-mode",
        choices=("per_thread", "shared"),
        default="per_thread",
        help=(
            "per_thread: each worker thread owns its own requests.Session "
            "(separate urllib3 connection pool — most thread-isolated, closest "
            "to production AsyncTrajectoryCollector pattern). "
            "shared: one Session reused by all threads (closer to aiohttp's "
            "shared-pool behavior; useful for A/B against the asyncio driver)."
        ),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Per-request timeout in seconds. Set high enough that wall-clock "
        "budget bites first.",
    )
    parser.add_argument(
        "--rollouts-per-thread",
        type=int,
        default=1,
        help=(
            "Default 1 = one thread per rollout (§3.1.D shape: connection-pool "
            "count grows linearly with concurrency). >1 = batch rollouts onto "
            "fewer threads sequentially, matching production's "
            "AsyncTrajectoryCollector pattern (one thread per prompt group, "
            "num_generations_per_prompt rollouts per thread). "
            "n_threads = ceil(concurrency / rollouts_per_thread)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "loaded" and args.input_jsonl is None:
        parser.error("--input-jsonl is required for --mode loaded")

    driver = ThreadLoadDriver(
        config_path=args.config,
        input_jsonl_path=args.input_jsonl,
        output_dir=args.output_dir,
        head_server_host=args.head_server_host,
        head_server_port=args.head_server_port,
        mode=args.mode,
        idle_window_s=args.idle_window_s,
        session_mode=args.session_mode,
        request_timeout_s=args.request_timeout_s,
        rollouts_per_thread=args.rollouts_per_thread,
    )
    driver.run()


if __name__ == "__main__":
    main()
