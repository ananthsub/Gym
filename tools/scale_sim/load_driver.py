# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-sim load driver.

Drives N concurrent ``/run`` requests through the **real** ``ServerClient`` /
aiohttp / ``RolloutCollectionHelper`` plumbing against gym sub-servers brought
up by ``ng_run``. Tracks per-rollout retry counts, error classes, and latency
percentiles, and dumps everything to ``results/<run_id>/``.

Run from ``tools/scale_sim/`` (after ``ng_run`` is up in another terminal)::

    python load_driver.py \
        --config configs/smoke.yaml \
        --input-jsonl data/smoke.jsonl \
        --output-dir results/<run_id>/

The driver mirrors the production ``RolloutCollectionHelper`` rollout loop:

- Inner retry on ``ServerDisconnectedError`` / ``ClientOSError`` lives inside
  ``nemo_gym.server_utils.request`` — the same layer production uses.
- Outer retry on ``ClientPayloadError`` lives here, matching the production
  per-rollout retry.

Per-rollout retry instrumentation is the piece the production stack does not
have; the driver records it so we can see retry storms as they build.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from aiohttp import ClientPayloadError
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_response_json,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)


# Make `instrumentation` importable when running this script directly from tools/scale_sim/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrumentation import KernelWatcher, ProcessMetricsSampler, RetryTracker  # noqa: E402


MAX_OUTER_RETRIES = 10  # cap on per-rollout outer retries before giving up


class LoadDriver:
    def __init__(
        self,
        config_path: Path,
        input_jsonl_path: Path,
        output_dir: Path,
        head_server_host: str,
        head_server_port: int,
        mode: str = "loaded",
        idle_window_s: float = 30.0,
    ) -> None:
        if mode not in ("loaded", "spinup_only"):
            raise ValueError(f"mode must be 'loaded' or 'spinup_only', got {mode!r}")
        self.mode = mode
        self.idle_window_s = idle_window_s

        self.config_path = config_path
        self.input_jsonl_path = input_jsonl_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg = OmegaConf.load(config_path)
        scale_sim_cfg = cfg.get("scale_sim")
        if scale_sim_cfg is None:
            raise ValueError(f"Config {config_path} has no top-level `scale_sim:` block.")
        self.scale_sim_cfg: DictConfig = scale_sim_cfg

        # agent_names is the multi-agent field; falls back to the existing
        # singleton agent_name field for single-agent configs (smoke, single_agent, etc.).
        # Per-row dispatch via agent_ref.name matches RolloutCollectionHelper exactly.
        agent_names_cfg = scale_sim_cfg.get("agent_names")
        if agent_names_cfg:
            self.agent_names: List[str] = list(agent_names_cfg)
        else:
            single = scale_sim_cfg.get("agent_name")
            if single is None:
                raise ValueError(
                    f"Config {config_path}: `scale_sim` must define either `agent_name` "
                    f"(single agent) or `agent_names: [list]` (multi-agent)."
                )
            self.agent_names = [str(single)]
        # Back-compat alias preserved for the per-cell summary.json field shape.
        self.agent_name: str = self.agent_names[0] if len(self.agent_names) == 1 else ",".join(self.agent_names)

        self.concurrency: int = int(scale_sim_cfg.concurrency)
        self.total_requests: int = int(scale_sim_cfg.total_requests)
        self.early_stop_failure_rate: float = float(scale_sim_cfg.get("early_stop_failure_rate", 0.10))
        self.early_stop_retry_rate: float = float(scale_sim_cfg.get("early_stop_retry_rate", 0.30))
        self.early_stop_wall_clock_s: float = float(scale_sim_cfg.get("early_stop_wall_clock_s", 600))
        self.early_stop_window_s: float = float(scale_sim_cfg.get("early_stop_window_s", 30))
        self.semaphore_enabled: bool = bool(scale_sim_cfg.get("semaphore_enabled", True))

        self.head_server_config = BaseServerConfig(host=head_server_host, port=head_server_port)

        # Output artifacts
        self.retry_jsonl_path = self.output_dir / "per_rollout_retries.jsonl"
        self.summary_json_path = self.output_dir / "summary.json"
        self.latencies_csv_path = self.output_dir / "latencies.csv"
        self.process_metrics_csv_path = self.output_dir / "process_metrics.csv"
        self.kernel_metrics_csv_path = self.output_dir / "kernel_metrics.csv"
        self.console_log_path = self.output_dir / "driver.log"

        self.tracker = RetryTracker(self.retry_jsonl_path)
        self._latencies: List[float] = []
        self._stop_requested = False
        self._stop_reason: Optional[str] = None
        self._run_wall_s: Optional[float] = None

    def _load_input_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with self.input_jsonl_path.open("rb") as f:
            for line in f:
                rows.append(orjson.loads(line))

        # Pad / cycle to total_requests.
        if not rows:
            raise ValueError(f"No rows in {self.input_jsonl_path}")
        if len(rows) < self.total_requests:
            print(f"Input has {len(rows)} rows, total_requests={self.total_requests}. Cycling input to fill.")
        cycled: List[Dict[str, Any]] = []
        n_agents = len(self.agent_names)
        for i in range(self.total_requests):
            row = dict(rows[i % len(rows)])
            row["task_index"] = i
            row["rollout_index"] = 0
            # Per-row dispatch via agent_ref.name — same as production
            # RolloutCollectionHelper._post_subroutine. Single-agent configs end up
            # with every row pointing at the same agent (back-compat). Multi-agent
            # configs round-robin so each agent gets total_requests / N rollouts.
            row["agent_ref"] = {"name": self.agent_names[i % n_agents]}
            cycled.append(row)
        return cycled

    async def run(self) -> None:
        # Load merged global config from the head server.
        server_client = ServerClient.load_from_global_config(self.head_server_config)
        if not is_global_aiohttp_client_setup():
            set_global_aiohttp_client(
                cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict)
            )

        if self.mode == "spinup_only":
            await self._run_spinup_only()
            return

        rows = self._load_input_rows()
        n_agents = len(self.agent_names)
        agent_label = (
            self.agent_names[0]
            if n_agents == 1
            else f"{n_agents} agents (round-robin: {self.agent_names[0]} … {self.agent_names[-1]})"
        )
        print(
            f"Driving {self.total_requests} requests at concurrency={self.concurrency} "
            f"against {agent_label} via head_server={self.head_server_config.host}:{self.head_server_config.port}"
        )

        # Instrumentation
        loop = asyncio.get_running_loop()
        sampler = ProcessMetricsSampler(self.process_metrics_csv_path, loop=loop)
        kernel_watcher = KernelWatcher(self.kernel_metrics_csv_path)
        sampler.start()
        kernel_watcher.start()

        # Periodic summary task — also drives early-stop.
        summary_task = asyncio.create_task(self._summary_and_early_stop_loop())

        semaphore = asyncio.Semaphore(self.concurrency) if self.semaphore_enabled else nullcontext()

        async def _post_subroutine(row: Dict[str, Any]) -> None:
            rollout_idx = row["task_index"]
            agent_name = row["agent_ref"]["name"]
            t0 = time.perf_counter()
            for attempt in range(1, MAX_OUTER_RETRIES + 1):
                if self._stop_requested:
                    self.tracker.record_completion(rollout_idx, succeeded=False, agent_name=agent_name)
                    return
                self.tracker.record_attempt(rollout_idx)
                try:
                    # Per-row dispatch via agent_ref.name — same as production.
                    res = await server_client.post(
                        server_name=agent_name,
                        url_path="/run",
                        json=row,
                    )
                    await raise_for_status(res)
                    await get_response_json(res)
                    self._latencies.append(time.perf_counter() - t0)
                    self.tracker.record_completion(rollout_idx, succeeded=True, agent_name=agent_name)
                    return
                except ClientPayloadError as e:
                    self.tracker.record_attempt(rollout_idx, error_class=type(e).__name__)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    self.tracker.record_attempt(rollout_idx, error_class=type(e).__name__)
                    if attempt >= MAX_OUTER_RETRIES:
                        break
                    await asyncio.sleep(0.5)
            self.tracker.record_completion(rollout_idx, succeeded=False, agent_name=agent_name)

        async def _wrapper(row: Dict[str, Any]) -> None:
            async with semaphore:  # type: ignore[union-attr]
                await _post_subroutine(row)

        run_start_s = time.perf_counter()
        try:
            futures = [asyncio.create_task(_wrapper(row)) for row in rows]
            for fut in tqdm(asyncio.as_completed(futures), total=len(futures), desc="rollouts"):
                await fut
                if self._stop_requested:
                    print(f"Early-stop triggered: {self._stop_reason}. Cancelling remaining requests.")
                    for f in futures:
                        f.cancel()
                    break
        finally:
            self._run_wall_s = time.perf_counter() - run_start_s
            summary_task.cancel()
            try:
                await summary_task
            except asyncio.CancelledError:
                pass
            sampler.stop()
            kernel_watcher.stop()
            self.tracker.flush()
            self.tracker.close()
            self._dump_final_summary()

    async def _run_spinup_only(self) -> None:
        """Sample idle resource cost of the running ng_run topology, no traffic.

        Used by the multi-agent pre-flight sweep to answer "what does N sub-servers
        existing cost on the head node" without driving any rollouts. The driver
        itself is the wrong process to sample sub-server RSS / FD counts from
        (it lives outside ng_run's process tree); KernelWatcher's host-wide
        /proc/net/sockstat + file-nr + loadavg samples are what carry the signal,
        plus the ProcessMetricsSampler on the driver process records its own
        idle cost as a control. Driver runs the sampler for ``idle_window_s``,
        then writes a spinup-shaped summary.
        """
        loop = asyncio.get_running_loop()
        sampler = ProcessMetricsSampler(self.process_metrics_csv_path, loop=loop)
        kernel_watcher = KernelWatcher(self.kernel_metrics_csv_path)
        sampler.start()
        kernel_watcher.start()
        print(
            f"[scale_sim] spinup_only: sampling idle resource state for {self.idle_window_s:.0f}s "
            f"(n_agents={len(self.agent_names)})",
            flush=True,
        )
        try:
            await asyncio.sleep(self.idle_window_s)
        finally:
            sampler.stop()
            kernel_watcher.stop()
            self.tracker.close()
            self._dump_spinup_summary()

    def _dump_spinup_summary(self) -> None:
        # Best-effort kernel-metric reading: take the last sampled row from
        # kernel_metrics.csv as the steady-state idle snapshot.
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
        # Same idea on the driver-process metrics — last sample is steady-state.
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
            "agent_names": list(self.agent_names),
            "n_agents": len(self.agent_names),
            "idle_window_s": self.idle_window_s,
            "idle_kernel": idle,
            "idle_driver_process": proc,
        }
        self.summary_json_path.write_text(json.dumps(summary, indent=2))
        print("\n=== scale_sim spinup_only summary ===")
        print(json.dumps(summary, indent=2))

    async def _summary_and_early_stop_loop(self) -> None:
        deadline = time.time() + self.early_stop_wall_clock_s
        while True:
            await asyncio.sleep(min(10.0, self.early_stop_window_s / 3))
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
                f"[scale_sim] window({self.early_stop_window_s}s): "
                f"completed={window_summary['n_rollouts']} "
                f"failure_rate={window_summary['failure_rate']:.2%} "
                f"retry_rate={window_summary['retry_rate']:.2%} "
                f"p_at_least_1_retry={window_summary['p_at_least_1_retry']:.2%}"
            )

    def _dump_final_summary(self) -> None:
        all_summary = self.tracker.summary_all()
        # Throughput: completed rollouts per second of wall clock. This is the
        # capacity number — latency percentiles describe one rollout's wait,
        # throughput describes how many the node clears per second.
        n_completed = all_summary.get("n_rollouts", 0)
        wall_s = self._run_wall_s
        throughput = (n_completed / wall_s) if (wall_s and wall_s > 0) else None
        # Latency percentiles
        latency_summary: Dict[str, Any]
        if self._latencies:
            sorted_lat = sorted(self._latencies)
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
            "input_jsonl_path": str(self.input_jsonl_path),
            "agent_name": self.agent_name,
            "agent_names": list(self.agent_names),
            "n_agents": len(self.agent_names),
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "semaphore_enabled": self.semaphore_enabled,
            "stop_reason": self._stop_reason,
            "wall_clock_s": wall_s,
            "throughput_rollouts_per_s": throughput,
            # completion_rate = fraction of attempted rollouts that finished within
            # the window. At saturation this drops far below 1 while failure_rate
            # stays near 0 — i.e. the cell is throughput-limited, not failing.
            "completion_rate": (n_completed / self.total_requests) if self.total_requests else None,
            # saturated = the cell hit the wall-clock cap rather than draining all
            # requests. Latency percentiles below are over completed rollouts only,
            # so they understate latency when saturated is true.
            "saturated": bool(self._stop_reason and "wall_clock" in str(self._stop_reason)),
            "retry_summary": all_summary,
            "latency_summary": latency_summary,
        }
        # Per-agent breakdown is empty for single-agent runs; non-empty for multi-agent.
        per_agent = self.tracker.summary_by_agent()
        if per_agent:
            summary["per_agent"] = per_agent
        self.summary_json_path.write_text(json.dumps(summary, indent=2))

        # Latency CSV for offline plotting
        with self.latencies_csv_path.open("w") as f:
            f.write("latency_s\n")
            for lat in self._latencies:
                f.write(f"{lat:.6f}\n")

        print("\n=== scale_sim summary ===")
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a sweep config YAML.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=False,
        default=None,
        help="JSONL of input rows. Required for --mode=loaded; ignored for --mode=spinup_only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write per_rollout_retries.jsonl, summary.json, etc. Defaults to results/<timestamp>/.",
    )
    parser.add_argument(
        "--head-server-host",
        default="0.0.0.0",
        help="Match the `head_server.host` in the YAML.",
    )
    parser.add_argument(
        "--head-server-port",
        type=int,
        default=5000,
        help="Match the `head_server.port` in the YAML. Default 5000 stays clear of ephemeral-port floors.",
    )
    parser.add_argument(
        "--mode",
        choices=("loaded", "spinup_only"),
        default="loaded",
        help=(
            "loaded (default): drive concurrent /run requests through the gym stack. "
            "spinup_only: skip dispatch, sample idle resource cost for --idle-window-s seconds (multi-agent pre-flight)."
        ),
    )
    parser.add_argument(
        "--idle-window-s",
        type=float,
        default=30.0,
        help="With --mode=spinup_only, how long to sample after sub-servers are ready.",
    )
    args = parser.parse_args()

    if args.mode == "loaded" and args.input_jsonl is None:
        parser.error("--input-jsonl is required for --mode=loaded.")
    if args.mode == "spinup_only" and args.input_jsonl is None:
        # spinup_only doesn't read inputs, but the LoadDriver's __init__ keeps
        # a path for the summary. Use the config path itself as a placeholder so
        # the field is non-None.
        args.input_jsonl = args.config

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("results") / args.config.stem / ts

    driver = LoadDriver(
        config_path=args.config,
        input_jsonl_path=args.input_jsonl,
        output_dir=args.output_dir,
        head_server_host=args.head_server_host,
        head_server_port=args.head_server_port,
        mode=args.mode,
        idle_window_s=args.idle_window_s,
    )
    asyncio.run(driver.run())


if __name__ == "__main__":
    main()
