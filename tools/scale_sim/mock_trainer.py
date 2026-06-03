# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mock trainer driver — drives trainer-side load on a gym Ray actor.

The direct load driver (``load_driver.py``) talks HTTP straight to the head
server and skips the Ray actor. This driver adds the actor back: it talks to a
``NemoGymActor`` with blocking ``ray.get`` calls, the same way the training
framework does. Use it to study failures that depend on the trainer/actor
boundary rather than on raw HTTP concurrency.

Two return shapes:

- ``sync_blocking``: ``thread_count`` threads each issue a blocking
  ``ray.get(actor.collect_batch.remote(rows))`` — one call per prompt group.
  Each call returns the whole batch as one object through Ray's object store.
  This is the shape under which the production ``ClientPayloadError`` was seen
  at high thread counts.

- ``lag_batched_stream``: the streaming-return shape. ``thread_count`` is sized
  to the lag the trainer tolerates (a handful) rather than the prompt-group
  count, and the actor returns results through an ``ObjectRefGenerator`` — one
  ``ray.get`` per row instead of one large return per batch. Each thread runs
  ``gen = actor.collect_batch_streaming.remote(rows)`` then
  ``for ref in gen: ray.get(ref)``.

Knobs:
    --mode {sync_blocking, lag_batched_stream}   Which return shape to run.
    --thread-count N        Number of trainer threads. For sync_blocking this
                            is the prompt-group count; for lag_batched_stream
                            it is the lag bound. Total in-flight rollouts =
                            thread_count * prompts_per_call.
    --prompts-per-call P    Rows per actor.collect_batch{,_streaming}.remote() call.
    --num-steps S           Total trainer steps; each step issues thread_count
                            concurrent calls.
    --config                YAML config the actor loads.
    --input-jsonl           Source of rollout rows.
    --output-dir            Where summary.json / metrics CSVs / per-call JSONL go.

Output artifacts (same conventions as load_driver.py):
    summary.json            Top-line metrics + per-error-class counts.
    per_call.jsonl          One line per actor call (per-row bodies dropped;
                            only response_bytes + status kept).
    process_metrics.csv     Driver-process RSS / FDs / CPU%.
    kernel_metrics.csv      Host tcp_inuse / file_nr / loadavg.
    trainer.log             Console output.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import ray
from omegaconf import OmegaConf


# Make `instrumentation` + `gym_actor` importable when run directly from tools/scale_sim/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_actor import NemoGymActor  # noqa: E402
from instrumentation import KernelWatcher, ProcessMetricsSampler  # noqa: E402


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class _MetricsSink:
    """Thread-safe collector for per-call metrics, written to per_call.jsonl."""

    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", buffering=1)
        self._lock = threading.Lock()
        self.per_row_latencies: List[float] = []
        self.per_call_latencies: List[float] = []
        self.error_class_counts: Dict[str, int] = {}
        self.n_calls = 0
        self.n_rows_succeeded = 0
        self.n_rows_failed = 0

    def record_call(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(orjson.dumps(record).decode() + "\n")
            self.n_calls += 1
            self.per_call_latencies.append(record["call_latency_s"])
            for row in record["per_row"]:
                if "latency_s" in row:
                    self.per_row_latencies.append(row["latency_s"])
                if row.get("status") == "ok":
                    self.n_rows_succeeded += 1
                else:
                    self.n_rows_failed += 1
                    cls = row.get("error_class") or "Unknown"
                    self.error_class_counts[cls] = self.error_class_counts.get(cls, 0) + 1
            # If the streaming generator died mid-batch, the trainer thread sees fewer
            # rows than it asked for. Count the missing ones as call-level failures
            # so the failure rate reflects what the trainer actually observed.
            n_requested = record.get("n_rows_in_call", len(record["per_row"]))
            n_returned = record.get("n_rows_returned", len(record["per_row"]))
            if n_requested > n_returned:
                missing = n_requested - n_returned
                self.n_rows_failed += missing
                cls = "MissingFromStream"
                self.error_class_counts[cls] = self.error_class_counts.get(cls, 0) + missing
            # Also record any call-level error (e.g. ray.exceptions.RayTaskError).
            if record.get("call_error_class"):
                cls = record["call_error_class"]
                self.error_class_counts[cls] = self.error_class_counts.get(cls, 0) + 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _load_rows(path: Path, total: int, agent_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            rows.append(orjson.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    cycled: List[Dict[str, Any]] = []
    for i in range(total):
        row = dict(rows[i % len(rows)])
        row["task_index"] = i
        row["rollout_index"] = 0
        row["agent_ref"] = {"name": agent_name}
        cycled.append(row)
    return cycled


def _worker_thread_sync_blocking(
    *,
    thread_id: int,
    work_q: "queue.Queue[Optional[Dict[str, Any]]]",
    actor: Any,
    sink: _MetricsSink,
    stop_event: threading.Event,
) -> None:
    """sync_blocking: one blocking ``ray.get`` per work item.

    Pulls work items off ``work_q``. Each item is a dict with ``step_idx`` +
    ``rows`` (the batch). For each item, issues
    ``ray.get(actor.collect_batch.remote(rows))`` and records the outcome.
    """
    while not stop_event.is_set():
        item = work_q.get()
        try:
            if item is None:
                return
            t0 = time.perf_counter()
            call_error_class: Optional[str] = None
            call_error_message: Optional[str] = None
            per_row: List[Dict[str, Any]] = []
            try:
                fut = actor.collect_batch.remote(item["rows"])
                per_row = ray.get(fut)
            except Exception as e:
                call_error_class = type(e).__name__
                call_error_message = str(e)[:500]
            call_latency_s = time.perf_counter() - t0
            sink.record_call(
                {
                    "step_idx": item["step_idx"],
                    "thread_id": thread_id,
                    "n_rows_in_call": len(item["rows"]),
                    "call_latency_s": call_latency_s,
                    "call_error_class": call_error_class,
                    "call_error_message": call_error_message,
                    "per_row": per_row,
                    "t_complete": time.time(),
                }
            )
        finally:
            work_q.task_done()


def _worker_thread_lag_batched_stream(
    *,
    thread_id: int,
    work_q: "queue.Queue[Optional[Dict[str, Any]]]",
    actor: Any,
    sink: _MetricsSink,
    stop_event: threading.Event,
    worker_max_retries: int = 0,
    worker_retry_base_s: float = 1.0,
) -> None:
    """lag_batched_stream: per-row streaming consumption via ``ObjectRefGenerator``.

    For each batch:
        gen = actor.collect_batch_streaming.remote(rows)
        for ref in gen:
            row = ray.get(ref)   # blocking, but one ref per rollout

    ``thread_count`` is sized to the trainer's lag bound, not to the
    prompt-group count.

    When ``worker_max_retries > 0``, mirrors the training framework's
    per-prompt-group retry behavior: if a partial-batch failure occurs
    (mid-stream exception, or any per-row error in the returned stream), the
    WHOLE prompt group is re-submitted up to ``worker_max_retries`` times with
    exponential backoff (``worker_retry_base_s * 2^attempt``). Only the final
    attempt's result is recorded; intermediate attempts are counted as
    ``worker_retry_attempts`` in the per-call record, so the trainer never sees
    a partially-successful prompt group.

    Extra streaming-mode metrics:
    - ``time_to_first_row_s``: from .remote() to the first yielded row.
    - ``per_ref_get_latency_s``: per-row ray.get cost.
    - ``worker_retry_attempts``: how many full-batch retries were spent
      (= 0 when no retries needed).
    """
    while not stop_event.is_set():
        item = work_q.get()
        try:
            if item is None:
                return
            t_call_start = time.perf_counter()
            worker_retry_attempts = 0
            final_per_row: List[Dict[str, Any]] = []
            final_time_to_first_row_s: Optional[float] = None
            final_per_ref_get_latency: List[float] = []
            call_error_class: Optional[str] = None
            call_error_message: Optional[str] = None

            for attempt in range(worker_max_retries + 1):
                t_attempt = time.perf_counter()
                attempt_per_row: List[Dict[str, Any]] = []
                attempt_per_ref_get: List[float] = []
                attempt_time_to_first_row: Optional[float] = None
                attempt_error_class: Optional[str] = None
                attempt_error_message: Optional[str] = None
                try:
                    gen = actor.collect_batch_streaming.remote(item["rows"])
                    for ref in gen:
                        get_t0 = time.perf_counter()
                        try:
                            row = ray.get(ref)
                        except Exception as e:
                            row = {
                                "row_idx": -1,
                                "status": "error",
                                "latency_s": time.perf_counter() - get_t0,
                                "error_class": type(e).__name__,
                                "error_message": str(e)[:500],
                            }
                        attempt_per_ref_get.append(time.perf_counter() - get_t0)
                        if attempt_time_to_first_row is None:
                            attempt_time_to_first_row = time.perf_counter() - t_attempt
                        attempt_per_row.append(row)
                except Exception as e:
                    attempt_error_class = type(e).__name__
                    attempt_error_message = str(e)[:500]

                attempt_had_error = (
                    attempt_error_class is not None
                    or len(attempt_per_row) < len(item["rows"])
                    or any(r.get("status") != "ok" for r in attempt_per_row)
                )

                if (not attempt_had_error) or attempt == worker_max_retries:
                    # Either clean success, or final attempt exhausted — record this one.
                    final_per_row = attempt_per_row
                    final_time_to_first_row_s = attempt_time_to_first_row
                    final_per_ref_get_latency = attempt_per_ref_get
                    call_error_class = attempt_error_class
                    call_error_message = attempt_error_message
                    break
                worker_retry_attempts += 1
                backoff_s = worker_retry_base_s * (2**attempt)
                time.sleep(backoff_s)

            call_latency_s = time.perf_counter() - t_call_start
            sink.record_call(
                {
                    "step_idx": item["step_idx"],
                    "thread_id": thread_id,
                    "n_rows_in_call": len(item["rows"]),
                    "n_rows_returned": len(final_per_row),
                    "call_latency_s": call_latency_s,
                    "time_to_first_row_s": final_time_to_first_row_s,
                    "per_ref_get_latency_s_p50": (
                        sorted(final_per_ref_get_latency)[len(final_per_ref_get_latency) // 2]
                        if final_per_ref_get_latency
                        else None
                    ),
                    "per_ref_get_latency_s_max": (
                        max(final_per_ref_get_latency) if final_per_ref_get_latency else None
                    ),
                    "call_error_class": call_error_class,
                    "call_error_message": call_error_message,
                    "worker_retry_attempts": worker_retry_attempts,
                    "per_row": final_per_row,
                    "t_complete": time.time(),
                }
            )
        finally:
            work_q.task_done()


WORKER_FN_BY_MODE = {
    "sync_blocking": _worker_thread_sync_blocking,
    "lag_batched_stream": _worker_thread_lag_batched_stream,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config for the gym sub-server topology. Same shape load_driver consumes; "
        "must define scale_sim.agent_name.",
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=sorted(WORKER_FN_BY_MODE.keys()),
        default="sync_blocking",
        help="Trainer-side shape. sync_blocking = pre-fix production "
        "(1 ray.get per batch, big return). lag_batched_stream = "
        "post-fix production (1 ray.get per row via "
        "ObjectRefGenerator, lag-bounded thread pool).",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        required=True,
        help="Number of trainer threads. For sync_blocking set this to the "
        "prompt-group count (256-2048); for lag_batched_stream set it to the "
        "lag bound (2-8).",
    )
    parser.add_argument(
        "--prompts-per-call",
        type=int,
        required=True,
        help="Rows per single actor.collect_batch{,_streaming}.remote() call.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Total trainer steps. Each step issues thread_count concurrent calls.",
    )
    # Post-fix retry layers (production-faithful when both are enabled):
    parser.add_argument(
        "--in-actor-retry-payload-error",
        action="store_true",
        help="If set, the actor's per-row coroutine retries ClientPayloadError "
        "indefinitely with --in-actor-retry-sleep-s between attempts. "
        "Matches production gym RolloutCollectionHelper._post_subroutine. "
        "Other errors still propagate to the trainer.",
    )
    parser.add_argument("--in-actor-retry-sleep-s", type=float, default=0.5)
    parser.add_argument(
        "--worker-max-retries",
        type=int,
        default=0,
        help="lag_batched_stream only: retry the WHOLE prompt group up to this "
        "many times on any partial-batch error. Matches the training "
        "framework's per-prompt-group retry (MAX_RETRIES=3).",
    )
    parser.add_argument(
        "--worker-retry-base-s",
        type=float,
        default=1.0,
        help="Exponential backoff base for --worker-max-retries: sleep = base * 2^attempt.",
    )
    parser.add_argument("--head-server-host", default="127.0.0.1")
    parser.add_argument("--head-server-port", type=int, default=5000)
    parser.add_argument("--spinup-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster to connect to. Default 'auto' picks up the pre-started cluster.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    agent_name = cfg["scale_sim"]["agent_name"]
    total_per_step = args.thread_count * args.prompts_per_call
    total_rows_needed = total_per_step * args.num_steps

    print(
        f"[mock_trainer] mode={args.mode} config={args.config.name} agent_name={agent_name}",
        f"thread_count={args.thread_count} prompts_per_call={args.prompts_per_call}",
        f"num_steps={args.num_steps} → {total_rows_needed} total rollouts "
        f"({total_per_step} per step, in-flight cap = {args.thread_count}×{args.prompts_per_call})",
        flush=True,
    )

    rows = _load_rows(args.input_jsonl, total_rows_needed, agent_name)

    # 1. Connect to Ray.
    ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=False)

    # 2. Spin up the actor. Use a detached actor name so failures + restarts
    # don't lose the head-server subprocess on us mid-run.
    actor_log_dir = args.output_dir / "actor_logs"
    # max_concurrency must exceed thread_count or Ray queues the excess calls
    # in front of the actor's own async fan-out — that would shift the
    # measurement from "actor under N concurrent ray.get pressure" to "ray's
    # default 1000-slot RPC queue". Set it to thread_count + headroom.
    actor_max_concurrency = max(args.thread_count + 64, 1024)
    actor = NemoGymActor.options(
        name=f"gym_actor_{os.getpid()}",
        lifetime=None,
        max_concurrency=actor_max_concurrency,
    ).remote(
        config_path=str(args.config),
        head_server_host=args.head_server_host,
        head_server_port=args.head_server_port,
        log_dir=str(actor_log_dir),
        spinup_timeout_s=args.spinup_timeout_s,
        in_actor_retry_payload_error=args.in_actor_retry_payload_error,
        in_actor_retry_sleep_s=args.in_actor_retry_sleep_s,
    )
    print("[mock_trainer] waiting for actor to spin up sub-servers …", flush=True)
    spinup_t0 = time.time()
    ready_info = ray.get(actor.wait_ready.remote(), timeout=args.spinup_timeout_s + 30)
    print(f"[mock_trainer] actor ready: {ready_info} ({time.time() - spinup_t0:.1f}s)", flush=True)

    # 3. Telemetry — same shape as load_driver so process_metrics.csv /
    # kernel_metrics.csv slot into the same analysis pipeline.
    process_metrics_csv = args.output_dir / "process_metrics.csv"
    kernel_metrics_csv = args.output_dir / "kernel_metrics.csv"
    per_call_jsonl = args.output_dir / "per_call.jsonl"
    summary_json = args.output_dir / "summary.json"

    # ProcessMetricsSampler uses a background threading.Thread for sampling; the
    # `loop` arg is only needed for asyncio event-loop-lag tracking, which doesn't
    # apply to the mock trainer (the trainer's hot path is blocking threads, not
    # asyncio). Pass loop=None and skip that metric.
    sampler = ProcessMetricsSampler(process_metrics_csv, loop=None, interval_s=1.0)
    kw = KernelWatcher(kernel_metrics_csv)
    sampler.start()
    kw.start()

    sink = _MetricsSink(per_call_jsonl)

    # 4. Build work queue and worker threads.
    work_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    stop_event = threading.Event()
    worker_fn = WORKER_FN_BY_MODE[args.mode]

    def _build_kwargs(thread_id: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            thread_id=thread_id,
            work_q=work_q,
            actor=actor,
            sink=sink,
            stop_event=stop_event,
        )
        if args.mode == "lag_batched_stream":
            kwargs["worker_max_retries"] = args.worker_max_retries
            kwargs["worker_retry_base_s"] = args.worker_retry_base_s
        return kwargs

    workers = [
        threading.Thread(
            target=worker_fn,
            kwargs=_build_kwargs(i),
            name=f"mock_trainer_thread_{i}",
            daemon=True,
        )
        for i in range(args.thread_count)
    ]
    for w in workers:
        w.start()
    print(f"[mock_trainer] {len(workers)} worker threads started", flush=True)

    # 5. Fire steps. At each step boundary we enqueue thread_count batches, one
    # per thread. Each thread pulls one batch and blocks on ray.get. After all
    # batches in the step complete (queue drains), record step wall-clock and
    # proceed to the next step.
    run_t0 = time.time()
    step_wall_clocks: List[float] = []
    try:
        for step_idx in range(args.num_steps):
            step_t0 = time.perf_counter()
            base = step_idx * total_per_step
            for tid in range(args.thread_count):
                start = base + tid * args.prompts_per_call
                end = start + args.prompts_per_call
                work_q.put({"step_idx": step_idx, "rows": rows[start:end]})
            # Wait until *all* batches in this step are done before issuing the next step.
            # This matches the trainer's "wait for batch → step → wait for next batch" loop.
            work_q.join()
            step_wall_clocks.append(time.perf_counter() - step_t0)
            print(
                f"[mock_trainer] step {step_idx + 1}/{args.num_steps} done in {step_wall_clocks[-1]:.1f}s "
                f"(succeeded={sink.n_rows_succeeded} failed={sink.n_rows_failed} errs={dict(sink.error_class_counts)})",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[mock_trainer] KeyboardInterrupt — draining …", flush=True)
    finally:
        # 6. Tear down.
        stop_event.set()
        for _ in workers:
            work_q.put(None)
        for w in workers:
            w.join(timeout=10.0)
        sampler.stop()
        kw.stop()
        sink.close()
        # Best-effort actor shutdown (closes ng_run).
        try:
            ray.get(actor.shutdown.remote(), timeout=30)
        except Exception:
            pass

    run_wall_s = time.time() - run_t0

    # 7. Write summary.json.
    succeeded = sink.n_rows_succeeded
    failed = sink.n_rows_failed
    total = succeeded + failed
    summary = {
        "mode": args.mode,
        "config_path": str(args.config),
        "input_jsonl": str(args.input_jsonl),
        "thread_count": args.thread_count,
        "prompts_per_call": args.prompts_per_call,
        "num_steps": args.num_steps,
        "in_actor_retry_payload_error": args.in_actor_retry_payload_error,
        "in_actor_retry_sleep_s": args.in_actor_retry_sleep_s,
        "worker_max_retries": args.worker_max_retries,
        "worker_retry_base_s": args.worker_retry_base_s,
        "rollouts_per_step_target": total_per_step,
        "n_steps_completed": len(step_wall_clocks),
        "run_wall_clock_s": run_wall_s,
        "rows": {
            "n_attempted": total,
            "n_succeeded": succeeded,
            "n_failed": failed,
            "failure_rate": (failed / total) if total else 0.0,
        },
        "error_class_counts": dict(sink.error_class_counts),
        "per_row_latency_s": {
            "n": len(sink.per_row_latencies),
            "p50": _percentile(sink.per_row_latencies, 0.50),
            "p99": _percentile(sink.per_row_latencies, 0.99),
            "max": max(sink.per_row_latencies, default=None),
            "mean": (statistics.fmean(sink.per_row_latencies) if sink.per_row_latencies else None),
        },
        "per_call_latency_s": {
            "n": len(sink.per_call_latencies),
            "p50": _percentile(sink.per_call_latencies, 0.50),
            "p99": _percentile(sink.per_call_latencies, 0.99),
            "max": max(sink.per_call_latencies, default=None),
        },
        "step_wall_clock_s": {
            "n": len(step_wall_clocks),
            "p50": _percentile(step_wall_clocks, 0.50),
            "p99": _percentile(step_wall_clocks, 0.99),
            "max": max(step_wall_clocks, default=None),
            "all": step_wall_clocks,
        },
        "straggler_blocking_factor": (
            _percentile(step_wall_clocks, 0.50) / _percentile(sink.per_row_latencies, 0.50)
            if step_wall_clocks and sink.per_row_latencies
            else None
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[mock_trainer] summary written → {summary_json}", flush=True)
    print(
        f"[mock_trainer] DONE. failure_rate={summary['rows']['failure_rate']:.4f} "
        f"error_classes={summary['error_class_counts']} "
        f"step_p50={summary['step_wall_clock_s']['p50']:.1f}s "
        f"row_p50={summary['per_row_latency_s']['p50']:.2f}s "
        f"straggler_factor={summary['straggler_blocking_factor']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
