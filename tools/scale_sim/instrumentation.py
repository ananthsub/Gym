# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instrumentation helpers for tools/scale_sim.

Three concerns:

1. ``RetryTracker`` — records per-rollout retry counts + error classes, computes
   sliding-window aggregate stats. Used by ``load_driver.py``.
2. ``ProcessMetricsSampler`` — background thread that samples RSS / FDs / CPU / asyncio
   loop lag every ~100 ms. Dumps to a CSV at shutdown.
3. ``KernelWatcher`` — separate process (spawned via ``Popen``) that samples
   ``/proc/net/sockstat`` etc. once a second. Out-of-process so it survives even when
   the head actor's event loop is wedged.

All artifacts live under a single ``output_dir`` that the caller picks (typically
``tools/scale_sim/results/<git_sha>/<run_id>/``).
"""

from __future__ import annotations

import asyncio
import csv
import os
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import psutil


# -----------------------------------------------------------------------------
# RetryTracker
# -----------------------------------------------------------------------------


@dataclass
class _RolloutRecord:
    rollout_idx: int
    n_attempts: int
    total_retry_wait_s: float
    error_classes: List[str]
    succeeded: bool
    completed_at: float
    # Optional: which agent name this rollout was dispatched to. Set when the
    # driver runs in multi-agent mode (Axis B). Always present at completion
    # time but may be None on partial records.
    agent_name: Optional[str] = None


class RetryTracker:
    """Per-rollout retry instrumentation. Thread-safe append, async-safe summary.

    Use ``record_attempt(rollout_idx, error_class)`` for each retry attempt and
    ``record_completion(rollout_idx, succeeded, agent_name)`` once the rollout
    finishes (or is abandoned). ``summary_window(n_seconds)`` returns sliding-window
    aggregates for the early-stop check. ``summary_by_agent()`` breaks the totals
    down per agent for Axis-B analysis (returns ``{}`` if no agent_name was passed).
    """

    def __init__(self, output_jsonl_path: Path) -> None:
        self._lock = threading.Lock()
        self._records: Dict[int, _RolloutRecord] = {}
        self._completed: Deque[_RolloutRecord] = deque()
        self._jsonl_path = output_jsonl_path
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_file = output_jsonl_path.open("w")

    def record_attempt(self, rollout_idx: int, error_class: Optional[str] = None) -> None:
        with self._lock:
            rec = self._records.get(rollout_idx)
            if rec is None:
                rec = _RolloutRecord(
                    rollout_idx=rollout_idx,
                    n_attempts=0,
                    total_retry_wait_s=0.0,
                    error_classes=[],
                    succeeded=False,
                    completed_at=0.0,
                )
                self._records[rollout_idx] = rec
            rec.n_attempts += 1
            if error_class is not None:
                rec.error_classes.append(error_class)

    def record_completion(
        self,
        rollout_idx: int,
        succeeded: bool,
        agent_name: Optional[str] = None,
    ) -> None:
        import orjson

        with self._lock:
            rec = self._records.pop(rollout_idx, None)
            if rec is None:
                rec = _RolloutRecord(
                    rollout_idx=rollout_idx,
                    n_attempts=1,
                    total_retry_wait_s=0.0,
                    error_classes=[],
                    succeeded=succeeded,
                    completed_at=time.time(),
                    agent_name=agent_name,
                )
            else:
                rec.succeeded = succeeded
                rec.completed_at = time.time()
                rec.agent_name = agent_name
            self._completed.append(rec)
            self._jsonl_file.write(
                orjson.dumps(
                    {
                        "rollout_idx": rec.rollout_idx,
                        "n_attempts": rec.n_attempts,
                        "n_retries": max(0, rec.n_attempts - 1),
                        "error_classes": rec.error_classes,
                        "succeeded": rec.succeeded,
                        "completed_at": rec.completed_at,
                        "agent_name": rec.agent_name,
                    }
                ).decode()
                + "\n"
            )

    def flush(self) -> None:
        with self._lock:
            self._jsonl_file.flush()

    def close(self) -> None:
        with self._lock:
            self._jsonl_file.close()

    def summary_all(self) -> Dict:
        with self._lock:
            completed = list(self._completed)
        return _summarize(completed)

    def summary_window(self, window_s: float) -> Dict:
        cutoff = time.time() - window_s
        with self._lock:
            completed = [r for r in self._completed if r.completed_at >= cutoff]
        return _summarize(completed)

    def summary_by_agent(self) -> Dict[str, Dict]:
        """Per-agent breakdown over all completed rollouts.

        Returns an empty dict if no rollout had an ``agent_name`` (single-agent runs).
        Otherwise returns ``{agent_name: <same-shape-as-summary_all>}`` with one row
        per agent that has at least one completion.
        """
        with self._lock:
            completed = list(self._completed)
        if not any(r.agent_name for r in completed):
            return {}
        by_agent: Dict[str, List[_RolloutRecord]] = {}
        for r in completed:
            if r.agent_name is None:
                continue
            by_agent.setdefault(r.agent_name, []).append(r)
        return {name: _summarize(recs) for name, recs in sorted(by_agent.items())}


def _summarize(records: List[_RolloutRecord]) -> Dict:
    n = len(records)
    if n == 0:
        return {
            "n_rollouts": 0,
            "failure_rate": 0.0,
            "retry_rate": 0.0,
            "p_at_least_1_retry": 0.0,
            "p_at_least_2_retries": 0.0,
            "p_at_least_5_retries": 0.0,
            "p_at_least_10_retries": 0.0,
            "n_attempts_mean": 0.0,
            "n_attempts_p50": 0,
            "n_attempts_p99": 0,
            "n_attempts_max": 0,
            "error_class_counts": {},
        }

    n_failed = sum(1 for r in records if not r.succeeded)
    n_with_retry = sum(1 for r in records if r.n_attempts > 1)
    attempts_sorted = sorted(r.n_attempts for r in records)

    def pct(p: float) -> int:
        return attempts_sorted[min(n - 1, int(n * p))]

    error_counter: Counter[str] = Counter()
    for r in records:
        for cls in r.error_classes:
            error_counter[cls] += 1

    return {
        "n_rollouts": n,
        "failure_rate": n_failed / n,
        "retry_rate": n_with_retry / n,
        "p_at_least_1_retry": sum(1 for r in records if r.n_attempts >= 2) / n,
        "p_at_least_2_retries": sum(1 for r in records if r.n_attempts >= 3) / n,
        "p_at_least_5_retries": sum(1 for r in records if r.n_attempts >= 6) / n,
        "p_at_least_10_retries": sum(1 for r in records if r.n_attempts >= 11) / n,
        "n_attempts_mean": sum(attempts_sorted) / n,
        "n_attempts_p50": pct(0.50),
        "n_attempts_p99": pct(0.99),
        "n_attempts_max": attempts_sorted[-1],
        "error_class_counts": dict(error_counter),
    }


# -----------------------------------------------------------------------------
# ProcessMetricsSampler
# -----------------------------------------------------------------------------


class ProcessMetricsSampler:
    """Background-thread sampler for the *current* process's RSS / FDs / CPU.

    Optionally also captures asyncio event-loop lag (the difference between
    ``loop.time()`` ticks scheduled to fire every 100 ms and when they actually
    fired). Loop lag is the headline M3 indicator.

    Usage::

        sampler = ProcessMetricsSampler(Path("metrics.csv"), loop=asyncio.get_running_loop())
        sampler.start()
        try:
            ...
        finally:
            sampler.stop()
    """

    def __init__(
        self,
        output_csv_path: Path,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        interval_s: float = 0.1,
    ) -> None:
        self._output_csv_path = output_csv_path
        self._loop = loop
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process(os.getpid())
        # Baseline cpu_percent so the first reading isn't 0.
        self._proc.cpu_percent(interval=None)

        # Loop-lag tracking
        self._loop_lag_samples: Deque[float] = deque(maxlen=64)

    def start(self) -> None:
        self._output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self._loop is not None:
            self._loop.call_later(self._interval_s, self._loop_tick, self._loop.time())
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _loop_tick(self, scheduled_for: float) -> None:
        if self._loop is None:
            return
        actual = self._loop.time()
        lag = max(0.0, actual - scheduled_for)
        self._loop_lag_samples.append(lag)
        if not self._stop_event.is_set():
            self._loop.call_later(self._interval_s, self._loop_tick, actual + self._interval_s)

    def _run(self) -> None:
        with self._output_csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "rss_mb", "num_fds", "cpu_pct", "loop_lag_p99_ms"])
            while not self._stop_event.is_set():
                try:
                    rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                    num_fds = self._proc.num_fds() if hasattr(self._proc, "num_fds") else -1
                    cpu_pct = self._proc.cpu_percent(interval=None)
                    if self._loop_lag_samples:
                        sorted_samples = sorted(self._loop_lag_samples)
                        p99_ms = sorted_samples[max(0, int(len(sorted_samples) * 0.99) - 1)] * 1000
                    else:
                        p99_ms = 0.0
                    writer.writerow([f"{time.time():.3f}", f"{rss_mb:.1f}", num_fds, f"{cpu_pct:.1f}", f"{p99_ms:.2f}"])
                    f.flush()
                except Exception:
                    # Don't let metrics sampling crash the run.
                    pass
                self._stop_event.wait(self._interval_s)


# -----------------------------------------------------------------------------
# KernelWatcher
# -----------------------------------------------------------------------------


_KERNEL_WATCHER_SCRIPT = r"""
import csv, sys, time
out_path = sys.argv[1]
interval_s = float(sys.argv[2])
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t", "tcp_inuse", "tcp_tw", "tcp_orphan", "file_nr_used", "loadavg_1m"])
    while True:
        try:
            with open("/proc/net/sockstat") as src:
                stats = src.read()
            tcp_inuse = tcp_tw = tcp_orphan = -1
            for line in stats.splitlines():
                if line.startswith("TCP:"):
                    parts = line.split()
                    for i, k in enumerate(parts):
                        if k == "inuse" and i + 1 < len(parts): tcp_inuse = int(parts[i + 1])
                        if k == "tw" and i + 1 < len(parts): tcp_tw = int(parts[i + 1])
                        if k == "orphan" and i + 1 < len(parts): tcp_orphan = int(parts[i + 1])
            try:
                with open("/proc/sys/fs/file-nr") as src:
                    file_nr_used = int(src.read().split()[0])
            except Exception:
                file_nr_used = -1
            try:
                with open("/proc/loadavg") as src:
                    loadavg_1m = float(src.read().split()[0])
            except Exception:
                loadavg_1m = -1.0
            writer.writerow([f"{time.time():.3f}", tcp_inuse, tcp_tw, tcp_orphan, file_nr_used, f"{loadavg_1m:.2f}"])
            f.flush()
        except Exception:
            pass
        time.sleep(interval_s)
"""


class KernelWatcher:
    """Out-of-process sampler for /proc/net/sockstat + file-nr + loadavg.

    Spawned as a separate process so it survives even when the head actor's event
    loop is wedged. Dumps one CSV at ``output_csv_path``.
    """

    def __init__(self, output_csv_path: Path, interval_s: float = 1.0) -> None:
        self._output_csv_path = output_csv_path
        self._interval_s = interval_s
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _KERNEL_WATCHER_SCRIPT, str(self._output_csv_path), str(self._interval_s)],
        )

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
