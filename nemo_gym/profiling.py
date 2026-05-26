# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import json
import os
import threading
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
from subprocess import run
from typing import Dict, List, Optional

import yappi
from gprof2dot import main as gprof2dot_main
from pydantic import BaseModel
from pydot import graph_from_dot_file


# -----------------------------------------------------------------------
# Lightweight per-process metric accumulator
# -----------------------------------------------------------------------
#
# This is separate from the yappi-based ``Profiler`` class below. yappi traces
# every Python call across the process; this records *explicit* labelled
# samples from call sites that opt in via ``record_metric``. Useful when you
# want a few specific timings or sizes (e.g. per-rollout wall-clock,
# per-HTTP request size) without the ~30 % overhead yappi adds.
#
# Used by nemo_gym/rollout_collection.py and nemo_gym/server_utils.py to
# emit:
#   - rollout/single_example_total       per-rollout wall-clock seconds
#   - http/request_serialize             per-request orjson.dumps seconds
#   - http/request_size_kb               per-request body bytes / 1024
#   - http/network_round_trip            per-request await client.request() seconds
#   - http/response_size_kb              per-response body bytes / 1024
#   - http/response_deserialize          per-response orjson.loads seconds
#
# Activation: ``enable_profiling(output_dir)`` from setup_profiling in
# nemo_gym/server_utils.py when ``profiling_enabled=True`` in the gym
# config. On process exit, summaries are dumped to
# ``<output_dir>/profiling_metrics_<pid>.json``.

_METRICS_LOCK = threading.Lock()
_METRICS: Dict[str, List[float]] = defaultdict(list)
_PROFILING_ENABLED: bool = False
_OUTPUT_DIR: Optional[Path] = None
_REGISTERED_ATEXIT: bool = False


def is_profiling_enabled() -> bool:
    """Hot-path check used by call sites to skip the recording overhead
    when profiling is off. A single attribute read; suitable for use on
    every HTTP request without measurable cost."""
    return _PROFILING_ENABLED


def record_metric(label: str, value: float) -> None:
    """Append a numeric sample under ``label``. Thread-safe.

    Cheap when profiling is on (one float append under one shared lock);
    cheaper when off (single bool check, returns immediately). Callers
    typically guard their per-call timing computation with
    ``is_profiling_enabled()`` so the perf_counter calls themselves are
    skipped in the off path.
    """
    if not _PROFILING_ENABLED:
        return
    with _METRICS_LOCK:
        _METRICS[label].append(float(value))


def enable_profiling(output_dir: Optional[Path] = None) -> None:
    """Turn metric recording on for this process. Idempotent.

    Registers an atexit handler on first call so accumulated metrics are
    persisted to disk on clean shutdown. If ``output_dir`` is None,
    metrics summaries are printed to stdout at exit instead of written
    to a file.
    """
    global _PROFILING_ENABLED, _OUTPUT_DIR, _REGISTERED_ATEXIT
    _PROFILING_ENABLED = True
    if output_dir is not None:
        _OUTPUT_DIR = Path(output_dir)
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _REGISTERED_ATEXIT:
        atexit.register(_dump_metrics)
        _REGISTERED_ATEXIT = True


def _summarize(values: List[float]) -> dict:
    """p50/p90/p99/max/count/sum summary for one label's series.

    Cheap O(n log n) sort each time — only runs once at process exit.
    Good enough for the ~10 labels we record; if cardinality grows we'd
    switch to a t-digest or HdrHistogram.
    """
    if not values:
        return {"count": 0}
    n = len(values)
    sorted_vals = sorted(values)

    def pct(p: float) -> float:
        return sorted_vals[min(n - 1, int(n * p))]

    return {
        "count": n,
        "sum": sum(values),
        "mean": sum(values) / n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": sorted_vals[-1],
    }


def _dump_metrics() -> None:
    """Persist accumulated metrics on process exit. Called via atexit.

    Writes ``<output_dir>/profiling_metrics_<pid>.json`` with one summary
    entry per label. If no output_dir was set, prints the summary to
    stdout instead. Robust to mid-write process death — the per-pid
    filename means parallel sub-servers don't clobber each other.
    """
    with _METRICS_LOCK:
        snapshot = {k: list(v) for k, v in _METRICS.items()}
    if not snapshot:
        return
    summary = {label: _summarize(values) for label, values in snapshot.items()}
    payload = {
        "pid": os.getpid(),
        "dumped_at": time.time(),
        "summary": summary,
    }
    if _OUTPUT_DIR is not None:
        out_path = _OUTPUT_DIR / f"profiling_metrics_{payload['pid']}.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"[profiling] wrote {len(snapshot)} metric labels to {out_path}")
        except Exception as e:
            # Don't let metric-dump failure crash an otherwise clean shutdown.
            print(f"[profiling] WARN: failed to write {out_path}: {e}")
    else:
        print(f"[profiling] metrics summary (pid={payload['pid']}):")
        print(json.dumps(summary, indent=2))


class Profiler(BaseModel):
    name: str
    base_profile_dir: Path

    # Used to clean up and filter out unnecessary information in the yappi log
    required_str: Optional[str] = None

    def model_post_init(self, context):
        assert " " not in self.name, f"Spaces are not allowed in profiler name, but got `{repr(self.name)}`"
        return super().model_post_init(context)

    def _check_for_dot_installation(self) -> None:  # pragma: no cover
        res = run("dot -V", shell=True, check=False, capture_output=True)
        if res.returncode == 0:
            return

        raise RuntimeError("""You must install dot in order to use this profiling too.
Please install dot using:
- Mac: `brew install graphviz`
- Linux: `apt update && apt install -y graphviz`""")

    def start(self) -> None:
        self._check_for_dot_installation()

        yappi.set_clock_type("CPU")
        yappi.start()
        print(f"🔍 Enabled profiling for {self.name}")

    def stop(self) -> None:
        print(f"🛑 Stopping profiler for {self.name}. Check {self.base_profile_dir} for the metrics!")
        yappi.stop()
        self.dump()

    def dump(self) -> None:
        self.base_profile_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.base_profile_dir / f"{self.name}.log"
        callgrind_path = self.base_profile_dir / f"{self.name}.callgrind"
        callgrind_dotfile_path = self.base_profile_dir / f"{self.name}.dot"
        callgrind_graph_path = self.base_profile_dir / f"{self.name}.png"

        yappi.get_func_stats().save(callgrind_path, type="CALLGRIND")
        gprof2dot_main(argv=f"--format=callgrind --output={callgrind_dotfile_path} -e 5 -n 5 {callgrind_path}".split())

        (graph,) = graph_from_dot_file(callgrind_dotfile_path)
        graph.write_png(callgrind_graph_path)

        buffer = StringIO()
        yappi.get_func_stats().print_all(
            out=buffer,
            columns={
                0: ("name", 200),
                1: ("ncall", 10),
                2: ("tsub", 8),
                3: ("ttot", 8),
                4: ("tavg", 8),
            },
        )

        buffer.seek(0)
        res = ""
        past_header = False
        for line in buffer:
            if not past_header or (self.required_str and self.required_str in line):
                res += line

            if line.startswith("name"):
                past_header = True

        with open(log_path, "w") as f:
            f.write(res)
