# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline throughput analysis from a cell's per-rollout records.

The driver's headline throughput is ``completed / wall_clock`` over the whole
run, which includes the ramp (opening tens of thousands of sockets, filling the
connector pool). At high concurrency the ramp is a large fraction of the window
and biases throughput downward. This reads the per-rollout completion timestamps
and reports a warm-up-excluded steady-state throughput plus a per-second
completion curve, so the ramp / plateau / collapse shape is visible.

Read-only: it never touches the running harness.

    python analyze_throughput.py results/workstation/concurrency_scaling/c8192
    python analyze_throughput.py <dir-or-per_rollout_retries.jsonl> --warmup-s 30 --write-curve
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_completion_times(jsonl_path: Path) -> List[float]:
    """Return sorted completed_at timestamps for succeeded rollouts."""
    times: List[float] = []
    with jsonl_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("succeeded") and rec.get("completed_at"):
                times.append(float(rec["completed_at"]))
    times.sort()
    return times


def analyze(jsonl_path: Path, warmup_s: float = 30.0) -> Dict[str, Any]:
    """Compute naive and warm-up-excluded steady-state throughput.

    Steady-state drops the first ``warmup_s`` seconds of completions and divides
    the remaining completions by the remaining elapsed time. Falls back to the
    naive number when the run is too short to exclude a warm-up.
    """
    times = _load_completion_times(jsonl_path)
    n = len(times)
    if n == 0:
        return {"n_completed": 0, "naive_rps": None, "steady_rps": None, "warmup_s": warmup_s}

    t0, t1 = times[0], times[-1]
    span = t1 - t0
    naive = (n / span) if span > 0 else None

    if span > 2 * warmup_s:
        cut = t0 + warmup_s
        plateau = [t for t in times if t >= cut]
        dur = t1 - cut
        steady = (len(plateau) / dur) if dur > 0 else None
    else:
        # Too short to separate ramp from plateau; steady == naive.
        steady = naive

    return {
        "n_completed": n,
        "span_s": round(span, 2),
        "naive_rps": naive,
        "steady_rps": steady,
        "warmup_s": warmup_s,
    }


def per_second_curve(jsonl_path: Path) -> List[int]:
    """Completions bucketed per second from the first completion."""
    times = _load_completion_times(jsonl_path)
    if not times:
        return []
    base = math.floor(times[0])
    buckets: Counter[int] = Counter(int(t - base) for t in times)
    return [buckets.get(s, 0) for s in range(int(times[-1] - base) + 1)]


def _resolve_jsonl(path: Path) -> Optional[Path]:
    if path.is_dir():
        cand = path / "per_rollout_retries.jsonl"
        return cand if cand.exists() else None
    return path if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="A cell results dir or a per_rollout_retries.jsonl file.")
    p.add_argument("--warmup-s", type=float, default=30.0, help="Seconds of ramp to exclude (default 30).")
    p.add_argument("--write-curve", action="store_true", help="Write throughput_curve.csv next to the input.")
    args = p.parse_args()

    jsonl = _resolve_jsonl(args.path)
    if jsonl is None:
        p.error(f"No per_rollout_retries.jsonl found at {args.path}")

    res = analyze(jsonl, args.warmup_s)
    print(json.dumps(res, indent=2))

    if args.write_curve:
        curve = per_second_curve(jsonl)
        out = jsonl.parent / "throughput_curve.csv"
        with out.open("w") as f:
            f.write("second,completions\n")
            for s, c in enumerate(curve):
                f.write(f"{s},{c}\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
