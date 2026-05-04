#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-aggregate multi-agent sweep cells under a results/<sha>/ directory.

The aggregator embedded in `run_multi_agent_sweep.sh` only runs once at the end
of the sweep and can miss cells if the sweep was interrupted or if cells landed
their `summary.json` after the aggregator ran. This script re-builds the master
CSV from whatever exists on disk *now*, with broken-out per-sub-sweep tables.

Stdlib-only (no venv activation needed). Usage::

    python tools/scale_sim/analyze_multi_agent.py /path/to/results/<sha>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: analyze_multi_agent.py <results_dir>")
        sys.exit(2)
    results_root = Path(sys.argv[1])
    if not results_root.is_dir():
        print(f"not a directory: {results_root}")
        sys.exit(2)

    rows: list[dict] = []
    for cell_dir in sorted(results_root.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("multi_agent_n"):
            continue
        summary_path = cell_dir / "summary.json"
        if not summary_path.exists():
            rows.append({"cell": cell_dir.name, "status": "no_summary"})
            continue
        try:
            s = json.loads(summary_path.read_text())
        except Exception as e:
            rows.append({"cell": cell_dir.name, "status": f"err: {e}"})
            continue
        base = {
            "cell": cell_dir.name,
            "mode": s.get("mode", "loaded"),
            "n_agents": s.get("n_agents"),
            "concurrency": s.get("concurrency"),
            "total_requests": s.get("total_requests"),
            "stop_reason": s.get("stop_reason"),
        }
        if base["mode"] == "spinup_only":
            idle = s.get("idle_kernel", {}) or {}
            proc = s.get("idle_driver_process", {}) or {}
            rows.append({
                **base,
                "tcp_inuse": idle.get("tcp_inuse"),
                "tcp_tw": idle.get("tcp_tw"),
                "file_nr_used": idle.get("file_nr_used"),
                "loadavg_1m": idle.get("loadavg_1m"),
                "driver_rss_mb": proc.get("rss_mb"),
            })
        else:
            retry = s.get("retry_summary", {}) or {}
            lat = s.get("latency_summary", {}) or {}
            per_agent = s.get("per_agent", {}) or {}
            per_n = [v.get("n_rollouts", 0) for v in per_agent.values()]
            rows.append({
                **base,
                "n_rollouts": retry.get("n_rollouts"),
                "failure_rate": retry.get("failure_rate"),
                "retry_rate": retry.get("retry_rate"),
                "p50_s": lat.get("p50_s"),
                "p99_s": lat.get("p99_s"),
                "max_s": lat.get("max_s"),
                "n_per_agent_min": min(per_n) if per_n else None,
                "n_per_agent_max": max(per_n) if per_n else None,
                "n_per_agent_count": len(per_agent),
            })

    out_csv = results_root / "multi_agent_master_full.csv"
    cols = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}\n{len(rows)} cells\n")

    print("=== Sub-sweep A: spinup_only ===")
    print(f"{'N':>5} {'tcp_inuse':>10} {'tcp_tw':>8} {'file_nr':>10} {'loadavg':>8}")
    for r in rows:
        if r.get("mode") == "spinup_only":
            n = r.get("n_agents")
            print(f"{str(n):>5} {str(r.get('tcp_inuse', '')):>10} {str(r.get('tcp_tw', '')):>8} {str(r.get('file_nr_used', '')):>10} {str(r.get('loadavg_1m', '')):>8}")

    print("\n=== Sub-sweep B: loaded fixed total=4096, r=10000 ===")
    hdr = f"{'N':>5} {'per-ag':>7} {'p50_s':>8} {'p99_s':>10} {'max_s':>10} {'fail':>6} {'retry':>6} {'n_done':>8} {'stop':>30}"
    print(hdr)
    for r in rows:
        if r.get("mode") == "loaded" and r.get("concurrency") == 4096 and r.get("total_requests") == 10000:
            n = r.get("n_agents") or 1
            per_agent = 4096 // n
            print(
                f"{str(n):>5} {per_agent:>7} "
                f"{(r.get('p50_s') or 0):>8.2f} {(r.get('p99_s') or 0):>10.2f} {(r.get('max_s') or 0):>10.2f} "
                f"{(r.get('failure_rate') or 0):>6.3f} {(r.get('retry_rate') or 0):>6.3f} "
                f"{str(r.get('n_rollouts', '')):>8} {str(r.get('stop_reason', ''))[:30]:>30}"
            )

    print("\n=== Sub-sweep C: loaded fixed per-agent=256 ===")
    print(hdr.replace("per-ag", "total "))
    for r in rows:
        if r.get("mode") == "loaded" and r.get("n_agents") and r.get("concurrency") == 256 * r.get("n_agents"):
            n = r["n_agents"]
            print(
                f"{str(n):>5} {r.get('concurrency', ''):>7} "
                f"{(r.get('p50_s') or 0):>8.2f} {(r.get('p99_s') or 0):>10.2f} {(r.get('max_s') or 0):>10.2f} "
                f"{(r.get('failure_rate') or 0):>6.3f} {(r.get('retry_rate') or 0):>6.3f} "
                f"{str(r.get('n_rollouts', '')):>8} {str(r.get('stop_reason', ''))[:30]:>30}"
            )

    failures = [r for r in rows if r.get("status") or (r.get("stop_reason") not in (None, ""))]
    if failures:
        print("\n=== Failures / early-stops ===")
        for r in failures:
            print(f"  {r['cell']}  status={r.get('status', '')}  stop={r.get('stop_reason', '')}")


if __name__ == "__main__":
    main()
