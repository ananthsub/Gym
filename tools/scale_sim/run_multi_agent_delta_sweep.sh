#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-agent validation delta — runs the focused subset of the full multi-agent
# matrix that exercises every new code path (multi-agent dispatch, spinup_only
# mode, generator) without committing to the full overnight sweep.
#
# Four cells, ~45-60 min wall-clock total:
#
#   1. N=4 spinup_only      (~5 min)    — generator + spinup_only mode + venv-sharing assumption
#   2. N=4 loaded           (~10 min)   — round-robin dispatch + per-agent summary
#   3. N=64 spinup_only     (~5-10 min) — first interesting spinup cost; predicted "spinup > 60s"
#   4. N=16 loaded          (~20-30 min) — first interesting loaded cell; total conc=4096 / per-agent=256
#
# After phase 4, aggregates per-cell summary.json files into a single
# `multi_agent_delta.csv` with the headline columns. Each cell wrapped in
# `|| true` so a single failure doesn't kill the rest.
#
# Designed for `run_on_slurm_baremetal.sh batch` via DRIVER_SCRIPT env var:
#
#     cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
#     DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_delta_sweep.sh \
#     JOB_NAME=nemo-gym-multi-agent-delta \
#       bash tools/scale_sim/run_on_slurm_baremetal.sh batch

set -uo pipefail

# ---------- Inside-container/venv setup ----------
GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${GYM_VENV}/bin/activate"
fi
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[multi-agent-delta] ulimit -n: $(ulimit -n)"
echo "[multi-agent-delta] hostname:  $(hostname)"
echo "[multi-agent-delta] gym path:  ${CONTAINER_GYM_PATH}"
echo "[multi-agent-delta] which ng_run: $(which ng_run || echo NOT FOUND)"
which ng_run >/dev/null 2>&1 || { echo "[multi-agent-delta] ng_run not on PATH — venv broken"; exit 1; }

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
echo "[multi-agent-delta] git sha:   ${GIT_SHA}"

RESULTS_ROOT="${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}"
mkdir -p "${RESULTS_ROOT}"

BASE_CFG="configs/multi_agent_base.yaml"
GEN_DIR="configs/_generated"
mkdir -p "${GEN_DIR}"

# ---------- Generate small input JSONL once (reused across loaded cells) ----------
echo
echo "[multi-agent-delta] Generating input JSONL..."
python data/generate_data.py --n 5000 --user-input-size-bytes 1024 \
  --output data/multi_agent_delta.jsonl

# ---------- Phase 1: N=4 spinup_only (validates generator + spinup_only mode) ----------
echo
echo "[multi-agent-delta] === Phase 1: N=4 spinup_only ==="
python configs/_gen_multi_agent.py \
  --base "${BASE_CFG}" --n-agents 4 \
  --out "${GEN_DIR}/multi_agent_delta_n4_spinup.yaml" >&2 || { echo "[multi-agent-delta] generator failed at N=4"; exit 1; }

python sweep_runner.py \
  --config "${GEN_DIR}/multi_agent_delta_n4_spinup.yaml" \
  --git-sha "${GIT_SHA}" \
  --head-server-host 127.0.0.1 --head-server-port 5000 \
  --driver-mode spinup_only --idle-window-s 15 \
  --spinup-timeout-s 300 --teardown-sleep-s 5 || \
  echo "[multi-agent-delta] WARN: Phase 1 failed (continuing)"

# Venv-sharing assumption check: count how many .venv dirs exist per server type.
# Should be exactly 1 each. If any of these are >1, the per-server-type venv-sharing
# assumption is wrong and we'd need the symlink fallback.
echo
echo "[multi-agent-delta] Venv-sharing check after Phase 1:"
echo "  resources_servers/synthetic_resources*/.venv:"
ls -d resources_servers/synthetic_resources*/.venv 2>/dev/null | sed 's/^/    /' || echo "    (none yet)"
echo "  responses_api_models/synthetic_model*/.venv:"
ls -d responses_api_models/synthetic_model*/.venv 2>/dev/null | sed 's/^/    /' || echo "    (none yet)"
echo "  responses_api_agents/simple_agent*/.venv:"
ls -d ../../responses_api_agents/simple_agent*/.venv 2>/dev/null | sed 's/^/    /' || echo "    (none yet)"

# ---------- Phase 2: N=4 loaded (validates round-robin + per-agent breakdown) ----------
echo
echo "[multi-agent-delta] === Phase 2: N=4 loaded (concurrency=1024, 4000 requests) ==="
python configs/_gen_multi_agent.py \
  --base "${BASE_CFG}" --n-agents 4 \
  --concurrency 1024 --total-requests 4000 \
  --out "${GEN_DIR}/multi_agent_delta_n4_loaded.yaml" >&2 || { echo "[multi-agent-delta] generator failed at N=4 loaded"; exit 1; }

python sweep_runner.py \
  --config "${GEN_DIR}/multi_agent_delta_n4_loaded.yaml" \
  --input-jsonl "${CONTAINER_GYM_PATH}/tools/scale_sim/data/multi_agent_delta.jsonl" \
  --git-sha "${GIT_SHA}" \
  --head-server-host 127.0.0.1 --head-server-port 5000 \
  --driver-mode loaded \
  --spinup-timeout-s 300 --teardown-sleep-s 5 || \
  echo "[multi-agent-delta] WARN: Phase 2 failed (continuing)"

# ---------- Phase 3: N=64 spinup_only (first non-trivial spinup data point) ----------
echo
echo "[multi-agent-delta] === Phase 3: N=64 spinup_only ==="
python configs/_gen_multi_agent.py \
  --base "${BASE_CFG}" --n-agents 64 \
  --out "${GEN_DIR}/multi_agent_delta_n64_spinup.yaml" >&2 || { echo "[multi-agent-delta] generator failed at N=64"; exit 1; }

python sweep_runner.py \
  --config "${GEN_DIR}/multi_agent_delta_n64_spinup.yaml" \
  --git-sha "${GIT_SHA}" \
  --head-server-host 127.0.0.1 --head-server-port 5000 \
  --driver-mode spinup_only --idle-window-s 30 \
  --spinup-timeout-s 600 --teardown-sleep-s 8 || \
  echo "[multi-agent-delta] WARN: Phase 3 failed (continuing)"

# ---------- Phase 4: N=16 loaded (first non-trivial loaded data point) ----------
echo
echo "[multi-agent-delta] === Phase 4: N=16 loaded (total_conc=4096, per-agent=256, 10000 requests) ==="
python configs/_gen_multi_agent.py \
  --base "${BASE_CFG}" --n-agents 16 \
  --concurrency 4096 --total-requests 10000 \
  --out "${GEN_DIR}/multi_agent_delta_n16_loaded.yaml" >&2 || { echo "[multi-agent-delta] generator failed at N=16 loaded"; exit 1; }

python sweep_runner.py \
  --config "${GEN_DIR}/multi_agent_delta_n16_loaded.yaml" \
  --input-jsonl "${CONTAINER_GYM_PATH}/tools/scale_sim/data/multi_agent_delta.jsonl" \
  --git-sha "${GIT_SHA}" \
  --head-server-host 127.0.0.1 --head-server-port 5000 \
  --driver-mode loaded \
  --spinup-timeout-s 300 --teardown-sleep-s 5 || \
  echo "[multi-agent-delta] WARN: Phase 4 failed (continuing)"

# ---------- Aggregate: only the four delta cells, into multi_agent_delta.csv ----------
echo
echo "[multi-agent-delta] Aggregating delta results..."
python <<PYEOF
import csv, json
from pathlib import Path

results_root = Path("${RESULTS_ROOT}")
# Pick out only the four cells this script generated (by stem prefix).
delta_stems = {
    "multi_agent_delta_n4_spinup",
    "multi_agent_delta_n4_loaded",
    "multi_agent_delta_n64_spinup",
    "multi_agent_delta_n16_loaded",
}

rows = []
for cell_dir in sorted(results_root.iterdir()):
    if not cell_dir.is_dir():
        continue
    # sweep_runner names the cell dir <cfg_stem>_<ts>, so strip the trailing _<ts>.
    stem = "_".join(cell_dir.name.split("_")[:-2]) if cell_dir.name.count("_") >= 2 else cell_dir.name
    if stem not in delta_stems:
        continue
    summary_path = cell_dir / "summary.json"
    if not summary_path.exists():
        rows.append({"cell": cell_dir.name, "status": "no_summary"})
        continue
    try:
        s = json.loads(summary_path.read_text())
    except Exception as e:
        rows.append({"cell": cell_dir.name, "status": f"summary_read_err: {e}"})
        continue

    base = {
        "cell": cell_dir.name,
        "stem": stem,
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
        # Cheap sanity stats on the per-agent breakdown — round-robin should give
        # roughly equal n_rollouts per agent.
        per_n = [v.get("n_rollouts", 0) for v in per_agent.values()]
        per_n_min = min(per_n) if per_n else None
        per_n_max = max(per_n) if per_n else None
        rows.append({
            **base,
            "n_rollouts": retry.get("n_rollouts"),
            "failure_rate": retry.get("failure_rate"),
            "retry_rate": retry.get("retry_rate"),
            "p50_s": lat.get("p50_s"),
            "p99_s": lat.get("p99_s"),
            "max_s": lat.get("max_s"),
            "n_per_agent_min": per_n_min,
            "n_per_agent_max": per_n_max,
            "n_per_agent_count": len(per_agent),
        })

if not rows:
    print(f"[multi-agent-delta] No delta cells found under {results_root}")
else:
    out_csv = results_root / "multi_agent_delta.csv"
    cols = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[multi-agent-delta] Delta summary: {out_csv}")
    headline = ["stem", "mode", "n_agents", "concurrency", "n_rollouts",
                "failure_rate", "retry_rate", "p50_s", "p99_s",
                "n_per_agent_count", "n_per_agent_min", "n_per_agent_max",
                "tcp_inuse", "file_nr_used", "stop_reason"]
    cols_present = [c for c in headline if any(c in r for r in rows)]
    print("\t".join(cols_present))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols_present))
PYEOF

echo
echo "[multi-agent-delta] Done. Delta results under ${RESULTS_ROOT}/multi_agent_delta.csv"
echo "[multi-agent-delta] If all four cells succeeded with failure_rate=0 and per_agent_count matches n_agents, the multi-agent code is good."
echo "[multi-agent-delta] To kick off the full multi-agent matrix:"
echo "[multi-agent-delta]   DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_sweep.sh \\"
echo "[multi-agent-delta]   JOB_NAME=nemo-gym-multi-agent \\"
echo "[multi-agent-delta]     bash tools/scale_sim/run_on_slurm_baremetal.sh batch"
