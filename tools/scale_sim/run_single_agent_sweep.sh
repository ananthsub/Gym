#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs the full single-agent scalability characterization sweep matrix end-to-end
# inside the container. Designed to be invoked by `run_on_slurm.sh batch` via the
# DRIVER_SCRIPT env var, but also runnable manually inside an interactive
# allocation.
#
# Topology: 1 head + 1 model + 1 resources + 1 simple_agent. For the multi-agent
# (N agents + N resources + 1 shared model) sweep, see run_multi_agent_sweep.sh.
#
# Time budget: ~2.5 hours wall-clock for the full set. Set TIME=4:00:00 (or
# higher) on the sbatch submission and use a qos without the 2h interactive cap.
#
# Each sweep is wrapped in `|| true` so one failure doesn't stop the rest. After
# all sweeps complete, aggregates per-experiment CSVs into a single master
# summary.

set -uo pipefail

# ---------- Inside-container setup ----------
GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

source "${GYM_VENV}/bin/activate"
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[single-agent] ulimit -n: $(ulimit -n)"
echo "[single-agent] hostname:  $(hostname)"
echo "[single-agent] gym path:  ${CONTAINER_GYM_PATH}"
echo "[single-agent] which ng_run: $(which ng_run || echo NOT FOUND)"

# GIT_SHA controls which results/<sha>/ dir the new cells land in. Defaults to
# the current HEAD, but can be overridden via env so a resume targets the same
# dir as the original (partial) run — useful when you've committed a fix on
# top of the run you're trying to recover.
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"
echo "[single-agent] git sha:   ${GIT_SHA}"

RESULTS_ROOT="${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}"
mkdir -p "${RESULTS_ROOT}"

# ---------- Resume support ----------
# Each sub-sweep below writes its aggregated `sweep_results.csv` only after all
# of its cells finish. If a previous job ran to wall-clock and got killed
# mid-way through one of the sub-sweeps, the *completed* sub-sweeps will have a
# `${RESULTS_ROOT}/<exp_name>_<ts>/sweep_results.csv` already on disk; the
# in-flight one won't. Re-running this script after bumping TIME will then
# automatically skip the completed sub-sweeps and only re-run the missing ones.
#
# Set FORCE_RERUN_ALL=1 to ignore the on-disk results and re-run everything
# from scratch (e.g. when changing knobs that aren't visible in the exp_name).
FORCE_RERUN_ALL="${FORCE_RERUN_ALL:-0}"

run_or_skip() {
    local exp_name="$1"; shift
    if [[ "${FORCE_RERUN_ALL}" != "1" ]] && \
       compgen -G "${RESULTS_ROOT}/${exp_name}_"*/sweep_results.csv >/dev/null 2>&1; then
        local existing
        existing=$(ls -1dt "${RESULTS_ROOT}/${exp_name}_"*/sweep_results.csv 2>/dev/null | head -1)
        echo
        echo "[single-agent] === ${exp_name} === SKIP (already complete: ${existing})"
        return 0
    fi
    echo
    echo "[single-agent] === ${exp_name} ==="
    "$@" || echo "[single-agent] WARN: ${exp_name} failed (continuing)"
}

# ---------- Generate input data ----------
echo "[single-agent] Generating input JSONLs..."
python data/generate_data.py --n 256   --user-input-size-bytes 256  --output data/smoke.jsonl
python data/generate_data.py --n 20000 --user-input-size-bytes 1024 --output data/sweep_20k.jsonl

# ---------- The sweeps ----------

# 1.1 Smoke (validates the harness against new code state)
run_or_skip 00_smoke python run_sweep.py \
    --base configs/smoke.yaml \
    --input-jsonl data/smoke.jsonl \
    --concurrency 16 \
    --output-tokens 32 \
    --git-sha "${GIT_SHA}" \
    --exp-name 00_smoke

# 1.2 Concurrency baseline at output=16K (today's typical Ultra V3 rollout body).
# Was a num_workers ablation, but gym's `num_workers > 1` mode is currently unusable:
#   - uvicorn multi-worker requires a top-level `app: FastAPI` in the module, which
#     simple_agent and our synthetic servers don't expose.
#   - gym's `run_command` sets PYTHONPATH to the server's own dir, not the parent —
#     so uvicorn can't import `<server_type>.<server_name>.app` either.
# Sub-server CPU scaling instead requires multi-instance (spawning N sub-server
# instances, each num_workers=1). That's the multi-agent sweep — see
# run_multi_agent_sweep.sh.
# Trimmed concurrency=8192 from this sub-sweep — the same point is already
# covered by 04_defect_5_ablation (concurrency=8192, semaphore=true). The
# remaining 1024/4096/16384 anchors define the curve shape (low / mid /
# predicted-onset).
run_or_skip 01_concurrency_baseline python run_sweep.py \
    --base configs/single_agent_base.yaml \
    --input-jsonl data/sweep_20k.jsonl \
    --concurrency 1024,4096,16384 \
    --output-tokens 16384 \
    --total-requests 20000 \
    --git-sha "${GIT_SHA}" \
    --exp-name 01_concurrency_baseline

# 1.3 output_tokens far-tail (16K → 512K, iso-memory zip) — the headline body-size cliff.
# All cells run with num_workers=1 (yaml default) since gym's multi-worker mode is broken.
# Trimmed the output_tokens=1M cell — past-cliff and ~10 min wall-clock-bound;
# the cliff is already visible at 524 K.
run_or_skip 02_output_tokens_far_tail python run_sweep.py \
    --base configs/single_agent_base.yaml \
    --input-jsonl data/sweep_20k.jsonl \
    --output-tokens 16384,131072,262144,524288 \
    --concurrency 8192,1024,512,256 \
    --total-requests 20000,4000,2000,1000 \
    --mode zip \
    --git-sha "${GIT_SHA}" \
    --exp-name 02_output_tokens_far_tail

# 1.4 hop-depth curve (HTTP hops per /run).
# Trimmed n_hops=64 — past-cliff and wall-clock-bound (only ~140 of 20K
# rollouts complete before early-stop fires). The cliff is already visible
# between hops=4 and hops=16.
run_or_skip 03_n_hops_curve python run_sweep.py \
    --base configs/single_agent_base.yaml \
    --input-jsonl data/sweep_20k.jsonl \
    --concurrency 8192 \
    --output-tokens 1024 \
    --n-hops 1,4,16 \
    --total-requests 20000 \
    --git-sha "${GIT_SHA}" \
    --exp-name 03_n_hops_curve

# 1.5 defect #5 ablation: RL-side dispatch semaphore on/off across two concurrencies.
run_or_skip 04_defect_5_ablation python run_sweep.py \
    --base configs/single_agent_base.yaml \
    --input-jsonl data/sweep_20k.jsonl \
    --concurrency 8192,16384 \
    --semaphore-enabled true,false \
    --output-tokens 16384 \
    --total-requests 20000 \
    --git-sha "${GIT_SHA}" \
    --exp-name 04_defect_5_ablation

# (Sub-server-count axis is the multi-agent sweep, run separately via
# run_multi_agent_sweep.sh.)

# ---------- Aggregate ----------
echo
echo "[single-agent] Aggregating results..."
python <<PYEOF
import glob, json
from pathlib import Path
import pandas as pd

results_root = Path("${RESULTS_ROOT}")
dfs = []
for csv_path in sorted(results_root.glob("*/sweep_results.csv")):
    exp_name = csv_path.parent.name
    df = pd.read_csv(csv_path)
    df.insert(0, "experiment", exp_name)
    dfs.append(df)

if dfs:
    master = pd.concat(dfs, ignore_index=True)
    master_csv = results_root / "single_agent_master.csv"
    master.to_csv(master_csv, index=False)
    print(f"\\n[single-agent] Master summary: {master_csv}")
    cols = [c for c in ["experiment","concurrency","num_workers","output_tokens","n_hops",
                        "semaphore_enabled","n_rollouts","failure_rate","retry_rate",
                        "p50_s","p99_s","max_s","stop_reason","rc"] if c in master.columns]
    print(master[cols].to_string())
else:
    print(f"[single-agent] No sweep_results.csv files found under {results_root}")
PYEOF

echo
echo "[single-agent] Done. Results under ${RESULTS_ROOT}"
