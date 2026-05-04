#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs ONE single-agent sub-sweep (selected via SINGLE_AGENT_SUBSWEEP env var).
# Designed to let you submit each sub-sweep as its own slurm job so the five
# of them run in parallel — end-to-end wall-clock becomes max(per-job) instead
# of sum(per-job).
#
# All sub-sweeps share the same `results/<git-sha>/` root, so once all jobs
# finish you can re-aggregate them with run_single_agent_sweep.sh's tail block,
# or just `python -c` over the per-sub-sweep sweep_results.csv files.
#
# Usage (one job per sub-sweep, all in parallel):
#
#     for SUB in 00_smoke 01_concurrency_baseline 02_output_tokens_far_tail \
#                03_n_hops_curve 04_defect_5_ablation; do
#       SINGLE_AGENT_SUBSWEEP=$SUB \
#       DRIVER_SCRIPT=tools/scale_sim/run_single_agent_subsweep.sh \
#       JOB_NAME=scale-single-agent-$SUB TIME=2:00:00 \
#         bash tools/scale_sim/run_on_slurm_baremetal.sh batch
#     done
#
# Each sub-sweep's predicted wall-clock (post-trim, with early_stop_wall_clock_s=300):
#   00_smoke                  ~1 min   (1 cell)
#   01_concurrency_baseline   ~25 min  (3 cells, 16K hits wall-clock)
#   02_output_tokens_far_tail ~30 min  (4 cells)
#   03_n_hops_curve           ~15 min  (3 cells, 16-hops hits wall-clock)
#   04_defect_5_ablation      ~25 min  (4 cells, 16K cells hit wall-clock)
# So end-to-end with parallel jobs ≈ 30 min, vs ~3-4h sequential.

set -uo pipefail

# ---------- Inside-container/venv setup (mirrors run_single_agent_sweep.sh) ----------
GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

source "${GYM_VENV}/bin/activate"
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[single-agent] subsweep=${SINGLE_AGENT_SUBSWEEP:-<unset>}"
echo "[single-agent] ulimit -n: $(ulimit -n)"
echo "[single-agent] hostname:  $(hostname)"
echo "[single-agent] gym path:  ${CONTAINER_GYM_PATH}"
echo "[single-agent] which ng_run: $(which ng_run || echo NOT FOUND)"

# GIT_SHA env override lets multiple parallel jobs share the same results dir
# even if you've committed a fix between submissions.
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"
echo "[single-agent] git sha:   ${GIT_SHA}"

RESULTS_ROOT="${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}"
mkdir -p "${RESULTS_ROOT}"

if [[ -z "${SINGLE_AGENT_SUBSWEEP:-}" ]]; then
    echo "[single-agent] ERROR: SINGLE_AGENT_SUBSWEEP env var is required."
    echo "[single-agent] Valid values: 00_smoke, 01_concurrency_baseline, 02_output_tokens_far_tail, 03_n_hops_curve, 04_defect_5_ablation"
    exit 2
fi

# ---------- Idempotent skip ----------
# If this sub-sweep already has a sweep_results.csv on disk under
# RESULTS_ROOT, skip — same semantics as run_single_agent_sweep.sh's
# run_or_skip helper.
if compgen -G "${RESULTS_ROOT}/${SINGLE_AGENT_SUBSWEEP}_"*/sweep_results.csv >/dev/null 2>&1; then
    existing=$(ls -1dt "${RESULTS_ROOT}/${SINGLE_AGENT_SUBSWEEP}_"*/sweep_results.csv 2>/dev/null | head -1)
    echo "[single-agent] === ${SINGLE_AGENT_SUBSWEEP} === SKIP (already complete: ${existing})"
    exit 0
fi

# ---------- Generate input data ----------
# Each parallel job needs the input JSONL on disk. generate_data is idempotent
# w.r.t. (n, seed) so concurrent generation is safe — different jobs may write
# the same bytes to the same file but the last writer wins on identical input.
echo "[single-agent] Generating input JSONL for ${SINGLE_AGENT_SUBSWEEP}..."
case "${SINGLE_AGENT_SUBSWEEP}" in
    00_smoke)
        python data/generate_data.py --n 256 --user-input-size-bytes 256 --output data/smoke.jsonl
        ;;
    *)
        python data/generate_data.py --n 20000 --user-input-size-bytes 1024 --output data/sweep_20k.jsonl
        ;;
esac

# ---------- Dispatch ----------
echo
echo "[single-agent] === ${SINGLE_AGENT_SUBSWEEP} ==="

case "${SINGLE_AGENT_SUBSWEEP}" in
    00_smoke)
        python run_sweep.py \
            --base configs/smoke.yaml \
            --input-jsonl data/smoke.jsonl \
            --concurrency 16 \
            --output-tokens 32 \
            --git-sha "${GIT_SHA}" \
            --exp-name 00_smoke
        ;;
    01_concurrency_baseline)
        python run_sweep.py \
            --base configs/single_agent_base.yaml \
            --input-jsonl data/sweep_20k.jsonl \
            --concurrency 1024,4096,16384 \
            --output-tokens 16384 \
            --total-requests 20000 \
            --git-sha "${GIT_SHA}" \
            --exp-name 01_concurrency_baseline
        ;;
    02_output_tokens_far_tail)
        python run_sweep.py \
            --base configs/single_agent_base.yaml \
            --input-jsonl data/sweep_20k.jsonl \
            --output-tokens 16384,131072,262144,524288 \
            --concurrency 8192,1024,512,256 \
            --total-requests 20000,4000,2000,1000 \
            --mode zip \
            --git-sha "${GIT_SHA}" \
            --exp-name 02_output_tokens_far_tail
        ;;
    03_n_hops_curve)
        python run_sweep.py \
            --base configs/single_agent_base.yaml \
            --input-jsonl data/sweep_20k.jsonl \
            --concurrency 8192 \
            --output-tokens 1024 \
            --n-hops 1,4,16 \
            --total-requests 20000 \
            --git-sha "${GIT_SHA}" \
            --exp-name 03_n_hops_curve
        ;;
    04_defect_5_ablation)
        python run_sweep.py \
            --base configs/single_agent_base.yaml \
            --input-jsonl data/sweep_20k.jsonl \
            --concurrency 8192,16384 \
            --semaphore-enabled true,false \
            --output-tokens 16384 \
            --total-requests 20000 \
            --git-sha "${GIT_SHA}" \
            --exp-name 04_defect_5_ablation
        ;;
    *)
        echo "[single-agent] ERROR: unknown SINGLE_AGENT_SUBSWEEP='${SINGLE_AGENT_SUBSWEEP}'"
        echo "[single-agent] Valid values: 00_smoke, 01_concurrency_baseline, 02_output_tokens_far_tail, 03_n_hops_curve, 04_defect_5_ablation"
        exit 2
        ;;
esac

rc=$?
echo
echo "[single-agent] === ${SINGLE_AGENT_SUBSWEEP} === finished with rc=${rc}"
echo "[single-agent] Results under ${RESULTS_ROOT}"
echo "[single-agent] To re-aggregate after all parallel jobs land, see run_single_agent_sweep.sh's tail block (or rerun it — its run_or_skip helper makes it a no-op for completed sub-sweeps and runs only the aggregator)."
exit "${rc}"
