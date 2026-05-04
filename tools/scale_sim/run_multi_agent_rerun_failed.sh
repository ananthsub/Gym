#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rerun only the multi-agent cells that failed in the original sweep. Same shape
# as run_multi_agent_sweep.sh but with a smaller cell list — the 11 cells whose
# summary.json never landed because of the Ray-state-leak between cells (now
# fixed in sweep_runner._pre_cell_cleanup).
#
# Designed for batch submission via run_on_slurm_baremetal.sh:
#
#     cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
#     DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_rerun_failed.sh \
#     JOB_NAME=nemo-gym-multi-agent-rerun \
#       bash tools/scale_sim/run_on_slurm_baremetal.sh batch
#
# After it finishes, re-run analyze_multi_agent.py to merge the new cells with
# the existing successful cells into a single master CSV.

set -uo pipefail

GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${GYM_VENV}/bin/activate"
fi
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[multi-agent-rerun] ulimit -n: $(ulimit -n)"
echo "[multi-agent-rerun] hostname:  $(hostname)"
echo "[multi-agent-rerun] which ng_run: $(which ng_run || echo NOT FOUND)"

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
echo "[multi-agent-rerun] git sha:   ${GIT_SHA}"

BASE_CFG="configs/multi_agent_base.yaml"
GEN_DIR="configs/_generated"
mkdir -p "${GEN_DIR}"

# Generate input data if missing — reuses the file from the previous sweep if
# present.
if [[ ! -f data/multi_agent_20k.jsonl ]]; then
  echo "[multi-agent-rerun] Generating input JSONL..."
  python data/generate_data.py --n 20000 --user-input-size-bytes 1024 --output data/multi_agent_20k.jsonl
fi

gen_cell() {
  local n="$1" conc="$2" reqs="$3" sem="${4:-true}"
  local out="${GEN_DIR}/multi_agent_n${n}_c${conc}_r${reqs}_sem${sem}.yaml"
  python configs/_gen_multi_agent.py \
    --base "${BASE_CFG}" --n-agents "${n}" \
    --concurrency "${conc}" --total-requests "${reqs}" \
    --semaphore-enabled "${sem}" \
    --out "${out}" >&2
  echo "${out}"
}

teardown_sleep_for() {
  local n="$1" s=$((n / 8))
  if (( s < 5 )); then s=5; fi
  if (( s > 60 )); then s=60; fi
  echo "$s"
}

spinup_timeout_for() {
  local n="$1" t=$((60 + n * 5))
  if (( t > 1200 )); then t=1200; fi
  echo "$t"
}

run_cell() {
  local n="$1" conc="$2" reqs="$3" mode="$4"
  echo
  echo "[multi-agent-rerun] --- N=${n} mode=${mode} concurrency=${conc} total_requests=${reqs} ---"
  local cfg
  cfg=$(gen_cell "${n}" "${conc}" "${reqs}" true)
  local extra=()
  if [[ "${mode}" == "spinup_only" ]]; then
    extra+=(--driver-mode spinup_only --idle-window-s 30)
  else
    extra+=(--driver-mode loaded --input-jsonl "${CONTAINER_GYM_PATH}/tools/scale_sim/data/multi_agent_20k.jsonl")
  fi
  python sweep_runner.py \
    --config "${cfg}" \
    --git-sha "${GIT_SHA}" \
    --head-server-host 127.0.0.1 --head-server-port 5000 \
    --spinup-timeout-s "$(spinup_timeout_for "${n}")" \
    --teardown-sleep-s "$(teardown_sleep_for "${n}")" \
    "${extra[@]}" || \
    echo "[multi-agent-rerun] WARN: cell N=${n} mode=${mode} failed (continuing)"
}

# ---------- The 11 cells that failed in the previous sweep ----------

# Sub-sweep A spinup_only: N=16, 32, 64, 128, 256
for N in 16 32 64 128 256; do
  run_cell "${N}" 1 1 spinup_only
done

# Sub-sweep B loaded c=4096 r=10000: N=32, 64, 128, 256
for N in 32 64 128 256; do
  run_cell "${N}" 4096 10000 loaded
done

# Sub-sweep C loaded fixed-per-agent=256: N=128 (32K total), N=256 (65K total)
for N in 128 256; do
  total_conc=$((256 * N))
  total_reqs=$((4000 * N))
  if (( total_reqs > 50000 )); then total_reqs=50000; fi
  run_cell "${N}" "${total_conc}" "${total_reqs}" loaded
done

# ---------- Merge into the existing master CSV ----------
echo
echo "[multi-agent-rerun] Re-aggregating with the previously-successful cells..."
python analyze_multi_agent.py \
  "${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}" || \
  echo "[multi-agent-rerun] WARN: aggregation failed"

echo
echo "[multi-agent-rerun] Done. Combined results: ${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}/multi_agent_master_full.csv"
