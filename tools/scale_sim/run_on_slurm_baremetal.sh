#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bare-metal (no container, no pyxis) Slurm launcher for the scale-sim harness.
#
# Companion to run_on_slurm.sh. That one assumes a pyxis container with a
# pre-baked /opt/ray_venvs/.../NemoGym venv. This one assumes:
#   - You already created `<repo>/.venv` via `uv venv && uv sync --extra dev`
#     in an interactive cpu allocation (one-time setup).
#   - The compute node can see `<repo>/.venv` over Lustre.
#   - The CPU partition on this cluster does not require --qos and does not
#     require --gres=gpu (we just want CPUs + RAM).
#
# Usage (interactive shell on a compute node):
#
#     bash tools/scale_sim/run_on_slurm_baremetal.sh interactive
#
# Usage (batch — runs the full single-agent sweep matrix overnight):
#
#     DRIVER_SCRIPT=tools/scale_sim/run_single_agent_sweep.sh \
#     JOB_NAME=scale-single-agent TIME=4:00:00 \
#       bash tools/scale_sim/run_on_slurm_baremetal.sh batch
#
# Usage (batch — single config via sweep_runner.py):
#
#     CONFIG=tools/scale_sim/configs/smoke.yaml \
#       bash tools/scale_sim/run_on_slurm_baremetal.sh batch

set -euo pipefail

MODE="${1:-batch}"

# ---------- Cluster defaults ----------
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemofw}"
PARTITION="${PARTITION:-cpu}"
# This cluster's cpu partition does not require a qos. Leave SLURM_QOS empty unless overridden.
SLURM_QOS="${SLURM_QOS:-}"
TIME="${TIME:-4:00:00}"
JOB_NAME="${JOB_NAME:-scale-sim}"
# 40-48 cores per node on this cluster; ask for all of them.
CPUS_PER_TASK="${CPUS_PER_TASK:-40}"
NODES="${NODES:-1}"

# ---------- Paths ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
GYM_DIR="${GYM_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GYM_VENV="${GYM_VENV:-${GYM_DIR}/.venv}"

CONFIG="${CONFIG:-tools/scale_sim/configs/smoke.yaml}"
INPUT_JSONL="${INPUT_JSONL:-tools/scale_sim/data/smoke.jsonl}"

# ---------- Sanity checks ----------
if [[ ! -d "${GYM_VENV}" ]]; then
  echo "[scale-sim] ERROR: ${GYM_VENV} not found."
  echo "[scale-sim]   Create it once on a compute node:"
  echo "[scale-sim]     salloc -A ${ACCOUNT} -p ${PARTITION} --nodes=1 --time=1:00:00 --exclusive --mem=0"
  echo "[scale-sim]     cd ${GYM_DIR}"
  echo "[scale-sim]     export UV_LINK_MODE=copy RAY_TMPDIR=/tmp"
  echo "[scale-sim]     uv venv --python 3.12 && uv sync --extra dev"
  exit 1
fi

# Bump submit-shell FDs; rlimits propagate through srun.
_HARD_FD=$(ulimit -Hn)
if [[ "${_HARD_FD}" == "unlimited" ]]; then
  ulimit -Sn "${TARGET_FD:-1048576}" 2>/dev/null || true
else
  ulimit -Sn "${TARGET_FD:-${_HARD_FD}}" 2>/dev/null || true
fi
echo "[scale-sim] submit-shell ulimit -Sn=$(ulimit -Sn) -Hn=${_HARD_FD}"

# Source the comprehensive Ray-GCS burst-tolerance tuning. This is the
# A12 workaround — see investigations/nemo-gym-scale-testing.md §10.3.3.
# Sourcing into the submit shell exports the RAY_* vars; sbatch's --export=ALL
# below carries them into the compute-node environment.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_ray_burst_env.sh"

# Other env vars every cell needs (non-Ray-burst).
COMMON_ENV_EXPORTS=(
  "RAY_TMPDIR=/tmp"
  "RAY_NODE_IP_ADDRESS=127.0.0.1"
  "UV_LINK_MODE=copy"
)

# Build the comma-separated --export string for srun/sbatch.
_EXPORT_STR="ALL"
for kv in "${COMMON_ENV_EXPORTS[@]}"; do _EXPORT_STR+=",${kv}"; done

echo "[scale-sim] ACCOUNT=${ACCOUNT} PARTITION=${PARTITION} QOS=${SLURM_QOS:-<none>} TIME=${TIME}"
echo "[scale-sim] GYM_DIR=${GYM_DIR}"
echo "[scale-sim] GYM_VENV=${GYM_VENV}"

# Generate smoke data on submit host if missing — small enough to do unconditionally.
if [[ "$INPUT_JSONL" == "tools/scale_sim/data/smoke.jsonl" ]] && [[ ! -f "${GYM_DIR}/${INPUT_JSONL}" ]]; then
  echo "[scale-sim] Generating ${INPUT_JSONL} on submit host"
  mkdir -p "$(dirname "${GYM_DIR}/${INPUT_JSONL}")"
  "${GYM_VENV}/bin/python" "${GYM_DIR}/tools/scale_sim/data/generate_data.py" \
    --n 256 --user-input-size-bytes 256 \
    --output "${GYM_DIR}/${INPUT_JSONL}"
fi

# Build the sbatch/srun option list, only emitting --qos when set.
_slurm_opts=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${TIME}"
  --job-name="${JOB_NAME}"
  --nodes="${NODES}"
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --exclusive
  --mem=0
)
[[ -n "${SLURM_QOS}" ]] && _slurm_opts+=(--qos="${SLURM_QOS}")

case "$MODE" in
  interactive)
    echo "[scale-sim] Launching interactive shell on a ${PARTITION} node…"
    echo "[scale-sim] Once attached, run:"
    echo "[scale-sim]   source ${GYM_VENV}/bin/activate"
    echo "[scale-sim]   cd ${GYM_DIR}/tools/scale_sim"
    echo "[scale-sim]   ulimit -n   # should be \$(ulimit -Sn) ≈ 1M"
    echo "[scale-sim]   python data/generate_data.py --n 256 --user-input-size-bytes 256 --output data/smoke.jsonl"
    echo "[scale-sim]   python sweep_runner.py --config configs/smoke.yaml --input-jsonl data/smoke.jsonl --head-server-host 127.0.0.1"
    echo
    exec srun \
      "${_slurm_opts[@]}" \
      --export="${_EXPORT_STR}" \
      --pty bash
    ;;

  batch)
    SUBMIT_DIR="$(pwd)"
    LOG_DIR="${SUBMIT_DIR}/scale-sim-logs/${JOB_NAME}-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    echo "[scale-sim] Submitting batch job, logs → ${LOG_DIR}"

    DRIVER_FILE="$LOG_DIR/driver.sh"
    if [[ -n "${DRIVER_SCRIPT:-}" ]]; then
      _abs_driver="${GYM_DIR}/${DRIVER_SCRIPT}"
      if [[ ! -f "${_abs_driver}" ]]; then
        echo "[scale-sim] ERROR: DRIVER_SCRIPT not found at ${_abs_driver}"
        exit 1
      fi
      # Export GYM_VENV and CONTAINER_GYM_PATH so container-defaulted scripts
      # like run_single_agent_sweep.sh point at the bare-metal venv + repo instead of
      # /opt/ray_venvs/... and /opt/nemo-rl/.... Both names are unfortunate on
      # bare metal, but they're what the existing scripts expect.
      cat > "$DRIVER_FILE" <<DRIVER_EOF
#!/bin/bash
# Wraps a custom driver script (e.g. run_single_agent_sweep.sh).
set -uo pipefail
export GYM_VENV=${GYM_VENV}
export CONTAINER_GYM_PATH=${GYM_DIR}
export RAY_TMPDIR=/tmp
export RAY_NODE_IP_ADDRESS=127.0.0.1
export UV_LINK_MODE=copy
# Re-source the Ray burst-tolerance tuning inside the compute-node shell.
# srun --export=ALL should already carry these from the submit shell, but
# Slurm's env propagation has cluster-specific edge cases — re-source to be
# safe. The A12 workaround per investigations/nemo-gym-scale-testing.md §10.3.3.
# shellcheck disable=SC1091
[[ -f ${GYM_DIR}/tools/scale_sim/_ray_burst_env.sh ]] && source ${GYM_DIR}/tools/scale_sim/_ray_burst_env.sh
source ${GYM_VENV}/bin/activate
_HARD=\$(ulimit -Hn); [[ "\$_HARD" == "unlimited" ]] && _HARD=1048576
ulimit -Sn "\$_HARD" 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[driver-wrap] GYM_VENV=\$GYM_VENV"
echo "[driver-wrap] CONTAINER_GYM_PATH=\$CONTAINER_GYM_PATH (bare-metal: same as GYM_DIR)"
echo "[driver-wrap] ulimit -n: \$(ulimit -n)"
echo "[driver-wrap] which ng_run: \$(which ng_run || echo NOT FOUND)"
echo "[driver-wrap] RAY_gcs_rpc_server_connect_timeout_s=\${RAY_gcs_rpc_server_connect_timeout_s:-<unset>}"
echo "[driver-wrap] RAY_gcs_server_rpc_server_thread_num=\${RAY_gcs_server_rpc_server_thread_num:-<unset>}"
exec bash ${_abs_driver}
DRIVER_EOF
      echo "[scale-sim] Using custom driver: ${DRIVER_SCRIPT}"
    else
      cat > "$DRIVER_FILE" <<DRIVER_EOF
#!/bin/bash
set -euo pipefail
source ${GYM_VENV}/bin/activate
cd ${GYM_DIR}/tools/scale_sim

_HARD=\$(ulimit -Hn); [[ "\$_HARD" == "unlimited" ]] && _HARD=1048576
ulimit -Sn "\$_HARD" 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[scale-sim] ulimit -n: \$(ulimit -n)"
echo "[scale-sim] hostname:  \$(hostname)"
echo "[scale-sim] which ng_run: \$(which ng_run || echo NOT FOUND)"
which ng_run >/dev/null 2>&1 || { echo "ng_run missing — re-run uv sync in ${GYM_DIR}"; exit 1; }

python sweep_runner.py \\
  --config $(printf "${GYM_DIR}/%s " ${CONFIG}) \\
  --input-jsonl ${GYM_DIR}/${INPUT_JSONL} \\
  --head-server-host 127.0.0.1 \\
  --git-sha "\$(cd ${GYM_DIR} && git rev-parse --short HEAD 2>/dev/null || echo local)"
DRIVER_EOF
    fi
    chmod +x "$DRIVER_FILE"

    sbatch_script="$LOG_DIR/sbatch.sh"
    {
      echo "#!/bin/bash"
      echo "#SBATCH --account=${ACCOUNT}"
      echo "#SBATCH --partition=${PARTITION}"
      echo "#SBATCH --time=${TIME}"
      echo "#SBATCH --job-name=${JOB_NAME}"
      echo "#SBATCH --nodes=${NODES}"
      echo "#SBATCH --ntasks=1"
      echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
      echo "#SBATCH --exclusive"
      echo "#SBATCH --mem=0"
      echo "#SBATCH --output=${LOG_DIR}/sbatch.out"
      echo "#SBATCH --error=${LOG_DIR}/sbatch.err"
      [[ -n "${SLURM_QOS}" ]] && echo "#SBATCH --qos=${SLURM_QOS}"
      echo
      echo "set -euo pipefail"
      echo "export RAY_TMPDIR=/tmp"
      echo "export RAY_NODE_IP_ADDRESS=127.0.0.1"
      echo "export UV_LINK_MODE=copy"
      echo
      echo "srun --export=${_EXPORT_STR} bash ${DRIVER_FILE}"
    } > "$sbatch_script"
    chmod +x "$sbatch_script"

    echo "[scale-sim] sbatch script: $sbatch_script"
    echo "[scale-sim] driver script: $DRIVER_FILE"
    sbatch "$sbatch_script"
    echo "[scale-sim] Tail logs: tail -f ${LOG_DIR}/sbatch.out"
    ;;

  *)
    echo "Unknown mode: $MODE. Use 'interactive' or 'batch'."
    exit 1
    ;;
esac
