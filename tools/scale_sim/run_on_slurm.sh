#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node Slurm launcher for the scale-sim harness.
#
# Bind-mounts your local nemo-gym checkout over the container's submodule path
# (`/opt/nemo-rl/3rdparty/Gym-workspace/Gym`) so changes under `tools/scale_sim/`
# are picked up without rebuilding the container. Same convention as
# `launch_ultra_pipeclean.sh`'s `NRL_GYM_DIR` overlay.
#
# This script does NOT use `ray.sub`. Scale-sim is single-node, and `ng_run` brings
# up its own local Ray cluster. We just need plain `srun --container-image`.
#
# Usage (interactive — opens a shell on a compute node, you run ng_run + load_driver
# inside the container yourself):
#
#     bash tools/scale_sim/run_on_slurm.sh interactive
#
# Usage (batch — submits an sbatch that runs the smoke config end-to-end):
#
#     CONFIG=tools/scale_sim/configs/smoke.yaml \
#     bash tools/scale_sim/run_on_slurm.sh batch
#
# Usage (batch — multi-cell sweep via sweep_runner.py):
#
#     CONFIG="tools/scale_sim/configs/axis_a_8k.yaml tools/scale_sim/configs/axis_c_8k_4mb.yaml" \
#     bash tools/scale_sim/run_on_slurm.sh batch
#
# Required env vars (defaults wired for the llmservice_nemotron_ultra cluster):
#
#     CONTAINER       — pyxis container image (.sqsh path).
#                       Default: rl.nightly.sqsh under llmservice_nemotron_ultra/nemo_rl/images.
#     ACCOUNT         — Slurm account.   Default: llmservice_nemotron_ultra
#     PARTITION       — Slurm partition. Default: batch
#     SLURM_QOS       — Slurm QoS.       Default: interactive  (capped at 2h walltime)
#     NRL_GYM_DIR     — Absolute path to your local nemo-gym checkout (the one with
#                       tools/scale_sim/ in it). Bind-mounted over the container's
#                       /opt/nemo-rl/3rdparty/Gym-workspace/Gym. Default: this script's repo root.
#
# Optional env vars:
#
#     CONFIG          — One or more sweep config paths (relative to NRL_GYM_DIR)
#                       for batch mode. Default: tools/scale_sim/configs/smoke.yaml.
#     INPUT_JSONL     — Pre-generated input data (relative to NRL_GYM_DIR).
#                       If left at the default and the file is missing, smoke data is
#                       generated on the fly using the host python3.
#     TIME            — Slurm walltime. Default 1:00:00 (must be <= 2h with QOS=interactive).
#     GPUS_PER_NODE   — GPUs to request per node. Default 4. The simulation is CPU-only,
#                       but the cluster requires gres on GPU partitions; just claim them.
#     CPUS_PER_TASK   — CPUs per task. Default 144 (matches GB200 NVL72).
#     EXTRA_MOUNTS    — Comma-separated additional --container-mounts entries.

set -euo pipefail

MODE="${1:-batch}"

# ---------- Defaults wired for the llmservice_nemotron_ultra cluster ----------
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.nightly.sqsh}"
ACCOUNT="${ACCOUNT:-llmservice_nemotron_ultra}"
# CPU-only by default. Override PARTITION=batch if you want a GPU node.
PARTITION="${PARTITION:-cpu}"
# Default qos and CPUs adapt to partition. Override either explicitly.
#   cpu     — AmpereOne, 96 CPUs/node; cpu-normal qos
#   batch   — GB200 NVL72, 144 CPUs/node (Slurm-default qos)
case "${PARTITION}" in
  cpu)
    SLURM_QOS="${SLURM_QOS:-cpu-normal}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-96}"
    ;;
  batch)
    SLURM_QOS="${SLURM_QOS:-}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-144}"
    ;;
  *)
    SLURM_QOS="${SLURM_QOS:-}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-}"
    ;;
esac
TIME="${TIME:-4:00:00}"
JOB_NAME="${JOB_NAME:-scale-sim}"
# Set GPUS_PER_NODE>0 only on GPU partitions.
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"

# Default NRL_GYM_DIR to the repo root inferred from this script's location:
# tools/scale_sim/run_on_slurm.sh → walk up two parents.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
NRL_GYM_DIR="${NRL_GYM_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

CONFIG="${CONFIG:-tools/scale_sim/configs/smoke.yaml}"
INPUT_JSONL="${INPUT_JSONL:-tools/scale_sim/data/smoke.jsonl}"

# Inside-container path where gym lives. Matches launch_ultra_pipeclean.sh's NRL_GYM_DIR overlay target.
CONTAINER_GYM_PATH="/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# Bind-mount: user's nemo-gym checkout overrides the container's submodule.
# Lustre is mounted as the production scripts do so /lustre/... paths inside the container resolve.
MOUNTS="/lustre:/lustre,${NRL_GYM_DIR}:${CONTAINER_GYM_PATH}"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

# Avoid the 107-byte AF_UNIX socket path limit on Lustre (Ray gotcha).
# Force Ray GCS to bind localhost so child processes inside the same container
# can reach it (the eth0 IP is sometimes unreachable from in-container peers).
COMMON_ENV="RAY_TMPDIR=/tmp,RAY_NODE_IP_ADDRESS=127.0.0.1"

# Set submit-shell soft FD limit to whatever the cluster's hard limit allows.
# rlimits inherit through srun -> pyxis -> container, so setting here propagates inside.
_HARD_FD=$(ulimit -Hn)
if [[ "${_HARD_FD}" == "unlimited" ]]; then
  ulimit -Sn "${TARGET_FD:-1048576}" 2>/dev/null || true
else
  ulimit -Sn "${TARGET_FD:-${_HARD_FD}}" 2>/dev/null || true
fi
echo "[scale-sim] submit-shell ulimit -Sn=$(ulimit -Sn) -Hn=${_HARD_FD}"

# Per-actor venv inside the container that has the full gym CLI (`ng_run`, etc.).
# Same venv production's NemoGym Ray actor uses, so no behavioural drift.
GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"

echo "[scale-sim] CONTAINER=${CONTAINER}"
echo "[scale-sim] ACCOUNT=${ACCOUNT} PARTITION=${PARTITION} QOS=${SLURM_QOS} TIME=${TIME}"
echo "[scale-sim] NRL_GYM_DIR=${NRL_GYM_DIR}"
echo "[scale-sim]   → mounts as ${CONTAINER_GYM_PATH} inside container"
echo "[scale-sim] mounts=${MOUNTS}"

# Generate smoke data on the host if missing — small enough to do unconditionally.
if [[ "$INPUT_JSONL" == "tools/scale_sim/data/smoke.jsonl" ]] && [[ ! -f "${NRL_GYM_DIR}/${INPUT_JSONL}" ]]; then
  echo "[scale-sim] Generating ${INPUT_JSONL} on submit host"
  mkdir -p "$(dirname "${NRL_GYM_DIR}/${INPUT_JSONL}")"
  python3 "${NRL_GYM_DIR}/tools/scale_sim/data/generate_data.py" \
    --n 256 --user-input-size-bytes 256 \
    --output "${NRL_GYM_DIR}/${INPUT_JSONL}"
fi

case "$MODE" in
  interactive)
    echo "[scale-sim] Launching interactive container on 1 node…"
    echo "[scale-sim] Once attached, activate the gym venv first:"
    echo "[scale-sim]   source ${GYM_VENV}/bin/activate"
    echo "[scale-sim]   cd tools/scale_sim   # already in ${CONTAINER_GYM_PATH}"
    echo "[scale-sim]   ulimit -n   # verify pyxis preserved the soft FD limit (should be $(ulimit -Sn))"
    echo "[scale-sim]   ng_run \"+config_paths=[configs/smoke.yaml]\"   # terminal A"
    echo "[scale-sim]   python load_driver.py --config configs/smoke.yaml --input-jsonl data/smoke.jsonl   # terminal B"
    echo "[scale-sim] To open a second shell into the same allocation:"
    echo "[scale-sim]   srun --jobid=\$SLURM_JOBID --overlap --container-name=scale-sim --container-workdir=${CONTAINER_GYM_PATH} --pty bash"
    echo "[scale-sim]   (then re-run \`source ${GYM_VENV}/bin/activate\` in that shell)"
    echo
    _opt_args=()
    if (( ${GPUS_PER_NODE} > 0 )); then
      _opt_args+=(--gres="gpu:${GPUS_PER_NODE}")
    fi
    if [[ -n "${CPUS_PER_TASK}" ]]; then
      _opt_args+=(--cpus-per-task="${CPUS_PER_TASK}")
    fi
    if [[ -n "${SLURM_QOS}" ]]; then
      _opt_args+=(--qos="${SLURM_QOS}")
    fi
    exec srun \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --time="$TIME" \
      --job-name="$JOB_NAME" \
      --nodes=1 \
      --ntasks=1 \
      "${_opt_args[@]}" \
      --exclusive \
      --mem=0 \
      --container-image="$CONTAINER" \
      --container-mounts="$MOUNTS" \
      --container-workdir="$CONTAINER_GYM_PATH" \
      --container-name=scale-sim \
      --no-container-mount-home \
      --export="ALL,${COMMON_ENV}" \
      --pty bash
    ;;

  batch)
    SUBMIT_DIR="$(pwd)"
    LOG_DIR="${SUBMIT_DIR}/scale-sim-logs/${JOB_NAME}-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    echo "[scale-sim] Submitting batch job, logs → ${LOG_DIR}"

    # Two ways to pick the driver:
    #   (a) DRIVER_SCRIPT=<path-relative-to-NRL_GYM_DIR> — use a pre-existing script.
    #       e.g. DRIVER_SCRIPT=tools/scale_sim/run_all_m1_sweeps.sh
    #   (b) Default: auto-generate a single sweep_runner.py invocation from CONFIG/INPUT_JSONL.
    DRIVER_FILE="$LOG_DIR/driver.sh"
    if [[ -n "${DRIVER_SCRIPT:-}" ]]; then
      _host_script_path="${NRL_GYM_DIR}/${DRIVER_SCRIPT}"
      if [[ ! -f "${_host_script_path}" ]]; then
        echo "[scale-sim] ERROR: DRIVER_SCRIPT not found at ${_host_script_path}"
        exit 1
      fi
      _container_script_path="${CONTAINER_GYM_PATH}/${DRIVER_SCRIPT}"
      cat > "$DRIVER_FILE" <<DRIVER_EOF
#!/bin/bash
# Wraps a custom driver script. The script is responsible for venv activation,
# cd, ulimit, and whatever sweep matrix it wants to run.
set -uo pipefail
exec bash ${_container_script_path}
DRIVER_EOF
      echo "[scale-sim] Using custom driver script: ${DRIVER_SCRIPT}"
    else
      # Default: single sweep_runner.py invocation from CONFIG + INPUT_JSONL.
      cat > "$DRIVER_FILE" <<DRIVER_EOF
#!/bin/bash
set -euo pipefail
source ${GYM_VENV}/bin/activate
cd ${CONTAINER_GYM_PATH}/tools/scale_sim
# Defensive bump in case pyxis dropped the rlimit on the way in. Use whatever the hard limit allows.
_HARD=$(ulimit -Hn); [[ "\$_HARD" == "unlimited" ]] && _HARD=1048576
ulimit -Sn "\$_HARD" 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[scale-sim] ulimit -n: \$(ulimit -n)"
echo "[scale-sim] hostname:  \$(hostname)"
echo "[scale-sim] gym path:  ${CONTAINER_GYM_PATH}"
echo "[scale-sim] gym venv:  ${GYM_VENV}"
echo "[scale-sim] which ng_run: \$(which ng_run || echo NOT FOUND)"
which ng_run >/dev/null 2>&1 || { echo "ng_run not on PATH after activating ${GYM_VENV} — set GYM_VENV to a venv that has gym installed"; exit 1; }

# Run the sweep. CONFIG may be a single path or a space-separated list (relative to gym root).
python sweep_runner.py \\
  --config $(printf "${CONTAINER_GYM_PATH}/%s " ${CONFIG}) \\
  --input-jsonl ${CONTAINER_GYM_PATH}/${INPUT_JSONL} \\
  --git-sha "\$(cd ${CONTAINER_GYM_PATH} && git rev-parse --short HEAD 2>/dev/null || echo local)"
DRIVER_EOF
    fi
    chmod +x "$DRIVER_FILE"

    sbatch_script="$LOG_DIR/sbatch.sh"
    cat > "$sbatch_script" <<SBATCH_EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
$( [[ -n "${SLURM_QOS}" ]] && echo "#SBATCH --qos=${SLURM_QOS}" )
$( (( ${GPUS_PER_NODE} > 0 )) && echo "#SBATCH --gres=gpu:${GPUS_PER_NODE}" )
$( [[ -n "${CPUS_PER_TASK}" ]] && echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" )
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/sbatch.out
#SBATCH --error=${LOG_DIR}/sbatch.err

set -euo pipefail

srun \\
  --container-image=${CONTAINER} \\
  --container-mounts=${MOUNTS},${LOG_DIR}:${LOG_DIR} \\
  --container-workdir=${CONTAINER_GYM_PATH} \\
  --container-name=scale-sim \\
  --no-container-mount-home \\
  --export=ALL,${COMMON_ENV} \\
  bash ${DRIVER_FILE}
SBATCH_EOF
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
