#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the scale-testing experiments on a Slurm node (or directly on a
# workstation), then write summaries under tools/scale_sim/findings/<LABEL>/ for
# side-by-side comparison across hardware.
#
# --- One-time setup (do this once, in an interactive allocation) ---
# The compute partition may have no internet, and a cold `uv sync` over Lustre
# wastes job wall-clock, so build the venv ahead of time rather than inside the
# batch job:
#
#   salloc -A coreai_dlalgo_nemofw -p cpu --nodes=1 --time=1:00:00 --exclusive --mem=0
#   cd <repo-root>
#   export UV_LINK_MODE=copy RAY_TMPDIR=/tmp
#   uv venv --python 3.12 && uv sync --extra dev
#   exit
#
# --- Submit the suite (defaults below already target this cluster) ---
#
#   LABEL=slurm-cpu sbatch tools/scale_sim/run_on_slurm.sh
#
# Override any default on the command line (sbatch flags beat #SBATCH lines):
#
#   LABEL=slurm-cpu sbatch -A <acct> -p <part> -q <qos> -t 03:59:00 tools/scale_sim/run_on_slurm.sh
#
# Run a subset by setting EXPERIMENTS (default: --all):
#
#   LABEL=slurm-cpu \
#   EXPERIMENTS="--experiment concurrency_scaling --experiment response_size_scaling" \
#       sbatch tools/scale_sim/run_on_slurm.sh
#
# It also runs directly (no Slurm) from the repo root, which is how you collect
# the workstation baseline:  LABEL=workstation bash tools/scale_sim/run_on_slurm.sh

# ---- Cluster defaults (CLI sbatch flags override these) ----
#SBATCH --job-name=scale-sim
#SBATCH --account=coreai_dlalgo_nemofw
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=scale-sim-%j.out
# NOTE: the cpu_short partition/QOS caps wall-clock at 4h. Keep --time<=03:59:00
# and submit one job per experiment in parallel (see submit_slurm_suite.sh)
# rather than running the whole suite in a single job.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f pyproject.toml ] || [ ! -d nemo_gym ]; then
  echo "Run from the NeMo Gym repo root (pyproject.toml + nemo_gym/ not found here)." >&2
  exit 1
fi

SCALE_SIM_DIR="tools/scale_sim"

# Loopback-pin Ray and use /tmp for its sockets (Lustre paths blow past the
# 107-byte AF_UNIX limit). RAY_NODE_IP_ADDRESS keeps Ray off multi-NIC routing.
export RAY_TMPDIR=/tmp
export RAY_NODE_IP_ADDRESS=127.0.0.1
export UV_LINK_MODE=copy

# GCS burst-tolerance tuning so ng_run reaches "All N/N servers ready" at high
# sub-server fan-out. No-op at low fan-out. See _ray_burst_env.sh for rationale.
# shellcheck disable=SC1091
source "${SCALE_SIM_DIR}/_ray_burst_env.sh"

LABEL="${LABEL:-slurm}"
EXPERIMENTS="${EXPERIMENTS:---all}"

# --- venv: require a pre-built .venv under Slurm; build it only off-cluster ---
if [ ! -d .venv ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: .venv not found, and building it inside a batch job is unreliable" >&2
    echo "       (compute partition may lack internet; cold uv sync eats wall-clock)." >&2
    echo "       Build it once in an interactive allocation, then resubmit:" >&2
    echo "         salloc -A coreai_dlalgo_nemofw -p cpu --nodes=1 --time=1:00:00 --exclusive --mem=0" >&2
    echo "         cd $(pwd) && export UV_LINK_MODE=copy RAY_TMPDIR=/tmp" >&2
    echo "         uv venv --python 3.12 && uv sync --extra dev" >&2
    exit 1
  fi
  echo "[scale-sim] no .venv found (off-cluster run) — building it now..."
  uv venv --python 3.12
  uv sync --extra dev
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Raise the soft FD limit to the hard limit so high-concurrency cells (up to
# 131072 concurrent rollouts open tens of thousands of sockets) are not capped
# by the default 1024. `ulimit -Sn unlimited` is invalid in bash, so when the
# hard limit is "unlimited" fall back to a large concrete value.
_HARD_FD="$(ulimit -Hn)"
if [ "${_HARD_FD}" = "unlimited" ]; then
  ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
else
  ulimit -Sn "${_HARD_FD}" 2>/dev/null || true
fi
echo "[scale-sim] ulimit -Sn=$(ulimit -Sn) -Hn=${_HARD_FD}"
echo "[scale-sim] host=$(hostname) cores=$(nproc) label=${LABEL}"
echo "[scale-sim] RAY burst tuning: gcs_rpc_threads=${RAY_gcs_server_rpc_server_thread_num} connect_timeout_s=${RAY_gcs_rpc_server_connect_timeout_s}"

# SMOKE=1 runs the fast end-to-end sanity check (smoke.yaml, 256 rollouts)
# through this exact env path instead of the full experiment suite. Use it to
# validate the harness comes up before committing to the long runs.
if [ "${SMOKE:-0}" = "1" ]; then
  echo "[scale-sim] SMOKE: generating smoke data + running smoke sweep"
  [ -f "${SCALE_SIM_DIR}/data/smoke.jsonl" ] || \
    python "${SCALE_SIM_DIR}/data/generate_data.py" \
      --n 256 --user-input-size-bytes 256 --output "${SCALE_SIM_DIR}/data/smoke.jsonl"
  python "${SCALE_SIM_DIR}/sweep_runner.py" \
    --config "${SCALE_SIM_DIR}/configs/smoke.yaml" \
    --input-jsonl "${SCALE_SIM_DIR}/data/smoke.jsonl" \
    --head-server-host 127.0.0.1 \
    --git-sha "$(git rev-parse --short HEAD 2>/dev/null || echo local)"
  echo "[scale-sim] SMOKE done. See ${SCALE_SIM_DIR}/results/"
  exit 0
fi

echo "[scale-sim] Running experiments (${EXPERIMENTS})"
python "${SCALE_SIM_DIR}/run_experiments.py" ${EXPERIMENTS} --label "${LABEL}"
echo "[scale-sim] Done. Summaries under ${SCALE_SIM_DIR}/findings/${LABEL}/"
