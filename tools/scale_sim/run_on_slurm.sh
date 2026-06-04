#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the scale-testing experiments on a Slurm node, then write summaries under
# tools/scale_sim/findings/<LABEL>/ for comparison with other hardware.
#
# Submit from the repo root, supplying your account and partition:
#
#   LABEL=slurm-cpu sbatch -A <account> -p <cpu-partition> -t 04:00:00 \
#       tools/scale_sim/run_on_slurm.sh
#
# Run a subset by setting EXPERIMENTS (default: --all), e.g.:
#
#   LABEL=slurm-cpu EXPERIMENTS="--experiment concurrency_scaling --experiment response_size_scaling" \
#       sbatch -A <account> -p <cpu-partition> tools/scale_sim/run_on_slurm.sh
#
# It can also be run directly (no Slurm) from the repo root: bash tools/scale_sim/run_on_slurm.sh

#SBATCH --job-name=scale-sim
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=scale-sim-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f pyproject.toml ] || [ ! -d nemo_gym ]; then
  echo "Run from the NeMo Gym repo root (pyproject.toml + nemo_gym/ not found here)." >&2
  exit 1
fi

export RAY_TMPDIR=/tmp UV_LINK_MODE=copy
LABEL="${LABEL:-slurm}"
EXPERIMENTS="${EXPERIMENTS:---all}"

if [ ! -d .venv ]; then
  uv venv --python 3.12
  uv sync --extra dev
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true

echo "Running experiments on label=${LABEL} (${EXPERIMENTS})"
python tools/scale_sim/run_experiments.py ${EXPERIMENTS} --label "${LABEL}"
echo "Done. Summaries under tools/scale_sim/findings/${LABEL}/"
