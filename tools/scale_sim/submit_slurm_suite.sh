#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit the scale-sim experiment suite as one Slurm job PER experiment, in
# parallel. The cpu_short partition/QOS caps wall-clock at 4h, so the full suite
# is split one-per-experiment: every job stays under the cap and end-to-end
# wall-clock becomes max(job) instead of sum(job).
# Each job runs on its own --exclusive node, so there are no port/Ray collisions
# between jobs. All jobs share LABEL=slurm-cpu, so their per-experiment CSVs land
# together in tools/scale_sim/findings/slurm-cpu/ for the report.
#
# Prereq: build .venv once (see run_on_slurm.sh header), then from the repo root:
#
#     bash tools/scale_sim/submit_slurm_suite.sh                 # all experiments
#     bash tools/scale_sim/submit_slurm_suite.sh concurrency_scaling agent_fan_out
#
# Override defaults via env: LABEL, TIME, ACCOUNT, PARTITION, QOS.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d nemo_gym ]; then
  echo "Run from the NeMo Gym repo root." >&2
  exit 1
fi

LABEL="${LABEL:-slurm-cpu}"
TIME="${TIME:-03:59:00}"            # cpu_short partition caps wall-clock at 4h.
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemofw}"
PARTITION="${PARTITION:-cpu_short}"
# The cpu_short partition applies its own QoS (p_cpu_short) automatically; our
# account is only associated with the 'normal' QoS, so do NOT pass --qos (it
# would be rejected as an invalid qos specification). Override only if needed.
QOS="${QOS:-}"

ALL_EXPERIMENTS=(
  concurrency_scaling
  response_size_scaling
  tool_call_depth_scaling
  work_per_step_sensitivity
  agent_fan_out
  burst_repro
  trainer_shape
  realistic_latency
)

if [ "$#" -gt 0 ]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

if [ ! -d .venv ]; then
  echo "ERROR: .venv not found. Build it once in an interactive allocation:" >&2
  echo "  salloc -A ${ACCOUNT} -p ${PARTITION} --nodes=1 --time=1:00:00 --exclusive --mem=0" >&2
  echo "  cd $(pwd) && export UV_LINK_MODE=copy RAY_TMPDIR=/tmp && uv venv --python 3.12 && uv sync --extra dev" >&2
  exit 1
fi

# Pre-generate the shared driver data BEFORE submitting. run_experiments.py's
# _ensure_data() lazily creates these on first use; with N parallel jobs sharing
# one repo on a network filesystem, that becomes a write race on the same files.
# Generating them here (idempotent — skipped if present) removes the race.
DATA_DIR="tools/scale_sim/data"
if [ ! -f "${DATA_DIR}/bench.jsonl" ]; then
  echo "[suite] pre-generating ${DATA_DIR}/bench.jsonl"
  .venv/bin/python "${DATA_DIR}/generate_data.py" \
    --n 2000 --user-input-size-bytes 512 --output "${DATA_DIR}/bench.jsonl"
fi
for declared in single_agent_10k.jsonl multi_agent_10k.jsonl; do
  [ -f "${DATA_DIR}/${declared}" ] || cp -f "${DATA_DIR}/bench.jsonl" "${DATA_DIR}/${declared}"
done

echo "[suite] label=${LABEL} time=${TIME} account=${ACCOUNT} partition=${PARTITION} qos=${QOS:-<none>}"
echo "[suite] submitting ${#EXPERIMENTS[@]} experiment job(s): ${EXPERIMENTS[*]}"

qos_opt=()
[ -n "${QOS}" ] && qos_opt=(--qos="${QOS}")

for exp in "${EXPERIMENTS[@]}"; do
  echo "[suite] === sbatch ${exp} (TIME=${TIME}) ==="
  LABEL="${LABEL}" EXPERIMENTS="--experiment ${exp}" \
    sbatch \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --time="${TIME}" \
      --job-name="scale-${exp}" \
      "${qos_opt[@]}" \
      tools/scale_sim/run_on_slurm.sh
done

echo
echo "[suite] all submitted. Watch:   squeue -u \$USER"
echo "[suite] results land in:        tools/scale_sim/findings/${LABEL}/<experiment>.csv"
echo "[suite] per-job stdout:         scale-sim-<jobid>.out (repo root)"
