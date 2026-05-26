#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-command launcher for the production-failure-reproduction sub-sweeps.
# Submits all four cells as parallel Slurm jobs:
#
#   1h_repro_chunked_latency       — Real GPU-shaped latency (1500 ms) AND
#                                     chunked / streaming responses on the
#                                     synthetic model. Most-likely candidate
#                                     for ClientPayloadError because the
#                                     synthetic harness has never exercised
#                                     the chunked code path until now.
#                                     [asyncio driver, c=4K and 8K]
#
#   1i_repro_chunked_disconnect    — Same as 1h plus deterministic mid-stream
#                                     disconnects (1 % rate, after 64 KB).
#                                     Confirms the consumer surfaces
#                                     ClientPayloadError when the wire
#                                     actually breaks. Pessimistic control;
#                                     production rates are lower.
#                                     [asyncio driver, c=4K]
#
#   1j_repro_burst_dispatch        — Step-boundary burst dispatch (4 096 rows
#                                     fired, 60 s idle, repeat). Reproduces
#                                     production's post-refit dispatch shape;
#                                     if server-side keepalive < client's,
#                                     next burst hits dead connections.
#                                     [asyncio driver, c=4K and 8K]
#
#   1k_repro_production_thread_shape  — Threaded driver at production's
#                                     thread count: 256 threads × 16
#                                     rollouts per thread (sequential per
#                                     thread). Tests whether the §3.1.D
#                                     failures still appear when connection-
#                                     pool count matches production (256-512
#                                     pools) rather than 4-16K pools.
#                                     [threaded driver, c=4K and 8K]
#
# Each becomes its own Slurm job so they run in parallel and you can read
# the results independently. Submit:
#
#     cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
#     bash tools/scale_sim/run_repro_sweep.sh
#
# After all four complete, the comparison matrix to populate is in
# investigations/nemo-gym-scale-results.md §3.1.E (pending data — to be
# filled in once these cells land).

set -euo pipefail

GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"
echo "[repro] git sha: ${GIT_SHA}"
echo "[repro] submitting 4 sub-sweeps as parallel Slurm jobs..."

# Each repro sub-sweep gets its own Slurm job. TIME budgets:
#   1h/1i: real-latency cells take ~6-10 s per rollout, 5 K rollouts × cells = ~1 hr cap
#   1j: step-burst with 60 s idle gaps; lots of wall-clock overhead, give it room
#   1k: threaded-driver cells; same wall-clock as §3.1.D cells, ~30 min suffices
declare -a SUB_TIME=(
    "1h_repro_chunked_latency:2:00:00"
    "1i_repro_chunked_disconnect:2:00:00"
    "1j_repro_burst_dispatch:2:00:00"
    "1k_repro_production_thread_shape:2:00:00"
)

for entry in "${SUB_TIME[@]}"; do
    SUB="${entry%%:*}"
    TIME="${entry#*:}"
    echo
    echo "[repro] === submitting ${SUB} (TIME=${TIME}) ==="
    SINGLE_AGENT_SUBSWEEP="${SUB}" \
    DRIVER_SCRIPT=tools/scale_sim/run_single_agent_subsweep.sh \
    JOB_NAME="scale-repro-${SUB}" \
    TIME="${TIME}" \
    GIT_SHA="${GIT_SHA}" \
        bash tools/scale_sim/run_on_slurm_baremetal.sh batch
done

echo
echo "[repro] All 4 sub-sweeps submitted. Watch with:"
echo "        squeue -u \$USER"
echo
echo "[repro] Once they complete, results land at:"
echo "        tools/scale_sim/results/${GIT_SHA}/1h_repro_chunked_latency_*/"
echo "        tools/scale_sim/results/${GIT_SHA}/1i_repro_chunked_disconnect_*/"
echo "        tools/scale_sim/results/${GIT_SHA}/1j_repro_burst_dispatch_*/"
echo "        tools/scale_sim/results/${GIT_SHA}/1k_repro_production_thread_shape_*/"
echo
echo "[repro] Quick error-class scan once they finish:"
echo "        for SUB in 1h_repro_chunked_latency 1i_repro_chunked_disconnect 1j_repro_burst_dispatch 1k_repro_production_thread_shape; do"
echo "          for f in tools/scale_sim/results/${GIT_SHA}/\${SUB}_*/concurrency-*/summary.json; do"
echo "            jq -r '\"\\(.concurrency)  retry_rate=\\(.retry_summary.retry_rate)  errors=\\(.retry_summary.error_class_counts)\"' < \"\$f\""
echo "          done"
echo "        done"
