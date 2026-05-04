#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-off validation: N=256 spinup_only for the multi-agent sweep, post-A10-fix.
# Confirms that the tempfile-based global-config IPC change in nemo_gym/cli.py
# allows ng_run to spawn 514 sub-servers without hitting Linux ARG_MAX.
#
# Submit via the bare-metal launcher:
#
#     cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
#     DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_n256_check.sh \
#     JOB_NAME=multi-agent-n256-check \
#       bash tools/scale_sim/run_on_slurm_baremetal.sh batch
#
# Look for in sbatch.out:
#   - `Argument list too long` → A10 fix didn't work, regression
#   - `[sweep] sub-servers ready after Ns` then summary printed → success
#   - tempfile created at /tmp/nemo_gym_global_config_<pid>.yaml during the
#     run, then deleted on shutdown

set -uo pipefail

GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${GYM_VENV}/bin/activate"
fi
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[multi-agent-n256-check] ulimit -n: $(ulimit -n)"
echo "[multi-agent-n256-check] hostname:  $(hostname)"
echo "[multi-agent-n256-check] which ng_run: $(which ng_run || echo NOT FOUND)"

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
echo "[multi-agent-n256-check] git sha:   ${GIT_SHA}"

# Snapshot current /tmp state — anything left behind by previous Ray clusters
# would blow up the new one. sweep_runner._pre_cell_cleanup also runs at the
# start of the cell, but show what's there for the record.
echo "[multi-agent-n256-check] pre-existing /tmp/ray*: $(ls -d /tmp/ray* 2>/dev/null | tr '\n' ' ')"
echo "[multi-agent-n256-check] pre-existing /tmp/nemo_gym_global_config_*: $(ls /tmp/nemo_gym_global_config_*.yaml 2>/dev/null | tr '\n' ' ')"
echo "[multi-agent-n256-check] pre-existing ng_run/raylet pids: $(pgrep -af 'raylet|gcs_server|plasma_store|ng_run' 2>/dev/null | head -3)"

GEN_DIR="configs/_generated"
mkdir -p "${GEN_DIR}"

echo
echo "[multi-agent-n256-check] Generating N=256 spinup_only config..."
python configs/_gen_multi_agent.py \
  --base configs/multi_agent_base.yaml --n-agents 256 \
  --out "${GEN_DIR}/multi_agent_n256_check.yaml"

echo
echo "[multi-agent-n256-check] Running spinup_only cell at N=256 (514 sub-servers)..."
echo "[multi-agent-n256-check] Expected wall-clock: ~3-5 min for spinup + 30s idle + teardown"
python sweep_runner.py \
  --config "${GEN_DIR}/multi_agent_n256_check.yaml" \
  --git-sha "${GIT_SHA}" \
  --head-server-host 127.0.0.1 --head-server-port 5000 \
  --driver-mode spinup_only --idle-window-s 30 \
  --spinup-timeout-s 1200 --teardown-sleep-s 32

# Capture the summary file (if produced) and check the headline results.
RESULTS_ROOT="${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}"
LATEST_CELL=$(ls -dt "${RESULTS_ROOT}"/multi_agent_n256_check_* 2>/dev/null | head -1)
echo
echo "[multi-agent-n256-check] Results dir: ${LATEST_CELL}"
if [[ -n "${LATEST_CELL}" ]] && [[ -f "${LATEST_CELL}/summary.json" ]]; then
  echo
  echo "[multi-agent-n256-check] === summary.json ==="
  cat "${LATEST_CELL}/summary.json"
  echo
  echo "[multi-agent-n256-check] === Last 30 lines of ng_run.log ==="
  tail -30 "${LATEST_CELL}/ng_run.log"
  echo
  echo "[multi-agent-n256-check] PASS: A10 fix validated. N=256 spinup completed successfully."
else
  echo "[multi-agent-n256-check] FAIL: No summary.json produced. Tail of ng_run.log:"
  if [[ -n "${LATEST_CELL}" ]]; then
    tail -60 "${LATEST_CELL}/ng_run.log" 2>/dev/null
    if grep -q "Argument list too long" "${LATEST_CELL}/ng_run.log" 2>/dev/null; then
      echo
      echo "[multi-agent-n256-check] FAIL REASON: A10 regression — still hitting ARG_MAX. Fix in nemo_gym/cli.py didn't take."
    fi
  fi
  exit 1
fi

# Confirm the materialized tempfile mechanism is working: there should be
# nothing left in /tmp from this run (RunHelper.shutdown cleans it up).
echo
echo "[multi-agent-n256-check] Post-shutdown /tmp/nemo_gym_global_config_*: $(ls /tmp/nemo_gym_global_config_*.yaml 2>/dev/null | tr '\n' ' ' || echo '<none — cleaned up>')"

echo
echo "[multi-agent-n256-check] Done."
