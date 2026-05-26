#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capture a py-spy flame graph from a gym sub-server inside a running
# NeMo RL production Slurm job — non-interactively. No need to know
# which Ray worker node has the gym actor scheduled.
#
# How it works:
#   1. Reads <JOB_ID>-attach.sh (created by ray.sub when the cluster
#      comes up) to know how to enter the per-worker containers.
#   2. Fans out a single capture script to every node in the job in
#      parallel (head + workers). Each container runs the same script.
#   3. The script searches its own PID namespace for a sub-server with
#      cwd matching $TARGET. If found, runs py-spy. If not, exits 0
#      silently. Exactly one node will find the target; the others
#      no-op cleanly.
#   4. SVGs land at $OUT_DIR (defaults to $PWD/pyspy-prod-<job>-<ts>/).
#      Pull them down to your laptop with `scp` and open in a browser.
#
# Prerequisites — set these BEFORE submitting the original ray.sub job:
#
#   1. Sub-servers must volunteer to be ptraced. Set this in the env
#      that goes to sbatch:
#
#          export NEMO_GYM_ALLOW_PROFILER_ATTACH=1
#          sbatch ray.sub
#
#      Slurm propagates env to the container; gym's BaseServer.run_webserver
#      reads this and calls prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) at
#      startup. Without it, py-spy attach fails with "Permission denied"
#      on clusters with kernel.yama.ptrace_scope=1 (most production
#      clusters).
#
#   2. py-spy must be reachable inside the container. Cheapest option:
#      bind-mount a static py-spy binary via MOUNTS. Once on a Lustre path:
#
#          mkdir -p $HOME/profiling-tools
#          curl -L https://github.com/benfred/py-spy/releases/download/v0.4.0/py-spy-0.4.0-x86_64-unknown-linux-musl.tar.gz \
#              | tar xz -C $HOME/profiling-tools
#          chmod +x $HOME/profiling-tools/py-spy
#
#      Then add to MOUNTS when launching ray.sub:
#
#          MOUNTS="$MOUNTS,$HOME/profiling-tools:/opt/profiling-tools"
#          sbatch ray.sub
#
#      The default $PYSPY path below assumes this layout. Override
#      $PYSPY if your binary is elsewhere (e.g. installed in the venv
#      at /opt/nemo_rl_venv/bin/py-spy).
#
# Usage:
#
#     # Capture simple_agent (default target) for 60s on whichever node has it:
#     bash tools/scale_sim/run_pyspy_prod.sh <JOB_ID>
#
#     # Different target / duration / rate:
#     TARGET=synthetic_model DURATION=120 RATE=200 \
#       bash tools/scale_sim/run_pyspy_prod.sh <JOB_ID>
#
#     # Custom output dir on Lustre:
#     OUT_DIR=/lustre/.../my-profile \
#       bash tools/scale_sim/run_pyspy_prod.sh <JOB_ID>

set -euo pipefail

# ----------------------------------------------------------------------
# Args + env
# ----------------------------------------------------------------------

JOB_ID="${1:-}"
if [[ -z "$JOB_ID" ]]; then
    echo "Usage: $0 <SLURM_JOB_ID>"
    echo "  TARGET=<simple_agent|synthetic_model|synthetic_resources>  (default: simple_agent)"
    echo "  DURATION=<seconds>  (default: 60)"
    echo "  RATE=<sample Hz>    (default: 100)"
    echo "  OUT_DIR=<path>      (default: \$PWD/pyspy-prod-<job>-<ts>)"
    echo "  PYSPY=<path>        (default: /opt/profiling-tools/py-spy in container)"
    exit 1
fi

TARGET="${TARGET:-simple_agent}"
DURATION="${DURATION:-60}"
RATE="${RATE:-100}"
OUT_DIR="${OUT_DIR:-$PWD/pyspy-prod-${JOB_ID}-$(date +%s)}"
PYSPY="${PYSPY:-/opt/profiling-tools/py-spy}"

mkdir -p "$OUT_DIR"
echo "[pyspy-prod] job_id=$JOB_ID  target=$TARGET  duration=${DURATION}s  rate=${RATE}Hz"
echo "[pyspy-prod] py-spy path inside container: $PYSPY"
echo "[pyspy-prod] output dir: $OUT_DIR"

# ----------------------------------------------------------------------
# Locate the per-job attach script that ray.sub generated when the
# cluster came up. It lives in the job's SLURM_SUBMIT_DIR; the user
# typically runs this from there. Allow override via ATTACH_SH.
# ----------------------------------------------------------------------

ATTACH_SH="${ATTACH_SH:-$PWD/${JOB_ID}-attach.sh}"
if [[ ! -f "$ATTACH_SH" ]]; then
    echo "[pyspy-prod] ERROR: attach script not found at $ATTACH_SH"
    echo "             ray.sub writes <JOB_ID>-attach.sh into its SLURM_SUBMIT_DIR."
    echo "             Either cd to that dir before running, or set ATTACH_SH=<path>."
    exit 1
fi

NUM_NODES=$(scontrol show job "$JOB_ID" 2>/dev/null | grep -oP 'NumNodes=\K[0-9]+' | head -1 || true)
if [[ -z "$NUM_NODES" ]]; then
    echo "[pyspy-prod] ERROR: could not resolve NumNodes from scontrol show job $JOB_ID."
    echo "             Is the job still running?"
    exit 1
fi
echo "[pyspy-prod] job has $NUM_NODES node(s); fanning out to all"

# ----------------------------------------------------------------------
# The capture script that runs INSIDE each container. Identical script,
# different containers — only the one that finds $TARGET captures.
# Variables are interpolated by the outer shell so they baked into the
# script string before being shipped via attach.sh.
# ----------------------------------------------------------------------

CAPTURE_CMD=$(cat <<EOF
set -e
PYSPY='${PYSPY}'
TARGET='${TARGET}'
DURATION='${DURATION}'
RATE='${RATE}'
OUT_DIR='${OUT_DIR}'

# Find the target sub-server PID by cwd-match. cwd is more robust than
# command-line match because every gym sub-server runs as
# "python -u app.py" (same cmdline) but with a per-server cwd. Mirrors
# tools/scale_sim/sweep_runner.py::_find_subserver_pid.
PID=""
for pid in \$(pgrep -u \$USER -f "python.*app\.py" 2>/dev/null || true); do
    cwd=\$(readlink /proc/\$pid/cwd 2>/dev/null || true)
    case "\$cwd" in
        */\${TARGET}*) PID=\$pid; break;;
    esac
done

if [[ -z "\$PID" ]]; then
    echo "[\$(hostname)] no \$TARGET process here; skipping"
    exit 0
fi

echo "[\$(hostname)] found \$TARGET pid=\$PID, capturing py-spy for \${DURATION}s..."

if [[ ! -x "\$PYSPY" ]]; then
    echo "[\$(hostname)] ERROR: \$PYSPY not found or not executable in container."
    echo "                see tools/scale_sim/run_pyspy_prod.sh header for setup."
    exit 1
fi

# Sanity-check ptrace permission. PR_SET_PTRACER_ANY (set by gym when
# NEMO_GYM_ALLOW_PROFILER_ATTACH=1) is process-local; py-spy attach
# will fail here if the env var was not in the gym sub-server's env at
# startup. The error is informative; surface it upstream.
mkdir -p "\$OUT_DIR"
SVG="\$OUT_DIR/\${TARGET}_\$(hostname)_pid\${PID}.svg"

set +e
"\$PYSPY" record --pid "\$PID" --rate "\$RATE" --duration "\$DURATION" -o "\$SVG"
RC=\$?
set -e

if [[ \$RC -ne 0 ]]; then
    echo "[\$(hostname)] py-spy exited rc=\$RC. Common causes:"
    echo "  - kernel.yama.ptrace_scope=1 AND gym sub-server didn't volunteer:"
    echo "    re-launch the original ray.sub job with NEMO_GYM_ALLOW_PROFILER_ATTACH=1"
    echo "  - py-spy version too old / wrong arch for the container's libc"
    exit \$RC
fi

echo "[\$(hostname)] wrote \$SVG"
EOF
)

# ----------------------------------------------------------------------
# Fan out. Each background srun runs in its own container; the one with
# $TARGET writes an SVG, others log "no $TARGET here" and exit 0. We
# wait for all of them.
#
# attach.sh's interface: arg=worker index (empty for head, 1..N-1 for
# workers); COMMAND env var is the bash to run inside the container.
# ----------------------------------------------------------------------

for ((i = 0; i < NUM_NODES; i++)); do
    if [[ $i -eq 0 ]]; then
        IDX=""  # head node
    else
        IDX="$i"
    fi
    COMMAND="$CAPTURE_CMD" bash "$ATTACH_SH" $IDX 2>&1 | sed "s/^/[node$i] /" &
done
wait

echo
echo "[pyspy-prod] all capture attempts complete"
echo "[pyspy-prod] artifacts in: $OUT_DIR"
ls -la "$OUT_DIR/" 2>/dev/null || echo "  (no SVGs written — see [node*] output above for diagnosis)"
