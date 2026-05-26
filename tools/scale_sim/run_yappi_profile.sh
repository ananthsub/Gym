#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Profile gym sub-servers with NeMo Gym's built-in yappi integration during
# one cell of load. Counterpart to ``run_pyspy_profile.sh``: same workload
# (single-agent, c=4096 by default, output_tokens=16384), different
# profiler. Use both side-by-side.
#
# Why both? py-spy and yappi answer different questions:
#   - py-spy gives a CPU-sample flame graph — what % of CPU is spent where.
#     Cheap, no instrumentation overhead, but no per-call cost.
#   - yappi gives per-function call counts and per-call time. Answers
#     "how many ms does ONE Pydantic.model_validate call cost on a 430 KB
#     body?", "how many serializations per rollout?", "what's the per-call
#     ser/de overhead in the production code path?". Has nontrivial
#     instrumentation overhead — DO NOT read the cell's latency numbers as
#     production-representative.
#
# What this script does:
#   1. Generates a base config that adds NeMo Gym's two profiling knobs
#      (``profiling_enabled: true`` + ``profiling_results_dirpath: <path>``)
#      to ``configs/single_agent_base.yaml``. Output dir is fixed (not
#      timestamped) so the artifacts are easy to find by name.
#   2. Runs a single cell at the chosen concurrency. Cell is shorter than
#      the §3.1.B budget (60s wall-clock, 5000 inputs) so yappi dump has
#      bounded data to process at teardown.
#   3. Bumps SCALE_SIM_TEARDOWN_TIMEOUT_S so each sub-server's lifespan
#      exit hook has time to run yappi's gprof2dot + pydot.write_png.
#
# Each sub-server (head, simple_agent, synthetic_model, synthetic_resources)
# writes 4 artifacts on shutdown:
#   - <name>.log       text dump: name, ncall, tsub, ttot, tavg per function
#   - <name>.callgrind binary callgrind format (kcachegrind / callgrind_annotate)
#   - <name>.dot       gprof2dot graph
#   - <name>.png       rendered call-graph image
#
# Knobs (env vars):
#   YAPPI_CELL_CONCURRENCY   — concurrency to drive at. Default: 4096.
#                              Match SCALE_SIM_PYSPY_TARGET's cell so the
#                              two profiles are comparable.
#   YAPPI_OUTPUT_TOKENS      — output_tokens. Default: 16384.
#   YAPPI_TOTAL_REQUESTS     — total inputs to dispatch. Default: 5000.
#                              Smaller than 20000 — yappi has bounded data
#                              to flush at shutdown.
#   YAPPI_WALLCLOCK_S        — per-cell wall-clock budget. Default: 60.
#                              Same reason as YAPPI_TOTAL_REQUESTS.
#
# Submit:
#
#     cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
#     DRIVER_SCRIPT=tools/scale_sim/run_yappi_profile.sh \
#     JOB_NAME=scale-yappi-profile TIME=1:00:00 \
#       bash tools/scale_sim/run_on_slurm_baremetal.sh batch
#
# Result lands at:
#   tools/scale_sim/results/<sha>/yappi_profile_c<conc>_<ts>/concurrency-*__output_tokens-*/  ← sweep_results + summary.json (cell-level)
#   tools/scale_sim/results/<sha>/_yappi_artifacts_c<conc>/<server_name>/<server_name>.log    ← per-call costs (open this!)
#   tools/scale_sim/results/<sha>/_yappi_artifacts_c<conc>/<server_name>/<server_name>.png    ← call graph
#
# Reading the .log file:
#   Sorted by ttot (total inclusive time) by default.
#   Columns: name, ncall, tsub, ttot, tavg
#     ncall: how many times this function was called during the cell
#     tsub:  total CPU time spent in this function exclusive of children
#     ttot:  total CPU time inclusive of children (what you'd see as a
#            wide bar on a flame graph)
#     tavg:  ttot / ncall — per-call CPU cost
#   Look for:
#     pydantic.main.BaseModel.model_validate        → tavg = per-call validation cost
#     pydantic.type_adapter.TypeAdapter.dump_json   → tavg = per-call serialization cost
#     orjson.loads / dumps                          → JSON ser/de baseline
#     simple_agent.app._create_response             → agent self-call cost
#     nemo_gym.server_utils.request                 → outbound aiohttp dispatch overhead

set -uo pipefail

GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${GYM_VENV}/bin/activate"
fi
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[yappi] ulimit -n: $(ulimit -n)"
echo "[yappi] hostname:  $(hostname)"
echo "[yappi] which ng_run: $(which ng_run || echo NOT FOUND)"

# yappi + gprof2dot + pydot must be importable. nemo_gym/profiling.py imports
# them at module load, so they're typically already in the venv used to run
# ng_run — but on a fresh worker node they may be missing. Install on the
# fly (mirroring how run_pyspy_profile.sh handles py-spy) so we fail
# gracefully instead of 5 minutes into a cell.
#
# Note: ng_run's process group spawns FastAPI sub-servers which import
# nemo_gym.server_utils → nemo_gym.profiling → yappi/gprof2dot/pydot. So
# the install needs to land in the SAME venv that ng_run uses; we infer
# that by resolving `which ng_run` to its python interpreter.
NG_RUN_PATH="$(command -v ng_run || true)"
if [[ -z "${NG_RUN_PATH}" ]]; then
    echo "[yappi] ERROR: ng_run not found on PATH"
    exit 1
fi
NG_RUN_PYTHON="$(dirname "${NG_RUN_PATH}")/python"
echo "[yappi] ng_run venv python: ${NG_RUN_PYTHON}"

ensure_pkg() {
    local pkg_import="$1"   # name to import-check (e.g. yappi)
    local pkg_install="$2"  # name to install (e.g. yappi). Usually same; differs for gprof2dot vs gprof2dot.
    if "${NG_RUN_PYTHON}" -c "import ${pkg_import}" 2>/dev/null; then
        local v
        v=$("${NG_RUN_PYTHON}" -c "import ${pkg_import} as m; print(getattr(m, '__version__', '?'))" 2>/dev/null || echo '?')
        echo "[yappi] ${pkg_import} present (version ${v})"
        return 0
    fi
    echo "[yappi] ${pkg_import} missing — installing ${pkg_install} into ng_run's venv..."
    if ! (uv pip install --python "${NG_RUN_PYTHON}" "${pkg_install}" 2>/dev/null \
          || "${NG_RUN_PYTHON}" -m pip install "${pkg_install}"); then
        echo "[yappi] ERROR: failed to install ${pkg_install}"
        return 1
    fi
    "${NG_RUN_PYTHON}" -c "import ${pkg_import}" 2>/dev/null || {
        echo "[yappi] ERROR: ${pkg_install} installed but ${pkg_import} still not importable"
        return 1
    }
    echo "[yappi] ${pkg_import} installed OK"
}

ensure_pkg yappi yappi || exit 1
ensure_pkg gprof2dot gprof2dot || exit 1
ensure_pkg pydot pydot || exit 1

# graphviz is required for the PNG render. If missing, the cell still
# produces .log + .callgrind + .dot which are the most useful artifacts.
if ! command -v dot >/dev/null 2>&1; then
    echo "[yappi] WARN: graphviz 'dot' not found. .png render will fail; .log/.callgrind/.dot will still land."
    echo "[yappi]       Install with: apt update && apt install -y graphviz"
fi

GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"
echo "[yappi] git sha: ${GIT_SHA}"

# Two operating modes:
#
# DEFAULT (low-load, drains naturally):
#   c=512, 500 reqs, 120 s wall-clock. Consumer drains in ~80-90 s.
#   All three sub-servers reach their lifespan exit cleanly and dump
#   yappi artifacts. Best for getting per-call costs (`tavg`).
#
# YAPPI_MATCH_PYSPY=1 (apples-to-apples vs §3.4):
#   c=4096, 20000 reqs, 5 min wall-clock — exactly the §3.4 py-spy cell's
#   load shape. Saturated single-agent. Direct A/B comparison of yappi vs
#   py-spy for the same workload. Combined with SCALE_SIM_DUMP_STATS_BEFORE_TEARDOWN=1
#   (set automatically below) so artifacts dump via /stats before SIGINT
#   triggers the agent's retry storm.
#
# Either way, SCALE_SIM_DUMP_STATS_BEFORE_TEARDOWN=1 is set so artifacts
# always land regardless of saturation level. The env var enables a
# pre-teardown hook in sweep_runner that hits GET /stats on every
# sub-server, triggering profiler.dump() in-place while servers are
# still healthy.
if [[ "${YAPPI_MATCH_PYSPY:-0}" == "1" ]]; then
    echo "[yappi] YAPPI_MATCH_PYSPY=1: matching §3.4 py-spy cell config"
    CELL_CONC="${YAPPI_CELL_CONCURRENCY:-4096}"
    OUTPUT_TOKENS="${YAPPI_OUTPUT_TOKENS:-16384}"
    TOTAL_REQUESTS="${YAPPI_TOTAL_REQUESTS:-20000}"
    WALLCLOCK_S="${YAPPI_WALLCLOCK_S:-300}"
else
    CELL_CONC="${YAPPI_CELL_CONCURRENCY:-512}"
    OUTPUT_TOKENS="${YAPPI_OUTPUT_TOKENS:-16384}"
    TOTAL_REQUESTS="${YAPPI_TOTAL_REQUESTS:-500}"
    WALLCLOCK_S="${YAPPI_WALLCLOCK_S:-120}"
fi

echo "[yappi] cell_concurrency=${CELL_CONC} output_tokens=${OUTPUT_TOKENS} total_requests=${TOTAL_REQUESTS} wallclock_s=${WALLCLOCK_S}"

# Bump the sub-server SIGINT-to-SIGKILL teardown timeout. yappi.Profiler.dump()
# runs gprof2dot + pydot.write_png in the lifespan exit hook, which can take
# 30-120s on a large callgraph. The default 15s is fine for non-profiling
# cells but truncates the dump here. 180s is a comfortable upper bound for
# the cell sizes this script produces.
export SCALE_SIM_TEARDOWN_TIMEOUT_S=180

# Trigger Profiler.dump() via each sub-server's /stats endpoint BEFORE
# we send SIGINT. This is the only reliable path for saturated cells
# (e.g. YAPPI_MATCH_PYSPY=1) where the agent's retry storm at teardown
# blocks its lifespan exit. The hook is a no-op when profiling_enabled
# isn't set, so it's safe to leave on always. See
# tools/scale_sim/sweep_runner.py::_trigger_yappi_dump_via_stats for the
# implementation.
export SCALE_SIM_DUMP_STATS_BEFORE_TEARDOWN=1

# Profiling artifacts dir. Fixed (not timestamped) so it's easy to find
# by name. Lives under tools/scale_sim/results/<sha>/ alongside the cell
# results dir. Path is relative to gym repo root because that's how
# nemo_gym.server_utils.setup_profiling resolves profiling_results_dirpath
# (against nemo_gym.WORKING_DIR — see nemo_gym/__init__.py).
PROFILE_DIR_REL="tools/scale_sim/results/${GIT_SHA}/_yappi_artifacts_c${CELL_CONC}"
PROFILE_DIR_ABS="${CONTAINER_GYM_PATH}/${PROFILE_DIR_REL}"
echo "[yappi] profile artifacts will land at: ${PROFILE_DIR_ABS}"

# Generate input data if missing.
if [[ ! -f data/sweep_20k.jsonl ]]; then
    python data/generate_data.py --n 20000 --user-input-size-bytes 1024 --output data/sweep_20k.jsonl
fi

# Generate the base config: layer NeMo Gym's profiling knobs plus a
# tighter per-cell budget on top of single_agent_base.yaml.
mkdir -p configs/_generated
BASE_CFG="configs/_generated/yappi_profile_c${CELL_CONC}_base.yaml"
python - <<PY
import yaml, pathlib
src = pathlib.Path("configs/single_agent_base.yaml")
dst = pathlib.Path("${BASE_CFG}")
cfg = yaml.safe_load(src.read_text())
# NeMo Gym's two top-level profiling knobs. ProfilingMiddlewareConfig
# in nemo_gym/server_utils.py reads these from the global config and,
# if profiling_enabled is true, registers Profiler(name=server_name,
# base_profile_dir=WORKING_DIR/profiling_results_dirpath/server_name)
# in each sub-server's lifespan_wrapper.
cfg["profiling_enabled"] = True
cfg["profiling_results_dirpath"] = "${PROFILE_DIR_REL}"
# Default: payload-size logging OFF so this is apples-to-apples with a
# pure profiling run (matches what run_pyspy_profile.sh does — no extra
# per-request stdout cost). Set YAPPI_PAYLOAD_DEBUG=1 in the environment
# to turn it back on (useful when correlating per-call costs with body
# sizes; not needed for direct yappi-vs-py-spy comparison).
import os as _os
cfg["global_aiohttp_client_request_debug"] = _os.environ.get("YAPPI_PAYLOAD_DEBUG", "0") == "1"
# Cell wall-clock budget.
cfg.setdefault("scale_sim", {})["early_stop_wall_clock_s"] = ${WALLCLOCK_S}
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[yappi] wrote base config: {dst}")
print(f"[yappi]   profiling_enabled: True")
print(f"[yappi]   profiling_results_dirpath: ${PROFILE_DIR_REL}")
print(f"[yappi]   global_aiohttp_client_request_debug: {cfg['global_aiohttp_client_request_debug']}  (set YAPPI_PAYLOAD_DEBUG=1 to enable)")
print(f"[yappi]   scale_sim.early_stop_wall_clock_s: ${WALLCLOCK_S}")
PY

# Sweep one cell. exp-name kept token-free (no 'simple_agent', 'synthetic_model',
# 'synthetic_resources') to avoid sweep_runner._pre_cell_cleanup's pkill regex
# matching the launcher's cmdline. See run_pyspy_profile.sh comment block.
EXP_NAME="yappi_profile_c${CELL_CONC}"
python run_sweep.py \
    --base "${BASE_CFG}" \
    --input-jsonl data/sweep_20k.jsonl \
    --concurrency "${CELL_CONC}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --total-requests "${TOTAL_REQUESTS}" \
    --git-sha "${GIT_SHA}" \
    --exp-name "${EXP_NAME}"

RC=$?

# After the cell runs, ng_run.log lives in the cell directory and contains
# the combined stdout of every sub-server (head, simple_agent, synthetic_model,
# synthetic_resources). gym's BaseServer.prefix_server_logs prepends
# `(<server_name>) ` to each line at runtime — see
# nemo_gym/server_utils.py — so we can cheaply split that combined file
# into one log per server next to the yappi artifacts. This puts ALL
# per-server data (yappi profile + stdout) in one place per server, which
# is what you actually want when correlating per-call yappi costs against
# the [payload] / ClientPayloadError / ClientOSError lines emitted in the
# server's stdout.
echo
echo "[yappi] Splitting ng_run.log per-server alongside the yappi artifacts..."
LATEST_CELL_LOG=$(ls -1dt "${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}/${EXP_NAME}_"*/concurrency-*/ng_run.log 2>/dev/null | head -1)
if [[ -n "${LATEST_CELL_LOG}" && -f "${LATEST_CELL_LOG}" ]]; then
    echo "[yappi]   source: ${LATEST_CELL_LOG}"
    # gym's BaseServer.prefix_server_logs prepends `(<server.config.name>) `
    # to every stdout line — that's the *configured instance name* from the
    # YAML (e.g. ``synthetic_simple_agent``, ``synthetic_resources_inst``,
    # ``synthetic_model_inst``), NOT the bare type names. Discover them at
    # runtime by scanning the combined log so we don't hard-code names that
    # drift if the YAML is renamed.
    INSTANCE_NAMES=$(grep -oE '^\([A-Za-z_][A-Za-z0-9_]*\) ' "${LATEST_CELL_LOG}" \
        | sort -u \
        | sed -E 's/^\((.+)\) /\1/')
    if [[ -z "${INSTANCE_NAMES}" ]]; then
        echo "[yappi]   WARN: no '(<instance>) ' prefixes found in ng_run.log."
        echo "[yappi]         BaseServer.prefix_server_logs may not have wrapped stdout."
    fi
    for SRV in ${INSTANCE_NAMES}; do
        SRV_DIR="${PROFILE_DIR_ABS}/${SRV}"
        mkdir -p "${SRV_DIR}"
        grep -F "(${SRV}) " "${LATEST_CELL_LOG}" > "${SRV_DIR}/${SRV}.stdout.log" || true
        N_LINES=$(wc -l < "${SRV_DIR}/${SRV}.stdout.log" 2>/dev/null || echo 0)
        echo "[yappi]   ${SRV}.stdout.log: ${N_LINES} lines → ${SRV_DIR}/${SRV}.stdout.log"
    done
    # Drop a copy of the full combined log alongside the artifacts so
    # everything for the run is reachable from one root.
    cp "${LATEST_CELL_LOG}" "${PROFILE_DIR_ABS}/ng_run.combined.log" 2>/dev/null || true
    echo "[yappi]   full combined log copied to: ${PROFILE_DIR_ABS}/ng_run.combined.log"
else
    echo "[yappi]   WARN: could not find ng_run.log under ${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}/${EXP_NAME}_*/."
    echo "[yappi]         Per-server stdout will still be available in the cell directory; just not split here."
fi
RESULTS_ROOT="${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}"
echo
echo "[yappi] run_sweep exit code: ${RC}"
echo
echo "[yappi] Cell results:"
echo "        ${RESULTS_ROOT}/${EXP_NAME}_*/"
echo
echo "[yappi] Per-server profile + stdout artifacts:"
echo "        Layout (yappi side, written by gym sub-servers on shutdown):"
echo "          ${PROFILE_DIR_ABS}/<ClassName>___<config.name>/<config.name>.log        # yappi: name, ncall, tsub, ttot, tavg per function"
echo "          ${PROFILE_DIR_ABS}/<ClassName>___<config.name>/<config.name>.callgrind  # kcachegrind / callgrind_annotate"
echo "          ${PROFILE_DIR_ABS}/<ClassName>___<config.name>/<config.name>.png        # call graph (graphviz required)"
echo "        Layout (stdout side, written by this launcher post-run):"
echo "          ${PROFILE_DIR_ABS}/<config.name>/<config.name>.stdout.log               # this server's stdout — payload-size lines, BaseServer prints, FastAPI access"
echo "          ${PROFILE_DIR_ABS}/ng_run.combined.log                                  # interleaved stdout of all sub-servers (raw)"
echo
echo "[yappi] What's in those names:"
echo "        <config.name>  is the server-instance name from the YAML (e.g. synthetic_simple_agent,"
echo "                       synthetic_model_inst, synthetic_resources_inst). Listed above as INSTANCE_NAMES."
echo "        <ClassName>    is the gym BaseServer subclass (e.g. BaseResponsesAPIAgent,"
echo "                       BaseResponsesAPIModel, BaseResourcesServer). gym writes the dir as"
echo "                       <ClassName>___<config.name> via get_session_middleware_key()."
echo
echo "[yappi] Quick checks (run on the host where the cell ran):"
echo "        ls -la ${PROFILE_DIR_ABS}/"
echo "        # The agent: per-call ser/de costs"
echo "        find ${PROFILE_DIR_ABS} -name 'synthetic_simple_agent.log' -exec head -50 {} \\;"
echo "        # The agent: stdout (payload-size lines, errors)"
echo "        find ${PROFILE_DIR_ABS} -name 'synthetic_simple_agent.stdout.log' -exec head -50 {} \\;"

exit "${RC}"
