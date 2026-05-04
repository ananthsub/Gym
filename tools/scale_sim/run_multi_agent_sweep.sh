#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs the full multi-agent (co-located sub-server count) sweep matrix
# end-to-end inside the container OR bare-metal venv. Designed to be invoked by
# `run_on_slurm{,_baremetal}.sh batch` via the DRIVER_SCRIPT env var.
#
# Topology is locked: N agents + N resources + 1 shared model. See
# `investigations/nemo-gym-multi-agent-design.md` for the matrix shape and
# predictions.
#
# Each sub-sweep is wrapped in `|| true` so one failure doesn't stop the rest.
# After all sub-sweeps complete, aggregates per-cell summaries into
# `tools/scale_sim/results/<sha>/multi_agent_master.csv`.

set -uo pipefail

# ---------- Inside-container/venv setup ----------
GYM_VENV="${GYM_VENV:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym}"
CONTAINER_GYM_PATH="${CONTAINER_GYM_PATH:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym}"

# Activate only if not already active. Bare-metal driver wrapper sources the venv
# already; container path also sources it; this is defensive against double-source
# on already-activated venvs causing benign warnings.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${GYM_VENV}/bin/activate"
fi
cd "${CONTAINER_GYM_PATH}/tools/scale_sim"

ulimit -Sn 1048576 2>/dev/null || ulimit -Sn 65535 2>/dev/null || true
echo "[multi-agent] ulimit -n: $(ulimit -n)"
echo "[multi-agent] hostname:  $(hostname)"
echo "[multi-agent] gym path:  ${CONTAINER_GYM_PATH}"
echo "[multi-agent] which ng_run: $(which ng_run || echo NOT FOUND)"

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
echo "[multi-agent] git sha:   ${GIT_SHA}"

RESULTS_ROOT="${CONTAINER_GYM_PATH}/tools/scale_sim/results/${GIT_SHA}"
mkdir -p "${RESULTS_ROOT}"

BASE_CFG="configs/multi_agent_base.yaml"
GEN_DIR="configs/_generated"
mkdir -p "${GEN_DIR}"

# ---------- Generate input data ----------
# One shared input file across all multi-agent cells (driver round-robins agent_ref).
echo
echo "[multi-agent] Generating input JSONL..."
python data/generate_data.py --n 20000 --user-input-size-bytes 1024 --output data/multi_agent_20k.jsonl

# ---------- Helpers ----------
# Generate cell config for a given (n_agents, concurrency, total_requests, [semaphore]).
# Echoes the path to the generated YAML.
gen_cell() {
  local n="$1" conc="$2" reqs="$3" sem="${4:-true}"
  local out="${GEN_DIR}/multi_agent_n${n}_c${conc}_r${reqs}_sem${sem}.yaml"
  python configs/_gen_multi_agent.py \
    --base "${BASE_CFG}" \
    --n-agents "${n}" \
    --concurrency "${conc}" \
    --total-requests "${reqs}" \
    --semaphore-enabled "${sem}" \
    --out "${out}" >&2
  echo "${out}"
}

# Calculate teardown sleep based on N (N//8, min 5s, max 60s) — gives kernel
# TIME_WAIT entries from 2N+1 sub-servers' connections time to drain.
teardown_sleep_for() {
  local n="$1"
  local s=$((n / 8))
  if (( s < 5 )); then s=5; fi
  if (( s > 60 )); then s=60; fi
  echo "$s"
}

# Spinup timeout grows with N — at large N, many uvicorn cold-starts are serialized.
spinup_timeout_for() {
  local n="$1"
  local t=$((60 + n * 5))   # 60s baseline + 5s per agent
  if (( t > 1200 )); then t=1200; fi
  echo "$t"
}

# ---------- Sweep matrix ----------

# Sub-sweep A. Spinup-only — what does N existing cost? (no traffic)
echo
echo "[multi-agent] === A: spinup-only ==="
for N in 1 4 16 32 64 128 256; do
  echo
  echo "[multi-agent] --- spinup_only n_agents=${N} ---"
  cfg=$(gen_cell "${N}" 1 1 true)
  python sweep_runner.py \
    --config "${cfg}" \
    --git-sha "${GIT_SHA}" \
    --head-server-host 127.0.0.1 --head-server-port 5000 \
    --driver-mode spinup_only \
    --idle-window-s 30 \
    --spinup-timeout-s "$(spinup_timeout_for "${N}")" \
    --teardown-sleep-s "$(teardown_sleep_for "${N}")" || \
    echo "[multi-agent] WARN: spinup_only n_agents=${N} failed (continuing)"
done

# Sub-sweep B. Loaded sweep — fixed total concurrency = 4096.
# Tests "does adding sub-servers cost something at fixed total load."
# Per-agent concurrency decreases as N grows: at N=256 each agent sees only 16 in-flight.
echo
echo "[multi-agent] === B: fixed total concurrency=4096 ==="
for N in 1 4 16 32 64 128 256; do
  echo
  echo "[multi-agent] --- loaded fixed_total n_agents=${N} ---"
  cfg=$(gen_cell "${N}" 4096 10000 true)
  python sweep_runner.py \
    --config "${cfg}" \
    --input-jsonl "${CONTAINER_GYM_PATH}/tools/scale_sim/data/multi_agent_20k.jsonl" \
    --git-sha "${GIT_SHA}" \
    --head-server-host 127.0.0.1 --head-server-port 5000 \
    --driver-mode loaded \
    --spinup-timeout-s "$(spinup_timeout_for "${N}")" \
    --teardown-sleep-s "$(teardown_sleep_for "${N}")" || \
    echo "[multi-agent] WARN: B n_agents=${N} failed (continuing)"
done

# Sub-sweep C. Loaded sweep — fixed per-agent concurrency = 256, total grows with N.
# Headline test: at N=256, total concurrency = 65536. Predicted to be where OS rejects
# new connections (EMFILE) or kernel TW table fills. Where we expect to find the
# scale cliff via the N axis.
echo
echo "[multi-agent] === C: fixed per-agent concurrency=256 ==="
for N in 1 4 16 32 64 128 256; do
  total_conc=$((256 * N))
  # Cap total_requests at 50K to keep wall-clock bounded at huge N.
  # At N=256 total_conc=65536, even 50K requests will saturate the consumer for minutes.
  total_reqs=$((4000 * N))
  if (( total_reqs > 50000 )); then total_reqs=50000; fi
  echo
  echo "[multi-agent] --- loaded fixed_per_agent n_agents=${N} total_conc=${total_conc} ---"
  cfg=$(gen_cell "${N}" "${total_conc}" "${total_reqs}" true)
  python sweep_runner.py \
    --config "${cfg}" \
    --input-jsonl "${CONTAINER_GYM_PATH}/tools/scale_sim/data/multi_agent_20k.jsonl" \
    --git-sha "${GIT_SHA}" \
    --head-server-host 127.0.0.1 --head-server-port 5000 \
    --driver-mode loaded \
    --spinup-timeout-s "$(spinup_timeout_for "${N}")" \
    --teardown-sleep-s "$(teardown_sleep_for "${N}")" || \
    echo "[multi-agent] WARN: C n_agents=${N} failed (continuing)"
done

# ---------- Aggregate ----------
echo
echo "[multi-agent] Aggregating results..."
python <<PYEOF
import csv, json
from pathlib import Path

results_root = Path("${RESULTS_ROOT}")
rows = []
# Each cell wrote summary.json under results/<sha>/<cfg_stem>_<ts>/.
for cell_dir in sorted(results_root.glob("multi_agent_n*")):
    summary_path = cell_dir / "summary.json"
    if not summary_path.exists():
        continue
    try:
        s = json.loads(summary_path.read_text())
    except Exception as e:
        print(f"[multi-agent] failed to read {summary_path}: {e}")
        continue

    base = {
        "cell": cell_dir.name,
        "mode": s.get("mode", "loaded"),
        "n_agents": s.get("n_agents"),
        "concurrency": s.get("concurrency"),
        "total_requests": s.get("total_requests"),
        "semaphore_enabled": s.get("semaphore_enabled"),
        "stop_reason": s.get("stop_reason"),
    }
    if base["mode"] == "spinup_only":
        idle = s.get("idle_kernel", {}) or {}
        proc = s.get("idle_driver_process", {}) or {}
        rows.append({
            **base,
            "tcp_inuse": idle.get("tcp_inuse"),
            "tcp_tw": idle.get("tcp_tw"),
            "file_nr_used": idle.get("file_nr_used"),
            "loadavg_1m": idle.get("loadavg_1m"),
            "driver_rss_mb": proc.get("rss_mb"),
        })
    else:
        retry = s.get("retry_summary", {}) or {}
        lat = s.get("latency_summary", {}) or {}
        rows.append({
            **base,
            "n_rollouts": retry.get("n_rollouts"),
            "failure_rate": retry.get("failure_rate"),
            "retry_rate": retry.get("retry_rate"),
            "p50_s": lat.get("p50_s"),
            "p99_s": lat.get("p99_s"),
            "max_s": lat.get("max_s"),
        })

if not rows:
    print(f"[multi-agent] No summary.json files found under {results_root}")
else:
    out_csv = results_root / "multi_agent_master.csv"
    cols = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[multi-agent] Master summary: {out_csv}")
    # Pretty-print the headline columns.
    headline = ["cell", "mode", "n_agents", "concurrency", "n_rollouts", "failure_rate", "retry_rate", "p50_s", "p99_s", "max_s", "stop_reason"]
    cols_present = [c for c in headline if any(c in r for r in rows)]
    print("\t".join(cols_present))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols_present))
PYEOF

echo
echo "[multi-agent] Done. Results under ${RESULTS_ROOT}"
