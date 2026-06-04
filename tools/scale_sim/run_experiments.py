# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the scale-testing experiments and write comparable summaries.

Each experiment varies one thing and holds everything else fixed. For every run
this writes:

- raw per-cell artifacts under ``results/<label>/...`` (gitignored)
- one compact summary CSV per experiment under ``findings/<label>/`` (committed)
- ``findings/<label>/host.json`` with the machine specs

The ``<label>`` names the hardware (e.g. ``workstation`` or ``slurm-cpu``) so
results from different machines sit side by side in the same branch.

Run interactively (a menu) or with flags for batch / cluster use:

    python run_experiments.py --list
    python run_experiments.py --experiment concurrency_scaling --label workstation
    python run_experiments.py --all --label slurm-cpu
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


SCALE_SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCALE_SIM_DIR))
import sweep_runner  # noqa: E402
from configs import gen_multi_agent  # noqa: E402


CONFIGS = SCALE_SIM_DIR / "configs"
DATA = SCALE_SIM_DIR / "data"
RESULTS = SCALE_SIM_DIR / "results"
FINDINGS = SCALE_SIM_DIR / "findings"
GENERATED = RESULTS / "_generated"

# Dotted paths into the single-agent config for the knobs experiments vary.
MODEL = "synthetic_model_inst.responses_api_models.synthetic_model"
RES = "synthetic_resources_inst.resources_servers.synthetic_resources"
AGENT = "synthetic_simple_agent.responses_api_agents.simple_agent"

# Approximate wire bytes per token once token ids + log probs + text are
# serialized to JSON (used only to annotate response-size cells).
BYTES_PER_TOKEN = 26


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def cleanup_processes() -> None:
    """Kill leftover ng_run / Ray / synthetic-server processes and clear Ray state.

    sweep_runner cleans up before each cell, but the last cell's pre-started Ray
    cluster survives the run, and a kill mid-run strands the active ng_run. We
    run this at startup (clean slate), at normal exit, and on SIGINT/SIGTERM so
    an interrupted run never leaves processes holding ports.
    """
    try:
        sweep_runner._pre_cell_cleanup()
    except Exception as e:  # never let cleanup raise
        print(f"[cleanup] warning: {e}", file=sys.stderr, flush=True)


def _on_signal(signum, _frame) -> None:
    print(f"\n[run_experiments] signal {signum} received — cleaning up leftover processes...", flush=True)
    cleanup_processes()
    sys.exit(130)


# --------------------------------------------------------------------------- #
# Host info
# --------------------------------------------------------------------------- #
def host_info(label: str) -> Dict[str, Any]:
    cpu_model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb: Optional[float] = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=SCALE_SIM_DIR).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        sha = ""
    return {
        "label": label,
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "mem_gb": mem_gb,
        "fd_limit_soft": soft,
        "fd_limit_hard": hard,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# --------------------------------------------------------------------------- #
# Experiment definitions
# --------------------------------------------------------------------------- #
def _fixed(output_tokens: int = 256, model_lat: int = 0, tool_lat: int = 0, verify_lat: int = 0, max_steps: int = 1):
    """Per-cell overrides applied on top of configs/single_agent.yaml."""
    return {
        f"{MODEL}.output_tokens": output_tokens,
        f"{MODEL}.prompt_tokens": 512,
        f"{MODEL}.async_latency_ms": model_lat,
        f"{RES}.tool.async_latency_ms": tool_lat,
        f"{RES}.verify.async_latency_ms": verify_lat,
        f"{AGENT}.max_steps": max_steps,
    }


def _concurrency_cells() -> List[Dict[str, Any]]:
    cells = []
    for c in [64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072]:
        cells.append(
            {
                "cell": f"c{c}",
                "value_col": "concurrency",
                "value": c,
                "concurrency": c,
                "total_requests": max(2 * c, 20000),
                "wall_clock_s": 240,
                "overrides": _fixed(),
            }
        )
    return cells


def _response_size_cells() -> List[Dict[str, Any]]:
    cells = []
    for tok in [16384, 65536, 262144, 1048576, 4194304, 10485760]:
        conc = 64 if tok <= 1048576 else 16
        cells.append(
            {
                "cell": f"tok{tok}",
                "value_col": "context_tokens",
                "value": tok,
                "extra": {"approx_body_mb": round(tok * BYTES_PER_TOKEN / 1e6, 1)},
                "concurrency": conc,
                "total_requests": max(4 * conc, 256),
                "wall_clock_s": 300,
                "overrides": _fixed(output_tokens=tok),
            }
        )
    return cells


def _depth_cells() -> List[Dict[str, Any]]:
    cells = []
    for steps in [1, 2, 4, 8]:
        cells.append(
            {
                "cell": f"hops{steps}",
                "value_col": "tool_calls_per_rollout",
                "value": steps,
                "concurrency": 64,
                "total_requests": 20000,
                "wall_clock_s": 240,
                "overrides": _fixed(max_steps=steps),
            }
        )
    return cells


def _work_per_step_cells() -> List[Dict[str, Any]]:
    cells = []
    for lat in [0, 50, 200, 1000]:
        cells.append(
            {
                "cell": f"work{lat}ms",
                "value_col": "work_per_call_ms",
                "value": lat,
                "concurrency": 64,
                "total_requests": 20000,
                "wall_clock_s": 240,
                "overrides": _fixed(model_lat=lat),
            }
        )
    return cells


def _fan_out_cells() -> List[Dict[str, Any]]:
    per_agent = 256
    cells = []
    for n in [1, 2, 4, 8, 16, 32]:
        total = per_agent * n
        cells.append(
            {
                "cell": f"n{n}",
                "value_col": "n_agents",
                "value": n,
                "extra": {"per_agent_concurrency": per_agent, "total_concurrency": total},
                "n_agents": n,
                "concurrency": total,
                "total_requests": max(2 * total, 20000),
                "wall_clock_s": 240,
            }
        )
    return cells


def _return_shape_cells() -> List[Dict[str, Any]]:
    # (mode, thread_count, prompts_per_call) — in-flight = thread_count * prompts_per_call.
    return [
        {
            "cell": "whole_batch",
            "value_col": "return_shape",
            "value": "whole_batch",
            "mode": "sync_blocking",
            "thread_count": 256,
            "prompts_per_call": 1,
            "num_steps": 5,
        },
        {
            "cell": "streaming",
            "value_col": "return_shape",
            "value": "streaming",
            "mode": "lag_batched_stream",
            "thread_count": 8,
            "prompts_per_call": 256,
            "num_steps": 5,
        },
    ]


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "concurrency_scaling": {
        "blurb": "Throughput and latency as simultaneous rollouts grow (tiny responses, one tool call, no added work).",
        "kind": "single_agent",
        "cells": _concurrency_cells,
    },
    "response_size_scaling": {
        "blurb": "Latency and throughput as the response body grows (token ids + log probs for 16K..10M context), fixed concurrency.",
        "kind": "single_agent",
        "cells": _response_size_cells,
    },
    "tool_call_depth_scaling": {
        "blurb": "Cost as each rollout makes more sequential tool/model calls, fixed concurrency.",
        "kind": "single_agent",
        "cells": _depth_cells,
    },
    "work_per_step_sensitivity": {
        "blurb": "How much framework overhead matters as each call does more real work (added per-call delay), fixed concurrency.",
        "kind": "single_agent",
        "cells": _work_per_step_cells,
    },
    "agent_fan_out": {
        "blurb": "Spreading load across more agent servers at a fixed per-agent load (1..32 agents).",
        "kind": "fan_out",
        "cells": _fan_out_cells,
    },
    "trainer_return_shape": {
        "blurb": "Whole-batch vs streaming returns through the Ray actor.",
        "kind": "return_shape",
        "cells": _return_shape_cells,
    },
}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _ensure_data() -> Path:
    """Create the driver input JSONL, plus the dataset files the configs declare.

    The load driver reads ``bench.jsonl`` directly; the agent configs also declare
    a dataset path that ng_run expects to exist at startup, so we point those at
    the same rows.
    """
    path = DATA / "bench.jsonl"
    if not path.exists():
        subprocess.run(
            [
                sys.executable,
                str(DATA / "generate_data.py"),
                "--n",
                "2000",
                "--user-input-size-bytes",
                "512",
                "--output",
                str(path),
            ],
            check=True,
        )
    for declared in ("single_agent_10k.jsonl", "multi_agent_10k.jsonl"):
        dst = DATA / declared
        if not dst.exists():
            dst.write_bytes(path.read_bytes())
    return path


def _read_load_summary(d: Path) -> Dict[str, Any]:
    s = json.loads((d / "summary.json").read_text())
    lat = s.get("latency_summary", {})
    retry = s.get("retry_summary", {})
    return {
        "throughput_rps": s.get("throughput_rollouts_per_s"),
        "p50_s": lat.get("p50_s"),
        "p99_s": lat.get("p99_s"),
        "failure_rate": retry.get("failure_rate"),
        "completed": retry.get("n_rollouts"),
        "wall_s": s.get("wall_clock_s"),
        "stop_reason": s.get("stop_reason"),
    }


def _read_trainer_summary(d: Path) -> Dict[str, Any]:
    s = json.loads((d / "summary.json").read_text())
    rows = s.get("rows", {})
    lat = s.get("per_row_latency_s", {})
    wall = s.get("run_wall_clock_s")
    done = rows.get("n_succeeded") or 0
    return {
        "throughput_rps": (done / wall) if wall else None,
        "p50_s": lat.get("p50"),
        "p99_s": lat.get("p99"),
        "failure_rate": rows.get("failure_rate"),
        "completed": done,
        "wall_s": wall,
        "stop_reason": None,
    }


def _materialize_single_agent(exp: str, cell: Dict[str, Any], label: str) -> Path:
    cfg = OmegaConf.load(CONFIGS / "single_agent.yaml")
    for path, val in cell["overrides"].items():
        OmegaConf.update(cfg, path, val, force_add=True)
    OmegaConf.update(cfg, "scale_sim.concurrency", cell["concurrency"])
    OmegaConf.update(cfg, "scale_sim.total_requests", cell["total_requests"])
    OmegaConf.update(cfg, "scale_sim.early_stop_wall_clock_s", cell["wall_clock_s"])
    out = GENERATED / label / f"{exp}_{cell['cell']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg))
    return out


def _materialize_fan_out(exp: str, cell: Dict[str, Any], label: str) -> Path:
    cfg_dict = gen_multi_agent.generate(CONFIGS / "multi_agent.yaml", cell["n_agents"])
    cfg = OmegaConf.create(cfg_dict)
    OmegaConf.update(cfg, "scale_sim.concurrency", cell["concurrency"])
    OmegaConf.update(cfg, "scale_sim.total_requests", cell["total_requests"])
    OmegaConf.update(cfg, "scale_sim.early_stop_wall_clock_s", cell["wall_clock_s"])
    out = GENERATED / label / f"{exp}_{cell['cell']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg))
    return out


def run_experiment(exp: str, label: str, input_jsonl: Path) -> List[Dict[str, Any]]:
    spec = EXPERIMENTS[exp]
    cells = spec["cells"]()
    rows: List[Dict[str, Any]] = []
    print(f"\n=== {exp} ({len(cells)} cells) — {spec['blurb']}", flush=True)
    for cell in cells:
        out_dir = RESULTS / label / exp / cell["cell"]
        print(f"\n--- {exp}/{cell['cell']} ({cell['value_col']}={cell['value']})", flush=True)
        try:
            if spec["kind"] == "single_agent":
                cfg = _materialize_single_agent(exp, cell, label)
                rc = sweep_runner._run_one_cell(
                    config_path=cfg,
                    input_jsonl=input_jsonl,
                    output_dir=out_dir,
                    head_server_host="0.0.0.0",
                    head_server_port=5000,
                    spinup_timeout_s=600.0,
                    driver_mode="loaded",
                    teardown_sleep_s=5.0,
                )
                metrics = _read_load_summary(out_dir) if rc == 0 else {"error": f"cell rc={rc}"}
            elif spec["kind"] == "fan_out":
                cfg = _materialize_fan_out(exp, cell, label)
                rc = sweep_runner._run_one_cell(
                    config_path=cfg,
                    input_jsonl=input_jsonl,
                    output_dir=out_dir,
                    head_server_host="0.0.0.0",
                    head_server_port=5000,
                    spinup_timeout_s=900.0,
                    driver_mode="loaded",
                    teardown_sleep_s=10.0,
                )
                metrics = _read_load_summary(out_dir) if rc == 0 else {"error": f"cell rc={rc}"}
            else:  # return_shape
                metrics = _run_return_shape_cell(cell, out_dir, input_jsonl)
        except Exception as e:  # keep the sweep going; record the failure
            metrics = {"error": f"{type(e).__name__}: {e}"}
        row = {cell["value_col"]: cell["value"], **cell.get("extra", {}), **metrics}
        rows.append(row)
        print(f"    -> {metrics}", flush=True)
    _write_findings(exp, label, rows)
    return rows


def _run_return_shape_cell(cell: Dict[str, Any], out_dir: Path, input_jsonl: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(SCALE_SIM_DIR / "mock_trainer.py"),
        "--config",
        str(CONFIGS / "actor_repro.yaml"),
        "--input-jsonl",
        str(input_jsonl),
        "--output-dir",
        str(out_dir),
        "--mode",
        cell["mode"],
        "--thread-count",
        str(cell["thread_count"]),
        "--prompts-per-call",
        str(cell["prompts_per_call"]),
        "--num-steps",
        str(cell["num_steps"]),
    ]
    rc = subprocess.run(cmd, cwd=SCALE_SIM_DIR).returncode
    return _read_trainer_summary(out_dir) if rc == 0 else {"error": f"trainer rc={rc}"}


def _write_findings(exp: str, label: str, rows: List[Dict[str, Any]]) -> None:
    FINDINGS.joinpath(label).mkdir(parents=True, exist_ok=True)
    out = FINDINGS / label / f"{exp}.csv"
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _interactive_pick() -> List[str]:
    names = list(EXPERIMENTS)
    print("\nExperiments:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n} — {EXPERIMENTS[n]['blurb']}")
    print("  a. all")
    choice = input("\nPick numbers (comma-separated) or 'a': ").strip().lower()
    if choice in ("a", "all", ""):
        return names
    picked = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(names):
            picked.append(names[int(part) - 1])
    return picked or names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", action="append", choices=list(EXPERIMENTS), help="Run one experiment (repeatable).")
    p.add_argument("--all", action="store_true", help="Run all experiments.")
    p.add_argument("--label", default=None, help="Hardware label for the results dir (e.g. workstation, slurm-cpu).")
    p.add_argument("--list", action="store_true", help="List experiments and exit.")
    p.add_argument("--cleanup", action="store_true", help="Kill leftover ng_run/Ray/server processes and exit.")
    args = p.parse_args()

    if args.list:
        for n, s in EXPERIMENTS.items():
            print(f"{n}\n    {s['blurb']}")
        return

    if args.cleanup:
        cleanup_processes()
        print("Cleanup complete.")
        return

    if args.all:
        to_run = list(EXPERIMENTS)
    elif args.experiment:
        to_run = args.experiment
    elif sys.stdin.isatty():
        to_run = _interactive_pick()
    else:
        p.error("Specify --all or --experiment, or run interactively from a terminal.")

    label = args.label or socket.gethostname().split(".")[0]
    FINDINGS.joinpath(label).mkdir(parents=True, exist_ok=True)
    info = host_info(label)
    (FINDINGS / label / "host.json").write_text(json.dumps(info, indent=2))
    print(f"host: {info['hostname']} | {info['cpu_count']} cores | {info['mem_gb']} GB | label={label}")

    # Clean any leftovers from a previous run, and make sure we clean up on exit
    # or interrupt so we never strand ng_run / Ray / server processes.
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    atexit.register(cleanup_processes)
    cleanup_processes()

    input_jsonl = _ensure_data()
    for exp in to_run:
        run_experiment(exp, label, input_jsonl)
    print(f"\nDone. Summaries under {FINDINGS / label}/")


if __name__ == "__main__":
    main()
