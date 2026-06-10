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
from analyze_throughput import analyze as analyze_throughput  # noqa: E402
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

# De-facto training-representative response body. The synthetic model emits this
# many output tokens (token ids + log probs + text) per response, ~1.7 MB on the
# wire. Held constant in every experiment except response_size_scaling (which
# sweeps it) and tool_call_depth_scaling (which uses a small body because the
# request body accumulates conversation history across hops — see _depth_cells).
TRAIN_RESPONSE_TOKENS = 65536

# Realistic model-latency proxy for the realistic_latency experiment (TESTPLAN
# #9 / Q5). The synthetic model can draw per-call latency from a Pareto
# distribution; these defaults mirror actor_repro.yaml's production-shaped
# values (median ~1.5 s, heavy tail capped at 2 min). RETUNE async_latency_ms /
# pareto_alpha from a real-vLLM calibration (TESTPLAN #10) when one is available;
# they are intentionally a single, obvious knob.
REAL_LATENCY_MS = 1500
REAL_LATENCY_PARETO_ALPHA = 1.5
REAL_LATENCY_PARETO_MAX_MS = 120000.0


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
def _fixed(
    output_tokens: int = TRAIN_RESPONSE_TOKENS,
    model_lat: int = 0,
    tool_lat: int = 0,
    verify_lat: int = 0,
    max_steps: int = 1,
):
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
    # Powers of 2 from the unloaded floor (1, 4, 16) through deep saturation
    # (131072), all at the 64k training body. No memory cap: the consumer
    # throttles, so actual peak in-flight memory is bounded by throughput x
    # latency (Little's law), not by offered concurrency (predicted vs actual
    # diverge ~10x at high load). total_requests is sized small at the floor so
    # those cells drain quickly for a clean L1, and capped at 20000 so saturated
    # cells go wall-bound on the 300s budget.
    for c in [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 131072]:
        cells.append(
            {
                "cell": f"c{c}",
                "value_col": "concurrency",
                "value": c,
                "concurrency": c,
                "total_requests": min(max(8 * c, 128), 20000),
                "wall_clock_s": 300,
                "overrides": _fixed(),
            }
        )
    return cells


def _realistic_latency_cells() -> List[Dict[str, Any]]:
    # TESTPLAN #9 / Q5: the concurrency sweep (#1) re-run with a realistic model
    # latency distribution instead of the near-zero synthetic proxy. Same 64k
    # body and grid as concurrency_scaling so the two are directly comparable;
    # the only change is the model draws per-call latency from a Pareto
    # distribution (median ~1.5 s, heavy tail). This shows whether the throughput
    # ceiling and the saturation knee shift once per-rollout latency dominates.
    cells = []
    for c in [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 131072]:
        overrides = _fixed()
        overrides[f"{MODEL}.async_latency_ms"] = REAL_LATENCY_MS
        overrides[f"{MODEL}.latency_dist"] = "pareto"
        overrides[f"{MODEL}.pareto_alpha"] = REAL_LATENCY_PARETO_ALPHA
        overrides[f"{MODEL}.pareto_min_ms"] = 0.0
        overrides[f"{MODEL}.pareto_max_ms"] = REAL_LATENCY_PARETO_MAX_MS
        cells.append(
            {
                "cell": f"c{c}",
                "value_col": "concurrency",
                "value": c,
                "concurrency": c,
                "total_requests": min(max(8 * c, 128), 20000),
                "wall_clock_s": 300,
                "overrides": overrides,
            }
        )
    return cells


def _response_size_cells() -> List[Dict[str, Any]]:
    cells = []
    # Power-of-2 sweep over response body size (token ids + log probs + text).
    # 64k (65536) is the de-facto training response size; the grid brackets it
    # symmetrically (16k..256k) and extends into the large-body cliff (512k..4M).
    # Concurrency is held fixed (low) so this isolates body-serialization cost
    # from the consumer event-loop ceiling that concurrency_scaling measures.
    for tok in [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]:
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
    # Powers of 2 out to 512 hops. Runs at concurrency=1 with a small (1k) body:
    # simple_agent accumulates prior outputs into each hop's request, so the
    # request body grows linearly with hop count (at 512 hops a 1k-token body is
    # already ~13 MB; a 64k body would be ~870 MB and OOM). c=1 isolates per-hop
    # cost from queueing, matching the prior extreme-hop-depth experiment.
    for steps in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        cells.append(
            {
                "cell": f"hops{steps}",
                "value_col": "tool_calls_per_rollout",
                "value": steps,
                "concurrency": 1,
                "total_requests": 8,
                "wall_clock_s": 900,
                "overrides": _fixed(output_tokens=1024, max_steps=steps),
            }
        )
    return cells


def _work_per_step_cells() -> List[Dict[str, Any]]:
    cells = []
    for lat in [0, 64, 256, 1024]:
        cells.append(
            {
                "cell": f"work{lat}ms",
                "value_col": "work_per_call_ms",
                "value": lat,
                "concurrency": 64,
                "total_requests": 20000,
                "wall_clock_s": 300,
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


def _trainer_shape_cells() -> List[Dict[str, Any]]:
    # Compare the two trainer->gym request shapes at MATCHED in-flight rollout
    # count (the gym/model sees the same N concurrent /run either way — only the
    # trainer<->actor return boundary differs). Both run on the fast
    # actor_stress.yaml so the storm completes in minutes, and both surface
    # error_class_counts so we can see whether production connection errors
    # appear with one shape but not the other.
    #
    #   threaded  (sync_blocking): the trainer spawns many threads, each blocking
    #     on one whole-batch RPC whose result crosses the Ray object store as a
    #     single object. This is the shape that produced production
    #     ClientPayloadError at high thread counts. To reach N in-flight we use
    #     N/16 threads each returning a 16-row batch (<=4096 OS threads at N=65536).
    #
    #   streaming (lag_batched_stream): ONE streaming RPC fans out all N rows and
    #     yields per-row refs back (ObjectRefGenerator) — many small transfers
    #     instead of one big one. This is the production shape today; scaling the
    #     batch size scales the single call's concurrency.
    #
    # Questions: does threaded surface connection errors as N grows while
    # streaming stays clean, and how high can the streaming shape scale?
    cells: List[Dict[str, Any]] = []
    threaded_batch = 16
    for in_flight in [256, 1024, 4096, 16384, 65536]:
        threads = in_flight // threaded_batch
        cells.append(
            {
                "cell": f"threaded_if{in_flight}",
                "value_col": "in_flight",
                "value": in_flight,
                "extra": {"shape": "threaded", "threads": threads, "batch_per_call": threaded_batch},
                "config": "actor_stress.yaml",
                "mode": "sync_blocking",
                "thread_count": threads,
                "prompts_per_call": threaded_batch,
                "num_steps": 1,
            }
        )
        cells.append(
            {
                "cell": f"stream_if{in_flight}",
                "value_col": "in_flight",
                "value": in_flight,
                "extra": {"shape": "streaming", "threads": 1, "batch_per_call": in_flight},
                "config": "actor_stress.yaml",
                "mode": "lag_batched_stream",
                "thread_count": 1,
                "prompts_per_call": in_flight,
                "num_steps": 1,
            }
        )
    return cells


def _burst_repro_cells() -> List[Dict[str, Any]]:
    # Burst + train-floor + refit-pause + overlapping-cycles, swept over response
    # body size, WITH the keepalive fix in place (server 30s > client 15s). The
    # question: does the connection-reset (ideally ClientPayloadError) reproduce
    # at larger bodies once the keepalive asymmetry is fixed? On loopback the body
    # must exceed the socket buffers (~16MB max here) for the server write to
    # block and create an in-flight window; smaller bodies are delivered to the
    # client receive buffer instantly. force_close at the top size is the control.
    common = dict(num_agents=16, num_prompts=256, per_prompt_calls=16, num_cycles=4, train_floor_s=30.0, refit_s=5.0)
    cells: List[Dict[str, Any]] = []
    for body in [65536, 131072, 262144, 524288]:
        cells.append(
            {
                "cell": f"baseline_{body // 1024}k",
                "value_col": "variant",
                "value": f"baseline_{body // 1024}k",
                "extra": {"body_tokens": body, "approx_body_mb": round(body * BYTES_PER_TOKEN / 1e6, 1), "force_close": False},
                "model_output_tokens": body,
                "spawn_jitter_s": 0.0,
                "force_close": False,
                **common,
            }
        )
    cells.append(
        {
            "cell": "forceclose_512k",
            "value_col": "variant",
            "value": "forceclose_512k",
            "extra": {"body_tokens": 524288, "approx_body_mb": round(524288 * BYTES_PER_TOKEN / 1e6, 1), "force_close": True},
            "model_output_tokens": 524288,
            "spawn_jitter_s": 0.0,
            "force_close": True,
            **common,
        }
    )
    return cells


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "concurrency_scaling": {
        "blurb": "Throughput and latency as simultaneous rollouts grow (1..131072) at the 64k training body, one tool call, no added work.",
        "kind": "single_agent",
        "cells": _concurrency_cells,
    },
    "response_size_scaling": {
        "blurb": "Latency and throughput as the response body grows in powers of 2 (16k..4M tokens), fixed concurrency.",
        "kind": "single_agent",
        "cells": _response_size_cells,
    },
    "tool_call_depth_scaling": {
        "blurb": "Per-hop cost as each rollout chains more tool/model calls (1..512 hops, small body, concurrency=1).",
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
    "burst_repro": {
        "blurb": "Reproduce the production ClientPayloadError via burst dispatch + train-floor + refit-pause + overlapping cycles; A/B spawn-jitter and keep-alive (16 agents, 8192 in-flight).",
        "kind": "burst",
        "cells": _burst_repro_cells,
    },
    "trainer_shape": {
        "blurb": "Threaded whole-batch RPCs vs one streaming RPC at matched in-flight (256..65536): do production connection errors appear with threaded but not streaming, and how high can streaming scale?",
        "kind": "return_shape",
        "cells": _trainer_shape_cells,
    },
    "realistic_latency": {
        "blurb": "Concurrency sweep (1..131072, 64k body) re-run with a Pareto model-latency distribution (~1.5s median, heavy tail) — does the ceiling/knee shift under realistic per-rollout latency?",
        "kind": "single_agent",
        "cells": _realistic_latency_cells,
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
    # Warm-up-excluded steady-state throughput from per-rollout timestamps. The
    # naive throughput includes the ramp; steady_rps is the plateau rate.
    prr = d / "per_rollout_retries.jsonl"
    steady = analyze_throughput(prr).get("steady_rps") if prr.exists() else None
    return {
        "throughput_rps": s.get("throughput_rollouts_per_s"),
        "steady_throughput_rps": steady,
        "p50_s": lat.get("p50_s"),
        "p99_s": lat.get("p99_s"),
        "failure_rate": retry.get("failure_rate"),
        "completion_rate": s.get("completion_rate"),
        "saturated": s.get("saturated"),
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
    attempted = rows.get("n_attempted") or 0
    return {
        "throughput_rps": (done / wall) if wall else None,
        "p50_s": lat.get("p50"),
        "p99_s": lat.get("p99"),
        "failure_rate": rows.get("failure_rate"),
        "error_class_counts": s.get("error_class_counts", {}),
        "completion_rate": (done / attempted) if attempted else None,
        "saturated": False,
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
    # Disable the collapse early-stop so a saturated cell runs the full window and
    # yields a real (low) throughput, instead of stopping the moment retries spike.
    OmegaConf.update(cfg, "scale_sim.early_stop_failure_rate", 1.0)
    OmegaConf.update(cfg, "scale_sim.early_stop_retry_rate", 1.0)
    out = GENERATED / label / f"{exp}_{cell['cell']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg))
    return out


def _materialize_fan_out(exp: str, cell: Dict[str, Any], label: str) -> Path:
    cfg_dict = gen_multi_agent.generate(CONFIGS / "multi_agent.yaml", cell["n_agents"])
    cfg = OmegaConf.create(cfg_dict)
    # Hold the shared model at the 64k training body, same as the single-agent
    # experiments, so fan-out is measured at a training-representative payload.
    model_key = gen_multi_agent._model_instance_name()
    OmegaConf.update(
        cfg,
        f"{model_key}.responses_api_models.synthetic_model.output_tokens",
        TRAIN_RESPONSE_TOKENS,
        force_add=True,
    )
    OmegaConf.update(cfg, "scale_sim.concurrency", cell["concurrency"])
    OmegaConf.update(cfg, "scale_sim.total_requests", cell["total_requests"])
    OmegaConf.update(cfg, "scale_sim.early_stop_wall_clock_s", cell["wall_clock_s"])
    OmegaConf.update(cfg, "scale_sim.early_stop_failure_rate", 1.0)
    OmegaConf.update(cfg, "scale_sim.early_stop_retry_rate", 1.0)
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
            elif spec["kind"] == "burst":
                metrics = _run_burst_cell(cell, out_dir, label)
            else:  # return_shape
                metrics = _run_return_shape_cell(cell, out_dir, input_jsonl)
        except Exception as e:  # keep the sweep going; record the failure
            metrics = {"error": f"{type(e).__name__}: {e}"}
        row = {cell["value_col"]: cell["value"], **cell.get("extra", {}), **metrics}
        rows.append(row)
        print(f"    -> {metrics}", flush=True)
        # Write findings after every cell so an interrupted run keeps partial results.
        _write_findings(exp, label, rows)
    return rows


def _materialize_burst(exp: str, cell: Dict[str, Any], label: str) -> tuple:
    """Generate an N-agent config for the burst test; return (config_path, agent_ports)."""
    cfg_dict = gen_multi_agent.generate(CONFIGS / "multi_agent.yaml", cell["num_agents"])
    cfg = OmegaConf.create(cfg_dict)
    # Non-blocking high-latency proxy: hold each agent->model call for ~Pareto(1.5s
    # median, heavy tail) via await asyncio.sleep (concurrent, does NOT block the
    # loop) and return a reference-sized ~430 KB body. This is the async
    # counterpart to the reference's blocking single-worker agent: connections to
    # the agents stay open across the train/refit gap because of genuine latency,
    # not loop serialization. Lets us test whether the connection-reset reproduces
    # from hold-time alone, without event-loop blocking.
    _mbase = f"{gen_multi_agent._model_instance_name()}.responses_api_models.synthetic_model"
    OmegaConf.update(cfg, f"{_mbase}.output_tokens", int(cell.get("model_output_tokens", 16384)), force_add=True)
    OmegaConf.update(cfg, f"{_mbase}.async_latency_ms", REAL_LATENCY_MS, force_add=True)
    OmegaConf.update(cfg, f"{_mbase}.latency_dist", "pareto", force_add=True)
    OmegaConf.update(cfg, f"{_mbase}.pareto_alpha", REAL_LATENCY_PARETO_ALPHA, force_add=True)
    OmegaConf.update(cfg, f"{_mbase}.pareto_min_ms", 0.0, force_add=True)
    OmegaConf.update(cfg, f"{_mbase}.pareto_max_ms", REAL_LATENCY_PARETO_MAX_MS, force_add=True)
    ports: List[int] = []
    for key, val in cfg_dict.items():
        agent = (val or {}).get("responses_api_agents", {}).get("simple_agent") if isinstance(val, dict) else None
        if agent and "port" in agent:
            ports.append(int(agent["port"]))
    ports.sort()
    out = GENERATED / label / f"{exp}_{cell['cell']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg))
    return out, ports


def _run_burst_cell(cell: Dict[str, Any], out_dir: Path, label: str) -> Dict[str, Any]:
    """Stand up N synthetic agents via ng_run, drive burst_driver.py, tear down."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path, agent_ports = _materialize_burst("burst_repro", cell, label)
    if not agent_ports:
        return {"error": "no agent ports resolved from generated config"}

    sweep_runner._pre_cell_cleanup()
    sweep_runner._pre_start_ray()

    ng_log_path = out_dir / "ng_run.log"
    ng_log = ng_log_path.open("w")
    ng_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ng_cmd = f'ng_run "+config_paths=[{cfg_path.as_posix()}]"'
    ng_proc = subprocess.Popen(
        ng_cmd,
        shell=True,
        cwd=SCALE_SIM_DIR,
        stdout=ng_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=ng_env,
    )
    try:
        ok, reason = sweep_runner._wait_for_port(
            "0.0.0.0", 5000, timeout_s=900.0, ng_proc=ng_proc, log_path=ng_log_path, phase_label="head:5000"
        )
        if not ok or not sweep_runner._wait_for_servers_ready(ng_log_path, timeout_s=900.0, ng_proc=ng_proc):
            return {"error": f"spinup failed ({reason if not ok else 'servers not ready'})"}

        burst_cmd = [
            sys.executable,
            "-u",
            str(SCALE_SIM_DIR / "burst_driver.py"),
            "--agent-ports",
            ",".join(str(p) for p in agent_ports),
            "--output-dir",
            str(out_dir),
            "--num-prompts-per-step",
            str(cell["num_prompts"]),
            "--per-prompt-calls",
            str(cell["per_prompt_calls"]),
            "--num-cycles",
            str(cell["num_cycles"]),
            "--train-floor-s",
            str(cell["train_floor_s"]),
            "--refit-s",
            str(cell["refit_s"]),
            "--spawn-jitter-s",
            str(cell["spawn_jitter_s"]),
        ]
        if cell.get("force_close"):
            burst_cmd.append("--force-close")
        subprocess.run(burst_cmd, cwd=SCALE_SIM_DIR)
    finally:
        try:
            os.killpg(os.getpgid(ng_proc.pid), signal.SIGINT)
            ng_proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(ng_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        ng_log.close()

    summ = out_dir / "summary.json"
    if not summ.exists():
        return {"error": "burst_driver wrote no summary"}
    s = json.loads(summ.read_text())
    errs = s.get("error_types", {})
    return {
        "failure_rate": s.get("failure_rate"),
        "successes": s.get("successes"),
        "failures": s.get("failures"),
        "client_payload_errors": errs.get("ClientPayloadError", 0),
        "error_types": json.dumps(errs),
    }


def _run_return_shape_cell(cell: Dict[str, Any], out_dir: Path, input_jsonl: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # mock_trainer.py connects with ray.init(address="auto"), so it needs a
    # running cluster. The single_agent/fan_out cells get this from
    # sweep_runner._run_one_cell; this cell calls mock_trainer directly, so we
    # must clear stale Ray state and pre-start the pinned-port cluster ourselves.
    # Without this the experiment only works as the last cell of --all (inheriting
    # a leftover cluster) and fails when run standalone on a fresh Slurm node.
    sweep_runner._pre_cell_cleanup()
    sweep_runner._pre_start_ray()
    cmd = [
        sys.executable,
        "-u",
        str(SCALE_SIM_DIR / "mock_trainer.py"),
        "--config",
        str(CONFIGS / cell.get("config", "actor_repro.yaml")),
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
