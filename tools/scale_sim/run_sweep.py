# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level sweep runner.

Takes a base config + a list of values for each knob, generates one config per
cell of the cross-product (or zipped together), and runs the whole matrix using
``sweep_runner._run_one_cell``. Aggregates each cell's ``summary.json`` into a
single CSV at the end so you can plot the results directly.

Designed for the §5 sweep matrix in `nemo-gym-scale-simulation-design.md` plus
the sub-server `num_workers` axis we discovered after the first 8K run.

Usage::

    cd tools/scale_sim/

    # Concurrency × num_workers cross product
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 1024,4096,8192,16384,32768 \\
        --num-workers 1,8 \\
        --exp-name axis_a_c_x_w

    # Sequence-length sweep — the right way to test body-size scaling.
    # output_tokens drives text + token_ids + log_probs sizes proportionally.
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 8192 \\
        --output-tokens 256,1024,4096,16384 \\
        --num-workers 8 \\
        --exp-name axis_c_output_tokens

    # Far-tail output-tokens sweep into 100K-1M range.
    # Concurrency MUST be scaled down inversely to keep host RAM bounded.
    # Matched-list zip mode: 16K@8K, 128K@1K, 256K@512, 512K@256, 1M@128 keeps
    # peak model-server body memory at ~80 GB across all cells.
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --output-tokens 16384,131072,262144,524288,1048576 \\
        --concurrency 8192,1024,512,256,128 \\
        --total-requests 20000,4000,2000,1000,500 \\
        --num-workers 8 \\
        --mode zip \\
        --exp-name output_tokens_far_tail

    # Long-thinking sweep: vary reasoning content while output stays fixed.
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 8192 \\
        --output-tokens 1024 \\
        --n-reasoning-items 1 \\
        --reasoning-tokens-per-item 0,8192,32768,131072 \\
        --num-workers 8 \\
        --exp-name axis_c_reasoning

    # Hop-depth sweep
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 8192 \\
        --n-hops 1,4,16,64 \\
        --exp-name axis_c_depth

    # Defect-#5 ablation: semaphore on/off at high concurrency
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 16384 \\
        --num-workers 8 \\
        --semaphore-enabled true,false \\
        --exp-name defect_5_ablation

    # Training-mode ablation: drop token_ids/log_probs to isolate their cost.
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 8192 \\
        --output-tokens 16384 \\
        --include-token-ids-and-log-probs true,false \\
        --num-workers 8 \\
        --exp-name training_mode_cost

    # Zip mode (parallel lists, no cross product) — useful when one parameter
    # implies another (e.g. larger concurrency needs larger total_requests)
    python run_sweep.py \\
        --base configs/axis_a_8k.yaml \\
        --concurrency 8192,16384,32768 \\
        --total-requests 10000,20000,40000 \\
        --num-workers 8 \\
        --mode zip \\
        --exp-name concurrency_isoload

Results land at ``results/<exp_name>_<ts>/`` with subdirectories per cell and a
top-level ``sweep_results.csv``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

# Make `sweep_runner._run_one_cell` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_runner import _run_one_cell  # noqa: E402


SCALE_SIM_DIR = Path(__file__).resolve().parent


# Each axis maps to one or more dotted paths into the merged config YAML. We
# support multiple paths because the same conceptual knob is set on multiple
# sub-servers (e.g. `num_workers` on resources, model, and agent).
KNOB_TO_PATHS: Dict[str, List[str]] = {
    # Load driver
    "concurrency": ["scale_sim.concurrency"],
    "total_requests": ["scale_sim.total_requests"],
    "semaphore_enabled": ["scale_sim.semaphore_enabled"],
    # Per-sub-server uvicorn worker count (the sub-server-side M3 axis)
    "num_workers": [
        "synthetic_resources_inst.resources_servers.synthetic_resources.num_workers",
        "synthetic_model_inst.responses_api_models.synthetic_model.num_workers",
        "synthetic_simple_agent.responses_api_agents.simple_agent.num_workers",
    ],
    # Hop depth (Axis C1)
    "n_hops": ["synthetic_simple_agent.responses_api_agents.simple_agent.max_steps"],
    # Sequence-length-driven payload knobs (Axis C2). Body bytes derived from these.
    "prompt_tokens": ["synthetic_model_inst.responses_api_models.synthetic_model.prompt_tokens"],
    "output_tokens": ["synthetic_model_inst.responses_api_models.synthetic_model.output_tokens"],
    "n_reasoning_items": ["synthetic_model_inst.responses_api_models.synthetic_model.n_reasoning_items"],
    "reasoning_tokens_per_item": [
        "synthetic_model_inst.responses_api_models.synthetic_model.reasoning_tokens_per_item",
    ],
    "include_token_ids_and_log_probs": [
        "synthetic_model_inst.responses_api_models.synthetic_model.include_token_ids_and_log_probs",
    ],
    # Tool-side payload (the function_call_output the agent receives)
    "tool_output_chars": [
        "synthetic_resources_inst.resources_servers.synthetic_resources.tool.body_size_bytes",
    ],
    "tool_body_shape": ["synthetic_resources_inst.resources_servers.synthetic_resources.tool.body_shape"],
    # Latency
    "tool_async_latency_ms": ["synthetic_resources_inst.resources_servers.synthetic_resources.tool.async_latency_ms"],
    "model_async_latency_ms": ["synthetic_model_inst.responses_api_models.synthetic_model.async_latency_ms"],
    "model_cpu_burn_ms": ["synthetic_model_inst.responses_api_models.synthetic_model.cpu_burn_ms"],
}


def _parse_value_list(raw: str, kind: str) -> List[Any]:
    """Parse comma-separated values, casting based on kind."""
    if raw is None or raw == "":
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    cast: Any
    if kind == "int":
        cast = int
    elif kind == "float":
        cast = float
    elif kind == "bool":
        cast = lambda s: s.lower() in ("true", "1", "yes")  # noqa: E731
    else:
        cast = str
    return [cast(p) for p in parts]


def _set_path(d: Dict, dotted: str, value: Any) -> None:
    """Set a value in a nested dict via dotted path. Creates missing intermediate dicts."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _generate_cells(
    knob_values: Dict[str, List[Any]],
    mode: str,
) -> List[Dict[str, Any]]:
    """Return list of {knob: value} dicts representing one cell each."""
    knobs = list(knob_values.keys())
    value_lists = [knob_values[k] for k in knobs]

    if mode == "cross_product":
        return [dict(zip(knobs, combo)) for combo in itertools.product(*value_lists)]
    if mode == "zip":
        lengths = {len(v) for v in value_lists}
        if len(lengths) > 1:
            raise ValueError(f"--mode zip requires equal-length value lists, got {dict(zip(knobs, value_lists))}")
        return [dict(zip(knobs, combo)) for combo in zip(*value_lists)]
    raise ValueError(f"Unknown mode: {mode}")


def _materialize_config(base_cfg: Dict, cell: Dict[str, Any]) -> Dict:
    """Apply a cell's overrides to the base config and return a new dict."""
    cfg = copy.deepcopy(base_cfg)
    for knob, value in cell.items():
        if knob not in KNOB_TO_PATHS:
            raise KeyError(f"Unknown sweep knob: {knob}. Add to KNOB_TO_PATHS in run_sweep.py.")
        for path in KNOB_TO_PATHS[knob]:
            _set_path(cfg, path, value)
    return cfg


def _cell_dir_name(cell: Dict[str, Any]) -> str:
    """Encode a cell's knob/value dict as a directory name.

    NOTE: must not contain `=`. Hydra parses `+config_paths=[<path>]` and
    interprets `=` in the path as a key-value override boundary, which fails
    with `no viable alternative at input '...concurrency='`. Use `-` between
    knob and value, `__` between knobs.
    """
    parts = []
    for k, v in sorted(cell.items()):
        s = str(v)
        if isinstance(v, bool):
            s = "T" if v else "F"
        # Compress numbers a bit
        try:
            n = int(v)
            if abs(n) >= 1024:
                s = f"{n // 1024}k" if n % 1024 == 0 else f"{n}"
        except (TypeError, ValueError):
            pass
        parts.append(f"{k}-{s}")
    return "__".join(parts)


def _load_summary(cell_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = cell_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None


def _flatten_summary(cell: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(cell)
    row["stop_reason"] = summary.get("stop_reason")
    rs = summary.get("retry_summary", {}) or {}
    ls = summary.get("latency_summary", {}) or {}
    row["n_rollouts"] = rs.get("n_rollouts")
    row["failure_rate"] = rs.get("failure_rate")
    row["retry_rate"] = rs.get("retry_rate")
    row["p_at_least_1_retry"] = rs.get("p_at_least_1_retry")
    row["n_attempts_max"] = rs.get("n_attempts_max")
    error_classes = rs.get("error_class_counts") or {}
    row["error_classes"] = ";".join(f"{k}={v}" for k, v in error_classes.items())
    row["p50_s"] = ls.get("p50_s")
    row["p90_s"] = ls.get("p90_s")
    row["p99_s"] = ls.get("p99_s")
    row["p999_s"] = ls.get("p999_s")
    row["max_s"] = ls.get("max_s")
    return row


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    print("\n=== sweep results ===")
    print(" | ".join(f"{c:<{widths[c]}}" for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Base config YAML to layer cell overrides on top of.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="JSONL of input rows. If omitted, derived from --base's `simple_agent.datasets[0].jsonl_fpath`.",
    )
    parser.add_argument("--exp-name", type=str, default="sweep", help="Experiment name; used in results dir name.")
    parser.add_argument(
        "--mode",
        choices=("cross_product", "zip"),
        default="cross_product",
        help="`cross_product` (default): every combination. `zip`: parallel lists, equal length.",
    )
    parser.add_argument("--head-server-host", default="0.0.0.0")
    parser.add_argument("--head-server-port", type=int, default=5000)
    parser.add_argument("--spinup-timeout-s", type=float, default=300.0)
    parser.add_argument("--git-sha", default=None, help="Override the git sha tag in results dir.")

    # ---- Load driver knobs ----
    parser.add_argument("--concurrency", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument("--total-requests", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument("--semaphore-enabled", type=str, default=None, help="Comma-separated bools.")
    # ---- Sub-server worker count (M3 sub-server axis) ----
    parser.add_argument(
        "--num-workers",
        type=str,
        default=None,
        help="Comma-separated ints. Applied to all 3 sub-servers (resources, model, agent).",
    )
    # ---- Hop depth (Axis C1) ----
    parser.add_argument("--n-hops", type=str, default=None, help="Comma-separated ints; sets simple_agent.max_steps.")
    # ---- Sequence-length payload knobs (Axis C2) ----
    parser.add_argument("--prompt-tokens", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument("--output-tokens", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument("--n-reasoning-items", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument("--reasoning-tokens-per-item", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument(
        "--include-token-ids-and-log-probs",
        type=str,
        default=None,
        help="Comma-separated bools. RL training default is true.",
    )
    # ---- Tool-output payload ----
    parser.add_argument("--tool-output-chars", type=str, default=None, help="Comma-separated ints.")
    parser.add_argument("--tool-body-shape", type=str, default=None, help="flat_padding,realistic_messages")
    # ---- Latency ----
    parser.add_argument("--tool-async-latency-ms", type=str, default=None)
    parser.add_argument("--model-async-latency-ms", type=str, default=None)
    parser.add_argument("--model-cpu-burn-ms", type=str, default=None)

    args = parser.parse_args()

    # Build the knob → values dict, only including knobs the user actually specified.
    knob_values: Dict[str, List[Any]] = {}
    if args.concurrency:
        knob_values["concurrency"] = _parse_value_list(args.concurrency, "int")
    if args.total_requests:
        knob_values["total_requests"] = _parse_value_list(args.total_requests, "int")
    if args.semaphore_enabled:
        knob_values["semaphore_enabled"] = _parse_value_list(args.semaphore_enabled, "bool")
    if args.num_workers:
        knob_values["num_workers"] = _parse_value_list(args.num_workers, "int")
    if args.n_hops:
        knob_values["n_hops"] = _parse_value_list(args.n_hops, "int")
    if args.prompt_tokens:
        knob_values["prompt_tokens"] = _parse_value_list(args.prompt_tokens, "int")
    if args.output_tokens:
        knob_values["output_tokens"] = _parse_value_list(args.output_tokens, "int")
    if args.n_reasoning_items:
        knob_values["n_reasoning_items"] = _parse_value_list(args.n_reasoning_items, "int")
    if args.reasoning_tokens_per_item:
        knob_values["reasoning_tokens_per_item"] = _parse_value_list(args.reasoning_tokens_per_item, "int")
    if args.include_token_ids_and_log_probs:
        knob_values["include_token_ids_and_log_probs"] = _parse_value_list(
            args.include_token_ids_and_log_probs, "bool"
        )
    if args.tool_output_chars:
        knob_values["tool_output_chars"] = _parse_value_list(args.tool_output_chars, "int")
    if args.tool_body_shape:
        knob_values["tool_body_shape"] = _parse_value_list(args.tool_body_shape, "str")
    if args.tool_async_latency_ms:
        knob_values["tool_async_latency_ms"] = _parse_value_list(args.tool_async_latency_ms, "float")
    if args.model_async_latency_ms:
        knob_values["model_async_latency_ms"] = _parse_value_list(args.model_async_latency_ms, "float")
    if args.model_cpu_burn_ms:
        knob_values["model_cpu_burn_ms"] = _parse_value_list(args.model_cpu_burn_ms, "float")

    if not knob_values:
        parser.error("Must specify at least one knob (e.g. --concurrency 8192,16384).")

    cells = _generate_cells(knob_values, args.mode)
    print(f"[sweep] Generated {len(cells)} cells from knobs={list(knob_values)} mode={args.mode}")

    # Load base config and figure out input JSONL.
    base_cfg = OmegaConf.to_container(OmegaConf.load(args.base), resolve=True)
    if args.input_jsonl is None:
        try:
            datasets = base_cfg["synthetic_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
            input_jsonl = Path(datasets[0]["jsonl_fpath"])
        except (KeyError, IndexError, TypeError):
            parser.error("Could not derive --input-jsonl from base config; please pass it explicitly.")
    else:
        input_jsonl = args.input_jsonl

    # Resolve relative paths against the cwd where the user invoked the script.
    # run_all_m1_sweeps.sh always cd's into tools/scale_sim/ before calling run_sweep.py,
    # so `data/smoke.jsonl` from a yaml resolves to <scale_sim>/data/smoke.jsonl.
    if not input_jsonl.is_absolute():
        input_jsonl = (Path.cwd() / input_jsonl).resolve()
    if not input_jsonl.exists():
        parser.error(
            f"Input JSONL not found: {input_jsonl}. "
            f"Generate with `python data/generate_data.py ...` from {Path.cwd()} first."
        )

    # Set up the experiment directory.
    git_sha = args.git_sha or "local"
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = SCALE_SIM_DIR / "results" / git_sha / f"{args.exp_name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = exp_dir / "configs"
    cells_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] exp_dir={exp_dir}")
    print(f"[sweep] input_jsonl={input_jsonl}")

    # Save the spec for reproducibility.
    (exp_dir / "spec.json").write_text(
        json.dumps(
            {
                "base": str(args.base),
                "input_jsonl": str(input_jsonl),
                "mode": args.mode,
                "knob_values": knob_values,
                "cells": cells,
            },
            indent=2,
            default=str,
        )
    )

    # Materialize per-cell config files and run them.
    cell_results: List[Tuple[Dict[str, Any], int, Path]] = []
    for i, cell in enumerate(cells):
        cell_name = _cell_dir_name(cell) or f"cell_{i:03d}"
        cell_dir = exp_dir / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)

        cell_config = _materialize_config(base_cfg, cell)
        cell_yaml = cells_dir / f"{cell_name}.yaml"
        OmegaConf.save(OmegaConf.create(cell_config), cell_yaml)

        print(f"\n[sweep] cell {i + 1}/{len(cells)}: {cell_name}")
        rc = _run_one_cell(
            config_path=cell_yaml,
            input_jsonl=input_jsonl,
            output_dir=cell_dir,
            head_server_host=args.head_server_host,
            head_server_port=args.head_server_port,
            spinup_timeout_s=args.spinup_timeout_s,
        )
        cell_results.append((cell, rc, cell_dir))
        print(f"[sweep] cell {cell_name} → rc={rc} dir={cell_dir}")

    # Aggregate.
    rows: List[Dict[str, Any]] = []
    for cell, rc, cell_dir in cell_results:
        summary = _load_summary(cell_dir)
        if summary is None:
            row = dict(cell)
            row.update({"rc": rc, "stop_reason": "no_summary"})
        else:
            row = _flatten_summary(cell, summary)
            row["rc"] = rc
        rows.append(row)

    csv_path = exp_dir / "sweep_results.csv"
    if rows:
        cols = list({k for r in rows for k in r.keys()})
        # Stable column order: knob columns first, then metrics
        knob_cols = [c for c in cols if c in KNOB_TO_PATHS]
        other_cols = [c for c in cols if c not in KNOB_TO_PATHS]
        ordered = knob_cols + sorted(other_cols)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print(f"\n[sweep] Aggregate CSV: {csv_path}")

    _print_table(rows)

    failures = [(c, rc) for c, rc, _ in cell_results if rc != 0]
    if failures:
        print(f"\n[sweep] {len(failures)} / {len(cells)} cells returned non-zero rc")
        sys.exit(1)


if __name__ == "__main__":
    main()
