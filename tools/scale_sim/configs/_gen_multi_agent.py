# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a multi-agent cell config from configs/multi_agent_base.yaml.

Topology is locked to N agents + N resources + 1 shared model — see
``investigations/nemo-gym-multi-agent-design.md``.

Usage::

    python tools/scale_sim/configs/_gen_multi_agent.py \
        --base tools/scale_sim/configs/multi_agent_base.yaml \
        --n-agents 64 \
        --out tools/scale_sim/configs/_generated/multi_agent_n64.yaml

The generator is deterministic: same N + same base → same YAML byte-for-byte.
This is what makes per-cell reruns reproducible.

Optional overrides (used by run_multi_agent_sweep.sh to vary the matrix without
editing the base file):

    --concurrency / --total-requests / --semaphore-enabled
        Override the matching scale_sim.* knobs for this cell.

The generator does NOT touch the per-server knobs (latencies, body sizes).
Those live in the base file and are stamped into every instance verbatim. If
a sweep wants to vary them, that's a separate base file.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml


def _resources_instance_name(i: int) -> str:
    return f"synthetic_resources_{i}"


def _agent_instance_name(i: int) -> str:
    return f"synthetic_simple_agent_{i}"


def _model_instance_name() -> str:
    return "shared_synthetic_model"


def generate(base_path: Path, n_agents: int) -> dict:
    """Stamp out the full cell config dict from the base template + N.

    Returns the dict; caller writes to disk. Pure function — no side effects.
    """
    if n_agents < 1:
        raise ValueError(f"n_agents must be >= 1, got {n_agents}")

    with base_path.open() as f:
        base = yaml.safe_load(f)

    if "multi_agent" not in base:
        raise ValueError(f"{base_path} has no top-level `multi_agent:` block — wrong base file?")

    meta = base["multi_agent"]
    model_knobs = meta["model"]
    resources_knobs = meta["resources"]
    agent_knobs = meta["agent"]
    port_base = int(meta.get("port_base", 5001))
    dataset_name = meta.get("dataset_name", "multi_agent")
    dataset_path = meta.get("dataset_jsonl_fpath", "data/multi_agent_10k.jsonl")

    # Output dict starts as a copy of base minus the meta `multi_agent` block.
    out = {k: deepcopy(v) for k, v in base.items() if k != "multi_agent"}

    # Port allocation: head reads from existing head_server.port (5000). Then
    # model=port_base, resources_i=port_base+1+i, agent_i=port_base+1+N+i.
    model_port = port_base
    resources_port_start = port_base + 1
    agent_port_start = port_base + 1 + n_agents

    # 1 shared model
    out[_model_instance_name()] = {
        "responses_api_models": {
            "synthetic_model": {
                "entrypoint": "app.py",
                **deepcopy(model_knobs),
                "port": model_port,
            }
        }
    }

    # N resources instances — same on-disk server type, different top-level keys.
    # Per the design doc, ng_run shares <server_type>/<name>/.venv/ across
    # all instances of the same type, so this scales to N=256 without N venvs.
    for i in range(n_agents):
        out[_resources_instance_name(i)] = {
            "resources_servers": {
                "synthetic_resources": {
                    "entrypoint": "app.py",
                    "domain": "other",
                    "verified": False,
                    **deepcopy(resources_knobs),
                    "port": resources_port_start + i,
                }
            }
        }

    # N agent instances — each pointing at its own resources but the shared model.
    for i in range(n_agents):
        out[_agent_instance_name(i)] = {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    **deepcopy(agent_knobs),
                    "resources_server": {
                        "type": "resources_servers",
                        "name": _resources_instance_name(i),
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": _model_instance_name(),
                    },
                    "datasets": [
                        {
                            "name": dataset_name,
                            "type": "example",
                            "jsonl_fpath": dataset_path,
                            "num_repeats": 1,
                        }
                    ],
                    "port": agent_port_start + i,
                }
            }
        }

    # scale_sim block already exists from the base file; add agent_names round-robin.
    if "scale_sim" not in out:
        out["scale_sim"] = {}
    out["scale_sim"]["agent_names"] = [_agent_instance_name(i) for i in range(n_agents)]
    # Drop legacy single-agent field if it's hanging around in the base.
    out["scale_sim"].pop("agent_name", None)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Path to multi_agent_base.yaml.")
    parser.add_argument("--n-agents", type=int, required=True, help="Number of agent instances to stamp out.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the generated YAML.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override scale_sim.concurrency for this cell.",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=None,
        help="Override scale_sim.total_requests for this cell.",
    )
    parser.add_argument(
        "--semaphore-enabled",
        type=str,
        default=None,
        choices=("true", "false"),
        help="Override scale_sim.semaphore_enabled for this cell.",
    )
    args = parser.parse_args()

    cfg = generate(args.base, args.n_agents)
    if args.concurrency is not None:
        cfg["scale_sim"]["concurrency"] = args.concurrency
    if args.total_requests is not None:
        cfg["scale_sim"]["total_requests"] = args.total_requests
    if args.semaphore_enabled is not None:
        cfg["scale_sim"]["semaphore_enabled"] = args.semaphore_enabled == "true"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        # default_flow_style=False keeps it human-readable.
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(
        f"Wrote {args.out} (n_agents={args.n_agents}, "
        f"total_subservers={2 * args.n_agents + 1}, "
        f"concurrency={cfg['scale_sim']['concurrency']}, "
        f"total_requests={cfg['scale_sim']['total_requests']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
