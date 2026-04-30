# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic JSONL of N rollout tasks for the scale-sim harness.

Each row matches the shape that `RolloutCollectionHelper` consumes:

    {
        "responses_create_params": {"input": [...], "tools": [...]},
        "verifier_metadata": {},
    }

Use ``user_input_size_bytes`` to dial the *request*-side body size axis, separate
from the response-side knobs on the synthetic_model / synthetic_resources servers.

Usage:

    python tools/scale_sim/data/generate_data.py \
        --n 8192 \
        --user-input-size-bytes 1024 \
        --output tools/scale_sim/data/synthetic_8k.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = "You are the scale-sim synthetic agent."


def _build_row(user_input_size_bytes: int) -> dict:
    user_text = "x" * max(1, user_input_size_bytes)
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "synthetic_tool",
                    "description": "Scale-sim synthetic tool. Returns a controllable-size payload.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
        },
        "verifier_metadata": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="Number of rows to emit.")
    parser.add_argument(
        "--user-input-size-bytes",
        type=int,
        default=512,
        help="Bytes of padding in the user message. Controls the request-side body size axis.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for _ in range(args.n):
            f.write(json.dumps(_build_row(args.user_input_size_bytes)) + "\n")

    print(
        f"Wrote {args.n} rows to {args.output} "
        f"(user_input_size_bytes={args.user_input_size_bytes})"
    )


if __name__ == "__main__":
    main()
