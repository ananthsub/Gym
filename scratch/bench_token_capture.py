# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark + profile the training token-capture write path.

Drives the real async hot path -- N concurrent rollouts, each doing M sequential
model calls via ``await asyncio.to_thread(append, entry)`` -- against several sink
variants so we can see where the time goes:

  current     TokenCaptureStore.append (global lock + flock + fsync per call)
  no_fsync    same, but os.fsync patched to a no-op  (isolates fsync cost)
  no_lock     no global lock, no flock, no fsync -- open('a')+write (cmunley-style)
  memory      dict[rollout_id] -> list[entry]        (pure in-memory floor)

Run from the repo root:  uv run python scratch/bench_token_capture.py
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import os
import pstats
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import orjson

from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


def make_entries(rollouts: int, calls: int) -> list[list[TokenEntry]]:
    """Pre-build realistic multi-turn entries (construction is not timed).

    Prompt grows each turn (history accumulates); generation ~500 tokens; one
    content-bearing output item, matching what the real sink writes.
    """
    per_rollout: list[list[TokenEntry]] = []
    for r in range(rollouts):
        rid = f"{r}-0"
        entries = []
        for c in range(calls):
            prompt_len = 1000 + c * 600
            gen_len = 500
            entries.append(
                TokenEntry(
                    rollout_id=rid,
                    model_call_id=f"{rid}-c{c}-{os.urandom(4).hex()}",
                    model="qwen2.5-7b",
                    prompt_token_ids=list(range(prompt_len)),
                    generation_token_ids=list(range(gen_len)),
                    generation_log_probs=[-0.12] * gen_len,
                    output_items=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "x" * 400}],
                        }
                    ],
                    created_at=time.time(),
                )
            )
        per_rollout.append(entries)
    return per_rollout


# --- sink variants: each returns an append(entry) callable -----------------------------------


def sink_current(root: Path) -> Callable[[TokenEntry], None]:
    store = TokenCaptureStore(root)
    return store.append


def sink_no_fsync(root: Path) -> Callable[[TokenEntry], None]:
    store = TokenCaptureStore(root)
    return store.append  # os.fsync is monkeypatched off around the run


def sink_no_lock(root: Path) -> Callable[[TokenEntry], None]:
    # cmunley-style: open('a')+write, no global lock, no flock, no fsync.
    root.mkdir(parents=True, exist_ok=True)

    def append(entry: TokenEntry) -> None:
        line = orjson.dumps(entry.model_dump(), option=orjson.OPT_APPEND_NEWLINE)
        with open(root / f"{entry.rollout_id}.tokens.jsonl", "ab") as handle:
            handle.write(line)

    return append


def sink_fsync_nolock(root: Path) -> Callable[[TokenEntry], None]:
    # Durable (fsync) but no global in-process lock and no flock: per-rollout files let
    # concurrent fsyncs run in parallel across the threadpool instead of serializing.
    root.mkdir(parents=True, exist_ok=True)

    def append(entry: TokenEntry) -> None:
        line = orjson.dumps(entry.model_dump(), option=orjson.OPT_APPEND_NEWLINE)
        fd = os.open(root / f"{entry.rollout_id}.tokens.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    return append


def sink_memory(_root: Path) -> Callable[[TokenEntry], None]:
    buf: dict[str, list[TokenEntry]] = defaultdict(list)

    def append(entry: TokenEntry) -> None:
        buf[entry.rollout_id].append(entry)

    return append


VARIANTS = {
    "current": sink_current,
    "fsync_nolock": sink_fsync_nolock,
    "no_fsync": sink_no_fsync,
    "no_lock": sink_no_lock,
    "memory": sink_memory,
}

# When >0, model a shared/network filesystem by adding fixed latency to every fsync.
_FSYNC_LATENCY_S = 0.0
_REAL_FSYNC = os.fsync


def _slow_fsync(fd: int) -> None:
    _REAL_FSYNC(fd)
    if _FSYNC_LATENCY_S:
        time.sleep(_FSYNC_LATENCY_S)


async def _drive(append: Callable[[TokenEntry], None], per_rollout: list[list[TokenEntry]]) -> None:
    async def one_rollout(entries: list[TokenEntry]) -> None:
        for entry in entries:
            await asyncio.to_thread(append, entry)

    await asyncio.gather(*(one_rollout(e) for e in per_rollout))


def run_variant(name: str, per_rollout: list[list[TokenEntry]]) -> dict:
    total = sum(len(e) for e in per_rollout)
    with tempfile.TemporaryDirectory(prefix=f"bench_{name}_") as tmp:
        # Variants that fsync go through os.fsync, so patch it to model slow storage (and to a
        # no-op for the no_fsync variant). no_lock/memory never call fsync.
        if name == "no_fsync":
            os.fsync = lambda fd: None  # type: ignore[assignment]
        else:
            os.fsync = _slow_fsync  # type: ignore[assignment]
        append = VARIANTS[name](Path(tmp))
        try:
            t0 = time.perf_counter()
            asyncio.run(_drive(append, per_rollout))
            dt = time.perf_counter() - t0
        finally:
            os.fsync = _REAL_FSYNC  # type: ignore[assignment]
    return {"name": name, "appends": total, "wall_s": dt, "per_s": total / dt, "ms_each": 1000 * dt / total}


def profile_current(per_rollout: list[list[TokenEntry]]) -> str:
    with tempfile.TemporaryDirectory(prefix="bench_prof_") as tmp:
        append = sink_current(Path(tmp))
        pr = cProfile.Profile()
        pr.enable()
        asyncio.run(_drive(append, per_rollout))
        pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    return s.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=256)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument(
        "--fsync-latency-ms", type=float, default=0.0, help="model a shared/network FS by adding latency to fsync"
    )
    args = ap.parse_args()

    global _FSYNC_LATENCY_S
    _FSYNC_LATENCY_S = args.fsync_latency_ms / 1000.0

    per_rollout = make_entries(args.rollouts, args.calls)
    total = sum(len(e) for e in per_rollout)
    print(f"workload: {args.rollouts} rollouts x {args.calls} calls = {total} appends")
    print(f"threadpool default_max_workers ~= {min(32, (os.cpu_count() or 1) + 4)}  cpu={os.cpu_count()}")
    print(f"injected fsync latency = {args.fsync_latency_ms} ms\n")

    print(f"{'variant':<14}{'appends':>9}{'wall_s':>10}{'appends/s':>13}{'ms/append':>12}")
    baseline = None
    for name in ("current", "fsync_nolock", "no_fsync", "no_lock", "memory"):
        r = run_variant(name, per_rollout)
        if baseline is None:
            baseline = r["per_s"]
        speedup = r["per_s"] / baseline
        print(
            f"{r['name']:<14}{r['appends']:>9}{r['wall_s']:>10.3f}{r['per_s']:>13.0f}"
            f"{r['ms_each']:>12.3f}   {speedup:>5.1f}x vs current"
        )

    print("\n=== cProfile of `current` (sorted by tottime) ===")
    print(profile_current(per_rollout))


if __name__ == "__main__":
    main()
