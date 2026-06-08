# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Burst driver — reproduces the production connection-reset failure.

The production ``ClientPayloadError("Response payload is not completed:
<ContentLengthError ... ConnectionResetError(104, 'Connection reset by peer')>")``
is not a steady-state throughput ceiling and not the Ray-actor boundary. It is a
connection-lifecycle failure that needs a specific temporal pattern, mirrored
here from the training framework's ``AsyncTrajectoryCollector._process_batch``:

  for prompt in range(num_prompts_per_step):
      [optional spawn jitter]
      inflight_sema.acquire()                  # cap = num_prompts * max_age
      Thread(_run_prompt_group_worker).start() # fans out per_prompt_calls POSTs

with a per-cycle train floor (the step's compute runs while the burst is in
flight), a refit pause (weight sync — the collector stops, connections idle),
and overlapping cycles (workers are NOT joined per cycle), all over one global
aiohttp client configured exactly like Gym's. Idle keep-alive connections held
across the train/refit gap get reset by the server, surfacing as
``ClientPayloadError`` on the next read.

This driver is plain aiohttp (no Ray, no gym actor) on purpose — the reproducer
showed the actor is incidental; the failure is the client + burst + slow step.

Knobs of interest:
  --spawn-jitter-s        spread the burst over N seconds (the mitigation under test)
  --connector-keepalive-timeout / --force-close   keep-alive A/B (force-close disables reuse)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp


class _LoopRunner:
    """One asyncio loop in a dedicated thread (mirrors Gym's single-loop client)."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="aio-loop")
        self._thread.start()
        self.session: Optional[aiohttp.ClientSession] = None

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        if self.session is not None:
            try:
                self.submit(self.session.close()).result(timeout=5)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


def make_client(
    runner: _LoopRunner, *, limit: int, limit_per_host: int, keepalive_timeout: Optional[float], force_close: bool
) -> aiohttp.ClientSession:
    """aiohttp.ClientSession matching Gym's global client, with keep-alive knobs."""

    async def _build() -> aiohttp.ClientSession:
        connector_kwargs = {"limit": limit, "limit_per_host": limit_per_host, "force_close": force_close}
        # keepalive_timeout only applies when not force-closing; None keeps aiohttp's default.
        if keepalive_timeout is not None and not force_close:
            connector_kwargs["keepalive_timeout"] = keepalive_timeout
        connector = aiohttp.TCPConnector(**connector_kwargs)
        return aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(),  # infinite, like Gym
            cookie_jar=aiohttp.DummyCookieJar(),
        )

    runner.session = runner.submit(_build()).result()
    return runner.session


@dataclass
class CycleStats:
    cycle: int
    spawn_started_at: float = 0.0
    spawn_finished_at: float = 0.0
    train_finished_at: float = 0.0
    successes: int = 0
    failures: int = 0
    error_types: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict:
        with self.lock:
            total = self.successes + self.failures
            return {
                "cycle": self.cycle,
                "spawn_burst_s": round(self.spawn_finished_at - self.spawn_started_at, 3),
                "successes": self.successes,
                "failures": self.failures,
                "error_rate_pct": round(100.0 * self.failures / max(1, total), 3),
                "error_types": dict(self.error_types),
            }


async def _post(session: aiohttp.ClientSession, url: str) -> Tuple[bool, Optional[str]]:
    try:
        async with session.post(url, json={"responses_create_params": {"input": []}, "verifier_metadata": {}}) as resp:
            await resp.read()  # full body read — where ClientPayloadError surfaces
            return True, None
    except Exception as e:  # noqa: BLE001 — capture every failure class
        return False, type(e).__name__


def run_one_cycle(
    *,
    cycle_idx: int,
    runner: _LoopRunner,
    session: aiohttp.ClientSession,
    agent_urls: List[str],
    num_prompts_per_step: int,
    per_prompt_calls: int,
    inflight_sema: threading.Semaphore,
    refit_pause_cleared: threading.Event,
    spawn_jitter_s: float,
    train_floor_s: float,
    refit_s: float,
) -> Tuple[CycleStats, List[threading.Thread]]:
    """One generation-burst -> train -> refit cycle (mirrors _process_batch)."""
    stats = CycleStats(cycle=cycle_idx)
    while not refit_pause_cleared.wait(timeout=1.0):
        pass
    stats.spawn_started_at = time.monotonic()
    threads: List[threading.Thread] = []

    async def _gather(url: str):
        return await asyncio.gather(*(_post(session, url) for _ in range(per_prompt_calls)))

    def _worker() -> None:
        try:
            results = runner.submit(_gather(random.choice(agent_urls))).result()
            succ = sum(1 for ok, _ in results if ok)
            errs: Counter[str] = Counter(err or "Unknown" for ok, err in results if not ok)
            with stats.lock:
                stats.successes += succ
                stats.failures += sum(errs.values())
                stats.error_types.update(errs)
        finally:
            inflight_sema.release()

    per_iter_jitter = (spawn_jitter_s / num_prompts_per_step) if spawn_jitter_s > 0 else 0.0
    for _ in range(num_prompts_per_step):
        while not refit_pause_cleared.wait(timeout=1.0):
            pass
        if per_iter_jitter > 0:
            time.sleep(random.uniform(0.0, per_iter_jitter))
        while not inflight_sema.acquire(timeout=1.0):
            pass
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        threads.append(t)

    stats.spawn_finished_at = time.monotonic()
    time.sleep(train_floor_s)  # step compute runs while burst is in flight
    stats.train_finished_at = time.monotonic()
    # Refit pause: collector stops, in-flight/pooled connections idle.
    refit_pause_cleared.clear()
    time.sleep(refit_s)
    refit_pause_cleared.set()
    # Deliberately do NOT join threads — cycles overlap, like production.
    return stats, threads


def _parse_ports(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(p) for p in spec.split(",")]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-ports", required=True, help="'5018-5033' or '5018,5019'")
    p.add_argument("--agent-host", default="127.0.0.1")
    p.add_argument("--num-prompts-per-step", type=int, default=512)
    p.add_argument("--per-prompt-calls", type=int, default=16)
    p.add_argument("--num-cycles", type=int, default=6)
    p.add_argument("--train-floor-s", type=float, default=30.0)
    p.add_argument("--refit-s", type=float, default=5.0)
    p.add_argument("--spawn-jitter-s", type=float, default=0.0)
    p.add_argument("--connector-limit", type=int, default=4096)
    p.add_argument("--connector-limit-per-host", type=int, default=4096)
    p.add_argument("--connector-keepalive-timeout", type=float, default=None)
    p.add_argument("--force-close", action="store_true", help="Disable keep-alive reuse (new connection per request).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent_urls = [f"http://{args.agent_host}:{port}/run" for port in _parse_ports(args.agent_ports)]
    print(
        f"[burst] {len(agent_urls)} agents | N={args.num_prompts_per_step} x {args.per_prompt_calls} calls "
        f"= {args.num_prompts_per_step * args.per_prompt_calls} in-flight | jitter={args.spawn_jitter_s}s "
        f"force_close={args.force_close} keepalive={args.connector_keepalive_timeout}",
        flush=True,
    )

    runner = _LoopRunner()
    session = make_client(
        runner,
        limit=args.connector_limit,
        limit_per_host=args.connector_limit_per_host,
        keepalive_timeout=args.connector_keepalive_timeout,
        force_close=args.force_close,
    )
    inflight_sema = threading.Semaphore(args.num_prompts_per_step)  # max_trajectory_age_steps = 1
    refit_pause_cleared = threading.Event()
    refit_pause_cleared.set()

    all_stats: List[CycleStats] = []
    all_threads: List[threading.Thread] = []
    try:
        for cycle in range(args.num_cycles):
            s, ts = run_one_cycle(
                cycle_idx=cycle,
                runner=runner,
                session=session,
                agent_urls=agent_urls,
                num_prompts_per_step=args.num_prompts_per_step,
                per_prompt_calls=args.per_prompt_calls,
                inflight_sema=inflight_sema,
                refit_pause_cleared=refit_pause_cleared,
                spawn_jitter_s=args.spawn_jitter_s,
                train_floor_s=args.train_floor_s,
                refit_s=args.refit_s,
            )
            all_stats.append(s)
            all_threads.extend(ts)
            print(f"[burst] cycle {cycle} snapshot: {json.dumps(s.to_dict())}", flush=True)
        for t in all_threads:
            t.join(timeout=120.0)
    finally:
        runner.stop()

    total_succ = sum(s.successes for s in all_stats)
    total_fail = sum(s.failures for s in all_stats)
    err_types: Counter[str] = Counter()
    for s in all_stats:
        err_types.update(s.error_types)
    summary = {
        "successes": total_succ,
        "failures": total_fail,
        "failure_rate": round(total_fail / max(1, total_succ + total_fail), 6),
        "error_types": dict(err_types),
        "num_cycles": len(all_stats),
        "per_cycle": [s.to_dict() for s in all_stats],
        "config": {
            "num_prompts_per_step": args.num_prompts_per_step,
            "per_prompt_calls": args.per_prompt_calls,
            "in_flight": args.num_prompts_per_step * args.per_prompt_calls,
            "spawn_jitter_s": args.spawn_jitter_s,
            "train_floor_s": args.train_floor_s,
            "refit_s": args.refit_s,
            "force_close": args.force_close,
            "connector_keepalive_timeout": args.connector_keepalive_timeout,
            "num_agents": len(agent_urls),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[burst] DONE failure_rate={summary['failure_rate']} errs={summary['error_types']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
