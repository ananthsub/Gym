# Race/concurrency reproductions for nemo_gym.token_id_capture (branch pr-2181).
#
# Experiments:
#   1. F2: concurrent FileLineageStore.resolve on a warm cache poisoning by_fingerprint.
#   2. Freeze-vs-put race (expected safe).
#   3. Multi-process read-after-write soak (expected safe).
#   4. Executor-starvation throughput probe.
#
# Run:  .venv/bin/python simulation/race_experiments.py
# Writes simulation/race_results.json.

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nemo_gym.token_id_capture.lineage import FileLineageStore, stamp_continuation  # noqa: E402
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage  # noqa: E402
from nemo_gym.token_id_capture.sink import (  # noqa: E402
    CaptureContext,
    capture_tokens,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.store import TokenCaptureStore  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "race_results.json"
SCRATCH = Path(tempfile.mkdtemp(prefix="race-exp-"))


def user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_item(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def make_entry(
    rollout_id: str,
    call_id: str,
    request_items: list[dict],
    output_text: str,
    n_tokens: int = 32,
    seed: int = 0,
) -> TokenEntry:
    gen = [(seed * 7919 + i) % 50000 for i in range(n_tokens)]
    entry = TokenEntry(
        rollout_id=rollout_id,
        model_call_id=call_id,
        prompt_token_ids=[1, 2, 3],
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
        output_items=[assistant_item(output_text)],
        created_at=time.time(),
    )
    stamp_continuation(entry, list(request_items))
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return entry


def new_dir(tag: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=f"{tag}-", dir=SCRATCH))
    return path


# --------------------------------------------------------------------------------------
# Experiment 1: F2 — concurrent resolve poisoning the warm-cache fingerprint index.
# --------------------------------------------------------------------------------------


def f2_trial(threads: int, n_new: int, n_tokens: int, switch_interval: float | None, use_api: bool) -> dict:
    tmp = new_dir("f2")
    old_interval = sys.getswitchinterval()
    rollout = "r-f2"
    try:
        store = TokenCaptureStore(tmp)
        lineage_store = FileLineageStore(tmp)
        request = [user_msg("question-0")]

        # Call A exists; warm the cursor cache so later refreshes share the cached RolloutLineage.
        store.append(make_entry(rollout, "call-A", request, "warm-response", n_tokens, seed=999))
        asyncio.run(lineage_store.resolve(rollout, request + [assistant_item("warm-response")]))

        # Now a NEW tail of entries lands (all committed by the store, all distinct outputs).
        continuations: list[tuple[str, list[dict]]] = []
        for i in range(n_new):
            text = f"response-{i}"
            store.append(make_entry(rollout, f"call-{i}", request, text, n_tokens, seed=i))
            continuations.append((f"call-{i}", request + [assistant_item(text)]))

        target = continuations[0][1]
        barrier = threading.Barrier(threads)
        errors: list[str] = []

        def worker() -> None:
            try:
                barrier.wait()
                if use_api:
                    asyncio.run(lineage_store.resolve(rollout, target))
                else:
                    lineage_store._resolve(rollout, target)
            except Exception as error:  # noqa: BLE001
                errors.append(repr(error))

        if switch_interval is not None:
            sys.setswitchinterval(switch_interval)
        threads_list = [threading.Thread(target=worker) for _ in range(threads)]
        for thread in threads_list:
            thread.start()
        for thread in threads_list:
            thread.join()
        sys.setswitchinterval(old_interval)

        # Inspect the shared cached index for duplicate call ids.
        duplicate_lists: dict[str, list[str]] = {}
        overcount = None
        cached = lineage_store._cache.get(rollout)
        if cached is not None:
            lineage = cached[2]
            for fingerprint, call_ids in lineage.by_fingerprint.items():
                if len(call_ids) != len(set(call_ids)):
                    duplicate_lists[fingerprint[:12]] = list(call_ids)
            expected_tokens = sum(node.cum_len for node in lineage.by_call_id.values())
            overcount = lineage.total_tokens - expected_tokens

        # A request that SHOULD be RESOLVED now comes back UNRESOLVED(ambiguous) forever.
        ambiguous_calls: list[str] = []
        for call_id, continuation in continuations:
            result = lineage_store._resolve(rollout, continuation)
            if result.status == ParentResolutionStatus.UNRESOLVED and result.reason == "ambiguous":
                ambiguous_calls.append(call_id)

        return {
            "duplicate_fingerprint_lists": duplicate_lists,
            "ambiguous_calls": ambiguous_calls,
            "total_tokens_overcount": overcount,
            "worker_errors": errors,
            "reproduced": bool(duplicate_lists) or bool(ambiguous_calls),
        }
    finally:
        sys.setswitchinterval(old_interval)
        shutil.rmtree(tmp, ignore_errors=True)


def f2_continuous_trial(resolvers: int, n_entries: int, n_tokens: int) -> dict:
    """Writer appends entries while resolver threads hammer resolve, default GIL switch interval.

    This mirrors production: capture (put) and request-time resolve run concurrently, so every
    append opens a fresh tail that multiple warm-cache resolvers may parse simultaneously.
    """
    tmp = new_dir("f2c")
    rollout = "r-f2c"
    try:
        store = TokenCaptureStore(tmp)
        lineage_store = FileLineageStore(tmp)
        request = [user_msg("question-0")]
        store.append(make_entry(rollout, "call-A", request, "warm-response", n_tokens, seed=999))
        target = request + [assistant_item("warm-response")]
        asyncio.run(lineage_store.resolve(rollout, target))  # warm the cursor cache

        stop = threading.Event()
        errors: list[str] = []

        def resolver() -> None:
            try:
                while not stop.is_set():
                    lineage_store._resolve(rollout, target)
            except Exception as error:  # noqa: BLE001
                errors.append(repr(error))

        threads = [threading.Thread(target=resolver) for _ in range(resolvers)]
        for thread in threads:
            thread.start()
        continuations: list[tuple[str, list[dict]]] = []
        for i in range(n_entries):
            text = f"response-{i}"
            store.append(make_entry(rollout, f"call-{i}", request, text, n_tokens, seed=i))
            continuations.append((f"call-{i}", request + [assistant_item(text)]))
        stop.set()
        for thread in threads:
            thread.join()

        duplicate_lists: dict[str, list[str]] = {}
        overcount = None
        cached = lineage_store._cache.get(rollout)
        if cached is not None:
            lineage = cached[2]
            for fingerprint, call_ids in lineage.by_fingerprint.items():
                if len(call_ids) != len(set(call_ids)):
                    duplicate_lists[fingerprint[:12]] = list(call_ids)
            overcount = lineage.total_tokens - sum(node.cum_len for node in lineage.by_call_id.values())
        ambiguous_calls = []
        for call_id, continuation in continuations:
            result = lineage_store._resolve(rollout, continuation)
            if result.status == ParentResolutionStatus.UNRESOLVED and result.reason == "ambiguous":
                ambiguous_calls.append(call_id)
        return {
            "duplicate_fingerprint_lists": duplicate_lists,
            "ambiguous_calls": ambiguous_calls,
            "total_tokens_overcount": overcount,
            "worker_errors": errors,
            "reproduced": bool(duplicate_lists) or bool(ambiguous_calls),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def experiment_f2() -> dict:
    configs = [
        # (name, trials, threads, n_new, n_tokens, switch_interval, use_api)
        ("api_default_interval_8t", 60, 8, 20, 400, None, True),
        ("sync_default_interval_8t", 70, 8, 20, 400, None, False),
        ("sync_1us_interval_8t", 70, 8, 20, 400, 1e-6, False),
    ]
    out: dict = {"configs": {}}
    first_evidence = None
    for name, trials, threads, n_new, n_tokens, interval, use_api in configs:
        reproduced = 0
        ambiguous_total = 0
        example = None
        t0 = time.perf_counter()
        for _ in range(trials):
            result = f2_trial(threads, n_new, n_tokens, interval, use_api)
            if result["reproduced"]:
                reproduced += 1
                ambiguous_total += len(result["ambiguous_calls"])
                if example is None:
                    example = result
        out["configs"][name] = {
            "trials": trials,
            "threads": threads,
            "new_tail_entries": n_new,
            "tokens_per_entry": n_tokens,
            "switch_interval": interval,
            "through_async_api": use_api,
            "reproduced_trials": reproduced,
            "reproduction_rate": reproduced / trials,
            "ambiguous_resolves_total": ambiguous_total,
            "example_corruption": example,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        if example is not None and first_evidence is None:
            first_evidence = {"config": name, **example}
        print(f"  [F2] {name}: {reproduced}/{trials} reproduced", flush=True)

    # Continuous-traffic config: concurrent writer + resolvers at the DEFAULT switch interval.
    name = "continuous_default_interval_4resolvers"
    trials = 20
    reproduced = 0
    ambiguous_total = 0
    example = None
    t0 = time.perf_counter()
    for _ in range(trials):
        result = f2_continuous_trial(resolvers=4, n_entries=200, n_tokens=200)
        if result["reproduced"]:
            reproduced += 1
            ambiguous_total += len(result["ambiguous_calls"])
            if example is None:
                example = result
    out["configs"][name] = {
        "trials": trials,
        "resolver_threads": 4,
        "entries_appended_during_resolves": 200,
        "tokens_per_entry": 200,
        "switch_interval": None,
        "reproduced_trials": reproduced,
        "reproduction_rate": reproduced / trials,
        "ambiguous_resolves_total": ambiguous_total,
        "example_corruption": example,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    if example is not None and first_evidence is None:
        first_evidence = {"config": name, **example}
    print(f"  [F2] {name}: {reproduced}/{trials} reproduced", flush=True)
    out["reproduced_overall"] = any(c["reproduced_trials"] > 0 for c in out["configs"].values())
    out["first_evidence"] = first_evidence
    return out


# --------------------------------------------------------------------------------------
# Experiment 2: freeze-vs-put race (expected safe).
# --------------------------------------------------------------------------------------


def freeze_trial(iteration: int) -> list[str]:
    tmp = new_dir("frz")
    rollout = "r-frz"
    violations: list[str] = []
    try:
        store = TokenCaptureStore(tmp)
        request = [user_msg("q")]
        acked: list[str] = []
        writer_error: dict = {}

        def writer() -> None:
            for i in range(100):
                entry = make_entry(rollout, f"c{i}", request, f"o{i}", 8, seed=i)
                try:
                    store.append(entry)
                    acked.append(f"c{i}")
                except RuntimeError as error:
                    writer_error["message"] = str(error)
                    return
            writer_error["message"] = ""  # finished all 100 without hitting the freeze

        thread = threading.Thread(target=writer)
        thread.start()
        deadline = time.time() + 30
        while len(acked) < 15 and thread.is_alive() and time.time() < deadline:
            time.sleep(0.0005)
        acked_before_freeze = list(acked)
        snapshot = store.freeze_now(rollout)
        thread.join(timeout=30)

        snapshot_ids = {entry.model_call_id for entry in snapshot.entries}
        missing = set(acked_before_freeze) - snapshot_ids
        if missing:
            violations.append(f"acked-before-freeze entries missing from snapshot: {sorted(missing)}")
        if writer_error.get("message") is None:
            violations.append("writer thread hung")
        elif writer_error["message"] == "":
            violations.append("writer finished all 100 puts before freeze took effect (timing miss)")
        elif "frozen" not in writer_error["message"]:
            violations.append(f"unexpected writer error: {writer_error['message']}")

        # Everything acked at freeze time must equal the snapshot exactly (no extra committed puts).
        final_acked = set(acked)
        if final_acked != snapshot_ids:
            violations.append(f"snapshot ids != acked ids (snapshot-only={snapshot_ids - final_acked}, acked-only={final_acked - snapshot_ids})")

        if iteration % 2 == 1:
            # Post-freeze mutation: mark_incomplete must bump version, making drop refuse.
            asyncio.run(store.mark_incomplete(rollout, "late-call"))
            state = store._read_state(rollout)
            if int(state["version"]) <= snapshot.version:
                violations.append("mark_incomplete did not bump version")
            dropped = asyncio.run(store.drop(rollout, snapshot_id=snapshot.snapshot_id, version=snapshot.version))
            if dropped:
                violations.append("drop succeeded despite post-freeze mutation")
        else:
            dropped = asyncio.run(store.drop(rollout, snapshot_id=snapshot.snapshot_id, version=snapshot.version))
            if not dropped:
                violations.append("drop failed although nothing changed after the snapshot")
        return violations
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def experiment_freeze() -> dict:
    iterations = 50
    all_violations: list = []
    timing_misses = 0
    t0 = time.perf_counter()
    for i in range(iterations):
        violations = freeze_trial(i)
        real = [v for v in violations if "timing miss" not in v]
        if any("timing miss" in v for v in violations):
            timing_misses += 1
        if real:
            all_violations.append({"iteration": i, "violations": real})
    print(f"  [freeze] {iterations} iterations, {len(all_violations)} violating, {timing_misses} timing misses", flush=True)
    return {
        "iterations": iterations,
        "violating_iterations": all_violations,
        "violations_count": len(all_violations),
        "timing_misses_freeze_after_writer_done": timing_misses,
        "passed": not all_violations,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------------------
# Experiment 3: multi-process read-after-write soak.
# --------------------------------------------------------------------------------------

N_SOAK_ROLLOUTS = int(os.environ.get("RACE_SOAK_ROLLOUTS", "20"))
N_SOAK_CALLS = int(os.environ.get("RACE_SOAK_CALLS", "20"))


def _soak_conv(rollout_index: int, upto_call: int) -> list[dict]:
    """Conversation as sent for call `upto_call`: base user turn plus outputs of calls 1..upto_call-1."""
    return [user_msg(f"soak-q{rollout_index}")] + [assistant_item(f"r{rollout_index}-c{k}") for k in range(1, upto_call)]


def soak_worker(role: int, root: str, q_in, q_out, results_q) -> None:
    async def main() -> None:
        store = TokenCaptureStore(root)
        lineage_store = FileLineageStore(root)
        latencies: list[float] = []
        failures: list[str] = []
        resolves = 0
        for j in range(N_SOAK_ROLLOUTS):
            rollout = f"soak-{j}"
            for k in range(1, N_SOAK_CALLS + 1):
                mine = (k % 2 == 1) if role == 1 else (k % 2 == 0)
                if not mine:
                    continue
                if k > 1:
                    # Wait for the peer's ack that call k-1 is durably put.
                    ack = await asyncio.to_thread(q_in.get, True, 60)
                    if ack != (j, k - 1):
                        failures.append(f"protocol desync: expected {(j, k - 1)}, got {ack}")
                        results_q.put((role, latencies, failures, resolves))
                        return
                    conv = _soak_conv(j, k)
                    t0 = time.perf_counter()
                    resolution = await lineage_store.resolve(rollout, conv)
                    latencies.append(time.perf_counter() - t0)
                    resolves += 1
                    expected_parent = f"{rollout}-c{k - 1}"
                    if resolution.status != ParentResolutionStatus.RESOLVED:
                        failures.append(
                            f"rollout {j} call {k}: expected RESOLVED parent {expected_parent}, "
                            f"got {resolution.status.value}/{resolution.reason}"
                        )
                    elif resolution.match.model_call_id != expected_parent:
                        failures.append(
                            f"rollout {j} call {k}: wrong parent {resolution.match.model_call_id}"
                        )
                entry = make_entry(rollout, f"{rollout}-c{k}", _soak_conv(j, k), f"r{j}-c{k}", 16, seed=100 * j + k)
                await store.put(entry)
                q_out.put((j, k))
            # Drain the trailing ack when the peer wrote the rollout's last call,
            # so the next rollout starts in sync.
            last_mine = (N_SOAK_CALLS % 2 == 1) if role == 1 else (N_SOAK_CALLS % 2 == 0)
            if not last_mine:
                ack = await asyncio.to_thread(q_in.get, True, 60)
                if ack != (j, N_SOAK_CALLS):
                    failures.append(f"protocol desync on trailing ack: expected {(j, N_SOAK_CALLS)}, got {ack}")
                    results_q.put((role, latencies, failures, resolves))
                    return
        results_q.put((role, latencies, failures, resolves))

    asyncio.run(main())


def experiment_soak() -> dict:
    tmp = new_dir("soak")
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    q12 = ctx.Queue()
    q21 = ctx.Queue()
    results_q = ctx.Queue()
    p1 = ctx.Process(target=soak_worker, args=(1, str(tmp), q21, q12, results_q))
    p2 = ctx.Process(target=soak_worker, args=(2, str(tmp), q12, q21, results_q))
    p1.start()
    p2.start()
    results = []
    try:
        for _ in range(2):
            results.append(results_q.get(True, 300))
    finally:
        p1.join(timeout=30)
        p2.join(timeout=30)
        for p in (p1, p2):
            if p.is_alive():
                p.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
    latencies = sorted(lat for _, lats, _, _ in results for lat in lats)
    failures = [f for _, _, fails, _ in results for f in fails]
    total_resolves = sum(r for _, _, _, r in results)
    out = {
        "rollouts": N_SOAK_ROLLOUTS,
        "calls_per_rollout": N_SOAK_CALLS,
        "total_resolves": total_resolves,
        "failures": failures,
        "failures_count": len(failures),
        "passed": not failures and total_resolves == N_SOAK_ROLLOUTS * (N_SOAK_CALLS - 1),
        "resolve_latency_ms": {
            "p50": round(1000 * statistics.quantiles(latencies, n=100)[49], 3) if len(latencies) >= 100 else None,
            "p99": round(1000 * statistics.quantiles(latencies, n=100)[98], 3) if len(latencies) >= 100 else None,
            "mean": round(1000 * statistics.fmean(latencies), 3) if latencies else None,
            "max": round(1000 * max(latencies), 3) if latencies else None,
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    print(f"  [soak] resolves={total_resolves} failures={len(failures)}", flush=True)
    return out


# --------------------------------------------------------------------------------------
# Experiment 4: executor-starvation throughput probe.
# --------------------------------------------------------------------------------------


class SlowSink:
    """Wrap the file store, adding a 20 ms blocking sleep inside put (slow storage)."""

    def __init__(self, store: TokenCaptureStore) -> None:
        self._store = store

    def _slow_append(self, entry: TokenEntry) -> None:
        time.sleep(0.020)
        self._store.append(entry)

    async def put(self, entry: TokenEntry) -> None:
        await asyncio.to_thread(self._slow_append, entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        await self._store.mark_incomplete(rollout_id, model_call_id)

    async def close(self) -> None:
        pass


def _op_response(op_id: int) -> dict:
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"out-{op_id}"}],
                "prompt_token_ids": [1, 2, 3],
                "generation_token_ids": [10 + op_id % 7, 11, 12, 13],
                "generation_log_probs": [-0.1, -0.1, -0.1, -0.1],
            }
        ]
    }


async def _capture_op(sink, lineage_store, rollout: str, continuation: list[dict], op_id: int) -> bool:
    context = CaptureContext(
        rollout_id=rollout,
        model_call_id=f"op-{op_id}",
        token_sink=sink,
        lineage_store=lineage_store,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(continuation)
        resolved = (
            context.parent_resolution is not None
            and context.parent_resolution.status == ParentResolutionStatus.RESOLVED
        )
        await capture_tokens(_op_response(op_id), request_messages=continuation)
        return context.committed and resolved
    finally:
        reset_token_sink(token)


async def _run_probe(tag: str, k: int, total_ops: int, root: Path, slow_sink: bool) -> dict:
    store = TokenCaptureStore(root)
    lineage_store = FileLineageStore(root)
    sink = SlowSink(store) if slow_sink else store
    n_rollouts = 64
    rollouts = [f"{tag}-r{i}" for i in range(n_rollouts)]
    continuations: dict[str, list[dict]] = {}
    for rollout in rollouts:
        request = [user_msg(f"seed-q-{rollout}")]
        seed_text = f"seed-out-{rollout}"
        entry = make_entry(rollout, f"{rollout}-seed", request, seed_text, 8, seed=1)
        store.append(entry)
        continuations[rollout] = request + [assistant_item(seed_text)]

    semaphore = asyncio.Semaphore(k)
    latencies: list[float] = []
    ok_count = 0

    async def one(i: int) -> None:
        nonlocal ok_count
        async with semaphore:
            rollout = rollouts[i % n_rollouts]
            t0 = time.perf_counter()
            ok = await _capture_op(sink, lineage_store, rollout, continuations[rollout], i)
            latencies.append(time.perf_counter() - t0)
            if ok:
                ok_count += 1

    t0 = time.perf_counter()
    await asyncio.gather(*[one(i) for i in range(total_ops)])
    wall = time.perf_counter() - t0
    latencies.sort()
    incomplete_markers = len(list(root.glob("*.incomplete")))
    return {
        "k_in_flight": k,
        "total_ops": total_ops,
        "slow_sink_20ms": slow_sink,
        "ops_per_sec": round(total_ops / wall, 1),
        "op_latency_ms": {
            "p50": round(1000 * statistics.quantiles(latencies, n=100)[49], 2),
            "p99": round(1000 * statistics.quantiles(latencies, n=100)[98], 2),
            "max": round(1000 * max(latencies), 2),
        },
        "resolved_and_committed": ok_count,
        "incomplete_markers": incomplete_markers,
        "wall_s": round(wall, 2),
    }


def experiment_executor() -> dict:
    total_ops = 768
    points = []
    for k in (8, 32, 128, 512):
        root = new_dir(f"exec-k{k}")
        try:
            point = asyncio.run(_run_probe(f"k{k}", k, total_ops, root, slow_sink=False))
        finally:
            shutil.rmtree(root, ignore_errors=True)
        points.append(point)
        print(f"  [exec] K={k}: {point['ops_per_sec']} ops/s p99={point['op_latency_ms']['p99']}ms", flush=True)
    root = new_dir("exec-slow")
    try:
        slow_point = asyncio.run(_run_probe("slow", 128, 512, root, slow_sink=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"  [exec] K=128 slow sink: {slow_point['ops_per_sec']} ops/s p99={slow_point['op_latency_ms']['p99']}ms", flush=True)
    return {
        "default_executor_max_workers": min(32, (os.cpu_count() or 1) + 4),
        "throughput_curve": points,
        "slow_sink_k128": slow_point,
    }


# --------------------------------------------------------------------------------------


def main() -> None:
    results: dict = {
        "meta": {
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }
    print("Experiment 1: F2 concurrent-resolve index poisoning", flush=True)
    results["f2_concurrent_resolve"] = experiment_f2()
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

    print("Experiment 2: freeze-vs-put race", flush=True)
    results["freeze_vs_put"] = experiment_freeze()
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

    print("Experiment 3: multi-process read-after-write soak", flush=True)
    results["multiprocess_soak"] = experiment_soak()
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

    print("Experiment 4: executor-starvation probe", flush=True)
    results["executor_probe"] = experiment_executor()
    results["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}", flush=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
