# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark suite for NeMo-Gym's token-id-capture stack (branch pr-2181).

Run from the worktree root:
    /Users/ananth/dev/Gym-tokidcap/.venv/bin/python simulation/bench_capture.py

Writes machine-readable results to simulation/bench_results.json.
All scratch data goes to a throwaway temp directory outside the repo.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import random
import resource
import statistics
import struct
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import orjson

from nemo_gym.token_id_capture.builder import project_main_chain_response, run_builder
from nemo_gym.token_id_capture.consumer import trajectories_for_rollout
from nemo_gym.token_id_capture.lineage import (
    FileLineageStore,
    assistant_fingerprint,
    conversation_digest,
    stamp_continuation,
)
from nemo_gym.token_id_capture.records import (
    _DIGEST_DOMAIN,
    DIGEST_VERSION,
    ParentResolutionStatus,
    TokenEntry,
    compute_digest,
    cumulative_tokens,
    encode_token_ids,
    stamp_lineage,
)
from nemo_gym.token_id_capture.store import TokenCaptureStore


REPO = Path("/Users/ananth/dev/Gym-tokidcap")
RESULTS_PATH = REPO / "simulation" / "bench_results.json"

# ---------------------------------------------------------------- data helpers

_WORDS = None


def _words(rng: random.Random) -> list[str]:
    global _WORDS
    if _WORDS is None:
        _WORDS = [
            "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10))) for _ in range(4000)
        ]
    return _WORDS


def rand_tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(50_000) for _ in range(n)]


def make_text(rng: random.Random, lo: int = 200, hi: int = 800) -> str:
    words = _words(rng)
    target = rng.randint(lo, hi)
    parts: list[str] = []
    size = 0
    while size < target:
        w = rng.choice(words)
        parts.append(w)
        size += len(w) + 1
    return " ".join(parts)


def make_tool_call(rng: random.Random, call_id: str) -> dict:
    # ~1KB of JSON arguments.
    args = {
        "query": make_text(rng, 300, 500),
        "filters": {"lang": "en", "top_k": rng.randint(1, 50), "tags": [make_text(rng, 20, 40) for _ in range(6)]},
        "cursor": hashlib.sha256(call_id.encode()).hexdigest(),
    }
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "search_documents", "arguments": json.dumps(args)},
    }


def make_conversation(rng: random.Random, turns: int) -> list[dict]:
    """Chat-style conversation with realistic message sizes and occasional tool calls."""
    messages: list[dict] = [{"role": "system", "content": make_text(rng, 300, 600)}]
    i = 0
    while len(messages) < turns:
        if i % 2 == 0:
            messages.append({"role": "user", "content": make_text(rng)})
        else:
            if rng.random() < 0.15 and len(messages) + 2 <= turns:
                call_id = f"call_{i}_{rng.randrange(1 << 30):08x}"
                messages.append(
                    {"role": "assistant", "content": make_text(rng, 100, 300), "tool_calls": [make_tool_call(rng, call_id)]}
                )
                messages.append({"role": "tool", "tool_call_id": call_id, "content": make_text(rng, 300, 800)})
            else:
                messages.append({"role": "assistant", "content": make_text(rng)})
        i += 1
    return messages[:turns]


def make_entry(
    rng: random.Random,
    rollout_id: str,
    call_id: str,
    prompt: list[int],
    gen: list[int],
    request_items: list[dict] | None = None,
    assistant_text: str | None = None,
) -> TokenEntry:
    text = assistant_text if assistant_text is not None else make_text(rng)
    entry = TokenEntry(
        rollout_id=rollout_id,
        model_call_id=call_id,
        model="bench_model",
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
        output_items=[{"type": "message", "role": "assistant", "content": text}],
        token_item_index=0,
        created_at=time.time(),
    )
    if request_items is not None:
        stamp_continuation(entry, request_items)
    return entry


def build_chained_rollout(
    rng: random.Random,
    rollout_id: str,
    n_calls: int,
    root_prompt: int,
    gen_len: int,
    interstitial: int,
    store: TokenCaptureStore | None = None,
) -> tuple[list[TokenEntry], list[dict]]:
    """Chained rollout: call k's prompt extends call k-1's cumulative tokens.

    Returns (entries, final_conversation_messages). Entries are stamped with
    continuation metadata and lineage (ROOT / RESOLVED chain), and optionally
    appended to a store.
    """
    entries: list[TokenEntry] = []
    messages: list[dict] = [
        {"role": "system", "content": make_text(rng, 300, 600)},
        {"role": "user", "content": make_text(rng)},
    ]
    prompt = rand_tokens(rng, root_prompt)
    parent_id: str | None = None
    for k in range(n_calls):
        call_id = f"{rollout_id}-call-{k:03d}"
        gen = rand_tokens(rng, gen_len)
        entry = make_entry(rng, rollout_id, call_id, prompt, gen, request_items=messages)
        if parent_id is None:
            stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
        else:
            stamp_lineage(entry, parent_id, parent_resolution=ParentResolutionStatus.RESOLVED)
        if store is not None:
            store.append(entry)
        entries.append(entry)
        messages = messages + list(entry.output_items) + [{"role": "user", "content": make_text(rng)}]
        prompt = cumulative_tokens(entry) + rand_tokens(rng, interstitial)
        parent_id = call_id
    return entries, messages


# ---------------------------------------------------------------- measurement

def pctl(values_ms: list[float], p: float) -> float:
    values = sorted(values_ms)
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]


def summarize(values_ms: list[float]) -> dict:
    return {
        "p50_ms": round(pctl(values_ms, 50), 4),
        "p99_ms": round(pctl(values_ms, 99), 4),
        "mean_ms": round(statistics.fmean(values_ms), 4),
        "reps": len(values_ms),
    }


# ---------------------------------------------------------------- benchmarks

def bench_append(rng: random.Random) -> dict:
    out: dict = {}
    reps_by_size = {1_000: 60, 8_000: 40, 64_000: 15}
    for size, reps in reps_by_size.items():
        with tempfile.TemporaryDirectory(prefix="ngbench-append-") as tmp:
            store = TokenCaptureStore(tmp)
            # Pre-build all entries so timing covers only store.append.
            entries = []
            for i in range(reps + 3):
                rid = f"r{size}-{i:03d}"
                entry = make_entry(
                    rng, rid, f"{rid}-c0", rand_tokens(rng, size), rand_tokens(rng, 128),
                    request_items=[{"role": "user", "content": make_text(rng)}],
                )
                stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
                entries.append(entry)
            for entry in entries[:3]:  # warmup
                store.append(entry)
            samples = []
            for entry in entries[3:]:
                t0 = time.perf_counter()
                store.append(entry)
                samples.append((time.perf_counter() - t0) * 1000)
            out[f"prompt_{size}"] = summarize(samples)
    return out


def bench_append_growth(rng: random.Random) -> dict:
    n_calls = 200
    with tempfile.TemporaryDirectory(prefix="ngbench-growth-") as tmp:
        store = TokenCaptureStore(tmp)
        rid = "growth"
        jsonl = store.path_for(rid)
        state = store.state_path_for(rid)
        latencies: list[float] = []
        total_written = 0
        prev_jsonl = 0
        prompt = rand_tokens(rng, 300)
        messages: list[dict] = [{"role": "user", "content": make_text(rng)}]
        parent_id: str | None = None
        for k in range(1, n_calls + 1):
            call_id = f"{rid}-call-{k:03d}"
            gen = rand_tokens(rng, 100)
            entry = make_entry(rng, rid, call_id, prompt, gen, request_items=messages)
            if parent_id is None:
                stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
            else:
                stamp_lineage(entry, parent_id, parent_resolution=ParentResolutionStatus.RESOLVED)
            t0 = time.perf_counter()
            store.append(entry)
            latencies.append((time.perf_counter() - t0) * 1000)
            new_jsonl = jsonl.stat().st_size
            total_written += (new_jsonl - prev_jsonl) + state.stat().st_size  # state is rewritten whole
            prev_jsonl = new_jsonl
            messages = messages + list(entry.output_items) + [{"role": "user", "content": make_text(rng)}]
            prompt = cumulative_tokens(entry) + rand_tokens(rng, 200)  # next prompt ~ (k+1)*300
            parent_id = call_id
        file_bytes = sum(p.stat().st_size for p in Path(tmp).iterdir())
        return {
            "append_ms_call_1": round(latencies[0], 4),
            "append_ms_call_50": round(latencies[49], 4),
            "append_ms_call_100": round(latencies[99], 4),
            "append_ms_call_200": round(latencies[199], 4),
            "total_ms_all_200": round(sum(latencies), 2),
            "final_dir_bytes": file_bytes,
            "final_jsonl_bytes": prev_jsonl,
            "cumulative_bytes_written": total_written,  # jsonl deltas + full state rewrite each call
        }


def bench_hashes(rng: random.Random) -> dict:
    out: dict = {}
    for turns, reps in ((10, 200), (50, 60), (200, 20)):
        conv = make_conversation(rng, turns)
        size_bytes = len(orjson.dumps(conv))
        for name, fn in (("fingerprint", assistant_fingerprint), ("conversation_digest", conversation_digest)):
            for _ in range(3):
                fn(conv)
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn(conv)
                samples.append((time.perf_counter() - t0) * 1000)
            stats = summarize(samples)
            stats["conv_bytes"] = size_bytes
            stats["mb_per_s"] = round((size_bytes / 1e6) / (stats["mean_ms"] / 1e3), 1)
            out[f"{name}_{turns}turns"] = stats
    return out


def bench_stamp_o_l2(rng: random.Random) -> dict:
    n_calls = 100
    totals = []
    per_call_first = per_call_last = 0.0
    for _rep in range(3):
        messages: list[dict] = [
            {"role": "system", "content": make_text(rng, 300, 600)},
            {"role": "user", "content": make_text(rng)},
        ]
        total = 0.0
        for k in range(n_calls):
            entry = make_entry(rng, "stamp", f"c{k}", rand_tokens(rng, 32), rand_tokens(rng, 32))
            t0 = time.perf_counter()
            stamp_continuation(entry, messages)
            dt = time.perf_counter() - t0
            total += dt
            if k == 0:
                per_call_first = dt * 1000
            if k == n_calls - 1:
                per_call_last = dt * 1000
            messages = messages + list(entry.output_items) + [{"role": "user", "content": make_text(rng)}]
            if rng.random() < 0.15:
                call_id = f"tc_{k}"
                messages.append({"role": "assistant", "content": None, "tool_calls": [make_tool_call(rng, call_id)]})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": make_text(rng, 300, 800)})
        totals.append(total)
    return {
        "cumulative_s_100_calls": round(statistics.median(totals), 4),
        "first_call_ms": round(per_call_first, 4),
        "last_call_ms": round(per_call_last, 4),
        "final_conversation_bytes": len(orjson.dumps(messages)),
    }


def _encode_token_ids_vectorized(token_ids: list[int]) -> bytes:
    if any(t < 0 for t in token_ids):
        raise ValueError("token ids must be non-negative")
    return struct.pack(">BQ", DIGEST_VERSION, len(token_ids)) + struct.pack(f">{len(token_ids)}Q", *token_ids)


def bench_digest_tokens(rng: random.Random) -> dict:
    out: dict = {}
    for size, reps in ((8_000, 100), (65_000, 30)):
        ids = rand_tokens(rng, size)
        baseline_bytes = encode_token_ids(ids)
        vector_bytes = _encode_token_ids_vectorized(ids)
        identical = baseline_bytes == vector_bytes
        vec_digest = hashlib.sha256(_DIGEST_DOMAIN + vector_bytes).hexdigest()
        digest_identical = vec_digest == compute_digest(ids)

        def timeit(fn, reps=reps):
            for _ in range(3):
                fn()
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1000)
            return summarize(samples)

        current = timeit(lambda: compute_digest(ids))
        vectorized = timeit(lambda: hashlib.sha256(_DIGEST_DOMAIN + _encode_token_ids_vectorized(ids)).hexdigest())
        encode_only = timeit(lambda: encode_token_ids(ids))
        encode_vec_only = timeit(lambda: _encode_token_ids_vectorized(ids))
        out[f"tokens_{size}"] = {
            "compute_digest": current,
            "vectorized_digest": vectorized,
            "encode_current": encode_only,
            "encode_vectorized": encode_vec_only,
            "bytes_identical": identical,
            "digest_identical": digest_identical,
            "digest_speedup_x": round(current["mean_ms"] / vectorized["mean_ms"], 2),
            "encode_speedup_x": round(encode_only["mean_ms"] / encode_vec_only["mean_ms"], 2),
        }
    return out


def bench_resolve(rng: random.Random) -> dict:
    async def _run() -> dict:
        with tempfile.TemporaryDirectory(prefix="ngbench-resolve-") as tmp:
            store = TokenCaptureStore(tmp)
            rid = "resolve-rollout"
            entries, messages = build_chained_rollout(
                rng, rid, n_calls=50, root_prompt=8_000, gen_len=128, interstitial=32, store=store
            )
            jsonl_bytes = store.path_for(rid).stat().st_size
            # ``messages`` already ends with the next user turn; it continues the last call.
            cold_samples = []
            for _ in range(5):
                fresh = FileLineageStore(tmp)
                t0 = time.perf_counter()
                res = await fresh.resolve(rid, messages)
                cold_samples.append((time.perf_counter() - t0) * 1000)
                assert res.status == ParentResolutionStatus.RESOLVED, res
                assert res.match.model_call_id == entries[-1].model_call_id
            warm_store = FileLineageStore(tmp)
            await warm_store.resolve(rid, messages)  # populate cache
            warm_samples = []
            for _ in range(20):
                t0 = time.perf_counter()
                res = await warm_store.resolve(rid, messages)
                warm_samples.append((time.perf_counter() - t0) * 1000)
                assert res.status == ParentResolutionStatus.RESOLVED
            return {
                "cold": summarize(cold_samples),
                "warm": summarize(warm_samples),
                "n_entries": 50,
                "jsonl_bytes": jsonl_bytes,
            }

    return asyncio.run(_run())


def bench_eol(rng: random.Random) -> dict:
    with tempfile.TemporaryDirectory(prefix="ngbench-eol-") as tmp:
        store = TokenCaptureStore(tmp)
        rid = "eol-rollout"
        entries, _ = build_chained_rollout(
            rng, rid, n_calls=50, root_prompt=4_000, gen_len=600, interstitial=600, store=store
        )
        final_context = len(cumulative_tokens(entries[-1]))
        jsonl_bytes = store.path_for(rid).stat().st_size
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
        t0 = time.perf_counter()
        built = trajectories_for_rollout(rid, [Path(tmp)])
        wall_ms = (time.perf_counter() - t0) * 1000
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert built is not None and built["rebuilt_response"] is not None, built
        assert built["mask_sample"] is False, built
        assert built["metrics"]["chains"] == 1 and built["metrics"]["roots"] == 1
        # Phase attribution on the now-frozen rollout (freeze_now is idempotent).
        t0 = time.perf_counter()
        snapshot = store.freeze_now(rid)
        t1 = time.perf_counter()
        out = run_builder(list(snapshot.entries))
        t2 = time.perf_counter()
        project_main_chain_response(rid, out, model="bench_model")
        t3 = time.perf_counter()
        return {
            "wall_ms": round(wall_ms, 2),
            "phase_freeze_read_ms": round((t1 - t0) * 1000, 1),
            "phase_run_builder_ms": round((t2 - t1) * 1000, 1),
            "phase_projection_ms": round((t3 - t2) * 1000, 1),
            "peak_rss_delta_mb": round((rss_after - rss_before) / 1e6, 1),
            "note": "ru_maxrss is a process peak; delta can under-report if an earlier bench peaked higher",
            "final_context_tokens": final_context,
            "n_calls": 50,
            "jsonl_bytes": jsonl_bytes,
        }


def bench_row_size(rng: random.Random) -> dict:
    out: dict = {}
    for n_calls in (10, 25, 50):
        rid = f"rowsize-{n_calls}"
        entries, _ = build_chained_rollout(
            rng, rid, n_calls=n_calls, root_prompt=2_000, gen_len=400, interstitial=200
        )
        raw_bytes = sum(len(orjson.dumps(e.model_dump(mode="json"))) for e in entries)
        built = run_builder(entries)
        response = project_main_chain_response(rid, built, model="bench_model")
        rebuilt_bytes = len(orjson.dumps(response))
        out[f"calls_{n_calls}"] = {
            "raw_entries_bytes": raw_bytes,
            "rebuilt_response_bytes": rebuilt_bytes,
            "ratio_rebuilt_over_raw": round(rebuilt_bytes / raw_bytes, 2),
            "final_context_tokens": len(cumulative_tokens(entries[-1])),
        }
    return out


def bench_serving_overhead(rng: random.Random) -> dict:
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
    from nemo_gym.server_utils import ServerClient
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

    def _completion(prompt: list[int], generation: list[int], content: str) -> dict:
        return {
            "id": f"completion-{hashlib.sha1(content.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": 0,
            "model": "dummy_model",
            "prompt_token_ids": prompt,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": generation,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{t}", "logprob": -0.1, "bytes": None, "top_logprobs": []}
                            for t in generation
                        ]
                    },
                }
            ],
        }

    n_turns = 10
    n_rollouts = 12
    prompt_root = 2_000
    gen_len = 64
    interstitial = 16

    def make_chain_completions() -> list[dict]:
        prompt = rand_tokens(rng, prompt_root)
        completions = []
        for k in range(n_turns):
            gen = rand_tokens(rng, gen_len)
            completions.append(_completion(prompt, gen, make_text(rng, 200, 400)))
            prompt = prompt + gen + rand_tokens(rng, interstitial)
        return completions

    class Engine:
        def __init__(self) -> None:
            self.queue: list[dict] = []

    def build_client(capture_dir: str | None) -> tuple[TestClient, Engine]:
        engine = Engine()

        async def serve_completion(**kwargs) -> dict:
            return engine.queue.pop(0)

        mock = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock.create_chat_completion = AsyncMock(side_effect=serve_completion)
        mock.create_tokenize = AsyncMock()
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="vllm_model",
            base_url="http://localhost:9999/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            return_token_id_information=True,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            supply_prefix_token_ids=True,
        )
        global_config = {"token_id_capture": {"enabled": True, "dir": capture_dir}} if capture_dir else {}
        model = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config))
        model._clients = [mock]
        return TestClient(model.setup_webserver()), engine

    def run_rollout(client: TestClient, engine: Engine, url_prefix: str) -> list[float]:
        engine.queue = make_chain_completions()
        messages = [{"role": "user", "content": make_text(rng, 200, 400)}]
        samples = []
        for _turn in range(n_turns):
            t0 = time.perf_counter()
            response = client.post(f"{url_prefix}/v1/chat/completions", json={"messages": messages})
            samples.append((time.perf_counter() - t0) * 1000)
            assert response.status_code == 200, response.text
            answer = response.json()["choices"][0]["message"]
            messages = messages + [
                {"role": "assistant", "content": answer.get("content")},
                {"role": "user", "content": make_text(rng, 200, 400)},
            ]
        return samples

    with tempfile.TemporaryDirectory(prefix="ngbench-serving-") as tmp:
        captured_client, captured_engine = build_client(tmp)
        plain_client, plain_engine = build_client(None)

        # Warmup one rollout on each path.
        run_rollout(captured_client, captured_engine, "/ng-rollout/warmup-cap/training-token-capture")
        run_rollout(plain_client, plain_engine, "")

        captured_ms: list[float] = []
        plain_ms: list[float] = []
        for i in range(n_rollouts):
            captured_ms += run_rollout(
                captured_client, captured_engine, f"/ng-rollout/bench-cap-{i:02d}/training-token-capture"
            )
            plain_ms += run_rollout(plain_client, plain_engine, "")

        # Sanity: capture actually wrote chained entries.
        store = TokenCaptureStore(tmp)
        entries = store.read_entries("bench-cap-00")
        assert len(entries) == n_turns
        resolved = sum(1 for e in entries if e.parent_resolution == ParentResolutionStatus.RESOLVED)
        assert entries[0].parent_resolution == ParentResolutionStatus.ROOT

        captured_stats = summarize(captured_ms)
        plain_stats = summarize(plain_ms)
        return {
            "captured": captured_stats,
            "plain": plain_stats,
            "p50_delta_ms_per_call": round(captured_stats["p50_ms"] - plain_stats["p50_ms"], 3),
            "mean_delta_ms_per_call": round(captured_stats["mean_ms"] - plain_stats["mean_ms"], 3),
            "n_turns": n_turns,
            "n_rollouts": n_rollouts,
            "resolved_parent_calls_rollout0": resolved,
        }


# ---------------------------------------------------------------- red flags

def evaluate_flags(results: dict) -> dict:
    flags = {}

    append_p99 = max(v["p99_ms"] for v in results["bench_append"].values())
    flags["append_p99_over_5ms"] = {"value_ms": append_p99, "flag": append_p99 > 5.0}

    min_mbps = min(v["mb_per_s"] for v in results["bench_hashes"].values())
    flags["hashes_under_100MBps"] = {"min_mb_per_s": min_mbps, "flag": min_mbps < 100.0}

    d65 = results["bench_digest_tokens"]["tokens_65000"]["compute_digest"]["p50_ms"]
    flags["digest_65k_over_5ms"] = {"value_ms": d65, "flag": d65 > 5.0}

    warm = results["bench_resolve"]["warm"]["p50_ms"]
    cold = results["bench_resolve"]["cold"]["p50_ms"]
    flags["resolve_warm_over_1ms"] = {"value_ms": warm, "flag": warm > 1.0}
    flags["resolve_cold_over_100ms"] = {"value_ms": cold, "flag": cold > 100.0}

    eol = results["bench_eol"]
    flags["eol_over_500ms"] = {"value_ms": eol["wall_ms"], "flag": eol["wall_ms"] > 500.0}
    flags["eol_rss_over_500MB"] = {"value_mb": eol["peak_rss_delta_mb"], "flag": eol["peak_rss_delta_mb"] > 500.0}

    delta = results["bench_serving_overhead"]["p50_delta_ms_per_call"]
    flags["serving_p50_delta_over_10ms"] = {"value_ms": delta, "flag": delta > 10.0}

    for entry in flags.values():
        entry["status"] = "FLAG" if entry["flag"] else "PASS"
    return flags


# ---------------------------------------------------------------- main

def main() -> None:
    rng = random.Random(20260821)
    results: dict = {}
    benches = [
        ("bench_append", bench_append),
        ("bench_append_growth", bench_append_growth),
        ("bench_hashes", bench_hashes),
        ("bench_stamp_o_l2", bench_stamp_o_l2),
        ("bench_digest_tokens", bench_digest_tokens),
        ("bench_resolve", bench_resolve),
        ("bench_eol", bench_eol),
        ("bench_row_size", bench_row_size),
        ("bench_serving_overhead", bench_serving_overhead),
    ]
    suite_start = time.perf_counter()
    for name, fn in benches:
        t0 = time.perf_counter()
        results[name] = fn(rng)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s", flush=True)

    results["red_flags"] = evaluate_flags(results)
    try:
        branch = subprocess.run(
            ["git", "-C", str(REPO), "branch", "--show-current"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        branch = "unknown"
    results["env"] = {
        "machine": f"{platform.machine()} {platform.platform()} (Apple Silicon Mac, local NVMe)",
        "python": sys.version.split()[0],
        "branch": branch,
        "suite_wall_s": round(time.perf_counter() - suite_start, 1),
    }
    RESULTS_PATH.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    print(f"\nresults -> {RESULTS_PATH}")
    print(json.dumps(results["red_flags"], indent=2))


if __name__ == "__main__":
    main()
