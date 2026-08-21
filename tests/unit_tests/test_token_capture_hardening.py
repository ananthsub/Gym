# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted tests for the hardening fixes: eviction semantics, the lazy lineage
index, store GC, the append-path durability relaxation, and config plumbing.

These exist because the one prototype change that shipped without a test
asserting its semantics (the cache bound) was exactly where a bug lived
(FIFO-instead-of-LRU). Eviction behavior is now asserted, not assumed.
"""

import asyncio
import os
import time

import pytest

from nemo_gym.token_id_capture import FileLineageStore, TokenCaptureStore, TokenEntry
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.lineage import stamp_continuation
from nemo_gym.token_id_capture.records import ParentResolutionStatus, stamp_lineage


def _committed_entry(rollout_id: str, call_id: str, prompt: list[int], gen: list[int], request: list[dict]):
    entry = TokenEntry(
        rollout_id=rollout_id,
        model_call_id=call_id,
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
        output_items=[{"type": "message", "role": "assistant", "content": f"answer {call_id}"}],
    )
    stamp_continuation(entry, request)
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return entry


def _write_rollout(store: TokenCaptureStore, rollout_id: str) -> list[dict]:
    request = [{"role": "user", "content": f"question {rollout_id}"}]
    store.append(_committed_entry(rollout_id, f"{rollout_id}-c1", [1, 2, 3], [4, 5], request))
    return request + [{"role": "assistant", "content": f"answer {rollout_id}-c1"}, {"role": "user", "content": "next"}]


class TestLineageCacheEviction:
    def test_eviction_is_lru_not_fifo(self, tmp_path):
        """Touching a rollout must protect it: the longest-LIVED rollout is the one
        lineage matters most for, so recency — not birth order — decides eviction."""
        store = TokenCaptureStore(tmp_path)
        lineage = FileLineageStore(tmp_path, max_cached_rollouts=2)
        continuations = {rid: _write_rollout(store, rid) for rid in ("r-a", "r-b", "r-c")}

        asyncio.run(lineage.resolve("r-a", continuations["r-a"]))
        asyncio.run(lineage.resolve("r-b", continuations["r-b"]))
        # Touch the OLDER rollout, then insert a third; FIFO would evict r-a anyway.
        asyncio.run(lineage.resolve("r-a", continuations["r-a"]))
        asyncio.run(lineage.resolve("r-c", continuations["r-c"]))

        assert "r-a" in lineage._cache, "recently-touched rollout must survive eviction"
        assert "r-b" not in lineage._cache, "least-recently-used rollout is the one evicted"

    def test_eviction_mid_rollout_is_a_performance_event_not_a_correctness_event(self, tmp_path):
        """An evicted live rollout must resolve identically after a cold re-tail."""
        store = TokenCaptureStore(tmp_path)
        lineage = FileLineageStore(tmp_path, max_cached_rollouts=1)
        continuation = _write_rollout(store, "live")
        first = asyncio.run(lineage.resolve("live", continuation))

        _write_rollout(store, "other")
        asyncio.run(lineage.resolve("other", [{"role": "user", "content": "x"}]))
        assert "live" not in lineage._cache

        again = asyncio.run(lineage.resolve("live", continuation))
        assert again.status == first.status == ParentResolutionStatus.RESOLVED
        assert again.match.model_call_id == first.match.model_call_id
        assert again.match.cumulative_token_ids == first.match.cumulative_token_ids


class TestLazyLineageIndex:
    def test_index_is_metadata_only_and_materializes_exact_tokens(self, tmp_path):
        store = TokenCaptureStore(tmp_path)
        lineage = FileLineageStore(tmp_path)
        continuation = _write_rollout(store, "lazy")

        resolution = asyncio.run(lineage.resolve("lazy", continuation))
        assert resolution.status == ParentResolutionStatus.RESOLVED
        assert list(resolution.match.cumulative_token_ids) == [1, 2, 3, 4, 5]
        # The cached node itself must not hold token arrays.
        node = lineage._cache["lazy"][2].by_call_id["lazy-c1"]
        assert node.cum_tokens is None
        assert node.entry_offset >= 0

    def test_materialization_fails_closed_on_a_corrupted_log(self, tmp_path):
        """A mutated file must not supply tokens from the wrong call: the digest
        interlock turns it into an UNRESOLVED lookup error, never a wrong prefix."""
        store = TokenCaptureStore(tmp_path)
        lineage = FileLineageStore(tmp_path)
        continuation = _write_rollout(store, "corrupt")
        asyncio.run(lineage.resolve("corrupt", continuation))  # warm the metadata index

        path = store.path_for("corrupt")
        content = path.read_bytes().replace(b'"prompt_token_ids":[1,2,3]', b'"prompt_token_ids":[9,2,3]')
        assert content != path.read_bytes(), "test setup: the payload must actually change"
        path.write_bytes(content)

        with pytest.raises(ValueError, match="digest|offset|points at"):
            index = lineage._cache["corrupt"][2]
            lineage._materialize("corrupt", index.by_call_id["corrupt-c1"], index)


class TestStoreGC:
    def test_sweep_retired_removes_only_old_retired_tombstones(self, tmp_path):
        store = TokenCaptureStore(tmp_path)
        for rid in ("done", "fresh", "live"):
            store.append(_committed_entry(rid, f"{rid}-c1", [1], [2], [{"role": "user", "content": "q"}]))
        for rid in ("done", "fresh"):
            snapshot = store.freeze_now(rid)
            assert asyncio.run(store.drop(rid, snapshot_id=snapshot.snapshot_id, version=snapshot.version))

        old = time.time() - 3600
        os.utime(store.state_path_for("done"), (old, old))

        removed = store.sweep_retired(older_than_seconds=600)
        assert removed == 1
        assert not store.state_path_for("done").exists()
        assert store.state_path_for("fresh").exists(), "recent tombstones are retained"
        assert store.path_for("live").exists(), "live rollouts are never touched"


class TestAppendDurabilityRelaxation:
    def test_state_lag_is_recovered_from_the_jsonl_tail(self, tmp_path):
        """The append-path state write is non-fsynced by design; the JSONL line is
        the durability guarantee. Simulate a lost state update and assert the tail
        index reconstruction still yields a complete, consistent snapshot."""
        store = TokenCaptureStore(tmp_path)
        request = [{"role": "user", "content": "q"}]
        store.append(_committed_entry("lag", "lag-c1", [1, 2], [3], request))
        state_after_first = store.state_path_for("lag").read_bytes()
        store.append(_committed_entry("lag", "lag-c2", [1, 2, 3, 9], [10], request))
        # Roll the state file back: the second append's state update "was lost".
        store.state_path_for("lag").write_bytes(state_after_first)

        snapshot = store.freeze_now("lag")
        assert {entry.model_call_id for entry in snapshot.entries} == {"lag-c1", "lag-c2"}
        assert snapshot.incomplete is False


class TestKillSwitchConfig:
    def test_fields_parse_and_default_off(self):
        config = TokenIdCaptureConfig.model_validate(
            {"token_id_capture": {"enabled": True, "dir": "/tmp/x", "max_mask_fraction": 0.5}}
        )
        assert config.token_id_capture.max_mask_fraction == 0.5
        assert config.token_id_capture.mask_fraction_min_samples == 50
        default = TokenIdCaptureConfig.model_validate({"token_id_capture": {"enabled": True, "dir": "/tmp/x"}})
        assert default.token_id_capture.max_mask_fraction is None


class TestPrefixProofSourceOrder:
    def test_message_bundle_outranks_top_level(self):
        from responses_api_models.vllm_model.app import VLLMModel

        response = {
            "prompt_token_ids": [9, 9, 9],
            "choices": [{"message": {"role": "assistant", "prompt_token_ids": [1, 2, 3]}}],
        }
        assert VLLMModel._generation_prompt_token_ids(response) == [1, 2, 3]

    def test_top_level_is_the_fallback(self):
        from responses_api_models.vllm_model.app import VLLMModel

        response = {"prompt_token_ids": [7, 8], "choices": [{"message": {"role": "assistant"}}]}
        assert VLLMModel._generation_prompt_token_ids(response) == [7, 8]


class TestDeltaRecords:
    """Schema-5 delta chains: O(T) storage with fail-closed reconstruction."""

    @staticmethod
    def _commit(store, lineage, rollout_id, call_id, prompt, gen, request):
        from nemo_gym.token_id_capture import CaptureContext, reset_token_sink, set_token_sink
        from nemo_gym.token_id_capture.sink import commit_entry, resolve_parent

        entry = TokenEntry(
            rollout_id=rollout_id,
            model_call_id=call_id,
            prompt_token_ids=prompt,
            generation_token_ids=gen,
            generation_log_probs=[-0.1] * len(gen),
            output_items=[{"type": "message", "role": "assistant", "content": f"answer {call_id}"}],
        )
        stamp_continuation(entry, request)
        context = CaptureContext(
            rollout_id=rollout_id,
            model_call_id=call_id,
            token_sink=store,
            lineage_store=lineage,
            delta_records=True,
        )
        token = set_token_sink(context)
        try:
            asyncio.run(resolve_parent(request))
            asyncio.run(commit_entry(entry))
        finally:
            reset_token_sink(token)
        return context

    def test_chain_stores_suffixes_and_reconstructs_exactly(self, tmp_path):
        from nemo_gym.token_id_capture.consumer import trajectories_from_source

        store = TokenCaptureStore(tmp_path)
        lineage = FileLineageStore(tmp_path)
        rid = "delta"
        request1 = [{"role": "user", "content": "q1"}]
        self._commit(store, lineage, rid, "c1", [1, 2, 3], [4, 5], request1)
        echo1 = {"role": "assistant", "content": "answer c1"}
        request2 = request1 + [echo1, {"role": "user", "content": "q2"}]
        ctx2 = self._commit(store, lineage, rid, "c2", [1, 2, 3, 4, 5, 6, 7], [8], request2)
        echo2 = {"role": "assistant", "content": "answer c2"}
        request3 = request2 + [echo2, {"role": "user", "content": "q3"}]
        ctx3 = self._commit(store, lineage, rid, "c3", [1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11], request3)

        # Continuations resolved through the lazy index, including a delta parent.
        assert ctx2.parent_resolution.status == ParentResolutionStatus.RESOLVED
        assert ctx3.parent_resolution.status == ParentResolutionStatus.RESOLVED
        assert list(ctx3.parent_resolution.match.cumulative_token_ids) == [1, 2, 3, 4, 5, 6, 7, 8]

        # On disk: c2/c3 store only their suffixes.
        entries = {e.model_call_id: e for e in store.read_entries(rid)}
        assert entries["c1"].prompt_is_delta is False
        assert entries["c2"].prompt_is_delta and entries["c2"].prompt_token_ids == [6, 7]
        assert entries["c3"].prompt_is_delta and entries["c3"].prompt_token_ids == [9]

        # Delivery reconstructs the full contiguous chain.
        built = asyncio.run(trajectories_from_source(rid, store))
        assert built["mask_sample"] is False
        output = built["rebuilt_response"]["output"]
        assert [item["generation_token_ids"] for item in output] == [[4, 5], [8], [10, 11]]
        assert output[2]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_a_broken_delta_chain_masks_instead_of_guessing(self, tmp_path):
        from nemo_gym.token_id_capture.consumer import trajectories_from_source

        store = TokenCaptureStore(tmp_path)
        orphan = TokenEntry(
            rollout_id="orphan",
            model_call_id="child",
            prompt_token_ids=[9],
            generation_token_ids=[10],
            generation_log_probs=[-0.1],
            output_items=[{"type": "message", "role": "assistant", "content": "a"}],
        )
        orphan.prompt_is_delta = True  # construction validates flag+parent together; stamp below supplies the parent
        stamp_continuation(orphan, [{"role": "user", "content": "q"}])
        stamp_lineage(
            orphan,
            "missing-parent",
            parent_resolution=ParentResolutionStatus.RESOLVED,
            cumulative=[1, 2, 9, 10],
        )
        store.append(orphan)

        built = asyncio.run(trajectories_from_source("orphan", store))
        assert built["mask_sample"] is True
        # The failed-build delivery path keeps only the error; the reason lives on
        # the builder output.
        from nemo_gym.token_id_capture.builder import prefix_merging

        out = prefix_merging(store.read_entries("orphan"))
        assert out.notes.parent_link_failures.get("delta_chain_unreconstructable") == 1
        assert out.notes.unresolved_parent_calls == ["child"]


    def test_delta_records_require_a_durable_log_backed_resolver(self):
        """The in-memory reference index cannot reconstruct a delta chain; it must
        refuse loudly rather than index a suffix as if it were the full sequence."""
        from nemo_gym.token_id_capture import InMemoryLineageStore

        entry = TokenEntry(
            rollout_id="r",
            model_call_id="child",
            prompt_token_ids=[9],
            generation_token_ids=[10],
            generation_log_probs=[-0.1],
            output_items=[{"type": "message", "role": "assistant", "content": "an answer"}],
        )
        entry.prompt_is_delta = True
        stamp_continuation(entry, [{"role": "user", "content": "q"}])
        stamp_lineage(
            entry, "parent", parent_resolution=ParentResolutionStatus.RESOLVED, cumulative=[1, 9, 10]
        )
        with pytest.raises(ValueError, match="durable-log-backed"):
            asyncio.run(InMemoryLineageStore().put(entry))
