# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Milestone 1 checkpoint cycle against a staged external data plane.

This is the shape the NeMo-RL deployment has: token arrays leave Gym through
a staging sink before each response is acknowledged and live in an external
memory-resident store (TransferQueue), while the Gym model server keeps only
token-free custody rows referencing them. A checkpoint therefore has two
halves — the external store's content-addressed cut and Gym's custody
commit — and a restore is not trustworthy until the custody manifest is
reconciled against the restored staging rows.

The simulation (``staged_data_plane``) runs both halves on one CPU through
the real Gym pipeline: real capture middleware and sink seam, real admission
pause/drain, real model-checkpoint commit/restore routes. The RL-side steps
(the staging store, its checkpoint cut, the reconciliation join) run inline
in the test the way the NemoGym actor will run them.

Pinned behaviors beyond the file-store e2e test: custody rows in Gym never
contain token arrays; the sacrificed attempt's staged rows survive the cut
as orphans (masked, cleaned up later) while its custody is excluded; and a
staged row missing at restore downgrades exactly its rollout to fresh
dispatch — a rollout is never marked trainable with missing training data.
"""

import pytest

from nemo_gym.checkpoint import MODEL_ADMISSION_URL_PREFIX, MODEL_CHECKPOINT_URL_PREFIX
from nemo_gym.rollout_correlation import ROLLOUT_ID_HEADER
from nemo_gym.token_id_capture.protocols import install_token_sink
from nemo_gym.token_id_capture.store import TokenCaptureStore
from tests.unit_tests.staged_data_plane import (
    STAGING_MANIFEST_NAME,
    FakeTransferQueue,
    StagedTokenSink,
    reconcile_custody_with_staging,
)
from tests.unit_tests.test_checkpoint_e2e import _Stack


@pytest.fixture
def sink_seam():
    # The installed-sink seam is process-global and resolved per request;
    # always clear it so no other test inherits a staging sink.
    try:
        yield install_token_sink
    finally:
        install_token_sink(None)


def _staged_stack(tmp_path, store_subdir: str, queue: FakeTransferQueue, install) -> _Stack:
    stack = _Stack(tmp_path, store_subdir)
    install(StagedTokenSink(queue=queue, custody_store=TokenCaptureStore(stack.store_dir)))
    return stack


async def _run_completed_rollouts(stack: _Stack) -> None:
    for task, rollout in ((0, 0), (0, 1), (1, 0)):
        result = await stack.run_rollout(task, rollout)
        assert result["reward"] == 1.0


def _pause_abort_and_commit(stack: _Stack, tmp_path) -> None:
    pause = stack.policy_client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9}
    )
    assert pause.json()["state"] == "paused"
    abort = stack.policy_client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
        json={"checkpoint_id": "ckpt-1", "rollout_id": "2-0", "attempt_index": 0},
    )
    assert abort.status_code == 200
    commit = stack.policy_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={"checkpoint_id": "ckpt-1", "checkpoint_dir": str(tmp_path / "ckpt")},
    )
    assert commit.status_code == 200
    assert commit.json()["rollouts"] == 3
    assert commit.json()["excluded_tombstoned"] == 1


def _restore_and_resume(stack: _Stack, tmp_path) -> None:
    restore = stack.policy_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "ckpt-2", "checkpoint_dir": str(tmp_path / "ckpt")},
    )
    assert restore.status_code == 200
    assert restore.json()["rollouts"] == 3
    resume = stack.policy_client.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "ckpt-2"})
    assert resume.status_code == 200


@pytest.mark.asyncio
async def test_staged_plane_checkpoint_restart_reconcile_cycle(tmp_path, monkeypatch, sink_seam) -> None:
    import nemo_gym.server_utils

    # ---- runtime: tokens stage externally, Gym keeps token-free custody ----
    queue_a = FakeTransferQueue()
    stack_a = _staged_stack(tmp_path, "custody-a", queue_a, sink_seam)
    monkeypatch.setattr(nemo_gym.server_utils, "request", stack_a.dispatch)

    await _run_completed_rollouts(stack_a)
    await stack_a.run_rollout(2, 0, agent="partial_agent")  # unfinished at the boundary

    custody_store = TokenCaptureStore(stack_a.store_dir)
    for rollout_id in ("0-0", "0-1", "1-0", "2-0"):
        for entry in custody_store.read_entries(rollout_id):
            # The plane split: no token array ever rests in Gym's ledger.
            assert entry.prompt_token_ids == []
            assert entry.generation_token_ids == []
            extras = entry.model_dump()
            staged = queue_a.get(extras["staging_key"])
            assert staged is not None
            assert len(staged["prompt_token_ids"]) == extras["staged_prompt_len"] > 0
    assert len(queue_a) == 4  # one staged row per generation call, 2-0 included

    # ---- prepare + save: drain, sacrifice, cut the staging store, commit custody ----
    _pause_abort_and_commit(stack_a, tmp_path)
    # The staging cut happens after the drain (finalizer quiescence stand-in).
    # It is content-addressed and includes everything staged — also the
    # sacrificed attempt's row, which custody no longer references.
    assert queue_a.checkpoint(tmp_path / "ckpt" / "tq") == {"rows": 4}

    # ---- restart: fresh Gym stack, staging store rebuilt only from the cut ----
    queue_b = FakeTransferQueue.restore(tmp_path / "ckpt" / "tq")
    stack_b = _staged_stack(tmp_path, "custody-b", queue_b, sink_seam)
    monkeypatch.setattr(nemo_gym.server_utils, "request", stack_b.dispatch)
    _restore_and_resume(stack_b, tmp_path)

    # ---- reconcile: the actor joins restored custody against restored staging ----
    report = reconcile_custody_with_staging(stack_b.store_dir, queue_b)
    assert report["trainable"] == ["0-0", "0-1", "1-0"]
    assert report["downgraded"] == {}
    # The sacrificed attempt's staged row survived the cut with no custody
    # row referencing it: an orphan, masked and left for normal cleanup.
    assert len(report["orphans"]) == 1
    assert report["orphans"][0].startswith("2-0/")

    # Every trainable rollout's staged payload carries the digest custody
    # recorded before the restart: the training data is exactly what was
    # acknowledged, not a reconstruction.
    restored_store = TokenCaptureStore(stack_b.store_dir)
    for rollout_id in report["trainable"]:
        for entry in restored_store.read_entries(rollout_id):
            extras = entry.model_dump()
            assert queue_b.digest_of(extras["staging_key"]) == extras["staging_digest"]

    # The fence survived the restart alongside the data.
    stale = stack_b.policy_client.post("/v1/responses", json={"input": "late"}, headers={ROLLOUT_ID_HEADER: "2-0"})
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_attempt"

    # ---- fresh dispatch: the replacement attempt stages under its own identity ----
    result = await stack_b.run_rollout(2, 0, attempt=1)
    assert result["reward"] == 1.0
    report = reconcile_custody_with_staging(stack_b.store_dir, queue_b)
    assert "2-0-a1" in report["trainable"]
    assert report["downgraded"] == {}


@pytest.mark.asyncio
async def test_missing_staged_row_downgrades_only_its_rollout(tmp_path, monkeypatch, sink_seam) -> None:
    import nemo_gym.server_utils

    queue_a = FakeTransferQueue()
    stack_a = _staged_stack(tmp_path, "custody-a", queue_a, sink_seam)
    monkeypatch.setattr(nemo_gym.server_utils, "request", stack_a.dispatch)

    await _run_completed_rollouts(stack_a)
    await stack_a.run_rollout(2, 0, agent="partial_agent")
    _pause_abort_and_commit(stack_a, tmp_path)
    queue_a.checkpoint(tmp_path / "ckpt" / "tq")

    # A torn staging cut: rollout 0-1's staged row is gone from the restored
    # store (deleted from the manifest before restore).
    import json

    manifest_path = tmp_path / "ckpt" / "tq" / STAGING_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    victim = next(key for key in manifest if key.startswith("0-1/"))
    del manifest[victim]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=1))

    queue_b = FakeTransferQueue.restore(tmp_path / "ckpt" / "tq")
    stack_b = _staged_stack(tmp_path, "custody-b", queue_b, sink_seam)
    monkeypatch.setattr(nemo_gym.server_utils, "request", stack_b.dispatch)
    _restore_and_resume(stack_b, tmp_path)

    report = reconcile_custody_with_staging(stack_b.store_dir, queue_b)
    # Only the rollout with the missing row is downgraded to fresh dispatch;
    # it is never marked trainable with incomplete training data. The others
    # are untouched.
    assert report["trainable"] == ["0-0", "1-0"]
    assert list(report["downgraded"]) == ["0-1"]
    assert "missing from the restored store" in report["downgraded"]["0-1"]


def test_fake_transfer_queue_rejects_conflicting_restage() -> None:
    queue = FakeTransferQueue()
    queue.stage("4-2/call-1", {"prompt_token_ids": [1], "generation_token_ids": [2], "generation_log_probs": [-0.1]})
    # Idempotent restage is a no-op...
    queue.stage("4-2/call-1", {"prompt_token_ids": [1], "generation_token_ids": [2], "generation_log_probs": [-0.1]})
    # ...but the same key with different tokens is corruption, mirroring the
    # TokenSink contract.
    with pytest.raises(ValueError, match="different payload"):
        queue.stage(
            "4-2/call-1", {"prompt_token_ids": [9], "generation_token_ids": [2], "generation_log_probs": [-0.1]}
        )
