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
"""Capture-ledger checkpoint commit and restore.

The exit criterion for Milestone 1 is that completed rollouts survive a
controller restart with byte-identical token rows. These tests pin the
mechanisms behind it: commit copies the token-capture store into the
checkpoint directory with the manifest written last (a torn commit leaves no
manifest and restore refuses it); tombstoned attempts are excluded from the
commit and re-fenced at restore; digests are verified before installing; and
the restored server boots paused so nothing serves until the coordinator
explicitly resumes. The call-id echo header is the join key callers record
against this ledger.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    ModelCallCaptureConfig,
    SimpleResponsesAPIModel,
    install_model_call_capture,
)
from nemo_gym.checkpoint import (
    LEDGER_MANIFEST_NAME,
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    MODEL_LEDGER_SUBDIR,
    AdmissionLimiter,
    CaptureLedgerCheckpointer,
    LedgerMismatchError,
    StaleAttemptError,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import MODEL_CALL_ID_HEADER, ROLLOUT_ID_HEADER
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


def _entry(rollout_id: str, call: int) -> TokenEntry:
    return TokenEntry(
        rollout_id=rollout_id,
        model_call_id=f"{rollout_id}-call-{call}",
        prompt_token_ids=[1, 2, 3, call],
        generation_token_ids=[10 + call, 11 + call],
        generation_log_probs=[-0.1, -0.2],
    )


def _seed_store(root) -> TokenCaptureStore:
    store = TokenCaptureStore(root)
    for rollout_id in ("4-2", "7-1-a2"):
        for call in range(2):
            store.append(_entry(rollout_id, call))
    return store


# --- checkpointer ---


def test_commit_restore_round_trip_preserves_identical_rows(tmp_path) -> None:
    store = _seed_store(tmp_path / "store-a")
    checkpoint_dir = tmp_path / "ckpt"
    summary = CaptureLedgerCheckpointer(store.root).commit(checkpoint_dir, checkpoint_id="ckpt-1", tombstones=[])
    assert summary["rollouts"] == 2
    assert summary["rows"] == 4
    assert summary["excluded_tombstoned"] == 0

    restored_root = tmp_path / "store-b"
    result = CaptureLedgerCheckpointer(restored_root).restore(checkpoint_dir)
    assert result["rollouts"] == 2
    assert result["rows"] == 4

    for rollout_id in ("4-2", "7-1-a2"):
        original = (store.root / f"{rollout_id}.tokens.jsonl").read_bytes()
        restored = (restored_root / f"{rollout_id}.tokens.jsonl").read_bytes()
        assert restored == original
    # The restored store is a working store: the same idempotent append is a no-op.
    restored_store = TokenCaptureStore(restored_root)
    restored_store.append(_entry("4-2", 0))
    assert len(restored_store.read_entries("4-2")) == 2


def test_commit_excludes_tombstoned_attempts_and_manifest_records_them(tmp_path) -> None:
    store = _seed_store(tmp_path / "store")
    checkpoint_dir = tmp_path / "ckpt"
    summary = CaptureLedgerCheckpointer(store.root).commit(
        checkpoint_dir, checkpoint_id="ckpt-1", tombstones=[("7-1", 2)]
    )
    assert summary["rollouts"] == 1
    assert summary["excluded_tombstoned"] == 1

    manifest = json.loads((checkpoint_dir / MODEL_LEDGER_SUBDIR / LEDGER_MANIFEST_NAME).read_text())
    assert list(manifest["rollouts"]) == ["4-2"]
    assert manifest["tombstones"] == [{"rollout_id": "7-1", "attempt_index": 2}]
    # The abandoned attempt's rows never entered the checkpoint.
    assert not (checkpoint_dir / MODEL_LEDGER_SUBDIR / "7-1-a2.tokens.jsonl").exists()


def test_restore_refuses_directory_without_manifest(tmp_path) -> None:
    store = _seed_store(tmp_path / "store")
    checkpoint_dir = tmp_path / "ckpt"
    CaptureLedgerCheckpointer(store.root).commit(checkpoint_dir, checkpoint_id="ckpt-1", tombstones=[])
    # Simulate a torn commit: files present, manifest missing.
    (checkpoint_dir / MODEL_LEDGER_SUBDIR / LEDGER_MANIFEST_NAME).unlink()
    with pytest.raises(LedgerMismatchError):
        CaptureLedgerCheckpointer(tmp_path / "store-b").restore(checkpoint_dir)


def test_restore_refuses_corrupted_ledger_file(tmp_path) -> None:
    store = _seed_store(tmp_path / "store")
    checkpoint_dir = tmp_path / "ckpt"
    CaptureLedgerCheckpointer(store.root).commit(checkpoint_dir, checkpoint_id="ckpt-1", tombstones=[])
    target = checkpoint_dir / MODEL_LEDGER_SUBDIR / "4-2.tokens.jsonl"
    target.write_bytes(target.read_bytes() + b'{"corrupt": true}\n')
    with pytest.raises(LedgerMismatchError):
        CaptureLedgerCheckpointer(tmp_path / "store-b").restore(checkpoint_dir)


# --- model server routes ---


def _model_server(tmp_path, name: str, store_subdir: str) -> SimpleResponsesAPIModel:
    class _Model(SimpleResponsesAPIModel):
        async def chat_completions(self, request):
            raise NotImplementedError

        async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            raise NotImplementedError

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {
        "token_id_capture": {"enabled": True, "dir": str(tmp_path / store_subdir)},
    }
    return _Model(
        config=BaseResponsesAPIModelConfig(host="", port=0, entrypoint="", name=name, instance_role="policy"),
        server_client=server_client,
    )


def test_commit_and_restore_through_routes_with_paused_bootstrap(tmp_path) -> None:
    # Original server: rows captured, then pause -> commit.
    server_a = _model_server(tmp_path, "policy", "store-a")
    client_a = TestClient(server_a.setup_webserver())
    _seed_store(tmp_path / "store-a")
    server_a.admission_limiter().abort_inflight("7-1-a2", 2)

    pause = client_a.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9})
    assert pause.status_code == 200
    commit = client_a.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={"checkpoint_id": "ckpt-1", "checkpoint_dir": str(tmp_path / "ckpt")},
    )
    assert commit.status_code == 200
    assert commit.json()["rollouts"] == 1
    assert commit.json()["excluded_tombstoned"] == 1

    # Commit is idempotent by checkpoint id: a retry replays the result.
    assert (
        client_a.post(
            f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
            json={"checkpoint_id": "ckpt-1", "checkpoint_dir": str(tmp_path / "ckpt")},
        ).json()
        == commit.json()
    )

    # Replacement server on a fresh store: restore installs the ledger and
    # boots paused; only an explicit resume reopens admission.
    server_b = _model_server(tmp_path, "policy", "store-b")
    client_b = TestClient(server_b.setup_webserver())
    restore = client_b.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "ckpt-2", "checkpoint_dir": str(tmp_path / "ckpt")},
    )
    assert restore.status_code == 200
    assert restore.json()["rollouts"] == 1
    assert restore.json()["tombstones"] == [{"rollout_id": "7-1", "attempt_index": 2}]

    parked = client_b.post("/v1/responses", json={"input": "hi"})
    assert parked.status_code == 409
    assert parked.json()["error"]["code"] == "checkpoint_parked"

    resume = client_b.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "ckpt-2"})
    assert resume.status_code == 200

    # The tombstone survived the restart: the abandoned attempt stays fenced.
    stale = client_b.post("/v1/responses", json={"input": "hi"}, headers={ROLLOUT_ID_HEADER: "7-1-a2"})
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_attempt"

    # The restored rows are byte-identical to the committed ones.
    original = (tmp_path / "store-a" / "4-2.tokens.jsonl").read_bytes()
    assert (tmp_path / "store-b" / "4-2.tokens.jsonl").read_bytes() == original


def test_restore_with_ledger_but_no_store_is_rejected(tmp_path) -> None:
    store = _seed_store(tmp_path / "store")
    CaptureLedgerCheckpointer(store.root).commit(tmp_path / "ckpt", checkpoint_id="ckpt-1", tombstones=[])

    class _Model(SimpleResponsesAPIModel):
        async def chat_completions(self, request):
            raise NotImplementedError

        async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            raise NotImplementedError

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server = _Model(
        config=BaseResponsesAPIModelConfig(host="", port=0, entrypoint="", name="policy", instance_role="policy"),
        server_client=server_client,
    )
    client = TestClient(server.setup_webserver())
    restore = client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "ckpt-2", "checkpoint_dir": str(tmp_path / "ckpt")},
    )
    assert restore.status_code == 409
    assert restore.json()["error"]["code"] == "ledger_mismatch"


# --- identity guard on tombstones ---


def test_tombstone_identity_guard_for_logical_ids_ending_in_attempt_suffix() -> None:
    limiter = AdmissionLimiter()
    # A logical id that legitimately ends in -a1, attempt 0: the suffix does
    # not agree with the explicit attempt, so it is part of the identity.
    limiter.abort_inflight("run-a1", 0)
    assert limiter.tombstones() == [("run-a1", 0)]
    with pytest.raises(StaleAttemptError):
        limiter.admit(rollout_id="run-a1", attempt_index=0)
    # The transport id "run-a1" without an explicit attempt still reads as
    # logical "run" attempt 1 and is not fenced.
    limiter.release(limiter.admit(rollout_id="run-a1"))


def test_install_tombstone_never_splits() -> None:
    limiter = AdmissionLimiter()
    limiter.install_tombstone("run-a1", 0)
    assert limiter.tombstones() == [("run-a1", 0)]


# --- call-id echo ---


def test_capture_routes_echo_model_call_id(tmp_path) -> None:
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses() -> JSONResponse:
        return JSONResponse({"id": "resp-1", "object": "response", "output": []})

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path / "captures"),
        model_server_name="policy",
    )
    client = TestClient(app)

    captured = client.post("/ng-rollout/4-2/v1/responses", json={"input": "hi"})
    assert captured.status_code == 200
    assert captured.headers[MODEL_CALL_ID_HEADER]

    # An un-prefixed call is not a capture call: no call id is minted or echoed.
    plain = client.post("/v1/responses", json={"input": "hi"})
    assert MODEL_CALL_ID_HEADER not in plain.headers
