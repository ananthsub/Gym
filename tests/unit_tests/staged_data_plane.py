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
"""CPU stand-in for the NeMo-RL staged-mode data plane.

In the NeMo-RL deployment, token arrays never rest in Gym: the generation
worker stages each call's tokens into TransferQueue synchronously before the
response is acknowledged, and the Gym model server keeps only token-free
custody rows that reference the staged data. TransferQueue's storage is
memory-resident and becomes durable only at an explicit checkpoint cut, so a
restore must reconcile Gym's custody manifest against the restored staging
rows before anything is marked trainable.

This module simulates that split on one CPU so the checkpoint choreography
can be tested end to end without torch, Ray, or TransferQueue (whose client
cannot even import inside Gym's light venv — TQ clients live only in
RL-owned processes):

- ``FakeTransferQueue`` plays the external staging store: in-memory rows,
  content-addressed checkpoint to a directory, restore from that directory.
  Anything staged after the cut does not survive a restart, exactly like the
  real store.
- ``StagedTokenSink`` plays the staging path (installed through Gym's token
  sink seam): it stages the arrays before ``put`` returns — the
  durable-before-ack contract — and appends a token-free custody row, with
  the staging key and payload digest, to the model server's capture store.
- ``reconcile_custody_with_staging`` plays the NemoGym actor's post-restore
  join: a rollout is trainable only when every custody row's staged payload
  exists with a matching digest; any miss downgrades that rollout to a
  fresh dispatch (never a continuation with missing training data); staged
  rows with no custody row are orphans left for masked cleanup.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import orjson

from nemo_gym.token_id_capture.records import TOKEN_FIELDS, TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


STAGING_MANIFEST_NAME = "staging-manifest.json"


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


class FakeTransferQueue:
    """Memory-resident staging store, durable only at checkpoint cuts."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def stage(self, key: str, payload: dict[str, Any]) -> str:
        """Durably-before-ack in spirit: the row is visible once this returns.

        Restaging the same key with the same payload is a no-op; a different
        payload under an existing key is a corruption signal, mirroring the
        TokenSink contract.
        """
        digest = _payload_digest(payload)
        existing = self._rows.get(key)
        if existing is not None and _payload_digest(existing) != digest:
            raise ValueError(f"staging key {key!r} reused with a different payload")
        self._rows[key] = payload
        return digest

    def get(self, key: str) -> Optional[dict[str, Any]]:
        return self._rows.get(key)

    def digest_of(self, key: str) -> Optional[str]:
        payload = self._rows.get(key)
        return None if payload is None else _payload_digest(payload)

    def __len__(self) -> int:
        return len(self._rows)

    def checkpoint(self, directory: Path) -> dict[str, Any]:
        """Content-addressed dump; the manifest is written last.

        The caller is responsible for quiescence (the model server drained,
        finalizers idle) — the real TransferQueue checkpoint has the same
        precondition.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, str] = {}
        for key, payload in sorted(self._rows.items()):
            digest = _payload_digest(payload)
            blob = directory / f"{digest}.json"
            if not blob.exists():
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".stage-", delete=False) as handle:
                    handle.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.replace(handle.name, blob)
            manifest[key] = digest
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".manifest-", delete=False) as handle:
            handle.write(json.dumps(manifest, sort_keys=True, indent=1).encode())
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(handle.name, directory / STAGING_MANIFEST_NAME)
        return {"rows": len(manifest)}

    @classmethod
    def restore(cls, directory: Path) -> "FakeTransferQueue":
        """A restarted store holds exactly what the cut captured — no more."""
        directory = Path(directory)
        manifest = json.loads((directory / STAGING_MANIFEST_NAME).read_text())
        queue = cls()
        for key, digest in manifest.items():
            payload = orjson.loads((directory / f"{digest}.json").read_bytes())
            if _payload_digest(payload) != digest:
                raise ValueError(f"staged blob for {key!r} does not match its manifest digest")
            queue._rows[key] = payload
        return queue


class StagedTokenSink:
    """Gym-side stand-in for the worker's staging sink.

    Receives the full ``TokenEntry`` from the capture pipeline, stages the
    token arrays into the external store, and appends a token-free custody
    row to the model server's capture store. Both writes complete before
    ``put`` returns, so the response is never acknowledged ahead of custody.
    """

    def __init__(self, queue: FakeTransferQueue, custody_store: TokenCaptureStore) -> None:
        self.queue = queue
        self.custody_store = custody_store

    @staticmethod
    def staging_key(rollout_id: str, model_call_id: str) -> str:
        return f"{rollout_id}/{model_call_id}"

    async def put(self, entry: TokenEntry) -> None:
        key = self.staging_key(entry.rollout_id, entry.model_call_id)
        payload = {
            "prompt_token_ids": entry.prompt_token_ids,
            "generation_token_ids": entry.generation_token_ids,
            "generation_log_probs": entry.generation_log_probs,
        }
        digest = self.queue.stage(key, payload)

        custody = entry.model_dump(mode="json")
        for field in TOKEN_FIELDS:
            if field in custody and custody[field] is not None:
                custody[field] = []
        custody["staging_key"] = key
        custody["staging_digest"] = digest
        custody["staged_prompt_len"] = len(entry.prompt_token_ids)
        custody["staged_generation_len"] = len(entry.generation_token_ids)
        self.custody_store.append(TokenEntry.model_validate(custody))

    async def close(self) -> None:
        pass


def reconcile_custody_with_staging(custody_root: Path, queue: FakeTransferQueue) -> dict[str, Any]:
    """The NemoGym actor's post-restore join of custody against staging.

    Returns ``trainable`` rollout ids (every custody row resolved with a
    matching digest), ``downgraded`` rollout ids mapped to the reason (any
    row missing or mismatched — the whole rollout becomes a fresh dispatch),
    and ``orphans``: staged keys no custody row references, left masked for
    normal cleanup.
    """
    custody_root = Path(custody_root)
    store = TokenCaptureStore(custody_root)
    suffix = ".tokens.state.json"
    rollout_ids = sorted(path.name[: -len(suffix)] for path in custody_root.glob(f"*{suffix}"))

    trainable: list[str] = []
    downgraded: dict[str, str] = {}
    referenced: set[str] = set()
    for rollout_id in rollout_ids:
        reason = None
        for entry in store.read_entries(rollout_id):
            extras = entry.model_dump()
            key = extras.get("staging_key")
            expected = extras.get("staging_digest")
            if key is None or expected is None:
                reason = f"custody row {entry.model_call_id} carries no staging reference"
                break
            referenced.add(key)
            actual = queue.digest_of(key)
            if actual is None:
                reason = f"staged row {key} is missing from the restored store"
                break
            if actual != expected:
                reason = f"staged row {key} does not match its custody digest"
                break
        if reason is None:
            trainable.append(rollout_id)
        else:
            downgraded[rollout_id] = reason

    orphans = sorted(key for key in queue._rows if key not in referenced)
    return {"trainable": trainable, "downgraded": downgraded, "orphans": orphans}
