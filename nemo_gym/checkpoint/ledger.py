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

The model server's capture ledger is the durable record of which token rows
exist for which rollout attempt: the ``TokenCaptureStore`` files (rows,
state, incomplete sentinels). Commit copies that ledger into the checkpoint
directory; restore installs it into a freshly started server so restored
custody records refer to exactly the rows that were committed.

Three properties make the copy a checkpoint rather than a backup:

- **Tombstone exclusion.** A rollout attempt force-closed at the prepare
  deadline must not restore: its rows describe an execution the restored run
  replaces with a fresh dispatch. Commit skips tombstoned attempts and
  records the tombstones in the manifest so the restored server re-installs
  the fence before serving anything.
- **Manifest-last ordering.** Every ledger file is written and fsynced
  before the manifest appears (temporary name, fsync, rename). A commit that
  died partway leaves no manifest, and restore refuses the directory instead
  of installing a torn ledger.
- **Digest verification.** The manifest records each rollout file's SHA-256.
  Restore verifies every installed file against it, so silent corruption in
  transit fails loudly at restore instead of surfacing as wrong training
  data later.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.checkpoint.admission import AdmissionLimiter
from nemo_gym.checkpoint.control import (
    CONTROL_URL_PREFIX,
    CheckpointPhase,
    ControlError,
    ControlFence,
)
from nemo_gym.checkpoint.model_admission import NotPolicyInstanceError
from nemo_gym.rollout_correlation import split_transport_rollout_id


MODEL_CHECKPOINT_URL_PREFIX = f"{CONTROL_URL_PREFIX}/model-checkpoint"
MODEL_LEDGER_SUBDIR = "model-ledger"
LEDGER_MANIFEST_NAME = "manifest.json"
LEDGER_SCHEMA_VERSION = 1

# Ledger files per rollout, in the token-capture store's naming scheme. Lock
# files are deliberately absent: locks are process-local, not state.
_LEDGER_SUFFIXES = (".tokens.jsonl", ".tokens.state.json", ".tokens.incomplete")


class LedgerMismatchError(ControlError):
    """The checkpoint directory does not match its manifest.

    A missing manifest means the commit tore partway; a digest mismatch
    means a file changed after commit. Either way the ledger must not be
    installed: restored custody would refer to rows that do not exist as
    committed.
    """

    code = "ledger_mismatch"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_fsynced(source: Path, target: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ledger-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as src:
                shutil.copyfileobj(src, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CaptureLedgerCheckpointer:
    """Commit and restore one token-capture store directory."""

    def __init__(self, store_root: Path) -> None:
        self.store_root = Path(store_root)

    def _rollout_ids(self) -> list[str]:
        suffix = ".tokens.state.json"
        return sorted(path.name[: -len(suffix)] for path in self.store_root.glob(f"*{suffix}"))

    def commit(self, checkpoint_dir: Path, *, checkpoint_id: str, tombstones: list[tuple[str, int]]) -> dict[str, Any]:
        """Copy the ledger into ``checkpoint_dir``; the caller has already drained.

        The store must be quiescent (admission paused) when this runs: the
        copy takes no locks because nothing may be writing.
        """
        ledger_dir = Path(checkpoint_dir) / MODEL_LEDGER_SUBDIR
        ledger_dir.mkdir(parents=True, exist_ok=True)
        fenced = set(tombstones)

        rollouts: dict[str, dict[str, Any]] = {}
        excluded = 0
        total_rows = 0
        for rollout_id in self._rollout_ids():
            if split_transport_rollout_id(rollout_id) in fenced:
                excluded += 1
                continue
            files: dict[str, str] = {}
            rows = 0
            for suffix in _LEDGER_SUFFIXES:
                source = self.store_root / f"{rollout_id}{suffix}"
                if not source.exists():
                    continue
                target = ledger_dir / source.name
                _copy_fsynced(source, target)
                files[source.name] = _file_digest(target)
                if suffix == ".tokens.jsonl":
                    rows = sum(1 for line in target.read_bytes().splitlines() if line.strip())
            rollouts[rollout_id] = {"files": files, "rows": rows}
            total_rows += rows
        _fsync_dir(ledger_dir)

        manifest = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "rollouts": rollouts,
            "tombstones": [
                {"rollout_id": rollout_id, "attempt_index": attempt} for rollout_id, attempt in sorted(tombstones)
            ],
        }
        payload = json.dumps(manifest, sort_keys=True, indent=1).encode()
        with tempfile.NamedTemporaryFile(dir=ledger_dir, prefix=".manifest-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, ledger_dir / LEDGER_MANIFEST_NAME)
        _fsync_dir(ledger_dir)

        return {
            "rollouts": len(rollouts),
            "rows": total_rows,
            "excluded_tombstoned": excluded,
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
        }

    def restore(self, checkpoint_dir: Path) -> dict[str, Any]:
        """Install a committed ledger into this store root and verify it."""
        ledger_dir = Path(checkpoint_dir) / MODEL_LEDGER_SUBDIR
        manifest_path = ledger_dir / LEDGER_MANIFEST_NAME
        if not manifest_path.exists():
            raise LedgerMismatchError(
                f"no ledger manifest at {manifest_path}; the commit tore partway or never ran, "
                f"so this directory must not be installed"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version", 0) > LEDGER_SCHEMA_VERSION:
            raise LedgerMismatchError(
                f"ledger manifest schema_version {manifest.get('schema_version')} is newer than this "
                f"reader ({LEDGER_SCHEMA_VERSION})"
            )

        self.store_root.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        for rollout_id, meta in manifest["rollouts"].items():
            for name, digest in meta["files"].items():
                source = ledger_dir / name
                if not source.exists() or _file_digest(source) != digest:
                    raise LedgerMismatchError(
                        f"ledger file {name} for rollout {rollout_id!r} is missing or does not match "
                        f"its committed digest; refusing to install a corrupted ledger"
                    )
                _copy_fsynced(source, self.store_root / name)
            total_rows += int(meta.get("rows", 0))
        _fsync_dir(self.store_root)

        return {
            "rollouts": len(manifest["rollouts"]),
            "rows": total_rows,
            "checkpoint_id": manifest.get("checkpoint_id"),
            "tombstones": list(manifest.get("tombstones", ())),
        }


class ModelCheckpointCommitRequest(BaseModel):
    checkpoint_id: str
    checkpoint_dir: str


class ModelCheckpointRestoreRequest(BaseModel):
    checkpoint_id: str
    checkpoint_dir: str


def install_model_checkpoint(
    app: FastAPI,
    *,
    fence: ControlFence,
    limiter: AdmissionLimiter,
    store_root_provider: Callable[[], Optional[Path]],
    instance_role: Literal["policy", "auxiliary"],
) -> None:
    """Register ``/ng-control/v1/model-checkpoint`` on a model-server app.

    Commit requires the prepared (drained) phase; restore runs on a freshly
    started server and leaves it paused, so nothing serves until the
    coordinator has restored every component and explicitly resumes.
    """

    def _require_policy() -> None:
        if instance_role != "policy":
            raise NotPolicyInstanceError(
                "this model-server instance is auxiliary (judge or simulator traffic); "
                "it produces no training tokens and has no capture ledger to checkpoint"
            )

    @app.post(f"{MODEL_CHECKPOINT_URL_PREFIX}/commit")
    async def model_checkpoint_commit(body: ModelCheckpointCommitRequest) -> dict[str, Any]:
        _require_policy()

        async def run() -> dict[str, Any]:
            store_root = store_root_provider()
            if store_root is None:
                return {"rollouts": 0, "rows": 0, "excluded_tombstoned": 0, "detail": "no token store configured"}
            checkpointer = CaptureLedgerCheckpointer(store_root)
            return checkpointer.commit(
                Path(body.checkpoint_dir), checkpoint_id=body.checkpoint_id, tombstones=limiter.tombstones()
            )

        return await fence.run_operation(
            body.checkpoint_id,
            "model-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )

    @app.post(f"{MODEL_CHECKPOINT_URL_PREFIX}/restore")
    async def model_checkpoint_restore(body: ModelCheckpointRestoreRequest) -> dict[str, Any]:
        _require_policy()

        async def run() -> dict[str, Any]:
            # The restored server boots into the paused state: nothing may be
            # admitted until every component is restored and the coordinator
            # explicitly resumes.
            limiter.close()
            store_root = store_root_provider()
            ledger_dir = Path(body.checkpoint_dir) / MODEL_LEDGER_SUBDIR
            if store_root is None:
                if (ledger_dir / LEDGER_MANIFEST_NAME).exists():
                    raise LedgerMismatchError(
                        "the checkpoint contains a capture ledger but this server has no token store "
                        "configured to install it into"
                    )
                return {"rollouts": 0, "rows": 0, "tombstones": [], "detail": "no token store configured"}
            result = CaptureLedgerCheckpointer(store_root).restore(Path(body.checkpoint_dir))
            for tombstone in result["tombstones"]:
                limiter.install_tombstone(tombstone["rollout_id"], tombstone["attempt_index"])
            return result

        return await fence.run_operation(
            body.checkpoint_id,
            "model-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
        )
