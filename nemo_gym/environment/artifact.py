# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed prepared datasets for reproducible Gym runs."""

import hashlib
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.lock import EnvironmentLockRecord
from nemo_gym.environment.protocol import PREPARED_ARTIFACT_SCHEMA_VERSION, RUNTIME_PROTOCOL_VERSION
from nemo_gym.global_config import get_global_config_dict


PREPARED_ARTIFACT_MANIFEST = "manifest.json"
TASK_ID_KEY = "task_id"
NEMO_GYM_ENVIRONMENT_LOCK_ENV_VAR = "NEMO_GYM_ENVIRONMENT_LOCK"
_RUNTIME_ROW_KEYS = frozenset({"_ng_task_index", "_ng_rollout_index"})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PreparedArtifactManifest(BaseModel):
    """Identity and provenance for one immutable prepared JSONL dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["nemo-gym/prepared-artifact/v1"] = PREPARED_ARTIFACT_SCHEMA_VERSION
    runtime_protocol: Literal["nemo-gym/runtime/v1"] = RUNTIME_PROTOCOL_VERSION
    artifact_id: str = ""
    split: str
    data_file: str
    data_sha256: str
    row_count: int = Field(ge=0)
    input_sha256: str
    config_sha256: str | None = None
    environment_lock_identity: str | None = None

    def canonical_payload(self) -> bytes:
        # Original bytes are provenance, while artifact identity follows normalized task content.
        payload = self.model_dump(mode="json", exclude={"artifact_id", "input_sha256"})
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)

    @model_validator(mode="after")
    def validate_identity(self) -> "PreparedArtifactManifest":
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.split) is None:
            raise ValueError("Prepared artifact split must contain only letters, numbers, '.', '_', and '-'")
        if self.data_file != f"{self.split}.jsonl":
            raise ValueError(f"Prepared artifact data_file must be {self.split}.jsonl")
        expected = _sha256(self.canonical_payload())
        if self.artifact_id and self.artifact_id != expected:
            raise ValueError(f"Prepared artifact identity mismatch: expected {expected}, got {self.artifact_id}")
        object.__setattr__(self, "artifact_id", expected)
        return self

    def canonical_json(self) -> str:
        return orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS).decode()


def _task_identity_payload(row: dict[str, Any]) -> bytes:
    identity_row = {key: value for key, value in row.items() if key != TASK_ID_KEY and key not in _RUNTIME_ROW_KEYS}
    return orjson.dumps(identity_row, option=orjson.OPT_SORT_KEYS)


def _materialize_rows(source_path: Path) -> tuple[bytes, int]:
    output = bytearray()
    task_id_counts: dict[str, int] = {}
    task_ids: set[str] = set()
    with source_path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSON in {source_path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ConfigError(f"Prepared artifact row {line_number} must be a JSON object")

            task_id = row.get(TASK_ID_KEY)
            if task_id is None:
                content_id = _sha256(_task_identity_payload(row))[:32]
                occurrence = task_id_counts.get(content_id, 0) + 1
                task_id_counts[content_id] = occurrence
                task_id = f"ng:{content_id}" if occurrence == 1 else f"ng:{content_id}:{occurrence}"
                row[TASK_ID_KEY] = task_id
            if not isinstance(task_id, str) or not task_id:
                raise ConfigError(f"Prepared artifact row {line_number} has an invalid task_id")
            if task_id in task_ids:
                raise ConfigError(f"Prepared artifact contains duplicate task_id {task_id!r}")
            task_ids.add(task_id)
            output.extend(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.extend(b"\n")
    return bytes(output), len(task_ids)


def create_prepared_artifact(
    source_path: Path,
    *,
    output_root: Path,
    split: str,
    config_sha256: str | None = None,
    environment_lock_identity: str | None = None,
) -> Path:
    """Create or reuse a digest-addressed prepared dataset directory."""
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"Prepared artifact source does not exist: {source_path}")

    data, row_count = _materialize_rows(source_path)
    data_file = f"{split}.jsonl"
    manifest = PreparedArtifactManifest(
        split=split,
        data_file=data_file,
        data_sha256=_sha256(data),
        row_count=row_count,
        input_sha256=_sha256(source_path.read_bytes()),
        config_sha256=config_sha256,
        environment_lock_identity=environment_lock_identity,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / manifest.artifact_id
    if destination.exists():
        load_prepared_artifact(destination)
        return destination

    with TemporaryDirectory(prefix=".nemo-gym-artifact-", dir=output_root) as temporary_dir:
        temporary = Path(temporary_dir)
        (temporary / data_file).write_bytes(data)
        (temporary / PREPARED_ARTIFACT_MANIFEST).write_text(manifest.canonical_json() + "\n", encoding="utf-8")
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
    load_prepared_artifact(destination)
    return destination


def load_prepared_artifact(path: Path) -> tuple[PreparedArtifactManifest, Path]:
    """Validate a prepared artifact and return its manifest and data path."""
    artifact_dir = path.expanduser().resolve()
    if artifact_dir.is_file():
        if artifact_dir.name != PREPARED_ARTIFACT_MANIFEST:
            raise ConfigError(f"Prepared artifact file must be named {PREPARED_ARTIFACT_MANIFEST}")
        artifact_dir = artifact_dir.parent
    manifest_path = artifact_dir / PREPARED_ARTIFACT_MANIFEST
    try:
        manifest = PreparedArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(f"Prepared artifact manifest is invalid: {exc}") from exc

    data_path = (artifact_dir / manifest.data_file).resolve()
    if not data_path.is_relative_to(artifact_dir):
        raise ConfigError(f"Prepared artifact data file escapes its directory: {manifest.data_file}")
    try:
        data = data_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Prepared artifact data is unavailable: {data_path}") from exc
    if _sha256(data) != manifest.data_sha256:
        raise ConfigError(f"Prepared artifact data digest mismatch: {data_path}")

    rows, row_count = _materialize_rows(data_path)
    if rows != data or row_count != manifest.row_count:
        raise ConfigError(f"Prepared artifact data is not canonical or has the wrong row count: {data_path}")
    if artifact_dir.name != manifest.artifact_id:
        raise ConfigError(
            f"Prepared artifact directory must equal its digest: expected {manifest.artifact_id}, got {artifact_dir.name}"
        )
    return manifest, data_path


def validate_prepared_artifact_lock(
    manifest: PreparedArtifactManifest,
    *,
    environment_lock_path: Path | None,
) -> None:
    """Require artifact and runtime lock identities to be declared together and match."""
    artifact_identity = manifest.environment_lock_identity
    if environment_lock_path is None:
        if artifact_identity is not None:
            raise ConfigError(
                "Prepared artifact declares an environment lock, but the runtime has no environment lock to verify"
            )
        return
    try:
        runtime_lock = EnvironmentLockRecord.model_validate_json(environment_lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(f"Runtime environment lock is invalid: {exc}") from exc
    if artifact_identity is None:
        raise ConfigError("Prepared artifact is not bound to the runtime environment lock")
    if artifact_identity != runtime_lock.identity:
        raise ConfigError(
            f"Prepared artifact lock mismatch: artifact requires {artifact_identity}, "
            f"runtime provides {runtime_lock.identity}"
        )


def prepare_artifact_cli() -> None:
    """Create a content-addressed artifact from an already-collated JSONL file."""
    config = get_global_config_dict()
    source = config.get("input_jsonl_fpath")
    output = config.get("output_dirpath")
    split = config.get("split")
    if not isinstance(source, str) or not source:
        raise ConfigError("gym eval prepare-artifact requires --input PATH")
    if not isinstance(output, str) or not output:
        raise ConfigError("gym eval prepare-artifact requires --output-dir DIR")
    if not isinstance(split, str) or not split:
        raise ConfigError("gym eval prepare-artifact requires --split NAME")

    lock_identity = None
    lock_path = config.get("environment_lock")
    if lock_path:
        try:
            lock = EnvironmentLockRecord.model_validate_json(Path(lock_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ConfigError(f"Environment lock is invalid: {exc}") from exc
        lock_identity = lock.identity

    destination = create_prepared_artifact(
        Path(source),
        output_root=Path(output),
        split=split,
        config_sha256=config.get("config_sha256"),
        environment_lock_identity=lock_identity,
    )
    print(destination)
