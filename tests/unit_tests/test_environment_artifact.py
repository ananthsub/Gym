# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.artifact import create_prepared_artifact, load_prepared_artifact


def _write_source(path: Path) -> None:
    rows = [
        {"responses_create_params": {"input": "first"}, "task_source": "example"},
        {"responses_create_params": {"input": "second"}, "task_source": "example"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_create_prepared_artifact_is_content_addressed_and_reusable(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    _write_source(source)

    first = create_prepared_artifact(
        source,
        output_root=tmp_path / "artifacts",
        split="benchmark",
        config_sha256="a" * 64,
        environment_lock_identity="b" * 64,
    )
    second = create_prepared_artifact(
        source,
        output_root=tmp_path / "artifacts",
        split="benchmark",
        config_sha256="a" * 64,
        environment_lock_identity="b" * 64,
    )

    assert first == second
    manifest, data_path = load_prepared_artifact(first)
    assert first.name == manifest.artifact_id
    assert manifest.row_count == 2
    assert manifest.environment_lock_identity == "b" * 64
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["task_id"].startswith("ng:") for row in rows)
    assert len({row["task_id"] for row in rows}) == 2


def test_prepared_artifact_identity_is_independent_of_json_formatting(tmp_path: Path) -> None:
    compact = tmp_path / "compact.jsonl"
    pretty = tmp_path / "pretty.jsonl"
    row = {"task_source": "example", "responses_create_params": {"input": "question"}}
    compact.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    pretty.write_text(
        '{"responses_create_params": {"input": "question"}, "task_source": "example"}\n',
        encoding="utf-8",
    )

    compact_artifact = create_prepared_artifact(compact, output_root=tmp_path / "artifacts", split="benchmark")
    pretty_artifact = create_prepared_artifact(pretty, output_root=tmp_path / "artifacts", split="benchmark")

    assert compact_artifact == pretty_artifact


def test_load_prepared_artifact_rejects_mutated_data(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    _write_source(source)
    artifact = create_prepared_artifact(source, output_root=tmp_path / "artifacts", split="benchmark")
    _, data_path = load_prepared_artifact(artifact)
    data_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="data digest mismatch"):
        load_prepared_artifact(artifact)


def test_create_prepared_artifact_rejects_duplicate_explicit_task_ids(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(
        '{"task_id":"duplicate","responses_create_params":{"input":"first"}}\n'
        '{"task_id":"duplicate","responses_create_params":{"input":"second"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate task_id"):
        create_prepared_artifact(source, output_root=tmp_path / "artifacts", split="benchmark")


def test_create_prepared_artifact_rejects_unsafe_split(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    _write_source(source)

    with pytest.raises(ValueError, match="Prepared artifact split"):
        create_prepared_artifact(source, output_root=tmp_path / "artifacts", split="../../escape")
