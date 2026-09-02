# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import nemo_gym.environment.image as environment_image
from nemo_gym import __version__
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.image import (
    build_packed_image,
    build_split_images,
    load_verified_environment_lock,
    packed_build_command,
)
from nemo_gym.environment.lock import (
    ComponentLockRecord,
    EnvironmentLockRecord,
    GymIdentity,
    PackageIdentity,
    component_content_digest,
)


_HASH = "0" * 64
_REQUIREMENTS_LOCK = f"example-package==1.0 --hash=sha256:{_HASH}\n"


def _write_lock(tmp_path: Path) -> tuple[Path, Path, EnvironmentLockRecord]:
    source_root = tmp_path / "source"
    component_dir = source_root / "resources_servers" / "example"
    component_dir.mkdir(parents=True)
    (component_dir / "requirements.txt").write_text("example-package==1.0\n", encoding="utf-8")
    for filename in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        (source_root / filename).write_text(filename, encoding="utf-8")
    (source_root / "nemo_gym").mkdir()

    component = ComponentLockRecord(
        instances=("example_server",),
        component_type="resources_servers",
        implementation="example",
        source_kind="requirements",
        source_path="resources_servers/example",
        platform="linux/amd64",
        python_version="3.13.14",
        uv_version="0.11.29",
        parent_constraints=("nemo-gym==1.0.0",),
        config_sha256=_HASH,
        content_sha256=component_content_digest(component_dir),
        requirements_sha256=environment_image.hashlib.sha256(_REQUIREMENTS_LOCK.encode()).hexdigest(),
        requirements_lock=_REQUIREMENTS_LOCK,
    )
    lock = EnvironmentLockRecord(
        platform="linux/amd64",
        python_version="3.13.14",
        uv_version="0.11.29",
        gym=GymIdentity(name="nemo-gym", version=__version__, source_kind="registry"),
        parent_packages=(
            PackageIdentity(name="ray", version="2.56.1"),
            PackageIdentity(name="openai", version="2.44.0"),
        ),
        config_sha256=_HASH,
        components=(component,),
    )
    lock_path = tmp_path / "environment.lock.json"
    lock_path.write_text(lock.canonical_json() + "\n", encoding="utf-8")
    return lock_path, source_root, lock


def test_load_verified_environment_lock_rejects_changed_component(tmp_path: Path) -> None:
    lock_path, source_root, _ = _write_lock(tmp_path)
    (source_root / "resources_servers/example/app.py").write_text("changed = True\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Component content changed"):
        load_verified_environment_lock(lock_path, source_root=source_root)


def test_component_source_can_come_from_external_search_root(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    external = tmp_path / "plugins/resources_servers/example"
    external.mkdir(parents=True)
    monkeypatch.setattr(
        environment_image,
        "_resolve_under_cwd_or_install",
        lambda relative_path, validator: external,
    )

    resolved = environment_image._component_source_path(source_root, Path("resources_servers/example"))

    assert resolved == external.resolve()


def test_load_verified_environment_lock_rejects_tampered_identity(tmp_path: Path) -> None:
    lock_path, source_root, _ = _write_lock(tmp_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["identity"] = "f" * 64
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="Environment lock is invalid"):
        load_verified_environment_lock(lock_path, source_root=source_root)


def test_packed_build_command_carries_reproducibility_labels(tmp_path: Path) -> None:
    _, _, lock = _write_lock(tmp_path)

    command = packed_build_command(
        context_root=tmp_path,
        lock=lock,
        base_image="registry.example/gym-base@sha256:abc",
        tag="registry.example/example:locked",
        platform="linux/amd64",
        load=True,
    )

    rendered = " ".join(command)
    assert "LOCK_IDENTITY=" + lock.identity in rendered
    assert "LOCK_SCHEMA=nemo-gym/environment-lock/v1" in rendered
    assert "--platform linux/amd64" in rendered
    assert "--load" in command


def test_oci_platform_normalizes_lock_target_triples() -> None:
    assert environment_image._oci_platform("x86_64-unknown-linux-gnu") == "linux/amd64"
    assert environment_image._oci_platform("aarch64-unknown-linux-gnu") == "linux/arm64"
    assert environment_image._oci_platform("linux/amd64") == "linux/amd64"


def test_build_packed_image_verifies_before_docker(monkeypatch, tmp_path: Path) -> None:
    lock_path, source_root, lock = _write_lock(tmp_path)
    docker_run = MagicMock()
    monkeypatch.setattr(environment_image.subprocess, "run", docker_run)
    monkeypatch.setattr(
        environment_image,
        "PACKED_DOCKERFILE",
        Path(environment_image.__file__).parents[2] / "docker/Dockerfile.packed",
    )

    build_packed_image(
        lock_path,
        source_root=source_root,
        base_image="registry.example/gym-base@sha256:abc",
        tag="example:locked",
        platform="linux/amd64",
    )

    command = docker_run.call_args.args[0]
    assert command[:3] == ["docker", "buildx", "build"]
    assert f"LOCK_IDENTITY={lock.identity}" in command


def test_packed_dockerfile_has_immutable_runtime_contract() -> None:
    dockerfile = (Path(environment_image.__file__).parents[2] / "docker/Dockerfile.packed").read_text(encoding="utf-8")

    assert "NEMO_GYM_RUNTIME_INSTALL_POLICY=require-existing" in dockerfile
    assert 'LABEL io.nvidia.nemo-gym.runtime.protocol="${RUNTIME_PROTOCOL}"' in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert 'ENTRYPOINT ["gym", "env", "start"]' in dockerfile
    runtime_stage = dockerfile.split(" AS runtime", maxsplit=1)[1]
    assert "uv sync" not in runtime_stage
    assert "pip install" not in runtime_stage


def test_build_split_images_writes_digest_pinned_bom(monkeypatch, tmp_path: Path) -> None:
    lock_path, source_root, lock = _write_lock(tmp_path)
    commands = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(command)
        metadata_path = Path(command[command.index("--metadata-file") + 1])
        metadata_path.write_text(
            json.dumps({"containerimage.digest": f"sha256:{len(commands):064x}"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(environment_image.subprocess, "run", fake_run)
    bom_path = tmp_path / "composition.bom.json"

    bom = build_split_images(
        lock_path,
        source_root=source_root,
        base_image="registry.example/gym-base@sha256:abc",
        repository="registry.example/gym/example",
        output_path=bom_path,
        platform="linux/amd64",
    )

    assert bom.environment_lock_identity == lock.identity
    assert [image.target for image in bom.images] == ["head", "resources_servers/example"]
    assert all("@sha256:" in image.image for image in bom.images)
    assert all("--push" in command for command in commands)
    assert json.loads(bom_path.read_text(encoding="utf-8"))["identity"] == bom.identity


def test_server_dockerfile_has_distinct_foreground_targets() -> None:
    dockerfile = (Path(environment_image.__file__).parents[2] / "docker/Dockerfile.server").read_text(encoding="utf-8")

    assert "FROM runtime AS component" in dockerfile
    assert 'ENTRYPOINT ["gym", "env", "start-server"]' in dockerfile
    assert "FROM runtime AS head" in dockerfile
    assert 'ENTRYPOINT ["gym", "env", "start-head"]' in dockerfile
