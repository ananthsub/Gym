# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
from omegaconf import DictConfig

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.image import CompositionBOM, CompositionImage
from nemo_gym.environment.launch_plan import render_launch_plan
from nemo_gym.environment.lock import ComponentLockRecord, EnvironmentLockRecord, GymIdentity


_HASH = "1" * 64
_IMAGE_DIGEST = f"sha256:{'2' * 64}"
_REQUIREMENTS = f"demo==1 --hash=sha256:{'3' * 64}\n"


def _write_artifacts(tmp_path: Path, *, instances: tuple[str, ...] = ("demo_server",)) -> tuple[Path, Path]:
    component = ComponentLockRecord(
        instances=("demo_server",),
        component_type="resources_servers",
        implementation="demo",
        source_kind="requirements",
        source_path="resources_servers/demo",
        platform="linux/amd64",
        python_version="3.13.14",
        uv_version="0.12.2",
        parent_constraints=(),
        config_sha256=_HASH,
        content_sha256=_HASH,
        requirements_sha256=hashlib.sha256(_REQUIREMENTS.encode()).hexdigest(),
        requirements_lock=_REQUIREMENTS,
    )
    lock = EnvironmentLockRecord(
        base_image=f"registry.example/base@sha256:{_HASH}",
        platform="linux/amd64",
        python_version="3.13.14",
        uv_version="0.12.2",
        gym_source_sha256=_HASH,
        gym=GymIdentity(name="nemo-gym", version="0.7.0", source_kind="registry"),
        parent_packages=(),
        config_sha256=_HASH,
        components=(component,),
    )
    bom = CompositionBOM(
        environment_lock_identity=lock.identity,
        images=(
            CompositionImage(
                target="head",
                instances=("head_server",),
                image=f"registry.example/head@{_IMAGE_DIGEST}",
                digest=_IMAGE_DIGEST,
            ),
            CompositionImage(
                target="resources_servers/demo",
                instances=instances,
                image=f"registry.example/demo@{_IMAGE_DIGEST}",
                digest=_IMAGE_DIGEST,
                component_content_sha256=_HASH,
            ),
        ),
    )
    lock_path = tmp_path / "environment.lock.json"
    bom_path = tmp_path / "composition.bom.json"
    lock_path.write_text(lock.canonical_json(), encoding="utf-8")
    bom_path.write_text(bom.canonical_json(), encoding="utf-8")
    return bom_path, lock_path


def _runtime_config(*, entrypoint: str = "app.py", port: int = 12001) -> DictConfig:
    return DictConfig(
        {
            "head_server": {"host": "0.0.0.0", "port": 11000},
            "demo_server": {
                "resources_servers": {
                    "demo": {
                        "entrypoint": entrypoint,
                        "domain": "math",
                        "host": "0.0.0.0",
                        "port": port,
                    }
                }
            },
        }
    )


def test_render_launch_plan_covers_head_and_component(tmp_path: Path) -> None:
    bom_path, lock_path = _write_artifacts(tmp_path)

    plan = render_launch_plan(
        bom_path=bom_path,
        lock_path=lock_path,
        runtime_config=_runtime_config(),
    )

    assert [service.instance for service in plan.services] == ["head_server", "demo_server"]
    assert plan.services[1].command == ("gym", "env", "start-server")
    assert plan.services[1].arguments == ("--instance", "demo_server")
    assert plan.services[1].ports == (12001,)
    assert plan.services[1].dependencies == ("head_server",)
    assert "NEMO_GYM_CONFIG_FILE" in plan.services[1].environment


def test_render_launch_plan_rejects_incomplete_bom(tmp_path: Path) -> None:
    bom_path, lock_path = _write_artifacts(tmp_path, instances=("other_server",))

    with pytest.raises(ConfigError, match="instance coverage mismatch"):
        render_launch_plan(
            bom_path=bom_path,
            lock_path=lock_path,
            runtime_config=_runtime_config(),
        )


def test_render_launch_plan_rejects_unsafe_entrypoint(tmp_path: Path) -> None:
    bom_path, lock_path = _write_artifacts(tmp_path)

    with pytest.raises(ConfigError, match="unsafe entrypoint"):
        render_launch_plan(
            bom_path=bom_path,
            lock_path=lock_path,
            runtime_config=_runtime_config(entrypoint="../app.py"),
        )


def test_render_launch_plan_rejects_unconfigured_port(tmp_path: Path) -> None:
    bom_path, lock_path = _write_artifacts(tmp_path)

    with pytest.raises(ConfigError, match="configure a port"):
        render_launch_plan(
            bom_path=bom_path,
            lock_path=lock_path,
            runtime_config=_runtime_config(port=-1),
        )
