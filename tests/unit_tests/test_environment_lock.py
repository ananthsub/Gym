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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig
from pydantic import ValidationError

import nemo_gym.cli.env
import nemo_gym.environment.lock as environment_lock
from nemo_gym.cli.setup_command import ComponentInstallPlan, component_install_plan
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.lock import (
    ComponentLockRecord,
    EnvironmentLockRecord,
    GymIdentity,
    PackageIdentity,
    build_environment_lock,
    compile_component_requirements,
    gym_build_source_digest,
    lock_environment,
    redacted_config_digest,
    validate_dependency_inputs,
)


_HASH = "a" * 64
_HASHED_REQUIREMENT = f"demo==1 \\\n    --hash=sha256:{_HASH}\n"


def _plan(component_dir: Path, source_kind: str = "requirements") -> ComponentInstallPlan:
    dependency_name = "requirements.txt" if source_kind == "requirements" else "pyproject.toml"
    (component_dir / dependency_name).write_text("demo==1\n", encoding="utf-8")
    return ComponentInstallPlan(
        component_dir=component_dir,
        venv_path=component_dir / ".venv",
        python_version="3.13.14",
        dependency_source=source_kind,
        head_server_dependencies=("ray[default]==2.49.0", "openai==2.44.0"),
        is_editable_gym_install=False,
        skip_venv_setup=False,
        use_python_flag=False,
        verbose=False,
        has_overrides=False,
        nemo_gym_install_flags="",
        nemo_gym_version_spec="",
    )


def _component_record(**overrides) -> ComponentLockRecord:
    values = {
        "instances": ("demo_instance",),
        "component_type": "resources_servers",
        "implementation": "demo",
        "source_kind": "requirements",
        "source_path": "resources_servers/demo",
        "platform": "x86_64-unknown-linux-gnu",
        "python_version": "3.13.14",
        "uv_version": "0.12.2",
        "parent_constraints": ("nemo-gym==0.7.0", "ray[default]==2.49.0", "openai==2.44.0"),
        "config_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "requirements_sha256": "3" * 64,
        "requirements_lock": _HASHED_REQUIREMENT,
    }
    values.update(overrides)
    return ComponentLockRecord(**values)


def test_lock_records_have_stable_canonical_identity() -> None:
    first_component = _component_record()
    second_component = _component_record()
    first = EnvironmentLockRecord(
        platform="x86_64-unknown-linux-gnu",
        python_version="3.13.14",
        uv_version="0.12.2",
        gym=GymIdentity(name="nemo-gym", version="0.7.0", source_kind="registry"),
        parent_packages=(
            PackageIdentity(name="ray", version="2.49.0"),
            PackageIdentity(name="openai", version="2.44.0"),
        ),
        config_sha256="4" * 64,
        components=(first_component,),
    )
    second = EnvironmentLockRecord.model_validate_json(first.canonical_json())

    assert first_component.identity == second_component.identity
    assert first.identity == second.identity
    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json())["schema_version"] == "nemo-gym/environment-lock/v1"


def test_lock_record_rejects_tampered_identity() -> None:
    record = _component_record()
    payload = record.model_dump(mode="json")
    payload["requirements_lock"] = "changed"

    with pytest.raises(ValidationError, match="lock identity mismatch"):
        ComponentLockRecord.model_validate(payload)


def test_gym_build_source_digest_covers_runtime_lock_and_image_recipe(tmp_path: Path) -> None:
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym/runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/Dockerfile.packed").write_text("FROM base\n", encoding="utf-8")
    (tmp_path / "docker/Dockerfile.server").write_text("FROM base\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    original = gym_build_source_digest(tmp_path)

    (tmp_path / "uv.lock").write_text("version = 2\n", encoding="utf-8")

    assert gym_build_source_digest(tmp_path) != original


def test_redacted_config_digest_does_not_depend_on_secret_values() -> None:
    first = {"model": {"api_key": "secret-one", "headers": ["Bearer secret-two"], "name": "same"}}
    second = {"model": {"api_key": "different", "headers": ["different"], "name": "same"}}

    assert redacted_config_digest(first) == redacted_config_digest(second)


@pytest.mark.parametrize(
    "dependency",
    [
        "--extra-index-url https://packages.example/simple",
        "demo @ git+https://github.com/example/demo.git@main",
        "demo @ https://user:password@example.com/demo.whl",
        "demo @ https://example.com/demo.whl?token=secret",
    ],
)
def test_strict_validation_rejects_mutable_or_secret_dependency_inputs(tmp_path: Path, dependency: str) -> None:
    plan = _plan(tmp_path)
    (tmp_path / "requirements.txt").write_text(dependency + "\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        validate_dependency_inputs(plan)


def test_strict_validation_accepts_full_git_commit(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    commit = "a" * 40
    (tmp_path / "requirements.txt").write_text(
        f"demo @ git+https://github.com/example/demo.git@{commit}\n",
        encoding="utf-8",
    )

    validate_dependency_inputs(plan)


def test_compile_uses_local_uv_hashes_platform_and_parent_constraints(monkeypatch, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    calls: list[tuple[list[str], dict]] = []
    compiled_inputs: list[str] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        compiled_inputs.append(Path(command[-1]).read_text(encoding="utf-8"))
        output = Path(command[command.index("--output-file") + 1])
        output.write_text(_HASHED_REQUIREMENT, encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(environment_lock.subprocess, "run", fake_run)
    monkeypatch.setenv("UV_INDEX_URL", "https://user:secret@example.com/simple")

    result = compile_component_requirements(
        plan,
        platform="linux/amd64",
        parent_constraints=("nemo-gym==0.7.0", "ray[default]==2.49.0", "openai==2.44.0"),
        uv_executable="/venv/bin/uv",
        strict=True,
    )

    command, kwargs = calls[0]
    assert command[:3] == ["/venv/bin/uv", "pip", "compile"]
    assert "--generate-hashes" in command
    assert command[command.index("--python-platform") + 1] == "x86_64-unknown-linux-gnu"
    assert "UV_INDEX_URL" not in kwargs["env"]
    assert compiled_inputs[0].splitlines() == [
        "nemo-gym==0.7.0",
        "ray[default]==2.49.0",
        "openai==2.44.0",
    ]
    assert result.startswith("demo==1")


def test_compile_rejects_unhashed_uv_output(monkeypatch, tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output-file") + 1])
        output.write_text("demo==1\n", encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(environment_lock.subprocess, "run", fake_run)

    with pytest.raises(ConfigError, match="unhashed dependency lock"):
        compile_component_requirements(
            plan,
            platform="x86_64-unknown-linux-gnu",
            parent_constraints=("nemo-gym==0.7.0",),
            uv_executable="/venv/bin/uv",
            strict=True,
        )


def test_compile_expands_editable_gym_source_dependencies(monkeypatch, tmp_path: Path) -> None:
    component_dir = tmp_path / "resources_servers" / "demo"
    component_dir.mkdir(parents=True)
    (component_dir / "requirements.txt").write_text("-e nemo-gym[dev] @ ../../\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "nemo-gym"\nversion = "0.7.0"\ndependencies = ["base-dep==1"]\n'
        '[project.optional-dependencies]\ndev = ["dev-dep==2"]\n',
        encoding="utf-8",
    )
    config = DictConfig(
        {
            "uv_venv_dir": str(tmp_path),
            "skip_venv_if_present": False,
            "python_version": "3.13.14",
            "head_server_deps": ["ray[default]==2.49.0", "openai==2.44.0"],
        }
    )
    plan = component_install_plan(component_dir, config)
    compiled_inputs: list[list[str]] = []

    def fake_run(command, **kwargs):
        compiled_inputs.append([Path(path).read_text(encoding="utf-8") for path in command[-3:]])
        output = Path(command[command.index("--output-file") + 1])
        output.write_text(_HASHED_REQUIREMENT, encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(environment_lock.subprocess, "run", fake_run)

    compile_component_requirements(
        plan,
        platform="linux/amd64",
        parent_constraints=("ray[default]==2.49.0", "openai==2.44.0"),
        uv_executable="/venv/bin/uv",
        strict=True,
    )

    component_input, gym_input, parent_input = compiled_inputs[0]
    assert component_input == "\n"
    assert gym_input.splitlines() == ["base-dep==1", "dev-dep==2"]
    assert parent_input.splitlines() == ["ray[default]==2.49.0", "openai==2.44.0"]


def test_component_plan_captures_runtime_dependency_decisions(tmp_path: Path) -> None:
    component_dir = tmp_path / "resources_servers" / "demo"
    component_dir.mkdir(parents=True)
    (component_dir / "requirements.txt").write_text("demo\n", encoding="utf-8")
    (component_dir / "overrides.txt").write_text("demo==1\n", encoding="utf-8")
    config = DictConfig(
        {
            "uv_venv_dir": str(tmp_path),
            "skip_venv_if_present": False,
            "python_version": "3.13.14",
            "head_server_deps": ["ray[default]==2.49.0", "openai==2.44.0"],
            "uv_pip_set_python": True,
            "pip_install_verbose": True,
        }
    )

    plan = component_install_plan(component_dir, config)

    assert plan.dependency_source == "requirements"
    assert plan.dependency_file == component_dir / "requirements.txt"
    assert plan.has_overrides
    assert plan.use_python_flag
    assert plan.verbose
    assert plan.head_server_dependencies[-1] == "openai==2.44.0"


def test_build_lock_uses_current_component_search_resolution(monkeypatch, tmp_path: Path) -> None:
    component_dir = tmp_path / "plugin" / "resources_servers" / "demo"
    component_dir.mkdir(parents=True)
    (component_dir / "requirements.txt").write_text("demo==1\n", encoding="utf-8")
    config = DictConfig(
        {
            "demo_instance": {
                "resources_servers": {
                    "demo": {
                        "entrypoint": "app.py",
                        "domain": "math",
                        "host": "127.0.0.1",
                        "port": 10001,
                        "api_key": "must-not-appear",
                    }
                }
            },
            "second_demo_instance": {
                "resources_servers": {
                    "demo": {
                        "entrypoint": "app.py",
                        "domain": "math",
                        "host": "127.0.0.1",
                        "port": 10002,
                    }
                }
            },
            "head_server_deps": ["ray[default]==2.49.0", "openai==2.44.0"],
            "python_version": "3.13.14",
            "uv_venv_dir": str(tmp_path),
            "skip_venv_if_present": False,
            "uv_pip_set_python": False,
            "pip_install_verbose": False,
        }
    )
    resolved_paths: list[Path] = []

    def fake_resolve(path: Path) -> Path:
        resolved_paths.append(path)
        return component_dir

    def fake_run(command, **kwargs):
        if command[1:] == ["--version"]:
            return SimpleNamespace(stdout="uv 0.12.2\n", stderr="")
        output = Path(command[command.index("--output-file") + 1])
        output.write_text(_HASHED_REQUIREMENT, encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(nemo_gym.cli.env, "_resolve_server_dir", fake_resolve)
    versions = {"nemo-gym": "0.7.0", "ray": "2.49.0", "openai": "2.44.0"}
    monkeypatch.setattr(environment_lock.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        environment_lock.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=versions[name], read_text=lambda filename: None),
    )
    monkeypatch.setattr(environment_lock.shutil, "which", lambda name: "/venv/bin/uv")
    monkeypatch.setattr(environment_lock.subprocess, "run", fake_run)

    lock = build_environment_lock(config, platform="x86_64-unknown-linux-gnu", strict=True)

    assert resolved_paths == [Path("resources_servers/demo")]
    assert len(lock.components) == 1
    assert lock.components[0].instances == ("demo_instance", "second_demo_instance")
    assert lock.components[0].source_path == "resources_servers/demo"
    assert lock.components[0].parent_constraints[:3] == (
        "nemo-gym==0.7.0",
        "ray[default]==2.49.0",
        "openai==2.44.0",
    )
    assert lock.gym == GymIdentity(name="nemo-gym", version="0.7.0", source_kind="registry")
    assert lock.parent_packages == (
        PackageIdentity(name="ray", version="2.49.0"),
        PackageIdentity(name="openai", version="2.44.0"),
    )
    assert "must-not-appear" not in lock.canonical_json()


def test_lock_environment_is_strict_and_uses_canonical_default_filename(monkeypatch, tmp_path: Path) -> None:
    config = DictConfig({"platform": "x86_64-unknown-linux-gnu"})
    expected = EnvironmentLockRecord(
        platform="x86_64-unknown-linux-gnu",
        python_version="3.13.14",
        uv_version="0.12.2",
        gym=GymIdentity(name="nemo-gym", version="0.7.0", source_kind="registry"),
        parent_packages=(
            PackageIdentity(name="ray", version="2.49.0"),
            PackageIdentity(name="openai", version="2.44.0"),
        ),
        config_sha256="4" * 64,
        components=(),
    )
    calls: list[tuple[str, bool]] = []

    def fake_build(global_config_dict, *, platform, strict):
        calls.append((platform, strict))
        return expected

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(environment_lock, "get_global_config_dict", lambda **kwargs: config)
    monkeypatch.setattr(environment_lock, "build_environment_lock", fake_build)

    lock_environment()

    output = tmp_path / "environment.lock.json"
    assert calls == [("x86_64-unknown-linux-gnu", True)]
    assert output.read_text(encoding="utf-8") == expected.canonical_json() + "\n"
