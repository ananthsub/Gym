# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.build_manifest import (
    component_source_closure,
    infer_component_source_dependencies,
    load_component_build_manifest,
)


def _component(root: Path, path: str, source: str = "") -> Path:
    component = root / path
    component.mkdir(parents=True)
    (component / "app.py").write_text(source, encoding="utf-8")
    return component


def test_source_closure_infers_transitive_cross_component_imports(tmp_path: Path) -> None:
    _component(tmp_path, "responses_api_models/leaf")
    _component(
        tmp_path,
        "responses_api_models/middle",
        "from responses_api_models.leaf.app import Server\n",
    )
    _component(
        tmp_path,
        "responses_api_models/root",
        "from responses_api_models.middle.app import Server\n",
    )

    closure = component_source_closure(
        "responses_api_models/root",
        resolve_component=lambda path: tmp_path / path,
    )

    assert closure == ("responses_api_models/leaf", "responses_api_models/middle")


def test_build_manifest_adds_dynamic_source_dependency_and_argv_step(tmp_path: Path) -> None:
    component = _component(tmp_path, "responses_api_agents/root")
    _component(tmp_path, "resources_servers/dynamic")
    (component / "component.build.yaml").write_text(
        """
schema_version: nemo-gym/component-build/v1
source_dependencies:
  - resources_servers/dynamic
build_steps:
  - argv: ["{python}", "prepare.py"]
    expected_paths: [".prepared/tool"]
""".lstrip(),
        encoding="utf-8",
    )

    manifest = load_component_build_manifest(component)
    closure = component_source_closure(
        "responses_api_agents/root",
        resolve_component=lambda path: tmp_path / path,
    )

    assert manifest.build_steps[0].argv == ("{python}", "prepare.py")
    assert closure == ("resources_servers/dynamic",)


def test_build_step_cycle_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    first = _component(
        tmp_path,
        "resources_servers/first",
        "from resources_servers.second.app import Server\n",
    )
    _component(
        tmp_path,
        "resources_servers/second",
        "from resources_servers.first.app import Server\n",
    )
    (first / "component.build.yaml").write_text(
        'schema_version: nemo-gym/component-build/v1\nbuild_steps:\n  - argv: ["true"]\n    expected_paths: [".built"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="cycle makes build ordering ambiguous"):
        component_source_closure(
            "resources_servers/first",
            resolve_component=lambda path: tmp_path / path,
        )


def test_import_inference_ignores_non_component_modules(tmp_path: Path) -> None:
    component = _component(
        tmp_path,
        "resources_servers/root",
        "import asyncio\nfrom nemo_gym.server_utils import request\n",
    )

    assert infer_component_source_dependencies(component) == ()
