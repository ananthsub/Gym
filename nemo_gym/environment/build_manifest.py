# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nemo_gym.config_types import ConfigError


COMPONENT_BUILD_MANIFEST_SCHEMA_VERSION = "nemo-gym/component-build/v1"
COMPONENT_BUILD_MANIFEST_NAME = "component.build.yaml"
COMPONENT_ROOTS = frozenset({"resources_servers", "responses_api_agents", "responses_api_models"})


def _safe_component_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or path.parts[0] not in COMPONENT_ROOTS:
        raise ValueError(f"component source dependency must be ROOT/IMPLEMENTATION, got {value!r}")
    return path.as_posix()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"expected path must be relative and cannot traverse parents, got {value!r}")
    return path.as_posix()


class ComponentBuildStep(BaseModel):
    """One argv-only build step and the paths it must materialize."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...] = Field(min_length=1)
    expected_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or "\x00" in argument for argument in value):
            raise ValueError("build-step argv entries must be non-empty and cannot contain NUL")
        return value

    @field_validator("expected_paths")
    @classmethod
    def validate_expected_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_safe_relative_path(path) for path in value)


class ComponentBuildManifest(BaseModel):
    """Versioned build-only metadata for a Gym component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nemo-gym/component-build/v1"] = COMPONENT_BUILD_MANIFEST_SCHEMA_VERSION
    source_dependencies: tuple[str, ...] = ()
    build_steps: tuple[ComponentBuildStep, ...] = ()

    @field_validator("source_dependencies")
    @classmethod
    def validate_source_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({_safe_component_path(path) for path in value}))
        return normalized


def load_component_build_manifest(component_dir: Path) -> ComponentBuildManifest:
    """Load optional build metadata from a component directory."""
    manifest_path = component_dir / COMPONENT_BUILD_MANIFEST_NAME
    if not manifest_path.is_file():
        return ComponentBuildManifest()
    try:
        value = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
        return ComponentBuildManifest.model_validate(value)
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid component build manifest {manifest_path}: {exc}") from exc


def infer_component_source_dependencies(component_dir: Path) -> tuple[str, ...]:
    """Infer direct cross-component imports from Python source."""
    dependencies: set[str] = set()
    for path in sorted(component_dir.rglob("*.py")):
        if any(part in {".venv", "__pycache__"} for part in path.relative_to(component_dir).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ConfigError(f"Cannot inspect component imports in {path}: {exc}") from exc
        for node in ast.walk(tree):
            names: Iterable[str]
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = (node.module,)
            else:
                continue
            for name in names:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] in COMPONENT_ROOTS:
                    dependencies.add(f"{parts[0]}/{parts[1]}")
    return tuple(sorted(dependencies))


def component_source_closure(
    component_path: str,
    *,
    resolve_component: Callable[[str], Path],
) -> tuple[str, ...]:
    """Return transitive inferred and explicitly declared source dependencies."""
    visiting: list[str] = []
    visited: set[str] = set()
    ordered: list[str] = []
    manifests: dict[str, ComponentBuildManifest] = {}

    def visit(path: str) -> None:
        normalized = _safe_component_path(path)
        if normalized in visiting:
            cycle = visiting[visiting.index(normalized) :] + [normalized]
            if any(manifests.get(item, ComponentBuildManifest()).build_steps for item in cycle):
                raise ConfigError(
                    f"Component source dependency cycle makes build ordering ambiguous: {' -> '.join(cycle)}"
                )
            return
        if normalized in visited:
            return
        component_dir = resolve_component(normalized)
        manifest = load_component_build_manifest(component_dir)
        manifests[normalized] = manifest
        visiting.append(normalized)
        dependencies = set(infer_component_source_dependencies(component_dir))
        dependencies.update(manifest.source_dependencies)
        dependencies.discard(normalized)
        for dependency in sorted(dependencies):
            visit(dependency)
        visiting.pop()
        visited.add(normalized)
        if normalized != component_path:
            ordered.append(normalized)

    visit(component_path)
    return tuple(ordered)
