# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate split-image composition and render an orchestrator-neutral launch plan."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from nemo_gym.config_types import ConfigError, ServerInstanceConfig, is_server_ref
from nemo_gym.environment.image import CompositionBOM
from nemo_gym.environment.lock import EnvironmentLockRecord
from nemo_gym.environment.protocol import LAUNCH_PLAN_SCHEMA_VERSION, RUNTIME_PROTOCOL_VERSION
from nemo_gym.global_config import DEFAULT_HEAD_SERVER_PORT, GlobalConfigDictParser, get_global_config_dict


CONFIG_MOUNT_PATH = "/etc/nemo-gym/runtime.yaml"


class ConfigMountContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    container_path: str = CONFIG_MOUNT_PATH
    read_only: bool = True
    environment_variable: str = "NEMO_GYM_CONFIG_FILE"


class LaunchService(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["head", "component"]
    instance: str
    component: str | None = None
    image: str
    command: tuple[str, ...]
    arguments: tuple[str, ...] = ()
    environment: dict[str, str]
    ports: tuple[int, ...]
    dependencies: tuple[str, ...] = ()


class LaunchPlan(BaseModel):
    """A secret-free description of containers an external orchestrator must run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nemo-gym/launch-plan/v1"] = LAUNCH_PLAN_SCHEMA_VERSION
    runtime_protocol: Literal["nemo-gym/runtime/v1"] = RUNTIME_PROTOCOL_VERSION
    identity: str = ""
    environment_lock_identity: str
    composition_bom_identity: str
    config_mount: ConfigMountContract = ConfigMountContract()
    external_requirements: tuple[str, ...] = ("ray_address",)
    services: tuple[LaunchService, ...]

    @model_validator(mode="after")
    def populate_or_validate_identity(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"identity"})
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.identity and self.identity != expected:
            raise ValueError(f"Launch plan identity mismatch: expected {expected}, got {self.identity}")
        object.__setattr__(self, "identity", expected)
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _load_models(bom_path: Path, lock_path: Path) -> tuple[CompositionBOM, EnvironmentLockRecord]:
    try:
        bom = CompositionBOM.model_validate_json(bom_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise ConfigError(f"Composition BOM is invalid: {exc}") from exc
    try:
        lock = EnvironmentLockRecord.model_validate_json(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise ConfigError(f"Environment lock is invalid: {exc}") from exc
    if bom.environment_lock_identity != lock.identity:
        raise ConfigError(
            f"Composition BOM requires environment lock {bom.environment_lock_identity}, got {lock.identity}"
        )
    return bom, lock


def _safe_entrypoint(value: str, *, instance: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ConfigError(f"Server instance {instance!r} has an unsafe entrypoint: {value!r}")


def _server_dependencies(server: ServerInstanceConfig) -> tuple[str, ...]:
    dependencies: set[str] = {"head_server"}

    def visit(value: Any) -> None:
        if isinstance(value, (DictConfig, Mapping)):
            reference = is_server_ref(value)
            if reference is not None:
                dependencies.add(reference.name)
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(server.get_inner_run_server_config_dict())
    dependencies.discard(server.name)
    return tuple(sorted(dependencies))


def render_launch_plan(
    *,
    bom_path: Path,
    lock_path: Path,
    runtime_config: DictConfig,
) -> LaunchPlan:
    """Validate build artifacts against resolved runtime config and render a launch plan."""
    bom, lock = _load_models(bom_path, lock_path)
    parser = GlobalConfigDictParser()
    servers = parser.filter_for_server_instance_configs(runtime_config)
    parser.raise_on_no_server_instances(runtime_config)

    head_images = [image for image in bom.images if image.target == "head"]
    if len(head_images) != 1 or head_images[0].instances != ("head_server",):
        raise ConfigError("Composition BOM must contain exactly one head image for head_server")
    component_images = {image.target: image for image in bom.images if image.target != "head"}
    if len(component_images) != len(bom.images) - 1:
        raise ConfigError("Composition BOM contains duplicate component targets")
    locked_components = {
        f"{component.component_type}/{component.implementation}": component for component in lock.components
    }
    if set(component_images) != set(locked_components):
        raise ConfigError(
            "Split composition component coverage mismatch: "
            f"lock={sorted(locked_components)}, bom={sorted(component_images)}"
        )
    for target, component in locked_components.items():
        image = component_images[target]
        if image.component_content_sha256 != component.content_sha256:
            raise ConfigError(f"Composition image source digest does not match the lock for {target}")

    locked_instances = {instance for component in lock.components for instance in component.instances}
    runtime_instances = {server.name for server in servers}
    bom_instances = {instance for image in component_images.values() for instance in image.instances}
    if locked_instances != runtime_instances or locked_instances != bom_instances:
        raise ConfigError(
            "Split composition instance coverage mismatch: "
            f"lock={sorted(locked_instances)}, runtime={sorted(runtime_instances)}, bom={sorted(bom_instances)}"
        )

    head_config = runtime_config.get("head_server", {})
    head_port = head_config.get("port", DEFAULT_HEAD_SERVER_PORT)
    if not isinstance(head_port, int) or not 1 <= head_port <= 65535:
        raise ConfigError(f"Head server port must be configured in 1..65535, got {head_port!r}")
    services = [
        LaunchService(
            role="head",
            instance="head_server",
            image=head_images[0].image,
            command=("gym", "env", "start-head"),
            environment={
                "NEMO_GYM_CONFIG_FILE": CONFIG_MOUNT_PATH,
                "NEMO_GYM_RUNTIME_INSTALL_POLICY": "require-existing",
            },
            ports=(head_port,),
        )
    ]
    used_ports = {head_port}
    for server in sorted(servers, key=lambda value: value.name):
        inner = server.get_inner_run_server_config()
        _safe_entrypoint(inner.entrypoint, instance=server.name)
        if inner.port is None or not 1 <= inner.port <= 65535:
            raise ConfigError(f"Server instance {server.name!r} must configure a port in 1..65535")
        if inner.port in used_ports:
            raise ConfigError(f"Port {inner.port} is assigned more than once in the split launch plan")
        used_ports.add(inner.port)
        implementation = next(iter(getattr(server, server.SERVER_TYPE)))
        target = f"{server.SERVER_TYPE}/{implementation}"
        image = component_images.get(target)
        if image is None or server.name not in image.instances:
            raise ConfigError(f"Composition BOM has no matching image for {server.name!r} ({target})")
        dependencies = _server_dependencies(server)
        unknown_dependencies = set(dependencies) - runtime_instances - {"head_server"}
        if unknown_dependencies:
            raise ConfigError(
                f"Server instance {server.name!r} has unknown launch dependencies: {sorted(unknown_dependencies)}"
            )
        services.append(
            LaunchService(
                role="component",
                instance=server.name,
                component=target,
                image=image.image,
                command=("gym", "env", "start-server"),
                arguments=("--instance", server.name),
                environment={
                    "NEMO_GYM_CONFIG_FILE": CONFIG_MOUNT_PATH,
                    "NEMO_GYM_RUNTIME_INSTALL_POLICY": "require-existing",
                    "RAY_ADDRESS": "${RAY_ADDRESS}",
                },
                ports=(inner.port,),
                dependencies=dependencies,
            )
        )

    return LaunchPlan(
        environment_lock_identity=lock.identity,
        composition_bom_identity=bom.identity,
        services=tuple(services),
    )


def render_launch_plan_cli() -> None:
    """Render a validated launch plan as canonical JSON."""
    config = get_global_config_dict()
    bom_path = Path(config.get("bom", "composition.bom.json")).expanduser()
    lock_path = Path(config.get("lock", "environment.lock.json")).expanduser()
    output_path = Path(config.get("output", "launch-plan.json")).expanduser()
    plan = render_launch_plan(bom_path=bom_path, lock_path=lock_path, runtime_config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan.canonical_json() + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({plan.identity})")
