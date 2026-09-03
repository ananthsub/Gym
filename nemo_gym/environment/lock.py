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

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Self
from urllib.parse import unquote, urlparse

from omegaconf import DictConfig, ListConfig, OmegaConf
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym import PARENT_DIR
from nemo_gym.cli.setup_command import ComponentInstallPlan, component_install_plan
from nemo_gym.config_types import ConfigError, ServerInstanceConfig
from nemo_gym.environment.build_manifest import (
    ComponentBuildStep,
    component_source_closure,
    load_component_build_manifest,
)
from nemo_gym.environment.protocol import ENVIRONMENT_LOCK_SCHEMA_VERSION, RUNTIME_PROTOCOL_VERSION
from nemo_gym.global_config import (
    HEAD_SERVER_DEPS_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.secret_utils import MASKED_VALUE, looks_like_secret_key


COMPONENT_LOCK_SCHEMA_VERSION = "nemo-gym/component-lock/v1"
_IGNORED_CONTENT_PARTS = frozenset({".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"})
_INDEX_DIRECTIVE = re.compile(
    r"^\s*(?:-i|--index-url|--extra-index-url|--index|--default-index|-f|--find-links|--trusted-host)\b"
)
_DEPENDENCY_INCLUDE = re.compile(r"^\s*(?:-r|--requirement|-c|--constraint)(?:\s+|=)(\S+)")
_VCS_URL = re.compile(r"(?:git|hg|svn|bzr)\+[^;\s\"']+", re.IGNORECASE)
_PINNED_GIT_REF = re.compile(r"git\+[^;\s\"']+@[0-9a-f]{40}(?:[#&;]|$)", re.IGNORECASE)
_CREDENTIAL_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+(?::[^/\s@]*)?@", re.IGNORECASE)
_SECRET_QUERY = re.compile(r"[?&](?:token|api[_-]?key|password|secret)=", re.IGNORECASE)
_SECRET_INTERPOLATION = re.compile(r"\$\{[^}]*(?:token|api[_-]?key|password|secret)[^}]*}", re.IGNORECASE)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIREMENT_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}\b", re.IGNORECASE)
_DIGEST_PINNED_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _CanonicalLockModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: str = ""

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"identity"})

    def expected_identity(self) -> str:
        return _sha256_text(_canonical_json(self.identity_payload()))

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def _populate_or_validate_identity(self) -> Self:
        expected = self.expected_identity()
        if self.identity and self.identity != expected:
            raise ValueError(f"lock identity mismatch: expected {expected}, got {self.identity}")
        object.__setattr__(self, "identity", expected)
        return self


class ComponentLockRecord(_CanonicalLockModel):
    """A versioned dependency lock for one configured Gym server component."""

    schema_version: Literal["nemo-gym/component-lock/v1"] = COMPONENT_LOCK_SCHEMA_VERSION
    instances: tuple[str, ...]
    component_type: Literal["responses_api_models", "resources_servers", "responses_api_agents"]
    implementation: str
    source_kind: Literal["requirements", "pyproject"]
    source_path: str
    platform: str
    python_version: str
    uv_version: str
    parent_constraints: tuple[str, ...]
    config_sha256: str = Field(pattern=_HASH_PATTERN)
    content_sha256: str = Field(pattern=_HASH_PATTERN)
    source_dependencies: tuple["SourceDependencyLock", ...] = ()
    build_steps: tuple[ComponentBuildStep, ...] = ()
    requirements_sha256: str = Field(pattern=_HASH_PATTERN)
    requirements_lock: str


class SourceDependencyLock(BaseModel):
    """A source-only component needed by a selected runtime component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    content_sha256: str = Field(pattern=_HASH_PATTERN)
    build_steps: tuple[ComponentBuildStep, ...] = ()


class PackageIdentity(BaseModel):
    """The exact version of a package in the parent Gym environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str


class GymIdentity(PackageIdentity):
    """The installed Gym version and the source from which it was installed."""

    source_kind: Literal["registry", "editable", "vcs", "direct_url"]
    source_url: str | None = None
    source_revision: str | None = None
    source_sha256: str | None = Field(default=None, pattern=_HASH_PATTERN)


class EnvironmentLockRecord(_CanonicalLockModel):
    """A versioned lock for the resolved config and all configured components."""

    schema_version: Literal["nemo-gym/environment-lock/v1"] = ENVIRONMENT_LOCK_SCHEMA_VERSION
    runtime_protocol: Literal["nemo-gym/runtime/v1"] = RUNTIME_PROTOCOL_VERSION
    strict: bool = True
    base_image: str
    platform: str
    python_version: str
    uv_version: str
    gym_source_sha256: str = Field(pattern=_HASH_PATTERN)
    gym: GymIdentity
    parent_packages: tuple[PackageIdentity, ...]
    config_sha256: str = Field(pattern=_HASH_PATTERN)
    components: tuple[ComponentLockRecord, ...]


def _redacted_value(value: Any, key: str | None = None) -> Any:
    """Return a plain JSON value with secret-shaped fields replaced."""
    if key is not None and looks_like_secret_key(key.lower()):
        return MASKED_VALUE
    if isinstance(value, (DictConfig, Mapping)):
        return {
            str(child_key): _redacted_value(child_value, str(child_key)) for child_key, child_value in value.items()
        }
    if isinstance(value, (ListConfig, list, tuple)):
        return [_redacted_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and (
        _CREDENTIAL_URL.search(value) or _SECRET_QUERY.search(value) or _SECRET_INTERPOLATION.search(value)
    ):
        return MASKED_VALUE
    return value


def redacted_config_digest(value: Any) -> str:
    """Hash resolved configuration after replacing secret-shaped fields."""
    plain = (
        OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else deepcopy(value)
    )
    return _sha256_text(_canonical_json(_redacted_value(plain)))


def _is_deployment_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in {"host", "port"} or normalized.endswith(
        ("_host", "_port", "_url", "_base_url", "_address", "_endpoint")
    )


def _build_config_value(value: Any, key: str | None = None) -> Any:
    if key is not None and (looks_like_secret_key(key) or _is_deployment_key(key)):
        return MASKED_VALUE
    if isinstance(value, (DictConfig, Mapping)):
        return {
            str(child_key): _build_config_value(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (ListConfig, list, tuple)):
        return [_build_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_config_digest(value: Any) -> str:
    """Hash behavior-affecting configuration without secrets or deployment locations."""
    plain = (
        OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else deepcopy(value)
    )
    return _sha256_text(_canonical_json(_build_config_value(plain)))


def component_content_digest(component_dir: Path) -> str:
    """Hash component file names, contents, and symlink targets in stable order."""
    hasher = hashlib.sha256()
    for path in sorted(component_dir.rglob("*"), key=lambda item: item.relative_to(component_dir).as_posix()):
        relative = path.relative_to(component_dir)
        if any(part in _IGNORED_CONTENT_PARTS for part in relative.parts):
            continue
        if path.is_dir():
            continue
        hasher.update(relative.as_posix().encode("utf-8"))
        hasher.update(b"\0")
        if path.is_symlink():
            hasher.update(os.readlink(path).encode("utf-8"))
        else:
            hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def gym_build_source_digest(source_root: Path) -> str:
    """Hash Gym runtime source, dependency locks, and reproducible image recipes."""
    source_root = source_root.resolve()
    hasher = hashlib.sha256()
    inputs = [
        source_root / "nemo_gym",
        source_root / "pyproject.toml",
        source_root / "uv.lock",
        source_root / "docker" / "Dockerfile.packed",
        source_root / "docker" / "Dockerfile.server",
    ]
    for path in inputs:
        if not path.exists():
            continue
        relative = path.relative_to(source_root).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        if path.is_dir():
            hasher.update(component_content_digest(path).encode("ascii"))
        else:
            hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _gym_identity(*, strict: bool) -> GymIdentity:
    """Describe the installed Gym distribution without relying on a mutable package name alone."""
    try:
        distribution = importlib.metadata.distribution("nemo-gym")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConfigError("Cannot create a lock because parent package 'nemo-gym' is unavailable") from exc

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return GymIdentity(name="nemo-gym", version=distribution.version, source_kind="registry")

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise ConfigError("Installed nemo-gym has malformed direct_url.json metadata") from exc
    source_url = direct_url.get("url")
    if not isinstance(source_url, str) or not source_url:
        raise ConfigError("Installed nemo-gym direct_url.json does not identify its source URL")
    if _CREDENTIAL_URL.search(source_url) or _SECRET_QUERY.search(source_url):
        raise ConfigError("Installed nemo-gym source URL contains credentials and cannot be written to a lock")

    directory_info = direct_url.get("dir_info", {})
    if isinstance(directory_info, Mapping) and directory_info.get("editable") is True:
        parsed_url = urlparse(source_url)
        source_path = Path(unquote(parsed_url.path)) if parsed_url.scheme == "file" else PARENT_DIR
        if not (source_path / "nemo_gym").is_dir():
            source_path = PARENT_DIR
        return GymIdentity(
            name="nemo-gym",
            version=distribution.version,
            source_kind="editable",
            source_sha256=gym_build_source_digest(source_path),
        )

    vcs_info = direct_url.get("vcs_info", {})
    if isinstance(vcs_info, Mapping) and vcs_info:
        revision = vcs_info.get("commit_id")
        if strict and (
            not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision, re.IGNORECASE) is None
        ):
            raise ConfigError(
                "Installed nemo-gym uses a mutable VCS source; reinstall it at a full Git commit SHA before locking"
            )
        return GymIdentity(
            name="nemo-gym",
            version=distribution.version,
            source_kind="vcs",
            source_url=source_url,
            source_revision=revision if isinstance(revision, str) else None,
        )

    archive_info = direct_url.get("archive_info", {})
    archive_hash = archive_info.get("hash") if isinstance(archive_info, Mapping) else None
    source_sha256 = archive_hash.removeprefix("sha256=") if isinstance(archive_hash, str) else None
    if source_sha256 is not None and _HASH_PATTERN.fullmatch(source_sha256) is None:
        source_sha256 = None
    return GymIdentity(
        name="nemo-gym",
        version=distribution.version,
        source_kind="direct_url",
        source_url=source_url,
        source_sha256=source_sha256,
    )


def _dependency_texts(plan: ComponentInstallPlan) -> list[tuple[Path, str]]:
    dependency_file = plan.dependency_file
    if dependency_file is None:
        raise ConfigError(f"Component {plan.component_dir} has no dependency source")
    pending = [dependency_file]
    overrides = plan.component_dir / "overrides.txt"
    if overrides.is_file():
        pending.append(overrides)

    texts: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    while pending:
        path = pending.pop(0).resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"Dependency input {path} does not exist") from exc
        texts.append((path, text))
        for line in text.splitlines():
            match = _DEPENDENCY_INCLUDE.match(line)
            if match:
                pending.append(path.parent / match.group(1))
    return texts


def _portable_requirements_input(plan: ComponentInstallPlan) -> str:
    """Remove the parent checkout source and anchor nested inputs to the component."""
    lines: list[str] = []
    for line in plan.dependency_file.read_text(encoding="utf-8").splitlines():
        if "../.." in line:
            continue
        match = _DEPENDENCY_INCLUDE.match(line)
        if match:
            included_path = (plan.dependency_file.parent / match.group(1)).resolve()
            line = line.replace(match.group(1), str(included_path), 1)
        lines.append(line)
    return "\n".join(lines) + "\n"


def _pyproject_requirements_input(path: Path, *, extras: Sequence[str] = ()) -> str:
    """Render a project's runtime dependencies as a compile input, excluding Gym itself."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Cannot lock malformed dependency file {path}: {exc}") from exc

    project = data.get("project", {})
    dependencies = list(project.get("dependencies", ()))
    optional_dependencies = project.get("optional-dependencies", {})
    for extra in extras:
        try:
            dependencies.extend(optional_dependencies[extra])
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"{path} does not define requested nemo-gym extra {extra!r}") from exc

    lines: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, str):
            raise ConfigError(f"{path} contains a non-string project dependency")
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as exc:
            raise ConfigError(f"{path} contains an invalid project dependency {dependency!r}") from exc
        if requirement.name.lower().replace("_", "-") != "nemo-gym":
            lines.append(dependency)
    return "\n".join(lines) + "\n"


def _requested_gym_extras(plan: ComponentInstallPlan) -> tuple[str, ...]:
    extras: set[str] = set()
    for _, text in _dependency_texts(plan):
        for match in re.finditer(r"\bnemo[-_]gym(?:\[([^\]]+)\])?", text, re.IGNORECASE):
            if match.group(1):
                extras.update(extra.strip() for extra in match.group(1).split(",") if extra.strip())
    return tuple(sorted(extras))


def _validate_pyproject_sources(path: Path, text: str) -> None:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Cannot lock malformed dependency file {path}: {exc}") from exc

    uv_config = data.get("tool", {}).get("uv", {})
    if uv_config.get("index"):
        raise ConfigError(f"{path} contains uv index directives; strict locks use the default package index")

    sources = uv_config.get("sources", {})
    if not isinstance(sources, Mapping):
        return
    for package, source in sources.items():
        if not isinstance(source, Mapping):
            continue
        if "index" in source:
            raise ConfigError(f"{path} selects a package index for {package!r}; strict locks use the default index")
        if "git" not in source:
            continue
        revision = source.get("rev")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision, re.IGNORECASE) is None:
            raise ConfigError(f"{path} uses a mutable VCS ref for {package!r}; strict locks require a full commit SHA")
        if "branch" in source or "tag" in source:
            raise ConfigError(f"{path} uses a mutable VCS branch or tag for {package!r}")


def validate_dependency_inputs(plan: ComponentInstallPlan, *, strict: bool = True) -> None:
    """Reject dependency inputs that can change resolution or expose credentials."""
    for path, text in _dependency_texts(plan):
        for line_number, line in enumerate(text.splitlines(), start=1):
            if strict and _INDEX_DIRECTIVE.match(line):
                raise ConfigError(
                    f"{path}:{line_number} contains an index directive; strict locks only use the configured resolver"
                )
            if _CREDENTIAL_URL.search(line) or _SECRET_QUERY.search(line) or _SECRET_INTERPOLATION.search(line):
                raise ConfigError(f"{path}:{line_number} contains credentials or a secret-bearing URL")
            vcs_urls = _VCS_URL.findall(line)
            if strict and any(_PINNED_GIT_REF.fullmatch(url) is None for url in vcs_urls):
                raise ConfigError(
                    f"{path}:{line_number} uses a mutable VCS ref; strict locks require a full Git commit SHA"
                )
        if strict and path.name == "pyproject.toml":
            _validate_pyproject_sources(path, text)


def _parent_package_identities() -> tuple[PackageIdentity, ...]:
    try:
        ray_version = importlib.metadata.version("ray")
        openai_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConfigError(f"Cannot create a lock because parent package {exc.name!r} is unavailable") from exc
    return (
        PackageIdentity(name="ray", version=ray_version),
        PackageIdentity(name="openai", version=openai_version),
    )


def _exact_parent_constraints(
    global_config_dict: DictConfig,
    gym: GymIdentity,
    parent_packages: Sequence[PackageIdentity],
) -> tuple[str, ...]:
    package_versions = {package.name: package.version for package in parent_packages}

    other_constraints = tuple(
        dependency
        for dependency in global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]
        if not dependency.lower().startswith(("ray[", "ray=", "openai[", "openai="))
    )
    return (
        f"nemo-gym=={gym.version}",
        f"ray[default]=={package_versions['ray']}",
        f"openai=={package_versions['openai']}",
        *other_constraints,
    )


def _clean_resolver_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_FIND_LINKS",
        "UV_INSECURE_HOST",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
    ):
        env.pop(key, None)
    return env


def _uv_executable() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise ConfigError("gym env lock requires the repository-supported uv executable on PATH")
    return executable


def _uv_version(uv_executable: str) -> str:
    result = subprocess.run(
        [uv_executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_resolver_environment(),
    )
    match = re.search(r"\buv\s+([^\s]+)", result.stdout)
    if match is None:
        raise ConfigError(f"Could not parse uv version from: {result.stdout.strip()!r}")
    return match.group(1)


def _uv_python_platform(platform: str) -> str:
    """Translate common OCI platform names to uv's target triples."""
    aliases = {
        "linux/amd64": "x86_64-unknown-linux-gnu",
        "linux/arm64": "aarch64-unknown-linux-gnu",
        "darwin/amd64": "x86_64-apple-darwin",
        "darwin/arm64": "aarch64-apple-darwin",
        "windows/amd64": "x86_64-pc-windows-msvc",
        "windows/arm64": "aarch64-pc-windows-msvc",
    }
    return aliases.get(platform, platform)


def compile_component_requirements(
    plan: ComponentInstallPlan,
    *,
    platform: str,
    parent_constraints: Sequence[str],
    uv_executable: str,
    strict: bool,
) -> str:
    """Compile one component with local uv and return its hashed requirements."""
    validate_dependency_inputs(plan, strict=strict)
    if plan.dependency_file is None or plan.dependency_source is None:
        raise ConfigError(f"Cannot lock skipped component plan for {plan.component_dir}")

    with TemporaryDirectory(prefix="nemo-gym-lock-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        parent_input = temporary_path / "parent-requirements.in"
        parent_input.write_text("\n".join(parent_constraints) + "\n", encoding="utf-8")
        output_path = temporary_path / "requirements.lock"

        source_files: list[str] = []
        if plan.dependency_source == "requirements":
            source_input = temporary_path / "component-requirements.in"
            source_input.write_text(_portable_requirements_input(plan), encoding="utf-8")
            source_files.append(str(source_input))
        else:
            source_input = temporary_path / "component-requirements.in"
            source_input.write_text(_pyproject_requirements_input(plan.dependency_file), encoding="utf-8")
            source_files.append(str(source_input))

        if plan.is_editable_gym_install:
            gym_input = temporary_path / "nemo-gym-requirements.in"
            gym_pyproject = plan.component_dir.resolve().parents[1] / "pyproject.toml"
            gym_input.write_text(
                _pyproject_requirements_input(gym_pyproject, extras=_requested_gym_extras(plan)),
                encoding="utf-8",
            )
            source_files.append(str(gym_input))
        source_files.append(str(parent_input))

        command = [
            uv_executable,
            "pip",
            "compile",
            "--generate-hashes",
            "--no-strip-extras",
            "--no-header",
            "--no-annotate",
            "--python-version",
            plan.python_version,
            "--python-platform",
            _uv_python_platform(platform),
            "--output-file",
            str(output_path),
        ]
        overrides_path = plan.component_dir / "overrides.txt"
        if overrides_path.is_file():
            command.extend(["--overrides", str(overrides_path)])
        command.extend(source_files)

        try:
            subprocess.run(
                command,
                cwd=plan.component_dir,
                check=True,
                capture_output=True,
                text=True,
                env=_clean_resolver_environment(),
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise ConfigError(f"uv failed to lock {plan.component_dir.name}: {detail}") from exc
        requirements_lock = output_path.read_text(encoding="utf-8")
        _validate_hash_complete_lock(requirements_lock, component_dir=plan.component_dir)
        return requirements_lock


def _validate_hash_complete_lock(requirements_lock: str, *, component_dir: Path) -> None:
    """Require a SHA-256 hash on every compiled requirement block."""
    blocks: list[str] = []
    current: list[str] = []
    for line in requirements_lock.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace() and current:
            current.append(line)
            continue
        if current:
            blocks.append("\n".join(current))
        current = [line]
    if current:
        blocks.append("\n".join(current))

    unhashed = [block.splitlines()[0] for block in blocks if _REQUIREMENT_HASH.search(block) is None]
    if unhashed:
        preview = ", ".join(repr(requirement) for requirement in unhashed[:3])
        raise ConfigError(
            f"uv produced an unhashed dependency lock for {component_dir}: {preview}. "
            "Use immutable package artifacts that uv can hash."
        )


def _component_coordinates(server: ServerInstanceConfig) -> tuple[str, str]:
    component_type = server.SERVER_TYPE
    implementation = next(iter(getattr(server, component_type)))
    return component_type, implementation


def _environment_config_digest(global_config_dict: DictConfig, servers: Sequence[ServerInstanceConfig]) -> str:
    digest_input = {
        "python_version": global_config_dict[PYTHON_VERSION_KEY_NAME],
        "head_server_dependencies": list(global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]),
        "components": {
            server.name: OmegaConf.to_container(server.server_type_config_dict, resolve=True) for server in servers
        },
    }
    return build_config_digest(digest_input)


def build_environment_lock(
    global_config_dict: DictConfig,
    *,
    base_image: str,
    platform: str,
    source_root: Path = PARENT_DIR,
    strict: bool = True,
) -> EnvironmentLockRecord:
    """Resolve and compile dependency locks for every configured server component."""
    if strict and _DIGEST_PINNED_IMAGE.fullmatch(base_image) is None:
        raise ConfigError("Strict environment locks require --base-image to use an OCI sha256 digest")
    source_root = source_root.resolve()
    if not (source_root / "nemo_gym").is_dir():
        raise ConfigError(f"Gym source root does not contain nemo_gym/: {source_root}")
    parser = GlobalConfigDictParser()
    servers = parser.filter_for_server_instance_configs(global_config_dict)
    parser.raise_on_no_server_instances(global_config_dict)
    gym = _gym_identity(strict=strict)
    parent_packages = _parent_package_identities()
    uv_executable = _uv_executable()
    uv_version = _uv_version(uv_executable)

    from nemo_gym.cli.env import _resolve_server_dir

    resolved_components: dict[str, Path] = {}

    def resolve_component(path: str) -> Path:
        if path not in resolved_components:
            resolved_components[path] = _resolve_server_dir(Path(path))
        return resolved_components[path]

    grouped_servers: dict[tuple[str, str], list[ServerInstanceConfig]] = {}
    for server in servers:
        coordinates = _component_coordinates(server)
        grouped_servers.setdefault(coordinates, []).append(server)

    components: list[ComponentLockRecord] = []
    for (component_type, implementation), component_servers in sorted(grouped_servers.items()):
        relative_source = Path(component_type, implementation)
        component_dir = resolve_component(relative_source.as_posix())
        plan = component_install_plan(component_dir, global_config_dict, respect_existing_venv=False)
        constraints = _exact_parent_constraints(global_config_dict, gym, parent_packages)
        if plan.is_editable_gym_install:
            constraints = tuple(constraint for constraint in constraints if not constraint.startswith("nemo-gym=="))
        requirements_lock = compile_component_requirements(
            plan,
            platform=platform,
            parent_constraints=constraints,
            uv_executable=uv_executable,
            strict=strict,
        )
        source_kind = plan.dependency_source
        if source_kind is None:
            raise ConfigError(f"Component {component_dir} has no dependency source")
        sorted_servers = sorted(component_servers, key=lambda item: item.name)
        source_dependencies = tuple(
            SourceDependencyLock(
                path=dependency,
                content_sha256=component_content_digest(resolve_component(dependency)),
                build_steps=load_component_build_manifest(resolve_component(dependency)).build_steps,
            )
            for dependency in component_source_closure(
                relative_source.as_posix(),
                resolve_component=resolve_component,
            )
        )
        build_steps = load_component_build_manifest(component_dir).build_steps
        components.append(
            ComponentLockRecord(
                instances=tuple(server.name for server in sorted_servers),
                component_type=component_type,
                implementation=implementation,
                source_kind=source_kind,
                source_path=relative_source.as_posix(),
                platform=platform,
                python_version=plan.python_version,
                uv_version=uv_version,
                parent_constraints=constraints,
                config_sha256=build_config_digest(
                    {
                        server.name: OmegaConf.to_container(server.server_type_config_dict, resolve=True)
                        for server in sorted_servers
                    }
                ),
                content_sha256=component_content_digest(component_dir),
                source_dependencies=source_dependencies,
                build_steps=build_steps,
                requirements_sha256=_sha256_text(requirements_lock),
                requirements_lock=requirements_lock,
            )
        )

    return EnvironmentLockRecord(
        strict=strict,
        base_image=base_image,
        platform=platform,
        python_version=str(global_config_dict[PYTHON_VERSION_KEY_NAME]),
        uv_version=uv_version,
        gym_source_sha256=gym_build_source_digest(source_root),
        gym=gym,
        parent_packages=parent_packages,
        config_sha256=_environment_config_digest(global_config_dict, servers),
        components=tuple(components),
    )


def lock_environment() -> None:
    """Generate a canonical environment lock without starting any services."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(offline=True)
    )
    platform = global_config_dict.get("platform")
    if not isinstance(platform, str) or not platform:
        raise ConfigError("gym env lock requires --platform PLATFORM")
    output_dir = Path(global_config_dict.get("output", "."))
    output = output_dir / "environment.lock.json"
    strict = global_config_dict.get("strict", True)
    if not isinstance(strict, bool):
        raise ConfigError("strict must be a boolean")

    base_image = global_config_dict.get("base_image")
    if not isinstance(base_image, str) or not base_image:
        raise ConfigError("gym env lock requires --base-image IMAGE@sha256:DIGEST")
    source_root = Path(global_config_dict.get("source_root", PARENT_DIR))

    lock = build_environment_lock(
        global_config_dict,
        base_image=base_image,
        platform=platform,
        source_root=source_root,
        strict=strict,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(lock.canonical_json() + "\n", encoding="utf-8")
    print(f"Wrote {output} ({lock.identity})")
