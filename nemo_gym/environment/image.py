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

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from nemo_gym import PARENT_DIR, __version__, _resolve_under_cwd_or_install
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.lock import (
    EnvironmentLockRecord,
    _validate_hash_complete_lock,
    component_content_digest,
    gym_build_source_digest,
)
from nemo_gym.environment.protocol import COMPOSITION_BOM_SCHEMA_VERSION, RUNTIME_PROTOCOL_VERSION
from nemo_gym.global_config import get_global_config_dict


PACKED_DOCKERFILE = PARENT_DIR / "docker" / "Dockerfile.packed"
SERVER_DOCKERFILE = PARENT_DIR / "docker" / "Dockerfile.server"
_OCI_PLATFORM_ALIASES = {
    "x86_64-unknown-linux-gnu": "linux/amd64",
    "aarch64-unknown-linux-gnu": "linux/arm64",
}


class CompositionImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    instances: tuple[str, ...]
    image: str
    digest: str
    component_content_sha256: str | None = None


class CompositionBOM(BaseModel):
    """Digest-pinned image composition produced from one environment lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["nemo-gym/composition-bom/v1"] = COMPOSITION_BOM_SCHEMA_VERSION
    runtime_protocol: Literal["nemo-gym/runtime/v1"] = RUNTIME_PROTOCOL_VERSION
    identity: str = ""
    environment_lock_identity: str
    images: tuple[CompositionImage, ...]

    def canonical_payload(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json", exclude={"identity"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @model_validator(mode="after")
    def validate_identity(self) -> "CompositionBOM":
        expected = hashlib.sha256(self.canonical_payload()).hexdigest()
        if self.identity and self.identity != expected:
            raise ValueError(f"Composition BOM identity mismatch: expected {expected}, got {self.identity}")
        object.__setattr__(self, "identity", expected)
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _oci_platform(platform: str) -> str:
    normalized = _OCI_PLATFORM_ALIASES.get(platform, platform)
    if "/" not in normalized:
        raise ConfigError(f"Unsupported target platform {platform!r}; use an OCI platform such as linux/amd64")
    return normalized


def _clean_package_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    ):
        env.pop(key, None)
    env["PIP_CONFIG_FILE"] = os.devnull
    env["UV_NO_CONFIG"] = "1"
    return env


def _component_source_path(source_root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ConfigError(f"Component source path must be relative and cannot traverse parents: {relative_path}")
    local_path = (source_root / relative_path).resolve()
    if local_path.is_dir():
        return local_path
    return _resolve_under_cwd_or_install(relative_path, validator=Path.is_dir).resolve()


def load_verified_environment_lock(
    lock_path: Path,
    *,
    source_root: Path,
    component_selector: str | None = None,
) -> EnvironmentLockRecord:
    """Load a lock and verify its identity, requirements, and current component contents."""
    try:
        lock = EnvironmentLockRecord.model_validate_json(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Environment lock does not exist: {lock_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"Environment lock is invalid: {exc}") from exc

    source_root = source_root.resolve()
    if lock.gym.version != __version__:
        raise ConfigError(f"Environment lock requires nemo-gym {lock.gym.version}, but this source is {__version__}")
    if lock.gym.source_kind == "editable" and lock.gym.source_sha256 is not None:
        actual_gym_digest = gym_build_source_digest(source_root)
        if actual_gym_digest != lock.gym.source_sha256:
            raise ConfigError(
                f"Gym source changed after locking: expected {lock.gym.source_sha256}, got {actual_gym_digest}. "
                "Run 'gym env lock' again."
            )
    selected_components = [
        component
        for component in lock.components
        if component_selector is None or f"{component.component_type}/{component.implementation}" == component_selector
    ]
    if component_selector is not None and not selected_components:
        raise ConfigError(f"Component {component_selector!r} is not present in the environment lock")
    for component in selected_components:
        requirements_digest = hashlib.sha256(component.requirements_lock.encode("utf-8")).hexdigest()
        if requirements_digest != component.requirements_sha256:
            raise ConfigError(
                f"Requirements lock digest changed for {component.component_type}/{component.implementation}"
            )
        component_dir = _component_source_path(source_root, Path(component.source_path))
        if not component_dir.is_dir():
            raise ConfigError(f"Locked component source is missing: {component.source_path}")
        actual_digest = component_content_digest(component_dir)
        if actual_digest != component.content_sha256:
            raise ConfigError(
                f"Component content changed for {component.component_type}/{component.implementation}: "
                f"expected {component.content_sha256}, got {actual_digest}. Run 'gym env lock' again."
            )
        _validate_hash_complete_lock(component.requirements_lock, component_dir=component_dir)
    return lock


def _copy_build_context(
    source_root: Path,
    context_root: Path,
    lock: EnvironmentLockRecord,
    *,
    component_selector: str | None = None,
) -> None:
    for filename in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        shutil.copy2(source_root / filename, context_root / filename)
    (context_root / "cache").mkdir()
    shutil.copytree(source_root / "nemo_gym", context_root / "nemo_gym", ignore=shutil.ignore_patterns("__pycache__"))
    for component in lock.components:
        if component_selector is not None:
            selector = f"{component.component_type}/{component.implementation}"
            if selector != component_selector:
                continue
        destination = context_root / component.source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        component_source = _component_source_path(source_root, Path(component.source_path))
        shutil.copytree(
            component_source,
            destination,
            symlinks=True,
            ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".ruff_cache"),
        )
    docker_dir = context_root / "docker"
    docker_dir.mkdir()
    shutil.copy2(PACKED_DOCKERFILE, docker_dir / PACKED_DOCKERFILE.name)
    shutil.copy2(SERVER_DOCKERFILE, docker_dir / SERVER_DOCKERFILE.name)
    shutil.copy2(PACKED_DOCKERFILE, context_root / "Dockerfile.packed")
    (context_root / "environment.lock.json").write_text(lock.canonical_json() + "\n", encoding="utf-8")


def packed_build_command(
    *,
    context_root: Path,
    lock: EnvironmentLockRecord,
    base_image: str,
    tag: str,
    platform: str | None = None,
    load: bool = False,
    push: bool = False,
) -> list[str]:
    """Render the Docker buildx invocation for a verified packed build context."""
    if load and push:
        raise ConfigError("--load and --push are mutually exclusive")
    command = [
        "docker",
        "buildx",
        "build",
        "--file",
        str(context_root / "Dockerfile.packed"),
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"LOCK_IDENTITY={lock.identity}",
        "--build-arg",
        f"LOCK_SCHEMA={lock.schema_version}",
        "--build-arg",
        f"GYM_VERSION={lock.gym.version}",
        "--build-arg",
        f"PYTHON_VERSION={lock.python_version}",
        "--build-arg",
        f"RUNTIME_PROTOCOL={lock.runtime_protocol}",
        "--build-arg",
        "TARGET_KIND=packed",
        "--tag",
        tag,
    ]
    if platform is not None:
        command.extend(["--platform", platform])
    if load:
        command.append("--load")
    if push:
        command.append("--push")
    command.append(str(context_root))
    return command


def build_packed_image(
    lock_path: Path,
    *,
    source_root: Path,
    base_image: str,
    tag: str,
    platform: str | None = None,
    load: bool = False,
    push: bool = False,
) -> None:
    """Verify a lock and build one packed image without copying runtime configuration."""
    if "://" in base_image and "@" in base_image.split("://", 1)[1].split("/", 1)[0]:
        raise ConfigError("Base image reference must not contain credentials")
    lock = load_verified_environment_lock(lock_path, source_root=source_root)
    locked_platform = _oci_platform(lock.platform)
    if platform is not None and _oci_platform(platform) != locked_platform:
        raise ConfigError(f"Requested platform {platform!r} does not match lock platform {lock.platform!r}")

    with TemporaryDirectory(prefix="nemo-gym-packed-image-") as temporary_dir:
        context_root = Path(temporary_dir)
        _copy_build_context(source_root, context_root, lock)
        command = packed_build_command(
            context_root=context_root,
            lock=lock,
            base_image=base_image,
            tag=tag,
            platform=locked_platform,
            load=load,
            push=push,
        )
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise ConfigError("Docker is required for 'gym env build-image'") from exc
        except subprocess.CalledProcessError as exc:
            raise ConfigError(f"Docker buildx failed with exit code {exc.returncode}") from exc


def _image_digest_from_metadata(metadata_path: Path) -> str:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Docker did not produce valid build metadata: {exc}") from exc
    digest = metadata.get("containerimage.digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        raise ConfigError(f"Docker build metadata has no valid container image digest: {metadata_path}")
    return digest


def _split_image_build_command(
    *,
    context_root: Path,
    metadata_path: Path,
    lock: EnvironmentLockRecord,
    base_image: str,
    tag: str,
    target: str,
    target_kind: str,
    component_selector: str,
    platform: str | None,
) -> list[str]:
    command = [
        "docker",
        "buildx",
        "build",
        "--file",
        str(context_root / "Dockerfile.server"),
        "--target",
        target,
        "--metadata-file",
        str(metadata_path),
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"LOCK_IDENTITY={lock.identity}",
        "--build-arg",
        f"LOCK_SCHEMA={lock.schema_version}",
        "--build-arg",
        f"GYM_VERSION={lock.gym.version}",
        "--build-arg",
        f"PYTHON_VERSION={lock.python_version}",
        "--build-arg",
        f"RUNTIME_PROTOCOL={lock.runtime_protocol}",
        "--build-arg",
        f"TARGET_KIND={target_kind}",
        "--build-arg",
        f"COMPONENT_SELECTOR={component_selector}",
        "--tag",
        tag,
        "--push",
    ]
    if platform is not None:
        command.extend(["--platform", platform])
    command.append(str(context_root))
    return command


def build_split_images(
    lock_path: Path,
    *,
    source_root: Path,
    base_image: str,
    repository: str,
    output_path: Path,
    platform: str | None = None,
) -> CompositionBOM:
    """Build and push one head image and one image per locked component."""
    if "://" in base_image and "@" in base_image.split("://", 1)[1].split("/", 1)[0]:
        raise ConfigError("Base image reference must not contain credentials")
    lock = load_verified_environment_lock(lock_path, source_root=source_root)
    locked_platform = _oci_platform(lock.platform)
    if platform is not None and _oci_platform(platform) != locked_platform:
        raise ConfigError(f"Requested platform {platform!r} does not match lock platform {lock.platform!r}")
    repository = repository.rstrip("/")
    if not repository or "://" in repository or "@" in repository:
        raise ConfigError("Image repository must be a registry/repository prefix without a URL scheme")

    targets: list[tuple[str, str, tuple[str, ...], str | None]] = [
        ("head", "head", ("head_server",), None),
        *[
            (
                f"{component.component_type}-{component.implementation}".replace("_", "-"),
                f"{component.component_type}/{component.implementation}",
                component.instances,
                component.content_sha256,
            )
            for component in lock.components
        ],
    ]
    images = []
    for image_name, selector, instances, content_digest in targets:
        is_head = selector == "head"
        component_selector = "" if is_head else selector
        with TemporaryDirectory(prefix=f"nemo-gym-{image_name}-") as temporary_dir:
            context_root = Path(temporary_dir)
            _copy_build_context(
                source_root,
                context_root,
                lock,
                component_selector="__head__" if is_head else component_selector,
            )
            shutil.copy2(SERVER_DOCKERFILE, context_root / "Dockerfile.server")
            metadata_path = context_root / "build-metadata.json"
            tag = f"{repository}/{image_name}:{lock.identity[:16]}"
            command = _split_image_build_command(
                context_root=context_root,
                metadata_path=metadata_path,
                lock=lock,
                base_image=base_image,
                tag=tag,
                target="head" if is_head else "component",
                target_kind="head" if is_head else "server",
                component_selector=component_selector,
                platform=locked_platform,
            )
            try:
                subprocess.run(command, check=True)
            except FileNotFoundError as exc:
                raise ConfigError("Docker is required for 'gym env build-images'") from exc
            except subprocess.CalledProcessError as exc:
                raise ConfigError(f"Docker buildx failed for {selector} with exit code {exc.returncode}") from exc
            digest = _image_digest_from_metadata(metadata_path)
            images.append(
                CompositionImage(
                    target=selector,
                    instances=instances,
                    image=f"{repository}/{image_name}@{digest}",
                    digest=digest,
                    component_content_sha256=content_digest,
                )
            )

    bom = CompositionBOM(environment_lock_identity=lock.identity, images=tuple(images))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bom.canonical_json() + "\n", encoding="utf-8")
    return bom


def build_image_cli() -> None:
    """Build the packed image requested through the Gym CLI."""
    config = get_global_config_dict()
    lock_path = Path(config.get("lock", "environment.lock.json")).expanduser()
    source_root = Path(config.get("source_root", PARENT_DIR)).expanduser()
    base_image = config.get("base_image")
    tag = config.get("tag")
    if not isinstance(base_image, str) or not base_image:
        raise ConfigError("gym env build-image requires --base-image IMAGE")
    if not isinstance(tag, str) or not tag:
        raise ConfigError("gym env build-image requires --tag TAG")
    build_packed_image(
        lock_path,
        source_root=source_root,
        base_image=base_image,
        tag=tag,
        platform=config.get("platform"),
        load=bool(config.get("load", False)),
        push=bool(config.get("push", False)),
    )


def build_split_images_cli() -> None:
    """Build and publish digest-pinned standalone server images."""
    config = get_global_config_dict()
    lock_path = Path(config.get("lock", "environment.lock.json")).expanduser()
    source_root = Path(config.get("source_root", PARENT_DIR)).expanduser()
    base_image = config.get("base_image")
    repository = config.get("repository")
    if not isinstance(base_image, str) or not base_image:
        raise ConfigError("gym env build-images requires --base-image IMAGE")
    if not isinstance(repository, str) or not repository:
        raise ConfigError("gym env build-images requires --repository PREFIX")
    build_split_images(
        lock_path,
        source_root=source_root,
        base_image=base_image,
        repository=repository,
        output_path=Path(config.get("output", "composition.bom.json")).expanduser(),
        platform=config.get("platform"),
    )


def install_locked_components(
    lock_path: Path,
    *,
    source_root: Path,
    venv_root: Path,
    component_selector: str | None = None,
    gym_wheel_dir: Path | None = None,
) -> None:
    """Install every hash-complete component lock into its packed runtime venv."""
    lock = load_verified_environment_lock(
        lock_path,
        source_root=source_root,
        component_selector=component_selector,
    )
    gym_wheel = None
    if lock.gym.source_kind == "editable":
        wheels = sorted(gym_wheel_dir.glob("*.whl")) if gym_wheel_dir is not None else []
        if len(wheels) != 1:
            raise ConfigError("Editable Gym image builds require exactly one locally built Gym wheel")
        gym_wheel = wheels[0]
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise ConfigError("Locked image installation requires uv in the build image")
    clean_env = _clean_package_environment()
    for component in lock.components:
        selector = f"{component.component_type}/{component.implementation}"
        if component_selector is not None and selector != component_selector:
            continue
        venv_path = venv_root / component.component_type / component.implementation / ".venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        requirements_path = venv_path / "requirements.lock"
        requirements_path.write_text(component.requirements_lock, encoding="utf-8")
        python = venv_path / "bin" / "python"
        subprocess.run(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                str(python),
                "--require-hashes",
                "--no-deps",
                "--requirement",
                str(requirements_path),
            ],
            check=True,
            env=clean_env,
        )
        if lock.gym.source_kind == "editable":
            subprocess.run(
                [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(gym_wheel)],
                check=True,
                env=clean_env,
            )
        requirements_path.unlink()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--component")
    parser.add_argument("--gym-wheel-dir", type=Path)
    args = parser.parse_args()
    install_locked_components(
        args.lock,
        source_root=args.source_root,
        venv_root=args.venv_root,
        component_selector=args.component,
        gym_wheel_dir=args.gym_wheel_dir,
    )


if __name__ == "__main__":
    _main()
