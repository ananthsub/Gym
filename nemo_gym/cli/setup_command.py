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
import importlib.metadata
import os
from dataclasses import dataclass
from os import environ
from pathlib import Path
from subprocess import Popen
from sys import stderr, stdout
from typing import IO, Any, Literal, Mapping

from omegaconf import DictConfig

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import (
    HEAD_SERVER_DEPS_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    NEMO_GYM_RUNTIME_INSTALL_POLICY_ENV_VAR_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    RUNTIME_INSTALL_POLICY_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_PIP_SET_PYTHON_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
    get_global_config_dict,
)


def _get_nemo_gym_install_flags() -> str:
    """
    Build uv pip install flags for nemo-gym in sub-venvs.

    Supports:
    - Pre-release versions via NEMO_GYM_ALLOW_PRERELEASE=true
    - Custom PyPI indexes via UV_INDEX_URL, UV_EXTRA_INDEX_URL, UV_INDEX_STRATEGY
    - Auto-detection of parent venv version for consistency

    Returns:
        String of flags to add to 'uv pip install nemo-gym'
        Example: "--pre --index-url https://test.pypi.org/simple/ ==0.2.1rc0"
    """
    flags = ""

    # 1. Pre-release flag
    allow_prerelease = os.getenv("NEMO_GYM_ALLOW_PRERELEASE", "").lower() == "true"
    if allow_prerelease:
        flags += "--pre "
        # When pre-releases are enabled, also use unsafe-best-match strategy if not already set
        if not os.getenv("UV_INDEX_STRATEGY"):
            flags += "--index-strategy unsafe-best-match "
        # Pin fastapi<1.0 to avoid broken test.pypi package
        flags += "'fastapi<1.0' "

    # 2. Index URLs (respects uv's standard env vars)
    index_url = os.getenv("UV_INDEX_URL")
    if index_url:
        flags += f"--index-url {index_url} "

    extra_index_url = os.getenv("UV_EXTRA_INDEX_URL")
    if extra_index_url:
        flags += f"--extra-index-url {extra_index_url} "

    # Explicit index strategy (overrides auto-set above)
    index_strategy = os.getenv("UV_INDEX_STRATEGY")
    if index_strategy:
        flags += f"--index-strategy {index_strategy} "

    return flags


def _get_nemo_gym_version_spec(is_editable_install: bool) -> str:
    """
    Detect nemo-gym version from parent venv and return version specifier.

    Args:
        is_editable_install: Whether nemo-gym is installed in editable mode in parent venv

    Returns:
        Version specifier string (e.g., "==0.2.1rc0") or empty string
    """
    # Don't pin version for editable installs (development mode)
    if is_editable_install:
        return ""

    try:
        parent_version = importlib.metadata.version("nemo-gym")
        # Pin to exact version for consistency between parent and sub-venvs
        return f"=={parent_version}"
    except importlib.metadata.PackageNotFoundError:
        # nemo-gym not installed in parent venv (shouldn't happen, but be safe)
        return ""


def get_venv_path(dir_path: Path, global_config_dict: DictConfig) -> Path:
    """Return the server venv path for the configured venv root."""
    root_venv_path = Path(global_config_dict[UV_VENV_DIR_KEY_NAME])
    if root_venv_path.resolve() != PARENT_DIR.resolve():
        return Path(root_venv_path, *dir_path.parts[-2:], ".venv").absolute()
    return (dir_path / ".venv").absolute()


@dataclass(frozen=True)
class ComponentInstallPlan:
    """Dependency and source decisions for one component venv.

    Constructing a plan only inspects local metadata. It does not create a venv,
    install packages, or mutate process state.
    """

    component_dir: Path
    venv_path: Path
    python_version: str
    dependency_source: Literal["pyproject", "requirements"] | None
    head_server_dependencies: tuple[str, ...]
    is_editable_gym_install: bool
    skip_venv_setup: bool
    use_python_flag: bool
    verbose: bool
    has_overrides: bool
    nemo_gym_install_flags: str
    nemo_gym_version_spec: str

    @property
    def dependency_file(self) -> Path | None:
        if self.dependency_source == "pyproject":
            return self.component_dir / "pyproject.toml"
        if self.dependency_source == "requirements":
            return self.component_dir / "requirements.txt"
        return None


def component_install_plan(
    dir_path: Path,
    global_config_dict: DictConfig,
    *,
    respect_existing_venv: bool = True,
) -> ComponentInstallPlan:
    """Return the install decisions shared by venv setup and lock generation."""
    component_dir = dir_path.absolute()
    venv_path = get_venv_path(component_dir, global_config_dict)
    skip_requested = bool(global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME])
    venv_is_ready = (venv_path / "bin/python").exists() and (venv_path / "bin/activate").exists()
    skip_venv_setup = respect_existing_venv and skip_requested and venv_is_ready

    dependency_source: Literal["pyproject", "requirements"] | None = None
    has_overrides = False
    is_editable_install = (component_dir.resolve() / "../../pyproject.toml").exists()
    install_flags = ""
    version_spec = ""

    if not skip_venv_setup:
        has_pyproject_toml = (component_dir / "pyproject.toml").exists()
        has_requirements_txt = (component_dir / "requirements.txt").exists()
        if has_pyproject_toml and has_requirements_txt:
            raise RuntimeError(
                f"Found both pyproject.toml and requirements.txt for uv venv setup in server dir: "
                f"{component_dir}. Please only use one or the other!"
            )
        if has_pyproject_toml:
            dependency_source = "pyproject"
        elif has_requirements_txt:
            dependency_source = "requirements"
            has_overrides = (component_dir / "overrides.txt").exists()
        else:
            raise RuntimeError(
                f"Missing pyproject.toml or requirements.txt for uv venv setup in server dir: {component_dir}"
            )

        if not is_editable_install:
            install_flags = _get_nemo_gym_install_flags()
            version_spec = _get_nemo_gym_version_spec(is_editable_install)

    return ComponentInstallPlan(
        component_dir=component_dir,
        venv_path=venv_path,
        python_version=str(global_config_dict[PYTHON_VERSION_KEY_NAME]),
        dependency_source=dependency_source,
        head_server_dependencies=tuple(global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]),
        is_editable_gym_install=is_editable_install,
        skip_venv_setup=skip_venv_setup,
        use_python_flag=bool(global_config_dict.get(UV_PIP_SET_PYTHON_KEY_NAME, False)),
        verbose=bool(global_config_dict.get(PIP_INSTALL_VERBOSE_KEY_NAME)),
        has_overrides=has_overrides,
        nemo_gym_install_flags=install_flags,
        nemo_gym_version_spec=version_spec,
    )


def _install_command(plan: ComponentInstallPlan) -> str:
    """Render the existing uv install command from a component plan."""
    verbose_flag = "-v " if plan.verbose else ""
    uv_pip_python_flag = f"--python {plan.venv_path / 'bin/python'} " if plan.use_python_flag else ""
    head_server_deps = " ".join(plan.head_server_dependencies)

    if plan.dependency_source == "pyproject":
        if plan.is_editable_gym_install:
            return f"""uv pip install {verbose_flag}{uv_pip_python_flag}'-e .' {head_server_deps}"""
        return (
            f"""uv pip install {verbose_flag}{uv_pip_python_flag}{plan.nemo_gym_install_flags}"""
            f"""nemo-gym{plan.nemo_gym_version_spec} && """
            f"""uv pip install {verbose_flag}{uv_pip_python_flag}--no-sources '-e .' {head_server_deps}"""
        )

    if plan.dependency_source == "requirements":
        override_flag = "--override overrides.txt " if plan.has_overrides else ""
        if plan.is_editable_gym_install:
            return (
                f"""uv pip install {verbose_flag}{uv_pip_python_flag}{override_flag}"""
                f"""-r requirements.txt {head_server_deps}"""
            )
        return (
            f"""(echo 'nemo-gym{plan.nemo_gym_version_spec}' && grep -v -F '../..' requirements.txt) | """
            f"""uv pip install {verbose_flag}{uv_pip_python_flag}{plan.nemo_gym_install_flags}"""
            f"""{override_flag}-r /dev/stdin {head_server_deps}"""
        )

    raise RuntimeError(f"Component install plan for {plan.component_dir} has no dependency source")


def setup_env_command(dir_path: Path, global_config_dict: DictConfig, prefix: str) -> str:
    configured_policy = global_config_dict.get(RUNTIME_INSTALL_POLICY_KEY_NAME, "allow-install")
    image_policy = os.getenv(NEMO_GYM_RUNTIME_INSTALL_POLICY_ENV_VAR_NAME, "allow-install")
    for source, policy in (("config", configured_policy), ("environment", image_policy)):
        if policy not in ("allow-install", "require-existing"):
            raise ConfigError(
                f"{RUNTIME_INSTALL_POLICY_KEY_NAME} from {source} must be 'allow-install' or "
                f"'require-existing', got {policy!r}"
            )
    runtime_install_policy = (
        "require-existing" if "require-existing" in (configured_policy, image_policy) else configured_policy
    )
    if runtime_install_policy == "require-existing":
        component_dir = dir_path.absolute()
        venv_path = get_venv_path(component_dir, global_config_dict)
        venv_activate_fpath = venv_path / "bin/activate"
        missing = [path for path in (venv_path / "bin/python", venv_activate_fpath) if not path.is_file()]
        if missing:
            raise ConfigError(
                f"Packed-image runtime requires the prebuilt component environment at {venv_path}; "
                f"missing {', '.join(str(path) for path in missing)}. Rebuild the image from the current lock."
            )
        return f"cd {component_dir} && source {venv_activate_fpath}"

    plan = component_install_plan(dir_path, global_config_dict)
    venv_activate_fpath = plan.venv_path / "bin/activate"
    if plan.skip_venv_setup:
        env_setup_cmd = f"source {venv_activate_fpath}"
    else:
        uv_venv_cmd = f"uv venv --seed --allow-existing --python {plan.python_version} {plan.venv_path}"
        install_cmd = _install_command(plan)
        prefix_cmd = f" > >(sed 's/^/({prefix}) /') 2> >(sed 's/^/({prefix}) /' >&2)"
        env_setup_cmd = f"{uv_venv_cmd}{prefix_cmd} && source {venv_activate_fpath} && {install_cmd}{prefix_cmd}"

    return f"cd {plan.component_dir} && {env_setup_cmd}"


def run_command(
    command: str,
    working_dir_path: Path,
    server_name: str = "",
    project_root: Path | None = None,
    *,
    global_config_dict: DictConfig | None = None,
    extra_env: Mapping[str, str] | None = None,
    stdout_target: IO[Any] | None = None,
    stderr_target: IO[Any] | None = None,
) -> Popen:
    if global_config_dict is None:
        global_config_dict = get_global_config_dict()

    work_dir = f"{working_dir_path.absolute()}"
    custom_env = environ.copy()
    # The server dir on PYTHONPATH lets `import app` work. When a caller passes `project_root` (the
    # dir containing resources_servers/, responses_api_agents/, ...), it's added so generated
    # `resources_servers.<name>.app`-style imports resolve from outside a repo checkout — opt-in, so
    # this generic helper doesn't bake a layout assumption in for its other callers.
    py_path_entries = [work_dir]
    if project_root is not None:
        py_path_entries.append(f"{project_root.absolute()}")
    existing_py_path = custom_env.get("PYTHONPATH")
    if existing_py_path:
        py_path_entries.append(existing_py_path)
    custom_env["PYTHONPATH"] = ":".join(py_path_entries)

    custom_env["UV_CACHE_DIR"] = global_config_dict[UV_CACHE_DIR_KEY_NAME]
    if extra_env is not None:
        custom_env.update(extra_env)

    log_dir = global_config_dict.get(NEMO_GYM_LOG_DIR_KEY_NAME)
    if log_dir:
        safe_name = (server_name or working_dir_path.name).replace("/", "_")
        log_path = Path(log_dir) / f"{safe_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = f"set -o pipefail; ({command}) 2>&1 | tee -a {log_path}"

    redirect_stdout = stdout if stdout_target is None else stdout_target
    redirect_stderr = stderr if stderr_target is None else stderr_target
    return Popen(
        command,
        executable="/bin/bash",
        shell=True,
        env=custom_env,
        stdout=redirect_stdout,
        stderr=redirect_stderr,
    )
