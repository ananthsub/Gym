# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.env as env_cli
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RUNTIME_INSTALL_POLICY_ENV_VAR_NAME,
)


def test_standalone_server_spec_selects_one_locked_component(monkeypatch, tmp_path: Path) -> None:
    component_dir = tmp_path / "resources_servers/example"
    component_dir.mkdir(parents=True)
    monkeypatch.setattr(env_cli, "_resolve_server_dir", lambda path: component_dir)
    config = OmegaConf.create(
        {
            "example_server": {
                "resources_servers": {
                    "example": {
                        "entrypoint": "app.py",
                    }
                }
            }
        }
    )

    server_dir, entrypoint = env_cli._standalone_server_spec(config, "example_server")

    assert server_dir == component_dir
    assert entrypoint == Path("app.py")


def test_standalone_server_execs_prebuilt_python_with_config_in_environment(monkeypatch, tmp_path: Path) -> None:
    component_dir = tmp_path / "resources_servers/example"
    venv = component_dir / ".venv"
    python = venv / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    config = OmegaConf.create({"server_instance": "example_server"})
    monkeypatch.setattr(env_cli, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(env_cli, "_standalone_server_spec", lambda *_: (component_dir, Path("app.py")))
    monkeypatch.setattr(env_cli, "get_venv_path", lambda *_: venv)
    monkeypatch.setattr(env_cli.os, "chdir", lambda path: None)
    captured = {}

    def fake_execve(executable, argv, child_env):
        captured.update(executable=executable, argv=argv, child_env=child_env)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(env_cli.os, "execve", fake_execve)

    with pytest.raises(RuntimeError, match="exec intercepted"):
        env_cli.start_standalone_server()

    assert captured["executable"] == str(python)
    assert captured["argv"] == [str(python), "app.py"]
    assert captured["child_env"][NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME] == "example_server"
    assert captured["child_env"][NEMO_GYM_RUNTIME_INSTALL_POLICY_ENV_VAR_NAME] == "require-existing"
    assert NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME in captured["child_env"]
    assert NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME not in os.environ
