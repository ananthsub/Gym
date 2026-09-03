# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.protocol import (
    RUNTIME_PROTOCOL_VERSION,
    RuntimeCapabilities,
    require_runtime_protocol,
    show_runtime_capabilities,
)


def test_runtime_capabilities_publish_versioned_offline_contract() -> None:
    capabilities = RuntimeCapabilities()

    assert capabilities.runtime_protocol == "nemo-gym/runtime/v1"
    assert capabilities.environment_lock_schema == "nemo-gym/environment-lock/v1"
    assert capabilities.prepared_artifact_schema == "nemo-gym/prepared-artifact/v1"
    assert capabilities.composition_bom_schema == "nemo-gym/composition-bom/v1"
    assert capabilities.launch_plan_schema == "nemo-gym/launch-plan/v1"
    assert capabilities.require_existing_runtime_policy is True
    assert capabilities.digest_pinned_base_images is True
    assert capabilities.prepared_artifact_lock_enforcement is True
    assert capabilities.launch_plan_rendering is True
    assert capabilities.offline_runtime_installation is False


def test_unknown_runtime_protocol_is_rejected() -> None:
    require_runtime_protocol(RUNTIME_PROTOCOL_VERSION)

    with pytest.raises(ConfigError, match="Unsupported runtime protocol"):
        require_runtime_protocol("nemo-gym/runtime/v2")


def test_capabilities_command_is_machine_readable(capsys) -> None:
    show_runtime_capabilities()

    assert json.loads(capsys.readouterr().out)["runtime_protocol"] == RUNTIME_PROTOCOL_VERSION
