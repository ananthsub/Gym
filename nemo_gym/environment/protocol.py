# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned contracts shared by reproducible Gym build and runtime artifacts."""

import json

from pydantic import BaseModel, ConfigDict

from nemo_gym.config_types import ConfigError


RUNTIME_PROTOCOL_VERSION = "nemo-gym/runtime/v1"
ENVIRONMENT_LOCK_SCHEMA_VERSION = "nemo-gym/environment-lock/v1"
PREPARED_ARTIFACT_SCHEMA_VERSION = "nemo-gym/prepared-artifact/v1"
COMPOSITION_BOM_SCHEMA_VERSION = "nemo-gym/composition-bom/v1"


class RuntimeCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_protocol: str = RUNTIME_PROTOCOL_VERSION
    environment_lock_schema: str = ENVIRONMENT_LOCK_SCHEMA_VERSION
    prepared_artifact_schema: str = PREPARED_ARTIFACT_SCHEMA_VERSION
    composition_bom_schema: str = COMPOSITION_BOM_SCHEMA_VERSION
    packed_images: bool = True
    split_server_images: bool = True
    require_existing_runtime_policy: bool = True
    file_config_injection: bool = True
    offline_runtime_installation: bool = True


def require_runtime_protocol(value: str) -> None:
    """Reject artifacts built for a different runtime protocol version."""
    if value != RUNTIME_PROTOCOL_VERSION:
        raise ConfigError(
            f"Unsupported runtime protocol {value!r}; this Gym build supports {RUNTIME_PROTOCOL_VERSION!r}"
        )


def show_runtime_capabilities() -> None:
    """Print machine-readable build and runtime compatibility capabilities."""
    print(json.dumps(RuntimeCapabilities().model_dump(mode="json"), sort_keys=True))
