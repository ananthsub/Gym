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
"""Checkpoint control plane shared by every Gym server.

Partial rollout checkpointing pauses, drains, commits, and restores Gym
servers in lockstep with the NeMo-RL training checkpoint. The pieces in this
package are the server-side mechanisms that make those control calls safe:

- ``control``: the ``/ng-control/v1`` capability declaration, checkpoint-id
  fencing, phase machine, and deadline plumbing every control route uses.
- ``admission``: the admission limiter that drains a server's data plane to
  a quiescent point, with lease propagation for nested calls and
  ``409 checkpoint_parked`` for callers that can safely re-issue.
- ``model_admission``: the ``/ng-control/v1/model-admission`` routes a
  policy model server exposes to the checkpoint coordinator.
"""

from nemo_gym.checkpoint.admission import (
    ADMISSION_LEASE_HEADER,
    GATED_MODEL_ROUTE_SUFFIXES,
    PLANE_HEADER,
    AdmissionLimiter,
    AdmissionMiddleware,
    AdmissionParkedError,
    AdmissionTicket,
    StaleAttemptError,
    admission_lease_context,
    current_admission_lease,
)
from nemo_gym.checkpoint.control import (
    CONTROL_SCHEMA_VERSION,
    CONTROL_URL_PREFIX,
    AdmissionState,
    CheckpointConflictError,
    CheckpointPhase,
    ControlCapabilities,
    ControlError,
    ControlFence,
    Deadline,
    InvalidPhaseError,
    MultiProcessCapability,
    StaleCheckpointError,
    install_control_plane,
    multi_process_capability_from_num_workers,
)
from nemo_gym.checkpoint.model_admission import (
    MODEL_ADMISSION_URL_PREFIX,
    NotPolicyInstanceError,
    install_model_admission,
)


__all__ = [
    "ADMISSION_LEASE_HEADER",
    "CONTROL_SCHEMA_VERSION",
    "CONTROL_URL_PREFIX",
    "GATED_MODEL_ROUTE_SUFFIXES",
    "MODEL_ADMISSION_URL_PREFIX",
    "PLANE_HEADER",
    "AdmissionLimiter",
    "AdmissionMiddleware",
    "AdmissionParkedError",
    "AdmissionState",
    "AdmissionTicket",
    "CheckpointConflictError",
    "CheckpointPhase",
    "ControlCapabilities",
    "ControlError",
    "ControlFence",
    "Deadline",
    "InvalidPhaseError",
    "MultiProcessCapability",
    "NotPolicyInstanceError",
    "StaleAttemptError",
    "StaleCheckpointError",
    "admission_lease_context",
    "current_admission_lease",
    "install_control_plane",
    "install_model_admission",
    "multi_process_capability_from_num_workers",
]
