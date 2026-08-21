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
"""Durable per-rollout session state for partial rollout checkpointing.

A paused or killed rollout can resume at its last *tool boundary*: the point
where model call N and all of its tool executions have completed. Two record
kinds make that boundary durable:

- ``SessionSnapshot``: a resources server's own serialization of one session's
  environment state, exported at a boundary.
- ``ToolBoundaryRecord``: the agent's conversation delta for one step (model
  output items plus tool outputs), committed *after* the matching snapshots.

Commit order is snapshot-then-boundary, so the boundary record is the commit
point: a snapshot without a matching boundary record is an aborted step and is
never restored. Everything is keyed by the rollout id (see
``nemo_gym.rollout_correlation``), never by the transport session cookie.

This package is deliberately dependency-light (stdlib + pydantic only), like
``nemo_gym.token_id_capture``: a training framework can consume the store
without importing the server stack.
"""

from nemo_gym.session_state.records import (
    SESSION_STATE_SCHEMA_VERSION,
    SessionSnapshot,
    ToolBoundaryRecord,
)
from nemo_gym.session_state.store import FileSessionStateStore, SessionStateStore


__all__ = [
    "SESSION_STATE_SCHEMA_VERSION",
    "SessionSnapshot",
    "ToolBoundaryRecord",
    "SessionStateStore",
    "FileSessionStateStore",
]
