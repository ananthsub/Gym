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
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# Writers and readers of these records live in different processes and can be
# deployed at different versions (an agent server, a resources server, and a
# training framework's recovery pass). Records outlive deploys, so readers must
# refuse a record from a newer schema instead of silently decoding a subset.
SESSION_STATE_SCHEMA_VERSION = 1


def _refuse_a_newer_record(value: Any) -> Any:
    if isinstance(value, int) and value > SESSION_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"record schema_version {value} is newer than this reader "
            f"(supports <= {SESSION_STATE_SCHEMA_VERSION}); upgrade the reader"
        )
    return value


class SessionSnapshot(BaseModel):
    """One resources server's serialization of one session's state at one boundary.

    ``state`` is opaque to the framework: the owning server defines it in
    ``export_session_state`` and consumes it in ``restore_session_state``. It must
    be JSON-serializable; servers whose real state lives in an external backend
    (e.g. a connectable sandbox) put the reconnect descriptor here, not the state.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = SESSION_STATE_SCHEMA_VERSION
    rollout_id: str
    server_name: str
    # The agent step this snapshot is valid at: the state after step
    # ``boundary_index``'s tool calls completed. 0 means post-seed, pre-step-1.
    boundary_index: int
    state: dict[str, Any]
    created_at: float

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        return _refuse_a_newer_record(value)


class ToolBoundaryRecord(BaseModel):
    """The agent's conversation delta for one completed step.

    ``output_items`` holds the Responses items appended during step
    ``boundary_index``: the model output items followed by their
    ``function_call_output`` results. Tool results become durable *here* — they
    do not appear in any model-call capture until the next call's prompt — so
    this record is what closes the tool-boundary durability gap.

    A record is appended only after every snapshot it references was written
    (commit order: snapshots, then boundary). Replays after a crash may append
    the same ``boundary_index`` twice; readers keep the last record per index.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = SESSION_STATE_SCHEMA_VERSION
    rollout_id: str
    # 1-based agent step, matching the agent loop's step counter.
    boundary_index: int
    output_items: list[dict[str, Any]]
    # Accumulated usage as of this boundary (a full snapshot, not a delta), so
    # resume restores usage from the latest record alone.
    usage: Optional[dict[str, Any]] = None
    # Whether the resources server wrote a separate SessionSnapshot file for
    # this boundary. False for stateless servers, servers without export
    # support, boundaries skipped by the snapshot cadence, and inline states.
    env_exported: bool = False
    # Small environment states ride inline instead of in a snapshot file: one
    # append and one fsync per boundary instead of two files and three fsyncs —
    # the difference between tolerable and hostile on metadata-bound shared
    # filesystems (Lustre). Servers inline states up to an export-size
    # threshold; larger states keep the separate snapshot file.
    env_state: Optional[dict[str, Any]] = None
    # The model response that produced this step: the resync anchor between
    # committed boundaries and captured token records. On resume, calls
    # at-or-before this anchor belong to the kept prefix; the dead attempt's
    # captured calls after it are orphans a finalizer must retire, never
    # splice — the resumed attempt regenerates them under its own attempt key.
    response_id: Optional[str] = None
    created_at: float

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        return _refuse_a_newer_record(value)

    @property
    def resumable(self) -> bool:
        """Whether the environment can be restored at this boundary."""
        return self.env_exported or self.env_state is not None


def select_resume_records(records: list["ToolBoundaryRecord"]) -> list["ToolBoundaryRecord"]:
    """Return the prefix of ``records`` ending at the latest resumable boundary.

    With a snapshot cadence > 1, boundaries between snapshots carry only the
    conversation delta; resuming there would pair a newer conversation with an
    older environment. Truncate to the last boundary that has environment state
    (inline or in a snapshot file). When no record carries state, the server is
    stateless and every boundary is resumable: return all records.
    """
    if any(record.resumable for record in records):
        while records and not records[-1].resumable:
            records = records[:-1]
    return records
