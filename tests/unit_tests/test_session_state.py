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
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.session_state import (
    SESSION_STATE_SCHEMA_VERSION,
    FileSessionStateStore,
    SessionSnapshot,
    SessionStateStore,
    ToolBoundaryRecord,
)
from nemo_gym.session_state.records import select_resume_records


def _snapshot(rollout_id: str = "7-0", boundary_index: int = 1, **overrides) -> SessionSnapshot:
    fields = dict(
        rollout_id=rollout_id,
        server_name="counter_server",
        boundary_index=boundary_index,
        state={"counter": 7},
        created_at=1.0,
    )
    fields.update(overrides)
    return SessionSnapshot(**fields)


def _boundary(rollout_id: str = "7-0", boundary_index: int = 1, **overrides) -> ToolBoundaryRecord:
    fields = dict(
        rollout_id=rollout_id,
        boundary_index=boundary_index,
        output_items=[{"type": "function_call_output", "call_id": f"c{boundary_index}", "output": "ok"}],
        env_exported=True,
        created_at=1.0,
    )
    fields.update(overrides)
    return ToolBoundaryRecord(**fields)


class TestRecords:
    def test_reader_refuses_newer_schema(self) -> None:
        newer = SESSION_STATE_SCHEMA_VERSION + 1
        with pytest.raises(ValidationError, match="newer than this reader"):
            SessionSnapshot.model_validate(_snapshot().model_dump() | {"schema_version": newer})
        with pytest.raises(ValidationError, match="newer than this reader"):
            ToolBoundaryRecord.model_validate(_boundary().model_dump() | {"schema_version": newer})

    def test_current_schema_roundtrips(self) -> None:
        snapshot = _snapshot()
        assert SessionSnapshot.model_validate(snapshot.model_dump()) == snapshot
        boundary = _boundary()
        assert ToolBoundaryRecord.model_validate(boundary.model_dump()) == boundary


class TestSelectResumeRecords:
    def test_stateless_records_are_all_resumable(self) -> None:
        records = [_boundary(boundary_index=i, env_exported=False) for i in (1, 2, 3)]
        assert select_resume_records(records) == records

    def test_truncates_to_latest_boundary_with_state(self) -> None:
        # Snapshot cadence: conversation-only boundaries after the last one
        # with environment state would pair a newer conversation with an older
        # environment, so they are dropped.
        records = [
            _boundary(boundary_index=1, env_exported=False, env_state={"counter": 7}),
            _boundary(boundary_index=2, env_exported=False),
            _boundary(boundary_index=3, env_exported=True),
            _boundary(boundary_index=4, env_exported=False),
        ]
        assert [record.boundary_index for record in select_resume_records(records)] == [1, 2, 3]

    def test_inline_state_counts_as_resumable(self) -> None:
        records = [_boundary(boundary_index=1, env_exported=False, env_state={"counter": 7})]
        assert select_resume_records(records) == records
        assert records[0].resumable

    def test_empty_records(self) -> None:
        assert select_resume_records([]) == []


class TestFileSessionStateStore:
    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileSessionStateStore(tmp_path), SessionStateStore)

    def test_construction_costs_no_filesystem_operations(self, tmp_path: Path) -> None:
        # A store is constructed per request; on metadata-bound shared
        # filesystems (Lustre) an eager mkdir per request would be a storm.
        root = tmp_path / "never-created"
        FileSessionStateStore(root)
        assert not root.exists()

    async def test_reads_tolerate_missing_root(self, tmp_path: Path) -> None:
        store = FileSessionStateStore(tmp_path / "never-created")
        assert await store.read_boundaries("7-0") == []
        assert await store.read_snapshot("7-0", "counter_server", 1) is None
        await store.clear_rollout("7-0")

    async def test_boundary_append_and_read(self, tmp_path: Path) -> None:
        store = FileSessionStateStore(tmp_path)
        await store.append_boundary(_boundary(boundary_index=1))
        await store.append_boundary(_boundary(boundary_index=2))
        records = await store.read_boundaries("7-0")
        assert [r.boundary_index for r in records] == [1, 2]
        latest = await store.latest_boundary("7-0")
        assert latest is not None and latest.boundary_index == 2

    async def test_duplicate_index_last_record_wins(self, tmp_path: Path) -> None:
        # A crash-then-replay appends the same boundary index twice.
        store = FileSessionStateStore(tmp_path)
        await store.append_boundary(_boundary(boundary_index=1, output_items=[{"type": "x", "attempt": 1}]))
        await store.append_boundary(_boundary(boundary_index=1, output_items=[{"type": "x", "attempt": 2}]))
        records = await store.read_boundaries("7-0")
        assert len(records) == 1
        assert records[0].output_items[0]["attempt"] == 2

    async def test_torn_trailing_line_is_skipped(self, tmp_path: Path) -> None:
        store = FileSessionStateStore(tmp_path)
        await store.append_boundary(_boundary(boundary_index=1))
        with open(tmp_path / "7-0" / "boundaries.jsonl", "a", encoding="utf-8") as f:
            f.write('{"schema_version": 1, "rollout_id": "7-0", "boundary_ind')
        records = await store.read_boundaries("7-0")
        assert [r.boundary_index for r in records] == [1]

    async def test_snapshot_roundtrip_and_exact_index(self, tmp_path: Path) -> None:
        store = FileSessionStateStore(tmp_path)
        await store.write_snapshot(_snapshot(boundary_index=1, state={"counter": 5}))
        await store.write_snapshot(_snapshot(boundary_index=2, state={"counter": 9}))
        first = await store.read_snapshot("7-0", "counter_server", 1)
        assert first is not None and first.state == {"counter": 5}
        # An orphan snapshot at a later index never shadows the requested one.
        assert (await store.read_snapshot("7-0", "counter_server", 3)) is None

    async def test_missing_rollout_reads_empty(self, tmp_path: Path) -> None:
        store = FileSessionStateStore(tmp_path)
        assert await store.read_boundaries("no-such") == []
        assert await store.latest_boundary("no-such") is None
        assert await store.read_snapshot("no-such", "counter_server", 1) is None

    async def test_cross_instance_visibility(self, tmp_path: Path) -> None:
        # A different store instance (another process after a restart) sees
        # everything an acknowledged write produced.
        await FileSessionStateStore(tmp_path).append_boundary(_boundary())
        await FileSessionStateStore(tmp_path).write_snapshot(_snapshot())
        fresh = FileSessionStateStore(tmp_path)
        assert len(await fresh.read_boundaries("7-0")) == 1
        assert (await fresh.read_snapshot("7-0", "counter_server", 1)) is not None

    async def test_clear_rollout_spares_dot_prefixed_siblings(self, tmp_path: Path) -> None:
        # Ids may contain dots; clearing "1-2" must not touch "1-2.5".
        store = FileSessionStateStore(tmp_path)
        await store.append_boundary(_boundary(rollout_id="1-2"))
        await store.append_boundary(_boundary(rollout_id="1-2.5"))
        await store.clear_rollout("1-2")
        assert await store.read_boundaries("1-2") == []
        assert len(await store.read_boundaries("1-2.5")) == 1

    async def test_invalid_names_are_rejected(self, tmp_path: Path) -> None:
        store = FileSessionStateStore(tmp_path)
        with pytest.raises(ValueError, match="rollout_id"):
            await store.read_boundaries("../escape")
        with pytest.raises(ValueError, match="rollout_id"):
            await store.read_boundaries(".hidden")
        with pytest.raises(ValueError, match="server_name"):
            await store.read_snapshot("7-0", "bad/name", 1)
