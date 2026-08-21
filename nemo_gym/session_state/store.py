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
import asyncio
import fcntl
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol, TypeVar, runtime_checkable

from pydantic import ValidationError

from nemo_gym.session_state.records import SessionSnapshot, ToolBoundaryRecord


# Rollout ids and server names become path components; keep the same character
# discipline as ``rollout_correlation.ROLLOUT_ID_PATTERN`` (not imported, to
# keep this module dependency-light for external consumers). No leading dot and
# no path separators, so a name is always a safe single component.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_name(value: str, what: str) -> str:
    if not (isinstance(value, str) and _NAME_PATTERN.match(value)):
        raise ValueError(f"{what} must match {_NAME_PATTERN.pattern}; got {value!r}")
    return value


_T = TypeVar("_T")

# A dedicated, bounded executor for store IO. On shared filesystems an fsync
# can take tens of milliseconds; funneling thousands of concurrent rollouts'
# writes through asyncio's small default to_thread pool would starve every
# other to_thread user in the process. A private pool isolates that pressure
# and naturally batches queue depth against slow storage.
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ng-session-state")


async def _run_io(fn: Callable[..., _T], *args: Any) -> _T:
    return await asyncio.get_running_loop().run_in_executor(_IO_EXECUTOR, partial(fn, *args))


@runtime_checkable
class SessionStateStore(Protocol):
    """Protocol seam for session-state persistence.

    The file-backed default below is the recommended implementation: session
    state is small, infrequent, opaque blobs on shared storage, co-located with
    the token capture dir. A framework may substitute its own backend, but
    unlike token staging there is no trainer-owned process that ever holds this
    state, so a data-plane transport rarely applies.
    """

    async def append_boundary(self, record: ToolBoundaryRecord) -> None: ...

    async def read_boundaries(self, rollout_id: str) -> list[ToolBoundaryRecord]: ...

    async def latest_boundary(self, rollout_id: str) -> Optional[ToolBoundaryRecord]: ...

    async def write_snapshot(self, snapshot: SessionSnapshot) -> None: ...

    async def read_snapshot(
        self, rollout_id: str, server_name: str, boundary_index: int
    ) -> Optional[SessionSnapshot]: ...

    async def clear_rollout(self, rollout_id: str) -> None: ...


class FileSessionStateStore:
    """Durable file-backed session-state store.

    Layout: one directory per rollout, so clearing a rollout can never touch a
    sibling whose id shares a prefix (ids may contain dots)::

        <root>/<rollout_id>/boundaries.jsonl              append-only ToolBoundaryRecords
        <root>/<rollout_id>/<server>.snapshot.<idx>.json  one SessionSnapshot per boundary
        <root>/<rollout_id>/.lock                         cross-process write lock

    Correctness properties:
    - Writes hold an exclusive per-rollout ``fcntl`` lock and fsync before
      returning, so an acknowledged record survives process death and is
      visible to any process on the same filesystem (multiple uvicorn workers,
      a restarted server, a recovery pass).
    - Snapshots are written to a temp file and published with ``os.replace``,
      so readers never observe a torn snapshot.
    - Snapshot files are per-boundary-index, so a crash between "snapshot
      written" and "boundary appended" leaves an orphan snapshot that restore
      never selects (restore reads the index named by the boundary record).
    - Boundary reads tolerate a torn trailing line (crash mid-append before
      fsync) by skipping it; replays may duplicate an index, and the last
      record per index wins.

    All state lives on disk; instances hold no memory between calls, so
    constructing one per request is correct.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        # No eager mkdir: constructing a store must cost zero filesystem
        # operations, because callers construct one per request. Write paths
        # create directories on demand (mkdir parents=True covers the root);
        # read paths tolerate a missing tree.
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    def _rollout_dir(self, rollout_id: str, *, create: bool) -> Path:
        path = self.root / _validate_name(rollout_id, "rollout_id")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _boundaries_path(self, rollout_id: str, *, create: bool = False) -> Path:
        return self._rollout_dir(rollout_id, create=create) / "boundaries.jsonl"

    def _snapshot_path(self, rollout_id: str, server_name: str, boundary_index: int, *, create: bool = False) -> Path:
        _validate_name(server_name, "server_name")
        return self._rollout_dir(rollout_id, create=create) / f"{server_name}.snapshot.{int(boundary_index)}.json"

    @contextmanager
    def _locked(self, rollout_id: str) -> Iterator[None]:
        lock_path = self._rollout_dir(rollout_id, create=True) / ".lock"
        with open(lock_path, "a") as lock_file:
            # Lustre mounts without ``-o flock`` reject advisory locks. Degrade
            # to unlocked writes rather than failing the rollout: in practice a
            # rollout has a single writer at a time (its driving agent), and a
            # boundary append is one small O_APPEND write, which the kernel
            # serializes.
            locked = True
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except OSError:
                locked = False
            try:
                yield
            finally:
                if locked:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # -- boundaries ----------------------------------------------------------

    def _append_boundary_sync(self, record: ToolBoundaryRecord) -> None:
        path = self._boundaries_path(record.rollout_id, create=True)
        line = record.model_dump_json() + "\n"
        with self._locked(record.rollout_id):
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    async def append_boundary(self, record: ToolBoundaryRecord) -> None:
        await _run_io(self._append_boundary_sync, record)

    def _read_boundaries_sync(self, rollout_id: str) -> list[ToolBoundaryRecord]:
        path = self._boundaries_path(rollout_id)
        if not path.exists():
            return []
        by_index: dict[int, ToolBoundaryRecord] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ToolBoundaryRecord.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValidationError):
                    # A torn trailing line is an aborted append: the writer never
                    # acknowledged it, so skipping it is correct, not lossy.
                    continue
                by_index[record.boundary_index] = record
        return [by_index[i] for i in sorted(by_index)]

    async def read_boundaries(self, rollout_id: str) -> list[ToolBoundaryRecord]:
        return await _run_io(self._read_boundaries_sync, rollout_id)

    async def latest_boundary(self, rollout_id: str) -> Optional[ToolBoundaryRecord]:
        records = await self.read_boundaries(rollout_id)
        return records[-1] if records else None

    # -- snapshots -----------------------------------------------------------

    def _write_snapshot_sync(self, snapshot: SessionSnapshot) -> None:
        path = self._snapshot_path(snapshot.rollout_id, snapshot.server_name, snapshot.boundary_index, create=True)
        tmp = path.with_name(path.name + ".tmp")
        with self._locked(snapshot.rollout_id):
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(snapshot.model_dump_json())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    async def write_snapshot(self, snapshot: SessionSnapshot) -> None:
        await _run_io(self._write_snapshot_sync, snapshot)

    def _read_snapshot_sync(self, rollout_id: str, server_name: str, boundary_index: int) -> Optional[SessionSnapshot]:
        path = self._snapshot_path(rollout_id, server_name, boundary_index)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return SessionSnapshot.model_validate(json.load(f))

    async def read_snapshot(self, rollout_id: str, server_name: str, boundary_index: int) -> Optional[SessionSnapshot]:
        return await _run_io(self._read_snapshot_sync, rollout_id, server_name, boundary_index)

    # -- lifecycle -----------------------------------------------------------

    def _clear_rollout_sync(self, rollout_id: str) -> None:
        path = self._rollout_dir(rollout_id, create=False)
        shutil.rmtree(path, ignore_errors=True)

    async def clear_rollout(self, rollout_id: str) -> None:
        """Rerun hygiene: remove a rollout's records before a fresh (non-resume) dispatch."""
        await _run_io(self._clear_rollout_sync, rollout_id)
