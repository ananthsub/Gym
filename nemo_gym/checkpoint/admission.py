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
"""Admission control for checkpoint drain.

A checkpoint must reach a moment where no request is changing state anywhere
in Gym. The admission limiter produces that moment for one server: closing it
stops new top-level operations while operations that were already accepted
run to completion, and the server reports ``paused`` only when nothing is in
flight.

Two refinements make the drain correct rather than merely quiet:

- **Leases.** An accepted operation may make nested calls (an agent loop
  calls the model; a tool calls a judge). Refusing those nested calls would
  deadlock the drain: the accepted operation can never finish. Every admitted
  root operation therefore mints a lease token; the token travels on
  ``x-ng-admission-lease`` to nested calls, and a nested call carrying the
  lease of a still-running root operation is admitted even while the server
  is draining. The lease dies with its root operation.
- **Parking.** A refused caller receives ``409 checkpoint_parked``, a
  deliberate signal (not an error) that says: this operation was not lost,
  re-issue it after the checkpoint. Callers that cannot park are the
  coordinator's problem to drain or sacrifice before the deadline.

A request counts as in flight from admission until its response completes.
On capture routes the model server writes durable custody before the
terminal response event, so response completion also marks custody
completion; a future custody hook can extend the window explicitly.
"""

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional
from uuid import uuid4

from starlette.responses import JSONResponse

from nemo_gym.checkpoint.control import AdmissionState, ControlError
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER, split_transport_rollout_id


ADMISSION_LEASE_HEADER = "x-ng-admission-lease"
PLANE_HEADER = "x-ng-plane"

# Suffixes of the generation routes a policy model server gates. Matching by
# suffix covers the plain routes and their ``/ng-rollout/<id>/...`` twins.
GATED_MODEL_ROUTE_SUFFIXES = ("/v1/responses", "/v1/chat/completions", "/v1/messages")

_ADMISSION_LEASE: ContextVar[Optional[str]] = ContextVar("nemo_gym_admission_lease", default=None)


def _resolve_identity(rollout_id: str, attempt_index: Optional[int]) -> tuple[str, int]:
    """``(logical_id, attempt)`` for a transport rollout id and optional explicit attempt.

    The ``-a{n}`` suffix is stripped only when it agrees with the explicit
    attempt (or no explicit attempt was given): a logical id that itself ends
    in ``-a{n}`` arrives with an explicit attempt that does not match its
    suffix, and stripping it would fence the wrong identity.
    """
    logical, derived = split_transport_rollout_id(rollout_id)
    if attempt_index is None:
        return logical, derived
    if derived != attempt_index:
        return rollout_id, attempt_index
    return logical, attempt_index


def current_admission_lease() -> Optional[str]:
    return _ADMISSION_LEASE.get()


@contextmanager
def admission_lease_context(lease: Optional[str]) -> Iterator[None]:
    token = _ADMISSION_LEASE.set(lease)
    try:
        yield
    finally:
        _ADMISSION_LEASE.reset(token)


class AdmissionParkedError(ControlError):
    """The server is draining or paused and this caller can safely park.

    Not a failure: the operation was refused before any state changed, so the
    caller re-issues it after the checkpoint completes.
    """

    code = "checkpoint_parked"


class StaleAttemptError(ControlError):
    """The rollout attempt was force-closed at a checkpoint deadline.

    Its call roots are tombstoned; a late call from the abandoned attempt
    must not write new state under an identity the restore already replaced.
    """

    code = "stale_attempt"


class AdmissionTicket:
    """One admitted in-flight operation."""

    __slots__ = ("ticket_id", "rollout_id", "attempt_index", "lease", "nested", "plane", "started_ts")

    def __init__(
        self,
        *,
        rollout_id: Optional[str],
        attempt_index: Optional[int],
        lease: str,
        nested: bool,
        plane: Optional[str],
    ) -> None:
        self.ticket_id = uuid4().hex
        self.rollout_id = rollout_id
        self.attempt_index = attempt_index
        self.lease = lease
        self.nested = nested
        self.plane = plane
        self.started_ts = time.time()


class AdmissionLimiter:
    """Atomic admission state machine for one server process.

    State changes are atomic with the admission test: both happen inside one
    event-loop step, so there is no window where a request is admitted
    against a state that a concurrent control call already changed.
    """

    def __init__(self) -> None:
        self.state = AdmissionState.ACCEPTING
        self._inflight: dict[str, AdmissionTicket] = {}
        self._root_leases: set[str] = set()
        self._tombstones: set[tuple[str, int]] = set()
        self._drained = asyncio.Event()
        self._drained.set()
        self._listeners: list[Callable[[], None]] = []

    # -- change listeners ----------------------------------------------------
    #
    # A multi-worker deployment reports each worker's in-flight count to a
    # service-level coordinator; the listener fires on every count change so
    # the report is event-driven instead of polled.

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self) -> None:
        for listener in list(self._listeners):
            listener()

    # -- admission -----------------------------------------------------------

    def admit(
        self,
        *,
        rollout_id: Optional[str] = None,
        attempt_index: Optional[int] = None,
        lease: Optional[str] = None,
        plane: Optional[str] = None,
    ) -> AdmissionTicket:
        if rollout_id is not None:
            logical, attempt = _resolve_identity(rollout_id, attempt_index)
            if (logical, attempt) in self._tombstones:
                raise StaleAttemptError(
                    f"rollout {logical!r} attempt {attempt} was force-closed at a checkpoint "
                    f"deadline; the restored run dispatched a replacement attempt"
                )

        nested = lease is not None and lease in self._root_leases
        if self.state != AdmissionState.ACCEPTING and not nested:
            raise AdmissionParkedError(
                f"admission is {self.state.value} for a checkpoint; park and re-issue this "
                f"operation after the checkpoint completes"
            )

        if nested:
            ticket = AdmissionTicket(
                rollout_id=rollout_id, attempt_index=attempt_index, lease=lease, nested=True, plane=plane
            )
        else:
            ticket = AdmissionTicket(
                rollout_id=rollout_id, attempt_index=attempt_index, lease="", nested=False, plane=plane
            )
            ticket.lease = ticket.ticket_id
            self._root_leases.add(ticket.lease)
        self._inflight[ticket.ticket_id] = ticket
        self._drained.clear()
        self._notify_listeners()
        return ticket

    def release(self, ticket: AdmissionTicket) -> None:
        # Idempotent: a force-closed ticket was already removed by abort.
        removed = self._inflight.pop(ticket.ticket_id, None)
        if removed is not None and not removed.nested:
            self._root_leases.discard(removed.lease)
        self._after_inflight_change()

    def _after_inflight_change(self) -> None:
        if not self._inflight:
            self._drained.set()
            if self.state == AdmissionState.DRAINING:
                self.state = AdmissionState.PAUSED
        self._notify_listeners()

    # -- control -------------------------------------------------------------

    def close(self) -> None:
        """Stop admitting new root operations. Atomic with the admission test."""
        if self.state == AdmissionState.ACCEPTING:
            self.state = AdmissionState.DRAINING if self._inflight else AdmissionState.PAUSED

    def resume(self) -> None:
        self.state = AdmissionState.ACCEPTING

    def abort_inflight(self, rollout_id: str, attempt_index: int) -> list[str]:
        """Force-close a rollout that missed the prepare deadline.

        Tombstones the ``(rollout_id, attempt)`` so any late call from the
        abandoned attempt is rejected, and stops counting its in-flight
        operations toward the drain. The underlying HTTP requests still run
        to completion; their writes are attributed to a tombstoned attempt
        and excluded from the checkpoint by the ledger.
        """
        logical, attempt = _resolve_identity(rollout_id, attempt_index)
        self._tombstones.add((logical, attempt))
        aborted = [
            ticket.ticket_id
            for ticket in self._inflight.values()
            if ticket.rollout_id is not None
            and _resolve_identity(ticket.rollout_id, ticket.attempt_index) == (logical, attempt)
        ]
        for ticket_id in aborted:
            ticket = self._inflight.pop(ticket_id)
            if not ticket.nested:
                self._root_leases.discard(ticket.lease)
        self._after_inflight_change()
        return aborted

    def install_tombstone(self, logical_rollout_id: str, attempt_index: int) -> None:
        """Install a fence for an already-logical identity (checkpoint restore).

        Unlike ``abort_inflight`` this never strips an ``-a{n}`` suffix: the
        manifest records logical ids, and re-splitting one that legitimately
        ends in ``-a{n}`` would fence the wrong identity.
        """
        self._tombstones.add((logical_rollout_id, attempt_index))

    def tombstones(self) -> list[tuple[str, int]]:
        return sorted(self._tombstones)

    # -- observation ---------------------------------------------------------

    async def wait_for_drained(self, timeout_s: float) -> bool:
        """Wait until nothing is in flight, up to ``timeout_s``. True if drained."""
        if timeout_s <= 0:
            return not self._inflight
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    def counts(self) -> dict[str, Any]:
        now = time.time()
        return {
            "state": self.state.value,
            "inflight_total": len(self._inflight),
            "waiters_total": 0,
            "inflight": [
                {
                    "rollout_id": ticket.rollout_id,
                    "attempt_index": ticket.attempt_index,
                    "nested": ticket.nested,
                    "plane": ticket.plane,
                    "age_seconds": round(now - ticket.started_ts, 3),
                }
                for ticket in self._inflight.values()
            ],
        }


class AdmissionMiddleware:
    """Gate a server's data-plane routes behind an ``AdmissionLimiter``.

    Only paths ending in one of ``gated_suffixes`` are gated; control routes
    and liveness stay reachable while the data plane is paused. The admitted
    operation's lease is installed in the request context so downstream calls
    made while handling it carry ``x-ng-admission-lease`` automatically.
    """

    def __init__(self, app: Any, limiter: AdmissionLimiter, gated_suffixes: tuple[str, ...]) -> None:
        self._app = app
        self._limiter = limiter
        self._gated_suffixes = tuple(gated_suffixes)

    def _gated(self, scope: dict[str, Any]) -> bool:
        if scope.get("type") != "http":
            return False
        path = scope.get("path", "")
        return path.endswith(self._gated_suffixes)

    @staticmethod
    def _headers(scope: dict[str, Any]) -> dict[str, str]:
        wanted = {ROLLOUT_ID_HEADER, ATTEMPT_INDEX_HEADER, ADMISSION_LEASE_HEADER, PLANE_HEADER}
        found: dict[str, str] = {}
        for name, value in scope.get("headers") or ():
            key = name.decode("latin-1").lower()
            if key in wanted:
                found[key] = value.decode("latin-1")
        return found

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not self._gated(scope):
            await self._app(scope, receive, send)
            return

        headers = self._headers(scope)
        attempt_raw = headers.get(ATTEMPT_INDEX_HEADER)
        try:
            attempt_index = int(attempt_raw) if attempt_raw is not None else None
        except ValueError:
            attempt_index = None

        try:
            ticket = self._limiter.admit(
                rollout_id=headers.get(ROLLOUT_ID_HEADER),
                attempt_index=attempt_index,
                lease=headers.get(ADMISSION_LEASE_HEADER),
                plane=headers.get(PLANE_HEADER),
            )
        except ControlError as e:
            response = JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.code, "detail": e.detail}},
                headers={"retry-after": "1"},
            )
            await response(scope, receive, send)
            return

        try:
            with admission_lease_context(ticket.lease):
                await self._app(scope, receive, send)
        finally:
            # The ASGI call returns only after the response (including a
            # streamed one) has finished sending, so releasing here keeps the
            # request in flight until its response completes.
            self._limiter.release(ticket)
