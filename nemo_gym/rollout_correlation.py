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
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_rollout_id", default=None)
_ATTEMPT_INDEX: ContextVar[Optional[int]] = ContextVar("nemo_gym_attempt_index", default=None)

# These headers propagate execution identity on every downstream Gym call,
# independent of observability and token capture, so a server never depends on
# the capture path prefix to learn it. The rollout-id header carries the same
# transport id a prefixed path would (session derivation keys off it); the
# attempt header carries the attempt index as its own field, which is
# authoritative over the ``-a{n}`` suffix.
ROLLOUT_ID_HEADER = "x-ng-rollout-id"
ATTEMPT_INDEX_HEADER = "x-ng-attempt-index"

# Echoed by the model server on capture routes with the call id it minted for
# the request. The caller records it as the join key between its own state
# (an agent boundary, a custody row) and the model server's ledger.
MODEL_CALL_ID_HEADER = "x-ng-model-call-id"

# The transport id appends ``-a{n}`` for re-dispatch attempts. The suffix is a
# capture and routing key, never the logical identity. This pattern recovers
# the split when only the transport id is available; header-carried values are
# authoritative because an explicit logical id may itself end in ``-a{n}``.
_ATTEMPT_SUFFIX_PATTERN = re.compile(r"^(?P<logical>.+)-a(?P<attempt>\d+)$")


def split_transport_rollout_id(rollout_id: Optional[str]) -> tuple[Optional[str], int]:
    """Split a transport rollout id into ``(logical_id, attempt_index)``.

    A missing or unsuffixed id has attempt index 0. The split is a fallback for
    path-only sources; callers holding explicit header values must prefer them.
    """
    if rollout_id is None:
        return None, 0
    match = _ATTEMPT_SUFFIX_PATTERN.match(rollout_id)
    if match is None:
        return rollout_id, 0
    return match.group("logical"), int(match.group("attempt"))


# A capture id is a path segment in ``/ng-rollout/<id>/...``.
# Restrict it to characters that survive a path round trip.
# Exclude leading dots because stores also use the id as a filename component.
# Middleware uses the same pattern.
ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def maybe_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[str]:
    """Build the capture key for a run request.

    An explicit ``_ng_rollout_id`` takes precedence.
    Otherwise derive ``"{task}-{rollout}"`` from the task and rollout indices.
    Re-dispatch attempts append ``-a{n}``.
    Writers and consumers must use this same identity.
    Reused task and rollout indices produce a repeated capture key.
    Use an explicit id when numbering restarts across dispatches.
    """
    if not isinstance(body, (BaseModel, Mapping)):
        return None

    def field(key: str) -> Any:
        return body.get(key) if isinstance(body, Mapping) else getattr(body, key, None)

    explicit = field(ROLLOUT_ID_KEY_NAME)
    if explicit is not None:
        # Reject malformed explicit ids instead of sanitizing them.
        # Rewriting would create a key the caller cannot look up.
        if not (isinstance(explicit, str) and ROLLOUT_ID_PATTERN.match(explicit)):
            raise ValueError(
                f"{ROLLOUT_ID_KEY_NAME} must be a string of letters, digits, dots, dashes or "
                f"underscores starting with a letter or digit; got {explicit!r}"
            )
        rollout_id = explicit
    else:
        task = field(TASK_INDEX_KEY_NAME)
        rollout = field(ROLLOUT_INDEX_KEY_NAME)
        if task is None or rollout is None:
            return None
        rollout_id = f"{task}-{rollout}"

    attempt = field(ATTEMPT_INDEX_KEY_NAME)
    if attempt is not None and int(attempt) > 0:
        rollout_id = f"{rollout_id}-a{int(attempt)}"
    return rollout_id


def execution_identity_from_run_body(
    body: BaseModel | Mapping[str, Any] | None,
) -> tuple[Optional[str], Optional[int]]:
    """``(transport_rollout_id, attempt_index)`` for a run request body.

    Unlike capture-key derivation, this never depends on whether observability
    or token capture is enabled: a run request that names its identity always
    establishes it. The transport id carries the ``-a{n}`` suffix for
    re-dispatch attempts; the attempt index is returned as its own value when
    the body names one, so it survives even when the logical id itself ends in
    ``-a{n}``.
    """
    rollout_id = maybe_rollout_id_from_run_body(body)
    if rollout_id is None:
        return None, None
    if not isinstance(body, (BaseModel, Mapping)):  # pragma: no cover - guarded by rollout_id above
        return rollout_id, None
    attempt = (
        body.get(ATTEMPT_INDEX_KEY_NAME) if isinstance(body, Mapping) else getattr(body, ATTEMPT_INDEX_KEY_NAME, None)
    )
    return rollout_id, int(attempt) if attempt is not None else None


def current_rollout_id() -> Optional[str]:
    return _ROLLOUT_ID.get()


def current_attempt_index() -> Optional[int]:
    """The attempt index of the current request context.

    Falls back to the suffix of the transport rollout id when no explicit
    attempt was set; ``None`` when there is no rollout context at all.
    """
    explicit = _ATTEMPT_INDEX.get()
    if explicit is not None:
        return explicit
    rollout_id = _ROLLOUT_ID.get()
    if rollout_id is None:
        return None
    return split_transport_rollout_id(rollout_id)[1]


def current_execution_identity() -> tuple[Optional[str], Optional[int]]:
    """``(logical_rollout_id, attempt_index)`` for the current context.

    The logical id strips the transport ``-a{n}`` suffix. An explicitly set
    attempt index wins over the suffix-derived one.
    """
    rollout_id = _ROLLOUT_ID.get()
    logical, derived = split_transport_rollout_id(rollout_id)
    explicit = _ATTEMPT_INDEX.get()
    if rollout_id is None:
        return None, explicit
    return logical, explicit if explicit is not None else derived


@contextmanager
def rollout_context(rollout_id: Optional[str], attempt_index: Optional[int] = None) -> Iterator[None]:
    token = _ROLLOUT_ID.set(rollout_id)
    attempt_token = _ATTEMPT_INDEX.set(attempt_index)
    try:
        yield
    finally:
        _ATTEMPT_INDEX.reset(attempt_token)
        _ROLLOUT_ID.reset(token)


class RolloutContextMiddleware:
    """Strip a rollout prefix and expose it to downstream Gym calls for this request."""

    # Match the same id characters as ``ROLLOUT_ID_PATTERN``.
    # Anchor the id between the prefix and the remaining path.
    _PREFIX = re.compile(
        rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>{ROLLOUT_ID_PATTERN.pattern.strip('^$')})(?P<rest>/.*)$"
    )

    def __init__(self, app: Any) -> None:
        self._app = app

    @staticmethod
    def _identity_from_headers(scope: dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
        rollout_id: Optional[str] = None
        attempt: Optional[int] = None
        for name, value in scope.get("headers") or ():
            key = name.decode("latin-1").lower()
            if key == ROLLOUT_ID_HEADER:
                candidate = value.decode("latin-1")
                if ROLLOUT_ID_PATTERN.match(candidate):
                    rollout_id = candidate
            elif key == ATTEMPT_INDEX_HEADER:
                try:
                    attempt = int(value.decode("latin-1"))
                except ValueError:
                    attempt = None
        return rollout_id, attempt

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        match = self._PREFIX.match(scope.get("path", ""))
        header_rollout_id, header_attempt = self._identity_from_headers(scope)

        if match is None:
            # No capture prefix: identity headers alone establish the context,
            # so every server learns the execution identity unconditionally.
            if header_rollout_id is None:
                await self._app(scope, receive, send)
                return
            with rollout_context(header_rollout_id, attempt_index=header_attempt):
                await self._app(scope, receive, send)
            return

        path = match.group("rest")
        scope = {**scope, "path": path, "raw_path": path.encode()}
        # The path segment is the transport id; an explicit attempt header is
        # authoritative over its ``-a{n}`` suffix.
        with rollout_context(match.group("rollout_id"), attempt_index=header_attempt):
            await self._app(scope, receive, send)
