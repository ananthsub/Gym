# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared state and opt-in resources-session routes."""

from dataclasses import dataclass

from fastapi import FastAPI, Request

from nemo_gym.episode import (
    AgentSeedSessionRequest,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
)
from nemo_gym.rollout_observability import AgentObservationBundle


AGENT_SESSION_HEADER = "X-NeMo-Gym-Agent-Session-Id"


@dataclass
class AgentSession:
    """Store state owned by one agent-server worker."""

    request: AgentSeedSessionRequest
    state: object
    activation_started: bool = False


@dataclass
class AgentCloseSessionResult:
    """Return data harvested while closing agent-owned session state."""

    agent_observations: AgentObservationBundle | None = None
    resources_cookies: dict[str, str] | None = None


class ResourcesSessionServerMixin:
    """Add an opt-in resources-session cleanup route."""

    def setup_resources_session_routes(self, app: FastAPI) -> None:
        app.post("/close_session")(self.close_resources_session)

    async def close_resources_session(
        self,
        request: Request,
        body: ResourcesCloseSessionRequest,
    ) -> ResourcesCloseSessionResponse:
        """Destroy resources-server-owned session state."""
        raise NotImplementedError
