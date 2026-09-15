# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in session routes for episode participants."""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request

from nemo_gym.episode import (
    AgentSessionCloseRequest,
    AgentSessionCloseResponse,
    AgentSessionCreateRequest,
    AgentSessionCreateResponse,
    ResourcesSessionCloseRequest,
    ResourcesSessionCloseResponse,
)


AGENT_SESSION_HEADER = "X-NeMo-Gym-Agent-Session-Id"


@dataclass
class AgentSession:
    """Store state owned by one agent-server worker."""

    request: AgentSessionCreateRequest
    state: Any


class AgentSessionServerMixin:
    """Add opt-in agent-session creation and cleanup routes."""

    _agent_sessions: dict[str, AgentSession]

    def initialize_agent_sessions(self) -> None:
        if getattr(self.config, "num_workers", None) not in (None, 1):
            raise ValueError("Agent sessions require num_workers=1")
        self._agent_sessions = {}

    def setup_agent_session_routes(self, app: FastAPI) -> None:
        app.post("/v1/agent_sessions")(self.create_agent_session)
        app.post("/v1/agent_sessions/close")(self.close_agent_session)

    async def create_agent_session(
        self,
        body: AgentSessionCreateRequest,
    ) -> AgentSessionCreateResponse:
        agent_session_id = f"agent-session-{uuid4().hex}"
        state = await self.open_agent_session(agent_session_id, body)
        self._agent_sessions[agent_session_id] = AgentSession(request=body, state=state)
        return AgentSessionCreateResponse(agent_session_id=agent_session_id)

    async def close_agent_session(
        self,
        body: AgentSessionCloseRequest,
    ) -> AgentSessionCloseResponse:
        session = self.require_agent_session(body.agent_session_id)
        observations = await self.teardown_agent_session(body.agent_session_id, session)
        del self._agent_sessions[body.agent_session_id]
        return AgentSessionCloseResponse(
            agent_session_id=body.agent_session_id,
            agent_observations=observations,
            resources_cookies=self.resources_cookies_for_session(session),
        )

    def require_agent_session(self, agent_session_id: str) -> AgentSession:
        try:
            return self._agent_sessions[agent_session_id]
        except KeyError as error:
            raise ValueError(f"Unknown agent_session_id: {agent_session_id}") from error

    async def open_agent_session(
        self,
        agent_session_id: str,
        body: AgentSessionCreateRequest,
    ) -> Any:
        """Create agent-owned session state."""
        return None

    async def teardown_agent_session(self, agent_session_id: str, session: AgentSession) -> Any:
        """Destroy agent-owned session state and return observations."""
        return None

    def resources_cookies_for_session(self, session: AgentSession) -> dict[str, str] | None:
        """Return resources cookies updated during the agent session."""
        return None


class ResourcesSessionServerMixin:
    """Add an opt-in resources-session cleanup route."""

    def setup_resources_session_routes(self, app: FastAPI) -> None:
        app.post("/close_session")(self.close_resources_session)

    async def close_resources_session(
        self,
        request: Request,
        body: ResourcesSessionCloseRequest,
    ) -> ResourcesSessionCloseResponse:
        """Destroy resources-server-owned session state."""
        raise NotImplementedError
