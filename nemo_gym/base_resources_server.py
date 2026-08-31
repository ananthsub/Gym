# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from fastapi import FastAPI
from pydantic import BaseModel


if TYPE_CHECKING:
    # Type-only: importing MCPTool at runtime would be circular (mcp_auto_exposure imports this
    # module) and would pull the mcp SDK into agent/model processes that never need it.
    from nemo_gym.mcp_auto_exposure import MCPTool

import json
from time import time

from fastapi import HTTPException, Request

from nemo_gym.checkpoint.control import ControlCapabilities
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import RolloutContextMiddleware, current_rollout_id
from nemo_gym.server_utils import SESSION_ID_KEY, BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.session_state import FileSessionStateStore, SessionSnapshot


# Framework routes for partial-rollout session checkpointing. Multi-segment on
# purpose: they are framework plumbing, not tools, and must never be harvested
# for MCP exposure (mcp_auto_exposure skips this prefix explicitly).
SESSION_STATE_URL_PREFIX = "/ng-session"

# States at or below this serialized size are returned inline in the export
# response so the agent can embed them in the boundary record: one append and
# one fsync per boundary instead of an extra snapshot file with its
# rename-and-directory-fsync dance — the difference that matters on
# metadata-bound shared filesystems (Lustre). Larger states keep the file.
SESSION_STATE_INLINE_MAX_BYTES = 32_768


NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
NEMO_GYM_MCP_METADATA_KEY = "mcp"
# Salt namespacing the signed MCP session token, so it can't be confused with another signer
# that happens to share the same session-middleware secret.
_MCP_TOKEN_SALT = "nemo-gym-mcp-session-token"


def normalize_tool_name(name: str, server_name: Optional[str] = None) -> str:
    """Map a trajectory tool-call name to the server's bare tool name.

    HTTP-driven agents record bare tool names ("email_reply_email"); MCP-native agents (e.g.
    Claude Code) record them namespaced per server ("mcp__workplace_assistant__email_reply_email").
    Verifiers compare trajectory names against dataset/ground-truth vocabulary, so names are
    normalized before verify sees them and rollouts score identically on both transports.
    Non-namespaced names pass through unchanged. When ``server_name`` is given, only that server's
    prefix is stripped (robust to tool names that themselves contain double underscores).
    This runs only for servers exposed over MCP and mirrors how MCP clients namespace tool names,
    so a real tool that is itself named ``mcp__<server>__x`` being stripped is accepted.
    """
    if not name.startswith("mcp__"):
        return name
    if server_name is not None:
        prefix = f"mcp__{server_name}__"
        return name[len(prefix) :] if name.startswith(prefix) else name
    _, sep, tool = name[len("mcp__") :].partition("__")
    return tool if sep else name


# Tool names that would collide with the resources server's own endpoints if advertised over MCP.
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "aggregate_metrics", "mcp"})


class ReverifyMode(str, Enum):
    STATELESS = "stateless"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    # Opt in to serve this server's tool routes over MCP; default off.
    expose_tools_over_mcp: bool = False
    # Root directory of the session-state store for partial rollout
    # checkpointing. Shared storage, typically co-located with the token
    # capture dir; the agent driving this server must point at the same dir.
    # None (the default) disables session checkpointing for this server.
    session_state_dir: Optional[str] = None
    # The mode of reverification (for gym eval reverify) of this server.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNKNOWN


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float


class BaseMultiRewardVerifyResponse(BaseVerifyResponse):
    """Base verify response for environments with multiple reward objectives.

    Subclass this response instead of declaring ``reward_components`` on an
    environment-specific ``BaseVerifyResponse`` subclass. The mapping is required, and
    its objective keys should remain consistent across every task in the environment.

    Set the inherited ``reward`` to the scalar aggregate expected by single-reward
    consumers. To include individual objectives in aggregate metrics, also expose them
    as top-level numeric fields because metrics do not descend into this mapping. See
    ``resources_servers/example_tool_call_multireward`` for a complete example.
    """

    reward_components: dict[str, float]


class BaseSeedSessionRequest(BaseModel):
    pass


class BaseSeedSessionResponse(BaseModel):
    pass


class SessionExportRequest(BaseModel):
    # The agent step this export is taken at (state after that step's tools).
    boundary_index: int
    # Caller accepts the state inline in the response (to embed in its
    # boundary record) instead of a snapshot file. Off by default so older
    # callers keep file semantics.
    inline_ok: bool = False


class SessionExportResponse(BaseModel):
    supported: bool
    exported: bool = False
    boundary_index: Optional[int] = None
    # Set when the state was returned inline instead of written to a file.
    inline: bool = False
    state: Optional[dict[str, Any]] = None
    detail: str = ""


class SessionRestoreRequest(BaseModel):
    boundary_index: int
    # Inline state from the boundary record; when present the server restores
    # from it directly instead of reading a snapshot file.
    state: Optional[dict[str, Any]] = None


class SessionRestoreResponse(BaseModel):
    supported: bool
    restored: bool = False
    detail: str = ""


class MCPServerMetadata(BaseModel):
    """Metadata returned from /seed_session for per-rollout Gym MCP access."""

    server_name: str
    url_path: str = "/mcp"
    transport: str = "http"
    headers: dict[str, str]


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig

    _CONTROL_COMPONENT = "resources_servers"

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        app.add_middleware(RolloutContextMiddleware)

        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(judge_failsafe(self.verify))
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.get("/reverify_mode")(self.get_reverify_mode)
        self.setup_session_state_routes(app)
        self.setup_control_plane(app)

        return app

    def control_capabilities(self) -> ControlCapabilities:
        capabilities = super().control_capabilities()
        if self.supports_session_state():
            capabilities.checkpoint_mode = "export_restore"
            capabilities.concurrency_contract = "serialized_per_session"
        return capabilities

    def setup_session_state_routes(self, app: FastAPI) -> None:
        """Register the session-checkpointing framework routes.

        Servers that build their own FastAPI app instead of calling
        ``SimpleResourcesServer.setup_webserver`` (e.g. GymnasiumServer) must
        call this themselves to be checkpointable.
        """
        app.post(f"{SESSION_STATE_URL_PREFIX}/export")(self.session_export)
        app.post(f"{SESSION_STATE_URL_PREFIX}/restore")(self.session_restore)

    def normalize_tool_name(self, name: str) -> str:
        """Strip this server's MCP namespace from a trajectory tool-call name (see module function)."""
        return normalize_tool_name(name, self.config.name or self.__class__.__name__)

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Optional[Any]) -> Optional[list["MCPTool"]]:
        """Return the MCP tools to expose (default: the auto-harvested typed POST routes).

        Override to exclude (filter harvested), add catch-all-backed tools (harvested + [catchall.tool(...)]),
        or disable (return None). 'catchall' is None unless the server has one parameterized catch-all route.
        """
        return harvested

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> Optional[list[str]]:
        """Per-session tool restriction: return the tool names allowed for this rollout's MCP token,
        or ``None`` (the default) for unrestricted. ``seed_body`` is the JSON body POSTed to
        ``/seed_session``.
        """
        return None

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    # -- session checkpointing (partial rollouts) ------------------------------
    #
    # Export runs in the process that owns the live session objects and is
    # driven by the agent at tool boundaries; the trainer never calls servers
    # directly. Both routes require a rollout context (``/ng-rollout/<id>/...``
    # prefix), which also binds the transport session id to the rollout id, so
    # a restored session is addressable by any worker or restarted process
    # without a cookie handoff.

    def supports_session_state(self) -> bool:
        """Capability declaration: whether this server can export/restore session state.

        Override to return True and implement ``export_session_state`` /
        ``restore_session_state``. Servers without support keep whole-rollout
        retry semantics; the agent treats them as stateless.
        """
        return False

    async def export_session_state(self, session_id: str) -> Optional[dict[str, Any]]:
        """Return a JSON-serializable snapshot of one session's state, or None if there is nothing to export.

        Servers whose real state lives in an external durable backend should
        return the reconnect descriptor (e.g. ``AsyncSandbox.serialize()``),
        not the state itself.
        """
        return None

    async def restore_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Rebind ``state`` (as produced by ``export_session_state``) to ``session_id``."""
        raise NotImplementedError

    def _session_state_store(self) -> Optional[FileSessionStateStore]:
        if not self.config.session_state_dir:
            return None
        return FileSessionStateStore(self.config.session_state_dir)

    @staticmethod
    def _require_rollout_id() -> str:
        rollout_id = current_rollout_id()
        if rollout_id is None:
            # Session state is keyed by rollout id, never by the session cookie:
            # a cookie-keyed snapshot could not be found again after a restart.
            raise HTTPException(
                status_code=400,
                detail="session-state routes require the /ng-rollout/<rollout_id>/ request prefix",
            )
        return rollout_id

    async def session_export(self, request: Request, body: SessionExportRequest) -> SessionExportResponse:
        store = self._session_state_store()
        if store is None or not self.supports_session_state():
            return SessionExportResponse(supported=False, detail="session state not enabled for this server")
        rollout_id = self._require_rollout_id()
        state = await self.export_session_state(request.session[SESSION_ID_KEY])
        if state is None:
            return SessionExportResponse(supported=True, detail="no session state to export")
        if body.inline_ok and len(json.dumps(state)) <= SESSION_STATE_INLINE_MAX_BYTES:
            # Small states ride back to the caller and into the boundary
            # record: no snapshot file, no extra fsyncs on shared storage.
            return SessionExportResponse(
                supported=True, exported=True, boundary_index=body.boundary_index, inline=True, state=state
            )
        await store.write_snapshot(
            SessionSnapshot(
                rollout_id=rollout_id,
                server_name=self.config.name or self.__class__.__name__,
                boundary_index=body.boundary_index,
                state=state,
                created_at=time(),
            )
        )
        return SessionExportResponse(supported=True, exported=True, boundary_index=body.boundary_index)

    async def session_restore(self, request: Request, body: SessionRestoreRequest) -> SessionRestoreResponse:
        store = self._session_state_store()
        if store is None or not self.supports_session_state():
            return SessionRestoreResponse(supported=False, detail="session state not enabled for this server")
        rollout_id = self._require_rollout_id()
        if body.state is not None:
            await self.restore_session_state(request.session[SESSION_ID_KEY], body.state)
            return SessionRestoreResponse(supported=True, restored=True)
        snapshot = await store.read_snapshot(
            rollout_id, self.config.name or self.__class__.__name__, body.boundary_index
        )
        if snapshot is None:
            return SessionRestoreResponse(supported=True, detail=f"no snapshot at boundary {body.boundary_index}")
        await self.restore_session_state(request.session[SESSION_ID_KEY], snapshot.state)
        return SessionRestoreResponse(supported=True, restored=True)

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compute aggregate metrics from verify responses.

        RewardProfiler provides baseline stats. Override compute_metrics() and/or
        get_key_metrics() for benchmark-specific customization.
        """
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )

    async def get_reverify_mode(self) -> ReverifyMode:
        return self.config.REVERIFY_MODE
