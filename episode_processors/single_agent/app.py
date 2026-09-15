# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources-backed single-agent episode processor."""

from typing import Any, Literal

from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import Body, FastAPI
from pydantic import ConfigDict

from nemo_gym.base_episode_processor import (
    BaseEpisodeProcessor,
    BaseEpisodeProcessorConfig,
    EpisodeContext,
    HandledEpisodeError,
)
from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT, AgentServerRef, ResourcesServerRef
from nemo_gym.episode import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    DirectResourcesToolAccess,
    EpisodeId,
    MCPResourcesToolAccess,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResponsesEpisodeResourcesVerifyRequest,
    ResponsesEpisodeVerificationInput,
    SingleAgentEpisodeFailure,
    SingleAgentEpisodeInput,
    SingleAgentEpisodeRequest,
    SingleAgentEpisodeResponse,
    SingleAgentEpisodeResult,
    TaskId,
)
from nemo_gym.episode_sessions import AGENT_SESSION_HEADER
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    TOKEN_ID_CAPTURE_BLOCK,
    get_first_server_config_dict,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import get_response_json, raise_for_status


class SingleAgentEpisodeProcessorConfig(BaseEpisodeProcessorConfig):
    """Bind one resources server and one agent server."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    resources_tool_transport: Literal["direct_http", "mcp"] = "direct_http"


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    return isinstance(error, (ClientConnectionError, TimeoutError))


class SingleAgentEpisodeProcessor(BaseEpisodeProcessor[SingleAgentEpisodeRequest, SingleAgentEpisodeResponse]):
    """Run seed, one agent activation, verification, and cleanup."""

    config: SingleAgentEpisodeProcessorConfig
    request_model = SingleAgentEpisodeRequest
    response_model = SingleAgentEpisodeResponse

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/run_legacy")(self.run_legacy)
        return app

    async def run_legacy(self, row: dict[str, Any] = Body()) -> dict[str, Any]:
        """Adapt one legacy flat rollout row at the processor boundary."""
        response = await self.run(self._legacy_request(row))
        if response.failure is not None:
            failure = {
                "_ng_failure_class": "episode_processor_failed",
                "_ng_failure_terminal": response.failure.terminal,
                "_ng_failure_message": response.failure.message,
                "_ng_failure_stage": response.failure.stage,
            }
            if response.failure.partial_response is not None:
                failure["_ng_failure_partial_response"] = response.failure.partial_response.model_dump(mode="json")
            return failure
        if response.result is None:
            raise ValueError("Successful episode response has no result")
        result = response.result.verification.model_dump(mode="json")
        if response.result.agent_observations is not None:
            result["ng_agent_observations"] = response.result.agent_observations.model_dump(mode="json")
        return result

    @staticmethod
    def _legacy_request(row: dict[str, Any]) -> SingleAgentEpisodeRequest:
        task_source = row.get(TASK_SOURCE_KEY_NAME)
        if not isinstance(task_source, str) or not task_source:
            raise ValueError("Episode processor rows require task_source")
        task_id = next(
            (str(row[key]) for key in ("task_id", "problem_id", "instance_id") if row.get(key) is not None),
            str(row[TASK_INDEX_KEY_NAME]),
        )
        rollout_id = maybe_rollout_id_from_run_body(row) or f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError(f"Invalid episode attempt: {attempt!r}")
        excluded = {
            RESPONSES_CREATE_PARAMS_KEY_NAME,
            AGENT_REF_KEY_NAME,
            TASK_SOURCE_KEY_NAME,
            SKILLS_REF_KEY_NAME,
            ROLLOUT_ID_KEY_NAME,
            TASK_INDEX_KEY_NAME,
            ROLLOUT_INDEX_KEY_NAME,
            ATTEMPT_INDEX_KEY_NAME,
        }
        return SingleAgentEpisodeRequest(
            episode_id=EpisodeId(rollout_id=rollout_id, attempt=attempt),
            task_id=TaskId(task_source=task_source, task_id=task_id),
            episode_input=SingleAgentEpisodeInput(
                responses_create_params=row[RESPONSES_CREATE_PARAMS_KEY_NAME],
                task_data={key: value for key, value in row.items() if key not in excluded},
            ),
        )

    def _agent_responses_path(self, request: SingleAgentEpisodeRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.agent_server.name,
        )
        token_capture = bool(block.get("enabled", False)) and (
            bool(block.get("all_agents", False)) or bool(agent_config.get("token_id_capture", False))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses"

    def _failure(
        self,
        *,
        stage: str,
        message: str,
        terminal: bool,
        partial_response: Any = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            SingleAgentEpisodeFailure(
                stage=stage,
                message=message[:2000],
                terminal=terminal,
                partial_response=partial_response,
            )
        )

    async def process(
        self,
        request: SingleAgentEpisodeRequest,
        context: EpisodeContext,
    ) -> SingleAgentEpisodeResponse:
        episode_input = request.episode_input
        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    episode_id=request.episode_id,
                    task_id=request.task_id,
                    task_data=episode_input.task_data,
                ),
            )
            await raise_for_status(seed_http_response)
            resources_cookies = _cookies(seed_http_response)
            seed = ResourcesSeedSessionResponse.model_validate(await get_response_json(seed_http_response))
        except Exception as error:
            raise self._failure(
                stage="seed",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        async def close_resources() -> None:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id),
                cookies=resources_cookies,
            )
            await raise_for_status(close_response)

        context.register_cleanup("resources session", close_resources)

        if self.config.resources_tool_transport == "mcp":
            if seed.resources_tools is None:
                raise self._failure(
                    stage="seed",
                    message="The resources server did not return MCP metadata",
                    terminal=True,
                )
            resources_access = MCPResourcesToolAccess(kind="mcp", metadata=seed.resources_tools)
        else:
            resources_access = DirectResourcesToolAccess(
                kind="direct_http",
                base_url=self.server_client._resolve_base_url(self.config.resources_server.name),
                cookies=resources_cookies,
            )
        try:
            agent_create_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    episode_id=request.episode_id,
                    resources_access=resources_access,
                    sandbox_access=seed.sandbox_access,
                ),
            )
            await raise_for_status(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
        except Exception as error:
            raise self._failure(
                stage="agent",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        close_result: AgentCloseSessionResponse | None = None

        async def close_agent() -> None:
            nonlocal close_result
            close_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(agent_session_id=agent_session.agent_session_id),
            )
            await raise_for_status(close_http_response)
            close_result = AgentCloseSessionResponse.model_validate(await get_response_json(close_http_response))

        agent_cleanup = context.register_cleanup("agent session", close_agent)
        agent_response = None
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=episode_input.responses_create_params,
                headers={AGENT_SESSION_HEADER: agent_session.agent_session_id},
            )
            await raise_for_status(agent_http_response)
            from nemo_gym.openai_utils import NeMoGymResponse

            agent_response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
        except Exception as error:
            raise self._failure(
                stage="agent",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        try:
            await agent_cleanup.close()
        except Exception as error:
            raise self._failure(
                stage="cleanup",
                message=str(error),
                terminal=True,
                partial_response=agent_response,
            ) from error
        if close_result is not None and close_result.resources_cookies is not None:
            resources_cookies = close_result.resources_cookies

        try:
            verify_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=ResponsesEpisodeResourcesVerifyRequest(
                    episode_id=request.episode_id,
                    verification_input=ResponsesEpisodeVerificationInput(
                        responses_create_params=episode_input.responses_create_params,
                        response=agent_response,
                    ),
                ),
                cookies=resources_cookies,
            )
            await raise_for_status(verify_http_response)
            verification = BaseVerifyResponse.model_validate(await get_response_json(verify_http_response))
        except Exception as error:
            raise self._failure(
                stage="verification",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
                partial_response=agent_response,
            ) from error

        return SingleAgentEpisodeResponse(
            episode_id=request.episode_id,
            task_id=request.task_id,
            result=SingleAgentEpisodeResult(
                verification=verification,
                agent_observations=close_result.agent_observations if close_result is not None else None,
            ),
        )


if __name__ == "__main__":
    SingleAgentEpisodeProcessor.run_webserver()
