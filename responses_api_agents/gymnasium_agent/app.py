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

"""Agent for GymnasiumServer resources servers (resources_servers.gymnasium) which implements the Gymnasium API."""

import logging
import uuid
from time import time
from typing import Any, List, Optional

from fastapi import Body, HTTPException, Request, Response
from pydantic import ConfigDict, Field, TypeAdapter

from nemo_gym.base_resources_server import (
    SESSION_STATE_URL_PREFIX,
    BaseRunRequest,
    BaseVerifyResponse,
    SessionExportResponse,
    SessionRestoreResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputItem,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import get_response_json, raise_for_status, rollout_path_prefix
from nemo_gym.session_state import FileSessionStateStore, ToolBoundaryRecord
from nemo_gym.session_state.records import select_resume_records
from resources_servers.gymnasium import EnvResetResponse, EnvStepResponse


_LOGGER = logging.getLogger(__name__)

# Run-body flag: resume from the last durable boundary instead of resetting.
_NG_RESUME_KEY = "_ng_resume"

_INPUT_ITEMS_ADAPTER = TypeAdapter(List[NeMoGymResponseInputItem])


class GymnasiumAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = Field(10, ge=1)
    # Root directory of the session-state store for partial rollout
    # checkpointing; must match the resources server's ``session_state_dir``.
    session_state_dir: Optional[str] = None
    # Export the environment every N steps (boundary 0 and boundary records
    # every step regardless). Raising this trades shared-filesystem IO for
    # regenerated steps on resume.
    session_state_snapshot_every_n_steps: int = 1


class GymnasiumAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class GymnasiumRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    terminated: bool = False
    truncated: bool = False
    info: dict = {}


class GymnasiumAgent(SimpleResponsesAPIAgent):
    config: GymnasiumAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_resp)
        result = NeMoGymResponse.model_validate(await get_response_json(model_resp))
        for k, v in model_resp.cookies.items():
            response.set_cookie(k, v)
        return result

    async def run(self, request: Request, body: GymnasiumAgentRunRequest) -> GymnasiumRunResponse:
        # Preserve auth/routing cookies and then merge in any session cookies
        # issued by the resource server during the rollout.
        env_cookies = dict(request.cookies)
        model_url_path = self.url_path_for_run("/v1/responses", body)

        # Session checkpointing: keyed by rollout id, independent of capture.
        session_rollout_id = maybe_rollout_id_from_run_body(body) if self.config.session_state_dir else None
        session_prefix = rollout_path_prefix(session_rollout_id) if session_rollout_id else ""
        session_store = FileSessionStateStore(self.config.session_state_dir) if session_rollout_id else None
        resume = session_rollout_id is not None and bool((body.model_extra or {}).get(_NG_RESUME_KEY))

        resume_records: list[ToolBoundaryRecord] = []
        if resume:
            # Skip /reset: rebuild the reset observation from the boundary-0
            # record and restore the environment at the latest boundary.
            resume_records = select_resume_records(await session_store.read_boundaries(session_rollout_id))
            if not resume_records or resume_records[0].boundary_index != 0:
                raise HTTPException(
                    status_code=409,
                    detail=f"resume requested but no boundary-0 record exists for rollout {session_rollout_id}",
                )
            latest_record = resume_records[-1]
            restore_body: dict = {"boundary_index": latest_record.boundary_index}
            if latest_record.env_state is not None:
                restore_body["state"] = latest_record.env_state
            restore_raw = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"{session_prefix}{SESSION_STATE_URL_PREFIX}/restore",
                json=restore_body,
                cookies=env_cookies,
            )
            await raise_for_status(restore_raw)
            restore = SessionRestoreResponse.model_validate(await get_response_json(restore_raw))
            if restore.supported and not restore.restored:
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot restore session for rollout {session_rollout_id}: {restore.detail}",
                )
            if restore_raw.cookies:
                env_cookies.update(restore_raw.cookies)
            boundary0_extras = resume_records[0].model_extra or {}
            reset_payload: Any = {
                "observation": boundary0_extras.get("reset_observation"),
                "info": boundary0_extras.get("reset_info") or {},
            }
        else:
            if session_rollout_id is not None:
                # Rerun hygiene: an abandoned attempt must not resume by accident.
                await session_store.clear_rollout(session_rollout_id)
            reset_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"{session_prefix}/reset",
                json=body.model_dump(),
                cookies=env_cookies,
            )
            await raise_for_status(reset_resp)
            if reset_resp.cookies:
                env_cookies.update(reset_resp.cookies)

        supports_explicit_close = False
        try:
            # A successful reset owns a stateful server slot even if response
            # decoding or schema validation fails, so validation belongs
            # inside the same cleanup boundary as the rollout itself.
            if not resume:
                reset_payload = await get_response_json(reset_resp)
            if isinstance(reset_payload, dict) and isinstance(reset_payload.get("info"), dict):
                supports_explicit_close = reset_payload["info"].get("supports_explicit_close") is True
            reset_data = EnvResetResponse.model_validate(reset_payload)
            if session_rollout_id is not None and not resume:
                # Boundary 0: the environment right after reset, plus the reset
                # observation, which lives in the base input rather than in the
                # step outputs and is otherwise lost to a resume.
                env_cookies = await self._commit_boundary(
                    session_store=session_store,
                    rollout_id=session_rollout_id,
                    session_prefix=session_prefix,
                    boundary_index=0,
                    output_items=[],
                    usage=None,
                    env_cookies=env_cookies,
                    extras={
                        "reset_observation": reset_data.observation,
                        "reset_info": reset_data.info,
                        "total_reward": 0.0,
                    },
                )
            result = await self._run_open_episode(
                body,
                model_url_path,
                reset_data,
                env_cookies,
                session_store=session_store,
                session_rollout_id=session_rollout_id,
                session_prefix=session_prefix,
                resume_records=resume_records,
            )
        except BaseException:
            if supports_explicit_close:
                # Preserve the original model/transport/cancellation failure.
                try:
                    await self._close_environment(env_cookies)
                except Exception:
                    _LOGGER.exception("Failed to close Gymnasium environment after rollout error")
            raise

        if not supports_explicit_close:
            return result
        try:
            await self._close_environment(env_cookies)
        except Exception as exc:
            _LOGGER.exception("Completed Gymnasium rollout, but environment cleanup failed")
            result = result.model_copy(
                update={
                    "info": {
                        **(result.info or {}),
                        "cleanup_warning": {
                            "operation": "close",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                }
            )
        return result

    async def _close_environment(self, env_cookies) -> None:
        """Close an environment that advertised the optional endpoint."""

        close_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close",
            json={},
            cookies=env_cookies,
        )
        await raise_for_status(close_resp)

    async def _commit_boundary(
        self,
        *,
        session_store: FileSessionStateStore,
        rollout_id: str,
        session_prefix: str,
        boundary_index: int,
        output_items: list[Any],
        usage: Optional[NeMoGymResponseUsage],
        env_cookies: Any,
        extras: dict[str, Any],
        take_snapshot: bool = True,
        response_id: Optional[str] = None,
    ) -> Any:
        """Make one boundary durable: environment state first, boundary record second.

        Small states come back inline and ride in the record (one append, one
        fsync); large states go to a snapshot file. The record is the commit
        point; a failed export on a stateful server skips the record so the
        store keeps its last consistent pair.
        """
        env_exported = False
        env_state = None
        if take_snapshot:
            try:
                export_raw = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"{session_prefix}{SESSION_STATE_URL_PREFIX}/export",
                    json={"boundary_index": boundary_index, "inline_ok": True},
                    cookies=env_cookies,
                )
                await raise_for_status(export_raw)
                export = SessionExportResponse.model_validate(await get_response_json(export_raw))
                if export_raw.cookies:
                    env_cookies = dict(env_cookies) | dict(export_raw.cookies)
            except Exception:
                # A failed export must not fail the live rollout; it only limits
                # how far back a future resume must go.
                return env_cookies
            if export.supported and not export.exported:
                return env_cookies
            env_state = export.state if export.inline else None
            env_exported = export.exported and not export.inline
        await session_store.append_boundary(
            ToolBoundaryRecord(
                rollout_id=rollout_id,
                boundary_index=boundary_index,
                output_items=[item.model_dump(mode="json") for item in output_items],
                usage=usage.model_dump(mode="json") if usage is not None else None,
                env_exported=env_exported,
                env_state=env_state,
                response_id=response_id,
                created_at=time(),
                **extras,
            )
        )
        return env_cookies

    async def _run_open_episode(
        self,
        body: GymnasiumAgentRunRequest,
        model_url_path: str,
        reset_data: EnvResetResponse,
        env_cookies,
        *,
        session_store: Optional[FileSessionStateStore] = None,
        session_rollout_id: Optional[str] = None,
        session_prefix: str = "",
        resume_records: Optional[list[ToolBoundaryRecord]] = None,
    ) -> GymnasiumRunResponse:
        """Drive an already-reset (or restored) episode; :meth:`run` owns its cleanup."""

        base_body = body.responses_create_params.model_copy(deep=True)
        if isinstance(base_body.input, str):
            base_body.input = [NeMoGymEasyInputMessage(role="user", content=base_body.input)]
        if reset_data.observation:
            base_body.input = list(base_body.input) + [
                NeMoGymEasyInputMessage(role="user", content=reset_data.observation)
            ]

        new_outputs = []
        total_reward = 0.0
        usage = None
        model_server_cookies = None
        step_data = EnvStepResponse(terminated=False, truncated=True, reward=0.0)
        last_model_response = None
        finished = False

        completed_steps = 0
        if resume_records and len(resume_records) > 1:
            # Records after boundary 0 hold each completed step's conversation
            # delta; agent-local loop state rides on the last one.
            for record in resume_records[1:]:
                new_outputs.extend(_INPUT_ITEMS_ADAPTER.validate_python(record.output_items))
            last_record = resume_records[-1]
            completed_steps = last_record.boundary_index
            total_reward = float((last_record.model_extra or {}).get("total_reward", 0.0))
            if last_record.usage is not None:
                usage = NeMoGymResponseUsage.model_validate(last_record.usage)
        if completed_steps >= self.config.max_steps:
            raise HTTPException(
                status_code=409,
                detail=f"resume requested but rollout {session_rollout_id} already used all {self.config.max_steps} steps",
            )

        for _ in range(self.config.max_steps - completed_steps):
            new_body = base_body.model_copy(update={"input": base_body.input + new_outputs})

            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_resp)
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp))
            model_server_cookies = model_resp.cookies
            last_model_response = model_response

            new_outputs.extend(model_response.output)
            step_items: list[Any] = list(model_response.output)

            usage = accumulate_response_usage(usage, model_response.usage)

            step_body = body.model_dump() | {"response": model_response.model_dump()}
            if (reset_data.info or {}).get("supports_step_idempotency") is True:
                step_body["_ng_step_request_id"] = uuid.uuid4().hex
            step_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"{session_prefix}/step",
                json=step_body,
                cookies=env_cookies,
            )
            await raise_for_status(step_resp)
            step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
            total_reward += step_data.reward
            if step_resp.cookies:
                env_cookies.update(step_resp.cookies)

            if step_data.terminated or step_data.truncated:
                finished = True
                break

            for tool_output in (step_data.info or {}).get("tool_outputs", []):
                function_call_output = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=tool_output["call_id"],
                    output=tool_output["output"],
                )
                new_outputs.append(function_call_output)
                step_items.append(function_call_output)

            if step_data.observation:
                observation_message = NeMoGymEasyInputMessage(role="user", content=step_data.observation)
                new_outputs.append(observation_message)
                step_items.append(observation_message)

            completed_steps += 1
            if session_store is not None and session_rollout_id is not None:
                cadence = max(1, self.config.session_state_snapshot_every_n_steps)
                env_cookies = await self._commit_boundary(
                    session_store=session_store,
                    rollout_id=session_rollout_id,
                    session_prefix=session_prefix,
                    boundary_index=completed_steps,
                    output_items=step_items,
                    usage=usage,
                    env_cookies=env_cookies,
                    extras={"total_reward": total_reward},
                    take_snapshot=completed_steps % cadence == 0,
                    response_id=model_response.id,
                )

        if not finished:
            step_data = step_data.model_copy(update={"truncated": True})

        last_model_response.output = new_outputs
        last_model_response.usage = usage

        return GymnasiumRunResponse(
            responses_create_params=base_body,
            response=last_model_response,
            reward=total_reward,
            terminated=step_data.terminated,
            truncated=step_data.truncated,
            info=step_data.info,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    GymnasiumAgent.run_webserver()
