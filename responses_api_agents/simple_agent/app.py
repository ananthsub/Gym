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
import json
from collections.abc import Mapping
from time import perf_counter, time
from typing import Any, List, Optional

from fastapi import HTTPException, Request, Response
from pydantic import ConfigDict, TypeAdapter, ValidationError

from nemo_gym.base_resources_server import (
    SESSION_STATE_URL_PREFIX,
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SessionExportResponse,
    SessionRestoreResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import get_response_json, raise_for_status, rollout_path_prefix
from nemo_gym.session_state import FileSessionStateStore, ToolBoundaryRecord
from nemo_gym.session_state.records import select_resume_records


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
# Run-body flag: resume this rollout from its last durable tool boundary
# instead of starting fresh. The caller (e.g. NeMo-RL recovery) sets it when
# redispatching an interrupted rollout under the same logical rollout id.
_NG_RESUME_KEY = "_ng_resume"
# Query param carrying the resume intent across the /run -> /v1/responses self-call.
_RESUME_QUERY_PARAM = "ng_session_resume"

_INPUT_ITEMS_ADAPTER = TypeAdapter(List[NeMoGymResponseInputItem])


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    # Root directory of the session-state store for partial rollout
    # checkpointing; must be the same directory the resources server was
    # configured with. None (the default) disables checkpointing.
    session_state_dir: Optional[str] = None
    # Export the environment every N tool steps (boundary records are still
    # appended every step — they are one cheap fsync). Raising this trades
    # shared-filesystem IO for regenerated steps on resume: recovery re-enters
    # at the latest boundary that has environment state.
    session_state_snapshot_every_n_steps: int = 1


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
        session_store: Optional[FileSessionStateStore] = None,
        session_resume: bool = False,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        invocation_id = "root"
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        trajectory_gaps: list[ObservationGap] = []
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        invocation_status = "completed"
        model_server_cookies = None

        # Session checkpointing is active only when the request carries a real
        # rollout id: state is keyed by rollout id, never by session cookie.
        session_active = session_store is not None and rollout_id != "unscoped"
        resources_prefix = rollout_path_prefix(rollout_id) if session_active else ""
        if session_active and session_resume:
            # Rebuild the loop position from durable tool-boundary records: the
            # conversation delta of every completed step, in order. The
            # environment restore already happened in run() before this call.
            boundary_records = select_resume_records(await session_store.read_boundaries(rollout_id))
            if not boundary_records:
                raise HTTPException(
                    status_code=409, detail=f"resume requested but no boundary records exist for rollout {rollout_id}"
                )
            for record in boundary_records:
                new_outputs.extend(_INPUT_ITEMS_ADAPTER.validate_python(record.output_items))
            last_record = boundary_records[-1]
            step = last_record.boundary_index
            if last_record.usage is not None:
                usage = NeMoGymResponseUsage.model_validate(last_record.usage)

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})
            if collect_trajectory:
                turn_timestamp = time()

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)
            step_items: list[Any] = list(output)
            if collect_trajectory:
                turn_model_calls = []
                if model_response.id:
                    model_call_ref = ModelCallRef(model_ref=self.config.model_server, response_id=model_response.id)
                    model_calls.append(model_call_ref)
                    turn_model_calls.append(model_call_ref)
                else:
                    trajectory_gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable", invocation_id=invocation_id, detail=f"turn:{step}"
                        )
                    )
                reasoning = [item.model_dump(mode="json") for item in output if item.type == "reasoning"] or None
                answer = [item for item in output if item.type != "reasoning"]
                turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation_id,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=step,
                        timestamp=turn_timestamp,
                        question=new_body.input,
                        answer=answer,
                        reasoning_content=reasoning,
                        step_count=len(tool_records),
                        model_calls=turn_model_calls,
                    )
                )

            usage = accumulate_response_usage(usage, model_response.usage)
            model_response.usage = None

            if model_response.incomplete_details:
                invocation_status = "incomplete"
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                if collect_trajectory:
                    started_at = time()
                    started_monotonic = perf_counter()
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})
                    if collect_trajectory:
                        error_type = type(e).__name__
                        tool_status = "failed"
                else:
                    # Resource-server errors are valid model-visible tool outputs.
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"{resources_prefix}/{output_function_call.name}",
                        json=parsed_arguments,
                        cookies=resources_server_cookies,
                    )
                    tool_output = (await api_response.content.read()).decode()
                    resources_server_cookies = api_response.cookies
                    if collect_trajectory:
                        completed = 200 <= api_response.status < 400
                        tool_status = "completed" if completed else "failed"
                        error_type = None if completed else f"http_{api_response.status}"

                if collect_trajectory:
                    tool_records.append(
                        TrajectoryToolCall(
                            invocation_id=invocation_id,
                            tool_call_id=output_function_call.call_id,
                            tool_name=output_function_call.name,
                            started_at=started_at,
                            completed_at=max(started_at, time()),
                            duration_ms=(perf_counter() - started_monotonic) * 1000,
                            timing_source="executor",
                            status=tool_status,
                            error_type=error_type,
                            output=tool_output,
                        )
                    )

                function_call_output = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(function_call_output)
                step_items.append(function_call_output)

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)

            if session_active and all_fn_calls:
                resources_server_cookies = await self._commit_tool_boundary(
                    session_store=session_store,
                    rollout_id=rollout_id,
                    resources_prefix=resources_prefix,
                    step=step,
                    step_items=step_items,
                    usage=usage,
                    resources_server_cookies=resources_server_cookies,
                    take_snapshot=step % max(1, self.config.session_state_snapshot_every_n_steps) == 0,
                    response_id=model_response.id,
                )

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                invocation_status = "incomplete"
                break

        model_response.output = new_outputs
        model_response.usage = usage
        trajectory = None
        if collect_trajectory:
            invocation = AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                model_calls=model_calls,
                conversation=[*body.input, *new_outputs],
            )
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=rollout_id,
                invocations=[invocation],
                turns=turns,
                tool_calls=tool_records,
                gaps=trajectory_gaps,
            )
        return model_response, trajectory, model_server_cookies, resources_server_cookies

    async def _commit_tool_boundary(
        self,
        *,
        session_store: FileSessionStateStore,
        rollout_id: str,
        resources_prefix: str,
        step: int,
        step_items: list[Any],
        usage: Optional[NeMoGymResponseUsage],
        resources_server_cookies: Any,
        take_snapshot: bool = True,
        response_id: Optional[str] = None,
    ) -> Any:
        """Make step ``step`` durable: environment state first, boundary record second.

        Small environment states come back inline from the export and ride in
        the boundary record itself — one append, one fsync, no snapshot file.
        Large states are written server-side as a snapshot file before the
        record is appended. The record is the commit point either way: if a
        stateful server's export fails, the record is NOT appended, so the
        store keeps its last consistent pair and a later resume re-enters
        there. With ``take_snapshot=False`` (snapshot cadence), the record is
        conversation-only and resume selects an earlier resumable boundary.
        """
        env_exported = False
        env_state = None
        if take_snapshot:
            try:
                export_raw = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"{resources_prefix}{SESSION_STATE_URL_PREFIX}/export",
                    json={"boundary_index": step, "inline_ok": True},
                    cookies=resources_server_cookies,
                )
                await raise_for_status(export_raw)
                export = SessionExportResponse.model_validate(await get_response_json(export_raw))
                resources_server_cookies = export_raw.cookies
            except Exception:
                # A failed export must not fail the live rollout; it only
                # limits how far back a future resume must go.
                return resources_server_cookies
            if export.supported and not export.exported:
                return resources_server_cookies
            env_state = export.state if export.inline else None
            env_exported = export.exported and not export.inline
        await session_store.append_boundary(
            ToolBoundaryRecord(
                rollout_id=rollout_id,
                boundary_index=step,
                output_items=[item.model_dump(mode="json") for item in step_items],
                usage=usage.model_dump(mode="json") if usage is not None else None,
                env_exported=env_exported,
                env_state=env_state,
                response_id=response_id,
                created_at=time(),
            )
        )
        return resources_server_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        session_store = None
        session_resume = False
        if self.config.session_state_dir and isinstance(rollout_id, str):
            session_store = FileSessionStateStore(self.config.session_state_dir)
            query_params = getattr(request, "query_params", None)
            session_resume = isinstance(query_params, Mapping) and query_params.get(_RESUME_QUERY_PARAM) == "1"
        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._create_episode(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=request.cookies,
            rollout_id=rollout_id or "unscoped",
            collect_trajectory=collect_trajectory,
            session_store=session_store,
            session_resume=session_resume,
        )
        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        if trajectory is not None:
            model_response = model_response.model_copy(
                update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies

        # Session checkpointing: keyed by rollout id (explicit _ng_rollout_id or
        # derived task/rollout indices), independent of capture correlation.
        session_rollout_id = maybe_rollout_id_from_run_body(body) if self.config.session_state_dir else None
        session_prefix = rollout_path_prefix(session_rollout_id) if session_rollout_id else ""
        resume = session_rollout_id is not None and bool((body.model_extra or {}).get(_NG_RESUME_KEY))

        if resume:
            # Restore the environment at the last durable tool boundary. The
            # rollout-prefixed path binds the resources server's session id to
            # the rollout id, so no saved cookie is needed.
            store = FileSessionStateStore(self.config.session_state_dir)
            resume_records = select_resume_records(await store.read_boundaries(session_rollout_id))
            if not resume_records:
                raise HTTPException(
                    status_code=409,
                    detail=f"resume requested but no boundary records exist for rollout {session_rollout_id}",
                )
            latest = resume_records[-1]
            restore_body: dict = {"boundary_index": latest.boundary_index}
            if latest.env_state is not None:
                restore_body["state"] = latest.env_state
            restore_raw = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"{session_prefix}{SESSION_STATE_URL_PREFIX}/restore",
                json=restore_body,
                cookies=cookies,
            )
            await raise_for_status(restore_raw)
            restore = SessionRestoreResponse.model_validate(await get_response_json(restore_raw))
            if restore.supported and not restore.restored:
                # A stateful server that cannot restore must fail the resume so
                # the caller falls back to abandon-and-redispatch; continuing
                # would pair the recovered conversation with a fresh environment.
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot restore session for rollout {session_rollout_id}: {restore.detail}",
                )
            cookies = restore_raw.cookies
        else:
            if session_rollout_id is not None:
                # Rerun hygiene: a fresh dispatch under an id that has stale
                # records (an abandoned attempt) must not resume by accident.
                await FileSessionStateStore(self.config.session_state_dir).clear_rollout(session_rollout_id)
            seed_session_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"{session_prefix}/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_session_response)
            cookies = seed_session_response.cookies

        run_path = self.url_path_for_run("/v1/responses", body)
        if session_rollout_id is not None and not run_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/"):
            run_path = f"{session_prefix}{run_path}"
        if resume:
            run_path = f"{run_path}?{_RESUME_QUERY_PARAM}=1"
        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=run_path,
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        model_response_json = await get_response_json(response)
        cookies = response.cookies

        trajectory = None
        expected_rollout_id = self.rollout_id_from_run(body)
        raw_trajectory = (
            model_response_json.pop(_INTERNAL_TRAJECTORY_KEY, None) if expected_rollout_id is not None else None
        )
        if isinstance(raw_trajectory, dict):
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            extra = body.model_extra or {}
            task_id = next(
                (
                    str(extra[key])
                    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
                    if extra.get(key) is not None
                ),
                "unknown",
            )
            rollout_id = expected_rollout_id or trajectory.rollout_id
            trajectory = trajectory.model_copy(
                update={
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "turns": [
                        turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id})
                        for turn in trajectory.turns
                    ],
                }
            )

        if self.config.skip_verification:
            result = body.model_dump() | {
                "response": model_response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify_request = SimpleAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": model_response_json}
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            result = await get_response_json(verify_response)
        if trajectory is not None:
            resolved = result.get("resolved")
            if isinstance(resolved, bool) and trajectory.turns:
                trajectory.turns[-1].resolved = resolved
            else:
                trajectory.gaps.append(ObservationGap(code="resolution_unavailable", invocation_id="root"))
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return SimpleAgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SimpleAgent.run_webserver()
