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
"""Throwaway: SimpleAgent H2 (model 409 kills run), H3 (resources 409 baked into boundary), H6 (max_steps off-by-one), resume leak."""

import asyncio
import importlib.util
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientResponseError, RequestInfo


ROOT = os.environ.get("GYM_STACK_ROOT", os.getcwd())
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "simple_tests", f"{ROOT}/responses_api_agents/simple_agent/tests/test_app.py"
)
simple_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simple_tests)

from nemo_gym.checkpoint.agent import AgentBoundaryRecord  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.rollout_correlation import rollout_context  # noqa: E402


def model_payload(output):
    return {
        "id": "resp",
        "created_at": 1.0,
        "model": "model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def fn_call(call_id="c1"):
    return {"type": "function_call", "call_id": call_id, "name": "tool", "arguments": "{}", "id": f"fc-{call_id}"}


def msg(text):
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def mock_409(code):
    body = json.dumps({"error": {"code": code, "detail": "closed"}})
    response = MagicMock(status=409, cookies={}, ok=False, headers={})
    response.read = AsyncMock(return_value=body)
    response.content.read = AsyncMock(return_value=body.encode())

    def _raise():
        raise ClientResponseError(
            RequestInfo(url="http://x", method="POST", headers={}, real_url="http://x"), (), status=409
        )

    response.raise_for_status = _raise
    return response


async def h3_resources_409_baked_into_boundary():
    agent, client = simple_tests._make_agent(False)
    participant = agent.checkpoint_participant()
    calls = iter(
        [
            simple_tests._mock_response(model_payload([fn_call("c1")])),
            mock_409("resources_admission_closed"),  # tool call during resources prepare
            simple_tests._mock_response(model_payload([msg("done")])),
        ]
    )

    async def post(*, url_path, **kwargs):
        return next(calls)

    client.post = AsyncMock(side_effect=post)
    with rollout_context("0-0", attempt_index=0, logical_rollout_id="0-0"):
        ex = await participant.begin("0-0", 0, task=asyncio.current_task())
        body = NeMoGymResponseCreateParamsNonStreaming(input="q")
        result, *_ = await agent._create_episode(body, model_url_path="/v1/responses")
        await participant.finish(ex, outcome="completed")
    print("H3 boundary output_items include tool output:", ex.boundary.output_items[-1]["output"])
    print("H3 episode completed normally, final output count:", len(result.output))


async def h2_model_409_fails_run():
    agent, client = simple_tests._make_agent(False)
    participant = agent.checkpoint_participant()
    calls = iter(
        [
            simple_tests._mock_response(model_payload([fn_call("c1")])),
            simple_tests._mock_response({}, content="tool-out"),
            mock_409("checkpoint_parked"),  # model paused before agent prepare
        ]
    )

    async def post(*, url_path, **kwargs):
        return next(calls)

    client.post = AsyncMock(side_effect=post)
    with rollout_context("0-1", attempt_index=0, logical_rollout_id="0-1"):
        ex = await participant.begin("0-1", 0, task=asyncio.current_task())
        body = NeMoGymResponseCreateParamsNonStreaming(input="q")
        try:
            await agent._create_episode(body, model_url_path="/v1/responses")
        except ClientResponseError as e:
            print("H2 model 409 escaped _create_episode as", type(e).__name__, e.status)
            await participant.finish(ex, outcome="failed")  # what the /run wrapper does
    print(
        "H2 execution state:",
        ex.state,
        "boundary_index:",
        ex.boundary.boundary_index,
        "records_for_commit:",
        len(participant.records_for_commit()),
    )


async def h6_continuation_at_max_steps_runs_extra_turn():
    agent, client = simple_tests._make_agent(False)
    agent.config.max_steps = 2
    participant = agent.checkpoint_participant()
    cont = AgentBoundaryRecord(
        rollout_id="0-2",
        attempt_index=0,
        boundary_index=2,
        output_items=[fn_call("c1"), {"type": "function_call_output", "call_id": "c1", "output": "x"}],
    )
    participant.install_restored([cont])
    await participant.resume()
    calls = iter(
        [
            simple_tests._mock_response(model_payload([fn_call("c2")])),
            simple_tests._mock_response({}, content="tool-out-2"),
            simple_tests._mock_response(model_payload([fn_call("c3")])),
            simple_tests._mock_response({}, content="tool-out-3"),
        ]
    )
    n = {"model": 0}

    async def post(*, url_path, **kwargs):
        if url_path.endswith("/v1/responses"):
            n["model"] += 1
        return next(calls)

    client.post = AsyncMock(side_effect=post)
    with rollout_context("0-2-a1", attempt_index=1, logical_rollout_id="0-2"):
        ex = await participant.begin("0-2", 1, task=asyncio.current_task())
        body = NeMoGymResponseCreateParamsNonStreaming(input="q")
        result, *_ = await agent._create_episode(
            body, model_url_path="/v1/responses", continuation=participant.continuation("0-2", 1)
        )
        await participant.finish(ex, outcome="completed")
    print(
        "H6 continuation with boundary_index == max_steps made",
        n["model"],
        "model calls (expected 0); boundary now",
        ex.boundary.boundary_index,
    )


async def resume_leaks_inner_handler_when_run_task_gone():
    participant_agent, _ = simple_tests._make_agent(False)
    p = participant_agent.checkpoint_participant()
    ex = await p.begin("0-3", 0, task=None)
    prep = asyncio.create_task(p.prepare(time.time() + 2))
    await asyncio.sleep(0)
    inner = asyncio.create_task(
        p.commit_boundary(AgentBoundaryRecord(rollout_id="0-3", attempt_index=0, boundary_index=1, output_items=[]))
    )
    await prep
    await p.finish(ex, outcome="cancelled")  # outer /run task cancelled while parked -> task=None
    await p.resume()
    await asyncio.sleep(0.05)
    print("resume(): execution state", ex.state, "; inner /v1/responses handler still parked:", not inner.done())
    inner.cancel()
    try:
        await inner
    except asyncio.CancelledError:
        pass


async def main():
    await h3_resources_409_baked_into_boundary()
    await h2_model_409_fails_run()
    await h6_continuation_at_max_steps_runs_extra_turn()
    await resume_leaks_inner_handler_when_run_task_gone()


asyncio.run(main())
