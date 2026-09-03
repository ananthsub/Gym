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
"""Throwaway: GymnasiumAgent boundary at max_steps -> restart continuation crash (H7)."""

import asyncio
import importlib.util
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock


ROOT = os.environ.get("GYM_STACK_ROOT", os.getcwd())
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "gym_tests", f"{ROOT}/responses_api_agents/gymnasium_agent/tests/test_app.py"
)
gym_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gym_tests)

from nemo_gym.checkpoint.agent import AgentBoundaryRecord, commit_agent_state, restore_agent_state  # noqa: E402
from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME  # noqa: E402
from nemo_gym.rollout_correlation import rollout_context  # noqa: E402
from responses_api_agents.gymnasium_agent.app import GymnasiumAgentRunRequest  # noqa: E402


async def produce_boundary_at_max_steps(tmp):
    agent = gym_tests._make_agent(max_steps=2)
    participant = agent.checkpoint_participant()
    payloads = {
        "/reset": [{"observation": "go", "info": {}}],
        "/ng-rollout/2-0/v1/responses": [gym_tests._model_response("a"), gym_tests._model_response("b")],
        "/step": [
            {"observation": "obs-1", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
            {"observation": "obs-2", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
        ],
    }
    at_step2 = asyncio.Event()
    proceed = asyncio.Event()
    model_calls = {"n": 0}

    async def _post(server_name, url_path, json=None, cookies=None, **kw):
        if url_path.endswith("/v1/responses"):
            model_calls["n"] += 1
            if model_calls["n"] == 2:
                at_step2.set()
                await proceed.wait()
        await asyncio.sleep(0)
        return gym_tests._FakeHttpResp(payloads[url_path].pop(0))

    agent.server_client.post = AsyncMock(side_effect=_post)
    req = MagicMock()
    req.cookies = {}
    body = GymnasiumAgentRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "x"}]},
        **{TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0},
    )

    async def run():
        with rollout_context("2-0", attempt_index=0, logical_rollout_id="2-0"):
            ex = await participant.begin("2-0", 0, task=asyncio.current_task())
            result = await agent.run(req, body)
            await participant.finish(ex, outcome="completed")
            return result

    task = asyncio.create_task(run())
    await at_step2.wait()  # step-1 boundary committed while RUNNING; model call for step 2 (== max_steps) in flight
    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    proceed.set()
    report = await prepare
    print("H7 prepare report:", report["executions"])
    summary = commit_agent_state(participant, tmp, checkpoint_id="c1")
    print("H7 committed records:", summary["records"])
    await participant.resume()
    result = await task
    print("H7 same-process resume result: truncated =", result.truncated)
    return agent


async def continue_after_restart(tmp):
    fresh = gym_tests._make_agent(max_steps=2)
    restore_agent_state(fresh.checkpoint_participant(), tmp)
    cont = fresh.checkpoint_participant().continuation("2-0", 1)
    print("H7 restored continuation boundary_index:", cont.boundary_index, "max_steps:", fresh.config.max_steps)
    await fresh.checkpoint_participant().resume()
    gym_tests._wire_mock_client(
        fresh, {"/ng-rollout/2-0-a1/v1/responses": [gym_tests._model_response("c")], "/step": []}
    )
    req = MagicMock()
    req.cookies = {}
    body = GymnasiumAgentRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "x"}]},
        **{TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, ATTEMPT_INDEX_KEY_NAME: 1},
    )
    with rollout_context("2-0-a1", attempt_index=1, logical_rollout_id="2-0"):
        try:
            result = await fresh.run(req, body)
            print("H7 continuation result:", result.truncated)
        except Exception as e:  # noqa: BLE001
            print("H7 continuation CRASH:", type(e).__name__, e)


async def off_by_one_when_not_at_max(tmp):
    """Boundary at step k < max_steps: does the continuation replay step k or start at k+1?"""
    fresh = gym_tests._make_agent(max_steps=3)
    cont = AgentBoundaryRecord(
        rollout_id="9-0",
        attempt_index=0,
        boundary_index=1,
        output_items=gym_tests._model_response("a")["output"]
        + [{"role": "user", "content": "obs-1", "type": "message"}],
        agent_state={
            "reset_data": {"observation": "go", "info": {}},
            "step_data": {"observation": "obs-1", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
            "total_reward": 0.0,
            "env_cookies": {},
        },
    )
    fresh.checkpoint_participant().install_restored([cont])
    await fresh.checkpoint_participant().resume()
    log = gym_tests._wire_mock_client(
        fresh,
        {
            "/ng-rollout/9-0-a1/v1/responses": [gym_tests._model_response("b"), gym_tests._model_response("c")],
            "/step": [
                {"observation": "obs-2", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                {"observation": "obs-3", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
            ],
        },
    )
    req = MagicMock()
    req.cookies = {}
    body = GymnasiumAgentRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "x"}]},
        **{TASK_INDEX_KEY_NAME: 9, ROLLOUT_INDEX_KEY_NAME: 0, ATTEMPT_INDEX_KEY_NAME: 1},
    )
    with rollout_context("9-0-a1", attempt_index=1, logical_rollout_id="9-0"):
        ex = await fresh.checkpoint_participant().begin("9-0", 1, task=asyncio.current_task())
        result = await fresh.run(req, body)
        await fresh.checkpoint_participant().finish(ex, outcome="completed")
    steps = [u for _s, u, _j in log if u == "/step"]
    print(
        "H7 continuation from boundary 1 with max_steps 3 executed",
        len(steps),
        "more steps (expect 2); truncated:",
        result.truncated,
    )


async def main():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        await produce_boundary_at_max_steps(Path(d))
        await continue_after_restart(Path(d))
    with tempfile.TemporaryDirectory() as d:
        await off_by_one_when_not_at_max(Path(d))


asyncio.run(main())
