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
"""End-to-end partial-rollout checkpointing through the real server stack.

Drives a real SimpleAgent app and a real StatefulCounterResourcesServer app
(with their middleware: rollout-prefix stripping, session binding, session
routes) over a scripted model. The rollout is killed mid-flight after two tool
steps; fresh server instances (empty memory, same session-state dir) then
resume it. The resumed model call must be the continuation call only — no
re-generation, no tool re-execution — and verification must see the restored
environment.
"""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient
from nemo_gym.session_state import FileSessionStateStore
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
)


_KILL = object()


def _usage() -> dict:
    return {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def _model_payload(step: int, output: list[dict]) -> dict:
    return {
        "id": f"resp_{step}",
        "created_at": 0.0,
        "model": "scripted",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": _usage(),
    }


def _increment_call(step: int, count: int) -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{step}",
        "call_id": f"call_{step}",
        "name": "increment_counter",
        "arguments": json.dumps({"count": count}),
        "status": "completed",
    }


def _final_message(step: int) -> dict:
    return {
        "type": "message",
        "id": f"msg_{step}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done", "annotations": []}],
    }


class _FakeHttpxAiohttpResponse:
    """Adapt an httpx TestClient response to the aiohttp surface the agent uses."""

    def __init__(self, response) -> None:
        self._response = response
        self.status = response.status_code
        self.ok = response.status_code < 400
        self.cookies = dict(response.cookies)
        self.content = self

    async def read(self) -> bytes:
        return self._response.content

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


class _FakeModelResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status = 200
        self.ok = True
        self.cookies: dict = {}
        self.content = self

    async def read(self) -> bytes:
        return orjson.dumps(self._payload)

    def raise_for_status(self) -> None:
        pass


class _Harness:
    """One 'process generation': fresh agent + resources server over a shared session-state dir."""

    def __init__(self, session_state_dir: str, model_script: list, snapshot_every_n_steps: int = 1) -> None:
        self.model_script = list(model_script)
        self.model_inputs: list[dict] = []

        resources_config = StatefulCounterResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="counter_server", session_state_dir=session_state_dir
        )
        self.resources_server = StatefulCounterResourcesServer(
            config=resources_config, server_client=MagicMock(spec=ServerClient)
        )
        self.resources_client = TestClient(self.resources_server.setup_webserver())

        agent_config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="session_agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            resources_server=ResourcesServerRef(type="resources_servers", name="counter_server"),
            session_state_dir=session_state_dir,
            session_state_snapshot_every_n_steps=snapshot_every_n_steps,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client.post = self._post
        self.agent = SimpleAgent(config=agent_config, server_client=server_client)
        self.agent_client = TestClient(self.agent.setup_webserver())

    async def _post(self, server_name: str, url_path: str, json: Any = None, cookies: Any = None, **kwargs: Any):
        if isinstance(json, BaseModel):
            # Mirror ServerClient.request's BaseModel handling.
            json = json.model_dump(exclude_unset=True)
        if server_name == "policy":
            self.model_inputs.append(json)
            step = self.model_script.pop(0)
            if step is _KILL:
                raise RuntimeError("simulated trainer restart: model connection lost")
            return _FakeModelResponse(step if isinstance(step, dict) else step(json))
        client = self.resources_client if server_name == "counter_server" else self.agent_client
        response = await asyncio.to_thread(client.post, url_path, json=json, cookies=dict(cookies or {}))
        return _FakeHttpxAiohttpResponse(response)

    async def run(self, body: dict):
        request = MagicMock(cookies={})
        return await self.agent.run(request, SimpleAgentRunRequest.model_validate(body))


_RUN_BODY = {
    "responses_create_params": {"input": [{"role": "user", "content": "increment the counter twice"}]},
    "_ng_rollout_id": "e2e-1",
    "initial_count": 5,
    "expected_count": 10,
}


class TestSessionCheckpointing:
    async def test_kill_and_resume_continues_without_reexecution(self, tmp_path) -> None:
        # ---- Phase 1: live rollout dies after two committed tool steps. ----
        first = _Harness(
            str(tmp_path),
            model_script=[
                _model_payload(1, [_increment_call(1, 2)]),
                _model_payload(2, [_increment_call(2, 3)]),
                _KILL,
            ],
        )
        with pytest.raises(RuntimeError, match="simulated trainer restart"):
            await first.run(_RUN_BODY)

        # Both boundaries are durable: conversation deltas and env state.
        # The counter is tiny, so its state rides inline in the record —
        # no snapshot file is ever written for it.
        store = FileSessionStateStore(tmp_path)
        boundaries = await store.read_boundaries("e2e-1")
        assert [record.boundary_index for record in boundaries] == [1, 2]
        assert all(record.resumable for record in boundaries)
        assert boundaries[-1].env_state == {"counter": 10}
        assert (await store.read_snapshot("e2e-1", "counter_server", 2)) is None
        assert boundaries[-1].response_id == "resp_2"

        # ---- Phase 2: fresh processes (empty memory), same store. ----
        second = _Harness(str(tmp_path), model_script=[_model_payload(3, [_final_message(3)])])
        assert second.resources_server.session_id_to_counter == {}

        result = await second.run(_RUN_BODY | {"_ng_resume": True})

        # Verification ran against the restored environment.
        assert result.reward == 1.0
        assert second.resources_server.session_id_to_counter == {"e2e-1": 10}

        # Exactly one model call on resume: the continuation, never a replay.
        assert len(second.model_inputs) == 1
        resumed_input = second.model_inputs[0]["input"]
        # Original user message from the redispatched row, then both replayed steps.
        assert resumed_input[0].get("role") == "user"
        assert [item.get("type") for item in resumed_input[1:]] == [
            "function_call",
            "function_call_output",
            "function_call",
            "function_call_output",
        ]
        assert [item.get("call_id") for item in resumed_input[1:]] == ["call_1", "call_1", "call_2", "call_2"]

        # No re-seeding and no tool re-execution happened on resume.
        response = result.response
        final_output = response["output"] if isinstance(response, dict) else response.output
        assert len(final_output) == 5  # 2 steps x (call + output) + final message
        usage = response["usage"] if isinstance(response, dict) else response.usage
        usage = usage if isinstance(usage, dict) else usage.model_dump()
        # 2 pre-kill calls + 1 resumed call, restored from the boundary record.
        assert usage["total_tokens"] == 45

    async def test_fresh_dispatch_clears_stale_records(self, tmp_path) -> None:
        # An abandoned attempt leaves records; a fresh (non-resume) dispatch
        # under the same id must start clean, not resume by accident.
        first = _Harness(str(tmp_path), model_script=[_model_payload(1, [_increment_call(1, 2)]), _KILL])
        with pytest.raises(RuntimeError):
            await first.run(_RUN_BODY)
        assert await FileSessionStateStore(tmp_path).read_boundaries("e2e-1") != []

        second = _Harness(
            str(tmp_path),
            model_script=[
                _model_payload(1, [_increment_call(1, 2)]),
                _model_payload(2, [_increment_call(2, 3)]),
                _model_payload(3, [_final_message(3)]),
            ],
        )
        result = await second.run(_RUN_BODY)
        assert result.reward == 1.0
        # The fresh run re-seeded and re-ran from scratch: three model calls.
        assert len(second.model_inputs) == 3

    async def test_snapshot_cadence_resumes_at_latest_resumable_boundary(self, tmp_path) -> None:
        # With snapshots every 2 steps, boundary 1 is conversation-only and
        # boundary 2 carries state. Killed at step 3, the rollout resumes at
        # boundary 2: the environment rewinds past step 3's orphaned effects
        # (env had 5+2+3+4=14 live; snapshot says 10) and step 3 regenerates.
        first = _Harness(
            str(tmp_path),
            model_script=[
                _model_payload(1, [_increment_call(1, 2)]),
                _model_payload(2, [_increment_call(2, 3)]),
                _model_payload(3, [_increment_call(3, 4)]),
                _KILL,
            ],
            snapshot_every_n_steps=2,
        )
        with pytest.raises(RuntimeError):
            await first.run(_RUN_BODY | {"expected_count": 14})

        store = FileSessionStateStore(tmp_path)
        boundaries = await store.read_boundaries("e2e-1")
        assert [(record.boundary_index, record.resumable) for record in boundaries] == [
            (1, False),
            (2, True),
            (3, False),
        ]

        second = _Harness(
            str(tmp_path),
            model_script=[_model_payload(4, [_increment_call(4, 4)]), _model_payload(5, [_final_message(5)])],
            snapshot_every_n_steps=2,
        )
        result = await second.run(_RUN_BODY | {"expected_count": 14, "_ng_resume": True})
        assert result.reward == 1.0
        # Two model calls on resume: the regenerated step 3 and the final
        # message; the regenerated step's prompt holds exactly steps 1-2.
        assert len(second.model_inputs) == 2
        first_resumed_input = second.model_inputs[0]["input"]
        assert [item.get("call_id") for item in first_resumed_input[1:]] == ["call_1", "call_1", "call_2", "call_2"]

    async def test_resume_without_records_fails_loudly(self, tmp_path) -> None:
        harness = _Harness(str(tmp_path), model_script=[])
        with pytest.raises(Exception, match="no boundary records"):
            await harness.run(_RUN_BODY | {"_ng_resume": True})
