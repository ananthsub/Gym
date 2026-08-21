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
"""Partial-rollout checkpointing on a real stateful pair: gymnasium_agent + blackjack.

Blackjack's per-session state includes a live unseeded ``random.Random`` — the
upcoming cards. Replay cannot recover that (a fresh RNG deals a fresh shoe), so
this is exactly the environment class that needs explicit export/restore. The
test kills an episode after one committed step, resumes it in fresh server
instances, and proves the continuation is *bit-exact*: the resumed dealer draws
the precise cards the durable RNG snapshot predicts.
"""

import asyncio
import json
import random
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.server_utils import ServerClient
from nemo_gym.session_state import FileSessionStateStore
from resources_servers.blackjack.app import _RANKS, BlackjackEnv, _fmt, _hand_value
from responses_api_agents.gymnasium_agent.app import (
    GymnasiumAgent,
    GymnasiumAgentConfig,
    GymnasiumAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
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


def _action_payload(step: int, action: str) -> dict:
    return {
        "id": f"resp_{step}",
        "created_at": 0.0,
        "model": "scripted",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": f"msg_{step}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": f"<action>{action}</action>", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": _usage(),
    }


class _FakeHttpxAiohttpResponse:
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
    """One 'process generation': fresh agent + blackjack server over a shared session-state dir."""

    def __init__(self, session_state_dir: str, model_script: list) -> None:
        self.model_script = list(model_script)
        self.model_inputs: list[dict] = []

        env_config = BaseResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="blackjack", session_state_dir=session_state_dir
        )
        self.env_server = BlackjackEnv(config=env_config, server_client=MagicMock(spec=ServerClient))
        self.env_client = TestClient(self.env_server.setup_webserver())

        agent_config = GymnasiumAgentConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="blackjack_agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            resources_server=ResourcesServerRef(type="resources_servers", name="blackjack"),
            max_steps=5,
            session_state_dir=session_state_dir,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client.post = self._post
        self.agent = GymnasiumAgent(config=agent_config, server_client=server_client)

    async def _post(self, server_name: str, url_path: str, json: Any = None, cookies: Any = None, **kwargs: Any):
        if isinstance(json, BaseModel):
            json = json.model_dump(exclude_unset=True)
        if server_name == "policy":
            self.model_inputs.append(json)
            step = self.model_script.pop(0)
            if step is _KILL:
                raise RuntimeError("simulated trainer restart: model connection lost")
            return _FakeModelResponse(step)
        response = await asyncio.to_thread(self.env_client.post, url_path, json=json, cookies=dict(cookies or {}))
        return _FakeHttpxAiohttpResponse(response)

    async def run(self, body: dict):
        request = MagicMock(cookies={})
        return await self.agent.run(request, GymnasiumAgentRunRequest.model_validate(body))


_RUN_BODY = {
    "responses_create_params": {"input": [{"role": "user", "content": "Play blackjack."}]},
    "_ng_rollout_id": "bj-1",
}


def _pick_seed() -> int:
    # A seed whose first hit does not bust, so phase 1 reaches a committed
    # non-terminal boundary. Consumption order mirrors reset()+step("hit").
    for seed in range(1000):
        rng = random.Random(seed)
        player = [rng.choice(_RANKS), rng.choice(_RANKS)]
        _dealer = [rng.choice(_RANKS), rng.choice(_RANKS)]
        if _hand_value(player + [rng.choice(_RANKS)]) <= 21:
            return seed
    raise AssertionError("no suitable seed found")


def _simulate_stand(snapshot_state: dict) -> dict:
    """Replicate step('stand') from a durable snapshot: the exact expected outcome."""
    player = list(snapshot_state["player"])
    dealer = list(snapshot_state["dealer"])
    rng = random.Random()
    rng_state = snapshot_state["rng_state"]
    rng.setstate((rng_state["version"], tuple(rng_state["internal"]), rng_state["gauss"]))
    while _hand_value(dealer) < 17:
        dealer.append(rng.choice(_RANKS))
    pv, dv = _hand_value(player), _hand_value(dealer)
    if dv > 21 or pv > dv:
        reward, result = 1.0, "win"
    elif pv == dv:
        reward, result = 0.0, "draw"
    else:
        reward, result = -1.0, "loss"
    return {
        "reward": reward,
        "info": {
            "result": result,
            "player": _fmt(player),
            "player_value": pv,
            "dealer": _fmt(dealer),
            "dealer_value": dv,
        },
    }


class TestBlackjackSessionCheckpointing:
    async def test_kill_and_resume_deals_bit_identical_cards(self, tmp_path, monkeypatch) -> None:
        seed = _pick_seed()

        class _SeededRandom(random.Random):
            def __init__(self) -> None:
                super().__init__(seed)

        # ---- Phase 1: seeded deal, one committed hit, then the process dies. ----
        first = _Harness(str(tmp_path), model_script=[_action_payload(1, "hit"), _KILL])
        with monkeypatch.context() as patched:
            patched.setattr(random, "Random", _SeededRandom)
            with pytest.raises(RuntimeError, match="simulated trainer restart"):
                await first.run(_RUN_BODY)

        store = FileSessionStateStore(tmp_path)
        boundaries = await store.read_boundaries("bj-1")
        assert [record.boundary_index for record in boundaries] == [0, 1]
        assert all(record.resumable for record in boundaries)
        reset_observation = (boundaries[0].model_extra or {})["reset_observation"]
        assert reset_observation.startswith("Your hand:")
        # The hand state is small, so it rides inline in the boundary record.
        durable_state = boundaries[1].env_state
        assert durable_state is not None and len(durable_state["player"]) == 3

        # The durable RNG state predicts the exact remainder of the episode.
        expected = _simulate_stand(durable_state)

        # ---- Phase 2: fresh processes (empty memory), same store, stand. ----
        second = _Harness(str(tmp_path), model_script=[_action_payload(2, "stand")])
        assert second.env_server.session_state == {}

        result = await second.run(_RUN_BODY | {"_ng_resume": True})

        assert result.terminated is True
        assert result.reward == expected["reward"]
        assert result.info == expected["info"]

        # Exactly one model call on resume, conditioned on the replayed
        # conversation: prompt, reset observation, the hit, its observation.
        assert len(second.model_inputs) == 1
        resumed_input = second.model_inputs[0]["input"]
        assert resumed_input[0]["content"] == "Play blackjack."
        assert resumed_input[1]["content"] == reset_observation
        assert "<action>hit</action>" in json.dumps(resumed_input[2])
        assert resumed_input[3]["content"].startswith("Your hand:")

        # Usage survived the kill: one call before, one after.
        assert result.response.usage.total_tokens == 30
        # Terminal step closed the restored session.
        assert second.env_server.session_state == {}

    async def test_fresh_dispatch_clears_stale_records(self, tmp_path, monkeypatch) -> None:
        seed = _pick_seed()

        class _SeededRandom(random.Random):
            def __init__(self) -> None:
                super().__init__(seed)

        first = _Harness(str(tmp_path), model_script=[_action_payload(1, "hit"), _KILL])
        with monkeypatch.context() as patched:
            patched.setattr(random, "Random", _SeededRandom)
            with pytest.raises(RuntimeError):
                await first.run(_RUN_BODY)
        assert await FileSessionStateStore(tmp_path).read_boundaries("bj-1") != []

        second = _Harness(str(tmp_path), model_script=[_action_payload(1, "stand")])
        result = await second.run(_RUN_BODY)
        assert result.terminated is True
        # A fresh (non-resume) dispatch re-reset from scratch: two-card hand.
        assert "[" in result.info["player"] and len(result.info["player"].split(",")) == 2

    async def test_resume_without_records_fails_loudly(self, tmp_path) -> None:
        harness = _Harness(str(tmp_path), model_script=[])
        with pytest.raises(Exception, match="no boundary-0 record"):
            await harness.run(_RUN_BODY | {"_ng_resume": True})
