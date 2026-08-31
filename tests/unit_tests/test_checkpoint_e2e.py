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
"""End-to-end Milestone 1 checkpoint simulation, fully on CPU.

One test walks the whole sequence the NemoGym actor drives in production,
against real Gym servers wired through real HTTP apps: run rollouts through
an agent that calls a policy model server with token capture on, pause the
policy admission, sacrifice the rollout that cannot finish before the
deadline, commit the capture ledger, restart onto fresh server processes,
restore, resume, and dispatch the replacement attempt.

The Milestone 1 exit criterion is asserted directly: every completed
rollout's token rows survive the restart byte-identical, the sacrificed
attempt's partial rows are excluded from the checkpoint and its identity
stays fenced after restore, and the replacement attempt runs and captures
fresh rows under its own identity.
"""

from urllib.parse import urlsplit

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict
from starlette.testclient import TestClient

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.checkpoint import MODEL_ADMISSION_URL_PREFIX, MODEL_CHECKPOINT_URL_PREFIX
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.rollout_correlation import ROLLOUT_ID_HEADER
from nemo_gym.server_utils import ServerClient, get_response_json


def _policy_response(seed: int) -> dict:
    return {
        "id": f"resp-{seed}",
        "created_at": 0.0,
        "model": "sim-policy",
        "object": "response",
        "output": [
            {
                "id": f"msg-{seed}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": f"answer-{seed}", "annotations": []}],
                # The simulated backend reports exact token metadata the way
                # vLLM does; the real capture pipeline stores it.
                "prompt_token_ids": [1, 2, 3, seed],
                "generation_token_ids": [100 + seed, 101 + seed],
                "generation_log_probs": [-0.25, -0.5],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class _PolicyModel(SimpleResponsesAPIModel):
    async def chat_completions(self, request):
        raise NotImplementedError

    async def responses(self, body) -> dict:
        seed = len(str(body.input)) % 97
        return _policy_response(seed)


class _Resources(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


class _RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class _Agent(SimpleResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body: _RunRequest) -> BaseVerifyResponse:
        policy = await self.server_client.post(
            server_name="policy",
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
        )
        response = orjson.loads(await policy.read())
        verify = await self.server_client.post(
            server_name="resources",
            url_path="/verify",
            json={
                "responses_create_params": body.responses_create_params.model_dump(),
                "response": response,
            },
        )
        return BaseVerifyResponse.model_validate(orjson.loads(await verify.read()))


class _PartialAgent(_Agent):
    """A rollout that stops after its first generation: unfinished at prepare."""

    async def run(self, body: _RunRequest) -> dict:  # type: ignore[override]
        policy = await self.server_client.post(
            server_name="policy",
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
        )
        await get_response_json(policy)
        return {"status": "interrupted before verification"}


class _Response:
    def __init__(self, response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self.cookies = response.cookies
        self._content = response.content

    async def read(self) -> bytes:
        return self._content


class _Stack:
    """One 'process generation' of the Gym stack: fresh servers, one store dir."""

    def __init__(self, tmp_path, store_subdir: str) -> None:
        self.store_dir = tmp_path / store_subdir
        config = OmegaConf.create(
            {
                "token_id_capture": {"enabled": True, "all_agents": True, "dir": str(self.store_dir)},
                "policy": {"responses_api_models": {"model": {"host": "policy.test", "port": 80}}},
                "resources": {"resources_servers": {"r": {"host": "resources.test", "port": 80}}},
                "agent": {"responses_api_agents": {"agent": {"host": "agent.test", "port": 80}}},
                "partial_agent": {"responses_api_agents": {"agent": {"host": "partial-agent.test", "port": 80}}},
            }
        )
        self.server_client = ServerClient(
            head_server_config=BaseServerConfig(host="head.test", port=80),
            global_config_dict=config,
        )
        self.policy = _PolicyModel(
            config=BaseResponsesAPIModelConfig(
                host="policy.test", port=80, entrypoint="", name="policy", instance_role="policy"
            ),
            server_client=self.server_client,
        )
        resources = _Resources(
            config=BaseResourcesServerConfig(host="resources.test", port=80, entrypoint="", name="resources"),
            server_client=self.server_client,
        )
        agent = _Agent(
            config=BaseResponsesAPIAgentConfig(host="agent.test", port=80, entrypoint="", name="agent"),
            server_client=self.server_client,
        )
        partial_agent = _PartialAgent(
            config=BaseResponsesAPIAgentConfig(host="partial-agent.test", port=80, entrypoint="", name="partial"),
            server_client=self.server_client,
        )
        self.policy_client = TestClient(self.policy.setup_webserver())
        self.clients = {
            "policy.test": self.policy_client,
            "resources.test": TestClient(resources.setup_webserver()),
            "agent.test": TestClient(agent.setup_webserver()),
            "partial-agent.test": TestClient(partial_agent.setup_webserver()),
        }

    async def dispatch(self, method: str, url: str, **kwargs) -> _Response:
        parsed = urlsplit(url)
        response = self.clients[parsed.hostname].request(
            method, parsed.path, json=kwargs.get("json"), headers=kwargs.get("headers")
        )
        return _Response(response)

    async def run_rollout(self, task: int, rollout: int, *, attempt: int = 0, agent: str = "agent") -> dict:
        result = await self.server_client.post(
            server_name=agent,
            url_path="/run",
            json={
                "_ng_task_index": task,
                "_ng_rollout_index": rollout,
                "_ng_attempt_index": attempt,
                "responses_create_params": {"input": f"solve {task}-{rollout} attempt {attempt}"},
            },
        )
        assert result.status == 200
        return orjson.loads(await result.read())

    def rows(self, rollout_id: str) -> bytes:
        path = self.store_dir / f"{rollout_id}.tokens.jsonl"
        return path.read_bytes() if path.exists() else b""


@pytest.mark.asyncio
async def test_milestone_1_checkpoint_restart_restore_cycle(tmp_path, monkeypatch) -> None:
    import nemo_gym.server_utils

    # ---- runtime: rollouts run and stage exact token rows ----
    stack_a = _Stack(tmp_path, "store-a")
    monkeypatch.setattr(nemo_gym.server_utils, "request", stack_a.dispatch)

    for task, rollout in ((0, 0), (0, 1), (1, 0)):
        result = await stack_a.run_rollout(task, rollout)
        assert result["reward"] == 1.0

    # Rollout 2-0 completed one generation but never reached verification:
    # it is the unfinished rollout at the checkpoint boundary.
    await stack_a.run_rollout(2, 0, agent="partial_agent")

    completed_rows = {rid: stack_a.rows(rid) for rid in ("0-0", "0-1", "1-0")}
    assert all(completed_rows.values())
    assert stack_a.rows("2-0")  # partial rows exist and must NOT survive

    # ---- prepare: close policy admission, drain, sacrifice the straggler ----
    pause = stack_a.policy_client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9}
    )
    assert pause.status_code == 200
    assert pause.json()["state"] == "paused"

    # New generation traffic parks instead of erroring.
    parked = stack_a.policy_client.post("/v1/responses", json={"input": "late"}, headers={ROLLOUT_ID_HEADER: "9-9"})
    assert parked.status_code == 409
    assert parked.json()["error"]["code"] == "checkpoint_parked"

    # deadline_policy restart_safe_rollouts: 2-0 cannot finish, so it is
    # fenced and recorded for fresh dispatch instead of aborting the checkpoint.
    abort = stack_a.policy_client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
        json={"checkpoint_id": "ckpt-1", "rollout_id": "2-0", "attempt_index": 0},
    )
    assert abort.status_code == 200

    # ---- commit: capture ledger into the checkpoint directory ----
    checkpoint_dir = tmp_path / "ckpt-1"
    commit = stack_a.policy_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={"checkpoint_id": "ckpt-1", "checkpoint_dir": str(checkpoint_dir)},
    )
    assert commit.status_code == 200
    assert commit.json()["rollouts"] == 3
    assert commit.json()["excluded_tombstoned"] == 1

    # ---- restart: brand-new server processes, empty store ----
    stack_b = _Stack(tmp_path, "store-b")
    monkeypatch.setattr(nemo_gym.server_utils, "request", stack_b.dispatch)

    restore = stack_b.policy_client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "ckpt-2", "checkpoint_dir": str(checkpoint_dir)},
    )
    assert restore.status_code == 200
    assert restore.json()["rollouts"] == 3

    # The restored server boots paused: nothing serves before resume.
    parked = stack_b.policy_client.post("/v1/responses", json={"input": "early"}, headers={ROLLOUT_ID_HEADER: "0-0"})
    assert parked.status_code == 409

    resume = stack_b.policy_client.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "ckpt-2"})
    assert resume.status_code == 200

    # ---- exit criterion: completed rollouts survived byte-identical ----
    for rollout_id, rows in completed_rows.items():
        assert stack_b.rows(rollout_id) == rows

    # The sacrificed attempt's partial rows did not survive, and its identity
    # stays fenced: a late call from the abandoned attempt is rejected.
    assert stack_b.rows("2-0") == b""
    stale = stack_b.policy_client.post(
        "/v1/responses", json={"input": "late write"}, headers={ROLLOUT_ID_HEADER: "2-0"}
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_attempt"

    # ---- fresh dispatch: the replacement attempt runs under its own identity ----
    result = await stack_b.run_rollout(2, 0, attempt=1)
    assert result["reward"] == 1.0
    assert stack_b.rows("2-0-a1")

    # Completed work kept flowing after resume too.
    result = await stack_b.run_rollout(3, 0)
    assert result["reward"] == 1.0
    assert stack_b.rows("3-0")
