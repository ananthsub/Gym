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
"""Admission control drains a server to a quiescent point without deadlock.

The pinned behaviors: state changes are atomic with the admission test; a
nested call under a still-running accepted operation stays admissible while
the server drains (refusing it would deadlock the drain); a refused caller
gets 409 checkpoint_parked, a park signal rather than an error; a rollout
attempt force-closed at the deadline is tombstoned so its late calls cannot
write under an identity the restore already replaced; and only policy
model-server instances gate — a pause sent to a judge instance is rejected.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.checkpoint import (
    ADMISSION_LEASE_HEADER,
    CONTROL_URL_PREFIX,
    GATED_MODEL_ROUTE_SUFFIXES,
    MODEL_ADMISSION_URL_PREFIX,
    AdmissionLimiter,
    AdmissionMiddleware,
    AdmissionParkedError,
    AdmissionState,
    StaleAttemptError,
    current_admission_lease,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import ROLLOUT_ID_HEADER
from nemo_gym.server_utils import ServerClient


# --- limiter ---


def test_close_with_nothing_inflight_pauses_immediately() -> None:
    limiter = AdmissionLimiter()
    limiter.close()
    assert limiter.state == AdmissionState.PAUSED


def test_drain_completes_when_last_inflight_releases() -> None:
    limiter = AdmissionLimiter()
    ticket = limiter.admit(rollout_id="4-2")
    limiter.close()
    assert limiter.state == AdmissionState.DRAINING
    limiter.release(ticket)
    assert limiter.state == AdmissionState.PAUSED


def test_new_root_operation_parks_while_draining() -> None:
    limiter = AdmissionLimiter()
    held = limiter.admit(rollout_id="4-2")
    limiter.close()
    with pytest.raises(AdmissionParkedError):
        limiter.admit(rollout_id="9-9")
    limiter.release(held)


def test_nested_call_under_accepted_operation_admitted_while_draining() -> None:
    limiter = AdmissionLimiter()
    root = limiter.admit(rollout_id="4-2")
    limiter.close()
    # The nested call carries the root operation's lease: refusing it would
    # deadlock the drain because the root could never finish.
    nested = limiter.admit(rollout_id="4-2", lease=root.lease)
    assert nested.nested
    limiter.release(nested)
    assert limiter.state == AdmissionState.DRAINING
    limiter.release(root)
    assert limiter.state == AdmissionState.PAUSED


def test_lease_dies_with_its_root_operation() -> None:
    limiter = AdmissionLimiter()
    root = limiter.admit(rollout_id="4-2")
    limiter.close()
    limiter.release(root)
    with pytest.raises(AdmissionParkedError):
        limiter.admit(rollout_id="4-2", lease=root.lease)


def test_unknown_lease_parks_while_draining() -> None:
    limiter = AdmissionLimiter()
    held = limiter.admit(rollout_id="4-2")
    limiter.close()
    with pytest.raises(AdmissionParkedError):
        limiter.admit(rollout_id="9-9", lease="not-a-real-lease")
    limiter.release(held)


def test_resume_reopens_admission() -> None:
    limiter = AdmissionLimiter()
    limiter.close()
    limiter.resume()
    assert limiter.state == AdmissionState.ACCEPTING
    limiter.release(limiter.admit(rollout_id="4-2"))


def test_abort_inflight_tombstones_and_unblocks_drain() -> None:
    limiter = AdmissionLimiter()
    stuck = limiter.admit(rollout_id="7-1-a2")
    limiter.close()
    aborted = limiter.abort_inflight("7-1-a2", 2)
    assert aborted == [stuck.ticket_id]
    # The stuck request no longer counts toward the drain.
    assert limiter.state == AdmissionState.PAUSED
    # Releasing the force-closed ticket later is a harmless no-op.
    limiter.release(stuck)
    assert limiter.counts()["inflight_total"] == 0

    limiter.resume()
    # The abandoned attempt is fenced permanently...
    with pytest.raises(StaleAttemptError):
        limiter.admit(rollout_id="7-1-a2")
    with pytest.raises(StaleAttemptError):
        limiter.admit(rollout_id="7-1", attempt_index=2)
    # ...but the replacement attempt is admissible.
    limiter.release(limiter.admit(rollout_id="7-1-a3"))
    assert limiter.tombstones() == [("7-1", 2)]


@pytest.mark.asyncio
async def test_wait_for_drained_long_poll() -> None:
    limiter = AdmissionLimiter()
    ticket = limiter.admit(rollout_id="4-2")
    limiter.close()

    async def release_soon() -> None:
        await asyncio.sleep(0.01)
        limiter.release(ticket)

    releaser = asyncio.create_task(release_soon())
    assert await limiter.wait_for_drained(timeout_s=1.0)
    await releaser
    assert limiter.state == AdmissionState.PAUSED


@pytest.mark.asyncio
async def test_wait_for_drained_times_out_with_stragglers() -> None:
    limiter = AdmissionLimiter()
    held = limiter.admit(rollout_id="4-2")
    limiter.close()
    assert not await limiter.wait_for_drained(timeout_s=0.01)
    limiter.release(held)


# --- middleware ---


def _gated_app(limiter: AdmissionLimiter) -> TestClient:
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses() -> dict:
        return {"lease": current_admission_lease(), "inflight": limiter.counts()["inflight_total"]}

    @app.get("/other")
    async def other() -> dict:
        return {"ok": True}

    app.add_middleware(AdmissionMiddleware, limiter=limiter, gated_suffixes=GATED_MODEL_ROUTE_SUFFIXES)
    return TestClient(app)


def test_middleware_admits_and_installs_lease_context() -> None:
    limiter = AdmissionLimiter()
    body = _gated_app(limiter).post("/v1/responses").json()
    # The request was in flight while handled and released afterwards.
    assert body["inflight"] == 1
    assert body["lease"]
    assert limiter.counts()["inflight_total"] == 0


def test_middleware_parks_new_calls_when_closed() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.close()
    response = client.post("/v1/responses", headers={ROLLOUT_ID_HEADER: "4-2"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "checkpoint_parked"
    assert response.headers["retry-after"] == "1"


def test_middleware_admits_leased_call_while_draining() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    root = limiter.admit(rollout_id="4-2")
    limiter.close()
    response = client.post("/v1/responses", headers={ADMISSION_LEASE_HEADER: root.lease})
    assert response.status_code == 200
    limiter.release(root)


def test_middleware_rejects_stale_attempt() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.close()
    limiter.abort_inflight("7-1", 0)
    limiter.resume()
    response = client.post("/v1/responses", headers={ROLLOUT_ID_HEADER: "7-1"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stale_attempt"


def test_middleware_leaves_ungated_paths_open_while_paused() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.close()
    assert client.get("/other").json() == {"ok": True}


# --- model server routes ---


def _model_server(instance_role: str) -> SimpleResponsesAPIModel:
    class _Model(SimpleResponsesAPIModel):
        async def chat_completions(self, request):
            raise NotImplementedError

        async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            return NeMoGymResponse.model_validate(
                {
                    "id": "resp-1",
                    "created_at": 0.0,
                    "model": "m",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                }
            )

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    return _Model(
        config=BaseResponsesAPIModelConfig(host="", port=0, entrypoint="", name="policy", instance_role=instance_role),
        server_client=server_client,
    )


def test_policy_model_server_pause_drain_resume_cycle() -> None:
    client = TestClient(_model_server("policy").setup_webserver())

    capabilities = client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert capabilities["instance_role"] == "policy"
    assert capabilities["admission_states"] == ["accepting", "draining", "paused"]

    # Generation works while accepting.
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 200

    pause = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9})
    assert pause.status_code == 200
    assert pause.json() == {
        "state": "paused",
        "workers": {"acknowledged": 1, "expected": 1},
        "inflight_total": 0,
        "waiters_total": 0,
    }

    # New generation parks; control routes stay reachable.
    parked = client.post("/v1/responses", json={"input": "hi"})
    assert parked.status_code == 409
    assert parked.json()["error"]["code"] == "checkpoint_parked"
    status = client.get(f"{MODEL_ADMISSION_URL_PREFIX}/status").json()
    assert status["state"] == "paused"
    assert status["per_worker"] == {"0": {"state": "paused", "inflight": 0}}

    # A duplicate pause replays; a competing checkpoint conflicts.
    replay = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9})
    assert replay.json() == pause.json()
    conflict = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-2", "deadline_ts": 4e9})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "checkpoint_conflict"

    resume = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "ckpt-1"})
    assert resume.json() == {
        "state": "accepting",
        "workers": {"acknowledged": 1, "expected": 1},
        "released_waiters": 0,
    }
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 200

    # The finished checkpoint id is stale for new operations; a new checkpoint proceeds.
    stale = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9})
    assert stale.status_code == 200  # replayed recorded result, not a new pause
    assert client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-3", "deadline_ts": 4e9}
    ).json()["state"] in {"paused", "draining"}


def test_abort_inflight_tombstones_via_route() -> None:
    server = _model_server("policy")
    client = TestClient(server.setup_webserver())
    client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9})

    abort = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
        json={"checkpoint_id": "ckpt-1", "rollout_id": "7-1-a2", "attempt_index": 2},
    )
    assert abort.status_code == 200
    status = client.get(f"{MODEL_ADMISSION_URL_PREFIX}/status").json()
    assert status["tombstones"] == [{"rollout_id": "7-1", "attempt_index": 2}]

    client.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "ckpt-1"})
    stale = client.post("/v1/responses", json={"input": "hi"}, headers={ROLLOUT_ID_HEADER: "7-1-a2"})
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_attempt"
    fresh = client.post("/v1/responses", json={"input": "hi"}, headers={ROLLOUT_ID_HEADER: "7-1-a3"})
    assert fresh.status_code == 200


@pytest.mark.asyncio
async def test_server_client_propagates_admission_lease(monkeypatch) -> None:
    import nemo_gym.server_utils
    from nemo_gym.checkpoint import admission_lease_context
    from nemo_gym.config_types import BaseServerConfig

    calls = []

    async def dispatch(method, url, **kwargs):
        calls.append(kwargs)

        class _Response:
            status = 200
            ok = True
            cookies: dict = {}

        return _Response()

    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)
    from omegaconf import OmegaConf

    client = ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=OmegaConf.create(
            {"judge": {"responses_api_models": {"m": {"host": "j.test", "port": 80}}}}
        ),
    )
    with admission_lease_context("lease-abc"):
        await client.post(server_name="judge", url_path="/v1/responses", json={})
    await client.post(server_name="judge", url_path="/v1/responses", json={})

    assert calls[0]["headers"][ADMISSION_LEASE_HEADER] == "lease-abc"
    assert ADMISSION_LEASE_HEADER not in (calls[1].get("headers") or {})


def test_auxiliary_model_server_never_gates_and_rejects_pause() -> None:
    client = TestClient(_model_server("auxiliary").setup_webserver())

    capabilities = client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert capabilities["instance_role"] == "auxiliary"
    assert capabilities["admission_states"] == ["accepting"]

    pause = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9})
    assert pause.status_code == 409
    assert pause.json()["error"]["code"] == "not_a_policy_instance"

    # Judge traffic keeps flowing: there is no admission middleware to close.
    assert client.post("/v1/responses", json={"input": "grade"}).status_code == 200
