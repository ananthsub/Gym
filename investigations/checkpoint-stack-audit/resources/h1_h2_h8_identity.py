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
"""H1/H2/H8: identity-less callers, capture-prefixed verify, register-before-success."""

import asyncio
import os
from unittest.mock import MagicMock

os.environ["NEMO_GYM_CHECKPOINT_CONTROL_TOKEN"] = "audit-token"

import httpx  # noqa: E402

from nemo_gym.checkpoint.resources import ResourcesSessionMiddleware  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponse  # noqa: E402
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER, RolloutContextMiddleware  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.example_session_state_mgmt.app import (  # noqa: E402
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)
from resources_servers.workplace_assistant.app import (  # noqa: E402
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
)


def _server(server_type, config_type):
    config = config_type(host="", port=0, entrypoint="", name="resources")
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return server_type(config=config, server_client=client)


def _hdr(rid, att):
    return {ROLLOUT_ID_HEADER: rid, ATTEMPT_INDEX_HEADER: str(att)}


async def counter_checks():
    server = _server(StatefulCounterResourcesServer, StatefulCounterResourcesServerConfig)
    app = server.setup_webserver()
    server.setup_liveness(app)  # mirrors run_webserver
    print("middleware order (outermost first):", [m.cls.__name__ for m in app.user_middleware])
    outer_idx = [m.cls for m in app.user_middleware].index(ResourcesSessionMiddleware)
    ctx_idx = [m.cls for m in app.user_middleware].index(RolloutContextMiddleware)
    print("ResourcesSessionMiddleware outside RolloutContextMiddleware:", outer_idx < ctx_idx)
    p = server.checkpoint_participant()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        for method, path in [("GET", "/"), ("GET", "/docs"), ("GET", "/openapi.json"), ("POST", "/mcp"),
                             ("POST", "/seed_session"), ("POST", "/ng-rollout/r1/seed_session"),
                             ("POST", "/verify"), ("GET", "/ng-control/v1/capabilities")]:
            r = await c.request(method, path, json={"initial_count": 1} if "seed" in path else {})
            print(f"  no-headers {method} {path:35s} -> {r.status_code} {r.text[:70]}")
        # capture-prefixed path with headers passes both middlewares
        r = await c.post("/ng-rollout/r1/seed_session", json={"initial_count": 5}, headers=_hdr("r1", 0))
        print("  headers+prefix seed:", r.status_code, "rev", r.headers.get("x-nemo-gym-resource-state-revision"))
        r = await c.post("/get_counter_value", headers=_hdr("r1", 0), cookies=r.cookies)
        print("  read /get_counter_value rev header:", r.headers.get("x-nemo-gym-resource-state-revision"), "(H3: reads bump revision)")
        # verify under capture prefix (observability on) -> not treated as terminal
        vb = {"responses_create_params": {"input": []}, "response": NeMoGymResponse(id="x", created_at=0, model="m", object="response", output=[], parallel_tool_calls=False, tool_choice="auto", tools=[]).model_dump(), "expected_count": 5}
        r = await c.post("/ng-rollout/r1/verify", json=vb, headers=_hdr("r1", 0))
        print("  prefixed verify status:", r.status_code, "| _revisions still has r1:", ("r1", 0) in p._revisions, "(H2)")
        # unprefixed verify -> terminal, retired
        r = await c.post("/ng-rollout/r5/seed_session", json={"initial_count": 5}, headers=_hdr("r5", 0))
        r = await c.post("/verify", json=vb, headers=_hdr("r5", 0))
        print("  bare verify status:", r.status_code, "| _revisions has r5:", ("r5", 0) in p._revisions)
        # 422 seed registers a key with no mapping
        r = await c.post("/seed_session", json={"bad": 1}, headers=_hdr("r2", 0))
        print("  422 seed:", r.status_code, "| _revisions has r2:", ("r2", 0) in p._revisions, "| mapping:", ("r2", 0) in server.execution_to_session)
        # tool call with identity but never seeded
        r = await c.post("/increment_counter", json={"count": 1}, headers=_hdr("r4", 0))
        print("  unseeded increment:", r.status_code, "| _revisions has r4:", ("r4", 0) in p._revisions, "| mapping:", ("r4", 0) in server.execution_to_session)
    try:
        import time; await p.prepare(time.time() + 10)
        print("  prepare: OK")
    except Exception as e:
        print("  prepare raised:", type(e).__name__, e, "| accepting:", p.accepting)


async def workplace_prefixed_verify():
    server = _server(WorkbenchResourcesServer, WorkbenchResourcesServerConfig)
    app = server.setup_webserver()
    p = server.checkpoint_participant()
    resp = NeMoGymResponse(id="x", created_at=0, model="m", object="response", output=[], parallel_tool_calls=False, tool_choice="auto", tools=[]).model_dump()
    vb = {"responses_create_params": {"input": []}, "response": resp, "ground_truth": [], "id": 1, "category": "c", "environment_name": "e"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/ng-rollout/w1/seed_session", json={}, headers=_hdr("w1", 0))
        print("workplace seed:", r.status_code)
        r = await c.post("/ng-rollout/w1/verify", json=vb, headers=_hdr("w1", 0))
        print("workplace prefixed verify:", r.status_code, "| _revisions has w1:", ("w1", 0) in p._revisions, "| mapping:", ("w1", 0) in server.execution_to_session)
    import time
    try:
        await p.prepare(time.time() + 10)
        print("workplace prepare: OK", p.status())
    except Exception as e:
        print("workplace prepare raised:", type(e).__name__, repr(e), "| accepting:", p.accepting)


asyncio.run(counter_checks())
asyncio.run(workplace_prefixed_verify())
