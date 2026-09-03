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
"""H6: terminal /step bookkeeping (bare and capture-prefixed) and close_session on checkpoint: ids."""

import asyncio
import os
import time
from unittest.mock import MagicMock

os.environ["NEMO_GYM_CHECKPOINT_CONTROL_TOKEN"] = "audit-token"

import httpx  # noqa: E402

from nemo_gym.base_resources_server import BaseResourcesServerConfig  # noqa: E402
from nemo_gym.checkpoint import ResourceSnapshot  # noqa: E402
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.blackjack.app import BlackjackEnv  # noqa: E402


def _server():
    cfg = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="resources")
    cl = MagicMock(spec=ServerClient)
    cl.global_config_dict = {}
    return BlackjackEnv(config=cfg, server_client=cl)


def _hdr(r, a):
    return {ROLLOUT_ID_HEADER: r, ATTEMPT_INDEX_HEADER: str(a)}


def _step_body(action):
    return {
        "responses_create_params": {"input": []},
        "response": {
            "id": "x", "created_at": 0, "model": "m", "object": "response", "parallel_tool_calls": False,
            "tool_choice": "auto", "tools": [],
            "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": f"<action>{action}</action>", "annotations": []}]}],
        },
    }


async def main():
    s = _server()
    app = s.setup_webserver()
    s.setup_exception_middleware(app)  # as run_webserver does
    p = s.checkpoint_participant()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        for prefix, rid in [("", "g1"), ("/ng-rollout/g2", "g2")]:
            r = await c.post(f"{prefix}/reset", json={"responses_create_params": {"input": []}}, headers=_hdr(rid, 0))
            ck = r.cookies
            r = await c.post(f"{prefix}/step", json=_step_body("stand"), headers=_hdr(rid, 0), cookies=ck)
            j = r.json()
            print(f"{prefix or '/'} terminal step: {r.status_code} terminated={j['terminated']} rev={r.headers.get('x-nemo-gym-resource-state-revision')}")
            print(f"   _revisions has {rid}: {(rid, 0) in p._revisions} | terminal_flag: {(rid, 0) in p._terminal_after_request} | "
                  f"session_state keys: {list(s.session_state)} | mapping: {(rid, 0) in s.execution_to_session} | lock kept: {(rid, 0) in p._locks}")
        # a terminal step whose handler raises after mark_terminal cannot happen in blackjack; instead show
        # a step for an identity that never reset (agent order) -> registered, no mapping
        r = await c.post("/step", json=_step_body("hit"), headers=_hdr("g3", 0))
        print("unreset step:", r.status_code, "| _revisions has g3:", ("g3", 0) in p._revisions, "| mapping:", ("g3", 0) in s.execution_to_session)
        try:
            await p.prepare(time.time() + 5)
            print("prepare ok:", p.status())
        except Exception as e:
            print("prepare raised:", type(e).__name__, repr(e))
        p.resume()
        p.retire("g3", 0)
        # restore + continuation with the OLD cookie and identity (g4, 1); terminal -> close_session('checkpoint:g4:a1')
        s.execution_to_session[("g4", 0)] = "old-cookie-sid"
        import random
        s.session_state["old-cookie-sid"] = {"player": ["10", "9"], "dealer": ["5", "5"], "rng": random.Random(1)}
        st = await s.export_checkpoint_state("g4", 0)
        await p.restore([ResourceSnapshot(rollout_id="g4", attempt_index=0, state_revision=2, state=st)])
        p.resume()
        r = await c.post("/step", json=_step_body("stand"), headers=_hdr("g4", 1), cookies={"BlackjackEnv___resources": "stale"})
        print("post-restore N+1 step:", r.status_code, r.json().get("info", {}).get("result"), "rev", r.headers.get("x-nemo-gym-resource-state-revision"))
        print("   session_state keys:", list(s.session_state), "| mapping:", sorted(s.execution_to_session), "| _revisions:", sorted(p._revisions))
        r = await c.post("/step", json=_step_body("stand"), headers=_hdr("g4", 0))
        print("stale attempt N step:", r.status_code, r.json()["error"]["code"])


asyncio.run(main())
