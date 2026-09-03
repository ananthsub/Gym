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
"""H4b: a /verify in flight when prepare materializes its key list retires the key; prepare then exports it and fails."""

import asyncio
import time

import httpx
from fastapi import FastAPI

from nemo_gym.checkpoint.resources import ResourcesCheckpointParticipant, ResourcesSessionMiddleware
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER


async def main():
    state = {("a", 0): {"v": 1}, ("b", 0): {"v": 1}}

    async def export(r, a):
        return dict(state[(r, a)])  # KeyError when the adapter already dropped the session (workplace verify finally:)

    async def restore(s):
        pass

    p = ResourcesCheckpointParticipant(export_state=export, restore_states=restore)
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()

    @app.post("/verify")
    async def verify():
        entered.set()
        await release.wait()
        state.pop(("b", 0))  # what the workplace/gymnasium adapters do on the terminal request
        return {"reward": 1.0}

    @app.post("/mutate")
    async def mutate():
        return {"ok": True}

    app.add_middleware(ResourcesSessionMiddleware, participant=p)
    ha = {ROLLOUT_ID_HEADER: "a", ATTEMPT_INDEX_HEADER: "0"}
    hb = {ROLLOUT_ID_HEADER: "b", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/mutate", headers=ha)
        await c.post("/mutate", headers=hb)
        inflight = asyncio.create_task(c.post("/verify", headers=hb))
        await entered.wait()
        prep = asyncio.create_task(p.prepare(time.time() + 5))
        await asyncio.sleep(0.05)
        release.set()
        resp = await inflight
        try:
            report = await prep
            print("prepare:", report, [(s.rollout_id, s.state) for s in p.prepared_snapshots()])
        except Exception as e:
            print("verify status:", resp.status_code, "| prepare raised:", type(e).__name__, repr(e), "| accepting:", p.accepting)
    print("_revisions after:", p._revisions)


asyncio.run(main())
