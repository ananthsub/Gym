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
"""H4: prepare's sorted() snapshot records a pre-mutation revision for an in-flight handler."""

import asyncio
import time

import httpx
from fastapi import FastAPI

from nemo_gym.checkpoint.resources import RESOURCE_STATE_REVISION_HEADER, ResourcesCheckpointParticipant, ResourcesSessionMiddleware
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER


async def main():
    state = {("a", 0): {"v": 0}, ("b", 0): {"v": 0}}

    async def export(r, a):
        return dict(state[(r, a)])

    async def restore(s):
        pass

    p = ResourcesCheckpointParticipant(export_state=export, restore_states=restore)
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    block = {"on": False}

    @app.post("/mutate/{key}")
    async def mutate(key: str):
        if key == "b" and block["on"]:
            entered.set()
            await release.wait()
        state[(key, 0)]["v"] += 1
        return state[(key, 0)]

    app.add_middleware(ResourcesSessionMiddleware, participant=p)
    hb = {ROLLOUT_ID_HEADER: "b", ATTEMPT_INDEX_HEADER: "0"}
    ha = {ROLLOUT_ID_HEADER: "a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/mutate/a", headers=ha)  # a rev 1
        await c.post("/mutate/b", headers=hb)  # b rev 1
        block["on"] = True
        inflight = asyncio.create_task(c.post("/mutate/b", headers=hb))
        await entered.wait()
        prep = asyncio.create_task(p.prepare(time.time() + 5))
        await asyncio.sleep(0.05)  # prepare has materialized sorted(items) and is exporting 'a' / waiting on b's lock
        release.set()
        resp, report = await asyncio.gather(inflight, prep)
    print("in-flight response revision header:", resp.headers[RESOURCE_STATE_REVISION_HEADER])
    for s in p.prepared_snapshots():
        print(f"snapshot {s.rollout_id}: state={s.state} state_revision={s.state_revision}")
    print("participant _revisions:", p._revisions)


asyncio.run(main())
