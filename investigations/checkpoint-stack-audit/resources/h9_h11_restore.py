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
"""H9/H11: fresh process accepts N+1 before restore; same-process restore leaves N live."""

import asyncio
import os
import time
from unittest.mock import MagicMock

os.environ["NEMO_GYM_CHECKPOINT_CONTROL_TOKEN"] = "audit-token"

import httpx  # noqa: E402

from nemo_gym.checkpoint import ResourceSnapshot  # noqa: E402
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.example_session_state_mgmt.app import (  # noqa: E402
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)


def _server():
    cfg = StatefulCounterResourcesServerConfig(host="", port=0, entrypoint="", name="resources")
    cl = MagicMock(spec=ServerClient)
    cl.global_config_dict = {}
    return StatefulCounterResourcesServer(config=cfg, server_client=cl)


def _hdr(r, a):
    return {ROLLOUT_ID_HEADER: r, ATTEMPT_INDEX_HEADER: str(a)}


async def pre_restore_acceptance():
    s = _server()
    app = s.setup_webserver()
    p = s.checkpoint_participant()
    print("fresh process accepting before any restore:", p.accepting)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/increment_counter", json={"count": 5}, headers=_hdr("r", 1))
        print("  N+1 increment before restore:", r.status_code, "rev", r.headers.get("x-nemo-gym-resource-state-revision"))
        print("  counters:", s.session_id_to_counter, "| mapping:", s.execution_to_session)
        await p.restore([ResourceSnapshot(rollout_id="r", attempt_index=0, state_revision=3, state={"counter": 17})])
        p.resume()
        r = await c.post("/get_counter_value", headers=_hdr("r", 1), cookies=r.cookies)
        print("  after restore+resume, N+1 reads:", r.json(), "rev", r.headers.get("x-nemo-gym-resource-state-revision"), "| counters:", s.session_id_to_counter)


async def same_process_restore():
    s = _server()
    app = s.setup_webserver()
    p = s.checkpoint_participant()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/seed_session", json={"initial_count": 1}, headers=_hdr("r", 0))
    await p.prepare(time.time() + 5)
    snaps = p.prepared_snapshots()
    await p.restore(snaps)  # restore on the SAME process that prepared
    p.resume()
    print("same-process restore: _revisions keys:", sorted(p._revisions), "| tombstones:", p._tombstones)
    await p.prepare(time.time() + 5)
    print("  second prepare exports:", [(x.rollout_id, x.attempt_index, x.state) for x in p.prepared_snapshots()])
    await p.restore(p.prepared_snapshots())
    print("  after 2nd restore: _revisions keys:", sorted(p._revisions), "| mapping:", sorted(s.execution_to_session))


asyncio.run(pre_restore_acceptance())
asyncio.run(same_process_restore())
