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
"""H7: adapter fidelity through a real JSON round trip (ResourceSnapshot -> json -> restore)."""

import asyncio
import json
import random
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.checkpoint import ResourceSnapshot
from nemo_gym.server_utils import ServerClient
from resources_servers.blackjack.app import BlackjackEnv
import resources_servers.example_multi_turn_gymnasium.app as _mt
ExampleMultiTurnGymnasiumServer = next(v for v in vars(_mt).values() if isinstance(v, type) and v.__module__ == _mt.__name__ and any(b.__name__ == "GymnasiumServer" for b in v.__mro__[1:]))
from resources_servers.gymnasium.base import GymnasiumServer
from resources_servers.workplace_assistant.app import _TOOLKITS, WorkbenchResourcesServer, WorkbenchResourcesServerConfig, get_tools


def _server(t, c):
    cfg = c(host="", port=0, entrypoint="", name="resources")
    cl = MagicMock(spec=ServerClient)
    cl.global_config_dict = {}
    return t(config=cfg, server_client=cl)


def roundtrip(snapshot: ResourceSnapshot) -> ResourceSnapshot:
    return ResourceSnapshot.model_validate_json(snapshot.model_dump_json())


async def blackjack():
    s = _server(BlackjackEnv, BaseResourcesServerConfig)
    rng = random.Random(7)
    s.execution_to_session[("r", 0)] = "sid"
    s.session_state["sid"] = {"player": ["A", "5"], "dealer": ["9", "2"], "rng": rng}
    st = await s.export_checkpoint_state("r", 0)
    print("blackjack getstate depth:", type(st["rng_state"]).__name__, len(st["rng_state"]), type(st["rng_state"][1]).__name__, len(st["rng_state"][1]))
    snap = roundtrip(ResourceSnapshot(rollout_id="r", attempt_index=1, state_revision=1, state=st))
    print("after JSON: rng_state is", type(snap.state["rng_state"]).__name__)
    expected = [rng.choice("23456789") for _ in range(5)]
    await s.restore_checkpoint_states([snap])
    got = [s.session_state["checkpoint:r:a1"]["rng"].choice("23456789") for _ in range(5)]
    print("blackjack rng continues identically after JSON round trip:", expected == got)


async def workplace():
    s = _server(WorkbenchResourcesServer, WorkbenchResourcesServerConfig)
    env = get_tools(_TOOLKITS)
    # mutate a few frames the way tools do
    env["functions"]["calendar_create_event"](event_name="Audit", participant_email="a@b.com", event_start="2023-12-01 10:00:00", duration="60")
    env["functions"]["email_send_email"](recipient="x@y.com", subject="s", body="b")
    env["functions"]["customer_relationship_manager_add_customer"](customer_name="Z", assigned_to_email="a@b.com", status="Qualified")
    env["functions"]["project_management_create_task"](task_name="T", assigned_to_email="a@b.com", list_name="Backlog", due_date="2023-12-05", board="Front end")
    s.execution_to_session[("r", 0)] = "sid"
    s.session_id_to_tool_env["sid"] = env
    before_search = {
        "cal": env["functions"]["calendar_search_events"](query="Audit"),
        "email": env["functions"]["email_search_emails"](query="s"),
        "crm": env["functions"]["customer_relationship_manager_search_customers"](customer_name="Z"),
        "pm": env["functions"]["project_management_search_tasks"](task_name="T"),
        "analytics": env["functions"]["analytics_total_visits_count"](time_min="2023-11-01", time_max="2023-11-30"),
    }
    st = await s.export_checkpoint_state("r", 0)
    snap = roundtrip(ResourceSnapshot(rollout_id="r", attempt_index=1, state_revision=1, state=st))
    await s.restore_checkpoint_states([snap])
    renv = s.session_id_to_tool_env["checkpoint:r:a1"]
    for name, container in env["containers"].items():
        for attr, frame in vars(container).items():
            if not isinstance(frame, pd.DataFrame):
                continue
            rf = getattr(renv["containers"][name], attr)
            eq = frame.equals(rf)
            dt_eq = list(frame.dtypes) == list(rf.dtypes)
            idx_eq = frame.index.equals(rf.index) and type(frame.index) is type(rf.index)
            flag = "" if (eq and dt_eq and idx_eq) else "   <-- DIFF"
            print(f"  {name}.{attr:24s} shape={frame.shape} equals={eq} dtypes_equal={dt_eq} index_equal={idx_eq}{flag}")
            if not dt_eq:
                print("     before:", dict(frame.dtypes.astype(str)))
                print("     after: ", dict(rf.dtypes.astype(str)))
            if not idx_eq:
                print("     index before:", type(frame.index).__name__, frame.index[:3].tolist(), "after:", type(rf.index).__name__, rf.index[:3].tolist())
    after_search = {
        "cal": renv["functions"]["calendar_search_events"](query="Audit"),
        "email": renv["functions"]["email_search_emails"](query="s"),
        "crm": renv["functions"]["customer_relationship_manager_search_customers"](customer_name="Z"),
        "pm": renv["functions"]["project_management_search_tasks"](task_name="T"),
        "analytics": renv["functions"]["analytics_total_visits_count"](time_min="2023-11-01", time_max="2023-11-30"),
    }
    for k in before_search:
        print(f"  tool output identical after restore [{k}]: {before_search[k] == after_search[k]}")
        if before_search[k] != after_search[k]:
            print("    before:", str(before_search[k])[:200])
            print("    after: ", str(after_search[k])[:200])
    # a follow-on mutation on the restored env
    try:
        out = renv["functions"]["calendar_create_event"](event_name="Audit2", participant_email="a@b.com", event_start="2023-12-02 10:00:00", duration="30")
        print("  post-restore create_event:", str(out)[:80])
    except Exception as e:
        print("  post-restore create_event raised:", repr(e))


async def gymnasium_base_types():
    class T(GymnasiumServer):
        async def step(self, *a, **k):
            return None, 0.0, True, False, {}

    s = _server(T, BaseResourcesServerConfig)
    print("tuple survives serialize?", type(s.serialize_session_state({"pos": (1, 2)})["pos"]).__name__)
    try:
        s.serialize_session_state({"arr": np.int64(3)})
    except TypeError as e:
        print("numpy scalar at prepare -> TypeError:", str(e)[:60])
    try:
        s.serialize_session_state({"s": {1, 2}})
    except TypeError as e:
        print("set at prepare -> TypeError:", str(e)[:60])


async def unadapted_subclass():
    s = _server(ExampleMultiTurnGymnasiumServer, BaseResourcesServerConfig)
    print("ExampleMultiTurnGymnasium checkpoint_state_enabled:", s.checkpoint_state_enabled())
    await s.reset({}, "sid")
    s.execution_to_session[("r", 0)] = "sid"
    print("  session_turns:", s.session_turns, "| session_state:", s.session_state)
    try:
        await s.export_checkpoint_state("r", 0)
    except Exception as e:
        print("  export raises:", type(e).__name__, e)


asyncio.run(blackjack())
asyncio.run(workplace())
asyncio.run(gymnasium_base_types())
asyncio.run(unadapted_subclass())
