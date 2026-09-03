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
"""H7b: which CRM fields change after a JSON round trip."""

import asyncio
from unittest.mock import MagicMock

import pandas as pd

from nemo_gym.checkpoint import ResourceSnapshot
from nemo_gym.server_utils import ServerClient
from resources_servers.workplace_assistant.app import _TOOLKITS, WorkbenchResourcesServer, WorkbenchResourcesServerConfig, get_tools


async def main():
    cfg = WorkbenchResourcesServerConfig(host="", port=0, entrypoint="", name="resources")
    cl = MagicMock(spec=ServerClient)
    cl.global_config_dict = {}
    s = WorkbenchResourcesServer(config=cfg, server_client=cl)
    env = get_tools(_TOOLKITS)
    env["functions"]["customer_relationship_manager_add_customer"](customer_name="Z", assigned_to_email="a@b.com", status="Qualified")
    s.execution_to_session[("r", 0)] = "sid"
    s.session_id_to_tool_env["sid"] = env
    before = env["functions"]["customer_relationship_manager_search_customers"](customer_name="Z")
    st = await s.export_checkpoint_state("r", 0)
    snap = ResourceSnapshot.model_validate_json(ResourceSnapshot(rollout_id="r", attempt_index=1, state_revision=1, state=st).model_dump_json())
    await s.restore_checkpoint_states([snap])
    renv = s.session_id_to_tool_env["checkpoint:r:a1"]
    after = renv["functions"]["customer_relationship_manager_search_customers"](customer_name="Z")
    for b, a in zip(before["customers"], after["customers"]):
        for k in b:
            if b[k] != a[k] and not (isinstance(b[k], float) and isinstance(a[k], float) and b[k] != b[k] and a[k] != a[k]):
                print(f"  field {k!r}: before={b[k]!r} ({type(b[k]).__name__}) after={a[k]!r} ({type(a[k]).__name__})")
    crm_b = env["containers"]["customer_relationship_manager"]._crm_data
    crm_a = renv["containers"]["customer_relationship_manager"]._crm_data
    last_b, last_a = crm_b.iloc[-1], crm_a.iloc[-1]
    print("  raw last row before:", {k: (repr(v), type(v).__name__) for k, v in last_b.items() if (v != v) or v is None or v == ""})
    print("  raw last row after: ", {k: (repr(v), type(v).__name__) for k, v in last_a.items() if (v != v) or v is None or v == ""})
    # also: does the email tool's HARDCODED_CURRENT_TIME logic survive? (strings kept as strings)
    em = renv["containers"]["email"]._emails
    print("  email sent_datetime dtype after restore:", em["sent_datetime"].dtype, "| sample:", em["sent_datetime"].iloc[0])
    print("  CRM dtypes after:", dict(crm_a.dtypes.astype(str)))
    print("  pandas", pd.__version__)


asyncio.run(main())
