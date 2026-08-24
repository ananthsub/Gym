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
"""Runtime routing of dataset rows to agent servers.

Dataset rows carry ``agent_ref.name``. Historically that name had to match an agent server
instance in the run config, which coupled prepared data to serving infrastructure: renaming an
agent or swapping harnesses meant re-processing every dataset. This module treats
``agent_ref.name`` as a *routing key* instead. The optional ``environments`` section of the
global config remaps routing keys to agent server instance names at dispatch time::

    environments:
      math_with_judge:
        agent: math_with_judge_simple_agent

(or ``+environments.math_with_judge.agent=...`` on the CLI). A routing key with no entry routes
to the key itself, so existing datasets behave exactly as before. Reporting and aggregation stay
keyed by the routing key; only network destinations use the resolved name. When a remap changes
the destination, the result is stamped with ``resolved_agent_ref`` to record which agent actually
ran the rollout.
"""

from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ENVIRONMENTS_KEY_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
)


def load_agent_routing_map(global_config_dict: Any) -> Dict[str, str]:
    """Parse the ``environments`` config section into a {routing_key: agent server name} map.

    Fails fast on malformed entries and on agent names that don't exist in the config, so a bad
    map surfaces before dispatch instead of as a mid-run routing error. Returns an empty map when
    the section is absent or the config is not a mapping (e.g. test doubles).
    """
    if not isinstance(global_config_dict, (Mapping, DictConfig)):
        return {}
    section = global_config_dict.get(ENVIRONMENTS_KEY_NAME)
    if section is None:
        return {}
    if isinstance(section, DictConfig):
        section = OmegaConf.to_container(section, resolve=True)
    if not isinstance(section, Mapping):
        raise ValueError(
            f"`{ENVIRONMENTS_KEY_NAME}` must be a mapping of routing keys to `{{agent: <agent server name>}}` "
            f"entries, got {type(section).__name__}"
        )

    routing: Dict[str, str] = {}
    for key, entry in section.items():
        agent = entry.get("agent") if isinstance(entry, Mapping) else None
        if not isinstance(agent, str) or not agent or set(entry) != {"agent"}:
            raise ValueError(
                f"Invalid `{ENVIRONMENTS_KEY_NAME}` entry for {str(key)!r}: expected "
                f"`{{agent: <agent server name>}}`, got {entry!r}"
            )
        routing[str(key)] = agent

    known_agents = agent_server_names(global_config_dict)
    unknown = {key: agent for key, agent in routing.items() if agent not in known_agents}
    if unknown:
        raise ValueError(
            f"`{ENVIRONMENTS_KEY_NAME}` routes to agent servers that are not in the config: {unknown}. "
            f"Configured agent servers: {sorted(known_agents)}"
        )
    return routing


def resolve_agent_name(row: Mapping, routing: Mapping[str, str]) -> Optional[str]:
    """Return the agent server instance that should run this row.

    The row's ``agent_ref.name`` is looked up in the routing map; a key with no entry is used
    verbatim (the pre-routing-map behavior). Returns None when the row has no routing key.
    """
    key = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
    if key is None:
        return None
    return routing.get(key, key)


def agent_server_names(global_config_dict: Any) -> set:
    """Names of the agent server instances configured at the top level of the global config."""
    names = set()
    for key in global_config_dict:
        if key in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
            continue
        value = global_config_dict.get(key)
        if isinstance(value, (Mapping, DictConfig)) and "responses_api_agents" in value:
            names.add(str(key))
    return names
