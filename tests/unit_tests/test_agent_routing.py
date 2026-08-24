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
import json
from asyncio import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.rollout_collection
from nemo_gym.agent_routing import agent_server_names, load_agent_routing_map, resolve_agent_name
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ENVIRONMENTS_KEY_NAME, RESOLVED_AGENT_REF_KEY_NAME
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


AGENTS_CONFIG = {
    "math_simple_agent": {"responses_api_agents": {"simple_agent": {}}},
    "swe_agent": {"responses_api_agents": {"mini_swe_agent": {}}},
    "math_resources": {"resources_servers": {"math_with_judge": {}}},
    "head_server": {"host": "127.0.0.1", "port": 11000},
}


class TestLoadAgentRoutingMap:
    def test_absent_section_returns_empty(self) -> None:
        assert load_agent_routing_map(AGENTS_CONFIG) == {}
        assert load_agent_routing_map({}) == {}

    def test_non_mapping_config_returns_empty(self) -> None:
        assert load_agent_routing_map(MagicMock()) == {}
        assert load_agent_routing_map(None) == {}

    def test_valid_map(self) -> None:
        config = {**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: {"math_with_judge": {"agent": "math_simple_agent"}}}
        assert load_agent_routing_map(config) == {"math_with_judge": "math_simple_agent"}

    def test_dictconfig_input(self) -> None:
        config = OmegaConf.create(
            {**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: {"math_with_judge": {"agent": "math_simple_agent"}}}
        )
        assert load_agent_routing_map(config) == {"math_with_judge": "math_simple_agent"}

    def test_section_not_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            load_agent_routing_map({**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: "math_simple_agent"})

    def test_entry_missing_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="math_with_judge"):
            load_agent_routing_map({**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: {"math_with_judge": {}}})

    def test_entry_not_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="math_with_judge"):
            load_agent_routing_map({**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: {"math_with_judge": "math_simple_agent"}})

    def test_entry_extra_keys_raises(self) -> None:
        section = {"math_with_judge": {"agent": "math_simple_agent", "agnet": "typo"}}
        with pytest.raises(ValueError, match="math_with_judge"):
            load_agent_routing_map({**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: section})

    def test_unknown_agent_target_raises_and_lists_configured_agents(self) -> None:
        config = {**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: {"math_with_judge": {"agent": "not_a_server"}}}
        with pytest.raises(ValueError, match="not_a_server") as exc_info:
            load_agent_routing_map(config)
        assert "math_simple_agent" in str(exc_info.value)


class TestResolveAgentName:
    def test_missing_agent_ref_returns_none(self) -> None:
        assert resolve_agent_name({}, {}) is None
        assert resolve_agent_name({AGENT_REF_KEY_NAME: None}, {}) is None
        assert resolve_agent_name({AGENT_REF_KEY_NAME: {}}, {"a": "b"}) is None

    def test_unmapped_key_routes_verbatim(self) -> None:
        row = {AGENT_REF_KEY_NAME: {"name": "swe_agent"}}
        assert resolve_agent_name(row, {}) == "swe_agent"
        assert resolve_agent_name(row, {"math_with_judge": "math_simple_agent"}) == "swe_agent"

    def test_mapped_key_routes_to_agent(self) -> None:
        row = {AGENT_REF_KEY_NAME: {"name": "math_with_judge"}}
        assert resolve_agent_name(row, {"math_with_judge": "math_simple_agent"}) == "math_simple_agent"


class TestAgentServerNames:
    def test_only_agent_servers_counted(self) -> None:
        assert agent_server_names(AGENTS_CONFIG) == {"math_simple_agent", "swe_agent"}

    def test_reserved_keys_skipped(self) -> None:
        config = {**AGENTS_CONFIG, ENVIRONMENTS_KEY_NAME: {"x": {"agent": "math_simple_agent"}}}
        assert agent_server_names(config) == {"math_simple_agent", "swe_agent"}


class TestRoutingEndToEnd:
    async def test_run_from_config_remaps_and_stamps_resolved_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mapped routing key dispatches to the mapped agent and stamps resolved_agent_ref on the
        result; an unmapped key routes verbatim and its result stays byte-identical to today's."""
        global_config = {
            **AGENTS_CONFIG,
            ENVIRONMENTS_KEY_NAME: {"math_with_judge": {"agent": "math_simple_agent"}},
        }
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "get_global_config_dict", MagicMock(return_value=global_config)
        )

        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "math_with_judge"}}),
            json.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "swe_agent"}}),
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
        )

        dispatched_rows = []

        class Helper(RolloutCollectionHelper):
            def run_examples(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    dispatched_rows.append(example)
                    future = Future()
                    future.set_result((example, {"reward": 1.0}))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        results = await Helper().run_from_config(config)

        by_routing_key = {r[AGENT_REF_KEY_NAME]["name"]: r for r in results}
        # The routing key stays on agent_ref for grouping; the remapped destination is recorded.
        assert by_routing_key["math_with_judge"][RESOLVED_AGENT_REF_KEY_NAME] == {"name": "math_simple_agent"}
        # An unmapped row's result carries no resolved_agent_ref at all.
        assert RESOLVED_AGENT_REF_KEY_NAME not in by_routing_key["swe_agent"]
        assert len(dispatched_rows) == 2
