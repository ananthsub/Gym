"""Claude Code CLI smoke test against Gym's /v1/messages with the id-reuse change.

Retires the top risk in TERMINAL_ATTRIBUTION.md section 6.1: does the real CLI
accept a Messages envelope whose id is the reused inner `resp_...` id, over the
real streaming path — and does capture record the same id the CLI keeps?
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from unittest.mock import MagicMock

import uvicorn
from omegaconf import DictConfig

import nemo_gym.global_config as global_config_module
import nemo_gym.server_utils as server_utils_module  # noqa: F401
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import TokenCaptureStore
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


global_config_module._GLOBAL_CONFIG_DICT = DictConfig({})

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "unit_tests"))
from test_capture_fake_backend_e2e import build_fake_backend  # noqa: E402


def serve(app) -> tuple[uvicorn.Server, int]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        pass
    return server, server.servers[0].sockets[0].getsockname()[1]


def main() -> int:
    capture_dir = tempfile.mkdtemp(prefix="cc-smoke-capture-")
    config_dir = tempfile.mkdtemp(prefix="cc-smoke-config-")
    _, backend_port = serve(build_fake_backend())

    gym = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="",
            name="policy_model",
            base_url=f"http://127.0.0.1:{backend_port}/v1",
            api_key="dummy",
            model="fake-model",
            return_token_id_information=True,
            uses_reasoning_parser=False,
        ),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"token_id_capture": {"enabled": True, "dir": capture_dir}},
        ),
    )
    _, gym_port = serve(gym.setup_webserver())

    rid = "cc-smoke-0-0"
    base_url = f"http://127.0.0.1:{gym_port}/ng-rollout/{rid}/training-token-capture"
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "dummy",
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "fake-model",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "fake-model",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "fake-model",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "fake-model",
        "CLAUDE_CODE_SUBAGENT_MODEL": "fake-model",
        "IS_SANDBOX": "1",
        "CLAUDE_CONFIG_DIR": config_dir,
    }
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--bare",
        "--model",
        "fake-model",
        "Reply with one word.",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=150)
    print("--- exit code:", proc.returncode)
    stream_ids = []
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if isinstance(message, dict) and message.get("id"):
            stream_ids.append(message["id"])
    print("--- message ids seen by the CLI:", stream_ids)
    if proc.returncode != 0:
        print("--- stderr tail:", proc.stderr[-2000:])
        print("--- stdout tail:", proc.stdout[-2000:])

    entries = TokenCaptureStore(capture_dir).read_entries(rid)
    captured = [(entry.model_call_id, entry.response_id) for entry in entries]
    print("--- captured (model_call_id, response_id):", captured)

    ok = (
        proc.returncode == 0
        and captured
        and all(response_id and response_id.startswith("resp_") for _, response_id in captured)
        and any(response_id in stream_ids for _, response_id in captured)
    )
    print("--- SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
