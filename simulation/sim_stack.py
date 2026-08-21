# Edge-case simulation for the token-id-capture stack.
#
# A deterministic fake inference engine and a scripted blackbox harness drive the REAL
# Gym vllm_model server (middleware, lineage, capture, prefix supply) end to end, then
# the REAL consumer (trajectories_from_source) rebuilds the rollout. Every delivered
# trajectory is checked against the engine's own ground truth of what it processed —
# the "fabrication detector": training data must never contain a token sequence the
# engine never saw, and must not silently omit calls that fed the reward.
#
# Tokenization model: word-level deterministic tokenizer over a chat template that is
# concatenation-safe — a faithful echo re-renders to exactly prompt+generation, so the
# prefix property holds by construction, and every harness/engine mutation breaks it in
# the same way the real failure mode does (different bytes => different tokens).
#
# Run: .venv/bin/python simulation/sim_scenarios.py  (from the worktree root)

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import (
    TokenCaptureStore,
    TokenEntry,
    install_token_sink,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def tokenize(text: str) -> list[int]:
    """Deterministic word-level tokenizer. Different bytes => different ids.

    Newlines are their own tokens, which makes tokenization concatenation-safe:
    tokenize(a + "\\n" + b) == tokenize(a) + tokenize("\\n") + tokenize(b).
    """
    tokens = []
    for chunk in text.replace("\n", " \n ").split(" "):
        if not chunk:
            continue
        digest = hashlib.md5(chunk.encode("utf-8")).digest()
        tokens.append(int.from_bytes(digest[:4], "big") % 49000 + 1000)
    return tokens


def render_message(message: dict, drift: bool = False) -> str:
    """The engine's chat template for one message.

    ``drift=True`` re-renders an assistant turn with different whitespace — the
    retokenization-drift failure mode (same visible text, different tokens).
    """
    role = message.get("role", "")
    content = message.get("content") or ""
    if drift and role == "assistant":
        # Token-visible re-render drift: same conversation, different bytes per word.
        content = " ".join(f"{word}​" for word in content.split(" "))
    text = f"<|{role}|> {content}".rstrip()
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments") or ""
        try:  # the ENGINE canonicalizes args in its template
            args = json.dumps(json.loads(args), sort_keys=True)
        except (TypeError, ValueError):
            pass
        text += f" <|tool|> {call.get('id')} {fn.get('name')} {args}"
    return text + " <|end|>"


def render_prompt(messages: list[dict], drift_history: bool = False) -> str:
    parts = []
    for i, message in enumerate(messages):
        historical = message.get("role") == "assistant"
        parts.append(render_message(message, drift=drift_history and historical))
    return "\n".join(parts)


def generation_tokens(assistant_message: dict) -> list[int]:
    """The tokens the engine 'samples' for a reply: exactly what a faithful re-render adds."""
    return tokenize("\n" + render_message(assistant_message))


@dataclass
class SimEngine:
    """Deterministic fake vLLM. Records ground truth of every call it served."""

    honor_prefix: bool = True          # apply required_prefix_token_ids when present
    drift_on_rerender: bool = False    # re-tokenize historical assistant turns differently
    proof_shape: str = "top"           # "top" | "bundle" | "none" (where prompt ids appear)
    scripted: list[dict] = field(default_factory=list)  # per-call {content, tool_calls?, empty_generation?}
    truth: list[tuple[list[int], list[int]]] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)
    calls: int = 0

    async def create_chat_completion(self, **body: Any) -> dict:
        self.requests.append(body)
        messages = body["messages"]
        required = body.get("required_prefix_token_ids")
        if required and self.honor_prefix:
            # Extend the exact prefix; render only what follows the last assistant turn.
            last_assistant = max(
                (i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1
            )
            suffix = messages[last_assistant + 1 :]
            prompt_ids = list(required) + tokenize("\n" + render_prompt(suffix)) if suffix else list(required)
        else:
            prompt_ids = tokenize(render_prompt(messages, drift_history=self.drift_on_rerender))

        script = self.scripted[self.calls] if self.calls < len(self.scripted) else {"content": f"answer {self.calls}"}
        self.calls += 1
        message: dict[str, Any] = {"role": "assistant", "content": script.get("content", "")}
        if script.get("tool_calls"):
            message["tool_calls"] = script["tool_calls"]
        gen_ids = [] if script.get("empty_generation") else generation_tokens(message)
        self.truth.append((list(prompt_ids), list(gen_ids)))

        response: dict[str, Any] = {
            "id": f"sim-{self.calls}",
            "object": "chat.completion",
            "created": 0,
            "model": "sim_model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": list(gen_ids),
                    "message": message,
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{t}", "logprob": -0.1, "bytes": None, "top_logprobs": []}
                            for t in gen_ids
                        ]
                    },
                }
            ],
        }
        if self.proof_shape == "top":
            response["prompt_token_ids"] = list(prompt_ids)
        elif self.proof_shape == "bundle":
            message["prompt_token_ids"] = list(prompt_ids)
        return response

    async def create_tokenize(self, **body: Any) -> dict:
        return {"tokens": tokenize(render_prompt(body.get("messages") or [], drift_history=self.drift_on_rerender))}


def build_server(engine: SimEngine, capture_dir: str | None, supply: bool) -> TestClient:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="sim_model",
        return_token_id_information=True,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        supply_prefix_token_ids=supply,
    )
    capture_block: dict[str, Any] = {"enabled": True}
    if capture_dir is not None:
        capture_block["dir"] = capture_dir
    global_config = {"token_id_capture": capture_block}
    model = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config))
    mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
    mock_client.create_chat_completion = AsyncMock(side_effect=engine.create_chat_completion)
    mock_client.create_tokenize = AsyncMock(side_effect=engine.create_tokenize)
    model._clients = [mock_client]
    return TestClient(model.setup_webserver(), raise_server_exceptions=False)


CAPTURE_PATH = "/ng-rollout/{rid}/training-token-capture/v1/chat/completions"


@dataclass
class Harness:
    """Scripted blackbox harness: drives the server, echoes history with mutations."""

    client: TestClient
    rollout_id: str
    echo_mutator: Callable[[dict], dict] | None = None           # rewrite the echoed assistant turn
    pre_echo_insert: Callable[[int], dict | None] | None = None  # item inserted between context and echo
    history: list[dict] = field(default_factory=list)
    statuses: list[int] = field(default_factory=list)

    def call(self, messages: list[dict]) -> dict | None:
        response = self.client.post(CAPTURE_PATH.format(rid=self.rollout_id), json={"messages": messages})
        self.statuses.append(response.status_code)
        if response.status_code != 200:
            return None
        return response.json()["choices"][0]["message"]

    def echo_of(self, assistant_message: dict) -> dict:
        echoed = {k: v for k, v in assistant_message.items() if k in ("role", "content", "tool_calls") and v}
        echoed.setdefault("role", "assistant")
        if self.echo_mutator is not None:
            echoed = self.echo_mutator(dict(echoed))
        return echoed

    def turn(self, user_text: str) -> dict | None:
        """One user turn: append, call, absorb the (possibly mutated) echo into history."""
        self.history.append({"role": "user", "content": user_text})
        reply = self.call(list(self.history))
        if reply is None:
            self.history.pop()
            return None
        if self.pre_echo_insert is not None:
            inserted = self.pre_echo_insert(len(self.statuses))
            if inserted is not None:
                self.history.append(inserted)
        self.history.append(self.echo_of(reply))
        return reply


class FailingSink:
    """Delegates to a real store, but on call N both put and mark_incomplete fail.

    ``begin_call`` (the intent) still succeeds — this models a backend that died
    DURING the generation window, after the intent landed. The pre-dispatch death
    is modeled by ``PreDispatchFailingSink``.
    """

    def __init__(self, store: TokenCaptureStore, fail_on_call: int) -> None:
        self.store = store
        self.fail_on_call = fail_on_call
        self.calls = 0

    async def begin_call(self, rollout_id: str, model_call_id: str) -> None:
        begin = getattr(self.store, "begin_call", None)
        if begin is not None:
            await begin(rollout_id, model_call_id)

    async def put(self, entry: TokenEntry) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated storage outage (put)")
        await self.store.put(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated storage outage (mark_incomplete)")
        await self.store.mark_incomplete(rollout_id, model_call_id)

    async def close(self) -> None:
        pass


class PreDispatchFailingSink(FailingSink):
    """The backend is already down when call N arrives: even the intent fails."""

    def __init__(self, store: TokenCaptureStore, fail_on_call: int) -> None:
        super().__init__(store, fail_on_call)
        self.intents = 0

    async def begin_call(self, rollout_id: str, model_call_id: str) -> None:
        self.intents += 1
        if self.intents == self.fail_on_call:
            raise RuntimeError("simulated storage outage (begin_call)")
        await super().begin_call(rollout_id, model_call_id)

    async def put(self, entry: TokenEntry) -> None:
        await self.store.put(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        await self.store.mark_incomplete(rollout_id, model_call_id)


class MemorySink:
    """A custom sink WITHOUT a paired lineage store (the F3 misconfiguration)."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, TokenEntry]] = {}
        self.incomplete: set[str] = set()

    async def put(self, entry: TokenEntry) -> None:
        self.entries.setdefault(entry.rollout_id, {})[entry.model_call_id] = entry

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        self.incomplete.add(rollout_id)

    async def close(self) -> None:
        pass


class MemorySource:
    def __init__(self, sink: MemorySink) -> None:
        self.sink = sink

    async def freeze(self, rollout_id: str):
        from nemo_gym.token_id_capture import TokenCaptureSnapshot

        return TokenCaptureSnapshot(
            rollout_id=rollout_id,
            entries=tuple(self.sink.entries.get(rollout_id, {}).values()),
            incomplete=rollout_id in self.sink.incomplete,
            snapshot_id="mem",
            version=1,
        )

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        return True

    async def close(self) -> None:
        pass


class DuplicatingSource:
    """Wraps a real source but duplicates one entry in the snapshot (at-least-once transport)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def freeze(self, rollout_id: str):
        from nemo_gym.token_id_capture import TokenCaptureSnapshot

        snapshot = await self.inner.freeze(rollout_id)
        entries = list(snapshot.entries)
        if entries:
            entries.append(entries[0].model_copy(deep=True))
        return TokenCaptureSnapshot(
            rollout_id=snapshot.rollout_id,
            entries=tuple(entries),
            incomplete=snapshot.incomplete,
            snapshot_id=snapshot.snapshot_id,
            version=snapshot.version,
        )

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        return await self.inner.drop(rollout_id, snapshot_id=snapshot_id, version=version)

    async def close(self) -> None:
        await self.inner.close()


def verify_no_fabrication(built: dict, engine: SimEngine) -> tuple[bool, str]:
    """A delivered trajectory must be a contiguous chain of calls the engine served."""
    rebuilt = built.get("rebuilt_response")
    if rebuilt is None:
        return True, "nothing delivered"
    served = {tuple(prompt): tuple(gen) for prompt, gen in engine.truth}
    cumulative: list[int] = []
    delivered_calls = 0
    for item in rebuilt.get("output", []):
        prompt = item.get("prompt_token_ids")
        gen = item.get("generation_token_ids")
        if prompt is None or gen is None:
            continue
        delivered_calls += 1
        if tuple(prompt) not in served:
            return False, "FABRICATION: delivered a prompt the engine never processed"
        if served[tuple(prompt)] != tuple(gen):
            return False, "FABRICATION: delivered generation differs from what the engine sampled"
        if cumulative and list(prompt[: len(cumulative)]) != cumulative:
            return False, "FABRICATION: delivered chain is not contiguous"
        cumulative = list(prompt) + list(gen)
    return True, f"delivered {delivered_calls} call(s), all match engine ground truth"


def uninstall_sinks() -> None:
    install_token_sink(None)
