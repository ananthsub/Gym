# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic resources server for scale-sim.

Implements `/verify` and `/synthetic_tool` with controllable latency, body size,
body shape, CPU work, and failure injection. Used by the scale-sim harness to
characterize where the nemo-gym architecture breaks under load. See
`tools/scale_sim/README.md` for the test plan.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


BodyShape = Literal["flat_padding", "realistic_messages"]
LatencyDist = Literal["fixed", "lognormal"]


class SyntheticEndpointConfig(BaseModel):
    """Per-endpoint knobs, applied identically to /synthetic_tool and /verify."""

    async_latency_ms: float = Field(
        default=0.0,
        description="`await asyncio.sleep(...)` — yields the event loop. Models I/O-bound waits.",
    )
    cpu_burn_ms: float = Field(
        default=0.0,
        description="`time.perf_counter()` busy-loop — holds the event loop. Models CPU-bound handler work.",
    )
    latency_dist: LatencyDist = Field(default="fixed")
    latency_lognormal_mu: float = 0.0
    latency_lognormal_sigma: float = 0.0
    body_size_bytes: int = Field(
        default=1024,
        description="Total byte length of the response body's variable portion.",
    )
    body_dist: LatencyDist = Field(default="fixed")
    body_lognormal_mu: float = 10.0
    body_lognormal_sigma: float = 2.0
    body_shape: BodyShape = Field(
        default="flat_padding",
        description=(
            "`flat_padding`: one big `data` string. Cheap orjson, single allocation. "
            "`realistic_messages`: a list of dict chunks, exercises real parse + GC."
        ),
    )
    realistic_chunk_bytes: int = Field(
        default=256,
        description="Approximate per-chunk size when body_shape=realistic_messages.",
    )


class SyntheticResourcesServerConfig(BaseResourcesServerConfig):
    tool: SyntheticEndpointConfig = Field(default_factory=SyntheticEndpointConfig)
    verify: SyntheticEndpointConfig = Field(default_factory=SyntheticEndpointConfig)

    inject_500_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="P(handler returns HTTP 500)")
    inject_close_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="P(handler returns truncated body to simulate mid-body RST). 0.0 today (placeholder).",
    )
    inject_hang_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="P(handler awaits effectively forever) — simulates wedged sub-server.",
    )
    inject_hang_seconds: float = Field(
        default=3600.0,
        description="Sleep duration when inject_hang fires.",
    )

    seed: int = Field(default=0, description="RNG seed for reproducibility within one process.")


class SyntheticToolRequest(BaseModel):
    # Tool args — accept anything since synthetic_model emits empty `{}`
    model_config = {"extra": "allow"}


def _sample_latency_seconds(cfg: SyntheticEndpointConfig, rng: random.Random) -> float:
    if cfg.latency_dist == "fixed":
        return cfg.async_latency_ms / 1000.0
    return rng.lognormvariate(cfg.latency_lognormal_mu, cfg.latency_lognormal_sigma) / 1000.0


def _sample_body_size(cfg: SyntheticEndpointConfig, rng: random.Random) -> int:
    if cfg.body_dist == "fixed":
        return cfg.body_size_bytes
    return max(0, int(rng.lognormvariate(cfg.body_lognormal_mu, cfg.body_lognormal_sigma)))


def _build_padding_body(target_bytes: int, shape: BodyShape, chunk_bytes: int) -> Dict[str, Any]:
    """Build a JSON-serializable dict whose serialized form is approximately `target_bytes`.

    `flat_padding` is one big string field (cheap parse, single allocation).
    `realistic_messages` is a list of small dict chunks (drives real parse + GC).
    """
    if target_bytes <= 0:
        return {"data": ""}

    if shape == "flat_padding":
        return {"data": "x" * target_bytes}

    # realistic_messages: many small dict allocations.
    # Each chunk is a dict like {"role": "tool", "content": "<text>", "idx": N}; serialized overhead ~40B per chunk.
    n_chunks = max(1, target_bytes // max(1, chunk_bytes))
    payload_per_chunk = max(1, chunk_bytes - 40)
    chunk_text = "x" * payload_per_chunk
    chunks: List[Dict[str, Any]] = [{"role": "tool", "content": chunk_text, "idx": i} for i in range(n_chunks)]
    return {"messages": chunks}


def _cpu_burn(milliseconds: float) -> None:
    """Busy-loop for the requested duration. Holds the event loop on purpose."""
    if milliseconds <= 0:
        return
    deadline = time.perf_counter() + (milliseconds / 1000.0)
    # Inline busy-loop. Touches `time.perf_counter()` so it isn't optimized away.
    while time.perf_counter() < deadline:
        pass


class SyntheticResourcesServer(SimpleResourcesServer):
    config: SyntheticResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._rng = random.Random(self.config.seed)
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/synthetic_tool")(self.synthetic_tool)
        return app

    async def _maybe_inject_failure(self) -> JSONResponse | None:
        cfg = self.config
        if cfg.inject_hang_rate > 0 and self._rng.random() < cfg.inject_hang_rate:
            await asyncio.sleep(cfg.inject_hang_seconds)
        if cfg.inject_500_rate > 0 and self._rng.random() < cfg.inject_500_rate:
            return JSONResponse(content={"error": "synthetic_500"}, status_code=500)
        return None

    async def _do_endpoint_work(self, endpoint_cfg: SyntheticEndpointConfig) -> Dict[str, Any]:
        latency_s = _sample_latency_seconds(endpoint_cfg, self._rng)
        if latency_s > 0:
            await asyncio.sleep(latency_s)
        _cpu_burn(endpoint_cfg.cpu_burn_ms)
        body_size = _sample_body_size(endpoint_cfg, self._rng)
        return _build_padding_body(
            target_bytes=body_size,
            shape=endpoint_cfg.body_shape,
            chunk_bytes=endpoint_cfg.realistic_chunk_bytes,
        )

    async def synthetic_tool(self, request: Request, body: SyntheticToolRequest) -> JSONResponse:
        failure = await self._maybe_inject_failure()
        if failure is not None:
            return failure
        payload = await self._do_endpoint_work(self.config.tool)
        return JSONResponse(content=payload)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        failure = await self._maybe_inject_failure()
        if failure is not None:
            # /verify must return a BaseVerifyResponse shape. Surface failure as reward=0.
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)
        # Do the same work pattern as /synthetic_tool but return a proper verify response.
        # The padding goes in a top-level field that isn't part of BaseVerifyResponse — we
        # extend the model with extra="allow" by wrapping with model_dump.
        latency_s = _sample_latency_seconds(self.config.verify, self._rng)
        if latency_s > 0:
            await asyncio.sleep(latency_s)
        _cpu_burn(self.config.verify.cpu_burn_ms)
        target_bytes = _sample_body_size(self.config.verify, self._rng)
        padding = _build_padding_body(
            target_bytes=target_bytes,
            shape=self.config.verify.body_shape,
            chunk_bytes=self.config.verify.realistic_chunk_bytes,
        )
        # Pydantic models from BaseVerifyResponse don't allow extra by default; serialize
        # ourselves to attach the padding field for size control.
        base = BaseVerifyResponse(**body.model_dump(), reward=1.0).model_dump()
        base["synthetic_padding"] = padding
        return JSONResponse(content=base)


if __name__ == "__main__":
    SyntheticResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # uvicorn worker re-import path; see vllm_model/app.py for the pattern.
    app = SyntheticResourcesServer.run_webserver()  # noqa: F401
