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
"""Probe: does abort_inflight actually terminate a request under the production middleware stack?

Production order (outermost first): exception_handling_middleware (BaseHTTPMiddleware, run_webserver),
AdmissionMiddleware, _CaptureMiddleware (pure ASGI), SessionMiddleware, add_session_id (BaseHTTPMiddleware).
"""

import asyncio
from asyncio import CancelledError

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym.checkpoint import GATED_MODEL_ROUTE_SUFFIXES, AdmissionLimiter, AdmissionMiddleware


async def main() -> None:
    app = FastAPI()
    limiter = AdmissionLimiter()
    admitted = asyncio.Event()
    handler_cancelled = asyncio.Event()

    @app.middleware("http")
    async def add_session_id(request: Request, call_next):
        request.session["session_id"] = "x"
        return await call_next(request)

    app.add_middleware(SessionMiddleware, secret_key="k")

    @app.post("/v1/responses")
    async def responses() -> dict:
        admitted.set()
        try:
            await asyncio.Event().wait()
        except CancelledError:
            handler_cancelled.set()
            raise
        return {"ok": True}

    app.add_middleware(AdmissionMiddleware, limiter=limiter, gated_suffixes=GATED_MODEL_ROUTE_SUFFIXES)

    @app.middleware("http")
    async def exception_handling_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except CancelledError:
            return JSONResponse(content="An unknown error occurred", status_code=500)
        except Exception as e:
            return JSONResponse(content=repr(e), status_code=500)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        req = asyncio.create_task(
            client.post(
                "/v1/responses",
                headers={"x-nemo-gym-rollout-id": "4-2", "x-nemo-gym-attempt-index": "0"},
                json={},
            )
        )
        await asyncio.wait_for(admitted.wait(), 5)
        limiter.close()
        aborted = limiter.abort_inflight("4-2", 0)
        print("aborted tickets:", len(aborted), "| limiter state:", limiter.state.value)
        try:
            response = await asyncio.wait_for(req, 5)
            print("client saw:", response.status_code, response.text[:80])
        except BaseException as e:
            print("client request raised:", type(e).__name__, e)
        print("handler cancelled:", handler_cancelled.is_set())
        print("inflight after abort:", limiter.counts()["inflight_total"], "| state:", limiter.state.value)


asyncio.run(main())
