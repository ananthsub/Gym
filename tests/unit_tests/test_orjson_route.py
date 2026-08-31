# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import orjson
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import GymORJSONResponse, ORJSONRoute, ServerClient, install_orjson_serving


class _Payload(BaseModel):
    name: str
    values: list[int]


def _app() -> FastAPI:
    app = FastAPI()
    install_orjson_serving(app)
    return app


def test_annotated_route_keeps_fastapi_response_contract():
    """Annotated routes serve through FastAPI's own pydantic-core path, untouched."""
    app = _app()

    @app.post("/echo", status_code=201)
    async def echo(body: _Payload) -> _Payload:
        return body

    payload = {"name": "x", "values": list(range(32))}
    with TestClient(app) as client:
        resp = client.post("/echo", json=payload)
    assert resp.status_code == 201
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == payload


def test_aliased_fields_keep_their_wire_names():
    """FastAPI serializes by alias; the serving changes must not alter that."""

    class Aliased(BaseModel):
        schema_: dict = Field(alias="schema")

        model_config = {"populate_by_name": True}

    app = _app()

    @app.post("/aliased")
    async def aliased() -> Aliased:
        return Aliased(schema_={"type": "object"})

    with TestClient(app) as client:
        served = client.post("/aliased").json()
    assert "schema" in served and "schema_" not in served


def test_base_annotation_filters_subclass_fields_like_fastapi():
    class _BasePayload(BaseModel):
        a: int

    class _SubPayload(_BasePayload):
        secret: str = "hidden"

    app = _app()

    @app.post("/filtered")
    async def filtered() -> _BasePayload:
        return _SubPayload(a=1)

    @app.post("/unfiltered")
    async def unfiltered() -> _SubPayload:
        return _SubPayload(a=2)

    with TestClient(app) as client:
        assert client.post("/filtered").json() == {"a": 1}
        assert client.post("/unfiltered").json() == {"a": 2, "secret": "hidden"}


def test_unannotated_dict_route_renders_with_orjson_and_falls_back():
    """Dict-returning routes get GymORJSONResponse; orjson-rejected values fall back."""
    import decimal

    app = _app()

    @app.post("/dict")
    async def dict_route():
        return {"ok": True}

    @app.post("/compat")
    async def compat():
        # jsonable_encoder coerces Decimal/set before render; a lone surrogate
        # survives into render, where the stdlib fallback must serve it.
        return {"text": "truncated \ud83d", "big": 2**70, "d": decimal.Decimal("1.5"), "s": {1}}

    @app.post("/stream")
    async def stream_route():
        return StreamingResponse(iter([b"a", b"b"]), media_type="text/event-stream")

    with TestClient(app) as client:
        assert client.post("/dict").content == orjson.dumps({"ok": True})
        resp = client.post("/compat")
        assert resp.status_code == 200
        assert resp.json() == {"text": "truncated \ud83d", "big": 2**70, "d": 1.5, "s": [1]}
        stream = client.post("/stream")
        assert stream.content == b"ab"
        assert stream.headers["content-type"].startswith("text/event-stream")


def test_explicit_response_class_is_respected():
    app = _app()

    @app.post("/text", response_class=PlainTextResponse)
    async def text_route() -> str:
        return "plain"

    with TestClient(app) as client:
        resp = client.post("/text")
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "plain"


def test_ingress_parses_with_orjson_and_falls_back_for_nonstandard_literals():
    app = _app()

    @app.post("/ingest")
    async def ingest(body: dict = Body()):
        return {"is_nan": body["v"] != body["v"]}

    nan_payload = json.dumps({"v": float("nan")})
    with TestClient(app) as client:
        resp = client.post("/ingest", content=nan_payload, headers={"Content-Type": "application/json"})
        assert resp.status_code == 200
        assert resp.json() == {"is_nan": True}
        assert client.post("/ingest", json={"v": 1.5}).json() == {"is_nan": False}


def test_kill_switch_leaves_stock_fastapi_serving(monkeypatch):
    import nemo_gym.server_utils as su

    monkeypatch.setattr(su, "_ORJSON_SERVING_DISABLED", True)
    app = FastAPI()
    su.install_orjson_serving(app)
    assert app.router.route_class is not su.ORJSONRoute


class _Res(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


def test_setup_webserver_installs_route_and_response_classes_and_serves_verify():
    server = _Res(
        config=BaseResourcesServerConfig(name="res", host="127.0.0.1", port=9, entrypoint="app.py"),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=10),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    assert app.router.route_class is ORJSONRoute
    assert app.router.default_response_class is GymORJSONResponse

    body = {
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
        "response": {
            "id": "resp_1",
            "created_at": 1.0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [],
        },
    }
    with TestClient(app) as client:
        resp = client.post("/verify", json=body)
    assert resp.status_code == 200, resp.text
    parsed = resp.json()
    assert parsed["reward"] == 1.0
    assert parsed["response"]["id"] == "resp_1"
