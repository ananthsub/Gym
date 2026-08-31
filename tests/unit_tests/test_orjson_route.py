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
from inspect import iscoroutinefunction

import orjson
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ORJSONRoute, ServerClient, install_orjson_serving


class _Payload(BaseModel):
    name: str
    values: list[int]


def _app() -> FastAPI:
    app = FastAPI()
    install_orjson_serving(app)
    return app


def test_model_return_serializes_with_orjson_and_skips_response_model():
    app = _app()

    @app.post("/echo")
    async def echo(body: _Payload) -> _Payload:
        return body

    payload = {"name": "x", "values": list(range(32))}
    with TestClient(app) as client:
        resp = client.post("/echo", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    # Byte-for-byte the orjson serialization of the model, not FastAPI's stdlib encode.
    assert resp.content == orjson.dumps(payload)


def test_dict_and_streaming_and_sync_returns():
    app = _app()

    @app.post("/dict")
    async def dict_route():
        return {"ok": True}

    @app.post("/stream")
    async def stream_route():
        return StreamingResponse(iter([b"a", b"b"]), media_type="text/event-stream")

    @app.post("/sync")
    def sync_route():
        return {"sync": 1}

    # A sync endpoint stays sync so FastAPI keeps threadpooling it.
    sync_api_route = next(r for r in app.routes if getattr(r, "path", None) == "/sync")
    assert not iscoroutinefunction(sync_api_route.endpoint)

    with TestClient(app) as client:
        assert client.post("/dict").content == orjson.dumps({"ok": True})
        stream = client.post("/stream")
        assert stream.content == b"ab"
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert client.post("/sync").content == orjson.dumps({"sync": 1})


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

    # stdlib json emits the non-standard NaN literal; orjson rejects it, the fallback accepts it.
    nan_payload = json.dumps({"v": float("nan")})
    with TestClient(app) as client:
        resp = client.post("/ingest", content=nan_payload, headers={"Content-Type": "application/json"})
        assert resp.status_code == 200
        assert resp.json() == {"is_nan": True}
        assert client.post("/ingest", json={"v": 1.5}).json() == {"is_nan": False}


class _Res(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


def test_setup_webserver_installs_route_class_and_serves_verify_via_orjson():
    server = _Res(
        config=BaseResourcesServerConfig(name="res", host="127.0.0.1", port=9, entrypoint="app.py"),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=10),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    assert app.router.route_class is ORJSONRoute

    verify_route = next(r for r in app.routes if getattr(r, "path", None) == "/verify")
    # Later route wrappers (MCP auto-exposure) unwrap to the model-returning handler via this marker.
    assert hasattr(verify_route.endpoint, "__nemo_gym_orjson_inner__")

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


class _BasePayload(BaseModel):
    a: int


class _SubPayload(_BasePayload):
    secret: str = "hidden"


def test_base_annotation_filters_subclass_fields_like_fastapi():
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


def test_exotic_return_falls_back_to_jsonable_encoder():
    class _Odd:
        pass

    app = _app()

    @app.post("/odd")
    async def odd():
        return {"obj": _Odd()}

    with TestClient(app) as client:
        resp = client.post("/odd")
    # jsonable_encoder coerces via vars(); the request must not 500.
    assert resp.status_code == 200
    assert resp.json() == {"obj": {}}


def test_numpy_scalars_serialize():
    np = __import__("pytest").importorskip("numpy")
    app = _app()

    @app.post("/np")
    async def np_route():
        return {"done": np.bool_(True), "score": np.float64(0.5)}

    with TestClient(app) as client:
        assert client.post("/np").json() == {"done": True, "score": 0.5}
