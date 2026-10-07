import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_service.config import Settings
from inference_service.main import create_app, get_vllm_client
from inference_service.schemas import GenerateRequest
from inference_service.vllm_client import InferenceError, VLLMClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_health_and_client_lifecycle():
    settings = Settings(_env_file=None, vllm_model="test-model", vllm_api_key="test-secret")
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        http = app.state.vllm_client.client
        assert str(http.base_url) == "http://127.0.0.1:8003/v1/"
        assert http.timeout.read == 60
        assert http.headers["Authorization"] == "Bearer test-secret"
        assert not http.is_closed
    assert http.is_closed


@pytest.mark.anyio
async def test_generate_endpoint_with_mocked_vllm():
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "http://vllm.test/v1/completions"
        assert json.loads(request.content) == {
            "model": "test-model", "prompt": " prompt\n", "max_tokens": 32,
            "temperature": 0.0, "stream": False,
        }
        return httpx.Response(200, json={
            "model": "test-model", "choices": [{"text": "Generated answer", "finish_reason": "stop"}]
        })

    app = create_app(Settings(_env_file=None))
    async with httpx.AsyncClient(
        base_url="http://vllm.test/v1/", transport=httpx.MockTransport(handler)
    ) as upstream:
        client = VLLMClient(upstream, "test-model")
        app.dependency_overrides[get_vllm_client] = lambda: client
        async with httpx.AsyncClient(
            base_url="http://inference.test/", transport=httpx.ASGITransport(app=app)
        ) as http:
            response = await http.post("/v1/generate", json={
                "prompt": " prompt\n", "max_tokens": 32, "temperature": 0
            })
    assert response.status_code == 200
    assert response.json() == {"answer": "Generated answer", "model": "test-model", "finish_reason": "stop"}


@pytest.mark.parametrize("payload", [
    {}, {"prompt": ""}, {"prompt": "  "}, {"prompt": 123},
    {"prompt": "p", "max_tokens": 0}, {"prompt": "p", "max_tokens": 8193},
    {"prompt": "p", "max_tokens": True}, {"prompt": "p", "max_tokens": "32"},
    {"prompt": "p", "temperature": -0.1}, {"prompt": "p", "temperature": 2.1},
    {"prompt": "p", "extra": True},
])
def test_invalid_requests(payload):
    with TestClient(create_app(Settings(_env_file=None))) as client:
        assert client.post("/v1/generate", json=payload).status_code == 422


def test_defaults_and_prompt_preservation():
    request = GenerateRequest(prompt=" prompt\n")
    assert request.prompt == " prompt\n"
    assert request.max_tokens == 256
    assert request.temperature == 0.2


@pytest.mark.anyio
@pytest.mark.parametrize("failure,status", [
    ("timeout", 504), ("connect", 502), (400, 502), (401, 502), (429, 502), (500, 502),
    (302, 502), ("invalid-json", 502), ("empty-choices", 502), ("missing-text", 502),
])
async def test_vllm_failures_propagate_cleanly(failure, status):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private details", request=request)
        if failure == "connect":
            raise httpx.ConnectError("private details", request=request)
        if isinstance(failure, int):
            return httpx.Response(failure, text="private details")
        if failure == "invalid-json":
            return httpx.Response(200, text="private details")
        choices = [] if failure == "empty-choices" else [{}]
        return httpx.Response(200, json={"model": "test-model", "choices": choices})

    app = create_app(Settings(_env_file=None))
    async with httpx.AsyncClient(
        base_url="http://vllm.test/v1/", transport=httpx.MockTransport(handler)
    ) as upstream:
        client = VLLMClient(upstream, "test-model")
        app.dependency_overrides[get_vllm_client] = lambda: client
        async with httpx.AsyncClient(
            base_url="http://inference.test/", transport=httpx.ASGITransport(app=app)
        ) as http:
            response = await http.post("/v1/generate", json={"prompt": "hello"})
    assert response.status_code == status
    assert "private details" not in response.text
    assert "detail" in response.json()


def test_environment_settings(monkeypatch):
    monkeypatch.setenv("INFERENCE_VLLM_BASE_URL", "http://vllm.test/v1/")
    monkeypatch.setenv("INFERENCE_VLLM_MODEL", "configured-model")
    monkeypatch.setenv("INFERENCE_VLLM_TIMEOUT_SECONDS", "12")
    settings = Settings(_env_file=None)
    assert str(settings.vllm_base_url) == "http://vllm.test/v1/"
    assert settings.vllm_model == "configured-model"
    assert settings.vllm_timeout_seconds == 12


@pytest.mark.parametrize("values", [
    {"vllm_base_url": "invalid"}, {"vllm_model": ""}, {"vllm_timeout_seconds": 0}
])
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)
