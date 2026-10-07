import json

import httpx
import pytest

from api_gateway.config import Settings
from api_gateway.inference_client import InferenceClient, InferenceError
from api_gateway.main import create_app, get_inference_client, get_retrieval_client
from api_gateway.prompts import build_context, build_prompt
from api_gateway.retrieval_client import RetrievalClient
from api_gateway.schemas import GenerateRequest, RetrievalMatch
from inference_service.config import Settings as InferenceSettings
from inference_service.main import create_app as create_inference_app, get_vllm_client
from inference_service.vllm_client import VLLMClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("empty", [False, True])
async def test_full_pipeline_with_mocked_vllm(empty):
    query = "What is RAG?"
    matches = [] if empty else [RetrievalMatch(document_id="doc", text="RAG retrieves context.", score=0.9)]
    prompt = build_prompt(query, build_context(matches))

    def retrieval_handler(request):
        assert str(request.url) == "http://retrieval.test/v1/retrieve"
        assert json.loads(request.content) == {"query": query, "top_k": 2}
        return httpx.Response(200, json={"matches": [match.model_dump() for match in matches]})

    def vllm_handler(request):
        assert str(request.url) == "http://vllm.test/v1/completions"
        assert json.loads(request.content) == {
            "model": "test-model", "prompt": prompt, "max_tokens": 64,
            "temperature": 0.1, "stream": False,
        }
        return httpx.Response(200, json={
            "model": "test-model", "choices": [{"text": "Model answer", "finish_reason": "stop"}]
        })

    inference_app = create_inference_app(InferenceSettings(_env_file=None))
    gateway_app = create_app(Settings(_env_file=None))
    async with httpx.AsyncClient(
        base_url="http://retrieval.test/", transport=httpx.MockTransport(retrieval_handler)
    ) as retrieval_http, httpx.AsyncClient(
        base_url="http://vllm.test/v1/", transport=httpx.MockTransport(vllm_handler)
    ) as vllm_http, httpx.AsyncClient(
        base_url="http://inference.test/", transport=httpx.ASGITransport(app=inference_app)
    ) as inference_http:
        vllm_client = VLLMClient(vllm_http, "test-model")
        retrieval_client = RetrievalClient(retrieval_http)
        inference_client = InferenceClient(inference_http)
        inference_app.dependency_overrides[get_vllm_client] = lambda: vllm_client
        gateway_app.dependency_overrides[get_retrieval_client] = lambda: retrieval_client
        gateway_app.dependency_overrides[get_inference_client] = lambda: inference_client
        async with httpx.AsyncClient(
            base_url="http://gateway.test/", transport=httpx.ASGITransport(app=gateway_app)
        ) as gateway:
            response = await gateway.post("/v1/query", json={
                "query": query, "top_k": 2, "max_tokens": 64, "temperature": 0.1
            })
    assert response.status_code == 200
    assert response.json() == {
        "status": "completed", "answer": "Model answer",
        "sources": [match.model_dump() for match in matches],
    }


@pytest.mark.anyio
@pytest.mark.parametrize("failure,status", [
    ("timeout", 504), ("connect", 502), (504, 504), (500, 502), (422, 502),
    ("bad-json", 502), ("bad-schema", 502),
])
async def test_gateway_inference_errors(failure, status):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private detail", request=request)
        if failure == "connect":
            raise httpx.ConnectError("private detail", request=request)
        if isinstance(failure, int):
            return httpx.Response(failure, text="private detail")
        return httpx.Response(200, text="not json" if failure == "bad-json" else "{}")

    app = create_app(Settings(_env_file=None))
    async with httpx.AsyncClient(
        base_url="http://retrieval.test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"matches": []})),
    ) as retrieval_http, httpx.AsyncClient(
        base_url="http://inference.test/", transport=httpx.MockTransport(handler)
    ) as inference_http:
        retrieval = RetrievalClient(retrieval_http)
        inference = InferenceClient(inference_http)
        app.dependency_overrides[get_retrieval_client] = lambda: retrieval
        app.dependency_overrides[get_inference_client] = lambda: inference
        async with httpx.AsyncClient(
            base_url="http://gateway.test/", transport=httpx.ASGITransport(app=app)
        ) as http:
            response = await http.post("/v1/query", json={"query": "hello"})
    assert response.status_code == status
    assert "private detail" not in response.text


def test_gateway_inference_settings(monkeypatch):
    monkeypatch.setenv("RAG_INFERENCE_SERVICE_URL", "http://inference.test/")
    monkeypatch.setenv("RAG_INFERENCE_TIMEOUT_SECONDS", "100")
    settings = Settings(_env_file=None)
    assert str(settings.inference_service_url) == "http://inference.test/"
    assert settings.inference_timeout_seconds == 100
