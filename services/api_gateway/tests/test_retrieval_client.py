from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api_gateway.config import Settings
from api_gateway.main import create_app, get_retrieval_client, get_inference_client
from api_gateway.retrieval_client import RetrievalClient, RetrievalError
from api_gateway.schemas import RetrieveResponse


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_http_request_and_response():
    import json

    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "http://retrieval.test/v1/retrieve"
        assert json.loads(request.content) == {"query": "hello", "top_k": 3}
        assert request.extensions["timeout"]["read"] == 7.0
        return httpx.Response(200, json={"matches": [
            {"document_id": "doc-1", "text": "chunk", "score": 0.9}
        ]})

    async with httpx.AsyncClient(
        base_url="http://retrieval.test/", timeout=7, transport=httpx.MockTransport(handler)
    ) as http:
        response = await RetrievalClient(http).retrieve("hello", 3)
    assert response.matches[0].model_dump() == {
        "document_id": "doc-1", "text": "chunk", "score": 0.9
    }


@pytest.mark.anyio
async def test_empty_matches():
    async with httpx.AsyncClient(
        base_url="http://retrieval.test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"matches": []})),
    ) as http:
        assert (await RetrievalClient(http).retrieve("hello")).matches == []


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 422, 500, 501, 503])
async def test_upstream_error(status):
    async with httpx.AsyncClient(
        base_url="http://retrieval.test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="private detail")),
    ) as http:
        with pytest.raises(RetrievalError) as error:
            await RetrievalClient(http).retrieve("hello")
    assert error.value.status_code == 502
    assert "private detail" not in error.value.detail


@pytest.mark.anyio
@pytest.mark.parametrize("exception,status", [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)])
async def test_transport_errors(exception, status):
    def handler(request):
        raise exception("private detail", request=request)

    async with httpx.AsyncClient(
        base_url="http://retrieval.test/", transport=httpx.MockTransport(handler)
    ) as http:
        with pytest.raises(RetrievalError) as error:
            await RetrievalClient(http).retrieve("hello")
    assert error.value.status_code == status


@pytest.mark.anyio
@pytest.mark.parametrize("body", ["not json", "{}", '{"matches":[{"document_id":"x"}]}'])
async def test_invalid_response(body):
    async with httpx.AsyncClient(
        base_url="http://retrieval.test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
    ) as http:
        with pytest.raises(RetrievalError, match="invalid response") as error:
            await RetrievalClient(http).retrieve("hello")
    assert error.value.status_code == 502


def test_gateway_returns_sources(inference_client):
    client = AsyncMock()
    client.retrieve.return_value = RetrieveResponse(matches=[
        {"document_id": "doc", "text": "retrieved text", "score": 0.75}
    ])
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_retrieval_client] = lambda: client
    app.dependency_overrides[get_inference_client] = lambda: inference_client
    with TestClient(app) as http:
        response = http.post("/v1/query", json={"query": "question", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["sources"] == [
        {"document_id": "doc", "text": "retrieved text", "score": 0.75}
    ]
    client.retrieve.assert_awaited_once_with("question", 2)


@pytest.mark.parametrize("status", [502, 504])
def test_gateway_maps_errors(status):
    client = AsyncMock()
    client.retrieve.side_effect = RetrievalError(status, "Retrieval failed")
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_retrieval_client] = lambda: client
    with TestClient(app) as http:
        response = http.post("/v1/query", json={"query": "hello"})
    assert response.status_code == status
    assert response.json() == {"detail": "Retrieval failed"}


def test_lifespan_configures_and_closes_client():
    app = create_app(Settings(
        _env_file=None, retrieval_service_url="http://retrieval.test/", retrieval_timeout_seconds=7
    ))
    with TestClient(app):
        http = app.state.retrieval_client.client
        assert str(http.base_url) == "http://retrieval.test/"
        assert http.timeout.read == 7
        assert not http.is_closed
    assert http.is_closed


def test_retrieval_configuration(monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_SERVICE_URL", "http://retrieval.test:8001")
    monkeypatch.setenv("RAG_RETRIEVAL_TIMEOUT_SECONDS", "4")
    settings = Settings(_env_file=None)
    assert str(settings.retrieval_service_url) == "http://retrieval.test:8001/"
    assert settings.retrieval_timeout_seconds == 4


@pytest.mark.parametrize("overrides", [
    {"retrieval_service_url": "not-a-url"}, {"retrieval_timeout_seconds": 0}
])
def test_invalid_configuration(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)
