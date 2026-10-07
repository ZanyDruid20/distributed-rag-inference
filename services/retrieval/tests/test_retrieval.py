import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from pydantic import ValidationError

from retrieval_service.config import Settings
from retrieval_service.main import create_app
from retrieval_service.schemas import DocumentRequest, RetrieveRequest
from retrieval_service.service import retrieve
from retrieval_service import service
from retrieval_service.schemas import RetrievalMatch


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "search", Mock(return_value=[]))
    with TestClient(create_app(Settings(_env_file=None, environment="test"))) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_retrieve_endpoint(client):
    response = client.post("/v1/retrieve", json={"query": "A query", "top_k": 3})
    assert response.status_code == 200
    assert response.json() == {"matches": []}
    service.search.assert_called_once_with("A query", 3)


@pytest.mark.parametrize("payload", [
    {}, {"text": "  "}, {"text": 123}, {"text": "x" * 1_000_001},
    {"text": "hello", "title": ""}, {"text": "hello", "extra": True},
])
def test_invalid_document(client, payload):
    assert client.post("/v1/documents", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {}, {"query": " "}, {"query": 123}, {"query": "x" * 10_001},
    {"query": "hello", "top_k": 0}, {"query": "hello", "top_k": 101},
    {"query": "hello", "top_k": "5"}, {"query": "hello", "top_k": True},
    {"query": "hello", "extra": True},
])
def test_invalid_retrieve(client, payload):
    assert client.post("/v1/retrieve", json=payload).status_code == 422


def test_request_defaults():
    assert RetrieveRequest(query=" hello ").query == "hello"
    assert RetrieveRequest(query="hello").top_k == 5
    assert DocumentRequest(text=" hello ").text == "hello"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_async_retrieval_service(monkeypatch):
    match = RetrievalMatch(document_id="doc", text="chunk", score=0.8)
    search = Mock(return_value=[match])
    monkeypatch.setattr(service, "search", search)
    response = await retrieve(RetrieveRequest(query="hello", top_k=2))
    assert response.matches == [match]
    search.assert_called_once_with("hello", 2)


def test_settings_isolated_from_gateway(monkeypatch):
    monkeypatch.setenv("RAG_APP_NAME", "Gateway")
    monkeypatch.setenv("RETRIEVAL_APP_NAME", "Test Retrieval")
    monkeypatch.setenv("RETRIEVAL_ENVIRONMENT", "test")
    settings = Settings(_env_file=None)
    assert create_app(settings).title == "Test Retrieval"
    assert settings.environment == "test"


def test_invalid_environment(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_ENVIRONMENT", "invalid")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openapi_contracts(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/v1/documents", "/v1/retrieve"):
        assert "200" in paths[path]["post"]["responses"]
    assert "501" not in paths["/v1/retrieve"]["post"]["responses"]
    assert "501" not in paths["/v1/documents"]["post"]["responses"]
