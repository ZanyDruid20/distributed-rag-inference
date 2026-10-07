import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api_gateway.config import Settings
from api_gateway.main import create_app, get_retrieval_client, get_inference_client
from api_gateway.schemas import QueryRequest, RetrieveResponse
from api_gateway.service import process_query


@pytest.fixture
def retrieval_client():
    client = AsyncMock()
    client.retrieve.return_value = RetrieveResponse(matches=[])
    return client


@pytest.fixture
def client(retrieval_client, inference_client):
    app = create_app(Settings(_env_file=None, environment="test"))
    app.dependency_overrides[get_retrieval_client] = lambda: retrieval_client
    app.dependency_overrides[get_inference_client] = lambda: inference_client
    with TestClient(app) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_generated_answer(client):
    response = client.post("/v1/query", json={"query": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["answer"] == "Generated answer"
    assert body["sources"] == []


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": "   "}, {"query": None},
    {"query": 42}, {"query": "x" * 10001},
    {"query": "hello", "unexpected": True},
])
def test_invalid_query(client, payload):
    assert client.post("/v1/query", json=payload).status_code == 422


def test_malformed_json(client):
    response = client.post("/v1/query", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_query_trims_whitespace():
    assert QueryRequest(query="  hello  ").query == "hello"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_async_service(retrieval_client, inference_client):
    response = await process_query(QueryRequest(query="hello"), retrieval_client, inference_client)
    assert response.status == "completed"
    assert response.answer == "Generated answer"
    assert response.sources == []
    retrieval_client.retrieve.assert_awaited_once_with("hello", 5)


def test_environment_settings(monkeypatch):
    monkeypatch.setenv("RAG_APP_NAME", "Test Gateway")
    monkeypatch.setenv("RAG_ENVIRONMENT", "production")
    settings = Settings(_env_file=None)
    assert settings.environment == "production"
    assert create_app(settings).title == "Test Gateway"


def test_dotenv_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_APP_NAME", raising=False)
    monkeypatch.delenv("RAG_ENVIRONMENT", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_APP_NAME=File Gateway\nRAG_ENVIRONMENT=test\n", encoding="utf-8")
    assert Settings(_env_file=env_file).app_name == "File Gateway"
    monkeypatch.setenv("RAG_APP_NAME", "Environment Gateway")
    assert Settings(_env_file=env_file).app_name == "Environment Gateway"


def test_invalid_environment(monkeypatch):
    monkeypatch.setenv("RAG_ENVIRONMENT", "invalid")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
