"""Ingestion/storage tests use real chunking and mocked batch embeddings."""

from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from retrieval_service import service
from retrieval_service.config import Settings
from retrieval_service.main import create_app
from retrieval_service.schemas import DocumentRequest
from retrieval_service.store import StoredChunk, add_chunks, clear_store, get_chunks


@pytest.fixture(autouse=True)
def isolated_store():
    clear_store()
    yield
    clear_store()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def embedder(monkeypatch):
    embedder = Mock(side_effect=lambda texts: [[float(i), 1.0] for i in range(len(texts))])
    monkeypatch.setattr(service, "embed_texts", embedder)
    return embedder


@pytest.mark.anyio
async def test_ingest_multiple_chunks(embedder):
    text = "a" * 450 + "b" * 450 + "c" * 200
    response = await service.ingest_document(DocumentRequest(text=text))
    expected = [text[:500], text[450:950], text[900:]]
    embedder.assert_called_once_with(expected)
    assert str(UUID(response.document_id)) == response.document_id
    assert response.chunk_count == 3
    assert get_chunks() == [
        StoredChunk(response.document_id, chunk, [float(i), 1.0])
        for i, chunk in enumerate(expected)
    ]


@pytest.mark.anyio
async def test_documents_have_distinct_ids_and_append(embedder):
    first = await service.ingest_document(DocumentRequest(text="first"))
    second = await service.ingest_document(DocumentRequest(text="second"))
    assert first.document_id != second.document_id
    assert first.chunk_count == second.chunk_count == 1
    assert [chunk.document_id for chunk in get_chunks()] == [first.document_id, second.document_id]
    assert [chunk.text for chunk in get_chunks()] == ["first", "second"]


@pytest.mark.anyio
async def test_embedding_failure_does_not_store_partial_document(embedder):
    existing = StoredChunk("existing", "previous", [1.0])
    add_chunks([existing])
    embedder.side_effect = RuntimeError("embedding failed")
    with pytest.raises(RuntimeError, match="embedding failed"):
        await service.ingest_document(DocumentRequest(text="new document"))
    assert get_chunks() == [existing]


@pytest.mark.anyio
async def test_embedding_count_mismatch_does_not_store(embedder):
    embedder.side_effect = None
    embedder.return_value = [[1.0]]
    with pytest.raises(ValueError):
        await service.ingest_document(DocumentRequest(text="x" * 1100))
    assert get_chunks() == []


def test_document_endpoint(embedder):
    with TestClient(create_app(Settings(_env_file=None, environment="test"))) as client:
        response = client.post("/v1/documents", json={"text": "hello", "title": "Example"})
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 1
    assert get_chunks() == [StoredChunk(body["document_id"], "hello", [0.0, 1.0])]
    embedder.assert_called_once_with(["hello"])


def test_clear_store():
    add_chunks([StoredChunk("one", "hello", [1.0])])
    clear_store()
    assert get_chunks() == []


def test_store_copies_inputs_and_returns_snapshots():
    chunks = [StoredChunk("one", "hello", [1.0])]
    add_chunks(chunks)
    chunks[0].embedding[0] = 99.0
    chunks.clear()
    snapshot = get_chunks()
    snapshot[0].text = "changed"
    snapshot[0].embedding[0] = 42.0
    snapshot.clear()
    assert get_chunks() == [StoredChunk("one", "hello", [1.0])]
