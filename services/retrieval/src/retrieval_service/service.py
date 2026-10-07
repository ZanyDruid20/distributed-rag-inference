"""Async adapters for ingestion and the user-owned search algorithm."""

from uuid import uuid4

from starlette.concurrency import run_in_threadpool

from retrieval_service.algorithms import chunk_text, search
from retrieval_service.embeddings import embed_texts
from retrieval_service.store import StoredChunk, add_chunks

from retrieval_service.schemas import (
    DocumentRequest,
    DocumentResponse,
    RetrieveRequest,
    RetrieveResponse,
)


async def ingest_document(request: DocumentRequest) -> DocumentResponse:
    """Chunk, embed, and store without blocking the async event loop."""
    return await run_in_threadpool(_ingest_document, request)


def _ingest_document(request: DocumentRequest) -> DocumentResponse:
    document_id = str(uuid4())
    texts = chunk_text(request.text, chunk_size=500, overlap=50)
    embeddings = embed_texts(texts)
    chunks = [
        StoredChunk(document_id=document_id, text=text, embedding=embedding)
        for text, embedding in zip(texts, embeddings, strict=True)
    ]
    # Commit only after the full batch has been embedded and constructed.
    add_chunks(chunks)
    return DocumentResponse(document_id=document_id, chunk_count=len(chunks))


async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Execute the existing user-owned search in a worker thread."""
    matches = await run_in_threadpool(search, request.query, request.top_k)
    return RetrieveResponse(matches=matches)
