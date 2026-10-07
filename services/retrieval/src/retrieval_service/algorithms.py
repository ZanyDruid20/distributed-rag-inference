"""User-owned algorithm TODOs. No chunking or search is implemented."""

from retrieval_service.schemas import RetrievalMatch
from retrieval_service.store import get_chunks
from retrieval_service.embeddings import embed_text


import math


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """TODO: Implement text chunking and validate chunk size/overlap semantics."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")
    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        result.append(chunk)
        start += chunk_size - overlap
    return result


def search(query: str, top_k: int) -> list[RetrievalMatch]:
    """TODO: Implement core retrieval/search against your chosen document store.

    Storage, indexing, scoring, and result ordering are intentionally undecided.
    Adapt this function's signature as needed when implementing those choices.
    """
    chunks = get_chunks()
    query_embedding = embed_text(query)
    matches = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        matches.append(RetrievalMatch(document_id=chunk.document_id, text=chunk.text, score=score))
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches[:top_k]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """TODO: Implement cosine similarity and define invalid-vector handling."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensions")
    dot_product = 0.0
    for x, y in zip(a,b):
        dot_product += x * y
    magnitude_a = 0.0
    magnitude_b = 0.0
    for x in a:
        magnitude_a += x ** 2
    magnitude_a = math.sqrt(magnitude_a)
    for y in b:
        magnitude_b += y ** 2
    magnitude_b = math.sqrt(magnitude_b)
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector")
    return dot_product / (magnitude_a * magnitude_b)