"""Deterministic ranking tests using real storage and cosine similarity."""

from unittest.mock import Mock

import pytest

from retrieval_service import algorithms
from retrieval_service.schemas import RetrievalMatch
from retrieval_service.store import StoredChunk, add_chunks, clear_store


@pytest.fixture(autouse=True)
def isolated_store():
    clear_store()
    yield
    clear_store()


@pytest.fixture
def embedder(monkeypatch):
    embedder = Mock(return_value=[1.0, 0.0])
    # Patch where search looks up the function, not its original module.
    monkeypatch.setattr(algorithms, "embed_text", embedder)
    return embedder


@pytest.fixture
def ranked_chunks():
    # Deliberately insert out of score order. Mock embeddings represent relevance
    # to the query, without requiring a model or testing its semantic quality.
    chunks = [
        StoredChunk("orthogonal", "Unrelated topic", [0.0, 1.0]),
        StoredChunk("opposite", "Opposite meaning", [-1.0, 0.0]),
        StoredChunk("relevant", "Best matching document", [1.0, 0.0]),
        StoredChunk("partial", "Partially matching document", [3.0, 4.0]),
    ]
    add_chunks(chunks)
    return chunks


def test_most_relevant_chunks_first(embedder, ranked_chunks):
    matches = algorithms.search("semantic query", top_k=2)
    assert [match.document_id for match in matches] == ["relevant", "partial"]
    embedder.assert_called_once_with("semantic query")


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_respects_top_k(embedder, ranked_chunks, top_k):
    matches = algorithms.search("semantic query", top_k=top_k)
    assert len(matches) == top_k
    assert [match.document_id for match in matches] == [
        "relevant", "partial", "orthogonal", "opposite"
    ][:top_k]


def test_fewer_chunks_than_top_k(embedder):
    add_chunks([StoredChunk("only", "Only document", [1.0, 0.0])])
    matches = algorithms.search("semantic query", top_k=5)
    assert len(matches) == 1
    assert matches[0].document_id == "only"


def test_empty_store(embedder):
    assert algorithms.search("semantic query", top_k=5) == []


def test_returned_fields_and_scores(embedder, ranked_chunks):
    matches = algorithms.search("semantic query", top_k=4)
    expected = [
        ("relevant", "Best matching document", 1.0),
        ("partial", "Partially matching document", 0.6),
        ("orthogonal", "Unrelated topic", 0.0),
        ("opposite", "Opposite meaning", -1.0),
    ]
    assert len(matches) == len(expected)
    for match, (document_id, text, score) in zip(matches, expected, strict=True):
        assert isinstance(match, RetrievalMatch)
        assert match.document_id == document_id
        assert match.text == text
        assert match.score == pytest.approx(score)


def test_descending_cosine_similarity(embedder, ranked_chunks):
    scores = [match.score for match in algorithms.search("semantic query", top_k=4)]
    assert scores == pytest.approx([1.0, 0.6, 0.0, -1.0])
    assert scores == sorted(scores, reverse=True)
