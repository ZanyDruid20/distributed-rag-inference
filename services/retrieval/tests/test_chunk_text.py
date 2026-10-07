"""Chunking behavior tests; the algorithm implementation is user-owned."""

import pytest

from retrieval_service.algorithms import chunk_text


def test_normal_chunking():
    assert chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=2) == [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
        "yz",
    ]


@pytest.mark.parametrize("chunk_size", [0, -1, -10])
def test_nonpositive_chunk_size_raises(chunk_size):
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=chunk_size, overlap=0)


@pytest.mark.parametrize("overlap", [-1, -10])
def test_negative_overlap_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=overlap)


@pytest.mark.parametrize("overlap", [10, 11, 20])
def test_overlap_at_least_chunk_size_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=overlap)


def test_empty_text():
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_text_shorter_than_chunk_size():
    assert chunk_text("hello", chunk_size=10, overlap=2) == ["hello"]
