"""Cosine similarity tests; the implementation is user-owned."""

import pytest

from retrieval_service.algorithms import cosine_similarity


def test_identical_vectors():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


@pytest.mark.parametrize("a,b", [([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 2])])
def test_different_lengths_raise(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


@pytest.mark.parametrize("a,b", [
    ([0, 0, 0], [1, 2, 3]),
    ([1, 2, 3], [0, 0, 0]),
    ([0, 0, 0], [0, 0, 0]),
])
def test_zero_vectors_raise(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_known_vectors():
    # Independently calculated: 32 / sqrt(14 * 77).
    assert cosine_similarity([1, 2, 3], [4, 5, 6]) == pytest.approx(
        0.9746318461970762
    )
