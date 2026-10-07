"""Offline wrapper tests: no model downloads or inference required."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from retrieval_service import embeddings
from retrieval_service.config import Settings


@pytest.fixture
def model(monkeypatch):
    model = Mock()
    model.encode.return_value = [[1, 0], [0, 1]]
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_load_model", Mock(return_value=model))
    return model


def test_single_text(model):
    model.encode.return_value = [[1, 0]]
    result = embeddings.embed_text("hello")
    assert result == [1.0, 0.0]
    assert all(type(value) is float for value in result)
    model.encode.assert_called_once_with(
        ["hello"], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )


def test_batch_preserves_order(model):
    result = embeddings.embed_texts(["first", "second"])
    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert all(type(value) is float for vector in result for value in vector)
    assert model.encode.call_args.args == (["first", "second"],)


def test_empty_batch_does_not_load(model):
    assert embeddings.embed_texts([]) == []
    embeddings._load_model.assert_not_called()
    model.encode.assert_not_called()


@pytest.mark.parametrize("text", ["", "  ", "\n"])
def test_blank_text_rejected(model, text):
    with pytest.raises(ValueError):
        embeddings.embed_text(text)
    embeddings._load_model.assert_not_called()


@pytest.mark.parametrize("texts", ["hello", [123], [None], ["valid", 42]])
def test_invalid_batch_rejected(model, texts):
    with pytest.raises(TypeError):
        embeddings.embed_texts(texts)
    embeddings._load_model.assert_not_called()


def test_model_reused_across_single_and_batch(model):
    embeddings.embed_text("first")
    embeddings.embed_texts(["second", "third"])
    embeddings._load_model.assert_called_once_with()


def test_concurrent_first_load(model):
    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded = list(executor.map(lambda _: embeddings._get_model(), range(8)))
    assert all(item is model for item in loaded)
    embeddings._load_model.assert_called_once_with()


def test_load_failure_is_retryable(model):
    embeddings._load_model.side_effect = [RuntimeError("model unavailable"), model]
    with pytest.raises(RuntimeError, match="model unavailable"):
        embeddings.embed_text("hello")
    assert embeddings._model is None
    assert embeddings.embed_text("hello") == [1.0, 0.0]


def test_model_configuration(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_EMBEDDING_MODEL", "local-model-path")
    assert Settings(_env_file=None).embedding_model == "local-model-path"


def test_loader_uses_configured_model_on_cpu(monkeypatch):
    import sys

    constructor = Mock()
    monkeypatch.setenv("RETRIEVAL_EMBEDDING_MODEL", "local-model-path")
    monkeypatch.setitem(sys.modules, "sentence_transformers", Mock(SentenceTransformer=constructor))
    assert embeddings._load_model() is constructor.return_value
    constructor.assert_called_once_with("local-model-path", device="cpu")
