"""Local CPU embeddings with lazy, process-wide model reuse."""

from threading import Lock
from typing import TYPE_CHECKING

from retrieval_service.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None
_model_lock = Lock()


def _load_model() -> "SentenceTransformer":
    # Import and model download happen only when embeddings are first requested.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(Settings().embedding_model, device="cpu")


def _get_model() -> "SentenceTransformer":
    global _model
    # Also prevent concurrent first calls from loading duplicate models.
    with _model_lock:
        if _model is None:
            _model = _load_model()
        return _model


def embed_text(text: str) -> list[float]:
    """Embed a nonblank string as a normalized vector of Python floats."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch in input order. An empty batch returns [] without loading."""
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must be a list of strings")
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain blank strings")
    if not texts:
        return []
    vectors = _get_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return [[float(value) for value in vector] for vector in vectors]
