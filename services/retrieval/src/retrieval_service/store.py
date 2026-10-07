"""Process-local chunk storage; contents are lost when the process exits."""

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock


@dataclass
class StoredChunk:
    document_id: str
    text: str
    embedding: list[float]


_chunks: list[StoredChunk] = []
_lock = Lock()


def add_chunks(chunks: list[StoredChunk]) -> None:
    """Append a complete batch, copying it so callers cannot mutate storage."""
    snapshot = deepcopy(chunks)
    with _lock:
        _chunks.extend(snapshot)


def get_chunks() -> list[StoredChunk]:
    """Return a snapshot in insertion order, safe to inspect or modify."""
    with _lock:
        return deepcopy(_chunks)


def clear_store() -> None:
    """Remove all chunks in this process."""
    with _lock:
        _chunks.clear()
