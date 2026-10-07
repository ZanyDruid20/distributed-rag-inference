"""Proposed API contracts for document ingestion and retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(strict=True, min_length=1, max_length=1_000_000)
    title: str | None = Field(default=None, strict=True, min_length=1, max_length=500)


class DocumentResponse(BaseModel):
    document_id: str
    chunk_count: int = Field(ge=0)


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(strict=True, min_length=1, max_length=10_000)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)


class RetrievalMatch(BaseModel):
    document_id: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    matches: list[RetrievalMatch]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class NotImplementedResponse(BaseModel):
    detail: str
