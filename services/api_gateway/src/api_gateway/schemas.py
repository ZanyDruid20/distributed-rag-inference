"""Public request and response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(strict=True, min_length=1, max_length=10000)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)
    max_tokens: int = Field(default=256, strict=True, ge=1, le=8192)
    temperature: float = Field(default=0.2, strict=True, ge=0, le=2, allow_inf_nan=False)


class RetrieveRequest(BaseModel):
    """Gateway-owned copy of the retrieval HTTP request contract."""

    query: str = Field(strict=True, min_length=1, max_length=10000)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)


class RetrievalMatch(BaseModel):
    """A retrieved chunk and its cosine similarity score."""

    document_id: str
    text: str
    score: float = Field(allow_inf_nan=False)


class RetrieveResponse(BaseModel):
    matches: list[RetrievalMatch]



class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(strict=True, min_length=1)
    max_tokens: int = Field(default=256, strict=True, ge=1, le=8192)
    temperature: float = Field(default=0.2, strict=True, ge=0, le=2, allow_inf_nan=False)

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class GenerateResponse(BaseModel):
    answer: str = Field(strict=True)
    model: str = Field(strict=True)
    finish_reason: str | None = None


class QueryResponse(BaseModel):
    status: Literal["completed"] = "completed"
    answer: str
    sources: list[RetrievalMatch] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
