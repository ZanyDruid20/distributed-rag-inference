from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class CompletionChoice(BaseModel):
    text: str = Field(strict=True)
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    model: str = Field(strict=True)
    choices: list[CompletionChoice] = Field(min_length=1)
