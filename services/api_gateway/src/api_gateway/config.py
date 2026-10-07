"""Settings from environment variables and optional .env."""

from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(default="Distributed RAG API Gateway", min_length=1)
    environment: Literal["development", "test", "production"] = "development"
    retrieval_service_url: HttpUrl = HttpUrl("http://127.0.0.1:8001")
    retrieval_timeout_seconds: float = Field(default=10.0, gt=0)
    inference_service_url: HttpUrl = HttpUrl("http://127.0.0.1:8002")
    inference_timeout_seconds: float = Field(default=90.0, gt=0)
