"""Retrieval settings, isolated from the gateway's RAG_ variables."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(default="Distributed RAG Retrieval Service", min_length=1)
    environment: Literal["development", "test", "production"] = "development"
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", min_length=1
    )
