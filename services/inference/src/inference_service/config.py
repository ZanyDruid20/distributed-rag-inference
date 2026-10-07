from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(default="Distributed RAG Inference Service", min_length=1)
    environment: Literal["development", "test", "production"] = "development"
    vllm_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8003/v1/")
    vllm_model: str = Field(default="your-served-model", min_length=1)
    vllm_timeout_seconds: float = Field(default=60.0, gt=0)
    vllm_api_key: SecretStr | None = None
