from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request

from inference_service.config import Settings
from inference_service.schemas import GenerateRequest, GenerateResponse, HealthResponse
from inference_service.service import generate
from inference_service.vllm_client import InferenceError, VLLMClient


def get_vllm_client(request: Request) -> VLLMClient:
    return request.app.state.vllm_client


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        headers = {}
        if settings.vllm_api_key is not None:
            headers["Authorization"] = "Bearer " + settings.vllm_api_key.get_secret_value()
        async with httpx.AsyncClient(
            base_url=str(settings.vllm_base_url).rstrip("/") + "/",
            timeout=settings.vllm_timeout_seconds, headers=headers,
        ) as http:
            application.state.vllm_client = VLLMClient(http, settings.vllm_model)
            yield

    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    application.state.settings = settings

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Process liveness; does not contact vLLM."""
        return HealthResponse()

    @application.post("/v1/generate", response_model=GenerateResponse)
    async def generate_endpoint(
        request: GenerateRequest, client: VLLMClient = Depends(get_vllm_client)
    ) -> GenerateResponse:
        try:
            return await generate(request, client)
        except InferenceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return application


app = create_app()
