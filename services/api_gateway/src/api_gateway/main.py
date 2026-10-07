"""FastAPI application and HTTP routes."""

from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request

from api_gateway.config import Settings
from api_gateway.inference_client import InferenceClient, InferenceError
from api_gateway.retrieval_client import RetrievalClient, RetrievalError
from api_gateway.schemas import HealthResponse, QueryRequest, QueryResponse
from api_gateway.service import process_query


def get_retrieval_client(request: Request) -> RetrievalClient:
    return request.app.state.retrieval_client


def get_inference_client(request: Request) -> InferenceClient:
    return request.app.state.inference_client


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with httpx.AsyncClient(
            base_url=str(settings.retrieval_service_url).rstrip("/") + "/",
            timeout=settings.retrieval_timeout_seconds,
        ) as client:
            async with httpx.AsyncClient(
                base_url=str(settings.inference_service_url).rstrip("/") + "/",
                timeout=settings.inference_timeout_seconds,
            ) as inference_http:
                application.state.retrieval_client = RetrievalClient(client)
                application.state.inference_client = InferenceClient(inference_http)
                yield

    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    application.state.settings = settings

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Process liveness only; does not probe downstream services."""
        return HealthResponse()

    @application.post("/v1/query", response_model=QueryResponse, tags=["query"])
    async def query(
        request: QueryRequest, retrieval_client: RetrievalClient = Depends(get_retrieval_client),
        inference_client: InferenceClient = Depends(get_inference_client),
    ) -> QueryResponse:
        try:
            return await process_query(request, retrieval_client, inference_client)
        except (RetrievalError, InferenceError) as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return application


app = create_app()
