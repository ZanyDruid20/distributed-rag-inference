"""Independent FastAPI retrieval application."""

from fastapi import FastAPI

from retrieval_service.config import Settings
from retrieval_service.schemas import (
    DocumentRequest,
    DocumentResponse,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from retrieval_service.service import ingest_document, retrieve


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    application = FastAPI(title=settings.app_name, version="0.1.0")
    application.state.settings = settings

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Process liveness; does not indicate that algorithms are implemented."""
        return HealthResponse()

    @application.post(
        "/v1/documents", response_model=DocumentResponse, tags=["documents"],
    )
    async def documents(request: DocumentRequest) -> DocumentResponse:
        return await ingest_document(request)

    @application.post(
        "/v1/retrieve", response_model=RetrieveResponse, tags=["retrieval"],
    )
    async def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
        return await retrieve(request)

    return application


app = create_app()
