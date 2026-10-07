"""Async HTTP boundary to the independently running retrieval service."""

import httpx
from pydantic import ValidationError

from api_gateway.schemas import RetrieveRequest, RetrieveResponse


class RetrievalError(Exception):
    """Safe downstream failure exposed by the gateway."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RetrievalClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def retrieve(self, query: str, top_k: int = 5) -> RetrieveResponse:
        payload = RetrieveRequest(query=query, top_k=top_k)
        try:
            response = await self.client.post("v1/retrieve", json=payload.model_dump())
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RetrievalError(504, "Retrieval service timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(502, "Retrieval service returned an error") from exc
        except httpx.RequestError as exc:
            raise RetrievalError(502, "Retrieval service is unavailable") from exc
        try:
            return RetrieveResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise RetrievalError(502, "Retrieval service returned an invalid response") from exc
