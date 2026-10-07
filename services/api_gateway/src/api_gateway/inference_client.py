"""HTTP client for the inference service, with no direct vLLM access."""

import httpx

from api_gateway.schemas import GenerateRequest, GenerateResponse


class InferenceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class InferenceClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            response = await self.client.post("v1/generate", json=request.model_dump())
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InferenceError(504, "Inference service timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = 504 if exc.response.status_code == 504 else 502
            raise InferenceError(status, "Inference service returned an error") from exc
        except httpx.RequestError as exc:
            raise InferenceError(502, "Inference service is unavailable") from exc
        try:
            return GenerateResponse.model_validate(response.json())
        except ValueError as exc:
            raise InferenceError(502, "Inference service returned an invalid response") from exc
