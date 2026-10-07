"""Non-streaming OpenAI-compatible completions over HTTP."""

import httpx

from inference_service.schemas import CompletionResponse, GenerateRequest, GenerateResponse


class InferenceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class VLLMClient:
    def __init__(self, client: httpx.AsyncClient, model: str):
        self.client = client
        self.model = model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            response = await self.client.post(
                "completions", json={"model": self.model, **request.model_dump(), "stream": False}
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InferenceError(504, "vLLM request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise InferenceError(502, "vLLM returned an error") from exc
        except httpx.RequestError as exc:
            raise InferenceError(502, "vLLM is unavailable") from exc
        try:
            completion = CompletionResponse.model_validate(response.json())
        except ValueError as exc:
            raise InferenceError(502, "vLLM returned an invalid response") from exc
        choice = completion.choices[0]
        return GenerateResponse(
            answer=choice.text, model=completion.model, finish_reason=choice.finish_reason
        )
