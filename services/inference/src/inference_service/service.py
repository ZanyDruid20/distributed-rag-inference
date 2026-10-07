from inference_service.schemas import GenerateRequest, GenerateResponse
from inference_service.vllm_client import VLLMClient


async def generate(request: GenerateRequest, client: VLLMClient) -> GenerateResponse:
    return await client.generate(request)
