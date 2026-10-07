"""Retrieve sources, construct a prompt, and request remote generation."""

from api_gateway.inference_client import InferenceClient
from api_gateway.prompts import build_context, build_prompt
from api_gateway.retrieval_client import RetrievalClient
from api_gateway.schemas import GenerateRequest, QueryRequest, QueryResponse


async def process_query(
    request: QueryRequest, retrieval_client: RetrievalClient, inference_client: InferenceClient
) -> QueryResponse:
    """Use the existing prompt builders and preserve retrieval sources."""
    retrieved = await retrieval_client.retrieve(request.query, request.top_k)
    context = build_context(retrieved.matches)
    prompt = build_prompt(request.query, context)
    generated = await inference_client.generate(GenerateRequest(
        prompt=prompt, max_tokens=request.max_tokens, temperature=request.temperature
    ))
    return QueryResponse(
        answer=generated.answer,
        sources=retrieved.matches,
    )
