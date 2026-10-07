"""Tests for user-owned builders and the HTTP-to-prompt integration boundary."""

import json
from unittest.mock import Mock

import httpx
import pytest

from api_gateway import prompts, service
from api_gateway.config import Settings
from api_gateway.main import create_app, get_retrieval_client, get_inference_client
from api_gateway.retrieval_client import RetrievalClient
from api_gateway.schemas import RetrievalMatch


@pytest.fixture
def matches():
    return [
        RetrievalMatch(document_id="doc-A", text="First chunk.\nSecond line.", score=0.9),
        RetrievalMatch(document_id="doc-B", text="Second chunk: café & details.", score=0.7),
    ]


def test_context_formats_multiple_matches(matches):
    assert prompts.build_context(matches) == (
        "[Source 1 | document_id=doc-A]\nFirst chunk.\nSecond line.\n\n"
        "[Source 2 | document_id=doc-B]\nSecond chunk: café & details."
    )


def test_source_numbering_starts_at_one(matches):
    context = prompts.build_context(matches)
    assert context.startswith("[Source 1 |")
    assert "[Source 2 |" in context
    assert "[Source 0 |" not in context


def test_document_ids_and_text_preserved(matches):
    context = prompts.build_context(matches)
    for match in matches:
        assert f"document_id={match.document_id}" in context
        assert match.text in context


def test_empty_matches_produce_empty_context():
    assert prompts.build_context([]) == ""


def test_prompt_includes_query_and_context(matches):
    context = prompts.build_context(matches)
    query = "What do these documents say?"
    prompt = prompts.build_prompt(query, context)
    assert query in prompt
    assert context in prompt
    assert f"Context:\n{context}" in prompt
    assert f"Question:\n{query}" in prompt


def test_prompt_requires_only_provided_context():
    prompt = prompts.build_prompt("question", "retrieved evidence")
    assert "using only the provided context" in prompt.lower()


def test_prompt_handles_empty_context():
    prompt = prompts.build_prompt("What is the answer?", "")
    assert "What is the answer?" in prompt
    assert "Context:\n\n" in prompt
    assert "say that you do not have enough information" in prompt.lower()
    assert "Answer:" in prompt


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mocked_http_matches_can_build_context_and_prompt(matches):
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/retrieve"
        assert json.loads(request.content) == {"query": "question", "top_k": 2}
        return httpx.Response(200, json={"matches": [match.model_dump() for match in matches]})

    async with httpx.AsyncClient(
        base_url="http://retrieval.test/", transport=httpx.MockTransport(handler)
    ) as http:
        retrieved = await RetrievalClient(http).retrieve("question", top_k=2)
    context = prompts.build_context(retrieved.matches)
    prompt = prompts.build_prompt("question", context)
    assert "[Source 1 | document_id=doc-A]" in context
    assert "[Source 2 | document_id=doc-B]" in context
    assert context in prompt
    assert "question" in prompt


@pytest.mark.anyio
@pytest.mark.parametrize("empty", [False, True], ids=["with-matches", "empty-context"])
async def test_gateway_invokes_context_prompt_pipeline(monkeypatch, matches, empty, inference_client):
    returned_matches = [] if empty else matches
    query = "What do these documents say?"
    expected_context = prompts.build_context(returned_matches)
    build_context = Mock(wraps=prompts.build_context)
    build_prompt = Mock(wraps=prompts.build_prompt)
    # Observe either module-qualified calls or imports at the service boundary.
    monkeypatch.setattr(prompts, "build_context", build_context)
    monkeypatch.setattr(prompts, "build_prompt", build_prompt)
    monkeypatch.setattr(service, "build_context", build_context, raising=False)
    monkeypatch.setattr(service, "build_prompt", build_prompt, raising=False)

    def handler(request):
        assert request.url.path == "/v1/retrieve"
        assert json.loads(request.content) == {"query": query, "top_k": 2}
        return httpx.Response(200, json={
            "matches": [match.model_dump() for match in returned_matches]
        })

    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_inference_client] = lambda: inference_client
    async with httpx.AsyncClient(
        base_url="http://retrieval.test/", transport=httpx.MockTransport(handler)
    ) as upstream:
        client = RetrievalClient(upstream)
        app.dependency_overrides[get_retrieval_client] = lambda: client
        async with httpx.AsyncClient(
            base_url="http://gateway.test/", transport=httpx.ASGITransport(app=app)
        ) as gateway:
            response = await gateway.post("/v1/query", json={"query": query, "top_k": 2})

    assert response.status_code == 200
    assert response.json()["sources"] == [match.model_dump() for match in returned_matches]
    build_context.assert_called_once_with(returned_matches)
    build_prompt.assert_called_once_with(query, expected_context)
    inference_client.generate.assert_awaited_once()
    assert expected_context in inference_client.generate.call_args.args[0].prompt
