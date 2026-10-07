from unittest.mock import AsyncMock

import pytest

from api_gateway.schemas import GenerateResponse


@pytest.fixture
def inference_client():
    client = AsyncMock()
    client.generate.return_value = GenerateResponse(
        answer="Generated answer", model="test-model", finish_reason="stop"
    )
    return client
