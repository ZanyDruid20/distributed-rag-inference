# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# All workspace metadata is needed to validate the shared lockfile.
COPY pyproject.toml uv.lock ./
COPY services/retrieval/pyproject.toml services/retrieval/pyproject.toml
COPY services/inference/pyproject.toml services/inference/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package distributed-rag-api-gateway --no-install-workspace
COPY services/api_gateway/src services/api_gateway/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package distributed-rag-api-gateway

FROM python:3.12-slim-bookworm AS runtime
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]
CMD ["uvicorn", "api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
