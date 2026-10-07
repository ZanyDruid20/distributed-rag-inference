# Distributed RAG Inference Platform

Foundation for a cloud-native, GPU-accelerated RAG inference platform. This phase
implements Python 3.11+ FastAPI gateway, retrieval, and inference services.
Retrieval uses local embeddings, in-memory ingestion, and user-owned cosine search.
The gateway builds a prompt using the user-owned builders and calls inference over
HTTP. Inference delegates generation to an external vLLM OpenAI-compatible server.
No model server, GPU setup, Docker, Redis, database, or Kubernetes is included.

## Layout

```text
services/api_gateway/
  src/api_gateway/
    config.py      # environment settings
    main.py        # application factory and HTTP routes
    schemas.py     # Pydantic API contracts
    service.py     # retrieval, prompt construction, and generation flow
    retrieval_client.py # async HTTP retrieval boundary
    inference_client.py # async HTTP inference boundary
    prompts.py     # user-owned build_context and build_prompt
  tests/test_gateway.py
services/retrieval/
  pyproject.toml
  src/retrieval_service/
    algorithms.py  # user-owned chunking, cosine similarity, and search
    embeddings.py  # lazy local CPU embedding model and wrappers
    config.py
    main.py
    schemas.py
    service.py     # async ingestion and retrieval adapters
    store.py       # process-local chunk storage
  tests/test_retrieval.py
services/inference/
  pyproject.toml
  src/inference_service/
    config.py
    main.py
    schemas.py
    service.py
    vllm_client.py # OpenAI-compatible HTTP completions adapter
  tests/test_inference.py
pyproject.toml
uv.lock
.env.example
```

The gateway is an installable package with a src layout. Future services can live
in sibling directories under `services/` with their own packages and dependencies.
The root pyproject packages the gateway; retrieval and inference have their own
pyprojects and are uv workspace members. The root dev group includes both packages
so `uv sync --locked` supports development and tests for all services. HTTP handlers
delegate work to async service functions.

## Run locally

Run from the repository root with `uv` installed and Python 3.11 or newer:

```powershell
uv sync --locked
Copy-Item .env.example .env
uv run --locked uvicorn api_gateway.main:app --host 127.0.0.1 --port 8000 --reload
```

`uv sync` creates `.venv` and installs all services and the default `dev` dependency
group. `uv run` uses that environment without manual activation. `uv.lock` pins
resolved dependencies; commit it with dependency changes. Use `uv add PACKAGE`
for runtime dependencies and `uv add --dev PACKAGE` for development dependencies.
On macOS/Linux, copy the environment file with `cp .env.example .env`.
Omit `--reload` when live development reload is unnecessary. To install only
runtime dependencies, use `uv sync --locked --no-dev`.

Settings use the `RAG_` prefix. Environment variables override `.env`, which is
loaded relative to the working directory. Supported settings are `RAG_APP_NAME`
and `RAG_ENVIRONMENT` (`development`, `test`, or `production`). The environment
label is configuration metadata; it does not enable deployment features.

## API

- `GET /health` returns `200` with `{"status":"ok"}`. This checks process liveness.
- `POST /v1/query` accepts `{"query":"What is RAG?", "top_k":5, "max_tokens":256, "temperature":0.2}`,
  calls retrieval and inference, and returns `200` with the generated answer and sources:

```json
{
  "status": "completed",
  "answer": "The generated answer from the model.",
  "sources": []
}
```

Queries must be strings containing 1–10,000 characters after trimming whitespace.
Unknown request fields and invalid inputs return `422`. `top_k` defaults to 5 and
must be an integer from 1 to 100. Sources contain each chunk's `document_id`,
`text`, and `score`; the example above represents no matches. Interactive API
documentation is at `/docs`.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/query -Method Post -ContentType 'application/json' -Body '{"query":"What is RAG?"}'
uv run --locked pytest
```

Tests cover health, generated responses, request validation, the async services,
and environment/.env configuration. No GPU or cloud resources are required.

## Phase 2: retrieval scaffold

Start retrieval in a separate terminal/process from the repository root:

```powershell
uv run --locked --package distributed-rag-retrieval uvicorn retrieval_service.main:app --host 127.0.0.1 --port 8001 --reload
```

Its docs are at `http://127.0.0.1:8001/docs`. The gateway runs on port 8000 and
calls retrieval over HTTP. Retrieval settings use `RETRIEVAL_APP_NAME` and
`RETRIEVAL_ENVIRONMENT`, with the same .env loading rules as the gateway.

- `GET /health` returns `200` with `{"status":"ok"}` for process liveness.
- `POST /v1/documents` accepts `{"text":"Document text", "title":"Optional title"}`.
  Text must contain 1-1,000,000 characters after trimming; title is optional and
  must contain 1-500 characters if supplied.
- `POST /v1/retrieve` accepts `{"query":"Question", "top_k":5}`. The query must
  contain 1-10,000 characters after trimming. `top_k` defaults to 5 and must be
  an integer between 1 and 100.

Document ingestion returns `200` with a generated `document_id` and `chunk_count`.
It uses the existing `chunk_text` function with 500-character chunks and a
50-character overlap, then embeds all chunks in one batch with `embed_texts`.
Chunking and embedding run in a worker thread. Each stored chunk contains only
`document_id`, `text`, and `embedding`; the optional title is not stored.
All chunks are committed together after embedding succeeds.

`store.get_chunks()` returns a copied snapshot and `store.clear_store()` clears
all chunks. Storage is local to one process and is lost on restart; separate
workers do not share it. No database or vector index is used.

Retrieval calls the existing `search()` in a worker thread and returns `200`.
Invalid inputs return `422`; unknown request fields are rejected. The response
has `matches` containing `document_id`, `text`, and `score`.

Chunking, cosine similarity, and search are user-owned implementations with
deterministic tests. HTTP and ingestion tests mock embedding/model calls.

```powershell
# All services
uv run --locked pytest
# Retrieval only
uv run --locked pytest services/retrieval/tests
```

## Local embeddings

`retrieval_service.embeddings` exposes `embed_text(text: str) -> list[float]` and
`embed_texts(texts: list[str]) -> list[list[float]]`. Batch results preserve input
order; an empty batch returns `[]`. Blank strings raise `ValueError` and nonstring
inputs raise `TypeError`. The vectors are normalized for cosine-based comparisons.

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Set
`RETRIEVAL_EMBEDDING_MODEL` to a different model identifier or a local model
directory before first use. The model is loaded once per process, lazily and on
CPU; concurrent first calls share that load. Restart the process to change models.
Importing the wrapper or calling `/health` does not load weights. The first real
embedding call downloads model files from Hugging Face unless already cached or
using a local directory. Subsequent inference runs locally.

```python
from retrieval_service.embeddings import embed_text, embed_texts

vector = embed_text("What is semantic retrieval?")
vectors = embed_texts(["First document", "Second document"])
```

Ingestion calls the batch wrapper in a worker thread; search uses the single-text wrapper.
Unit tests mock model loading and encoding to run without
network access or model weights; they test the wrapper rather than model quality.

## Phase 3: gateway retrieval connection

Start retrieval on port 8001 and the gateway on port 8000 in separate terminals
using the commands above. Ingest documents directly through retrieval's
`POST /v1/documents`, then call the gateway's `POST /v1/query`.

Configure `RAG_RETRIEVAL_SERVICE_URL` (default `http://127.0.0.1:8001`) and
`RAG_RETRIEVAL_TIMEOUT_SECONDS` (default 10) before starting the gateway. Its
lifespan owns a reusable HTTPX AsyncClient with connection pooling and closes it
on shutdown. The gateway has its own retrieval wire models and does not import
the retrieval package. Downstream timeouts return `504`; connection failures,
non-success responses, and invalid response bodies return `502` with safe messages.
`/health` checks only gateway liveness.

`api_gateway.prompts.build_context(matches)` and
`api_gateway.prompts.build_prompt(query, context)` are user-owned implementations.
The query service invokes both, sends the resulting prompt to inference, and
returns the generated answer while preserving retrieved chunks in `sources`.
Client tests use HTTPX MockTransport and require no running service or GPU model.

## Phase 4: inference service

Run the inference adapter in its own terminal from the repository root:

```powershell
uv run --locked --package distributed-rag-inference-service uvicorn inference_service.main:app --host 127.0.0.1 --port 8002 --reload
```

Docs: `http://127.0.0.1:8002/docs`. `GET /health` is process liveness only and
does not require a vLLM server. `POST /v1/generate` accepts:

```json
{"prompt":"Answer using the provided context...", "max_tokens":256, "temperature":0.2}
```

The prompt must be a nonblank string and is preserved exactly. `max_tokens`
defaults to 256 and must be an integer from 1 to 8192. `temperature` defaults to
0.2 and must be a finite number from 0 to 2. Unknown fields return `422`.
Successful responses contain `answer`, `model`, and nullable `finish_reason`:

```json
{"answer":"Generated text", "model":"your-served-model", "finish_reason":"stop"}
```

The adapter sends `model`, `prompt`, `max_tokens`, `temperature`, and
`stream:false` to `INFERENCE_VLLM_BASE_URL` plus `completions`, then extracts
the first choice's text. The gateway calls only the inference service's
`/v1/generate`, never vLLM. Each service owns and closes reusable async HTTP clients.
No vLLM or OpenAI SDK dependency is installed. Model weights are not loaded here.
There are no automatic generation retries or streaming in this phase.

Settings (environment variables override `.env`):

| Variable | Default / purpose |
| --- | --- |
| `RAG_INFERENCE_SERVICE_URL` | `http://127.0.0.1:8002` |
| `RAG_INFERENCE_TIMEOUT_SECONDS` | `90` |
| `INFERENCE_APP_NAME` | `Distributed RAG Inference Service` |
| `INFERENCE_ENVIRONMENT` | `development` (`development`, `test`, `production`) |
| `INFERENCE_VLLM_BASE_URL` | `http://127.0.0.1:8003/v1/`; include `/v1/` |
| `INFERENCE_VLLM_MODEL` | `your-served-model`; replace with the exact served name |
| `INFERENCE_VLLM_TIMEOUT_SECONDS` | `60` |
| `INFERENCE_VLLM_API_KEY` | Optional bearer token for an authenticated vLLM server |

Timeouts return `504`; connection failures, non-2xx upstream responses, or malformed
responses return `502` with safe messages. A downstream inference `504` remains
`504` at the gateway. `/health` does not assert model readiness.

Before live generation, configure an external reachable vLLM server supporting
`/v1/completions`, its exact model name, and its API key if enabled. Adjust token
limits to its model context window and keep the gateway timeout above the inference
timeout. None of the GPU/model setup is performed in this phase. Retrieval's
in-memory documents must be ingested again after that process restarts.

```powershell
uv run --locked pytest
uv run --locked pytest services/inference/tests
```

Tests use mocked vLLM HTTP responses, including the full gateway-to-inference
request path. They verify protocol handling rather than real model output quality.
