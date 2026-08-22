# syntax=docker/dockerfile:1.7
# Multi-stage. CPU-only torch keeps the image ~1.2GB instead of ~4GB with CUDA wheels.

# ---------- builder ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# CPU-only torch: pull from the PyTorch CPU index instead of the default CUDA wheels.
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match

COPY pyproject.toml README.md ./
COPY app ./app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1001 app && \
    useradd --uid 1001 --gid app --create-home app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Bake model downloads into a writable, predictable location.
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/sentence_transformers

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY --chown=app:app app ./app
COPY --chown=app:app static ./static
COPY --chown=app:app alembic.ini ./alembic.ini
COPY --chown=app:app migrations ./migrations

USER app

# Pre-download the embedding + reranker models at build time so the first request
# is not a 90-second cold start. Failure is non-fatal (offline builds still work).
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=BAAI/bge-reranker-base
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}'); \
print('models cached')" || echo "WARN: model pre-cache skipped"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
