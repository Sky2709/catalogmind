#!/usr/bin/env bash
# Codespace bootstrap. Runs once, on container creation.
#
# Deliberately does NOT start the datastore stack: Codespaces bills by wall-clock
# uptime, and four idle containers waste the 120 free core-hours/month. Start it
# with `make up` when you actually need it, and `make down` when you don't.

set -euo pipefail

echo "==> installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "==> creating venv on Python 3.12"
uv venv --python 3.12

echo "==> installing dependencies (torch CPU wheels)"
# The default index serves CUDA builds on Linux (~2.5GB). We only ever run CPU
# inference, and the 15GB Codespaces storage quota is not worth spending on CUDA.
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
UV_INDEX_STRATEGY=unsafe-best-match \
  uv pip install -e ".[dev]"

echo "==> seeding .env if absent"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    created .env from .env.example"
  echo "    NOTE: set ANTHROPIC_API_KEY (Codespaces secret ANTHROPIC_API_KEY is picked up automatically if configured)"
fi

# A Codespaces secret named ANTHROPIC_API_KEY arrives as an env var; fold it into .env
# so the app's pydantic-settings loader sees it without any special-casing.
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}|" .env
  echo "    ANTHROPIC_API_KEY injected from Codespaces secret"
fi

echo "==> pre-caching embedding + reranker models"
# ~150MB. Done now so the first search is not a cold-start download.
uv run python - <<'PY' || echo "    WARN: model pre-cache failed (will download on first use)"
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("BAAI/bge-reranker-base")
print("    models cached")
PY

echo "==> verifying"
uv run python -m pytest tests/unit -q || true

cat <<'BANNER'

  CatalogMind codespace ready.

    make up        start weaviate + postgres + mongo + redis
    make migrate   apply database migrations
    make seed      provision demo merchants + ingest catalogs
    make dev       API on :8000  (Swagger at /docs)
    make test      unit tests (no stack needed)
    make down      stop the stack  <- do this when idle, it saves core-hours

BANNER
