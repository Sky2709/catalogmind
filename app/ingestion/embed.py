"""Local CPU embeddings.

Runs `BAAI/bge-small-en-v1.5` through sentence-transformers on the CPU. Chosen over a
hosted embedding API for one practical reason: the alpha sweep re-queries the golden
sets hundreds of times, and per-call rate limits or costs would make that experiment
either slow or expensive. Local inference makes retrieval evaluation free and
unlimited, which is what lets the numbers in the README be measured rather than
asserted.

Two details that quietly decide retrieval quality:

**Asymmetric encoding.** BGE is trained with an instruction prefix on the *query* side
only. Passages are embedded bare; queries get
``"Represent this sentence for searching relevant passages: "``. Embedding both sides
identically is the most common way to leave recall on the table with these models, and
it fails silently — everything still returns results, just worse ones.

**Normalisation.** Vectors are L2-normalised, so cosine similarity reduces to a dot
product and Weaviate's COSINE distance behaves as the model was trained to expect.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BGE's retrieval instruction. Applied to queries only - see module docstring.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once per process.

    Loading costs seconds and hundreds of megabytes of RAM; doing it per request would
    dominate latency. Cached so the API, the ingestion worker and the eval harness all
    share one instance.
    """
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("loading embedding model %s (cpu)", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    logger.info(
        "loaded %s: dim=%s max_seq_len=%s",
        settings.embedding_model,
        _dimension_of(model),
        model.max_seq_length,
    )
    return model


def _dimension_of(model: SentenceTransformer) -> int | None:
    """Output dimension, across sentence-transformers versions.

    v6 renamed `get_sentence_embedding_dimension` to `get_embedding_dimension` and
    warns on the old name. Prefer the new one, fall back for older installs.
    """
    getter = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    return getter()


def embedding_dimension() -> int:
    """Actual output dimension of the loaded model.

    sentence-transformers types this as optional because some module stacks cannot
    report it. For our models it is always present, and a silent `None` here would
    surface much later as a Weaviate dimension mismatch on insert - so fail loudly.
    """
    dim = _dimension_of(get_model())
    if dim is None:
        raise RuntimeError(
            f"{get_settings().embedding_model} did not report an embedding dimension"
        )
    return int(dim)


def embed_documents(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> list[list[float]]:
    """Embed catalog text. No instruction prefix — this is the passage side."""
    if not texts:
        return []
    vectors = get_model().encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str, *, use_instruction: bool | None = None) -> list[float]:
    """Embed a shopper's query, with BGE's retrieval instruction prefix.

    Defaults to the `embed_query_instruction` setting; pass explicitly so the eval
    harness can measure what the prefix is actually worth on our data rather than
    taking the model card's word for it.
    """
    return embed_queries([text], use_instruction=use_instruction)[0]


def embed_queries(
    texts: Sequence[str],
    *,
    use_instruction: bool | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    if not texts:
        return []
    if use_instruction is None:
        use_instruction = get_settings().embed_query_instruction
    prepared = [QUERY_INSTRUCTION + t for t in texts] if use_instruction else list(texts)
    vectors = get_model().encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# --- async wrappers ---------------------------------------------------------------

# Inference is synchronous, CPU-bound and releases the GIL inside torch. Running it in
# the default executor keeps the FastAPI event loop responsive instead of stalling
# every other in-flight request for the duration of an encode.


async def aembed_documents(
    texts: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[list[float]]:
    return await asyncio.to_thread(embed_documents, texts, batch_size=batch_size)


async def aembed_query(text: str, *, use_instruction: bool | None = None) -> list[float]:
    return await asyncio.to_thread(embed_query, text, use_instruction=use_instruction)


def warm_up() -> None:
    """Force model load and one encode at startup.

    The first encode is materially slower than steady state (lazy kernel init inside
    torch). Paying that during application startup keeps it out of the p99 of the
    first real shopper request.
    """
    get_model()
    embed_documents(["warm up"])
    logger.info("embedding model warm")
