"""Optional LLM tracing via Langfuse, added 2026-08-26.

`CLAUDE.md` is explicit that Claude is called inside LangGraph nodes with the raw
`anthropic` SDK, never a LangChain/LangGraph model wrapper - deliberate, to keep
prompt-cache checkpoints, the Haiku/Sonnet router, and `output_config.effort`
working. That means Langfuse's automatic LangChain `CallbackHandler` has nothing to
hook into here: its auto-capture triggers on a `BaseChatModel` invocation, which
never happens in this codebase. So this module does not use the callback handler
(and does not add `langchain` as a dependency just to import it) - it manually
opens Langfuse spans/generations around the two real seams: one span per chat turn
(`app/routers/chat.py`), one generation per Bedrock call (`app/llm/graph.py`).
Both use `start_as_current_observation`, which is OpenTelemetry-context-based, so a
generation opened inside `agent()` nests correctly under the turn span opened in
the router - same context-propagation mechanism, not two disconnected traces.
"""

from __future__ import annotations

from langfuse import Langfuse

from app.config import get_settings

_client: Langfuse | None = None


def langfuse_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def get_langfuse_client() -> Langfuse:
    """Lazy singleton, same shape as `get_redis_client()`/`get_mongo_client()` -
    only ever called when `langfuse_enabled()` is true, so callers never pay for
    an unconfigured client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url or None,
        )
    return _client
