"""Structured per-request logging - what `app/obs/metrics.py`'s aggregate counters
can't give you: what a *specific* tool call actually asked for and got back.

Before this, the only place a tool call's query/filters and the products it
returned ever existed was the ephemeral SSE stream sent to the browser
(`app/llm/graph.py`'s `writer({"type": "tool_call", ...})`) - nothing was written
down server-side. A live bug report ("suit, women" surfacing women's shoes
alongside blazers) turned out to be a stale semantic-cache entry, and there was no
way to reconstruct what the *original* tool call had actually searched for to
confirm the mechanism - only to prove the symptom was real. This module exists so
that gap doesn't repeat: every tool call and its result are logged as one JSON
line each, tagged by tenant/conversation_id, so a specific exchange can be grepped
back out of the logs after the fact.

`structlog` was already a declared dependency, unused anywhere in the codebase -
wired up here rather than adding a new one.
"""

from __future__ import annotations

import structlog


def configure_structlog() -> None:
    """Call once, at process startup - `app/main.py`'s lifespan does this
    alongside the existing `logging.basicConfig`. Idempotent (structlog allows
    re-configuring), but there's never a reason to call it more than once per
    process.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A bound structlog logger backed by a stdlib logger of the same name -
    `Settings.log_level` (set via `logging.basicConfig` in `app/main.py`)
    applies to it the same way it does to every other logger in this codebase.
    """
    return structlog.get_logger(name)
