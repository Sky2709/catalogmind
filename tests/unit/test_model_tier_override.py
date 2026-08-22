"""`ChatState.force_tier` - the eval-only override that lets
`eval/measure_model_router.py` run the same message through both model tiers.
Fully mocked (`get_bedrock_client` patched with a minimal fake stream, no real
Bedrock call), proving the override actually changes which model gets requested
and that production's own code path (no `force_tier` key at all) is untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import TextBlock, Usage

from app.config import get_settings
from app.llm import graph as graph_module
from app.llm.graph import agent


class _FakeStream:
    """Just enough of `client.messages.stream()`'s async-context-manager shape
    for `agent()`'s `_call()` to run to completion with a plain text answer -
    no tool use, no streamed delta events (an empty answer is fine for this
    test, which only cares about which `model` kwarg was requested)."""

    def __init__(self, model: str) -> None:
        self._model = model

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return MagicMock(
            content=[TextBlock(type="text", text="ok")],
            usage=Usage(input_tokens=10, output_tokens=2),
        )


def _state(*, force_tier: str | None, **overrides: Any) -> Any:
    base = {
        "tenant": "demo-electronics-in",
        "merchant_id": 1,
        "conversation_id": "test-conversation",
        "messages": [{"role": "user", "content": "compare two things vs each other"}],
        "tool_call_rounds": 0,
        "model_used": None,
        "cache_hit": False,
        "final_answer": "",
        "citations": [],
        "force_no_tools": False,
    }
    if force_tier is not None:
        base["force_tier"] = force_tier
    base.update(overrides)
    return base


@pytest.fixture
def fake_bedrock(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    requested_kwargs: list[dict[str, Any]] = []

    def _stream(**kwargs: Any) -> _FakeStream:
        requested_kwargs.append(kwargs)
        return _FakeStream(kwargs["model"])

    async def _no_op_record_llm_usage(**kwargs: Any) -> None:
        return None

    fake_client = MagicMock()
    fake_client.messages.stream = _stream
    monkeypatch.setattr(graph_module, "get_bedrock_client", lambda: fake_client)
    monkeypatch.setattr(graph_module, "observe_chat_tokens", lambda **kwargs: None)
    # This is a no-stack unit test (`make test`) - `record_llm_usage` opens a real
    # Postgres session, which doesn't belong here any more than a real Bedrock call
    # does. Mocked out the same way `observe_chat_tokens` already is.
    monkeypatch.setattr(graph_module, "record_llm_usage", _no_op_record_llm_usage)
    monkeypatch.setattr(graph_module, "get_stream_writer", lambda: lambda event: None)
    return requested_kwargs


async def test_force_tier_fast_requests_the_fast_model(
    fake_bedrock: list[dict[str, Any]],
) -> None:
    # The message itself reads as "reasoning" via the real lexical router
    # ("compare... vs") - proving the override actually overrides, not just
    # agreeing with what classify_complexity would have picked anyway.
    state = _state(force_tier="fast")
    result = await agent(state)
    assert fake_bedrock[0]["model"] == get_settings().model_fast
    assert result["model_used"] == get_settings().model_fast
    assert "output_config" not in fake_bedrock[0]  # reasoning-tier-only extras


async def test_force_tier_reasoning_requests_the_reasoning_model(
    fake_bedrock: list[dict[str, Any]],
) -> None:
    state = _state(force_tier="reasoning", messages=[{"role": "user", "content": "hi"}])
    result = await agent(state)
    assert fake_bedrock[0]["model"] == get_settings().model_reasoning
    assert result["model_used"] == get_settings().model_reasoning
    assert fake_bedrock[0]["output_config"] == {"effort": "high"}


async def test_no_force_tier_key_falls_back_to_the_real_router(
    fake_bedrock: list[dict[str, Any]],
) -> None:
    """Production's `/chat` handler never sets `force_tier` at all - `.get()`
    must default to `None` and `classify_complexity` must run normally."""
    state = _state(force_tier=None, messages=[{"role": "user", "content": "hi"}])
    assert "force_tier" not in state
    result = await agent(state)
    assert fake_bedrock[0]["model"] == get_settings().model_fast
    assert result["model_used"] == get_settings().model_fast
