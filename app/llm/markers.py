"""The structured marker protocol Claude is instructed to follow
(`app/llm/prompting.py::_SYSTEM_TEXT`), replacing free-text heuristics for
relevance/citation/stat-claim verification with plain, exact string matching.

Why this exists: `app/llm/refusal_text.py`'s `_REFUSAL_CUE` and
`app/llm/citations.py`'s old `_SKU_SHAPED_TOKEN` regex both tried to infer a
structural fact (did the model refuse? did it cite a real product?) by guessing
at the surface shape of freely generated prose. That guess can never be complete -
an open-ended generator has unlimited ways to phrase the same fact, proven live
this session when a real refusal ("didn't find... don't match") slipped past a
cue list that had already been widened twice. The fix is not a better guess, it's
not guessing at all: Claude declares the fact directly, in one of three fixed,
visible ASCII formats, and detection becomes an exact string/substring check.

Three markers, one family (`[[KEYWORD:...]]` or the bare `[[KEYWORD]]`),
deliberately not a bare bracket or plain-English phrase - unlikely to collide
with real retail copy (a title like "10-Pack [Blue]" has a single bracket, never
a doubled one immediately followed by one of these three keywords):

- `[[NO_MATCH]]` - the first characters of the answer, if and only if none of the
  retrieved candidates are a genuine match for the request.
- `[[SKU:X]]` - wraps every product SKU Claude cites.
- `[[STAT:N]]` - wraps every `get_catalog_stats`-backed figure Claude states.

Live-verified (2026-08-22) against real Bedrock calls on both
`anthropic.claude-haiku-4-5` and `anthropic.claude-sonnet-5`: both tiers reliably
emit `[[NO_MATCH]]` for a genuine refusal and wrap every real cited SKU in
`[[SKU:X]]`, with zero hallucinated markers across 6 real test calls, including a
combined case with the prose-trimming system-prompt instruction shipped earlier
the same session. See `PROGRESS.md`'s dated entry for the full account of why the
regex approach was retired.
"""

from __future__ import annotations

import re

NO_MATCH_MARKER = "[[NO_MATCH]]"

_SKU_MARKER = re.compile(r"\[\[SKU:([^\[\]]+)\]\]")
_STAT_MARKER = re.compile(r"\[\[STAT:(-?\d+(?:\.\d+)?)\]\]")


def is_no_match(answer: str) -> bool:
    """True if `answer` opens with the exact `[[NO_MATCH]]` marker - a plain
    string check, never a guess at refusal phrasing."""
    return answer.strip().startswith(NO_MATCH_MARKER)


def extract_cited_skus(answer: str) -> list[str]:
    """Every SKU Claude wrapped in a `[[SKU:...]]` marker, in order, not
    deduplicated (callers that need de-duplication already do it themselves,
    e.g. `find_hallucinated_citations`)."""
    return _SKU_MARKER.findall(answer)


def extract_stat_markers(answer: str) -> list[str]:
    """Every numeric literal Claude wrapped in a `[[STAT:...]]` marker, as raw
    strings (callers convert to `Decimal` themselves so a malformed marker can't
    crash verification - see callers for that handling)."""
    return _STAT_MARKER.findall(answer)
