"""Offline, one-time category/sub-category enrichment for the three demo catalogs.

None of the three raw feeds has a usable category signal (see
`app/ingestion/taxonomy.py`'s docstring). This script classifies every product
against that module's locked taxonomy using a forced tool/function call.

Deliberately **not** part of `app/ingestion/pipeline.py`: this is a real, paid
LLM call, run once (or re-run only for new/changed products) as a standalone
step, not on the interactive ingestion or search path - the same "LLM calls
belong in generation and in offline ingestion-time attribute extraction"
carve-out CLAUDE.md's "no LLM on the retrieval hot path" invariant already
allows. Results are written to a resumable, git-ignored JSONL artifact
(`data/processed/category_enrichment/<tenant>.jsonl`), one line appended per
completed batch - a crash mid-run only loses the in-flight batch, and a
re-run skips identity keys already present, which also transparently handles
re-classifying only new/changed products on a future feed update.

**Two providers, `--provider {gemini,bedrock}`, default `gemini`.** This is a
deliberate, dated, SCOPED exception to CLAUDE.md's "do not reintroduce Gemini
anywhere" rule - not a reversal of the Day 5 provider decision. Made
2026-08-23 at the user's explicit request, specifically because this is a
one-off ~30,000-item offline batch job and they didn't want to spend AWS
Bedrock quota on it. The original reason Gemini was rolled back (a live chat
request hanging for minutes with no clean error when its quota ran out
mid-turn) doesn't transfer to this script: there is no live request to hang -
a Gemini failure here just fails/retries a batch, offline, with nobody
waiting on it. Nothing in `app/llm/` (the live chat/search/ingestion path)
imports `google-genai` or reads `GEMINI_API_KEY` - that boundary is the
difference between "a scoped exception" and "reintroducing Gemini." See
`app/config.py::Settings.gemini_api_key` and PROGRESS.md for the same account.
The Bedrock/Haiku path (this script's original implementation, tool-forced via
the same idiom already used in `eval/measure_superlative_heuristic.py::_judge`)
is kept fully working behind `--provider bedrock`, not deleted - both share the
identical provider-agnostic JSON-schema tool definition (`_classify_tool`).

**Identity key is per-catalog, not a uniform "dedupe by title"** - verified
against the raw feeds before deciding this (see PROGRESS.md for the full
investigation):
  - fashion/home-goods: exact title text. Fashion's 1,012 duplicate-title
    groups were directly checked to have 0/1,012 unstable gender or brand -
    true colour/size variants of one listing, safe to classify once.
  - electronics: the SKU (ASIN) - `name` is hard-truncated at exactly 125
    characters on 59.7% of rows, so two genuinely different products can
    truncate to an identical string; deduping by title there would silently
    misclassify one of them.

Run: `.venv/bin/python -m scripts.enrich_categories demo-fashion-in --pilot 120`
(the `--pilot` flag caps how many *distinct identity keys* get classified -
use it for the validation pass before spending on the full catalog; omit it
for a full run once the pilot is approved). Requires the `enrichment` extra
(`uv sync --extra dev --extra enrichment`) for `--provider gemini`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import anthropic
from anthropic.types import ToolChoiceToolParam, ToolParam, ToolUseBlock
from google import genai
from google.genai import types as gtypes

from app.config import get_settings
from app.ingestion.taxonomy import UNCATEGORIZED, taxonomy_for
from app.llm.client import ANTHROPIC_TRANSIENT_ERRORS, dispose_bedrock_client, get_bedrock_client
from app.llm.pricing import estimate_cost_usd
from app.retry import with_retry

# Gemini pricing lives here, not in `app/llm/pricing.py` - that table is for the
# shared per-merchant Bedrock cost ledger (`app/llm/cost_tracking.py`), and this
# script's containment promise depends on nothing Gemini-specific leaking into
# `app/llm/`. Rate confirmed via web search 2026-08-23 (introductory pricing,
# through 2026-12-31 per Google's announcement) - same "dated, sourced, flagged"
# discipline `app/llm/pricing.py`'s own docstring already uses.
GEMINI_MODEL = "gemini-3.7-flash"
_GEMINI_INPUT_PER_MTOK = Decimal("0.75")
_GEMINI_OUTPUT_PER_MTOK = Decimal("3.75")


def _gemini_cost_usd(input_tokens: int, output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * _GEMINI_INPUT_PER_MTOK
        + Decimal(output_tokens) * _GEMINI_OUTPUT_PER_MTOK
    ) / Decimal(1_000_000)


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("enrich_categories")

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/category_enrichment")

BATCH_SIZE = 40
ENRICH_CONCURRENCY_LIMIT = 5
_enrich_semaphore = asyncio.Semaphore(ENRICH_CONCURRENCY_LIMIT)

# A sustained ~800-call batch job is exactly the shape that trips Bedrock
# throttling - `app/llm/client.py::ANTHROPIC_TRANSIENT_ERRORS` deliberately
# excludes `RateLimitError` for the interactive chat path (low volume, blind
# backoff on a rate limit there can compound the problem); that reasoning
# doesn't transfer here, so this script gets its own retry policy: the same
# transient-error set plus `RateLimitError`, with a longer backoff than the
# interactive default.
_BATCH_RETRYABLE_ERRORS = (*ANTHROPIC_TRANSIENT_ERRORS, anthropic.RateLimitError)
_BATCH_RETRY_BASE_DELAY = 2.0
_BATCH_RETRY_MAX_DELAY = 45.0
_BATCH_RETRY_ATTEMPTS = 5

_ASIN = re.compile(r"/dp/([A-Z0-9]{10})")

DESCRIPTION_CHARS = 150


@dataclass(frozen=True)
class CatalogSpec:
    tenant: str
    files: tuple[Path, ...]
    title_column: str
    description_column: str | None
    hint_column: str | None  # e.g. home-goods' sparse `rank-sub`
    dedupe_by_title: bool


def _catalog_specs() -> dict[str, CatalogSpec]:
    home_dir = DATA_DIR / "home-shein"
    return {
        "demo-fashion-in": CatalogSpec(
            tenant="demo-fashion-in",
            files=(DATA_DIR / "fashion-myntra" / "Myntra_fashion_products.csv",),
            title_column="name",
            description_column="description",
            hint_column=None,
            dedupe_by_title=True,
        ),
        "demo-electronics-in": CatalogSpec(
            tenant="demo-electronics-in",
            files=(DATA_DIR / "electronics-amazon" / "electronics_product.csv",),
            title_column="name",
            description_column=None,
            hint_column=None,
            dedupe_by_title=False,
        ),
        "demo-home-goods": CatalogSpec(
            tenant="demo-home-goods",
            files=(
                home_dir / "us-shein-home_and_kitchen-3719.csv",
                home_dir / "us-shein-home_textile-3883.csv",
                home_dir / "us-shein-tools_and_home_improvement-3903.csv",
            ),
            title_column="goods-title-link",
            description_column=None,
            hint_column="rank-sub",
            dedupe_by_title=True,
        ),
    }


@dataclass
class WorkItem:
    key: str  # identity key: title text, or ASIN for electronics
    title: str
    description: str
    hint: str | None


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = field(default_factory=lambda: Decimal(0))


def _row_title(spec: CatalogSpec, row: Mapping[str, str]) -> str:
    title = (row.get(spec.title_column) or "").strip()
    if not title and spec.tenant == "demo-home-goods":
        # Same fallback `SheinHomeGoodsAdapter.preprocess` already applies - two
        # of the three source files carry a "Best Sellers"-tagged row's title in
        # `goods-title-link--jump` instead, leaving `goods-title-link` empty.
        title = (row.get("goods-title-link--jump") or "").strip()
    return title


def _row_key(spec: CatalogSpec, row: Mapping[str, str], title: str) -> str | None:
    if spec.tenant == "demo-electronics-in":
        match = _ASIN.search(row.get("link") or "")
        return match.group(1) if match else None
    return title or None


def load_work_items(spec: CatalogSpec) -> list[WorkItem]:
    seen: dict[str, WorkItem] = {}
    for path in spec.files:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                title = _row_title(spec, row)
                if not title:
                    continue
                key = _row_key(spec, row, title)
                if not key:
                    continue
                if spec.dedupe_by_title and key in seen:
                    continue
                description = (
                    (row.get(spec.description_column) or "")[:DESCRIPTION_CHARS]
                    if spec.description_column
                    else ""
                )
                hint = None
                if spec.hint_column:
                    raw_hint = (row.get(spec.hint_column) or "").strip()
                    hint = raw_hint.removeprefix("in ").strip() or None
                seen[key] = WorkItem(key=key, title=title, description=description, hint=hint)
    return list(seen.values())


def _prepare_output(output_path: Path) -> set[str]:
    """Sync (blocking) I/O, run off the event loop via `asyncio.to_thread` - just
    a directory create plus a small JSONL read, but `ruff`'s ASYNC240 correctly
    flags raw `pathlib.Path` blocking calls inside an `async def` body."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        return set()
    keys: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        keys.update(entry["classifications"].keys())
    return keys


def _classify_tool(taxonomy: tuple[str, ...]) -> ToolParam:
    return {
        "name": "classify_products",
        "description": (
            "Record the single best-fit category+subcategory for each numbered "
            "product, chosen from the fixed list of allowed values. Use "
            f"{UNCATEGORIZED!r} only when a product genuinely does not fit any "
            "other option - not as a default for anything mildly ambiguous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "The product's number in this batch.",
                            },
                            "category_path": {
                                "type": "string",
                                "enum": list(taxonomy),
                            },
                        },
                        "required": ["index", "category_path"],
                    },
                }
            },
            "required": ["classifications"],
        },
    }


def _batch_prompt(batch: list[WorkItem]) -> str:
    lines = ["Classify each of these products:\n"]
    for i, item in enumerate(batch):
        parts = [f"{i}. {item.title}"]
        if item.description:
            parts.append(f"(description: {item.description})")
        if item.hint:
            parts.append(f"(scraper category hint: {item.hint})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _extract_result(
    tenant: str, taxonomy: tuple[str, ...], batch: list[WorkItem], by_index: dict[int, str]
) -> dict[str, str]:
    """Shared across both providers: `identity_key -> category_path`, falling
    back to `UNCATEGORIZED` for any index the model omitted or returned outside
    the allowed enum - the forced schema constrains this at the API level on
    both providers, but validating defensively here costs nothing and turns a
    hypothetical SDK/model surprise into a labelled fallback instead of a
    `KeyError`."""
    result: dict[str, str] = {}
    for i, item in enumerate(batch):
        result[item.key] = by_index.get(i, UNCATEGORIZED)
        if i not in by_index:
            logger.warning("%s: no valid classification for %r, defaulting", tenant, item.title)
    return result


async def _classify_batch_bedrock(
    tenant: str, taxonomy: tuple[str, ...], batch: list[WorkItem], usage: UsageTotals
) -> dict[str, str]:
    """One Haiku call via Bedrock, forced to return a `category_path` per item -
    the tool-forced idiom already used in
    `eval/measure_superlative_heuristic.py::_judge`."""
    client = get_bedrock_client()
    tool = _classify_tool(taxonomy)
    tool_choice: ToolChoiceToolParam = {"type": "tool", "name": "classify_products"}

    async def _call() -> Any:
        async with _enrich_semaphore:
            return await client.messages.create(
                model=get_settings().model_fast,
                max_tokens=BATCH_SIZE * 40,
                tools=[tool],
                tool_choice=tool_choice,
                messages=[{"role": "user", "content": _batch_prompt(batch)}],
            )

    response = await with_retry(
        _call,
        retryable=_BATCH_RETRYABLE_ERRORS,
        attempts=_BATCH_RETRY_ATTEMPTS,
        base_delay=_BATCH_RETRY_BASE_DELAY,
        max_delay=_BATCH_RETRY_MAX_DELAY,
    )

    usage.input_tokens += response.usage.input_tokens
    usage.output_tokens += response.usage.output_tokens
    usage.cost_usd += estimate_cost_usd(
        get_settings().model_fast, response.usage.input_tokens, response.usage.output_tokens
    )

    call = cast(ToolUseBlock, next(b for b in response.content if b.type == "tool_use"))
    raw: dict[str, Any] = dict(call.input) if isinstance(call.input, dict) else {}
    classifications = cast(list[dict[str, Any]], raw.get("classifications") or [])
    by_index: dict[int, str] = {}
    for entry in classifications:
        idx, cat = entry.get("index"), entry.get("category_path")
        if isinstance(idx, int) and cat in taxonomy:
            by_index[idx] = cat

    return _extract_result(tenant, taxonomy, batch, by_index)


_GEMINI_RETRYABLE_ERRORS = (
    genai.errors.ServerError,
    genai.errors.ClientError,  # covers 429s too - no separate rate-limit type to exclude
)


async def _classify_batch_gemini(
    tenant: str, taxonomy: tuple[str, ...], batch: list[WorkItem], usage: UsageTotals
) -> dict[str, str]:
    """One Gemini 3.7 Flash call, forced to call `classify_products` via
    `FunctionCallingConfigMode.ANY` - reuses the exact same provider-agnostic
    JSON-schema dict (`_classify_tool`) the Bedrock path builds, passed as
    `parameters_json_schema` (accepts plain lowercase JSON Schema directly,
    confirmed against the installed SDK - `parameters` would need the
    Google-specific uppercase `Schema`/`Type` shape instead, deliberately not
    used here to keep one shared schema builder)."""
    client = _gemini_client()
    tool_def = _classify_tool(taxonomy)
    function = gtypes.FunctionDeclaration(
        name=tool_def["name"],
        description=tool_def["description"],
        parameters_json_schema=tool_def["input_schema"],
    )
    config = gtypes.GenerateContentConfig(
        tools=[gtypes.Tool(function_declarations=[function])],
        tool_config=gtypes.ToolConfig(
            function_calling_config=gtypes.FunctionCallingConfig(mode="ANY")
        ),
        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
    )

    async def _call() -> Any:
        async with _enrich_semaphore:
            return await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=_batch_prompt(batch),
                config=config,
            )

    response = await with_retry(
        _call,
        retryable=_GEMINI_RETRYABLE_ERRORS,
        attempts=_BATCH_RETRY_ATTEMPTS,
        base_delay=_BATCH_RETRY_BASE_DELAY,
        max_delay=_BATCH_RETRY_MAX_DELAY,
    )

    usage_meta = response.usage_metadata
    input_tokens = usage_meta.prompt_token_count or 0
    output_tokens = usage_meta.candidates_token_count or 0
    usage.input_tokens += input_tokens
    usage.output_tokens += output_tokens
    usage.cost_usd += _gemini_cost_usd(input_tokens, output_tokens)

    parts = response.candidates[0].content.parts if response.candidates else []
    call = next((p for p in parts if p.function_call is not None), None)
    raw: dict[str, Any] = dict(call.function_call.args) if call and call.function_call else {}
    classifications = cast(list[dict[str, Any]], raw.get("classifications") or [])
    by_index: dict[int, str] = {}
    for entry in classifications:
        idx, cat = entry.get("index"), entry.get("category_path")
        if isinstance(idx, int) and cat in taxonomy:
            by_index[idx] = cat

    return _extract_result(tenant, taxonomy, batch, by_index)


_gemini_client_singleton: genai.Client | None = None


def _gemini_client() -> genai.Client:
    global _gemini_client_singleton
    if _gemini_client_singleton is None:
        api_key = get_settings().gemini_api_key
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set - required for --provider gemini "
                "(the default for this script). Set it in .env, or pass "
                "--provider bedrock to use Claude Haiku via Bedrock instead."
            )
        _gemini_client_singleton = genai.Client(api_key=api_key)
    return _gemini_client_singleton


async def _classify_batch(
    provider: str, tenant: str, taxonomy: tuple[str, ...], batch: list[WorkItem], usage: UsageTotals
) -> dict[str, str]:
    if provider == "gemini":
        return await _classify_batch_gemini(tenant, taxonomy, batch, usage)
    return await _classify_batch_bedrock(tenant, taxonomy, batch, usage)


async def enrich(tenant: str, pilot: int | None, provider: str) -> None:
    spec = _catalog_specs()[tenant]
    taxonomy = taxonomy_for(tenant)
    items = load_work_items(spec)
    logger.info("%s: %s distinct identity keys to classify", tenant, len(items))

    output_path = OUTPUT_DIR / f"{tenant}.jsonl"
    already_done = await asyncio.to_thread(_prepare_output, output_path)
    items = [item for item in items if item.key not in already_done]
    logger.info("%s: %s already classified, %s remaining", tenant, len(already_done), len(items))

    if pilot is not None:
        items = items[:pilot]
        logger.info("%s: --pilot %s, capping to first %s remaining items", tenant, pilot, pilot)

    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    usage = UsageTotals()

    def _append(result: dict[str, str]) -> None:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"classifications": result}) + "\n")

    async def _run_one(batch: list[WorkItem]) -> None:
        result = await _classify_batch(provider, tenant, taxonomy, batch, usage)
        await asyncio.to_thread(_append, result)

    for i in range(0, len(batches), ENRICH_CONCURRENCY_LIMIT):
        wave = batches[i : i + ENRICH_CONCURRENCY_LIMIT]
        await asyncio.gather(*[_run_one(b) for b in wave])
        logger.info(
            "%s: %s/%s batches done", tenant, min(i + len(wave), len(batches)), len(batches)
        )

    logger.info(
        "%s: done. input_tokens=%s output_tokens=%s cost=$%.4f",
        tenant,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )
    if pilot is not None and items:
        per_item = usage.cost_usd / len(items)
        logger.info(
            "%s: $%.6f/item measured this run - extrapolate to the full catalog "
            "before deciding whether to run without --pilot",
            tenant,
            per_item,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tenant", choices=sorted(_catalog_specs()))
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="Classify only the first N not-yet-classified items - for the validation pass.",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "bedrock"),
        default="gemini",
        help=(
            "gemini (default): Gemini 3.7 Flash, a scoped exception made "
            "2026-08-23 to avoid spending AWS Bedrock quota on this one-off "
            "batch - see module docstring. bedrock: Claude Haiku, this "
            "script's original implementation, kept working."
        ),
    )
    args = parser.parse_args()

    try:
        await enrich(args.tenant, args.pilot, args.provider)
    finally:
        if args.provider == "bedrock":
            await dispose_bedrock_client()


if __name__ == "__main__":
    asyncio.run(main())
