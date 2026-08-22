"""Application settings.

Everything is environment-driven (12-factor); see .env.example for the full surface.
Settings are read once and cached so importing this module is cheap anywhere.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- app ----
    environment: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    api_title: str = "CatalogMind"
    api_version: str = "0.1.0"

    admin_token: str = "dev-admin-token-change-me"
    """Guards merchant provisioning.

    Creating a merchant mints a credential and allocates a Weaviate tenant, so it is an
    operator action rather than a public one. Enforced unconditionally - there is no
    "unset means open" path, because that is exactly how a demo default becomes a
    production hole. The development value is published in .env.example on purpose, and
    `validate_production()` refuses to let it reach prod.
    """

    # ---- LLM ----
    bedrock_api_key: str = Field(
        "",
        validation_alias="AWS_BEARER_TOKEN_BEDROCK",
        description=(
            "A Bedrock long-term API key (bearer token), not full AWS IAM "
            "credentials - generated from the Bedrock console. Named to match the "
            "exact env var the `anthropic[bedrock]` SDK and AWS's own tooling "
            "expect, rather than the project's usual `<field>` -> `<FIELD>` "
            "convention, so a shell that already exports it for the AWS CLI works "
            "here without translation."
        ),
    )
    aws_region: str = "us-east-1"
    model_reasoning: str = "anthropic.claude-sonnet-5"
    model_fast: str = "anthropic.claude-haiku-4-5"
    """Provider history, both changes deliberate and dated, not silent drift - see
    `PROGRESS.md`'s "Decisions already locked" section for the full reasoning each
    time:

    1. Anthropic (original plan) -> Google Gemini (2026-08-21): the user had a
       Gemini key on hand, not an Anthropic one.
    2. Gemini -> Anthropic **via AWS Bedrock** (2026-08-21, same day): rolled back
       after Gemini quota ran out mid-build and a live chat request hung for
       minutes with no clean error - the user has Bedrock access and asked to move
       there instead of the direct Anthropic API.

    Bedrock exposes Claude through the same Messages API shape as the direct API
    (`anthropic[bedrock]`'s `AsyncAnthropicBedrockMantle` client - confirmed against
    the reference Bedrock user guide the user supplied, not guessed), which is why
    this rollback restores almost the entire original pre-Gemini design intent:
    `thinking={"type": "adaptive"}`, `output_config={"effort": ...}`, prompt-cache
    checkpoints via `cache_control`, and cost metering off `response.usage` all work
    the same way here as they would against the direct API - the only things that
    actually change are the client class, auth (a Bedrock bearer token via
    `AWS_BEARER_TOKEN_BEDROCK`, not `ANTHROPIC_API_KEY`), and the model ID strings
    (`anthropic.claude-sonnet-5`/`anthropic.claude-haiku-4-5` - Bedrock's bare,
    Messages-API-style names; the boto3 Invoke/Converse APIs would need the fully
    dated `-v1:0` modelId form instead, which this project does not use).
    """

    # ---- Weaviate ----
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_grpc_port: int = 50051

    # ---- Postgres ----
    postgres_dsn: str = "postgresql+asyncpg://catalogmind:catalogmind@localhost:5433/catalogmind"

    # ---- MongoDB ----
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "catalogmind"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6380/0"

    # ---- embeddings (local, CPU) ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"

    embed_query_instruction: bool = True
    """Prefix queries with BGE's retrieval instruction (asymmetric encoding).

    Measured on the real 170-query golden set (`eval/sweep_query_instruction.py`,
    rerank off throughout to isolate the embedding stage, same isolation rationale as
    `eval/sweep_alpha.py`): nDCG@10 0.9216 with the instruction vs 0.8518 without
    (+0.0698), recall@10 0.9408 vs 0.8591. An earlier 4-document smoke test had shown
    no benefit, but that set was saturated (MRR 1.0 in both arms) and proved nothing -
    this is the real measurement that settles it. Kept `True`.
    """

    # ---- retrieval behaviour ----
    default_alpha: float = Field(0.5, ge=0.0, le=1.0)
    dynamic_alpha_enabled: bool = True
    rerank_enabled: bool = True
    retrieve_top_k: int = 10
    """How many hybrid-search candidates get reranked, held over from the Day 3
    hardening pass as a quality/latency tradeoff to settle on the golden sets rather
    than guess.

    `scripts/bench_search.py` measured reranking cost as a function of pool depth:
    2.4s at k=10, 6.1s at k=25, 8.8s at k=50 (`bge-reranker-base` on this dev CPU) -
    reranking is the expensive stage by ~190x over hybrid search alone. Weighed
    against `eval/retrieval_eval.py`'s one rigorous, bias-free quality measurement
    (`rerank_at_judgment_depth` vs `rerank_off`, pool matched to each query's own
    judgment count so the metric never penalises reranking for finding something
    real outside the judged set): the fair nDCG@10 lift is **-0.0313** - negative,
    not merely small. (An earlier run of this same comparison, before this file's
    `retrieve_top_k` default was corrected here and `WeaviateHybridRetriever.search`
    still had the `request.limit`-truncation bug this docstring describes below,
    measured -0.0537 at the old `retrieve_top_k=50` default - same conclusion,
    slightly larger negative gap.) The `retrieve_top_k=50` **shipped-config**
    comparison (as opposed to the fair one) looked far worse still at the time
    (nDCG@10 collapsed to 0.64), but that number was invalidated by exactly the
    judgment-pool-depth confound the fair pass exists to remove (manual inspection
    found real, correct promotions of unjudged items in that run - not proof
    reranking is harmful at depth 50, just unmeasurable with this golden set without
    re-judging it deeper, which was judged not worth the effort). At the now-shipped
    `retrieve_top_k=10`, shipped-config and the fair comparison read almost
    identically (see `eval/results/report.md`) - there is little headroom left for
    reranking to reach past what the metric already judged, which is itself a
    consequence of shrinking this value.

    With no rigorous evidence that any pool deeper than what was fairly tested
    (judgment depths cluster at 1, 4-8, 15) buys quality, and a real latency cost
    that scales with depth, lowered from 50 to 10 - the cheapest benchmarked point,
    matching the depth this measurement actually covers rather than paying for an
    unproven deeper pool. `rerank_enabled` itself stays on (a locked decision, see
    `PROGRESS.md`) - this is a depth call, not a verdict on reranking altogether.
    Not a hard ceiling on results returned: `WeaviateHybridRetriever.search` fetches
    `max(retrieve_top_k, request.limit)` candidates, so a caller asking for more than
    10 results still gets a full page (`app/retrieval/hybrid.py`) - a real bug this
    change surfaced, since `request.limit` goes up to 100 and nothing previously
    guaranteed `retrieve_top_k` covered it.
    """
    rerank_top_k: int = 10

    # ---- caching ----
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = Field(0.95, ge=0.0, le=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def weaviate_http_url(self) -> str:
        return f"http://{self.weaviate_host}:{self.weaviate_port}"

    def validate_production(self) -> None:
        """Refuse to start prod with settings that are only acceptable locally.

        Called from the app lifespan. Crashing at boot is the correct behaviour here -
        a deployment that silently serves with the published admin token is worse than
        one that fails to start.
        """
        if self.environment != "prod":
            return
        problems = []
        if self.admin_token == "dev-admin-token-change-me":
            problems.append("admin_token is still the published development default")
        if not self.bedrock_api_key:
            problems.append("bedrock_api_key is empty")
        if problems:
            raise RuntimeError("unsafe production configuration: " + "; ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
