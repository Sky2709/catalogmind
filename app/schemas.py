"""Request and response models for the public API.

Kept separate from `app.models.db` on purpose: the database schema and the API contract
change for different reasons and at different rates. Leaking ORM objects out of
handlers is how internal columns end up in public responses by accident.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.db import JobStatus
from app.retrieval.base import SearchFilters

TENANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


class MerchantCreate(BaseModel):
    """Provisioning request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant": "acme-fashion",
                "name": "Acme Fashion",
                "default_currency": "INR",
                "rerank_enabled": True,
            }
        }
    )

    tenant: str = Field(
        ...,
        min_length=2,
        max_length=63,
        description=(
            "Immutable slug. Becomes the Weaviate tenant and the MongoDB collection "
            "suffix, so it cannot be changed after creation."
        ),
    )
    name: str = Field(..., min_length=1, max_length=255)
    default_currency: str | None = Field(
        None, min_length=3, max_length=3, description="ISO 4217, e.g. INR"
    )
    rerank_enabled: bool = True
    alpha_override: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Pin the hybrid alpha, bypassing the dynamic router. Usually leave null.",
    )
    column_mapping: dict[str, Any] | None = Field(
        None, description="Feed column mapping, so onboarding needs no code change."
    )

    @field_validator("tenant")
    @classmethod
    def _validate_tenant(cls, v: str) -> str:
        # Mirrors the Postgres CHECK constraint. Validating here as well turns a 500
        # from a constraint violation into a readable 422.
        if not TENANT_PATTERN.match(v):
            raise ValueError(
                "tenant must be lowercase alphanumeric, may contain '-' or '_', "
                "and must start with a letter or digit"
            )
        return v

    @field_validator("default_currency")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class MerchantOut(BaseModel):
    """A merchant, as returned to its owner."""

    model_config = ConfigDict(from_attributes=True)

    tenant: str
    name: str
    default_currency: str | None
    rerank_enabled: bool
    alpha_override: float | None
    created_at: datetime


class ApiKeyOut(BaseModel):
    """A key's metadata. Never contains the secret."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    key_prefix: str
    label: str | None
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class MerchantCreated(BaseModel):
    """Provisioning response. The one and only time the key is visible."""

    merchant: MerchantOut
    api_key: str = Field(
        ...,
        description=(
            "Store this now. Only a hash is persisted, so it cannot be shown again - "
            "a lost key must be replaced by issuing a new one."
        ),
    )
    warning: str = "Save this API key now. It will never be shown again."


class ApiKeyCreated(BaseModel):
    """Rotation response."""

    api_key: str
    key: ApiKeyOut
    warning: str = "Save this API key now. It will never be shown again."


class ApiKeyCreate(BaseModel):
    label: str | None = Field(
        None, max_length=128, description="Human note, e.g. 'staging' or 'ci'"
    )


class ErrorResponse(BaseModel):
    detail: str


class IngestionAccepted(BaseModel):
    """Response to a catalog upload. The job runs after this returns - see
    `GET /v1/merchants/{tenant}/ingestion/{job_id}` for progress."""

    job_id: int
    status: JobStatus
    rows_total: int = Field(..., description="Rows read from the upload, before parsing")
    message: str = "Ingestion started. Poll GET .../ingestion/{job_id} for progress."


class IngestionJobOut(BaseModel):
    """One ingestion job's progress and outcome."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: JobStatus
    source_filename: str | None
    rows_total: int
    rows_indexed: int = Field(..., description="New or changed rows embedded and upserted")
    rows_failed: int
    rows_skipped: int = Field(..., description="Unchanged by content hash - re-embed skipped")
    rows_deleted: int = Field(
        ..., description="Removed for being absent from a full_sync feed - 0 unless requested"
    )
    errors: dict[str, Any] | None = Field(
        None, description="Failure reasons aggregated, plus a capped sample of raw rows"
    )
    error_message: str | None = Field(None, description="Set only when the whole job crashed")
    started_at: datetime
    finished_at: datetime | None


class SearchQuery(BaseModel):
    """A search request body. No `tenant` field - see `app.deps` - it always comes
    from the authenticated API key, never from anything the caller can write."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "waterproof hiking boots size 10",
                "limit": 10,
                "filters": {"in_stock_only": True},
            }
        }
    )

    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)
    alpha: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Hybrid blend override. 0.0 = pure keyword, 1.0 = pure vector. Omit to "
            "use this merchant's `alpha_override` if set, else the dynamic router."
        ),
    )
    rerank: bool | None = Field(
        None,
        description="Override this merchant's `rerank_enabled` default for this call.",
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ChatRequest(BaseModel):
    """A chat turn. No `tenant` field - see `app.deps` - it always comes from the
    authenticated API key, never from anything the caller can write."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "I need waterproof hiking boots, size 10",
                "conversation_id": None,
            }
        }
    )

    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(
        None,
        description=(
            "Omit on the first message of a conversation - the response's SSE "
            "stream carries the id to send back on the next turn. Backed by "
            "LangGraph's in-memory checkpointer (`app/llm/graph.py`): scoped to "
            "this process's uptime, not a durable chat history."
        ),
    )


class ModelUsageBreakdown(BaseModel):
    """One model's slice of a merchant's `GET .../usage` totals."""

    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: Decimal


class UsageSummary(BaseModel):
    """A merchant's LLM spend - `app/routers/usage.py`, backed by `LlmUsage`
    (`app/models/db.py`). Cost figures are estimates from `app/llm/pricing.py`'s
    published rates, not a reconciled AWS bill."""

    total_cost_usd: Decimal
    total_calls: int
    by_model: list[ModelUsageBreakdown]
