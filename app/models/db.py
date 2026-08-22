"""Relational schema: merchants, their API keys, and their ingestion jobs.

Postgres owns the *control plane* — who a merchant is, what key authenticates them,
what settings they run under. Weaviate owns the *data plane*, one tenant per merchant.
The two are joined by `Merchant.tenant`, which is the string handed to
`collection.with_tenant(...)`.

**API keys are never stored.** Only a SHA-256 hash and a short display prefix are
persisted, so a database dump does not hand over working credentials. The plaintext key
is returned exactly once, at creation, and cannot be recovered afterwards — the same
contract Stripe and GitHub use, and the one a reviewer expects to see.
"""

from __future__ import annotations

import enum
import hashlib
import secrets
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# Keys look like: cm_live_3f9a2b...  The prefix is stored in clear so a merchant can
# tell two keys apart in a UI without us ever holding the secret.
API_KEY_PREFIX = "cm_live_"
API_KEY_BYTES = 32
DISPLAY_PREFIX_LENGTH = len(API_KEY_PREFIX) + 8


def generate_api_key() -> str:
    """A fresh key. `secrets` (not `random`) because this is a credential."""
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(API_KEY_BYTES)}"


def hash_api_key(key: str) -> str:
    """SHA-256 of the key.

    Deliberately not bcrypt/argon2. Those are for *low-entropy human passwords*, where
    slowness is the defence against guessing. An API key here carries 256 bits of
    entropy from `secrets`, so brute force is already impossible and a slow hash would
    only add latency to every single authenticated request.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def display_prefix(key: str) -> str:
    return key[:DISPLAY_PREFIX_LENGTH]


class Merchant(Base):
    """One shop. Maps 1:1 to a Weaviate tenant and a MongoDB collection."""

    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    tenant: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        doc="Weaviate tenant name. Slug-safe, immutable once created.",
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # --- per-merchant retrieval settings ---
    # A real multi-tenant product cannot assume every catalog wants identical search
    # behaviour: a latency-sensitive merchant may want reranking off, and a merchant
    # whose catalog is all part numbers may want a permanently low alpha.
    rerank_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    alpha_override: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Pin the hybrid alpha for this merchant, bypassing the dynamic router.",
    )
    default_currency: Mapped[str | None] = mapped_column(String(3), nullable=True)

    # Stored ColumnMapping. Keeping this in the database is what makes onboarding
    # merchant N+1 a configuration change rather than a deploy.
    column_mapping: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="merchant", cascade="all, delete-orphan"
    )
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="merchant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "alpha_override IS NULL OR (alpha_override >= 0 AND alpha_override <= 1)",
            name="ck_merchants_alpha_range",
        ),
        CheckConstraint("tenant ~ '^[a-z0-9][a-z0-9_-]{1,62}$'", name="ck_merchants_tenant_slug"),
    )

    def __repr__(self) -> str:
        return f"<Merchant {self.tenant!r}>"


class ApiKey(Base):
    """A credential belonging to exactly one merchant.

    Multiple live keys per merchant is intentional: it is the only way to rotate a
    credential without downtime — issue the new one, migrate callers, revoke the old.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(
        ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False, index=True
    )

    key_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True, doc="SHA-256 hex digest"
    )
    key_prefix: Mapped[str] = mapped_column(
        String(32), nullable=False, doc="First few chars, shown in UIs to identify a key"
    )
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    merchant: Mapped[Merchant] = relationship(back_populates="api_keys")

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    def __repr__(self) -> str:
        state = "active" if self.is_active else "revoked"
        return f"<ApiKey {self.key_prefix}... {state}>"


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # Distinct from FAILED: rows were indexed, but some were rejected. A merchant needs
    # to know their catalog is live *and* that 400 rows need fixing.
    PARTIAL = "partial"


class IngestionJob(Base):
    """One catalog upload, tracked so `GET /ingestion/{id}` can report real progress."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(
        ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=False, length=16),
        nullable=False,
        default=JobStatus.PENDING,
    )
    source_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)

    rows_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_skipped: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, doc="Unchanged by content hash - delta detection"
    )
    rows_deleted: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Removed because absent from a full_sync feed - always 0 for a partial/delta upload",
    )

    # Capped sample of failures plus aggregate counts by reason. Storing all 400k
    # errors from a bad feed would be useless to read and expensive to hold.
    errors: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    merchant: Mapped[Merchant] = relationship(back_populates="ingestion_jobs")

    __table_args__ = (
        # The status endpoint always asks "latest jobs for this merchant".
        Index("ix_ingestion_jobs_merchant_started", "merchant_id", "started_at"),
    )

    def __repr__(self) -> str:
        return f"<IngestionJob {self.id} {self.status.value}>"


class LlmUsage(Base):
    """One Claude invocation's token usage and estimated dollar cost - Day 6's
    per-merchant cost ledger (`app/llm/cost_tracking.py`).

    One row per Bedrock call, not per chat turn: a multi-round tool-calling turn
    makes several calls, each with its own `response.usage`, so this is the
    granularity that adds up correctly. No `merchant` relationship - the usage
    endpoint (`app/routers/usage.py`) always aggregates by `merchant_id` directly,
    so a `merchant.llm_usage` traversal would have no caller.

    Isolation here is only as strong as the `WHERE merchant_id = ...` filter every
    query against this table must use - unlike Weaviate's structurally per-tenant
    shards, Postgres has no row-level security in this codebase (confirmed: no
    `CREATE POLICY` anywhere), so this table relies on query discipline the same
    way `ApiKey`/`IngestionJob` already do.
    """

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(
        ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_creation_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 8dp: at Haiku's $1/$5-per-million rates a single short call can cost a small
    # fraction of a cent - fewer decimal places would round real, cheap calls to $0.
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 8), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # The usage endpoint always asks "this merchant's spend, optionally in a
        # date range" - same shape as `ix_ingestion_jobs_merchant_started`.
        Index("ix_llm_usage_merchant_created", "merchant_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<LlmUsage merchant={self.merchant_id} model={self.model} cost={self.cost_usd}>"
