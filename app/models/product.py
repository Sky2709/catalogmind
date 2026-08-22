"""The canonical product schema.

Every source feed — Flipkart CSV, Google Merchant Center XML, a bespoke JSONL dump —
is normalised into `Product` by an adapter in `app/ingestion/adapters/`. Nothing
downstream of normalisation ever sees a source-specific field.

Design notes:
  * `sku` is the merchant-scoped natural key. Uniqueness is per tenant, never global.
  * `attributes` is deliberately open. Catalogs disagree about what a "product" has,
    and forcing a closed schema is how you end up unable to onboard merchant N+1.
  * `content_hash` drives delta detection so re-ingesting a feed touches only what
    actually changed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    """A normalised, tenant-scoped catalog item."""

    sku: str = Field(..., description="Merchant-scoped unique identifier", min_length=1)
    title: str = Field(..., min_length=1)
    description: str = ""

    brand: str | None = None
    category_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical category, broad to narrow: ['Apparel', 'Men', 'Shirts']",
    )

    price: Decimal | None = None
    currency: str | None = Field(None, description="ISO 4217, e.g. INR, USD")
    original_price: Decimal | None = None

    in_stock: bool = True
    rating: float | None = Field(None, ge=0, le=5)
    review_count: int | None = Field(None, ge=0)

    image_url: str | None = None
    product_url: str | None = None

    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Open key/value bag: colour, size, material, wattage, ...",
    )

    updated_at: datetime | None = None

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, v: str | None) -> str | None:
        return v.upper() if v else v

    @field_validator("category_path", mode="before")
    @classmethod
    def _split_category(cls, v: Any) -> Any:
        """Accept 'Apparel > Men > Shirts' or 'a/b/c' as well as a real list."""
        if isinstance(v, str):
            for sep in (">", "/", "|"):
                if sep in v:
                    return [part.strip() for part in v.split(sep) if part.strip()]
            return [v.strip()] if v.strip() else []
        return v

    def embedding_text(self) -> str:
        """The text actually sent to the embedding model.

        Field order matters: title first (highest signal, survives truncation),
        then brand and category, then description, then flattened attributes.
        Keep this stable — changing it invalidates every stored vector.
        """
        parts = [self.title]
        if self.brand:
            parts.append(self.brand)
        if self.category_path:
            parts.append(" ".join(self.category_path))
        if self.description:
            parts.append(self.description)
        if self.attributes:
            parts.extend(f"{k}: {v}" for k, v in sorted(self.attributes.items()) if v is not None)
        return "\n".join(parts)

    def content_hash(self) -> str:
        """Stable hash over the semantically meaningful fields.

        Excludes `updated_at` so a feed that only bumps timestamps does not force a
        full re-embed. This is what makes re-ingestion incremental.
        """
        payload = self.model_dump(mode="json", exclude={"updated_at"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IngestionError(BaseModel):
    """One row that failed, retained so merchants can fix their feed."""

    row_number: int
    reason: str
    raw: dict[str, Any] | None = None
