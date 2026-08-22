"""Adapters for the three demo catalogs seeded by `scripts/seed.py`.

Per the module docstring in `app.ingestion.adapters.base`: a plain `ColumnMapping`
should be the default, and reaching for a `FeedAdapter` subclass should feel like an
escalation. All three demo feeds now earn that escalation - Myntra fashion joined
electronics and home-goods here on 2026-08-22, when its packed-together `images`
column turned out to be the reason product images never reached the chat UI (a plain
`ColumnMapping` has no way to pick one URL out of a delimited blob; see
`MyntraFashionAdapter` below). Before that date it onboarded with a `ColumnMapping`
alone precisely because nothing needed its image column parsed correctly yet.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from app.ingestion.adapters.base import ColumnMapping, FeedAdapter

_ASIN = re.compile(r"/dp/([A-Z0-9]{10})")


class MyntraFashionAdapter(FeedAdapter):
    """Myntra fashion export whose `images` column packs every shot for a listing
    into one `"~"`-delimited string, e.g. `"url1.jpg ~ url2.jpg ~ url3.jpg"` - the
    one thing a declarative `ColumnMapping` genuinely cannot express (it can only
    name a source column, not reshape its value), which is exactly what earns this
    the escalation to a real adapter.

    `Product` has a single `image_url`, not a gallery, so this keeps the first
    listed image and discards the rest - Myntra always lists the primary product
    shot first (verified against all 12,491 rows: every row's first `~`-separated
    segment is itself a valid `http(s)` URL, none empty or malformed).
    """

    name = "myntra-fashion"

    def __init__(self) -> None:
        super().__init__(
            ColumnMapping(
                sku="sku",
                title="name",
                description="description",
                brand="brand",
                price="price",
                currency="currency",
                in_stock="in_stock",
                image_url="_image_url",
                attribute_columns=("gender",),
            )
        )

    def preprocess(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        row = dict(row)
        images = row.get("images") or ""
        first_image = next((part.strip() for part in images.split("~") if part.strip()), None)
        row["_image_url"] = first_image
        return row


class AmazonElectronicsAdapter(FeedAdapter):
    """Amazon India electronics export with no id column at all.

    The SKU is the ASIN embedded in the product link - every row in this feed happens
    to have one, but a real onboarding would still need to check that before relying
    on it, which is exactly why this is code and not a `ColumnMapping`.
    """

    name = "amazon-electronics"

    def __init__(self) -> None:
        super().__init__(
            ColumnMapping(
                sku="_asin",
                title="name",
                category="main_category",
                price="discount_price",
                original_price="actual_price",
                currency=None,
                default_currency="INR",
                rating="ratings",
                review_count="no_of_ratings",
                image_url="image",
                product_url="link",
                attribute_columns=("sub_category",),
            )
        )

    def preprocess(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        row = dict(row)
        match = _ASIN.search(row.get("link") or "")
        row["_asin"] = match.group(1) if match else None
        return row


class SheinHomeGoodsAdapter(FeedAdapter):
    """The deliberately messy one: no id, no brand, no stock flag, and a "title" field
    that is really SEO keyword-stuffing rather than a product name.

    The SKU is synthesised from a hash of that title - the only thing in the feed that
    resembles a stable identity. Two products that scraped to an identical title
    collide into the same SKU; for a feed with no id column at all, that is the honest
    behaviour, not a bug to route around.
    """

    name = "shein-home-goods"

    def __init__(self, default_category: str) -> None:
        super().__init__(
            ColumnMapping(
                sku="_sku",
                title="goods-title-link",
                price="price",
                currency=None,
                default_currency="USD",
                category="_category",
                attribute_columns=("selling_proposition", "discount"),
            )
        )
        self._default_category = default_category

    def preprocess(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        row = dict(row)
        title = row.get("goods-title-link") or ""
        row["_sku"] = f"shein-{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"

        rank_sub = (row.get("rank-sub") or "").strip()
        row["_category"] = rank_sub.removeprefix("in ").strip() or self._default_category
        return row
