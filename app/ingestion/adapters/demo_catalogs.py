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
from app.ingestion.taxonomy import UNCATEGORIZED, load_enrichment_map

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

    The feed carries no category column at all - `category="_display_category"`
    comes from `scripts/enrich_categories.py`'s offline LLM classification
    (`data/processed/category_enrichment/demo-fashion-in.jsonl`), looked up by
    exact title text, the same identity key that script used for this catalog.
    See `app/ingestion/taxonomy.py` for the locked taxonomy and PROGRESS.md for
    why this exists - a "skincare" query was returning makeup/beauty-kit
    products with nothing to structurally tell them apart.
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
                category="_display_category",
                attribute_columns=("gender",),
            )
        )
        self._enrichment = load_enrichment_map("demo-fashion-in")

    def preprocess(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        row = dict(row)
        images = row.get("images") or ""
        first_image = next((part.strip() for part in images.split("~") if part.strip()), None)
        row["_image_url"] = first_image
        title = (row.get("name") or "").strip()
        row["_display_category"] = self._enrichment.get(title, UNCATEGORIZED)
        return row


class AmazonElectronicsAdapter(FeedAdapter):
    """Amazon India electronics export with no id column at all.

    The SKU is the ASIN embedded in the product link - every row in this feed happens
    to have one, but a real onboarding would still need to check that before relying
    on it, which is exactly why this is code and not a `ColumnMapping`.

    `main_category`/`sub_category` are a placebo - every one of the 9,600 rows
    shares the identical value in each, confirmed by direct inspection, not
    assumed. `category="_display_category"` replaces them with
    `scripts/enrich_categories.py`'s offline LLM classification
    (`data/processed/category_enrichment/demo-electronics-in.jsonl`), looked up
    by ASIN (the identity key that script used for this catalog, not title -
    `name` is hard-truncated at 125 characters on 59.7% of rows, so two
    genuinely different products can truncate to an identical string). See
    `app/ingestion/taxonomy.py` and PROGRESS.md.
    """

    name = "amazon-electronics"

    def __init__(self) -> None:
        super().__init__(
            ColumnMapping(
                sku="_asin",
                title="name",
                category="_display_category",
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
        self._enrichment = load_enrichment_map("demo-electronics-in")

    def preprocess(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        row = dict(row)
        match = _ASIN.search(row.get("link") or "")
        asin = match.group(1) if match else None
        row["_asin"] = asin
        row["_display_category"] = self._enrichment.get(asin or "", UNCATEGORIZED)
        return row


class SheinHomeGoodsAdapter(FeedAdapter):
    """The deliberately messy one: no id, no brand, no stock flag, and a "title" field
    that is really SEO keyword-stuffing rather than a product name.

    The SKU is synthesised from a hash of the title *and* derived category - the only
    things in the feed that resemble a stable identity. Two products that scraped to
    an identical title **and** landed in the same category collide into the same SKU;
    for a feed with no id column at all, that is the honest behaviour, not a bug to
    route around. Scoping the hash by category (not title alone) is a real, measured
    fix: of three concatenated per-category source files, 752 identical-title pairs
    turned out to span *different* categories (the same SEO-stuffed boilerplate title
    reused across unrelated listings) - hashing on title alone let a later file's row
    silently overwrite an earlier file's differently-priced, differently-categorized
    product (`parse_and_dedupe`'s last-wins merge). Same-category title collisions
    still collide deliberately - that is a real re-scrape duplicate, not a bug.

    Two of the three source files also split a row's title across two possible
    columns: most rows carry it in `goods-title-link`, but rows flagged with a
    "Best Sellers" rank badge instead carry it in `goods-title-link--jump` (with the
    product URL in `goods-title-link--jump href`) and leave `goods-title-link` empty.
    A prior version of this adapter only ever read the first column, silently
    dropping 77 real products as "missing title" even though the title existed one
    column over in the same row - see PROGRESS.md.

    `_category` (above, `rank-sub`-derived) feeds the SKU digest and is
    deliberately **not** what gets shown/filtered as the product's category -
    `rank-sub` is empty for 78.9% of rows and, where present, an ungrouped
    939-value long tail with no hierarchy. `category="_display_category"`
    (a *different* row field) carries `scripts/enrich_categories.py`'s offline
    LLM classification instead
    (`data/processed/category_enrichment/demo-home-goods.jsonl`, looked up by
    title). Keeping these as two separate fields is deliberate, not an
    oversight: the SKU digest must never change again after the cross-category
    collision fix above shipped, so nothing here may overwrite `_category`
    before it's hashed - see `tests/unit/test_demo_catalogs.py`'s SKU-stability
    test and PROGRESS.md.
    """

    name = "shein-home-goods"

    def __init__(self, default_category: str) -> None:
        super().__init__(
            ColumnMapping(
                sku="_sku",
                title="_title",
                price="price",
                currency=None,
                default_currency="USD",
                category="_display_category",
                product_url="_product_url",
                attribute_columns=("selling_proposition", "discount"),
            )
        )
        self._default_category = default_category
        self._enrichment = load_enrichment_map("demo-home-goods")

    def preprocess(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        row = dict(row)
        title = (row.get("goods-title-link") or row.get("goods-title-link--jump") or "").strip()
        row["_title"] = title
        row["_product_url"] = row.get("goods-title-link--jump href") or None

        # SKU-hash input only - never overwritten by the enrichment lookup below.
        rank_sub = (row.get("rank-sub") or "").strip()
        category = rank_sub.removeprefix("in ").strip() or self._default_category
        row["_category"] = category

        digest = hashlib.sha1(f"{category}:{title}".encode()).hexdigest()[:12]
        row["_sku"] = f"shein-{digest}"

        row["_display_category"] = self._enrichment.get(title, UNCATEGORIZED)
        return row
