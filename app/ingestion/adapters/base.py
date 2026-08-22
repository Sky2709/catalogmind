"""Source adapters: turn one merchant's feed format into canonical `Product` rows.

Onboarding merchant N+1 must not require a code change. That is the single hardest
requirement in this project and the reason adapters exist as a first-class concept
rather than a pile of per-merchant scripts.

The design splits into two layers:

* **`ColumnMapping`** — declarative. Which source column feeds which canonical field,
  plus per-source constants (currency, rating scale). A new merchant whose CSV merely
  uses different column names needs *data*, not code: a mapping can be built from a
  dict, stored per-merchant in Postgres, and supplied at ingestion time.
* **`FeedAdapter`** — code, for the cases declarative mapping genuinely cannot express:
  a nested JSON payload, an XML feed, attributes encoded as a delimited blob.

Most merchants need only the first. Reaching for a subclass should feel like an
escalation, and if it stops feeling that way the mapping layer is too weak.

Adapters never raise on a bad row. They yield `Product | IngestionError`, so a feed
with 400 broken rows out of 50,000 still indexes the 49,600 good ones and hands the
merchant a precise list of what to fix.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.ingestion.normalize import (
    clean_text,
    detect_currency,
    normalise_sku,
    parse_bool,
    parse_int,
    parse_price,
    parse_rating,
    split_categories,
)
from app.models.product import IngestionError, Product

# One source row, already parsed out of CSV/JSON/XML into a flat mapping.
RawRow = Mapping[str, Any]

# What an adapter yields per row: a usable product, or a recorded failure.
RowResult = Product | IngestionError

MAX_DESCRIPTION_CHARS = 4000


@dataclass(frozen=True)
class ColumnMapping:
    """Declarative source-column -> canonical-field mapping.

    Each field names the *source* column. `None` means the source does not provide it.
    `attribute_columns` is the escape hatch: any column listed there is folded into
    `Product.attributes` under its own name, which is how catalogs with wildly
    different specs (wattage, fabric, screen size) share one schema.
    """

    sku: str = "sku"
    title: str = "title"
    description: str | None = "description"
    brand: str | None = "brand"
    category: str | None = "category"
    price: str | None = "price"
    original_price: str | None = None
    currency: str | None = None
    in_stock: str | None = None
    rating: str | None = None
    review_count: str | None = None
    image_url: str | None = None
    product_url: str | None = None

    attribute_columns: tuple[str, ...] = ()

    # --- per-source constants, applied when the feed itself does not carry them ---
    default_currency: str | None = None
    """Authoritative when the source's currency is known out of band. Beats symbol
    sniffing, which cannot distinguish USD from CAD."""

    rating_scale: float = 5.0
    """The source's rating scale. Supplied here rather than guessed per value."""

    default_in_stock: bool = True
    """Feeds that omit a stock column generally list only sellable items."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColumnMapping:
        """Build from stored configuration, ignoring unknown keys.

        Ignoring unknowns rather than raising is deliberate: a mapping persisted by a
        newer version of the service must not break an older worker mid-rollout.
        """
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in data.items() if k in known}
        if "attribute_columns" in payload and payload["attribute_columns"] is not None:
            payload["attribute_columns"] = tuple(payload["attribute_columns"])
        return cls(**payload)


class FeedAdapter:
    """Feed adapter. Usable as-is with a ColumnMapping.

    Deliberately NOT an ABC: the declarative path is the common case, so
    `FeedAdapter(mapping)` must work directly. Subclassing is the escape hatch for
    feeds a mapping cannot express, via the `preprocess`/`postprocess` hooks.
    """

    name: str = "base"

    def __init__(self, mapping: ColumnMapping | None = None) -> None:
        self.mapping = mapping or ColumnMapping()

    # --- the two hooks a subclass may override ---

    def preprocess(self, row: RawRow) -> RawRow:
        """Reshape a source row before mapping. Default: unchanged.

        This is where a nested JSON payload gets flattened, or a delimited spec blob
        gets exploded into real columns.
        """
        return row

    def postprocess(self, product: Product, row: RawRow) -> Product:
        """Adjust the mapped product. Default: unchanged."""
        return product

    # --- the mapping engine, shared by every adapter ---

    def parse_row(self, row: RawRow, row_number: int) -> RowResult:
        """Map one source row. Never raises."""
        try:
            prepared = self.preprocess(row)
            product = self._map(prepared)
            return self.postprocess(product, prepared)
        except _RowRejectedError as exc:
            return IngestionError(row_number=row_number, reason=str(exc), raw=dict(row))
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the feed
            return IngestionError(
                row_number=row_number,
                reason=f"{type(exc).__name__}: {exc}",
                raw=dict(row),
            )

    def parse(self, rows: Iterable[RawRow], *, start: int = 1) -> Iterator[RowResult]:
        """Map an entire feed lazily, so a 5M-row file never lands in memory."""
        for offset, row in enumerate(rows, start=start):
            yield self.parse_row(row, offset)

    # --- internals ---

    def _get(self, row: RawRow, column: str | None) -> Any:
        return row.get(column) if column else None

    def _map(self, row: RawRow) -> Product:
        m = self.mapping

        sku = normalise_sku(self._get(row, m.sku))
        if not sku:
            raise _RowRejectedError(f"missing or empty SKU (column {m.sku!r})")

        title = clean_text(self._get(row, m.title))
        if not title:
            raise _RowRejectedError(f"missing or empty title (column {m.title!r})")

        raw_price = self._get(row, m.price)
        price = parse_price(raw_price)

        # Currency precedence: explicit column > adapter constant > symbol sniffing.
        # Sniffing is last because it cannot distinguish USD from CAD or AUD.
        currency = clean_text(self._get(row, m.currency)).upper() or None
        if not currency:
            currency = m.default_currency
        if not currency and isinstance(raw_price, str):
            currency = detect_currency(raw_price)

        attributes: dict[str, Any] = {}
        for column in m.attribute_columns:
            value = row.get(column)
            cleaned = clean_text(value)
            if cleaned:
                attributes[column] = cleaned

        return Product(
            sku=sku,
            title=title,
            description=clean_text(self._get(row, m.description), max_length=MAX_DESCRIPTION_CHARS),
            brand=clean_text(self._get(row, m.brand)) or None,
            category_path=split_categories(self._get(row, m.category)),
            price=price,
            original_price=parse_price(self._get(row, m.original_price)),
            currency=currency,
            in_stock=(
                parse_bool(self._get(row, m.in_stock), default=m.default_in_stock)
                if m.in_stock
                else m.default_in_stock
            ),
            rating=parse_rating(self._get(row, m.rating), source_scale=m.rating_scale),
            review_count=parse_int(self._get(row, m.review_count)),
            image_url=clean_text(self._get(row, m.image_url)) or None,
            product_url=clean_text(self._get(row, m.product_url)) or None,
            attributes=attributes,
            updated_at=datetime.now(UTC),
        )


class _RowRejectedError(Exception):
    """A row that cannot become a Product. Carries the merchant-facing reason."""


@dataclass
class ParseStats:
    """Outcome counts for one feed. Surfaced on the ingestion job endpoint."""

    total: int = 0
    ok: int = 0
    failed: int = 0
    duplicates: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record_ok(self) -> None:
        self.total += 1
        self.ok += 1

    def record_failure(self, reason: str) -> None:
        self.total += 1
        self.failed += 1
        # Bucket by reason prefix so "missing or empty SKU (column 'id')" aggregates
        # rather than producing 400 unique one-off strings in the report.
        key = reason.split("(")[0].strip()
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def record_duplicate(self) -> None:
        self.duplicates += 1

    @property
    def success_rate(self) -> float:
        return self.ok / self.total if self.total else 0.0


def deduplicate(
    results: Iterable[RowResult], stats: ParseStats | None = None
) -> Iterator[RowResult]:
    """Drop repeated SKUs, keeping the *last* occurrence.

    Last-wins because feeds are commonly appended to, so a later row for the same SKU
    is the more recent truth. This buffers products in memory to see the final
    occurrence; errors stream through untouched.

    Duplicate SKUs are not hypothetical — they appear in most public retail dumps, and
    left alone they would create two Weaviate objects competing for the same query and
    splitting their own relevance.
    """
    seen: dict[str, Product] = {}
    order: list[str] = []

    for result in results:
        if isinstance(result, IngestionError):
            yield result
            continue
        if result.sku in seen and stats is not None:
            stats.record_duplicate()
        elif result.sku not in seen:
            order.append(result.sku)
        seen[result.sku] = result

    for sku in order:
        yield seen[sku]
