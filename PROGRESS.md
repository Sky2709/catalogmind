# CatalogMind — Progress Tracker

**This file is the single source of truth for what is done and what is next.**
Claude updates it at the end of every work session. If you are ever confused, read this.

---

## Where we are right now

| | |
|---|---|
| **Current day** | Day 6 🚧 IN PROGRESS. Generation-quality eval done (2026-08-21, 218 real chat scenarios, groundedness 0.982/answer-hit 0.941/refusal-correctness 0.958). **Unplanned, real-bug-driven scope addition (2026-08-22)**: manual live testing of `/ui/` found the chat agent confidently answering a superlative price question wrong (claimed nothing above ₹2,499; real answer ₹58,854) - not on the original 7-day plan, but a genuine correctness bug found by using the product, so fixed properly rather than deferred. Agent's tool surface expanded from 1 tool to 3 (`search_catalog` with real filters, new `get_catalog_stats`, new `get_product_detail`) plus a claim-verification layer - see "Agentic RAG expansion" below for the full account, including a second real live bug (a Bedrock 400 on turn 2+ from an SDK response type that doesn't survive LangGraph's checkpoint round-trip). **Second unplanned fix, same day**: product images turned out to never reach the chat UI at all - three separate drops (Myntra's `image_url` never mapped, `image_url` stripped from every chat citation regardless of catalog, no `<img>` in the UI at all) - see "Image pipeline fix" below. **Third, same day**: once images made the product cards richer, the redundancy of Claude *also* re-listing every result in prose became obvious - trimmed per the user's explicit choice, see "Trim redundant prose" below (which also live-surfaced a real, pre-existing SKU-digit-transposition case that the existing hallucination detector correctly caught). **Fourth, same day**: a correct refusal ("this catalog doesn't carry underwear") still showed 8 irrelevant product cards beneath it - fixed by reusing the eval harness's refusal detector in production for the first time, see "Suppress product cards on a refusal answer" below. **Fifth and largest, same day**: that same regex-based detector missed a second real refusal ("black tshirt for men") within one turn of shipping - user called out regex-guessing an LLM's free text as fundamentally wrong for a production agentic RAG system, asked for a full audit and a real fix for all of it. Retired every free-text correctness heuristic on the chat agent's *own generated output* (refusal detection, SKU-citation verification, stat-claim verification) in favour of a structured `[[NO_MATCH]]`/`[[SKU:X]]`/`[[STAT:N]]` marker protocol Claude declares explicitly - live-tested twice (a 6-call Bedrock smoke test, then all 7 real failing/edge queries through the actual chat graph) before and after implementation. See "Retired every free-text correctness heuristic" below for the full account, including a first hypothesis (a pure retrieval-score threshold) that real measurement disproved before it shipped. **Phase 2 of that same plan, same day**: measured the two remaining named lexical heuristics (`model_router.py`'s escalation logic, `has_superlative_language`) against real outcomes for the first time - the router measurement caught and fixed a real, separate semantic-cache bug (a write path with no `semantic_cache_enabled` guard) before it could give a false reading; the superlative-language measurement found and fixed 3 confirmed vocabulary gaps ("priciest," "average," "total number of") via an LLM-judge comparison, naming 2 more as accepted limitations rather than force a fix. See both Phase 2 entries below. **Sixth, same day**: manual `/ui/` transcript review (user-initiated, not a bug report) surfaced a real frontend leak - the `[[NO_MATCH]]` marker rendered as literal visible bracket text when Claude narrated before a tool call, because the display-strip regex only matched a leading marker and pre-tool-call narration pushed it out of position 0. Fixed by resetting the frontend's raw-text buffer on every `tool_call` event; the same transcript also reconfirmed the already-accepted `refuses()` backstop limitation live, named but left as-is per explicit instruction. See "`[[NO_MATCH]]` marker leaked as raw bracket text" below. **Seventh, same day**: per-merchant cost tracking and the Helm chart both landed, closing two of the three remaining Day 6 items (only the CI eval gate is still open). Planning this surfaced a third, unplanned finding - `app/routers/chat.py` accepted a client-supplied `conversation_id` with no tenant-scoping, a real cross-tenant chat-history leak via LangGraph's checkpointer - fixed the same session, ahead of the two originally-requested items. See "Cost tracking, Helm chart, and a real conversation-isolation leak" below for the full account. |
| **Last verified** | 2026-08-22 |
| **Tests passing** | **511** (457 unit + 54 integration, all real - 0 skipped, real stack + real `AWS_BEARER_TOKEN_BEDROCK`) |
| **Stack** | Running (weaviate · postgres · mongo · redis) — 3 demo merchants seeded |
| **Blocked on you?** | No |

---

## Day 4's two open questions — both settled this session (2026-08-21)

**1. Does the BGE query instruction help?** Yes, measurably. No script had ever
actually tested it — `embed_query_instruction: bool = True` was following the model
card, backed only by an inconclusive 4-document smoke test. Wrote
`eval/sweep_query_instruction.py`, ran it against the real 170-query golden set
(rerank off, to isolate the embedding stage): **nDCG@10 0.9216 with the instruction
vs 0.8518 without (+0.0698)**, recall@10 0.9408 vs 0.8591. Kept `True`, docstring in
`app/config.py` now cites the real numbers instead of the smoke test.

**2. `retrieve_top_k`: keep 50 or drop it?** Dropped to **10**. The corrected,
per-query-judgment-depth-matched `eval.retrieval_eval` run (the one left paused at
the end of the last session) measured reranking's fair nDCG@10 lift as **negative**
(-0.0313 at the final, corrected numbers), against a real latency cost that scales
with pool depth (~2.4s/call at k=10 vs ~8.8s/call at k=50,
`scripts/bench_search.py`). No pool depth showed a positive quality return, so the
cheapest benchmarked depth was kept. `rerank_enabled` itself stays on — that's a
separately locked decision (see "Decisions already locked" below), this was a depth
call only. Full reasoning in `app/config.py`'s `retrieve_top_k` docstring.

**Two real bugs caught while finishing this, both fixed:**
- **`.env`/`.env.example` silently overrode the Python default.** The first attempt
  to change `retrieve_top_k` only edited `app/config.py`'s class default — but
  `RETRIEVE_TOP_K=50` was hardcoded in both env files, and `pydantic-settings` reads
  those over the class default. The first rerun after the "fix" produced numbers
  byte-identical to the old run, which is what caught it. Fixed both env files;
  verified `get_settings().retrieve_top_k` actually resolves to 10 before re-running.
- **`request.limit`-truncation bug in `WeaviateHybridRetriever.search`**
  (`app/retrieval/hybrid.py`), found while sanity-checking what a lower
  `retrieve_top_k` default would even mean: `retrieve_top_k` was a fixed per-instance
  Weaviate fetch size, but `SearchRequest.limit` goes up to 100 per call — any caller
  asking for more results than the fetch size silently got fewer than they asked
  for. Already latent at the old default of 50; would have bitten far more often at
  10. Fixed by fetching `max(retrieve_top_k, request.limit)` instead. All 384 tests
  still pass.

`eval/report.py` also had **hardcoded** `retrieve_top_k=50` strings in the generated
report text — not derived from settings at all, silently violating this project's
own "published numbers are generated, never typed" rule the moment the default
changed. Fixed to read `get_settings().retrieve_top_k` live; `eval/results/report.md`
regenerated and now internally consistent (shipped-config and the fair-comparison
numbers read almost identically at k=10, which makes sense — there's little pool
headroom left for reranking to reach past what's already returned).

---

## How to resume in a NEW chat

If this conversation gets too long or you start fresh, do this:

1. Open a new Claude Code session **in the WSL folder** (`~/catalogmind`)
2. Paste exactly this:

   > Read CLAUDE.md and PROGRESS.md, then continue from where the tracker says we are.

That's it. `CLAUDE.md` explains the project rules and environment; this file says what's
done. Between them the new session has everything it needs. You do **not** need to
re-explain the Rezolve job, the plan, or the WSL setup.

The full 7-day blueprint lives at
`/mnt/c/Users/AKASH/.claude/plans/stateless-churning-sparrow.md` if deeper context is
needed (that is the WSL path; sessions run inside WSL).

---

## Daily startup checklist

```powershell
# 1. In Windows PowerShell, ONCE per login (keeps the databases alive):
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\akash\catalogmind\scripts\wsl-keepalive.ps1
```

```bash
# 2. In WSL:
cd ~/catalogmind
make up        # start databases, waits until they really answer
make test      # confirm nothing is broken (should say 457 passed)
```

---

## The 7 days

### Day 0 — Setup ✅ DONE

- [x] Repo scaffolding, dependency manifest, Makefile
- [x] Docker Compose stack (weaviate, postgres, mongo, redis)
- [x] Settings module + health/readiness endpoints
- [x] WSL2 + Docker Engine installed and working
- [x] Moved project into WSL at `~/catalogmind`
- [x] **Gate passed:** `/health/ready` returns 200 with all four dependencies OK

### Day 1 — Merchant accounts & security ✅ DONE

Building the part where a shop signs up and gets its own private, isolated space.

- [x] Weaviate multi-tenant schema + tenant lifecycle (`app/retrieval/weaviate_client.py`)
- [x] Postgres tables: merchants, api_keys, ingestion_jobs (`app/models/db.py`)
- [x] Alembic migration, applied and verified against the live database
      (`migrations/versions/f5205fcc61fc_*.py`)
- [x] `POST /v1/merchants` — sign a shop up, return its API key (`app/routers/merchants.py`)
- [x] API-key authentication in one dependency; key alone decides the tenant (`app/deps.py`)
- [x] Key rotation + revocation, admin-guarded provisioning
- [x] **Isolation test: PASSING.** Shop A's key returns zero results for a SKU that only
      exists in Shop B — proved at the storage layer *and* the HTTP boundary
      (`tests/integration/test_tenant_isolation.py`, `test_http_isolation.py`)

### Day 2 — Load real catalogs ✅ DONE

- [x] Feed normaliser — messy prices, HTML, encodings (`app/ingestion/normalize.py`)
- [x] Source adapters + column mapping (`app/ingestion/adapters/base.py`)
- [x] Local embeddings on CPU (`app/ingestion/embed.py`) — throughput is a function of
      text length, not a single number: ~590 docs/sec for a title alone down to ~55
      docs/sec for a full product description (measured directly, see below). The
      "289 docs/sec" this line used to say had no test or script behind it anywhere in
      the repo — replaced after finding that out the hard way.
- [x] Ingestion pipeline: batching, delta detection, per-row error reporting
      (`app/ingestion/pipeline.py`). `parse_and_dedupe` and `partition_by_delta` are
      pure/I/O-free by design — unit-tested directly in `tests/unit/test_pipeline.py`
      without touching the stack.
- [x] `POST /v1/merchants/{tenant}/catalog:ingest` + `GET .../ingestion/{job_id}`
      (`app/routers/ingestion.py`). Runs as a FastAPI `BackgroundTask`, not a queue —
      deliberate for now, see the module docstring; `run_ingestion_job(job_id, rows)`
      already takes the exact shape an `arq` task would, so upgrading later is a
      call-site change. **Caught a real bug via the integration test:** the job row
      must be committed *before* scheduling the background task, because
      `BackgroundTasks` run before a `yield`-dependency's post-response cleanup
      (`session_scope`'s commit) — the background task's own session was looking up an
      uncommitted row. Fixed with an explicit `await session.commit()` in the handler.
- [x] Delta detection confirmed against live Weaviate: batch `insert_many` with a
      deterministic id (`generate_uuid5(sku, tenant)`) upserts on a repeated id rather
      than erroring — this is what makes re-ingestion idempotent (see
      `weaviate_client.py::upsert_products`).
- [x] Mongo raw-row storage, one `raw_products_{tenant}` collection per merchant
      (`app/mongo.py`), upserted by SKU for replay/audit.
- [x] Pick and licence-check real datasets (Indian fashion / electronics / messy home).
      Downloaded into `data/raw/` (git-ignored); provenance + licence recorded in
      `data/SOURCES.md`:
      - **Fashion:** Myntra Fashion Products, CC0-1.0, ~12.8K rows
      - **Electronics:** Amazon Electronics Products 10k, ODbL-1.0, ~9.6K rows.
        First pick (Datafiniti's electronics pricing set) was **rejected** —
        CC-BY-NC-SA-4.0 is too ambiguous for a public repo + live demo.
      - **Home goods (deliberately messy):** 3 category files from the "Dirty
        E-Commerce Data" set, ODC-By, ~11.5K rows combined. No SKU/id column at all —
        will need a real `FeedAdapter` subclass, not just a `ColumnMapping`.
- [x] Seed all three demo merchants (`scripts/seed.py`, `app/ingestion/adapters/demo_catalogs.py`).
      Talks to Postgres/Weaviate/Mongo directly (not over HTTP), so `make seed` works
      without `make dev` running. Idempotent — re-running reuses existing merchants and
      skips re-embedding unchanged rows. Fashion onboards with a plain `ColumnMapping`
      (config, no code); electronics and home-goods each needed a real `FeedAdapter`
      subclass — electronics has no SKU column (derived from the ASIN in the product
      link), home-goods has no id column *or* brand/stock/currency (SKU synthesised
      from a hash of the title). Confirms the mapping-vs-adapter escalation story the
      adapter layer was built around, on real data rather than a made-up example.
      **First real run, seeded:**
      - `demo-fashion-in`: indexed 12,491, 0 failed, 0 duplicates
      - `demo-electronics-in`: indexed 9,512, 0 failed, 88 duplicate ASINs (collapsed correctly)
      - `demo-home-goods`: indexed 9,839, 77 failed (`missing or empty title`), 1,587
        duplicate titles collapsed via the SKU hash — the deliberately-messy catalog
        living up to its name.
      **Caught a real perf bug from this run:** `_store_raw_rows` was upserting to
      Mongo one document at a time in a loop — seeding took ~15 minutes end to end,
      almost entirely serial round trips, not the CPU embedding pass sitting next to
      it. Fixed with a chunked `bulk_write`; a repeat (idempotent, so delta detection
      skips all re-embedding) dropped to **5.9 seconds** for all three catalogs
      combined. Worth remembering next time ingestion "feels slow" — check for a loop
      of single-document I/O before blaming the model.
- [x] Chunk/batch sizes across all three batching sites (Mongo `bulk_write`, Weaviate
      `insert_many`, CPU embedding) put through a properly rigorous benchmark, not a
      quick sweep — real `Product` objects from the actual seeded catalogs (not
      synthetic stand-ins), 10 reps per size (embedding: 5, it's ~100x more expensive
      per trial), both a fresh-insert and an upsert-into-existing-data shape, and a
      concurrent-load test for each site. Committed as `scripts/bench_ingestion.py` so
      it's rerunnable, not a one-off. Full reasoning lives in the constants' docstrings
      in `app/ingestion/pipeline.py`; the headlines:
      - **`MONGO_BULK_BATCH_SIZE`: 500 → 2000.** A first, less rigorous pass had landed
        on 1,000; the properly repeated (10x) dual-shape test found 2,000 gave a
        reproducible ~25% edge over 1,000 in *both* shapes independently.
      - **Weaviate `insert_many` batch size: kept at 128.** Re-tested with real
        `Product` properties instead of the toy objects the first pass used — result
        didn't change (fresh-insert slightly preferred 64, upsert slightly preferred
        128, 128 was never the worst in either shape; bigger batches were consistently
        worse in both).
      - **CPU embedding batch size: decoupled from Weaviate's, set to 16
        (`EMBED_MODEL_BATCH_SIZE`).** This was the one genuinely wrong existing value:
        `_embed_and_upsert` was silently inheriting `embed.py`'s unrelated default of
        64 for the actual model call, never explicitly tuned. Measured against the
        real, mixed-length text this pipeline actually embeds (72-3,065 chars, not a
        fixed synthetic length): batch 64 ran ~51.5 docs/sec vs ~60.9 at 16 - an ~18%
        loss that had gone unnoticed. Also tested length-bucketing (sorting texts by
        length before batching, the "proper" fix for padding waste) - it moved
        throughput by under 1% either way, so the much simpler fix (smaller batches)
        was kept and bucketing wasn't implemented.
      - **Concurrency, measured for all three sites:** Mongo and Weaviate both saturate
        around 2 concurrent tenant-writers on this dev stack - aggregate throughput
        barely improves beyond that while each tenant's own effective rate drops by
        more than half. CPU embedding is *no better* once tested correctly (see below)
        - none of the three should be assumed to scale cleanly if multiple merchants'
        ingestion jobs ever run at once; a concurrency limiter is a real future need,
        not implemented here.
      - **Three of my own bugs caught and fixed during this, worth remembering:** (1) a
        "mixed corpus" for the embedding test that was silently 100% one catalog
        because of a slicing bug after concatenating dict values; (2) a crash from
        calling `asyncio.run()` inside code already running inside an event loop; (3)
        an embedding concurrency test that initially *split one shared job* across N
        threads instead of giving each concurrent caller its own full job - which
        produced a falsely optimistic "concurrency helps" result. Corrected, it showed
        the opposite: 2 concurrent full embedding jobs showed no aggregate improvement
        over 1, and 4 concurrent jobs each took ~3.3x longer than running alone -
        consistent with the Mongo/Weaviate finding, not an exception to it.
- [x] Ingestion data-quality report (`app/ingestion/quality.py`, pure/unit-tested +
      `scripts/ingestion_quality_report.py`, orchestrates against live data). Answers
      "did ingestion parse and store this correctly" - a narrower, earlier question
      than `eval/`'s retrieval-quality metrics (nDCG/recall/MRR), which need a search
      endpoint and a golden query set that don't exist until Day 3/4. Checks: parse
      coverage, per-field completeness, price/rating/description-length distributions
      (percentiles, not just mean), category-field concentration, an **exhaustive**
      Weaviate storage round-trip check (every product in every catalog, not a
      sample - one iterator pass per tenant, not one round trip per SKU), and a
      human-readable spot-check printout. All three catalogs: zero storage drift.
      **Anomaly thresholds are grounded in named, sourced statistical methods, not
      picked by eyeballing three catalogs** - each was found to have a real problem
      on the first honest attempt, which is exactly why this mattered:
      - **Price outliers**: Tukey's "far-out" IQR fence (k=3), applied on the *log* of
        price - raw-scale Tukey false-positived on a smooth, entirely legitimate
        geometric price spread (real prices are right-skewed/log-normal; a linear
        fence eventually flags any long tail). Verified against both a real junk
        value and a real legitimate spread before trusting it.
      - **Category concentration**: Herfindahl-Hirschman Index, the US DOJ/FTC
        merger-guideline concentration measure (bands: >0.25 high, 0.15-0.25
        moderate), replacing a hand-picked "one value is >90%" cutoff.
      - **Duplicate rate**: Grubbs' test for a single outlier, using the real
        published critical-value table (source: NIST-derived, chem.libretexts.org),
        self-calibrated against the other catalogs in the same run rather than a
        fixed percentage. Caught a real bug in the first version of this: a flat
        "z-score > 2" rule is *mathematically incapable of ever firing* with only 3
        catalogs (the max possible z-score for one differing value among n peers is
        capped at sqrt(n-1) ~1.41 at n=3) - it looked validated but was silently
        untestable at this project's actual scale. Grubbs' per-n critical values
        (1.155 at n=3) exist specifically to avoid that trap.
      **Real, confirmed findings from the fully-validated version:**
      - Home-goods' max price is **$888,888** - a literal junk/placeholder row in the
        source CSV (title: `"This Is A Very Goods Product Name A A A A !!! 00000..."`),
        parsed correctly (it *is* what the row says) but not a real price. Left in
        place, not silently filtered - flagging it for a human to decide is the right
        behaviour for a demo of messy-data handling.
      - Home-goods' duplicate rate (14%) is a genuine Grubbs-test outlier relative to
        fashion (0%) and electronics (0.9%) - expected, given its SKU is synthesised
        from a title hash rather than a real source ID, but now actually detected
        rather than silently within an untestable threshold.
      - Electronics' category field is **highly concentrated** (HHI=1.00 - literally
        every one of 9,512 products shares the single value `"tv, audio & cameras"`)
        and home-goods' is concentrated too (HHI=0.62 - 79% default-fallback into
        "Home Goods" when the source lacked a sub-category) - both parse fine, both
        weak as a filter/ranking signal.
      - Electronics and home-goods both have 0% description coverage (embeddings for
        both lean entirely on title).
      - **A real false positive caught and fixed the same way**: fashion's Tukey
        price-fence flagged a ₹63,090 item - which turned out to be a genuine Garmin
        Forerunner 945 GPS smartwatch, alongside real MOVADO/SEIKO luxury watches.
        No purely statistical price test can distinguish "the top of a real premium
        cluster" from "one junk value" without product-type context it doesn't have -
        that's a structural limit of the method, not a bug to tune away. Fixed by
        having the report print the actual highest-priced products next to any such
        flag, so a human can tell the difference in seconds instead of going back to
        the raw CSV (which is what it took to catch this one).
- [x] Embedding *semantic* quality checks (`app/ingestion/embedding_quality.py`,
      pure/unit-tested + `scripts/embedding_quality_report.py`). Answers "does the
      vector space actually carry meaning" without needing a search endpoint or a
      golden query set - average pairwise similarity (a collapse sanity ceiling),
      group contrast (do products sharing a real, unused-by-the-embedding attribute
      land closer together than random pairs?), a nearest-neighbour spot check, and
      a handful of hand-written probe queries (explicitly *not* a substitute for Day
      4's real golden set - just enough to catch an obviously broken pipeline early).
      **Real findings:** embeddings are not collapsed (avg similarity 0.59-0.63, well
      below 1.0, across all three catalogs). Fashion shows strong measured structure
      (brand contrast +0.274 - DKNY clusters with DKNY, Gini and Jony jeans with Gini
      and Jony jeans). Electronics and home-goods show weaker *measured* contrast
      (+0.05-0.06), but that is at least partly because the only available grouping
      variables for them are weak (a rough keyword-guessed product type for
      electronics; a category field that is 79% generic fallback for home-goods) -
      their nearest-neighbour spot checks show real, correct clustering regardless
      (Redmi phones with Redmi phones, OPPO with OPPO, food-storage containers with
      food-storage containers). **The one genuinely actionable finding**: generic
      category queries underperform pure vector search - "smartphone" retrieves a
      tripod and a phone stand in the top-3, not an actual phone; "laptop" retrieves
      a laptop table and a mouse. Product titles are dominated by specific
      brand+model tokens ("Redmi 10 Power", "OPPO A74 5G") rather than generic
      category words, so a broad query has weak vector-similarity to any single
      listing. This is real, catalog-specific empirical evidence for exactly the
      failure mode the planned hybrid search + dynamic alpha routing (Day 3/4)
      exists to address - not a hypothetical risk, something already observed.

### Production-hardening pass (5 gaps closed, not part of the original Day 2 plan)

A prior review flagged five concrete gaps: no CI, no retry/backoff, no rate limiting,
no stale-product deletion, no concurrency limiter despite measured contention. All
five closed, tested, and verified against the live stack - not just written.

- [x] **Retry/backoff** (`app/ingestion/retry.py`, 5 unit tests with a fake clock -
      no real waiting). Exponential backoff + jitter, scoped deliberately narrow: only
      genuinely transient failures (`WeaviateConnectionError`/`WeaviateTimeoutError`/
      `WeaviateGRPCUnavailableError`, pymongo's `ConnectionFailure`) get retried. A
      schema/validation error is never retried - retrying can't fix it and would only
      delay surfacing a real problem. Wired around the three actual I/O calls in
      `app/ingestion/pipeline.py`: the Weaviate content-hash fetch, the Weaviate
      upsert, and the Mongo bulk write.
- [x] **Concurrency limiter** (`INGESTION_CONCURRENCY_LIMIT = 2` in `pipeline.py`, an
      `asyncio.Semaphore`; 2 unit tests proving it actually caps concurrent holders and
      fully releases afterward). The number is not a guess - it's the exact saturation
      point `scripts/bench_ingestion.py` measured for Mongo, Weaviate, and CPU
      embedding. Documented limitation: this is a per-process semaphore, so it does
      not help once the app runs as more than one worker process - a real gap for a
      future queue-based worker to close with a shared (e.g. Redis-backed) limiter.
- [x] **Stale-product deletion** (`full_sync` param on `ingest()`/`run_ingestion_job`/
      the `POST .../catalog:ingest` endpoint; `wv.delete_products_by_sku`, verified
      against live Weaviate before wiring in; new `rows_deleted` column via Alembic
      migration `1677b437ffb6`; 2 integration tests). Deliberately **opt-in, default
      off**: a merchant sending today's 50 price updates must never have their other
      49,950 products silently deleted for not being mentioned. Only a caller that
      knows a feed is the complete, current catalog should request `full_sync=true`.
- [x] **Rate limiting** (`app/rate_limit.py` + `app/redis_client.py`, 6 unit tests
      against a tiny in-memory fake + 3 integration tests against the real Redis).
      Fixed-window counter (`INCR` + `EXPIRE` on first increment only, so a steady
      trickle of requests can't hold one window open forever), 10 uploads/60s, keyed
      by tenant from the API key so one merchant's uploads can never count against
      another's. Applied as a FastAPI dependency on the ingest endpoint specifically,
      not globally.
- [x] **CI** (`.github/workflows/ci.yml`). Two jobs mirroring `make test` vs
      `make test-all`: a fast lint+typecheck+unit job with no datastores, and an
      integration job that boots the real stack via `make up`. Deliberately runs the
      *same* `make` targets a human runs locally rather than reimplementing the same
      checks in YAML, so local dev and CI cannot quietly drift apart. Every individual
      command it invokes (`make install/lint/typecheck/test/up/migrate/test-all`) was
      verified locally, one at a time - but the workflow itself has **not** been run
      by GitHub Actions, because there is no GitHub remote yet (still on the pending
      list below). Deliberately excludes an eval-quality gate - that needs a golden
      query set and a working search endpoint, neither of which exist until Day 3/4.

### Day 3 — Search ✅ DONE

- [x] Hybrid search (keyword + vector blended) against Weaviate
      (`app/retrieval/hybrid.py::WeaviateHybridRetriever`). Implements the `Retriever`
      protocol that was already scaffolded in `base.py`. Pipeline per request:
      classify + pick alpha (`alpha_router.py`, no I/O) → embed the query (same BGE
      model/instruction as ingestion) → `collection.query.hybrid(alpha=..., vector=...,
      fusion_type=RELATIVE_SCORE)` for `retrieve_top_k` candidates → optional rerank →
      slice to `request.limit`. Every stage timed into
      `SearchResponse.stage_timings_ms`.
- [x] Cross-encoder reranking (`app/retrieval/rerank.py`, `BAAI/bge-reranker-base`,
      local CPU - same reasoning as `embed.py` for local BGE: sweeps need free,
      unlimited re-scoring). `rerank_text()` deliberately mirrors
      `Product.embedding_text()`'s field order (title, brand, category, description,
      attributes) but never includes the SKU - the identifier query class is meant to
      be won by BM25, not patched up by the reranker; a unit test
      (`test_hybrid.py`/`test_search.py`) pins that a raw SKU search only surfaces
      top-1 with `rerank=false`.
- [x] `POST /v1/merchants/{tenant}/search` (`app/routers/search.py`). Request body has
      no `tenant` field (`SearchQuery` in `schemas.py`) - tenant is always the
      authenticated merchant, per `app/deps.py`'s invariant. Precedence for `alpha`
      and `rerank`: explicit request value → merchant's stored
      `alpha_override`/`rerank_enabled` → dynamic router / global default. That
      resolution happens in the router handler, not the retriever, so `Retriever`
      stays generic (tenants and queries only, no merchant-config awareness).
- [x] **New shared-connection pattern for Weaviate**
      (`weaviate_client.py::get_shared_client`/`dispose_shared_client`). Ingestion and
      merchant provisioning still open a short-lived client per call (one-off
      operations; a gRPC handshake is invisible next to an embedding batch). Search
      runs on every query, so it gets a process-wide cached client instead - same
      lazy-singleton shape as `get_redis_client`/`get_mongo_client`, adapted with an
      `asyncio.Lock` because Weaviate's async client needs an explicit `await
      .connect()` that the other two don't. Wired into `main.py`'s lifespan shutdown.
- [x] Structured pre-filters (`price`, `brands`, `categories`, `in_stock_only`) via
      Weaviate `Filter` objects, ANDed. Brand uses `equal` (OR'd across the list) for
      exact scalar match rather than `contains_any`'s tokenised-keyword semantics;
      `category_path` uses `contains_any` correctly since it's a real `TEXT_ARRAY`
      (array membership, not text search).
- [x] 21 new tests: 8 unit (`test_hybrid.py`, pure filter/rerank-text/hit-assembly
      helpers, no live Weaviate needed) + 5 unit (`test_rerank.py`, real model on CPU,
      same tradeoff as `test_embed.py`) + 13 integration (`test_search.py`, real stack
      end to end - identifier/exploratory query classes, alpha override, all four
      filter kinds, rerank on/off, and a dedicated cross-tenant isolation check at the
      search endpoint itself, matching the rigor of
      `test_tenant_isolation.py`/`test_http_isolation.py`).
      **Caught a real bug in the test fixture, not the app**: the CSV builder used a
      naive `",".join()` instead of the `csv` module, so a description field
      containing a literal comma ("Grippy sole, ankle support") silently shifted every
      later column - `csv.DictReader` then produced a `None` key for the overflow
      field, which crashed `_bulk_write`'s Mongo insert with `InvalidDocument` deep in
      the ingestion pipeline. Nothing wrong in `app/`; fixed by writing the test CSV
      properly quoted via `csv.writer`.

### Production-hardening pass #2 (5 gaps closed, found via self-review of Day 3)

Asked what would actually justify calling Day 3 "production grade" rather than just
"functionally correct." Answer: the 13 search integration tests prove the wiring
works, not that it performs or survives real load - those are different claims. Five
concrete gaps named and closed, the last one turning up a genuinely severe finding
that reshaped its own fix.

- [x] **Rate limiting on `/search`** (`SEARCH_RATE_LIMIT = 120`/60s per tenant,
      `app/routers/search.py`, reusing `app/rate_limit.py`). Set far higher than
      ingestion's 10/60s deliberately - search is the normal per-shopper-query path,
      not an occasional upload, so this exists only to stop a runaway client, not to
      constrain real traffic. 3 new integration tests mirroring
      `test_rate_limit.py`'s coverage of the ingest limiter.
- [x] **Resilience**: the one Weaviate I/O call on the search hot path
      (`collection.query.hybrid()`) now goes through the same `with_retry` +
      transient-error tuple ingestion already uses, so a dropped connection or
      timeout is retried instead of becoming a 500. Moved `with_retry` out of
      `app/ingestion/retry.py` to top-level `app/retry.py` and
      `WEAVIATE_TRANSIENT_ERRORS` out of `pipeline.py` into `weaviate_client.py` in
      the same change - both were ingestion-only in name only, and search needed the
      exact same tools.
- [x] **Warm-up at startup**: both the embedder and the reranker are now loaded and
      run once (`warm_up()`) in `main.py`'s lifespan, concurrently via
      `asyncio.to_thread` + `asyncio.gather`. First real shopper request no longer
      eats a multi-second cold model load. Confirmed lifespan does not fire under the
      test client construction this project's tests use (`ASGITransport` without a
      lifespan manager), so this adds no time to the suite.
- [x] **Observability**: `app/obs/metrics.py` - Prometheus histograms for
      `embed_ms`/`hybrid_search_ms`/`rerank_ms`/`total_ms` (labelled by `stage` only,
      deliberately never by tenant - that's the textbook Prometheus cardinality
      mistake) plus a request counter, scraped at `/metrics`
      (`app/routers/metrics.py`, unauthenticated like `/health`, hidden from the
      public OpenAPI schema). `histogram_quantile()` gets p50/p95/p99 correctly
      across every request and, eventually, every replica - the per-response
      `stage_timings_ms` field can't do that on its own, it only ever describes one
      request.
- [x] **Rerank latency under concurrent load, measured** (`scripts/bench_search.py`,
      committed and rerunnable, same rigor bar as `bench_ingestion.py`; shared
      `percentiles()`/`print_latency_table()` helpers extracted to
      `scripts/bench_utils.py` so both scripts report identically). This is the gap
      that mattered most:
      - **Rerank cost scales with candidate count, and it's not cheap**: reranking
        real fetched candidates from `demo-fashion-in` against a real query cost a
        mean of **2.4s at k=10, 6.1s at k=25, 8.8s at k=50** (10 reps each,
        `bge-reranker-base` on this dev CPU). Hybrid search alone for the same query
        took 47ms - reranking is **~190x** the cost of the search stage it sits on
        top of. `retrieve_top_k=50` (the shipped default) means every reranked
        search pays the ~8.8s number today.
      - **Concurrency doesn't just fail to help, it's actively pathological.**
        Aggregate throughput for full embed+hybrid+rerank calls fell monotonically:
        **0.19 → 0.14 → 0.09 → 0.04 req/sec** at concurrency 1/2/4/8. Per-call p50
        latency: **3.8s → 11.6s → 35.9s → 193.5s.** The jump from 4 to 8 is the
        damning number - 8 serialized ~8.7s rerank calls would cost a queued caller
        roughly 70s worst-case, not the observed 193.5s p50. That gap (~3x worse than
        pure queueing predicts) is consistent with CPU cache/thread contention inside
        torch, not useful parallel work - each `CrossEncoder.predict()` call already
        tries to use multiple threads, and 8 of them fighting over a handful of cores
        makes every one of them slower than running alone would have. This is a
        sharper version of Day 2's CPU-embedding concurrency finding, not a new
        phenomenon - but nobody had checked whether reranking behaved the same way
        until now.
      - **Fix applied**: `RERANK_CONCURRENCY_LIMIT = 1`
        (`app/retrieval/rerank.py`, an `asyncio.Semaphore`, same shape as
        `INGESTION_CONCURRENCY_LIMIT`; 2 new unit tests proving it actually caps
        concurrent holders and fully releases afterward). This does not make
        reranking fast - it turns the *pathological* blowup back into plain,
        predictable queueing. The underlying ~8.8s-per-search cost at
        `retrieve_top_k=50` is still there and is a real open question, not resolved
        by this pass - see below.

**Decision on the ~8.8s rerank cost**: two separate moves, not one.

- [x] **Done now, because it's free, not a tradeoff**: reranking is skipped
      automatically for a query containing a real SKU/model-number-shaped token
      (`has_identifier_shaped_token()`, `app/retrieval/alpha_router.py`). `rerank_text()`
      never includes the SKU, so a cross-encoder scoring "DW-4402B" against title text
      has no real signal to act on - paying `retrieve_top_k` cross-encoder passes there
      is pure cost, not a quality/latency tradeoff. Deliberately **not** gated on
      `classification.query_class == IDENTIFIER`: that class also catches a plain short
      query like "hiking boots" via its no-competing-signal fallback (nothing else
      fired, so IDENTIFIER wins by default) even though it has no product-code token at
      all - and that kind of query genuinely can benefit from reranking. Caught this
      distinction by actually running `classify("hiking boots")` while implementing the
      fix, not by assuming the class label meant what its name suggests.
      **A real bug surfaced and fixed while wiring this in**: the first cut skipped
      reranking for a real SKU query in the unit-level predicate but not through the
      actual `/search` endpoint - the router was resolving
      `rerank = body.rerank if body.rerank is not None else merchant.rerank_enabled`
      into a concrete `bool` *before* the retriever ever saw the request, so
      `request.rerank` was never `None` in practice and the new heuristic had no state
      to act on. Fixed by splitting the field in two: `SearchRequest.rerank` (an
      explicit per-call override, still nullable) and a new `SearchRequest.rerank_default`
      (the merchant's stored preference) - so precedence is now correctly explicit call
      value → identifier heuristic → merchant default → global setting, with the
      heuristic sitting *between* the two states the router used to collapse into one.
      `alpha` didn't need the same fix: the retriever has no alpha heuristic sitting
      between "caller said nothing" and "merchant's stored value," so collapsing it at
      the router is safe there and only `rerank` needed the extra field. 15 new/updated
      tests across `test_alpha_router.py` (the predicate itself, including the
      "hiking boots" false-positive), `test_search.py` (end-to-end through the real
      endpoint), and `test_rerank_concurrency.py`/`test_obs_metrics.py` from the rest of
      this hardening pass.
- [ ] **Held for Day 4, not guessed**: `retrieve_top_k=50` itself. It is also now a
      measured ~8.8s-per-search cost whenever reranking *does* run (the shipped
      default, for every non-identifier query). Lowering it would cut rerank latency
      close to linearly (2.4s at k=10) but feeds the reranker fewer candidates to
      recover from a mediocre hybrid ranking - a real quality/latency tradeoff, not a
      free win, and exactly the kind of number this project's own rule ("alpha values
      are measured, not guessed") says should be tuned against the golden sets rather
      than picked from a hunch. Flagged for a decision before or alongside the Day 4
      sweep.

### Day 4 — Measure it ⭐ the part that impresses them ✅ DONE

- [x] Metric maths: nDCG, recall, precision, MRR (`eval/metrics.py`)
- [x] Query classifier + dynamic alpha router (`app/retrieval/alpha_router.py`)
- [x] Golden query sets — ~60 labelled queries per merchant, **170 total** (59
      fashion / 55 electronics / 56 home goods — the blueprint's "~60" was approximate;
      each catalog's real structure decided the exact split, see below). Every
      judgment anchored to a real product, live-verified against the tenant this
      session (`eval/golden/demo_fashion_in.py`, `demo_electronics_in.py`,
      `demo_home_goods.py`; construction helpers in `scripts/build_golden_sets.py`;
      loader in `eval/golden/__init__.py`). Phase A (10 fashion queries) was reviewed
      and approved before scaling to the full set.
      **Per-catalog construction had to differ**, driven by what sampling the live
      data actually showed: fashion has real numeric SKUs and populated
      brand/gender fields, so identifier queries use the raw SKU and attribute
      queries lean on brand+garment-type (plus required Hinglish/code-mixed queries
      for the Indian tenant, e.g. `"shaadi ke liye kurta women"`). Electronics has
      `brand=null` on every row and a single-value `category_path` (Day 2's own
      HHI=1.0 finding) - identifier queries use a distinctive model-name phrase from
      the real title instead of the ASIN, verified to land top-1. Home goods has
      synthetic hash SKUs no shopper would ever type and directly surfaced Day 2's
      ~14% duplicate-rate finding *during construction*: two identifier candidates
      that should have had one clear target instead matched 2-3 near-identical
      listings for the same real product - kept as ATTRIBUTE queries with every
      variant judged relevant rather than papered over, and documented in
      `demo_home_goods.py`'s module docstring.
      **A real classifier bug caught building the sample, fixed immediately** (not
      deferred): running the first 10 fashion queries through the actual retriever
      exposed that `alpha_router.py`'s `_EXPLORATORY_CUE` regex listed
      "casual"/"formal" as occasion words, so "Raymond formal shirt" misclassified as
      EXPLORATORY off that one word alone - in fashion e-commerce those words name a
      garment *style*, not an occasion the way "wedding"/"party" genuinely are. Not
      cosmetic: it took one query's nDCG@10 from ~1.0 to 0.0 at the wrong class's
      alpha. Fixed by moving both words to `_ATTRIBUTE_CUE`; 7 new tests in
      `test_alpha_router.py` pin the fix without swallowing genuinely occasion-shaped
      queries that happen to use the same words. All tests pass, nothing broke.
- [x] **The alpha sweep** — `eval/sweep_alpha.py`, run against all 170 golden queries,
      11 alpha values, reranking off throughout (isolates the hybrid-blend stage; also
      the only computationally feasible option given reranking's measured ~8.8s/call
      cost). **Measured, not guessed, per-class best alpha**:
      identifier=0.1 (nDCG@10=1.0000), attribute=0.5 (nDCG@10=0.9935),
      exploratory=0.8 (nDCG@10=0.9265) - the exact low/mid/high shape the blueprint
      predicted, now with real numbers behind it. **Best single fixed alpha across
      all three classes: 0.5** (macro nDCG@10=0.8920) vs **dynamic routing**
      (each class its own best alpha): macro nDCG@10=0.9733 - a measured
      **+0.0813** advantage for dynamic alpha over the best possible single fixed
      value. `eval/results/tuned_alpha.json` now holds these real values;
      `alpha_router.py` loads them instead of `PRIOR_ALPHA` guesses starting now.
      Chart at `eval/results/alpha_sweep.png`.
      **A second and third real bug caught by this pass, not by inspection.**
      The first retrieval-quality run (rerank on, `retrieve_top_k=50`) showed nDCG@10
      *collapsing* by roughly half everywhere reranking touched (overall 0.93 →
      0.52). Investigating one case (`"wireless bluetooth mouse"`) showed the
      reranker correctly promoting two genuinely relevant items (a Zebronics mouse
      at hybrid-rank 14, an HP 430 mouse at rank 11) that were simply never in this
      project's own judgment pool - built by hand-verifying only the top ~6-15 hits
      per query, shallower than the 50-candidate pool reranking actually searches. A
      metric has no way to distinguish "found something good outside the pool" from
      "found something irrelevant," so it scored both as wrong - the textbook IR
      judgment-pool-depth problem (the reason TREC pooling exists), not a reranker
      defect, confirmed by manual inspection rather than assumed.
      **Fixed the comparison, not the golden set**: re-verifying 170 queries at depth
      50 was judged not worth the effort for what it would prove. `WeaviateHybridRetriever`
      gained an explicit `retrieve_top_k` constructor override (production's
      singleton retriever never passes it, so this doesn't touch production
      behaviour), and `eval/retrieval_eval.py` runs a third pass comparing rerank-off
      against reranking with the candidate pool matched to depth.
      **First attempt at that third pass used one blanket `retrieve_top_k=15` for
      every query and still measured a large, suspicious negative lift** (-0.2264
      overall). Direct inspection of a specific query
      (`"office ke liye formal shirt mard"`, judged to depth 6) showed the exact same
      pool-depth problem recurring at smaller scale: many golden queries -
      especially exploratory and Hinglish ones - were verified shallower than 15
      (often 4-8 items), so a 15-candidate pool still let reranking correctly
      promote unjudged-but-real items past the verified ones. Fixed by making the
      comparison's pool depth **per-query** (`retrieve_top_k = len(query.judgments)`
      for that specific query, via a small cache of retrievers keyed by pool size),
      not one constant for all 170.
      **That same investigation surfaced a third, independent, genuine bug**: fashion's
      identifier queries are bare numeric SKUs (`"10015819"`, no letters at all), and
      `has_identifier_shaped_token()`'s `_IDENTIFIER_TOKEN` regex required at least
      one letter in every one of its three original branches - so fashion's
      identifier queries were never recognized as identifier-shaped, never skipped
      reranking, and collapsed from nDCG@10=1.0 to 0.0158 exactly the way
      Day 3's original fix was supposed to prevent. Fixed by adding a fourth
      alternative (`^\d{6,}$` - 6+ digits, chosen so it doesn't fire on ordinary
      quantities like "500ml" or "size 10") to `_IDENTIFIER_TOKEN`; 6 new tests in
      `test_alpha_router.py` pin both the fix and that short non-SKU numbers
      ("999", "12345") stay unaffected. Confirmed live: fashion identifier queries
      now correctly report `reranked=False` and hold nDCG@10=1.0 even at
      `retrieve_top_k=50`. 384 tests pass throughout all of this, nothing broke.
- [x] Settle the open question: does the BGE query instruction actually help?
      **Yes** — measured +0.0698 nDCG@10 on the real golden set
      (`eval/sweep_query_instruction.py`). See "Day 4's two open questions" above.
- [x] Settle the `retrieve_top_k=50` question held over from the Day 3 hardening pass.
      **Dropped to 10** — the fair, judgment-depth-matched comparison measured a
      negative quality lift from reranking at every depth tested, so the deep
      (and much more expensive) pool bought nothing. Two real bugs caught and fixed
      along the way (an env-file override that silently no-opped the first attempt,
      and a `request.limit`-truncation bug in `WeaviateHybridRetriever.search`). See
      "Day 4's two open questions" above and `app/config.py`'s `retrieve_top_k`
      docstring for full reasoning.

### Day 5 — The chat assistant ✅ DONE, live-verified on Bedrock

**Provider rolled back from Google Gemini to Anthropic Claude via AWS Bedrock,
2026-08-21 - the same day as the Gemini switch.** Two pivots in one session, both
at the user's explicit direction, both dated rather than silently overwritten:
Anthropic (original plan) → Gemini (a Gemini key was on hand) → Anthropic **via
Bedrock** (Gemini's prepayment quota ran out mid-build, plus a live chat request had
hung for minutes with no clean error; the user has Bedrock access and asked to move
there). See "Decisions already locked" below for the dated note, and "LangGraph
scope" for the current, Bedrock-native rationale.

**Orchestrated with LangGraph** — see "LangGraph scope" below for exactly where it may
and may not be used.

**Rewritten for Bedrock and live-verified the same session (2026-08-21).** A real
`AWS_BEARER_TOKEN_BEDROCK` landed, and all 3 live integration tests in
`tests/integration/test_chat.py` passed on the **first real attempt** - not because
nothing went wrong, but because a cheap, minimal, non-graph smoke test (three plain
`messages.create` calls, no LangGraph, no tool use, pennies not dollars) caught a
real config bug *before* spending a full graph run on it. `make lint`/`make
typecheck`/full `make test-all`: **412 passed, 0 skipped.**

Concrete design (`app/llm/`), same shape as before the rollback - the LangGraph
state machine, model router, semantic cache, and citation checker are all
provider-agnostic and did not need to change: a LangGraph agent with a
`search_catalog` tool bound directly to the existing `get_retriever().search()` (no
new retrieval code), a semantic-cache short-circuit in front of the LLM call, a
lexical model router (`anthropic.claude-haiku-4-5` default, escalate to
`anthropic.claude-sonnet-5` on comparison-language/multi-constraint messages or a
second unresolved tool-call round), SSE streaming with distinct
`tool_call`/`token`/`citations` event types, and a post-hoc uncited-SKU check that
feeds a metric (not a stream-blocking gate — the answer is already streaming by the
time the check runs).

- [x] LangGraph state graph for the conversation (nodes, tool calls, multi-turn state
      via LangGraph's own checkpointer, keyed by `conversation_id` — in-memory for
      this single-process demo, documented as the scale limit) — `app/llm/graph.py`.
      Live-confirmed: a real tool-call round trip through `search_catalog` and back
      completed correctly.
- [x] Claude calls made with the **raw `anthropic` SDK inside the nodes** (via its
      Bedrock client, `AsyncAnthropicBedrockMantle`), not a LangChain/LangGraph model
      wrapper — `app/llm/client.py`, `app/llm/graph.py`. Live-confirmed against real
      Bedrock endpoints for both models.
- [x] Grounded answers with SKU citations (prompt-level constraint in
      `app/llm/prompting.py` + post-hoc detection in `app/llm/citations.py`) -
      live-confirmed citing the correct seeded SKU (`BOOT-WP-10`) for a real query
      against `demo-fashion-in`.
- [x] Model router (cheap model for lookups, strong model for hard questions,
      `app/llm/model_router.py`, unit tested, provider-agnostic - unchanged)
- [x] Streaming responses + a minimal chat page (`app/routers/chat.py` SSE endpoint,
      `static/index.html` - vanilla JS, no React; unchanged from the Gemini build
      except the event *payloads* it renders, which are provider-agnostic anyway)
- [x] Semantic caching to cut cost and latency (`app/llm/semantic_cache.py`, Redis,
      reuses the already-declared `semantic_cache_enabled`/`semantic_cache_threshold`
      settings; provider-agnostic, unchanged from the Gemini build) - live-confirmed
      a repeated query returns `cached: true` with no second Bedrock call.

**Bedrock facts confirmed against the reference user guide the user supplied
(`bedrock-ug.pdf`), not guessed - the same discipline the Gemini build's mistakes
argued for**:
- Model IDs are Bedrock's bare, Messages-API-style names -
  `anthropic.claude-sonnet-5` and `anthropic.claude-haiku-4-5` - not the fully dated
  `-v1:0` modelId strings the boto3 Invoke/Converse APIs need (unused here). Both
  confirmed live-reachable.
- The Messages API is exposed via `anthropic[bedrock]`'s
  `AsyncAnthropicBedrockMantle` client, auth'd with a Bedrock long-term API key
  (`AWS_BEARER_TOKEN_BEDROCK`) - not full AWS IAM access/secret keys, and a
  genuinely different, simpler credential than either the Anthropic or Gemini plans
  needed. Live-confirmed.
- Prompt caching needs a **4,096-token minimum per cache checkpoint** to actually
  cache anything - confirmed directly from the guide, not assumed. This project's
  system prompt (a few hundred tokens) is nowhere near that floor on its own; the
  `cache_control` marker is still attached (correctly positions the prefix for when
  real multi-turn history crosses the threshold) but a cache hit is not claimed for
  a short single-turn prefix, because that hasn't been measured and would violate
  this project's own "measured, not guessed" rule.
- Two timeout layers (a constructor-level `timeout=` plus a per-chunk
  `asyncio.wait_for` in `app/llm/graph.py`) were built in from the start this time,
  carrying forward the Gemini-era hang lesson rather than waiting to rediscover it.
  No real Bedrock stall has actually occurred yet to exercise this for real, so the
  per-chunk enforcement is still inherited caution, not something a real hang has
  confirmed on this provider - genuinely different from the finding below, which
  *was* confirmed live.

**One real bug caught live, and caught cheaply** - a direct sequel to the
"adaptive thinking" line in the guide that turned out to only be true for one of
the two models: a minimal, non-graph smoke test (plain `messages.create`, `max_
tokens=50`, no tool use - the cheapest possible real call) tried `thinking:
{"type": "adaptive"}` + `output_config.effort` on both models before touching the
full agent. Sonnet 5 accepted both exactly as the guide said. **Haiku 4.5 rejected
both outright**: `"adaptive thinking is not supported on this model"` and
`"This model does not support the effort parameter"` - two separate 400s, not one.
Further live checks (still outside the graph, still cheap) found Haiku 4.5 only
accepts the older `thinking: {"type": "enabled", "budget_tokens": N}` shape (which
also requires `max_tokens > budget_tokens`, a real constraint worth remembering),
or no `thinking`/`output_config` at all. Given Haiku is specifically the
fast/cheap-lookup tier, forcing it into the older explicit-budget reasoning mode
would spend real thinking-token cost for exactly the queries meant to avoid that -
so `app/llm/graph.py` now sends `thinking`/`output_config.effort` **only** for the
Sonnet-5/reasoning tier, and neither for Haiku. Confirmed correct immediately after
by running the full `test_chat.py` suite for real - all 3 passed on the first try
with the fix already in place, no repeated failed attempts this time, unlike the
Gemini build's several rounds of live debugging for its own real bugs.

**Why the cheap-check-first approach mattered**: this bug would have failed inside
the full LangGraph agent too (same 400), but finding it via three tiny,
non-streaming, no-tool-use calls cost a fraction of a cent and one exchange, instead
of discovering it mid-way through a real multi-round tool-calling conversation the
way several of the Gemini-era bugs were found. Worth carrying forward as a general
practice: when switching LLM providers, smoke-test the raw API surface directly
before wiring it into the full agent, not after.

**`scripts/bench_chat.py` not yet run against Bedrock** - the Gemini-era numbers
below are real, but they describe a different provider's latency profile entirely
and must not be quoted as if they still apply to Bedrock. Run it when real Bedrock
latency numbers are wanted; same cost-discipline reasoning as before (~30 real paid
calls, not spent without being asked).

**Two more real bugs found live in the browser, same session (2026-08-21), after
the automated suite was already green** - a reminder that `make test-all` passing
doesn't mean the UI actually works; both were only found by testing `/ui/` in a
real browser and reading DevTools evidence rather than guessing:

1. **`static/index.html` rendered nothing at all for a successful response.**
   Root cause: `sse-starlette`'s `EventSourceResponse` defaults to `\r\n` as its SSE
   line separator (`DEFAULT_SEPARATOR = "\r\n"` in the installed package), so every
   event actually ends with `\r\n\r\n` on the wire - but the client's hand-rolled SSE
   parser was splitting the buffered stream on a literal `"\n\n"`, which never
   occurs. The parsing loop's `indexOf` stayed `-1` forever, so no event was ever
   parsed, no exception was thrown, and nothing rendered - server-side curl and
   Network-tab evidence both looked perfectly correct, which is what made this one
   deceptive. Fixed by normalizing `\r\n` → `\n` on each decoded chunk before
   splitting (`static/index.html`).
2. **A second turn in the same conversation could permanently corrupt that
   conversation's history with a 400 from Anthropic** (`tool_use ids were found
   without tool_result blocks immediately after`). Root cause: Claude will emit
   *two* `tool_use` blocks in one turn for a compound ask (e.g. "anything for women
   in red?" right after an unrelated shirt query), but `app/llm/graph.py`'s
   `agent()`/`tool_node()` only ever read and answered `tool_use_blocks[0]` - every
   other piece of state (`tool_call_rounds`, `citations` as a single result set)
   already assumed one call per round. The second, unanswered `tool_use` id got
   persisted into the LangGraph checkpointer's message history and poisoned every
   later turn on that `conversation_id`, since the corrupted history gets resent
   to Anthropic on every subsequent call. Fixed by setting
   `tool_choice: {"type": "auto", "disable_parallel_tool_use": True}` on the
   Bedrock call (confirmed as a real, current parameter directly from the installed
   `anthropic` SDK's `ToolChoiceAutoParam` type, not guessed) - caps Claude to one
   tool call per round, matching what the rest of the graph actually handles.

---

#### Gemini era (2026-08-21, superseded same day) — kept as history, not current

Everything in this subsection describes the Gemini build that preceded the Bedrock
rollback above. The code it describes (`google-genai`, `gemini-3.1-pro-preview`,
`gemini-3.7-flash`, `GenerateContentConfig`, etc.) **no longer exists in this
repo** - kept here only because the bugs found and lessons learned were real, and
several of them (the timeout/hang fix, the round-cap `force_no_tools` fix, the
citation false-positive) were carried forward into the Bedrock rebuild rather than
re-discovered. Do not use any model ID, SDK class, or code reference below as if it
still applies to this codebase.

**Model IDs corrected after a real key confirmed reality**: `gemini-3-pro` (this
build's first guess) turned out **not to exist** as a text model at all - only
`gemini-3-pro-image-preview`/`-image` do. `client.models.list()` against the live
API confirmed `gemini-3.1-pro-preview` and `gemini-3.7-flash` are the real,
reachable IDs; both smoke-tested with a real `generate_content` call before being
locked in. One real, measured cost worth remembering: even at `thinking_level="low"`,
`gemini-3.1-pro-preview` spent ~94 "thinking" tokens answering a two-word question -
a small `max_output_tokens` can silently starve the actual answer of any budget.

**Six real bugs caught, all worth remembering:**
- **A genuine hang, found by just leaving the chat page open**: a real `/chat`
  request sat with an established, healthy-looking TCP connection to Gemini,
  `200 OK` already received on the streaming response, and then produced *nothing*
  - no chunk, no error - for over two minutes. `google-genai`'s `Client` has no
  request timeout configured by default, so there was nothing to bound it: the SSE
  connection to the browser would have hung forever. Root cause on Gemini's side
  couldn't be confirmed from this end (the TCP connection stayed open the whole
  time, so it wasn't a network drop) - what's fixed is CatalogMind's side of the
  contract: `app/llm/client.py` now sets `GEMINI_TIMEOUT_MS = 60_000` via
  `HttpOptions`, generous relative to `bench_chat.py`'s measured p99 (28.1s for a
  *whole* turn), so a stuck call fails as a retryable `httpx.TimeoutException`
  instead of hanging indefinitely. Caught a second issue while fixing this one:
  `with_retry` was wrapping the *entire* streamed call, including any tokens
  already written to the live SSE stream - retrying after a mid-stream failure
  would have re-run the whole call and duplicated/garbled output the client
  already received. Fixed by tracking whether any token was written and, if so,
  converting a transient failure into a hard (non-retried) error instead -
  matching this module's own documented tradeoff ("once streaming has started, a
  failure is reported, not silently retried") that the code hadn't actually
  enforced until now. Also dropped retry attempts for this call from the default
  3 to 2, so one bad call costs at most two timeouts, not three.
- **A false alarm while debugging the above, instructive on its own**: replaying
  the exact failing query first looked broken twice in a row (reproducing the
  earlier round-cap bug's symptom exactly - 3 tool-call events, 0 token events) -
  but this wasn't the round-cap bug resurfacing, it was `uvicorn --reload` not
  having finished restarting yet before the retry curl fired, so those two
  attempts hit stale pre-fix code. A third attempt, after actually confirming the
  reload had settled, showed the fix working correctly. Worth remembering:
  "reproduced the same symptom" doesn't always mean "same bug" - check what code
  was actually running before concluding a fix failed.
- **The one only actually clicking around the `/ui` page could catch**: asked
  "something to wear in christmas" (an exploratory query) through the real chat
  page and got back a user bubble with nothing after it - no answer, no error, no
  citations. Root cause: `MAX_TOOL_CALL_ROUNDS` capped how many times `tool_node`
  would run, but nothing stopped the *agent* call right at the cap from requesting
  yet another tool call anyway - `search_catalog` was still in its `tools` list.
  `_route_after_agent` then had nowhere to send that request, `final_answer` was
  never set, and the client had a real, `200 OK`, fully-formed SSE stream with
  nothing in it to render. Every unit test used scripted single-round fakes and
  every existing integration test happened to resolve in one round, so nothing
  before this had ever driven the loop to its actual boundary. Fixed by adding
  `ChatState.force_no_tools`: once `tool_call_rounds` hits the cap, `tool_node`
  still answers the *pending* call (the Gemini API requires that), but the next
  `agent()` call omits `tools` from its `GenerateContentConfig` entirely - not a
  prompt asking it to stop, an actual removal of the capability - which guarantees
  a real text answer within one more turn. **Verifying this took two false starts,
  both instructive**: replaying the exact failing query first returned a full,
  well-formed answer - but only because it hit the semantic cache from a prior run,
  which never exercises the fixed code path at all. The next two fresh (non-cached)
  attempts reproduced the *original* failure exactly (3 tool-call events, 0 token
  events) - not because the fix was wrong, but because `uvicorn --reload` hadn't
  actually finished restarting before those curls fired, so they hit stale code.
  Confirmed for real only after adding temporary debug logging of
  `state["tool_call_rounds"]`/`force_no_tools`/`config.tools` and re-running once
  the reload had settled: round 2 correctly showed `force_no_tools=True`,
  `config.tools=None`, zero function calls, and a real streamed answer. Removed the
  debug logging once confirmed; a plain fresh query afterward resolved normally in
  2 rounds with 16 real token events - the fix holds.
- **A second bug the same session surfaced**, in `static/index.html` this time: the
  streamed-token handler did `answerEl.textContent += payload.text`, and setting
  `textContent` replaces *all* of an element's children, not just its text -  so on
  a semantic-cache hit (where the `citations` event, and its product cards, arrive
  *before* any `token` event) the very first token appended would silently delete
  the product cards that had just been rendered. Fixed by giving the assistant
  message a dedicated child `<span>` for streamed text, so appending to it can
  never touch sibling elements like the product-card container.
- **The one only a live call could catch**: Gemini 3 attaches a `thought_signature`
  to a function-call `Part` and rejects (400) a follow-up turn if that signature
  wasn't carried forward unchanged. `agent()`'s tool-call handling had been
  rebuilding a fresh `Part` from the extracted `name`/`args` via
  `Part.from_function_call(...)`, which silently drops it - passed every unit test
  (which never exercises a real multi-turn Gemini call) and failed the instant a
  live integration test actually looped through a tool call. Fixed by keeping and
  reusing the *original* `Part` object verbatim (`_AgentTurn.function_call_parts`,
  `app/llm/graph.py`) instead of reconstructing one. This is the concrete case for
  why the live integration tests exist at all, not just the unit-level fakes.
- A citation false-positive: an earlier version of `citations.py` flagged ordinary
  measurement words like "500ml"/"4k" as hallucinated SKUs, the same class of bug
  `alpha_router.py`'s `_MEASUREMENT` regex exists to prevent on the
  query-classification side - fixed by requiring a hyphen/underscore separator for
  the mixed letters+digits case, caught by this module's own unit tests before it
  ever reached a live call.
- `google-genai`'s streaming response is a pydantic model that doesn't support
  attaching arbitrary attributes - an early version tried to stash accumulated
  text/function-calls back onto the last response chunk and would have failed at
  runtime the first time it actually ran. Fixed by assembling the turn's result into
  a small local dataclass (`_AgentTurn`) instead of mutating the SDK's response
  object.

**`scripts/bench_chat.py` run for real (2026-08-21, user explicitly asked for the
spend)** - ~30 real paid Gemini calls, same rigor as `bench_search.py`. Real numbers:

- **Cold turn (real Gemini call, tool call + answer): p50=10.6s, p95/p99=28.1s,
  mean=11.6s.** Slow, and worth being honest about - this is a multi-hop round trip
  (search_catalog tool call, then a second generation) on top of `gemini-3.7-flash`'s
  own latency, not a single fast call.
- **Cache hit: p50=29.3ms, p95/p99=36.5ms** - roughly **360-700x faster** than a
  cold turn. The semantic cache isn't a marginal optimisation here, it's the
  difference between an 11-second wait and an instant response for any repeated
  question.
- **Concurrency - the open question this script existed to answer, settled**:
  aggregate throughput *increased* with concurrency (0.12 → 0.25 → 0.42 req/sec at
  concurrency 1/2/4), unlike the reranker's measured pathological collapse in the
  Day 3 hardening pass (0.19 → 0.14 → 0.09 → 0.04 req/sec at the same concurrency
  levels). Per-call p95/p99 latency does grow under load (up to ~16-19s at
  concurrency=4), but the earlier CPU-bound-contention finding does **not**
  generalise to a network-bound external API call - a real, measured result, not an
  assumption carried over from the reranker's very different failure mode.

*(End of the Gemini-era history block. Everything above this point in Day 5 is
about a provider this codebase no longer uses.)*

### Day 6 — Automation 🚧 IN PROGRESS

- [x] **Generation quality eval: groundedness, hallucination rate** (2026-08-21).
      Built small first (12 scenarios), got a clean 1.0/1.0/1.0 pass, then
      **scaled to full parity with retrieval's 170-query golden set** the same
      session - a 12-scenario "well established" claim wasn't credible on its
      own, and scaling up is what actually surfaced the real bugs below.
      `eval/golden_chat/` - **218 total scenarios**: 170 grounded (every
      positively-judged query from `eval/golden/`, auto-derived via
      `_grounded_scenarios()` so there's no separate hand-labelling effort -
      reusing already-verified retrieval anchors to test a genuinely different
      question, whether the *generated answer* names the right SKU, not
      whether it was retrieved) + 48 hand-picked refusal probes (~16/tenant,
      built by pointing a real anchor query from *another* tenant's golden set
      at this one - cross-tenant absence as a mechanical ground truth).
      `eval/generation_metrics.py` - pure, unit-tested scoring: groundedness
      reuses `app.llm.citations.find_hallucinated_citations` directly (the same
      function production runs on every real turn, not a reimplementation), a
      price-mismatch heuristic scored only when unambiguous (matching
      `eval/metrics.py`'s "undefined isn't zero" rule), and a lexical
      refusal-cue detector. `eval/generation_eval.py` runs every scenario
      through the real, compiled `get_chat_graph()` - real Bedrock calls, real
      retrieval underneath - with the semantic cache force-disabled for the run.
      **Final measured result at n=218: groundedness 0.982, answer-hit-rate
      0.941, refusal-correctness 0.958** (`eval/results/generation_eval.json`),
      up from 0.784/0.941/0.750 on the first full-scale run - only 15 of 218
      scenarios left unresolved after the fixes below, mostly further instances
      of the same paraphrase pattern already understood and two genuinely
      ambiguous scenario constructions, not new unknowns.
      **Real bugs found, none hypothetical - the full run earned its cost**:
      1. **A genuine production bug in the chat graph**: `ChatState.
         force_no_tools` correctly stopped the model from *requesting* another
         search once `MAX_TOOL_CALL_ROUNDS` was spent, but said nothing about
         what to do instead - so the model would sometimes narrate a search it
         could no longer run as its entire final answer. Fixed with
         `app/llm/prompting.py::FORCE_ANSWER_NUDGE`, an extra uncached
         system-prompt block appended only on that turn.
      2. **A second genuine production bug**: `tool_node` *overwrote*
         `state["citations"]` on every search round instead of accumulating -
         so a final answer that legitimately cited a product found in an
         *earlier* round (still visible in the model's own message history) got
         flagged as a fabricated citation by `validate_and_store`'s own check,
         which feeds the live `catalogmind_chat_hallucinated_citations_total`
         metric. Fixed by merging across rounds, deduped by SKU
         (`app/llm/graph.py::tool_node`).
      3. **The dominant false-positive class, found at scale**: the original
         hallucination check only ever compared a flagged token against
         retrieved products' `sku` field - but real e-commerce titles routinely
         carry their *own* model number, EAN barcode, or spec text distinct
         from the internal SKU ("Sony **WH-1000XM5**", "Ayesha Women Aviator
         Sunglasses **8903705152451**"), and the model was correctly, faithfully
         quoting real retrieved data every single time this was checked by
         hand. Fixed generally in `app/llm/citations.py`: check retrieved
         products' title/brand text too, not just `sku`.
      4. **A narrower, paraphrase-shaped remainder of the same class**: a model
         sometimes *reworded* a real spec instead of quoting it verbatim
         ("30Hr Battery" -> "30-hour", a title's "1/2/3/4 Seater" -> "1-4", two
         real prices "$16.40"/"$17.10" -> "16-17"). The substring check alone
         can't catch a reworded number, so `_ORDINARY_HYPHENATED_IDIOM` was
         widened twice: first from just "N-in-1" to any digit-hyphen-*word*
         shape ("2-Pack", "3-year"), then to digit-hyphen-*digit* ranges
         ("1-4", "16-17", "20-50cm") once the fuller run surfaced that variant
         too. Verified safe against swallowing a real SKU: none of this
         project's actual SKU shapes (bare digits, a bare ASIN, `shein-<hex>`)
         start with digits-then-hyphen-then-a-bare-word-or-number.
      5. **A distinct fifth case**: a model that answered an identifier lookup
         *without calling the search tool* (asking "want me to search for
         product 10071599?" instead) got flagged for echoing back a SKU the
         **shopper themselves** typed - not a fabrication, the customer's own
         words. Fixed by also exempting any token present verbatim in the
         user's own message (`find_hallucinated_citations`'s new optional
         `user_message` parameter).
      6. **The refusal-cue heuristic needed widening twice**, each time on a
         real hedged phrasing the live run produced and the phrase list didn't
         yet cover ("we don't *appear to* carry X", "I *wasn't able to find*
         any X... I *don't want to* recommend these"). Explicitly **not**
         claimed as a principled fix - it is, and remains, a lexical cue list
         that will keep missing novel phrasings; a genuinely principled
         solution would be a small LLM-judge classification call, which was
         discussed but not built this session (cost/determinism trade-off, and
         it would need its own calibration against a hand-labelled sample to
         avoid trading one unvalidated approximation for another).
      7. **Two refusal scenarios turned out to have a wrong ground truth, not a
         model or detector bug**: the Myntra fashion catalog genuinely carries
         some home-textile items (coasters, duvet covers, bath mats) alongside
         clothing, so three fashion-refusal probes anchored on those categories
         found a real match - correct model behaviour against a false "this
         is absent" assumption. Swapped for electronics anchors (zero domain
         leakage observed there). A fourth ("kids party wear for a birthday")
         was genuinely ambiguous (this catalog has party *accessories*, just no
         clothing) rather than a clean absence: swapped for footwear, confirmed
         absent by Day 2's own data-quality report.
      Every fix re-verified: `make lint`/`make typecheck` clean, full
      `pytest tests/` (unit + integration, real stack + real Bedrock) at
      **445 passed**. The last two fixes (#4's digit-range widening and #5's
      user-message exemption) are unit-tested but were **not** re-verified with
      another full 218-scenario paid run - a deliberate cost/time call, not an
      oversight; the 0.982/0.941/0.958 numbers above predate them slightly.
      **Deliberately not attempted**: a general hallucinated-*attribute* rate
      (color/material/size claims, not just price) - needs either an LLM judge
      or much larger structured ground truth, neither built. Price stayed
      unscored on every run so far (no answer happened to state a price in the
      one-product/one-price shape the check requires) - not a claim the check
      doesn't work, just that it hasn't fired yet.
      **A live process mistake worth remembering, not just a finding**: the
      first full 218-scenario attempt was killed mid-run after ~15 minutes,
      misdiagnosed as hung from a stale-looking log - the redirected stdout was
      block-buffered (not line-buffered, since it wasn't a tty), so `tail`
      showed a frozen line while the process had actually progressed cleanly
      through ~119 scenarios. That run's real Bedrock spend was wasted (results
      only get written at the very end). Fixed by rerunning with `python -u`
      for every subsequent long eval run - unbuffered output is now the
      default for anything background-monitored this way.
- [x] **Agentic RAG expansion: catalog-wide stats + exact lookup tools** (2026-08-22).
      **Not on the original 7-day plan** - found by manually using the shipped
      `/ui/` chat page after Day 6's generation eval was already green, the same
      "make test-all passing doesn't mean the product actually works" lesson
      Day 5 already learned once. Two distinct real bugs, one capability gap,
      fixed properly because they're real correctness/reliability problems, not
      scope creep for its own sake.
      **Bug 1 - a real Bedrock 400 on turn 2+ of a real conversation**: asked a
      follow-up question in the same chat, got `Error: Error code: 400 -
      messages.1.content.0.tool_use.toolset_name: Extra inputs are not
      permitted`. Root cause, confirmed with a direct `model_dump(exclude_unset=
      True)` check, not guessed: the installed `anthropic` SDK's `ToolUseBlock`
      *response* type carries newer fields (`toolset_name`, `caller`) that get
      marked "set" during response parsing (the server's JSON explicitly
      includes them, even as `null`), so echoing that exact object back as
      *request* input on the next turn resent them - Bedrock's backend rejects
      both as unrecognized. **First fix attempt was incomplete**: sanitizing
      only the message being stored that turn didn't survive LangGraph's
      checkpoint serialize/deserialize round trip (confirmed live via
      `graph.aget_state()` - a cleanly-built object came back with all six
      fields marked "set" after one round trip). Real fix had to live at the
      actual failure boundary: `app/llm/graph.py::_sanitize_messages_for_request`
      now rebuilds every `tool_use` block fresh, right before *every* outgoing
      API call, not once at storage time. Verified live through 3 real
      sequential turns in the same conversation after the fix.
      **Bug 2 - the flagship one**: asked "what's the highest priced item for
      men" and "is there anything above ₹10,000", the agent confidently said no
      (claimed nothing above ₹2,499) - real answer (verified directly against
      Weaviate): a ₹58,854 MOVADO watch, and 94 products exceed ₹10,000. Root
      cause: `search_catalog` was the agent's only tool, returning 5
      relevance-ranked results per call - structurally incapable of answering a
      superlative/threshold/count question, yet the model answered with false
      confidence instead of recognizing the limit.
      **Before building a fix, walked every plausible shopper question shape
      against the proposed design** (a "have you thought of every kind of
      query" check, not just patching the one bug found) - full table in the
      approved plan (`/home/akash/.claude/plans/snoopy-hopping-axolotl.md`).
      Found two more real gaps this way, not by inspection: (a) a superlative
      scoped by free text ("cheapest waterproof jacket") isn't answerable by a
      structured-filter aggregate at all - `search_catalog` gained an optional
      `sort_by` (fetches a wider relevance pool, sorts it for real, rather than
      guessing from 5); (b) `rating`/`review_count` are real indexed properties
      the original price-only stats design ignored - generalized to a
      `metric` parameter, free once the mechanism exists at all.
      **A third real finding, caught live while testing the fix, not assumed**:
      the worked "resolve the flagship bug via `get_catalog_stats(category=
      "Men")`" design in the plan was itself wrong against real data -
      `demo-fashion-in`'s `category_path` is **empty for every product**;
      gender lives in unindexed `attributes_json` (`{"gender":"Men"}`), never a
      filterable property. "For men" is a free-text-scoped question on this
      catalog, not a structured one - confirmed by testing `stats()` live and
      getting a wrongly-confident `count=0` before catching it. `search_sorted`'s
      pool bumped 30→50 in response, **and, confirmed with the user, the model
      is instructed to hedge a `search_sorted`-backed superlative claim
      explicitly** ("best among candidates found," not `get_catalog_stats`-level
      certainty) rather than just relying on a bigger pool to make the same
      failure rarer instead of impossible.
      **What shipped** (full detail and exact signatures in the approved plan
      file above): `app/retrieval/hybrid.py` gained `stats()`, `get_by_skus()`,
      `search_sorted()`; `app/retrieval/base.py` gained `CatalogStats`,
      `SearchHit.rating`, and bounded `SearchFilters.brands`/`.categories`
      (unbounded lists became reachable via LLM-constructed tool calls, not
      just a human-crafted HTTP body). `app/llm/prompting.py`'s tool surface
      grew from one tool to three (`search_catalog` now takes real
      price/brand/category/stock/sort_by args instead of query-only;
      new `get_catalog_stats`; new `get_product_detail`, batched up to 5 SKUs).
      `app/llm/graph.py::tool_node` rewritten as a name-keyed dispatch table -
      tenant is still hardcoded from `state["tenant"]` in every handler, never
      from the model's own tool-call JSON, preserving the existing
      tenant-isolation pattern exactly (a "generic dispatch" refactor is
      exactly the kind of change that could have quietly reintroduced a
      caller/model-controlled tenant filter). `ChatState.citations` became a
      mixed ledger (`"kind": "product"` / `"kind": "stats"` dict entries, never
      a real class instance stored in state - deliberately, to avoid repeating
      Bug 1's exact checkpoint-serde problem) - the outward SSE event and the
      semantic cache stay product-kind-only, so neither's wire/storage format
      changed. New `app/llm/claims.py` (`find_stat_claim_mismatch`,
      `find_unverified_quantitative_refusal` - a "nothing matches" refusal
      about a quantifiable condition is only trusted if backed by a confirmed
      `count=0`, `has_superlative_language`) and `app/llm/price_text.py` (the
      price-mention regex extracted out of `eval/generation_metrics.py` so
      production and eval share one implementation instead of two that could
      drift). `app/obs/metrics.py`'s hallucination-only counter generalized to
      `catalogmind_chat_claim_mismatches_total{claim_type}` (hallucinated
      citation / stat mismatch / unverified quantitative refusal / superlative
      without a stats call - the last one confirmed with the user as
      observability-only, not a forced retry, matching
      `find_hallucinated_citations`'s existing "detection, not prevention"
      stance) plus a new `catalogmind_chat_tool_calls_total{tool}` leading
      indicator. `static/index.html`'s status line made `payload.tool`-aware
      (the `tool_call` SSE event's payload changed from a bare `query` string
      to a structured `input` dict, since two of the three tools have no
      query). Two new integration tests extend
      `tests/integration/test_tenant_isolation.py` to the new aggregate/exact-
      lookup read paths specifically (neither goes through `.query.hybrid()`/
      `.query.bm25()`, so the existing search-isolation tests didn't cover
      them). New `tests/unit/test_tool_node.py` - a fully mocked (`get_retriever()`
      patched, no live stack) dispatch test, the one gap analysis flagged as
      highest-leverage: going from one tool to three sharing one dispatch
      function is exactly the shape of change this session found breaking only
      at real-call scale, repeatedly.
      **Live-verified end to end after the fix** (real Bedrock, real Weaviate,
      not just unit tests passing): "is there anything above ₹10,000?" now
      correctly answers "Yes, 94 items... from ₹10,125 to ₹63,090, average
      ₹18,171" - matching the real Weaviate aggregate exactly. "Highest priced
      item for men" now gets a `get_catalog_stats` call, a confirmed `count=0`
      (correct, given the schema reality above), and an honest request for
      clarification instead of a fabricated number - more conservative than
      the plan's anticipated `search_sorted` fallback, but correct and safe.
      **483 tests passing** (432 unit + 51 integration, real stack + real
      Bedrock), `make lint`/`make typecheck` clean throughout.
      **Still open from this piece of work** (P6/P7 in the approved plan, not
      done this session): a new `"aggregate"` `eval/golden_chat` scenario kind
      with ground truth computed live from Weaviate (never hand-typed, per this
      project's own rule) to give this new capability the same eval coverage
      grounded/refusal scenarios already have; and a `MAX_TOOL_CALL_ROUNDS`
      re-measurement gated on that eval existing, not raised on a guess.
- [x] **Image pipeline fix: three separate drops, not one** (2026-08-22). Found by
      the user asking why product images never showed up in the chat UI despite two
      of the three demo feeds carrying real image URLs. Traced end to end (model →
      adapter → Weaviate → search API → chat tool result → citations → frontend)
      before touching anything, rather than guessing at a single cause:
      1. **Myntra's `image_url` was never mapped at all.** Its `images` column packs
         every shot for a listing into one `"~"`-delimited string
         (`"url1.jpg~url2.jpg~url3.jpg"`) - a declarative `ColumnMapping` can only
         name a source column, not reshape its value, so this had been left
         deliberately unmapped since Day 2 (see `scripts/seed.py`'s old comment,
         removed now). Fixed by escalating Myntra to a real `MyntraFashionAdapter`
         (`app/ingestion/adapters/demo_catalogs.py`), matching the same
         mapping-vs-adapter escalation pattern electronics and home-goods already
         needed - all three demo feeds now use adapters, so the demo-only
         "config-only onboarding" scaffolding in `scripts/seed.py`
         (`DemoCatalog.column_mapping`, `_mapping_dict`) was dead code and removed
         rather than left unused. Verified against all 12,491 Myntra rows that the
         first `~`-separated segment is always a non-empty, valid `http(s)` URL
         before trusting "take the first" as the rule (`tests/unit/test_demo_catalogs.py`).
      2. **Every chat citation stripped `image_url`, for every catalog, including
         Electronics where the field was already correct all the way through
         Weaviate.** `app/llm/prompting.py::_citable` is deliberately minimal - its
         own comment says every extra field is real token cost paid on every
         `search_catalog`/`get_product_detail` call - and that dict was reused
         unchanged for both what Claude sees in the tool_result *and* what the
         frontend receives as `citations`. Simply adding `image_url` to it would
         have paid real per-call token cost for a field the model never reasons
         about or cites. Fixed by splitting the two: `_citable` stays exactly as it
         was (model-facing, no image), a new `hits_to_evidence()` builds a richer
         superset (`_evidence`, adds `image_url`) straight from the same
         `SearchHit`s for the browser-facing `citations` event - so Claude's
         tool_result and the shopper's product cards can never drift out of sync on
         sku/title/etc, but only the latter ever carries the URL. Regression-tested
         directly (`tests/unit/test_tool_node.py::test_citations_carry_image_url_but_tool_result_does_not`
         asserts the tool_result text literally never contains the image URL) and
         live end-to-end (`tests/integration/test_chat.py`'s golden-path test now
         asserts a real seeded image survives ingest → search → citation).
      3. **The frontend never rendered an `<img>` element at all**, even for a
         product whose `image_url` reached it correctly. Fixed in
         `static/index.html::renderProducts`. **A pre-existing XSS gap found and
         fixed in the same edit, per CLAUDE.md's "fix it immediately" rule**: every
         product field (title, sku, price, currency - all merchant-feed-controlled,
         not app-controlled) was interpolated straight into `card.innerHTML` with no
         escaping at all, and the new `<img src="...">` would have been a second,
         worse vector (attribute-breakout, not just a stray tag) if added the same
         way. Added `escapeHtml()` for every interpolated text value and
         `isSafeImageUrl()` (requires `http(s)://`, blocking `javascript:`/`data:`)
         before ever using a value as `src`; a dead/missing image removes itself on
         `onerror` rather than showing a broken-image icon (expected for home-goods,
         which has no image column in its source feed at all - not a bug, the
         dataset genuinely doesn't have one).
      **Data re-seeded after the fix**: Myntra's `image_url` changed for every one
      of its ~12,491 products, which changes `content_hash()` for all of them, so
      `make seed`'s delta detection correctly treats every row as changed and
      re-embeds the fashion catalog (Electronics/home-goods content-hashes were
      unaffected, so their re-run is a no-op past the first pass, per the existing
      idempotency guarantee).
      **9 new/changed tests**: `tests/unit/test_demo_catalogs.py` (new - the
      adapter's `preprocess` split logic and its edge cases: multi-URL, single URL,
      missing/empty/whitespace-only `images`, ragged delimiter spacing), plus the
      tool_node split-shape regression test above. `tests/integration/conftest.py`'s
      shared `CATALOG` fixture gained a real `image` column (only `BOOT-WP-10` has
      one, deliberately, so a test can tell "this product's image survived" apart
      from "every row happens to have one") - `test_search.py` shares the fixture
      unaffected since nothing there asserted on row shape or image fields.
- [x] **Trim redundant prose for multi-result answers** (2026-08-22, same day as
      the image pipeline fix, found by the user testing `/ui/` right after it).
      Once product cards started showing images, it became obvious that a
      "search for X" answer was showing the same 5 products *twice*: once as
      Claude's own numbered prose list (title/price/stock per item, since
      nothing told it not to), once as the deterministic product-card grid built
      from `citations` (`app/llm/prompting.py::hits_to_evidence`) - not a
      double-render bug (confirmed by tracing the code: exactly one `citations`
      SSE event per turn, exactly one `renderProducts()` call), just two
      UI elements independently describing the same results. **User's explicit
      choice, asked directly rather than guessed**: trim Claude's prose, keep
      both elements (over: drop the text, drop the cards, or leave as-is).
      Added a paragraph to `_SYSTEM_TEXT` (`app/llm/prompting.py`) instructing a
      brief framing sentence plus one cited SKU for a multi-result answer,
      reserving a full per-item description for a single-product ask -
      deliberately scoped to the *listing* case only, not the single-item
      lookup path, specifically so it wouldn't touch `answer_hit_rate`'s
      literal-SKU-substring check (`eval/generation_metrics.py::score_scenario`)
      for grounded scenarios that already expect one specific SKU named.
      **Live-verified, not just unit-tested** (no test pins exact system-prompt
      text, so this needed a real run): "need a traditional saree for a wedding"
      now gets a 2-sentence answer citing one SKU instead of a 5-item list;
      "tell me more about SKU 10258247" still gets full brand/price/stock/
      category detail - confirming the trim didn't overcorrect the single-item
      path. All 3 real `test_chat.py` integration cases still pass.
      **A real, pre-existing bug surfaced by this live check, not caused by
      it**: the same "wedding saree" run cited `SKU: 10258487` in prose - a
      digit-transposed near-miss of the real, actually-retrieved
      `10238487` (Varkala Silk Sarees Magenta Banarasi Saree). Confirmed this is
      exactly what `find_hallucinated_citations` already exists to catch (it
      correctly flagged `10258487` as unmatched against the 5 real retrieved
      SKUs when replayed against the real detector) - the "detection, not
      prevention" stance from the Agentic RAG expansion work above is already
      live and did its job (the metric fires, the answer still streams). **Not
      fixed further this session** - the existing behavior is a deliberate,
      previously-confirmed-with-the-user design choice, and this is one
      observed instance of an LLM digit-transcription slip, not something this
      prompt change made more likely by any measured evidence. Worth watching
      via `catalogmind_chat_claim_mismatches_total{claim_type="hallucinated_citation"}`
      if it shows up as a real, non-trivial rate rather than reaching for a
      prevention mechanism on a single anecdote.
- [x] **Suppress product cards on a refusal answer** (2026-08-22, same day, found
      by the user live-testing "red underwear" against `demo-electronics-in`).
      Claude's prose was actually correct and well-hedged ("this catalog doesn't
      carry underwear... I can't guarantee catalog-wide certainty without
      `get_catalog_stats`") - but the product-card grid beneath it still showed
      8 unrelated `search_catalog` misses (phone cases, laptop sleeves, a
      fountain pen) as if they were the answer, visually contradicting the text
      right above them. Root cause: `_run_search_catalog`'s evidence gets
      accumulated into `state["citations"]` and displayed unconditionally,
      regardless of whether the model's own final answer judged those results
      irrelevant.
      Fixed by reusing the existing refusal-detection heuristic
      (`eval/generation_metrics.py`'s `_REFUSAL_CUE`/`refuses`, already scoring
      `kind="refusal"` eval scenarios) in production for the first time -
      extracted to a new shared `app/llm/refusal_text.py` so eval and
      `app/llm/graph.py::validate_and_store` share one implementation instead of
      a second copy that could drift, the exact same split already done for
      `app/llm/price_text.py`. When `refuses(answer)` is true, the `citations`
      SSE event and the semantic-cache write both get an empty product list -
      deliberately **only affecting what's displayed and cached**, not what the
      hallucination/stat-mismatch checks above it are scored against (those
      still see the full, unfiltered evidence, so a fabricated SKU hidden
      inside a refusal-shaped answer is still caught).
      **Known, accepted tradeoff, not silently assumed away**: this is the same
      lexical-cue heuristic already documented as imprecise - it can suppress a
      legitimately useful card if a genuinely helpful answer hedges with a
      refusal-shaped phrase while still recommending a substitute ("I don't have
      an exact match, but this similar item might work: SKU-X"). Accepted for
      now because the failure mode it fixes (visually contradicting a correct
      refusal) is worse than the rarer one it risks.
      4 new unit tests (`tests/unit/test_validate_and_store.py`, fully mocked -
      `get_stream_writer`/`aembed_query`/`cache_store`, no real Bedrock/Redis
      call) pin: refusal hides products from both the SSE event and the cache
      write, a non-refusal answer still shows/caches them, hallucination
      detection still runs on the unfiltered evidence, and a stats-backed answer
      is still never cached at all regardless of refusal.
      **Live-verified against the real bug**: a fresh, never-asked query ("green
      cotton socks size 10" against `demo-electronics-in`, chosen to avoid the
      semantic cache replaying the pre-fix "red underwear" answer) now returns
      `cached=false` with **0** products alongside a correct refusal, versus the
      real live case that started this fix showing 8 irrelevant cards.
      **Also cleared the semantic cache for all three demo tenants** (`redis-cli`
      via `app.redis_client`, deleting only the three `semcache:{tenant}` keys -
      552 entries total, nothing else in Redis touched) - the pre-fix "red
      underwear" answer was already cached with its unfiltered product list, and
      would have kept replaying it verbatim (with `cached: true`) for up to the
      cache's 24h TTL otherwise, undermining the fix for exactly the query that
      found the bug.
- [x] **Retired every free-text correctness heuristic in the chat agent, replaced
      with a structured marker protocol Claude declares explicitly** (2026-08-22,
      same day, the fifth and largest fix). Triggered by a fifth real live bug:
      "black tshirt for men" against `demo-electronics-in` got a correct refusal
      in prose ("didn't find... don't match") that the `_REFUSAL_CUE` regex from
      the previous fix - already widened twice before this session - still
      missed, showing 5 irrelevant bag/briefcase cards under a correct refusal.
      User's reaction, verbatim: regex-guessing an open-ended generator's
      phrasing is not a production-grade fix for an agentic RAG system, and asked
      for a complete audit of every such heuristic plus a real remediation plan -
      "think as an AI engineer," not another patch.
      **Full audit performed** (two parallel Explore agents, every `re.compile`
      under `app/`/`eval`/`scripts` read and categorized): heuristics parsing
      *the LLM's own generated answer* to verify a claim (`refuses()`,
      `find_hallucinated_citations`'s SKU-shaped-token regex,
      `find_stat_claim_mismatch`/`find_unverified_quantitative_refusal`'s price/
      negation regexes) are the genuinely broken category - open-ended input,
      no ground truth to measure against, structurally incomplete by
      construction. Heuristics parsing *the incoming query* for routing
      (`alpha_router.py`, `model_router.py`) are a different, lower-risk
      category - `alpha_router` already measured against the 170-query golden
      set via `eval/sweep_alpha.py`; `model_router` isn't, a real but
      lower-priority gap (cost/latency, not customer-facing correctness) held
      for later. Heuristics parsing *structured source-feed data* during
      ingestion (currency sniffing, HTML stripping, ASIN extraction) are a
      different, legitimate problem domain entirely, untouched.
      **First hypothesis tested and disproven with real data, not assumed
      correct**: proposed a per-query-class score threshold on
      `SearchHit.score` (a real signal already computed by
      `WeaviateHybridRetriever` and already discarded before
      `ChatState.citations` is built). Measured it for real - 170 golden queries'
      judged-relevant-hit scores vs. ~48 real refusal-scenario top-hit scores,
      through the live retriever, grouped by class. Result: `identifier` class
      separates cleanly (0.72–1.00 positive, no negative examples), but
      `attribute` and `exploratory` do not - a real match scored as low as
      0.0086, and two real refusal queries' top (wrong) hit scored 0.9665 and
      0.8552, squarely inside the positive range. Traced both outliers to real
      keyword coincidences: "Do you sell the Samsung Galaxy S23 5G?" (against
      `demo-home-goods`) top-scored a "10pcs **Galaxy** Print Gift Bag"; "Do you
      sell a 4K smart Android TV?" top-scored a "Universal Remote... Compatible
      With Lg **Tv** Remote" - a fused BM25+vector score cannot tell "shares a
      word" from "is the thing asked for," and no per-class cutoff separates
      these distributions in the real data. The naive threshold plan was
      abandoned, not patched, once the numbers said so.
      **What actually works, per the same measurement**: Claude's own semantic
      read of "is this a genuine match" - correct in every real case observed
      this session (bags vs. shirts, electronics vs. underwear, gift bag vs.
      phone, remote vs. TV), from title text alone, unprompted, with no score
      ever shown to it. The fix is not a better score-based guess; it's getting
      that already-correct judgment out of free prose (which any detector must
      then re-guess) and into an explicit, structural declaration.
      **Unified marker protocol shipped** (`app/llm/markers.py`, new): Claude is
      now instructed (`app/llm/prompting.py::_SYSTEM_TEXT`) to declare three
      facts in a fixed, visible-ASCII format instead of free prose - `[[NO_MATCH]]`
      as the literal first characters of an answer when nothing genuinely
      matches (replaces `refuses()` as the primary signal in
      `app/llm/graph.py::validate_and_store`); `[[SKU:X]]` wrapping every cited
      product (replaces `find_hallucinated_citations`'s whole regex history -
      the SKU-shaped-token pattern, the hyphenated-idiom carve-out, the
      title/brand substring check, the shopper's-own-words exemption, all of it
      existing only because guessing which prose substrings are a citation is
      inherently ambiguous; exact set membership needs none of that); `[[STAT:N]]`
      wrapping every `get_catalog_stats`-backed figure (replaces
      `_QUANTIFIABLE_NEGATION_CUE`'s "guess every phrasing of nothing-above-X" -
      "nothing above X"/"the max is X"/"everything is under X" all reduce to the
      same checkable `[[STAT:X]]` regardless of phrasing). `refuses()` stays wired
      in as a real backstop (a compliance miss - marker not emitted but prose
      still reads as a refusal - still gets caught), not deleted.
      **Live-tested before committing to the design, twice, not just argued
      for**: (1) 6 real Bedrock `messages.create` smoke-test calls (both
      `anthropic.claude-haiku-4-5` and `anthropic.claude-sonnet-5`, no
      LangGraph/streaming, same discipline as the Day 5 Bedrock-migration
      smoke tests) - both tiers emitted `[[NO_MATCH]]` correctly for a real
      refusal case and wrapped every real cited SKU in `[[SKU:X]]` for a real
      match case, zero hallucinated markers, and composed cleanly with the
      prose-trimming instruction already shipped earlier the same day. (2) All 7
      real failing/edge queries from this session re-run end-to-end through the
      actual `get_chat_graph()` after implementation - "red underwear," "green
      cotton socks size 10," "black tshirt for men" (the one that started this
      fix), "Samsung Galaxy S23 5G" and "4K smart Android TV" against
      `demo-home-goods` specifically (the two keyword-coincidence catalogs the
      measurement surfaced) - every one now returns `[[NO_MATCH]]`, zero
      product cards, `cached=false`. Semantic cache cleared a second time before
      this run (a handful of stale entries from mid-session diagnostic queries).
      **A real near-miss caught while writing the plan itself, not swept under
      the rug**: an early draft used a literal NUL control byte (`\x00`) as the
      example marker instead of describing one in words - caught only because
      `grep`/`Edit`'s exact-string matching silently failed against text visibly
      on screen. Fixed to a plain, visible `[[NO_MATCH]]`-style ASCII marker
      before implementation, and kept as a documented argument for that choice
      rather than quietly corrected away.
      **Frontend** (`static/index.html`): a `displayTextFor()` transform strips/
      resolves markers from the accumulated streamed text before it's shown -
      `[[NO_MATCH]]` prefix removed, `[[SKU:X]]`/`[[STAT:N]]` collapsed to their
      bare value - applied identically on live streaming and on a cache replay
      (both paths ultimately send the same `token` events). Verified by
      simulating the transform against real captured raw answers: no `[[`
      survives in any displayed text. Known, accepted minor cosmetic limitation:
      a marker split across two streamed chunks briefly shows raw bracket text
      for one render until the closing `]]` arrives - not worth a buffering
      scheme for a demo chat UI.
      **`eval/generation_metrics.py::score_scenario`** dropped its now-unused
      `message` parameter (the SKU-shaped-token exemption it fed no longer
      exists) and its refusal check now mirrors production exactly:
      `is_no_match(answer) or refuses(answer)`, not a separate reimplementation.
      **Explicitly deferred, named not silently dropped**: measuring
      `model_router.py`'s escalation heuristic against real outcomes the way
      `alpha_router` already is (a cost/latency lever, not a customer-facing
      correctness gate - lower priority); a periodic offline LLM-judge check on
      `has_superlative_language`'s accuracy. Both real gaps, both lower stakes,
      both need a paid eval run to do properly - not run without asking first,
      per this project's cost discipline.
      **9 new/rewritten test files**: `test_markers.py` (new), `test_citations.py`
      (rewritten - the whole false-positive-carveout suite retired along with
      the regex it tested, replaced by ~7 marker-based tests), `test_claims.py`
      (rewritten for `[[STAT:N]]`), `test_validate_and_store.py` (marker-primary
      + `refuses()`-backstop coverage, plus a real mocked hallucination-inside-
      a-no-match-answer case), `test_generation_metrics.py` (marker-format
      answers throughout, a new backstop-path test). **443 unit tests** (down
      slightly from 445 - the retired regex needed far more tests to pin its
      many carve-outs than the marker check needs to pin nothing being
      guessed), **494 total** with the 51 real integration tests, all passing;
      `make lint`/`make typecheck` clean; all 3 real `test_chat.py` cases still
      green.
      Full architecture writeup, the complete audit table, and the security
      review (ReDoS check on every retired/kept regex, the marker-injection
      analysis, why this introduces no new trust boundary) live in the approved
      plan file the session used: `/home/akash/.claude/plans/moonlit-riding-hummingbird.md`.
- [x] **Phase 2 (partial): measured `model_router.py`'s escalation heuristic
      against real outcomes for the first time** (2026-08-22, same day). Named
      in the marker-protocol plan as a real but lower-priority gap
      (`app/llm/model_router.py`'s docstring already admitted "not a golden
      set... a cost/latency lever, not a quality-scored classifier like
      `alpha_router.classify`") - measured properly rather than left as a
      permanent guess, "yes move ahead as planned."
      **A real finding before spending anything**: checking `classify_complexity()`
      against all 218 existing `eval/golden_chat` messages (free, no LLM call)
      found **zero** that naturally escalate to the reasoning tier at round 0 -
      every one is a short, templated single-item lookup. The golden-chat set
      alone cannot test whether escalation helps, because it never exercises the
      escalation path. New `eval/measure_model_router.py` (mirrors
      `eval/sweep_alpha.py`'s shape) uses two groups instead: 15 ordinary
      golden-chat messages the router already leaves on the fast tier
      (BASELINE), plus 5 hand-authored messages grounded in real live-pulled
      SKUs that genuinely trigger each real signal - comparison language,
      3+ constraint joins, 25+ tokens (ESCALATION_PROBES, confirmed by running
      the real, unmocked `classify_complexity()` against them before spending
      anything). New `ChatState.force_tier` (`app/llm/graph.py`) - an eval-only
      override, same "eval override, production never passes it" shape as
      `WeaviateHybridRetriever`'s `retrieve_top_k` constructor override - lets
      the same message run through the real, compiled chat graph forced to each
      tier, not a simplified re-implementation. 3 new unit tests
      (`test_model_tier_override.py`, fully mocked, no real Bedrock call) pin
      that the override actually changes the requested model and that
      production's own code path (no `force_tier` key at all) is untouched.
      **Scoped and confirmed before spending real money**, per the plan's own
      explicit commitment: presented the concrete scope (20 probes x 2 tiers =
      40 real Bedrock calls, ~$1-3 estimated) via `AskUserQuestion` before
      running - user confirmed "run it now."
      **First run produced a wrong answer, caught before being trusted**: 2/20
      probes showed the *fast* tier's `final_answer` as a **literal empty
      string**, scoring as a rules-based "the expensive tier did worse" result.
      Investigated rather than reported at face value: re-running the same two
      probes with the semantic cache explicitly disabled and cleared produced
      completely different, substantive, correct answers both times - meaning
      the empty answers were never a real generation failure to begin with, they
      were a cache-replay artifact. **The real, separate bug found**:
      `validate_and_store`'s `cache_store` call had no `semantic_cache_enabled`
      guard at all, unlike `maybe_serve_from_cache`'s read side - the
      measurement script disabled the setting for its *lookups* only (to keep
      its own runs from replaying stale results), but every scored turn,
      including that real one-off empty/broken response, still got written
      unconditionally into the **shared** semantic cache, where a real shopper's
      semantically similar query could have been served that broken answer
      later, for up to the cache's 24h TTL. Fixed by gating the write on
      `get_settings().semantic_cache_enabled` (also skips a wasted embedding
      call when caching is off entirely) - 1 new regression test
      (`test_semantic_cache_disabled_skips_the_write_too_not_just_the_read`).
      **Re-ran clean after the fix** (cache cleared across all three tenants
      first). Real result: 2/20 probes still changed outcome by tier, but now in
      the expected direction (reasoning better, not worse) - `router-probe-long-01`
      (a genuine, router-flagged long/multi-constraint question) went from an
      incomplete fast-tier answer (literally "Let me get more details on the
      Realme Smart TV Stick 4K to answer your question better:" - a narrated,
      unfinished tool-call plan, not a real answer) to a complete, correct
      reasoning-tier one - **this validates the router's existing length/
      constraint-join escalation signal working as intended**, not a gap.
      `homegoods-refusal-001` (a BASELINE message the router does *not*
      escalate) showed the same "narrated, unfinished tool-call" pattern on
      fast but a complete, correct `[[NO_MATCH]]` refusal on reasoning -
      **a genuinely more ambiguous result**: a fresh, independent third sample
      of the *same* fast-tier query (real run-to-run variance, not determinism)
      produced a different, actually-correct-but-unmarked refusal ("this catalog
      may not currently stock wireless Bluetooth earbuds") that neither
      `[[NO_MATCH]]` nor the `refuses()` backstop would catch either - a live,
      third example of the same compliance-miss class already named as an
      accepted tradeoff in the marker-protocol plan, not a new problem, and not
      chased into another regex patch for the reason that whole rework exists.
      **Honest conclusion, not oversold**: at n=20 with this much observed
      run-to-run variance, this sample is too small and too noisy to conclude
      the router should broaden its escalation criteria - one BASELINE probe
      shows a *possible* under-escalation signal, but a same-query, same-tier
      re-run contradicts it. The unambiguous, confident output of this
      measurement is the real cache-correctness bug it caught and fixed, not a
      verdict on the router. A larger, dedicated sample would be needed to say
      more, and would cost more - not run without asking first, per this
      project's cost discipline. Full raw results in
      `eval/results/model_router_measurement.json`.
- [x] **Phase 2 (complete): measured `has_superlative_language` against an
      independent LLM judge, found and fixed real confirmed gaps** (2026-08-22,
      same day, continuing "yes move ahead as planned" after the model-router
      measurement). Same free-check-first discipline: checking the heuristic
      against all 218 `eval/golden_chat` messages found it flags **zero** of
      them too (same root cause as the model-router finding - the templated
      messages never contain superlative language either), so a new, small,
      hand-authored `PROBES` list in `eval/measure_superlative_heuristic.py`
      (14 messages: 4 classic phrasings already in the word list, 6 deliberate
      paraphrases designed to test the dangerous false-negative direction, 4
      ordinary non-aggregate questions) replaces the uninformative golden-chat
      sweep, same reasoning as the router script.
      **The judge's verdict is a forced tool call** (`judge_needs_stats`), not
      parsed free text - the same "structural declaration over regex-guessing"
      principle from the marker-protocol rework, applied to the judge itself so
      its own output isn't one more thing to guess at.
      **First run's judge disagreed with the heuristic on "anything above 10000
      rupees?"** (heuristic correctly says yes; judge said no, reasoning it
      "can be answered by finding relevant matching products"). Recognised as a
      likely judge error, not a heuristic flaw, before accepting it: this is
      exactly the flagship production bug's question shape, and this project's
      own system prompt already states a search miss never proves absence -
      only a confirmed `get_catalog_stats` count=0 does. The generic judge
      prompt never carried that domain-specific asymmetry. Added it explicitly
      (`_JUDGE_CONTEXT`) and re-ran for a few cents rather than trust a result
      already suspected wrong - the disagreement resolved cleanly.
      **Clean result: 8/14 agreement, 0 false positives, 6/6 confirmed real
      misses** - every hand-designed paraphrase probe was independently
      confirmed by the corrected judge as a genuine gap: "priciest," "average
      price"/"average rating" (an actual `get_catalog_stats` metric with *no*
      cue at all before this), "total number of," a spelled-out number
      threshold ("over ten thousand rupees"), and a vague distribution question
      ("what price range do most products fall into").
      **Fixed three of the six, named the other two as accepted limitations
      rather than force them in** - `_SUPERLATIVE_CUE` (`app/llm/claims.py`)
      gained "priciest" and "average" and "total number of," all three
      unambiguous synonyms with no real false-positive risk (verified: "what do
      you *mean* by slim fit?" stays correctly unflagged - "mean" was
      deliberately *not* added as an "average" synonym for exactly this
      reason, and was never one of the measured probes to begin with). The
      spelled-out-number and vague-price-range gaps were left unfixed and
      named in a comment: the first needs a number-word parser, not a keyword,
      to catch generally; the second would trade a false negative for a new
      false positive (a looser "what's your price range" question doesn't
      always need an exact aggregate) rather than actually fixing anything.
      4 new unit tests pin the three real fixes, the "mean" false-positive
      guard, and the accepted spelled-out-number gap as documented, intentional
      behaviour. **501 tests total** (450 unit + 51 integration), lint/typecheck
      clean. Full raw judge output in
      `eval/results/superlative_heuristic_measurement.json`.
      **Phase 2 is now complete** - both real gaps named in the marker-protocol
      plan (`model_router.py`'s escalation heuristic, `has_superlative_language`)
      have been measured against real outcomes for the first time, matching the
      "measured, not guessed" discipline this project already held `alpha_router`
      to. Full plan file, including the Phase 1 marker-protocol rework this
      built on:
      `/home/akash/.claude/plans/moonlit-riding-hummingbird.md`.
- [x] **`[[NO_MATCH]]` marker leaked as raw bracket text to the customer, live-
      found via manual `/ui/` testing after Phase 1/2 shipped** (2026-08-22,
      same day). User pasted a real, extended chat transcript for review, not a
      bug report - reading it closely surfaced two separate real issues, only
      one asked to be fixed:
      1. **The frontend bug, fixed**: Claude often narrates before calling a
         tool ("I'll search for iPhone 17 phone covers in the catalog.") in the
         same visible answer bubble as its eventual real answer. `static/
         index.html`'s `displayTextFor()` only ever strips a *leading*
         `[[NO_MATCH]]` (`^\s*\[\[NO_MATCH\]\]`), but the accumulated
         `rawAnswer` buffer spans the *whole turn's* streamed tokens, narration
         included - so once the real, marker-led answer arrived after the tool
         call, the marker was no longer at position 0 of the buffer, the strip
         regex never matched, and the literal text "[[NO_MATCH]] The catalog
         doesn't have..." rendered straight into the chat bubble. Confirmed by
         the raw brackets visible verbatim in the pasted transcript for both
         "phonecover for iphone 17" and "iphoe 15." **Server-side consequence
         was unaffected** - `is_no_match()` checks `state["final_answer"]`
         (only the final round's own text), so product-card suppression
         already worked correctly in both cases; this was a pure display leak,
         not a citation/relevance bug.
         Fixed by resetting `rawAnswer = ""` whenever a `tool_call` SSE event
         fires - any text streamed before a tool call is disposable narration
         already conveyed by the status line ("🔍 Searching catalog..."), so
         discarding it means the *next* segment's own marker is genuinely
         leading again. Verified with a faithful line-by-line simulation of the
         exact same regex (no Node available in this environment, and this
         project has no JS test infra at all - `static/index.html` is
         explicitly "a demo, not the primary interface") against three real
         scenarios: the exact reported leak (now clean), an ordinary
         genuine-match answer with `[[SKU:...]]` citations (still renders
         correctly), and a synthetic two-tool-call-round case (still resolves
         to just the final segment). Confirmed the live regex in the file byte-
         for-byte matches what was simulated, not assumed.
      2. **Named but not fixed, per explicit instruction**: the same transcript
         also reconfirmed the already-documented `refuses()` backstop
         imprecision live - "samsung mobile" got a clean, correctly-reasoned
         refusal in prose but no `[[NO_MATCH]]` marker and no backstop match
         ("not Samsung mobile phones themselves" doesn't hit any cue), so 5
         accessory cards displayed under a "no" answer; the near-identical
         "realme mobile" query happened to phrase its refusal as "doesn't have
         ... themselves," which *does* match the backstop, and correctly
         suppressed its cards. Same accepted tradeoff already named in the
         marker-protocol plan (a compliance miss can still slip past the
         lexical backstop) - now observed live rather than only tested
         synthetically, not chased into another regex patch for the same
         reason that whole rework exists.
- [ ] **CI eval gate** — a PR fails automatically if search quality drops
- [x] Per-merchant cost tracking
- [x] Helm chart / k8s manifests

### Cost tracking, Helm chart, and a real conversation-isolation leak (2026-08-22)

**Found while planning, fixed first, ahead of the two requested items**: grounding
the cost-tracking design in `app/routers/chat.py` surfaced that `ChatRequest.conversation_id`
(`app/schemas.py`) is accepted verbatim from the client with no format constraint,
and LangGraph's `InMemorySaver` (`app/llm/graph.py`) keys every checkpoint by that
value alone - nothing composed tenant into the key. A merchant reusing another
merchant's `conversation_id` would resume that merchant's checkpointed message
history (`ChatState.messages`'s `operator.add` accumulation) into their own turn -
a real, unmitigated cross-tenant leak against `CLAUDE.md`'s non-negotiable
invariant #1, not a hypothetical, and not covered by `test_tenant_isolation.py`
(Weaviate/product-level isolation only, never the chat checkpointer). Fixed by
composing the checkpointer's key as `f"{merchant.tenant}:{conversation_id}"` in
`app/routers/chat.py::_stream` - invisible to the client, which still only ever
sees the bare `conversation_id`. New test in `test_chat.py`
(`test_conversation_id_is_isolated_by_tenant`) proves it directly against
LangGraph's own `aget_state` API rather than inferring it from LLM prose, reusing
the one real call the test already needed.

**Per-merchant cost tracking**: `app/llm/graph.py`'s `agent` node already read
`response.usage` for a cross-tenant Prometheus counter
(`observe_chat_tokens`/`app/obs/metrics.py`) - that module's own docstring named
the exact gap this closes: per-tenant cost needs a queryable store (Postgres), not
a metrics label. New `LlmUsage` table (`app/models/db.py`, migration
`7260b347df31`), one row per Bedrock invocation (not per chat turn - a
multi-round tool-calling turn makes several calls, each with its own `usage`).
`app/llm/pricing.py` holds $/token rates for `anthropic.claude-sonnet-5`/
`anthropic.claude-haiku-4-5`, web-search-confirmed (2026-08-22, AWS/Anthropic's
own pricing pages - no internal pricing doc exists anywhere in this repo or the
Bedrock guide) with the same "documented, dated, flagged" discipline
`alpha_router.py`'s `PRIOR_ALPHA` already uses. **Sonnet 5's $2/$10-per-million
rate is promotional and expires 2026-08-31** (then $3/$15) - if `pricing.py`
still reads $2/$10 well after that date, re-verify against AWS's Bedrock pricing
page before trusting a cost figure it produces. `app/llm/cost_tracking.py::record_llm_usage`
writes the ledger row, awaited inline (not fire-and-forget - a billing ledger's
durability requirement is stronger than a cache warm or a metrics counter, and
the write costs single-digit milliseconds next to a multi-second Bedrock call),
wrapped in `try/except` so a ledger-write failure can never turn a successful
streamed answer into a 500. New `GET /v1/merchants/{tenant}/usage`
(`app/routers/usage.py`) aggregates by model with an isolation caveat documented
in its own docstring: unlike Weaviate's structural per-tenant shards, this is
Postgres, so isolation is only as strong as the `WHERE merchant_id = ...` filter
(same as `ApiKey`/`IngestionJob` today - this codebase has no Postgres RLS despite
`app/main.py`'s API description claiming otherwise, confirmed by grep, a
pre-existing documentation/reality gap not introduced or fixed here).

Adding `ChatState.merchant_id`/`conversation_id` as required fields broke three
call sites that construct chat state by hand and invoke the compiled graph
directly, bypassing `chat.py` - a real, live failure (`KeyError: 'merchant_id'`),
not caught by `make typecheck` (which only checks `app`/`eval`, not `scripts/`,
and the two eval scripts build `turn_input` as a plain untyped dict rather than a
`ChatState`-annotated one). Fixed all three: `eval/generation_eval.py` and
`eval/measure_model_router.py` now resolve and cache each tenant's real
`Merchant.id` before invoking the graph (so their real Bedrock spend also
produces real ledger rows); `scripts/bench_chat.py` does the same once for its
fixed `TENANT`. `tests/unit/test_model_tier_override.py` (a no-stack unit test
that calls `agent()` directly with a hand-built state) got placeholder
`merchant_id`/`conversation_id` values plus a mock for `record_llm_usage` itself
- it was already mocking `observe_chat_tokens` for the identical reason, a real
Postgres write doesn't belong in a unit test any more than a real Bedrock call
does.

**A real bug in the fix's own first draft, caught by actually rendering the
Helm chart rather than trusting the YAML by inspection**: the Postgres password
needed by both the `postgres` Secret and the API's `POSTGRES_DSN` was resolved
by a shared `_helpers.tpl` template, `catalogmind.postgresPassword`, intended to
generate the password once and reuse it everywhere. It didn't - `randAlphaNum`
is not memoized across separate `{{ include }}` call sites, so the two callers
each independently generated a *different* random password, confirmed by
running `helm template` for real and diffing the two base64 secret values, not
assumed from reading the template. Fixed by caching the resolved value into
`.Values.postgres.password` on first computation (Helm's `.Values` is a mutable
map shared across every template in one render) - re-rendered and confirmed the
two secrets agree.

**Helm chart** (`infra/helm/`, using the empty scaffold that already existed):
full stack per the user's explicit choice - StatefulSets for Weaviate/Postgres/
Mongo/Redis (mirroring `infra/docker-compose.yml`'s env vars verbatim, not
re-derived) plus a Deployment/Service/ConfigMap/Secret/HPA/Ingress for the API
and a pre-install/pre-upgrade migration Job running `alembic upgrade head`.
Weaviate is fixed at 1 replica, not a values knob - its single-node raft
bootstrap (`CLUSTER_HOSTNAME=node1`/`RAFT_JOIN=node1`/`RAFT_BOOTSTRAP_EXPECT=1`)
only works for exactly one node, the same constraint Day 0 already documented for
Compose. Weaviate's readiness/liveness probes use `httpGet` against
`/v1/.well-known/ready`/`/live` rather than an `exec` probe - the image is
distroless (no shell), the same reason Compose ships with no container
healthcheck at all, but Kubernetes' `httpGet` probe doesn't need a shell, so this
is genuine in-cluster readiness Compose structurally couldn't have. `NOTES.txt`
says explicitly that hand-rolled stateful services are for local/demo
reproducibility, not a production recommendation - a real deployment would use
managed datastores. New `make helm-lint` target (`helm lint` + `helm template`,
no cluster needed) and a third, independent CI job
(`.github/workflows/ci.yml`, using `azure/setup-helm`) run it on every PR.
Verified for real: `helm` isn't installed in this environment, so a portable
binary was downloaded to the session scratchpad (not system-wide) specifically
to run `helm lint`/`helm template` against the actual chart before calling this
done - which is what caught the password bug above.

### Day 7 — Ship it

- [ ] README with the measured results and architecture diagram
- [ ] `DECISIONS.md` — tradeoffs + "what breaks at 1,000 merchants"
- [ ] Deploy a live demo
- [ ] 3-minute screen recording

---

## Things you need to do (only these)

| When | What | Status |
|---|---|---|
| Any time | Run the keepalive after each Windows login | ongoing |
| Before Day 5 (Bedrock) | Get a Bedrock long-term API key (Bedrock console → API keys), paste into `.env` as `AWS_BEARER_TOKEN_BEDROCK` | done (2026-08-21) |
| Before Day 6 | Create an empty public GitHub repo named `catalogmind` | **not done** |
| Day 2 | Kaggle account + API token, so real datasets can be downloaded | done |

Nothing above is blocking today.

---

## If it stops working

| Symptom | Fix |
|---|---|
| Everything is slow / databases missing | Run the keepalive script; then `make up` |
| `make up` hangs on weaviate | Normal after idle — it re-elects a raft leader, ~25s |
| Tests fail after pulling changes | `make install` then `make test` |
| Total reset (destroys data, keeps code) | `make nuke && make up && make migrate && make seed` |

---

## LangGraph scope — read before adding it anywhere

LangGraph is **in**, but only for the Day 5 conversational agent.

| Layer | Framework? |
|---|---|
| Ingestion, normalisation, embeddings | **No.** Plain Python. |
| Retrieval: hybrid search, alpha routing, reranking | **No.** This is the part the job is judged on; a framework would hide it. |
| Eval harness and the alpha sweep | **No.** Must measure our own pipeline, not a wrapper's. |
| Day 5 chat agent: state, tool calls, multi-turn flow | **Yes** — LangGraph. |

**Claude is called through the raw `anthropic` SDK (via its Bedrock client,
`AsyncAnthropicBedrockMantle`) inside LangGraph nodes** — *not* through a
LangChain/LangGraph model wrapper. LangGraph supplies the state machine; we keep the
API call itself. That preserves four things a wrapper would take away:

1. **Prompt-cache control.** Caching is prefix-exact via `cache_control`
   checkpoints; a test asserts `response.usage.cache_read_input_tokens > 0` once a
   scenario is large enough to actually clear Bedrock's 4,096-token-per-checkpoint
   minimum (confirmed against the reference Bedrock user guide) - not before, since
   this project's own rule is measured, not guessed.
2. **The model router.** Haiku vs Sonnet chosen per query, with the decision logged.
3. **The adaptive-thinking + `output_config.effort` dial**, tuned per route. Sonnet
   5's card confirms adaptive thinking is always on with a configurable effort
   level; Haiku 4.5's card doesn't give the same detail, so `app/llm/graph.py`
   currently assumes the same shape for both and flags that as unconfirmed, not
   settled.
4. **Per-tenant token/cost metering**, read straight off `response.usage`.

If a future change wants a LangChain/LangGraph model wrapper, that is a real
architectural decision and needs re-deciding, not a quiet swap. (This section has
now been rewritten twice in one day - Anthropic-specific → Gemini-native →
Bedrock-native - each time for the same reason: a real provider change, not a quiet
drift. See "Decisions already locked" below for the dated history.)

## Decisions already locked (don't relitigate)

- Timeline: ~1 week, "Strong" tier
- Runs inside WSL, not Windows (company antivirus blocks Windows↔WSL networking)
- Docker **Engine**, not Docker Desktop (free at any company size)
- LLM: **Anthropic Claude via AWS Bedrock** — `anthropic.claude-sonnet-5` for
  reasoning, `anthropic.claude-haiku-4-5` for cheap lookups. Full history, same day
  (2026-08-21), each change deliberate and dated, not silent drift:
  1. Original plan: Anthropic Claude direct API.
  2. → **Google Gemini** (`gemini-3.1-pro-preview`/`gemini-3.7-flash`, model IDs
     corrected after a real key confirmed `gemini-3-pro` doesn't exist as a text
     model) — the user had a Gemini key on hand, not an Anthropic one.
  3. → **Anthropic Claude again, but via AWS Bedrock** — Gemini's prepayment quota
     ran out mid-build and a live chat request had hung for minutes with no clean
     error; the user has Bedrock access and asked to move there. Model IDs and SDK
     confirmed against the reference Bedrock user guide the user supplied, not
     guessed.
  See the Day 5 section above (both the current Bedrock design and the Gemini-era
  history block) and `CLAUDE.md`'s "LLM usage" section for the full mechanics.
- Embeddings: local BGE on CPU (free, unlimited, needed for the sweep)
- 3 contrasting demo catalogs; reranking enabled and measured
- Interface: Swagger + a minimal chat page. **No React frontend.**
- If time runs short, cut in this order: Helm → offload endpoint → post-purchase agent
  → load test. **Never cut the eval harness.**
