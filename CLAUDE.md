# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**CatalogMind** — a multi-tenant conversational commerce API. A merchant POSTs a product
feed and gets a grounded, streaming shopping assistant scoped strictly to their catalog.

It is a portfolio project built to target a specific **AI/ML Engineer** role at Rezolve AI
PLC (Nasdaq: RZLV). Every architectural choice traces back to a line in that job
description. That context matters when judging tradeoffs: **the goal is demonstrable
engineering depth on a narrow slice, not feature breadth.**

The full approved blueprint lives at
`/mnt/c/Users/AKASH/.claude/plans/stateless-churning-sparrow.md`
(the same file is `C:\Users\AKASH\.claude\plans\stateless-churning-sparrow.md` from
Windows — use the `/mnt/c` form, since sessions run inside WSL).
Read it before proposing anything structural.

## Read PROGRESS.md first

`PROGRESS.md` is the live tracker: what is done, what is next, what the user still owes.
**Read it at the start of every session, and update it at the end of every session** —
tick completed items, move the "Current day" marker, refresh the test count. It is what
lets the user (and a fresh session) resume without re-deriving state from the git log.

## Current state

Days 0–5 complete and live-verified (infra, tenant isolation, ingestion, hybrid
search + eval harness, and the chat agent - now on Claude via AWS Bedrock, after a
same-day provider round-trip: Anthropic → Gemini → Anthropic/Bedrock, each change
dated in `PROGRESS.md`, not silent drift). Day 6 in progress: generation-quality
eval is done; CI gate, cost tracking, and Helm are still open. The chat agent's
tool surface also grew from 1 tool to 3 (catalog-wide stats + exact lookup, on
top of search) after live manual testing found a real correctness bug outside
the original plan - see `PROGRESS.md`'s "Agentic RAG expansion" entry. This line
is a one-sentence pointer only - `PROGRESS.md` is the live tracker with the real
detail; update *that* file every session, and only bump this line when the day
count moves. Days 1–7 are specified in the plan file, §11.

## Where this runs — WSL is canonical

**Everything happens inside WSL2 Ubuntu 26.04 at `~/catalogmind`.** Run every command
there: `uv`, `pytest`, `make`, `docker`. This matches GitHub Actions CI and the
production Dockerfile exactly, so "works locally" and "works in CI" are the same claim.

`C:\Users\AKASH\rezolve` is a **stale copy**, kept only as a backup. Do not edit it — it
will diverge. Claude Code can edit the live tree through
`\\wsl.localhost\Ubuntu\home\akash\catalogmind\...`, which the Read/Write/Edit tools
handle fine.

### Why not the Windows host

Two separate corporate-security blocks, in sequence:

1. **WSL2 VM creation** was blocked by Seqrite Endpoint Protection
   (`Wsl/Service/CreateInstance/0xd0000022`). **IT has fixed this** — WSL now works.
2. **Windows→WSL networking is still blocked.** Services listening in WSL are
   unreachable from Windows. Confirmed not fixable from our side: NAT mode fails,
   `networkingMode=mirrored` fails (verified active — the VM's `eth1` carries the host's
   Wi-Fi IP), and Hyper-V firewall `LoopbackEnabled` is already `True`. Decisive test: a
   plain WSL listener **accepts a TCP connection** from Windows but the HTTP payload
   times out, while Docker-published ports do not connect at all. Connection established,
   data dropped — stateful inspection, i.e. Seqrite's firewall.

Do not send the user back to the BIOS: VT-x is enabled.
`Win32_Processor.VirtualizationFirmwareEnabled` reads `False` only because a hypervisor
already owns VT-x — a reporting artifact, not a fault.

`.devcontainer/` is retained as a working fallback and for the "Open in Codespaces"
badge, not because we currently need it.

## Daily startup

```powershell
# once per Windows login, from PowerShell:
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\akash\catalogmind\scripts\wsl-keepalive.ps1
```

```bash
# then, inside WSL:
cd ~/catalogmind && make up     # blocks until weaviate actually answers
```

**Why the keepalive matters.** WSL2 stops an idle VM after ~60s, taking the Docker stack
with it. `restart: unless-stopped` then revives the containers on the next command, so
the symptom is baffling rather than obvious: containers perpetually "Up N seconds" with
`RestartCount=0`, and Weaviate redoing its ~25s raft election every time, making
readiness look randomly intermittent. There is **no config key** for this in WSL 2.7.3 —
`vmIdleTimeout` is rejected as unknown under both `[wsl2]` and `[experimental]` (both
verified). Holding a WSL session open is the only reliable fix.

Without the keepalive nothing breaks permanently — data lives in named volumes — the
first command after idle just pays ~25s while Weaviate re-elects. `make up` waits for it.

## Docker

Docker **Engine** (not Desktop) runs inside WSL: v29.7.2, Compose v5.5.0, systemd-managed.
Chosen over Desktop because it is free at any company size — Desktop requires a paid
licence above 250 employees / $10M revenue, which is a live question on a company laptop.

### Weaviate gotchas, both already hit

- **Single-node raft must be bootstrapped explicitly.** Weaviate 1.25+ routes schema
  through raft even with one node. Without `RAFT_JOIN=node1` + `RAFT_BOOTSTRAP_EXPECT=1`
  it enters follower state, loops `attempting to join` against itself, and the HTTP API
  never opens.
- **The image is distroless** — no `sh`, `curl`, `wget`, `nc — so a container healthcheck
  is impossible. There is deliberately none. An earlier
  `["CMD","/bin/weaviate","--help"]` reported *healthy* for a container whose API was
  dead, and `depends_on: service_healthy` would have gated on that lie. Assert readiness
  via `/v1/.well-known/ready` from outside instead.

## Commands

```bash
make install     # uv venv on Python 3.12 + all deps
make up          # weaviate + postgres + mongo + redis (datastores only)
make dev         # API on the host with reload -> localhost:8000/docs
make migrate     # alembic upgrade head
make seed        # provision 3 demo merchants + ingest catalogs
make test        # unit tests, no stack needed
make test-all    # includes integration tests (requires `make up`)
make lint fmt typecheck
make sweep       # the alpha experiment -> eval/results/tuned_alpha.json
make eval        # full eval suite; regenerates every number in the README
```

Compose profiles: default = 4 datastores · `--profile app` = +API container (CI) ·
`--profile offload` = +MinIO (Weaviate tenant OFFLOADED state).

### Port remapping — do not "fix" this

This machine runs a **native Windows Postgres on 5432 and Redis on 6379** belonging to
other projects. The compose stack therefore binds **5433 → 5432** and **6380 → 6379** on
the host, and `.env.example` / `app/config.py` point at those host ports.

Do not revert these to the defaults: it would either fail to bind or, worse, silently
connect CatalogMind to another project's database. Service-to-service traffic *inside*
the compose network still uses the standard ports (`postgres:5432`, `redis:6379`) — only
host-side mappings are shifted.

## Non-negotiable invariants

These are the things the whole project is arguing for. Breaking one silently defeats the
purpose of building it.

1. **Tenant isolation is enforced by the engine, never the caller.** Weaviate *native*
   multi-tenancy (`collection.with_tenant(...)`), Postgres RLS, per-merchant Mongo
   collections. Never add a `merchant_id` where-filter as the isolation mechanism — that
   is precisely the shortcut this project exists to reject. There is a must-pass CI test:
   merchant A's key must return zero results for a SKU that only exists in merchant B.

2. **Tenant comes from the API key, never the request body.** Resolved in one FastAPI
   dependency. No handler reads a tenant from user-supplied payload.

3. **Published numbers are generated, never typed.** Every figure in `README.md` comes
   from `make eval`. If you cannot regenerate it, do not publish it.

4. **Alpha values are measured, not guessed.** `app/retrieval/alpha_router.py` loads
   `eval/results/tuned_alpha.json` produced by the sweep. `PRIOR_ALPHA` is a documented
   fallback for a fresh clone — not a place to hand-tune.

5. **No LLM call on the retrieval hot path.** Query classification is lexical heuristics.
   LLM calls belong in generation and in offline ingestion-time attribute extraction.

6. **`Product.embedding_text()` is frozen.** Changing field order or content invalidates
   every stored vector. If it must change, that is a re-index migration, not an edit.

## Conventions

- **Python 3.12**, pinned. The system has 3.14 installed — do not use it; ML wheels lag.
  Always `uv run` or the `.venv`.
- `from __future__ import annotations` at the top of every module.
- Async throughout — asyncpg, motor, redis.asyncio, Weaviate async client.
- Pydantic v2 for all boundaries; `pydantic-settings` for config (never read `os.environ`
  directly outside `app/config.py`).
- Ruff, line length 100.
- Type hints everywhere; `mypy app` must stay clean.
- Docstrings explain **why**, not what. The existing modules set the tone — match it.
  This code is read by a hiring CTO; a comment that explains a tradeoff is worth more
  than one that narrates the next line.

## LLM usage

**Provider is Anthropic Claude, via AWS Bedrock — not the direct Anthropic API, and
not Google Gemini.** Two pivots happened the same day (2026-08-21), both at the
user's explicit direction, both dated rather than silently overwritten: Anthropic
(original plan) → Gemini (a Gemini key was on hand, not an Anthropic one) → Anthropic
**via Bedrock** (rolled back after Gemini's quota ran out mid-build and a live chat
request hung for minutes with no clean error; the user has Bedrock access). See
`PROGRESS.md`'s "Decisions already locked" section for the full dated history - do
not re-litigate it, and do not reintroduce `google-genai`/Gemini anywhere.

- `anthropic.claude-sonnet-5` for multi-constraint reasoning, `anthropic.claude-
  haiku-4-5` for lookups and bulk eval. Model IDs come from `app/config.py` - these
  are Bedrock's bare, Messages-API-style names (confirmed against the reference
  Bedrock user guide the user supplied), **not** the fully dated `-v1:0` modelId
  strings the boto3 Invoke/Converse APIs need (this project doesn't use those APIs).
- SDK is `anthropic[bedrock]`'s `AsyncAnthropicBedrockMantle`
  (`from anthropic import AsyncAnthropicBedrockMantle`) — the Messages-API-compatible
  Bedrock client, not the lower-level boto3 client and not the direct-API
  `AsyncAnthropic` class. Auth is a Bedrock long-term API key (bearer token) via
  `AWS_BEARER_TOKEN_BEDROCK`, generated from the Bedrock console - **not** full AWS
  IAM access/secret keys, and not `ANTHROPIC_API_KEY`.
- Because Bedrock exposes the same Messages API shape as the direct API, almost the
  entire original pre-Gemini design intent survives unchanged: `thinking={"type":
  "adaptive"}`, `output_config={"effort": ...}`, prompt-cache checkpoints via
  `cache_control`, cost metering off `response.usage`. **But only for Sonnet 5** -
  confirmed live, not just from the guide: Haiku 4.5 rejects both
  `thinking={"type":"adaptive"}` ("adaptive thinking is not supported on this
  model") and `output_config.effort` ("This model does not support the effort
  parameter") outright. It only accepts the older `thinking={"type":"enabled",
  "budget_tokens":N}` shape (which also needs `max_tokens > budget_tokens`) or
  neither param at all. `app/llm/graph.py` sends `thinking`/`output_config` only on
  the reasoning (Sonnet 5) path for exactly this reason - not because Haiku
  couldn't use *some* thinking config, but because forcing the cheap/fast tier into
  the older explicit-budget mode would spend real thinking-token cost on the
  queries specifically meant to avoid that.
- Prompt caching needs a **minimum of 4,096 tokens in one cache checkpoint** to
  actually cache anything (confirmed against the reference guide) - `cache_control`
  on a short system-only prefix is accepted but does nothing below that floor.
  `app/llm/prompting.py`'s system prompt is a few hundred tokens; the marker is
  still attached (correctly positions this prefix for when real multi-turn history
  crosses the threshold) but claiming a cache hit on a short single-turn prefix
  today would be asserting something not measured. Assert
  `response.usage.cache_read_input_tokens > 0` only once a test scenario actually
  clears 4,096 tokens for real - not before.
- Cost metering off `response.usage` (`input_tokens`, `output_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`) - same field name the
  original pre-Gemini plan specified; Gemini's `response.usage_metadata` naming
  does not apply here.
- **Two timeout layers, applied from the start this time rather than rediscovered**:
  a real Gemini-era hang sat for two minutes with `200 OK` already received and zero
  further progress, and the SDK's own documented timeout setting did not reliably
  bound a stall *between* streamed chunks. `app/llm/client.py` sets a constructor-
  level `timeout=` as a first, best-effort layer, but `app/llm/graph.py` wraps each
  streamed chunk in `asyncio.wait_for` as the layer actually relied on - not yet
  re-verified against a live Bedrock stall, so treat that as inherited caution from
  the Gemini build, not a confirmed fact about Bedrock specifically, until it's
  actually been tested here too.
- **Claude can emit more than one `tool_use` block in a single turn** for a
  compound ask (e.g. two unrelated product searches in one message) - confirmed
  live, not a hypothetical, after it broke a real conversation: `app/llm/graph.py`
  only ever answers `tool_use_blocks[0]`, so a second, unanswered `tool_use` id
  persisted into the LangGraph checkpointer's message history and made Anthropic
  reject every later call on that conversation with `tool_use ids were found
  without tool_result blocks immediately after`. Fixed by setting
  `tool_choice: {"type": "auto", "disable_parallel_tool_use": True}` on every call
  that has `tools` attached - confirmed real from the installed `anthropic` SDK's
  `ToolChoiceAutoParam` type, not guessed. Keep this set as long as the graph
  design assumes one tool call per round (`tool_call_rounds`, `citations` as a
  single result set) - lifting it would need `tool_node` to handle and answer
  multiple simultaneous `tool_use` blocks, not just the cap removed. Now that
  the agent has three tools (`search_catalog`/`get_catalog_stats`/
  `get_product_detail`, added 2026-08-22) instead of one, the temptation for
  the model to request two *different* tools in one compound turn is higher
  than before, not lower - this constraint matters more, not less.
- **The installed `anthropic` SDK's `ToolUseBlock` *response* type carries
  fields (`toolset_name`, `caller`) that Bedrock's backend rejects as *request*
  input on a replayed conversation** - confirmed live (2026-08-22), not a
  hypothetical: a real second chat turn failed with `messages.1.content.0.
  tool_use.toolset_name: Extra inputs are not permitted`. Root cause, confirmed
  with `model_dump(exclude_unset=True)`, not guessed: a block parsed from a
  real API response has those fields marked "set" (the server's JSON
  explicitly includes them, even as `null`), so echoing that exact object back
  resends them - a *freshly constructed* `ToolUseBlock` with only `id`/`name`/
  `input`/`type` never touches them, so `exclude_unset` correctly drops them.
  **Sanitizing only at message-storage time was not enough** - LangGraph's
  checkpoint serialize/deserialize round trip doesn't preserve pydantic's
  unset-vs-set distinction for an unregistered type (confirmed via
  `graph.aget_state()` - a cleanly-built object came back "dirty" after one
  round trip), so the real fix lives at `app/llm/graph.py::
  _sanitize_messages_for_request`, applied fresh on *every* outgoing call, not
  once when a message is first built. If a future SDK bump adds yet another
  response-only field to a content block type this project echoes back into
  history, expect the same failure shape and the same fix location.
- **Claude reliably follows a self-defined inline text protocol, across both
  model tiers** - confirmed live (2026-08-22), not assumed. A family of regex
  heuristics that tried to guess citation/refusal/stat-claim intent from
  Claude's free-form generated prose (`app/llm/refusal_text.py`,
  `app/llm/citations.py`, `app/llm/claims.py`) was retired after one of them
  missed a second real live refusal within one turn of shipping a fix for the
  first miss - proof the approach was structurally incomplete, not
  under-tuned (full story, including a first-hypothesis retrieval-score
  threshold that real measurement disproved before it shipped, in
  `PROGRESS.md`'s dated entry). Replaced with `app/llm/markers.py`: Claude
  declares three facts in a fixed, visible-ASCII format instead of prose the
  code then has to guess the meaning of - `[[NO_MATCH]]` as the literal first
  characters of a refusal, `[[SKU:X]]` wrapping every cited product,
  `[[STAT:N]]` wrapping every `get_catalog_stats`-backed figure. Verified on
  real Bedrock calls across both `anthropic.claude-haiku-4-5` and
  `anthropic.claude-sonnet-5` - 100% marker compliance, zero hallucinated
  markers, observed across every real call made since (dozens, spanning
  single-shot smoke tests through full multi-round tool-calling
  conversations, confirmed again by auditing real answers for an unmarked
  citation and finding none). **This is now the standing pattern for
  verifying any claim Claude makes about the catalog.** If a new claim type
  needs verifying, extend the marker family (`app/llm/markers.py`) - do not
  reintroduce a regex that guesses at free-text phrasing; that is precisely
  the mistake this replaced.
- There is no repo or environment skill for the Bedrock/Anthropic API. Before
  writing or changing a call, check the *current* Bedrock user guide (the PDF the
  user supplied) or web search rather than trusting training-time knowledge -
  guessing at API specifics has already produced real bugs in this build (a wrong
  Gemini model ID, a dropped `thought_signature`); do not repeat that pattern here
  just because the provider changed.

### LangGraph is scoped to the Day 5 agent only

Orchestration for the conversational agent uses LangGraph. **Nothing else may.**
Ingestion, retrieval, alpha routing, reranking and the eval harness stay plain Python —
those are the parts the job description is actually testing, and a framework would
abstract exactly the depth we are trying to show.

Inside LangGraph nodes, call Claude with the **raw `anthropic` SDK** (via its Bedrock
client), never a LangChain/LangGraph model wrapper. The wrapper would cost us
prompt-cache prefix control (`cache_control` checkpoints, a test asserts
`cache_read_input_tokens > 0` once a scenario is large enough to actually cache), the
Haiku/Sonnet model router, the adaptive-thinking + `output_config.effort` dial, and
per-tenant cost metering off `response.usage`.
See PROGRESS.md § "LangGraph scope".

## Cost discipline

Target spend for the whole build is ~$15–40. Run sweeps on Haiku; reserve Sonnet for
the final scored pass. Retrieval eval is free (embeddings are local CPU). Check
before launching any loop that makes thousands of calls.

## Scope guard

Explicitly out of scope — do not add: crypto/stablecoin checkout, geolocation/geofencing,
a mobile app, model fine-tuning, a React/Next.js frontend, or anything reproducing
Rezolve's branding, product names, or copy. This is an independent implementation of a
public product category on public datasets.

If the week slips, the fixed cut order is: Helm → offload endpoint → post-purchase agent
→ load test. **Never cut the eval harness** — it is the differentiator.

## Data

`data/raw/` and `data/processed/` are git-ignored. Dataset provenance and licence terms go
in `data/SOURCES.md`. Verify a licence before ingesting anything.
