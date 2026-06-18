# LARA — Architecture & Design Decisions

This document explains *why* LARA is built the way it is. For setup and API
reference, see [README.md](README.md).

## System Overview

Four Docker services: `frontend` (React SPA served by nginx, which also
proxies `/api/*`), `backend` (FastAPI under gunicorn with 2 uvicorn workers),
`redis` (all shared mutable state), and `chroma` (vector store in server
mode). The backend publishes no host port — every request crosses the nginx
proxy, which is what makes the `X-Real-IP` header trustworthy for rate
limiting ([backend/app/limiter.py](backend/app/limiter.py)).

## Key Decisions

| Concern | Choice | Why |
|---------|--------|-----|
| RAG framework | None — hand-written pipeline | Every step is a named, independently testable function; no framework indirection to debug |
| LLM | Groq, Llama 3.3 70B | Free tier, ~500 tok/s, no GPU needed |
| LLM fallback | Cerebras (same model family) | Second free provider; Redis-backed circuit breakers route between them |
| Embeddings | `NeuML/pubmedbert-base-embeddings` (local, 768-dim) | Domain-specific for medical text; no API cost or key; Apache 2.0 |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (toggle: `RERANKER_ENABLED`) | Better relevance ordering than raw cosine distance |
| Vector store | ChromaDB in **server mode** (own container) | Survives backend restarts; embedded mode can't be shared by 2 gunicorn workers |
| Shared state | Redis 7 with AOF | Sessions, chat history, upload jobs, DailyMed cache, rate limits, and circuit-breaker state must be consistent across workers |
| Drug data | DailyMed (NLM) | U.S. government public domain, no key needed |
| Dependency mgmt | `uv` with committed lockfile | Reproducible builds; CI and Docker both install `--frozen` |

## The RAG Pipeline

Every step is a named, independently testable function under
[backend/app/services/](backend/app/services/) (orchestrated by
[rag_pipeline.py](backend/app/services/rag_pipeline.py)):

1. **`chunk_text()`** — paragraph/sentence-aware, 1000 chars with 100-char overlap.
2. **`embed()`** — local sentence-transformers on a dedicated thread pool,
   sized separately from the general blocking-I/O pool so embedding bursts
   can't starve PDF parsing and vice versa.
3. **`store()`** — one ChromaDB collection **per session** (cosine space).
   Collection-per-session makes isolation structural (no `where session_id=`
   filter to forget) and makes session deletion a single `delete_collection`.
4. **`retrieve()`** — top-k=20 with a distance threshold. Multi-drug sessions
   query **per drug** so one drug's leaflet can't monopolise the context
   (`_retrieve_diverse`).
5. **`rerank()`** — cross-encoder rescoring (shares the embed thread pool,
   since embedding and reranking are sequential within a request).
6. **`trim_to_budget()`** — tiktoken-counted context budget; the prescription
   block is counted first, leaflet chunks fill the remainder.
7. **`generate()` / `generate_stream()`** — hallucination-guarded prompt; the
   model must end with a `CITED: drug/section, ...` footer, which the
   pipeline parses to filter the returned source list down to what the answer
   actually used.

### Hallucination guard

The system prompt ([backend/app/services/llm_client.py](backend/app/services/llm_client.py))
restricts answers to the provided context, mandates inline citations, and
defines a fixed refusal sentence when the context lacks the answer. The
`CITED:` footer is machine-parsed and stripped before display; if parsing
fails (LLM format drift), the full source list is returned rather than none.

## Prescription Ingestion

Uploads return **202 + job id** immediately; ingestion runs as a background
task and the frontend polls `/upload/status/{job_id}`. Bundled **sample
prescriptions** (`GET /samples`, `POST /samples/{sample_id}`) follow the same
202 job contract; their leaflets are seeded into the Redis DailyMed cache at
startup so the demo never needs a live DailyMed call
([backend/app/routes/samples.py](backend/app/routes/samples.py)).

Extraction is tiered ([backend/app/services/prescription_parser.py](backend/app/services/prescription_parser.py)):

1. **Quarantine** — prescription text is scanned line-by-line against
   prompt-injection patterns on an ASCII-folded copy (defeats Unicode
   lookalike substitutions). A flagged prescription is rejected outright.
2. **LLM extraction** — structured JSON (drug, dosage, frequency, duration,
   instructions); every returned drug name must pass an allowlist regex.
3. **Regex/spaCy fallback** — if the LLM is down or returns garbage,
   a bullet-format extractor (then a name-only extractor) takes over.
   Which tier fired is a Prometheus counter.

## LLM Routing & Circuit Breakers

`Groq → Cerebras` failover with one circuit breaker per provider
([backend/app/services/circuit_breaker.py](backend/app/services/circuit_breaker.py)):

```text
CLOSED ──(N consecutive failures)──► OPEN ──(cooldown elapsed)──► HALF_OPEN
HALF_OPEN ──probe succeeds──► CLOSED          HALF_OPEN ──probe fails──► OPEN
```

- Breaker state lives in **Redis** so both gunicorn workers trip and reset
  together; failure recording is a Lua script (atomic increment, timestamp,
  and TTL), and the TTL means stuck state self-heals.
- Breakers **fail open** when Redis is down: worst case all workers hammer a
  degraded provider, which retries and the provider fallback still absorb —
  preferable to refusing traffic because the *metadata* store hiccuped.
- Streaming failover: if Groq dies mid-stream after tokens were already sent,
  the stream emits a `reset` event so the client discards the partial text
  before the regenerated fallback answer arrives (Cerebras has no streaming
  API, so its answer arrives as one chunk).

## Security Model (and its honest limits)

- **One static API key** (`X-API-Key`), compared timing-safely, baked into
  the frontend bundle at build time. This is *bot/scraper friction, not
  authentication*: anyone can extract the key from the JS bundle.
- **Per-session ownership token**: `POST /session` issues a high-entropy
  `session_token` ([backend/app/routes/session.py](backend/app/routes/session.py));
  only its sha256 is stored server-side. Session-scoped routes require the raw
  token as `X-Session-Token` and reject a mismatch with 403
  ([backend/app/dependencies.py](backend/app/dependencies.py)) — so a guessed or
  leaked session id alone cannot read another caller's session.
- **Session isolation** is also structural: a UUID session id maps to its own
  Chroma collection and Redis keys, making isolation a property of the data
  layout rather than a `where session_id=` filter.
- **Rate limiting** is per-client-IP via `X-Real-IP`, which only nginx can
  set because the backend has no published port.
- **Prompt-injection defence in depth**: quarantine (above) + drug-name
  allowlist + a system prompt that treats prescription text as untrusted.
- Sessions expire after 2 h (Redis TTL); a background sweep deletes the
  Chroma collections of expired sessions. Expired session → HTTP 410, and
  the frontend silently starts a fresh one.

## Operational Concerns

- **Health**: `GET /health` aggregates Chroma, Redis, embedder, LLM keys, and
  breaker states; returns 503 when degraded. Docker healthchecks gate startup
  ordering (frontend waits for a healthy backend).
- **Observability**: Prometheus metrics at `/metrics` (provider call
  outcomes, extraction tiers, retrieval chunk counts) plus JSON logs with a
  per-request `X-Request-ID`.
- **Testing**: ML-model-loading tests are marked `ml` and excluded from the
  PR pipeline, which runs with `HF_HUB_OFFLINE=1` so an accidental model
  download fails loudly instead of slowing CI. A scheduled weekly workflow
  runs the full suite with real models. Redis logic — including the breaker's
  Lua script — is tested against `fakeredis[lua]`.

## Known Trade-offs / Future Work

- **In-process ingestion jobs**: a worker restart strands a job in
  `processing` (the frontend's poll times out after 2 minutes). A durable
  queue (e.g. arq/Celery) is the upgrade path.
- **Shared static API key**: see Security Model above.
- **Interaction detection is lexical**: it flags drug B's *name* inside drug
  A's official Drug Interactions section. It misses brand-name synonyms and
  drug-*class* warnings, and absence of a flag is not evidence of safety
  ([backend/app/services/interaction_detector.py](backend/app/services/interaction_detector.py)).
- **Context budget** reserves room for the prescription block, the system
  prompt, and conversation history (the history reservation is capped at
  `max_history_tokens`), then fills the remainder with leaflet chunks. Because
  that reservation is capped rather than hard-trimmed, an unusually long chat
  can still push the assembled prompt over the configured budget.
- **TLS** terminates nowhere in this stack — deployment behind a TLS proxy
  is assumed.
