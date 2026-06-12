<div align="center">

<img src="frontend/public/mascot.png" alt="LARA mascot" width="110">

<h1>LARA</h1>

**L**eaflet **A**ssistant **R**AG **A**pplication

Upload a prescription PDF and ask questions about your medications —<br>
answers grounded in official FDA drug leaflets from [DailyMed](https://dailymed.nlm.nih.gov/dailymed/) (NLM),<br>
streamed token-by-token with per-answer source citations.

[![CI](https://github.com/nemofromdreamland/LARA/actions/workflows/ci.yml/badge.svg)](https://github.com/nemofromdreamland/LARA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-f39237.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/docs-architecture_%26_design-1a2744.svg)](ARCHITECTURE.md)

[How it works](#how-it-works) · [Setup](#setup) · [API](#api) · [Testing](#testing--ci) · **[Architecture & design decisions →](ARCHITECTURE.md)**

<!-- TODO: record a 30-second demo (upload → drugs extracted → streamed answer with sources)
     and embed it here:  ![LARA demo](docs/demo.gif) -->

</div>

---

## How It Works

1. **Upload** — you submit a prescription PDF; ingestion runs as an async job (HTTP 202 + status polling)
2. **Extract** — LARA pulls structured medication entries (drug, dosage, frequency): LLM-first, with regex/spaCy fallback tiers, behind a prompt-injection quarantine
3. **Fetch** — official leaflet sections come from DailyMed and are cached in Redis
4. **Index** — leaflets are chunked, embedded locally (PubMedBERT), and stored in a **per-session ChromaDB collection**
5. **Retrieve** — questions are answered per-drug (every prescribed drug gets representation), reranked by a local cross-encoder, and trimmed to a token budget
6. **Answer** — tokens stream over SSE, generated **only** from the retrieved leaflet context; the LLM must cite which `drug/section` labels it used, and the source list is filtered to those citations

## Architecture

```text
Browser ──► frontend (nginx :5173)
              ├── serves the React SPA
              └── /api/* ──► backend (FastAPI + gunicorn×2, no host port)
                              ├── RAG pipeline: chunk → embed → store → retrieve
                              │                 → rerank → trim → generate
                              ├── embedder: sentence-transformers PubMedBERT (local)
                              ├── reranker: BAAI/bge-reranker-base (local)
                              ├── LLM: Groq (Llama 3.3 70B) ──► Cerebras fallback,
                              │        Redis-backed circuit breakers per provider
                              ├── DailyMed client (httpx + tenacity, Redis cache)
                              ├── chroma (vector store, server mode) ── per-session collections
                              └── redis (sessions, chat history, upload jobs,
                                         rate limits, circuit-breaker state)
```

Four Docker services: `frontend`, `backend`, `redis`, `chroma`. The backend deliberately has **no published host port** — all traffic goes through the nginx proxy, so clients can't spoof the `X-Real-IP` header that rate limiting keys on.

Design rationale and trade-offs: see [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- A free [Groq API key](https://console.groq.com/keys) (optionally a [Cerebras](https://cloud.cerebras.ai) key for the fallback provider)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/nemofromdreamland/LARA.git
cd LARA
```

### 2. Configure environment variables (two files)

```bash
cp .env.example backend/.env   # read by the backend container (env_file)
cp .env.example .env           # read by docker-compose for the frontend build arg
```

Edit **`backend/.env`** and set:

```ini
LARA_API_KEY=<a strong random secret>   # e.g. `openssl rand -base64 32`
GROQ_API_KEY=<your Groq API key>
```

Edit **`.env`** (repo root) and set `LARA_API_KEY` to the **same value** — docker-compose injects it into the frontend bundle at build time. If you rotate the key, change both files and rebuild the frontend image.

### 3. Start with Docker Compose

```bash
docker-compose up --build
```

> First boot downloads the two local ML models (PubMedBERT embedder + bge reranker, ~1 GB total) into a Docker volume — the backend healthcheck allows up to 5 minutes for this. Subsequent starts are fast.

Open <http://localhost:5173> — the SPA and the API (proxied under `/api/`) are both served by nginx.

The backend is not directly reachable from the host in Docker. For Swagger/OpenAPI docs, run the backend locally (below) and open <http://localhost:8000/docs>.

### 4. Stop

```bash
docker-compose down       # keep data
docker-compose down -v    # also remove vector DB, Redis data, and model cache
```

## Local Development (backend)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). The backend needs Redis and ChromaDB, which you can run from the same compose file:

```bash
docker-compose up -d redis chroma   # Redis on :6379, Chroma on :8001

cd backend
uv sync --dev                  # install all dependencies
cp ../.env.example .env        # then set LARA_API_KEY + GROQ_API_KEY
uv run task dev                # dev server on :8000 (Swagger at /docs)
```

### Available tasks

```bash
uv run task test         # full pytest suite (loads real ML models)
uv run task test-light   # suite without ml-marked tests (what CI runs)
uv run task lint         # ruff check
uv run task fmt          # ruff format
uv run task check        # fmt-check + lint + test (CI gate, 80% coverage minimum)
```

## API

All endpoints except `GET /health` require the `X-API-Key` header.

| Method | Endpoint | Body / params | Response |
|--------|----------|---------------|----------|
| POST | `/session` | — | `{ session_id }` |
| POST | `/upload` | `session_id`, `file` (PDF, ≤20 MB) | **202** `{ job_id, status: "processing" }` |
| GET | `/upload/status/{job_id}` | `?session_id=` | `{ status, drugs_found[], missing_leaflets[], error }` |
| POST | `/chat` | `{ session_id, question }` | `{ answer, sources[] }` |
| POST | `/chat/stream` | `{ session_id, question }` | SSE: `token` / `reset` / `sources` / `done` events |
| POST | `/interactions` | `{ session_id }` | `{ pairs_checked, interactions[] }` — lexical scan of official *Drug Interactions* sections (name matches only; not a pharmacological model) |
| GET | `/health` | — | `{ status, components }`; **503** when degraded |
| GET | `/metrics` | — | Prometheus metrics |

Sessions are isolated by `session_id` (UUID) with a 2-hour TTL — no login. An expired session returns **410** and the frontend transparently starts a new one. Conversation history lives server-side in Redis.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LARA_API_KEY` | **Yes** | — | Static API key checked on every request (timing-safe); also baked into the frontend bundle at build time |
| `GROQ_API_KEY` | Yes* | — | Groq API key for Llama 3.3 70B inference (*required when `LLM_PROVIDER=groq`, the default) |
| `CEREBRAS_API_KEY` | No | — | Fallback LLM provider (same model family) |
| `LLM_PROVIDER` | No | `groq` | Primary provider: `groq` or `cerebras` |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection (compose overrides to the `redis` service) |
| `CHROMA_HOST` / `CHROMA_PORT` | No | `localhost` / `8001` | ChromaDB server (compose overrides to the `chroma` service) |
| `FRONTEND_ORIGIN` | No | `http://localhost:5173` | CORS allowed origin |
| `RERANKER_ENABLED` | No | `true` | Toggle the cross-encoder reranking stage |

Tuning knobs (retrieval top-k and distance threshold, context token budget, rate limits, circuit-breaker thresholds, thread-pool sizes, TTLs) are all env-configurable — see [`backend/app/config.py`](backend/app/config.py).

## Testing & CI

- ~4,300 lines of tests against ~2,400 lines of application code; 80% coverage gate.
- Redis behaviour (including the circuit breaker's Lua script) is tested against `fakeredis[lua]` — no Redis server needed.
- Tests that load real ML model weights are marked `ml`. The [PR pipeline](.github/workflows/ci.yml) runs everything else fully offline (`HF_HUB_OFFLINE=1` guarantees no silent model downloads); a [weekly workflow](.github/workflows/ci-full.yml) runs the full suite with real models.

## Troubleshooting

**`docker-compose up` fails: `env file ./backend/.env not found`** — you skipped Setup step 2; both env files are required.

**Frontend loads but every request returns 401** — `LARA_API_KEY` differs between root `.env` and `backend/.env`. Set them to the same value and rebuild: `docker-compose up --build frontend`.

**Backend stays unhealthy for several minutes on first start** — it is downloading the embedding and reranker models; watch progress with `docker-compose logs -f backend`.

**DailyMed returns no leaflet for a drug** — DailyMed covers FDA-approved drugs; generic names work best (e.g. `lisinopril` rather than `Prinivil`). Missing drugs are reported per-upload in `missing_leaflets`.

**Groq rate limit (429)** — handled automatically: the request falls back to Cerebras (if `CEREBRAS_API_KEY` is set), and a Redis-backed circuit breaker stops hammering Groq until its cooldown expires. Without a Cerebras key, sustained 429s surface as 503s.

## License

MIT — see [LICENSE](LICENSE).

Drug leaflet data is sourced from [DailyMed](https://dailymed.nlm.nih.gov/dailymed/) (U.S. National Library of Medicine), public domain.

## Disclaimer

> **LARA is not a substitute for professional medical advice — always consult your doctor or pharmacist.**
> It provides information from official drug leaflets only and cannot account for your personal medical situation.
