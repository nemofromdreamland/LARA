# LARA — Claude Code Session Context

## Project
LARA (Leaflet Assistant RAG Application) — full-stack RAG app for querying FDA drug leaflets from uploaded prescriptions.

Full specification: `SPEC.md`

## Key Architecture Decisions

| Concern | Choice | Reason |
|---------|--------|--------|
| Dependency management | `uv` (not pip/venv) | Fast, reproducible, lockfile-based |
| LLM | Groq + Llama 3.3 70B | Free tier, ~500 tok/s, no GPU |
| LLM fallback | Cerebras Inference (same model) | Also free tier |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Fully local, MIT license, no API key |
| Vector store | ChromaDB (embedded, SQLite) | Local, Apache 2.0, native metadata filtering |
| Drug data source | DailyMed NLM API | U.S. government public domain, no key needed |
| PDF parsing | PyMuPDF (fitz) | Fast, MIT license |
| HTTP client | httpx (async) | Replaces requests |
| Retry logic | tenacity | Exponential backoff for DailyMed (flaky) |
| Linter/formatter | ruff | Replaces flake8 + black + isort |
| Framework | FastAPI + Uvicorn | Async, OpenAPI docs free |
| Frontend | React + Vite | Standard SPA |
| Containerization | Docker + docker-compose | 2 services: frontend, backend |
| No LangChain | Hand-written RAG pipeline | Portfolio clarity, explicit steps |

## User Isolation
- `session_id` (UUID) is the **only** isolation mechanism — no auth system
- Every ChromaDB chunk is stored with `session_id` metadata
- All queries filter with `where={"session_id": session_id}`

## RAG Pipeline Steps (all named, all independently testable)
1. `chunk_text()` — sliding window, 500 chars, 50-char overlap
2. `embed()` — sentence-transformers local
3. `store()` — ChromaDB with `{session_id, drug_name, section}` metadata
4. `retrieve()` — semantic search + metadata filter, top_k=5
5. `generate()` — Groq API with hallucination-guard system prompt

## LLM Hallucination Guard (enforced in llm_client.py)
> "You are LARA, a medical information assistant. Answer ONLY using the context provided below. If the context does not contain the answer, respond: 'This information is not available in the provided leaflets.' Never add information from general knowledge. Always cite the section (e.g. 'According to the Warnings section...')."

## Directory Layout
```
LARA/
├── frontend/          # React + Vite (Step 8)
├── backend/
│   ├── pyproject.toml
│   ├── main.py
│   └── app/
│       ├── config.py
│       ├── routes/    # session, upload, chat, health
│       ├── services/  # pdf_parser, drug_extractor, dailymed,
│       │              # chunker, embedder, vector_store,
│       │              # llm_client, rag_pipeline
│       └── models/    # schemas.py
├── docker-compose.yml
├── .env.example
├── .gitignore
├── SPEC.md
└── CLAUDE.md          # (this file)
```

## Build Sequence (incremental — stop after each step for confirmation)
| Step | Deliverable |
|------|-------------|
| 1 | Scaffold (this step) |
| 2 | Docker + docker-compose |
| 3 | FastAPI skeleton: /health, /session, schemas |
| 4 | pdf_parser + drug_extractor + unit tests |
| 5 | DailyMed client + chunker + tests (respx) |
| 6 | embedder + vector_store + /upload + tests |
| 7 | llm_client + rag_pipeline + /chat + tests |
| 8 | React frontend wired to backend |

## Dev Commands (from backend/)
```bash
uv sync --dev          # install all deps
uv run task dev        # start dev server on :8000
uv run task test       # run pytest
uv run task lint       # ruff check
uv run task fmt        # ruff format
uv run task check      # fmt-check + lint + test
```

## API Endpoints
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | /session | — | `{ session_id }` |
| POST | /upload | `session_id`, `file` (PDF) | `{ drugs_found, status }` |
| POST | /chat | `session_id`, `question` | `{ answer, sources[] }` |
| GET | /health | — | `{ status }` |

## Licenses (all permissive)
- FastAPI: MIT | ChromaDB: Apache 2.0 | sentence-transformers: Apache 2.0
- PyMuPDF: AGPL (Docker-isolated, not distributed) | httpx: BSD | tenacity: Apache 2.0
- Llama 3.3: Meta Community License (open weights)
