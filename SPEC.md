# LARA — Full Project Specification

## Project Name
LARA — Leaflet Assistant RAG Application

## Goal
Build a full-stack, containerized (Docker) web application that allows users to upload a prescription (PDF only) and ask questions about the prescribed drugs using official drug leaflet data (not the prescription text itself).

The application must use a Retrieval-Augmented Generation (RAG) pipeline grounded in trusted medical sources.

## Target Users
Primary audience is USA-based users. Drug data is sourced from DailyMed (NLM), which covers FDA-approved drugs.

## Core User Flow
1. User accesses a landing page (single-page app)
2. A unique `session_id` is generated for the user
3. User uploads a prescription in PDF format
4. Backend extracts text from the PDF
5. System identifies drug names from the extracted text
6. System fetches official drug leaflet data from the DailyMed (NLM) API
7. Leaflet data is processed, chunked, and embedded
8. Each embedded chunk is stored with metadata: `session_id`, `drug_name`, `section`
9. User asks questions in a chat interface
10. System retrieves only chunks filtered by `session_id`
11. System generates grounded answers based only on that filtered context

## Technical Stack

### Frontend
- React (Vite)

### Backend
- FastAPI (Python 3.11+)
- Dependency management: `uv`
- Global CLI tools: `pipx` (installs `uv`, `ruff` in isolated envs)

### LLM
- Groq API free tier — Llama 3.3 70B
  - ~500 tok/s inference, no GPU required
  - Free tier: 6,000 tokens/min
- Fallback: Cerebras Inference free tier (also Llama 3.3)
- Provider abstraction via `LLMProvider` enum — swap with one env var

### Embeddings
- sentence-transformers — `all-MiniLM-L6-v2`
  - Runs fully locally, MIT licensed, no API key

### Vector Database
- ChromaDB (local, embedded)
  - Apache 2.0 licensed
  - Native `where=` filtering for `session_id` isolation
  - SQLite-backed for small collections
  - Persisted via Docker volume
  - Upgrade path: Qdrant Cloud free tier

### PDF Parsing
- PyMuPDF (fitz)

### HTTP Client
- httpx — async

### Retry Logic
- tenacity — retry with exponential backoff for DailyMed API

### External API
- DailyMed (NLM) — https://dailymed.nlm.nih.gov/dailymed
  - U.S. government public domain, no API key required
  - Sections of interest: boxed_warnings, indications, dosage, contraindications,
    drug_interactions, adverse_reactions, warnings

### Containerization
- Docker + docker-compose (2 services: frontend, backend)

### RAG Pipeline
- No LangChain — explicit, hand-written pipeline
- Steps: `chunk_text()` → `embed()` → `store()` → `retrieve()` → `generate()`

## Dev Tooling

```toml
[tool.taskipy.tasks]
dev       = "uvicorn app.main:app --reload --port 8000"
test      = "pytest"
test-v    = "pytest -v"
lint      = "ruff check app tests"
fmt       = "ruff format app tests"
fmt-check = "ruff format --check app tests"
check     = "task fmt-check && task lint && task test"
```

- **ruff** — linter + formatter
- **pytest** — test runner
- **pytest-asyncio** — async test support (`asyncio_mode = "auto"`)
- **pytest-cov** — coverage (gate: 80% minimum)
- **respx** — mock httpx requests in tests
- **anyio** — async test backend
- **taskipy** — task runner via `uv run task <name>`

## Architecture Requirements
- Services: frontend, backend
- Docker Compose
- Persist vector DB using volumes (`./chroma_data`)
- Backend orchestrates full RAG pipeline
- Support multiple users via `session_id` (no authentication)
- LLM inference is remote (Groq) — no Ollama

## Backend Module Structure
```
backend/
├── pyproject.toml
├── main.py
└── app/
    ├── config.py
    ├── routes/
    │   ├── session.py
    │   ├── upload.py
    │   ├── chat.py
    │   └── health.py
    ├── services/
    │   ├── pdf_parser.py
    │   ├── drug_extractor.py
    │   ├── dailymed.py
    │   ├── chunker.py
    │   ├── embedder.py
    │   ├── vector_store.py
    │   ├── llm_client.py
    │   └── rag_pipeline.py
    └── models/
        └── schemas.py
```

## Test Structure
```
backend/tests/
├── conftest.py
├── unit/
│   ├── test_config.py
│   ├── test_pdf_parser.py
│   ├── test_drug_extractor.py
│   ├── test_chunker.py
│   ├── test_dailymed.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   └── test_llm_client.py
└── integration/
    ├── test_upload_route.py
    └── test_chat_route.py
```

## Functional Requirements
- Generate a `session_id` (UUID)
- Store embeddings with metadata: `session_id`, `drug_name`, `section`
- Metadata filtering in vector search: `where={"session_id": session_id}`
- Extract drug names via regex suffix heuristic + Rx-line pattern
- Query DailyMed NLM, parse LOINC-coded sections
- Chunk leaflet text (500 chars, 50-char overlap)
- Semantic retrieval with metadata filtering (top_k=5)
- Generate answers ONLY from retrieved context
- Include source attribution (drug name + section)
- Retry DailyMed calls with exponential backoff (tenacity, 3 attempts)

## LLM Hallucination Guard
System prompt enforced in `llm_client.py`:

> "You are LARA, a medical information assistant. Answer ONLY using the context provided below. If the context does not contain the answer, respond: 'This information is not available in the provided leaflets.' Never add information from general knowledge. Always cite the section (e.g. 'According to the Warnings section...')."

## Non-Functional Requirements
- No login/password system
- `session_id` is the only isolation mechanism
- Must avoid hallucinations — if not in context, say so explicitly
- Keep implementation simple and modular
- Code must be clean and portfolio-ready
- All libraries: MIT or Apache 2.0 licensed
- Everything deployable for free (Groq free tier + free cloud host)

## API Design

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | /session | — | `{ session_id }` |
| POST | /upload | `session_id`, `file` (PDF) | `{ drugs_found, status }` |
| POST | /chat | `session_id`, `question` | `{ answer, sources[] }` |
| GET | /health | — | `{ status }` |

## RAG Pipeline Detail

### Ingestion (/upload)
1. Extract text from PDF — PyMuPDF
2. Extract drug names — regex heuristic
3. Fetch each drug from DailyMed — httpx + tenacity
4. Parse LOINC sections into `LeafletSection` dataclasses
5. Chunk each section — sliding window
6. Embed chunks — sentence-transformers local
7. Store in Chroma with `{session_id, drug_name, section}` metadata

### Query (/chat)
1. Embed question — sentence-transformers
2. Query Chroma with `where={"session_id": session_id}`
3. Retrieve top-5 chunks
4. Build prompt with SYSTEM_PROMPT + context + question
5. Call Groq API (Llama 3.3 70B)
6. Return answer + source list

## Step-by-Step Build Plan

### Phase 1 — Core (MVP)
| Step | Deliverable | Test |
|------|-------------|------|
| 1 | Git init, folder scaffold, config files, pyproject.toml | `git log`, `uv sync` |
| 2 | docker-compose.yml + Dockerfiles | `docker-compose up` → both start |
| 3 | FastAPI skeleton: main.py, /health, /session, schemas | `curl /health → 200` |
| 4 | pdf_parser + drug_extractor + unit tests | `uv run task test` passes |
| 5 | DailyMed client + chunker + tests (respx) | mocked fetch returns LeafletSection list |
| 6 | embedder + vector_store + /upload + tests | Chroma has docs with correct metadata |
| 7 | llm_client + rag_pipeline + /chat + tests | mocked LLM returns grounded answer |
| 8 | React frontend: upload panel + chat UI | Full e2e flow in browser |

### Phase 2 — Reliability & Resilience
| Step | Deliverable |
|------|-------------|
| 9 | Groq → Cerebras fallback with circuit breaker |
| 10 | Session expiry: background job deletes old Chroma docs |
| 11 | Graceful DailyMed miss handling |

### Phase 3 — Quality & Accuracy
| Step | Deliverable |
|------|-------------|
| 12 | Replace regex drug extractor with spaCy NER |
| 13 | Streaming responses via SSE on /chat |
| 14 | Drug interaction detection: cross-reference multiple drugs |

### Phase 4 — Scale & Infrastructure
| Step | Deliverable |
|------|-------------|
| 15 | Migrate vector store to Qdrant Cloud free tier |
| 16 | CI pipeline: GitHub Actions running `task check` |

## Git & GitHub Requirements
- Initialize Git from the beginning
- Commit conventions: `feat:` / `fix:` / `chore:` / `refactor:`
- Monorepo structure
- README.md with Docker setup, usage, troubleshooting
- `.env.example` with all required keys documented

## .gitignore
```
# Python
__pycache__/
*.pyc
.venv/
.env

# Frontend
node_modules/
dist/
build/

# Vector DB
chroma_data/
*.db

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```
