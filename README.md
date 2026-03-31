# ![LARA mascot](frontend/public/mascot.png) LARA — Leaflet Assistant RAG Application

Upload a prescription PDF and ask questions about your medications — answers grounded in official FDA drug leaflets from DailyMed (NLM).

## How It Works

1. You upload a prescription PDF
2. LARA extracts drug names from the text
3. Official leaflet data is fetched from DailyMed (NLM)
4. Leaflets are chunked, embedded, and stored in a session-scoped vector database
5. You ask questions in a chat interface
6. Answers are generated **only** from the retrieved leaflet context — no hallucination

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- A free [Groq API key](https://console.groq.com/keys)

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd LARA
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

### 4. Stop

```bash
docker-compose down
```

To also remove the persisted vector database:

```bash
docker-compose down -v
```

## Local Development (backend only)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd backend
uv sync --dev        # install all dependencies
cp ../.env.example .env   # configure env vars
uv run task dev      # start dev server on :8000
```

### Available tasks

```bash
uv run task test       # run pytest
uv run task test-v     # pytest verbose
uv run task lint       # ruff check
uv run task fmt        # ruff format
uv run task check      # fmt-check + lint + test (CI gate)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key for Llama 3.3 70B inference |
| `CEREBRAS_API_KEY` | No | — | Fallback LLM provider |
| `LLM_PROVIDER` | No | `groq` | Active provider: `groq` or `cerebras` |
| `CHROMA_PATH` | No | `/data/chroma` | ChromaDB persistence directory |
| `FRONTEND_ORIGIN` | No | `http://localhost:5173` | CORS allowed origin |

## Architecture

```
User → Frontend (React/Vite)
         ↓ HTTP
       Backend (FastAPI)
         ├── PDF Parser (PyMuPDF)
         ├── Drug Extractor (regex)
         ├── DailyMed Client (httpx + tenacity)
         ├── Chunker (sliding window)
         ├── Embedder (sentence-transformers, local)
         ├── Vector Store (ChromaDB, local)
         └── LLM Client (Groq → Cerebras fallback)
```

Sessions are isolated by `session_id` (UUID) — no login required.

## Troubleshooting

**`docker-compose up` fails on port conflict**
```bash
# Check what's using the port
netstat -ano | findstr :8000
# Change ports in docker-compose.yml if needed
```

**`uv sync` fails — Python version mismatch**

Ensure Python 3.11+ is installed:
```bash
python --version
uv python install 3.11
```

**DailyMed API returns no results for a drug**

DailyMed covers FDA-approved drugs. Generic names work best (e.g., `lisinopril` not `Prinivil`). Brand names are also indexed but may have multiple entries.

**Groq rate limit (429)**

Free tier limit is 6,000 tokens/minute. Set `LLM_PROVIDER=cerebras` in `.env` as a fallback.

## License

MIT — see [LICENSE](LICENSE)

Drug leaflet data is sourced from [DailyMed](https://dailymed.nlm.nih.gov/dailymed/) (U.S. National Library of Medicine), public domain.
