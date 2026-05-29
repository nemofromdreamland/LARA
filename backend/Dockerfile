# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /code

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching.
# Both pyproject.toml AND uv.lock are needed: the lock pins versions while
# pyproject.toml's [tool.uv.sources] redirects torch to the CPU-only wheel
# index so Docker doesn't download the 500 MB CUDA variant.
COPY pyproject.toml uv.lock ./

# Force CPU-only torch variant (no CUDA packages in the container).
ENV UV_TORCH_BACKEND=cpu

# Install production dependencies only (no dev group).
# --frozen: use uv.lock for version pinning, don't re-resolve.
# Cache mount persists downloaded wheels across Docker builds, so a
# failed/retried build doesn't re-download packages that already succeeded.
# en_core_web_sm is declared as a URL dependency in pyproject.toml and
# installed automatically — no separate spacy download step required.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Copy application code
COPY app/ app/
COPY main.py .

# Add venv binaries to PATH
ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
