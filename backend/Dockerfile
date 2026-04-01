FROM python:3.11-slim

WORKDIR /code

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml .

# Install production dependencies only (no dev group)
# en_core_web_sm is declared as a URL dependency in pyproject.toml and
# installed automatically — no separate spacy download step required.
RUN uv sync --no-dev --no-editable

# Copy application code
COPY app/ app/
COPY main.py .

# Add venv binaries to PATH
ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
