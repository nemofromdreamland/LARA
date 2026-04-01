import logging
from collections.abc import AsyncGenerator

import groq as groq_sdk
import httpx
from groq import AsyncGroq

from app.config import LLMProvider, settings
from app.services.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are LARA, a medical information assistant. "
    "Answer ONLY using the context provided below. "
    "The context includes the patient's prescription and official FDA drug "
    "leaflet sections. "
    "If the context does not contain the answer, respond: "
    "'This information is not available in the provided leaflets.' "
    "Never add information from general knowledge. "
    "Always cite the source "
    "(e.g. 'According to the Pregnancy section of the sertraline leaflet...')."
)

_MODEL_GROQ = "llama-3.3-70b-versatile"
_MODEL_CEREBRAS = "llama3.3-70b"

# Module-level breaker — shared across all requests in the process.
# Trips after 3 consecutive Groq failures; resets after 60 s cooldown.
_groq_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

# Errors that indicate a transient Groq problem worth falling back from.
_GROQ_TRANSIENT = (
    groq_sdk.RateLimitError,
    groq_sdk.InternalServerError,
    groq_sdk.APIConnectionError,
)


def _build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


async def generate(context: str, question: str) -> str:
    """Generate an answer from context + question.

    Routing logic:
    - LLM_PROVIDER=cerebras  → Cerebras only (no Groq attempt)
    - LLM_PROVIDER=groq (default):
        1. If circuit breaker is OPEN → skip directly to Cerebras fallback
        2. Try Groq
           - Success               → record success, return answer
           - Transient error (429 / 5xx / connection) → record failure,
             fall back to Cerebras
           - Other errors          → re-raise immediately
    """
    prompt = _build_prompt(context, question)

    if settings.llm_provider == LLMProvider.cerebras:
        return await _call_cerebras(prompt)

    # --- Groq path with circuit breaker ---
    if not _groq_breaker.allow_request():
        logger.warning("Groq circuit open — falling back to Cerebras")
        return await _call_cerebras(prompt)

    try:
        result = await _call_groq(prompt)
        _groq_breaker.record_success()
        return result
    except _GROQ_TRANSIENT as exc:
        _groq_breaker.record_failure()
        logger.warning("Groq transient error (%s) — falling back to Cerebras", exc)
        return await _call_cerebras(prompt)


async def generate_stream(context: str, question: str) -> AsyncGenerator[str, None]:
    """Stream answer tokens for *context* + *question*.

    Yields raw text chunks as they arrive.  Applies the same circuit-breaker
    routing as generate():
    - Cerebras provider or circuit OPEN → yields full response as one chunk.
    - Groq (default): streams tokens; on transient error falls back to
      Cerebras and yields the full response as one chunk.
    """
    prompt = _build_prompt(context, question)

    if settings.llm_provider == LLMProvider.cerebras:
        yield await _call_cerebras(prompt)
        return

    if not _groq_breaker.allow_request():
        logger.warning("Groq circuit open — streaming fallback to Cerebras")
        yield await _call_cerebras(prompt)
        return

    try:
        async for chunk in _stream_groq(prompt):
            yield chunk
        _groq_breaker.record_success()
    except _GROQ_TRANSIENT as exc:
        _groq_breaker.record_failure()
        logger.warning(
            "Groq transient error during stream (%s) — falling back to Cerebras", exc
        )
        yield await _call_cerebras(prompt)


async def _call_groq(prompt: str) -> str:
    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


async def _stream_groq(prompt: str) -> AsyncGenerator[str, None]:
    """Yield text tokens from Groq's streaming API."""
    client = AsyncGroq(api_key=settings.groq_api_key)
    stream = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def _call_cerebras(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.cerebras_api_key}"},
            json={
                "model": _MODEL_CEREBRAS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
