import logging
from collections.abc import AsyncGenerator

import groq as groq_sdk
import httpx
from groq import AsyncGroq
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import LLMProvider, settings
from app.services.circuit_breaker import CircuitBreaker
from app.utils import get_request_id

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

# Singleton Groq client — reuses the underlying httpx connection pool across requests.
_groq_client: AsyncGroq | None = None


def _get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


# Errors that indicate a transient Groq problem worth falling back from.
_GROQ_TRANSIENT = (
    groq_sdk.RateLimitError,
    groq_sdk.InternalServerError,
    groq_sdk.APIConnectionError,
)


def _build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


async def call_llm(
    system_prompt: str, user_message: str, temperature: float = 0.0
) -> str:
    """Generic LLM call with provider routing and circuit breaker.

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
    rid = get_request_id()

    if settings.llm_provider == LLMProvider.cerebras:
        return await _call_cerebras(user_message, system_prompt, temperature)

    if not _groq_breaker.allow_request():
        logger.info(
            "LLM fallback activated — Groq circuit open",
            extra={"request_id": rid},
        )
        return await _call_cerebras(user_message, system_prompt, temperature)

    try:
        result = await _call_groq(user_message, system_prompt, temperature)
        _groq_breaker.record_success()
        return result
    except _GROQ_TRANSIENT as exc:
        _groq_breaker.record_failure()
        logger.info(
            "LLM fallback activated — Groq transient error: %s", exc,
            extra={"request_id": rid},
        )
        return await _call_cerebras(user_message, system_prompt, temperature)


async def generate(context: str, question: str) -> str:
    """Generate a RAG answer grounded in *context* for *question*."""
    return await call_llm(SYSTEM_PROMPT, _build_prompt(context, question))


async def generate_stream(context: str, question: str) -> AsyncGenerator[str, None]:
    """Stream answer tokens for *context* + *question*.

    Yields raw text chunks as they arrive.  Applies the same circuit-breaker
    routing as generate():
    - Cerebras provider or circuit OPEN → yields full response as one chunk.
    - Groq (default): streams tokens; on transient error falls back to
      Cerebras and yields the full response as one chunk.
    """
    rid = get_request_id()
    prompt = _build_prompt(context, question)

    if settings.llm_provider == LLMProvider.cerebras:
        yield await _call_cerebras(prompt)
        return

    if not _groq_breaker.allow_request():
        logger.info(
            "LLM fallback activated — Groq circuit open (stream)",
            extra={"request_id": rid},
        )
        yield await _call_cerebras(prompt)
        return

    try:
        async for chunk in _stream_groq(prompt):
            yield chunk
        _groq_breaker.record_success()
    except _GROQ_TRANSIENT as exc:
        _groq_breaker.record_failure()
        logger.info(
            "LLM fallback activated — Groq transient error during stream: %s", exc,
            extra={"request_id": rid},
        )
        yield await _call_cerebras(prompt)


async def _call_groq(
    prompt: str, system_prompt: str = SYSTEM_PROMPT, temperature: float = 0.0
) -> str:
    client = _get_groq_client()
    response = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


async def _stream_groq(
    prompt: str, system_prompt: str = SYSTEM_PROMPT
) -> AsyncGenerator[str, None]:
    """Yield text tokens from Groq's streaming API."""
    client = _get_groq_client()
    stream = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def _is_cerebras_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


@retry(
    retry=retry_if_exception(_is_cerebras_transient),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _call_cerebras(
    prompt: str, system_prompt: str = SYSTEM_PROMPT, temperature: float = 0.0
) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.cerebras_api_key}"},
            json={
                "model": _MODEL_CEREBRAS,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
