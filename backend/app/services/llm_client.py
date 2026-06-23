import logging
import re
from collections.abc import AsyncGenerator

import groq as groq_sdk
import httpx
from groq import AsyncGroq
from prometheus_client import Counter
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import LLMProvider, settings
from app.services.circuit_breaker import RedisCircuitBreaker
from app.utils import get_request_id

logger = logging.getLogger(__name__)


class ServiceUnavailableError(Exception):
    """Raised when all LLM providers have open circuit breakers."""


class _GroqEmptyResponseError(Exception):
    """Groq returned empty/blank content — treat as transient to trigger fallback."""


# Sentinel yielded by generate_stream when partial tokens must be discarded:
# Groq died mid-stream and the fallback provider regenerates from scratch, so
# the consumer must drop everything received before this marker. Follows the
# same in-band convention as rag_pipeline's [SOURCES]/[DONE] sentinels.
STREAM_RESET = "[RESET]"


_LLM_CALLS = Counter(
    "lara_llm_provider_calls_total",
    "LLM provider call outcomes",
    ["provider", "outcome"],
)


# MULTILINE so $ anchors to line-end, not string-end; handles one or more
# trailing blank lines that LLMs frequently append after the CITED: footer.
_CITED_RE = re.compile(r"\nCITED:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def strip_cited_line(text: str) -> tuple[str, list[tuple[str, str]] | None]:
    """Remove the trailing CITED: line from LLM output and parse it.

    Returns (clean_text, cited) where cited distinguishes three cases so callers
    can tell "the model cited nothing" from "there was no usable footer":
      - ``None`` — no CITED footer, or a footer with no parseable drug/section
        pair (format drift). Caller should fall back to showing all sources.
      - ``[]`` — the footer is exactly ``CITED: none``: the model cited nothing,
        so zero sources should be shown.
      - ``[(drug, section), ...]`` — the parsed citations.
    """
    m = _CITED_RE.search(text)
    if not m:
        return text, None

    cited_raw = m.group(1).strip()
    clean_text = text[: m.start()].rstrip()

    if cited_raw.lower() == "none":
        return clean_text, []

    pairs: list[tuple[str, str]] = []
    for entry in cited_raw.split(","):
        entry = entry.strip()
        if "/" in entry:
            drug, _, section = entry.partition("/")
            pairs.append((drug.strip().lower(), section.strip().lower()))
    # Footer present but unparseable (no drug/section) → treat as "no footer".
    return clean_text, pairs or None


SYSTEM_PROMPT = (
    "You are LARA, a medical information assistant. "
    "Answer ONLY using the context provided below. "
    "The context includes the patient's prescription and official FDA drug "
    "leaflet sections. Each section is labelled as [drug_name — section_name]. "
    "If the context does not contain the answer, respond: "
    "'This information is not available in the provided leaflets.' "
    "Never add information from general knowledge. "
    "Always cite the source inline "
    "(e.g. 'According to the warnings section of the metformin leaflet...'). "
    "After your answer, on a new line, write exactly: "
    "CITED: drug1/section1, drug2/section2 "
    "listing only the [drug — section] labels you actually drew on. "
    "If you drew on none, write: CITED: none"
)

_MODEL_GROQ = "llama-3.3-70b-versatile"
_MODEL_CEREBRAS = "llama3.3-70b"

# Redis-backed breakers — state is shared across all uvicorn workers.
# Groq: trips after 3 consecutive failures; resets after 60 s cooldown.
_groq_breaker = RedisCircuitBreaker("groq", failure_threshold=3, cooldown_seconds=60.0)
# Cerebras fallback: trips after configurable threshold; longer cooldown.
_cerebras_breaker = RedisCircuitBreaker(
    "cerebras",
    failure_threshold=settings.cerebras_cb_failure_threshold,
    cooldown_seconds=settings.cerebras_cb_cooldown_seconds,
)

# Singleton Groq client — reuses the underlying httpx connection pool across requests.
_groq_client: AsyncGroq | None = None

# Singleton httpx client for Cerebras — avoids per-request TCP+TLS handshake.
_cerebras_client: httpx.AsyncClient | None = None


async def init_cerebras_client() -> None:
    """Create the shared Cerebras httpx client. Call from app lifespan startup."""
    global _cerebras_client
    _cerebras_client = httpx.AsyncClient(timeout=30.0)


async def close_cerebras_client() -> None:
    """Close the shared Cerebras httpx client. Call from app lifespan teardown."""
    global _cerebras_client
    if _cerebras_client is not None:
        await _cerebras_client.aclose()
        _cerebras_client = None


def _get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.groq_timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
        )
    return _groq_client


# Errors that indicate a transient Groq problem worth falling back from.
_GROQ_TRANSIENT = (
    groq_sdk.RateLimitError,
    groq_sdk.InternalServerError,
    groq_sdk.APIConnectionError,
    _GroqEmptyResponseError,
)


def _build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


def _build_messages(
    system_prompt: str,
    history: list[dict],
    current_prompt: str,
) -> list[dict]:
    """Assemble the full messages list: system → history turns → current user turn."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": current_prompt})
    return messages


async def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    history: list[dict] | None = None,
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
    hist = history or []

    if settings.llm_provider == LLMProvider.cerebras:
        return await _call_cerebras(user_message, system_prompt, temperature, hist)

    if not await _groq_breaker.allow_request():
        logger.info(
            "LLM fallback activated — Groq circuit open",
            extra={"request_id": rid},
        )
        return await _call_cerebras(user_message, system_prompt, temperature, hist)

    try:
        result = await _call_groq(user_message, system_prompt, temperature, hist)
        await _groq_breaker.record_success()
        _LLM_CALLS.labels(provider="groq", outcome="success").inc()
        return result
    except _GROQ_TRANSIENT as exc:
        await _groq_breaker.record_failure()
        _LLM_CALLS.labels(provider="groq", outcome="fallback").inc()
        logger.info(
            "LLM fallback activated — Groq transient error: %s",
            exc,
            extra={"request_id": rid},
        )
        return await _call_cerebras(user_message, system_prompt, temperature, hist)


async def generate(
    context: str, question: str, history: list[dict] | None = None
) -> str:
    """Generate a RAG answer grounded in *context* for *question*."""
    return await call_llm(
        SYSTEM_PROMPT, _build_prompt(context, question), history=history
    )


async def generate_stream(
    context: str, question: str, history: list[dict] | None = None
) -> AsyncGenerator[str, None]:
    """Stream answer tokens for *context* + *question*.

    Applies the same circuit-breaker routing as generate():
    - Groq (default): streams tokens from Groq's native streaming API; on
      transient error falls back to Cerebras and yields the full response as
      one chunk. If Groq dies *after* yielding tokens, a STREAM_RESET sentinel
      is yielded first so the consumer discards the partial Groq text instead
      of displaying it concatenated with the full fallback answer.
    - Cerebras provider or Groq circuit OPEN: yields the full response as one
      chunk (Cerebras has no native streaming API).
    """
    rid = get_request_id()
    prompt = _build_prompt(context, question)
    hist = history or []

    if settings.llm_provider == LLMProvider.cerebras:
        async for chunk in _stream_cerebras(prompt, history=hist):
            yield chunk
        return

    if not await _groq_breaker.allow_request():
        logger.info(
            "LLM fallback activated — Groq circuit open (stream)",
            extra={"request_id": rid},
        )
        async for chunk in _stream_cerebras(prompt, history=hist):
            yield chunk
        return

    yielded_any = False
    try:
        async for chunk in _stream_groq(prompt, history=hist):
            yielded_any = True
            yield chunk
        await _groq_breaker.record_success()
    except _GROQ_TRANSIENT as exc:
        await _groq_breaker.record_failure()
        logger.info(
            "LLM fallback activated — Groq transient error during stream: %s",
            exc,
            extra={"request_id": rid},
        )
        if yielded_any:
            yield STREAM_RESET
        async for chunk in _stream_cerebras(prompt, history=hist):
            yield chunk


async def _call_groq(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.0,
    history: list[dict] | None = None,
) -> str:
    client = _get_groq_client()
    response = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=_build_messages(system_prompt, history or [], prompt),
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise _GroqEmptyResponseError("Groq returned empty content")
    return content


async def _stream_groq(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield text tokens from Groq's streaming API."""
    client = _get_groq_client()
    stream = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=_build_messages(system_prompt, history or [], prompt),
        temperature=0.0,
        stream=True,
    )
    saw_content = False
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            saw_content = True
            yield content
    if not saw_content:
        # An all-empty stream is a filtered/degenerate completion; raise so the
        # caller falls back to Cerebras (no tokens yielded → seamless failover).
        raise _GroqEmptyResponseError("Groq stream produced no content")


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
async def _call_cerebras_http(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.0,
    history: list[dict] | None = None,
) -> str:
    """Raw Cerebras HTTP call with tenacity retries. Use _call_cerebras instead."""
    payload = {
        "model": _MODEL_CEREBRAS,
        "messages": _build_messages(system_prompt, history or [], prompt),
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {settings.cerebras_api_key}"}
    url = "https://api.cerebras.ai/v1/chat/completions"

    if _cerebras_client is not None:
        response = await _cerebras_client.post(url, headers=headers, json=payload)
    else:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def _call_cerebras(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.0,
    history: list[dict] | None = None,
) -> str:
    """Cerebras call with CB guard. Raises ServiceUnavailableError if CB open."""
    if not await _cerebras_breaker.allow_request():
        _LLM_CALLS.labels(provider="cerebras", outcome="failure").inc()
        raise ServiceUnavailableError("Cerebras circuit breaker is open")
    try:
        result = await _call_cerebras_http(prompt, system_prompt, temperature, history)
        await _cerebras_breaker.record_success()
        _LLM_CALLS.labels(provider="cerebras", outcome="success").inc()
        return result
    except Exception as exc:
        if _is_cerebras_transient(exc):
            await _cerebras_breaker.record_failure()
            _LLM_CALLS.labels(provider="cerebras", outcome="failure").inc()
        raise


async def _stream_cerebras(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.0,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Cerebras fallback: fetches the full response and yields it as a single chunk.

    Cerebras has no native streaming API; the previous word-by-word sleep loop added
    ~0.02 s × word_count of artificial latency with no client benefit.
    The SSE layer in the route handler emits one token event for the whole text.
    """
    text = await _call_cerebras(prompt, system_prompt, temperature, history)
    yield text
