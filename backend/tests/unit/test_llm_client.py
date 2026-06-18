from unittest.mock import AsyncMock, MagicMock, patch

import groq as groq_sdk
import httpx
import pytest

import app.services.llm_client as llm_module
from app.services.llm_client import (
    STREAM_RESET,
    SYSTEM_PROMPT,
    ServiceUnavailableError,
    _build_prompt,
    _call_cerebras,
    call_llm,
    generate,
    generate_stream,
    strip_cited_line,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before each test."""
    llm_module._groq_client = None
    yield
    llm_module._groq_client = None


def test_build_prompt_contains_context_and_question():
    prompt = _build_prompt("Some context.", "What is the dose?")
    assert "Some context." in prompt
    assert "What is the dose?" in prompt


def test_build_prompt_structure():
    prompt = _build_prompt("ctx", "q")
    assert prompt.index("ctx") < prompt.index("q")


def test_system_prompt_enforces_grounding():
    assert "ONLY" in SYSTEM_PROMPT
    assert "not available in the provided leaflets" in SYSTEM_PROMPT
    assert "cite the source" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# strip_cited_line
# ---------------------------------------------------------------------------


def test_strip_cited_line_parses_pairs():
    text = "Some answer.\nCITED: metformin/warnings, aspirin/dosage"
    clean, pairs = strip_cited_line(text)
    assert clean == "Some answer."
    assert ("metformin", "warnings") in pairs
    assert ("aspirin", "dosage") in pairs


def test_strip_cited_line_none_returns_empty_pairs():
    text = "Some answer.\nCITED: none"
    clean, pairs = strip_cited_line(text)
    assert clean == "Some answer."
    assert pairs == []


def test_strip_cited_line_no_cited_returns_original():
    text = "Some answer with no footer."
    clean, pairs = strip_cited_line(text)
    assert clean == text
    assert pairs == []


def test_strip_cited_line_trailing_blank_lines():
    """Trailing newlines after CITED: must not prevent parsing (MULTILINE fix)."""
    text = "Some answer.\nCITED: metformin/warnings\n\n"
    clean, pairs = strip_cited_line(text)
    assert ("metformin", "warnings") in pairs
    assert clean == "Some answer."


def test_strip_cited_line_single_trailing_newline():
    text = "Answer.\nCITED: drugA/dosage\n"
    clean, pairs = strip_cited_line(text)
    assert ("druga", "dosage") in pairs


def test_strip_cited_line_lowercases_pairs():
    text = "Answer.\nCITED: Metformin/Warnings"
    _, pairs = strip_cited_line(text)
    assert pairs == [("metformin", "warnings")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_groq_client(content: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    return mock_client


def _mock_cerebras_client(content: str) -> AsyncMock:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    return mock_http


# ---------------------------------------------------------------------------
# Normal routing
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
async def test_generate_calls_groq_by_default(mock_groq_cls, mock_settings):
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "fake-key"

    mock_client = _mock_groq_client("Groq answer.")
    mock_groq_cls.return_value = mock_client

    result = await generate("some context", "some question")
    assert result == "Groq answer."
    mock_client.chat.completions.create.assert_called_once()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
async def test_generate_passes_system_prompt(mock_groq_cls, mock_settings):
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "fake-key"

    mock_client = _mock_groq_client("answer")
    mock_groq_cls.return_value = mock_client

    await generate("ctx", "q")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
async def test_generate_calls_cerebras_when_configured(mock_client_cls, mock_settings):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.cerebras
    mock_settings.cerebras_api_key = "fake-cerebras-key"

    mock_http = _mock_cerebras_client("Cerebras answer.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("context", "question")
    assert result == "Cerebras answer."


# ---------------------------------------------------------------------------
# Automatic fallback: Groq transient error → Cerebras
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_falls_back_to_cerebras_on_rate_limit(
    mock_breaker, mock_groq_cls, mock_client_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_breaker.allow_request = AsyncMock(return_value=True)
    mock_breaker.record_failure = AsyncMock()
    mock_breaker.record_success = AsyncMock()

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        side_effect=groq_sdk.RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body={}
        )
    )

    mock_http = _mock_cerebras_client("Cerebras fallback.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("ctx", "q")
    assert result == "Cerebras fallback."
    mock_breaker.record_failure.assert_called_once()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_falls_back_to_cerebras_on_internal_server_error(
    mock_breaker, mock_groq_cls, mock_client_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_breaker.allow_request = AsyncMock(return_value=True)
    mock_breaker.record_failure = AsyncMock()
    mock_breaker.record_success = AsyncMock()

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        side_effect=groq_sdk.InternalServerError(
            "server error", response=MagicMock(status_code=500), body={}
        )
    )

    mock_http = _mock_cerebras_client("Cerebras fallback.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("ctx", "q")
    assert result == "Cerebras fallback."
    mock_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# Empty/blank Groq completion → transient → Cerebras fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_content", [None, "", "   "])
@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_empty_groq_content_falls_back_to_cerebras(
    mock_breaker, mock_groq_cls, mock_client_cls, mock_settings, empty_content
):
    """A None/blank Groq completion is treated as transient, routing to Cerebras."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_breaker.allow_request = AsyncMock(return_value=True)
    mock_breaker.record_failure = AsyncMock()
    mock_breaker.record_success = AsyncMock()

    mock_groq_cls.return_value = _mock_groq_client(empty_content)

    mock_http = _mock_cerebras_client("Cerebras fallback.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("ctx", "q")
    assert result == "Cerebras fallback."
    mock_breaker.record_failure.assert_called_once()
    mock_breaker.record_success.assert_not_called()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_groq_stream_no_content_falls_back_to_cerebras(
    mock_groq_breaker, mock_groq_cls, mock_client_cls, mock_cerebras_cb, mock_settings
):
    """A Groq stream that yields only empty deltas falls back to Cerebras with no
    STREAM_RESET (nothing usable was yielded, so the failover is seamless)."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_groq_breaker.allow_request = AsyncMock(return_value=True)
    mock_groq_breaker.record_failure = AsyncMock()
    mock_groq_breaker.record_success = AsyncMock()
    mock_cerebras_cb.allow_request = AsyncMock(return_value=True)
    mock_cerebras_cb.record_success = AsyncMock()
    mock_cerebras_cb.record_failure = AsyncMock()

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        return_value=_make_groq_stream_chunks(None, "", None)
    )

    mock_http = _mock_cerebras_client("Cerebras fallback.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Cerebras fallback."]
    assert STREAM_RESET not in tokens
    mock_groq_breaker.record_failure.assert_called_once()
    mock_groq_breaker.record_success.assert_not_called()


# ---------------------------------------------------------------------------
# Circuit breaker open → skip Groq entirely
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_skips_groq_when_circuit_open(
    mock_breaker, mock_groq_cls, mock_client_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_breaker.allow_request = AsyncMock(return_value=False)  # circuit is OPEN

    mock_http = _mock_cerebras_client("Cerebras (circuit open).")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("ctx", "q")
    assert result == "Cerebras (circuit open)."
    mock_groq_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Non-transient Groq errors are re-raised (not swallowed)
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_non_transient_groq_error_raises(
    mock_breaker, mock_groq_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_breaker.allow_request = AsyncMock(return_value=True)
    mock_breaker.record_failure = AsyncMock()
    mock_breaker.record_success = AsyncMock()

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        side_effect=groq_sdk.AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body={}
        )
    )

    with pytest.raises(groq_sdk.AuthenticationError):
        await generate("ctx", "q")

    mock_breaker.record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Success records on the breaker
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_groq_success_records_on_breaker(
    mock_breaker, mock_groq_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_breaker.allow_request = AsyncMock(return_value=True)
    mock_breaker.record_success = AsyncMock()

    mock_client = _mock_groq_client("ok")
    mock_groq_cls.return_value = mock_client

    await generate("ctx", "q")
    mock_breaker.record_success.assert_called_once()


# ---------------------------------------------------------------------------
# generate_stream
# ---------------------------------------------------------------------------


def _make_groq_stream_chunks(*tokens: str):
    """Build async-iterable mock chunks for Groq streaming API."""

    async def _aiter():
        for t in tokens:
            chunk = MagicMock()
            chunk.choices[0].delta.content = t
            yield chunk

    return _aiter()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_generate_stream_yields_tokens(
    mock_breaker, mock_groq_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_breaker.allow_request = AsyncMock(return_value=True)
    mock_breaker.record_success = AsyncMock()

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        return_value=_make_groq_stream_chunks("Hello", " world", "!")
    )

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Hello", " world", "!"]


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client.httpx.AsyncClient")
async def test_generate_stream_cerebras_yields_single_chunk(
    mock_client_cls, mock_cb, mock_settings
):
    """Cerebras yields the full response as one chunk (no native streaming API)."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.cerebras
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_cb.allow_request = AsyncMock(return_value=True)
    mock_cb.record_success = AsyncMock()
    mock_cb.record_failure = AsyncMock()

    mock_http = _mock_cerebras_client("Full answer.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Full answer."]


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_generate_stream_falls_back_on_rate_limit(
    mock_groq_breaker, mock_groq_cls, mock_client_cls, mock_cerebras_cb, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_groq_breaker.allow_request = AsyncMock(return_value=True)
    mock_groq_breaker.record_failure = AsyncMock()
    mock_groq_breaker.record_success = AsyncMock()
    mock_cerebras_cb.allow_request = AsyncMock(return_value=True)
    mock_cerebras_cb.record_success = AsyncMock()
    mock_cerebras_cb.record_failure = AsyncMock()

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        side_effect=groq_sdk.RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body={}
        )
    )

    mock_http = _mock_cerebras_client("Cerebras fallback.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Cerebras fallback."]
    mock_groq_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# Cerebras retry on transient errors
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
async def test_cerebras_retries_on_503_then_succeeds(mock_client_cls, mock_settings):
    """_call_cerebras retries transient 5xx and returns on eventual success."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.cerebras
    mock_settings.cerebras_api_key = "fake-cerebras-key"

    # First call: 503 error response
    fail_response = MagicMock()
    fail_response.status_code = 503
    fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable", request=MagicMock(), response=fail_response
    )

    # Second call: success
    ok_response = MagicMock()
    ok_response.raise_for_status = MagicMock()
    ok_response.json.return_value = {
        "choices": [{"message": {"content": "Retry succeeded."}}]
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=[fail_response, ok_response])
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("ctx", "q")
    assert result == "Retry succeeded."
    assert mock_http.post.call_count == 2


# ---------------------------------------------------------------------------
# call_llm — generic communication layer
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
async def test_call_llm_uses_given_system_prompt(mock_groq_cls, mock_settings):
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "fake-key"

    mock_client = _mock_groq_client("extracted result")
    mock_groq_cls.return_value = mock_client

    custom_prompt = "You are an extraction tool."
    result = await call_llm(custom_prompt, "some user text")

    assert result == "extracted result"
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == custom_prompt
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "some user text"


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
async def test_call_llm_passes_temperature(mock_groq_cls, mock_settings):
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "fake-key"

    mock_client = _mock_groq_client("ok")
    mock_groq_cls.return_value = mock_client

    await call_llm("sys", "user msg", temperature=0.7)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Cerebras circuit breaker
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client._groq_breaker")
async def test_both_circuit_breakers_open_raises_service_unavailable(
    mock_groq_breaker, mock_cerebras_cb, mock_settings
):
    """When both CBs are open, ServiceUnavailableError is raised immediately."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_groq_breaker.allow_request = AsyncMock(return_value=False)
    mock_cerebras_cb.allow_request = AsyncMock(return_value=False)

    with pytest.raises(ServiceUnavailableError):
        await generate("ctx", "q")


@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client._call_cerebras_http")
async def test_cerebras_cb_records_failure_after_transient_error(mock_http, mock_cb):
    # Verify _call_cerebras records a CB failure when _call_cerebras_http throws.
    mock_cb.allow_request = AsyncMock(return_value=True)
    mock_cb.record_failure = AsyncMock()
    mock_cb.record_success = AsyncMock()

    fail_response = MagicMock()
    fail_response.status_code = 503
    mock_http.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=fail_response
    )

    with pytest.raises(httpx.HTTPStatusError):
        await _call_cerebras("some prompt")

    mock_cb.record_failure.assert_called_once()
    mock_cb.record_success.assert_not_called()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client.httpx.AsyncClient")
async def test_stream_cerebras_yields_single_chunk(
    mock_client_cls, mock_cb, mock_settings
):
    """_stream_cerebras yields the full response as one chunk (no native streaming)."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.cerebras
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_cb.allow_request = AsyncMock(return_value=True)
    mock_cb.record_success = AsyncMock()
    mock_cb.record_failure = AsyncMock()

    mock_http = _mock_cerebras_client("Hello world from Cerebras")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    from app.services.llm_client import _stream_cerebras

    tokens = []
    async for chunk in _stream_cerebras("some prompt"):
        tokens.append(chunk)

    assert tokens == ["Hello world from Cerebras"]


# ---------------------------------------------------------------------------
# generate_stream — Groq fails MID-stream (after yielding tokens)
# ---------------------------------------------------------------------------


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_generate_stream_midstream_failure_falls_back_after_partial_tokens(
    mock_groq_breaker, mock_groq_cls, mock_client_cls, mock_cerebras_cb, mock_settings
):
    """Groq dying after it already yielded tokens still falls back to Cerebras.

    Contract: a STREAM_RESET sentinel is yielded between the partial Groq
    tokens and the regenerated Cerebras answer, so the consumer knows to
    discard everything received before the marker.
    """
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_groq_breaker.allow_request = AsyncMock(return_value=True)
    mock_groq_breaker.record_failure = AsyncMock()
    mock_groq_breaker.record_success = AsyncMock()
    mock_cerebras_cb.allow_request = AsyncMock(return_value=True)
    mock_cerebras_cb.record_success = AsyncMock()
    mock_cerebras_cb.record_failure = AsyncMock()

    async def _dying_stream():
        for t in ("Partial ", "answer"):
            chunk = MagicMock()
            chunk.choices[0].delta.content = t
            yield chunk
        raise groq_sdk.InternalServerError(
            "boom", response=MagicMock(status_code=500), body={}
        )

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        return_value=_dying_stream()
    )

    mock_http = _mock_cerebras_client("Complete fallback answer.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == [
        "Partial ",
        "answer",
        STREAM_RESET,
        "Complete fallback answer.",
    ]
    mock_groq_breaker.record_failure.assert_called_once()
    mock_groq_breaker.record_success.assert_not_called()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client._cerebras_breaker")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_generate_stream_failure_before_first_token_emits_no_reset(
    mock_groq_breaker, mock_groq_cls, mock_client_cls, mock_cerebras_cb, mock_settings
):
    """If Groq dies before yielding anything, the failover is seamless —
    no STREAM_RESET sentinel, just the fallback answer."""
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_groq_breaker.allow_request = AsyncMock(return_value=True)
    mock_groq_breaker.record_failure = AsyncMock()
    mock_groq_breaker.record_success = AsyncMock()
    mock_cerebras_cb.allow_request = AsyncMock(return_value=True)
    mock_cerebras_cb.record_success = AsyncMock()
    mock_cerebras_cb.record_failure = AsyncMock()

    async def _dies_immediately():
        raise groq_sdk.InternalServerError(
            "boom", response=MagicMock(status_code=500), body={}
        )
        yield  # pragma: no cover — makes this an async generator

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        return_value=_dies_immediately()
    )

    mock_http = _mock_cerebras_client("Fallback answer.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Fallback answer."]
    mock_groq_breaker.record_failure.assert_called_once()
