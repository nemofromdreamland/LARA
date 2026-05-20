from unittest.mock import AsyncMock, MagicMock, patch

import groq as groq_sdk
import httpx
import pytest

import app.services.llm_client as llm_module
from app.services.llm_client import (
    SYSTEM_PROMPT,
    _build_prompt,
    call_llm,
    generate,
    generate_stream,
)


@pytest.fixture(autouse=True)
def reset_groq_singleton():
    """Reset the module-level Groq singleton before each test.

    Without this, the first test to call _get_groq_client() caches a client,
    and subsequent tests that patch AsyncGroq never see their mock used.
    """
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
    mock_breaker.allow_request.return_value = True

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
    mock_breaker.allow_request.return_value = True

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
    mock_breaker.allow_request.return_value = False  # circuit is OPEN

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
    mock_breaker.allow_request.return_value = True

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
    mock_breaker.allow_request.return_value = True

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
    mock_breaker.allow_request.return_value = True

    mock_groq_cls.return_value.chat.completions.create = AsyncMock(
        return_value=_make_groq_stream_chunks("Hello", " world", "!")
    )

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Hello", " world", "!"]


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client._groq_breaker")
async def test_generate_stream_cerebras_yields_single_chunk(
    mock_breaker, mock_client_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.cerebras
    mock_settings.cerebras_api_key = "fake-cerebras-key"

    mock_http = _mock_cerebras_client("Full answer.")
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    tokens = []
    async for t in generate_stream("ctx", "q"):
        tokens.append(t)

    assert tokens == ["Full answer."]


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.httpx.AsyncClient")
@patch("app.services.llm_client.AsyncGroq")
@patch("app.services.llm_client._groq_breaker")
async def test_generate_stream_falls_back_on_rate_limit(
    mock_breaker, mock_groq_cls, mock_client_cls, mock_settings
):
    from app.config import LLMProvider

    mock_settings.llm_provider = LLMProvider.groq
    mock_settings.groq_api_key = "fake-key"
    mock_settings.cerebras_api_key = "fake-cerebras-key"
    mock_breaker.allow_request.return_value = True

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
    mock_breaker.record_failure.assert_called_once()


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
