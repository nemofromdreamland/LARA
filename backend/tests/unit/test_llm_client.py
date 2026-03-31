from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm_client import SYSTEM_PROMPT, _build_prompt, generate


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
    assert "cite the section" in SYSTEM_PROMPT


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
async def test_generate_calls_groq_by_default(mock_groq_cls, mock_settings):
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "fake-key"

    mock_choice = MagicMock()
    mock_choice.message.content = "Groq answer."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_groq_cls.return_value = mock_client

    result = await generate("some context", "some question")
    assert result == "Groq answer."
    mock_client.chat.completions.create.assert_called_once()


@patch("app.services.llm_client.settings")
@patch("app.services.llm_client.AsyncGroq")
async def test_generate_passes_system_prompt(mock_groq_cls, mock_settings):
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "fake-key"

    mock_choice = MagicMock()
    mock_choice.message.content = "answer"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
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

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Cerebras answer."}}]
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await generate("context", "question")
    assert result == "Cerebras answer."
