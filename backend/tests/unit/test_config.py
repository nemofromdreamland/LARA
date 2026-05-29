import pytest

from app.config import LLMProvider, Settings


def test_default_provider():
    s = Settings(groq_api_key="test-key")
    assert s.llm_provider == LLMProvider.groq


def test_llm_provider_enum_values():
    assert LLMProvider.groq == "groq"
    assert LLMProvider.cerebras == "cerebras"


def test_groq_without_key_raises():
    with pytest.raises(ValueError, match="groq_api_key"):
        Settings(llm_provider=LLMProvider.groq, groq_api_key="")


def test_ttl_below_minimum_raises():
    with pytest.raises(ValueError):
        Settings(groq_api_key="k", session_ttl_seconds=299)


def test_ttl_at_minimum_passes():
    s = Settings(groq_api_key="k", session_ttl_seconds=300)
    assert s.session_ttl_seconds == 300


def test_lara_api_key_empty_raises():
    with pytest.raises(ValueError, match="lara_api_key"):
        Settings(groq_api_key="k", lara_api_key="")
