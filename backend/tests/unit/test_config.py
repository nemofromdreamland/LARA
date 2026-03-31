from app.config import LLMProvider, Settings


def test_default_provider():
    s = Settings()
    assert s.llm_provider == LLMProvider.groq


def test_llm_provider_enum_values():
    assert LLMProvider.groq == "groq"
    assert LLMProvider.cerebras == "cerebras"
