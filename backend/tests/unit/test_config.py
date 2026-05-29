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


def test_embed_pool_workers_default():
    s = Settings(groq_api_key="k")
    assert s.embed_pool_workers == 4


def test_embed_pool_workers_below_minimum_raises():
    with pytest.raises(ValueError):
        Settings(groq_api_key="k", embed_pool_workers=0)


def test_embed_pool_workers_exceeds_thread_pool_raises():
    with pytest.raises(ValueError, match="embed_pool_workers"):
        Settings(groq_api_key="k", thread_pool_workers=4, embed_pool_workers=8)


def test_embed_pool_workers_equal_to_thread_pool_ok():
    s = Settings(groq_api_key="k", thread_pool_workers=4, embed_pool_workers=4)
    assert s.embed_pool_workers == 4
