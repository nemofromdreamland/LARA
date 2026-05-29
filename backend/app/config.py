from enum import Enum

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    groq = "groq"
    cerebras = "cerebras"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: LLMProvider = LLMProvider.groq
    lara_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    frontend_origin: str = "http://localhost:5173"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = Field(default=7200, ge=300)  # min 5 min, default 2 hours
    max_context_chars: int = 32_000
    retrieval_distance_threshold: float = Field(default=0.65, ge=0.0, le=2.0)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    dailymed_cache_ttl_seconds: int = 86_400
    upload_rate_limit: str = "5/minute"
    chat_rate_limit: str = "20/minute"
    groq_timeout_seconds: float = 30.0
    cerebras_cb_failure_threshold: int = 5
    cerebras_cb_cooldown_seconds: float = 120.0
    thread_pool_workers: int = 8
    embed_pool_workers: int = Field(default=4, ge=1)
    cleanup_interval_seconds: int = Field(default=1800, ge=60)
    reranker_enabled: bool = True

    @model_validator(mode="after")
    def _check_api_key(self) -> "Settings":
        if self.llm_provider == LLMProvider.groq and not self.groq_api_key:
            raise ValueError("groq_api_key must be set when llm_provider is 'groq'.")
        if self.llm_provider == LLMProvider.cerebras and not self.cerebras_api_key:
            raise ValueError(
                "cerebras_api_key must be set when llm_provider is 'cerebras'."
            )
        if not self.lara_api_key:
            raise ValueError("lara_api_key must be set.")
        return self

    @model_validator(mode="after")
    def _check_embed_pool_workers(self) -> "Settings":
        if self.embed_pool_workers > self.thread_pool_workers:
            raise ValueError(
                f"embed_pool_workers ({self.embed_pool_workers}) must be <= "
                f"thread_pool_workers ({self.thread_pool_workers})."
            )
        return self


settings = Settings()
