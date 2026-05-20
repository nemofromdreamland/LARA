from enum import Enum

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    groq = "groq"
    cerebras = "cerebras"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: LLMProvider = LLMProvider.groq
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    chroma_path: str = "./data/chroma"
    frontend_origin: str = "http://localhost:5173"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = Field(default=7200, ge=300)  # min 5 min, default 2 hours
    max_context_chars: int = 12_000
    dailymed_cache_ttl_seconds: int = 86_400
    upload_rate_limit: str = "5/minute"
    chat_rate_limit: str = "20/minute"

    @model_validator(mode="after")
    def _check_api_key(self) -> "Settings":
        if self.llm_provider == LLMProvider.groq and not self.groq_api_key:
            raise ValueError("groq_api_key must be set when llm_provider is 'groq'.")
        if self.llm_provider == LLMProvider.cerebras and not self.cerebras_api_key:
            raise ValueError(
                "cerebras_api_key must be set when llm_provider is 'cerebras'."
            )
        return self


settings = Settings()
