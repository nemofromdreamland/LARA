from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    groq = "groq"
    cerebras = "cerebras"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: LLMProvider = LLMProvider.groq
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    chroma_path: str = "/data/chroma"
    frontend_origin: str = "http://localhost:5173"
    session_ttl_seconds: int = 7200  # 2 hours
    expiry_interval_seconds: int = 600  # run eviction every 10 minutes


settings = Settings()
