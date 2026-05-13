"""Centralized configuration via pydantic-settings.

Single source of truth for env vars. Import `settings` anywhere instead of reading
`os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── app
    app_env: Literal["local", "staging", "prod"] = "local"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # ─── llm
    openai_api_key: str = "sk-missing"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ─── alpha vantage
    alphavantage_api_key: str = "demo"
    alphavantage_rate_per_min: int = 5
    alphavantage_rate_per_day: int = 25

    # ─── storage
    database_url: str = "postgresql+asyncpg://finsight:finsight@localhost:5432/finsight"
    database_url_sync: str = "postgresql://finsight:finsight@localhost:5432/finsight"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "sec_filings"

    # ─── sec
    sec_user_agent: str = "FinSight Research contact@example.com"

    # ─── misc
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
