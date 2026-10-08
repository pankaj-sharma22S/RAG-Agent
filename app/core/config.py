"""Centralized configuration and environment settings for FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Info
    APP_NAME: str = "Knowledge AI Backend"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    API_V1_PREFIX: str = "/api/v1"

    # Database & Storage
    DATABASE_URL: str = Field(default="sqlite:///./data/dev.db")
    UPLOAD_DIR: str = Field(default=str(BASE_DIR / "app" / "data" / "uploads"))
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024  # 25 MB

    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Supabase & Authentication
    SUPABASE_URL: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL", ""))
    SUPABASE_KEY: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY", ""))
    SUPABASE_JWT_SECRET: str = Field(default_factory=lambda: os.getenv("SUPABASE_JWT_SECRET", ""))
    SUPABASE_JWT_ALGORITHMS: List[str] = ["HS256", "RS256", "ES256"]
    SUPABASE_JWT_AUDIENCE: str = Field(default="authenticated")
    DEV_AUTH_BYPASS: bool = Field(default=False)

    # LLM & Observability
    PRIMARY_PROVIDER: str = Field(default="gemini")
    DEFAULT_LLM_PROVIDER: str = Field(default="gemini")
    GEMINI_API_KEY: str = Field(default="")
    LANGSMITH_TRACING: bool = Field(default=False)

    # Rate Limiting & Timeouts
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60
    REQUEST_TIMEOUT_SECONDS: int = 120
    RATE_LIMIT_ENABLED: bool = Field(default=True)

    # Redis Connection & Semantic Response Cache
    REDIS_URL: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    REDIS_HOST: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    REDIS_PORT: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    REDIS_PASSWORD: str | None = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD", None))
    REDIS_DB: int = Field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    REDIS_CONNECT_TIMEOUT: float = Field(default=0.5)
    REDIS_SOCKET_TIMEOUT: float = Field(default=0.5)

    SEMANTIC_CACHE_ENABLED: bool = Field(default=True)
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.92)
    SEMANTIC_CACHE_TTL_SECONDS: int = Field(default=3600)


settings = Settings()

# Ensure critical directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "data", exist_ok=True)
os.makedirs(BASE_DIR / "app" / "data", exist_ok=True)
