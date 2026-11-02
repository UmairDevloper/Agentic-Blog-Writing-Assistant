"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from the environment and project .env file."""

    model_config = SettingsConfigDict(
        env_prefix="SEO_BLOG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    writing_mode: Literal["batched", "per_section"] = "batched"
    max_words_per_call: int = Field(default=1500, ge=300, le=4000)
    max_images: int = Field(default=3, ge=0, le=4)
    image_mode: Literal["placeholder", "generate", "off"] = "placeholder"
    text_model_reasoning_effort: Literal["low", "medium", "high"] | None = "low"

    text_model_provider: Literal["groq"] | None = None
    text_model_name: str | None = None
    text_model_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SEO_BLOG_GROQ_API_KEY",
            "GROQ_API_KEY",
            "SEO_BLOG_TEXT_MODEL_API_KEY",
        ),
    )
    text_model_timeout_seconds: int = Field(default=120, ge=1, le=600)
    text_model_max_concurrency: int = Field(default=2, ge=1, le=10)
    text_model_max_attempts: int = Field(default=4, ge=1, le=8)
    text_model_max_completion_tokens: int = Field(default=4096, ge=256, le=32768)
    text_model_requests_per_second: float | None = Field(default=None, gt=0)

    search_provider: Literal["tavily"] | None = None
    search_api_key: SecretStr | None = None
    search_timeout_seconds: int = Field(default=30, ge=1, le=120)
    search_max_results: int = Field(default=5, ge=1, le=20)
    search_max_attempts: int = Field(default=3, ge=1, le=5)
    search_queries_per_article: int = Field(default=2, ge=1, le=3)

    image_provider: Literal["huggingface"] | None = None
    image_model_name: str | None = None
    image_inference_provider: str = "auto"
    image_num_inference_steps: int | None = Field(default=4, ge=1, le=50)
    huggingface_token: SecretStr | None = None
    image_timeout_seconds: int = Field(default=120, ge=1, le=600)

    @field_validator(
        "text_model_provider", "search_provider", "image_provider", mode="before"
    )
    @classmethod
    def normalize_provider_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for the running application."""
    return Settings()
