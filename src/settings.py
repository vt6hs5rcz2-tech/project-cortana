"""Environment settings for Project Cortana."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = field(repr=False)
    openai_model: str


def load_settings() -> Settings:
    """Load and validate Project Cortana settings."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to the private .env file."
        )

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
    )
