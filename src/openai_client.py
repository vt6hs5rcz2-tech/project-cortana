"""OpenAI client creation for Project Cortana."""

from openai import OpenAI

from src.settings import Settings


def create_openai_client(settings: Settings) -> OpenAI:
    """Create an authenticated OpenAI client from validated settings."""
    return OpenAI(api_key=settings.openai_api_key)
