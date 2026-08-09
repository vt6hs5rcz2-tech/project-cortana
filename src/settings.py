"""Environment settings for Project Cortana."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    ALLOWED_REALTIME_MODELS,
    ALLOWED_REALTIME_VOICES,
    ALLOWED_TRANSCRIPTION_MODELS,
    ALLOWED_TTS_MODELS,
    ALLOWED_TTS_VOICES,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = field(repr=False)
    openai_model: str
    google_oauth_client_file: Path | None = None
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    realtime_model: str = DEFAULT_REALTIME_MODEL
    realtime_voice: str = DEFAULT_REALTIME_VOICE


def load_settings() -> Settings:
    """Load and validate Project Cortana settings."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"
    oauth_client_raw = os.getenv("CORTANA_GOOGLE_OAUTH_CLIENT_FILE", "").strip()
    transcription_model = (
        os.getenv("CORTANA_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL).strip()
        or DEFAULT_TRANSCRIPTION_MODEL
    )
    tts_model = (
        os.getenv("CORTANA_TTS_MODEL", DEFAULT_TTS_MODEL).strip() or DEFAULT_TTS_MODEL
    )
    tts_voice = (
        os.getenv("CORTANA_TTS_VOICE", DEFAULT_TTS_VOICE).strip() or DEFAULT_TTS_VOICE
    )
    realtime_model = (
        os.getenv("CORTANA_REALTIME_MODEL", DEFAULT_REALTIME_MODEL).strip()
        or DEFAULT_REALTIME_MODEL
    )
    realtime_voice = (
        os.getenv("CORTANA_REALTIME_VOICE", DEFAULT_REALTIME_VOICE).strip()
        or DEFAULT_REALTIME_VOICE
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to the private .env file."
        )

    if transcription_model not in ALLOWED_TRANSCRIPTION_MODELS:
        raise ValueError(
            "CORTANA_TRANSCRIPTION_MODEL is not an allowed transcription model."
        )
    if tts_model not in ALLOWED_TTS_MODELS:
        raise ValueError("CORTANA_TTS_MODEL is not an allowed text-to-speech model.")
    if tts_voice not in ALLOWED_TTS_VOICES:
        raise ValueError("CORTANA_TTS_VOICE is not an allowed text-to-speech voice.")
    if realtime_model not in ALLOWED_REALTIME_MODELS:
        raise ValueError(
            "CORTANA_REALTIME_MODEL is not an allowed realtime voice model."
        )
    if realtime_voice not in ALLOWED_REALTIME_VOICES:
        raise ValueError(
            "CORTANA_REALTIME_VOICE is not an allowed realtime voice."
        )

    oauth_client_file: Path | None = None
    if oauth_client_raw:
        oauth_client_file = Path(oauth_client_raw)

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        google_oauth_client_file=oauth_client_file,
        transcription_model=transcription_model,
        tts_model=tts_model,
        tts_voice=tts_voice,
        realtime_model=realtime_model,
        realtime_voice=realtime_voice,
    )
