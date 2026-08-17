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

MISSING_API_KEY_MESSAGE = (
    "Cortana: OPENAI_API_KEY is missing. Add it to your .env file."
)
INVALID_TRANSCRIPTION_MODEL_MESSAGE = (
    "Cortana: CORTANA_TRANSCRIPTION_MODEL has an invalid value."
)
INVALID_TTS_MODEL_MESSAGE = "Cortana: CORTANA_TTS_MODEL has an invalid value."
INVALID_TTS_VOICE_MESSAGE = "Cortana: CORTANA_TTS_VOICE has an invalid value."
INVALID_REALTIME_MODEL_MESSAGE = (
    "Cortana: CORTANA_REALTIME_MODEL has an invalid value."
)
INVALID_REALTIME_VOICE_MESSAGE = (
    "Cortana: CORTANA_REALTIME_VOICE has an invalid value."
)
GENERIC_CONFIGURATION_MESSAGE = (
    "Cortana: Required configuration is invalid. Check your .env file."
)
INVALID_DATA_DIR_MESSAGE = (
    "Cortana: CORTANA_DATA_DIR is invalid or not writable."
)


class SettingsError(ValueError):
    """Bounded configuration error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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

    def has_api_key(self) -> bool:
        """Return True when a non-blank API key is present."""
        return bool(self.openai_api_key.strip())


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
        raise SettingsError(
            "missing_api_key",
            "OPENAI_API_KEY is missing. Add it to the private .env file.",
        )

    if transcription_model not in ALLOWED_TRANSCRIPTION_MODELS:
        raise SettingsError(
            "invalid_transcription_model",
            "CORTANA_TRANSCRIPTION_MODEL is not an allowed transcription model.",
        )
    if tts_model not in ALLOWED_TTS_MODELS:
        raise SettingsError(
            "invalid_tts_model",
            "CORTANA_TTS_MODEL is not an allowed text-to-speech model.",
        )
    if tts_voice not in ALLOWED_TTS_VOICES:
        raise SettingsError(
            "invalid_tts_voice",
            "CORTANA_TTS_VOICE is not an allowed text-to-speech voice.",
        )
    if realtime_model not in ALLOWED_REALTIME_MODELS:
        raise SettingsError(
            "invalid_realtime_model",
            "CORTANA_REALTIME_MODEL is not an allowed realtime voice model.",
        )
    if realtime_voice not in ALLOWED_REALTIME_VOICES:
        raise SettingsError(
            "invalid_realtime_voice",
            "CORTANA_REALTIME_VOICE is not an allowed realtime voice.",
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


def user_facing_settings_error(error: BaseException) -> str:
    """Map a settings failure to a bounded user-facing configuration message."""
    code = getattr(error, "code", "")
    mapped = {
        "missing_api_key": MISSING_API_KEY_MESSAGE,
        "invalid_transcription_model": INVALID_TRANSCRIPTION_MODEL_MESSAGE,
        "invalid_tts_model": INVALID_TTS_MODEL_MESSAGE,
        "invalid_tts_voice": INVALID_TTS_VOICE_MESSAGE,
        "invalid_realtime_model": INVALID_REALTIME_MODEL_MESSAGE,
        "invalid_realtime_voice": INVALID_REALTIME_VOICE_MESSAGE,
        "invalid_data_directory": INVALID_DATA_DIR_MESSAGE,
    }
    if code in mapped:
        return mapped[code]

    text = str(error)
    if "OPENAI_API_KEY" in text:
        return MISSING_API_KEY_MESSAGE
    if "CORTANA_TTS_VOICE" in text:
        return INVALID_TTS_VOICE_MESSAGE
    if "CORTANA_TTS_MODEL" in text:
        return INVALID_TTS_MODEL_MESSAGE
    if "CORTANA_TRANSCRIPTION_MODEL" in text:
        return INVALID_TRANSCRIPTION_MODEL_MESSAGE
    if "CORTANA_REALTIME_MODEL" in text:
        return INVALID_REALTIME_MODEL_MESSAGE
    if "CORTANA_REALTIME_VOICE" in text:
        return INVALID_REALTIME_VOICE_MESSAGE
    return GENERIC_CONFIGURATION_MESSAGE
