"""Narrow accidental-turn filter for realtime multimodal playback.

Suppresses only low-information fragments that arrive as an early barge-in
during assistant playback. Legitimate short barge-ins stay allowed.
"""

from __future__ import annotations

import re

from src.conversation_intelligence import normalize_user_utterance
from src.realtime_stop_command import is_explicit_stop_command

# Must remain answerable even as one-word barge-ins.
PROTECTED_SHORT_TURNS = frozenset(
    {
        "stop",
        "no",
        "wait",
        "ram",
        "tuesday",
        "yes",
        "yeah",
        "yep",
        "nah",
        "ok",
        "okay",
        "what",
        "why",
        "help",
        "cancel",
        "quit",
        "hello",
        "hey",
        "cortana",
        "monday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "go",
        "back",
        "continue",
        "repeat",
    }
)

_MAX_PLAYBACK_FRAGMENT_SECONDS = 1.5
_MAX_FRAGMENT_CHARS = 12
_MAX_FRAGMENT_WORDS = 2
_INCOMPLETE_COPULA_FORMS = frozenset({"it", "its", "it s", "it is"})


def is_accidental_playback_turn(
    transcript: str,
    *,
    barged_during_playback: bool,
    seconds_after_playback_start: float | None,
) -> bool:
    """Return True only when multiple accidental-turn signals agree."""
    if not barged_during_playback:
        return False
    if seconds_after_playback_start is None:
        return False
    if seconds_after_playback_start < 0:
        return False
    if seconds_after_playback_start > _MAX_PLAYBACK_FRAGMENT_SECONDS:
        return False
    if is_explicit_stop_command(transcript):
        return False
    return is_low_information_fragment(transcript)


def is_incomplete_copula_fragment(transcript: str) -> bool:
    """Return True for truncated copula-only speech such as ``It's...``.

    Does not match substantive short turns like ``It's Tuesday.`` or
    ``It's broken.``
    """
    normalized = normalize_user_utterance(transcript)
    stripped = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped in _INCOMPLETE_COPULA_FORMS


def is_low_information_fragment(transcript: str) -> bool:
    """Return True for tiny noise-like fragments, not real short commands."""
    if is_incomplete_copula_fragment(transcript):
        return True
    normalized = normalize_user_utterance(transcript)
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return True
    if normalized in PROTECTED_SHORT_TURNS:
        return False
    words = normalized.split()
    if any(word in PROTECTED_SHORT_TURNS for word in words):
        return False
    if len(words) > _MAX_FRAGMENT_WORDS:
        return False
    if len(normalized) > _MAX_FRAGMENT_CHARS:
        return False
    if not any(char.isascii() and char.isalpha() for char in normalized):
        return True
    if len(words) == 1:
        return words[0] not in PROTECTED_SHORT_TURNS
    return all(len(word) <= 4 for word in words)
