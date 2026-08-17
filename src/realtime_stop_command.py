"""Deterministic explicit stop-command detection for realtime sessions."""

from __future__ import annotations

import re

from src.conversation_intelligence import normalize_user_utterance

_STOP_COMMANDS = frozenset(
    {
        "stop",
        "stop talking",
        "stop speaking",
        "be quiet",
        "quiet",
        "cancel that",
        "never mind",
        "stop talking cortana",
        "cortana stop",
        "cortana stop talking",
        "cortana be quiet",
    }
)

_NEGATED_PREFIXES = (
    "don't ",
    "do not ",
    "dont ",
    "please don't ",
    "please do not ",
)


def is_explicit_stop_command(transcript: str) -> bool:
    """Return True for standalone stop/silence commands only.

    Longer phrases that merely contain the word stop are not control intent.
    """
    normalized = normalize_user_utterance(transcript)
    normalized = re.sub(r"[^\w\s']+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in _NEGATED_PREFIXES):
        return False
    if normalized in _STOP_COMMANDS:
        return True
    words = normalized.split()
    if len(words) <= 4 and words[0] == "cortana" and " ".join(words[1:]) in _STOP_COMMANDS:
        return True
    return False
