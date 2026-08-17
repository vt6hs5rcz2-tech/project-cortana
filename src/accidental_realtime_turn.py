"""Narrow accidental-turn filter for realtime voice and multimodal.

Suppresses empty, punctuation-only, low-information, and conservative
self-echo transcripts. Legitimate short barge-ins stay allowed.
This is not acoustic echo cancellation and not speaker identification.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

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
_MAX_RECENT_ASSISTANT_UTTERANCES = 3
_ASSISTANT_ECHO_WINDOW_SECONDS = 5.0
_ECHO_TAIL_SECONDS = 1.5
_MIN_OVERLAP_CHARS = 12
_HIGH_TOKEN_JACCARD = 0.85


class RealtimeTurnRejectReason(str, Enum):
    """Bounded reasons a realtime transcript is not meaningful activity."""

    EMPTY = "empty"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    ACCIDENTAL_FRAGMENT = "accidental_fragment"
    SELF_ECHO = "self_echo"


@dataclass(frozen=True)
class RecentAssistantUtterance:
    """One short-lived assistant utterance used for conservative echo checks."""

    normalized: str
    tokens: tuple[str, ...]
    spoken_at_monotonic: float


def _alnum_normalized(transcript: str) -> str:
    normalized = normalize_user_utterance(transcript)
    stripped = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


def is_punctuation_only_transcript(transcript: str) -> bool:
    """Return True when the transcript has visible marks but no words."""
    stripped = transcript.strip()
    if not stripped:
        return False
    return not _alnum_normalized(stripped)


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
    stripped = _alnum_normalized(transcript)
    return stripped in _INCOMPLETE_COPULA_FORMS


def is_low_information_fragment(transcript: str) -> bool:
    """Return True for tiny noise-like fragments, not real short commands."""
    if is_incomplete_copula_fragment(transcript):
        return True
    normalized = _alnum_normalized(transcript)
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


def _prefix_or_suffix_overlap(left: str, right: str) -> bool:
    if len(left) < _MIN_OVERLAP_CHARS or len(right) < _MIN_OVERLAP_CHARS:
        return False
    return (
        left.startswith(right)
        or right.startswith(left)
        or left.endswith(right)
        or right.endswith(left)
    )


def _token_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def _playback_proximity(
    *,
    playback_active: bool,
    seconds_since_playback_end: float | None,
) -> bool:
    if playback_active:
        return True
    if seconds_since_playback_end is None:
        return False
    return 0.0 <= seconds_since_playback_end <= _ECHO_TAIL_SECONDS


def is_likely_self_echo(
    transcript: str,
    *,
    recent_assistant: Sequence[RecentAssistantUtterance],
    playback_active: bool,
    seconds_since_playback_end: float | None,
) -> bool:
    """Return True only when playback proximity and text overlap both agree.

    Exact repetition after playback has ended is not enough. The user may
    intentionally repeat Cortana. This is not full acoustic echo cancellation.
    """
    if is_explicit_stop_command(transcript):
        return False
    normalized = _alnum_normalized(transcript)
    if not normalized:
        return False
    if not _playback_proximity(
        playback_active=playback_active,
        seconds_since_playback_end=seconds_since_playback_end,
    ):
        return False
    user_tokens = tuple(normalized.split())
    for utterance in recent_assistant:
        text_match = False
        if normalized == utterance.normalized:
            text_match = True
        elif _prefix_or_suffix_overlap(normalized, utterance.normalized):
            text_match = True
        elif _token_jaccard(user_tokens, utterance.tokens) >= _HIGH_TOKEN_JACCARD:
            text_match = True
        if text_match:
            return True
    return False


def classify_realtime_user_transcript(
    transcript: str,
    *,
    barged_during_playback: bool = False,
    seconds_after_playback_start: float | None = None,
    playback_active: bool = False,
    seconds_since_playback_end: float | None = None,
    recent_assistant: Sequence[RecentAssistantUtterance] = (),
) -> RealtimeTurnRejectReason | None:
    """Return a reject reason, or None when the transcript is meaningful.

    Stop/control commands are never rejected here so the session can consume
    them before idle handling. Length alone is not a reject rule.
    """
    if is_explicit_stop_command(transcript):
        return None
    if transcript == "":
        return RealtimeTurnRejectReason.EMPTY
    if not transcript.strip():
        return RealtimeTurnRejectReason.WHITESPACE
    if is_punctuation_only_transcript(transcript):
        return RealtimeTurnRejectReason.PUNCTUATION
    if is_incomplete_copula_fragment(transcript):
        return RealtimeTurnRejectReason.ACCIDENTAL_FRAGMENT
    if is_accidental_playback_turn(
        transcript,
        barged_during_playback=barged_during_playback,
        seconds_after_playback_start=seconds_after_playback_start,
    ):
        return RealtimeTurnRejectReason.ACCIDENTAL_FRAGMENT
    if is_low_information_fragment(transcript):
        return RealtimeTurnRejectReason.ACCIDENTAL_FRAGMENT
    if is_likely_self_echo(
        transcript,
        recent_assistant=recent_assistant,
        playback_active=playback_active,
        seconds_since_playback_end=seconds_since_playback_end,
    ):
        return RealtimeTurnRejectReason.SELF_ECHO
    return None


class RecentAssistantSpeechBuffer:
    """Bounded recent assistant text for conservative self-echo checks."""

    def __init__(
        self,
        *,
        max_utterances: int = _MAX_RECENT_ASSISTANT_UTTERANCES,
        window_seconds: float = _ASSISTANT_ECHO_WINDOW_SECONDS,
    ) -> None:
        self._max_utterances = max_utterances
        self._window_seconds = window_seconds
        self._utterances: deque[RecentAssistantUtterance] = deque()

    def remember(self, text: str, *, now: float) -> None:
        normalized = _alnum_normalized(text)
        if not normalized:
            return
        self.prune(now)
        utterance = RecentAssistantUtterance(
            normalized=normalized,
            tokens=tuple(normalized.split()),
            spoken_at_monotonic=now,
        )
        self._utterances.append(utterance)
        while len(self._utterances) > self._max_utterances:
            self._utterances.popleft()

    def prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._utterances and self._utterances[0].spoken_at_monotonic < cutoff:
            self._utterances.popleft()

    def recent(self, now: float) -> tuple[RecentAssistantUtterance, ...]:
        self.prune(now)
        return tuple(self._utterances)

    def clear(self) -> None:
        self._utterances.clear()
