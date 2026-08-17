"""Bounded idle timeout for explicit realtime voice and multimodal sessions.

Tracks time since the last accepted meaningful user interaction. Microphone
bytes, VAD speech_started, and rejected transcripts do not reset the timer.
"""

from __future__ import annotations

from collections.abc import Callable

from src.config import (
    REALTIME_IDLE_TIMEOUT_SECONDS,
    bounded_realtime_idle_timeout_seconds,
)

REALTIME_IDLE_TIMEOUT_MESSAGE = "Cortana: I'll stop listening for now."


class RealtimeIdleWatch:
    """One session-local idle timer using monotonic time."""

    def __init__(
        self,
        *,
        timeout_seconds: float = REALTIME_IDLE_TIMEOUT_SECONDS,
        monotonic_fn: Callable[[], float],
    ) -> None:
        self._timeout_seconds = bounded_realtime_idle_timeout_seconds(
            timeout_seconds
        )
        self._monotonic = monotonic_fn
        self._last_meaningful_user_activity_monotonic = 0.0
        self._triggered = False

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def last_meaningful_user_activity_monotonic(self) -> float:
        return self._last_meaningful_user_activity_monotonic

    def start(self) -> None:
        """Arm the timer when a realtime session successfully starts."""
        self._last_meaningful_user_activity_monotonic = self._monotonic()
        self._triggered = False

    def mark_meaningful_user_activity(self) -> None:
        """Record one accepted meaningful user interaction.

        Call only after transcript validity, Stop/control routing,
        accidental-turn filtering, and background/self-echo gating.
        """
        if self._triggered:
            return
        self._last_meaningful_user_activity_monotonic = self._monotonic()

    def consume_timeout(self) -> bool:
        """Return True exactly once when the idle timeout has elapsed."""
        if self._triggered:
            return False
        last = self._last_meaningful_user_activity_monotonic
        if last <= 0.0:
            return False
        if self._monotonic() - last < self._timeout_seconds:
            return False
        self._triggered = True
        return True

    def clear(self) -> None:
        """Drop session-local idle state during cleanup."""
        self._last_meaningful_user_activity_monotonic = 0.0
        self._triggered = False
