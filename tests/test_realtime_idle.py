"""Idle timeout helper and conservative realtime noise/self-echo classification."""

from __future__ import annotations

from src.accidental_realtime_turn import (
    RecentAssistantSpeechBuffer,
    RealtimeTurnRejectReason,
    classify_realtime_user_transcript,
    is_likely_self_echo,
    is_punctuation_only_transcript,
)
from src.config import (
    MAX_REALTIME_IDLE_TIMEOUT_SECONDS,
    MIN_REALTIME_IDLE_TIMEOUT_SECONDS,
    REALTIME_IDLE_TIMEOUT_SECONDS,
    bounded_realtime_idle_timeout_seconds,
)
from src.realtime_idle import RealtimeIdleWatch


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_idle_watch_does_not_timeout_before_deadline() -> None:
    clock = FakeClock()
    watch = RealtimeIdleWatch(timeout_seconds=10.0, monotonic_fn=clock)
    watch.start()
    clock.advance(9.9)
    assert watch.consume_timeout() is False
    assert watch.triggered is False


def test_idle_watch_times_out_once_after_deadline() -> None:
    clock = FakeClock()
    watch = RealtimeIdleWatch(timeout_seconds=10.0, monotonic_fn=clock)
    watch.start()
    clock.advance(10.0)
    assert watch.consume_timeout() is True
    assert watch.triggered is True
    clock.advance(5.0)
    assert watch.consume_timeout() is False


def test_meaningful_activity_resets_idle_watch() -> None:
    clock = FakeClock()
    watch = RealtimeIdleWatch(timeout_seconds=10.0, monotonic_fn=clock)
    watch.start()
    clock.advance(9.0)
    watch.mark_meaningful_user_activity()
    clock.advance(9.0)
    assert watch.consume_timeout() is False
    clock.advance(1.0)
    assert watch.consume_timeout() is True


def test_idle_timeout_bounds() -> None:
    assert REALTIME_IDLE_TIMEOUT_SECONDS == 10.0
    assert MIN_REALTIME_IDLE_TIMEOUT_SECONDS == 5.0
    assert MAX_REALTIME_IDLE_TIMEOUT_SECONDS == 300.0
    assert bounded_realtime_idle_timeout_seconds(10.0) == 10.0
    assert (
        bounded_realtime_idle_timeout_seconds(1.0)
        == MIN_REALTIME_IDLE_TIMEOUT_SECONDS
    )
    assert (
        bounded_realtime_idle_timeout_seconds(999.0)
        == MAX_REALTIME_IDLE_TIMEOUT_SECONDS
    )
    assert (
        bounded_realtime_idle_timeout_seconds("nope")
        == REALTIME_IDLE_TIMEOUT_SECONDS
    )


def test_empty_whitespace_and_punctuation_are_rejected() -> None:
    assert (
        classify_realtime_user_transcript("")
        is RealtimeTurnRejectReason.EMPTY
    )
    assert (
        classify_realtime_user_transcript("   \n")
        is RealtimeTurnRejectReason.WHITESPACE
    )
    assert is_punctuation_only_transcript("...") is True
    assert (
        classify_realtime_user_transcript("...")
        is RealtimeTurnRejectReason.PUNCTUATION
    )
    assert (
        classify_realtime_user_transcript("It's...")
        is RealtimeTurnRejectReason.ACCIDENTAL_FRAGMENT
    )


def test_legitimate_short_turns_are_not_rejected() -> None:
    for phrase in ("No", "Yes", "Wait", "Stop", "Okay", "Tuesday", "RAM"):
        assert classify_realtime_user_transcript(phrase) is None, phrase


def test_self_echo_during_playback_is_rejected() -> None:
    clock = FakeClock()
    buffer = RecentAssistantSpeechBuffer()
    buffer.remember("The day after Monday is Tuesday.", now=clock.t)
    assert (
        is_likely_self_echo(
            "The day after Monday is Tuesday.",
            recent_assistant=buffer.recent(clock.t),
            playback_active=True,
            seconds_since_playback_end=None,
        )
        is True
    )
    assert (
        classify_realtime_user_transcript(
            "The day after Monday is Tuesday.",
            playback_active=True,
            recent_assistant=buffer.recent(clock.t),
        )
        is RealtimeTurnRejectReason.SELF_ECHO
    )


def test_intentional_repeat_after_playback_is_not_echo() -> None:
    clock = FakeClock()
    buffer = RecentAssistantSpeechBuffer()
    buffer.remember("Tuesday.", now=clock.t)
    clock.advance(3.0)
    assert (
        is_likely_self_echo(
            "Tuesday.",
            recent_assistant=buffer.recent(clock.t),
            playback_active=False,
            seconds_since_playback_end=3.0,
        )
        is False
    )
    assert (
        classify_realtime_user_transcript(
            "Tuesday.",
            playback_active=False,
            seconds_since_playback_end=3.0,
            recent_assistant=buffer.recent(clock.t),
        )
        is None
    )


def test_correction_is_not_self_echo() -> None:
    clock = FakeClock()
    buffer = RecentAssistantSpeechBuffer()
    buffer.remember("The remote is black.", now=clock.t)
    assert (
        is_likely_self_echo(
            "No, the remote is blue.",
            recent_assistant=buffer.recent(clock.t),
            playback_active=True,
            seconds_since_playback_end=None,
        )
        is False
    )
    assert (
        classify_realtime_user_transcript(
            "No, the remote is blue.",
            playback_active=True,
            recent_assistant=buffer.recent(clock.t),
        )
        is None
    )
