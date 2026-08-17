"""Deterministic explicit stop-command detection."""

from __future__ import annotations

from src.accidental_realtime_turn import is_accidental_playback_turn
from src.realtime_stop_command import is_explicit_stop_command


def test_standalone_stop_commands_match() -> None:
    for phrase in (
        "Stop",
        "Stop!",
        "Stop.",
        "Stop talking",
        "Stop talking.",
        "Stop speaking",
        "Be quiet",
        "Quiet",
        "Cancel that",
        "Never mind",
        "Stop talking, Cortana.",
        "Cortana, stop",
        "Cortana stop talking",
    ):
        assert is_explicit_stop_command(phrase) is True, phrase


def test_non_command_stop_phrases_do_not_match() -> None:
    for phrase in (
        "Don't stop",
        "Do not stop",
        "What is a stop sign?",
        "Stop the server",
        "Stop the timer",
        "Tell me about stop-motion animation",
        "Please don't stop",
        "I never mind the noise",
        "Stop talking, Cortana. When I tell you to stop, that means stop talking.",
    ):
        assert is_explicit_stop_command(phrase) is False, phrase


def test_stop_outranks_accidental_fragment_filter() -> None:
    assert (
        is_accidental_playback_turn(
            "Stop",
            barged_during_playback=True,
            seconds_after_playback_start=0.2,
        )
        is False
    )
    assert (
        is_accidental_playback_turn(
            "No",
            barged_during_playback=True,
            seconds_after_playback_start=0.2,
        )
        is False
    )
    assert (
        is_accidental_playback_turn(
            "Wait",
            barged_during_playback=True,
            seconds_after_playback_start=0.2,
        )
        is False
    )
