"""Narrow accidental-turn filter for multimodal playback fragments."""

from __future__ import annotations

from src.accidental_realtime_turn import (
    is_accidental_playback_turn,
    is_incomplete_copula_fragment,
    is_low_information_fragment,
)


def test_protected_short_turns_are_not_fragments() -> None:
    for phrase in ("Stop.", "No.", "Wait.", "RAM.", "Tuesday."):
        assert is_low_information_fragment(phrase) is False


def test_noise_fragments_are_low_information() -> None:
    assert is_low_information_fragment("Acaba.") is True
    assert is_low_information_fragment("好了。") is True
    assert is_low_information_fragment("I...") is True
    assert is_low_information_fragment("um") is True


def test_meaningful_short_phrases_are_not_fragments() -> None:
    assert is_low_information_fragment("Thank you.") is False
    assert is_low_information_fragment("What object am I holding up?") is False
    assert is_low_information_fragment("Bring me baby.") is False


def test_truncated_its_ellipsis_is_incomplete_copula_fragment() -> None:
    assert is_incomplete_copula_fragment("It's...") is True
    assert is_incomplete_copula_fragment("It's") is True
    assert is_low_information_fragment("It's...") is True
    assert is_incomplete_copula_fragment("It's broken.") is False
    assert is_incomplete_copula_fragment("It's Tuesday.") is False
    assert is_incomplete_copula_fragment("It's mine.") is False
    assert is_low_information_fragment("It's broken.") is False
    assert is_low_information_fragment("It's Tuesday.") is False


def test_truncated_its_during_playback_is_accidental() -> None:
    assert (
        is_accidental_playback_turn(
            "It's...",
            barged_during_playback=True,
            seconds_after_playback_start=0.4,
        )
        is True
    )
    assert (
        is_accidental_playback_turn(
            "It's broken.",
            barged_during_playback=True,
            seconds_after_playback_start=0.4,
        )
        is False
    )
    assert (
        is_accidental_playback_turn(
            "It's Tuesday.",
            barged_during_playback=True,
            seconds_after_playback_start=0.4,
        )
        is False
    )


def test_suppression_requires_early_playback_barge_in() -> None:
    assert (
        is_accidental_playback_turn(
            "Acaba.",
            barged_during_playback=True,
            seconds_after_playback_start=0.4,
        )
        is True
    )
    assert (
        is_accidental_playback_turn(
            "Acaba.",
            barged_during_playback=False,
            seconds_after_playback_start=0.4,
        )
        is False
    )
    assert (
        is_accidental_playback_turn(
            "Acaba.",
            barged_during_playback=True,
            seconds_after_playback_start=3.0,
        )
        is False
    )
    assert (
        is_accidental_playback_turn(
            "Stop.",
            barged_during_playback=True,
            seconds_after_playback_start=0.4,
        )
        is False
    )
