"""Tests for Milestone 25 realtime microphone frame capture."""

from __future__ import annotations

from queue import Full, Queue
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import (
    REALTIME_VOICE_FRAME_BYTES,
    REALTIME_VOICE_INPUT_QUEUE_FRAMES,
    REALTIME_VOICE_SAMPLE_RATE_HZ,
)
from src.realtime_voice_input import (
    RealtimeAudioFrame,
    RealtimeMicrophoneStream,
    RealtimeVoiceInputValidationError,
)


def _valid_pcm() -> bytes:
    return b"\x00\x00" * (REALTIME_VOICE_FRAME_BYTES // 2)


def test_realtime_audio_frame_accepts_canonical_frame() -> None:
    frame = RealtimeAudioFrame(
        pcm_bytes=_valid_pcm(),
        sample_rate=REALTIME_VOICE_SAMPLE_RATE_HZ,
        channels=1,
        sample_width_bytes=2,
        sequence=0,
    )
    assert len(frame.pcm_bytes) == REALTIME_VOICE_FRAME_BYTES


def test_realtime_audio_frame_rejects_wrong_size() -> None:
    with pytest.raises(RealtimeVoiceInputValidationError):
        RealtimeAudioFrame(
            pcm_bytes=b"\x00\x00",
            sample_rate=REALTIME_VOICE_SAMPLE_RATE_HZ,
            channels=1,
            sample_width_bytes=2,
            sequence=0,
        )


def test_realtime_audio_frame_rejects_wrong_rate() -> None:
    with pytest.raises(RealtimeVoiceInputValidationError):
        RealtimeAudioFrame(
            pcm_bytes=_valid_pcm(),
            sample_rate=16_000,
            channels=1,
            sample_width_bytes=2,
            sequence=0,
        )


def test_microphone_init_does_not_open_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = {"count": 0}

    class Boom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            opened["count"] += 1
            raise AssertionError("opened")

    monkeypatch.setitem(
        __import__("sys").modules,
        "sounddevice",
        type("SD", (), {"RawInputStream": Boom})(),
    )
    queue: Queue[RealtimeAudioFrame] = Queue(maxsize=REALTIME_VOICE_INPUT_QUEUE_FRAMES)
    stream = RealtimeMicrophoneStream(frame_queue=queue)
    assert opened["count"] == 0
    assert stream.is_active is False


def test_microphone_start_enqueues_frame_and_stop_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_holder: dict[str, Any] = {}

    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]
            callback_holder["samplerate"] = kwargs["samplerate"]
            callback_holder["blocksize"] = kwargs["blocksize"]
            self.closed = False

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        __import__("sys").modules,
        "sounddevice",
        type("SD", (), {"RawInputStream": FakeStream})(),
    )
    monkeypatch.setattr("src.realtime_voice_input.sys.platform", "win32")

    queue: Queue[RealtimeAudioFrame] = Queue(maxsize=REALTIME_VOICE_INPUT_QUEUE_FRAMES)
    stream = RealtimeMicrophoneStream(frame_queue=queue)
    stream.start()
    assert callback_holder["samplerate"] == REALTIME_VOICE_SAMPLE_RATE_HZ
    pcm = _valid_pcm()
    callback_holder["callback"](pcm, 480, None, None)
    frame = queue.get_nowait()
    assert frame.sequence == 0
    assert frame.pcm_bytes == pcm
    stream.stop()
    assert stream.is_active is False


def test_microphone_overflow_invokes_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_holder: dict[str, Any] = {}

    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "sounddevice",
        type("SD", (), {"RawInputStream": FakeStream})(),
    )
    monkeypatch.setattr("src.realtime_voice_input.sys.platform", "win32")

    queue: Queue[RealtimeAudioFrame] = Queue(maxsize=1)
    overflow = MagicMock()
    stream = RealtimeMicrophoneStream(frame_queue=queue, on_overflow=overflow)
    stream.start()
    pcm = _valid_pcm()
    callback_holder["callback"](pcm, 480, None, None)
    with pytest.raises(Full):
        queue.put_nowait(
            RealtimeAudioFrame(
                pcm_bytes=pcm,
                sample_rate=REALTIME_VOICE_SAMPLE_RATE_HZ,
                channels=1,
                sample_width_bytes=2,
                sequence=99,
            )
        )
    # Fill already has one; next callback should overflow.
    callback_holder["callback"](pcm, 480, None, None)
    overflow.assert_called_once()
    stream.stop()
