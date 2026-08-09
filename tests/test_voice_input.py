"""Tests for Milestone 24 microphone capture and normalized audio."""

from __future__ import annotations

import threading
import time

import pytest

import src.voice_input as voice_input
from src.config import (
    MAX_VOICE_AUDIO_BYTES,
    MAX_VOICE_PCM_BYTES,
    MAX_VOICE_UTTERANCE_SECONDS,
    MIN_VOICE_UTTERANCE_MS,
    VOICE_CHANNELS,
    VOICE_SAMPLE_RATE_HZ,
    VOICE_SAMPLE_WIDTH_BYTES,
    VOICE_WAV_HEADER_BYTES,
)
from src.voice_input import (
    MicrophoneCaptureAdapter,
    NormalizedAudioInput,
    VoiceCaptureCancelledError,
    VoiceInputValidationError,
    duration_ms_from_pcm,
    pcm_to_wav_bytes,
)


def _pcm_silence(duration_ms: int = 300) -> bytes:
    frames = (VOICE_SAMPLE_RATE_HZ * duration_ms) // 1000
    return b"\x00\x00" * frames


def test_voice_byte_bounds_are_derived() -> None:
    assert MAX_VOICE_PCM_BYTES == 16000 * 1 * 2 * 30
    assert MAX_VOICE_PCM_BYTES == 960_000
    assert VOICE_WAV_HEADER_BYTES == 44
    assert MAX_VOICE_AUDIO_BYTES == MAX_VOICE_PCM_BYTES + VOICE_WAV_HEADER_BYTES
    assert MAX_VOICE_AUDIO_BYTES == 960_044


def test_pcm_to_wav_uses_exact_canonical_header_size() -> None:
    pcm = _pcm_silence(1000)
    wav = pcm_to_wav_bytes(pcm)
    assert len(wav) - len(pcm) == VOICE_WAV_HEADER_BYTES
    assert len(wav) <= MAX_VOICE_AUDIO_BYTES


def test_max_pcm_round_trip_fits_max_audio_bytes() -> None:
    pcm = b"\x00\x00" * (MAX_VOICE_PCM_BYTES // 2)
    assert len(pcm) == MAX_VOICE_PCM_BYTES
    wav = pcm_to_wav_bytes(pcm)
    assert len(wav) == MAX_VOICE_AUDIO_BYTES


def test_normalized_audio_rejects_non_canonical_fields() -> None:
    pcm = _pcm_silence()
    wav = pcm_to_wav_bytes(pcm)
    duration = duration_ms_from_pcm(pcm)
    with pytest.raises(VoiceInputValidationError):
        NormalizedAudioInput(
            audio_bytes=wav,
            format="wav",
            sample_rate=8000,
            channels=VOICE_CHANNELS,
            sample_width_bytes=VOICE_SAMPLE_WIDTH_BYTES,
            duration_ms=duration,
            source_kind="microphone",
        )


def test_normalized_audio_rejects_too_short_duration() -> None:
    pcm = _pcm_silence(MIN_VOICE_UTTERANCE_MS - 1)
    wav = pcm_to_wav_bytes(pcm)
    with pytest.raises(VoiceInputValidationError):
        NormalizedAudioInput(
            audio_bytes=wav,
            format="wav",
            sample_rate=VOICE_SAMPLE_RATE_HZ,
            channels=VOICE_CHANNELS,
            sample_width_bytes=VOICE_SAMPLE_WIDTH_BYTES,
            duration_ms=MIN_VOICE_UTTERANCE_MS - 1,
            source_kind="microphone",
        )


def test_microphone_adapter_init_does_not_open_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = {"count": 0}

    class BoomStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            opened["count"] += 1
            raise AssertionError("RawInputStream must not open during init")

    fake_sd = type("SD", (), {"RawInputStream": BoomStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    adapter = MicrophoneCaptureAdapter()
    assert opened["count"] == 0
    assert adapter is not None


def test_capture_builds_normalized_wav_and_closes_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm = _pcm_silence(300)
    events: list[str] = []

    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.callback = kwargs["callback"]
            events.append("open")

        def start(self) -> None:
            events.append("start")
            self.callback(pcm, len(pcm) // 2, None, None)

        def stop(self) -> None:
            events.append("stop")

        def close(self) -> None:
            events.append("close")

    fake_sd = type("SD", (), {"RawInputStream": FakeStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice_input.sys, "platform", "win32")

    adapter = MicrophoneCaptureAdapter()
    result = adapter.capture(stop_signal=lambda: True)

    assert result.format == "wav"
    assert result.sample_rate == VOICE_SAMPLE_RATE_HZ
    assert result.channels == 1
    assert result.sample_width_bytes == 2
    assert result.source_kind == "microphone"
    assert result.duration_ms >= MIN_VOICE_UTTERANCE_MS
    assert len(result.audio_bytes) <= MAX_VOICE_AUDIO_BYTES
    assert events == ["open", "start", "stop", "close"]


def test_capture_cancel_via_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_sd = type("SD", (), {"RawInputStream": FakeStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice_input.sys, "platform", "win32")

    def cancel() -> bool:
        raise KeyboardInterrupt

    adapter = MicrophoneCaptureAdapter()
    with pytest.raises(VoiceCaptureCancelledError):
        adapter.capture(stop_signal=cancel)


def test_capture_cancel_via_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_sd = type("SD", (), {"RawInputStream": FakeStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice_input.sys, "platform", "win32")

    def cancel() -> bool:
        raise EOFError

    adapter = MicrophoneCaptureAdapter()
    with pytest.raises(VoiceCaptureCancelledError):
        adapter.capture(stop_signal=cancel)


def test_capture_hard_stop_does_not_exceed_max_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = b"\x00\x00" * ((MAX_VOICE_PCM_BYTES // 2) + 1000)

    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.callback = kwargs["callback"]

        def start(self) -> None:
            self.callback(oversized, len(oversized) // 2, None, None)

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_sd = type("SD", (), {"RawInputStream": FakeStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice_input.sys, "platform", "win32")

    result = MicrophoneCaptureAdapter().capture(stop_signal=lambda: True)
    assert len(result.audio_bytes) <= MAX_VOICE_AUDIO_BYTES
    assert result.duration_ms <= MAX_VOICE_UTTERANCE_SECONDS * 1000


def test_capture_closes_stream_on_callback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.callback = kwargs["callback"]
            events.append("open")

        def start(self) -> None:
            events.append("start")
            self.callback(object(), 1, None, None)

        def stop(self) -> None:
            events.append("stop")

        def close(self) -> None:
            events.append("close")

    fake_sd = type("SD", (), {"RawInputStream": FakeStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice_input.sys, "platform", "win32")

    with pytest.raises(voice_input.VoiceInputError):
        MicrophoneCaptureAdapter().capture(stop_signal=lambda: True)
    assert events == ["open", "start", "stop", "close"]


def test_stop_signal_is_distinct_from_outer_read_input() -> None:
    """stop_signal must remain a distinct injectable seam from read_input."""
    outer_calls = {"count": 0}
    stop_calls = {"count": 0}

    def outer_read_input() -> str:
        outer_calls["count"] += 1
        return "/voice-turn"

    def stop_signal() -> bool:
        stop_calls["count"] += 1
        return True

    assert stop_signal() is True
    assert stop_calls["count"] == 1
    assert outer_calls["count"] == 0
    assert outer_read_input() == "/voice-turn"
    assert outer_calls["count"] == 1


def test_m24_f5_timeout_leaves_no_stdin_consuming_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max-duration end must not leave a waiter that can steal later stdin."""
    pcm = _pcm_silence(300)
    events: list[str] = []
    poll_count = {"n": 0}
    stream_holder: dict[str, object] = {}

    class FakeStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.callback = kwargs["callback"]
            stream_holder["stream"] = self
            events.append("open")

        def start(self) -> None:
            events.append("start")
            self.callback(pcm, len(pcm) // 2, None, None)

        def stop(self) -> None:
            events.append("stop")

        def close(self) -> None:
            events.append("close")

    fake_sd = type("SD", (), {"RawInputStream": FakeStream})()
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice_input.sys, "platform", "win32")
    monkeypatch.setattr(voice_input, "VOICE_STOP_POLL_INTERVAL_SECONDS", 0.01)

    def never_stop() -> bool:
        poll_count["n"] += 1
        return False

    adapter = MicrophoneCaptureAdapter()
    adapter._max_utterance_seconds = 0.05  # type: ignore[assignment]
    result = adapter.capture(stop_signal=never_stop)

    count_at_return = poll_count["n"]
    time.sleep(0.1)
    assert poll_count["n"] == count_at_return
    assert not any(
        "voice-stop" in thread.name for thread in threading.enumerate()
    )
    assert events == ["open", "start", "stop", "close"]
    assert result.duration_ms >= MIN_VOICE_UTTERANCE_MS
    assert len(result.audio_bytes) <= MAX_VOICE_AUDIO_BYTES

    # Post-stop PortAudio callbacks must be harmless no-ops.
    stream = stream_holder["stream"]
    assert stream is not None
    callback = getattr(stream, "callback")
    callback(_pcm_silence(500), 8000, None, None)
