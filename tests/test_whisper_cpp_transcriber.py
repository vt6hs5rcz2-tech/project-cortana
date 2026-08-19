from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from src.voice_input import NormalizedAudioInput
from src.whisper_cpp_transcriber import (
    WhisperCppTranscriber,
    WhisperCppTranscriberError,
)


def _audio() -> NormalizedAudioInput:
    return NormalizedAudioInput(
        audio_bytes=b"RIFF" + (b"\x00" * 100),
        format="wav",
        sample_rate=16_000,
        channels=1,
        sample_width_bytes=2,
        duration_ms=1000,
        source_kind="microphone",
    )


def test_whisper_cpp_transcriber_returns_stripped_stdout() -> None:
    transcriber = WhisperCppTranscriber(
        executable_path=Path("whisper-cli.exe"),
        model_path=Path("ggml-small.en.bin"),
    )

    completed = CompletedProcess(
        args=[],
        returncode=0,
        stdout="  hello world  \n",
        stderr="",
    )

    with patch(
        "src.whisper_cpp_transcriber.subprocess.run",
        return_value=completed,
    ) as run_mock:
        result = transcriber.transcribe(_audio())

    assert result == "hello world"

    command = run_mock.call_args.args[0]

    assert command[0] == "whisper-cli.exe"
    assert command[1:3] == ["-m", "ggml-small.en.bin"]
    assert "-nt" in command
    assert "-np" in command


def test_whisper_cpp_transcriber_maps_nonzero_exit_to_error() -> None:
    transcriber = WhisperCppTranscriber(
        executable_path=Path("whisper-cli.exe"),
        model_path=Path("ggml-small.en.bin"),
    )

    completed = CompletedProcess(
        args=[],
        returncode=2,
        stdout="",
        stderr="failure",
    )

    with patch(
        "src.whisper_cpp_transcriber.subprocess.run",
        return_value=completed,
    ):
        with pytest.raises(WhisperCppTranscriberError, match="exit code 2"):
            transcriber.transcribe(_audio())


def test_whisper_cpp_transcriber_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        WhisperCppTranscriber(
            executable_path=Path("whisper-cli.exe"),
            model_path=Path("ggml-small.en.bin"),
            timeout_seconds=0,
        )
from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import patch

import pytest

from src.whisper_cpp_transcriber import WhisperCppTranscriber


def test_whisper_cpp_transcriber_maps_timeout_to_error() -> None:
    transcriber = WhisperCppTranscriber(
        executable_path=Path("whisper-cli.exe"),
        model_path=Path("ggml-small.en.bin"),
        timeout_seconds=1.0,
    )

    with patch(
        "src.whisper_cpp_transcriber.subprocess.run",
        side_effect=TimeoutExpired(
            cmd=["whisper-cli.exe"],
            timeout=1.0,
        ),
    ):
        with pytest.raises(
            WhisperCppTranscriberError,
            match="timed out",
        ):
            transcriber.transcribe(_audio())


def test_whisper_cpp_transcriber_uses_temporary_wav_path() -> None:
    transcriber = WhisperCppTranscriber(
        executable_path=Path("whisper-cli.exe"),
        model_path=Path("ggml-small.en.bin"),
    )

    completed = __import__("subprocess").CompletedProcess(
        args=[],
        returncode=0,
        stdout="hello",
        stderr="",
    )

    captured_path: Path | None = None

    def fake_run(command, **kwargs):
        nonlocal captured_path
        del kwargs
        captured_path = Path(command[4])
        assert captured_path.exists()
        return completed

    with patch(
        "src.whisper_cpp_transcriber.subprocess.run",
        side_effect=fake_run,
    ):
        assert transcriber.transcribe(_audio()) == "hello"

    assert captured_path is not None
    assert captured_path.exists() is False

def test_whisper_cpp_transcriber_maps_start_failure_to_error() -> None:
    transcriber = WhisperCppTranscriber(
        executable_path=Path("missing-whisper-cli.exe"),
        model_path=Path("ggml-small.en.bin"),
    )

    with patch(
        "src.whisper_cpp_transcriber.subprocess.run",
        side_effect=FileNotFoundError("missing executable"),
    ):
        with pytest.raises(
            WhisperCppTranscriberError,
            match="could not start",
        ):
            transcriber.transcribe(_audio())
