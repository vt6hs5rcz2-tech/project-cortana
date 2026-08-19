from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from src.local_speech_transcriber import LocalSpeechTranscriber
from src.voice_input import NormalizedAudioInput


class WhisperCppTranscriberError(RuntimeError):
    """Raised when local whisper.cpp transcription fails."""


class WhisperCppTranscriber(LocalSpeechTranscriber):
    """Optional local STT adapter backed by the whisper.cpp CLI."""

    def __init__(
        self,
        *,
        executable_path: Path,
        model_path: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self._executable_path = executable_path
        self._model_path = model_path
        self._timeout_seconds = timeout_seconds

    def transcribe(self, audio: NormalizedAudioInput) -> str:
        with tempfile.TemporaryDirectory(prefix="cortana-whisper-") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            wav_path.write_bytes(audio.audio_bytes)

            try:
                completed = subprocess.run(
                    [
                        str(self._executable_path),
                        "-m",
                        str(self._model_path),
                        "-f",
                        str(wav_path),
                        "-nt",
                        "-np",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise WhisperCppTranscriberError(
                    "whisper.cpp transcription timed out."
                ) from error
            except OSError as error:
                raise WhisperCppTranscriberError(
                    "whisper.cpp transcription could not start."
                ) from error

        if completed.returncode != 0:
            raise WhisperCppTranscriberError(
                f"whisper.cpp transcription failed with exit code "
                f"{completed.returncode}."
            )

        return completed.stdout.strip()
