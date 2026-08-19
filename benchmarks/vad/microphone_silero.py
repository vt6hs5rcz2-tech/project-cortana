from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd


MODEL_PATH = Path(__file__).with_name("silero_vad.onnx")

SOURCE_RATE = 24_000
TARGET_RATE = 16_000
CHANNELS = 1
BLOCK_MS = 20
BLOCK_SAMPLES = SOURCE_RATE * BLOCK_MS // 1000
TARGET_WINDOW = 512
CONTEXT = 64


def resample_24k_to_16k(samples: np.ndarray) -> np.ndarray:
    target_length = int(round(len(samples) * TARGET_RATE / SOURCE_RATE))
    source_positions = np.arange(len(samples), dtype=np.float32)
    target_positions = np.linspace(
        0,
        len(samples) - 1,
        target_length,
        dtype=np.float32,
    )
    return np.interp(
        target_positions,
        source_positions,
        samples,
    ).astype(np.float32)


def main() -> None:
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        del frames, time_info
        if status:
            print(f"audio status: {status}")
        audio_queue.put(indata[:, 0].copy())

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT), dtype=np.float32)
    sample_rate = np.array(TARGET_RATE, dtype=np.int64)
    buffer = np.empty(0, dtype=np.float32)

    print("Silero microphone test")
    print("Stay silent for a few seconds, then speak normally.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        with sd.InputStream(
            samplerate=SOURCE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            callback=callback,
        ):
            while True:
                chunk = audio_queue.get()
                converted = resample_24k_to_16k(chunk)
                buffer = np.concatenate((buffer, converted))

                while len(buffer) >= TARGET_WINDOW:
                    window = buffer[:TARGET_WINDOW]
                    buffer = buffer[TARGET_WINDOW:]

                    model_input = np.concatenate(
                        (context, window.reshape(1, -1)),
                        axis=1,
                    )

                    output, state = session.run(
                        None,
                        {
                            "input": model_input,
                            "state": state,
                            "sr": sample_rate,
                        },
                    )

                    probability = float(output[0][0])
                    context = model_input[:, -CONTEXT:]

                    if probability >= 0.50:
                        label = "SPEECH"
                    elif probability >= 0.20:
                        label = "maybe"
                    else:
                        label = "silence"

                    print(f"{label:7}  probability={probability:.3f}")

    except KeyboardInterrupt:
        print()
        print("Stopped.")


if __name__ == "__main__":
    main()
