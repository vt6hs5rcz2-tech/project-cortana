from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_PATH = Path(__file__).with_name("silero_vad.onnx")

SOURCE_RATE = 24_000
TARGET_RATE = 16_000
SOURCE_FRAME_MS = 20
SOURCE_FRAME_SAMPLES = SOURCE_RATE * SOURCE_FRAME_MS // 1000
TARGET_WINDOW_SAMPLES = 512


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

    # Five Cortana frames = 100 ms of 24 kHz audio.
    source = np.zeros(SOURCE_FRAME_SAMPLES * 5, dtype=np.float32)
    resampled = resample_24k_to_16k(source)

    if len(resampled) < TARGET_WINDOW_SAMPLES:
        raise RuntimeError("Not enough samples for one Silero window.")

    state = np.zeros((2, 1, 128), dtype=np.float32)
    sample_rate = np.array(TARGET_RATE, dtype=np.int64)

    window = resampled[:TARGET_WINDOW_SAMPLES].reshape(1, -1)

    # Warm up once before timing.
    _, state = session.run(
        None,
        {
            "input": window,
            "state": state,
            "sr": sample_rate,
        },
    )

    iterations = 1000
    start = time.perf_counter()

    probability = 0.0

    for _ in range(iterations):
        output, state = session.run(
            None,
            {
                "input": window,
                "state": state,
                "sr": sample_rate,
            },
        )
        probability = float(output[0][0])

    elapsed = time.perf_counter() - start
    average_ms = elapsed * 1000 / iterations

    print(f"model: {MODEL_PATH}")
    print(f"source frame: {SOURCE_FRAME_SAMPLES} samples @ {SOURCE_RATE} Hz")
    print(f"resampled buffer: {len(resampled)} samples @ {TARGET_RATE} Hz")
    print(f"Silero window: {TARGET_WINDOW_SAMPLES} samples")
    print(f"iterations: {iterations}")
    print(f"average inference: {average_ms:.4f} ms")
    print(f"final silence probability: {probability:.6f}")


if __name__ == "__main__":
    main()
