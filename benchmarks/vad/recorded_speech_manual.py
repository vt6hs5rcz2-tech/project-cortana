import sounddevice as sd
import numpy as np
import onnxruntime as ort

SOURCE_RATE = 24000
TARGET_RATE = 16000
WINDOW = 512
CONTEXT = 64

print("Speak normally for 5 seconds...")
x = sd.rec(
    int(5 * SOURCE_RATE),
    samplerate=SOURCE_RATE,
    channels=1,
    dtype="float32",
)
sd.wait()

x = x[:, 0]

src = np.arange(len(x), dtype=np.float32)
dst = np.linspace(
    0,
    len(x) - 1,
    int(len(x) * TARGET_RATE / SOURCE_RATE),
    dtype=np.float32,
)
y = np.interp(dst, src, x).astype(np.float32)

session = ort.InferenceSession(
    r"benchmarks\vad\silero_vad.onnx",
    providers=["CPUExecutionProvider"],
)

state = np.zeros((2, 1, 128), dtype=np.float32)
context = np.zeros((1, CONTEXT), dtype=np.float32)
sample_rate = np.array(TARGET_RATE, dtype=np.int64)

probabilities = []

for i in range(0, len(y) - WINDOW + 1, WINDOW):
    window = y[i:i + WINDOW].reshape(1, -1)

    model_input = np.concatenate(
        (context, window),
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

    probabilities.append(float(output[0][0]))
    context = model_input[:, -CONTEXT:]

print("windows:", len(probabilities))
print("max probability:", max(probabilities))
print("mean probability:", sum(probabilities) / len(probabilities))
print(
    "speech windows >=0.5:",
    sum(p >= 0.5 for p in probabilities),
)
print(
    "speech windows >=0.2:",
    sum(p >= 0.2 for p in probabilities),
)
