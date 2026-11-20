import argparse
import csv
from pathlib import Path
import wave

import numpy as np
from openwakeword.model import Model

RATE, BLOCK = 16000, 1280
THRESHOLDS = (0.35, 0.20, 0.10)


def get_peak(model, path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"{path}: 需要 16 kHz、单声道、16-bit PCM WAV")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    audio = np.pad(audio, (0, RATE + (-(len(audio) + RATE) % BLOCK)))
    np.random.seed(0)
    model.reset()
    return max(
        float(model.predict(audio[i:i + BLOCK])["hey_nailong"])
        for i in range(0, len(audio), BLOCK)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", type=Path)
    parser.add_argument("--model", type=Path, default=Path("hey_nailong.onnx"))
    args = parser.parse_args()

    positives = sorted((args.clips / "NAILONG").rglob("*.wav"))
    negatives = sorted((args.clips / "NEG").rglob("*.wav"))
    negatives += sorted((args.clips / "MILK_DRAGON").rglob("*.wav"))
    if not positives or not negatives:
        parser.error("需要 NAILONG 文件夹里的正样本，以及 NEG 或 MILK_DRAGON 里的负样本。")
    if not args.model.is_file():
        parser.error(f"找不到模型 {args.model}")

    model = Model(wakeword_models=[str(args.model)], inference_framework="onnx")
    rows = [(path, label, get_peak(model, path))
            for label, paths in ((1, positives), (0, negatives)) for path in paths]
    with open("nailong_results.csv", "w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["file", "expected", "peak", *[f"trigger_{t:.2f}" for t in THRESHOLDS]])
        for path, label, peak in rows:
            writer.writerow([path.relative_to(args.clips), label, peak,
                             *[int(peak >= t) for t in THRESHOLDS]])

    for t in THRESHOLDS:
        hits = sum(label == 1 and peak >= t for _, label, peak in rows)
        false = sum(label == 0 and peak >= t for _, label, peak in rows)
        print(f"{t:.2f}  命中 {hits}/{len(positives)}  误触发 {false}/{len(negatives)}")
    print("逐条分数已保存到 nailong_results.csv")


if __name__ == "__main__":
    main()
