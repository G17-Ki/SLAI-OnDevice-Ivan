#!/usr/bin/env python3
"""Compare canonical CPU SenseVoice output with the exact SM8850 QNN fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
import wave
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 WAV: {path}")
        sample_rate = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, sample_rate


def normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in value if character.isalnum())


def decode(recognizer: object, samples: np.ndarray, sample_rate: int) -> dict[str, object]:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    started = time.perf_counter_ns()
    recognizer.decode_stream(stream)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    result = stream.result
    text = str(getattr(result, "text", result)).strip()
    language = str(getattr(result, "lang", ""))
    return {"text": text, "language": language, "decode_ms": elapsed_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--qnn-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    global np, sherpa_onnx
    import numpy as np
    import sherpa_onnx

    qnn = json.loads(args.qnn_smoke.read_text(encoding="utf-8"))
    qnn_by_wav = {run["wav"]: json.loads(next(line for line in run["stdout"].splitlines() if line.startswith("{"))) for run in qnn["runs"]}

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(args.model_dir / "model.int8.onnx"),
        tokens=str(args.model_dir / "tokens.txt"),
        num_threads=2,
        provider="cpu",
        language="auto",
        use_itn=True,
        debug=False,
    )

    runs = []
    for name in ("en.wav", "zh.wav"):
        path = args.fixture_dir / name
        samples, sample_rate = read_pcm16(path)
        padded = np.pad(samples, (0, max(0, sample_rate * 10 - len(samples))))[: sample_rate * 10]
        variable = decode(recognizer, samples, sample_rate)
        fixed_10s = decode(recognizer, padded, sample_rate)
        qnn_result = qnn_by_wav[name]
        runs.append(
            {
                "wav": name,
                "wav_sha256": sha256(path),
                "duration_seconds": len(samples) / sample_rate,
                "sample_rate": sample_rate,
                "qnn": {"text": qnn_result["text"], "language": qnn_result["lang"]},
                "cpu_variable": variable,
                "cpu_fixed_10s": fixed_10s,
                "exact_text_parity_variable": variable["text"] == qnn_result["text"],
                "exact_text_parity_fixed_10s": fixed_10s["text"] == qnn_result["text"],
                "language_parity_variable": variable["language"] == qnn_result["lang"],
                "language_parity_fixed_10s": fixed_10s["language"] == qnn_result["lang"],
                "normalized_text_parity_variable": normalized(variable["text"]) == normalized(qnn_result["text"]),
                "normalized_text_parity_fixed_10s": normalized(fixed_10s["text"]) == normalized(qnn_result["text"]),
            }
        )

    output = {
        "schema_version": 1,
        "cpu_provider": "sherpa-onnx SenseVoiceSmall INT8 / CPU",
        "sherpa_onnx_version": sherpa_onnx.__version__,
        "model_sha256": sha256(args.model_dir / "model.int8.onnx"),
        "tokens_sha256": sha256(args.model_dir / "tokens.txt"),
        "settings": {"language": "auto", "use_itn": True, "num_threads": 2},
        "runs": runs,
        "parity": {
            run["wav"][:2]: "pass"
            if run["exact_text_parity_fixed_10s"] and run["language_parity_fixed_10s"]
            else "fail"
            for run in runs
        },
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
