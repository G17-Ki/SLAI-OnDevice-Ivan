#!/usr/bin/env python3
"""Lane F F2 — host CPU reference for the ES/FR Parakeet route.

Runs the pinned sherpa-onnx Parakeet-TDT-0.6b-v3 INT8 offline transducer over a
list of 16 kHz mono WAVs, once per --threads setting, and records per-WAV
transcripts, decode wall time and real-time factor.

This is a HOST measurement on an Apple M2. It is not phone performance and it
is not QNN execution: the INT8 ONNX graphs run on the CPU provider only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"{path} is not mono ({samples.shape[1]} channels)")
    if rate != 16000:
        raise ValueError(f"{path} is {rate} Hz, expected 16000")
    return samples[:, 0], rate


def build_recognizer(model_dir: Path, threads: int) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider="cpu",
        debug=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--wav", action="append", required=True,
                        help="lang=path pairs, e.g. es=/abs/es.wav")
    parser.add_argument("--threads", action="append", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    thread_counts = arguments.threads or [4]

    model_dir: Path = arguments.model_dir
    model_files = {
        name: {
            "bytes": (model_dir / name).stat().st_size,
            "sha256": sha256_file(model_dir / name),
        }
        for name in (
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        )
    }

    inputs = []
    for pair in arguments.wav:
        language, _, raw_path = pair.partition("=")
        path = Path(raw_path)
        samples, rate = read_mono_16k(path)
        inputs.append(
            {
                "language": language,
                "path": str(path),
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sample_count": int(samples.shape[0]),
                "duration_seconds": float(samples.shape[0]) / rate,
                "_samples": samples,
            }
        )

    runs = []
    for threads in thread_counts:
        init_started = time.perf_counter()
        recognizer = build_recognizer(model_dir, threads)
        init_seconds = time.perf_counter() - init_started

        results = []
        for item in inputs:
            texts: set[str] = set()
            factors: list[float] = []
            seconds: list[float] = []
            last = None
            for _ in range(max(1, arguments.repeats)):
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, item["_samples"])
                decode_started = time.perf_counter()
                recognizer.decode_stream(stream)
                decode_seconds = time.perf_counter() - decode_started
                last = stream.result
                texts.add(last.text.strip())
                seconds.append(decode_seconds)
                factors.append(decode_seconds / item["duration_seconds"])
            text = sorted(texts)[0]
            results.append(
                {
                    "language": item["language"],
                    "wav": item["name"],
                    "wav_sha256": item["sha256"],
                    "duration_seconds": item["duration_seconds"],
                    "repeats": len(factors),
                    "decode_seconds_all": seconds,
                    "decode_seconds_median": float(np.median(seconds)),
                    "real_time_factor_all": factors,
                    "real_time_factor_median": float(np.median(factors)),
                    "real_time_factor_min": min(factors),
                    "real_time_factor_max": max(factors),
                    "text": text,
                    "deterministic_across_repeats": len(texts) == 1,
                    "distinct_texts": sorted(texts),
                    "tokens": list(last.tokens),
                    "timestamps_seconds": [float(v) for v in last.timestamps],
                    "non_empty": bool(text),
                }
            )
            print(f"[threads={threads}] {item['language']}: rtf_median="
                  f"{np.median(factors):.3f} (n={len(factors)}, "
                  f"{min(factors):.3f}-{max(factors):.3f}) "
                  f"stable={len(texts) == 1} :: {text}")
        del recognizer

        runs.append(
            {
                "num_threads": threads,
                "recognizer_init_seconds": init_seconds,
                "results": results,
            }
        )

    payload = {
        "schema_version": 1,
        "purpose": "Lane F host CPU reference for the ES/FR Parakeet route",
        "evidence_class": "HOST",
        "not_a_measurement_of": [
            "Nubia NX809J / SM8850 phone latency",
            "QNN or HTP execution",
            "word error rate against a scored reference corpus",
        ],
        "runtime": {
            "host": f"{platform.platform()} {platform.machine()}",
            "python": platform.python_version(),
            "sherpa_onnx": sherpa_onnx.__version__,
            "provider": "cpu",
            "model_type": "nemo_transducer",
            "decoding_method": "greedy_search",
            "sample_rate_hz": 16000,
            "feature_dim": 80,
            "maximum_resident_set_bytes":
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "model": {
            "directory": model_dir.name,
            "files": model_files,
        },
        "inputs": [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in inputs
        ],
        "runs": runs,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
