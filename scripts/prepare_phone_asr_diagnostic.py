#!/usr/bin/env python3
"""Materialize the exact 18-utterance ASR diagnostic used by the Colab report."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from path_config import WORK_ROOT

N_SAMPLES = 18
SEED = 20260715
DATASET = "hf-internal-testing/librispeech_asr_dummy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_ROOT / "phone_asr_diagnostic",
    )
    return parser.parse_args()


def load_audio(value: dict[str, Any]) -> np.ndarray:
    source: Any
    if value.get("bytes") is not None:
        source = io.BytesIO(value["bytes"])
    elif value.get("path"):
        source = value["path"]
    else:
        raise ValueError("Dataset audio sample has neither bytes nor path")
    wave, rate = sf.read(source, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
    return wave.astype("float32", copy=False)


def peak_safe(wave: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    return wave * (0.98 / peak) if peak > 0.98 else wave


def mix_at_snr(signal: np.ndarray, interference: np.ndarray, snr_db: float) -> np.ndarray:
    if interference.size < signal.size:
        repeats = int(math.ceil(signal.size / max(1, interference.size)))
        interference = np.tile(interference, repeats)
    interference = interference[: signal.size]
    signal_rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(interference * interference) + 1e-12))
    scale = signal_rms / (10 ** (snr_db / 20) * noise_rms)
    return peak_safe(signal + interference * scale).astype("float32")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, "clean", split="validation").cast_column("audio", Audio(decode=False))
    samples = [
        {"wave": load_audio(row["audio"]), "text": row["text"]}
        for row in dataset.select(range(min(N_SAMPLES, len(dataset))))
    ]
    rng = np.random.default_rng(SEED)
    conditions: dict[str, list[np.ndarray]] = {
        "clean": [sample["wave"] for sample in samples],
        "gaussian_10db": [
            mix_at_snr(sample["wave"], rng.standard_normal(sample["wave"].shape).astype("float32"), 10.0)
            for sample in samples
        ],
        "competing_speech_5db": [
            mix_at_snr(sample["wave"], samples[(index + 7) % len(samples)]["wave"], 5.0)
            for index, sample in enumerate(samples)
        ],
    }
    entries = []
    for condition, waves in conditions.items():
        condition_dir = args.output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        for index, wave in enumerate(waves):
            path = condition_dir / f"sample_{index:02d}.wav"
            sf.write(path, wave, 16000, subtype="PCM_16")
            entries.append(
                {
                    "condition": condition,
                    "sample_index": index,
                    "file": str(path.relative_to(args.output_dir)),
                    "reference": samples[index]["text"],
                    "audio_seconds": len(wave) / 16000,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "dataset": f"{DATASET} clean/validation",
        "sample_count": len(samples),
        "seed": SEED,
        "encoding": "mono PCM16 WAV at 16 kHz",
        "conditions": {
            "clean": "unaltered except PCM16 serialization",
            "gaussian_10db": "deterministic white noise at 10 dB SNR",
            "competing_speech_5db": "sample (index + 7) mod 18 mixed at 5 dB target-to-interferer ratio",
        },
        "entries": entries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
