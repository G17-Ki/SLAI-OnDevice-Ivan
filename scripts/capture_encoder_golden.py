#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import soundfile as sf
import torch
import torchaudio

from component_utils import (
    VENDOR_ROOT,
    WORK_ROOT,
    load_encoder_adapter,
    preprocessor_config,
)
from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        default=VENDOR_ROOT / "assets" / "question.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/artifacts/encoder-golden-question",
    )
    parser.add_argument("--mel-frames", type=int, default=80)
    return parser.parse_args()


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    wav_array, sample_rate = sf.read(args.audio, dtype="float32")
    if wav_array.ndim == 2:
        wav_array = wav_array.mean(axis=1)
    wav = torch.from_numpy(wav_array).unsqueeze(0)
    if sample_rate != 16_000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16_000)
    wav_len = torch.tensor([wav.shape[1]], dtype=torch.long)

    prep_conf = preprocessor_config(deterministic=True)
    preprocessor = AudioToMelSpectrogramPreprocessor(**asdict(prep_conf)).eval()

    started = time.perf_counter()
    mel, mel_len = preprocessor(wav, wav_len)
    preprocess_seconds = time.perf_counter() - started

    requested_frames = min(args.mel_frames, int(mel_len[0]))
    mel = mel[:, :, :requested_frames].contiguous()
    mel_len = torch.tensor([requested_frames], dtype=torch.long)

    memory_before_load = rss_bytes()
    load_started = time.perf_counter()
    component = load_encoder_adapter(dtype=torch.float32)
    load_seconds = time.perf_counter() - load_started
    memory_after_load = rss_bytes()

    inference_started = time.perf_counter()
    with torch.inference_mode():
        adapted, encoded_len = component(mel, mel_len)
    inference_seconds = time.perf_counter() - inference_started
    memory_after_inference = rss_bytes()

    torch.save({"mel": mel, "mel_len": mel_len}, args.output_dir / "inputs.pt")
    torch.save(
        {"adapted": adapted, "encoded_len": encoded_len},
        args.output_dir / "outputs.pt",
    )

    summary = {
        "audio": str(args.audio),
        "original_sample_rate": sample_rate,
        "audio_seconds": len(wav_array) / sample_rate,
        "deterministic_preprocessor": True,
        "input_shapes": {"mel": list(mel.shape), "mel_len": list(mel_len.shape)},
        "output_shapes": {
            "adapted": list(adapted.shape),
            "encoded_len": list(encoded_len.shape),
        },
        "output_statistics": {
            "adapted_min": float(adapted.min()),
            "adapted_max": float(adapted.max()),
            "adapted_mean": float(adapted.mean()),
            "adapted_std": float(adapted.std()),
            "encoded_len": encoded_len.tolist(),
        },
        "timings": {
            "preprocess_seconds": preprocess_seconds,
            "component_load_seconds": load_seconds,
            "component_inference_seconds": inference_seconds,
        },
        "memory": {
            "rss_before_component_load_bytes": memory_before_load,
            "rss_after_component_load_bytes": memory_after_load,
            "rss_after_inference_bytes": memory_after_inference,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
