#!/usr/bin/env python3
"""Reconstruct host-side waveforms from remote detokenizer neural outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from liquid_audio.detokenizer import ISTFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cosine(actual: np.ndarray, golden: np.ndarray) -> float:
    a = actual.astype(np.float64).ravel()
    g = golden.astype(np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(g)
    return float(np.dot(a, g) / denominator) if denominator else 1.0


def waveform_metrics(actual: np.ndarray, golden: np.ndarray) -> dict[str, float]:
    a = actual.astype(np.float64).ravel()
    g = golden.astype(np.float64).ravel()
    delta = a - g
    rmse = float(np.sqrt(np.mean(delta * delta)))
    golden_rms = float(np.sqrt(np.mean(g * g)))
    projection = float(np.dot(a, g) / np.dot(g, g)) * g if np.dot(g, g) else g
    noise = a - projection
    si_sdr = 10.0 * np.log10(
        (np.dot(projection, projection) + 1e-20) / (np.dot(noise, noise) + 1e-20)
    )
    return {
        "cosine_similarity": cosine(a, g),
        "rmse": rmse,
        "golden_rms": golden_rms,
        "normalized_rmse": rmse / golden_rms if golden_rms else 0.0,
        "si_sdr_db": float(si_sdr),
        "actual_peak_abs": float(np.max(np.abs(a))),
        "golden_peak_abs": float(np.max(np.abs(g))),
    }


def reconstruct(log_abs: np.ndarray, angle: np.ndarray) -> np.ndarray:
    istft = ISTFT(1280, 320, 1280, padding="same").eval()
    with torch.inference_mode():
        magnitude = torch.from_numpy(log_abs.astype(np.float32)).exp()
        phase = torch.from_numpy(angle.astype(np.float32))
        waveform = istft(torch.polar(magnitude, phase))
    return waveform.numpy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.remote_outputs) as archive:
        actual_log = archive["actual__log_abs"]
        golden_log = archive["golden__log_abs"]
        actual_angle = archive["actual__angle"]
        golden_angle = archive["golden__angle"]

    actual_wave = reconstruct(actual_log, actual_angle)
    golden_wave = reconstruct(golden_log, golden_angle)
    wrapped_phase = np.angle(np.exp(1j * (actual_angle - golden_angle)))
    actual_spec = np.exp(actual_log.astype(np.float64) + 1j * actual_angle.astype(np.float64))
    golden_spec = np.exp(golden_log.astype(np.float64) + 1j * golden_angle.astype(np.float64))

    sf.write(args.output_dir / "remote.wav", actual_wave[0], 24_000, subtype="FLOAT")
    sf.write(args.output_dir / "golden.wav", golden_wave[0], 24_000, subtype="FLOAT")
    report = {
        "remote_outputs": str(args.remote_outputs),
        "waveform_shape": list(actual_wave.shape),
        "waveform_seconds": float(actual_wave.shape[-1] / 24_000),
        "waveform": waveform_metrics(actual_wave, golden_wave),
        "wrapped_phase_mean_abs_radians": float(np.mean(np.abs(wrapped_phase))),
        "wrapped_phase_p95_abs_radians": float(np.percentile(np.abs(wrapped_phase), 95)),
        "complex_spectrogram_normalized_rmse": float(
            np.linalg.norm(actual_spec - golden_spec) / np.linalg.norm(golden_spec)
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
