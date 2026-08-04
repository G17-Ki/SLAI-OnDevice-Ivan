#!/usr/bin/env python3
"""Small dependency-light helpers for the Supertonic v81 chain experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON so an interrupted run never leaves partial state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def describe_array(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = (
        bool(np.isfinite(array).all())
        if np.issubdtype(array.dtype, np.number)
        else True
    )
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "min": float(array.min()) if array.size and finite else None,
        "max": float(array.max()) if array.size and finite else None,
        "mean": (
            float(array.astype(np.float64).mean()) if array.size and finite else None
        ),
        "sha256": sha256_array(array),
    }


def compare_arrays(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> dict[str, Any]:
    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)
    result: dict[str, Any] = {
        "actual": describe_array(actual_array),
        "reference": describe_array(reference_array),
        "atol": atol,
        "rtol": rtol,
    }
    if actual_array.shape != reference_array.shape:
        result.update({"passed": False, "reason": "shape_mismatch"})
        return result
    if not result["actual"]["finite"] or not result["reference"]["finite"]:
        result.update({"passed": False, "reason": "non_finite"})
        return result

    actual64 = actual_array.astype(np.float64, copy=False).ravel()
    reference64 = reference_array.astype(np.float64, copy=False).ravel()
    error = actual64 - reference64
    absolute = np.abs(error)
    rmse = float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0
    reference_rms = (
        float(np.sqrt(np.mean(np.square(reference64)))) if reference64.size else 0.0
    )
    denominator = float(np.linalg.norm(actual64) * np.linalg.norm(reference64))
    result.update(
        {
            "passed": bool(
                np.allclose(actual_array, reference_array, atol=atol, rtol=rtol)
            ),
            "max_abs_error": float(absolute.max()) if absolute.size else 0.0,
            "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            "p95_abs_error": (
                float(np.percentile(absolute, 95)) if absolute.size else 0.0
            ),
            "rmse": rmse,
            "reference_rms": reference_rms,
            "normalized_rmse": rmse / reference_rms if reference_rms else 0.0,
            "cosine_similarity": (
                float(np.dot(actual64, reference64) / denominator)
                if denominator
                else 1.0
            ),
        }
    )
    return result


def _stft_magnitude(
    samples: np.ndarray, *, n_fft: int = 1024, hop_length: int = 256
) -> np.ndarray:
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.size))
    frame_count = 1 + max(0, (mono.size - n_fft) // hop_length)
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    frames = np.stack([mono[start : start + n_fft] for start in starts])
    frames *= np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1))


def audio_metrics(
    actual: np.ndarray, reference: np.ndarray, *, sample_rate: int
) -> dict[str, Any]:
    actual_mono = np.asarray(actual, dtype=np.float32).reshape(-1)
    reference_mono = np.asarray(reference, dtype=np.float32).reshape(-1)
    sample_count = min(actual_mono.size, reference_mono.size)
    actual_aligned = actual_mono[:sample_count]
    reference_aligned = reference_mono[:sample_count]
    result = {
        "sample_rate": sample_rate,
        "actual_samples": int(actual_mono.size),
        "reference_samples": int(reference_mono.size),
        "aligned_samples": int(sample_count),
        "waveform": compare_arrays(
            actual_aligned, reference_aligned, atol=5e-2, rtol=5e-2
        ),
    }
    if sample_count == 0:
        return result

    actual_mag = _stft_magnitude(actual_aligned)
    reference_mag = _stft_magnitude(reference_aligned)
    frame_count = min(actual_mag.shape[0], reference_mag.shape[0])
    actual_mag = actual_mag[:frame_count]
    reference_mag = reference_mag[:frame_count]
    difference = actual_mag - reference_mag
    reference_norm = float(np.linalg.norm(reference_mag))
    log_actual = np.log1p(actual_mag)
    log_reference = np.log1p(reference_mag)
    result["spectral"] = {
        "n_fft": 1024,
        "hop_length": 256,
        "frames": int(frame_count),
        "spectral_convergence": (
            float(np.linalg.norm(difference) / reference_norm)
            if reference_norm
            else 0.0
        ),
        "log_magnitude_rmse": float(
            np.sqrt(np.mean(np.square(log_actual - log_reference)))
        ),
        "log_magnitude_mae": float(np.mean(np.abs(log_actual - log_reference))),
    }
    return result


def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes(order="C"))
