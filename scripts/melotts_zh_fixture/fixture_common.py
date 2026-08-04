#!/usr/bin/env python3
"""Pure NumPy host operations shared by local and remote MeloTTS runs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


MAX_SEQ_LEN = 512
UPSAMPLED_MAX_SEQ_LEN = 1536
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_OVERLAP = 12
DECODER_Z_TIME_DIM = 64
UPSAMPLE_FACTOR = 512
SAMPLE_RATE = 44_100


def generate_path(duration: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """NumPy equivalent of QAIHM 0.58.0 ``generate_path``."""
    if duration.ndim != 3 or mask.ndim != 4:
        raise ValueError("duration must be rank 3 and mask rank 4")
    batch, _, target_len, source_len = mask.shape
    if duration.shape != (batch, 1, source_len):
        raise ValueError("duration and mask source dimensions do not match")
    cumulative = np.cumsum(duration, axis=-1).reshape(batch * source_len)
    positions = np.arange(target_len, dtype=cumulative.dtype)
    path = (positions[None, :] < cumulative[:, None]).astype(mask.dtype)
    path = path.reshape(batch, source_len, target_len)
    previous = np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = path - previous
    return path[:, None].transpose(0, 1, 3, 2) * mask


def decoder_windows(z: np.ndarray, y_length: int) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Build the official 40-new-frame / 12-overlap decoder calls."""
    if z.ndim != 3 or z.shape[2] != UPSAMPLED_MAX_SEQ_LEN:
        raise ValueError(f"z must have shape [batch, channels, {UPSAMPLED_MAX_SEQ_LEN}]")
    if not 1 <= y_length <= UPSAMPLED_MAX_SEQ_LEN:
        raise ValueError("y_length is outside the fixed flow extent")
    windows: list[np.ndarray] = []
    schedule: list[dict[str, Any]] = []
    first = np.zeros((z.shape[0], z.shape[1], DECODER_Z_TIME_DIM), dtype=z.dtype)
    copied = MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP
    first[:, :, :copied] = z[:, :, :copied]
    windows.append(first)
    schedule.append(
        {
            "call_index": 0,
            "source_start_frame": 0,
            "source_end_frame_exclusive": copied,
            "kept_audio_start_sample": 0,
            "kept_audio_end_sample_exclusive": MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR,
        }
    )
    total = MAX_DEC_SEQ_LEN
    while total < y_length:
        start = total - DEC_SEQ_OVERLAP
        end = total + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP
        window = np.zeros((z.shape[0], z.shape[1], DECODER_Z_TIME_DIM), dtype=z.dtype)
        source_end = min(end, z.shape[2])
        copied = max(source_end - start, 0)
        window[:, :, :copied] = z[:, :, start:source_end]
        windows.append(window)
        schedule.append(
            {
                "call_index": len(windows) - 1,
                "source_start_frame": start,
                "source_end_frame_exclusive": source_end,
                "kept_audio_start_sample": DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR,
                "kept_audio_end_sample_exclusive": (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP) * UPSAMPLE_FACTOR,
            }
        )
        total += MAX_DEC_SEQ_LEN
    if len(windows) != math.ceil(y_length / MAX_DEC_SEQ_LEN):
        raise AssertionError("decoder call count does not match the official schedule")
    return windows, schedule


def stitch_decoder_audio(chunks: list[np.ndarray], y_length: int) -> np.ndarray:
    if not chunks:
        raise ValueError("at least one decoder output is required")
    pieces: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        flat = np.asarray(chunk).reshape(-1)
        if flat.size != DECODER_Z_TIME_DIM * UPSAMPLE_FACTOR:
            raise ValueError("decoder output does not contain 32,768 samples")
        start = 0 if index == 0 else DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR
        end = MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR if index == 0 else (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP) * UPSAMPLE_FACTOR
        pieces.append(flat[start:end])
    return np.concatenate(pieces)[: y_length * UPSAMPLE_FACTOR].astype(np.float32, copy=False)


def tensor_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    finite = np.isfinite(value) if np.issubdtype(value.dtype, np.floating) else np.ones(value.shape, dtype=bool)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "finite": bool(finite.all()),
        "min": float(value.min()) if value.size else None,
        "max": float(value.max()) if value.size else None,
    }
