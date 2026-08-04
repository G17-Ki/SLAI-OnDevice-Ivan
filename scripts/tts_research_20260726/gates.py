#!/usr/bin/env python3
"""Waveform validity gates shared by every candidate in the 2026-07-26 run.

These gates are deliberately fail-closed. A candidate that cannot produce a
finite, non-degenerate waveform is rejected before any latency or
intelligibility number is allowed to influence a ranking, and non-finite PCM is
never written out as if it were valid audio.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# Thresholds are documented here rather than inlined so a reviewer can see the
# whole acceptance surface in one place.
CLIP_LEVEL = 0.999
MIN_RMS = 0.005          # below this the "audio" is effectively silence
# A DC offset is a real defect (wasted headroom, thump on playback) but it is
# removable with a one-pole DC blocker, so it is not treated as a correctness
# failure until it is large enough to eat meaningful headroom. The warning band
# starts at 1% of full scale, which is the conventional "significant DC" level.
WARN_ABS_DC = 0.01
MAX_ABS_DC = 0.10
MAX_SILENCE_S = 1.2      # longest internal gap tolerated inside one utterance
SILENCE_LEVEL = 1e-4     # |x| below this counts as silence
MAX_CLIP_FRACTION = 0.001
REPEAT_WINDOW_S = 0.25   # collapse detector window
REPEAT_CORR = 0.995      # normalized correlation that counts as a repeat
REPEAT_MAX_RUN = 6       # consecutive identical windows tolerated


@dataclass
class GateResult:
    passed: bool
    hard_failures: list
    warnings: list
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _longest_silence_seconds(samples: np.ndarray, sample_rate: int) -> float:
    quiet = np.abs(samples) < SILENCE_LEVEL
    if not quiet.any():
        return 0.0
    # Run-length encode the quiet mask.
    changes = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    bounds = np.concatenate(([0], changes + 1, [quiet.size]))
    longest = 0
    for start, end in zip(bounds[:-1], bounds[1:]):
        if quiet[start]:
            longest = max(longest, end - start)
    return float(longest / sample_rate)


def _repeated_pattern_run(samples: np.ndarray, sample_rate: int) -> int:
    """Longest run of consecutive near-identical windows.

    A vocoder that has collapsed into a loop emits the same short window over
    and over. Normalized correlation catches that without being fooled by an
    ordinary sustained vowel, which still drifts in amplitude and phase.
    """
    window = int(REPEAT_WINDOW_S * sample_rate)
    if window <= 0 or samples.size < window * 3:
        return 0
    count = samples.size // window
    windows = samples[: count * window].reshape(count, window).astype(np.float64)
    norms = np.linalg.norm(windows, axis=1)
    best_run = 0
    run = 0
    for i in range(1, count):
        denom = norms[i] * norms[i - 1]
        if denom <= 0:
            run = 0
            continue
        corr = float(np.dot(windows[i], windows[i - 1]) / denom)
        if corr >= REPEAT_CORR:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return best_run


def evaluate(samples, sample_rate: int) -> GateResult:
    """Run every waveform gate. Never raises on degenerate input."""
    arr = np.asarray(samples, dtype=np.float32)
    hard: list = []
    warn: list = []

    if arr.size == 0:
        return GateResult(
            passed=False,
            hard_failures=["empty_waveform"],
            warnings=[],
            metrics={"num_samples": 0, "sample_rate_hz": int(sample_rate)},
        )

    finite_mask = np.isfinite(arr)
    nonfinite = int(np.count_nonzero(~finite_mask))
    # Every downstream statistic is computed on the finite subset so a single
    # NaN cannot poison the metrics and hide a second, different defect.
    finite = arr[finite_mask]
    if finite.size == 0:
        return GateResult(
            passed=False,
            hard_failures=["all_samples_non_finite"],
            warnings=[],
            metrics={
                "num_samples": int(arr.size),
                "nonfinite_count": nonfinite,
                "sample_rate_hz": int(sample_rate),
            },
        )

    duration_s = float(arr.size / sample_rate)
    peak = float(np.max(np.abs(finite)))
    rms = float(np.sqrt(np.mean(np.square(finite.astype(np.float64)))))
    dc = float(np.mean(finite.astype(np.float64)))
    clip_count = int(np.count_nonzero(np.abs(finite) >= CLIP_LEVEL))
    clip_fraction = clip_count / float(finite.size)
    longest_silence_s = _longest_silence_seconds(finite, sample_rate)
    repeat_run = _repeated_pattern_run(finite, sample_rate)

    if nonfinite:
        hard.append(f"non_finite_pcm:{nonfinite}")
    if rms < MIN_RMS:
        hard.append(f"near_silent_rms:{rms:.6f}")
    if clip_fraction > MAX_CLIP_FRACTION:
        hard.append(f"clipping_fraction:{clip_fraction:.6f}")
    if repeat_run >= REPEAT_MAX_RUN:
        hard.append(f"repeated_pattern_run:{repeat_run}")
    if abs(dc) > MAX_ABS_DC:
        hard.append(f"severe_dc_offset:{dc:.6f}")
    elif abs(dc) > WARN_ABS_DC:
        warn.append(f"dc_offset_needs_blocker:{dc:.6f}")
    if longest_silence_s > MAX_SILENCE_S:
        hard.append(f"internal_silence_s:{longest_silence_s:.3f}")

    if peak > 0.99:
        warn.append(f"peak_near_full_scale:{peak:.4f}")
    if peak < 0.05:
        warn.append(f"very_low_peak:{peak:.4f}")
    if duration_s < 0.15:
        warn.append(f"very_short_output_s:{duration_s:.3f}")

    return GateResult(
        passed=not hard,
        hard_failures=hard,
        warnings=warn,
        metrics={
            "num_samples": int(arr.size),
            "sample_rate_hz": int(sample_rate),
            "audio_duration_s": duration_s,
            "nonfinite_count": nonfinite,
            "peak_abs": peak,
            "rms": rms,
            "dc_mean": dc,
            "clip_count": clip_count,
            "clip_fraction": clip_fraction,
            "dc_over_rms": float(dc / rms) if rms > 0 else None,
            "longest_silence_s": longest_silence_s,
            "repeated_window_run": repeat_run,
        },
    )
