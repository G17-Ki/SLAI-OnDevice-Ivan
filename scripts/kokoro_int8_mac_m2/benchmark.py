#!/usr/bin/env python3
"""Bounded Kokoro v1.1-zh INT8 benchmark through Sherpa-ONNX.

Kokoro's callback is emitted after a complete sentence. It is a useful
time-to-first-playable-PCM measurement only when the caller flushes short
clauses/sentences; it is not frame-level vocoder streaming.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import sherpa_onnx
import soundfile as sf


CASES = [
    {
        "id": "en_short",
        "lang": "en",
        "speaker": "af_maple",
        "sid": 0,
        "text": "Sure, I can help with that.",
    },
    {
        "id": "zh_short",
        "lang": "zh",
        "speaker": "zf_001",
        "sid": 3,
        "text": "当然可以，我可以帮你。",
    },
    {
        "id": "en_fact",
        "lang": "en",
        "speaker": "af_maple",
        "sid": 0,
        "text": "Paris is the capital of France.",
    },
    {
        "id": "zh_fact",
        "lang": "zh",
        "speaker": "zf_001",
        "sid": 3,
        "text": "巴黎是法国的首都。",
    },
    {
        "id": "en_chunked",
        "lang": "en",
        "speaker": "af_maple",
        "sid": 0,
        "text": (
            "I found three nearby cafes that are still open. The closest one "
            "is two minutes away. Would you like walking directions?"
        ),
    },
    {
        "id": "zh_chunked",
        "lang": "zh",
        "speaker": "zf_001",
        "sid": 3,
        "text": (
            "我找到三家还在营业的咖啡馆。最近的一家步行只需两分钟。"
            "需要我为你导航吗？"
        ),
    },
]

UNSUPPORTED_LANGUAGE_FACTS = {
    "es": "Not declared by the v1.1-zh model card or embedded ONNX metadata.",
    "ja": "Not declared by the v1.1-zh model card or embedded ONNX metadata.",
}


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(np.asarray(values, dtype=np.float64), 95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


class PeakRss:
    def __init__(self, interval_s: float = 0.005):
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRss":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def create_tts(model_dir: Path, threads: int):
    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=str(model_dir / "model.int8.onnx"),
        voices=str(model_dir / "voices.bin"),
        tokens=str(model_dir / "tokens.txt"),
        data_dir=str(model_dir / "espeak-ng-data"),
        lexicon=",".join(
            map(str, [model_dir / "lexicon-us-en.txt", model_dir / "lexicon-zh.txt"])
        ),
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro,
            num_threads=threads,
            provider="cpu",
            debug=False,
        ),
        rule_fsts=",".join(
            map(
                str,
                [
                    model_dir / "phone-zh.fst",
                    model_dir / "date-zh.fst",
                    model_dir / "number-zh.fst",
                ],
            )
        ),
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError("Invalid Kokoro model configuration")
    return sherpa_onnx.OfflineTts(config)


def pcm_hash(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def audio_stats(
    samples: np.ndarray, sample_rate: int
) -> dict[str, float | int | str | None]:
    samples = np.asarray(samples, dtype=np.float32)
    finite_mask = np.isfinite(samples)
    nonfinite_count = int(np.count_nonzero(~finite_mask))
    finite_samples = samples[finite_mask]
    if finite_samples.size:
        peak_abs: float | None = float(np.max(np.abs(finite_samples)))
        rms: float | None = float(np.sqrt(np.mean(np.square(finite_samples))))
        dc_mean: float | None = float(np.mean(finite_samples))
        clip_count = int(np.count_nonzero(np.abs(finite_samples) >= 0.999))
    else:
        peak_abs = None
        rms = None
        dc_mean = None
        clip_count = 0
    audio_status = (
        "ok"
        if nonfinite_count == 0
        else "all_nonfinite"
        if finite_samples.size == 0
        else "contains_nonfinite"
    )
    return {
        "num_samples": int(samples.size),
        "sample_rate_hz": int(sample_rate),
        "audio_duration_s": float(samples.size / sample_rate),
        "audio_status": audio_status,
        "peak_abs": peak_abs,
        "rms": rms,
        "dc_mean": dc_mean,
        "clip_count": clip_count,
        "nonfinite_count": nonfinite_count,
        "pcm_f32_sha256": pcm_hash(samples),
    }


def synthesize(
    tts,
    case: dict[str, object],
    repeat: int,
    samples_dir: Path,
    preload_rss_bytes: int,
) -> dict[str, object]:
    callback_times: list[float] = []
    callback_sizes: list[int] = []
    callback_progress: list[float] = []
    callback_hashes: list[str] = []
    first_chunk: np.ndarray | None = None
    start = time.perf_counter()

    def callback(samples: np.ndarray, progress: float) -> int:
        nonlocal first_chunk
        chunk = np.asarray(samples, dtype=np.float32)
        callback_times.append(time.perf_counter() - start)
        callback_sizes.append(int(chunk.size))
        callback_progress.append(float(progress))
        callback_hashes.append(pcm_hash(chunk))
        if first_chunk is None:
            first_chunk = chunk.copy()
        # Sherpa's native callback convention uses non-zero to continue.
        return 1

    with PeakRss() as rss:
        audio = tts.generate(
            str(case["text"]),
            sid=int(case["sid"]),
            speed=1.0,
            callback=callback,
        )
    total_s = time.perf_counter() - start
    samples = np.asarray(audio.samples, dtype=np.float32)
    stats = audio_stats(samples, audio.sample_rate)

    wav_path = None
    first_chunk_path = None
    if repeat == 0 and int(stats["nonfinite_count"]) == 0:
        wav_path = samples_dir / f"{case['id']}.wav"
        sf.write(wav_path, samples, audio.sample_rate, subtype="PCM_16")
        if first_chunk is not None:
            first_chunk_path = samples_dir / f"{case['id']}.first_chunk.wav"
            sf.write(first_chunk_path, first_chunk, audio.sample_rate, subtype="PCM_16")

    return {
        "case_id": case["id"],
        "lang": case["lang"],
        "speaker": case["speaker"],
        "sid": case["sid"],
        "text": case["text"],
        "repeat_index": repeat,
        "first_callback_s": callback_times[0] if callback_times else None,
        "callback_times_s": callback_times,
        "callback_chunk_samples": callback_sizes,
        "callback_progress": callback_progress,
        "callback_pcm_f32_sha256": callback_hashes,
        "callback_chunk_count": len(callback_times),
        "first_chunk_pcm_f32_sha256": callback_hashes[0] if callback_hashes else None,
        "total_s": total_s,
        "rtf": total_s / float(stats["audio_duration_s"]),
        "peak_rss_bytes": rss.peak,
        "peak_incremental_vs_preload_bytes": rss.peak - preload_rss_bytes,
        "wav_path": f"samples/{wav_path.name}" if wav_path else None,
        "first_chunk_wav_path": (
            f"samples/{first_chunk_path.name}" if first_chunk_path else None
        ),
        **stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_start = time.perf_counter()
    with PeakRss() as load_rss:
        tts = create_tts(args.model_dir, args.threads)
    load_s = time.perf_counter() - load_start
    rss_after = process.memory_info().rss

    warmups = []
    if args.warmup:
        for case in (
            {"id": "warmup_en", "lang": "en", "speaker": "af_maple", "sid": 0, "text": "Hello."},
            {"id": "warmup_zh", "lang": "zh", "speaker": "zf_001", "sid": 3, "text": "你好。"},
        ):
            warmups.append(synthesize(tts, case, -1, samples_dir, rss_after))

    rss_after_warmup = process.memory_info().rss
    runs = []
    for case in CASES:
        for repeat in range(args.repeats):
            result = synthesize(tts, case, repeat, samples_dir, rss_after_warmup)
            runs.append(result)
            print(
                f"{result['case_id']} run={repeat + 1}: "
                f"first={result['first_callback_s']:.3f}s "
                f"total={result['total_s']:.3f}s rtf={result['rtf']:.3f}",
                flush=True,
            )

    summary = {}
    for case in CASES:
        selected = [r for r in runs if r["case_id"] == case["id"]]
        hashes = sorted({str(r["pcm_f32_sha256"]) for r in selected})
        first_hashes = sorted(
            {str(r["first_chunk_pcm_f32_sha256"]) for r in selected}
        )
        summary[str(case["id"])] = {
            "lang": case["lang"],
            "speaker": case["speaker"],
            "sid": case["sid"],
            "text": case["text"],
            "first_callback_s": distribution(
                [float(r["first_callback_s"]) for r in selected]
            ),
            "total_s": distribution([float(r["total_s"]) for r in selected]),
            "rtf": distribution([float(r["rtf"]) for r in selected]),
            "audio_duration_s": distribution(
                [float(r["audio_duration_s"]) for r in selected]
            ),
            "peak_rss_bytes": distribution(
                [float(r["peak_rss_bytes"]) for r in selected]
            ),
            "peak_incremental_vs_preload_bytes": distribution(
                [float(r["peak_incremental_vs_preload_bytes"]) for r in selected]
            ),
            "callback_chunk_count": sorted(
                {int(r["callback_chunk_count"]) for r in selected}
            ),
            "unique_full_pcm_hashes": hashes,
            "unique_first_chunk_pcm_hashes": first_hashes,
            "bit_identical_across_repeats": len(hashes) == 1 and len(first_hashes) == 1,
        }

    report = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "Apple Silicon host CPU benchmark; not phone or NPU evidence",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sherpa_onnx": sherpa_onnx.__version__,
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "soundfile": sf.__version__,
        },
        "settings": {
            "num_threads": args.threads,
            "repeats": args.repeats,
            "speed": 1.0,
            "warmup": args.warmup,
            "max_num_sentences": 1,
            "resident_model_instances": 1,
        },
        "model_artifact": args.model_dir.name,
        "declared_supported_languages": ["en", "zh"],
        "unsupported_language_facts": UNSUPPORTED_LANGUAGE_FACTS,
        "load": {
            "load_s": load_s,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "incremental_rss_bytes": rss_after - rss_before,
            "peak_rss_bytes": load_rss.peak,
            "rss_after_warmup_bytes": rss_after_warmup,
        },
        "warmups": warmups,
        "warmup_failures": [
            {
                "case_id": warmup["case_id"],
                "lang": warmup["lang"],
                "text": warmup["text"],
                "audio_status": warmup["audio_status"],
                "num_samples": warmup["num_samples"],
                "nonfinite_count": warmup["nonfinite_count"],
                "pcm_f32_sha256": warmup["pcm_f32_sha256"],
            }
            for warmup in warmups
            if int(warmup["nonfinite_count"]) > 0
        ],
        "measured_output_gate": {
            "all_measured_case_outputs_finite": all(
                int(run["nonfinite_count"]) == 0 for run in runs
            ),
            "all_measured_case_outputs_unclipped": all(
                int(run["clip_count"]) == 0 for run in runs
            ),
        },
        "summary_by_case": summary,
        "runs": runs,
    }
    output_path = args.output_dir / "benchmark.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
