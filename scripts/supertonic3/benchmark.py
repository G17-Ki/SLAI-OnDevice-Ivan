#!/usr/bin/env python3
"""Benchmark Supertonic 3 INT8 through Sherpa-ONNX.

The callback is emitted after each completed text chunk. It is therefore a
measure of time-to-first-playable PCM when the first chunk is played as soon as
it arrives; it is not frame-level vocoder streaming.
"""

from __future__ import annotations

import argparse
import json
import math
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
    {"id": "en_short", "lang": "en", "text": "Sure, I can help with that."},
    {"id": "es_short", "lang": "es", "text": "Claro, puedo ayudarte con eso."},
    {"id": "de_short", "lang": "de", "text": "Natürlich, ich kann dir dabei helfen."},
    {"id": "ja_short", "lang": "ja", "text": "もちろん、お手伝いできます。"},
    {"id": "ko_short", "lang": "ko", "text": "물론이죠, 제가 도와드릴게요."},
    {"id": "ru_short", "lang": "ru", "text": "Конечно, я могу с этим помочь."},
    {
        "id": "en_sentence",
        "lang": "en",
        "text": (
            "The weather should stay clear this afternoon, so you can leave "
            "the umbrella at home."
        ),
    },
    {
        "id": "en_chunked",
        "lang": "en",
        "max_len": 64,
        "text": (
            "I found three nearby cafés that are still open. The closest one "
            "is two minutes away, and it has outdoor seating. Would you like "
            "walking directions?"
        ),
    },
]


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
    supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=str(model_dir / "duration_predictor.int8.onnx"),
        text_encoder=str(model_dir / "text_encoder.int8.onnx"),
        vector_estimator=str(model_dir / "vector_estimator.int8.onnx"),
        vocoder=str(model_dir / "vocoder.int8.onnx"),
        tts_json=str(model_dir / "tts.json"),
        unicode_indexer=str(model_dir / "unicode_indexer.bin"),
        voice_style=str(model_dir / "voice.bin"),
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=supertonic,
            num_threads=threads,
            provider="cpu",
            debug=False,
        )
    )
    if not config.validate():
        raise ValueError("Invalid Supertonic model configuration")
    return sherpa_onnx.OfflineTts(config)


def audio_stats(samples: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    samples = np.asarray(samples, dtype=np.float32)
    return {
        "num_samples": int(samples.size),
        "sample_rate_hz": int(sample_rate),
        "audio_duration_s": float(samples.size / sample_rate),
        "peak_abs": float(np.max(np.abs(samples))),
        "rms": float(np.sqrt(np.mean(np.square(samples)))),
        "dc_mean": float(np.mean(samples)),
        "clip_count": int(np.count_nonzero(np.abs(samples) >= 0.999)),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(samples))),
    }


def synthesize(
    tts,
    case: dict[str, object],
    repeat: int,
    output_dir: Path,
    sid: int,
    steps: int,
    speed: float,
) -> dict[str, object]:
    config = sherpa_onnx.GenerationConfig()
    config.sid = sid
    config.num_steps = steps
    config.speed = speed

    # Do not mutate config.extra with config.extra["lang"] = ... . Pybind
    # returns that map by value, so the mutation is silently lost. Assign the
    # complete map to make non-English synthesis actually take effect.
    extra = {"lang": str(case["lang"]), "seed": "0"}
    if case.get("max_len") is not None:
        extra["max_len"] = str(case["max_len"])
    config.extra = extra

    callback_times: list[float] = []
    callback_sizes: list[int] = []
    first_chunk: np.ndarray | None = None
    start = time.perf_counter()

    def callback(samples: np.ndarray, _progress: float) -> int:
        nonlocal first_chunk
        callback_times.append(time.perf_counter() - start)
        callback_sizes.append(int(samples.size))
        if first_chunk is None:
            first_chunk = np.asarray(samples, dtype=np.float32).copy()
        return 1

    with PeakRss() as rss:
        audio = tts.generate(str(case["text"]), config, callback)
    total_s = time.perf_counter() - start
    samples = np.asarray(audio.samples, dtype=np.float32)
    stats = audio_stats(samples, audio.sample_rate)

    wav_path = None
    first_chunk_path = None
    if repeat == 0:
        wav_path = output_dir / f"{case['id']}.wav"
        sf.write(wav_path, samples, audio.sample_rate, subtype="PCM_16")
        if first_chunk is not None:
            first_chunk_path = output_dir / f"{case['id']}.first_chunk.wav"
            sf.write(first_chunk_path, first_chunk, audio.sample_rate, subtype="PCM_16")

    return {
        "case_id": case["id"],
        "lang": case["lang"],
        "text": case["text"],
        "max_len": case.get("max_len"),
        "repeat_index": repeat,
        "first_callback_s": callback_times[0] if callback_times else None,
        "callback_times_s": callback_times,
        "callback_chunk_samples": callback_sizes,
        "callback_chunk_count": len(callback_times),
        "total_s": total_s,
        "rtf": total_s / float(stats["audio_duration_s"]),
        "peak_rss_bytes": rss.peak,
        "wav_path": str(wav_path) if wav_path else None,
        "first_chunk_wav_path": str(first_chunk_path) if first_chunk_path else None,
        **stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--sid", type=int, default=6)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    cases = CASES if not args.cases_json else json.loads(args.cases_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_start = time.perf_counter()
    with PeakRss() as load_rss:
        tts = create_tts(args.model_dir, args.threads)
    load_s = time.perf_counter() - load_start
    rss_after = process.memory_info().rss

    if args.warmup:
        synthesize(
            tts,
            {"id": "warmup", "lang": "en", "text": "Hello."},
            1,
            args.output_dir,
            args.sid,
            args.steps,
            args.speed,
        )

    runs = []
    for case in cases:
        for repeat in range(args.repeats):
            result = synthesize(
                tts, case, repeat, args.output_dir, args.sid, args.steps, args.speed
            )
            runs.append(result)
            print(
                f"{result['case_id']} run={repeat + 1}: "
                f"first={result['first_callback_s']:.3f}s "
                f"total={result['total_s']:.3f}s rtf={result['rtf']:.3f}",
                flush=True,
            )

    summary = {}
    for case in cases:
        selected = [r for r in runs if r["case_id"] == case["id"]]
        summary[str(case["id"])] = {
            "lang": case["lang"],
            "text": case["text"],
            "max_len": case.get("max_len"),
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
            "callback_chunk_count": sorted(
                {int(r["callback_chunk_count"]) for r in selected}
            ),
        }

    report = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sherpa_onnx": sherpa_onnx.__version__,
            "numpy": np.__version__,
        },
        "settings": {
            "num_threads": args.threads,
            "repeats": args.repeats,
            "num_steps": args.steps,
            "speed": args.speed,
            "sid": args.sid,
            "warmup": args.warmup,
        },
        "model_dir": str(args.model_dir.resolve()),
        "load": {
            "load_s": load_s,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "incremental_rss_bytes": rss_after - rss_before,
            "peak_rss_bytes": load_rss.peak,
        },
        "summary_by_case": summary,
        "runs": runs,
    }
    output_path = args.output_dir / "benchmark.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
