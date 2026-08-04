#!/usr/bin/env python3
"""Measure warm latency, memory, language detection, WER, and CER."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import psutil
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio


CHAR_PRIMARY_LANGUAGES = {"ja", "ko", "zh"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return cleaned.split()


def normalize_chars(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        character
        for character in normalized
        if not character.isspace()
        and unicodedata.category(character)[0] not in {"P", "S"}
    ]


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def error_record(reference: str, hypothesis: str) -> dict[str, object]:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    reference_chars = normalize_chars(reference)
    hypothesis_chars = normalize_chars(hypothesis)
    word_edits = edit_distance(reference_words, hypothesis_words)
    char_edits = edit_distance(reference_chars, hypothesis_chars)
    return {
        "normalized_reference_words": " ".join(reference_words),
        "normalized_hypothesis_words": " ".join(hypothesis_words),
        "normalized_reference_chars": "".join(reference_chars),
        "normalized_hypothesis_chars": "".join(hypothesis_chars),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "wer": word_edits / max(len(reference_words), 1),
        "char_edits": char_edits,
        "reference_chars": len(reference_chars),
        "cer": char_edits / max(len(reference_chars), 1),
    }


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class PeakRssSampler:
    def __init__(self, interval_s: float = 0.005) -> None:
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.peak_bytes = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRssSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def transcribe_once(
    model: WhisperModel,
    audio,
    requested_language: str | None,
    decoder_profile: str,
) -> tuple[dict[str, object], int]:
    if decoder_profile == "fast_greedy":
        decode_options = {
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "compression_ratio_threshold": None,
            "log_prob_threshold": None,
            "no_speech_threshold": None,
            "without_timestamps": True,
            "max_new_tokens": 128,
        }
    else:
        decode_options = {
            "beam_size": 5,
            "best_of": 5,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "without_timestamps": False,
            "max_new_tokens": None,
        }
    start_ns = time.perf_counter_ns()
    with PeakRssSampler() as rss_sampler:
        segments, info = model.transcribe(
            audio,
            language=requested_language,
            task="transcribe",
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
            **decode_options,
        )
        api_return_ns = time.perf_counter_ns()
        first_segment_ns = None
        transcript_parts = []
        segment_count = 0
        for segment in segments:
            if first_segment_ns is None:
                first_segment_ns = time.perf_counter_ns()
            segment_count += 1
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)
        final_ns = time.perf_counter_ns()
    if first_segment_ns is None:
        first_segment_ns = final_ns
    transcript = " ".join(transcript_parts).strip()
    return (
        {
            "requested_language": requested_language,
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "transcript": transcript,
            "segment_count": segment_count,
            "api_return_ms": (api_return_ns - start_ns) / 1e6,
            "first_segment_ms": (first_segment_ns - start_ns) / 1e6,
            "final_ms": (final_ns - start_ns) / 1e6,
            "peak_rss_mb": rss_sampler.peak_bytes / (1024 * 1024),
        },
        rss_sampler.peak_bytes,
    )


def aggregate_quality(runs: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for mode in sorted({str(run["mode"]) for run in runs}):
        first_repeats = [
            run for run in runs if run["mode"] == mode and run["repeat_index"] == 0
        ]
        language_records: dict[str, object] = {}
        for language in sorted({str(run["language"]) for run in first_repeats}):
            language_runs = [run for run in first_repeats if run["language"] == language]
            word_edits = sum(int(run["errors"]["word_edits"]) for run in language_runs)
            reference_words = sum(
                int(run["errors"]["reference_words"]) for run in language_runs
            )
            char_edits = sum(int(run["errors"]["char_edits"]) for run in language_runs)
            reference_chars = sum(
                int(run["errors"]["reference_chars"]) for run in language_runs
            )
            primary_metric = "cer" if language in CHAR_PRIMARY_LANGUAGES else "wer"
            language_records[language] = {
                "samples": len(language_runs),
                "wer": word_edits / max(reference_words, 1),
                "word_edits": word_edits,
                "reference_words": reference_words,
                "cer": char_edits / max(reference_chars, 1),
                "char_edits": char_edits,
                "reference_chars": reference_chars,
                "primary_metric": primary_metric,
                "primary_error_rate": (
                    char_edits / max(reference_chars, 1)
                    if primary_metric == "cer"
                    else word_edits / max(reference_words, 1)
                ),
            }

        latin = [run for run in first_repeats if run["language"] not in CHAR_PRIMARY_LANGUAGES]
        char_primary = [run for run in first_repeats if run["language"] in CHAR_PRIMARY_LANGUAGES]
        latin_word_edits = sum(int(run["errors"]["word_edits"]) for run in latin)
        latin_reference_words = sum(int(run["errors"]["reference_words"]) for run in latin)
        cjk_char_edits = sum(int(run["errors"]["char_edits"]) for run in char_primary)
        cjk_reference_chars = sum(int(run["errors"]["reference_chars"]) for run in char_primary)
        mode_record = {
            "languages": language_records,
            "latin_micro_wer": latin_word_edits / max(latin_reference_words, 1),
            "char_primary_micro_cer": cjk_char_edits / max(cjk_reference_chars, 1),
            "macro_primary_error_rate": statistics.mean(
                record["primary_error_rate"] for record in language_records.values()
            ),
        }
        if mode == "auto":
            mode_record["language_detection_accuracy"] = sum(
                run["detected_language"] == run["language"] for run in first_repeats
            ) / max(len(first_repeats), 1)
        output[mode] = mode_record
    return output


def aggregate_latency(runs: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for mode in sorted({str(run["mode"]) for run in runs}):
        mode_runs = [run for run in runs if run["mode"] == mode]
        mode_output: dict[str, object] = {}
        for metric in ("api_return_ms", "first_segment_ms", "final_ms", "rtf"):
            values = [float(run[metric]) for run in mode_runs]
            mode_output[metric] = {
                "median": statistics.median(values),
                "p95": percentile(values, 95),
                "min": min(values),
                "max": max(values),
            }
        per_language = {}
        for language in sorted({str(run["language"]) for run in mode_runs}):
            language_runs = [run for run in mode_runs if run["language"] == language]
            per_language[language] = {
                "first_segment_ms_median": statistics.median(
                    float(run["first_segment_ms"]) for run in language_runs
                ),
                "final_ms_median": statistics.median(
                    float(run["final_ms"]) for run in language_runs
                ),
                "rtf_median": statistics.median(float(run["rtf"]) for run in language_runs),
            }
        mode_output["per_language"] = per_language
        output[mode] = mode_output
    return output


def sysctl_value(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=("fixed", "auto"), default=["fixed", "auto"])
    parser.add_argument(
        "--decoder-profile",
        choices=("fast_greedy", "quality_beam5"),
        default="fast_greedy",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    sample_root = manifest_path.parent
    decoded_samples = []
    for sample in manifest["samples"]:
        audio_path = sample_root / sample["audio_path"]
        decode_start = time.perf_counter_ns()
        audio = decode_audio(str(audio_path), sampling_rate=16000)
        decode_ms = (time.perf_counter_ns() - decode_start) / 1e6
        decoded_samples.append(
            {
                **sample,
                "audio": audio,
                "duration_decoded_s": len(audio) / 16000.0,
                "decode_ms": decode_ms,
            }
        )

    process = psutil.Process()
    gc.collect()
    rss_before_model_bytes = process.memory_info().rss
    load_start = time.perf_counter_ns()
    model = WhisperModel(
        str(args.model_dir.resolve()),
        device="cpu",
        compute_type="int8",
        cpu_threads=args.cpu_threads,
        num_workers=1,
        local_files_only=True,
    )
    load_ms = (time.perf_counter_ns() - load_start) / 1e6
    rss_after_model_bytes = process.memory_info().rss

    warmup_records = []
    warmup_sample = decoded_samples[0]
    for mode in args.modes:
        requested_language = warmup_sample["language"] if mode == "fixed" else None
        warmup_result, _ = transcribe_once(
            model,
            warmup_sample["audio"],
            requested_language,
            args.decoder_profile,
        )
        warmup_records.append({"mode": mode, **warmup_result})

    runs = []
    process_peak_bytes = max(rss_before_model_bytes, rss_after_model_bytes)
    for mode in args.modes:
        for sample in decoded_samples:
            for repeat_index in range(args.repeats):
                requested_language = sample["language"] if mode == "fixed" else None
                result, peak_bytes = transcribe_once(
                    model,
                    sample["audio"],
                    requested_language,
                    args.decoder_profile,
                )
                process_peak_bytes = max(process_peak_bytes, peak_bytes)
                errors = error_record(sample["reference"], result["transcript"])
                run = {
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "sample_id": sample["sample_id"],
                    "language": sample["language"],
                    "reference": sample["reference"],
                    "audio_sha256": sample["audio_sha256"],
                    "duration_s": sample["duration_decoded_s"],
                    "rtf": result["final_ms"] / 1000.0 / sample["duration_decoded_s"],
                    **result,
                    "errors": errors,
                }
                runs.append(run)
                print(
                    f"{args.model_name} {mode} {sample['sample_id']} "
                    f"r{repeat_index}: {result['final_ms']:.1f} ms "
                    f"{result['transcript']!r}",
                    flush=True,
                )

    model_files = []
    for path in sorted(args.model_dir.resolve().iterdir()):
        if path.is_file():
            model_files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )

    output = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_profile": {
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": args.cpu_threads,
            "num_workers": 1,
            "repeats": args.repeats,
            "modes": args.modes,
            "decoder_profile": args.decoder_profile,
            "decoder": (
                "greedy; temperature=0; no fallback; no timestamps"
                if args.decoder_profile == "fast_greedy"
                else "beam size 5; standard fallback thresholds; timestamps"
            ),
            "audio_input": "pre-decoded mono float32 PCM at 16 kHz",
            "latency_scope": "model feature extraction through stable segment/final transcript; file decode excluded",
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "macos_product": platform.mac_ver()[0],
            "hardware_model": sysctl_value("hw.model"),
            "chip_brand": sysctl_value("machdep.cpu.brand_string"),
            "physical_memory_bytes": int(sysctl_value("hw.memsize") or 0),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("faster-whisper", "ctranslate2", "numpy", "psutil", "av")
        },
        "model": {
            "name": args.model_name,
            "repo_id": args.model_repo,
            "revision": args.model_revision,
            "files": model_files,
            "total_file_bytes": sum(record["bytes"] for record in model_files),
            "load_ms": load_ms,
            "rss_before_model_mb": rss_before_model_bytes / (1024 * 1024),
            "rss_after_model_mb": rss_after_model_bytes / (1024 * 1024),
            "rss_after_load_increment_mb": (
                rss_after_model_bytes - rss_before_model_bytes
            )
            / (1024 * 1024),
            "peak_rss_mb": process_peak_bytes / (1024 * 1024),
            "incremental_peak_rss_mb": (
                process_peak_bytes - rss_before_model_bytes
            )
            / (1024 * 1024),
        },
        "dataset": manifest["dataset"],
        "sample_count": len(decoded_samples),
        "audio_total_s": sum(sample["duration_decoded_s"] for sample in decoded_samples),
        "decode_ms": [
            {
                "sample_id": sample["sample_id"],
                "decode_ms": sample["decode_ms"],
                "duration_decoded_s": sample["duration_decoded_s"],
            }
            for sample in decoded_samples
        ],
        "warmups": warmup_records,
        "latency": aggregate_latency(runs),
        "quality": aggregate_quality(runs),
        "runs": runs,
        "caveats": [
            "This is a 21-clip diagnostic fixture, not a statistically powered ASR benchmark.",
            "Faster-Whisper INT8 on a Mac CPU is not the Qualcomm AI Hub graph or an HTP-v81 measurement.",
            "first_segment_ms is time to the first stable Faster-Whisper segment, not internal decoder first-token latency.",
            "The fixed-language mode assumes the app already knows the selected user language.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
