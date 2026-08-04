#!/usr/bin/env python3
"""Benchmark pinned SenseVoiceSmall INT8 with fixed locales and bounded resources."""

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
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import psutil


SUPPORTED_LANGUAGES = ("en", "ja", "ko", "zh")
CHAR_PRIMARY_LANGUAGES = {"ja", "ko", "zh"}
MODEL_ARCHIVE_NAME = (
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
)
MODEL_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    + MODEL_ARCHIVE_NAME
)
MODEL_ARCHIVE_SHA256 = (
    "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e"
)
MODEL_RELEASE_ASSET_ID = 288366523
MODEL_RELEASE_ASSET_CREATED_UTC = "2025-09-01T08:32:52Z"


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


def decode_audio(path: Path) -> tuple[np.ndarray, float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "pipe:1",
    ]
    start_ns = time.perf_counter_ns()
    raw = subprocess.check_output(command)
    decode_ms = (time.perf_counter_ns() - start_ns) / 1e6
    return np.frombuffer(raw, dtype="<f4").copy(), decode_ms


def serializable_result_metadata(result: object) -> dict[str, object]:
    output: dict[str, object] = {"repr": str(result)}
    for name in ("lang", "emotion", "event", "tokens", "timestamps", "json"):
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        try:
            json.dumps(value)
            output[name] = value
        except TypeError:
            output[name] = str(value)
    return output


def decode_once(recognizer: object, audio: np.ndarray) -> dict[str, object]:
    start_ns = time.perf_counter_ns()
    with PeakRssSampler() as rss_sampler:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        result = stream.result
        transcript_value = getattr(result, "text", result)
        if callable(transcript_value):
            transcript_value = transcript_value()
        transcript = str(transcript_value).strip()
        metadata = serializable_result_metadata(result)
    final_ns = time.perf_counter_ns()
    final_ms = (final_ns - start_ns) / 1e6
    return {
        "transcript": transcript,
        "first_stable_ms": final_ms,
        "final_ms": final_ms,
        "peak_rss_mb": rss_sampler.peak_bytes / (1024 * 1024),
        "result_metadata": metadata,
    }


def worker_main(args: argparse.Namespace) -> int:
    import sherpa_onnx

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    sample_root = manifest_path.parent
    selected = [
        sample
        for sample in manifest["samples"]
        if sample["language"] == args.worker_language
    ]
    if len(selected) != 3:
        raise ValueError(
            f"expected three {args.worker_language} samples, found {len(selected)}"
        )

    decoded_samples = []
    for sample in selected:
        audio_path = sample_root / sample["audio_path"]
        audio, decode_ms = decode_audio(audio_path)
        decoded_samples.append(
            {
                **sample,
                "audio": audio,
                "decode_ms": decode_ms,
                "duration_decoded_s": len(audio) / 16000.0,
            }
        )

    process = psutil.Process()
    gc.collect()
    rss_before_model_bytes = process.memory_info().rss
    with PeakRssSampler() as global_peak:
        load_start_ns = time.perf_counter_ns()
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str((args.model_dir / "model.int8.onnx").resolve()),
            tokens=str((args.model_dir / "tokens.txt").resolve()),
            num_threads=args.cpu_threads,
            provider="cpu",
            language=args.worker_language,
            use_itn=True,
            debug=False,
        )
        load_ms = (time.perf_counter_ns() - load_start_ns) / 1e6
        rss_after_model_bytes = process.memory_info().rss

        warmup = decode_once(recognizer, decoded_samples[0]["audio"])
        runs = []
        for sample in decoded_samples:
            for repeat_index in range(args.repeats):
                result = decode_once(recognizer, sample["audio"])
                errors = error_record(sample["reference"], result["transcript"])
                run = {
                    "mode": "fixed",
                    "repeat_index": repeat_index,
                    "sample_id": sample["sample_id"],
                    "language": sample["language"],
                    "requested_language": args.worker_language,
                    "reference": sample["reference"],
                    "audio_sha256": sample["audio_sha256"],
                    "duration_s": sample["duration_decoded_s"],
                    "rtf": result["final_ms"] / 1000.0 / sample["duration_decoded_s"],
                    **result,
                    "errors": errors,
                }
                runs.append(run)
                print(
                    f"SenseVoiceSmall {args.worker_language} {sample['sample_id']} "
                    f"r{repeat_index}: {result['final_ms']:.1f} ms "
                    f"{result['transcript']!r}",
                    flush=True,
                )

    output = {
        "language": args.worker_language,
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
        "load_ms": load_ms,
        "rss_before_model_mb": rss_before_model_bytes / (1024 * 1024),
        "rss_after_model_mb": rss_after_model_bytes / (1024 * 1024),
        "rss_after_load_increment_mb": (
            rss_after_model_bytes - rss_before_model_bytes
        )
        / (1024 * 1024),
        "peak_rss_mb": global_peak.peak_bytes / (1024 * 1024),
        "incremental_peak_rss_mb": (
            global_peak.peak_bytes - rss_before_model_bytes
        )
        / (1024 * 1024),
        "warmup": warmup,
        "runs": runs,
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return 0


def aggregate_quality(runs: list[dict[str, object]]) -> dict[str, object]:
    first_repeats = [run for run in runs if run["repeat_index"] == 0]
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
    cjk = [run for run in first_repeats if run["language"] in CHAR_PRIMARY_LANGUAGES]
    latin_word_edits = sum(int(run["errors"]["word_edits"]) for run in latin)
    latin_reference_words = sum(int(run["errors"]["reference_words"]) for run in latin)
    cjk_char_edits = sum(int(run["errors"]["char_edits"]) for run in cjk)
    cjk_reference_chars = sum(int(run["errors"]["reference_chars"]) for run in cjk)
    return {
        "fixed": {
            "languages": language_records,
            "latin_micro_wer": latin_word_edits / max(latin_reference_words, 1),
            "char_primary_micro_cer": cjk_char_edits / max(cjk_reference_chars, 1),
            "macro_primary_error_rate": statistics.mean(
                record["primary_error_rate"] for record in language_records.values()
            ),
        }
    }


def aggregate_latency(runs: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for metric in ("first_stable_ms", "final_ms", "rtf"):
        values = [float(run[metric]) for run in runs]
        output[metric] = {
            "median": statistics.median(values),
            "p95": percentile(values, 95),
            "min": min(values),
            "max": max(values),
        }
    output["per_language"] = {}
    for language in sorted({str(run["language"]) for run in runs}):
        language_runs = [run for run in runs if run["language"] == language]
        output["per_language"][language] = {
            "final_ms_median": statistics.median(
                float(run["final_ms"]) for run in language_runs
            ),
            "final_ms_p95": percentile(
                [float(run["final_ms"]) for run in language_runs], 95
            ),
            "rtf_median": statistics.median(
                float(run["rtf"]) for run in language_runs
            ),
        }
    return {"fixed": output}


def sysctl_value(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--languages", nargs="+", choices=SUPPORTED_LANGUAGES, default=list(SUPPORTED_LANGUAGES)
    )
    parser.add_argument("--worker-language", choices=SUPPORTED_LANGUAGES)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.worker_language:
        if args.worker_output is None:
            parser.error("--worker-output is required in worker mode")
        return worker_main(args)

    model_file = args.model_dir / "model.int8.onnx"
    tokens_file = args.model_dir / "tokens.txt"
    for required in (model_file, tokens_file):
        if not required.is_file():
            raise FileNotFoundError(required)

    worker_records = []
    with tempfile.TemporaryDirectory(prefix="sensevoice-benchmark-") as temp_dir:
        for language in args.languages:
            worker_output = Path(temp_dir) / f"{language}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--model-dir",
                str(args.model_dir.resolve()),
                "--output",
                str(args.output.resolve()),
                "--repeats",
                str(args.repeats),
                "--cpu-threads",
                str(args.cpu_threads),
                "--worker-language",
                language,
                "--worker-output",
                str(worker_output),
            ]
            subprocess.run(command, check=True)
            worker_records.append(json.loads(worker_output.read_text()))

    runs = [run for worker in worker_records for run in worker["runs"]]
    manifest = json.loads(args.manifest.resolve().read_text())
    model_files = []
    for path in sorted(args.model_dir.resolve().iterdir()):
        if path.is_file():
            model_files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )

    archive_record = {
        "name": MODEL_ARCHIVE_NAME,
        "url": MODEL_ARCHIVE_URL,
        "github_release_tag": "asr-models",
        "github_asset_id": MODEL_RELEASE_ASSET_ID,
        "asset_created_utc": MODEL_RELEASE_ASSET_CREATED_UTC,
        "expected_sha256": MODEL_ARCHIVE_SHA256,
    }
    if args.archive and args.archive.is_file():
        archive_record.update(
            {
                "local_bytes": args.archive.stat().st_size,
                "local_sha256": sha256(args.archive),
            }
        )

    output = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_profile": {
            "device": "cpu",
            "runtime": "sherpa-onnx",
            "model_quantization": "dynamic UINT8 weights for MatMul (archive declaration)",
            "cpu_threads": args.cpu_threads,
            "worker_processes_concurrent": 1,
            "repeats": args.repeats,
            "mode": "fixed locale",
            "languages": args.languages,
            "use_itn": True,
            "decoder": "non-autoregressive CTC greedy search",
            "audio_input": "pre-decoded mono float32 PCM at 16 kHz",
            "latency_scope": (
                "stream creation, waveform acceptance, feature extraction, model decode, "
                "and final transcript; file decode excluded"
            ),
            "partial_output": "none; first_stable_ms equals final_ms",
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
            "ffmpeg": subprocess.check_output(
                ["ffmpeg", "-version"], text=True
            ).splitlines()[0],
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("sherpa-onnx", "numpy", "psutil")
        },
        "model": {
            "name": "SenseVoiceSmall INT8",
            "source_model": "iic/SenseVoiceSmall",
            "source_mirror": "FunAudioLLM/SenseVoiceSmall",
            "source_revision": None,
            "source_revision_status": (
                "not declared by the dated Sherpa-ONNX conversion archive; "
                "the archive SHA-256 pins the executable artifact"
            ),
            "conversion": "k2-fsa/sherpa-onnx dated SenseVoice INT8 asset",
            "archive": archive_record,
            "files": model_files,
            "total_file_bytes": sum(record["bytes"] for record in model_files),
            "load_ms_median": statistics.median(
                worker["load_ms"] for worker in worker_records
            ),
            "load_ms_by_language": {
                worker["language"]: worker["load_ms"] for worker in worker_records
            },
            "rss_after_load_increment_mb_max": max(
                worker["rss_after_load_increment_mb"] for worker in worker_records
            ),
            "incremental_peak_rss_mb": max(
                worker["incremental_peak_rss_mb"] for worker in worker_records
            ),
            "memory_by_language": {
                worker["language"]: {
                    key: worker[key]
                    for key in (
                        "rss_before_model_mb",
                        "rss_after_model_mb",
                        "rss_after_load_increment_mb",
                        "peak_rss_mb",
                        "incremental_peak_rss_mb",
                    )
                }
                for worker in worker_records
            },
            "weights_license": {
                "name": "FunASR Model Open Source License Agreement v1.1",
                "spdx": None,
                "url": "https://github.com/modelscope/FunASR/blob/58830eca4012644aac0c3218c3ccc7d98f003fda/MODEL_LICENSE",
                "notes": (
                    "custom model license; attribution and model-name retention required; "
                    "not represented as an OSI-approved license"
                ),
            },
            "runtime_license": {
                "name": "Apache-2.0",
                "url": "https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE",
            },
        },
        "dataset": manifest["dataset"],
        "sample_count": sum(worker["sample_count"] for worker in worker_records),
        "audio_total_s": sum(worker["audio_total_s"] for worker in worker_records),
        "decode_ms": [record for worker in worker_records for record in worker["decode_ms"]],
        "warmups": {
            worker["language"]: worker["warmup"] for worker in worker_records
        },
        "latency": aggregate_latency(runs),
        "quality": aggregate_quality(runs),
        "runs": runs,
        "worker_records": [
            {key: value for key, value in worker.items() if key != "runs"}
            for worker in worker_records
        ],
        "caveats": [
            "This is a 12-clip diagnostic fixture, not a statistically powered ASR benchmark.",
            "SenseVoiceSmall supports zh/en/ja/ko/yue; unsupported Spanish, German, and French clips were excluded.",
            "This Mac CPU measurement is not a Nubia phone or Hexagon HTP-v81 measurement.",
            "The model is offline/non-streaming; it does not expose partial transcripts in this path.",
            "Fixed locale assumes the app already knows or caches the selected user language.",
            "Each language ran in a fresh sequential worker so only one model copy was resident at a time.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
