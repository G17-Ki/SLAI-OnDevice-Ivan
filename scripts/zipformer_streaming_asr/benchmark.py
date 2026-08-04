#!/usr/bin/env python3
"""Reproduce the fixed English/Mandarin streaming Zipformer quality gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import psutil


SAMPLE_RATE = 16_000
LANGUAGES = ("en", "zh")
NUM_THREADS = 2
WARMUP_RUNS = 1
REPEATS = 3
TAIL_MS = 660
PARTIAL_CHUNK_MS = 100
MODEL_REPO = "pfluo/k2fsa-zipformer-chinese-english-mixed"
MODEL_REVISION = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"
EXPECTED_MODEL_SHA256 = {
    "encoder": "e92df89d84ae5cb1571a2489c85d5d385c31fb8022892a240bd8c4c301b3ecc8",
    "decoder": "d91b63c84f27f1d822244ced4cc2b9f7fb8a60a328f13b860408b8889dd0a0ed",
    "joiner": "16e901969495d5704a27a929e396a2ff9cc6fbfdeafdc6418e8c88b370113af6",
    "tokens": "a8e0e4ec53810e433789b54a5c0134a7eaa2ffca595a6334d54c00da858841d3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


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


def canonical_language(value: object) -> str | None:
    language = str(value).strip().lower().replace("_", "-")
    if language == "en" or language.startswith("en-"):
        return "en"
    if language in {"zh", "cmn"} or language.startswith("zh-"):
        return "zh"
    return None


def first_present(record: dict[str, object], names: Sequence[str]) -> object:
    for name in names:
        value = record.get(name)
        if value is not None and str(value) != "":
            return value
    raise ValueError(f"manifest row is missing all of these fields: {', '.join(names)}")


def resolve_audio_path(raw_path: str, dataset_dir: Path, manifest_dir: Path) -> Path:
    source = Path(raw_path).expanduser()
    candidates = [source] if source.is_absolute() else []
    if not source.is_absolute():
        candidates.extend(
            [
                dataset_dir / source,
                manifest_dir / source,
                dataset_dir / "samples" / source,
                manifest_dir / "samples" / source,
                dataset_dir / source.name,
                manifest_dir / source.name,
                dataset_dir / "samples" / source.name,
                manifest_dir / "samples" / source.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"cannot resolve audio {raw_path!r}; tried:\n  {tried}")


def portable_path_label(value: object) -> str:
    path = Path(str(value))
    return path.name if path.is_absolute() else path.as_posix()


def load_samples(manifest_path: Path, dataset_dir: Path) -> list[dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("samples")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("manifest must be a list or an object with a 'samples' list")

    samples = []
    for raw_record in rows:
        if not isinstance(raw_record, dict):
            raise ValueError("every manifest sample must be a JSON object")
        language = canonical_language(raw_record.get("language", ""))
        if language not in LANGUAGES:
            continue
        audio_value = first_present(raw_record, ("audio_path", "audio", "path"))
        reference = str(first_present(raw_record, ("reference", "text", "transcript")))
        audio_path = resolve_audio_path(
            str(audio_value), dataset_dir, manifest_path.parent
        )
        sample_id = str(raw_record.get("sample_id") or Path(str(audio_value)).stem)
        samples.append(
            {
                "sample_id": sample_id,
                "language": language,
                "reference": reference,
                "manifest_audio_path": portable_path_label(audio_value),
                "audio_path": audio_path,
                "duration_metadata_s": raw_record.get(
                    "duration_metadata_s", raw_record.get("duration")
                ),
            }
        )

    samples.sort(key=lambda sample: (str(sample["language"]), str(sample["sample_id"])))
    counts = Counter(str(sample["language"]) for sample in samples)
    missing = [language for language in LANGUAGES if counts[language] == 0]
    if missing:
        raise ValueError(f"manifest has no selected samples for: {', '.join(missing)}")
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"sample_id values must be unique: {duplicates}")
    return samples


def locate_model_files(
    model_dir: Path, overrides: dict[str, Path | None]
) -> dict[str, Path]:
    candidates = {
        "encoder": (
            "exp/encoder-epoch-99-avg-1.int8.onnx",
            "encoder-epoch-99-avg-1.int8.onnx",
            "encoder.int8.onnx",
            "encoder.onnx",
        ),
        "decoder": (
            "exp/decoder-epoch-99-avg-1.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "decoder.onnx",
        ),
        "joiner": (
            "exp/joiner-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "joiner.int8.onnx",
            "joiner.onnx",
        ),
        "tokens": ("data/lang_char_bpe/tokens.txt", "tokens.txt"),
    }
    resolved = {}
    for component, relative_candidates in candidates.items():
        override = overrides[component]
        if override is not None:
            candidate = override.expanduser()
            if not candidate.is_absolute():
                candidate = model_dir / candidate
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"--{component} does not identify a file: {candidate}"
                )
            resolved[component] = candidate.resolve()
            continue
        for relative_path in relative_candidates:
            candidate = model_dir / relative_path
            if candidate.is_file():
                resolved[component] = candidate.resolve()
                break
        else:
            expected = ", ".join(relative_candidates)
            raise FileNotFoundError(
                f"missing {component} in {model_dir}; expected {expected}"
            )
    return resolved


def verify_model_files(
    model_dir: Path,
    paths: dict[str, Path],
    allow_unpinned_model: bool,
) -> tuple[list[dict[str, object]], bool]:
    records = []
    checkpoint_matches_pinned_revision = True
    for component in ("encoder", "decoder", "joiner", "tokens"):
        path = paths[component]
        actual_sha256 = sha256(path)
        expected_sha256 = EXPECTED_MODEL_SHA256[component]
        matches_pinned_sha256 = actual_sha256 == expected_sha256
        checkpoint_matches_pinned_revision &= matches_pinned_sha256
        if not matches_pinned_sha256 and not allow_unpinned_model:
            raise ValueError(
                f"{component} SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}. Pass --allow-unpinned-model only for a "
                "deliberate comparison checkpoint."
            )
        try:
            displayed_path = path.relative_to(model_dir).as_posix()
        except ValueError:
            displayed_path = path.name
        records.append(
            {
                "component": component,
                "path": displayed_path,
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
                "pinned_sha256": expected_sha256,
                "matches_pinned_sha256": matches_pinned_sha256,
            }
        )
    return records, checkpoint_matches_pinned_revision


def ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    )
    return completed.stdout.splitlines()[0]


def decode_audio(path: Path) -> tuple[np.ndarray, float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]
    start_ns = time.perf_counter_ns()
    completed = subprocess.run(command, check=True, capture_output=True)
    decode_ms = (time.perf_counter_ns() - start_ns) / 1e6
    audio = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32)
    audio /= 32768.0
    if len(audio) == 0:
        raise ValueError(f"ffmpeg decoded no samples from {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"ffmpeg produced non-finite samples for {path}")
    return audio, decode_ms


def result_text(recognizer: object, stream: object) -> str:
    result = recognizer.get_result(stream)
    value = getattr(result, "text", result)
    if callable(value):
        value = value()
    return str(value).strip()


def drain_ready(recognizer: object, stream: object) -> tuple[int, float]:
    decode_calls = 0
    decode_compute_ms = 0.0
    while recognizer.is_ready(stream):
        start_ns = time.perf_counter_ns()
        recognizer.decode_stream(stream)
        decode_compute_ms += (time.perf_counter_ns() - start_ns) / 1e6
        decode_calls += 1
    return decode_calls, decode_compute_ms


def decode_final(recognizer: object, audio: np.ndarray) -> dict[str, object]:
    tail_samples = round(SAMPLE_RATE * TAIL_MS / 1000)
    start_ns = time.perf_counter_ns()
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    stream.accept_waveform(SAMPLE_RATE, np.zeros(tail_samples, dtype=np.float32))
    stream.input_finished()
    decode_calls, decode_compute_ms = drain_ready(recognizer, stream)
    hypothesis = result_text(recognizer, stream)
    final_ms = (time.perf_counter_ns() - start_ns) / 1e6
    return {
        "hypothesis": hypothesis,
        "final_ms": final_ms,
        "decode_compute_ms": decode_compute_ms,
        "decode_calls": decode_calls,
    }


def incremental_probe(recognizer: object, audio: np.ndarray) -> dict[str, object]:
    chunk_samples = round(SAMPLE_RATE * PARTIAL_CHUNK_MS / 1000)
    tail_samples = round(SAMPLE_RATE * TAIL_MS / 1000)
    stream = recognizer.create_stream()
    total_decode_calls = 0
    total_decode_compute_ms = 0.0
    first_nonempty_audio_fed_ms = None
    first_nonempty_accumulated_decode_ms = None
    first_partial = None

    def decode_and_probe(audio_fed_samples: int) -> None:
        nonlocal total_decode_calls
        nonlocal total_decode_compute_ms
        nonlocal first_nonempty_audio_fed_ms
        nonlocal first_nonempty_accumulated_decode_ms
        nonlocal first_partial
        while recognizer.is_ready(stream):
            start_ns = time.perf_counter_ns()
            recognizer.decode_stream(stream)
            total_decode_compute_ms += (time.perf_counter_ns() - start_ns) / 1e6
            total_decode_calls += 1
            partial = result_text(recognizer, stream)
            if partial and first_partial is None:
                first_nonempty_audio_fed_ms = audio_fed_samples / SAMPLE_RATE * 1000.0
                first_nonempty_accumulated_decode_ms = total_decode_compute_ms
                first_partial = partial

    fed_samples = 0
    for offset in range(0, len(audio), chunk_samples):
        chunk = audio[offset : offset + chunk_samples]
        stream.accept_waveform(SAMPLE_RATE, chunk)
        fed_samples += len(chunk)
        decode_and_probe(fed_samples)

    stream.accept_waveform(SAMPLE_RATE, np.zeros(tail_samples, dtype=np.float32))
    decode_and_probe(fed_samples)
    input_finished_start_ns = time.perf_counter_ns()
    stream.input_finished()
    decode_and_probe(fed_samples)
    post_input_finished_ms = (time.perf_counter_ns() - input_finished_start_ns) / 1e6
    return {
        "first_nonempty_audio_fed_ms": first_nonempty_audio_fed_ms,
        "first_nonempty_accumulated_decode_ms": first_nonempty_accumulated_decode_ms,
        "first_partial": first_partial,
        "final": result_text(recognizer, stream),
        "total_decode_compute_ms": total_decode_compute_ms,
        "decode_calls": total_decode_calls,
        "post_input_finished_ms": post_input_finished_ms,
    }


def aggregate_quality(per_sample: list[dict[str, object]]) -> dict[str, object]:
    quality = {}
    for language, metric, edit_key, denominator_key in (
        ("en", "wer", "word_edits", "reference_words"),
        ("zh", "cer", "char_edits", "reference_chars"),
    ):
        selected = [sample for sample in per_sample if sample["language"] == language]
        edits = sum(int(sample["errors"][edit_key]) for sample in selected)
        denominator = sum(int(sample["errors"][denominator_key]) for sample in selected)
        if denominator == 0:
            raise ValueError(f"zero {metric} denominator for {language}")
        quality[language] = {
            "metric": metric,
            "edits": edits,
            "denominator": denominator,
            "error_rate": edits / denominator,
        }
    return quality


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the pinned streaming Zipformer on EN/ZH samples."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--encoder", type=Path, help="encoder path, relative to --model-dir"
    )
    parser.add_argument(
        "--decoder", type=Path, help="decoder path, relative to --model-dir"
    )
    parser.add_argument(
        "--joiner", type=Path, help="joiner path, relative to --model-dir"
    )
    parser.add_argument(
        "--tokens", type=Path, help="token-table path, relative to --model-dir"
    )
    parser.add_argument(
        "--allow-unpinned-model",
        action="store_true",
        help="permit a comparison checkpoint whose SHA-256 values differ",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required and was not found on PATH")
    model_dir = args.model_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

    model_paths = locate_model_files(
        model_dir,
        {
            "encoder": args.encoder,
            "decoder": args.decoder,
            "joiner": args.joiner,
            "tokens": args.tokens,
        },
    )
    model_records, checkpoint_matches_pinned_revision = verify_model_files(
        model_dir, model_paths, args.allow_unpinned_model
    )
    samples = load_samples(manifest_path, dataset_dir)
    decoded_samples = []
    for sample in samples:
        audio, ffmpeg_decode_ms = decode_audio(Path(sample["audio_path"]))
        decoded_samples.append(
            {
                **sample,
                "audio": audio,
                "audio_bytes": Path(sample["audio_path"]).stat().st_size,
                "audio_sha256": sha256(Path(sample["audio_path"])),
                "ffmpeg_decode_ms": ffmpeg_decode_ms,
                "duration_decoded_s": len(audio) / SAMPLE_RATE,
            }
        )

    import sherpa_onnx

    process = psutil.Process()
    gc.collect()
    rss_before_model_bytes = process.memory_info().rss
    with PeakRssSampler() as peak_sampler:
        load_start_ns = time.perf_counter_ns()
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_paths["tokens"]),
            encoder=str(model_paths["encoder"]),
            decoder=str(model_paths["decoder"]),
            joiner=str(model_paths["joiner"]),
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
            debug=False,
        )
        model_load_ms = (time.perf_counter_ns() - load_start_ns) / 1e6
        rss_after_model_bytes = process.memory_info().rss

        warmup_results = []
        for warmup_index in range(WARMUP_RUNS):
            warmup = decode_final(recognizer, decoded_samples[0]["audio"])
            warmup_results.append({"warmup_index": warmup_index, **warmup})

        runs = []
        for sample in decoded_samples:
            for repeat_index in range(REPEATS):
                with PeakRssSampler() as run_peak:
                    decoded = decode_final(recognizer, sample["audio"])
                errors = error_record(sample["reference"], decoded["hypothesis"])
                runs.append(
                    {
                        "sample_id": sample["sample_id"],
                        "language": sample["language"],
                        "repeat_index": repeat_index,
                        "duration_s": sample["duration_decoded_s"],
                        "reference": sample["reference"],
                        **decoded,
                        "rtf": decoded["final_ms"]
                        / 1000.0
                        / sample["duration_decoded_s"],
                        "peak_rss_mib": run_peak.peak_bytes / (1024 * 1024),
                        "errors": errors,
                    }
                )
                print(
                    f"{sample['language']} {sample['sample_id']} r{repeat_index}: "
                    f"{decoded['final_ms']:.1f} ms {decoded['hypothesis']!r}",
                    file=sys.stderr,
                    flush=True,
                )

        streaming = []
        for sample in decoded_samples:
            streaming.append(
                {
                    "sample_id": sample["sample_id"],
                    "language": sample["language"],
                    **incremental_probe(recognizer, sample["audio"]),
                }
            )

    per_sample = []
    for sample in decoded_samples:
        sample_runs = [run for run in runs if run["sample_id"] == sample["sample_id"]]
        hypothesis_counts = Counter(str(run["hypothesis"]) for run in sample_runs)
        representative_hypothesis = hypothesis_counts.most_common(1)[0][0]
        final_values = [float(run["final_ms"]) for run in sample_runs]
        probe = next(
            item for item in streaming if item["sample_id"] == sample["sample_id"]
        )
        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "language": sample["language"],
                "reference": sample["reference"],
                "hypothesis": representative_hypothesis,
                "hypotheses_deterministic": len(hypothesis_counts) == 1,
                "final_ms_median": statistics.median(final_values),
                "final_ms_p95": percentile(final_values, 95),
                "first_nonempty_audio_fed_ms": probe[
                    "first_nonempty_audio_fed_ms"
                ],
                "first_partial": probe["first_partial"],
                "errors": error_record(
                    str(sample["reference"]), representative_hypothesis
                ),
            }
        )

    all_final_ms = [float(run["final_ms"]) for run in runs]
    all_rtfs = [float(run["rtf"]) for run in runs]
    result = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "model_repo": MODEL_REPO if checkpoint_matches_pinned_revision else None,
            "model_revision": (
                MODEL_REVISION if checkpoint_matches_pinned_revision else None
            ),
            "checkpoint_matches_pinned_revision": (
                checkpoint_matches_pinned_revision
            ),
            "pinned_reference": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
            },
            "model_files": model_records,
            "manifest_filename": manifest_path.name,
            "manifest_sha256": sha256(manifest_path),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sherpa_onnx": importlib.metadata.version("sherpa-onnx"),
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "ffmpeg_first_line": ffmpeg_version(),
            "provider": "cpu",
        },
        "method": {
            "languages": list(LANGUAGES),
            "threads": NUM_THREADS,
            "warmup_runs": WARMUP_RUNS,
            "repeats_per_clip": REPEATS,
            "tail_ms": TAIL_MS,
            "partial_probe_chunk_ms": PARTIAL_CHUNK_MS,
            "decoding_method": "greedy_search",
            "audio": (
                "ffmpeg mono 16 kHz signed PCM16 decode outside timed regions; "
                "normalize to float32 by dividing by 32768"
            ),
            "timed_region": (
                "stream creation, waveform and tail acceptance, input_finished, "
                "decode loop, and final result retrieval"
            ),
            "normalization": (
                "Unicode NFKC, casefold, remove punctuation/symbols; "
                "English WER and Mandarin CER"
            ),
            "sample_count": len(decoded_samples),
        },
        "runtime": {
            "model_load_ms": model_load_ms,
            "rss_before_model_mib": rss_before_model_bytes / (1024 * 1024),
            "rss_after_model_mib": rss_after_model_bytes / (1024 * 1024),
            "post_load_rss_delta_mib": (
                rss_after_model_bytes - rss_before_model_bytes
            )
            / (1024 * 1024),
            "peak_rss_mib": peak_sampler.peak_bytes / (1024 * 1024),
            "peak_incremental_rss_mib": (
                peak_sampler.peak_bytes - rss_before_model_bytes
            )
            / (1024 * 1024),
            "final_ms_median": statistics.median(all_final_ms),
            "final_ms_p95": percentile(all_final_ms, 95),
            "rtf_median": statistics.median(all_rtfs),
            "rtf_p95": percentile(all_rtfs, 95),
        },
        "quality": aggregate_quality(per_sample),
        "audio_inputs": [
            {
                "sample_id": sample["sample_id"],
                "language": sample["language"],
                "manifest_audio_path": sample["manifest_audio_path"],
                "audio_bytes": sample["audio_bytes"],
                "audio_sha256": sample["audio_sha256"],
                "duration_metadata_s": sample["duration_metadata_s"],
                "duration_decoded_s": sample["duration_decoded_s"],
                "ffmpeg_decode_ms": sample["ffmpeg_decode_ms"],
            }
            for sample in decoded_samples
        ],
        "warmup": warmup_results,
        "per_sample": per_sample,
        "runs": runs,
        "streaming": streaming,
    }
    serialized = json.dumps(
        result, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    json.loads(
        output_path.read_text(encoding="utf-8"),
        parse_constant=reject_json_constant,
    )
    print(output_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
