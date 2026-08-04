#!/usr/bin/env python3
"""Run a bounded Italian ASR intelligibility check over Piper WAVs.

Models are loaded sequentially and constrained to two CPU threads.  The output
records public-safe labels rather than local absolute model paths.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import unicodedata
from typing import Sequence


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

from faster_whisper import WhisperModel


def parse_mapping(value: str, kind: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"{kind} must use LABEL=PATH")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"invalid {kind} label: {label!r}")
    return label, Path(raw_path).expanduser().resolve()


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


def score(reference: str, transcript: str) -> dict[str, object]:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(transcript)
    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(transcript)
    word_edits = edit_distance(ref_words, hyp_words)
    char_edits = edit_distance(ref_chars, hyp_chars)
    return {
        "normalized_reference": " ".join(ref_words),
        "normalized_transcript": " ".join(hyp_words),
        "word_edits": word_edits,
        "reference_words": len(ref_words),
        "wer": word_edits / max(len(ref_words), 1),
        "char_edits": char_edits,
        "reference_chars": len(ref_chars),
        "cer": char_edits / max(len(ref_chars), 1),
        "exact_normalized_match": ref_words == hyp_words,
    }


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.name


def macos_pressure_level() -> int | None:
    if platform.system() != "Darwin":
        return None
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(completed.stdout.strip())


def write_output(path: Path, value: dict[str, object]) -> None:
    value["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Faster-Whisper model directory; repeat for sequential models",
    )
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="WAV input; repeat for local and remote samples",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--language", default="it")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed model/sample rows from an existing output JSON",
    )
    args = parser.parse_args()

    if args.beam_size < 1:
        parser.error("--beam-size must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    models = [parse_mapping(value, "model") for value in args.model]
    samples = [parse_mapping(value, "sample") for value in args.sample]
    for _, path in (*models, *samples):
        if not path.exists():
            parser.error(f"input does not exist: {path}")

    output_path = args.output.expanduser().resolve()
    if args.resume:
        if not output_path.is_file():
            parser.error("--resume requires an existing --output JSON")
        output = json.loads(output_path.read_text())
        if output.get("reference") != args.reference:
            parser.error("resume reference differs from the existing output")
        if output.get("language") != args.language:
            parser.error("resume language differs from the existing output")
        expected_samples = [label for label, _ in samples]
        existing_samples = [item.get("label") for item in output.get("samples", [])]
        if existing_samples != expected_samples:
            parser.error("resume sample labels/order differ from the existing output")
        output["status"] = "running"
        output.pop("aborted_before_model", None)
        output.pop("aborted_before_sample", None)
        output.pop("memory_pressure_level", None)
        output.setdefault("resume_utc", []).append(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
    else:
        output = {
            "schema_version": 1,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "running",
            "purpose": (
                "ASR intelligibility backcheck; not a perceptual naturalness or "
                "speaker-similarity evaluation"
            ),
            "reference": args.reference,
            "language": args.language,
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "protocol": {
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 2,
                "num_workers": 1,
                "beam_size": args.beam_size,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "models_loaded_sequentially": True,
                "resume_supported": True,
            },
            "samples": [
                {"label": label, "wav": display_path(path, repo_root)}
                for label, path in samples
            ],
            "models": [],
        }

    model_results = list(output.get("models", []))
    existing_models = {
        str(item.get("label")): item for item in model_results if item.get("label")
    }
    for model_label, model_path in models:
        model_result = existing_models.get(model_label)
        completed_samples = {
            str(item.get("sample"))
            for item in (model_result or {}).get("results", [])
            if item.get("sample")
        }
        if completed_samples.issuperset(label for label, _ in samples):
            continue
        pressure = macos_pressure_level()
        if pressure not in (None, 1):
            output["status"] = "aborted_host_memory_pressure"
            output["aborted_before_model"] = model_label
            output["memory_pressure_level"] = pressure
            output["models"] = model_results
            write_output(output_path, output)
            return 2
        load_start = time.perf_counter()
        model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1,
        )
        load_seconds = time.perf_counter() - load_start
        if model_result is None:
            model_result = {
                "label": model_label,
                "model_directory_name": model_path.name,
                "load_seconds": load_seconds,
                "results": [],
            }
            model_results.append(model_result)
            existing_models[model_label] = model_result
        else:
            model_result.setdefault("resume_load_seconds", []).append(load_seconds)
        sample_results = list(model_result.get("results", []))
        for sample_label, sample_path in samples:
            if sample_label in completed_samples:
                continue
            pressure = macos_pressure_level()
            if pressure not in (None, 1):
                model_result["results"] = sample_results
                output["status"] = "aborted_host_memory_pressure"
                output["aborted_before_sample"] = sample_label
                output["memory_pressure_level"] = pressure
                output["models"] = model_results
                write_output(output_path, output)
                del model
                gc.collect()
                return 2
            start = time.perf_counter()
            segments, info = model.transcribe(
                str(sample_path),
                language=args.language,
                task="transcribe",
                beam_size=args.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            materialized = list(segments)
            elapsed = time.perf_counter() - start
            transcript = " ".join(
                segment.text.strip() for segment in materialized if segment.text.strip()
            ).strip()
            sample_results.append(
                {
                    "sample": sample_label,
                    "transcript": transcript,
                    "elapsed_seconds": elapsed,
                    "detected_language": info.language,
                    "language_probability": info.language_probability,
                    "segments": [
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip(),
                        }
                        for segment in materialized
                    ],
                    **score(args.reference, transcript),
                }
            )
            completed_samples.add(sample_label)
            model_result["results"] = sample_results
            output["models"] = model_results
            write_output(output_path, output)
        model_result["results"] = sample_results
        output["models"] = model_results
        write_output(output_path, output)
        del model
        gc.collect()

    output["status"] = "completed"
    output["models"] = model_results
    write_output(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
