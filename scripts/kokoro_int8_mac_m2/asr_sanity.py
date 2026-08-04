#!/usr/bin/env python3
"""One-pass SenseVoiceSmall sanity check for Kokoro samples and nearby audio.

This is not a perceptual metric. Fixed-locale recognizers run in sequential
worker processes so only one ASR model is resident at a time and all memory is
released when each worker exits.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import psutil


SAMPLES = [
    {
        "id": "kokoro_en_short",
        "kind": "kokoro_quality_sanity",
        "language": "en",
        "audio_path": "reports/android_phone/streaming_response/kokoro_int8_mac_m2/samples/en_short.wav",
        "reference": "Sure, I can help with that.",
    },
    {
        "id": "kokoro_en_fact",
        "kind": "kokoro_quality_sanity",
        "language": "en",
        "audio_path": "reports/android_phone/streaming_response/kokoro_int8_mac_m2/samples/en_fact.wav",
        "reference": "Paris is the capital of France.",
    },
    {
        "id": "kokoro_en_chunked",
        "kind": "kokoro_quality_sanity",
        "language": "en",
        "audio_path": "reports/android_phone/streaming_response/kokoro_int8_mac_m2/samples/en_chunked.wav",
        "reference": (
            "I found three nearby cafes that are still open. The closest one "
            "is two minutes away. Would you like walking directions?"
        ),
    },
    {
        "id": "kokoro_zh_short",
        "kind": "kokoro_quality_sanity",
        "language": "zh",
        "audio_path": "reports/android_phone/streaming_response/kokoro_int8_mac_m2/samples/zh_short.wav",
        "reference": "当然可以，我可以帮你。",
    },
    {
        "id": "kokoro_zh_fact",
        "kind": "kokoro_quality_sanity",
        "language": "zh",
        "audio_path": "reports/android_phone/streaming_response/kokoro_int8_mac_m2/samples/zh_fact.wav",
        "reference": "巴黎是法国的首都。",
    },
    {
        "id": "kokoro_zh_chunked",
        "kind": "kokoro_quality_sanity",
        "language": "zh",
        "audio_path": "reports/android_phone/streaming_response/kokoro_int8_mac_m2/samples/zh_chunked.wav",
        "reference": (
            "我找到三家还在营业的咖啡馆。最近的一家步行只需两分钟。"
            "需要我为你导航吗？"
        ),
    },
    {
        "id": "supertonic_ja_request",
        "kind": "separate_crosscheck_not_kokoro",
        "language": "ja",
        "audio_path": "reports/android_phone/streaming_response/multilingual_request_samples/ja_request.wav",
        "reference": "フランスの首都はどこですか。",
    },
    {
        "id": "lfm_jp_tts_01",
        "kind": "separate_crosscheck_not_kokoro",
        "language": "ja",
        "audio_path": "reports/android_phone/streaming_response/lfm_jp_q4_mac_m2/audio/tts-01.wav",
        "reference": "こんにちは、お元気ですか？",
    },
    {
        "id": "supertonic_ko_request",
        "kind": "separate_crosscheck_not_kokoro",
        "language": "ko",
        "audio_path": "reports/android_phone/streaming_response/multilingual_request_samples/ko_request.wav",
        "reference": "프랑스의 수도는 어디인가요?",
    },
]


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


def errors(reference: str, hypothesis: str, language: str) -> dict[str, object]:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    word_edits = edit_distance(ref_words, hyp_words)
    char_edits = edit_distance(ref_chars, hyp_chars)
    primary = "wer" if language == "en" else "cer"
    return {
        "normalized_reference_words": " ".join(ref_words),
        "normalized_hypothesis_words": " ".join(hyp_words),
        "normalized_reference_chars": "".join(ref_chars),
        "normalized_hypothesis_chars": "".join(hyp_chars),
        "word_edits": word_edits,
        "reference_words": len(ref_words),
        "wer": word_edits / max(len(ref_words), 1),
        "char_edits": char_edits,
        "reference_chars": len(ref_chars),
        "cer": char_edits / max(len(ref_chars), 1),
        "primary_metric": primary,
        "primary_error_rate": (
            word_edits / max(len(ref_words), 1)
            if primary == "wer"
            else char_edits / max(len(ref_chars), 1)
        ),
    }


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
    start = time.perf_counter()
    raw = subprocess.check_output(command)
    return np.frombuffer(raw, dtype="<f4").copy(), time.perf_counter() - start


def worker(args: argparse.Namespace) -> int:
    import sherpa_onnx

    manifest = json.loads(args.manifest.read_text())
    selected = [x for x in manifest if x["language"] == args.worker_language]
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(args.model_dir / "model.int8.onnx"),
        tokens=str(args.model_dir / "tokens.txt"),
        num_threads=2,
        provider="cpu",
        language=args.worker_language,
        use_itn=True,
        debug=False,
    )
    load_s = time.perf_counter() - start
    rss_after = process.memory_info().rss
    results = []
    peak = rss_after
    for sample in selected:
        audio, decode_s = decode_audio(args.repo_root / sample["audio_path"])
        start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        inference_s = time.perf_counter() - start
        result = stream.result
        transcript_value = getattr(result, "text", result)
        if callable(transcript_value):
            transcript_value = transcript_value()
        transcript = str(transcript_value).strip()
        peak = max(peak, process.memory_info().rss)
        results.append(
            {
                **sample,
                "transcript": transcript,
                "audio_duration_s": len(audio) / 16000.0,
                "decode_s": decode_s,
                "inference_s": inference_s,
                "rtf": inference_s / (len(audio) / 16000.0),
                "errors": errors(sample["reference"], transcript, sample["language"]),
            }
        )
        print(f"{sample['id']}: {transcript!r}", flush=True)

    output = {
        "language": args.worker_language,
        "fixed_locale": True,
        "num_threads": 2,
        "load_s": load_s,
        "rss_before_bytes": rss_before,
        "rss_after_load_bytes": rss_after,
        "incremental_rss_bytes": rss_after - rss_before,
        "observed_peak_rss_bytes": peak,
        "results": results,
    }
    args.worker_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return 0


def aggregate(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    manifest = args.output_dir / "asr_manifest.json"
    manifest.write_text(
        json.dumps(SAMPLES, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )

    workers = []
    for language in ("en", "zh", "ja", "ko"):
        output = raw_dir / f"sensevoice_{language}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--repo-root",
            str(args.repo_root),
            "--model-dir",
            str(args.model_dir),
            "--output-dir",
            str(args.output_dir),
            "--manifest",
            str(manifest),
            "--worker-language",
            language,
            "--worker-output",
            str(output),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "2",
            }
        )
        subprocess.run(command, check=True, env=environment)
        workers.append(json.loads(output.read_text()))
        gc.collect()

    all_results = [result for item in workers for result in item["results"]]
    kokoro = [x for x in all_results if x["kind"] == "kokoro_quality_sanity"]
    summary = {}
    for language in ("en", "zh"):
        selected = [x for x in kokoro if x["language"] == language]
        metric = "wer" if language == "en" else "cer"
        edits_key = "word_edits" if metric == "wer" else "char_edits"
        count_key = "reference_words" if metric == "wer" else "reference_chars"
        edits = sum(int(x["errors"][edits_key]) for x in selected)
        count = sum(int(x["errors"][count_key]) for x in selected)
        summary[language] = {
            "samples": len(selected),
            "primary_metric": metric,
            "micro_error_rate": edits / max(count, 1),
            "edits": edits,
            "reference_units": count,
        }

    final = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge": "SenseVoiceSmall INT8 through sherpa-onnx==1.13.4",
        "judge_scope": (
            "One-pass fixed-locale intelligibility sanity only; not MOS, "
            "naturalness, speaker similarity, or human verification."
        ),
        "model_artifact": args.model_dir.name,
        "kokoro_summary": summary,
        "results": all_results,
        "worker_resource_records": [
            {key: value for key, value in worker.items() if key != "results"}
            for worker in workers
        ],
    }
    output = args.output_dir / "asr_sanity.json"
    output.write_text(
        json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--worker-language", choices=("en", "zh", "ja", "ko"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_language:
        if args.manifest is None or args.worker_output is None:
            parser.error("workers require --manifest and --worker-output")
        return worker(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
