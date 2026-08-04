#!/usr/bin/env python3
"""Retranscribe generated samples as an intelligibility smoke check."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel
import jiwer


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="tiny")
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark_json.read_text())
    selected = {
        run["case_id"]: run
        for run in benchmark["runs"]
        if run["repeat_index"] == 0 and run["wav_path"]
    }

    load_start = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    load_s = time.perf_counter() - load_start
    results = []

    for case_id, run in selected.items():
        start = time.perf_counter()
        segments, info = model.transcribe(
            run["wav_path"],
            language=run["lang"],
            task="transcribe",
            beam_size=5,
            vad_filter=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        reference_norm = normalize(run["text"])
        transcript_norm = normalize(transcript)
        results.append(
            {
                "case_id": case_id,
                "lang": run["lang"],
                "reference": run["text"],
                "transcript": transcript,
                "normalized_reference": reference_norm,
                "normalized_transcript": transcript_norm,
                "wer": jiwer.wer(reference_norm, transcript_norm),
                "cer": jiwer.cer(reference_norm, transcript_norm),
                "asr_elapsed_s": time.perf_counter() - start,
                "asr_language_probability": info.language_probability,
                "wav_path": run["wav_path"],
            }
        )
        print(case_id, repr(transcript), flush=True)

    output = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": f"faster-whisper:{args.model}",
        "device": "cpu",
        "compute_type": "int8",
        "load_s": load_s,
        "warning": (
            "Whisper-tiny retranscription is an intelligibility smoke test, "
            "not a perceptual quality or naturalness evaluation."
        ),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
