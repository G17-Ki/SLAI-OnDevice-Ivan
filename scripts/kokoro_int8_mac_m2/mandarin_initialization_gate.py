#!/usr/bin/env python3
"""Distinguish Kokoro first-Mandarin initialization failures from text length.

Each fresh-start scenario runs in its own process so a prior synthesis cannot
silently initialize model state.  A final boundary sweep runs only after one
discarded Mandarin warm-up and checks short chunks repeatedly.  The run is
sequential and keeps at most one Kokoro instance resident.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx

from benchmark import audio_stats, create_tts


FRESH_SCENARIOS: dict[str, list[tuple[str, str, int]]] = {
    "zh_one_char_first": [("zh", "你。", 3), ("zh", "你。", 3)],
    "zh_short_first": [("zh", "你好。", 3), ("zh", "你好。", 3)],
    "zh_fact_first": [
        ("zh", "巴黎是法国的首都。", 3),
        ("zh", "巴黎是法国的首都。", 3),
    ],
    "zh_long_first": [
        ("zh", "当然可以，我可以帮你。", 3),
        ("zh", "当然可以，我可以帮你。", 3),
    ],
    "en_then_zh_short": [
        ("en", "Hello.", 0),
        ("zh", "你好。", 3),
        ("zh", "你好。", 3),
    ],
}

BOUNDARY_TEXTS = (
    "你。",
    "你好。",
    "你好呀。",
    "欢迎。",
    "请稍等。",
    "我来帮你。",
    "请稍等一下。",
)


def synthesize(tts: Any, language: str, text: str, sid: int) -> dict[str, Any]:
    first_callback_s: float | None = None
    started = time.perf_counter()

    def callback(samples: np.ndarray, _progress: float) -> int:
        nonlocal first_callback_s
        if first_callback_s is None:
            first_callback_s = time.perf_counter() - started
        return 1

    audio = tts.generate(text, sid=sid, speed=1.0, callback=callback)
    total_s = time.perf_counter() - started
    stats = audio_stats(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
    return {
        "language": language,
        "text": text,
        "text_codepoints": len(text),
        "sid": sid,
        "first_callback_s": first_callback_s,
        "total_s": total_s,
        **stats,
    }


def worker(model_dir: Path, threads: int, scenario: str) -> int:
    tts = create_tts(model_dir, threads)
    if scenario == "post_warmup_boundary":
        sequence = [("zh", "你好。", 3)]
        for text in BOUNDARY_TEXTS:
            sequence.extend(("zh", text, 3) for _ in range(3))
    else:
        sequence = FRESH_SCENARIOS[scenario]
    results = [synthesize(tts, *item) for item in sequence]
    print(json.dumps({"scenario": scenario, "results": results}, ensure_ascii=False, allow_nan=False))
    return 0


def orchestrate(model_dir: Path, threads: int, output: Path) -> int:
    scenarios = [*FRESH_SCENARIOS, "post_warmup_boundary"]
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-dir",
            str(model_dir),
            "--threads",
            str(threads),
            "--worker",
            scenario,
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed_s = time.perf_counter() - started
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            records.append({"status": "completed", "elapsed_s": elapsed_s, **payload})
        else:
            records.append(
                {
                    "scenario": scenario,
                    "status": "failed",
                    "elapsed_s": elapsed_s,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        print(f"{scenario}: {records[-1]['status']} ({elapsed_s:.2f}s)", flush=True)

    successful = [record for record in records if record["status"] == "completed"]
    first_mandarin = []
    post_warmup = []
    for record in successful:
        results = record["results"]
        if record["scenario"] == "post_warmup_boundary":
            post_warmup.extend(results[1:])
            continue
        for result in results:
            if result["language"] == "zh":
                first_mandarin.append(
                    {
                        "scenario": record["scenario"],
                        **result,
                    }
                )
                break

    document = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "Apple Silicon host CPU safety diagnostic; not phone or NPU evidence",
        "purpose": (
            "Determine whether non-finite Kokoro Mandarin PCM is tied to the first "
            "Mandarin synthesis after model construction or only to short text."
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sherpa_onnx": sherpa_onnx.__version__,
            "numpy": np.__version__,
        },
        "settings": {
            "num_threads": threads,
            "fresh_process_per_initialization_scenario": True,
            "post_warmup_repeats_per_text": 3,
            "resident_model_instances": 1,
        },
        "summary": {
            "fresh_first_mandarin_calls": len(first_mandarin),
            "fresh_first_mandarin_nonfinite": sum(
                int(result["nonfinite_count"]) > 0 for result in first_mandarin
            ),
            "post_warmup_calls": len(post_warmup),
            "post_warmup_nonfinite": sum(
                int(result["nonfinite_count"]) > 0 for result in post_warmup
            ),
            "post_warmup_clipped": sum(
                int(result["clip_count"]) > 0 for result in post_warmup
            ),
        },
        "scenarios": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document["summary"], indent=2))
    print(f"wrote {output}")
    return 0 if all(record["status"] == "completed" for record in records) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=[*FRESH_SCENARIOS, "post_warmup_boundary"])
    args = parser.parse_args()
    if args.worker:
        return worker(args.model_dir, args.threads, args.worker)
    if args.output is None:
        parser.error("--output is required outside worker mode")
    return orchestrate(args.model_dir, args.threads, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
