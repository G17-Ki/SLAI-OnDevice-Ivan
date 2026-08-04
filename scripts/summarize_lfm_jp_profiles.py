#!/usr/bin/env python3
"""Combine measured runs from Japanese LFM profile JSON files."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "first_text_ms",
    "first_pcm_ms",
    "total_ms",
    "audio_seconds",
    "largest_pcm_arrival_gap_ms",
    "stream_generation_rtf",
    "character_error_rate_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def main() -> None:
    args = parse_args()
    runs: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    for path in args.inputs:
        document = json.loads(path.read_text(encoding="utf-8"))
        source_metadata.append(
            {
                "path": str(path),
                "mode": document.get("mode"),
                "model": document.get("model"),
                "system_prompt": document.get("system_prompt"),
                "input_audio": document.get("input_audio"),
                "input_audio_sha256": document.get("input_audio_sha256"),
                "input_text": document.get("input_text"),
                "max_tokens": document.get("max_tokens"),
            }
        )
        for run in document.get("runs", []):
            if run.get("measured"):
                copied = dict(run)
                copied["source"] = str(path)
                runs.append(copied)

    metric_summary: dict[str, dict[str, float] | None] = {}
    for metric in METRICS:
        values = [
            float(value)
            for run in runs
            if (value := run.get(metric)) is not None
        ]
        metric_summary[metric] = summarize(values)

    transcripts = [str(run.get("transcript", "")) for run in runs]
    pcm_hashes = [str(run["pcm_sha256"]) for run in runs if run.get("pcm_sha256")]
    output = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": source_metadata,
        "measured_runs": len(runs),
        "summary": metric_summary,
        "unique_transcripts": sorted(set(transcripts)),
        "unique_pcm_sha256": sorted(set(pcm_hashes)),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
