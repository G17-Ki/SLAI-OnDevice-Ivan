#!/usr/bin/env python3
"""Combine Tiny/Base raw benchmark JSON into public tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def row_for(result: dict, mode: str) -> dict[str, object]:
    latency = result["latency"][mode]
    quality = result["quality"][mode]
    return {
        "model": result["model"]["name"],
        "decoder_profile": result["benchmark_profile"].get(
            "decoder_profile", "fast_greedy"
        ),
        "mode": mode,
        "model_mib": result["model"]["total_file_bytes"] / (1024 * 1024),
        "incremental_peak_rss_mb": result["model"]["incremental_peak_rss_mb"],
        "first_segment_ms_median": latency["first_segment_ms"]["median"],
        "first_segment_ms_p95": latency["first_segment_ms"]["p95"],
        "final_ms_median": latency["final_ms"]["median"],
        "final_ms_p95": latency["final_ms"]["p95"],
        "rtf_median": latency["rtf"]["median"],
        "rtf_p95": latency["rtf"]["p95"],
        "latin_micro_wer": quality["latin_micro_wer"],
        "char_primary_micro_cer": quality["char_primary_micro_cer"],
        "macro_primary_error_rate": quality["macro_primary_error_rate"],
        "language_detection_accuracy": quality.get("language_detection_accuracy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text()) for path in args.inputs]
    rows = [row_for(result, mode) for result in results for mode in result["latency"]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    transcript_rows = []
    latency_rows = []
    for result in results:
        decoder_profile = result["benchmark_profile"].get(
            "decoder_profile", "fast_greedy"
        )
        for run in result["runs"]:
            if run["repeat_index"] != 0:
                continue
            transcript_rows.append(
                {
                    "model": result["model"]["name"],
                    "decoder_profile": decoder_profile,
                    "mode": run["mode"],
                    "sample_id": run["sample_id"],
                    "language": run["language"],
                    "detected_language": run["detected_language"],
                    "reference": run["reference"],
                    "transcript": run["transcript"],
                    "wer": run["errors"]["wer"],
                    "cer": run["errors"]["cer"],
                    "first_segment_ms": run["first_segment_ms"],
                    "final_ms": run["final_ms"],
                    "rtf": run["rtf"],
                }
            )
        for mode, mode_latency in result["latency"].items():
            for language, language_latency in mode_latency["per_language"].items():
                latency_rows.append(
                    {
                        "model": result["model"]["name"],
                        "decoder_profile": decoder_profile,
                        "mode": mode,
                        "language": language,
                        **language_latency,
                    }
                )

    transcript_path = output_dir / "transcripts.csv"
    with transcript_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transcript_rows[0]))
        writer.writeheader()
        writer.writerows(transcript_rows)

    latency_path = output_dir / "latency_by_language.csv"
    with latency_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latency_rows[0]))
        writer.writeheader()
        writer.writerows(latency_rows)

    comparison = {
        "schema_version": 1,
        "rows": rows,
        "per_language": {
            (
                result["model"]["name"]
                + ":"
                + result["benchmark_profile"].get("decoder_profile", "fast_greedy")
            ): result["quality"]
            for result in results
        },
    }
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Generated benchmark summary",
        "",
        "| Model | Decoder | Language mode | Stable first segment median / P95 | Final median / P95 | Median RTF | Peak model RSS delta | Latin WER | JA/KO/ZH CER | Language ID |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        detection = row["language_detection_accuracy"]
        lines.append(
            f"| {row['model']} | {row['decoder_profile']} | {row['mode']} | "
            f"{row['first_segment_ms_median']:.1f} / {row['first_segment_ms_p95']:.1f} ms | "
            f"{row['final_ms_median']:.1f} / {row['final_ms_p95']:.1f} ms | "
            f"{row['rtf_median']:.3f} | {row['incremental_peak_rss_mb']:.1f} MiB | "
            f"{100 * row['latin_micro_wer']:.1f}% | "
            f"{100 * row['char_primary_micro_cer']:.1f}% | "
            f"{('-' if detection is None else f'{100 * detection:.1f}%')} |"
        )
    markdown_path = output_dir / "benchmark_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
