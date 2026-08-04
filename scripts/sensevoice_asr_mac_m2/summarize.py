#!/usr/bin/env python3
"""Create compact CSV/JSON artifacts from the raw SenseVoice benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


LANGUAGES = ("en", "ja", "ko", "zh")
CHAR_LANGUAGES = {"ja", "ko", "zh"}


def percentile(values: list[float], percentile_value: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def fixed_runs(raw: dict[str, object]) -> list[dict[str, object]]:
    return [
        run
        for run in raw["runs"]
        if run.get("mode") == "fixed" and run["language"] in LANGUAGES
    ]


def quality_for_subset(runs: list[dict[str, object]]) -> dict[str, object]:
    first = [run for run in runs if run["repeat_index"] == 0]
    languages = {}
    for language in LANGUAGES:
        selected = [run for run in first if run["language"] == language]
        word_edits = sum(int(run["errors"]["word_edits"]) for run in selected)
        reference_words = sum(int(run["errors"]["reference_words"]) for run in selected)
        char_edits = sum(int(run["errors"]["char_edits"]) for run in selected)
        reference_chars = sum(int(run["errors"]["reference_chars"]) for run in selected)
        primary = "cer" if language in CHAR_LANGUAGES else "wer"
        languages[language] = {
            "samples": len(selected),
            "wer": word_edits / max(reference_words, 1),
            "cer": char_edits / max(reference_chars, 1),
            "word_edits": word_edits,
            "reference_words": reference_words,
            "char_edits": char_edits,
            "reference_chars": reference_chars,
            "primary_metric": primary,
            "primary_error_rate": (
                char_edits / max(reference_chars, 1)
                if primary == "cer"
                else word_edits / max(reference_words, 1)
            ),
        }
    cjk = [run for run in first if run["language"] in CHAR_LANGUAGES]
    cjk_edits = sum(int(run["errors"]["char_edits"]) for run in cjk)
    cjk_chars = sum(int(run["errors"]["reference_chars"]) for run in cjk)
    return {
        "languages": languages,
        "en_wer": languages["en"]["wer"],
        "ja_ko_zh_micro_cer": cjk_edits / max(cjk_chars, 1),
        "macro_primary_error_rate": statistics.mean(
            record["primary_error_rate"] for record in languages.values()
        ),
    }


def comparison_row(name: str, raw: dict[str, object]) -> dict[str, object]:
    runs = fixed_runs(raw)
    final_values = [float(run.get("final_ms", run.get("first_segment_ms"))) for run in runs]
    rtf_values = [float(run["rtf"]) for run in runs]
    quality = quality_for_subset(runs)
    profile = raw["benchmark_profile"]
    return {
        "model": name,
        "runtime": profile.get("runtime", "faster-whisper"),
        "cpu_threads": profile["cpu_threads"],
        "sample_count": len({run["sample_id"] for run in runs}),
        "final_ms_median": statistics.median(final_values),
        "final_ms_p95": percentile(final_values, 95),
        "rtf_median": statistics.median(rtf_values),
        "model_file_mib": raw["model"]["total_file_bytes"] / (1024 * 1024),
        "incremental_peak_rss_mb": raw["model"]["incremental_peak_rss_mb"],
        "en_wer": quality["en_wer"],
        "ja_ko_zh_micro_cer": quality["ja_ko_zh_micro_cer"],
        "macro_primary_error_rate": quality["macro_primary_error_rate"],
        "languages": quality["languages"],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensevoice", type=Path, required=True)
    parser.add_argument("--whisper-base", type=Path)
    parser.add_argument("--whisper-small", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sensevoice = json.loads(args.sensevoice.read_text())
    rows = [comparison_row("SenseVoiceSmall INT8", sensevoice)]
    if args.whisper_base:
        rows.append(comparison_row("Whisper Base INT8", json.loads(args.whisper_base.read_text())))
    if args.whisper_small:
        rows.append(comparison_row("Whisper Small INT8", json.loads(args.whisper_small.read_text())))

    public_fields = [
        "model",
        "runtime",
        "cpu_threads",
        "sample_count",
        "final_ms_median",
        "final_ms_p95",
        "rtf_median",
        "model_file_mib",
        "incremental_peak_rss_mb",
        "en_wer",
        "ja_ko_zh_micro_cer",
        "macro_primary_error_rate",
    ]
    write_csv(
        output_dir / "summary.csv",
        public_fields,
        [{field: row[field] for field in public_fields} for row in rows],
    )

    sensevoice_runs = fixed_runs(sensevoice)
    transcript_rows = []
    for run in sensevoice_runs:
        if run["repeat_index"] != 0:
            continue
        transcript_rows.append(
            {
                "sample_id": run["sample_id"],
                "language": run["language"],
                "reference": run["reference"],
                "hypothesis": run["transcript"],
                "wer": run["errors"]["wer"],
                "cer": run["errors"]["cer"],
                "primary_metric": (
                    "cer" if run["language"] in CHAR_LANGUAGES else "wer"
                ),
                "final_ms_first_repeat": run["final_ms"],
            }
        )
    write_csv(
        output_dir / "transcripts.csv",
        list(transcript_rows[0]),
        transcript_rows,
    )

    latency_rows = []
    for language in LANGUAGES:
        selected = [run for run in sensevoice_runs if run["language"] == language]
        values = [float(run["final_ms"]) for run in selected]
        latency_rows.append(
            {
                "language": language,
                "runs": len(selected),
                "final_ms_median": statistics.median(values),
                "final_ms_p95": percentile(values, 95),
                "rtf_median": statistics.median(float(run["rtf"]) for run in selected),
            }
        )
    write_csv(
        output_dir / "latency_by_language.csv",
        list(latency_rows[0]),
        latency_rows,
    )

    comparison = {
        "schema_version": 1,
        "scope": "same 12 EN/JA/KO/ZH Common Voice clips; fixed locale",
        "normalization": "Unicode NFKC, case fold, punctuation/symbol removal, whitespace normalization",
        "rows": rows,
        "comparison_caveat": (
            "Quality is directly comparable on the same clips and normalizer. "
            "Latency is host evidence but thread counts and runtimes differ."
        ),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "model_manifest.json").write_text(
        json.dumps(sensevoice["model"], indent=2, ensure_ascii=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
