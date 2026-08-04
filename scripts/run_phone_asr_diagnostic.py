#!/usr/bin/env python3
"""Compare baseline and memory-saving LFM runtime configurations on the phone."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from profile_native_android_app import PACKAGE, request_once, run, wait_ready
from run_android_memory_matrix import configure
from path_config import WORK_ROOT


CONFIGURATIONS = [
    {"name": "repack_ctx512", "runtime_args": ["-c", "512"]},
    {"name": "no_repack_ctx512", "runtime_args": ["-c", "512", "--no-repack"]},
    {
        "name": "memory_gate_candidate",
        "runtime_args": [
            "-c",
            "512",
            "--no-repack",
            "-b",
            "64",
            "-ub",
            "16",
            "-ctk",
            "q4_0",
            "-ctv",
            "q4_0",
        ],
    },
    {
        "name": "memory_gate_f16kv",
        "runtime_args": ["-c", "512", "--no-repack", "-b", "64", "-ub", "16"],
    },
    {
        "name": "memory_gate_ub32",
        "runtime_args": ["-c", "512", "--no-repack", "-b", "128", "-ub", "32"],
    },
    {
        "name": "memory_gate_ub128_q4kv",
        "runtime_args": [
            "-c",
            "512",
            "--no-repack",
            "-b",
            "512",
            "-ub",
            "128",
            "-ctk",
            "q4_0",
            "-ctv",
            "q4_0",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=WORK_ROOT / "phone_asr_diagnostic",
    )
    parser.add_argument(
        "--configuration",
        action="append",
        choices=[config["name"] for config in CONFIGURATIONS],
        help="Configuration to run; repeat as needed. Defaults to repack and no-repack context-512.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for index, reference_word in enumerate(reference, 1):
        current = [index]
        for offset, prediction_word in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (reference_word != prediction_word),
                )
            )
        previous = current
    return previous[-1]


def corpus_wer(references: list[str], predictions: list[str]) -> float:
    reference_words = [word for text in references for word in normalize_text(text).split()]
    prediction_words = [word for text in predictions for word in normalize_text(text).split()]
    return 100.0 * edit_distance(reference_words, prediction_words) / max(1, len(reference_words))


def read_status_value(field: str) -> int:
    pid = run(["adb", "shell", "pidof", "liblfmserver.so"]).strip().split()[0]
    status = run(["adb", "exec-out", "run-as", PACKAGE, "cat", f"/proc/{pid}/status"])
    match = re.search(rf"^{re.escape(field)}:\s+(\d+)\s+kB", status, flags=re.MULTILINE)
    return int(match.group(1)) if match else 0


def write_report(path: Path, summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# Native Android 18-utterance ASR diagnostic",
        "",
        "This is the same dummy LibriSpeech subset, deterministic noise seed, competing-speech "
        "pairing, and text normalizer used by the earlier Colab diagnostic. WAV serialization is "
        "PCM16 for the Android HTTP interface.",
        "",
        "| Runtime | Condition | WER | Mean first text | Median total | VmHWM | Failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['configuration']} | {row['condition']} | {row['wer_percent']:.2f}% | "
            f"{row['mean_first_text_ms']:.1f} ms | {row['median_total_ms']:.1f} ms | "
            f"{row['vmhwm_mib']:.1f} MiB | {row['failures']} |"
        )
    lines += [
        "",
        "## No-repack quality delta",
        "",
        "| Condition | Baseline WER | No-repack WER | Delta | 0.3 pp gate |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparisons:
        gate = "PASS" if row["delta_wer_pp"] <= 0.3 else "FAIL"
        lines.append(
            f"| {row['condition']} | {row['baseline_wer_percent']:.2f}% | "
            f"{row['candidate_wer_percent']:.2f}% | {row['delta_wer_pp']:+.2f} pp | {gate} |"
        )
    lines += [
        "",
        "This 18-utterance suite is a controlled diagnostic, not a publication-scale benchmark. "
        "Phone results are not AR1 or glasses results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = args.corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare the corpus first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    output_dir = args.output_dir or repo_root / "reports/android_phone/asr_diagnostic" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    run(["adb", "forward", "tcp:18080", "tcp:8080"])

    detail_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    conditions = list(manifest["conditions"])
    selected_names = set(args.configuration or ("repack_ctx512", "no_repack_ctx512"))
    selected_configurations = [config for config in CONFIGURATIONS if config["name"] in selected_names]
    try:
        for config in selected_configurations:
            configure(config["runtime_args"])
            run(["adb", "shell", "am", "force-stop", PACKAGE])
            run(["adb", "exec-out", "run-as", PACKAGE, "sh", "-c", ": > files/lfm-server.log"], check=False)
            run(["adb", "shell", "am", "start", "-n", f"{PACKAGE}/.MainActivity"])
            wait_ready(timeout=60)
            for entry in manifest["entries"]:
                path = args.corpus_dir / entry["file"]
                started = time.perf_counter()
                try:
                    result = request_once(path, "asr", "phone_diagnostic", entry["sample_index"], measured=True)
                    prediction = result["transcript"]
                    error = None
                except BaseException as caught:
                    result = {"first_text_ms": None, "total_ms": (time.perf_counter() - started) * 1000}
                    prediction = ""
                    error = f"{type(caught).__name__}: {caught}"
                detail_rows.append(
                    {
                        "configuration": config["name"],
                        "runtime_args": " ".join(config["runtime_args"]),
                        "condition": entry["condition"],
                        "sample_index": entry["sample_index"],
                        "audio_seconds": entry["audio_seconds"],
                        "reference": entry["reference"],
                        "prediction": prediction,
                        "first_text_ms": result.get("first_text_ms"),
                        "total_ms": result.get("total_ms"),
                        "error": error,
                    }
                )
            vmhwm_mib = read_status_value("VmHWM") / 1024
            config_rows = [row for row in detail_rows if row["configuration"] == config["name"]]
            for condition in conditions:
                rows = [row for row in config_rows if row["condition"] == condition]
                valid = [row for row in rows if row["error"] is None]
                summaries.append(
                    {
                        "configuration": config["name"],
                        "runtime_args": " ".join(config["runtime_args"]),
                        "condition": condition,
                        "samples": len(rows),
                        "failures": len(rows) - len(valid),
                        "audio_seconds": sum(row["audio_seconds"] for row in rows),
                        "wer_percent": corpus_wer(
                            [row["reference"] for row in rows],
                            [row["prediction"] for row in rows],
                        ),
                        "mean_first_text_ms": statistics.fmean(
                            row["first_text_ms"] for row in valid if row["first_text_ms"] is not None
                        ),
                        "median_total_ms": statistics.median(row["total_ms"] for row in valid),
                        "vmhwm_mib": vmhwm_mib,
                    }
                )
            log = run(
                ["adb", "exec-out", "run-as", PACKAGE, "cat", "files/lfm-server.log"],
                check=False,
                timeout=30,
            )
            (output_dir / f"lfm-server-{config['name']}.log").write_text(log, encoding="utf-8")
    finally:
        configure([])
        subprocess.run(["adb", "shell", "am", "force-stop", PACKAGE], check=False)
        subprocess.run(
            ["adb", "shell", "am", "start", "-n", f"{PACKAGE}/.MainActivity"],
            check=False,
            capture_output=True,
        )

    comparisons = []
    if {"repack_ctx512", "no_repack_ctx512"}.issubset(selected_names):
        for condition in conditions:
            baseline = next(
                row
                for row in summaries
                if row["configuration"] == "repack_ctx512" and row["condition"] == condition
            )
            candidate = next(
                row
                for row in summaries
                if row["configuration"] == "no_repack_ctx512" and row["condition"] == condition
            )
            comparisons.append(
                {
                    "condition": condition,
                    "baseline_wer_percent": baseline["wer_percent"],
                    "candidate_wer_percent": candidate["wer_percent"],
                    "delta_wer_pp": candidate["wer_percent"] - baseline["wer_percent"],
                }
            )

    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as file:
        for row in detail_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": manifest["dataset"],
        "seed": manifest["seed"],
        "normalization": "lowercase; strip control tokens and punctuation except apostrophes; collapse whitespace",
        "configurations": selected_configurations,
        "summary": summaries,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "REPORT.md", summaries, comparisons)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
