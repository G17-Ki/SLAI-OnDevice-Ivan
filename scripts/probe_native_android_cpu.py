#!/usr/bin/env python3
"""Measure app/model CPU core-seconds for repeated native Android chat requests."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_native_android_app import request_once  # noqa: E402


PACKAGE = "ai.liquid.lfmdemo"


def shell(command: str) -> str:
    return subprocess.run(
        ["adb", "shell", command],
        text=True,
        capture_output=True,
        check=True,
        errors="replace",
    ).stdout.replace("\r", "").strip()


def cpu_ticks(pid: int) -> int:
    text = shell(f"cat /proc/{pid}/stat")
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[11]) + int(fields[12])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_audio = repo_root / "vendor" / "liquid-audio" / "assets" / "question.wav"
    audio = args.audio or default_audio
    if not audio.is_file():
        raise FileNotFoundError(audio)
    subprocess.run(["adb", "forward", "tcp:18080", "tcp:8080"], check=True, capture_output=True)
    clk_tck = int(shell("getconf CLK_TCK"))
    rows = []
    for index in range(1, args.runs + 1):
        app_pid = int(shell(f"pidof {PACKAGE}").split()[0])
        model_pid = int(shell("pidof liblfmserver.so").split()[0])
        app_before = cpu_ticks(app_pid)
        model_before = cpu_ticks(model_pid)
        result = request_once(audio, "chat", "cpu_probe_chat", index, measured=True)
        app_after = cpu_ticks(app_pid)
        model_after = cpu_ticks(model_pid)
        wall_s = result["total_ms"] / 1000.0
        app_cpu_s = (app_after - app_before) / clk_tck
        model_cpu_s = (model_after - model_before) / clk_tck
        rows.append(
            {
                "run_index": index,
                "wall_s": wall_s,
                "app_cpu_s": app_cpu_s,
                "model_cpu_s": model_cpu_s,
                "combined_cpu_s": app_cpu_s + model_cpu_s,
                "app_average_cores": app_cpu_s / wall_s,
                "model_average_cores": model_cpu_s / wall_s,
                "combined_average_cores": (app_cpu_s + model_cpu_s) / wall_s,
                "first_text_ms": result["first_text_ms"],
                "first_audio_ms": result["first_audio_ms"],
                "total_ms": result["total_ms"],
            }
        )
    report = {
        "clock_ticks_per_second": clk_tck,
        "runs": rows,
        "median_model_average_cores": statistics.median(row["model_average_cores"] for row in rows),
        "median_app_average_cores": statistics.median(row["app_average_cores"] for row in rows),
        "median_combined_average_cores": statistics.median(row["combined_average_cores"] for row in rows),
        "max_combined_average_cores": max(row["combined_average_cores"] for row in rows),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
