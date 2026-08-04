#!/usr/bin/env python3
"""Run reproducible context, KV, mmap, batch, or audio-length memory matrices."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from profile_native_android_app import PACKAGE, default_asset_dir, run


MATRICES: dict[str, list[dict[str, Any]]] = {
    "context": [
        {"name": f"ctx_{size}", "args": ["-c", str(size)]}
        for size in (512, 1024, 2048, 4096, 8192)
    ],
    "kv": [
        {"name": "kv_f16", "args": ["-c", "512", "--no-repack", "-ctk", "f16", "-ctv", "f16"]},
        {"name": "kv_q8_0", "args": ["-c", "512", "--no-repack", "-ctk", "q8_0", "-ctv", "q8_0"]},
        {"name": "kv_q4_0", "args": ["-c", "512", "--no-repack", "-ctk", "q4_0", "-ctv", "q4_0"]},
    ],
    "mmap": [
        {"name": "mmap_default", "args": ["-c", "512"]},
        {"name": "no_mmap", "args": ["-c", "512", "--no-mmap"]},
        {"name": "mlock", "args": ["-c", "512", "--mlock"]},
        {"name": "no_mmap_mlock", "args": ["-c", "512", "--no-mmap", "--mlock"]},
    ],
    "batch": [
        {"name": "batch_default", "args": ["-c", "512", "--no-repack"]},
        {"name": "batch_1024_ubatch_256", "args": ["-c", "512", "--no-repack", "-b", "1024", "-ub", "256"]},
        {"name": "batch_512_ubatch_128", "args": ["-c", "512", "--no-repack", "-b", "512", "-ub", "128"]},
        {"name": "batch_256_ubatch_64", "args": ["-c", "512", "--no-repack", "-b", "256", "-ub", "64"]},
    ],
    "repack": [
        {"name": "repack_default", "args": ["-c", "512"]},
        {"name": "no_repack", "args": ["-c", "512", "--no-repack"]},
    ],
    "gate": [
        {"name": "no_repack_default", "args": ["-c", "512", "--no-repack"]},
        {
            "name": "no_repack_b128_ub32",
            "args": ["-c", "512", "--no-repack", "-b", "128", "-ub", "32"],
        },
        {
            "name": "no_repack_b64_ub16",
            "args": ["-c", "512", "--no-repack", "-b", "64", "-ub", "16"],
        },
        {
            "name": "no_repack_b64_ub16_q4kv",
            "args": [
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
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=(*MATRICES, "audio"))
    parser.add_argument("--mode", choices=("asr", "chat"), default="asr")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--leave-last", action="store_true")
    return parser.parse_args()


def configure(arguments: list[str]) -> None:
    if not arguments:
        subprocess.run(
            ["adb", "exec-out", "run-as", PACKAGE, "rm", "-f", "files/runtime.args"],
            check=False,
            capture_output=True,
        )
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as file:
        for argument in arguments:
            file.write(argument + "\n")
        local = Path(file.name)
    remote = "/data/local/tmp/lfm-runtime.args"
    try:
        subprocess.run(["adb", "push", str(local), remote], check=True, capture_output=True)
        subprocess.run(
            ["adb", "exec-out", "run-as", PACKAGE, "cp", remote, "files/runtime.args"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["adb", "shell", "rm", "-f", remote], check=True, capture_output=True)
    finally:
        local.unlink(missing_ok=True)


def synthetic_30_seconds(source: Path, output: Path) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    bytes_per_frame = params.nchannels * params.sampwidth
    target_frames = int(params.framerate * 30)
    repeats = (target_frames * bytes_per_frame + len(frames) - 1) // len(frames)
    payload = (frames * repeats)[: target_frames * bytes_per_frame]
    with wave.open(str(output), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(payload)


def audio_matrix(output_dir: Path) -> list[dict[str, Any]]:
    assets = default_asset_dir()
    synthetic = output_dir / "synthetic_30s.wav"
    synthetic_30_seconds(assets / "asr.wav", synthetic)
    return [
        {"name": "audio_4p9s", "args": [], "audio": assets / "question.wav"},
        {"name": "audio_18p4s", "args": [], "audio": assets / "asr.wav"},
        {"name": "audio_30s", "args": [], "audio": synthetic},
    ]


def extract_row(name: str, arguments: list[str], summary: dict[str, Any]) -> dict[str, Any]:
    peak = summary["captures"]["active_peak"]["summary"]
    idle = summary["captures"]["idle"]["summary"]
    request = summary["request"]
    categories = peak["category_totals_kib"]
    return {
        "name": name,
        "runtime_args": " ".join(arguments) or "<default>",
        "idle_rss_mib": idle["smaps_totals_kib"]["Rss"] / 1024,
        "active_peak_rss_mib": peak["smaps_totals_kib"]["Rss"] / 1024,
        "vmhwm_mib": int(peak["status"].get("VmHWM", 0)) / 1024,
        "gguf_rss_mib": categories.get("gguf_weights", {}).get("Rss", 0) / 1024,
        "anonymous_rss_mib": categories.get("anonymous", {}).get("Rss", 0) / 1024,
        "first_text_ms": request.get("first_text_ms"),
        "first_audio_ms": request.get("first_audio_ms"),
        "total_ms": request.get("total_ms"),
        "exact_normalized": request.get("exact_normalized"),
        "transcript": request.get("transcript"),
        "error": None,
    }


def write_outputs(output_dir: Path, matrix: str, rows: list[dict[str, Any]]) -> None:
    (output_dir / "matrix.json").write_text(
        json.dumps({"matrix": matrix, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "name",
        "runtime_args",
        "idle_rss_mib",
        "active_peak_rss_mib",
        "vmhwm_mib",
        "gguf_rss_mib",
        "anonymous_rss_mib",
        "first_text_ms",
        "first_audio_ms",
        "total_ms",
        "exact_normalized",
        "transcript",
        "error",
    ]
    with (output_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Android {matrix} memory matrix",
        "",
        "Each row is one fresh model-process launch followed by one request. Values are exploratory "
        "single-run measurements; they are not latency distributions.",
        "",
        "| Configuration | Args | Idle RSS MiB | Active RSS MiB | VmHWM MiB | Anonymous MiB | Total ms | Exact ASR |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['name']} | `{row.get('runtime_args', '')}` | — | — | — | — | — | failed |")
            continue
        exact = "—" if row.get("exact_normalized") is None else str(row["exact_normalized"])
        lines.append(
            f"| {row['name']} | `{row['runtime_args']}` | {row['idle_rss_mib']:.1f} | "
            f"{row['active_peak_rss_mib']:.1f} | {row['vmhwm_mib']:.1f} | "
            f"{row['anonymous_rss_mib']:.1f} | {row['total_ms']:.1f} | {exact} |"
        )
    lines += [
        "",
        "For failed configurations, inspect that configuration's `failure.json`. Every successful "
        "configuration retains its smaps tables and server load log.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if "device" not in run(["adb", "get-state"]):
        raise RuntimeError("No authorized Android device")
    repo_root = Path(__file__).resolve().parents[1]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    output_dir = args.output_dir or repo_root / "reports/android_phone/memory_matrix" / f"{stamp}_{args.matrix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    configurations = audio_matrix(output_dir) if args.matrix == "audio" else MATRICES[args.matrix]
    profiler = Path(__file__).with_name("profile_android_memory_attribution.py")
    rows: list[dict[str, Any]] = []
    try:
        for config in configurations:
            name = config["name"]
            arguments = config["args"]
            run_dir = output_dir / name
            configure(arguments)
            command = [
                sys.executable,
                str(profiler),
                "--mode",
                args.mode,
                "--output-dir",
                str(run_dir),
            ]
            selected_audio = config.get("audio") or args.audio
            if selected_audio is not None:
                command += ["--audio", str(selected_audio)]
            try:
                subprocess.run(command, check=True, text=True)
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                rows.append(extract_row(name, arguments, summary))
            except BaseException as error:
                run_dir.mkdir(parents=True, exist_ok=True)
                failure = {
                    "name": name,
                    "runtime_args": arguments,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                (run_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                rows.append(
                    {
                        "name": name,
                        "runtime_args": " ".join(arguments) or "<default>",
                        "error": str(error),
                    }
                )
    finally:
        if not args.leave_last:
            configure([])
            subprocess.run(["adb", "shell", "am", "force-stop", PACKAGE], check=False)
            subprocess.run(
                ["adb", "shell", "am", "start", "-n", f"{PACKAGE}/.MainActivity"],
                check=False,
                capture_output=True,
            )
    write_outputs(output_dir, args.matrix, rows)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
