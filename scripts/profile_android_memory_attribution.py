#!/usr/bin/env python3
"""Attribute native Android LFM residency at idle and during generation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from profile_native_android_app import PACKAGE, default_asset_dir, request_once, run, wait_ready


HEADER = re.compile(
    r"^(?P<start>[0-9a-f]+)-(?P<end>[0-9a-f]+)\s+"
    r"(?P<perms>\S+)\s+(?P<offset>[0-9a-f]+)\s+"
    r"(?P<device>\S+)\s+(?P<inode>\d+)(?:\s+(?P<path>.*))?$"
)
MEMORY_FIELDS = {
    "Size",
    "Rss",
    "Pss",
    "Shared_Clean",
    "Shared_Dirty",
    "Private_Clean",
    "Private_Dirty",
    "Referenced",
    "Anonymous",
    "AnonHugePages",
    "Swap",
    "SwapPss",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=default_asset_dir() / "question.wav")
    parser.add_argument("--mode", choices=("asr", "chat"), default="chat")
    parser.add_argument("--interval", type=float, default=0.20)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def adb_run_as(*arguments: str, check: bool = True, timeout: float = 120) -> str:
    return run(
        ["adb", "exec-out", "run-as", PACKAGE, *arguments],
        check=check,
        timeout=timeout,
    )


def model_pid() -> int:
    output = run(["adb", "shell", "pidof", "liblfmserver.so"]).strip()
    if not output:
        raise RuntimeError("The native LFM model process is not running")
    return int(output.split()[0])


def read_process_file(pid: int, name: str) -> str:
    return adb_run_as("cat", f"/proc/{pid}/{name}", timeout=60)


def sanitize_path(path: str) -> str:
    path = re.sub(
        rf"/data/(?:user/0|data)/{re.escape(PACKAGE)}/files/models/",
        "/app-private/models/",
        path,
    )
    path = re.sub(
        rf"/data/app/[^/]*{re.escape(PACKAGE)}[^/]*/",
        "/data/app/<package-install>/",
        path,
    )
    return path


def category_for(path: str) -> tuple[str, str]:
    clean = sanitize_path(path.strip())
    lower = clean.lower()
    if lower.endswith(".gguf"):
        name = Path(clean).name
        return "gguf_weights", name
    if not clean or clean.startswith("[anon:") or clean in {"[heap]", "[stack]"}:
        return "anonymous", clean or "<anonymous>"
    if lower.endswith(".so") or ".so!" in lower or ".so (deleted)" in lower:
        return "shared_libraries", Path(clean.split("!")[0]).name
    if any(token in lower for token in (".apk", ".dex", ".oat", ".vdex", ".art", ".jar")):
        return "android_runtime_files", Path(clean.split("!")[0]).name
    if clean.startswith("/dev/"):
        return "device_mappings", clean
    if clean.startswith("["):
        return "special_mappings", clean
    return "other_file_backed", clean


def parse_smaps(text: str) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        header = HEADER.match(line)
        if header:
            if current is not None:
                mappings.append(current)
            current = {
                **header.groupdict(),
                "path": header.group("path") or "",
                **{field: 0 for field in MEMORY_FIELDS},
            }
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key not in MEMORY_FIELDS:
            continue
        match = re.search(r"(\d+)", value)
        if match:
            current[key] = int(match.group(1))
    if current is not None:
        mappings.append(current)
    for mapping in mappings:
        category, detail = category_for(mapping["path"])
        mapping["category"] = category
        mapping["detail"] = detail
        mapping["path"] = sanitize_path(mapping["path"])
    return mappings


def parse_status(text: str) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        match = re.fullmatch(r"(\d+)\s+kB", value)
        result[key] = int(match.group(1)) if match else value
    return result


def summarize(mappings: list[dict[str, Any]], status: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    totals = {field: 0 for field in MEMORY_FIELDS}
    for mapping in mappings:
        key = (mapping["category"], mapping["detail"])
        row = grouped.setdefault(
            key,
            {
                "category": mapping["category"],
                "detail": mapping["detail"],
                "mapping_count": 0,
                **{field: 0 for field in MEMORY_FIELDS},
            },
        )
        row["mapping_count"] += 1
        for field in MEMORY_FIELDS:
            row[field] += mapping[field]
            totals[field] += mapping[field]
    rows = sorted(grouped.values(), key=lambda item: item["Rss"], reverse=True)
    category_totals: dict[str, dict[str, int]] = defaultdict(lambda: {field: 0 for field in MEMORY_FIELDS})
    for row in rows:
        for field in MEMORY_FIELDS:
            category_totals[row["category"]][field] += row[field]
    return {
        "status": status,
        "smaps_totals_kib": totals,
        "category_totals_kib": dict(category_totals),
        "rows": rows,
    }


def capture(pid: int) -> dict[str, Any]:
    smaps = read_process_file(pid, "smaps")
    status = parse_status(read_process_file(pid, "status"))
    mappings = parse_smaps(smaps)
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": pid,
        "raw_smaps": smaps,
        "summary": summarize(mappings, status),
    }


def mib(kib: int | float) -> str:
    return f"{kib / 1024:.1f}"


def write_rows(path: Path, captures: dict[str, dict[str, Any]]) -> None:
    fields = [
        "capture",
        "category",
        "detail",
        "mapping_count",
        "Size",
        "Rss",
        "Pss",
        "Shared_Clean",
        "Shared_Dirty",
        "Private_Clean",
        "Private_Dirty",
        "Anonymous",
        "Swap",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, capture_item in captures.items():
            for item in capture_item["summary"]["rows"]:
                writer.writerow({"capture": name, **{field: item.get(field, 0) for field in fields[1:]}})


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Android LFM memory attribution",
        "",
        f"Created: {report['created_utc']}",
        "",
        "All values are KiB-derived MiB from `/proc/<pid>/smaps`. The active capture is the "
        "highest-RSS sample observed while one request was executing.",
        "",
        "| Capture | smaps RSS | VmRSS | VmHWM | PSS |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("idle", "active_peak", "after"):
        item = report["captures"][name]["summary"]
        status = item["status"]
        lines.append(
            f"| {name} | {mib(item['smaps_totals_kib']['Rss'])} | "
            f"{mib(int(status.get('VmRSS', 0)))} | {mib(int(status.get('VmHWM', 0)))} | "
            f"{mib(item['smaps_totals_kib']['Pss'])} |"
        )
    lines += ["", "## Active-peak attribution", "", "| Category | RSS MiB | PSS MiB |", "|---|---:|---:|"]
    peak = report["captures"]["active_peak"]["summary"]
    for category, values in sorted(
        peak["category_totals_kib"].items(), key=lambda pair: pair[1]["Rss"], reverse=True
    ):
        lines.append(f"| {category} | {mib(values['Rss'])} | {mib(values['Pss'])} |")
    lines += ["", "## Per-GGUF active residency", "", "| File | RSS MiB | PSS MiB |", "|---|---:|---:|"]
    ggufs = [item for item in peak["rows"] if item["category"] == "gguf_weights"]
    for item in ggufs:
        lines.append(f"| `{item['detail']}` | {mib(item['Rss'])} | {mib(item['Pss'])} |")
    lines += [
        "",
        "## Request",
        "",
        f"- Mode: `{report['request']['mode']}`.",
        f"- Input: `{report['request']['audio']}`.",
        f"- First text: {report['request']['first_text_ms']:.1f} ms.",
        f"- First audio: {report['request']['first_audio_ms']:.1f} ms."
        if report["request"]["first_audio_ms"] is not None
        else "- First audio: not applicable.",
        f"- Total: {report['request']['total_ms']:.1f} ms.",
        "",
        "The raw peak smaps file has app-private installation paths normalized. "
        "`mapping_groups.csv` contains every grouped mapping for independent checking.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    repo_root = Path(__file__).resolve().parents[1]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    output_dir = args.output_dir or repo_root / "reports/android_phone/memory_attribution" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    if "device" not in run(["adb", "get-state"]):
        raise RuntimeError("No authorized Android device")
    run(["adb", "forward", "tcp:18080", "tcp:8080"])
    if not args.no_restart:
        run(["adb", "shell", "am", "force-stop", PACKAGE])
        adb_run_as("sh", "-c", ": > files/lfm-server.log", check=False)
        run(["adb", "shell", "am", "start", "-n", f"{PACKAGE}/.MainActivity"])
        wait_ready(timeout=60)
    time.sleep(args.settle_seconds)

    pid = model_pid()
    captures: dict[str, dict[str, Any]] = {"idle": capture(pid)}
    request_result: dict[str, Any] = {}
    request_error: list[BaseException] = []

    def run_request() -> None:
        try:
            request_result.update(request_once(args.audio, args.mode, "memory_attribution", 1, measured=True))
        except BaseException as error:  # Preserve the profiler output if the request fails.
            request_error.append(error)

    worker = threading.Thread(target=run_request, name="memory-attribution-request")
    worker.start()
    active_samples: list[dict[str, Any]] = []
    while worker.is_alive():
        active_samples.append(capture(pid))
        time.sleep(args.interval)
    worker.join()
    if request_error:
        raise request_error[0]
    if not active_samples:
        active_samples.append(capture(pid))
    captures["active_peak"] = max(
        active_samples,
        key=lambda item: item["summary"]["smaps_totals_kib"]["Rss"],
    )
    time.sleep(args.settle_seconds)
    captures["after"] = capture(pid)

    for name, item in captures.items():
        raw = item.pop("raw_smaps")
        sanitized = "\n".join(
            sanitize_path(line) if HEADER.match(line) else line for line in raw.splitlines()
        )
        (output_dir / f"{name}.smaps.txt").write_text(sanitized + "\n", encoding="utf-8")

    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "package": PACKAGE,
        "pid": pid,
        "sampling_interval_s": args.interval,
        "active_sample_count": len(active_samples),
        "request": request_result,
        "captures": captures,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_rows(output_dir / "mapping_groups.csv", captures)
    write_report(output_dir / "REPORT.md", report)
    server_log = adb_run_as("cat", "files/lfm-server.log", check=False, timeout=30)
    (output_dir / "lfm-server.log").write_text(server_log, encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
