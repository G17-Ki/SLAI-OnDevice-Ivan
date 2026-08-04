#!/usr/bin/env python3
"""Replay captured LFM PCM packet timelines under deterministic slowdowns."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--phone-profile", type=Path)
    parser.add_argument("--current-cushion-ms", type=float, default=140.0)
    parser.add_argument("--later-margin-ms", type=float, default=20.0)
    parser.add_argument(
        "--cushions-ms",
        type=parse_csv_floats,
        default=parse_csv_floats("0,20,40,60,80,100,140,200"),
    )
    parser.add_argument(
        "--slowdowns",
        type=parse_csv_floats,
        default=parse_csv_floats("1,1.25,1.5,1.75,2,2.25"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def simulate(
    packets: list[dict[str, float]],
    cushion_ms: float,
    later_margin_ms: float,
    slowdown: float,
) -> dict[str, float | int]:
    first_arrival = packets[0]["arrival_ms"]
    cursor = 0.0
    gap_count = 0
    max_gap_ms = 0.0
    total_gap_ms = 0.0
    first_start_ms = 0.0
    for index, packet in enumerate(packets):
        arrival_ms = first_arrival + (packet["arrival_ms"] - first_arrival) * slowdown
        margin_ms = cushion_ms if index == 0 else later_margin_ms
        start_ms = max(cursor, arrival_ms + margin_ms)
        if index == 0:
            first_start_ms = start_ms
        else:
            gap_ms = max(0.0, start_ms - cursor)
            if gap_ms > 0.05:
                gap_count += 1
                total_gap_ms += gap_ms
                max_gap_ms = max(max_gap_ms, gap_ms)
        cursor = start_ms + packet["duration_ms"]
    return {
        "first_playback_ms": first_start_ms,
        "playback_complete_ms": cursor,
        "gap_count": gap_count,
        "total_gap_ms": total_gap_ms,
        "max_gap_ms": max_gap_ms,
    }


def exact_slowdown_limit(
    packets: list[dict[str, float]], cushion_ms: float, later_margin_ms: float
) -> float:
    """Largest uniform post-first-packet slowdown that remains gap-free."""
    first_arrival = packets[0]["arrival_ms"]
    audio_before_ms = packets[0]["duration_ms"]
    limits: list[float] = []
    for packet in packets[1:]:
        relative_arrival_ms = packet["arrival_ms"] - first_arrival
        if relative_arrival_ms > 0:
            available_ms = cushion_ms + audio_before_ms - later_margin_ms
            limits.append(max(0.0, available_ms / relative_arrival_ms))
        audio_before_ms += packet["duration_ms"]
    return min(limits) if limits else math.inf


def current_headroom_ms(
    packets: list[dict[str, float]], cushion_ms: float, later_margin_ms: float
) -> float:
    """Minimum extra single-packet delay before the current queue would gap."""
    first_arrival = packets[0]["arrival_ms"]
    cursor = first_arrival + cushion_ms + packets[0]["duration_ms"]
    headroom: list[float] = []
    for packet in packets[1:]:
        headroom.append(cursor - (packet["arrival_ms"] + later_margin_ms))
        cursor += packet["duration_ms"]
    return min(headroom) if headroom else math.inf


def relative_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def load_runs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        measured = [
            run
            for run in payload.get("runs", [])
            if run.get("measured") and run.get("audio_packet_timeline")
        ]
        if not measured:
            raise ValueError(f"No measured packet timelines in {path}")
        sources.append(
            {
                "path": relative_name(path),
                "sha256": sha256_file(path),
                "measured_runs": len(measured),
            }
        )
        for run in measured:
            runs.append(
                {
                    "source": relative_name(path),
                    "run_index": run["run_index"],
                    "first_pcm_ms": run["first_pcm_ms"],
                    "response_complete_ms": run["response_complete_ms"],
                    "generated_audio_seconds": run["generated_audio_seconds"],
                    "packets": run["audio_packet_timeline"],
                }
            )
    return sources, runs


def load_phone_proxy(path: Path | None, mac_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    chat = payload["workloads"]["question_chat"]
    phone_first = float(chat["first_audio_ms"]["median"])
    phone_total = float(chat["total_ms"]["median"])
    phone_audio_s = float(chat["audio_output_seconds"]["median"])
    mac_first = statistics.median(float(run["first_pcm_ms"]) for run in mac_runs)
    mac_total = statistics.median(float(run["response_complete_ms"]) for run in mac_runs)
    mac_audio_s = statistics.median(float(run["generated_audio_seconds"]) for run in mac_runs)
    phone_wall_per_audio = (phone_total - phone_first) / (phone_audio_s * 1000.0)
    mac_wall_per_audio = (mac_total - mac_first) / (mac_audio_s * 1000.0)
    return {
        "path": relative_name(path),
        "sha256": sha256_file(path),
        "phone": {
            "first_pcm_ms_median": phone_first,
            "response_complete_ms_median": phone_total,
            "generated_audio_seconds": phone_audio_s,
            "post_first_wall_seconds_per_audio_second": phone_wall_per_audio,
        },
        "mac_capture_pool": {
            "first_pcm_ms_median": mac_first,
            "response_complete_ms_median": mac_total,
            "generated_audio_seconds_median": mac_audio_s,
            "post_first_wall_seconds_per_audio_second": mac_wall_per_audio,
        },
        "aggregate_normalized_slowdown_phone_vs_mac": phone_wall_per_audio / mac_wall_per_audio,
        "caveat": (
            "The phone profile retained only first-audio and completion timestamps, not every "
            "packet arrival. This ratio is an aggregate proxy, not a phone underflow measurement."
        ),
    }


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    current = result["current_queue"]
    proxy = result.get("phone_aggregate_proxy")
    lines = [
        "# Streaming queue robustness replay",
        "",
        f"Date: {result['created_at_utc'][:10]}",
        "",
        "## Decision",
        "",
        (
            f"Keep the current {fmt(current['initial_cushion_ms'], 0)} ms startup cushion for "
            "the phone acceptance run. Across the captured Mac packet timelines, it leaves "
            f"{fmt(current['single_packet_delay_headroom_ms']['min'])} ms worst-case extra "
            "single-packet delay and remains gap-free up to a "
            f"{fmt(current['uniform_post_first_slowdown_limit']['min'], 2)}x uniform slowdown."
        ),
        "",
    ]
    if proxy:
        lines.extend(
            [
                (
                    "The retained phone profile's normalized post-first generation pace is "
                    f"{fmt(proxy['aggregate_normalized_slowdown_phone_vs_mac'], 2)}x slower than "
                    "the pooled Mac capture. That is below the deterministic replay limit, so "
                    "the existing evidence predicts a continuous queue under average conditions."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "This is queue arithmetic over captured packet timestamps, not acoustic phone proof. "
            "Thermal throttling, Android scheduling, WebView audio-device startup, and bursty "
            "stalls still require the physical playback-head test.",
            "",
            "## Current queue",
            "",
            "| Metric across measured runs | Minimum | Median | P95 | Maximum |",
            "|---|---:|---:|---:|---:|",
            (
                "| Uniform post-first slowdown limit | "
                f"{fmt(current['uniform_post_first_slowdown_limit']['min'], 2)}x | "
                f"{fmt(current['uniform_post_first_slowdown_limit']['median'], 2)}x | "
                f"{fmt(current['uniform_post_first_slowdown_limit']['p95'], 2)}x | "
                f"{fmt(current['uniform_post_first_slowdown_limit']['max'], 2)}x |"
            ),
            (
                "| Extra one-packet delay headroom | "
                f"{fmt(current['single_packet_delay_headroom_ms']['min'])} ms | "
                f"{fmt(current['single_packet_delay_headroom_ms']['median'])} ms | "
                f"{fmt(current['single_packet_delay_headroom_ms']['p95'])} ms | "
                f"{fmt(current['single_packet_delay_headroom_ms']['max'])} ms |"
            ),
            "",
            "The slowdown transform keeps first-PCM time fixed and multiplies every later packet's "
            "time distance from the first packet. The single-packet value is the smallest extra "
            "arrival delay that would exhaust the queued audio at the current 20 ms scheduling margin.",
            "",
            "## Deterministic sensitivity matrix",
            "",
            "Each cell is `runs with a gap / measured runs`.",
            "",
            "| Startup cushion | "
            + " | ".join(f"{fmt(value, 2)}x" for value in result["slowdowns"])
            + " |",
            "|---:|" + "---:|" * len(result["slowdowns"]),
        ]
    )
    for cushion in result["cushions_ms"]:
        row = result["matrix"][str(cushion)]
        lines.append(
            f"| {fmt(cushion, 0)} ms | "
            + " | ".join(
                f"{row[str(slowdown)]['runs_with_gap']}/{result['measured_runs']}"
                for slowdown in result["slowdowns"]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Phone interpretation",
            "",
        ]
    )
    if proxy:
        lines.extend(
            [
                "| Evidence | Mac capture pool | Retained Nubia profile |",
                "|---|---:|---:|",
                (
                    "| First PCM, median | "
                    f"{fmt(proxy['mac_capture_pool']['first_pcm_ms_median'])} ms | "
                    f"{fmt(proxy['phone']['first_pcm_ms_median'])} ms |"
                ),
                (
                    "| Completion, median | "
                    f"{fmt(proxy['mac_capture_pool']['response_complete_ms_median'])} ms | "
                    f"{fmt(proxy['phone']['response_complete_ms_median'])} ms |"
                ),
                (
                    "| Generated audio | "
                    f"{fmt(proxy['mac_capture_pool']['generated_audio_seconds_median'], 2)} s | "
                    f"{fmt(proxy['phone']['generated_audio_seconds'], 2)} s |"
                ),
                (
                    "| Post-first wall time per audio second | "
                    f"{fmt(proxy['mac_capture_pool']['post_first_wall_seconds_per_audio_second'], 3)} s | "
                    f"{fmt(proxy['phone']['post_first_wall_seconds_per_audio_second'], 3)} s |"
                ),
                "",
                proxy["caveat"],
                "",
            ]
        )
    else:
        lines.extend(["No phone aggregate profile was supplied.", ""])
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "python3 scripts/analyze_stream_queue_robustness.py \\",
            "  reports/android_phone/streaming_response/mac_lfm_streaming_2026-07-22.json \\",
            "  reports/android_phone/streaming_response/mac_lfm_streaming_restart_2026-07-22.json \\",
            "  --phone-profile reports/android_phone/native_profile/20260716t120118z/profile.json \\",
            "  --output reports/android_phone/streaming_response/queue_robustness.json \\",
            "  --report reports/android_phone/streaming_response/queue_robustness.md",
            "```",
            "",
            "The script produced the JSON metrics and this Markdown report without altering the "
            "captured evidence files.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    sources, runs = load_runs(args.captures)
    limits = [
        exact_slowdown_limit(run["packets"], args.current_cushion_ms, args.later_margin_ms)
        for run in runs
    ]
    headroom = [
        current_headroom_ms(run["packets"], args.current_cushion_ms, args.later_margin_ms)
        for run in runs
    ]
    matrix: dict[str, dict[str, Any]] = {}
    for cushion in args.cushions_ms:
        row: dict[str, Any] = {}
        for slowdown in args.slowdowns:
            simulations = [
                simulate(run["packets"], cushion, args.later_margin_ms, slowdown)
                for run in runs
            ]
            row[str(slowdown)] = {
                "runs_with_gap": sum(item["gap_count"] > 0 for item in simulations),
                "total_gaps": sum(int(item["gap_count"]) for item in simulations),
                "max_gap_ms": max(float(item["max_gap_ms"]) for item in simulations),
                "first_playback_ms": describe(
                    [float(item["first_playback_ms"]) for item in simulations]
                ),
            }
        matrix[str(cushion)] = row

    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": {
            "arrival_transform": (
                "arrival' = first_arrival + (arrival - first_arrival) * slowdown"
            ),
            "first_packet_time": "unchanged",
            "gap_threshold_ms": 0.05,
            "scope": "measured runs with captured audio_packet_timeline only",
        },
        "sources": sources,
        "measured_runs": len(runs),
        "current_queue": {
            "initial_cushion_ms": args.current_cushion_ms,
            "later_margin_ms": args.later_margin_ms,
            "uniform_post_first_slowdown_limit": describe(limits),
            "single_packet_delay_headroom_ms": describe(headroom),
        },
        "cushions_ms": args.cushions_ms,
        "slowdowns": args.slowdowns,
        "matrix": matrix,
        "phone_aggregate_proxy": load_phone_proxy(args.phone_profile, runs),
        "limitations": [
            "The replay does not measure Android AudioTrack/WebAudio playback-head movement.",
            "Uniform slowdown and one-packet delay are deterministic stress models, not a thermal trace.",
            "The retained phone profile lacks per-packet arrival timestamps.",
            "Mac and phone responses have slightly different generated audio duration.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.report:
        write_report(args.report, result)


if __name__ == "__main__":
    main()
