#!/usr/bin/env python3
"""Measure streamed LFM text/audio timing and simulate the Android playback queue."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="16 kHz mono WAV input")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--mode", choices=("chat", "asr"), default="chat")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--initial-cushion-ms", type=float, default=140.0)
    parser.add_argument("--later-margin-ms", type=float, default=20.0)
    parser.add_argument(
        "--compare-cushions-ms",
        default="100,140,200,300,400,500",
        help="Comma-separated startup cushions to replay against the captured packet timeline",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; at n <= 10, p95 is the sample maximum."""
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def payload(audio: Path, mode: str, max_tokens: int | None) -> bytes:
    is_chat = mode == "chat"
    request = {
        "model": "LFM2.5-Audio-1.5B",
        "stream": True,
        "modalities": ["text", "audio"] if is_chat else ["text"],
        "temperature": 0,
        "max_tokens": max_tokens or (192 if is_chat else 256),
        "messages": [
            {
                "role": "system",
                "content": "Respond with interleaved text and audio." if is_chat else "Perform ASR.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
    }
    return json.dumps(request, separators=(",", ":")).encode("utf-8")


def decode_audio_delta(delta: dict[str, Any]) -> tuple[int, int] | None:
    """Return (sample_count, sample_rate) for legacy float32 or current PCM16."""
    legacy = delta.get("audio_chunk")
    current = delta.get("audio")
    item = legacy or current
    if not item:
        return None
    data = item if isinstance(item, str) else item.get("data")
    if not data:
        return None
    sample_rate = 24000 if isinstance(item, str) else int(item.get("sample_rate", 24000))
    encoding = (
        ("pcm16" if current else "float32")
        if isinstance(item, str)
        else str(item.get("format") or item.get("encoding") or ("pcm16" if current else "float32"))
    )
    byte_count = len(base64.b64decode(data))
    normalized_encoding = encoding.lower()
    if any(marker in normalized_encoding for marker in ("float32", "f32", "32f")):
        bytes_per_sample = 4
    elif any(marker in normalized_encoding for marker in ("pcm16", "s16", "16le", "int16")):
        bytes_per_sample = 2
    else:
        bytes_per_sample = 4 if legacy else 2
    return byte_count // bytes_per_sample, sample_rate


def is_speakable_prefix(text: str) -> bool:
    """Heuristic boundary for when a standalone TTS engine could start safely."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    words = re.findall(r"[\w']+", compact, flags=re.UNICODE)
    cjk_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", compact)
    content_chars = len(re.sub(r"\W", "", compact, flags=re.UNICODE))
    if re.search(r"[.!?。！？][\"'”’）)]?$", compact) and content_chars >= 12:
        return True
    if re.search(r"[,;:，、；：][\"'”’）)]?$", compact) and (len(words) >= 4 or len(cjk_chars) >= 12):
        return True
    return len(words) >= 8 or len(cjk_chars) >= 24


def simulate_playback(
    packets: list[dict[str, float]], initial_cushion_ms: float, later_margin_ms: float
) -> dict[str, Any]:
    cursor = 0.0
    underflows = 0
    gap_count = 0
    total_gap_ms = 0.0
    max_gap_ms = 0.0
    trace: list[dict[str, float | bool]] = []
    for index, packet in enumerate(packets):
        arrival = packet["arrival_ms"] / 1000.0
        duration = packet["duration_ms"] / 1000.0
        is_first = index == 0
        prior_cursor = cursor
        underflow = not is_first and prior_cursor < arrival
        if underflow:
            underflows += 1
        margin = initial_cushion_ms if is_first else later_margin_ms
        start = max(prior_cursor, arrival + margin / 1000.0)
        gap_ms = 0.0 if is_first else max(0.0, start - prior_cursor) * 1000.0
        if gap_ms > 0.05:
            gap_count += 1
            total_gap_ms += gap_ms
            max_gap_ms = max(max_gap_ms, gap_ms)
        cursor = start + duration
        trace.append(
            {
                "arrival_ms": packet["arrival_ms"],
                "duration_ms": packet["duration_ms"],
                "scheduled_start_ms": start * 1000.0,
                "buffer_ahead_before_ms": (prior_cursor - arrival) * 1000.0 if not is_first else 0.0,
                "gap_ms": gap_ms,
                "underflow": underflow,
            }
        )
    return {
        "initial_cushion_ms": initial_cushion_ms,
        "later_margin_ms": later_margin_ms,
        "first_playback_scheduled_ms": None if not trace else trace[0]["scheduled_start_ms"],
        "playback_complete_scheduled_ms": None if not trace else cursor * 1000.0,
        "underflows": underflows,
        "gap_count": gap_count,
        "total_gap_ms": total_gap_ms,
        "max_gap_ms": max_gap_ms,
        "trace": trace,
    }


def request_once(args: argparse.Namespace, run_index: int, measured: bool) -> dict[str, Any]:
    request = urllib.request.Request(
        args.url,
        data=payload(args.audio, args.mode, args.max_tokens),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    headers_at: float | None = None
    first_event_at: float | None = None
    first_text_at: float | None = None
    first_speakable_at: float | None = None
    first_speakable_text: str | None = None
    first_pcm_at: float | None = None
    previous_pcm_at: float | None = None
    largest_pcm_arrival_gap = 0.0
    text_parts: list[str] = []
    packets: list[dict[str, float]] = []
    audio_chunks = 0
    audio_samples = 0
    sample_rate: int | None = None
    first_chunk_samples: int | None = None
    event_count = 0

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        headers_at = time.perf_counter()
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            now = time.perf_counter()
            if first_event_at is None:
                first_event_at = now
            event = json.loads(data)
            event_count += 1
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if text:
                if first_text_at is None:
                    first_text_at = now
                text_parts.append(text)
                current_text = "".join(text_parts).strip()
                if first_speakable_at is None and is_speakable_prefix(current_text):
                    first_speakable_at = now
                    first_speakable_text = current_text

            decoded = decode_audio_delta(delta)
            if decoded is None:
                continue
            chunk_samples, chunk_rate = decoded
            if first_pcm_at is None:
                first_pcm_at = now
                first_chunk_samples = chunk_samples
            if previous_pcm_at is not None:
                largest_pcm_arrival_gap = max(largest_pcm_arrival_gap, now - previous_pcm_at)
            previous_pcm_at = now
            audio_chunks += 1
            audio_samples += chunk_samples
            sample_rate = chunk_rate

            packets.append(
                {
                    "arrival_ms": (now - started) * 1000.0,
                    "duration_ms": chunk_samples / chunk_rate * 1000.0,
                }
            )

    ended = time.perf_counter()
    cushions = [
        float(value.strip())
        for value in args.compare_cushions_ms.split(",")
        if value.strip()
    ]
    if args.initial_cushion_ms not in cushions:
        cushions.append(args.initial_cushion_ms)
    simulations = {
        f"{cushion:g}": simulate_playback(packets, cushion, args.later_margin_ms)
        for cushion in sorted(set(cushions))
    }
    selected = simulations[f"{args.initial_cushion_ms:g}"]
    return {
        "run_index": run_index,
        "measured": measured,
        "mode": args.mode,
        "audio": args.audio.name,
        "url": args.url,
        "headers_ms": None if headers_at is None else (headers_at - started) * 1000,
        "first_event_ms": None if first_event_at is None else (first_event_at - started) * 1000,
        "first_text_ms": None if first_text_at is None else (first_text_at - started) * 1000,
        "first_speakable_text_ms": (
            None if first_speakable_at is None else (first_speakable_at - started) * 1000
        ),
        "first_speakable_text": first_speakable_text,
        "first_pcm_ms": None if first_pcm_at is None else (first_pcm_at - started) * 1000,
        "first_playback_scheduled_ms": selected["first_playback_scheduled_ms"],
        "response_complete_ms": (ended - started) * 1000,
        "playback_complete_scheduled_ms": selected["playback_complete_scheduled_ms"],
        "audio_stream_tail_buffer_ms": (
            None
            if not audio_chunks
            else float(selected["playback_complete_scheduled_ms"]) - (ended - started) * 1000
        ),
        "largest_pcm_arrival_gap_ms": largest_pcm_arrival_gap * 1000,
        "playback_underflows_simulated": selected["underflows"],
        "playback_gap_count_simulated": selected["gap_count"],
        "playback_max_gap_ms_simulated": selected["max_gap_ms"],
        "playback_total_gap_ms_simulated": selected["total_gap_ms"],
        "playback_simulations": simulations,
        "audio_packet_timeline": packets,
        "audio_chunks": audio_chunks,
        "audio_samples": audio_samples,
        "sample_rate": sample_rate,
        "first_chunk_ms": (
            None
            if first_chunk_samples is None or sample_rate is None
            else first_chunk_samples / sample_rate * 1000
        ),
        "generated_audio_seconds": (
            None if not audio_samples or sample_rate is None else audio_samples / sample_rate
        ),
        "sse_events": event_count,
        "text": "".join(text_parts).strip(),
    }


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if run["measured"]]
    keys = (
        "first_text_ms",
        "first_speakable_text_ms",
        "first_pcm_ms",
        "first_playback_scheduled_ms",
        "response_complete_ms",
        "playback_complete_scheduled_ms",
        "largest_pcm_arrival_gap_ms",
        "generated_audio_seconds",
    )
    timing: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(run[key]) for run in measured if run[key] is not None]
        if values:
            timing[key] = describe(values)
    return {
        "measured_runs": len(measured),
        "timing": timing,
        "runs_with_simulated_underflow": sum(
            bool(run["playback_underflows_simulated"]) for run in measured
        ),
        "total_simulated_underflows": sum(
            int(run["playback_underflows_simulated"]) for run in measured
        ),
        "runs_with_simulated_gap": sum(bool(run["playback_gap_count_simulated"]) for run in measured),
        "max_simulated_gap_ms": max(
            (float(run["playback_max_gap_ms_simulated"]) for run in measured), default=0.0
        ),
        "cushion_comparison": {
            cushion: {
                "runs_with_gap": sum(
                    bool(run["playback_simulations"][cushion]["gap_count"]) for run in measured
                ),
                "total_gaps": sum(
                    int(run["playback_simulations"][cushion]["gap_count"]) for run in measured
                ),
                "max_gap_ms": max(
                    (
                        float(run["playback_simulations"][cushion]["max_gap_ms"])
                        for run in measured
                    ),
                    default=0.0,
                ),
                "first_playback_scheduled_ms": describe(
                    [
                        float(run["playback_simulations"][cushion]["first_playback_scheduled_ms"])
                        for run in measured
                        if run["playback_simulations"][cushion]["first_playback_scheduled_ms"]
                        is not None
                    ]
                ),
            }
            for cushion in measured[0]["playback_simulations"]
        }
        if measured
        else {},
    }


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file does not exist: {args.audio}")
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be positive and --warmups cannot be negative")

    runs: list[dict[str, Any]] = []
    total = args.warmups + args.runs
    for index in range(total):
        measured = index >= args.warmups
        result = request_once(args, index, measured)
        runs.append(result)
        phase = "measure" if measured else "warmup"
        print(
            f"{phase} {index + 1}/{total}: "
            f"text={result['first_text_ms']!s} ms, "
            f"pcm={result['first_pcm_ms']!s} ms, "
            f"scheduled={result['first_playback_scheduled_ms']!s} ms, "
            f"done={result['response_complete_ms']:.1f} ms, "
            f"underflows={result['playback_underflows_simulated']}",
            flush=True,
        )

    document = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_audio": {
            "filename": args.audio.name,
            "sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        },
        "playback_simulation": {
            "initial_cushion_ms": args.initial_cushion_ms,
            "later_margin_ms": args.later_margin_ms,
            "note": (
                "Scheduled playback is a deterministic simulation of the Android WebView queue. "
                "It is not an acoustic measurement of the speaker output."
            ),
        },
        "summary": build_summary(runs),
        "runs": runs,
    }
    output = args.output
    if output is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
        output = Path("reports/android_phone/streaming_response") / f"stream_profile_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(json.dumps(document["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
