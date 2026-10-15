#!/usr/bin/env python3
"""Pull and summarize the Android streamed-audio monotonic telemetry log."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


PACKAGE = "ai.liquid.lfmdemo.dev"
TELEMETRY_PATH = "files/streaming-telemetry.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Summarize an existing pull instead of contacting ADB.",
    )
    parser.add_argument(
        "--clear-after",
        action="store_true",
        help="Delete only the app-private telemetry file after a successful pull.",
    )
    return parser.parse_args()


def run(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        errors="replace",
        check=False,
    )


def pull_from_phone() -> str:
    devices = run(["adb", "devices", "-l"])
    connected = [
        line
        for line in devices.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if len(connected) != 1:
        raise SystemExit(
            f"Expected exactly one authorized ADB device, found {len(connected)}"
        )
    pulled = run(
        ["adb", "exec-out", "run-as", PACKAGE, "cat", TELEMETRY_PATH],
        timeout=60,
    )
    if pulled.returncode != 0:
        raise SystemExit(
            "Could not read app-private telemetry. Install and run the "
            f"instrumented APK first. ADB said: {pulled.stderr.strip()}"
        )
    return pulled.stdout


def strict_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSON on telemetry line {line_number}: {error}")
        if not isinstance(event, dict):
            raise SystemExit(f"Telemetry line {line_number} is not a JSON object")
        name = event.get("event")
        timestamp = event.get("monotonic_ms")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"Telemetry line {line_number} has no event name")
        if not isinstance(timestamp, int | float) or not math.isfinite(timestamp):
            raise SystemExit(f"Telemetry line {line_number} has no finite monotonic_ms")
        events.append(event)
    if not events:
        raise SystemExit("Telemetry log contains no events")
    return events


def first(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((event for event in events if event["event"] == name), None)


def last(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((event for event in reversed(events) if event["event"] == name), None)


def timestamp(event: dict[str, Any] | None) -> float | None:
    return None if event is None else float(event["monotonic_ms"])


def difference(
    start: dict[str, Any] | None, end: dict[str, Any] | None
) -> float | None:
    start_ms = timestamp(start)
    end_ms = timestamp(end)
    if start_ms is None or end_ms is None:
        return None
    return end_ms - start_ms


def detail_number(event: dict[str, Any] | None, key: str) -> float | None:
    if event is None:
        return None
    value = (event.get("details") or {}).get(key)
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def scheduled_request_latency(
    request: dict[str, Any] | None, queued: dict[str, Any] | None
) -> float | None:
    queued_after_request = difference(request, queued)
    scheduled_delay = detail_number(queued, "scheduled_delay_ms")
    if queued_after_request is not None and scheduled_delay is not None:
        return queued_after_request + scheduled_delay
    return detail_number(queued, "scheduled_request_elapsed_ms")


def output_head_request_latency(
    request: dict[str, Any] | None, output: dict[str, Any] | None
) -> float | None:
    if request is not None and output is not None:
        request_performance = request.get("performance_ms")
        output_performance = detail_number(output, "performance_time")
        if (
            isinstance(request_performance, int | float)
            and math.isfinite(request_performance)
            and output_performance is not None
        ):
            return output_performance - float(request_performance)
    return detail_number(output, "request_elapsed_ms")


def summarize_turn(turn_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda event: float(event["monotonic_ms"]))
    stop = first(ordered, "input_stop_requested")
    wav = first(ordered, "wav_ready")
    request = first(ordered, "request_start")
    text = first(ordered, "first_text_sse")
    pcm = first(ordered, "first_pcm_sse")
    queued = first(ordered, "first_buffer_queued")
    output = first(ordered, "output_head_result")
    voiced_scheduled = first(ordered, "first_voiced_audio_scheduled")
    voiced_output = first(ordered, "first_voiced_output_head_estimate")
    response_complete = first(ordered, "response_complete")
    tts_text_complete = first(ordered, "tts_text_sse_complete")
    lfm_tts_complete = first(ordered, "lfm_tts_response_complete")
    lfm_interleaved_complete = first(ordered, "lfm_interleaved_response_complete")
    generation_complete = (
        tts_text_complete
        or lfm_tts_complete
        or lfm_interleaved_complete
        or response_complete
    )
    language_resolved = last(ordered, "language_resolved")
    tts_init_requested = first(ordered, "native_tts_init_requested")
    tts_ready = first(ordered, "native_tts_init_ready")
    tts_first_chunk = first(ordered, "native_tts_chunk_queued")
    tts_first_start = first(ordered, "native_tts_utterance_start")
    tts_first_audio_proxy = first(ordered, "native_tts_first_audio_start_proxy")
    tts_queue_closed = first(ordered, "native_tts_queue_closed") or first(
        ordered, "native_tts_response_complete"
    )
    tts_terminal = first(ordered, "native_tts_playback_terminal")
    tts_last_done = last(ordered, "native_tts_utterance_done")
    tts_chunks = [event for event in ordered if event["event"] == "native_tts_chunk_queued"]
    tts_cancels = [event for event in ordered if event["event"] == "native_tts_queue_cancel"]
    error_names = {
        "turn_error",
        "native_tts_init_error",
        "native_tts_error",
        "native_tts_utterance_error",
    }
    errors = [event for event in ordered if event["event"] in error_names]
    underflows = [event for event in ordered if event["event"] == "queue_underflow"]
    quality_warnings = [
        event for event in ordered if event["event"] == "response_quality_warning"
    ]
    completion_details = (generation_complete or {}).get("details") or {}
    language_details = (language_resolved or {}).get("details") or {}
    native_tts_used = tts_init_requested is not None
    terminal_status = (
        (tts_terminal.get("details") or {}).get("status") if tts_terminal else None
    )
    complete_event = tts_terminal if native_tts_used else generation_complete
    is_complete = (
        tts_text_complete is not None
        and terminal_status == "completed"
        and not errors
        if native_tts_used
        else generation_complete is not None and not errors
    )
    return {
        "turn_id": turn_id,
        "mode": next((event.get("mode") for event in ordered if event.get("mode")), None),
        "events": ordered,
        "event_names": [event["event"] for event in ordered],
        "timing_ms": {
            "manual_input_stop_to_wav_ready": difference(stop, wav),
            "manual_input_stop_to_request_start": difference(stop, request),
            "request_to_first_text_sse": difference(request, text),
            "request_to_first_pcm_sse": difference(request, pcm),
            "request_to_first_buffer_queued_event": difference(request, queued),
            "request_to_scheduled_first_buffer": scheduled_request_latency(
                request, queued
            ),
            "request_to_output_head": output_head_request_latency(request, output),
            "request_to_first_voiced_audio_scheduled": detail_number(
                voiced_scheduled, "request_elapsed_ms"
            ),
            "request_to_first_voiced_output_head_estimate": detail_number(
                voiced_output, "request_elapsed_ms"
            ),
            "request_to_native_tts_init_ready": difference(request, tts_ready),
            "request_to_native_tts_first_chunk_queued": difference(
                request, tts_first_chunk
            ),
            "request_to_native_tts_utterance_start": difference(
                request, tts_first_start
            ),
            "request_to_native_tts_first_audio_start_proxy": difference(
                request, tts_first_audio_proxy
            ),
            "request_to_tts_text_sse_complete": difference(
                request, tts_text_complete
            ),
            "request_to_native_tts_queue_closed": difference(
                request, tts_queue_closed
            ),
            "request_to_native_tts_playback_terminal": difference(
                request, tts_terminal
            ),
            "request_to_native_tts_last_utterance_done": difference(
                request, tts_last_done
            ),
            "request_to_complete": difference(request, complete_event),
        },
        "output_head_status": (output.get("details") or {}).get("status")
        if output
        else None,
        "queue_underflows": len(underflows),
        "asr": {
            "selected_language": language_details.get("selected_language"),
            "resolved_language": language_details.get("resolved_language"),
            "backend": language_details.get("backend"),
            "provider": language_details.get("provider"),
            "npu_executed": language_details.get("npu_executed"),
            "npu_selected": language_details.get("npu_selected"),
        },
        "output_quality": {
            "finish_reason": completion_details.get("finish_reason"),
            "response_token_limit": completion_details.get("response_token_limit"),
            "pcm_duration_seconds": completion_details.get("pcm_duration_seconds"),
            "sse_events": completion_details.get("sse_events"),
            "repetition": completion_details.get("repetition"),
            "warning_reasons": [
                (event.get("details") or {}).get("reason")
                for event in quality_warnings
                if (event.get("details") or {}).get("reason")
            ],
        },
        "native_tts": {
            "used": native_tts_used,
            "resolved_locale": (tts_ready.get("details") or {}).get("resolved_locale")
            if tts_ready
            else None,
            "voice_name": (tts_ready.get("details") or {}).get("voice_name")
            if tts_ready
            else None,
            "network_connection_required": (
                tts_ready.get("details") or {}
            ).get("network_connection_required")
            if tts_ready
            else None,
            "chunks_queued": len(tts_chunks),
            "queue_cancelled": bool(tts_cancels),
            "text_sse_complete": tts_text_complete is not None,
            "queue_closed": tts_queue_closed is not None,
            "playback_terminal": tts_terminal is not None,
            "playback_terminal_status": terminal_status,
        },
        "errors": errors,
        "complete": is_complete,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def metric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def aggregate(turns: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            name
            for turn in turns
            for name in turn.get("timing_ms", {})
        }
    )
    metrics = {}
    for name in metric_names:
        values = [
            float(turn["timing_ms"][name])
            for turn in turns
            if turn["timing_ms"].get(name) is not None
        ]
        metrics[name] = metric_summary(values)
    return {
        "turns": len(turns),
        "complete_turns": sum(bool(turn["complete"]) for turn in turns),
        "turns_with_underflow": sum(turn["queue_underflows"] > 0 for turn in turns),
        "total_underflows": sum(turn["queue_underflows"] for turn in turns),
        "native_tts_turns": sum(bool(turn["native_tts"]["used"]) for turn in turns),
        "native_tts_cancelled_turns": sum(
            bool(turn["native_tts"]["queue_cancelled"]) for turn in turns
        ),
        "native_tts_playback_completed_turns": sum(
            turn["native_tts"]["playback_terminal_status"] == "completed"
            for turn in turns
        ),
        "native_tts_playback_failed_turns": sum(
            turn["native_tts"]["playback_terminal_status"] == "failed"
            for turn in turns
        ),
        "npu_executed_turns": sum(
            turn.get("asr", {}).get("npu_executed") is True for turn in turns
        ),
        "generation_length_cap_turns": sum(
            turn.get("output_quality", {}).get("finish_reason") == "length"
            for turn in turns
        ),
        "quality_warning_turns": sum(
            bool(turn.get("output_quality", {}).get("warning_reasons"))
            for turn in turns
        ),
        "repetition_warning_turns": sum(
            "repeated_four_grams"
            in turn.get("output_quality", {}).get("warning_reasons", [])
            for turn in turns
        ),
        "output_head_status_counts": {
            status: sum(turn["output_head_status"] == status for turn in turns)
            for status in sorted(
                {turn["output_head_status"] for turn in turns if turn["output_head_status"]}
            )
        },
        "timing_ms": metrics,
    }


def markdown(summary: dict[str, Any]) -> str:
    rows = [
        "# Android streaming telemetry summary",
        "",
        f"Generated: {summary['created_at_utc']}",
        "",
        "This trace uses Android elapsed-realtime timestamps. For microphone turns,",
        "`input_stop_requested` is the user's/manual stop proxy; it is not an",
        "acoustically measured last-speech sample or a production VAD-close event.",
        "",
        "## Result",
        "",
        f"- Turns: {summary['aggregate']['turns']}",
        f"- Complete turns: {summary['aggregate']['complete_turns']}",
        f"- Turns with queue underflow: {summary['aggregate']['turns_with_underflow']}",
        f"- Total underflows: {summary['aggregate']['total_underflows']}",
        f"- Native Android TTS turns: {summary['aggregate']['native_tts_turns']}",
        f"- Native TTS playback-completed turns: "
        f"{summary['aggregate']['native_tts_playback_completed_turns']}",
        f"- Native TTS playback-failed turns: "
        f"{summary['aggregate']['native_tts_playback_failed_turns']}",
        f"- Native TTS turns cancelled/flushed by user lifecycle: "
        f"{summary['aggregate']['native_tts_cancelled_turns']}",
        f"- Turns whose ASR executed on NPU: {summary['aggregate']['npu_executed_turns']}",
        f"- Turns that reached the generation cap: "
        f"{summary['aggregate']['generation_length_cap_turns']}",
        f"- Turns with output-quality warnings: "
        f"{summary['aggregate']['quality_warning_turns']}",
        f"- Turns with repetition warnings: "
        f"{summary['aggregate']['repetition_warning_turns']}",
        "",
        "| Metric | Count | Median (ms) | P95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in summary["aggregate"]["timing_ms"].items():
        def show(value: float | None) -> str:
            return "—" if value is None else f"{value:.1f}"

        rows.append(
            f"| `{name}` | {values['count']} | {show(values['median'])} | "
            f"{show(values['p95'])} | {show(values['max'])} |"
        )
    rows.extend(
        [
            "",
            "The WebAudio output-head timestamp excludes the downstream Android",
            "mixer/HAL, DAC, speaker, and acoustic propagation. External recording is",
            "still required for the final audible-onset claim. Likewise, Android TTS",
            "`UtteranceProgressListener.onStart` is only a native playback-start proxy;",
            "it is not a microphone-observed acoustic onset.",
            "For native TTS, `tts_text_sse_complete` means only that model text ended,",
            "and `native_tts_queue_closed` means only that no more clauses will be",
            "submitted. A turn is complete only when `native_tts_playback_terminal`",
            "has status `completed`.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    text = (
        args.input_jsonl.expanduser().read_text(encoding="utf-8")
        if args.input_jsonl
        else pull_from_phone()
    )
    events = strict_events(text)
    raw_path = output_dir / "telemetry.jsonl"
    raw_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unscoped = []
    for event in events:
        turn_id = event.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            grouped[turn_id].append(event)
        else:
            unscoped.append(event)
    turns = [summarize_turn(turn_id, rows) for turn_id, rows in sorted(grouped.items())]
    summary = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(args.input_jsonl) if args.input_jsonl else "adb run-as app-private telemetry",
        "raw_event_count": len(events),
        "unscoped_native_events": unscoped,
        "turns": turns,
        "aggregate": aggregate(turns),
        "interpretation": {
            "manual_stop_is_not_speech_end": True,
            "output_head_is_not_acoustic_onset": True,
            "first_voiced_output_head_includes_pcm_silence_scan": True,
            "first_voiced_output_head_is_still_not_external_acoustic_onset": True,
            "native_tts_on_start_is_not_acoustic_onset": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(markdown(summary), encoding="utf-8")

    if args.clear_after and not args.input_jsonl:
        cleared = run(["adb", "shell", "run-as", PACKAGE, "rm", "-f", TELEMETRY_PATH])
        if cleared.returncode != 0:
            raise SystemExit(f"Pulled successfully but could not clear telemetry: {cleared.stderr}")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
