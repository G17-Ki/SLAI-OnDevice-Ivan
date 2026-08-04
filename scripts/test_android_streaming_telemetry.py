#!/usr/bin/env python3
"""Dependency-free checks for Android streaming telemetry summarization."""

from __future__ import annotations

import json
from typing import Any

from pull_android_streaming_telemetry import aggregate, strict_events, summarize_turn


def event(
    name: str,
    at: float,
    *,
    turn_id: str = "turn-1",
    mode: str = "tts",
    **details: Any,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "turn_id": turn_id,
        "event": name,
        "mode": mode,
        "monotonic_ms": at,
        "performance_ms": at,
        "details": details,
    }


def main() -> None:
    rows = [
        event("input_stop_requested", 1000.0),
        event("wav_ready", 1060.0),
        event("request_start", 1100.0),
        event(
            "language_resolved",
            1180.0,
            selected_language="auto",
            resolved_language="ru",
            backend="zipformer_ru_cpu",
            provider="cpu",
            npu_executed=True,
            npu_selected=False,
        ),
        event("native_tts_init_requested", 1110.0, requested_locale="ja-JP"),
        event(
            "native_tts_init_ready",
            1200.0,
            resolved_locale="ja-JP",
            voice_name="offline-ja",
            network_connection_required=False,
        ),
        event("first_text_sse", 1500.0),
        event("native_tts_chunk_queued", 1510.0, chunk_index=0),
        event("native_tts_utterance_start", 1575.0, chunk_index=0),
        event("native_tts_first_audio_start_proxy", 1580.0, chunk_index=0),
        event("first_pcm_sse", 1600.0),
        event(
            "first_buffer_queued",
            1605.0,
            scheduled_delay_ms=140.0,
            scheduled_request_elapsed_ms=9999.0,
        ),
        event(
            "output_head_result",
            1748.0,
            status="observed",
            performance_time=1748.0,
            request_elapsed_ms=9999.0,
        ),
        event(
            "tts_text_sse_complete",
            2800.0,
            finish_reason="length",
            response_token_limit=256,
            pcm_duration_seconds=0.0,
            sse_events=42,
            repetition={
                "token_count": 20,
                "gram_size": 4,
                "duplicate_gram_ratio": 0.3,
                "max_gram_count": 3,
            },
        ),
        event(
            "response_quality_warning",
            2801.0,
            reason="repeated_four_grams",
            finish_reason="length",
        ),
        event(
            "response_quality_warning",
            2802.0,
            reason="generation_length_cap",
            finish_reason="length",
        ),
        event("native_tts_queue_closed", 2820.0, pending_utterances=1),
        event("native_tts_utterance_done", 3000.0, pending_utterances=0),
        event(
            "native_tts_playback_terminal",
            3010.0,
            status="completed",
            pending_utterances=0,
        ),
    ]
    parsed = strict_events("".join(json.dumps(row) + "\n" for row in rows))
    turn = summarize_turn("turn-1", parsed)
    assert turn["complete"] is True
    assert turn["queue_underflows"] == 0
    assert turn["timing_ms"]["manual_input_stop_to_wav_ready"] == 60.0
    assert turn["timing_ms"]["request_to_first_text_sse"] == 400.0
    assert turn["timing_ms"]["request_to_first_pcm_sse"] == 500.0
    assert turn["timing_ms"]["request_to_scheduled_first_buffer"] == 645.0
    assert turn["timing_ms"]["request_to_output_head"] == 648.0
    assert turn["timing_ms"]["request_to_native_tts_init_ready"] == 100.0
    assert turn["timing_ms"]["request_to_native_tts_first_chunk_queued"] == 410.0
    assert turn["timing_ms"]["request_to_native_tts_first_audio_start_proxy"] == 480.0
    assert turn["timing_ms"]["request_to_native_tts_last_utterance_done"] == 1900.0
    assert turn["timing_ms"]["request_to_tts_text_sse_complete"] == 1700.0
    assert turn["timing_ms"]["request_to_native_tts_queue_closed"] == 1720.0
    assert turn["timing_ms"]["request_to_native_tts_playback_terminal"] == 1910.0
    assert turn["timing_ms"]["request_to_complete"] == 1910.0
    assert turn["native_tts"]["used"] is True
    assert turn["native_tts"]["resolved_locale"] == "ja-JP"
    assert turn["native_tts"]["network_connection_required"] is False
    assert turn["native_tts"]["chunks_queued"] == 1
    assert turn["native_tts"]["text_sse_complete"] is True
    assert turn["native_tts"]["queue_closed"] is True
    assert turn["native_tts"]["playback_terminal_status"] == "completed"
    assert turn["asr"]["backend"] == "zipformer_ru_cpu"
    assert turn["asr"]["provider"] == "cpu"
    assert turn["asr"]["npu_executed"] is True
    assert turn["asr"]["npu_selected"] is False
    assert turn["output_quality"]["finish_reason"] == "length"
    assert turn["output_quality"]["response_token_limit"] == 256
    assert turn["output_quality"]["pcm_duration_seconds"] == 0.0
    assert turn["output_quality"]["sse_events"] == 42
    assert turn["output_quality"]["repetition"]["max_gram_count"] == 3
    assert turn["output_quality"]["warning_reasons"] == [
        "repeated_four_grams",
        "generation_length_cap",
    ]
    summary = aggregate([turn])
    assert summary["turns"] == 1
    assert summary["complete_turns"] == 1
    assert summary["native_tts_turns"] == 1
    assert summary["native_tts_playback_completed_turns"] == 1
    assert summary["npu_executed_turns"] == 1
    assert summary["generation_length_cap_turns"] == 1
    assert summary["quality_warning_turns"] == 1
    assert summary["repetition_warning_turns"] == 1
    assert summary["timing_ms"]["request_to_output_head"]["p95"] == 648.0

    pending_rows = [row for row in rows if row["event"] != "native_tts_playback_terminal"]
    pending_turn = summarize_turn("turn-1", pending_rows)
    assert pending_turn["native_tts"]["queue_closed"] is True
    assert pending_turn["native_tts"]["playback_terminal"] is False
    assert pending_turn["complete"] is False

    stuck_rows = [
        row
        for row in rows
        if row["event"]
        not in {"native_tts_utterance_done", "native_tts_playback_terminal"}
    ]
    stuck_turn = summarize_turn("turn-1", stuck_rows)
    assert stuck_turn["native_tts"]["queue_closed"] is True
    assert stuck_turn["native_tts"]["playback_terminal"] is False
    assert stuck_turn["complete"] is False

    failed_rows = pending_rows + [
        event("native_tts_utterance_error", 3005.0, error_code=-1),
        event(
            "native_tts_playback_terminal",
            3010.0,
            status="failed",
            pending_utterances=0,
        ),
    ]
    failed_turn = summarize_turn("turn-1", failed_rows)
    assert failed_turn["native_tts"]["playback_terminal_status"] == "failed"
    assert failed_turn["complete"] is False
    assert failed_turn["errors"]

    cancelled_rows = pending_rows + [
        event("native_tts_queue_cancel", 2900.0, reason="new_input"),
        event(
            "native_tts_playback_terminal",
            2910.0,
            status="cancelled",
            pending_utterances=0,
        ),
    ]
    cancelled_turn = summarize_turn("turn-1", cancelled_rows)
    assert cancelled_turn["native_tts"]["playback_terminal_status"] == "cancelled"
    assert cancelled_turn["native_tts"]["queue_cancelled"] is True
    assert cancelled_turn["complete"] is False

    chat_rows = [
        event("request_start", 100.0, turn_id="chat-1", mode="chat"),
        event("response_complete", 500.0, turn_id="chat-1", mode="chat"),
    ]
    chat_turn = summarize_turn("chat-1", chat_rows)
    assert chat_turn["native_tts"]["used"] is False
    assert chat_turn["complete"] is True
    assert chat_turn["timing_ms"]["request_to_complete"] == 400.0

    direct_rows = [
        event("request_start", 100.0, turn_id="direct-1", mode="chat"),
        event("first_text_sse", 260.0, turn_id="direct-1", mode="chat"),
        event("first_pcm_sse", 280.0, turn_id="direct-1", mode="chat"),
        event(
            "lfm_interleaved_response_complete",
            900.0,
            turn_id="direct-1",
            mode="chat",
            finish_reason="stop",
            response_token_limit=512,
            pcm_duration_seconds=2.4,
            sse_events=18,
        ),
    ]
    direct_turn = summarize_turn("direct-1", direct_rows)
    assert direct_turn["complete"] is True
    assert direct_turn["timing_ms"]["request_to_complete"] == 800.0
    assert direct_turn["output_quality"]["finish_reason"] == "stop"
    assert direct_turn["output_quality"]["response_token_limit"] == 512

    exact_lfm_rows = [
        event("request_start", 100.0, turn_id="exact-lfm-1", mode="chat"),
        event(
            "lfm_tts_response_complete",
            1100.0,
            turn_id="exact-lfm-1",
            mode="chat",
            finish_reason="stop",
        ),
    ]
    exact_lfm_turn = summarize_turn("exact-lfm-1", exact_lfm_rows)
    assert exact_lfm_turn["complete"] is True
    assert exact_lfm_turn["timing_ms"]["request_to_complete"] == 1000.0
    print("android streaming telemetry tests passed")


if __name__ == "__main__":
    main()
