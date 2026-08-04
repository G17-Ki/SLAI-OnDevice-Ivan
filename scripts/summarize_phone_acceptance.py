#!/usr/bin/env python3
"""Create a compact public-safe summary from a Nubia WebView acceptance run.

The raw harness output is intentionally not publication-ready: it contains
temporary fixture paths, a WebView target identifier, full prompts, and stack
locations. This script retains the behavioral and timing evidence while
dropping those machine-local details.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        events.append(value)
    return events


def rounded(value: Any) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    rounded_value = round(number, 3)
    return int(rounded_value) if rounded_value.is_integer() else rounded_value


def compact_history(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    limits = value.get("limits") if isinstance(value.get("limits"), dict) else {}
    return {
        "turns": value.get("turns"),
        "characters": value.get("characters"),
        "approximate_tokens": value.get("approximateTokens"),
        "limits": {
            "max_turns": limits.get("maxTurns"),
            "max_characters": limits.get("maxCharacters"),
            "max_approximate_tokens": limits.get("maxApproxTokens"),
        },
    }


def event_session(event: dict[str, Any]) -> str | None:
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    return event.get("session_id") or details.get("session_id")


def event_counts(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = collections.Counter(str(event.get("event", "")) for event in events)
    counter.pop("", None)
    return dict(sorted(counter.items()))


def derive_case_verdict(case: dict[str, Any]) -> str:
    assertions = (
        case.get("acceptanceAssertions")
        if isinstance(case.get("acceptanceAssertions"), dict)
        else {}
    )
    all_passed = assertions.get("allPassed")
    expectations = (
        case.get("expectations") if isinstance(case.get("expectations"), dict) else {}
    )
    if all_passed is True:
        return (
            "pass_expected_block"
            if expectations.get("outcome") == "blocked"
            else "pass"
        )
    if all_passed is False:
        return "fail"
    if case.get("thrown"):
        return "error_unclassified"
    return "unclassified"


def compact_quiet_rapid(value: dict[str, Any]) -> dict[str, Any]:
    snapshots = value.get("snapshots") if isinstance(value.get("snapshots"), list) else []
    by_test = {
        str(item.get("test")): item
        for item in snapshots
        if isinstance(item, dict) and item.get("test")
    }
    quiet_started = by_test.get("quiet_started", {})
    quiet_stopped = by_test.get("quiet_stopped", {})
    rapid = []
    for index in range(1, 4):
        item = by_test.get(f"rapid_{index}", {})
        state = item.get("state") if isinstance(item.get("state"), dict) else {}
        rapid.append(
            {
                "cycle": index,
                "start_accepted": item.get("accepted"),
                "stop_accepted": item.get("stopped"),
                "final_phase": state.get("phase"),
                "last_error": state.get("lastError"),
            }
        )
    start_state = (
        quiet_started.get("state")
        if isinstance(quiet_started.get("state"), dict)
        else {}
    )
    stop_state = (
        quiet_stopped.get("state")
        if isinstance(quiet_stopped.get("state"), dict)
        else {}
    )
    final_state = value.get("finalState") if isinstance(value.get("finalState"), dict) else {}
    return {
        "mode": value.get("mode"),
        "input_mode": value.get("inputMode"),
        "quiet_session": {
            "start_accepted": quiet_started.get("accepted"),
            "reached_phase": start_state.get("phase"),
            "stop_accepted": quiet_stopped.get("accepted"),
            "final_phase": stop_state.get("phase"),
            "last_error": stop_state.get("lastError"),
        },
        "rapid_cycles": rapid,
        "final_phase": final_state.get("phase"),
        "last_error": final_state.get("lastError"),
        "visible_error": value.get("error"),
    }


def turn_metrics_by_turn(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in events:
        turn_id = event.get("turn_id")
        if not turn_id:
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        item = result.setdefault(str(turn_id), {})
        name = event.get("event")
        if name == "asr_request_complete":
            item["asr"] = {
                "selected_language": details.get("selected_language"),
                "resolved_language": details.get("resolved_language"),
                "first_text_ms": rounded(details.get("first_text_ms")),
                "backend": details.get("backend"),
                "provider": details.get("provider"),
                "asr_companion": details.get("asr_companion"),
                "npu_executed": details.get("npu_executed"),
                "npu_selected": details.get("npu_selected"),
                "npu_gate": details.get("npu_gate"),
                "npu_fallback_reason": details.get("npu_fallback_reason"),
                "candidate_summaries": details.get("candidate_summaries"),
            }
        elif name == "response_backend_request_start":
            item["response_backend"] = details.get("response_backend")
            item["response_model"] = details.get("response_model")
        elif name == "first_text_sse":
            item["first_response_token_ms"] = rounded(details.get("request_elapsed_ms"))
            item["response_backend_first_token_ms"] = rounded(
                details.get("response_backend_elapsed_ms")
            )
            if details.get("response_backend") is not None:
                item["response_backend"] = details.get("response_backend")
        elif name in {
            "response_complete",
            "tts_text_sse_complete",
            "lfm_tts_response_complete",
            "lfm_interleaved_response_complete",
        }:
            first_output_head = rounded(details.get("first_output_head_ms"))
            first_native_tts = rounded(details.get("first_native_tts_start_proxy_ms"))
            item.update(
                {
                    "completion_event": name,
                    "first_text_ms": rounded(details.get("first_text_ms")),
                    "first_pcm_ms": rounded(details.get("first_pcm_ms")),
                    "first_audio_scheduled_ms": rounded(
                        details.get("first_audio_scheduled_ms")
                    ),
                    "first_output_head_ms": first_output_head,
                    "first_voiced_pcm_offset_ms": rounded(
                        details.get("first_voiced_pcm_offset_ms")
                    ),
                    "first_voiced_scheduled_ms": rounded(
                        details.get("first_voiced_scheduled_ms")
                    ),
                    "first_voiced_output_head_ms": rounded(
                        details.get("first_voiced_output_head_ms")
                    ),
                    "first_native_tts_start_proxy_ms": first_native_tts,
                    "first_audio_proxy_ms": (
                        first_output_head
                        if first_output_head is not None
                        else first_native_tts
                    ),
                    "response_complete_ms": rounded(details.get("request_elapsed_ms")),
                    "output_head_status": details.get("output_head_status"),
                    "playback_underflows": details.get("playback_underflows", 0),
                    "audio_chunks": details.get("audio_chunks", 0),
                    "pcm_duration_seconds": rounded(
                        details.get("pcm_duration_seconds")
                    ),
                    "finish_reason": details.get("finish_reason"),
                    "sse_events": details.get("sse_events"),
                    "response_token_limit": details.get("response_token_limit"),
                    "repetition": details.get("repetition"),
                    "native_tts_locale": details.get("native_tts_locale"),
                    "native_tts_chunks_submitted": details.get(
                        "native_tts_chunks_submitted"
                    ),
                    "native_tts_chunks_confirmed": details.get(
                        "native_tts_chunks_confirmed"
                    ),
                    "voice_backend": details.get("voice_backend"),
                    "history_mutated": details.get("history_mutated"),
                    "context_enabled": details.get("context_enabled"),
                    "history_turns": details.get("history_turns"),
                    "backend": details.get("backend"),
                    "provider": details.get("provider"),
                    "npu_executed": details.get("npu_executed"),
                    "npu_selected": details.get("npu_selected"),
                    "response_backend": details.get("response_backend")
                    or item.get("response_backend"),
                    "response_model": details.get("response_model")
                    or item.get("response_model"),
                }
            )
        elif name == "response_quality_warning":
            item["response_quality_warning"] = {
                "reason": details.get("reason"),
                "finish_reason": details.get("finish_reason"),
                "response_token_limit": details.get("response_token_limit"),
                "token_count": details.get("token_count"),
                "gram_size": details.get("gram_size"),
                "duplicate_gram_ratio": rounded(details.get("duplicate_gram_ratio")),
                "max_gram_count": details.get("max_gram_count"),
            }
        elif name == "native_tts_utterance_start":
            monotonic_ms = rounded(event.get("monotonic_ms"))
            current = item.get("native_tts_first_start_monotonic_ms")
            if monotonic_ms is not None and (current is None or monotonic_ms < current):
                item["native_tts_first_start_monotonic_ms"] = monotonic_ms
        elif name == "native_tts_playback_terminal":
            item["native_tts_playback_terminal"] = {
                "status": details.get("status"),
                "message": details.get("message"),
            }
            item["native_tts_terminal_monotonic_ms"] = rounded(event.get("monotonic_ms"))
    for item in result.values():
        start = item.get("native_tts_first_start_monotonic_ms")
        terminal = item.get("native_tts_terminal_monotonic_ms")
        item["native_tts_playback_span_ms"] = (
            rounded(float(terminal) - float(start))
            if start is not None and terminal is not None and float(terminal) >= float(start)
            else None
        )
    return result


def compact_companion_fetch(fetch: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_language": fetch.get("requestedLanguage"),
        "method": fetch.get("method"),
        "content_type": fetch.get("contentType"),
        "wav_bytes": fetch.get("bodyBytes"),
        "http_status": fetch.get("httpStatus"),
        "http_ok": fetch.get("httpOk"),
        "fetch_error": fetch.get("fetchError"),
    }


def compact_response_fetch(fetch: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_backend": fetch.get("requestBackend"),
        "request_model": fetch.get("requestModelIdentity"),
        "response_model": fetch.get("responseModelIdentity"),
        "max_tokens": fetch.get("maxTokens"),
        "thinking_enabled": fetch.get("enableThinking"),
        "http_status": fetch.get("httpStatus"),
        "http_ok": fetch.get("httpOk"),
        "finish_reason": fetch.get("finishReason"),
        "reasoning_content_characters": fetch.get("reasoningContentCharacters"),
        "think_tag_seen": fetch.get("thinkTagSeen"),
        "response_characters": fetch.get("responseCharacters"),
        "sse_done_seen": fetch.get("sseDoneSeen"),
        "sse_parse_errors": fetch.get("sseParseErrors"),
        "fetch_error": fetch.get("fetchError"),
        "inspection_error": fetch.get("responseInspectionError"),
    }


def model_identity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).rsplit("/", 1)[-1]
    return normalized[:-5] if normalized.lower().endswith(".gguf") else normalized


def compact_case(case: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    thrown = case.get("thrown") if isinstance(case.get("thrown"), dict) else None
    fetches = case.get("fetches") if isinstance(case.get("fetches"), list) else []
    language_route = (
        case.get("languageRoute") if isinstance(case.get("languageRoute"), dict) else {}
    )
    asr_facts = case.get("asrFacts") if isinstance(case.get("asrFacts"), dict) else {}
    response_request = case.get("responseRequestText")
    companion_fetches = [
        compact_companion_fetch(fetch)
        for fetch in fetches
        if fetch.get("stage") == "asr_companion"
    ]
    response_fetches = [
        compact_response_fetch(fetch)
        for fetch in fetches
        if fetch.get("stage") in {"response", "response_companion"}
    ]
    metric = metrics.get(str(case.get("turnId"))) or {}
    expected = case.get("expectations") if isinstance(case.get("expectations"), dict) else {}
    response_fetch = response_fetches[-1] if response_fetches else {}
    observed_backend = metric.get("response_backend") or response_fetch.get("request_backend")
    observed_model = model_identity(
        metric.get("response_model")
        or response_fetch.get("response_model")
        or response_fetch.get("request_model")
    )
    terminal = (
        metric.get("native_tts_playback_terminal")
        if isinstance(metric.get("native_tts_playback_terminal"), dict)
        else {}
    )
    telemetry_assertions = {
        "response_backend": (
            observed_backend == expected.get("responseBackend")
            if expected.get("responseBackend")
            else None
        ),
        "response_model": (
            observed_model == expected.get("responseModel")
            if expected.get("responseModel")
            else None
        ),
        "finish_reason": (
            (metric.get("finish_reason") or response_fetch.get("finish_reason"))
            == expected.get("finishReason")
            if expected.get("finishReason")
            else None
        ),
        "max_first_audio_ms": (
            metric.get("first_audio_proxy_ms") is not None
            and float(metric["first_audio_proxy_ms"]) <= float(expected["maxFirstAudioMs"])
            if expected.get("maxFirstAudioMs") is not None
            else None
        ),
        "min_native_tts_playback_ms": (
            metric.get("native_tts_playback_span_ms") is not None
            and float(metric["native_tts_playback_span_ms"])
            >= float(expected["minNativeTtsPlaybackMs"])
            if expected.get("minNativeTtsPlaybackMs") is not None
            else None
        ),
        "tts_terminal_status": (
            terminal.get("status") == expected.get("ttsTerminalStatus")
            if expected.get("ttsTerminalStatus")
            else None
        ),
    }
    raw_assertions = (
        case.get("acceptanceAssertions")
        if isinstance(case.get("acceptanceAssertions"), dict)
        else {}
    )
    classified = [
        value
        for key, value in {**raw_assertions, **telemetry_assertions}.items()
        if key != "allPassed" and isinstance(value, bool)
    ]
    overall_passed = (
        all(classified)
        if classified
        else raw_assertions.get("allPassed")
        if isinstance(raw_assertions.get("allPassed"), bool)
        else None
    )
    verdict = derive_case_verdict(
        {
            "expectations": expected,
            "acceptanceAssertions": {"allPassed": overall_passed},
            "thrown": thrown,
        }
    )
    return {
        "name": case.get("name"),
        "verdict": verdict,
        "server_warmth": case.get("serverWarmth"),
        "requested_mode": case.get("requestedMode"),
        "selected_language": case.get("selectedLanguage"),
        "resolved_language": language_route.get("language"),
        "context_enabled_requested": case.get("contextEnabledRequested"),
        "context_enabled_observed": case.get("historyEnabled"),
        "context_switch_aria_checked": case.get("historySwitchAriaChecked"),
        "voice_backend": case.get("voiceBackend"),
        "asr": {
            "backend": asr_facts.get("backend"),
            "provider": asr_facts.get("provider"),
            "npu_executed": asr_facts.get("npu_executed"),
            "npu_selected": asr_facts.get("npu_selected"),
            "npu_gate": asr_facts.get("npu_gate"),
            "npu_fallback_reason": asr_facts.get("npu_fallback_reason"),
            "candidate_summaries": asr_facts.get("candidate_summaries"),
        },
        "response": case.get("response"),
        "transcript": case.get("transcript"),
        "response_request_characters": (
            len(response_request) if isinstance(response_request, str) else 0
        ),
        "visible_error": case.get("error"),
        "thrown_error": None if thrown is None else thrown.get("message"),
        "expected": expected,
        "acceptance_assertions": raw_assertions,
        "telemetry_assertions": telemetry_assertions,
        "all_assertions_passed": overall_passed,
        "timer_text": case.get("timer"),
        "harness_elapsed_ms_including_playback_drain": rounded(case.get("elapsedMs")),
        "latency": metric or None,
        "history_before": compact_history(case.get("historyBefore")),
        "history_after": compact_history(case.get("historyAfter")),
        "protocol_assertions": case.get("protocolAssertions"),
        "request_stages": [fetch.get("stage") for fetch in fetches],
        "companion_requests": companion_fetches,
        "response_requests": response_fetches,
        "drain": case.get("drain"),
    }


def compact_static(matrix: dict[str, Any]) -> dict[str, Any]:
    static = matrix.get("staticInspection") if isinstance(matrix.get("staticInspection"), dict) else {}
    native = static.get("nativeBridge") if isinstance(static.get("nativeBridge"), dict) else {}
    response = (
        static.get("responseCompanion")
        if isinstance(static.get("responseCompanion"), dict)
        else {}
    )
    return {
        "model_ready": static.get("modelReady"),
        "device_status": static.get("deviceStatus"),
        "language_options": static.get("languageOptions"),
        "controls": static.get("controls"),
        "history_enabled": static.get("historyEnabled"),
        "history_switch": static.get("historySwitch"),
        "asr_companion": static.get("asrCompanion"),
        "response_companion": {
            "backend": response.get("backend"),
            "request_model": model_identity(response.get("requestModel")),
            "health_http_status": (response.get("health") or {}).get("httpStatus"),
            "models_http_status": (response.get("models") or {}).get("httpStatus"),
            "model_ids": response.get("modelIds"),
        },
        "native_bridge": {
            "present": native.get("present"),
            "telemetry": native.get("telemetry"),
            "tts": native.get("tts"),
            "vad_start": native.get("vadStart"),
            "vad_resume": native.get("vadResume"),
            "vad_stop": native.get("vadStop"),
            "tts_engine_available": native.get("ttsEngineAvailable"),
            "default_tts_locale": native.get("defaultTtsLocale"),
        },
        "history": compact_history(static.get("history")),
        "service_worker": {
            "controlled": (static.get("serviceWorker") or {}).get("controlled"),
            "script_url": (static.get("serviceWorker") or {}).get("scriptURL"),
            "cache_keys": (static.get("serviceWorker") or {}).get("cacheKeys"),
        }
        if isinstance(static.get("serviceWorker"), dict)
        else None,
    }


def final_vad_summary(
    events: list[dict[str, Any]],
    endpoint_status: dict[str, Any],
    endpoint_stop: dict[str, Any],
) -> dict[str, Any]:
    state = endpoint_status.get("state") if isinstance(endpoint_status.get("state"), dict) else {}
    session_id = state.get("sessionId")
    session_events = [event for event in events if event_session(event) == session_id]
    endpoints = [event for event in session_events if event.get("event") == "vad_endpoint"]
    resumed = [event for event in session_events if event.get("event") == "vad_resumed"]
    stopped = [event for event in session_events if event.get("event") == "vad_stopped"]
    endpoint = endpoints[-1] if endpoints else {}
    resume = resumed[-1] if resumed else {}
    stop = stopped[-1] if stopped else {}
    endpoint_details = endpoint.get("details") if isinstance(endpoint.get("details"), dict) else {}
    stop_details = stop.get("details") if isinstance(stop.get("details"), dict) else {}
    native_endpoints = [
        event for event in session_events if event.get("event") == "native_vad_endpoint"
    ]
    native_resumes = [
        event for event in session_events if event.get("event") == "native_vad_resumed"
    ]
    native_endpoint = native_endpoints[-1] if native_endpoints else endpoint
    native_resume = native_resumes[-1] if native_resumes else resume
    endpoint_mono = native_endpoint.get("monotonic_ms")
    resume_mono = native_resume.get("monotonic_ms")

    if endpoint_mono is not None and resume_mono is not None:
        endpoint_to_resume = rounded(float(resume_mono) - float(endpoint_mono))
    else:
        endpoint_to_resume = None

    speech_events = [
        event
        for event in session_events
        if event.get("event") == "vad_speech_confirmed"
        and float(event.get("monotonic_ms", float("inf")))
        <= float(endpoint.get("monotonic_ms", float("-inf")))
    ]
    sequence_start = speech_events[-1] if speech_events else endpoint
    start = float(sequence_start.get("monotonic_ms", float("-inf")))
    end = float(stop.get("monotonic_ms", float("inf")))
    sequence_allowlist = {
        "vad_speech_confirmed",
        "vad_hangover",
        "native_vad_endpoint",
        "vad_endpoint",
        "native_vad_output_gate",
        "vad_output_gate",
        "wav_ready",
        "request_start",
        "asr_request_start",
        "language_resolved",
        "response_complete",
        "lfm_tts_response_complete",
        "lfm_interleaved_response_complete",
        "vad_assistant_output_settled",
        "vad_resume_requested",
        "native_vad_resumed",
        "native_vad_listening",
        "vad_resumed",
        "vad_listening",
        "native_vad_stopped",
        "vad_stopped",
    }
    raw_sequence = [
        event.get("event")
        for event in events
        if start <= float(event.get("monotonic_ms", float("-inf"))) <= end
        and event.get("event") in sequence_allowlist
        and (event_session(event) in {None, session_id})
    ]
    sequence = []
    for name in raw_sequence:
        if not sequence or sequence[-1] != name:
            sequence.append(name)

    response_events = [
        event
        for event in events
        if event.get("event")
        in {
            "response_complete",
            "tts_text_sse_complete",
            "lfm_tts_response_complete",
            "lfm_interleaved_response_complete",
        }
        and start <= float(event.get("monotonic_ms", float("-inf"))) <= end
    ]
    response_details = (
        response_events[-1].get("details")
        if response_events and isinstance(response_events[-1].get("details"), dict)
        else {}
    )

    return {
        "start_accepted": True,
        "endpoint_reason": endpoint_details.get("endpoint_reason"),
        "sample_rate": endpoint_details.get("sample_rate"),
        "pcm_bytes": endpoint_details.get("pcm_bytes"),
        "wav_bytes": endpoint_details.get("wav_bytes"),
        "audio_duration_ms": endpoint_details.get("audio_duration_ms"),
        "payload_present_at_callback": endpoint_details.get("payload_present"),
        "payload_retained_in_public_summary": False,
        "transcript": endpoint_status.get("response"),
        "asr_first_text_ms": rounded(response_details.get("first_text_ms")),
        "request_complete_ms": rounded(response_details.get("request_elapsed_ms")),
        "endpoint_to_native_resume_ms": endpoint_to_resume,
        "resumed_to_listening": bool(resumed),
        "final_stop_accepted": endpoint_stop.get("accepted"),
        "final_state": endpoint_stop.get("state"),
        "frames_processed": stop_details.get("frames_processed"),
        "samples_processed": stop_details.get("samples_processed"),
        "confirmed_utterances": stop_details.get("confirmed_utterances"),
        "endpoints": stop_details.get("endpoints"),
        "trailing_silence_endpoints": stop_details.get("trailing_silence_endpoints"),
        "rejected_short_bursts": stop_details.get("rejected_short_bursts"),
        "sequence": sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-telemetry", type=Path, required=True)
    parser.add_argument("--full-telemetry", type=Path, required=True)
    parser.add_argument("--vad-quiet-rapid", type=Path, required=True)
    parser.add_argument("--vad-endpoint-status", type=Path, required=True)
    parser.add_argument("--vad-endpoint-stop", type=Path, required=True)
    parser.add_argument("--matrix-logcat", type=Path, required=True)
    parser.add_argument("--full-logcat", type=Path, required=True)
    parser.add_argument("--device-facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version-code", type=int, required=True)
    parser.add_argument("--version-name", required=True)
    parser.add_argument("--apk-sha256", required=True)
    args = parser.parse_args()

    matrix = load_json(args.matrix)
    matrix_events = load_jsonl(args.matrix_telemetry)
    all_events = load_jsonl(args.full_telemetry)
    quiet_rapid = load_json(args.vad_quiet_rapid)
    endpoint_status = load_json(args.vad_endpoint_status)
    endpoint_stop = load_json(args.vad_endpoint_stop)
    device = load_json(args.device_facts)
    metrics = turn_metrics_by_turn(matrix_events)
    compact_cases = [
        compact_case(case, metrics) for case in matrix.get("cases", [])
    ]

    compact_matrix = {
        "schema_version": 2,
        "candidate": {
            "application_id": "ai.liquid.lfmdemo",
            "version_code": args.version_code,
            "version_name": args.version_name,
            "apk_sha256": args.apk_sha256,
        },
        "device": {
            "product": device.get("product"),
            "soc": device.get("soc_model"),
            "android_release": device.get("android_release"),
            "abi": device.get("abi"),
            "htp_arch": "v81",
        },
        "run": {
            "started_at": matrix.get("startedAt"),
            "finished_at": matrix.get("finishedAt"),
            "harness_failure": matrix.get("failure"),
            "case_count": len(matrix.get("cases", [])),
            "verdict_counts": dict(
                sorted(collections.Counter(
                    case["verdict"] for case in compact_cases
                ).items())
            ),
            "blocked_file_input_events": (matrix.get("cleanup") or {}).get(
                "blockedFileInputEvents"
            ),
        },
        "static_inspection": compact_static(matrix),
        "response_service_preflight": matrix.get("responseServicePreflight"),
        "cases": compact_cases,
    }

    response_latency_by_warmth: dict[str, list[float]] = collections.defaultdict(list)
    for case in compact_cases:
        warmth = case.get("server_warmth")
        latency = case.get("latency") if isinstance(case.get("latency"), dict) else {}
        first_token = latency.get("response_backend_first_token_ms")
        if warmth and first_token is not None:
            response_latency_by_warmth[str(warmth)].append(float(first_token))
    compact_response_latency = {
        warmth: {
            "runs": len(values),
            "first_token_ms": [rounded(value) for value in values],
            "median_first_token_ms": rounded(statistics.median(values)),
            "maximum_first_token_ms": rounded(max(values)),
        }
        for warmth, values in sorted(response_latency_by_warmth.items())
    }

    audio_turns = [
        {"turn": turn, **value}
        for turn, value in metrics.items()
        if value.get("first_audio_proxy_ms") is not None
    ]
    first_audio_proxies = [
        float(item["first_audio_proxy_ms"]) for item in audio_turns
    ]
    output_heads = [
        float(item["first_output_head_ms"])
        for item in audio_turns
        if item.get("first_output_head_ms") is not None
    ]
    full_counts = event_counts(all_events)
    matrix_counts = event_counts(matrix_events)
    telemetry_summary = {
        "schema_version": 2,
        "matrix": {
            "events": len(matrix_events),
            "event_counts": matrix_counts,
            "turn_errors": matrix_counts.get("turn_error", 0),
            "turn_cancellations": matrix_counts.get("turn_cancelled", 0),
            "audio_turn_count": len(audio_turns),
            "playback_underflows_total": sum(
                int(item.get("playback_underflows") or 0) for item in audio_turns
            ),
            "maximum_first_output_head_ms": rounded(max(output_heads)) if output_heads else None,
            "all_first_output_heads_below_3000_ms": bool(output_heads)
            and max(output_heads) < 3000,
            "maximum_first_audio_proxy_ms": (
                rounded(max(first_audio_proxies)) if first_audio_proxies else None
            ),
            "all_first_audio_proxies_below_3000_ms": bool(first_audio_proxies)
            and max(first_audio_proxies) < 3000,
            "response_quality_warning_turns": sum(
                bool(item.get("response_quality_warning")) for item in metrics.values()
            ),
            "finish_reason_counts": dict(
                sorted(collections.Counter(
                    str(item.get("finish_reason"))
                    for item in metrics.values()
                    if item.get("finish_reason") is not None
                ).items())
            ),
            "response_backend_first_token_by_warmth": compact_response_latency,
            "audio_turns": [
                {"case_index": index, **{key: value for key, value in item.items() if key != "turn"}}
                for index, item in enumerate(audio_turns, 1)
            ],
        },
        "full_run": {
            "events": len(all_events),
            "event_counts": full_counts,
            "turn_errors": full_counts.get("turn_error", 0),
            "turn_cancellations": full_counts.get("turn_cancelled", 0),
            "vad_errors": full_counts.get("vad_error", 0),
            "npu_executed_true_events": sum(
                1
                for event in all_events
                if isinstance(event.get("details"), dict)
                and event["details"].get("npu_executed") is True
            ),
        },
        "vad": {
            "quiet_and_rapid": compact_quiet_rapid(quiet_rapid),
            "final_prerecorded_endpoint": final_vad_summary(
                all_events, endpoint_status, endpoint_stop
            ),
            "voice_response_playback_acoustically_exercised": False,
        },
        "crash_anr_evidence": {
            "matrix_logcat_bytes": args.matrix_logcat.stat().st_size,
            "full_logcat_bytes": args.full_logcat.stat().st_size,
            "independent_logcat_proof_available": args.matrix_logcat.stat().st_size > 0
            or args.full_logcat.stat().st_size > 0,
            "claim": (
                "Non-empty logcat evidence was retained for independent crash/ANR review."
                if args.matrix_logcat.stat().st_size > 0
                or args.full_logcat.stat().st_size > 0
                else "Harness and telemetry completed without error; zero-byte logcat captures do not independently prove crash/ANR absence."
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matrix_results.json").write_text(
        json.dumps(compact_matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "telemetry_summary.json").write_text(
        json.dumps(telemetry_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
