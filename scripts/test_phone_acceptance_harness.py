#!/usr/bin/env python3
"""Deterministic tests for the v0.4.2 phone-acceptance tooling."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER_PATH = ROOT / "scripts/summarize_phone_acceptance.py"
HARNESS_PATH = ROOT / "scripts/run_phone_webview_acceptance.mjs"
PLAN_PATH = ROOT / "scripts/phone_acceptance_plans/qwen3_1_7b_physical.json"
JA_KO_PLAN_PATH = (
    ROOT / "scripts/phone_acceptance_plans/qwen2_5_ja_ko_current.json"
)

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("phone_acceptance_summary", SUMMARIZER_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class VerdictTest(unittest.TestCase):
    def test_verdicts_come_from_case_expectations(self) -> None:
        self.assertEqual(
            SUMMARY.derive_case_verdict(
                {
                    "expectations": {"outcome": "complete"},
                    "acceptanceAssertions": {"allPassed": True},
                }
            ),
            "pass",
        )
        self.assertEqual(
            SUMMARY.derive_case_verdict(
                {
                    "expectations": {"outcome": "blocked"},
                    "acceptanceAssertions": {"allPassed": True},
                }
            ),
            "pass_expected_block",
        )
        self.assertEqual(
            SUMMARY.derive_case_verdict(
                {"acceptanceAssertions": {"allPassed": False}}
            ),
            "fail",
        )
        self.assertEqual(
            SUMMARY.derive_case_verdict({"thrown": {"message": "unexpected"}}),
            "error_unclassified",
        )


class TelemetryTest(unittest.TestCase):
    def test_multilingual_and_long_output_metadata_are_joined_by_turn(self) -> None:
        events = [
            {
                "event": "asr_request_complete",
                "turn_id": "turn-1",
                "details": {
                    "selected_language": "auto",
                    "resolved_language": "zh",
                    "backend": "sensevoice_qnn",
                    "provider": "qnn",
                    "asr_companion": True,
                    "npu_executed": True,
                    "npu_selected": True,
                    "candidate_summaries": [{"backend": "sensevoice_qnn"}],
                },
            },
            {
                "event": "response_complete",
                "turn_id": "turn-1",
                "details": {
                    "first_text_ms": 510.25,
                    "first_output_head_ms": 880.5,
                    "request_elapsed_ms": 4200,
                    "pcm_duration_seconds": 11.75,
                    "finish_reason": "stop",
                    "sse_events": 42,
                    "response_token_limit": 320,
                    "repetition": {
                        "duplicate_gram_ratio": 0.0,
                        "max_gram_count": 1,
                    },
                    "npu_executed": True,
                },
            },
            {
                "event": "response_quality_warning",
                "turn_id": "turn-1",
                "details": {
                    "reason": "repeated_four_grams",
                    "duplicate_gram_ratio": 0.3,
                    "max_gram_count": 3,
                },
            },
            {
                "event": "tts_text_sse_complete",
                "turn_id": "turn-2",
                "details": {
                    "first_native_tts_start_proxy_ms": 1200,
                    "request_elapsed_ms": 2500,
                    "finish_reason": "stop",
                    "voice_backend": "android_native_tts",
                    "response_backend": "qwen3_1_7b_q4_cpu",
                    "response_model": "../model/Qwen3-1.7B-Q4_0.gguf",
                },
            },
            {
                "event": "native_tts_utterance_start",
                "turn_id": "turn-2",
                "monotonic_ms": 3000,
                "details": {},
            },
            {
                "event": "native_tts_playback_terminal",
                "turn_id": "turn-2",
                "monotonic_ms": 14500,
                "details": {"status": "completed"},
            },
        ]

        metrics = SUMMARY.turn_metrics_by_turn(events)
        self.assertEqual(metrics["turn-1"]["asr"]["backend"], "sensevoice_qnn")
        self.assertTrue(metrics["turn-1"]["asr"]["npu_executed"])
        self.assertEqual(metrics["turn-1"]["first_audio_proxy_ms"], 880.5)
        self.assertEqual(metrics["turn-1"]["finish_reason"], "stop")
        self.assertEqual(
            metrics["turn-1"]["response_quality_warning"]["max_gram_count"], 3
        )
        self.assertEqual(metrics["turn-2"]["first_audio_proxy_ms"], 1200)
        self.assertEqual(
            metrics["turn-2"]["native_tts_playback_terminal"]["status"],
            "completed",
        )
        self.assertEqual(metrics["turn-2"]["native_tts_playback_span_ms"], 11500)
        self.assertEqual(metrics["turn-2"]["response_backend"], "qwen3_1_7b_q4_cpu")

    def test_telemetry_expectations_affect_public_verdict(self) -> None:
        case = {
            "name": "long_native_tts",
            "requestedMode": "chat",
            "selectedLanguage": "en",
            "turnId": "turn-long",
            "expectations": {
                "outcome": "complete",
                "responseBackend": "qwen3_1_7b_q4_cpu",
                "responseModel": "Qwen3-1.7B-Q4_0",
                "finishReason": "stop",
                "maxFirstAudioMs": 3000,
                "minNativeTtsPlaybackMs": 10000,
                "ttsTerminalStatus": "completed",
            },
            "acceptanceAssertions": {"allPassed": True},
            "fetches": [],
        }
        metrics = {
            "turn-long": {
                "response_backend": "qwen3_1_7b_q4_cpu",
                "response_model": "../model/Qwen3-1.7B-Q4_0.gguf",
                "finish_reason": "stop",
                "first_audio_proxy_ms": 1700,
                "native_tts_playback_span_ms": 12000,
                "native_tts_playback_terminal": {"status": "completed"},
            }
        }

        compact = SUMMARY.compact_case(case, metrics)
        self.assertEqual(compact["verdict"], "pass")
        self.assertTrue(compact["all_assertions_passed"])
        self.assertTrue(all(
            value for value in compact["telemetry_assertions"].values()
            if isinstance(value, bool)
        ))

        metrics["turn-long"]["first_audio_proxy_ms"] = 3100
        failed = SUMMARY.compact_case(case, metrics)
        self.assertEqual(failed["verdict"], "fail")
        self.assertFalse(failed["telemetry_assertions"]["max_first_audio_ms"])

    def test_public_case_summary_keeps_metadata_but_drops_request_text(self) -> None:
        case = {
            "name": "context_on_recall",
            "requestedMode": "chat",
            "selectedLanguage": "en",
            "contextEnabledRequested": True,
            "historyEnabled": True,
            "historySwitchAriaChecked": "true",
            "languageRoute": {"language": "en"},
            "asrFacts": {
                "backend": "sensevoice_qnn",
                "provider": "qnn",
                "npu_executed": True,
            },
            "responseRequestText": "The secret phrase is cobalt tulip.",
            "expectations": {"outcome": "complete"},
            "acceptanceAssertions": {"allPassed": True},
            "fetches": [
                {
                    "stage": "asr_companion",
                    "requestedLanguage": "en",
                    "method": "POST",
                    "contentType": "audio/wav",
                    "bodyBytes": 64044,
                    "httpStatus": 200,
                    "httpOk": True,
                    "url": "http://127.0.0.1:6010/asr?language=en",
                }
            ],
        }

        compact = SUMMARY.compact_case(case, {})
        rendered = json.dumps(compact)
        self.assertEqual(compact["verdict"], "pass")
        self.assertEqual(compact["asr"]["provider"], "qnn")
        self.assertEqual(compact["companion_requests"][0]["wav_bytes"], 64044)
        self.assertEqual(compact["response_request_characters"], 34)
        self.assertNotIn("cobalt tulip", rendered)
        self.assertNotIn("127.0.0.1", rendered)


class JavaScriptHarnessSourceTest(unittest.TestCase):
    def test_context_and_companion_capture_are_present(self) -> None:
        source = HARNESS_PATH.read_text(encoding="utf-8")
        for marker in (
            'stage: "asr_companion"',
            'stage: isResponseCompanion',
            '"response_companion"',
            "function setContextExpression(enabled)",
            "const accepted = setHistoryEnabled(requested);",
            "expectedResponseRequestIncludes",
            "expectedCompanionLanguages",
            "expectedTranscriptScript",
            "expectedResponseScript",
            "expectedMaxDuplicateGramRatio",
            "expectedResponseBackend",
            "expectedResponseModel",
            "expectedVoiceBackend",
            "expectedFinishReason",
            "expectedNoReasoning",
            "maxFirstAudioMs",
            "minNativeTtsPlaybackMs",
            "minPcmDurationMs",
            "expectedTtsTerminalStatus",
            "ttsPolicy",
            "expectedTtsGenerationLanguage",
            "expectedAsrLanguageHint",
            "expectedNpuSelected",
            "maxTtsFirstChunkMs",
            "maxTtsRtf",
            "maxTtsDroppedChunks",
            "collectNativeTtsEvidence",
            "nativeTtsAcceptanceAssertions",
            "window.__lfmNativeTtsCapture",
            "responseThinkingDisabled(responseCompanionFetches)",
            "responseServicePreflight",
            "asrRequestPlanValid",
            "acceptanceAssertions.allPassed",
            "schemaVersion: 2",
        ):
            self.assertIn(marker, source)
        self.assertIn("expected.maxTtsDroppedChunks !== null", source)
        self.assertIn('["chat", "tts"].includes(testCase.mode)', source)
        self.assertNotIn("expected.maxDroppedChunks !== null", source)
        self.assertIn('typeof asrFacts?.npu_executed === "boolean"', source)
        self.assertIn('typeof asrFacts?.npu_selected === "boolean"', source)
        self.assertNotIn("Boolean(asrFacts?.npu_executed)", source)
        self.assertNotIn("Boolean(asrFacts?.npu_selected)", source)
        self.assertIn("payload.summary = summarizePayload(payload)", source)
        self.assertIn("if (!ok) process.exitCode = 1", source)

    def test_physical_plan_is_qwen_17b_and_covers_required_modes(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(plan["expectedResponseModel"], "Qwen3-1.7B-Q4_0")
        self.assertEqual(plan["expectedResponseBackend"], "qwen3_1_7b_q4_cpu")
        self.assertTrue(plan["expectedNoReasoning"])
        cases = {case["name"]: case for case in plan["cases"]}
        for name in (
            "cold_explicit_en_capital",
            "warm_explicit_en_capital",
            "context_off_recall_isolated",
            "context_on_recall",
            "context_disable_and_clear",
            "auto_en",
            "explicit_zh",
            "auto_zh",
            "explicit_ru",
            "auto_ru",
            "explicit_fr",
            "explicit_it",
            "explicit_es",
            "long_en_natural_voice",
        ):
            self.assertIn(name, cases)
        self.assertGreaterEqual(cases["long_en_natural_voice"]["minPcmDurationMs"], 10000)
        self.assertEqual(
            cases["long_en_natural_voice"]["expectedVoiceBackend"],
            "lfm_audio_natural_tts_streaming",
        )
        for case in plan["cases"]:
            self.assertTrue((ROOT / case["audio"]).is_file(), case["audio"])

    def test_ja_ko_plan_requires_question_meaning_and_positive_answer(self) -> None:
        plan = json.loads(JA_KO_PLAN_PATH.read_text(encoding="utf-8"))
        cases = {case["name"]: case for case in plan["cases"]}
        japanese = cases["explicit_ja_full_turn"]
        korean = cases["explicit_ko_full_turn"]
        self.assertIn("どこ", japanese["expectedTranscriptIncludes"])
        self.assertEqual(japanese["expectedResponseIncludes"], "パリです")
        self.assertIn("パリではありません", japanese["expectedResponseExcludes"])
        self.assertIn("어디", korean["expectedTranscriptIncludes"])
        self.assertEqual(korean["expectedResponseIncludes"], "파리입니다")
        self.assertIn("파리가 아닙니다", korean["expectedResponseExcludes"])


if __name__ == "__main__":
    unittest.main()
