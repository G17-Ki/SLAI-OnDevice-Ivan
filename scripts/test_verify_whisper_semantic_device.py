#!/usr/bin/env python3
"""Unit tests for the isolated Whisper semantic overlay verifier."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import verify_whisper_semantic_device as verifier


class SemanticVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "gate_id": "whisper_small_bundled_en_semantic_candidate_v1",
            "build_flag": "WHISPER_SEMANTIC_DECODE_ENABLED=true",
            "language_code": "en",
            "language_token_id": 50259,
            "prompt_token_ids": [50258, 50259, 50359, 50363],
            "feature_source": "bundled_question_en_native_frontend",
            "max_decoder_calls": 199,
            "maximum_generated_tokens": 196,
            "eot_token_id": 50257,
            "allowed_generated_token_min": 0,
            "allowed_generated_token_max": 50257,
            "suppressed_generated_token_min": 50258,
            "suppressed_generated_token_max": 51864,
            "attention_mask_mode": "cumulative_suffix_uint16",
            "reference_fixture": "en",
            "reference_wav_sha256": "unused-by-direct-test",
            "reference_text": (
                "What is the capital of France? Answer in one short sentence."
            ),
            "require_eot": True,
            "max_wer": 0.25,
            "max_cer": 0.20,
            "max_semantic_compute_ms": 2500.0,
        }
        generated = [100, 101, 50257]
        decoder_calls = 3 + len(generated)
        step_times = [4.0] * decoder_calls
        self.semantic = {
            "enabled": True,
            "status": "candidate_decode_eot",
            "feature_source": "bundled_question_en_native_frontend",
            "language_code": "en",
            "language_token_id": 50259,
            "prompt_token_ids": [50258, 50259, 50359, 50363],
            "decoder_calls": decoder_calls,
            "decoder_calls_expected": decoder_calls,
            "generated_token_count": len(generated),
            "generated_token_ids": generated,
            "generated_token_ids_include_eot": True,
            "eot_reached": True,
            "termination_reason": "end_of_transcript",
            "maximum_generated_tokens": 196,
            "candidate_transcript": (
                "What is the capital of France? Answer in one short sentence."
            ),
            "candidate_transcript_skips_eot": True,
            "input_token_ids": [50258, 50259, 50359, 50363, 100, 101],
            "attention_mask_mode": "cumulative_suffix_uint16",
            "attention_first_unmasked_indices": [199, 198, 197, 196, 195, 194],
            "allowed_generated_token_min": 0,
            "allowed_generated_token_max": 50257,
            "suppressed_generated_token_min": 50258,
            "suppressed_generated_token_max": 51864,
            "encoder_execute_ms": 200.0,
            "decoder_execute_ms": sum(step_times),
            "decoder_loop_ms": 30.0,
            "decoder_step_ms": step_times,
            "semantic_transcript_claimed": False,
            "real_audio_transcription": False,
        }

    def verify(self, mutation=None):
        semantic = copy.deepcopy(self.semantic)
        if mutation is not None:
            mutation(semantic)
        return verifier.verify_semantic_decode(self.contract, semantic)

    def assert_rejected(self, mutation) -> None:
        with self.assertRaises(RuntimeError):
            self.verify(mutation)

    def test_exact_candidate_passes(self) -> None:
        result = self.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["transcript_score"]["wer"], 0.0)

    def test_committed_semantic_contract_passes(self) -> None:
        spec = verifier.frontend_verifier.load_object(verifier.DEFAULT_SEMANTIC_SPEC)
        contract = verifier.semantic_contract(spec)
        self.assertEqual(
            contract["gate_id"],
            "whisper_small_bundled_en_semantic_candidate_v1",
        )

    def test_japanese_contract_and_candidate_pass(self) -> None:
        spec = verifier.frontend_verifier.load_object(verifier.DEFAULT_SEMANTIC_SPEC)
        overlay = spec["semantic_decode_overlay"]
        overlay.update(
            {
                "gate_id": "whisper_small_bundled_ja_semantic_candidate_v1",
                "build_flag": (
                    "WHISPER_SEMANTIC_DECODE_ENABLED=true;WHISPER_SEMANTIC_LANGUAGE=ja"
                ),
                "language_code": "ja",
                "language_token_id": 50266,
                "prompt_token_ids": [50258, 50266, 50359, 50363],
                "feature_source": "bundled_question_ja_native_frontend",
                "reference_fixture": "ja",
                "reference_wav_sha256": (
                    "0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761"
                ),
                "reference_text": "フランスの首都は、どこですか？",
                "max_wer": 1.0,
            }
        )
        contract = verifier.semantic_contract(spec)
        semantic = copy.deepcopy(self.semantic)
        semantic.update(
            {
                "feature_source": "bundled_question_ja_native_frontend",
                "language_code": "ja",
                "language_token_id": 50266,
                "prompt_token_ids": [50258, 50266, 50359, 50363],
                "candidate_transcript": " フランスの首都は、どこですか？",
                "input_token_ids": [50258, 50266, 50359, 50363, 100, 101],
            }
        )
        result = verifier.verify_semantic_decode(contract, semantic)
        self.assertEqual(result["transcript_score"]["cer"], 0.0)

    def test_zh_ko_ru_contracts_and_candidates_pass(self) -> None:
        cases = {
            "zh": {
                "token": 50260,
                "wav": "bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63",
                "text": "请用中文简单介绍一下北京。",
                "max_wer": 1.0,
                "max_cer": 0.20,
            },
            "ko": {
                "token": 50264,
                "wav": "0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de",
                "text": "프랑스의 수도는 어디인가요?",
                "max_wer": 0.34,
                "max_cer": 0.20,
            },
            "ru": {
                "token": 50263,
                "wav": "3545b8a20e3f320174a4e4dee57ff77268f30f79526eef91c86c778e3c180683",
                "text": "Расскажи коротко о Москве.",
                "max_wer": 0.25,
                "max_cer": 0.10,
            },
        }
        for language, case in cases.items():
            with self.subTest(language=language):
                spec = verifier.frontend_verifier.load_object(
                    verifier.DEFAULT_SEMANTIC_SPEC
                )
                spec["frontend_profile"] = "v2"
                spec["device_output_contract"]["frontend"]["fixtures"][language] = {
                    "wav_sha256": case["wav"]
                }
                overlay = spec["semantic_decode_overlay"]
                overlay.update(
                    {
                        "gate_id": (
                            f"whisper_small_bundled_{language}_semantic_candidate_v1"
                        ),
                        "build_flag": (
                            "WHISPER_FRONTEND_PROFILE=v2;"
                            "WHISPER_SEMANTIC_DECODE_ENABLED=true;"
                            f"WHISPER_SEMANTIC_LANGUAGE={language}"
                        ),
                        "language_code": language,
                        "language_token_id": case["token"],
                        "prompt_token_ids": [50258, case["token"], 50359, 50363],
                        "feature_source": (
                            f"bundled_question_{language}_native_frontend"
                        ),
                        "reference_fixture": language,
                        "reference_wav_sha256": case["wav"],
                        "reference_text": case["text"],
                        "max_wer": case["max_wer"],
                        "max_cer": case["max_cer"],
                    }
                )
                contract = verifier.semantic_contract(spec)
                semantic = copy.deepcopy(self.semantic)
                semantic.update(
                    {
                        "feature_source": (
                            f"bundled_question_{language}_native_frontend"
                        ),
                        "language_code": language,
                        "language_token_id": case["token"],
                        "prompt_token_ids": [50258, case["token"], 50359, 50363],
                        "candidate_transcript": case["text"],
                        "input_token_ids": [
                            50258,
                            case["token"],
                            50359,
                            50363,
                            100,
                            101,
                        ],
                    }
                )
                result = verifier.verify_semantic_decode(contract, semantic)
                self.assertEqual(result["transcript_score"]["cer"], 0.0)

    def test_committed_zh_ko_ru_semantic_contracts_pass(self) -> None:
        report_root = verifier.REPORT_ROOT
        for language in ("zh", "ko", "ru"):
            with self.subTest(language=language):
                path = report_root / (
                    "nubia_whisper_android_frontend_multilang_v2_semantic_"
                    f"{language}_probe_spec.json"
                )
                spec = verifier.frontend_verifier.load_object(path)
                contract = verifier.semantic_contract(spec)
                self.assertEqual(contract["language_code"], language)

    def test_punctuation_and_case_are_normalized(self) -> None:
        result = self.verify(
            lambda value: value.__setitem__(
                "candidate_transcript",
                "WHAT IS THE CAPITAL OF FRANCE — ANSWER IN ONE SHORT SENTENCE!",
            )
        )
        self.assertEqual(result["transcript_score"]["wer"], 0.0)

    def test_non_cumulative_attention_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["attention_first_unmasked_indices"].__setitem__(1, 199)
        )

    def test_suppressed_generated_token_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["generated_token_ids"].__setitem__(0, 50258)
        )

    def test_missing_eot_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value["generated_token_ids"].__setitem__(-1, 42)
        )

    def test_decoder_call_invariant_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value.__setitem__("decoder_calls", value["decoder_calls"] + 1)
        )

    def test_autoregressive_input_mismatch_is_rejected(self) -> None:
        self.assert_rejected(lambda value: value["input_token_ids"].__setitem__(-1, 99))

    def test_wer_failure_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value.__setitem__("candidate_transcript", "unrelated words")
        )

    def test_latency_failure_is_rejected(self) -> None:
        self.assert_rejected(lambda value: value.__setitem__("decoder_loop_ms", 2400.0))

    def test_boolean_timing_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value.__setitem__("encoder_execute_ms", True)
        )

    def test_extra_field_is_rejected(self) -> None:
        self.assert_rejected(lambda value: value.__setitem__("untrusted", True))

    def test_replacement_character_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value.__setitem__(
                "candidate_transcript", value["candidate_transcript"] + "\ufffd"
            )
        )

    def test_nul_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value.__setitem__(
                "candidate_transcript", value["candidate_transcript"] + "\x00"
            )
        )

    def test_disallowed_control_is_rejected(self) -> None:
        self.assert_rejected(
            lambda value: value.__setitem__(
                "candidate_transcript", value["candidate_transcript"] + "\x01"
            )
        )

    def test_line_break_controls_remain_allowed(self) -> None:
        result = self.verify(
            lambda value: value.__setitem__(
                "candidate_transcript", "\n" + value["candidate_transcript"] + "\r\n"
            )
        )
        self.assertEqual(result["transcript_score"]["wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
