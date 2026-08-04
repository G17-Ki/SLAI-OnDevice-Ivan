#!/usr/bin/env python3
"""Verify the bounded English semantic overlay on the isolated Whisper probe.

The existing frontend verifier remains authoritative for whole-APK identity,
the app-private payload, QNN/HTP execution, FastRPC/DMA evidence, frontend
numerics, and protected-package hashes. This overlay adds only the semantic
candidate contract and deliberately does not promote the probe to production.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/streaming_response/whisper_v81"
FRONTEND_VERIFIER_PATH = ROOT / "scripts/verify_whisper_android_frontend_device.py"
DEFAULT_SEMANTIC_SPEC = REPORT_ROOT / (
    "nubia_whisper_android_frontend_semantic_en_probe_spec.json"
)
DEFAULT_BASE_SPEC = REPORT_ROOT / "nubia_whisper_ort_qnn_probe_spec.json"
LANGUAGE_TOKEN_IDS = {
    "en": 50259,
    "zh": 50260,
    "ru": 50263,
    "ko": 50264,
    "ja": 50266,
}


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import verification helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


frontend_verifier = load_module(
    "verify_whisper_android_frontend_device", FRONTEND_VERIFIER_PATH
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not an integer",
    )
    if minimum is not None:
        require(value >= minimum, f"{label} is below {minimum}")
    return value


def finite_positive(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    converted = float(value)
    require(math.isfinite(converted) and converted > 0.0, f"{label} is not positive")
    return converted


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return cleaned.split()


def normalized_chars(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        character
        for character in normalized
        if not character.isspace()
        and unicodedata.category(character)[0] not in {"P", "S"}
    ]


def require_valid_transcript_text(text: str) -> None:
    require("\ufffd" not in text, "candidate transcript contains U+FFFD")
    for character in text:
        code_point = ord(character)
        category = unicodedata.category(character)
        require(code_point != 0, "candidate transcript contains NUL")
        require(category != "Cs", "candidate transcript contains a surrogate")
        require(
            category != "Cc" or character in {"\t", "\n", "\r"},
            f"candidate transcript contains control U+{code_point:04X}",
        )


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def transcript_score(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    reference_chars = normalized_chars(reference)
    hypothesis_chars = normalized_chars(hypothesis)
    require(reference_words and reference_chars, "semantic reference is empty")
    word_edits = edit_distance(reference_words, hypothesis_words)
    character_edits = edit_distance(reference_chars, hypothesis_chars)
    return {
        "reference_text": reference,
        "candidate_transcript": hypothesis,
        "normalized_reference_words": " ".join(reference_words),
        "normalized_candidate_words": " ".join(hypothesis_words),
        "reference_word_count": len(reference_words),
        "candidate_word_count": len(hypothesis_words),
        "word_edits": word_edits,
        "wer": word_edits / len(reference_words),
        "normalized_reference_chars": "".join(reference_chars),
        "normalized_candidate_chars": "".join(hypothesis_chars),
        "reference_char_count": len(reference_chars),
        "candidate_char_count": len(hypothesis_chars),
        "character_edits": character_edits,
        "cer": character_edits / len(reference_chars),
        "normalized_exact_match": reference_words == hypothesis_words,
    }


def semantic_contract(frontend_spec: dict[str, Any]) -> dict[str, Any]:
    contract = frontend_spec.get("semantic_decode_overlay")
    require(isinstance(contract, dict), "semantic decode overlay is missing")
    required = {
        "gate_id",
        "build_flag",
        "language_code",
        "language_token_id",
        "prompt_token_ids",
        "feature_source",
        "max_decoder_calls",
        "maximum_generated_tokens",
        "eot_token_id",
        "allowed_generated_token_min",
        "allowed_generated_token_max",
        "suppressed_generated_token_min",
        "suppressed_generated_token_max",
        "attention_mask_mode",
        "reference_fixture",
        "reference_wav_sha256",
        "reference_text",
        "require_eot",
        "max_wer",
        "max_cer",
        "max_semantic_compute_ms",
    }
    require(set(contract) == required, "semantic decode contract keys differ")
    language_code = contract["language_code"]
    require(language_code in LANGUAGE_TOKEN_IDS, "unsupported semantic language")
    require(
        contract["gate_id"]
        == f"whisper_small_bundled_{language_code}_semantic_candidate_v1",
        "semantic gate ID mismatch",
    )
    frontend_profile = frontend_spec.get("frontend_profile", "v1")
    require(frontend_profile in {"v1", "v2"}, "unsupported frontend profile")
    profile_languages = {
        "v1": {"en", "ja"},
        "v2": set(LANGUAGE_TOKEN_IDS),
    }
    require(
        language_code in profile_languages[frontend_profile],
        "semantic language is unavailable in the frontend profile",
    )
    expected_build_flag = ""
    if frontend_profile == "v2":
        expected_build_flag = "WHISPER_FRONTEND_PROFILE=v2;"
    expected_build_flag += "WHISPER_SEMANTIC_DECODE_ENABLED=true"
    if language_code != "en":
        expected_build_flag += f";WHISPER_SEMANTIC_LANGUAGE={language_code}"
    require(
        contract["build_flag"] == expected_build_flag,
        "semantic build flag mismatch",
    )
    language_token_id = LANGUAGE_TOKEN_IDS[language_code]
    require(
        contract["language_token_id"] == language_token_id,
        "semantic language token mismatch",
    )
    require(
        contract["prompt_token_ids"] == [50258, language_token_id, 50359, 50363],
        "semantic prompt mismatch",
    )
    require(
        contract["feature_source"]
        == f"bundled_question_{language_code}_native_frontend",
        "feature source mismatch",
    )
    require(contract["max_decoder_calls"] == 199, "decoder-call bound mismatch")
    require(contract["maximum_generated_tokens"] == 196, "token bound mismatch")
    require(contract["eot_token_id"] == 50257, "EOT token mismatch")
    require(
        contract["allowed_generated_token_min"] == 0
        and contract["allowed_generated_token_max"] == 50257,
        "allowed token bounds mismatch",
    )
    require(
        contract["suppressed_generated_token_min"] == 50258
        and contract["suppressed_generated_token_max"] == 51864,
        "suppressed token bounds mismatch",
    )
    require(
        contract["attention_mask_mode"] == "cumulative_suffix_uint16",
        "attention mask contract mismatch",
    )
    require(
        contract["reference_fixture"] == language_code,
        "reference fixture mismatch",
    )
    fixtures = frontend_spec["device_output_contract"]["frontend"]["fixtures"]
    require(language_code in fixtures, "semantic fixture is absent from frontend spec")
    expected_wav = fixtures[language_code]
    require(
        contract["reference_wav_sha256"] == expected_wav["wav_sha256"],
        "semantic/reference WAV mismatch",
    )
    require(contract["require_eot"] is True, "semantic gate must require EOT")
    for name in ("max_wer", "max_cer"):
        value = contract[name]
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= float(value) <= 1.0,
            f"invalid semantic {name}",
        )
    finite_positive(contract["max_semantic_compute_ms"], "max_semantic_compute_ms")
    return contract


def verify_semantic_decode(
    contract: dict[str, Any], semantic: dict[str, Any]
) -> dict[str, Any]:
    required_fields = {
        "enabled",
        "status",
        "feature_source",
        "language_code",
        "language_token_id",
        "prompt_token_ids",
        "decoder_calls",
        "decoder_calls_expected",
        "generated_token_count",
        "generated_token_ids",
        "generated_token_ids_include_eot",
        "eot_reached",
        "termination_reason",
        "maximum_generated_tokens",
        "candidate_transcript",
        "candidate_transcript_skips_eot",
        "input_token_ids",
        "attention_mask_mode",
        "attention_first_unmasked_indices",
        "allowed_generated_token_min",
        "allowed_generated_token_max",
        "suppressed_generated_token_min",
        "suppressed_generated_token_max",
        "encoder_execute_ms",
        "decoder_execute_ms",
        "decoder_loop_ms",
        "decoder_step_ms",
        "semantic_transcript_claimed",
        "real_audio_transcription",
    }
    require(isinstance(semantic, dict), "semantic decode report is not an object")
    require(set(semantic) == required_fields, "semantic decode report keys differ")
    require(semantic["enabled"] is True, "semantic decode was not enabled")
    require(
        semantic["status"] == "candidate_decode_eot",
        "semantic decode did not terminate at EOT",
    )
    require(
        semantic["feature_source"] == contract["feature_source"],
        "semantic feature source mismatch",
    )
    require(
        semantic["language_code"] == contract["language_code"],
        "semantic language mismatch",
    )
    require(
        semantic["language_token_id"] == contract["language_token_id"],
        "semantic language token mismatch",
    )
    require(
        semantic["prompt_token_ids"] == contract["prompt_token_ids"],
        "semantic prompt mismatch",
    )
    require(
        semantic["attention_mask_mode"] == contract["attention_mask_mode"],
        "semantic attention mode mismatch",
    )
    require(
        semantic["maximum_generated_tokens"] == contract["maximum_generated_tokens"],
        "semantic token budget mismatch",
    )
    require(
        semantic["allowed_generated_token_min"]
        == contract["allowed_generated_token_min"],
        "allowed token minimum mismatch",
    )
    require(
        semantic["allowed_generated_token_max"]
        == contract["allowed_generated_token_max"],
        "allowed token maximum mismatch",
    )
    require(
        semantic["suppressed_generated_token_min"]
        == contract["suppressed_generated_token_min"],
        "suppressed token minimum mismatch",
    )
    require(
        semantic["suppressed_generated_token_max"]
        == contract["suppressed_generated_token_max"],
        "suppressed token maximum mismatch",
    )
    require(
        semantic["semantic_transcript_claimed"] is False, "probe made a semantic claim"
    )
    require(
        semantic["real_audio_transcription"] is False,
        "probe crossed the real-audio claim boundary",
    )

    decoder_calls = strict_int(semantic["decoder_calls"], "decoder_calls", minimum=1)
    require(
        decoder_calls <= contract["max_decoder_calls"], "decoder call bound exceeded"
    )
    generated_count = strict_int(
        semantic["generated_token_count"], "generated_token_count", minimum=1
    )
    generated = semantic["generated_token_ids"]
    require(isinstance(generated, list), "generated token IDs are not an array")
    require(len(generated) == generated_count, "generated token count mismatch")
    require(
        generated_count <= contract["maximum_generated_tokens"],
        "generated token budget exceeded",
    )
    for index, token in enumerate(generated):
        strict_int(token, f"generated_token_ids[{index}]", minimum=0)
        require(
            contract["allowed_generated_token_min"]
            <= token
            <= contract["allowed_generated_token_max"],
            f"generated token outside allowed domain: {token}",
        )
    require(semantic["eot_reached"] is True, "semantic EOT was not reached")
    require(
        semantic["termination_reason"] == "end_of_transcript",
        "termination reason mismatch",
    )
    require(
        generated[-1] == contract["eot_token_id"], "final generated token is not EOT"
    )
    require(
        contract["eot_token_id"] not in generated[:-1],
        "EOT appeared before the final generated token",
    )

    prompt = contract["prompt_token_ids"]
    expected_calls = len(prompt) - 1 + generated_count
    require(decoder_calls == expected_calls, "decoder call/token invariant mismatch")
    require(
        semantic["decoder_calls_expected"] == expected_calls,
        "reported expected decoder calls mismatch",
    )
    require(
        semantic["generated_token_ids_include_eot"] is True,
        "generated-token EOT inclusion contract mismatch",
    )
    require(
        semantic["candidate_transcript_skips_eot"] is True,
        "candidate transcript EOT filtering contract mismatch",
    )
    input_tokens = semantic["input_token_ids"]
    require(isinstance(input_tokens, list), "input token IDs are not an array")
    require(len(input_tokens) == decoder_calls, "input token count mismatch")
    for index, token in enumerate(input_tokens):
        strict_int(token, f"input_token_ids[{index}]", minimum=0)
    require(input_tokens[: len(prompt)] == prompt, "forced prompt inputs mismatch")
    require(
        input_tokens[len(prompt) :] == generated[:-1],
        "autoregressive generated-token inputs mismatch",
    )
    first_unmasked = semantic["attention_first_unmasked_indices"]
    require(isinstance(first_unmasked, list), "attention progression is not an array")
    require(
        first_unmasked == [199 - step for step in range(decoder_calls)],
        "attention mask is not the cumulative right-aligned suffix progression",
    )

    encoder_ms = finite_positive(semantic["encoder_execute_ms"], "encoder_execute_ms")
    decoder_execute_ms = finite_positive(
        semantic["decoder_execute_ms"], "decoder_execute_ms"
    )
    decoder_loop_ms = finite_positive(semantic["decoder_loop_ms"], "decoder_loop_ms")
    step_times = semantic["decoder_step_ms"]
    require(isinstance(step_times, list), "decoder step timings are not an array")
    require(len(step_times) == decoder_calls, "decoder step timing count mismatch")
    checked_step_times = [
        finite_positive(value, f"decoder_step_ms[{index}]")
        for index, value in enumerate(step_times)
    ]
    require(
        abs(decoder_execute_ms - round(sum(checked_step_times), 3)) <= 0.002,
        "decoder step/aggregate timing mismatch",
    )
    require(
        decoder_loop_ms + 0.002 >= decoder_execute_ms,
        "decoder loop timing is too small",
    )
    semantic_compute_ms = encoder_ms + decoder_loop_ms
    require(
        semantic_compute_ms <= float(contract["max_semantic_compute_ms"]),
        "semantic compute latency exceeded the bounded gate",
    )

    transcript = semantic["candidate_transcript"]
    require(isinstance(transcript, str), "candidate transcript is not text")
    require(transcript.strip(), "candidate transcript is empty")
    require_valid_transcript_text(transcript)
    score = transcript_score(contract["reference_text"], transcript)
    require(score["wer"] <= float(contract["max_wer"]), "semantic WER gate failed")
    require(score["cer"] <= float(contract["max_cer"]), "semantic CER gate failed")
    return {
        "status": "pass",
        "decoder_calls": decoder_calls,
        "generated_token_count": generated_count,
        "eot_reached": True,
        "semantic_compute_ms": round(semantic_compute_ms, 3),
        "encoder_execute_ms": encoder_ms,
        "decoder_execute_ms": decoder_execute_ms,
        "decoder_loop_ms": decoder_loop_ms,
        "transcript_score": score,
    }


def verify_files(
    semantic_spec_path: Path,
    base_spec_path: Path,
    local_verify_path: Path,
    device_report_path: Path,
    evidence_path: Path,
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    base_result = frontend_verifier.verify_files(
        semantic_spec_path,
        base_spec_path,
        local_verify_path,
        device_report_path,
        evidence_path,
        expected_run_id=expected_run_id,
    )
    frontend_spec = frontend_verifier.load_object(semantic_spec_path)
    device_report = frontend_verifier.load_object(device_report_path)
    contract = semantic_contract(frontend_spec)
    semantic_result = verify_semantic_decode(
        contract, device_report.get("semantic_decode")
    )
    return {
        "schema_version": 1,
        "gate_id": contract["gate_id"],
        "frontend_gate_id": frontend_spec["gate_id"],
        "package": base_result["package"],
        "run_id": expected_run_id,
        "status": "pass",
        "installed_apk_sha256": base_result["installed_apk_sha256"],
        "cpu_ep_fallback_disabled": base_result["cpu_ep_fallback_disabled"],
        "fastrpc_cdsp_fd_open": base_result["fastrpc_cdsp_fd_open"],
        "accepted_packages_unchanged": base_result["accepted_packages_unchanged"],
        "semantic_transcript_claimed_by_app": False,
        "production_runtime_integration": False,
        "semantic_decode": semantic_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-spec", type=Path, default=DEFAULT_SEMANTIC_SPEC)
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--local-verify", type=Path, required=True)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = verify_files(
        arguments.semantic_spec.resolve(),
        arguments.base_spec.resolve(),
        arguments.local_verify.resolve(),
        arguments.device_report.resolve(),
        arguments.evidence.resolve(),
        expected_run_id=arguments.run_id,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
