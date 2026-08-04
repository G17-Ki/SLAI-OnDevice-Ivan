#!/usr/bin/env python3
"""Build Lane G G3 short-input safety reports from frozen MOSS artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path


OUT_D = Path(__file__).resolve().parent
SYNTH_PATH = OUT_D / "artifacts" / "moss" / "results" / "g3_synthesis.json"
D3_PATH = OUT_D / "D3.json"
RUNNER_PATH = OUT_D / "run_moss_g.py"
TTS_WEIGHT_PATH = OUT_D / "artifacts" / "moss" / "tts" / "pytorch_model.bin"
CODEC_WEIGHT_PATH = (
    OUT_D
    / "artifacts"
    / "moss"
    / "audio_tokenizer"
    / "model-00001-of-00001.safetensors"
)
RESULT_JSON_PATH = OUT_D / "G3.json"
RESULT_MD_PATH = OUT_D / "G3.md"

SYNTH_SHA256 = "f8f7a2fb48f38204db4f11d9e9cdbcbc24586f28056ce998db789dde0026ac15"
D3_SHA256 = "c68e1d37be7bacb362de0ca1563d5831ad80f86f283db2a03f93f672a020f5f0"
TTS_WEIGHT_SHA256 = "24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5"
CODEC_WEIGHT_SHA256 = "34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67"
STORAGE_CAP_BYTES = 3_000_000_000

LANGUAGES = ("en", "es", "fr", "ru", "zh")
TEXTS = {
    "en": ("A", "OK", "Yes", "Hello", "Hello there", "Hello there friend"),
    "es": ("A", "Sí", "Sol", "Hola", "Hola mundo", "Hola mundo ahora"),
    "fr": ("À", "Ça", "Oui", "Bonjour", "Bonjour Marie", "Bonjour mon ami"),
    "ru": ("Я", "Да", "Нет", "Привет", "Привет мир", "Привет мой друг"),
    "zh": ("好", "好的", "没问题", "你", "你好", "你好呀"),
}

EXPECTED_CAP_RECORD_IDS = (
    "en_01_r1",
    "en_04_r1",
    "en_05_r1",
    "es_01_r1",
    "es_04_r1",
    "fr_01_r1",
    "fr_02_r1",
    "fr_05_r1",
    "zh_01_r1",
    "zh_01_r2",
    "zh_01_r3",
    "zh_02_r1",
    "zh_02_r2",
    "zh_03_r1",
    "zh_03_r2",
    "zh_03_r3",
    "zh_04_r2",
    "zh_04_r3",
    "zh_05_r1",
    "zh_05_r2",
    "zh_05_r3",
    "zh_06_r1",
    "zh_06_r3",
)

SERIALIZATION = {
    "source": "raw 48 kHz stereo decoder tensor",
    "operation": "arithmetic mean across two channels",
    "retained_format": "mono PCM16 WAV",
}

PLAUSIBLE_DURATION_UNRESOLVED_REASON = (
    "Lane G requires plausible duration on every short input, but no text-relative "
    "duration rule was predeclared. Passing the broad registered [0.15, 30] s "
    "waveform-duration gate and avoiding the frame cap therefore cannot be promoted "
    "to a plausible-duration PASS."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def relative(path: Path) -> str:
    return str(path.relative_to(OUT_D))


def record_id(row: dict) -> str:
    return f"{row['case_id']}_r{row['repeat']}"


def stable_seed(case_id: str, repeat: int) -> int:
    payload = f"lane-g-moss-v1:g3:{case_id}:{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def expected_inventory() -> list[tuple[str, str, int, str]]:
    expected = []
    for language in LANGUAGES:
        for index, text in enumerate(TEXTS[language], 1):
            repeats = (1, 2, 3) if language == "zh" else (1,)
            for repeat in repeats:
                expected.append((language, f"{language}_{index:02d}", repeat, text))
    return expected


def directory_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def validate_record(row: dict, sampling: dict) -> None:
    rid = record_id(row)
    if row.get("schema_version") != 1 or row.get("suite") != "g3":
        raise SystemExit(f"invalid schema/suite: {rid}")
    if row.get("seed") != stable_seed(row["case_id"], row["repeat"]):
        raise SystemExit(f"deterministic seed mismatch: {rid}")
    if row.get("sampling") != sampling:
        raise SystemExit(f"sampling settings changed within G3: {rid}")
    if row.get("prepared_text") != row.get("raw_text"):
        raise SystemExit(f"text was changed before synthesis: {rid}")
    if row.get("external_text_normalization") != "DISABLED_EXACT_FROZEN_INPUT":
        raise SystemExit(f"unexpected text-normalization mode: {rid}")
    if row.get("max_new_frames") != 120:
        raise SystemExit(f"unexpected generation cap: {rid}")
    if row.get("max_new_frames_hit") != (
        row.get("generated_frames", -1) >= row.get("max_new_frames", 0)
    ):
        raise SystemExit(f"cap-hit flag disagrees with frame count: {rid}")

    raw = row.get("raw_waveform_gate", {})
    output = row.get("output", {})
    if raw.get("channels") != 2 or raw.get("sample_rate_hz") != 48_000:
        raise SystemExit(f"raw decoder tensor is not recorded as 48 kHz stereo: {rid}")
    if not raw.get("passed") or not all(raw.get("assertions", {}).values()):
        raise SystemExit(f"registered raw waveform gate failed: {rid}")
    if output.get("channels") != 1 or output.get("sample_rate_hz") != 48_000:
        raise SystemExit(f"retained WAV is not 48 kHz mono: {rid}")
    if output.get("format") != "WAV" or output.get("subtype") != "PCM_16":
        raise SystemExit(f"retained output format changed: {rid}")
    if not output.get("passed") or not all(output.get("assertions", {}).values()):
        raise SystemExit(f"registered serialized waveform gate failed: {rid}")
    if row.get("serialization") != SERIALIZATION:
        raise SystemExit(f"serialization path changed: {rid}")
    if not row.get("passed"):
        raise SystemExit(f"synthesis artifact did not retain a registered-gate pass: {rid}")
    if raw.get("frames") != output.get("frames"):
        raise SystemExit(f"raw/output frame counts differ: {rid}")
    if raw.get("sample_count") != 2 * output.get("sample_count", 0):
        raise SystemExit(f"raw/output channel sample counts disagree: {rid}")
    if raw.get("duration_s") != output.get("duration_s"):
        raise SystemExit(f"raw/output durations differ: {rid}")

    wav_path = OUT_D / output.get("path", "")
    if not wav_path.is_file():
        raise SystemExit(f"retained WAV missing: {wav_path}")
    if wav_path.stat().st_size != output.get("bytes"):
        raise SystemExit(f"retained WAV size mismatch: {rid}")
    if sha256_file(wav_path) != output.get("sha256"):
        raise SystemExit(f"retained WAV hash mismatch: {rid}")


def validate_inputs(synth: dict, d3: dict) -> None:
    for path, expected_hash, label in (
        (SYNTH_PATH, SYNTH_SHA256, "G3 synthesis artifact"),
        (D3_PATH, D3_SHA256, "registered D3 result"),
        (TTS_WEIGHT_PATH, TTS_WEIGHT_SHA256, "TTS weights"),
        (CODEC_WEIGHT_PATH, CODEC_WEIGHT_SHA256, "codec weights"),
    ):
        if not path.is_file():
            raise SystemExit(f"required file missing: {path}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise SystemExit(f"{label} hash mismatch: expected {expected_hash}, got {actual}")

    if synth.get("suite") != "g3" or synth.get("evidence_class") != "HOST":
        raise SystemExit("unexpected G3 artifact identity")
    records = synth.get("records", [])
    observed_inventory = [
        (row.get("language"), row.get("case_id"), row.get("repeat"), row.get("raw_text"))
        for row in records
    ]
    if observed_inventory != expected_inventory():
        raise SystemExit("G3 input inventory/order changed")
    if (synth.get("attempted"), synth.get("passed"), synth.get("failed")) != (42, 42, 0):
        raise SystemExit("top-level registered-gate counts are not 42/42/0")
    if synth.get("max_new_frames_hit_count") != 23:
        raise SystemExit("top-level cap-hit count is not 23")
    sampling = synth.get("runtime", {}).get("sampling")
    if not isinstance(sampling, dict) or sampling.get("do_sample") is not True:
        raise SystemExit("G3 was not recorded as sampled generation")
    for row in records:
        validate_record(row, sampling)
    cap_ids = tuple(record_id(row) for row in records if row["max_new_frames_hit"])
    if cap_ids != EXPECTED_CAP_RECORD_IDS:
        raise SystemExit(f"exact cap-hit inventory changed: {cap_ids}")

    model = synth.get("model", {})
    if model.get("tts_weight_sha256") != TTS_WEIGHT_SHA256:
        raise SystemExit("artifact TTS-weight hash mismatch")
    if model.get("codec_weight_sha256") != CODEC_WEIGHT_SHA256:
        raise SystemExit("artifact codec-weight hash mismatch")
    if d3.get("gate") != {
        "finite": "all PCM samples finite",
        "nonempty": "sample_count > 0",
        "minimum_rms_dbfs_exclusive": -50,
        "minimum_rms_linear_exclusive": 0.0031622776601683794,
        "duration_seconds_inclusive": [0.15, 30.0],
        "sustained_clipping_dbfs": -0.1,
        "sustained_clipping_linear": 0.9885530946569389,
        "failure_rule": "fail if a contiguous run above the clipping threshold exceeds 5 ms",
    }:
        raise SystemExit("registered D3 waveform gate changed")


def cap_record(row: dict) -> dict:
    output = row["output"]
    return {
        "record_id": record_id(row),
        "language": row["language"],
        "case_id": row["case_id"],
        "repeat": row["repeat"],
        "text": row["raw_text"],
        "unicode_character_count": len(row["raw_text"]),
        "seed": row["seed"],
        "generated_frames": row["generated_frames"],
        "max_new_frames": row["max_new_frames"],
        "duration_s": output["duration_s"],
        "raw_stereo_registered_gate_passed": row["raw_waveform_gate"]["passed"],
        "serialized_mono_registered_gate_passed": output["passed"],
        "wav": {
            "path": output["path"],
            "bytes": output["bytes"],
            "sha256": output["sha256"],
        },
        "supplemental_disposition": "FAIL_MAX_NEW_FRAMES_HIT_PROBABLE_NONTERMINATION_OR_TRUNCATION",
    }


def attempt_record(row: dict) -> dict:
    cap_hit = row["max_new_frames_hit"]
    return {
        "record_id": record_id(row),
        "case_id": row["case_id"],
        "repeat": row["repeat"],
        "language": row["language"],
        "text": row["raw_text"],
        "unicode_character_count": len(row["raw_text"]),
        "external_text_normalization": row["external_text_normalization"],
        "seed": row["seed"],
        "seed_recomputed": stable_seed(row["case_id"], row["repeat"]),
        "seed_verified": row["seed"] == stable_seed(row["case_id"], row["repeat"]),
        "sampling": row["sampling"],
        "prompt_name": row["prompt_name"],
        "prompt": row["prompt"],
        "text_tokenization": row["text_tokenization"],
        "generated_frames": row["generated_frames"],
        "max_new_frames": row["max_new_frames"],
        "max_new_frames_hit": cap_hit,
        "audio_token_shape": row["audio_token_shape"],
        "audio_token_sha256": row["audio_token_sha256"],
        "timing": row["timing"],
        "raw_stereo_waveform_gate": row["raw_waveform_gate"],
        "serialization": row["serialization"],
        "serialized_mono_waveform_gate": row["output"],
        "registered_waveform_gate_verdict": "PASS",
        "supplemental_cap_gate_verdict": "FAIL" if cap_hit else "PASS",
        "lane_required_plausible_duration_verdict": (
            "UNRESOLVED_NO_PREDECLARED_TEXT_RELATIVE_THRESHOLD"
        ),
        "effective_safety_verdict": (
            "UNSAFE_PROBABLE_NONTERMINATION_OR_TRUNCATION"
            if cap_hit
            else "UNRESOLVED_PLAUSIBLE_DURATION"
        ),
        "warnings": [],
    }


def language_result(language: str, records: list[dict]) -> dict:
    durations = [row["output"]["duration_s"] for row in records]
    cap_rows = [row for row in records if row["max_new_frames_hit"]]
    non_cap_rows = [row for row in records if not row["max_new_frames_hit"]]
    if cap_rows:
        verdict = "UNSAFE_PROBABLE_NONTERMINATION_OR_TRUNCATION_CAP_HIT"
        warning = None
    else:
        verdict = "UNRESOLVED_PLAUSIBLE_DURATION_NO_TEXT_RELATIVE_THRESHOLD"
        warning = None
    return {
        "language": language,
        "attempted": len(records),
        "registered_raw_stereo_gate_passed": sum(
            row["raw_waveform_gate"]["passed"] is True for row in records
        ),
        "registered_serialized_mono_gate_passed": sum(
            row["output"]["passed"] is True for row in records
        ),
        "supplemental_cap_gate_passed": len(records) - len(cap_rows),
        "supplemental_cap_gate_failed": len(cap_rows),
        "effective_safe_attempts": 0,
        "effective_unresolved_attempts": len(non_cap_rows),
        "effective_unsafe_attempts": len(cap_rows),
        "cap_hit_record_ids": [record_id(row) for row in cap_rows],
        "unresolved_record_ids": [record_id(row) for row in non_cap_rows],
        "duration_s": {
            "min": min(durations),
            "median": statistics.median(durations),
            "max": max(durations),
        },
        "verdict": verdict,
        "warning": warning,
    }


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict) -> str:
    lines = [
        "# G3 — MOSS short-input safety",
        "",
        "MOSS-TTS-Nano is **unsafe on this G3 probe for EN, ES, FR, and ZH**: 23/42 sampled outputs stopped exactly at the 120-frame ceiling, the predeclared supplemental probable-nontermination/truncation disqualifier. The other 19 draws passed the broad registered waveform gate and avoided the cap, but none is promoted to a lane-required plausible-duration PASS because no text-relative rule was predeclared. RU is therefore unresolved, not safe.",
        "",
        "All 42 attempts passed the original registered waveform gate both on the raw 48 kHz stereo decoder tensor and on the retained mono PCM16 WAV. The cap rule is reported separately; it does not rewrite that gate.",
        "",
        "| Language | Attempts | Registered raw/mono gate | Cap hits | Effective safe / unresolved / unsafe | Duration min/median/max | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["language_results"]:
        duration = row["duration_s"]
        lines.append(
            f"| {row['language'].upper()} | {row['attempted']} | "
            f"{row['registered_raw_stereo_gate_passed']}/{row['registered_serialized_mono_gate_passed']} | "
            f"{row['supplemental_cap_gate_failed']} | {row['effective_safe_attempts']} / "
            f"{row['effective_unresolved_attempts']} / {row['effective_unsafe_attempts']} | "
            f"{duration['min']:.2f}/{duration['median']:.2f}/{duration['max']:.2f} s | `{row['verdict']}` |"
        )

    lines.extend(
        [
            "",
            "## Exact cap-hit cases",
            "",
            "Every row below passed the registered waveform gate but fails the supplemental cap gate.",
            "",
            "| Language | Record | Input | Seed | Frames | Duration | WAV |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["exact_cap_hit_records"]:
        lines.append(
            f"| {row['language'].upper()} | `{row['record_id']}` | {markdown_escape(row['text'])} | "
            f"{row['seed']} | {row['generated_frames']}/{row['max_new_frames']} | "
            f"{row['duration_s']:.2f} s | `{row['wav']['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Plausible-duration warning",
            "",
            "All 19 non-cap draws are `UNRESOLVED_PLAUSIBLE_DURATION`, not PASS. The broad registered [0.15, 30] s interval is a waveform sanity gate, not a text-relative plausibility rule. Conspicuous examples include `en_03_r1` (`Yes`, 8.32 s), `ru_01_r1` (`Я`, 8.72 s), and `ru_02_r1` (`Да`, 6.48 s). These examples motivate caution but are not used to invent a post-hoc numeric threshold.",
            "",
            "## Gates and execution order",
            "",
            "Registered waveform gate, preserved exactly from D3: non-empty; all samples finite; RMS > −50 dBFS; duration in [0.15, 30] s; and no contiguous run above −0.1 dBFS longer than 5 ms.",
            "",
            "The synthesis harness checked the raw stereo decoder tensor first. Only a raw-gate pass was downmixed by arithmetic mean to mono PCM16, serialized, checked again with the same gate, and retained. This ordering prevents downmix from hiding a raw-channel catastrophe. All 42 raw and all 42 serialized checks passed.",
            "",
            "Supplemental gate: `max_new_frames_hit` fails when generated frames reach the configured 120-frame ceiling. A hit is evidence of probable nontermination or truncation, not proof of which one occurred; either disposition is unsafe for a short-input route.",
            "",
            "Lane-required plausible-duration review: `UNRESOLVED` for every non-cap draw because no text-relative rule was predeclared. This does not rewrite the registered waveform gate or the cap-gate result.",
            "",
            "## Reproducibility and limits",
            "",
            f"- Artifact: `artifacts/moss/results/g3_synthesis.json`, SHA-256 `{payload['artifacts']['synthesis']['sha256']}` (`HOST`).",
            f"- Registered gate source: `D3.json`, SHA-256 `{payload['registered_waveform_gate']['source_sha256']}` (`EXACT`).",
            "- The exact execution-time `run_moss_g.py` hash was not retained for G3 (`UNKNOWN`). The synthesis receipt itself pins model/codec revisions and hashes, prompt hashes, fixed seeds and sampling, token hashes, waveform metrics, and retained WAV hashes. The current runner is recorded separately and is not misattributed as the executing file.",
            f"- Generation used sampling (`do_sample=true`) with fixed settings and deterministic SHA-256-derived seeds; all 42 seeds were recomputed and matched. Non-ZH cases have one draw each; each ZH text has three draws.",
            "- Deterministic seeds make these draws reproducible but do not estimate the full stochastic failure probability.",
            "- The raw stereo tensors were measured before downmix but were not retained, so their recorded metrics cannot be independently recomputed from the mono WAVs. The 42 retained WAV hashes and sizes were rechecked.",
            "- The English prompt was used for EN/ES/FR/RU and the Mandarin prompt for ZH; this probe does not cover speaker or prompt diversity.",
            "- Waveform/cap safety does not establish intelligibility, correct language, naturalness, licensing, or product acceptance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    synth = read_json(SYNTH_PATH)
    d3 = read_json(D3_PATH)
    validate_inputs(synth, d3)
    records = synth["records"]
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_language[row["language"]].append(row)

    language_results = [language_result(lang, by_language[lang]) for lang in LANGUAGES]
    cap_rows = [row for row in records if row["max_new_frames_hit"]]
    non_cap_rows = [row for row in records if not row["max_new_frames_hit"]]
    plausibility_warning = {
        "id": "PLAUSIBLE_DURATION_UNRESOLVED_NO_PREDECLARED_TEXT_RELATIVE_THRESHOLD",
        "affected_non_cap_attempt_count": len(non_cap_rows),
        "affected_record_ids": [record_id(row) for row in non_cap_rows],
        "affected_record_ids_by_language": {
            language: [
                record_id(row)
                for row in non_cap_rows
                if row["language"] == language
            ]
            for language in LANGUAGES
        },
        "conspicuous_examples": [
            {
                "record_id": "en_03_r1",
                "text": "Yes",
                "duration_s": 8.32,
            },
            {
                "record_id": "ru_01_r1",
                "text": "Я",
                "duration_s": 8.72,
            },
            {
                "record_id": "ru_02_r1",
                "text": "Да",
                "duration_s": 6.48,
            },
        ],
        "route_clearance_disqualifying": True,
        "reason": PLAUSIBLE_DURATION_UNRESOLVED_REASON,
    }
    verdict_by_language = {
        row["language"]: (
            "FAIL"
            if row["registered_raw_stereo_gate_passed"] != row["attempted"]
            or row["registered_serialized_mono_gate_passed"] != row["attempted"]
            or row["supplemental_cap_gate_failed"] > 0
            else "UNRESOLVED"
        )
        for row in language_results
    }
    routes = [
        {
            "engine": "moss",
            "language": row["language"],
            "attempted": row["attempted"],
            "waveform_gate_passed": row["registered_serialized_mono_gate_passed"],
            "waveform_gate_failed": (
                row["attempted"] - row["registered_serialized_mono_gate_passed"]
            ),
            "generation_cap_hits": row["supplemental_cap_gate_failed"],
            "cap_hit_cases": list(row["cap_hit_record_ids"]),
            "duration_s": row["duration_s"],
            "verdict": verdict_by_language[row["language"]],
            "verdict_detail": row["verdict"],
            "warning": (
                plausibility_warning
                if verdict_by_language[row["language"]] == "UNRESOLVED"
                else row["warning"]
            ),
        }
        for row in language_results
    ]
    attempts = [attempt_record(row) for row in records]
    storage_before = directory_bytes(OUT_D)
    registered_gate = d3["gate"]

    payload = {
        "schema_version": 1,
        "sprint_id": "G3",
        "lane": "G",
        "status": "COMPLETE_WITH_CAP_FAILURES_AND_PLAUSIBLE_DURATION_UNRESOLVED",
        "evidence_class": "HOST",
        "exact_hash_evidence_class": "EXACT",
        "final_verdict": {
            "overall": "FAIL",
            "by_language": verdict_by_language,
            "registered_waveform_gate": "PASS_42_OF_42",
            "supplemental_cap_gate": "FAIL_23_OF_42",
            "effective_safe_attempts": 0,
            "effective_unsafe_attempts": 23,
            "effective_unresolved_attempts": 19,
            "safe_languages": [],
            "unsafe_languages": ["en", "es", "fr", "zh"],
            "unresolved_languages": ["ru"],
            "plausible_duration_warning_is_route_clearance_disqualifying": True,
        },
        "headline": {
            "attempted": 42,
            "registered_raw_stereo_gate_passed": 42,
            "registered_serialized_mono_gate_passed": 42,
            "max_new_frames_hit_count": 23,
            "effective_safe": 0,
            "effective_unsafe": 23,
            "effective_unresolved": 19,
            "unsafe_language_count": 4,
            "safe_language_count": 0,
            "unresolved_language_count": 1,
        },
        "model": synth["model"],
        "registered_waveform_gate": {
            "source_path": relative(D3_PATH),
            "source_sha256": sha256_file(D3_PATH),
            "definition": registered_gate,
            "definition_canonical_sha256": sha256_json(registered_gate),
            "preserved_without_modification": True,
            "application_order": [
                "raw 48 kHz stereo decoder tensor",
                "registered waveform gate",
                "arithmetic-mean downmix to mono",
                "PCM16 WAV serialization",
                "same registered waveform gate on retained file",
            ],
            "raw_stereo_passed": 42,
            "serialized_mono_passed": 42,
        },
        "supplemental_probable_nontermination_or_truncation_gate": {
            "id": "MAX_NEW_FRAMES_HIT",
            "predeclared_for_G3_aggregation": True,
            "rule": "FAIL when generated_frames >= max_new_frames",
            "configured_max_new_frames": 120,
            "failure_label": "UNSAFE_PROBABLE_NONTERMINATION_OR_TRUNCATION",
            "reason": (
                "A short-input output that reaches the hard generation ceiling did not demonstrate normal termination; "
                "the retained audio may be nonterminated or truncated."
            ),
            "relationship_to_registered_gate": (
                "Separate supplemental disqualifier; it does not modify or retroactively reinterpret the registered waveform gate."
            ),
            "passed": 19,
            "failed": 23,
        },
        "protocol": {
            "battery": {
                "texts_by_language": {lang: list(TEXTS[lang]) for lang in LANGUAGES},
                "attempts_by_language": {
                    "en": 6,
                    "es": 6,
                    "fr": 6,
                    "ru": 6,
                    "zh": 18,
                },
                "repeat_policy": "one draw per non-ZH text; three draws per ZH text",
                "total_attempts": 42,
            },
            "generation": {
                "sampling": synth["runtime"]["sampling"],
                "do_sample": True,
                "seed_derivation": (
                    "big-endian uint32 from first four bytes of SHA-256('lane-g-moss-v1:g3:{case_id}:{repeat}')"
                ),
                "all_seeds_recomputed_and_matched": all(
                    row["seed_verified"] for row in attempts
                ),
                "external_text_preprocessing": synth["runtime"]["text_preprocessing"],
                "sample_rate_hz": synth["runtime"]["sample_rate_hz"],
                "decoder_channels": synth["runtime"]["channels"],
                "audio_codebooks": synth["runtime"]["nq"],
                "attention_implementation": synth["runtime"][
                    "attention_implementation"
                ],
            },
            "prompts": synth["prompt_records"],
            "host": synth["host"],
        },
        "language_results": language_results,
        "routes": routes,
        "exact_cap_hit_records": [cap_record(row) for row in cap_rows],
        "warnings": [plausibility_warning],
        "attempts": attempts,
        "artifacts": {
            "synthesis": {
                "path": relative(SYNTH_PATH),
                "bytes": SYNTH_PATH.stat().st_size,
                "sha256": sha256_file(SYNTH_PATH),
            },
            "runner_execution": {
                "path": relative(RUNNER_PATH),
                "sha256": None,
                "evidence_class": "UNKNOWN",
                "note": (
                    "G3 executed before the untracked runner's exact source hash was retained. "
                    "The synthesis receipt pins model revisions, weights, prompts, seeds, sampling, "
                    "text preprocessing, generated-token hashes, and output hashes, but the exact "
                    "execution-time Python file hash is unavailable."
                ),
            },
            "runner_current_reproduction_version": {
                "path": relative(RUNNER_PATH),
                "bytes": RUNNER_PATH.stat().st_size,
                "sha256": sha256_file(RUNNER_PATH),
                "evidence_class": "EXACT",
                "note": "Current committed-candidate runner; not misattributed as the G3 execution-time file.",
            },
            "tts_weights": {
                "path": relative(TTS_WEIGHT_PATH),
                "bytes": TTS_WEIGHT_PATH.stat().st_size,
                "sha256": sha256_file(TTS_WEIGHT_PATH),
            },
            "codec_weights": {
                "path": relative(CODEC_WEIGHT_PATH),
                "bytes": CODEC_WEIGHT_PATH.stat().st_size,
                "sha256": sha256_file(CODEC_WEIGHT_PATH),
            },
            "retained_wav_count": 42,
        },
        "limits": [
            "A generation-cap hit identifies probable nontermination or truncation but does not distinguish between them.",
            "Non-ZH inputs have one sampled draw each; ZH inputs have three. These are not stochastic failure-rate estimates.",
            "Raw stereo tensors were measured before downmix but not retained, so raw metrics cannot be independently recomputed from mono files.",
            "The exact execution-time runner source hash was not retained for G3; the synthesis receipt still pins the effective model, prompt, seed, sampling, token, and waveform evidence.",
            "No text-relative duration threshold was predeclared. Because Lane G nevertheless requires plausible duration on every output, all 19 non-cap draws remain unresolved rather than being promoted to PASS.",
            "The English prompt served EN/ES/FR/RU and the Mandarin prompt served ZH; speaker and prompt diversity were not tested.",
            "This short-input safety probe does not establish intelligibility, correct language, naturalness, licensing, or product acceptance.",
        ],
        "reproducibility": {
            "helper_path": relative(Path(__file__).resolve()),
            "helper_sha256": sha256_file(Path(__file__).resolve()),
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "out_d_regular_file_bytes_before_G3_reports": storage_before,
        },
    }
    atomic_write_json(RESULT_JSON_PATH, payload)
    atomic_write_text(RESULT_MD_PATH, render_markdown(payload))
    storage_after = directory_bytes(OUT_D)
    if storage_after >= STORAGE_CAP_BYTES:
        raise SystemExit(
            f"storage cap exceeded after report write: {storage_after} >= {STORAGE_CAP_BYTES}"
        )
    print(f"wrote {RESULT_JSON_PATH}")
    print(f"wrote {RESULT_MD_PATH}")
    print(
        f"registered=42/42 cap_hits={len(cap_rows)}/42 "
        f"unsafe=en,es,fr,zh unresolved=ru storage={storage_after}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
