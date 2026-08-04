#!/usr/bin/env python3
"""Transcribe and score the fixed Lane G G4 reduced-n MOSS screen.

The harness reads exactly ten predeclared D1 item IDs in EN, ES, FR, and RU.
EN/ES/FR use the already pinned faster-whisper-tiny reference route. Russian
uses only the existing pinned GigaAM-v2 CTC INT8 route. Matched-human ASR
transcripts are reused from the completed D1 artifacts.

The two ASR runtimes live in different retained Lane D environments. The run
subcommand orchestrates them sequentially, then invokes the frozen D1 scorer
environment to produce G4.json and G4.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


OUT_D = Path(__file__).resolve().parent
ARTIFACTS = OUT_D / "artifacts"
MOSS_ARTIFACTS = ARTIFACTS / "moss"
G4_AUDIO_ROOT = MOSS_ARTIFACTS / "audio" / "g4"
RESULT_DIR = MOSS_ARTIFACTS / "results"
CHECKPOINT_PATH = RESULT_DIR / "g4_eval_asr_records.jsonl"
ASR_RESULT_PATH = RESULT_DIR / "g4_eval_asr_results.json"
DRY_RUN_PATH = RESULT_DIR / "g4_eval_dry_run.json"
SCORING_RESULT_PATH = RESULT_DIR / "g4_eval_scoring.json"
G4_SYNTHESIS_PATH = RESULT_DIR / "g4_synthesis.json"
G4_JSON_PATH = OUT_D / "G4.json"
G4_MD_PATH = OUT_D / "G4.md"
G1_PATH = OUT_D / "G1.json"
MOSS_RUNNER_PATH = OUT_D / "run_moss_g.py"

D1_CORPUS_PATH = OUT_D / "D1_corpus.json"
D1_SCORING_DIR = ARTIFACTS / "d1_scoring"
D1_NORMALIZER_PATH = D1_SCORING_DIR / "normalization.py"
D1_FIXTURES_PATH = D1_SCORING_DIR / "normalization_fixtures.json"
D1_FASTER_RESULT_PATH = D1_SCORING_DIR / "faster_whisper_results.json"
D1_GIGAAM_RESULT_PATH = D1_SCORING_DIR / "gigaam_results.json"

MAIN_PYTHON = ARTIFACTS / "main" / "venv" / "bin" / "python"
SHERPA_PYTHON = ARTIFACTS / "matcha" / "venv" / "bin" / "python"

LANGUAGES = ("en", "es", "fr", "ru")
WHISPER_LANGUAGES = ("en", "es", "fr")
G4_ITEM_IDS = (
    "fleurs_02",
    "fleurs_03",
    "fleurs_04",
    "fleurs_08",
    "fleurs_11",
    "fleurs_17",
    "fleurs_23",
    "fleurs_32",
    "fleurs_34",
    "fleurs_40",
)

CLASSIFICATION = "EXPLORATORY_REDUCED_N"
TH13_STATUS = "INSUFFICIENT_N_10_LT_60"
RANKING_INADMISSIBLE = True
NO_IMPUTATION = True
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20_260_801
CONTROL_THRESHOLD_PP = 2.0
REDUCED_N_POINT_SCREEN_THRESHOLD_PP = 5.0
THREADS = 2
SYNTH_RUNNER_EXECUTED_SHA256 = (
    "9b3a853ad70c57ef2cc7ef00d2ce1d277e7707f694c5834c5463babdacbf05f2"
)
SYNTH_RUNNER_EXECUTED_BYTES = 27_051
SYNTH_RUNNER_BINDING_CUTOFF_UTC = "2026-08-02T09:20:34Z"
SYNTH_RUNNER_UNKNOWN_KEYS = frozenset(
    [("en", item_id) for item_id in G4_ITEM_IDS]
    + [("es", "fleurs_02"), ("es", "fleurs_03")]
)

NORMALIZER_EXPECTED_SHA256 = (
    "d754293b250e6532b1254f46afbd3a4b7fc36383e64cd44ac3ef1d79493bc989"
)
FASTER_CONFIG_SHA256 = (
    "723e825a01380848fcf05e7c13b0a26a36534fd9fd73ac7f5d6f11d60a763705"
)
GIGAAM_CONFIG_SHA256 = (
    "bb05111ec7be13be98dae283ce45f139a60f8cb8335d08d8335efbad5cebb7da"
)
FASTER_REPO = "Systran/faster-whisper-tiny"
FASTER_REVISION = "d90ca5fe260221311c53c58e660288d3deb8d356"
GIGAAM_ARTIFACT = "sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19"
GIGAAM_MODEL_ROOT = D1_SCORING_DIR / "models" / GIGAAM_ARTIFACT
GIGAAM_MODEL_PATH = GIGAAM_MODEL_ROOT / "model.int8.onnx"
GIGAAM_TOKENS_PATH = GIGAAM_MODEL_ROOT / "tokens.txt"

FASTER_DECODE_CONFIG = {
    "task": "transcribe",
    "beam_size": 5,
    "best_of": 5,
    "patience": 1.0,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "word_timestamps": False,
    "vad_filter": False,
}

GIGAAM_DECODE_CONFIG = {
    "sample_rate_hz": 16000,
    "feature_dim": 80,
    "decoding_method": "greedy_search",
    "provider": "cpu",
    "threads": THREADS,
    "inverse_text_normalization": "none",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
    return rows


def relative(path: Path) -> str:
    return str(path.relative_to(OUT_D))


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"required JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_corpus() -> tuple[dict, dict[str, dict]]:
    payload = load_json(D1_CORPUS_PATH)
    rows = payload.get("items")
    if not isinstance(rows, list) or len(rows) != 40:
        raise SystemExit("D1_corpus.json must contain exactly 40 items")
    by_id = {row.get("item_id"): row for row in rows}
    if len(by_id) != 40:
        raise SystemExit("D1 corpus item IDs are not unique")
    if any(item_id not in by_id for item_id in G4_ITEM_IDS):
        raise SystemExit("one or more predeclared G4 IDs are absent from D1 corpus")
    selected = {item_id: by_id[item_id] for item_id in G4_ITEM_IDS}
    for item_id, row in selected.items():
        texts = row.get("texts", {})
        for language in LANGUAGES:
            if not isinstance(texts.get(language), str) or not texts[language]:
                raise SystemExit(f"missing {language} text for {item_id}")
    return payload, selected


def wav_record(path: Path) -> dict:
    info = sf.info(path)
    if info.frames <= 0 or info.duration <= 0:
        raise SystemExit(f"empty G4 WAV: {path}")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }


def expected_audio_paths() -> dict[tuple[str, str], Path]:
    return {
        (language, item_id): G4_AUDIO_ROOT / language / f"{item_id}.wav"
        for language in LANGUAGES
        for item_id in G4_ITEM_IDS
    }


def build_audio_inventory() -> dict:
    expected = expected_audio_paths()
    actual = (
        {path for path in G4_AUDIO_ROOT.rglob("*.wav")}
        if G4_AUDIO_ROOT.is_dir()
        else set()
    )
    expected_set = set(expected.values())
    unexpected = sorted(relative(path) for path in actual - expected_set)
    records = []
    missing = []
    for (language, item_id), path in expected.items():
        if path.is_file():
            records.append(
                {
                    "language": language,
                    "item_id": item_id,
                    "status": "present",
                    "wav": wav_record(path),
                }
            )
        else:
            missing.append({"language": language, "item_id": item_id})
            records.append(
                {
                    "language": language,
                    "item_id": item_id,
                    "status": "missing",
                    "path": relative(path),
                }
            )
    present_by_language = {
        language: sum(
            row["status"] == "present"
            for row in records
            if row["language"] == language
        )
        for language in LANGUAGES
    }
    complete = not missing and not unexpected and len(actual) == 40
    complete_languages = [
        language
        for language in LANGUAGES
        if present_by_language[language] == len(G4_ITEM_IDS)
    ]
    return {
        "root": relative(G4_AUDIO_ROOT),
        "expected_item_ids": list(G4_ITEM_IDS),
        "expected_languages": list(LANGUAGES),
        "expected_clip_count": 40,
        "present_clip_count": sum(present_by_language.values()),
        "present_by_language": present_by_language,
        "complete_languages": complete_languages,
        "incomplete_languages": [
            language
            for language in LANGUAGES
            if language not in complete_languages
        ],
        "missing": missing,
        "unexpected_wavs": unexpected,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "records": records,
    }


def build_synthesis_receipt(audio_inventory: dict) -> dict:
    """Validate the upstream G4 synthesis receipt without weakening its gates."""
    if not G4_SYNTHESIS_PATH.is_file():
        return {
            "path": relative(G4_SYNTHESIS_PATH),
            "status": "MISSING",
            "record_count": 0,
            "expected_record_count": 40,
            "by_language": {
                language: {
                    "record_count": 0,
                    "failed_record_count": 0,
                    "timeout_count": 0,
                    "waveform_passed_count": 0,
                    "max_new_frames_hit_count": 0,
                    "max_new_frames_unknown_count": 0,
                    "waveform_and_cap_safe_count": 0,
                    "waveform_and_cap_safe": False,
                    "named_failures": [],
                }
                for language in LANGUAGES
            },
            "named_failures": [],
        }

    payload = load_json(G4_SYNTHESIS_PATH)
    if payload.get("suite") != "g4":
        raise SystemExit("unexpected suite in g4_synthesis.json")
    g1 = load_json(G1_PATH)
    g1_tts = g1["primary_pytorch_pins"]["tts"]
    g1_codec = g1["primary_pytorch_pins"]["audio_tokenizer"]
    expected_model = {
        "tts_repo": g1_tts["repo"],
        "tts_revision": g1_tts["revision"],
        "tts_weight_sha256": g1_tts["weight_artifact"]["sha256"],
        "tts_parameters": g1["headline"]["tts_unique_learned_parameters"][
            "value"
        ],
        "codec_repo": g1_codec["repo"],
        "codec_revision": g1_codec["revision"],
        "codec_weight_sha256": g1_codec["weight_artifact"]["sha256"],
        "codec_parameters": g1["headline"][
            "audio_tokenizer_unique_learned_parameters"
        ]["value"],
        "combined_parameters": g1["headline"]["pair_unique_learned_parameters"][
            "value"
        ],
        "source_revision": g1["runtime_reproducibility"][
            "upstream_wrapper_source"
        ]["revision"],
    }
    actual_model = payload.get("model") or {}
    model_pin_matches = all(
        actual_model.get(key) == value for key, value in expected_model.items()
    )
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise SystemExit("g4_synthesis.json records must be a list")

    expected_keys = {
        (language, item_id)
        for language in LANGUAGES
        for item_id in G4_ITEM_IDS
    }
    by_key: dict[tuple[str, str], dict] = {}
    duplicate_keys = []
    unexpected_keys = []
    for row in rows:
        key = (row.get("language"), row.get("item_id"))
        if key not in expected_keys:
            unexpected_keys.append(list(key))
            continue
        if key in by_key:
            duplicate_keys.append(list(key))
            continue
        by_key[key] = row

    audio_by_key = {
        (row["language"], row["item_id"]): row["wav"]
        for row in audio_inventory["records"]
        if row["status"] == "present"
    }
    compact_records = []
    for language in LANGUAGES:
        for item_id in G4_ITEM_IDS:
            key = (language, item_id)
            row = by_key.get(key)
            if row is None:
                continue
            output = row.get("output") or {}
            raw_gate = row.get("raw_waveform_gate") or {}
            wav = audio_by_key.get(key) or {}
            output_identity_matches = bool(
                output.get("path") == wav.get("path")
                and output.get("sha256") == wav.get("sha256")
                and output.get("bytes") == wav.get("bytes")
            )
            waveform_passed = bool(
                row.get("passed")
                and raw_gate.get("passed")
                and output.get("passed")
                and output_identity_matches
            )
            raw_max_new_frames_hit = row.get("max_new_frames_hit")
            max_new_frames_hit = (
                raw_max_new_frames_hit
                if isinstance(raw_max_new_frames_hit, bool)
                else None
            )
            cap_safe = max_new_frames_hit is False
            compact_records.append(
                {
                    "language": language,
                    "item_id": item_id,
                    "case_id": row.get("case_id"),
                    "suite_matches": row.get("suite") == "g4",
                    "case_id_matches": row.get("case_id")
                    == f"{language}_{item_id}",
                    "synthesis_runner_provenance": (
                        synthesis_runner_row_provenance(language, item_id)
                    ),
                    "passed": bool(row.get("passed")),
                    "raw_waveform_gate_passed": bool(raw_gate.get("passed")),
                    "file_waveform_gate_passed": bool(output.get("passed")),
                    "output_identity_matches": output_identity_matches,
                    "waveform_passed": waveform_passed,
                    "max_new_frames": row.get("max_new_frames"),
                    "generated_frames": row.get("generated_frames"),
                    "max_new_frames_hit": max_new_frames_hit,
                    "cap_safe": cap_safe,
                    "waveform_and_cap_safe": waveform_passed and cap_safe,
                    "failure_reason": row.get("failure_reason"),
                    "failure_scope": row.get("failure_scope"),
                    "exception": row.get("exception"),
                    "output_planned_path": row.get("output_planned_path"),
                    "output": output,
                }
            )

    missing_keys = sorted(expected_keys - set(by_key))
    structural_complete = bool(
        len(rows) == 40
        and len(by_key) == 40
        and not missing_keys
        and not duplicate_keys
        and not unexpected_keys
        and payload.get("attempted") == 40
        and payload.get("evidence_class") == "HOST"
        and model_pin_matches
        and payload.get("passed")
        == sum(bool(row.get("passed")) for row in rows)
        and payload.get("failed")
        == sum(not bool(row.get("passed")) for row in rows)
        and payload.get("max_new_frames_hit_count")
        == sum(row.get("max_new_frames_hit") is True for row in rows)
        and len(compact_records) == 40
        and all(
            row["suite_matches"]
            and row["case_id_matches"]
            and (
                not row["passed"]
                or (
                    row["waveform_passed"]
                    and row["cap_safe"]
                    and row["output_identity_matches"]
                )
            )
            for row in compact_records
        )
    )
    by_language = {}
    for language in LANGUAGES:
        selected = [
            row for row in compact_records if row["language"] == language
        ]
        named_failures = []
        for row in selected:
            if row["passed"]:
                continue
            exception = row.get("exception") or {}
            named_failures.append(
                {
                    "language": language,
                    "item_id": row["item_id"],
                    "case_id": row.get("case_id"),
                    "failure_type": (
                        exception.get("type")
                        or row.get("failure_reason")
                        or "SYNTHESIS_FAILED"
                    ),
                    "failure_message": (
                        exception.get("message")
                        or row.get("failure_scope")
                        or row.get("failure_reason")
                    ),
                    "max_new_frames_hit": row["max_new_frames_hit"],
                    "output_retained": bool(row.get("output")),
                }
            )
        by_language[language] = {
            "record_count": len(selected),
            "failed_record_count": sum(not row["passed"] for row in selected),
            "timeout_count": sum(
                (row.get("exception") or {}).get("type") == "TimeoutError"
                for row in selected
            ),
            "waveform_passed_count": sum(
                row["waveform_passed"] for row in selected
            ),
            "max_new_frames_hit_count": sum(
                row["max_new_frames_hit"] is True for row in selected
            ),
            "max_new_frames_unknown_count": sum(
                row["max_new_frames_hit"] is None for row in selected
            ),
            "waveform_and_cap_safe_count": sum(
                row["waveform_and_cap_safe"] for row in selected
            ),
            "waveform_and_cap_safe": bool(
                len(selected) == 10
                and all(row["waveform_and_cap_safe"] for row in selected)
            ),
            "named_failures": named_failures,
        }
    all_named_failures = [
        failure
        for language in LANGUAGES
        for failure in by_language[language]["named_failures"]
    ]
    return {
        "path": relative(G4_SYNTHESIS_PATH),
        "bytes": G4_SYNTHESIS_PATH.stat().st_size,
        "sha256": sha256_file(G4_SYNTHESIS_PATH),
        "status": "COMPLETE" if structural_complete else "INVALID",
        "evidence_class": payload.get("evidence_class"),
        "model": payload.get("model"),
        "expected_model_from_g1": expected_model,
        "model_pin_matches_g1": model_pin_matches,
        "synthesis_runner_provenance": synthesis_runner_provenance(),
        "runtime": payload.get("runtime"),
        "host": payload.get("host"),
        "record_count": len(rows),
        "expected_record_count": 40,
        "reported_attempted": payload.get("attempted"),
        "reported_passed": payload.get("passed"),
        "reported_failed": payload.get("failed"),
        "reported_max_new_frames_hit_count": payload.get(
            "max_new_frames_hit_count"
        ),
        "missing_keys": [list(key) for key in missing_keys],
        "duplicate_keys": duplicate_keys,
        "unexpected_keys": unexpected_keys,
        "by_language": by_language,
        "named_failures": all_named_failures,
        "records": compact_records,
    }


def verify_recorded_file(record: dict) -> dict:
    path_text = record.get("resolved_path") or record.get("path")
    if not path_text:
        raise SystemExit(f"model provenance record lacks path: {record}")
    path = Path(path_text)
    if not path.is_absolute():
        path = OUT_D / path
    if not path.is_file():
        raise SystemExit(f"pinned model artifact missing: {path}")
    actual = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if actual["bytes"] != record.get("bytes"):
        raise SystemExit(f"pinned model size mismatch: {path}")
    if actual["sha256"] != record.get("sha256"):
        raise SystemExit(f"pinned model hash mismatch: {path}")
    return actual


def select_human_records(
    payload: dict,
    selected_corpus: dict[str, dict],
    languages: tuple[str, ...],
) -> dict[tuple[str, str], dict]:
    selected = {}
    for row in payload.get("records", []):
        language = row.get("language")
        item_id = row.get("item_id")
        if (
            language not in languages
            or item_id not in G4_ITEM_IDS
            or row.get("source_kind") != "matched_human"
            or row.get("group_id") != f"human_fleurs_{language}"
        ):
            continue
        key = (language, item_id)
        if key in selected:
            raise SystemExit(f"duplicate matched-human transcript: {key}")
        expected_reference = selected_corpus[item_id]["texts"][language]
        if row.get("reference_text") != expected_reference:
            raise SystemExit(f"matched-human reference mismatch: {key}")
        selected[key] = row
    expected_keys = {
        (language, item_id)
        for language in languages
        for item_id in G4_ITEM_IDS
    }
    if set(selected) != expected_keys:
        missing = sorted(expected_keys - set(selected))
        extra = sorted(set(selected) - expected_keys)
        raise SystemExit(
            f"matched-human transcript selection mismatch missing={missing} extra={extra}"
        )
    return selected


def load_reference_bundle(
    selected_corpus: dict[str, dict],
) -> tuple[dict[tuple[str, str], dict], dict]:
    faster = load_json(D1_FASTER_RESULT_PATH)
    gigaam = load_json(D1_GIGAAM_RESULT_PATH)
    if faster.get("config_sha256") != FASTER_CONFIG_SHA256:
        raise SystemExit("faster-whisper D1 config hash changed")
    if gigaam.get("config_sha256") != GIGAAM_CONFIG_SHA256:
        raise SystemExit("GigaAM D1 config hash changed")
    faster_model = faster.get("model", {})
    gigaam_model = gigaam.get("model", {})
    if faster_model.get("repo") != FASTER_REPO:
        raise SystemExit("unexpected faster-whisper repository")
    if faster_model.get("revision") != FASTER_REVISION:
        raise SystemExit("unexpected faster-whisper revision")
    if faster_model.get("compute_type") != "float32":
        raise SystemExit("faster-whisper compute type is not frozen float32")
    if faster_model.get("decoding") != FASTER_DECODE_CONFIG:
        raise SystemExit("faster-whisper decoding config changed")
    if gigaam_model.get("artifact_name") != GIGAAM_ARTIFACT:
        raise SystemExit("unexpected GigaAM artifact")
    if gigaam_model.get("decoding") != GIGAAM_DECODE_CONFIG:
        raise SystemExit("GigaAM decoding config changed")
    faster_files = [
        verify_recorded_file(record) for record in faster_model.get("files", [])
    ]
    gigaam_files = [
        verify_recorded_file(record) for record in gigaam_model.get("files", [])
    ]
    if len(faster_files) != 4:
        raise SystemExit("faster-whisper provenance must contain four pinned files")
    if not GIGAAM_MODEL_PATH.is_file() or not GIGAAM_TOKENS_PATH.is_file():
        raise SystemExit("pinned GigaAM model or token file is missing")
    human = select_human_records(faster, selected_corpus, WHISPER_LANGUAGES)
    human.update(select_human_records(gigaam, selected_corpus, ("ru",)))
    public = {
        "human_transcript_count": len(human),
        "selected_ids_per_language": len(G4_ITEM_IDS),
        "faster_whisper": {
            "artifact_path": relative(D1_FASTER_RESULT_PATH),
            "artifact_bytes": D1_FASTER_RESULT_PATH.stat().st_size,
            "artifact_sha256": sha256_file(D1_FASTER_RESULT_PATH),
            "config_sha256": FASTER_CONFIG_SHA256,
            "repo": FASTER_REPO,
            "revision": FASTER_REVISION,
            "snapshot_path": faster_model.get("snapshot_path"),
            "compute_type": "float32",
            "decoding": FASTER_DECODE_CONFIG,
            "verified_files": faster_files,
            "languages": list(WHISPER_LANGUAGES),
        },
        "gigaam": {
            "artifact_path": relative(D1_GIGAAM_RESULT_PATH),
            "artifact_bytes": D1_GIGAAM_RESULT_PATH.stat().st_size,
            "artifact_sha256": sha256_file(D1_GIGAAM_RESULT_PATH),
            "config_sha256": GIGAAM_CONFIG_SHA256,
            "artifact_name": GIGAAM_ARTIFACT,
            "prebuilt_quantization": "INT8",
            "decoding": GIGAAM_DECODE_CONFIG,
            "verified_files": gigaam_files,
            "languages": ["ru"],
            "route_assertion": "Russian uses GigaAM only; Whisper is forbidden.",
        },
    }
    return human, public


def runtime_probe(interpreter: Path, modules: tuple[str, ...]) -> dict:
    if not interpreter.is_file():
        return {
            "interpreter": str(interpreter),
            "status": "MISSING",
            "modules": list(modules),
        }
    code = (
        "import importlib.metadata,json,sys;"
        f"mods={list(modules)!r};"
        "out={};"
        "\nfor name in mods:\n"
        "  try:\n"
        "    module=__import__(name);"
        "    out[name]=getattr(module,'__version__',None) or "
        "importlib.metadata.version(name.replace('_','-'))\n"
        "  except Exception as e:\n"
        "    out[name]='ERROR:'+repr(e)\n"
        "print(json.dumps({'python':sys.version.split()[0],'modules':out},sort_keys=True))"
    )
    result = subprocess.run(
        [str(interpreter), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "interpreter": str(interpreter),
        "status": "READY" if result.returncode == 0 else "ERROR",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "modules": list(modules),
    }


def exact_commands() -> dict:
    script = Path(__file__).resolve()
    return {
        "dry_run": f"{MAIN_PYTHON} {script} dry-run",
        "full_orchestrated_run": f"{MAIN_PYTHON} {script} run",
        "whisper_only": f"{MAIN_PYTHON} {script} transcribe-whisper",
        "ru_only": f"{SHERPA_PYTHON} {script} transcribe-ru",
        "score_only": f"{MAIN_PYTHON} {script} score",
    }


def synthesis_runner_provenance() -> dict:
    current = {
        "path": relative(MOSS_RUNNER_PATH),
        "bytes": MOSS_RUNNER_PATH.stat().st_size,
        "sha256": sha256_file(MOSS_RUNNER_PATH),
    }
    receipt_rows = []
    if G4_SYNTHESIS_PATH.is_file():
        receipt_rows = load_json(G4_SYNTHESIS_PATH).get("records") or []
    receipt_rewrite = {
        "records_with_case_config_sha256": sum(
            isinstance(row.get("case_config_sha256"), str)
            for row in receipt_rows
        ),
        "records_marked_checkpoint_migrated_from_legacy": sum(
            row.get("checkpoint_migrated_from_legacy") is True
            for row in receipt_rows
        ),
        "case_timeout_s_values": sorted(
            {
                row.get("case_config", {}).get("case_timeout_s")
                for row in receipt_rows
                if row.get("case_config") is not None
            }
        ),
        "evidence_class": "EXACT",
        "provenance_interpretation": (
            "Post-run checkpoint compatibility/migration metadata only. It "
            "does not prove which runner generated a waveform, and it must "
            "not upgrade the 12 execution-runner-UNKNOWN rows."
        ),
    }
    return {
        "scope": "PARTIAL_ROW_BINDING",
        "observed_runner_for_bound_rows": {
            "path": relative(MOSS_RUNNER_PATH),
            "bytes": SYNTH_RUNNER_EXECUTED_BYTES,
            "sha256": SYNTH_RUNNER_EXECUTED_SHA256,
            "evidence_class": "HOST",
            "command": (
                "artifacts/main/venv/bin/python run_moss_g.py --suite g4 g5 "
                "--case-timeout-s 300"
            ),
            "observation": (
                "Captured after the resumed live G4/G5 process started; the "
                "first row bound to this observation is ES fleurs_04."
            ),
        },
        "binding_cutoff_utc": SYNTH_RUNNER_BINDING_CUTOFF_UTC,
        "bound_row_count": 28,
        "bound_row_scope": "ES fleurs_04 onward: ES remaining 8 + FR 10 + RU 10",
        "unknown_execution_runner_row_count": 12,
        "unknown_execution_runner_rows": [
            {"language": language, "item_id": item_id}
            for language, item_id in sorted(
                SYNTH_RUNNER_UNKNOWN_KEYS,
                key=lambda key: (LANGUAGES.index(key[0]), G4_ITEM_IDS.index(key[1])),
            )
        ],
        "current_worktree_file": current,
        "current_matches_observed_bound_runner": bool(
            current["bytes"] == SYNTH_RUNNER_EXECUTED_BYTES
            and current["sha256"] == SYNTH_RUNNER_EXECUTED_SHA256
        ),
        "receipt_rewrite": receipt_rewrite,
        "provenance_rule": (
            "Only the 28 rows from ES fleurs_04 onward bind to the observed "
            "runner hash. EN 10 plus ES fleurs_02 and fleurs_03 predate the "
            "observed PID and remain execution-runner UNKNOWN. Later worktree "
            "integrity edits are not claimed to have executed for any row."
        ),
    }


def synthesis_runner_row_provenance(language: str, item_id: str) -> dict:
    if (language, item_id) in SYNTH_RUNNER_UNKNOWN_KEYS:
        return {
            "status": "UNKNOWN_EXECUTION_RUNNER",
            "evidence_class": "UNKNOWN",
            "reason": (
                "Checkpoint row predates the observed resumed PID; no exact "
                "runner hash was captured for this row."
            ),
        }
    return {
        "status": "BOUND_TO_OBSERVED_RUNNER",
        "evidence_class": "HOST",
        "runner_sha256": SYNTH_RUNNER_EXECUTED_SHA256,
        "runner_bytes": SYNTH_RUNNER_EXECUTED_BYTES,
        "binding_cutoff_utc": SYNTH_RUNNER_BINDING_CUTOFF_UTC,
    }


def build_plan() -> tuple[dict, dict[str, dict], dict[tuple[str, str], dict]]:
    corpus, selected_corpus = load_selected_corpus()
    human, reference_public = load_reference_bundle(selected_corpus)
    audio = build_audio_inventory()
    synthesis_receipt = build_synthesis_receipt(audio)
    if synthesis_receipt["status"] != "COMPLETE":
        status = "WAITING_FOR_G4_SYNTHESIS_RECEIPT"
    elif audio["status"] == "COMPLETE":
        status = "READY_FOR_TRANSCRIPTION"
    else:
        status = "READY_FOR_PARTIAL_LANGUAGE_TRANSCRIPTION"
    plan = {
        "schema_version": 1,
        "task": "Lane G G4 reduced-n all-four-language MOSS ASR backcheck",
        "status": status,
        "classification": CLASSIFICATION,
        "ranking_inadmissible": RANKING_INADMISSIBLE,
        "th13": TH13_STATUS,
        "no_imputation": NO_IMPUTATION,
        "fixed_item_ids": list(G4_ITEM_IDS),
        "languages": list(LANGUAGES),
        "n_per_language": 10,
        "corpus": {
            "path": relative(D1_CORPUS_PATH),
            "sha256": sha256_file(D1_CORPUS_PATH),
            "selected_items_sha256": sha256_json(
                [
                    {
                        "item_id": item_id,
                        "texts": {
                            language: selected_corpus[item_id]["texts"][language]
                            for language in LANGUAGES
                        },
                    }
                    for item_id in G4_ITEM_IDS
                ]
            ),
        },
        "audio_inventory": audio,
        "synthesis_receipt": synthesis_receipt,
        "synthesis_runner_provenance": synthesis_runner_provenance(),
        "reference_asr": reference_public,
        "normalization": {
            "source_path": relative(D1_NORMALIZER_PATH),
            "expected_source_sha256": NORMALIZER_EXPECTED_SHA256,
            "actual_source_sha256": sha256_file(D1_NORMALIZER_PATH),
            "fixtures_path": relative(D1_FIXTURES_PATH),
            "fixtures_sha256": sha256_file(D1_FIXTURES_PATH),
            "required_variants": ["primary", "basic"],
        },
        "statistics": {
            "point_estimator": "ratio_of_sums_sum_errors_div_sum_reference_units",
            "paired_delta": "synth_error_rate_minus_matched_human_error_rate",
            "bootstrap_B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.random.PCG64",
            "ci_method": "paired_utterance_bootstrap_percentile",
            "percentiles": [2.5, 97.5],
            "cluster_unit": "utterance",
        },
        "runtime_environments": {
            "faster_whisper_and_scoring": runtime_probe(
                MAIN_PYTHON,
                (
                    "faster_whisper",
                    "numpy",
                    "soundfile",
                    "transformers",
                    "cn2an",
                    "num2words",
                    "opencc",
                ),
            ),
            "gigaam": runtime_probe(
                SHERPA_PYTHON, ("sherpa_onnx", "numpy", "soundfile")
            ),
        },
        "commands": exact_commands(),
        "script": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if plan["normalization"]["actual_source_sha256"] != NORMALIZER_EXPECTED_SHA256:
        raise SystemExit("frozen D1 normalizer source hash changed")
    for environment in plan["runtime_environments"].values():
        if environment["status"] != "READY":
            raise SystemExit(f"required retained runtime is not ready: {environment}")
        if "ERROR:" in environment["stdout"]:
            raise SystemExit(f"required runtime import failed: {environment}")
    return plan, selected_corpus, human


def competing_moss_synthesis_processes() -> list[dict]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,etime=,%cpu=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = []
    own_pid = os.getpid()
    for line in result.stdout.splitlines():
        if "run_moss_g.py" not in line:
            continue
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        pid = int(parts[0])
        if pid == own_pid:
            continue
        matches.append(
            {
                "pid": pid,
                "elapsed": parts[1],
                "cpu_percent": float(parts[2]),
                "command": parts[3],
            }
        )
    return matches


def require_synthesis_complete(plan: dict) -> None:
    audio = plan["audio_inventory"]
    if audio["unexpected_wavs"]:
        raise SystemExit(f"unexpected G4 WAV files: {audio['unexpected_wavs']}")
    synthesis_receipt = plan["synthesis_receipt"]
    if synthesis_receipt["status"] != "COMPLETE":
        raise SystemExit(
            "G4 synthesis receipt is not complete and valid: "
            f"{synthesis_receipt['status']}"
        )


def complete_audio_languages(
    plan: dict, route_languages: tuple[str, ...] = LANGUAGES
) -> tuple[str, ...]:
    complete = set(plan["audio_inventory"]["complete_languages"])
    return tuple(language for language in route_languages if language in complete)


def refuse_during_moss_synthesis() -> None:
    matches = competing_moss_synthesis_processes()
    if matches:
        raise SystemExit(
            "refusing CPU-heavy ASR while MOSS synthesis is running: "
            + json.dumps(matches, ensure_ascii=False)
        )


def checkpoint_key(record: dict) -> tuple:
    return (
        record["asr_route_id"],
        record["asr_config_sha256"],
        record["language"],
        record["item_id"],
        record["wav"]["sha256"],
    )


def current_checkpoint_records(
    plan: dict, *, route_languages: tuple[str, ...] = LANGUAGES
) -> dict[tuple, dict]:
    all_rows = read_jsonl(CHECKPOINT_PATH)
    by_key = {checkpoint_key(row): row for row in all_rows}
    config_by_language = {
        "en": FASTER_CONFIG_SHA256,
        "es": FASTER_CONFIG_SHA256,
        "fr": FASTER_CONFIG_SHA256,
        "ru": GIGAAM_CONFIG_SHA256,
    }
    route_by_language = {
        "en": "faster_whisper_tiny",
        "es": "faster_whisper_tiny",
        "fr": "faster_whisper_tiny",
        "ru": "gigaam_v2_ctc_int8",
    }
    wav_by_key = {
        (row["language"], row["item_id"]): row["wav"]
        for row in plan["audio_inventory"]["records"]
        if row["status"] == "present"
    }
    current = {}
    for language in route_languages:
        for item_id in G4_ITEM_IDS:
            wav = wav_by_key.get((language, item_id))
            if wav is None:
                continue
            key = (
                route_by_language[language],
                config_by_language[language],
                language,
                item_id,
                wav["sha256"],
            )
            if key in by_key:
                current[key] = by_key[key]
    return current


def record_base(
    *,
    language: str,
    item_id: str,
    item: dict,
    wav: dict,
    route_id: str,
    config_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "task": "Lane G G4",
        "classification": CLASSIFICATION,
        "asr_route_id": route_id,
        "asr_config_sha256": config_sha256,
        "language": language,
        "group_id": "moss_g4",
        "source_kind": "synthesized",
        "engine_id": "moss_tts_nano_100m",
        "item_id": item_id,
        "sentence_id": item.get("sentence_id"),
        "categories": item.get("categories", []),
        "reference_text": item["texts"][language],
        "wav": wav,
    }


def transcribe_whisper_one(
    *,
    model,
    language: str,
    item_id: str,
    item: dict,
    path: Path,
    wav: dict,
) -> dict:
    started = time.perf_counter_ns()
    segments_iter, info = model.transcribe(
        str(path),
        language=language,
        task=FASTER_DECODE_CONFIG["task"],
        beam_size=FASTER_DECODE_CONFIG["beam_size"],
        best_of=FASTER_DECODE_CONFIG["best_of"],
        patience=FASTER_DECODE_CONFIG["patience"],
        temperature=FASTER_DECODE_CONFIG["temperature"],
        compression_ratio_threshold=FASTER_DECODE_CONFIG[
            "compression_ratio_threshold"
        ],
        log_prob_threshold=FASTER_DECODE_CONFIG["log_prob_threshold"],
        no_speech_threshold=FASTER_DECODE_CONFIG["no_speech_threshold"],
        condition_on_previous_text=FASTER_DECODE_CONFIG[
            "condition_on_previous_text"
        ],
        without_timestamps=FASTER_DECODE_CONFIG["without_timestamps"],
        word_timestamps=FASTER_DECODE_CONFIG["word_timestamps"],
        vad_filter=FASTER_DECODE_CONFIG["vad_filter"],
    )
    segments = list(segments_iter)
    decode_s = (time.perf_counter_ns() - started) / 1e9
    text = "".join(segment.text for segment in segments).strip()
    return {
        **record_base(
            language=language,
            item_id=item_id,
            item=item,
            wav=wav,
            route_id="faster_whisper_tiny",
            config_sha256=FASTER_CONFIG_SHA256,
        ),
        "asr_text_raw": text,
        "segments": [
            {
                "id": int(segment.id),
                "start_s": float(segment.start),
                "end_s": float(segment.end),
                "text": segment.text,
                "temperature": float(segment.temperature),
                "avg_logprob": float(segment.avg_logprob),
                "compression_ratio": float(segment.compression_ratio),
                "no_speech_prob": float(segment.no_speech_prob),
            }
            for segment in segments
        ],
        "recognizer_info": {
            "requested_language": language,
            "detected_language": info.language,
            "language_probability": float(info.language_probability),
            "duration_s": float(info.duration),
            "duration_after_vad_s": float(info.duration_after_vad),
        },
        "timing": {
            "decode_s": decode_s,
            "decode_rtf": decode_s / wav["duration_s"],
        },
        "transcribed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def transcribe_ru_one(
    *,
    recognizer,
    item_id: str,
    item: dict,
    path: Path,
    wav: dict,
) -> dict:
    read_start = time.perf_counter_ns()
    samples_2d, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    read_s = (time.perf_counter_ns() - read_start) / 1e9
    samples = (
        samples_2d[:, 0]
        if samples_2d.shape[1] == 1
        else samples_2d.mean(axis=1, dtype=np.float32)
    )
    if samples.size == 0 or not np.isfinite(samples).all():
        raise SystemExit(f"invalid GigaAM input waveform: {path}")
    stream = recognizer.create_stream()
    accept_start = time.perf_counter_ns()
    stream.accept_waveform(int(sample_rate), samples)
    accept_s = (time.perf_counter_ns() - accept_start) / 1e9
    decode_start = time.perf_counter_ns()
    recognizer.decode_stream(stream)
    decode_s = (time.perf_counter_ns() - decode_start) / 1e9
    result = stream.result
    return {
        **record_base(
            language="ru",
            item_id=item_id,
            item=item,
            wav=wav,
            route_id="gigaam_v2_ctc_int8",
            config_sha256=GIGAAM_CONFIG_SHA256,
        ),
        "asr_text_raw": result.text,
        "asr_tokens_raw": list(result.tokens),
        "asr_timestamps_raw": [float(value) for value in result.timestamps],
        "timing": {
            "read_s": read_s,
            "accept_waveform_and_resample_s": accept_s,
            "decode_s": decode_s,
            "decode_rtf": decode_s / wav["duration_s"],
        },
        "transcribed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def transcribe_whisper(*, force: bool, max_new_clips: int | None) -> int:
    plan, selected_corpus, _ = build_plan()
    require_synthesis_complete(plan)
    refuse_during_moss_synthesis()
    route_languages = complete_audio_languages(plan, WHISPER_LANGUAGES)

    existing = {} if force else current_checkpoint_records(
        plan, route_languages=route_languages
    )
    wav_by_key = {
        (row["language"], row["item_id"]): row["wav"]
        for row in plan["audio_inventory"]["records"]
        if row["status"] == "present"
    }
    jobs = []
    for language in route_languages:
        for item_id in G4_ITEM_IDS:
            wav = wav_by_key[(language, item_id)]
            key = (
                "faster_whisper_tiny",
                FASTER_CONFIG_SHA256,
                language,
                item_id,
                wav["sha256"],
            )
            if key not in existing:
                jobs.append((language, item_id, wav))
    if max_new_clips is not None:
        jobs = jobs[:max_new_clips]
    if jobs:
        from faster_whisper import WhisperModel

        snapshot = plan["reference_asr"]["faster_whisper"]["snapshot_path"]
        model = WhisperModel(
            snapshot,
            device="cpu",
            compute_type="float32",
            cpu_threads=THREADS,
            num_workers=1,
            local_files_only=True,
        )
        for language, item_id, wav in jobs:
            path = OUT_D / wav["path"]
            record = transcribe_whisper_one(
                model=model,
                language=language,
                item_id=item_id,
                item=selected_corpus[item_id],
                path=path,
                wav=wav,
            )
            append_jsonl(CHECKPOINT_PATH, record)
            print(
                f"{language} {item_id} decode={record['timing']['decode_s']:.3f}s "
                f"rtf={record['timing']['decode_rtf']:.3f} "
                f"empty={not bool(record['asr_text_raw'])}",
                flush=True,
            )
    write_asr_result(plan)
    return 0


def transcribe_ru(*, force: bool, max_new_clips: int | None) -> int:
    plan, selected_corpus, _ = build_plan()
    require_synthesis_complete(plan)
    refuse_during_moss_synthesis()
    route_languages = complete_audio_languages(plan, ("ru",))

    existing = {} if force else current_checkpoint_records(
        plan, route_languages=route_languages
    )
    wav_by_key = {
        (row["language"], row["item_id"]): row["wav"]
        for row in plan["audio_inventory"]["records"]
        if row["status"] == "present"
    }
    jobs = []
    if route_languages:
        for item_id in G4_ITEM_IDS:
            wav = wav_by_key[("ru", item_id)]
            key = (
                "gigaam_v2_ctc_int8",
                GIGAAM_CONFIG_SHA256,
                "ru",
                item_id,
                wav["sha256"],
            )
            if key not in existing:
                jobs.append((item_id, wav))
    if max_new_clips is not None:
        jobs = jobs[:max_new_clips]
    if jobs:
        import sherpa_onnx

        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(GIGAAM_MODEL_PATH),
            tokens=str(GIGAAM_TOKENS_PATH),
            num_threads=THREADS,
            sample_rate=GIGAAM_DECODE_CONFIG["sample_rate_hz"],
            feature_dim=GIGAAM_DECODE_CONFIG["feature_dim"],
            decoding_method=GIGAAM_DECODE_CONFIG["decoding_method"],
            debug=False,
            provider=GIGAAM_DECODE_CONFIG["provider"],
        )
        for item_id, wav in jobs:
            path = OUT_D / wav["path"]
            record = transcribe_ru_one(
                recognizer=recognizer,
                item_id=item_id,
                item=selected_corpus[item_id],
                path=path,
                wav=wav,
            )
            append_jsonl(CHECKPOINT_PATH, record)
            print(
                f"ru {item_id} decode={record['timing']['decode_s']:.3f}s "
                f"rtf={record['timing']['decode_rtf']:.3f} "
                f"empty={not bool(record['asr_text_raw'])}",
                flush=True,
            )
    write_asr_result(plan)
    return 0


def summarize_asr_records(records: list[dict], plan: dict) -> dict:
    by_language = {}
    for language in LANGUAGES:
        selected = [row for row in records if row["language"] == language]
        decode = [row["timing"]["decode_s"] for row in selected]
        rtf = [row["timing"]["decode_rtf"] for row in selected]
        by_language[language] = {
            "record_count": len(selected),
            "expected_record_count": 10,
            "audio_clip_count": plan["audio_inventory"][
                "present_by_language"
            ][language],
            "empty_transcript_count": sum(
                not row.get("asr_text_raw") for row in selected
            ),
            "input_duration_total_s": float(
                sum(row["wav"]["duration_s"] for row in selected)
            ),
            "decode_total_s": float(sum(decode)),
            "decode_s_median": (
                float(statistics.median(decode)) if decode else None
            ),
            "decode_rtf_median": (
                float(statistics.median(rtf)) if rtf else None
            ),
        }
        by_language[language]["complete_for_scoring"] = bool(
            by_language[language]["audio_clip_count"] == 10
            and by_language[language]["record_count"] == 10
        )
    return {
        "record_count": len(records),
        "expected_record_count": 40,
        "complete": len(records) == 40,
        "complete_language_count": sum(
            row["complete_for_scoring"] for row in by_language.values()
        ),
        "by_language": by_language,
    }


def write_asr_result(plan: dict) -> dict:
    records = list(current_checkpoint_records(plan).values())
    records.sort(key=lambda row: (LANGUAGES.index(row["language"]), row["item_id"]))
    payload = {
        "schema_version": 1,
        "task": "Lane G G4 reference-ASR transcripts",
        "classification": CLASSIFICATION,
        "evidence_class": "HOST",
        "ranking_inadmissible": True,
        "th13": TH13_STATUS,
        "no_imputation": True,
        "fixed_item_ids": list(G4_ITEM_IDS),
        "audio_inventory": plan["audio_inventory"],
        "synthesis_receipt": plan["synthesis_receipt"],
        "synthesis_runner_provenance": plan[
            "synthesis_runner_provenance"
        ],
        "reference_asr": plan["reference_asr"],
        "checkpoint_path": relative(CHECKPOINT_PATH),
        "checkpoint_sha256": (
            sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
        },
        "summary": summarize_asr_records(records, plan),
        "records": records,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(ASR_RESULT_PATH, payload)
    return payload


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1],
                    distance[i - 1][j],
                    distance[i][j - 1],
                )
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j and distance[i][j] == distance[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            raise AssertionError("Levenshtein backtrace reached invalid state")
    errors = substitutions + deletions + insertions
    if errors != distance[-1][-1]:
        raise AssertionError("edit-count reconciliation failed")
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "reference_units": len(reference),
        "hypothesis_units": len(hypothesis),
    }


def score_record(
    *,
    normalization,
    language: str,
    item_id: str,
    variant: str,
    record: dict,
    arm: str,
) -> dict:
    reference = normalization.normalize(
        language, record["reference_text"], variant
    )
    hypothesis = normalization.normalize(
        language, record.get("asr_text_raw", ""), variant
    )
    if not reference["tokens"]:
        raise SystemExit(
            f"empty normalized reference: {language}/{item_id}/{variant}"
        )
    counts = edit_counts(reference["tokens"], hypothesis["tokens"])
    return {
        "language": language,
        "item_id": item_id,
        "arm": arm,
        "normalizer_variant": variant,
        "normalizer_id": reference["normalizer_id"],
        "reference_text_raw": record["reference_text"],
        "hypothesis_text_raw": record.get("asr_text_raw", ""),
        "reference_normalized": reference["normalized_text"],
        "hypothesis_normalized": hypothesis["normalized_text"],
        "reference_tokens": reference["tokens"],
        "hypothesis_tokens": hypothesis["tokens"],
        **counts,
        "utterance_error_rate": counts["errors"] / counts["reference_units"],
        "wav_path": record["wav"]["path"],
        "wav_sha256": record["wav"]["sha256"],
    }


def bootstrap_draws() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(
        0, len(G4_ITEM_IDS), size=(BOOTSTRAP_B, len(G4_ITEM_IDS)), endpoint=False
    )


def arm_summary(rows: list[dict], draws: np.ndarray) -> dict:
    errors = np.asarray([row["errors"] for row in rows], dtype=np.float64)
    refs = np.asarray([row["reference_units"] for row in rows], dtype=np.float64)
    bootstrap = errors[draws].sum(axis=1) / refs[draws].sum(axis=1)
    point = float(errors.sum() / refs.sum())
    return {
        "n": len(rows),
        "substitutions": sum(row["substitutions"] for row in rows),
        "deletions": sum(row["deletions"] for row in rows),
        "insertions": sum(row["insertions"] for row in rows),
        "errors": int(errors.sum()),
        "reference_units": int(refs.sum()),
        "point": point,
        "point_pp": point * 100.0,
        "ci95_lo": float(np.percentile(bootstrap, 2.5)),
        "ci95_hi": float(np.percentile(bootstrap, 97.5)),
        "ci95_lo_pp": float(np.percentile(bootstrap, 2.5) * 100.0),
        "ci95_hi_pp": float(np.percentile(bootstrap, 97.5) * 100.0),
    }


def paired_summary(human: list[dict], synth: list[dict]) -> dict:
    human_by_id = {row["item_id"]: row for row in human}
    synth_by_id = {row["item_id"]: row for row in synth}
    expected = set(G4_ITEM_IDS)
    if set(human_by_id) != expected or set(synth_by_id) != expected:
        raise AssertionError("paired G4 comparison lacks exactly ten fixed IDs")
    ordered_human = [human_by_id[item_id] for item_id in G4_ITEM_IDS]
    ordered_synth = [synth_by_id[item_id] for item_id in G4_ITEM_IDS]
    refs = np.asarray(
        [row["reference_units"] for row in ordered_human], dtype=np.float64
    )
    synth_refs = np.asarray(
        [row["reference_units"] for row in ordered_synth], dtype=np.float64
    )
    if not np.array_equal(refs, synth_refs):
        raise AssertionError("paired normalized reference lengths differ")
    human_errors = np.asarray(
        [row["errors"] for row in ordered_human], dtype=np.float64
    )
    synth_errors = np.asarray(
        [row["errors"] for row in ordered_synth], dtype=np.float64
    )
    draws = bootstrap_draws()
    denominator = refs[draws].sum(axis=1)
    human_boot = human_errors[draws].sum(axis=1) / denominator
    synth_boot = synth_errors[draws].sum(axis=1) / denominator
    delta_boot = synth_boot - human_boot
    human_point = float(human_errors.sum() / refs.sum())
    synth_point = float(synth_errors.sum() / refs.sum())
    delta = synth_point - human_point
    return {
        "human": arm_summary(ordered_human, draws),
        "synth": arm_summary(ordered_synth, draws),
        "delta": {
            "point": delta,
            "point_pp": delta * 100.0,
            "ci95_lo": float(np.percentile(delta_boot, 2.5)),
            "ci95_hi": float(np.percentile(delta_boot, 97.5)),
            "ci95_lo_pp": float(np.percentile(delta_boot, 2.5) * 100.0),
            "ci95_hi_pp": float(np.percentile(delta_boot, 97.5) * 100.0),
            "n": len(G4_ITEM_IDS),
            "bootstrap_B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "ci_method": "paired_utterance_bootstrap_percentile_ratio_of_sums",
            "cluster_unit": "utterance",
        },
    }


def normalizer_control(primary: dict, basic: dict) -> dict:
    control = {
        "threshold_pp": CONTROL_THRESHOLD_PP,
        "primary_human_rate_pp": primary["human"]["point_pp"],
        "basic_human_rate_pp": basic["human"]["point_pp"],
        "human_abs_difference_pp": abs(
            primary["human"]["point_pp"] - basic["human"]["point_pp"]
        ),
        "primary_synth_rate_pp": primary["synth"]["point_pp"],
        "basic_synth_rate_pp": basic["synth"]["point_pp"],
        "synth_abs_difference_pp": abs(
            primary["synth"]["point_pp"] - basic["synth"]["point_pp"]
        ),
        "primary_matched_delta_pp": primary["delta"]["point_pp"],
        "basic_matched_delta_pp": basic["delta"]["point_pp"],
        "matched_delta_abs_difference_pp": abs(
            primary["delta"]["point_pp"] - basic["delta"]["point_pp"]
        ),
    }
    control["cells"] = {
        "human_error_rate": {
            "abs_difference_pp": control["human_abs_difference_pp"],
            "status": (
                "UNSTABLE"
                if control["human_abs_difference_pp"] > CONTROL_THRESHOLD_PP
                else "STABLE"
            ),
        },
        "synth_error_rate": {
            "abs_difference_pp": control["synth_abs_difference_pp"],
            "status": (
                "UNSTABLE"
                if control["synth_abs_difference_pp"] > CONTROL_THRESHOLD_PP
                else "STABLE"
            ),
        },
        "matched_human_delta": {
            "abs_difference_pp": control[
                "matched_delta_abs_difference_pp"
            ],
            "status": (
                "UNSTABLE"
                if control["matched_delta_abs_difference_pp"]
                > CONTROL_THRESHOLD_PP
                else "STABLE"
            ),
        },
    }
    control["status"] = (
        "UNSTABLE"
        if any(cell["status"] == "UNSTABLE" for cell in control["cells"].values())
        else "STABLE"
    )
    control["ranking_inadmissible"] = True
    control["ranking_inadmissible_reasons"] = [
        CLASSIFICATION,
        TH13_STATUS,
    ]
    if control["status"] == "UNSTABLE":
        control["ranking_inadmissible_reasons"].append(
            "NORMALIZER_CONTROL_UNSTABLE"
        )
    return control


def load_frozen_normalizer():
    if str(D1_SCORING_DIR) not in sys.path:
        sys.path.insert(0, str(D1_SCORING_DIR))
    import normalization

    if normalization.source_sha256() != NORMALIZER_EXPECTED_SHA256:
        raise SystemExit("frozen normalizer import hash mismatch")
    receipt = normalization.validate_golden_fixtures(D1_FIXTURES_PATH)
    if receipt.get("status") != "PASS" or receipt.get("fixture_count") != 40:
        raise SystemExit("frozen normalizer fixtures failed")
    return normalization, receipt


def render_markdown(payload: dict) -> str:
    lines = [
        "# G4 — MOSS-TTS-Nano other-language reduced-n screen",
        "",
        "## Verdict",
        "",
        (
            "This is an **EXPLORATORY_REDUCED_N** matched-human ASR backcheck "
            "with planned n=10 per language. Only complete ten-item arms are "
            "scored. Every row is ranking-inadmissible and "
            "TH-13 is **INSUFFICIENT_N_10_LT_60**. No transcript, failed clip, "
            "or metric cell was imputed."
        ),
        "",
        (
            "EN, ES, and FR use the exact cached faster-whisper-tiny route from "
            "D1. RU uses only the existing GigaAM-v2 CTC INT8 route; Russian "
            "was never sent through Whisper."
        ),
        "",
        (
            "Scored complete arms: "
            + (", ".join(payload["headline"]["scored_languages"]) or "none")
            + ". Unscored incomplete arms: "
            + (
                ", ".join(
                    payload["headline"]["unscored_incomplete_languages"]
                )
                or "none"
            )
            + "."
        ),
        "",
        "## Matched-human deltas",
        "",
        (
            "| Language | human WER % | MOSS WER % | primary delta pp "
            "[95% paired CI] | waveform + cap safe | +5 pp screen | "
            "coverage verdict | basic-control delta pp | normalizer control |"
        ),
        "|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for language in LANGUAGES:
        comparison = payload["comparisons"][language]
        result = payload["languages"][language]
        if comparison["status"] != "SCORED_COMPLETE_10_OF_10":
            safety = (
                "PASS"
                if result["waveform_and_cap_safety"][
                    "waveform_and_cap_safe"
                ]
                else "FAIL"
            )
            lines.append(
                "| {lang} | — | — | UNSCORED ({audio}/10 WAV, "
                "{asr}/10 ASR) | {safety} | NOT RUN | FAIL | — | "
                "NOT RUN |".format(
                    lang=language.upper(),
                    audio=comparison["available_audio_n"],
                    asr=comparison["available_asr_n"],
                    safety=safety,
                )
            )
            continue
        primary = comparison["primary"]
        basic = comparison["basic_control"]
        control = comparison["normalizer_control"]
        lines.append(
            "| {lang} | {human:.3f} | {synth:.3f} | "
            "{delta:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
            "{safety} | {point_screen} | {coverage} | "
            "{basic_delta:+.3f} | {control} |".format(
                lang=language.upper(),
                human=primary["human"]["point_pp"],
                synth=primary["synth"]["point_pp"],
                delta=primary["delta"]["point_pp"],
                lo=primary["delta"]["ci95_lo_pp"],
                hi=primary["delta"]["ci95_hi_pp"],
                safety=(
                    "PASS"
                    if result["waveform_and_cap_safety"][
                        "waveform_and_cap_safe"
                    ]
                    else "FAIL"
                ),
                point_screen=(
                    "PASS"
                    if result["reduced_n_point_screen"]["passed"]
                    else "FAIL"
                ),
                coverage=result["coverage_verdict"],
                basic_delta=basic["delta"]["point_pp"],
                control=control["status"],
            )
        )
    lines.extend(["", "## Named synthesis failures", ""])
    named_failures = payload["synthesis_receipt"].get("named_failures", [])
    if named_failures:
        for failure in named_failures:
            lines.append(
                "- {language} `{item_id}` / `{case_id}`: **{failure_type}**"
                " — {failure_message}; max_new_frames_hit={cap_hit}; "
                "output_retained={output_retained}.".format(
                    language=failure["language"].upper(),
                    item_id=failure["item_id"],
                    case_id=failure["case_id"],
                    failure_type=failure["failure_type"],
                    failure_message=failure.get("failure_message")
                    or "no message recorded",
                    cap_hit=(
                        "unknown"
                        if failure["max_new_frames_hit"] is None
                        else str(failure["max_new_frames_hit"]).lower()
                    ),
                    output_retained=str(failure["output_retained"]).lower(),
                )
            )
    else:
        lines.append("- None.")
    for language in LANGUAGES:
        comparison = payload["comparisons"][language]
        if comparison["status"] == "SCORED_COMPLETE_10_OF_10":
            continue
        lines.append(
            "- {language} arm unscored with no imputation. Missing WAV IDs: "
            "{wav}; missing ASR IDs: {asr}.".format(
                language=language.upper(),
                wav=", ".join(comparison["missing_audio_item_ids"]) or "none",
                asr=", ".join(comparison["missing_asr_item_ids"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "Delta is synthesized error rate minus the same-ID matched-human "
            "error rate. Point estimates are ratio-of-sums, never means of "
            "utterance WER. Confidence intervals use the paired utterance "
            "bootstrap with B=10,000 and seed 20260801.",
            "",
            (
                "The compatibility `PASS` screen requires all ten clips in a "
                "language to pass both raw/file waveform gates, none to hit "
                "the generation-frame cap, and the primary matched-human "
                "point delta to be <= +5 pp. A PASS is still exploratory, "
                "ranking-inadmissible, and insufficient for TH-13."
            ),
            "",
            "## Protocol controls",
            "",
            (
                f"- Fixed IDs: {', '.join(G4_ITEM_IDS)}. No other D1 item was "
                "read or scored."
            ),
            (
                "- Frozen normalizer SHA-256 "
                f"{NORMALIZER_EXPECTED_SHA256}; all 40 golden fixtures passed."
            ),
            (
                "- Synthesis runner provenance is split. The 28 rows from ES "
                "`fleurs_04` onward bind to SHA-256 "
                f"{SYNTH_RUNNER_EXECUTED_SHA256}, "
                f"{SYNTH_RUNNER_EXECUTED_BYTES:,} bytes [HOST], cutoff "
                f"{SYNTH_RUNNER_BINDING_CUTOFF_UTC}. EN 10 plus ES "
                "`fleurs_02` and `fleurs_03` remain execution-runner UNKNOWN. "
                "Later worktree edits are not claimed to have executed for "
                "any row."
            ),
            (
                "- The suite receipt was later rewritten in checkpoint-"
                "reuse/migration mode: all 40 rows now carry case-config "
                "fingerprints and are marked migrated from legacy. Those "
                "fields are compatibility assertions, not waveform-execution "
                "evidence, and do not upgrade the 12 UNKNOWN runner rows."
            ),
            (
                "- Primary and BasicTextNormalizer controls are both reported. "
                "A control difference above 2 pp is marked unstable."
            ),
            (
                "- Classification is EXPLORATORY_REDUCED_N, "
                "ranking_inadmissible=true, and TH13="
                "INSUFFICIENT_N_10_LT_60 on every language."
            ),
            (
                "- No imputation: every scored language pairs 10 human and "
                "10 MOSS transcripts by exact item ID; an incomplete arm is "
                "left wholly unscored."
            ),
            (
                "- The configured 300 s case timeout uses Python SIGALRM as a "
                "best-effort interrupt. It is not a process-level watchdog, "
                "and a native inference call can surface the timeout late; "
                "300 s must not be read as a hard per-case wall-clock bound."
            ),
            "",
            "## Limits",
            "",
            (
                "The n=10 screen can expose a gross multilingual failure but "
                "cannot clear TH-13, establish ranking, or replace the planned "
                "n>=60 dual-reference screen. It does not measure naturalness, "
                "speaker similarity, wrong-language rate, phone behavior, or "
                "commercial rights."
            ),
            "",
            (
                "Full ASR records and utterance-level S/I/D counts are under "
                "artifacts/moss/results. Audio and caches remain git-ignored."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def score() -> int:
    plan, selected_corpus, human_records = build_plan()
    require_synthesis_complete(plan)
    asr_payload = write_asr_result(plan)
    synth_records = {
        (row["language"], row["item_id"]): row
        for row in asr_payload["records"]
    }
    normalization, fixture_receipt = load_frozen_normalizer()
    comparisons = {}
    utterance_scores = []
    summary_rows = []
    for language in LANGUAGES:
        present_audio_ids = {
            row["item_id"]
            for row in plan["audio_inventory"]["records"]
            if row["language"] == language and row["status"] == "present"
        }
        present_asr_ids = {
            item_id
            for record_language, item_id in synth_records
            if record_language == language
        }
        missing_audio_ids = [
            item_id for item_id in G4_ITEM_IDS if item_id not in present_audio_ids
        ]
        missing_asr_ids = [
            item_id for item_id in G4_ITEM_IDS if item_id not in present_asr_ids
        ]
        incomplete_ids = [
            item_id
            for item_id in G4_ITEM_IDS
            if item_id in set(missing_audio_ids) | set(missing_asr_ids)
        ]
        complete_for_scoring = bool(
            not missing_audio_ids
            and not missing_asr_ids
            and len(present_audio_ids) == 10
            and len(present_asr_ids) == 10
        )
        if not complete_for_scoring:
            comparison = {
                "language": language,
                "metric": "WER",
                "status": "UNSCORED_INCOMPLETE_ARM",
                "classification": CLASSIFICATION,
                "n": 0,
                "expected_n": 10,
                "available_audio_n": len(present_audio_ids),
                "available_asr_n": len(present_asr_ids),
                "missing_audio_item_ids": missing_audio_ids,
                "missing_asr_item_ids": missing_asr_ids,
                "primary": None,
                "basic_control": None,
                "normalizer_control": None,
                "ranking_inadmissible": True,
                "th13": TH13_STATUS,
                "imputation": {
                    "performed": False,
                    "imputed_item_count": 0,
                    "missing_item_count": len(incomplete_ids),
                    "status": "NO_IMPUTATION_ARM_LEFT_UNSCORED",
                },
            }
            comparisons[language] = comparison
            summary_rows.append(
                {
                    "language": language,
                    "metric": "WER",
                    "status": "UNSCORED_INCOMPLETE_ARM",
                    "n": 0,
                    "expected_n": 10,
                    "available_audio_n": len(present_audio_ids),
                    "available_asr_n": len(present_asr_ids),
                    "human_rate_pp": None,
                    "synth_rate_pp": None,
                    "matched_delta_pp": None,
                    "matched_delta_ci95_lo_pp": None,
                    "matched_delta_ci95_hi_pp": None,
                    "basic_matched_delta_pp": None,
                    "normalizer_control_status": "NOT_RUN_INCOMPLETE_ARM",
                    "classification": CLASSIFICATION,
                    "ranking_inadmissible": True,
                    "th13": TH13_STATUS,
                    "imputation_performed": False,
                }
            )
            continue
        variant_results = {}
        for variant in ("primary", "basic"):
            human_scored = []
            synth_scored = []
            for item_id in G4_ITEM_IDS:
                expected_reference = selected_corpus[item_id]["texts"][language]
                human = human_records[(language, item_id)]
                synth = synth_records[(language, item_id)]
                if human.get("reference_text") != expected_reference:
                    raise SystemExit(
                        f"human reference mismatch: {language}/{item_id}"
                    )
                if synth.get("reference_text") != expected_reference:
                    raise SystemExit(
                        f"synth reference mismatch: {language}/{item_id}"
                    )
                human_row = score_record(
                    normalization=normalization,
                    language=language,
                    item_id=item_id,
                    variant=variant,
                    record=human,
                    arm="matched_human",
                )
                synth_row = score_record(
                    normalization=normalization,
                    language=language,
                    item_id=item_id,
                    variant=variant,
                    record=synth,
                    arm="moss_tts_nano_100m",
                )
                if human_row["reference_tokens"] != synth_row["reference_tokens"]:
                    raise AssertionError("paired reference token mismatch")
                human_scored.append(human_row)
                synth_scored.append(synth_row)
                utterance_scores.extend((human_row, synth_row))
            variant_results[variant] = paired_summary(
                human_scored, synth_scored
            )
        primary = variant_results["primary"]
        basic = variant_results["basic"]
        control = normalizer_control(primary, basic)
        comparison = {
            "language": language,
            "metric": "WER",
            "status": "SCORED_COMPLETE_10_OF_10",
            "classification": CLASSIFICATION,
            "n": 10,
            "fixed_item_ids": list(G4_ITEM_IDS),
            "primary": primary,
            "basic_control": basic,
            "normalizer_control": control,
            "ranking_inadmissible": True,
            "th13": TH13_STATUS,
            "imputation": {
                "performed": False,
                "imputed_item_count": 0,
                "missing_item_count": 0,
                "status": "NO_IMPUTATION_COMPLETE_10_OF_10",
            },
        }
        comparisons[language] = comparison
        summary_rows.append(
            {
                "language": language,
                "metric": "WER",
                "n": 10,
                "human_rate_pp": primary["human"]["point_pp"],
                "synth_rate_pp": primary["synth"]["point_pp"],
                "matched_delta_pp": primary["delta"]["point_pp"],
                "matched_delta_ci95_lo_pp": primary["delta"]["ci95_lo_pp"],
                "matched_delta_ci95_hi_pp": primary["delta"]["ci95_hi_pp"],
                "basic_matched_delta_pp": basic["delta"]["point_pp"],
                "normalizer_control_status": control["status"],
                "classification": CLASSIFICATION,
                "ranking_inadmissible": True,
                "th13": TH13_STATUS,
                "imputation_performed": False,
            }
        )
    language_results = {}
    language_coverage = {}
    for language in LANGUAGES:
        comparison = comparisons[language]
        synthesis_safety = plan["synthesis_receipt"]["by_language"][language]
        scored_complete_arm = (
            comparison["status"] == "SCORED_COMPLETE_10_OF_10"
        )
        delta_pp = (
            comparison["primary"]["delta"]["point_pp"]
            if scored_complete_arm
            else None
        )
        point_screen_passed = bool(
            scored_complete_arm
            and delta_pp is not None
            and np.isfinite(delta_pp)
            and delta_pp <= REDUCED_N_POINT_SCREEN_THRESHOLD_PP
        )
        waveform_and_cap_safe = bool(
            synthesis_safety["waveform_and_cap_safe"]
        )
        coverage_verdict = (
            "PASS"
            if scored_complete_arm
            and waveform_and_cap_safe
            and point_screen_passed
            else "FAIL"
        )
        language_coverage[language] = coverage_verdict
        language_results[language] = {
            "language": language,
            "coverage_verdict": coverage_verdict,
            "waveform_and_cap_safety": synthesis_safety,
            "reduced_n_point_screen": {
                "metric": "primary_matched_human_delta_pp",
                "operator": "<=",
                "threshold_pp": REDUCED_N_POINT_SCREEN_THRESHOLD_PP,
                "observed_pp": delta_pp,
                "passed": point_screen_passed,
                "status": (
                    "RUN_COMPLETE_10_OF_10"
                    if scored_complete_arm
                    else "NOT_RUN_INCOMPLETE_ARM"
                ),
            },
            "comparison": comparison,
            "scoring_status": comparison["status"],
            "classification": CLASSIFICATION,
            "reduced_n_exploratory": True,
            "ranking_inadmissible": True,
            "th13": TH13_STATUS,
            "no_imputation": True,
        }
    overall_other_four = (
        "PASS"
        if all(value == "PASS" for value in language_coverage.values())
        else "FAIL"
    )
    final_language_coverage = {
        **language_coverage,
        "zh": "UNRESOLVED",
    }
    scored_languages = [
        language
        for language in LANGUAGES
        if comparisons[language]["status"] == "SCORED_COMPLETE_10_OF_10"
    ]
    unscored_languages = [
        language for language in LANGUAGES if language not in scored_languages
    ]
    report_status = (
        "COMPLETE_EXPLORATORY_REDUCED_N"
        if not unscored_languages
        else "COMPLETE_PARTIAL_EXPLORATORY_REDUCED_N"
    )
    payload = {
        "schema_version": 1,
        "sprint_id": "G4",
        "lane": "G",
        "task": "MOSS-TTS-Nano EN/ES/FR/RU reduced-n matched-human ASR backcheck",
        "status": report_status,
        "evidence_class": "HOST",
        "classification": CLASSIFICATION,
        "ranking_inadmissible": True,
        "ranking_inadmissible_reasons": [
            CLASSIFICATION,
            TH13_STATUS,
        ],
        "th13": TH13_STATUS,
        "no_imputation": True,
        "headline": {
            "language_count": 4,
            "scored_language_count": len(scored_languages),
            "scored_languages": scored_languages,
            "unscored_incomplete_languages": unscored_languages,
            "n_per_language": 10,
            "total_synth_clips": 40,
            "total_synth_clips_present": plan["audio_inventory"][
                "present_clip_count"
            ],
            "all_same_id_pairs_complete": not unscored_languages,
            "ranking_inadmissible": True,
            "th13": TH13_STATUS,
            "classification": CLASSIFICATION,
        },
        "final_verdict": {
            "language_coverage": final_language_coverage,
            "overall_other_four": overall_other_four,
            "reduced_n_exploratory": True,
            "single_engine_safe_all_five": False,
            "ranking_inadmissible": True,
            "th13": TH13_STATUS,
            "screen_definition": {
                "waveform_and_cap_safety_required": True,
                "primary_matched_human_delta_threshold_pp": (
                    REDUCED_N_POINT_SCREEN_THRESHOLD_PP
                ),
                "primary_matched_human_delta_operator": "<=",
                "pass_interpretation": (
                    "Exploratory reduced-n technical screen only; never a "
                    "ranking, TH-13 clearance, ship decision, or all-five "
                    "safety claim."
                ),
            },
        },
        "fixed_item_ids": list(G4_ITEM_IDS),
        "corpus": plan["corpus"],
        "audio_inventory": plan["audio_inventory"],
        "synthesis_receipt": plan["synthesis_receipt"],
        "synthesis_runner_provenance": plan[
            "synthesis_runner_provenance"
        ],
        "reference_asr": plan["reference_asr"],
        "asr_transcript_artifact": {
            "path": relative(ASR_RESULT_PATH),
            "bytes": ASR_RESULT_PATH.stat().st_size,
            "sha256": sha256_file(ASR_RESULT_PATH),
            "record_count": len(asr_payload["records"]),
        },
        "normalization": {
            "source_path": relative(D1_NORMALIZER_PATH),
            "source_sha256": normalization.source_sha256(),
            "package_versions": normalization.package_versions(),
            "golden_fixtures": fixture_receipt,
            "variants": ["primary", "basic"],
            "basic_control_threshold_pp": CONTROL_THRESHOLD_PP,
        },
        "statistics": plan["statistics"],
        "languages": language_results,
        "comparisons": comparisons,
        "summary_rows": summary_rows,
        "utterance_scores": utterance_scores,
        "limitations": [
            "Exploratory n=10 per language; TH-13 requires n>=60.",
            "Single reference ASR per language family; no dual-reference screen.",
            "No ranking or ship decision may be inferred from these rows.",
            "No naturalness, speaker-similarity, wrong-language, phone, or panel measurement.",
            (
                "Incomplete language arms were left wholly unscored; no "
                "partial-arm metric or imputation was produced."
            ),
            (
                "The Python SIGALRM timeout is best-effort rather than a "
                "process-level native-call watchdog; a blocked native call "
                "can exceed the nominal 300-second setting before the "
                "TimeoutError is delivered."
            ),
        ],
        "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_write_json(SCORING_RESULT_PATH, payload)
    atomic_write_json(G4_JSON_PATH, payload)
    atomic_write_text(G4_MD_PATH, render_markdown(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "G4_json": str(G4_JSON_PATH),
                "G4_md": str(G4_MD_PATH),
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def dry_run() -> int:
    plan, _, _ = build_plan()
    atomic_write_json(DRY_RUN_PATH, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "classification": plan["classification"],
                "ranking_inadmissible": plan["ranking_inadmissible"],
                "th13": plan["th13"],
                "fixed_item_ids": plan["fixed_item_ids"],
                "audio_present": plan["audio_inventory"]["present_clip_count"],
                "audio_expected": plan["audio_inventory"]["expected_clip_count"],
                "synthesis_receipt_status": plan["synthesis_receipt"]["status"],
                "missing": plan["audio_inventory"]["missing"],
                "unexpected_wavs": plan["audio_inventory"]["unexpected_wavs"],
                "human_transcript_count": plan["reference_asr"][
                    "human_transcript_count"
                ],
                "commands": plan["commands"],
                "dry_run_artifact": relative(DRY_RUN_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def orchestrate(*, force: bool) -> int:
    plan, _, _ = build_plan()
    require_synthesis_complete(plan)
    commands = [
        [str(MAIN_PYTHON), str(Path(__file__).resolve()), "transcribe-whisper"],
        [str(SHERPA_PYTHON), str(Path(__file__).resolve()), "transcribe-ru"],
        [str(MAIN_PYTHON), str(Path(__file__).resolve()), "score"],
    ]
    if force:
        commands[0].append("--force")
        commands[1].append("--force")
    for command in commands:
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, cwd=OUT_D, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")

    whisper = subparsers.add_parser("transcribe-whisper")
    whisper.add_argument("--force", action="store_true")
    whisper.add_argument("--max-new-clips", type=int)

    russian = subparsers.add_parser("transcribe-ru")
    russian.add_argument("--force", action="store_true")
    russian.add_argument("--max-new-clips", type=int)

    subparsers.add_parser("score")

    run = subparsers.add_parser("run")
    run.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "dry-run":
        return dry_run()
    if args.command == "transcribe-whisper":
        return transcribe_whisper(
            force=args.force, max_new_clips=args.max_new_clips
        )
    if args.command == "transcribe-ru":
        return transcribe_ru(force=args.force, max_new_clips=args.max_new_clips)
    if args.command == "score":
        return score()
    if args.command == "run":
        return orchestrate(force=args.force)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
