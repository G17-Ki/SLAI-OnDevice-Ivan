#!/usr/bin/env python3
"""Build Lane G's additive listening, route-table, and summary deliverables.

The default invocation is deliberately read-only.  Pass ``--apply`` only after
G1--G4 are final and the retained MOSS synthesis artifacts are complete.
Historical Lane D reports and route rows are treated as immutable inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


OUT_D = Path(__file__).resolve().parent
LISTENING_DIR = OUT_D / "artifacts" / "listening"
MOSS_DIR = OUT_D / "artifacts" / "moss"
MIXED_DIR = MOSS_DIR / "audio" / "g2_mixed"
G5_SYNTHESIS_PATH = MOSS_DIR / "results" / "g5_synthesis.json"
D5_PROMPTS_PATH = OUT_D / "D5_prompts.json"
D5_REPORT_PATH = OUT_D / "D5.json"
MAX_BYTES = 3_000_000_000
G4_G5_EXECUTION_RUNNER_SHA256 = (
    "9b3a853ad70c57ef2cc7ef00d2ce1d277e7707f694c5834c5463babdacbf05f2"
)
G4_G5_EXECUTION_RUNNER_BYTES = 27_051
G4_G5_PROCESS_STARTED_UTC = "2026-08-02T09:20:34Z"
G4_G5_CASE_TIMEOUT_S = 300
D5_PROMPTS_SHA256 = (
    "5bfb957efc25482f6b8d9c8f90e5af8e55b99d11fc982c83815c5ea2fd782b20"
)
D5_LISTENING_WAV_COUNT = 278
D5_LISTENING_MANIFEST_SHA256 = (
    "dac26b1fb522c6c730435054c33be122f56bcf78b2b0813464cab4b193dcaed6"
)
MOSS_SAMPLING = {
    "do_sample": True,
    "text_temperature": 1.0,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 0.8,
    "audio_top_p": 0.95,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.2,
    "use_kv_cache": True,
}

LANGUAGE_ORDER = ("en", "zh", "es", "fr", "ru")
MIXED_PAIR_ORDER = ("wifi", "5g", "digits", "cpu", "api_json", "vpn")
MIXED_PAIR_TEXT = {
    "wifi": ("WiFi密码", "密码", "Is ‘WiFi’ actually spoken?"),
    "5g": ("5G网络", "网络", "Is ‘5G’ actually spoken?"),
    "digits": ("订单号是5072", "订单号是", "Are all four digits spoken?"),
    "cpu": ("CPU温度达到82度", "温度达到82度", "Are ‘CPU’ and 82 both spoken?"),
    "api_json": ("API返回24条JSON记录", "返回24条记录", "Are both ‘API’ and ‘JSON’ spoken?"),
    "vpn": ("VPN每10分钟断开", "每10分钟断开", "Is ‘VPN’ actually spoken?"),
}

RMS_FLOOR = 0.0031622776601683794
CLIP_THRESHOLD = 0.9885530946569389
MAX_CLIP_RUN_S = 0.005
MIN_DURATION_S = 0.15
MAX_DURATION_S = 30.0

ALLOWED_EVIDENCE_CLASSES = {
    "EXACT",
    "UB",
    "VENDOR",
    "BENCHMARK",
    "HOST",
    "AIHUB-PROFILED",
    "PRIOR",
    "PROJECTED",
    "UNKNOWN",
}

# Canonical JSON SHA-256 of every pre-G non-MOSS route row in tts_table.json.
LANE_D_ROUTE_ROWS_SHA256 = (
    "128b85645f76ec9309e76f0561635aac369753f85f66dfed0185922e8de79af2"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_logical_bytes(root: Path) -> int:
    """Count stored logical bytes without following venv/interpreter symlinks."""
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            total += path.lstat().st_size
        elif path.is_file():
            total += path.stat().st_size
    return total


def stored_file_bytes(path: Path) -> int:
    """Return this directory entry's stored bytes without dereferencing links."""
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    return 0


def stable_seed(suite: str, case_id: str, repeat: int = 1) -> int:
    payload = f"lane-g-moss-v1:{suite}:{case_id}:{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def clip_manifest_sha256(paths: list[Path]) -> str:
    manifest = "".join(
        f"{path.name}\t{sha256_file(path)}\t{path.stat().st_size}\n"
        for path in sorted(paths, key=lambda item: item.name)
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_evidence_classes(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("evidence_class"):
                assert child in ALLOWED_EVIDENCE_CLASSES, (key, child)
            elif key == "evidence_classes":
                assert isinstance(child, list) and child
                assert set(child).issubset(ALLOWED_EVIDENCE_CLASSES), child
            assert_evidence_classes(child)
    elif isinstance(value, list):
        for child in value:
            assert_evidence_classes(child)


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask.reshape(-1):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def inspect_wav(path: Path) -> dict[str, Any]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    finite = bool(values.size and np.isfinite(values).all())
    duration_s = float(values.shape[0] / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(values * values))) if finite else None
    peak = float(np.max(np.abs(values))) if finite else None
    longest_clip_samples = 0
    if finite:
        for channel in range(values.shape[1]):
            longest_clip_samples = max(
                longest_clip_samples,
                longest_true_run(np.abs(values[:, channel]) >= CLIP_THRESHOLD),
            )
    longest_clip_s = float(longest_clip_samples / sample_rate) if sample_rate else 0.0
    assertions = {
        "nonempty": bool(values.size and values.shape[0] > 0),
        "finite": finite,
        "rms_above_floor": bool(rms is not None and rms > RMS_FLOOR),
        "duration_in_range": MIN_DURATION_S <= duration_s <= MAX_DURATION_S,
        "no_sustained_clipping": longest_clip_s <= MAX_CLIP_RUN_S,
    }
    info = sf.info(path)
    return {
        "filename": path.name,
        "relative_path": str(path.relative_to(OUT_D)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate_hz": int(sample_rate),
        "channels": int(values.shape[1]),
        "frames": int(values.shape[0]),
        "duration_s": duration_s,
        "rms": rms,
        "peak_abs": peak,
        "longest_contiguous_clip_run_s": longest_clip_s,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }


def expected_listening_names() -> list[str]:
    matrix = {
        "en": ("kokoro", "piper", "supertonic"),
        "zh": ("kokoro", "matcha", "moss"),
        "es": ("kokoro", "piper", "supertonic"),
        "fr": ("kokoro", "piper", "supertonic"),
        "ru": ("piper", "silero", "supertonic"),
    }
    return [
        f"{language}_{engine}_{index:02d}.wav"
        for language in LANGUAGE_ORDER
        for engine in matrix[language]
        for index in range(1, 21)
    ]


def parse_listening_filename(filename: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"(en|zh|es|fr|ru)_([a-z0-9_]+)_(\d{2})\.wav", filename)
    assert match, filename
    return match.group(1), match.group(2), int(match.group(3))


def scan_listening_bundle() -> dict[str, Any]:
    expected = expected_listening_names()
    expected_set = set(expected)
    paths = sorted(LISTENING_DIR.glob("*.wav"), key=lambda path: path.name)
    present_set = {path.name for path in paths}
    d5 = load_json(D5_REPORT_PATH)
    assert sha256_file(D5_PROMPTS_PATH) == D5_PROMPTS_SHA256
    assert d5["prompt_manifest"]["file_sha256"] == D5_PROMPTS_SHA256
    assert d5["inventory"]["existing_wav_count"] == D5_LISTENING_WAV_COUNT
    assert (
        d5["inventory"]["clip_hash_manifest_sha256"]
        == D5_LISTENING_MANIFEST_SHA256
    )
    lane_d_paths = [
        path
        for path in paths
        if re.fullmatch(r"zh_moss_\d{2}\.wav", path.name) is None
    ]
    lane_d_manifest = clip_manifest_sha256(lane_d_paths)
    assert len(lane_d_paths) == D5_LISTENING_WAV_COUNT, len(lane_d_paths)
    assert lane_d_manifest == D5_LISTENING_MANIFEST_SHA256, lane_d_manifest
    clips = []
    format_counter: Counter[tuple[str, str, int, int]] = Counter()
    for path in paths:
        record = inspect_wav(path)
        language, engine, index = parse_listening_filename(path.name)
        record.update(
            {"language": language, "engine": engine, "listen_index": index}
        )
        clips.append(record)
        format_counter[
            (
                record["format"],
                record["subtype"],
                record["sample_rate_hz"],
                record["channels"],
            )
        ] += 1
    formats = [
        {
            "format": key[0],
            "subtype": key[1],
            "sample_rate_hz": key[2],
            "channels": key[3],
            "count": count,
        }
        for key, count in sorted(format_counter.items())
    ]
    missing = sorted(expected_set - present_set)
    extra = sorted(present_set - expected_set)
    return {
        "expected_wav_count": len(expected),
        "existing_wav_count": len(paths),
        "missing_expected_count": len(missing),
        "extra_unexpected_count": len(extra),
        "missing_expected_filenames": missing,
        "extra_unexpected_filenames": extra,
        "total_wav_bytes": sum(path.stat().st_size for path in paths),
        "clip_hash_manifest_sha256": clip_manifest_sha256(paths),
        "clip_hash_manifest_definition": (
            "SHA-256 of lexicographically sorted UTF-8 lines: "
            "filename<TAB>file_sha256<TAB>bytes<LF>"
        ),
        "lane_d_frozen_identity": {
            "existing_non_moss_wav_count": len(lane_d_paths),
            "expected_non_moss_wav_count": D5_LISTENING_WAV_COUNT,
            "clip_hash_manifest_sha256": lane_d_manifest,
            "expected_clip_hash_manifest_sha256": D5_LISTENING_MANIFEST_SHA256,
            "preserved": True,
        },
        "all_existing_wavs_readable": True,
        "all_existing_wavs_pass_waveform_gate": all(
            clip["passed"] for clip in clips
        ),
        "formats": formats,
        "clips": clips,
    }


def load_mixed_pairs() -> list[dict[str, Any]]:
    synthesis = load_json(MOSS_DIR / "results" / "g2_mixed_synthesis.json")
    by_case = {row["case_id"]: row for row in synthesis["records"]}
    rows = []
    for pair_id in MIXED_PAIR_ORDER:
        full_text, ablated_text, check = MIXED_PAIR_TEXT[pair_id]
        pair: dict[str, Any] = {
            "pair_id": pair_id,
            "full_text": full_text,
            "ablated_text": ablated_text,
            "listening_check": check,
        }
        for role in ("full", "ablated"):
            case = by_case[f"{pair_id}_{role}"]
            output = case.get("output")
            assert output is not None and case.get("passed"), (pair_id, role)
            path = OUT_D / output["path"]
            inspected = inspect_wav(path)
            assert inspected["passed"]
            assert inspected["sha256"] == output["sha256"]
            pair[role] = {
                "relative_path": inspected["relative_path"],
                "bytes": inspected["bytes"],
                "sha256": inspected["sha256"],
                "duration_s": inspected["duration_s"],
                "max_new_frames_hit": bool(case.get("max_new_frames_hit")),
                "generated_frames": case.get("generated_frames"),
                "max_new_frames": case.get("max_new_frames"),
            }
        pair["automatic_coverage_disposition"] = (
            "UNRESOLVED_FRAME_CAP_HIT_REQUIRES_LISTENING"
            if pair["full"]["max_new_frames_hit"]
            or pair["ablated"]["max_new_frames_hit"]
            else "UNRESOLVED_WAVEFORM_DIFFERENCE_IS_NOT_SPOKEN_SPAN_PROOF"
        )
        rows.append(pair)
    return rows


def lane_d_route_rows(table: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        language: [
            copy.deepcopy(row)
            for row in table["languages"][language]["routes"]
            if row.get("engine_id") != "moss"
        ]
        for language in LANGUAGE_ORDER
    }


def require_enum(value: Any, path: str) -> str:
    assert value in {"PASS", "FAIL", "UNRESOLVED"}, (path, value)
    return str(value)


def route_from_g3(g3: dict[str, Any], language: str) -> dict[str, Any]:
    rows = [row for row in g3.get("routes", []) if row.get("language") == language]
    assert len(rows) == 1, (language, len(rows))
    return rows[0]


def g3_route_stats(g3: dict[str, Any], language: str) -> tuple[dict[str, Any], int, int, int]:
    """Return a compact compatibility view of a canonical G3 route row.

    The evaluator may call the waveform-pass and frame-cap counters either by
    their compact report names or by the synthesis-summary names.  Unknown
    schemas still fail closed.
    """

    route = route_from_g3(g3, language)
    attempted = route.get("attempted")
    passed = route.get("waveform_gate_passed", route.get("passed"))
    cap_hits = route.get(
        "generation_cap_hits", route.get("max_new_frames_hit_count")
    )
    assert isinstance(attempted, int) and attempted >= 0, route
    assert isinstance(passed, int) and passed >= 0, route
    assert isinstance(cap_hits, int) and cap_hits >= 0, route
    return route, attempted, passed, cap_hits


def expected_g5_cases() -> list[dict[str, Any]]:
    prompts = load_json(D5_PROMPTS_PATH)
    rows = prompts["languages"]["zh"]
    assert len(rows) == 20
    cases = []
    for prompt in rows:
        index = int(prompt["listen_index"])
        case_id = f"zh_moss_{index:02d}"
        cases.append(
            {
                "listen_index": index,
                "case_id": case_id,
                "filename": f"{case_id}.wav",
                "text": prompt["text"],
                "seed": stable_seed("g5", case_id),
                "prompt_name": "zh_1.wav",
                "max_new_frames": 375,
                "output_planned_path": f"artifacts/listening/{case_id}.wav",
            }
        )
    cases.sort(key=lambda row: row["listen_index"])
    assert [row["listen_index"] for row in cases] == list(range(1, 21))
    return cases


def expected_moss_model(g1: dict[str, Any]) -> dict[str, Any]:
    tts = g1["primary_pytorch_pins"]["tts"]
    codec = g1["primary_pytorch_pins"]["audio_tokenizer"]
    return {
        "tts_repo": tts["repo"],
        "tts_revision": tts["revision"],
        "tts_weight_sha256": tts["weight_artifact"]["sha256"],
        "tts_parameters": g1["headline"]["tts_unique_learned_parameters"]["value"],
        "codec_repo": codec["repo"],
        "codec_revision": codec["revision"],
        "codec_weight_sha256": codec["weight_artifact"]["sha256"],
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


def failure_label(row: dict[str, Any]) -> str | None:
    if bool(row.get("max_new_frames_hit")):
        return "generation frame cap"
    exception = row.get("exception") or {}
    label = exception.get("type") or row.get("failure_reason")
    if label is None:
        return None
    return str(label).replace("|", "/").replace("\n", " ")


def load_g5_synthesis_receipt(
    *, g1: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    assert G5_SYNTHESIS_PATH.is_file(), G5_SYNTHESIS_PATH
    payload = load_json(G5_SYNTHESIS_PATH)
    assert payload.get("suite") == "g5"
    assert payload.get("evidence_class") == "HOST"
    model = expected_moss_model(g1)
    assert payload.get("model") == model
    assert payload.get("runtime", {}).get("sampling") == MOSS_SAMPLING
    rows = payload.get("records")
    assert isinstance(rows, list)
    expected = expected_g5_cases()
    expected_by_id = {row["case_id"]: row for row in expected}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        assert case_id in expected_by_id, case_id
        assert case_id not in by_id, case_id
        by_id[case_id] = row
    assert set(by_id) == set(expected_by_id)
    assert payload.get("attempted") == len(rows) == 20
    assert payload.get("passed") == sum(row.get("passed") is True for row in rows)
    assert payload.get("failed") == sum(row.get("passed") is not True for row in rows)
    assert payload.get("max_new_frames_hit_count") == sum(
        bool(row.get("max_new_frames_hit")) for row in rows
    )

    clips_by_name = {clip["filename"]: clip for clip in inventory["clips"]}
    prompt_path = MOSS_DIR / "prompts" / "zh_1.wav"
    assert prompt_path.is_file()
    prompt_sha256 = sha256_file(prompt_path)
    compact = []
    for expected_row in expected:
        row = by_id[expected_row["case_id"]]
        filename = expected_row["filename"]
        clip = clips_by_name.get(filename)
        assert row.get("suite") == "g5"
        assert row.get("repeat") == 1
        assert row.get("language") == "zh"
        assert row.get("raw_text") == expected_row["text"]
        assert row.get("prepared_text") == expected_row["text"]
        assert row.get("external_text_normalization") == (
            "DISABLED_EXACT_FROZEN_INPUT"
        )
        assert row.get("seed") == expected_row["seed"]
        assert row.get("sampling") == MOSS_SAMPLING
        assert row.get("prompt_name") == expected_row["prompt_name"]
        assert row.get("max_new_frames") == expected_row["max_new_frames"]
        assert row.get("output_planned_path") == expected_row["output_planned_path"]
        expected_case_config = {
            "generation_protocol_version": 1,
            "tts_revision": model["tts_revision"],
            "tts_weight_sha256": model["tts_weight_sha256"],
            "codec_revision": model["codec_revision"],
            "codec_weight_sha256": model["codec_weight_sha256"],
            "source_revision": model["source_revision"],
            "suite": "g5",
            "case_id": expected_row["case_id"],
            "repeat": 1,
            "seed": expected_row["seed"],
            "language": "zh",
            "raw_text": expected_row["text"],
            "prompt_name": expected_row["prompt_name"],
            "max_new_frames": expected_row["max_new_frames"],
            "output_planned_path": expected_row["output_planned_path"],
            "sampling": MOSS_SAMPLING,
            "external_text_normalization": "DISABLED_EXACT_FROZEN_INPUT",
            "case_timeout_s": G4_G5_CASE_TIMEOUT_S,
        }
        if row.get("case_config") is not None:
            assert row.get("generation_protocol_version") == 1
            assert row.get("case_config") == expected_case_config
            assert row.get("case_config_sha256") == canonical_sha256(
                expected_case_config
            )
        prompt = row.get("prompt") or {}
        assert prompt.get("path") == "artifacts/moss/prompts/zh_1.wav"
        assert prompt.get("sha256") == prompt_sha256
        execution_timestamp = row.get("run_started_utc") or row.get("run_finished_utc")
        assert isinstance(execution_timestamp, str)
        assert execution_timestamp >= G4_G5_PROCESS_STARTED_UTC, (
            expected_row["case_id"],
            execution_timestamp,
        )

        receipt_passed = row.get("passed") is True
        output = row.get("output")
        output_identity_matches = False
        if receipt_passed:
            assert isinstance(output, dict)
            assert clip is not None, filename
            assert row.get("raw_waveform_gate", {}).get("passed") is True
            assert output.get("passed") is True
            output_identity_matches = bool(
                output.get("path") == expected_row["output_planned_path"]
                and output.get("sha256") == clip["sha256"]
                and output.get("bytes") == clip["bytes"]
            )
            assert output_identity_matches, expected_row["case_id"]
            assert clip["passed"], filename
        else:
            assert output is None, expected_row["case_id"]
            assert clip is None, (
                expected_row["case_id"],
                "failed receipt row has a stale listening WAV",
            )

        cap_hit = bool(row.get("max_new_frames_hit"))
        listening_available = receipt_passed and output_identity_matches
        safe = listening_available and not cap_hit
        label = failure_label(row)
        disposition = (
            "RETAINED_RECEIPT_BOUND_SAFE"
            if safe
            else (
                "RETAINED_RECEIPT_BOUND_FRAME_CAP"
                if listening_available
                else "NO_RETAINED_WAV"
            )
        )
        compact.append(
            {
                **expected_row,
                "receipt_passed": receipt_passed,
                "generated_frames": row.get("generated_frames"),
                "max_new_frames_hit": cap_hit,
                "listening_available": listening_available,
                "waveform_safe": listening_available,
                "safe": safe,
                "disposition": disposition,
                "failure_label": label,
                "failure_reason": row.get("failure_reason"),
                "exception": row.get("exception"),
                "run_started_utc": row.get("run_started_utc"),
                "run_finished_utc": row.get("run_finished_utc"),
                "output": (
                    {
                        "relative_path": clip["relative_path"],
                        "bytes": clip["bytes"],
                        "sha256": clip["sha256"],
                        "duration_s": clip["duration_s"],
                    }
                    if listening_available
                    else None
                ),
            }
        )

    bound_names = {
        row["filename"] for row in compact if row["listening_available"]
    }
    present_moss_names = {
        clip["filename"] for clip in inventory["clips"] if clip["engine"] == "moss"
    }
    assert present_moss_names == bound_names, (present_moss_names, bound_names)
    current_runner = OUT_D / "run_moss_g.py"
    return {
        "path": str(G5_SYNTHESIS_PATH.relative_to(OUT_D)),
        "bytes": G5_SYNTHESIS_PATH.stat().st_size,
        "sha256": sha256_file(G5_SYNTHESIS_PATH),
        "suite": "g5",
        "attempted": payload["attempted"],
        "passed": payload["passed"],
        "failed": payload["failed"],
        "max_new_frames_hit_count": payload["max_new_frames_hit_count"],
        "model": copy.deepcopy(payload["model"]),
        "runtime": copy.deepcopy(payload["runtime"]),
        "host": copy.deepcopy(payload["host"]),
        "records": compact,
        "execution_provenance": {
            "executed_runner": {
                "path": "run_moss_g.py",
                "bytes": G4_G5_EXECUTION_RUNNER_BYTES,
                "sha256": G4_G5_EXECUTION_RUNNER_SHA256,
                "evidence_class": "HOST",
                "scope": "ALL_20_G5_RECORDS",
                "process_started_utc": G4_G5_PROCESS_STARTED_UTC,
                "receipt_embeds_runner_identity": False,
                "note": (
                    "The runner hash and byte count were observed immediately after the "
                    "G4/G5 process started. G5 had no pre-existing rows or WAVs, and all "
                    "20 record timestamps are after that process start."
                ),
            },
            "receipt_rewrite": {
                "records_with_case_config_sha256": sum(
                    isinstance(row.get("case_config_sha256"), str) for row in rows
                ),
                "records_marked_checkpoint_migrated_from_legacy": sum(
                    row.get("checkpoint_migrated_from_legacy") is True for row in rows
                ),
                "note": (
                    "The retained suite receipt may have been rewritten later by the "
                    "hardened runner in checkpoint-reuse/migration mode. That rewrite "
                    "does not replace the separately recorded execution-time runner "
                    "identity for waveform generation."
                ),
            },
            "current_worktree_file": {
                "path": "run_moss_g.py",
                "bytes": current_runner.stat().st_size,
                "sha256": sha256_file(current_runner),
            },
            "current_matches_executed": bool(
                current_runner.stat().st_size == G4_G5_EXECUTION_RUNNER_BYTES
                and sha256_file(current_runner) == G4_G5_EXECUTION_RUNNER_SHA256
            ),
        },
    }


def make_g5(
    inventory: dict[str, Any],
    mixed_pairs: list[dict[str, Any]],
    synthesis_receipt: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    expected_moss = [f"zh_moss_{index:02d}.wav" for index in range(1, 21)]
    by_name = {clip["filename"]: clip for clip in inventory["clips"]}
    cases = synthesis_receipt["records"]
    assert [row["filename"] for row in cases] == expected_moss
    moss_present = [row["filename"] for row in cases if row["listening_available"]]
    moss_waveform_safe = [row["filename"] for row in cases if row["waveform_safe"]]
    moss_safe = [row["filename"] for row in cases if row["safe"]]
    moss_missing = [row["filename"] for row in cases if not row["listening_available"]]
    moss_cap_hits = [row["filename"] for row in cases if row["max_new_frames_hit"]]
    moss_failures = [
        {
            "filename": row["filename"],
            "case_id": row["case_id"],
            "failure_label": row["failure_label"] or "no retained WAV",
            "failure_reason": row["failure_reason"],
            "exception": row["exception"],
        }
        for row in cases
        if not row["listening_available"]
    ]
    assert set(moss_present) == {
        name for name in expected_moss if name in by_name
    }
    complete_safe = len(moss_safe) == 20
    known_d_missing = ["zh_kokoro_09.wav", "zh_kokoro_14.wav"]
    assert set(known_d_missing).issubset(inventory["missing_expected_filenames"])
    naturalness = (
        "PENDING_USER_LISTENING"
        if moss_present
        else "UNRESOLVED_NO_RETAINED_G5_CLIPS"
    )
    status = (
        "COMPLETE_PENDING_USER_LISTENING"
        if complete_safe
        else (
            "PARTIAL_PENDING_USER_LISTENING"
            if moss_present
            else "INCOMPLETE_NO_MOSS_LISTENING_CLIPS"
        )
    )
    payload = {
        "schema_version": 1,
        "sprint_id": "G5",
        "lane": "G",
        "status": status,
        "evidence_class": "HOST",
        "final_verdict": {
            "moss_zh_listening_clips_complete_and_safe": complete_safe,
            "moss_zh_listening_clip_count": len(moss_safe),
            "naturalness": naturalness,
            "mixed_script_coverage": "PENDING_USER_LISTENING",
        },
        "filename_contract": "<lang>_<engine>_<idx>.wav with two-digit 01..20 indexes",
        "inventory": {
            key: inventory[key]
            for key in (
                "expected_wav_count",
                "existing_wav_count",
                "missing_expected_count",
                "extra_unexpected_count",
                "missing_expected_filenames",
                "extra_unexpected_filenames",
                "total_wav_bytes",
                "clip_hash_manifest_sha256",
                "clip_hash_manifest_definition",
                "lane_d_frozen_identity",
                "all_existing_wavs_readable",
                "all_existing_wavs_pass_waveform_gate",
            )
        },
        "formats": inventory["formats"],
        "waveform_gate": {
            "finite": "all PCM samples finite",
            "nonempty": "sample_count > 0",
            "minimum_rms_dbfs_exclusive": -50,
            "minimum_rms_linear_exclusive": RMS_FLOOR,
            "duration_seconds_inclusive": [MIN_DURATION_S, MAX_DURATION_S],
            "sustained_clipping_dbfs": -0.1,
            "sustained_clipping_linear": CLIP_THRESHOLD,
            "failure_rule": "fail if a contiguous run above the clipping threshold exceeds 5 ms",
        },
        "known_lane_d_omissions": known_d_missing,
        "synthesis_receipt": synthesis_receipt,
        "moss_zh": {
            "expected": 20,
            "present": len(moss_present),
            "waveform_safe": len(moss_waveform_safe),
            "safe": len(moss_safe),
            "missing": moss_missing,
            "frame_cap_hit": moss_cap_hits,
            "failures": moss_failures,
            "clips": [by_name[name] for name in moss_present],
            "cases": cases,
            "listening_disposition": "USER_JUDGMENT_REQUIRED_NO_PANEL",
        },
        "mixed_script_pairs": mixed_pairs,
        "listening_workflow": {
            "guide": "LISTEN.md",
            "primary_comparison": "Listen to MOSS next to Matcha for each Mandarin row 01..20.",
            "priority_rows": [12, 13, 14],
            "mixed_script_instruction": "For each full/ablated pair, decide whether the Latin acronym or digits are actually spoken; duration or waveform differences alone do not prove coverage.",
        },
        "artifacts": {
            "listening_directory": "artifacts/listening/",
            "mixed_script_directory": "artifacts/moss/audio/g2_mixed/",
            "guide": "LISTEN.md",
            "g5_synthesis_receipt": str(G5_SYNTHESIS_PATH.relative_to(OUT_D)),
        },
    }
    missing_moss_text = (
        "none"
        if not moss_missing
        else ", ".join(f"`{name}`" for name in moss_missing)
    )
    markdown = f"""# G5 — MOSS Mandarin listening bundle

Status: **{payload['status']}**. The Lane G addition contains **{len(moss_present)}/20** receipt-bound MOSS Mandarin WAVs and **{len(moss_safe)}/20** clear both the waveform and generation-cap gates. The complete Lane D + G listening inventory is **{inventory['existing_wav_count']}/{inventory['expected_wav_count']} WAVs** ({inventory['total_wav_bytes']:,} bytes). The frozen Lane D omissions remain `zh_kokoro_09.wav` and `zh_kokoro_14.wav`; missing MOSS G5 WAVs: {missing_moss_text}.

Use [`LISTEN.md`](LISTEN.md) as the working sheet. In the Mandarin table, each retained MOSS clip is immediately beside Matcha at the same index and source text; any failed case is shown as an explicit no-WAV cell rather than a dead link. Start with rows **12–14**, then work through 01–20. A separate six-pair mixed-script section compares each full acronym/digit input with an ablated Han-only control.

The mixed-script decision is intentionally **pending personal listening**. All 12 probe WAVs passed the waveform gate, but a waveform difference does not prove the Latin/digit span was spoken, and four probe outputs hit their generation-frame cap.

- Ordered listening-manifest SHA-256: `{inventory['clip_hash_manifest_sha256']}`
- Frozen Lane D non-MOSS manifest SHA-256: `{inventory['lane_d_frozen_identity']['clip_hash_manifest_sha256']}`
- G5 synthesis receipt SHA-256: `{synthesis_receipt['sha256']}`
- Executed synthesis runner SHA-256: `{G4_G5_EXECUTION_RUNNER_SHA256}` (all 20 G5 rows; receipt itself does not embed runner identity)
- Naturalness: `{naturalness}`
- No listening panel was run; safety and licence evidence cannot be overridden by preference.
"""
    return payload, markdown


def make_moss_route(
    *,
    language: str,
    g1: dict[str, Any],
    g2: dict[str, Any],
    g3: dict[str, Any],
    g4: dict[str, Any],
    g5: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    g2_final = g2["final_verdict"]
    d1_complete = bool(g2_final["d1_complete_and_scored"])
    assert g2["headline"]["d1_attempted"] == 40
    assert g2["headline"]["d1_safe_and_scored"] == 40
    zh_screen = require_enum(
        g2_final["zh_point_estimate_technical_screen"],
        "G2.final_verdict.zh_point_estimate_technical_screen",
    )
    mixed = require_enum(
        g2_final["mixed_script_coverage"],
        "G2.final_verdict.mixed_script_coverage",
    )
    g3_final = g3["final_verdict"]
    g3_by_language = g3_final["by_language"]
    short_safety = require_enum(
        g3_by_language[language], f"G3.final_verdict.by_language.{language}"
    )
    g4_coverage = require_enum(
        g4["final_verdict"]["language_coverage"].get(language, "UNRESOLVED"),
        f"G4.final_verdict.language_coverage.{language}",
    )
    g5_safe = bool(g5["final_verdict"]["moss_zh_listening_clips_complete_and_safe"])
    g3_route, g3_attempted, g3_passed, g3_cap_hits = g3_route_stats(g3, language)
    expected_g3_attempts = 18 if language == "zh" else 6
    assert g3_attempted == expected_g3_attempts
    assert g3_route.get("verdict") == short_safety
    if short_safety == "PASS":
        assert g3_passed == g3_attempted and g3_cap_hits == 0
    elif short_safety == "FAIL":
        assert g3_passed < g3_attempted or g3_cap_hits > 0
    else:
        assert g3_passed == g3_attempted and g3_cap_hits == 0
        warning = g3_route.get("warning") or {}
        assert warning.get("route_clearance_disqualifying") is True

    if language == "zh":
        technical_candidate = (
            d1_complete
            and zh_screen == "PASS"
            and mixed == "PASS"
            and short_safety == "PASS"
            and g5_safe
        )
        unsafe = short_safety == "FAIL" or not g5_safe
        if unsafe:
            technical = (
                g3_route.get("verdict_detail")
                or "UNSAFE_ON_G3_SHORT_INPUT_PROBE"
                if short_safety == "FAIL"
                else "INCOMPLETE_G5_LISTENING_WAVEFORM_ROUTE"
            )
            overall = "REJECT_UNSAFE_DIAGNOSTIC_ONLY"
            bucket = "unsafe"
        elif technical_candidate:
            technical = "SPRINT_CANDIDATE_NOT_GATE_CLEARED"
            overall = "TECHNICAL_CANDIDATE_COMMERCIAL_AND_LISTENING_BLOCKED"
            bucket = "technical"
        elif zh_screen == "FAIL" or mixed == "FAIL":
            technical = "TECHNICAL_SCREEN_FAILED"
            overall = "REJECT_TECHNICAL_SCREEN_AND_COMMERCIAL_BLOCKED"
            bucket = "limited"
        else:
            technical = "EVIDENCE_INCOMPLETE_PENDING_MIXED_SCRIPT_OR_LISTENING"
            overall = "EVIDENCE_INCOMPLETE_AND_COMMERCIAL_BLOCKED"
            bucket = "limited"
    else:
        unsafe = short_safety == "FAIL"
        if unsafe:
            technical = "UNSAFE_ON_G3_SHORT_INPUT_PROBE"
            overall = "REJECT_UNSAFE_DIAGNOSTIC_ONLY"
            bucket = "unsafe"
        elif g4_coverage == "FAIL":
            technical = "REDUCED_N_G4_TECHNICAL_SCREEN_FAILED"
            overall = "REJECT_REDUCED_N_SCREEN_AND_COMMERCIAL_BLOCKED"
            bucket = "limited"
        elif g4_coverage == "PASS":
            technical = "REDUCED_N_G4_EXPLORATORY_PASS_NOT_D1_COMPLETE"
            overall = "EVIDENCE_INCOMPLETE_AND_COMMERCIAL_BLOCKED"
            bucket = "limited"
        else:
            technical = "REDUCED_N_G4_UNRESOLVED_NOT_D1_COMPLETE"
            overall = "EVIDENCE_INCOMPLETE_AND_COMMERCIAL_BLOCKED"
            bucket = "limited"

    licence = g1["licence"]["weight_and_export_licence_scope"]
    assert licence["commercially_unblocked"] is False
    row: dict[str, Any] = {
        "engine_id": "moss",
        "display_name": "MOSS-TTS-Nano-100M + MOSS-Audio-Tokenizer-Nano",
        "model": "OpenMOSS-Team/MOSS-TTS-Nano-100M + MOSS-Audio-Tokenizer-Nano",
        "revision": g1["primary_pytorch_pins"]["tts"]["revision"],
        "codec_revision": g1["primary_pytorch_pins"]["audio_tokenizer"]["revision"],
        "voice_or_style": "upstream built-in zh_1 reference" if language == "zh" else "upstream built-in en_4 reference",
        "model_license": "HF metadata Apache-2.0; required root licence absent from pinned weight/export snapshots",
        "license_disposition": licence["disposition"],
        "technical": technical,
        "commercial": "BLOCKED_PINNED_WEIGHT_REDISTRIBUTION_SCOPE_UNLICENSED",
        "overall": overall,
        "g1": {
            "tts_parameters": g1["headline"]["tts_unique_learned_parameters"],
            "codec_parameters": g1["headline"]["audio_tokenizer_unique_learned_parameters"],
            "combined_parameters": g1["headline"]["pair_unique_learned_parameters"],
            "onnx_tied_weight_trap": g1["headline"]["onnx_tied_weight_trap"],
        },
        "g3": copy.deepcopy(g3_route),
        "commercial_evidence": {
            "hf_metadata": g1["licence"]["hf_card_metadata"],
            "snapshot_license_file_present": g1["licence"]["immutable_snapshot_manifest_check"]["license_file_present_in_any_snapshot"],
            "upstream_card_instruction": g1["licence"]["tts_card_explicit_text"],
        },
        "notes": [
            "All retained Lane G synthesis used local CPU and is labelled HOST.",
            "Sampling and an upstream built-in reference voice were used; no greedy/reference-audio comparison is inferred.",
            "Apache-2.0 HF metadata does not override the pinned README's explicit no-redistribution-until-root-LICENSE instruction.",
        ],
    }
    if language == "zh":
        row["g2"] = {
            "status": g2.get("status"),
            "headline": copy.deepcopy(g2.get("headline")),
            "final_verdict": copy.deepcopy(g2_final),
        }
        row["g5"] = {
            "expected": 20,
            "present": g5["moss_zh"]["present"],
            "safe": g5["moss_zh"]["safe"],
            "missing": g5["moss_zh"]["missing"],
            "listening_disposition": "PENDING_USER_LISTENING",
        }
        row["notes"].append(
            "Mixed-script coverage remains governed only by G2's explicit verdict; full/ablated waveform differences are not proof that Latin spans were spoken."
        )
    else:
        language_result = copy.deepcopy(g4.get("languages", {}).get(language, {}))
        row["g4"] = {
            "coverage_verdict": g4_coverage,
            "result": language_result,
            "scope": "REDUCED_N_EXPLORATORY",
        }
        row["g5"] = {"status": "NO_G5_LISTENING_CLIPS_FOR_THIS_LANGUAGE"}
    return row, bucket


def update_table(
    base: dict[str, Any],
    *,
    g1: dict[str, Any],
    g2: dict[str, Any],
    g3: dict[str, Any],
    g4: dict[str, Any],
    g5: dict[str, Any],
) -> dict[str, Any]:
    table = copy.deepcopy(base)
    historical = lane_d_route_rows(table)
    assert canonical_sha256(historical) == LANE_D_ROUTE_ROWS_SHA256
    table["task"] = "Lane D + G five-language TTS route table"
    table["status"] = "NO_SINGLE_SAFE_COMMERCIALLY_UNBLOCKED_FIVE_LANGUAGE_ENGINE"
    table["evidence_classes"] = sorted(
        set(table.get("evidence_classes", [])) | {"VENDOR"}
    )

    moss_buckets: dict[str, str] = {}
    for language in LANGUAGE_ORDER:
        language_row = table["languages"][language]
        language_row["routes"] = [
            row for row in language_row["routes"] if row.get("engine_id") != "moss"
        ]
        for key in (
            "technical_candidates_on_sprint_evidence",
            "candidate_routes_without_an_identified_lane_commercial_blocker",
            "unsafe_routes",
            "limited_or_evidence_incomplete_routes",
            "no_lane_evidence_or_unsupported",
        ):
            language_row[key] = [engine for engine in language_row[key] if engine != "moss"]
        moss_row, bucket = make_moss_route(
            language=language, g1=g1, g2=g2, g3=g3, g4=g4, g5=g5
        )
        language_row["routes"].append(moss_row)
        moss_buckets[language] = bucket
        if bucket == "technical":
            language_row["technical_candidates_on_sprint_evidence"].append("moss")
        elif bucket == "unsafe":
            language_row["unsafe_routes"].append("moss")
        else:
            language_row["limited_or_evidence_incomplete_routes"].append("moss")

    zh_bucket = moss_buckets["zh"]
    table["headline"].update(
        {
            "single_engine_safe_all_five": False,
            "single_engine_commercially_unblocked_all_five": False,
            "general_zh_ship_route_from_tested_set": False,
            "reason": (
                "MOSS was measured, but it is not a shippable Mandarin route on current evidence: "
                f"its ZH technical bucket is {zh_bucket}, personal listening is pending, and the pinned weight/export snapshots lack the root licence file required by their own README."
            ),
        }
    )
    # The pre-G best-supported routes were routes without an identified lane
    # commercial blocker.  MOSS remains blocked and is therefore never added.
    assert "moss" not in table["headline"]["best_supported_sprint_routes"]["zh"]
    table["interpretation"]["naturalness"] = (
        "USER_LISTENING_REQUIRED; MOSS ZH clips and mixed-script pairs are in LISTEN.md"
    )
    table["cross_cutting_findings"] = [
        finding
        for finding in table["cross_cutting_findings"]
        if not finding.startswith("MOSS Lane G:")
    ] + [
        "MOSS Lane G: 139,280,896 combined learned parameters [PROJECTED from two EXACT component counts]; the official ONNX TTS export redundantly materialises tied codebook tables.",
        "MOSS Lane G: HF metadata says Apache-2.0, but all pinned weight/export snapshots lack a root licence file and the TTS README says to treat redistribution as unlicensed until it appears.",
        "MOSS Lane G: reduced-n EN/ES/FR/RU evidence remains exploratory and cannot establish a safe single-engine five-language replacement.",
    ]
    table["evidence"].update(
        {
            "G1": "G1.json",
            "G2": "G2.json",
            "G3": "G3.json",
            "G4": "G4.json",
            "G5": "G5.json",
            "listening_guide": "LISTEN.md",
        }
    )
    table["lane_g_moss"] = {
        "historical_lane_d_route_rows_sha256": LANE_D_ROUTE_ROWS_SHA256,
        "zh_bucket": zh_bucket,
        "commercially_unblocked": False,
        "commercial_blocker": g1["licence"]["weight_and_export_licence_scope"]["disposition"],
        "naturalness": "PENDING_USER_LISTENING",
    }
    assert canonical_sha256(lane_d_route_rows(table)) == LANE_D_ROUTE_ROWS_SHA256
    return table


def replace_marked_block(text: str, marker: str, block: str, anchor: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    wrapped = f"{start}\n{block.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(wrapped, text)
    assert anchor in text, anchor
    return text.replace(anchor, wrapped + "\n\n" + anchor, 1)


def build_mixed_listen_block(mixed_pairs: list[dict[str, Any]]) -> str:
    lines = [
        "## MOSS Mandarin mixed-script probes",
        "",
        "Listen to each **full** clip, then its **ablated** Han-only control. Mark the row only after deciding whether the removed acronym/digits were actually spoken. Different duration or waveform is not enough. `FRAME CAP` means at least one member reached the configured generation limit, so treat that pair as especially uncertain.",
        "",
        "| Done | probe | full input | full clip | ablated input | ablated clip | decision (yes / no / unclear) | check |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pair in mixed_pairs:
        cap = pair["full"]["max_new_frames_hit"] or pair["ablated"]["max_new_frames_hit"]
        tag = " **FRAME CAP**" if cap else ""
        lines.append(
            "| ☐ | `{pair_id}`{tag} | {full_text} | [{full_name}]({full_path}) | "
            "{ablated_text} | [{ablated_name}]({ablated_path}) |  | {check} |".format(
                pair_id=pair["pair_id"],
                tag=tag,
                full_text=pair["full_text"],
                full_name=Path(pair["full"]["relative_path"]).name,
                full_path=pair["full"]["relative_path"],
                ablated_text=pair["ablated_text"] or "(empty span control)",
                ablated_name=Path(pair["ablated"]["relative_path"]).name,
                ablated_path=pair["ablated"]["relative_path"],
                check=pair["listening_check"],
            )
        )
    return "\n".join(lines)


def update_listen(
    base: str,
    *,
    inventory: dict[str, Any],
    mixed_pairs: list[dict[str, Any]],
    zh_bucket: str,
    g5: dict[str, Any],
) -> str:
    text = base.replace("# Lane D listening guide", "# Lane D + G listening guide", 1)
    intro = (
        "**Lane G priority:** judge MOSS Mandarin next to Matcha using the same 20 prompts. "
        "Start with rows 12–14, then complete the six mixed-script full/ablated probes. "
        "MOSS remains commercially blocked on the pinned weight-licence evidence, so listening "
        "answers quality/coverage only."
    )
    text = replace_marked_block(
        text,
        "LANE_G_INTRO",
        intro,
        "## Quick verdict sheet",
    )
    text = text.replace(
        "Suggested order: (1) EN/ES/FR row by row, (2) ZH indices 12–14 first to catch dropped acronyms, then the other ZH rows, (3) RU Supertonic vs Silero with special attention to stress; Piper RU is diagnostic only.",
        "Suggested order: (1) ZH rows 12–14, MOSS beside Matcha; (2) all six MOSS mixed-script full/ablated pairs; (3) the remaining ZH rows; (4) the original EN/ES/FR and RU comparisons.",
    )

    zh_start = text.index("## Mandarin Chinese (`zh`)")
    zh_end = text.index("## Spanish (`es`)", zh_start)
    prefix, zh, suffix = text[:zh_start], text[zh_start:zh_end], text[zh_end:]
    moss_cases = {row["filename"]: row for row in g5["moss_zh"]["cases"]}
    if "`moss`" not in zh.splitlines()[2]:
        moss_row_symbol = "⛔" if zh_bucket == "unsafe" else "⚠"
        moss_label = (
            "⛔ G3-unsafe + licence-blocked"
            if zh_bucket == "unsafe"
            else "⚠ measured; licence/listening pending"
        )
        zh = zh.replace(
            "Engine columns: `kokoro` ⛔ non-finite route · `matcha` ⚠ OOV + licence",
            "Engine columns: `kokoro` ⛔ non-finite route · `matcha` ⚠ OOV + licence · "
            f"`moss` {moss_label}",
            1,
        )
        zh = zh.replace(
            "| Done | # | focus | source text | kokoro | matcha |",
            "| Done | # | focus | source text | kokoro | matcha | moss |",
            1,
        )
        zh = zh.replace(
            "|---|---:|---|---|---|---|",
            "|---|---:|---|---|---|---|---|",
            1,
        )
        rows = []
        for line in zh.splitlines():
            if line.startswith("| ☐ |"):
                match = re.match(r"\| ☐ \| (\d{2}) \|", line)
                assert match, line
                index = int(match.group(1))
                filename = f"zh_moss_{index:02d}.wav"
                path = f"artifacts/listening/{filename}"
                case = moss_cases[filename]
                line = line.rstrip()
                assert line.endswith("|"), line
                if case["listening_available"]:
                    cap = " **FRAME CAP**" if case["max_new_frames_hit"] else ""
                    moss_cell = (
                        f"{moss_row_symbol} [{filename}]({path}){cap}"
                    )
                else:
                    label = case["failure_label"] or "synthesis failure"
                    moss_cell = f"⛔ no WAV — {label}"
                line = line[:-1].rstrip() + f" | {moss_cell} |"
            rows.append(line)
        zh = "\n".join(rows) + ("\n" if zh.endswith("\n") else "")
    text = prefix + zh + suffix
    text = replace_marked_block(
        text,
        "LANE_G_MIXED",
        build_mixed_listen_block(mixed_pairs),
        "## Spanish (`es`)",
    )

    moss_missing = g5["moss_zh"]["missing"]
    moss_missing_text = "none" if not moss_missing else ", ".join(moss_missing)
    waveform_statement = (
        f"All {inventory['existing_wav_count']} existing listening files are readable, finite, non-silent, 0.15–30 s, and free of sustained clipping above -0.1 dBFS for more than 5 ms."
        if inventory["all_existing_wavs_pass_waveform_gate"]
        else "At least one existing listening file fails the registered waveform gate; see G5.json."
    )
    bundle = "\n".join(
        [
            "## Bundle integrity",
            "",
            f"- Existing listening WAVs: **{inventory['existing_wav_count']}/{inventory['expected_wav_count']}**. Frozen Lane D omissions: `zh_kokoro_09.wav`, `zh_kokoro_14.wav`. Missing MOSS G5 WAVs: {moss_missing_text}.",
            f"- Receipt-bound MOSS G5 WAVs: **{g5['moss_zh']['present']}/20**; waveform-safe: **{g5['moss_zh']['waveform_safe']}/20**; waveform + generation-cap safe: **{g5['moss_zh']['safe']}/20**.",
            f"- Existing listening WAV bytes: **{inventory['total_wav_bytes']:,}**.",
            f"- Ordered clip-hash manifest SHA-256: `{inventory['clip_hash_manifest_sha256']}`.",
            f"- Frozen Lane D non-MOSS manifest SHA-256: `{inventory['lane_d_frozen_identity']['clip_hash_manifest_sha256']}`.",
            f"- G5 synthesis receipt SHA-256: `{g5['synthesis_receipt']['sha256']}`.",
            "- Prompt manifest SHA-256: `5bfb957efc25482f6b8d9c8f90e5af8e55b99d11fc982c83815c5ea2fd782b20`.",
            f"- {waveform_statement}",
            "- The 12 mixed-script probe WAVs are linked separately above and are not part of the 300-file row-aligned listening inventory.",
            "",
        ]
    )
    bundle_index = text.index("## Bundle integrity")
    text = text[:bundle_index] + bundle
    return text


def make_summary_addendum(
    *,
    g1: dict[str, Any],
    g2: dict[str, Any],
    g3: dict[str, Any],
    g4: dict[str, Any],
    g5: dict[str, Any],
    table: dict[str, Any],
    out_d_bytes: int,
) -> str:
    zh_row = next(
        row
        for row in table["languages"]["zh"]["routes"]
        if row["engine_id"] == "moss"
    )
    final = g2["final_verdict"]
    cap_count = sum(
        pair[role]["max_new_frames_hit"]
        for pair in g5["mixed_script_pairs"]
        for role in ("full", "ablated")
    )
    return f"""<!-- LANE_G_ADDENDUM_START -->
## Lane G addendum — MOSS-TTS-Nano

**MOSS is not a shippable Mandarin route on current Lane G evidence.** Its ZH route disposition is `{zh_row['overall']}`: the technical screen is `{zh_row['technical']}`, personal naturalness/mixed-script listening is still pending, and redistribution remains blocked because the pinned weight/export snapshots do not contain the root licence file required by their own README.

- **G1 — count and package audit:** MOSS TTS is **117,311,232 learned parameters `[EXACT]`** and its audio tokenizer is **21,969,664 `[EXACT]`**, or **139,280,896 `[PROJECTED]`** together. This exactly matches Lane A's component counts. The official ONNX TTS export was confirmed to materialise tied audio-codebook tables redundantly, with three physical copies total.
- **G2 — Mandarin first:** the full 40-item ZH D1 synthesis arm completed and was scored: `{final['zh_point_estimate_technical_screen']}` on the point-estimate screen. Mixed-script coverage is explicitly `{final['mixed_script_coverage']}`; all 12 probe WAVs passed the waveform gate, but **{cap_count}** hit the generation-frame cap and waveform differences were not promoted to spoken-span evidence.
- **G3 — short inputs:** `{g3['final_verdict']['overall']}` across the frozen 1–3-character battery. Per-language verdicts: `{json.dumps(g3['final_verdict']['by_language'], ensure_ascii=False, sort_keys=True)}`.
- **G4 — EN/ES/FR/RU:** reduced-n results remain **exploratory/incomplete**, so `single_engine_safe_all_five` stays false. Coverage verdicts: `{json.dumps(g4['final_verdict']['language_coverage'], ensure_ascii=False, sort_keys=True)}`.
- **G5 — personal listening:** **{g5['moss_zh']['present']}/20** MOSS ZH clips are bound to the G5 synthesis receipt and **{g5['moss_zh']['safe']}/20** clear both the waveform and generation-cap gates. Retained clips appear beside Matcha in `LISTEN.md`; failed cases are explicit no-WAV cells. Six full/ablated mixed-script pairs follow in a separate yes/no/unclear checklist. Naturalness is `{g5['final_verdict']['naturalness']}`.
- **Licence:** HF metadata says Apache-2.0 `[VENDOR]`, but no root licence/notice file exists in any pinned TTS, tokenizer, or ONNX snapshot `[EXACT]`. The TTS README explicitly says to treat redistribution as unlicensed until that file appears; MOSS is therefore excluded from every commercially-unblocked candidate list.
- **Storage and boundaries:** no training, fine-tuning, quantization, export, phone work, or supplied cloud credentials were used. Final `out/d` footprint: **{out_d_bytes:,} bytes**, below the 3,000,000,000-byte cap.
<!-- LANE_G_ADDENDUM_END -->"""


def update_summary(base: str, block: str) -> str:
    pattern = re.compile(
        r"<!-- LANE_G_ADDENDUM_START -->.*?<!-- LANE_G_ADDENDUM_END -->",
        re.DOTALL,
    )
    if pattern.search(base):
        return pattern.sub(block, base).rstrip() + "\n"
    return base.rstrip() + "\n\n" + block.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write G5.json/G5.md, tts_table.json, LISTEN.md, and the Lane G summary addendum",
    )
    args = parser.parse_args()

    reports = {name: load_json(OUT_D / f"{name}.json") for name in ("G1", "G2", "G3", "G4")}
    for payload in reports.values():
        assert_evidence_classes(payload)
    base_table = load_json(OUT_D / "tts_table.json")
    historical_hash = canonical_sha256(lane_d_route_rows(base_table))
    assert historical_hash == LANE_D_ROUTE_ROWS_SHA256, historical_hash

    inventory = scan_listening_bundle()
    mixed_pairs = load_mixed_pairs()
    synthesis_receipt = load_g5_synthesis_receipt(
        g1=reports["G1"], inventory=inventory
    )
    g5, g5_md = make_g5(inventory, mixed_pairs, synthesis_receipt)
    table = update_table(
        base_table,
        g1=reports["G1"],
        g2=reports["G2"],
        g3=reports["G3"],
        g4=reports["G4"],
        g5=g5,
    )
    listen = update_listen(
        (OUT_D / "LISTEN.md").read_text(encoding="utf-8"),
        inventory=inventory,
        mixed_pairs=mixed_pairs,
        zh_bucket=table["lane_g_moss"]["zh_bucket"],
        g5=g5,
    )

    current_bytes = tree_logical_bytes(OUT_D)
    rendered_without_summary = {
        OUT_D / "G5.json": json.dumps(g5, ensure_ascii=False, indent=2) + "\n",
        OUT_D / "G5.md": g5_md,
        OUT_D / "tts_table.json": json.dumps(table, ensure_ascii=False, indent=2) + "\n",
        OUT_D / "LISTEN.md": listen,
    }
    projected_without_summary = current_bytes
    for path, rendered in rendered_without_summary.items():
        projected_without_summary -= stored_file_bytes(path)
        projected_without_summary += len(rendered.encode("utf-8"))
    # The byte count changes the summary text itself. Two iterations are enough
    # to converge because only the decimal digit string can change length.
    projected_bytes = projected_without_summary
    summary = ""
    for _ in range(2):
        summary_block = make_summary_addendum(
            g1=reports["G1"],
            g2=reports["G2"],
            g3=reports["G3"],
            g4=reports["G4"],
            g5=g5,
            table=table,
            out_d_bytes=projected_bytes,
        )
        summary = update_summary(
            (OUT_D / "LANE_SUMMARY.md").read_text(encoding="utf-8"), summary_block
        )
        summary_path = OUT_D / "LANE_SUMMARY.md"
        projected_bytes = projected_without_summary
        projected_bytes -= stored_file_bytes(summary_path)
        projected_bytes += len(summary.encode("utf-8"))

    preview = {
        "mode": "APPLY" if args.apply else "READ_ONLY_PREVIEW",
        "g2_final_verdict": reports["G2"]["final_verdict"],
        "g3_final_verdict": reports["G3"]["final_verdict"],
        "g4_final_verdict": reports["G4"]["final_verdict"],
        "g5_final_verdict": g5["final_verdict"],
        "zh_moss_bucket": table["lane_g_moss"]["zh_bucket"],
        "listening_inventory": g5["inventory"],
        "mixed_script_pair_count": len(mixed_pairs),
        "historical_lane_d_route_rows_sha256": historical_hash,
        "current_out_d_bytes": current_bytes,
        "conservative_projected_out_d_bytes": projected_bytes,
        "storage_cap_bytes": MAX_BYTES,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.apply:
        return 0

    write_json(OUT_D / "G5.json", g5)
    (OUT_D / "G5.md").write_text(g5_md, encoding="utf-8")
    write_json(OUT_D / "tts_table.json", table)
    (OUT_D / "LISTEN.md").write_text(listen, encoding="utf-8")
    (OUT_D / "LANE_SUMMARY.md").write_text(summary, encoding="utf-8")
    final_bytes = tree_logical_bytes(OUT_D)
    assert final_bytes == projected_bytes, (final_bytes, projected_bytes)
    assert final_bytes < MAX_BYTES, (final_bytes, MAX_BYTES)
    print(f"wrote Lane G integration; out/d bytes={final_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
