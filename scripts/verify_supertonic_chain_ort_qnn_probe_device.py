#!/usr/bin/env python3
"""Independently verify future Supertonic four-stage QNN phone evidence.

This verifier never invokes ADB.  It consumes a committed trust-root spec, the
host whole-APK report, an archived app-private report, hash-bound raw /proc
artifacts, and a collector evidence document.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/model_sweep_20260729"
DEFAULT_SPEC = REPORT_ROOT / "supertonic_chain_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = REPORT_ROOT / "supertonic_chain_ort_qnn_apk_local_verify.json"
GATE_ID = "supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1"
PACKAGE = "ai.liquid.lfmdemo.supertonicchainprobe"
COMPLETION_MARKER = "SUPERTONIC_CHAIN_PROBE_REPORT_COMPLETE_V1"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MODEL_ROOT = (
    "/data/user/0/ai.liquid.lfmdemo.supertonicchainprobe/files/"
    "supertonic_chain_probe/payload-v1"
)
EXPECTED_REPORT_KEYS = {
    "schema_version",
    "gate_id",
    "package",
    "uid",
    "ort_version",
    "available_providers",
    "qnn_provider_requested",
    "cpu_ep_fallback_disabled",
    "backend_path",
    "model_storage",
    "runtime_storage",
    "host_transport_required",
    "source_format",
    "symbolic_dimension_overrides",
    "sessions_created",
    "chain",
    "timing_ms",
    "full_neural_chain",
    "chained_pcm_tensor",
    "full_tts",
    "audio_file_output",
    "audio_playback",
    "audio_quality_accepted",
    "host_reference_comparisons_promoted",
    "production_integration",
    "accepted_packages_mutated",
    "status",
    "claim_boundary",
    "evidence_schema_version",
    "run_id",
    "run_trigger",
    "started_at_unix_ms",
    "started_at_utc",
    "completed_at_unix_ms",
    "completed_at_utc",
    "elapsed_wall_ms",
    "elapsed_monotonic_ms",
    "report_complete",
    "completion_marker",
    "report_path",
}


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    def invalid_constant(item: str) -> Any:
        raise RuntimeError(f"invalid JSON numeric constant in {label}: {item}")

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON in {label}") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"expected JSON object in {label}")
    return decoded


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return strict_json_bytes(path.read_bytes(), str(path))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError(f"invalid integer for {label}: {value!r}")
    return value


def number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"invalid finite number for {label}: {value!r}")
    result = float(value)
    if minimum is not None and result < minimum:
        raise RuntimeError(f"number below minimum for {label}: {result}")
    return result


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(f"{label} field inventory mismatch: {actual!r}")
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"invalid SHA-256 for {label}")
    return value


def utc_unix_ms(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"invalid UTC timestamp for {label}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"invalid UTC timestamp for {label}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"timestamp is not UTC for {label}")
    milliseconds = parsed.timestamp() * 1000.0
    rounded = round(milliseconds)
    if abs(milliseconds - rounded) > 1.0e-6:
        raise RuntimeError(f"timestamp is not millisecond-exact for {label}")
    return int(rounded)


def verify_stats(value: Any, expected_elements: int, label: str) -> dict[str, Any]:
    stats = exact_keys(
        value,
        {
            "elements",
            "finite",
            "min",
            "max",
            "mean",
            "rms",
            "max_abs",
            "nonzero_values",
            "values_over_unit_range",
            "sha256_float32le",
        },
        label,
    )
    require(stats["finite"] is True, f"{label} is not finite")
    require(
        integer(stats["elements"], label + ".elements", minimum=1) == expected_elements,
        f"{label} element count mismatch",
    )
    minimum = number(stats["min"], label + ".min")
    maximum = number(stats["max"], label + ".max")
    mean = number(stats["mean"], label + ".mean")
    rms = number(stats["rms"], label + ".rms", minimum=0.0)
    max_abs = number(stats["max_abs"], label + ".max_abs", minimum=0.0)
    require(minimum <= mean <= maximum, f"{label} mean is outside min/max")
    require(
        max_abs + 1.0e-12 >= max(abs(minimum), abs(maximum)),
        f"{label} max_abs mismatch",
    )
    nonzero = integer(stats["nonzero_values"], label + ".nonzero_values")
    clipped = integer(
        stats["values_over_unit_range"], label + ".values_over_unit_range"
    )
    require(nonzero <= expected_elements, f"{label} nonzero count is impossible")
    require(clipped <= expected_elements, f"{label} clipped count is impossible")
    require_sha256(stats["sha256_float32le"], label + ".sha256_float32le")
    return {
        **stats,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "rms": rms,
        "max_abs": max_abs,
    }


def verify_comparison(value: Any, expected_elements: int, label: str) -> dict[str, Any]:
    comparison = exact_keys(
        value,
        {
            "diagnostic_only",
            "promoted",
            "elements",
            "max_abs_error",
            "mean_abs_error",
            "rmse",
            "cosine_similarity",
            "actual_sha256",
            "reference_sha256",
        },
        label,
    )
    require(
        comparison["diagnostic_only"] is True and comparison["promoted"] is False,
        f"{label} crossed the diagnostic-only boundary",
    )
    require(
        integer(comparison["elements"], label + ".elements", minimum=1)
        == expected_elements,
        f"{label} element count mismatch",
    )
    for field in ("max_abs_error", "mean_abs_error", "rmse"):
        number(comparison[field], label + "." + field, minimum=0.0)
    cosine = number(comparison["cosine_similarity"], label + ".cosine_similarity")
    require(-1.000001 <= cosine <= 1.000001, f"{label} cosine is invalid")
    require_sha256(comparison["actual_sha256"], label + ".actual_sha256")
    require_sha256(comparison["reference_sha256"], label + ".reference_sha256")
    return comparison


def onnx_dtype(dtype: str) -> str:
    mapping = {"float32": "FLOAT", "int64": "INT64"}
    if dtype not in mapping:
        raise RuntimeError(f"unsupported authoritative ONNX dtype: {dtype}")
    return mapping[dtype]


def verify_nodes(
    value: Any,
    expected: dict[str, Any],
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        raise RuntimeError(f"{label} node count mismatch")
    actual: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        node = exact_keys(item, {"name", "dtype", "shape"}, f"{label}[{index}]")
        name = node["name"]
        if not isinstance(name, str) or name in actual:
            raise RuntimeError(f"{label} has an invalid/duplicate node name")
        actual[name] = node
    require(set(actual) == set(expected), f"{label} node names mismatch")
    for name, contract in expected.items():
        node = actual[name]
        require(
            node["dtype"] == onnx_dtype(contract["dtype"]),
            f"{label}.{name} dtype mismatch",
        )
        require(node["shape"] == contract["shape"], f"{label}.{name} shape mismatch")


def verify_contract(value: Any, model: dict[str, Any], label: str) -> None:
    contract = exact_keys(value, {"inputs", "outputs"}, label)
    verify_nodes(contract["inputs"], model["inputs"], label + ".inputs")
    verify_nodes(contract["outputs"], model["outputs"], label + ".outputs")


def verify_reference_stage(
    stage: dict[str, Any],
    *,
    model: dict[str, Any],
    component: str,
    elements: int,
    label: str,
) -> dict[str, Any]:
    require(stage["component"] == component, f"{label} component mismatch")
    require(stage["session_created"] is True, f"{label} session was not created")
    verify_contract(stage["contract"], model, label + ".contract")
    output = verify_stats(stage["output"], elements, label + ".output")
    comparison = verify_comparison(
        stage["reference_comparison"], elements, label + ".reference_comparison"
    )
    require(
        comparison["actual_sha256"] == output["sha256_float32le"],
        f"{label} output/comparison hash mismatch",
    )
    return output


def verify_chain(report: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    chain = exact_keys(
        report["chain"],
        {
            "duration_predictor",
            "text_encoder",
            "vector_estimator",
            "vocoder",
            "vector_steps",
            "sample_rate",
            "wav_length_from_device_duration",
            "fixed_latent_length",
            "real_latent_length",
            "waveform_samples",
            "waveform_rms",
            "finite_non_silent_pcm_tensor",
        },
        "report.chain",
    )
    duration = exact_keys(
        chain["duration_predictor"],
        {
            "component",
            "session_created",
            "contract",
            "output",
            "reference_comparison",
            "diagnostic_absolute_tolerance",
            "reference_within_diagnostic_tolerance",
            "finite_shape_gate_passed",
        },
        "duration stage",
    )
    duration_output = verify_reference_stage(
        duration,
        model=spec["models"]["duration_predictor"],
        component="duration_predictor",
        elements=1,
        label="duration stage",
    )
    require(
        duration["finite_shape_gate_passed"] is True,
        "duration finite/shape gate failed",
    )
    require(
        number(
            duration["diagnostic_absolute_tolerance"],
            "duration diagnostic tolerance",
            minimum=0.0,
        )
        == spec["fixture"]["duration_absolute_tolerance"],
        "duration diagnostic tolerance mismatch",
    )
    require(
        isinstance(duration["reference_within_diagnostic_tolerance"], bool),
        "duration diagnostic comparison flag is not boolean",
    )

    text = exact_keys(
        chain["text_encoder"],
        {
            "component",
            "session_created",
            "contract",
            "output",
            "reference_comparison",
            "finite_shape_gate_passed",
        },
        "text-encoder stage",
    )
    verify_reference_stage(
        text,
        model=spec["models"]["text_encoder"],
        component="text_encoder",
        elements=1 * 256 * 32,
        label="text-encoder stage",
    )
    require(
        text["finite_shape_gate_passed"] is True,
        "text encoder finite/shape gate failed",
    )

    vector = exact_keys(
        chain["vector_estimator"],
        {
            "component",
            "session_created",
            "contract",
            "steps",
            "step_count",
            "finite_shape_gate_passed",
            "masked_padding_zero",
        },
        "vector stage",
    )
    require(vector["component"] == "vector_estimator", "vector component mismatch")
    require(vector["session_created"] is True, "vector session was not created")
    verify_contract(
        vector["contract"], spec["models"]["vector_estimator"], "vector contract"
    )
    require(
        integer(vector["step_count"], "vector step_count", minimum=1) == 4,
        "vector step count mismatch",
    )
    require(
        vector["finite_shape_gate_passed"] is True
        and vector["masked_padding_zero"] is True,
        "vector finite/shape/mask gate failed",
    )
    steps = vector["steps"]
    if not isinstance(steps, list) or len(steps) != 4:
        raise RuntimeError("vector step report inventory mismatch")
    for index, raw_step in enumerate(steps):
        step = exact_keys(
            raw_step,
            {
                "step",
                "total_steps",
                "execute_ms",
                "output",
                "reference_comparison",
                "finite_shape_gate_passed",
                "masked_padding_zero",
            },
            f"vector step {index}",
        )
        require(
            integer(step["step"], f"vector step {index}.step") == index
            and integer(step["total_steps"], f"vector step {index}.total_steps") == 4,
            f"vector step {index} index/total mismatch",
        )
        number(step["execute_ms"], f"vector step {index}.execute_ms", minimum=0.0)
        output = verify_stats(
            step["output"], 1 * 144 * 32, f"vector step {index}.output"
        )
        comparison = verify_comparison(
            step["reference_comparison"],
            1 * 144 * 32,
            f"vector step {index}.reference",
        )
        require(
            comparison["actual_sha256"] == output["sha256_float32le"],
            f"vector step {index} output/comparison hash mismatch",
        )
        require(
            step["finite_shape_gate_passed"] is True
            and step["masked_padding_zero"] is True,
            f"vector step {index} finite/shape/mask gate failed",
        )

    vocoder = exact_keys(
        chain["vocoder"],
        {
            "component",
            "session_created",
            "contract",
            "output",
            "reference_comparison",
            "finite_shape_gate_passed",
            "non_silent",
            "audio_quality_accepted",
        },
        "vocoder stage",
    )
    vocoder_output = verify_reference_stage(
        vocoder,
        model=spec["models"]["vocoder"],
        component="vocoder",
        elements=98_304,
        label="vocoder stage",
    )
    require(
        vocoder["finite_shape_gate_passed"] is True
        and vocoder["non_silent"] is True
        and vocoder["audio_quality_accepted"] is False,
        "vocoder finite/non-silence/claim gate failed",
    )
    require(vocoder_output["rms"] > 1.0e-5, "vocoder PCM tensor is silent")

    require(chain["vector_steps"] == 4, "chain vector-step count mismatch")
    require(chain["sample_rate"] == 44_100, "chain sample rate mismatch")
    require(chain["fixed_latent_length"] == 32, "fixed latent length mismatch")
    require(chain["real_latent_length"] == 31, "real latent length mismatch")
    require(chain["waveform_samples"] == 98_304, "waveform sample count mismatch")
    wav_length = integer(
        chain["wav_length_from_device_duration"], "device-derived wav length", minimum=1
    )
    require(
        30 * 3_072 < wav_length <= 31 * 3_072,
        "device duration left the fixed latent bucket",
    )
    waveform_rms = number(chain["waveform_rms"], "chain waveform_rms", minimum=0.0)
    require(
        abs(waveform_rms - vocoder_output["rms"]) <= 1.0e-12,
        "waveform RMS fields disagree",
    )
    require(
        chain["finite_non_silent_pcm_tensor"] is True,
        "chain did not assert finite non-silent PCM",
    )
    return {
        "duration": duration_output,
        "vocoder": vocoder_output,
        "wav_length": wav_length,
        "vector_steps": 4,
    }


def parse_status(status: str) -> dict[str, Any]:
    uid = re.search(
        r"^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
        status,
        re.MULTILINE,
    )
    rss = re.search(r"^VmRSS:\s+(\d+) kB\s*$", status, re.MULTILINE)
    if uid is None or rss is None:
        raise RuntimeError("raw /proc/status is missing UID or VmRSS")
    return {
        "uid_fields": [int(item) for item in uid.groups()],
        "status_rss_kib": int(rss.group(1)),
    }


def parse_smaps(smaps: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for field, key in (
        ("Rss", "rss_kib"),
        ("Pss", "pss_kib"),
        ("Private_Dirty", "private_dirty_kib"),
        ("Swap", "swap_kib"),
    ):
        values = [
            int(item)
            for item in re.findall(
                rf"^{re.escape(field)}:\s+(\d+) kB\s*$", smaps, re.MULTILINE
            )
        ]
        if not values:
            raise RuntimeError(f"raw /proc/smaps is missing {field}")
        output[key] = sum(values)
    return output


def parse_process_snapshot(
    status: str,
    maps: str,
    fds: str,
    smaps: str,
    app_install_root: str,
) -> dict[str, Any]:
    if not app_install_root.startswith("/data/app/") or app_install_root.endswith("/"):
        raise RuntimeError("invalid installed-app root for process parsing")
    mapped_paths: list[str] = []
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped_paths.append(fields[5].removesuffix(" (deleted)"))
    libraries = sorted(
        {
            PurePosixPath(path).name
            for path in mapped_paths
            if PurePosixPath(path).name.startswith("lib")
            and ".so" in PurePosixPath(path).name
        }
    )
    app_owned = sorted(
        path for path in mapped_paths if path.startswith(app_install_root + "/")
    )
    qcom_dmabuf_count = sum("/dmabuf:qcom," in line for line in fds.splitlines())
    return {
        **parse_status(status),
        **parse_smaps(smaps),
        "loaded_libraries": libraries,
        "app_owned_mappings": app_owned,
        "fastrpc_cdsp_fd_open": "/dev/fastrpc-cdsp" in fds,
        "qcom_dmabuf_count": qcom_dmabuf_count,
    }


def verify_raw_artifacts(
    process: dict[str, Any],
    evidence_dir: Path,
    app_install_root: str,
) -> dict[str, Any]:
    records = process.get("raw_artifacts")
    if not isinstance(records, dict) or set(records) != {
        "status",
        "maps",
        "fd",
        "smaps",
    }:
        raise RuntimeError("raw /proc artifact inventory mismatch")
    raw: dict[str, str] = {}
    for label, record in records.items():
        exact_keys(record, {"path", "bytes", "sha256"}, f"raw artifact {label}")
        path_value = record["path"]
        if (
            not isinstance(path_value, str)
            or PurePosixPath(path_value).name != path_value
            or path_value in {"", ".", ".."}
        ):
            raise RuntimeError(f"unsafe raw artifact path: {path_value!r}")
        path = evidence_dir / path_value
        require(path.is_file(), f"missing raw artifact: {path}")
        require(
            path.stat().st_size
            == integer(record["bytes"], f"{label}.bytes", minimum=1),
            f"raw artifact byte count mismatch: {label}",
        )
        require(
            sha256(path) == require_sha256(record["sha256"], f"{label}.sha256"),
            f"raw artifact SHA-256 mismatch: {label}",
        )
        raw[label] = path.read_text(encoding="utf-8")
    parsed = parse_process_snapshot(
        raw["status"], raw["maps"], raw["fd"], raw["smaps"], app_install_root
    )
    summary = {
        key: value
        for key, value in process.items()
        if key not in {"pid", "report_owner_uid", "raw_artifacts"}
    }
    require(
        summary == parsed, "collector process summary differs from raw /proc evidence"
    )
    return parsed


def verify_timing(report: dict[str, Any]) -> dict[str, float]:
    timing = exact_keys(
        report["timing_ms"],
        {
            "duration_session_load",
            "duration_execute",
            "text_encoder_session_load",
            "text_encoder_execute",
            "vector_session_load",
            "vector_execute_four_steps",
            "vocoder_session_load",
            "vocoder_execute",
            "session_load_total",
            "neural_execute_total",
            "total",
        },
        "report.timing_ms",
    )
    parsed = {
        key: number(value, "timing_ms." + key, minimum=0.0)
        for key, value in timing.items()
    }
    for key in (
        "duration_session_load",
        "duration_execute",
        "text_encoder_session_load",
        "text_encoder_execute",
        "vector_session_load",
        "vector_execute_four_steps",
        "vocoder_session_load",
        "vocoder_execute",
        "session_load_total",
        "neural_execute_total",
        "total",
    ):
        require(parsed[key] > 0.0, f"non-positive timing: {key}")
    load_sum = sum(
        parsed[key]
        for key in (
            "duration_session_load",
            "text_encoder_session_load",
            "vector_session_load",
            "vocoder_session_load",
        )
    )
    execute_sum = sum(
        parsed[key]
        for key in (
            "duration_execute",
            "text_encoder_execute",
            "vector_execute_four_steps",
            "vocoder_execute",
        )
    )
    require(
        abs(load_sum - parsed["session_load_total"]) <= 0.01,
        "session-load totals disagree",
    )
    require(
        abs(execute_sum - parsed["neural_execute_total"]) <= 0.01,
        "execution totals disagree",
    )
    require(
        parsed["total"] + 0.01 >= load_sum + execute_sum,
        "total timing is shorter than neural work",
    )
    return parsed


def verify_report(
    report: dict[str, Any],
    spec: dict[str, Any],
    installed: dict[str, Any],
    expected_run_id: str,
) -> dict[str, Any]:
    exact_keys(report, EXPECTED_REPORT_KEYS, "app-private report")
    require(report["schema_version"] == 1, "app report schema mismatch")
    require(report["evidence_schema_version"] == 1, "app evidence schema mismatch")
    require(report["gate_id"] == GATE_ID, "app report gate_id mismatch")
    require(report["package"] == PACKAGE, "app report package mismatch")
    require(report["run_id"] == expected_run_id, "app report run_id mismatch")
    require(
        report["run_trigger"] == spec["device_output_contract"]["run_trigger"],
        "app report trigger mismatch",
    )
    require(report["report_complete"] is True, "app report is incomplete")
    require(
        report["completion_marker"] == COMPLETION_MARKER, "completion marker mismatch"
    )
    require(
        report["report_path"] == spec["device_output_contract"]["report_path"],
        "app report publication path mismatch",
    )
    started = integer(report["started_at_unix_ms"], "started_at_unix_ms", minimum=1)
    completed = integer(
        report["completed_at_unix_ms"], "completed_at_unix_ms", minimum=1
    )
    require(completed >= started, "app report completion precedes start")
    require(
        utc_unix_ms(report["started_at_utc"], "started_at_utc") == started,
        "start UTC/unix mismatch",
    )
    require(
        utc_unix_ms(report["completed_at_utc"], "completed_at_utc") == completed,
        "completion UTC/unix mismatch",
    )
    require(
        integer(report["elapsed_wall_ms"], "elapsed_wall_ms") == completed - started,
        "wall elapsed time mismatch",
    )
    elapsed_monotonic = number(
        report["elapsed_monotonic_ms"], "elapsed_monotonic_ms", minimum=0.0
    )
    require(elapsed_monotonic > 0.0, "monotonic elapsed time is not positive")

    uid = integer(report["uid"], "app report uid", minimum=10_000)
    require(uid == installed["uid"], "app report UID mismatch")
    require(report["ort_version"] == "1.25.0", "ORT version mismatch")
    providers = report["available_providers"]
    require(
        isinstance(providers, list)
        and providers
        and len(providers) == len(set(providers))
        and all(isinstance(item, str) for item in providers),
        "available-provider inventory is invalid",
    )
    require("QNNExecutionProvider" in providers, "QNN EP is unavailable")
    require(report["qnn_provider_requested"] is True, "QNN EP was not requested")
    require(
        report["cpu_ep_fallback_disabled"] is True, "CPU EP fallback was not disabled"
    )
    require(report["host_transport_required"] is False, "host transport was required")
    require(report["source_format"] == "raw_onnx", "source model format mismatch")
    require(
        report["symbolic_dimension_overrides"]
        == {"batch_size": 1, "text_length": 32, "latent_length": 32},
        "symbolic-dimension override mismatch",
    )
    require(
        report["sessions_created"]
        == {
            "duration_predictor": True,
            "text_encoder": True,
            "vector_estimator": True,
            "vocoder": True,
        },
        "four-session creation inventory mismatch",
    )
    require(report["model_storage"] == MODEL_ROOT, "app-private model root mismatch")
    app_root = installed["apk_path"].removesuffix("/base.apk")
    expected_runtime = app_root + "/lib/arm64"
    require(
        report["runtime_storage"] == expected_runtime, "app-owned runtime root mismatch"
    )
    require(
        report["backend_path"] == expected_runtime + "/libQnnHtp.so",
        "QNN backend path mismatch",
    )
    require(report["full_neural_chain"] is True, "full neural chain did not complete")
    require(report["chained_pcm_tensor"] is True, "chained PCM tensor was not produced")
    for field in (
        "full_tts",
        "audio_file_output",
        "audio_playback",
        "audio_quality_accepted",
        "host_reference_comparisons_promoted",
        "production_integration",
        "accepted_packages_mutated",
    ):
        require(report[field] is False, f"claim/mutation boundary violated: {field}")
    require(
        report["status"] == "four_component_chain_probe_pass",
        "app chain probe did not pass",
    )
    require(
        report["claim_boundary"]
        == (
            "All four raw ONNX components executed as one deterministic chain with "
            "finite shape-safe tensors and non-silent PCM. Host reference comparisons "
            "are diagnostic only; no audio quality, language identity, playback, "
            "promoted latency, unplugged durability, or production integration is claimed."
        ),
        "app claim boundary mismatch",
    )
    chain = verify_chain(report, spec)
    timing = verify_timing(report)
    require(
        elapsed_monotonic + 0.01 >= timing["total"],
        "published elapsed time is shorter than probe timing",
    )
    return {
        "uid": uid,
        "started_at_unix_ms": started,
        "completed_at_unix_ms": completed,
        "elapsed_monotonic_ms": elapsed_monotonic,
        "chain": chain,
        "timing_ms": timing,
    }


def verify_resident_payload(
    resident: dict[str, Any], spec: dict[str, Any], report: dict[str, Any]
) -> None:
    exact_keys(
        resident,
        {"root", "kib", "payload_manifest", "models", "all_four_models_resident"},
        "resident payload",
    )
    require(
        resident["root"] == report["model_storage"] == MODEL_ROOT,
        "resident root mismatch",
    )
    kib = integer(resident["kib"], "resident payload KiB", minimum=1)
    require(
        kib <= spec["device_policy"]["max_resident_payload_kib"],
        "resident payload exceeds cap",
    )
    manifest = resident["payload_manifest"]
    exact_keys(manifest, {"path", "bytes", "sha256"}, "resident payload manifest")
    require(
        manifest["path"] == MODEL_ROOT + "/payload_manifest.json",
        "resident manifest path mismatch",
    )
    require(
        manifest["bytes"] == spec["payload_manifest"]["bytes"],
        "resident manifest size mismatch",
    )
    require(
        manifest["sha256"] == spec["payload_manifest"]["sha256"],
        "resident manifest hash mismatch",
    )
    models = resident["models"]
    require(
        isinstance(models, dict) and set(models) == set(spec["models"]),
        "resident four-model inventory mismatch",
    )
    for component, expected in spec["models"].items():
        record = models[component]
        exact_keys(record, {"path", "bytes", "sha256"}, f"resident model {component}")
        require(
            record["path"] == MODEL_ROOT + "/models/" + expected["filename"],
            f"resident model path mismatch: {component}",
        )
        require(
            record["bytes"] == expected["bytes"],
            f"resident model size mismatch: {component}",
        )
        require(
            record["sha256"] == expected["sha256"],
            f"resident model hash mismatch: {component}",
        )
    require(
        resident["all_four_models_resident"] is True,
        "resident model completion flag is false",
    )


def verify_documents(
    spec: dict[str, Any],
    local: dict[str, Any],
    report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    expected_run_id: str,
    spec_path: Path,
    local_path: Path,
    report_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    require(
        RUN_ID_RE.fullmatch(expected_run_id) is not None, "invalid requested run_id"
    )
    require(
        spec.get("schema_version") == 2 and spec.get("gate_id") == GATE_ID,
        "spec identity mismatch",
    )
    require(local.get("schema_version") == 2, "local verifier schema mismatch")
    require(
        local.get("gate_id") == GATE_ID and local.get("package") == PACKAGE,
        "local verifier identity mismatch",
    )
    require(local.get("status") == "pass", "local whole-APK verification did not pass")
    require(
        local.get("apk", {}).get("bytes") == spec["apk"]["bytes"]
        and local.get("apk", {}).get("sha256") == spec["apk"]["sha256"],
        "local whole-APK trust root mismatch",
    )
    require(
        local.get("authoritative_spec", {}).get("sha256") == sha256(spec_path),
        "local report does not bind the authoritative spec",
    )

    exact_keys(
        evidence,
        {
            "schema_version",
            "gate_id",
            "package",
            "run_id",
            "run",
            "device",
            "installed_package",
            "accepted_packages",
            "process",
            "resident_payload",
            "authoritative_spec",
            "local_verifier_report",
            "device_report",
            "collection",
            "publication",
            "status",
            "claim_boundary",
        },
        "device evidence",
    )
    require(evidence["schema_version"] == 1, "device evidence schema mismatch")
    require(
        evidence["gate_id"] == GATE_ID and evidence["package"] == PACKAGE,
        "device evidence identity mismatch",
    )
    require(evidence["run_id"] == expected_run_id, "device evidence run_id mismatch")
    require(
        evidence["run"]
        == {
            "requested_run_id": expected_run_id,
            "trigger": spec["device_output_contract"]["run_trigger"],
        },
        "device requested-run correlation mismatch",
    )
    require(
        evidence["status"] == "collected_unverified", "device evidence was pre-promoted"
    )
    require(
        evidence["claim_boundary"]
        == "Evidence collection only; independent verification decides promotion.",
        "collector claim boundary mismatch",
    )

    spec_record = exact_keys(
        evidence["authoritative_spec"],
        {"path", "bytes", "sha256"},
        "evidence spec record",
    )
    require(
        spec_record["bytes"] == spec_path.stat().st_size,
        "evidence spec byte count mismatch",
    )
    require(spec_record["sha256"] == sha256(spec_path), "evidence spec hash mismatch")
    local_record = exact_keys(
        evidence["local_verifier_report"],
        {"path", "bytes", "sha256"},
        "evidence local verifier record",
    )
    require(
        local_record["bytes"] == local_path.stat().st_size,
        "local report byte count mismatch",
    )
    require(local_record["sha256"] == sha256(local_path), "local report hash mismatch")
    device_record = exact_keys(
        evidence["device_report"],
        {
            "path",
            "device_path",
            "partial_device_path",
            "partial_exists_after_capture",
            "bytes",
            "sha256",
        },
        "evidence device report record",
    )
    require(
        device_record["device_path"] == spec["device_output_contract"]["report_path"],
        "device report path contract mismatch",
    )
    require(
        device_record["partial_device_path"]
        == spec["device_output_contract"]["partial_report_path"],
        "partial report path contract mismatch",
    )
    require(
        device_record["partial_exists_after_capture"] is False,
        "partial app report still existed",
    )
    require(
        device_record["bytes"] == report_path.stat().st_size,
        "archived device report byte mismatch",
    )
    require(
        device_record["sha256"] == sha256(report_path),
        "archived device report hash mismatch",
    )

    publication = exact_keys(
        evidence["publication"],
        {"current_alias_published", "verification_published"},
        "collector publication state",
    )
    require(
        publication
        == {"current_alias_published": False, "verification_published": False},
        "collector published evidence before independent verification",
    )
    collection = exact_keys(
        evidence["collection"],
        {
            "collected_at_unix_ms",
            "collected_at_utc",
            "matching_failed_closed_report_archived_before_error",
            "partial_report_absent",
        },
        "collection metadata",
    )
    collected_ms = integer(
        collection["collected_at_unix_ms"], "collection unix time", minimum=1
    )
    require(
        utc_unix_ms(collection["collected_at_utc"], "collection UTC") == collected_ms,
        "collection UTC/unix mismatch",
    )
    require(
        collection["matching_failed_closed_report_archived_before_error"] is True,
        "failure archival policy missing",
    )
    require(
        collection["partial_report_absent"] is True,
        "partial report absence was not proven",
    )

    installed = exact_keys(
        evidence["installed_package"],
        {
            "package",
            "apk_path",
            "bytes",
            "sha256",
            "version_code",
            "version_name",
            "uid",
            "uid_sources",
            "pid",
        },
        "installed package",
    )
    require(installed["package"] == PACKAGE, "installed package mismatch")
    require(
        isinstance(installed["apk_path"], str)
        and installed["apk_path"].startswith("/data/app/")
        and installed["apk_path"].endswith("/base.apk"),
        "installed APK path mismatch",
    )
    require(
        installed["bytes"] == spec["apk"]["bytes"]
        and installed["sha256"] == spec["apk"]["sha256"],
        "installed whole APK mismatch",
    )
    require(
        installed["version_code"] == spec["apk_policy"]["version_code"],
        "installed versionCode mismatch",
    )
    require(
        installed["version_name"] == spec["apk_policy"]["version_name"],
        "installed versionName mismatch",
    )
    uid = integer(installed["uid"], "installed UID", minimum=10_000)
    require(
        installed["uid_sources"] == {"dumpsys": uid, "package_manager": uid},
        "installed UID sources disagree",
    )
    pid = integer(installed["pid"], "installed PID", minimum=1)

    device = exact_keys(
        evidence["device"],
        {"serial", "model", "soc", "soc_source", "soc_raw", "abi", "airplane_mode_on"},
        "device identity",
    )
    policy = spec["device_policy"]
    require(
        isinstance(device["serial"], str) and device["serial"], "device serial is empty"
    )
    require(device["model"] == policy["model"], "wrong device model")
    require(device["soc"] == policy["soc"], "wrong device SoC")
    require(
        isinstance(device["soc_source"], str) and device["soc_source"],
        "SoC source is empty",
    )
    require(
        isinstance(device["soc_raw"], str) and device["soc_raw"],
        "raw SoC identity is empty",
    )
    require(device["abi"] == policy["abi"], "wrong device ABI")
    require(
        integer(device["airplane_mode_on"], "airplane mode")
        == policy["airplane_mode_on"],
        "airplane-mode gate failed",
    )

    accepted = evidence["accepted_packages"]
    require(
        isinstance(accepted, dict) and set(accepted) == set(spec["accepted_packages"]),
        "accepted-package evidence inventory mismatch",
    )
    for package, expected in spec["accepted_packages"].items():
        record = exact_keys(
            accepted[package],
            {"path", "bytes", "sha256"},
            f"accepted package {package}",
        )
        require(
            isinstance(record["path"], str) and record["path"].endswith("/base.apk"),
            f"accepted package path mismatch: {package}",
        )
        integer(record["bytes"], f"accepted package bytes: {package}", minimum=1)
        require(
            record["sha256"] == expected["sha256"], f"accepted APK changed: {package}"
        )

    report_summary = verify_report(report, spec, installed, expected_run_id)
    process = evidence["process"]
    exact_keys(
        process,
        {
            "pid",
            "report_owner_uid",
            "uid_fields",
            "status_rss_kib",
            "rss_kib",
            "pss_kib",
            "private_dirty_kib",
            "swap_kib",
            "loaded_libraries",
            "app_owned_mappings",
            "fastrpc_cdsp_fd_open",
            "qcom_dmabuf_count",
            "raw_artifacts",
        },
        "process evidence",
    )
    require(process["pid"] == pid, "process PID mismatch")
    require(process["report_owner_uid"] == uid, "report owner UID mismatch")
    app_root = installed["apk_path"].removesuffix("/base.apk")
    parsed_process = verify_raw_artifacts(process, evidence_path.parent, app_root)
    require(parsed_process["uid_fields"] == [uid] * 4, "process UID fields mismatch")
    require(
        parsed_process["status_rss_kib"] <= policy["max_process_rss_kib"],
        "process exceeds RSS cap",
    )
    require(
        parsed_process["rss_kib"] <= policy["max_process_rss_kib"],
        "smaps RSS exceeds cap",
    )
    require(parsed_process["pss_kib"] > 0, "process PSS is not positive")
    loaded = set(parsed_process["loaded_libraries"])
    require(
        set(policy["required_loaded_libraries"]).issubset(loaded),
        "required QNN/ORT/FastRPC maps are missing",
    )
    app_paths = parsed_process["app_owned_mappings"]
    for library in policy["required_app_owned_libraries"]:
        require(
            any(path.endswith("/" + library) for path in app_paths),
            f"app-owned mapping missing: {library}",
        )
    require(
        parsed_process["fastrpc_cdsp_fd_open"] is True, "FastRPC CDSP fd is missing"
    )
    require(
        parsed_process["qcom_dmabuf_count"] >= policy["minimum_qcom_dmabuf_count"],
        "QCOM DMA evidence is missing",
    )

    verify_resident_payload(evidence["resident_payload"], spec, report)
    return {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "package": PACKAGE,
        "run_id": expected_run_id,
        "status": "pass",
        "claim": "isolated_raw_onnx_four_component_qnn_htp_chain_execution",
        "installed_apk_sha256": installed["sha256"],
        "all_four_qnn_sessions_created": True,
        "four_vector_steps_executed": True,
        "cpu_ep_fallback_disabled": True,
        "finite_non_silent_pcm_tensor": True,
        "pcm_audio_quality_accepted": False,
        "host_reference_comparisons_promoted": False,
        "fastrpc_cdsp_fd_open": True,
        "qcom_dmabuf_count": parsed_process["qcom_dmabuf_count"],
        "rss_mib": round(parsed_process["rss_kib"] / 1024, 3),
        "pss_mib": round(parsed_process["pss_kib"] / 1024, 3),
        "resident_payload_mib": round(evidence["resident_payload"]["kib"] / 1024, 3),
        "timing_ms": report_summary["timing_ms"],
        "chain": report_summary["chain"],
        "accepted_packages_unchanged": True,
        "current_alias_published": False,
        "full_tts": False,
        "device_mutated_by_verifier": False,
        "claim_boundary": spec["claim_boundary"],
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial output exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--device-report", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    run_id = arguments.run_id
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise RuntimeError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    spec_path = arguments.spec.expanduser().resolve()
    local_path = arguments.local_verify.expanduser().resolve()
    report_path = (
        (
            arguments.device_report
            or REPORT_ROOT / f"supertonic_chain_device_report_{run_id}.json"
        )
        .expanduser()
        .resolve()
    )
    evidence_path = (
        (
            arguments.evidence
            or REPORT_ROOT / f"supertonic_chain_device_evidence_{run_id}.json"
        )
        .expanduser()
        .resolve()
    )
    result = verify_documents(
        load_object(spec_path),
        load_object(local_path),
        load_object(report_path),
        load_object(evidence_path),
        expected_run_id=run_id,
        spec_path=spec_path,
        local_path=local_path,
        report_path=report_path,
        evidence_path=evidence_path,
    )
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if arguments.output is not None:
        atomic_write(arguments.output.expanduser().resolve(), encoded)
    print(encoded.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
