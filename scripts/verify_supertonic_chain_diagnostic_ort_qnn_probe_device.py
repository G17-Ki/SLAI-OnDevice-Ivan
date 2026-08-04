#!/usr/bin/env python3
"""Independently verify a Supertonic stage-diagnostic phone report.

This verifier never invokes ADB. A structurally complete failed-closed report is
a valid diagnostic outcome, not a full-chain pass or production promotion.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/model_sweep_20260729"
DEFAULT_SPEC = REPORT_ROOT / "supertonic_chain_diagnostic_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = (
    REPORT_ROOT / "supertonic_chain_diagnostic_ort_qnn_apk_local_verify.json"
)
HOST_VERIFIER_PATH = (
    ROOT / "scripts/verify_supertonic_chain_diagnostic_ort_qnn_probe_apk.py"
)
BASE_DEVICE_VERIFIER_PATH = ROOT / "scripts/verify_supertonic_chain_ort_qnn_probe_device.py"
GATE_ID = (
    "supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2"
)
PACKAGE = "ai.liquid.lfmdemo.supertonicchaindiagprobe"
VERSION_NAME = "0.2.0-stage-diagnostic-raw-onnx-qnn-probe"
COMPLETION_MARKER = "SUPERTONIC_CHAIN_DIAGNOSTIC_REPORT_COMPLETE_V2"
DIAGNOSTIC_MARKER = "SUPERTONIC_CHAIN_STAGE_DIAGNOSTIC_V2"
REPORT_PATH = (
    "/data/user/0/ai.liquid.lfmdemo.supertonicchaindiagprobe/files/"
    "supertonic_chain_diagnostic_probe/reports/latest.json"
)
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ORDERED_STAGES = [
    "payload_install",
    "runtime_initialization",
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
]
SESSION_STAGES = [
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
]
RUNTIME_PHASES = {
    "native_library_load",
    "ort_environment_create",
    "provider_validation",
    "fixture_load",
}
COMPONENT_PHASES = {
    "session_create",
    "contract_validation",
    "execute",
    "output_validation",
}
FALSE_CLAIMS = {
    "full_tts": False,
    "audio_file_output": False,
    "audio_playback": False,
    "audio_quality_accepted": False,
    "host_reference_comparisons_promoted": False,
    "production_integration": False,
    "accepted_packages_mutated": False,
}


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import verifier helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


host_verifier = load_module("verify_supertonic_chain_diagnostic_apk", HOST_VERIFIER_PATH)
base_device = load_module("verify_supertonic_chain_device_base", BASE_DEVICE_VERIFIER_PATH)


def sha256(path: Path) -> str:
    return host_verifier.sha256(path)


def strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    return host_verifier.base_verifier.strict_json_bytes(value, label)


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return strict_json_bytes(path.read_bytes(), str(path))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(f"{label} field inventory mismatch: {actual!r}")
    return value


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
    require(parsed.tzinfo is not None, f"timestamp lacks timezone: {label}")
    milliseconds = parsed.timestamp() * 1000.0
    rounded = round(milliseconds)
    require(abs(milliseconds - rounded) <= 1.0e-6, f"timestamp precision mismatch: {label}")
    return int(rounded)


def verify_identity_record(
    record: Any, path: Path, label: str, *, expected_path: str | None = None
) -> None:
    exact_keys(record, {"path", "bytes", "sha256"}, label)
    if expected_path is not None:
        require(record["path"] == expected_path, f"{label} path mismatch")
    require(
        integer(record["bytes"], label + ".bytes", minimum=1) == path.stat().st_size,
        f"{label} byte count mismatch",
    )
    require(
        require_sha256(record["sha256"], label + ".sha256") == sha256(path),
        f"{label} SHA-256 mismatch",
    )


def verify_persistence_metadata(
    report: dict[str, Any], spec: dict[str, Any], expected_run_id: str
) -> dict[str, Any]:
    require(report["schema_version"] == 1, "app report schema mismatch")
    require(report["evidence_schema_version"] == 1, "evidence schema mismatch")
    require(report["gate_id"] == GATE_ID, "app report gate mismatch")
    require(report["package"] == PACKAGE, "app report package mismatch")
    require(report["run_id"] == expected_run_id, "app report run_id mismatch")
    require(
        report["run_trigger"] == spec["device_output_contract"]["run_trigger"],
        "app report trigger mismatch",
    )
    require(report["report_complete"] is True, "app report is incomplete")
    require(report["completion_marker"] == COMPLETION_MARKER, "marker mismatch")
    require(report["report_path"] == REPORT_PATH, "report path mismatch")
    started = integer(report["started_at_unix_ms"], "started_at_unix_ms", minimum=1)
    completed = integer(
        report["completed_at_unix_ms"], "completed_at_unix_ms", minimum=started
    )
    require(
        utc_unix_ms(report["started_at_utc"], "started_at_utc") == started,
        "started UTC/unix mismatch",
    )
    require(
        utc_unix_ms(report["completed_at_utc"], "completed_at_utc") == completed,
        "completed UTC/unix mismatch",
    )
    require(
        integer(report["elapsed_wall_ms"], "elapsed_wall_ms") == completed - started,
        "wall elapsed mismatch",
    )
    elapsed = number(
        report["elapsed_monotonic_ms"], "elapsed_monotonic_ms", minimum=0.0
    )
    require(elapsed > 0.0, "monotonic elapsed is not positive")
    return {"started_at_unix_ms": started, "completed_at_unix_ms": completed}


def verify_payload_stage(stage: Any) -> None:
    value = exact_keys(
        stage,
        {
            "component",
            "manifest_verified",
            "assets_verified",
            "native_libraries_verified",
            "model_storage",
            "runtime_storage",
        },
        "payload_install report",
    )
    require(value["component"] == "payload_install", "payload component mismatch")
    for field in ("manifest_verified", "assets_verified", "native_libraries_verified"):
        require(value[field] is True, f"payload report did not assert {field}")
    require(
        value["model_storage"].startswith("/data/user/0/" + PACKAGE + "/files/"),
        "model storage is not app-private",
    )
    require(
        value["runtime_storage"].startswith("/data/app/")
        and value["runtime_storage"].endswith("/lib/arm64"),
        "runtime storage is not the installed APK native directory",
    )


def verify_runtime_stage(stage: Any) -> None:
    value = exact_keys(
        stage,
        {
            "component",
            "native_libraries_loaded",
            "ort_version",
            "available_providers",
            "qnn_provider_available",
            "fixture_files_loaded",
        },
        "runtime_initialization report",
    )
    require(value["component"] == "runtime_initialization", "runtime component mismatch")
    require(value["native_libraries_loaded"] is True, "native runtime was not loaded")
    require(value["ort_version"] == "1.25.0", "ORT version mismatch")
    require(
        isinstance(value["available_providers"], list)
        and "QNNExecutionProvider" in value["available_providers"],
        "QNN provider absent from runtime report",
    )
    require(value["qnn_provider_available"] is True, "QNN availability mismatch")
    require(value["fixture_files_loaded"] is True, "fixture load not completed")


def verify_reference_component(
    stage: Any, source: dict[str, Any], component: str, elements: int
) -> None:
    value = stage
    expected_keys = {
        "component",
        "session_created",
        "contract",
        "output",
        "reference_comparison",
        "finite_shape_gate_passed",
    }
    if component == "duration_predictor":
        expected_keys |= {
            "diagnostic_absolute_tolerance",
            "reference_within_diagnostic_tolerance",
        }
    if component == "vocoder":
        expected_keys |= {"non_silent", "audio_quality_accepted"}
    exact_keys(value, expected_keys, component + " completed report")
    output = base_device.verify_reference_stage(
        value,
        model=source["models"][component],
        component=component,
        elements=elements,
        label=component + " completed report",
    )
    require(value["finite_shape_gate_passed"] is True, "finite/shape gate mismatch")
    if component == "duration_predictor":
        require(
            number(value["diagnostic_absolute_tolerance"], "duration tolerance")
            == source["fixture"]["duration_absolute_tolerance"],
            "duration diagnostic tolerance mismatch",
        )
        require(
            isinstance(value["reference_within_diagnostic_tolerance"], bool),
            "duration diagnostic flag is not boolean",
        )
    if component == "vocoder":
        require(value["non_silent"] is True, "vocoder report is silent")
        require(value["audio_quality_accepted"] is False, "audio quality was promoted")
        require(output["rms"] > 1.0e-5, "vocoder output is silent")


def verify_vector_stage(stage: Any, source: dict[str, Any]) -> None:
    value = exact_keys(
        stage,
        {
            "component",
            "session_created",
            "contract",
            "steps",
            "step_count",
            "finite_shape_gate_passed",
            "masked_padding_zero",
        },
        "vector completed report",
    )
    require(value["component"] == "vector_estimator", "vector component mismatch")
    require(value["session_created"] is True, "vector session not created")
    base_device.verify_contract(
        value["contract"], source["models"]["vector_estimator"], "vector contract"
    )
    require(value["step_count"] == 4, "vector step count mismatch")
    require(value["finite_shape_gate_passed"] is True, "vector finite gate mismatch")
    require(value["masked_padding_zero"] is True, "vector mask gate mismatch")
    verify_vector_step_reports(value["steps"], expected_count=4)


def verify_vector_step_reports(value: Any, *, expected_count: int) -> None:
    require(
        isinstance(value, list) and len(value) == expected_count,
        "completed vector-step report count mismatch",
    )
    for index, raw in enumerate(value):
        step = exact_keys(
            raw,
            {
                "step",
                "total_steps",
                "execute_ms",
                "output",
                "reference_comparison",
                "finite_shape_gate_passed",
                "masked_padding_zero",
            },
            f"completed vector step {index}",
        )
        require(step["step"] == index and step["total_steps"] == 4, "step index mismatch")
        number(step["execute_ms"], f"vector step {index} execute_ms", minimum=0.0)
        output = base_device.verify_stats(
            step["output"], 144 * 32, f"completed vector step {index}.output"
        )
        comparison = base_device.verify_comparison(
            step["reference_comparison"],
            144 * 32,
            f"completed vector step {index}.reference",
        )
        require(
            comparison["actual_sha256"] == output["sha256_float32le"],
            "vector output/reference hash mismatch",
        )
        require(step["finite_shape_gate_passed"] is True, "vector finite gate false")
        require(step["masked_padding_zero"] is True, "vector masked padding false")


def verify_progress(
    report: dict[str, Any], spec: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    require(report["diagnostic_schema_version"] == 2, "diagnostic schema mismatch")
    require(report["diagnostic_marker"] == DIAGNOSTIC_MARKER, "diagnostic marker mismatch")
    completed = report["completed_stages"]
    require(isinstance(completed, list), "completed_stages is not an array")
    require(
        completed == ORDERED_STAGES[: len(completed)],
        "completed stages are not an ordered prefix",
    )
    require(
        report["completed_stage_count"] == len(completed),
        "completed_stage_count mismatch",
    )
    stage_reports = report["completed_stage_reports"]
    require(
        isinstance(stage_reports, dict) and set(stage_reports) == set(completed),
        "completed stage-report inventory mismatch",
    )
    if "payload_install" in completed:
        verify_payload_stage(stage_reports["payload_install"])
    if "runtime_initialization" in completed:
        verify_runtime_stage(stage_reports["runtime_initialization"])
    if "duration_predictor" in completed:
        verify_reference_component(
            stage_reports["duration_predictor"], source, "duration_predictor", 1
        )
    if "text_encoder" in completed:
        verify_reference_component(
            stage_reports["text_encoder"], source, "text_encoder", 256 * 32
        )
    if "vector_estimator" in completed:
        verify_vector_stage(stage_reports["vector_estimator"], source)
    if "vocoder" in completed:
        verify_reference_component(
            stage_reports["vocoder"], source, "vocoder", 98_304
        )

    sessions = exact_keys(
        report["sessions_created"], set(SESSION_STAGES), "sessions_created"
    )
    require(all(isinstance(value, bool) for value in sessions.values()), "session flag type")
    for stage in SESSION_STAGES:
        if stage in completed:
            require(sessions[stage] is True, f"completed stage lacks session: {stage}")

    vector_steps = integer(
        report["completed_vector_steps"], "completed_vector_steps"
    )
    require(vector_steps <= 4, "too many completed vector steps")
    verify_vector_step_reports(
        report["completed_vector_step_reports"], expected_count=vector_steps
    )
    if "vector_estimator" in completed:
        require(vector_steps == 4, "completed vector stage lacks four steps")
    elif ORDERED_STAGES.index("vector_estimator") > len(completed):
        require(vector_steps == 0, "vector steps completed before vector stage")

    success = report["status"] == "four_component_chain_stage_diagnostic_pass"
    if success:
        require(
            report["diagnostic_outcome"] == "all_stages_completed",
            "success diagnostic outcome mismatch",
        )
        require(completed == ORDERED_STAGES, "success did not complete all stages")
        require(all(sessions.values()), "success did not create all sessions")
        require(vector_steps == 4, "success did not complete four vector steps")
        return {"kind": "pass", "completed_stages": completed}

    require(
        report["status"] == "failed_closed_stage_diagnostic",
        "unexpected diagnostic status",
    )
    require(
        report["diagnostic_outcome"] == "failure_localized",
        "failure diagnostic outcome mismatch",
    )
    stage = report["failure_stage"]
    phase = report["failure_phase"]
    allowed_stages = set(ORDERED_STAGES) | {"report_assembly"}
    require(stage in allowed_stages, "invalid failure_stage")
    if stage == "payload_install":
        require(phase == "install_and_hash", "invalid payload failure phase")
    elif stage == "runtime_initialization":
        require(phase in RUNTIME_PHASES, "invalid runtime failure phase")
    elif stage in SESSION_STAGES:
        require(phase in COMPONENT_PHASES, "invalid component failure phase")
    else:
        require(phase == "output_validation", "invalid report-assembly phase")
    expected_prefix = (
        ORDERED_STAGES
        if stage == "report_assembly"
        else ORDERED_STAGES[: ORDERED_STAGES.index(stage)]
    )
    require(completed == expected_prefix, "failure completed-stage prefix mismatch")
    if stage in SESSION_STAGES:
        if phase == "session_create":
            require(sessions[stage] is False, "failed session was marked created")
        else:
            require(sessions[stage] is True, "post-create failure lacks session flag")
    has_vector_index = "failure_vector_step" in report
    expected_vector_index = stage == "vector_estimator" and phase in {
        "execute",
        "output_validation",
    }
    require(has_vector_index == expected_vector_index, "vector failure index presence mismatch")
    if expected_vector_index:
        step = integer(report["failure_vector_step"], "failure_vector_step")
        require(step < 4, "failure vector step is out of range")
        require(vector_steps == step, "completed vector steps do not precede failure step")
    elif stage == "vector_estimator":
        require(vector_steps == 0, "vector steps exist before execution")
    elif stage in {"vocoder", "report_assembly"}:
        require(vector_steps == 4, "post-vector failure lacks four completed steps")
    else:
        require(vector_steps == 0, "pre-vector failure has completed vector steps")
    return {
        "kind": "failure",
        "failure_stage": stage,
        "failure_phase": phase,
        "failure_vector_step": report.get("failure_vector_step"),
        "completed_stages": completed,
    }


def verify_report(
    report: dict[str, Any],
    spec: dict[str, Any],
    source: dict[str, Any],
    expected_run_id: str,
) -> dict[str, Any]:
    success = report.get("status") == "four_component_chain_stage_diagnostic_pass"
    persistence_keys = {
        "evidence_schema_version",
        "package",
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
    progress_keys = {
        "diagnostic_schema_version",
        "diagnostic_marker",
        "completed_stages",
        "completed_stage_count",
        "completed_stage_reports",
        "sessions_created",
        "completed_vector_steps",
        "completed_vector_step_reports",
        "diagnostic_outcome",
    }
    if success:
        base_keys = set(base_device.EXPECTED_REPORT_KEYS)
        exact_keys(report, base_keys | progress_keys, "successful diagnostic report")
    else:
        failure_keys = {
            "schema_version",
            "gate_id",
            "status",
            "error_type",
            "error",
            "cpu_ep_fallback_disabled",
            "full_neural_chain",
            "chained_pcm_tensor",
            *FALSE_CLAIMS,
            "failure_stage",
            "failure_phase",
        }
        if "failure_vector_step" in report:
            failure_keys.add("failure_vector_step")
        exact_keys(
            report,
            failure_keys | persistence_keys | progress_keys,
            "failed diagnostic report",
        )
        require(
            isinstance(report["error_type"], str) and report["error_type"],
            "failure error_type is empty",
        )
        require(isinstance(report["error"], str), "failure error is not a string")
    timing = verify_persistence_metadata(report, spec, expected_run_id)
    require(report["cpu_ep_fallback_disabled"] is True, "CPU fallback was not disabled")
    for field, expected in FALSE_CLAIMS.items():
        require(report[field] is expected, f"claim boundary crossed: {field}")
    result = verify_progress(report, spec, source)
    if success:
        require(report["full_neural_chain"] is True, "success lacks full chain")
        require(report["chained_pcm_tensor"] is True, "success lacks chained PCM")
        base_device.verify_chain(report, source)
        base_device.verify_timing(report)
    else:
        require(report["full_neural_chain"] is False, "failure claims full chain")
        require(report["chained_pcm_tensor"] is False, "failure claims chained PCM")
    return {**result, **timing}


def verify_local_report(
    local: dict[str, Any], spec: dict[str, Any], spec_path: Path
) -> None:
    require(local.get("schema_version") == 3, "local verifier schema mismatch")
    require(local.get("gate_id") == GATE_ID, "local verifier gate mismatch")
    require(local.get("package") == PACKAGE, "local verifier package mismatch")
    require(local.get("status") == "pass", "local whole-APK verifier did not pass")
    require(local.get("device_mutated") is False, "local verifier mutated device")
    require(local.get("apk", {}).get("bytes") == spec["apk"]["bytes"], "local APK bytes")
    require(local.get("apk", {}).get("sha256") == spec["apk"]["sha256"], "local APK SHA")
    authoritative = local.get("authoritative_spec", {})
    require(authoritative.get("bytes") == spec_path.stat().st_size, "local spec bytes")
    require(authoritative.get("sha256") == sha256(spec_path), "local spec SHA")
    require(
        authoritative.get("whole_apk_identity_required_before_zip_inspection") is True,
        "local verifier did not bind whole APK first",
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
    _, source = host_verifier.validate_spec(spec, spec_path)
    verify_local_report(local, spec, spec_path)
    result = verify_report(report, spec, source, expected_run_id)
    exact_keys(
        evidence,
        {
            "schema_version",
            "gate_id",
            "package",
            "run_id",
            "prelaunch",
            "launch",
            "device",
            "installed_package",
            "accepted_packages_before",
            "accepted_packages_after",
            "authoritative_spec",
            "local_verifier_report",
            "device_report",
            "collection",
            "publication",
            "status",
            "claim_boundary",
        },
        "collector evidence",
    )
    require(evidence["schema_version"] == 2, "collector schema mismatch")
    require(evidence["gate_id"] == GATE_ID, "collector gate mismatch")
    require(evidence["package"] == PACKAGE, "collector package mismatch")
    require(evidence["run_id"] == expected_run_id, "collector run_id mismatch")
    prelaunch = exact_keys(
        evidence["prelaunch"],
        {
            "package_force_stopped",
            "latest_report_deleted",
            "partial_report_deleted",
            "latest_report_absent_before_launch",
            "partial_report_absent_before_launch",
            "cleanup_completed_at_unix_ms",
        },
        "prelaunch evidence",
    )
    for field in (
        "package_force_stopped",
        "latest_report_deleted",
        "partial_report_deleted",
        "latest_report_absent_before_launch",
        "partial_report_absent_before_launch",
    ):
        require(prelaunch[field] is True, f"prelaunch hygiene failed: {field}")
    cleanup_ms = integer(
        prelaunch["cleanup_completed_at_unix_ms"], "cleanup_completed_at_unix_ms", minimum=1
    )
    launch = exact_keys(
        evidence["launch"],
        {
            "component",
            "run_id",
            "run_probe_extra",
            "requested_at_unix_ms",
            "am_start_output_sha256",
        },
        "launch evidence",
    )
    require(launch["component"] == PACKAGE + "/.ProbeActivity", "launch component")
    require(launch["run_id"] == expected_run_id, "launch run_id mismatch")
    require(launch["run_probe_extra"] is True, "run_probe extra missing")
    require(
        integer(launch["requested_at_unix_ms"], "launch requested_at", minimum=cleanup_ms)
        >= cleanup_ms,
        "launch preceded cleanup",
    )
    require_sha256(launch["am_start_output_sha256"], "am start output")
    installed = exact_keys(
        evidence["installed_package"],
        {"package", "apk_path", "bytes", "sha256", "version_code", "version_name"},
        "installed package",
    )
    require(installed["package"] == PACKAGE, "installed package mismatch")
    require(
        isinstance(installed["apk_path"], str)
        and installed["apk_path"].startswith("/data/app/")
        and installed["apk_path"].endswith("/base.apk"),
        "installed APK path mismatch",
    )
    require(installed["bytes"] == spec["apk"]["bytes"], "installed APK bytes mismatch")
    require(installed["sha256"] == spec["apk"]["sha256"], "installed APK SHA mismatch")
    require(installed["version_code"] == 1, "installed versionCode mismatch")
    require(installed["version_name"] == VERSION_NAME, "installed versionName mismatch")
    before = evidence["accepted_packages_before"]
    after = evidence["accepted_packages_after"]
    require(isinstance(before, dict) and set(before) == set(spec["accepted_packages"]), "accepted before inventory")
    require(after == before, "accepted packages changed during diagnostic run")
    for package, record in before.items():
        exact_keys(record, {"path", "bytes", "sha256"}, "accepted package " + package)
        require(record["sha256"] == spec["accepted_packages"][package]["sha256"], "accepted package SHA mismatch")

    device = exact_keys(
        evidence["device"], {"serial", "model", "soc", "abi", "airplane_mode_on"}, "device"
    )
    require(isinstance(device["serial"], str) and device["serial"], "device serial empty")
    require(device["model"] == "NX809J", "device model mismatch")
    require(device["soc"] == "SM8850", "device SoC mismatch")
    require(device["abi"] == "arm64-v8a", "device ABI mismatch")
    require(device["airplane_mode_on"] == 1, "airplane mode was not enabled")
    verify_identity_record(evidence["authoritative_spec"], spec_path, "authoritative spec")
    verify_identity_record(evidence["local_verifier_report"], local_path, "local verifier")
    verify_identity_record(evidence["device_report"], report_path, "device report")
    report_record = evidence["device_report"]
    require(report_record["path"] == str(report_path), "device report archive path mismatch")
    collection = exact_keys(
        evidence["collection"],
        {
            "collected_at_unix_ms",
            "collected_at_utc",
            "matching_run_id",
            "matching_completion_marker",
            "partial_report_absent_after_capture",
            "diagnostic_status",
            "failure_stage",
            "failure_phase",
            "failure_vector_step",
        },
        "collection summary",
    )
    collected_ms = integer(collection["collected_at_unix_ms"], "collected_at", minimum=1)
    require(
        utc_unix_ms(collection["collected_at_utc"], "collected_at_utc") == collected_ms,
        "collection UTC/unix mismatch",
    )
    require(collection["matching_run_id"] is True, "collector did not correlate run_id")
    require(
        collection["matching_completion_marker"] is True,
        "collector did not correlate completion marker",
    )
    require(
        collection["partial_report_absent_after_capture"] is True,
        "partial report remained after capture",
    )
    require(collection["diagnostic_status"] == report["status"], "status summary mismatch")
    require(collection["failure_stage"] == report.get("failure_stage"), "failure stage summary mismatch")
    require(collection["failure_phase"] == report.get("failure_phase"), "failure phase summary mismatch")
    require(collection["failure_vector_step"] == report.get("failure_vector_step"), "failure step summary mismatch")
    require(
        evidence["publication"]
        == {"current_alias_published": False, "verification_published": False},
        "collector published an unverified alias",
    )
    require(evidence["status"] == "collected_unverified", "collector status mismatch")
    require(evidence_path.is_file(), "collector evidence path is unreadable")
    status = (
        "diagnostic_chain_pass_verified"
        if result["kind"] == "pass"
        else "diagnostic_failure_localized_verified"
    )
    return {
        "schema_version": 2,
        "gate_id": GATE_ID,
        "package": PACKAGE,
        "run_id": expected_run_id,
        "status": status,
        "diagnostic_result": result,
        "apk": spec["apk"],
        "spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": sha256(spec_path),
        },
        "local_verifier_report": {
            "path": str(local_path),
            "bytes": local_path.stat().st_size,
            "sha256": sha256(local_path),
        },
        "device_report": {
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": sha256(report_path),
        },
        "collector_evidence": {
            "path": str(evidence_path),
            "bytes": evidence_path.stat().st_size,
            "sha256": sha256(evidence_path),
        },
        "promotion": {
            "full_tts": False,
            "audio_quality": False,
            "latency": False,
            "production_integration": False,
        },
        "claim_boundary": (
            "Stage-localized compatibility evidence only. Even a chain pass does not "
            "promote audio quality, latency, language identity, or production readiness."
        ),
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
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    require(RUN_ID_RE.fullmatch(arguments.run_id) is not None, "invalid run_id")
    spec_path = arguments.spec.expanduser().resolve()
    local_path = arguments.local_verify.expanduser().resolve()
    report_path = arguments.report.expanduser().resolve()
    evidence_path = arguments.evidence.expanduser().resolve()
    verified = verify_documents(
        load_object(spec_path),
        load_object(local_path),
        load_object(report_path),
        load_object(evidence_path),
        expected_run_id=arguments.run_id,
        spec_path=spec_path,
        local_path=local_path,
        report_path=report_path,
        evidence_path=evidence_path,
    )
    encoded = (json.dumps(verified, indent=2, sort_keys=True) + "\n").encode()
    if arguments.output is not None:
        atomic_write(arguments.output.expanduser().resolve(), encoded)
    print(encoded.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
