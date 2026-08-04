#!/usr/bin/env python3
"""Fail-closed verification of the bundled-WAV frontend device evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/streaming_response/whisper_v81"
APK_VERIFIER_PATH = ROOT / "scripts/verify_whisper_android_frontend_apk.py"
DEFAULT_FRONTEND_SPEC = REPORT_ROOT / "nubia_whisper_android_frontend_probe_spec.json"
DEFAULT_BASE_SPEC = REPORT_ROOT / "nubia_whisper_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = REPORT_ROOT / "whisper_android_frontend_local_verify.json"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_PROCESS_LIBRARIES = {
    "libonnxruntime.so",
    "libonnxruntime4j_jni.so",
    "libQnnHtp.so",
    "libQnnHtpV81Stub.so",
    "libQnnSystem.so",
    "libcdsprpc.so",
    "libwhisper_frontend.so",
}


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import verification helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


apk_verifier = load_module(
    "verify_whisper_android_frontend_apk",
    APK_VERIFIER_PATH,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise RuntimeError(f"invalid JSON numeric constant in {path}: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def positive_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    encoded = float(value)
    require(math.isfinite(encoded) and encoded > 0.0, f"invalid timing: {label}")
    return encoded


def parse_utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), f"invalid UTC {label}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"invalid UTC {label}") from error


def parse_process_snapshot(status: str, maps: str, fds: str) -> dict[str, Any]:
    uid_match = re.search(
        r"^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$",
        status,
        re.MULTILINE,
    )
    require(uid_match is not None, "process status is missing Uid fields")
    libraries = sorted(
        {
            Path(token).name
            for line in maps.splitlines()
            for token in line.split()
            if token.startswith("/") and ".so" in Path(token).name
        }
    )
    dma_markers = ("dmabuf", "dma_buf", "dma-buf")
    return {
        "uid_fields": [int(value) for value in uid_match.groups()],
        "loaded_libraries": libraries,
        "fastrpc_cdsp_fd_open": "/dev/fastrpc-cdsp" in fds,
        "qcom_dmabuf_count": sum(
            any(marker in line.lower() for marker in dma_markers)
            for line in fds.splitlines()
        ),
    }


def validate_member_inventory(
    actual: Any,
    expected_records: list[dict[str, Any]],
    *,
    prefix: str,
    label: str,
) -> None:
    require(isinstance(actual, list), f"local {label} inventory is not an array")
    expected = {
        prefix + record["path"]: {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in expected_records
    }
    require(len(actual) == len(expected), f"local {label} inventory mismatch")
    seen: set[str] = set()
    for record in actual:
        require(
            isinstance(record, dict)
            and set(record) == {"path", "bytes", "sha256", "compressed_bytes"},
            f"local {label} member schema mismatch",
        )
        path = record["path"]
        require(
            isinstance(path, str) and path in expected and path not in seen,
            f"local {label} member identity mismatch",
        )
        seen.add(path)
        require(
            record["bytes"] == expected[path]["bytes"]
            and record["sha256"] == expected[path]["sha256"],
            f"local {label} member identity mismatch: {path}",
        )
        compressed = record["compressed_bytes"]
        require(
            isinstance(compressed, int)
            and not isinstance(compressed, bool)
            and compressed >= 0,
            f"local {label} compressed byte count is invalid: {path}",
        )
    require(seen == set(expected), f"local {label} inventory mismatch")


def validate_local_verify_report(
    local_verify: dict[str, Any],
    frontend_spec: dict[str, Any],
    base_spec: dict[str, Any],
    *,
    frontend_spec_path: Path,
    base_spec_path: Path,
) -> None:
    require(isinstance(local_verify, dict), "local verifier report is not an object")
    profile = frontend_spec.get("frontend_profile", "v1")
    require(
        isinstance(profile, str) and profile in {"v1", "v2"},
        "unsupported local frontend profile",
    )
    expected_fields = {
        "schema_version",
        "gate_id",
        "base_gate_id",
        "package",
        "status",
        "apk",
        "trust_root",
        "archive",
        "binary",
        "verification_toolchain",
        "device_mutated",
        "claim_boundary",
    }
    if profile == "v2":
        expected_fields.add("frontend_profile")
    require(
        set(local_verify) == expected_fields,
        "local verifier report field inventory mismatch",
    )
    require(local_verify["schema_version"] == 1, "wrong local verifier schema")
    require(
        local_verify["gate_id"] == frontend_spec["gate_id"],
        "local frontend gate mismatch",
    )
    require(
        local_verify["base_gate_id"] == base_spec["gate_id"], "local base gate mismatch"
    )
    require(
        local_verify["package"] == frontend_spec["target"]["package"],
        "local package mismatch",
    )
    require(local_verify["status"] == "pass", "local frontend APK verification failed")
    if profile == "v2":
        require(
            local_verify["frontend_profile"] == "v2", "local frontend profile mismatch"
        )

    apk = local_verify["apk"]
    require(
        isinstance(apk, dict) and set(apk) == {"path", "bytes", "sha256"},
        "local APK identity schema mismatch",
    )
    require(
        isinstance(apk["path"], str)
        and bool(apk["path"])
        and apk["bytes"] == frontend_spec["apk"]["bytes"]
        and apk["sha256"] == frontend_spec["apk"]["sha256"],
        "local whole-APK trust-root mismatch",
    )

    trust_root = local_verify["trust_root"]
    require(
        isinstance(trust_root, dict)
        and set(trust_root)
        == {
            "path",
            "bytes",
            "sha256",
            "base_spec_sha256",
            "whole_apk_identity_required",
        },
        "local trust-root schema mismatch",
    )
    recorded_spec_path = trust_root["path"]
    require(
        isinstance(recorded_spec_path, str)
        and Path(recorded_spec_path).resolve() == frontend_spec_path.resolve(),
        "local frontend trust-root path mismatch",
    )
    require(
        trust_root["bytes"] == frontend_spec_path.stat().st_size
        and trust_root["sha256"] == sha256(frontend_spec_path),
        "local frontend trust-root identity mismatch",
    )
    require(
        trust_root["base_spec_sha256"] == sha256(base_spec_path),
        "local base trust-root identity mismatch",
    )
    require(
        trust_root["whole_apk_identity_required"] is True,
        "local verifier did not require whole-APK identity",
    )

    archive = local_verify["archive"]
    require(
        isinstance(archive, dict)
        and set(archive)
        == {
            "status",
            "base_payload_manifest_sha256",
            "base_verified_members",
            "frontend_manifest_sha256",
            "frontend_verified_members",
            "frontend_native_library",
        },
        "local archive evidence schema mismatch",
    )
    require(archive["status"] == "pass", "local archive verification failed")
    require(
        archive["base_payload_manifest_sha256"] == base_spec["payload_manifest_sha256"],
        "local base payload manifest mismatch",
    )
    require(
        archive["frontend_manifest_sha256"]
        == frontend_spec["frontend_manifest"]["sha256"],
        "local frontend manifest mismatch",
    )
    require(
        isinstance(archive["base_verified_members"], list),
        "local base member inventory is not an array",
    )
    validate_member_inventory(
        [
            record
            for record in archive["base_verified_members"]
            if isinstance(record, dict)
            and str(record.get("path", "")).startswith("lib/")
        ],
        base_spec["required_native_libraries"],
        prefix="lib/",
        label="base native-library",
    )
    base_asset_count = len(base_spec["required_assets"])
    require(
        len(archive["base_verified_members"])
        == base_asset_count + len(base_spec["required_native_libraries"]),
        "local base member inventory mismatch",
    )
    validate_member_inventory(
        [
            record
            for record in archive["base_verified_members"]
            if isinstance(record, dict)
            and str(record.get("path", "")).startswith("assets/")
        ],
        base_spec["required_assets"],
        prefix="assets/",
        label="base asset",
    )
    validate_member_inventory(
        archive["frontend_verified_members"],
        frontend_spec["frontend_assets"],
        prefix="assets/",
        label="frontend asset",
    )
    expected_frontend_library = {
        "path": apk_verifier.FRONTEND_LIBRARY_MEMBER,
        "bytes": frontend_spec["frontend_native_library"]["bytes"],
        "sha256": frontend_spec["frontend_native_library"]["sha256"],
        "elf_class": 64,
        "byte_order": "little_endian",
        "machine": "aarch64",
    }
    require(
        archive["frontend_native_library"] == expected_frontend_library,
        "local frontend JNI evidence mismatch",
    )

    binary = local_verify["binary"]
    require(
        isinstance(binary, dict)
        and set(binary) == {"android_manifest", "signing", "alignment"},
        "local binary evidence schema mismatch",
    )
    apk_policy = base_spec["apk_policy"]
    expected_manifest = {
        "status": "pass",
        "package": apk_policy["package"],
        "version_code": apk_policy["version_code"],
        "version_name": apk_policy["version_name"],
        "compile_sdk": apk_policy["compile_sdk"],
        "min_sdk": apk_policy["min_sdk"],
        "target_sdk": apk_policy["target_sdk"],
        "launcher": apk_policy["launcher"],
        "permissions": apk_policy["permissions"],
        "native_abis": apk_policy["native_abis"],
        "required_native_libraries": apk_policy["required_native_libraries"],
        "optional_native_libraries": apk_policy["optional_native_libraries"],
        "optional_features": apk_policy["required_optional_features"],
    }
    require(
        binary["android_manifest"] == expected_manifest,
        "local AndroidManifest evidence mismatch",
    )
    require(
        binary["signing"] == {"status": "pass", **apk_policy["signing"]},
        "local signing evidence mismatch",
    )
    alignment = binary["alignment"]
    require(
        isinstance(alignment, dict)
        and set(alignment) == {"status", "alignment_bytes", "output_sha256"}
        and alignment["status"] == "pass"
        and alignment["alignment_bytes"] == 4
        and isinstance(alignment["output_sha256"], str)
        and SHA256_RE.fullmatch(alignment["output_sha256"]) is not None,
        "local alignment evidence mismatch",
    )

    toolchain = local_verify["verification_toolchain"]
    expected_tool_names = set(base_spec["verification_toolchain"])
    require(
        isinstance(toolchain, dict)
        and set(toolchain) == expected_tool_names | {"java_home"},
        "local verification toolchain inventory mismatch",
    )
    for name, expected in base_spec["verification_toolchain"].items():
        actual = toolchain[name]
        require(
            isinstance(actual, dict)
            and set(actual) == {"path", "bytes", "sha256"}
            and isinstance(actual["path"], str)
            and Path(actual["path"]).name == expected["relative_path"]
            and actual["bytes"] == expected["bytes"]
            and actual["sha256"] == expected["sha256"],
            f"local verification tool mismatch: {name}",
        )
    require(
        isinstance(toolchain["java_home"], str)
        and Path(toolchain["java_home"]).is_absolute(),
        "local Java toolchain path is invalid",
    )
    require(local_verify["device_mutated"] is False, "local verifier mutated a device")
    require(
        local_verify["claim_boundary"]
        == (
            "Local APK identity and frontend-overlay packaging only; no Android "
            "execution, target numerical result, QNN execution, or device mutation."
        ),
        "local verifier claim boundary mismatch",
    )


def verify_frontend_fixture(
    name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    require(
        actual.get("wav_sha256") == expected["wav_sha256"], f"{name} WAV hash mismatch"
    )
    require(
        actual.get("wav_pcm_frames") == expected["wav_pcm_frames"],
        f"{name} PCM frame count mismatch",
    )
    require(
        actual.get("golden_tensor_sha256") == expected["golden_tensor_sha256"],
        f"{name} golden tensor mismatch",
    )
    produced = actual.get("produced_tensor_sha256")
    require(
        isinstance(produced, str) and SHA256_RE.fullmatch(produced) is not None,
        f"{name} produced tensor SHA-256 is invalid",
    )
    differing = actual.get("differing_uint16_values")
    maximum_delta = actual.get("max_uint16_delta")
    fraction = actual.get("differing_fraction")
    require(
        isinstance(differing, int)
        and not isinstance(differing, bool)
        and differing >= 0,
        f"{name} differing count invalid",
    )
    require(
        differing <= policy["max_differing_values"],
        f"{name} differing count exceeds bound",
    )
    require(
        isinstance(maximum_delta, int)
        and not isinstance(maximum_delta, bool)
        and 0 <= maximum_delta <= policy["max_uint16_delta"],
        f"{name} maximum delta exceeds bound",
    )
    require(
        isinstance(fraction, (int, float)) and not isinstance(fraction, bool),
        f"{name} differing fraction invalid",
    )
    require(
        abs(float(fraction) - differing / 240_000.0) <= 1.0e-12,
        f"{name} differing fraction/count mismatch",
    )
    require(
        float(fraction) <= policy["max_differing_fraction"],
        f"{name} differing fraction exceeds bound",
    )
    require(actual.get("bounded_gate_passed") is True, f"{name} bound did not pass")
    require(
        isinstance(actual.get("exact_sha256_match"), bool),
        f"{name} exact hash flag invalid",
    )
    exact = actual["exact_sha256_match"]
    require(
        exact == (produced == expected["golden_tensor_sha256"]),
        f"{name} exact hash flag/hash mismatch",
    )
    require(
        exact == (differing == 0 and maximum_delta == 0),
        f"{name} exact hash flag/delta mismatch",
    )
    compute_ms = positive_number(
        actual.get("frontend_compute_ms"), f"{name} frontend_compute_ms"
    )
    return {
        "produced_tensor_sha256": produced,
        "differing_uint16_values": differing,
        "differing_fraction": float(fraction),
        "max_uint16_delta": maximum_delta,
        "frontend_compute_ms": compute_ms,
    }


def verify_documents(
    frontend_spec: dict[str, Any],
    base_spec: dict[str, Any],
    local_verify: dict[str, Any],
    device_report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    expected_run_id: str,
    local_verify_sha256: str,
    device_report_sha256: str,
    frontend_spec_path: Path,
    base_spec_path: Path,
) -> dict[str, Any]:
    require(frontend_spec.get("schema_version") == 1, "wrong frontend spec schema")
    require(base_spec.get("schema_version") == 2, "wrong base spec schema")
    validate_local_verify_report(
        local_verify,
        frontend_spec,
        base_spec,
        frontend_spec_path=frontend_spec_path,
        base_spec_path=base_spec_path,
    )
    gate_id = frontend_spec["gate_id"]
    base_gate_id = frontend_spec["base_probe"]["gate_id"]
    package = frontend_spec["target"]["package"]
    require(base_spec.get("gate_id") == base_gate_id, "base gate mismatch")
    require(
        RUN_ID_RE.fullmatch(expected_run_id) is not None, "expected run_id is invalid"
    )

    require(
        local_verify.get("status") == "pass", "local frontend APK verification failed"
    )
    require(local_verify.get("gate_id") == gate_id, "local frontend gate mismatch")
    require(
        local_verify.get("base_gate_id") == base_gate_id, "local base gate mismatch"
    )
    require(local_verify.get("package") == package, "local package mismatch")
    require(
        local_verify.get("apk")
        == {
            "path": local_verify["apk"].get("path"),
            "bytes": frontend_spec["apk"]["bytes"],
            "sha256": frontend_spec["apk"]["sha256"],
        },
        "local whole-APK trust-root mismatch",
    )

    require(evidence.get("schema_version") == 2, "wrong evidence schema")
    require(evidence.get("gate_id") == gate_id, "evidence gate mismatch")
    require(evidence.get("package") == package, "evidence package mismatch")
    require(evidence.get("run_id") == expected_run_id, "evidence run_id mismatch")
    require(
        evidence.get("run", {}).get("requested_run_id") == expected_run_id
        and evidence.get("run", {}).get("trigger") == "automation_intent",
        "collector run contract mismatch",
    )
    require(
        evidence.get("partial_report_absent") is True, "partial report still exists"
    )
    expected_report_path = frontend_spec["device_output_contract"]["report_path"]
    require(
        evidence["device_report"].get("device_path") == expected_report_path,
        "device report path mismatch",
    )
    require(
        evidence["device_report"].get("partial_device_path")
        == expected_report_path + ".partial"
        and evidence["device_report"].get("partial_exists_after_capture") is False,
        "partial report publication contract failed",
    )
    require(
        evidence["local_verifier_report"]["sha256"] == local_verify_sha256,
        "local verifier report SHA-256 mismatch",
    )
    require(
        evidence["device_report"]["sha256"] == device_report_sha256,
        "app-private report SHA-256 mismatch",
    )
    require(evidence["device_report"]["bytes"] > 0, "device report is empty")

    installed = evidence["installed_package"]
    require(installed.get("package") == package, "installed package mismatch")
    require(
        installed.get("bytes") == frontend_spec["apk"]["bytes"],
        "installed APK bytes mismatch",
    )
    require(
        installed.get("sha256") == frontend_spec["apk"]["sha256"],
        "installed APK hash mismatch",
    )
    require(
        installed.get("version_code") == frontend_spec["apk_policy"]["version_code"],
        "installed version code mismatch",
    )
    require(
        installed.get("version_name") == frontend_spec["apk_policy"]["version_name"],
        "installed version name mismatch",
    )
    uid = installed.get("uid")
    require(isinstance(uid, int) and uid >= 10_000, "installed UID is invalid")

    device = evidence["device"]
    require(
        device.get("model") == frontend_spec["target"]["model"], "wrong device model"
    )
    require(device.get("soc") == frontend_spec["target"]["soc"], "wrong device SoC")
    require(device.get("abi") == frontend_spec["target"]["abi"], "wrong device ABI")
    require(device.get("airplane_mode_on") == 1, "airplane-mode gate failed")

    for accepted_package, expected in frontend_spec["accepted_packages"].items():
        require(
            evidence["accepted_packages_before"].get(accepted_package, {}).get("sha256")
            == expected["sha256"],
            f"accepted APK pre-run hash mismatch: {accepted_package}",
        )
        require(
            evidence["accepted_packages"].get(accepted_package, {}).get("sha256")
            == expected["sha256"],
            f"accepted APK hash mismatch: {accepted_package}",
        )

    process = evidence["process"]
    require(process.get("pid") == installed.get("pid"), "process PID mismatch")
    require(process.get("uid_fields") == [uid, uid, uid, uid], "process UID mismatch")
    require(process.get("report_owner_uid") == uid, "report owner UID mismatch")
    require(device_report.get("uid") == uid, "app report UID mismatch")
    require(
        REQUIRED_PROCESS_LIBRARIES.issubset(set(process.get("loaded_libraries", []))),
        "required QNN/ORT/frontend libraries are not all mapped",
    )
    require(process.get("fastrpc_cdsp_fd_open") is True, "FastRPC CDSP fd is missing")
    require(process.get("qcom_dmabuf_count", 0) > 0, "QCOM DMA buffers are missing")
    raw_artifacts = process.get("raw_artifacts")
    require(
        isinstance(raw_artifacts, dict)
        and set(raw_artifacts) == {"status", "maps", "fd"},
        "raw process artifact inventory mismatch",
    )
    for label, record in raw_artifacts.items():
        require(isinstance(record, dict), f"raw process artifact invalid: {label}")
        artifact_name = record.get("path")
        require(
            isinstance(artifact_name, str)
            and artifact_name
            and Path(artifact_name).name == artifact_name,
            f"raw process artifact path invalid: {label}",
        )
        require(
            isinstance(record.get("bytes"), int)
            and not isinstance(record["bytes"], bool)
            and record["bytes"] > 0,
            f"raw process artifact bytes invalid: {label}",
        )
        require(
            isinstance(record.get("sha256"), str)
            and SHA256_RE.fullmatch(record["sha256"]) is not None,
            f"raw process artifact SHA-256 invalid: {label}",
        )
    installed_frontend = evidence["installed_frontend_library"]
    expected_frontend = frontend_spec["frontend_native_library"]
    require(
        installed_frontend.get("bytes") == expected_frontend["bytes"],
        "installed frontend JNI bytes mismatch",
    )
    require(
        installed_frontend.get("sha256") == expected_frontend["sha256"],
        "installed frontend JNI hash mismatch",
    )

    require(
        device_report.get("evidence_schema_version") == 1,
        "app evidence schema mismatch",
    )
    require(device_report.get("gate_id") == base_gate_id, "app base gate mismatch")
    require(device_report.get("package") == package, "app report package mismatch")
    require(
        device_report.get("run_id") == expected_run_id, "app report run_id mismatch"
    )
    contract = frontend_spec["device_output_contract"]
    require(
        device_report.get("run_trigger") == contract["run_trigger"],
        "run trigger mismatch",
    )
    require(device_report.get("report_complete") is True, "report is incomplete")
    require(
        device_report.get("completion_marker") == contract["completion_marker"],
        "completion marker mismatch",
    )
    require(
        device_report.get("report_path") == contract["report_path"],
        "report path mismatch",
    )
    require(
        device_report.get("accepted_packages_mutated") is False,
        "accepted-package claim violated",
    )
    require(
        device_report.get("real_audio_transcription") is False,
        "real-audio claim violated",
    )
    started_ms = device_report.get("started_at_unix_ms")
    completed_ms = device_report.get("completed_at_unix_ms")
    require(
        isinstance(started_ms, int)
        and not isinstance(started_ms, bool)
        and isinstance(completed_ms, int)
        and not isinstance(completed_ms, bool),
        "unix timestamps missing",
    )
    require(completed_ms >= started_ms, "report completion precedes start")
    elapsed_wall_ms = device_report.get("elapsed_wall_ms")
    require(
        isinstance(elapsed_wall_ms, int)
        and not isinstance(elapsed_wall_ms, bool)
        and elapsed_wall_ms == completed_ms - started_ms,
        "wall elapsed time mismatch",
    )
    positive_number(device_report.get("elapsed_monotonic_ms"), "elapsed_monotonic_ms")
    started_utc = parse_utc(device_report.get("started_at_utc"), "start")
    completed_utc = parse_utc(device_report.get("completed_at_utc"), "completion")
    require(completed_utc >= started_utc, "UTC timestamps are reversed")
    require(
        round(started_utc.timestamp() * 1000) == started_ms
        and round(completed_utc.timestamp() * 1000) == completed_ms,
        "UTC/unix timestamps disagree",
    )

    runtime = base_spec["embedded_manifest"]["runtime"]
    require(
        device_report.get("ort_version") == runtime["onnx_runtime"],
        "ORT version mismatch",
    )
    require(
        runtime["qnn_provider"] in device_report.get("available_providers", []),
        "QNN provider unavailable",
    )
    require(
        device_report.get("qnn_provider_requested") is True, "QNN was not requested"
    )
    require(
        device_report.get("cpu_ep_fallback_disabled") is True,
        "CPU EP fallback was not disabled",
    )
    require(
        device_report.get("host_transport_required") is False, "host transport required"
    )
    require(
        device_report.get("encoder_session_created") is True, "encoder session missing"
    )
    require(
        device_report.get("decoder_session_created") is True, "decoder session missing"
    )
    require(
        device_report.get("status") == "compatibility_probe_pass", "probe did not pass"
    )
    require(
        isinstance(device_report.get("claim_boundary"), str)
        and "not final" in device_report["claim_boundary"],
        "top-level claim boundary missing",
    )

    backend_path = device_report.get("backend_path", "")
    code_root = installed["apk_path"].removesuffix("/base.apk")
    require(
        backend_path.startswith(code_root)
        and backend_path.endswith("/lib/arm64/libQnnHtp.so"),
        "QNN backend is not installed/app-owned",
    )
    model_storage = device_report.get("model_storage", "")
    require(
        model_storage.startswith(f"/data/user/0/{package}/files/"),
        "model storage is not app-private",
    )
    resident = evidence["resident_payload"]
    require(resident.get("root") == model_storage, "resident payload root mismatch")
    expected_models = {
        item["path"].removeprefix("whisper_probe/models/"): item["bytes"]
        for item in base_spec["required_assets"]
        if item["path"].startswith("whisper_probe/models/")
    }
    require(
        resident.get("model_files") == expected_models,
        "resident model inventory mismatch",
    )
    require(resident.get("kib", 0) > 0, "resident payload size invalid")

    frontend_report = device_report["bundled_audio_frontend"]
    frontend_contract = contract["frontend"]
    require(frontend_report.get("gate_id") == gate_id, "nested frontend gate mismatch")
    require(
        frontend_report.get("manifest_sha256")
        == frontend_spec["frontend_manifest"]["sha256"],
        "frontend manifest hash mismatch",
    )
    require(
        frontend_report.get("implementation") == frontend_contract["implementation"],
        "frontend implementation mismatch",
    )
    require(
        frontend_report.get("feature_dtype") == frontend_contract["feature_dtype"],
        "frontend dtype mismatch",
    )
    require(
        frontend_report.get("feature_shape") == frontend_contract["feature_shape"],
        "frontend shape mismatch",
    )
    require(
        frontend_report.get("feature_layout") == frontend_contract["feature_layout"],
        "frontend layout mismatch",
    )
    require(frontend_report.get("bounded_gate_passed") is True, "frontend bound failed")
    require(
        frontend_report.get("status") == "bundled_wav_frontend_bound_pass",
        "frontend status mismatch",
    )
    require(
        frontend_report.get("native_library_sha256")
        == contract["native_frontend_sha256"],
        "runtime frontend JNI hash mismatch",
    )
    require(
        frontend_report.get("real_audio_frontend") is False,
        "frontend claim boundary violated",
    )
    require(
        frontend_report.get("real_audio_transcription") is False,
        "frontend transcription claim violated",
    )
    require(
        frontend_report.get("microphone_input") is False, "microphone claim violated"
    )
    require(
        isinstance(frontend_report.get("claim_boundary"), str)
        and "does not enable microphone" in frontend_report["claim_boundary"],
        "frontend claim boundary missing",
    )
    policy = frontend_report["verification_policy"]
    for field in ("max_uint16_delta", "max_differing_values", "max_differing_fraction"):
        require(
            policy.get(field) == frontend_contract[field],
            f"frontend policy mismatch: {field}",
        )
    require(
        policy.get("target_acceptance_promoted") is False,
        "target bound was improperly promoted",
    )
    fixtures = frontend_report.get("fixtures")
    expected_fixture_names = set(frontend_contract["fixtures"])
    require(
        isinstance(fixtures, dict)
        and expected_fixture_names
        and "en" in expected_fixture_names
        and set(fixtures) == expected_fixture_names,
        "frontend fixture inventory mismatch",
    )
    fixture_results = {
        name: verify_frontend_fixture(
            name,
            fixtures[name],
            expected,
            frontend_contract,
        )
        for name, expected in frontend_contract["fixtures"].items()
    }
    frontend_timings = frontend_report["frontend_compute_ms"]
    expected_timing_names = expected_fixture_names | {"aggregate"}
    require(
        isinstance(frontend_timings, dict)
        and set(frontend_timings) == expected_timing_names,
        "frontend timing inventory mismatch",
    )
    for name in sorted(expected_timing_names):
        positive_number(frontend_timings.get(name), f"frontend_compute_ms.{name}")
    for name in sorted(expected_fixture_names):
        require(
            float(frontend_timings[name])
            == fixture_results[name]["frontend_compute_ms"],
            f"frontend {name} timing disagrees with fixture",
        )
    positive_number(
        frontend_report.get("frontend_verification_total_ms"),
        "frontend_verification_total_ms",
    )
    expected_aggregate = round(
        sum(float(frontend_timings[name]) for name in expected_fixture_names), 3
    )
    require(
        abs(float(frontend_timings["aggregate"]) - expected_aggregate) <= 0.001,
        "frontend aggregate timing mismatch",
    )
    require(
        float(frontend_report["frontend_verification_total_ms"]) + 0.002
        >= float(frontend_timings["aggregate"]),
        "frontend verification total is below compute aggregate",
    )

    decode = device_report["synthetic_decode"]
    require(
        decode.get("feature_source") == contract["feature_source"],
        "wrong encoder feature source",
    )
    require(decode.get("feature_dtype") == "uint16", "decode feature dtype mismatch")
    require(decode.get("encoder_execute_passed") is True, "encoder execution failed")
    require(decode.get("decoder_execute_passed") is True, "decoder execution failed")
    require(
        decode.get("decoder_steps") == contract["decoder"]["steps"],
        "decoder step mismatch",
    )
    require(
        decode.get("attention_unmasked_indices")
        == contract["decoder"]["attention_unmask_order"],
        "attention update mismatch",
    )
    require(
        len(decode.get("token_ids", [])) == contract["decoder"]["steps"],
        "token count mismatch",
    )
    require(
        all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in decode["token_ids"]
        ),
        "token IDs are invalid",
    )
    require(
        decode.get("first_token_id") == decode["token_ids"][0],
        "first token ID mismatch",
    )
    require(
        decode.get("semantic_transcript_claimed") is False,
        "semantic claim is forbidden",
    )
    positive_number(decode.get("encoder_execute_ms"), "decode.encoder_execute_ms")
    positive_number(decode.get("decoder_execute_ms"), "decode.decoder_execute_ms")
    positive_number(decode.get("decoder_loop_ms"), "decode.decoder_loop_ms")
    decode_steps = decode.get("decoder_step_ms", [])
    require(
        len(decode_steps) == contract["decoder"]["steps"],
        "decode step timing count mismatch",
    )
    for index, value in enumerate(decode_steps):
        positive_number(value, f"decode.decoder_step_ms[{index}]")
    require(
        abs(float(decode["decoder_execute_ms"]) - round(sum(decode_steps), 3)) <= 0.002,
        "decode aggregate/step timing mismatch",
    )
    require(
        float(decode["decoder_loop_ms"]) + 0.002 >= float(decode["decoder_execute_ms"]),
        "decode loop is below model execution time",
    )

    timings = device_report["timing_ms"]
    for field in (
        "encoder_session_load",
        "decoder_session_load",
        "encoder_execute",
        "decoder_execute",
        "decoder_loop",
        "execute",
        "total",
    ):
        positive_number(timings.get(field), f"timing_ms.{field}")
    top_steps = timings.get("decoder_step", [])
    require(
        len(top_steps) == contract["decoder"]["steps"],
        "top-level decoder step timing count mismatch",
    )
    for index, value in enumerate(top_steps):
        positive_number(value, f"timing_ms.decoder_step[{index}]")
    require(top_steps == decode_steps, "nested/top-level decoder step timings differ")
    require(
        timings["encoder_execute"] == decode["encoder_execute_ms"]
        and timings["decoder_execute"] == decode["decoder_execute_ms"]
        and timings["decoder_loop"] == decode["decoder_loop_ms"],
        "nested/top-level execute timings differ",
    )
    require(
        float(timings["execute"]) + 0.002
        >= float(timings["encoder_execute"]) + float(timings["decoder_loop"]),
        "overall execute timing is inconsistent",
    )

    return {
        "schema_version": 1,
        "gate_id": gate_id,
        "base_gate_id": base_gate_id,
        "package": package,
        "run_id": expected_run_id,
        "status": "pass",
        "installed_apk_sha256": installed["sha256"],
        "frontend_native_sha256": installed_frontend["sha256"],
        "frontend_fixtures": fixture_results,
        "feature_source": decode["feature_source"],
        "decoder_steps": decode["decoder_steps"],
        "token_ids": decode["token_ids"],
        "timing_ms": timings,
        "cpu_ep_fallback_disabled": True,
        "fastrpc_cdsp_fd_open": True,
        "accepted_packages_unchanged": True,
        "report_complete": True,
        "real_audio_transcription": False,
    }


def verify_files(
    frontend_spec_path: Path,
    base_spec_path: Path,
    local_verify_path: Path,
    device_report_path: Path,
    evidence_path: Path,
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    frontend_spec = load_object(frontend_spec_path)
    base_spec = load_object(base_spec_path)
    local_verify = load_object(local_verify_path)
    device_report = load_object(device_report_path)
    evidence = load_object(evidence_path)
    frontend_profile = frontend_spec.get("frontend_profile", "v1")
    apk_verifier.validate_frontend_contract(
        frontend_spec,
        base_spec,
        frontend_profile=frontend_profile,
    )
    require(
        evidence["local_verifier_report"]["bytes"] == local_verify_path.stat().st_size,
        "local verifier report byte count mismatch",
    )
    require(
        evidence["device_report"]["bytes"] == device_report_path.stat().st_size,
        "device report byte count mismatch",
    )
    raw_values: dict[str, str] = {}
    for label, record in evidence.get("process", {}).get("raw_artifacts", {}).items():
        artifact_path = evidence_path.parent / record.get("path", "")
        require(artifact_path.is_file(), f"missing raw process artifact: {label}")
        require(
            artifact_path.stat().st_size == record.get("bytes"),
            f"raw process artifact byte count mismatch: {label}",
        )
        require(
            sha256(artifact_path) == record.get("sha256"),
            f"raw process artifact SHA-256 mismatch: {label}",
        )
        try:
            raw_values[label] = artifact_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"raw process artifact is not UTF-8: {label}") from error
    require(
        set(raw_values) == {"status", "maps", "fd"},
        "raw process artifacts are incomplete",
    )
    parsed_process = parse_process_snapshot(
        raw_values["status"], raw_values["maps"], raw_values["fd"]
    )
    for field, value in parsed_process.items():
        require(
            evidence["process"].get(field) == value,
            f"raw/parsed process evidence mismatch: {field}",
        )
    return verify_documents(
        frontend_spec,
        base_spec,
        local_verify,
        device_report,
        evidence,
        expected_run_id=expected_run_id,
        local_verify_sha256=sha256(local_verify_path),
        device_report_sha256=sha256(device_report_path),
        frontend_spec_path=frontend_spec_path,
        base_spec_path=base_spec_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-spec", type=Path, default=DEFAULT_FRONTEND_SPEC)
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = verify_files(
        arguments.frontend_spec.resolve(),
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
