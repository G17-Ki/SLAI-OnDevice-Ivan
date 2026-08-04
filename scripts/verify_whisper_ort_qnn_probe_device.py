#!/usr/bin/env python3
"""Fail-closed verification of isolated Whisper ORT-QNN device evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/streaming_response/whisper_v81"
DEFAULT_SPEC = REPORT_ROOT / "nubia_whisper_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = REPORT_ROOT / "nubia_whisper_ort_qnn_probe_local_verify.json"
DEFAULT_DEVICE_REPORT = (
    REPORT_ROOT / "nubia_whisper_ort_qnn_probe_device_rerun_20260729.json"
)
DEFAULT_DEVICE_EVIDENCE = (
    REPORT_ROOT / "nubia_whisper_ort_qnn_probe_device_current.json"
)
REQUIRED_PROCESS_LIBRARIES = {
    "libonnxruntime.so",
    "libonnxruntime4j_jni.so",
    "libQnnHtp.so",
    "libQnnHtpV81Stub.so",
    "libQnnSystem.so",
    "libcdsprpc.so",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_documents(
    spec: dict[str, Any],
    local_verify: dict[str, Any],
    device_report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    local_verify_sha256: str,
    device_report_sha256: str,
) -> dict[str, Any]:
    require(spec.get("schema_version") == 2, "wrong authoritative spec schema")
    require(evidence.get("schema_version") == 1, "wrong device-evidence schema")

    gate_id = spec["gate_id"]
    package = spec["target"]["package"]
    require(evidence.get("gate_id") == gate_id, "device gate_id mismatch")
    require(local_verify.get("gate_id") == gate_id, "local gate_id mismatch")
    require(device_report.get("gate_id") == gate_id, "app-report gate_id mismatch")
    require(evidence.get("package") == package, "device package mismatch")
    require(local_verify.get("package") == package, "local package mismatch")
    require(device_report.get("package") == package, "app-report package mismatch")

    require(local_verify.get("status") == "pass", "local APK verification did not pass")
    require(
        evidence["local_verifier_report"]["sha256"] == local_verify_sha256,
        "local verifier report SHA-256 mismatch",
    )
    require(
        evidence["device_report"]["sha256"] == device_report_sha256,
        "app-private report SHA-256 mismatch",
    )
    require(
        evidence["device_report"]["bytes"] > 0,
        "app-private report byte count is invalid",
    )

    installed = evidence["installed_package"]
    local_apk = local_verify["apk"]
    require(installed["package"] == package, "installed package mismatch")
    require(installed["sha256"] == local_apk["sha256"], "installed APK hash mismatch")
    require(installed["bytes"] == local_apk["bytes"], "installed APK size mismatch")
    require(installed["version_code"] == 1, "installed version code mismatch")
    require(
        installed["version_name"] == spec["apk_policy"]["version_name"],
        "installed version name mismatch",
    )

    device = evidence["device"]
    require(device["model"] == "NX809J", "wrong device model")
    require(device["soc"] == spec["target"]["soc"], "wrong device SoC")
    require(device["abi"] == spec["target"]["abi"], "wrong device ABI")
    require(device["airplane_mode_on"] == 1, "offline airplane-mode gate failed")

    accepted = evidence["accepted_packages"]
    for accepted_package, expected in spec["accepted_packages"].items():
        require(
            accepted.get(accepted_package, {}).get("sha256") == expected["sha256"],
            f"accepted APK hash mismatch: {accepted_package}",
        )

    uid = installed["uid"]
    process = evidence["process"]
    require(isinstance(uid, int) and uid >= 10000, "installed app UID is invalid")
    require(process["pid"] == installed["pid"], "process PID mismatch")
    require(process["uid_fields"] == [uid, uid, uid, uid], "process UID fields mismatch")
    require(device_report["uid"] == uid, "app-report UID mismatch")
    require(process["report_owner_uid"] == uid, "app-private report owner mismatch")

    runtime = spec["embedded_manifest"]["runtime"]
    require(device_report["ort_version"] == runtime["onnx_runtime"], "ORT version mismatch")
    require(
        runtime["qnn_provider"] in device_report["available_providers"],
        "QNN provider is unavailable",
    )
    require(device_report["qnn_provider_requested"] is True, "QNN was not requested")
    require(
        device_report["cpu_ep_fallback_disabled"] is True,
        "CPU EP fallback was not disabled",
    )
    require(device_report["host_transport_required"] is False, "host transport required")
    require(device_report["real_audio_transcription"] is False, "claim boundary violated")
    require(
        device_report["backend_path"].endswith("/lib/arm64/libQnnHtp.so"),
        "wrong QNN backend path",
    )
    code_root = installed["apk_path"].removesuffix("/base.apk")
    require(device_report["backend_path"].startswith(code_root), "backend is not app-owned")
    require(
        device_report["model_storage"].startswith(f"/data/user/0/{package}/files/"),
        "model storage is not app-private",
    )
    resident_payload = evidence["resident_payload"]
    require(
        resident_payload["root"] == device_report["model_storage"],
        "resident payload root mismatch",
    )
    require(resident_payload["kib"] > 0, "resident payload size is invalid")
    expected_model_files = {
        item["path"].removeprefix("whisper_probe/models/"): item["bytes"]
        for item in spec["required_assets"]
        if item["path"].startswith("whisper_probe/models/")
    }
    require(
        resident_payload["model_files"] == expected_model_files,
        "resident model file inventory mismatch",
    )

    require(device_report["encoder_session_created"] is True, "encoder session missing")
    require(device_report["decoder_session_created"] is True, "decoder session missing")
    decode = device_report["synthetic_decode"]
    require(decode["feature_source"] == "prequantized_fixture", "wrong feature source")
    require(decode["feature_dtype"] == "uint16", "wrong feature dtype")
    require(decode["encoder_execute_passed"] is True, "encoder execution failed")
    require(decode["decoder_execute_passed"] is True, "decoder execution failed")
    expected_decode = spec["embedded_manifest"]["decode_contract"]
    require(decode["decoder_steps"] == expected_decode["steps"], "decoder step mismatch")
    require(
        decode["attention_unmasked_indices"]
        == expected_decode["attention_unmask_order"],
        "attention-mask update mismatch",
    )
    require(
        len(decode["token_ids"]) == expected_decode["steps"],
        "generated token count mismatch",
    )
    require(decode["semantic_transcript_claimed"] is False, "semantic claim is forbidden")
    require(
        device_report["status"] == "compatibility_probe_pass",
        "app compatibility probe did not pass",
    )

    loaded = set(process["loaded_libraries"])
    require(
        REQUIRED_PROCESS_LIBRARIES.issubset(loaded),
        "required QNN/ORT/FastRPC libraries are not all mapped",
    )
    require(process["fastrpc_cdsp_fd_open"] is True, "FastRPC CDSP fd is missing")
    require(process["qcom_dmabuf_count"] > 0, "QCOM DMA buffers are missing")

    timings = device_report["timing_ms"]
    for name in ("encoder_session_load", "decoder_session_load", "execute", "total"):
        require(timings[name] > 0, f"invalid timing: {name}")

    return {
        "schema_version": 1,
        "gate_id": gate_id,
        "package": package,
        "status": "pass",
        "claim": "isolated_ort_qnn_htp_compatibility_probe",
        "uid": uid,
        "pid": process["pid"],
        "installed_apk_sha256": installed["sha256"],
        "cpu_ep_fallback_disabled": True,
        "fastrpc_cdsp_fd_open": True,
        "qcom_dmabuf_count": process["qcom_dmabuf_count"],
        "resident_payload_kib": resident_payload["kib"],
        "decoder_steps": decode["decoder_steps"],
        "token_ids": decode["token_ids"],
        "timing_ms": timings,
        "real_audio_transcription": False,
        "accepted_packages_unchanged": True,
    }


def verify_files(
    spec_path: Path,
    local_verify_path: Path,
    device_report_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    spec = load_object(spec_path)
    local_verify = load_object(local_verify_path)
    device_report = load_object(device_report_path)
    evidence = load_object(evidence_path)
    require(
        evidence["local_verifier_report"]["bytes"] == local_verify_path.stat().st_size,
        "local verifier report byte count mismatch",
    )
    require(
        evidence["device_report"]["bytes"] == device_report_path.stat().st_size,
        "app-private report byte count mismatch",
    )
    return verify_documents(
        spec,
        local_verify,
        device_report,
        evidence,
        local_verify_sha256=sha256(local_verify_path),
        device_report_sha256=sha256(device_report_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--device-report", type=Path, default=DEFAULT_DEVICE_REPORT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_DEVICE_EVIDENCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify_files(
        args.spec.resolve(),
        args.local_verify.resolve(),
        args.device_report.resolve(),
        args.evidence.resolve(),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
