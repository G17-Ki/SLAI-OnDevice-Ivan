#!/usr/bin/env python3
"""Fail-closed verification of Supertonic raw-ONNX QNN phone evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/model_sweep_20260729"
DEFAULT_SPEC = REPORT_ROOT / "supertonic_duration_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = REPORT_ROOT / "supertonic_duration_ort_qnn_apk_local_verify.json"
PACKAGE = "ai.liquid.lfmdemo.supertonicprobe"
GATE_ID = "supertonic3_duration_raw_onnx_ort_qnn_sm8850_probe_v1"
COMPLETION_MARKER = "SUPERTONIC_DURATION_PROBE_REPORT_COMPLETE_V1"
REQUIRED_LIBRARIES = {
    "libQnnHtp.so",
    "libQnnHtpV81Stub.so",
    "libQnnSystem.so",
    "libcdsprpc.so",
    "libonnxruntime.so",
    "libonnxruntime4j_jni.so",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--device-report", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report_path = (args.device_report or REPORT_ROOT / f"supertonic_duration_device_report_{args.run_id}.json").resolve()
    evidence_path = (args.evidence or REPORT_ROOT / f"supertonic_duration_device_evidence_{args.run_id}.json").resolve()
    spec_path = args.spec.resolve()
    local_path = args.local_verify.resolve()

    spec = load(spec_path)
    local = load(local_path)
    report = load(report_path)
    evidence = load(evidence_path)
    require(spec.get("gate_id") == GATE_ID, "authoritative gate_id mismatch")
    require(local.get("status") == "pass", "local whole-APK verification failed")
    require(local.get("apk") == {
        "path": local["apk"]["path"],
        "bytes": spec["apk"]["bytes"],
        "sha256": spec["apk"]["sha256"],
    }, "local APK trust-root mismatch")
    require(evidence.get("gate_id") == GATE_ID, "device-evidence gate_id mismatch")
    require(evidence.get("package") == PACKAGE, "device-evidence package mismatch")
    require(evidence.get("run_id") == args.run_id, "device-evidence run_id mismatch")

    report_record = evidence["app_private_report"]
    require(report_record["bytes"] == report_path.stat().st_size, "device-report size mismatch")
    require(report_record["sha256"] == sha256(report_path), "device-report hash mismatch")
    require(evidence["authoritative_spec"]["sha256"] == sha256(spec_path), "spec hash mismatch")
    require(evidence["local_verifier_report"]["sha256"] == sha256(local_path), "local verifier hash mismatch")

    installed = evidence["installed_package"]
    require(installed["package"] == PACKAGE, "installed package mismatch")
    require(installed["bytes"] == spec["apk"]["bytes"], "installed APK size mismatch")
    require(installed["sha256"] == spec["apk"]["sha256"], "installed APK hash mismatch")
    require(installed["version_code"] == spec["apk_policy"]["version_code"], "versionCode mismatch")
    require(installed["version_name"] == spec["apk_policy"]["version_name"], "versionName mismatch")
    require(installed["uid"] >= 10000, "invalid application UID")

    device = evidence["device"]
    require(device["model"] == "NX809J", "wrong phone model")
    require(device["soc"] == "SM8850", "wrong phone SoC")
    require(device["abi"] == "arm64-v8a", "wrong phone ABI")
    require(device["airplane_mode_on"] == 1, "airplane-mode gate failed")
    for package, expected in spec["accepted_packages"].items():
        require(
            evidence["accepted_packages"][package]["sha256"] == expected["sha256"],
            f"accepted APK changed: {package}",
        )

    require(report.get("gate_id") == GATE_ID, "app-report gate_id mismatch")
    require(report.get("package") == PACKAGE, "app-report package mismatch")
    require(report.get("run_id") == args.run_id, "app-report run_id mismatch")
    require(report.get("report_complete") is True, "app report is incomplete")
    require(report.get("completion_marker") == COMPLETION_MARKER, "completion marker mismatch")
    require(report.get("uid") == installed["uid"], "app-report UID mismatch")
    require(report.get("ort_version") == "1.25.0", "ORT version mismatch")
    require("QNNExecutionProvider" in report.get("available_providers", []), "QNN EP unavailable")
    require(report.get("qnn_provider_requested") is True, "QNN EP was not requested")
    require(report.get("cpu_ep_fallback_disabled") is True, "CPU EP fallback was not disabled")
    require(report.get("session_created") is True, "QNN session was not created")
    require(report.get("source_format") == "raw_onnx", "wrong model format")
    require(report.get("component") == "duration_predictor", "wrong component")
    require(report.get("host_transport_required") is False, "host transport required")
    require(report.get("status") == "duration_component_probe_pass", "duration probe did not pass")
    require(report.get("full_tts") is False and report.get("audio_output") is False, "claim boundary violated")
    execution = report["execution"]
    require(execution["finite"] is True, "duration output is non-finite")
    require(execution["numerical_sanity_passed"] is True, "duration numerical gate failed")
    require(execution["absolute_error"] <= execution["absolute_tolerance"], "duration error exceeds tolerance")
    require(report["timing_ms"]["session_load"] > 0, "invalid session-load timing")
    require(report["timing_ms"]["execute"] > 0, "invalid execution timing")

    process = evidence["process"]
    require(process["uid_fields"] == [installed["uid"]] * 4, "process UID mismatch")
    require(process["report_owner_uid"] == installed["uid"], "report owner mismatch")
    require(REQUIRED_LIBRARIES.issubset(set(process["loaded_libraries"])), "required QNN/FastRPC maps missing")
    require(process["fastrpc_cdsp_fd_open"] is True, "FastRPC CDSP fd missing")
    require(process["qcom_dmabuf_count"] > 0, "QCOM DMA buffers missing")
    require(process["rss_kib"] <= 700 * 1024, "candidate process exceeds 700 MiB RSS cap")
    require(process["pss_kib"] > 0, "invalid PSS evidence")
    for record in process["raw_artifacts"].values():
        artifact = evidence_path.parent / record["path"]
        require(artifact.stat().st_size == record["bytes"], "raw /proc artifact size mismatch")
        require(sha256(artifact) == record["sha256"], "raw /proc artifact hash mismatch")

    resident = evidence["resident_payload"]
    require(resident["root"] == report["model_storage"], "resident payload root mismatch")
    require(resident["model"]["bytes"] == spec["model"]["bytes"], "resident model size mismatch")
    require(resident["model"]["sha256"] == spec["model"]["sha256"], "resident model hash mismatch")
    require(resident["kib"] < 650 * 1024, "resident payload exceeds storage cap")
    require(report["backend_path"].startswith(installed["apk_path"].removesuffix("/base.apk")), "QNN backend is not app-owned")

    result = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "package": PACKAGE,
        "run_id": args.run_id,
        "status": "pass",
        "claim": "isolated_raw_onnx_duration_qnn_htp_execution",
        "installed_apk_sha256": installed["sha256"],
        "cpu_ep_fallback_disabled": True,
        "fastrpc_cdsp_fd_open": True,
        "qcom_dmabuf_count": process["qcom_dmabuf_count"],
        "rss_mib": round(process["rss_kib"] / 1024, 3),
        "pss_mib": round(process["pss_kib"] / 1024, 3),
        "resident_payload_mib": round(resident["kib"] / 1024, 3),
        "duration": execution,
        "timing_ms": report["timing_ms"],
        "accepted_packages_unchanged": True,
        "full_tts": False,
        "claim_boundary": spec["claim_boundary"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
