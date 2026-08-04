#!/usr/bin/env python3
"""Profile pinned Whisper Small W8A16 release contexts on exact SM8850.

The Qualcomm release archive is generated for the SM8850-AD Galaxy variant
and contains compiled QNN contexts plus interface metadata, but no matching
sample inputs or source-model goldens.  This resume-safe probe is therefore
strictly profile-only: it establishes cross-SKU context portability, latency,
memory, and operator placement on the exact Snapdragon 8 Elite Gen 5 QRD.
It does not establish numerical equivalence or transcript quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import qai_hub as hub

from profile_zipformer_release_contexts import (
    atomic_json,
    classify_profile,
    profile_metrics,
    sha256,
)


TARGET_DEVICE = "Snapdragon 8 Elite Gen 5 QRD"
TARGET_OS = "16"
TARGET_CHIPSET = "sm8850"
SOURCE_CHIPSET = "sm8850-ad"
SOURCE_DEVICE = "Samsung Galaxy S26"
QAIRT_VERSION = "2.45"
QAIRT_BUILD = "2.45.0.260326154327"
RELEASE_VERSION = "qai-hub-models 0.58.0"
ARCHIVE_SHA256 = "f798189b2e98c6ec139a3ad26ddc6cd2d41ee19c0ef87fab99f8d5fdc456c79e"
METADATA_SHA256 = "a288221ad7825eb63b0113708aa4c9c9d12d9a2932398811fc7376008be731cd"
EXPECTED_SHA256 = {
    "encoder": "af6e8d5514306c4e7fe54dc709880a9c1743b094257b15694e45bc688753a5a9",
    "decoder": "41e048ff84c10205082c264e8ca6894c9bcfaf54e7eef19316ec35567b693388",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", choices=tuple(EXPECTED_SHA256), required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--confirm-cloud", action="store_true")
    return parser.parse_args()


def memory_pressure_level() -> int:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(completed.stdout.strip())


def validate_metadata(model_dir: Path) -> dict[str, object]:
    metadata_path = model_dir / "metadata.json"
    if sha256(metadata_path) != METADATA_SHA256:
        raise SystemExit("Refusing release package with unpinned metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chipset = metadata.get("chipset_attributes", {})
    required = {
        "model_id": "whisper_small_quantized",
        "runtime": "qnn_context_binary",
        "precision": "w8a16",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise SystemExit(
                f"Unexpected metadata {key}: expected {expected!r}, "
                f"got {metadata.get(key)!r}"
            )
    if metadata.get("tool_versions", {}).get("qairt") != QAIRT_BUILD:
        raise SystemExit("Release metadata has an unexpected QAIRT build")
    if chipset.get("htp_version") != 81:
        raise SystemExit("Release metadata is not an HTP-v81 context")
    if SOURCE_CHIPSET not in chipset.get("aliases", []):
        raise SystemExit("Release metadata is not pinned to SM8850-AD")
    if chipset.get("reference_device") != SOURCE_DEVICE:
        raise SystemExit("Release metadata has an unexpected source device")
    return metadata


def main() -> int:
    args = parse_args()
    if not args.confirm_cloud:
        raise SystemExit("Cloud work requires --confirm-cloud")
    if not os.environ.get("TMUX"):
        raise SystemExit("Cloud work must run inside detached tmux")
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    pressure = memory_pressure_level()
    if pressure != 1:
        raise SystemExit(
            f"Refusing cloud preparation at memory-pressure level {pressure}; expected 1"
        )

    model_dir = args.model_dir.expanduser().resolve()
    model_path = model_dir / f"{args.component}.bin"
    output_dir = args.output_dir.expanduser().resolve()
    state_path = output_dir / "state.json"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    validate_metadata(model_dir)
    actual_sha256 = sha256(model_path)
    if actual_sha256 != EXPECTED_SHA256[args.component]:
        raise SystemExit(
            f"Refusing unpinned {args.component}: expected "
            f"{EXPECTED_SHA256[args.component]}, got {actual_sha256}"
        )

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "release": {
                "package": RELEASE_VERSION,
                "model": "Whisper Small Quantized",
                "runtime": "qnn_context_binary",
                "precision": "w8a16",
                "archive_sha256": ARCHIVE_SHA256,
                "metadata_sha256": METADATA_SHA256,
                "generated_for_chipset": SOURCE_CHIPSET,
                "generated_for_device": SOURCE_DEVICE,
                "generated_for_htp": "v81",
                "generated_with_qairt": QAIRT_BUILD,
            },
            "target": {
                "device": TARGET_DEVICE,
                "os": TARGET_OS,
                "chipset": TARGET_CHIPSET,
                "htp": "v81",
                "requested_qairt": QAIRT_VERSION,
            },
            "method": {
                "profile_iterations": args.iterations,
                "purpose": "sm8850-ad_to_sm8850_context_portability_probe",
                "numerical_gate": "unavailable_without_matching_sample_inputs_and_golden",
                "transcript_gate": "unavailable_without_end_to_end_pre_and_post_processing",
                "inference_submission": False,
            },
            "components": {},
        }
    if state["method"]["profile_iterations"] != args.iterations:
        raise SystemExit("Existing state uses a different --iterations value")
    if state["release"]["archive_sha256"] != ARCHIVE_SHA256:
        raise SystemExit("Existing state is pinned to a different release archive")
    if args.component == "decoder":
        encoder_status = state["components"].get("encoder", {}).get("status")
        if encoder_status != "profile_passed_strict_npu":
            raise SystemExit(
                "Refusing decoder submission before the smaller encoder canary "
                "passes strict NPU placement"
            )
    item = state["components"].setdefault(
        args.component,
        {
            "artifact": {
                "filename": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": actual_sha256,
            },
            "model_id": None,
            "profile_job_id": None,
            "status": "ready",
        },
    )
    if item["artifact"]["sha256"] != actual_sha256:
        raise SystemExit("Existing state is pinned to a different component artifact")
    atomic_json(state_path, state)

    client = hub.Client()
    device = hub.Device(TARGET_DEVICE, TARGET_OS, ["framework:qnn"])
    if item["model_id"]:
        model = client.get_model(item["model_id"])
    else:
        model = client.upload_model(
            str(model_path),
            name=f"lfm-demo-whisper-small-w8a16-{args.component}-s26-v81-release",
        )
        item["model_id"] = str(model.model_id)
        item["status"] = "uploaded"
        atomic_json(state_path, state)

    if item["profile_job_id"]:
        job = client.get_job(item["profile_job_id"])
    else:
        job = client.submit_profile_job(
            model=model,
            device=device,
            options=(
                f"--qairt_version {QAIRT_VERSION} "
                f"--max_profiler_iterations {args.iterations}"
            ),
            name=f"lfm-demo-whisper-small-w8a16-{args.component}-sm8850-profile",
            retry=False,
        )
        item["profile_job_id"] = str(job.job_id)
        item["status"] = "profile_submitted"
        atomic_json(state_path, state)

    try:
        status = job.wait(timeout=args.wait_timeout)
    except TimeoutError:
        item["status"] = "profile_pending"
        atomic_json(state_path, state)
        return 3
    item["job_status"] = {
        "code": status.code,
        "message": status.message,
        "finished": status.finished,
        "success": status.success,
    }
    if not status.success:
        item["status"] = "profile_failed" if status.finished else "profile_pending"
        atomic_json(state_path, state)
        return 2 if status.finished else 3

    profile = job.download_profile()
    if not isinstance(profile, dict):
        raise TypeError("AI Hub returned a non-dictionary profile payload")
    component_dir = output_dir / args.component
    atomic_json(component_dir / "profile.json", profile)
    placement = classify_profile(profile)
    atomic_json(component_dir / "placement.json", placement)
    item["metrics"] = profile_metrics(profile)
    item["placement"] = {
        "operation_count": placement["operation_count"],
        "counts": placement["counts"],
        "strict_npu_pass": placement["strict_npu_pass"],
    }
    item["status"] = (
        "profile_passed_strict_npu"
        if placement["strict_npu_pass"]
        else "profile_completed_with_fallback_or_unknown"
    )
    atomic_json(state_path, state)
    print(json.dumps({"component": args.component, **item}, indent=2))
    return 0 if placement["strict_npu_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
