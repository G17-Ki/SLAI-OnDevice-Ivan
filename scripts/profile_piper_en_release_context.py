#!/usr/bin/env python3
"""Profile Qualcomm's pinned Piper-English HTP-v81 release on plain SM8850.

The public package is generated for the Galaxy S26 ``sm8850-ad`` variant. This
resume-safe runner uploads its six QNN context binaries unchanged and profiles
them on Qualcomm AI Hub's ``Snapdragon 8 Elite Gen 5 QRD`` (plain ``sm8850``).
The archive contains no matching inputs or golden outputs, so the experiment is
deliberately profile-only: it measures portability, placement, latency, and
estimated inference memory, not numerical equivalence or speech quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
QAIRT_API_VERSION = "2.45"
QAIRT_BUILD = "2.45.0.260326154327"
RELEASE_VERSION = "qai-hub-models 0.58.0"
ARCHIVE_FILENAME = (
    "pipertts_en-voice_ai-float-"
    "qualcomm_snapdragon_8_elite_gen5_for_galaxy.zip"
)
ARCHIVE_BYTES = 69_259_115
ARCHIVE_SHA256 = "950f20398da2d0c3549b8ee5cc73ce53e4e093cbe430015d143ed3988483f1d6"

COMPONENTS = {
    "encoder": {
        "bytes": 15_745_024,
        "sha256": "bf4dbefa9fff825f88f0c2ee99406edb0afdc466be2fc0254959926bf76445a4",
    },
    "sdp": {
        "bytes": 2_060_288,
        "sha256": "22500478e7a7454da903c9baa591496a6ce9dc380a45ff439daff36400491eef",
    },
    "flow": {
        "bytes": 19_685_376,
        "sha256": "05b163e537b3839c4cadc6b06a6a1ff50363245d0726fe1a935b436ed56bb217",
    },
    "decoder": {
        "bytes": 3_547_136,
        "sha256": "fe1d54c604c5eb630dfbc7a17b392f408c84e65cfc4d78755d8910729286cd4c",
    },
    "charsiu_encoder": {
        "bytes": 30_457_856,
        "sha256": "c98451981422d2af0621d56a65b1efd846ab662596dc50abb433963224df2eb5",
    },
    "charsiu_decoder": {
        "bytes": 11_608_064,
        "sha256": "04d088f17e7cfb995fd8d792f515c279890620a15d40c89ca5f6c0314204cc9e",
    },
}

SUPPLEMENTARY_FILES = {
    "bert_normalizer.bin": {
        "bytes": 264_948,
        "sha256": "5d2e6435b23d107aa4cb4a87009ff1b1d6ee4069a866648b07377569b879d093",
        "classification": "native_preprocessing_asset_not_qnn_graph",
    },
    "config.json": {
        "bytes": 1_634,
        "sha256": "ab6246360d660adb4c878b8e527a697cab5e7b09af41b48220f6914d59eb353d",
        "classification": "runtime_configuration_not_qnn_graph",
    },
    "metadata.json": {
        "bytes": 15_585,
        "sha256": "7b23c1cabfe20f56484a6f09cb86a91d836c3acb93fa3df8f3f54c41778ad701",
        "classification": "release_metadata_not_qnn_graph",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        action="append",
        choices=tuple(COMPONENTS),
        help="Profile only this component; repeat as needed. Defaults to all six.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--confirm-cloud", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.wait_timeout < 1:
        parser.error("--wait-timeout must be positive")
    return args


def pinned_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = sha256(path)
    if size != expected["bytes"] or digest != expected["sha256"]:
        raise SystemExit(
            f"Refusing unpinned {path.name}: expected {expected['bytes']} bytes / "
            f"{expected['sha256']}, got {size} bytes / {digest}"
        )
    return {"filename": path.name, "bytes": size, "sha256": digest}


def validate_release(model_dir: Path, archive: Path) -> dict[str, Any]:
    archive_item = pinned_file(
        archive,
        {"bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256},
    )
    if archive_item["filename"] != ARCHIVE_FILENAME:
        raise SystemExit(
            f"Expected archive filename {ARCHIVE_FILENAME}, got {archive_item['filename']}"
        )

    supplementary = {
        name: {
            **pinned_file(model_dir / name, expected),
            "classification": expected["classification"],
        }
        for name, expected in SUPPLEMENTARY_FILES.items()
    }
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_id") != "pipertts_en":
        raise SystemExit("Unexpected model_id in metadata.json")
    if metadata.get("runtime") != "voice_ai" or metadata.get("precision") != "float":
        raise SystemExit("Unexpected runtime or precision in metadata.json")
    if metadata.get("tool_versions", {}).get("qairt") != QAIRT_BUILD:
        raise SystemExit("Unexpected QAIRT build in metadata.json")
    attributes = metadata.get("chipset_attributes", {})
    if SOURCE_CHIPSET not in attributes.get("aliases", []):
        raise SystemExit("Release metadata does not identify the expected source chipset")
    if attributes.get("htp_version") != 81:
        raise SystemExit("Release metadata does not identify HTP v81")
    if set(metadata.get("model_files", {})) != {
        f"{component}.bin" for component in COMPONENTS
    }:
        raise SystemExit("Release model-file inventory does not match the pinned six graphs")
    return {"archive": archive_item, "supplementary_files": supplementary}


def status_dict(status: Any) -> dict[str, Any]:
    return {
        "code": status.code,
        "message": status.message,
        "finished": status.finished,
        "success": status.success,
    }


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    completed = [
        item
        for item in state["components"].values()
        if item.get("status") in {
            "profile_passed_strict_npu",
            "profile_completed_with_fallback_or_unknown",
        }
    ]
    latencies = [
        item.get("metrics", {}).get("latency_ms")
        for item in completed
        if item.get("metrics", {}).get("latency_ms") is not None
    ]
    counts = {
        key: sum(item.get("placement", {}).get("counts", {}).get(key, 0) for item in completed)
        for key in (
            "hexagon_npu",
            "cpu_fallback",
            "unsupported_or_export_failure",
        )
    }
    return {
        "requested_component_count": len(state["components"]),
        "completed_component_count": len(completed),
        "all_components_strict_npu_pass": bool(completed)
        and len(completed) == len(state["components"])
        and all(item.get("placement", {}).get("strict_npu_pass") for item in completed),
        "component_latency_sum_ms": round(sum(latencies), 6) if latencies else None,
        "placement_counts": counts,
    }


def main() -> int:
    args = parse_args()
    if not args.confirm_cloud:
        raise SystemExit("Cloud work requires --confirm-cloud")
    if not os.environ.get("TMUX"):
        raise SystemExit("Cloud work must run inside detached tmux")

    model_dir = args.model_dir.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    state_path = output_dir / "state.json"
    selected = list(dict.fromkeys(args.component or COMPONENTS.keys()))
    release_inventory = validate_release(model_dir, archive)

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "release": {
                "package": RELEASE_VERSION,
                "runtime": "voice_ai",
                "precision": "float",
                "generated_for_chipset": SOURCE_CHIPSET,
                "generated_for_htp": "v81",
                "generated_with_qairt": QAIRT_BUILD,
                **release_inventory,
            },
            "target": {
                "device": TARGET_DEVICE,
                "os": TARGET_OS,
                "chipset": TARGET_CHIPSET,
                "htp": "v81",
                "requested_qairt": QAIRT_API_VERSION,
            },
            "method": {
                "profile_iterations": args.iterations,
                "purpose": "sm8850-ad_to_sm8850_context_portability_probe",
                "numerical_gate": "unavailable_without_matching_release_inputs_and_goldens",
                "intelligibility_gate": "unavailable_without_real_preprocessing_and_inference",
                "inference_submitted": False,
                "compile_submitted": False,
            },
            "components": {},
        }
    if state["method"]["profile_iterations"] != args.iterations:
        raise SystemExit("Existing state uses a different --iterations value")
    if state["release"]["archive"]["sha256"] != ARCHIVE_SHA256:
        raise SystemExit("Existing state is pinned to a different archive")

    for component in selected:
        model_path = model_dir / f"{component}.bin"
        artifact = pinned_file(model_path, COMPONENTS[component])
        item = state["components"].setdefault(
            component,
            {
                "artifact": artifact,
                "model_id": None,
                "profile_job_id": None,
                "status": "ready",
            },
        )
        if item["artifact"] != artifact:
            raise SystemExit(f"Existing state is pinned to a different {component} artifact")
    state["summary"] = build_summary(state)
    atomic_json(state_path, state)

    client = hub.Client()
    device = hub.Device(TARGET_DEVICE, TARGET_OS, ["framework:qnn"])
    jobs: dict[str, Any] = {}

    # Submit every missing profile before waiting, so the cloud can provision
    # the small independent contexts without serial host-side idle time.
    for component in selected:
        item = state["components"][component]
        model_path = model_dir / f"{component}.bin"
        if item["model_id"]:
            model = client.get_model(item["model_id"])
        else:
            model = client.upload_model(
                str(model_path),
                name=f"lfm-demo-piper-en-release-{component}-sm8850-ad",
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
                    f"--qairt_version {QAIRT_API_VERSION} "
                    f"--max_profiler_iterations {args.iterations}"
                ),
                name=f"lfm-demo-piper-en-release-{component}-sm8850-profile",
                retry=False,
            )
            item["profile_job_id"] = str(job.job_id)
            item["status"] = "profile_submitted"
            atomic_json(state_path, state)
        jobs[component] = job

    exit_code = 0
    for component in selected:
        item = state["components"][component]
        job = jobs[component]
        try:
            status = job.wait(timeout=args.wait_timeout)
        except TimeoutError:
            item["status"] = "profile_pending"
            state["summary"] = build_summary(state)
            atomic_json(state_path, state)
            exit_code = max(exit_code, 3)
            continue
        item["job_status"] = status_dict(status)
        if not status.success:
            item["status"] = "profile_failed" if status.finished else "profile_pending"
            state["summary"] = build_summary(state)
            atomic_json(state_path, state)
            exit_code = max(exit_code, 2 if status.finished else 3)
            continue

        profile = job.download_profile()
        if not isinstance(profile, dict):
            raise TypeError("AI Hub returned a non-dictionary profile payload")
        component_dir = output_dir / component
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
        if not placement["strict_npu_pass"]:
            exit_code = max(exit_code, 4)
        state["summary"] = build_summary(state)
        atomic_json(state_path, state)

    print(json.dumps(state, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
