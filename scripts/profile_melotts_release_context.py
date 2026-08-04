#!/usr/bin/env python3
"""Profile a pinned public MeloTTS Voice AI context on an exact SM8850 QRD.

The Qualcomm release package currently publishes HTP-v81 contexts for the
SM8850-AD Galaxy variant, but not a downloadable package for plain SM8850.
This resume-safe probe uploads one pinned context and asks AI Hub to profile it
on ``Snapdragon 8 Elite Gen 5 QRD``.  A pass is portability, latency, memory,
and placement evidence only; it is not a numerical or intelligibility gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
QAIRT_VERSION = "2.45"
RELEASE_VERSION = "qai-hub-models 0.58.0"
EXPECTED_SHA256 = {
    "decoder": "d5ab0849b39e0289db84baf0ffa1964d2fa3004c3bc70de833910cf808393895",
    "encoder": "dbd5fbe6a09c4d4de973e9cc5e9a41ed7b7706bedace62390b8d91cd41d19090",
    "flow": "836982f3e2afe440450fa33ea769abf570874acf1ffa12ecb4159222fe24a6f4",
    "bert_wrapper": "45fd58047d0234d3287f806eec34a994b868a9a7452a5dceb34014ac1603566a",
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


def main() -> int:
    args = parse_args()
    if not args.confirm_cloud:
        raise SystemExit("Cloud work requires --confirm-cloud")
    if not os.environ.get("TMUX"):
        raise SystemExit("Cloud work must run inside detached tmux")
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    model_path = args.model_dir.expanduser().resolve() / f"{args.component}.bin"
    output_dir = args.output_dir.expanduser().resolve()
    state_path = output_dir / "state.json"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
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
                "runtime": "voice_ai",
                "precision": "mixed_with_float",
                "generated_for_chipset": SOURCE_CHIPSET,
                "generated_for_htp": "v81",
                "generated_with_qairt": "2.45.0.260326154327",
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
                "numerical_gate": "unavailable_without_matching_source_golden",
                "intelligibility_gate": "requires_real_preprocessing_and_inference",
            },
            "components": {},
        }
    if state["method"]["profile_iterations"] != args.iterations:
        raise SystemExit("Existing state uses a different --iterations value")
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
        raise SystemExit("Existing state is pinned to a different artifact")
    atomic_json(state_path, state)

    client = hub.Client()
    device = hub.Device(TARGET_DEVICE, TARGET_OS, ["framework:qnn"])
    if item["model_id"]:
        model = client.get_model(item["model_id"])
    else:
        model = client.upload_model(
            str(model_path),
            name=f"lfm-demo-melotts-zh-{args.component}-sm8850-ad-release",
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
            name=f"lfm-demo-melotts-zh-{args.component}-sm8850-portability",
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
