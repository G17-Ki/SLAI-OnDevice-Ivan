#!/usr/bin/env python3
"""Resume-safe AI Hub profile for the pinned SM8850 SenseVoice context.

The upstream archive contains a complete QNN context binary but no matching
source graph suitable for an AI Hub numerical comparison.  This probe is
therefore deliberately profile-only: it measures exact-QRD compatibility,
latency, memory, and operator placement without claiming transcript parity.
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


EXPECTED_SHA256 = "522763135f536bc43e9d122bac39254f49df4afbb003462d6d8bcdb77051de30"
TARGET_DEVICE = "Snapdragon 8 Elite Gen 5 QRD"
TARGET_OS = "16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-model-id",
        help="Reuse an already uploaded pinned context instead of uploading it again.",
    )
    parser.add_argument(
        "--qairt-version",
        default="2.40",
        help="Match the QNN version that generated the pinned context first.",
    )
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

    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    state_path = output_dir / "state.json"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    actual_sha256 = sha256(model_path)
    if actual_sha256 != EXPECTED_SHA256:
        raise SystemExit(
            f"Refusing unpinned SenseVoice context: expected {EXPECTED_SHA256}, "
            f"got {actual_sha256}"
        )

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "artifact": {
                "filename": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": actual_sha256,
                "upstream_generator": "QNN SDK 2.40.0.251030",
                "fixed_audio_context_s": 10,
            },
            "target": {
                "device": TARGET_DEVICE,
                "os": TARGET_OS,
                "chipset": "sm8850",
                "htp": "v81",
                "requested_qairt": args.qairt_version,
            },
            "method": {
                "profile_iterations": args.iterations,
                "numerical_gate": "unavailable_without_matching_source_golden",
                "transcript_gate": "requires_phone_or_matching_QNN_runtime",
            },
            "model_id": args.reuse_model_id,
            "profile_job_id": None,
            "status": "ready",
        }
    if state["target"]["requested_qairt"] != args.qairt_version:
        raise SystemExit("Existing state uses a different --qairt-version")
    if args.reuse_model_id and state["model_id"] != args.reuse_model_id:
        raise SystemExit("Existing state uses a different --reuse-model-id")
    if state["method"]["profile_iterations"] != args.iterations:
        raise SystemExit("Existing state uses a different --iterations value")
    atomic_json(state_path, state)

    client = hub.Client()
    device = hub.Device(TARGET_DEVICE, TARGET_OS, ["framework:qnn"])
    if state["model_id"]:
        model = client.get_model(state["model_id"])
    else:
        model = client.upload_model(
            str(model_path), name="lfm-demo-sensevoice-release-sm8850-qnn240"
        )
        state["model_id"] = str(model.model_id)
        state["status"] = "uploaded"
        atomic_json(state_path, state)

    if state["profile_job_id"]:
        job = client.get_job(state["profile_job_id"])
    else:
        job = client.submit_profile_job(
            model=model,
            device=device,
            options=(
                f"--qairt_version {args.qairt_version} "
                f"--max_profiler_iterations {args.iterations}"
            ),
            name="lfm-demo-sensevoice-release-sm8850-profile",
            retry=False,
        )
        state["profile_job_id"] = str(job.job_id)
        state["status"] = "profile_submitted"
        atomic_json(state_path, state)

    try:
        status = job.wait(timeout=args.wait_timeout)
    except TimeoutError:
        state["status"] = "profile_pending"
        atomic_json(state_path, state)
        return 3
    state["job_status"] = {
        "code": status.code,
        "message": status.message,
        "finished": status.finished,
        "success": status.success,
    }
    if not status.success:
        state["status"] = "profile_failed" if status.finished else "profile_pending"
        atomic_json(state_path, state)
        return 2 if status.finished else 3

    profile = job.download_profile()
    if not isinstance(profile, dict):
        raise TypeError("AI Hub returned a non-dictionary profile payload")
    atomic_json(output_dir / "profile.json", profile)
    placement = classify_profile(profile)
    atomic_json(output_dir / "placement.json", placement)
    state["metrics"] = profile_metrics(profile)
    state["placement"] = {
        "operation_count": placement["operation_count"],
        "counts": placement["counts"],
        "strict_npu_pass": placement["strict_npu_pass"],
    }
    state["status"] = (
        "profile_passed_strict_npu"
        if placement["strict_npu_pass"]
        else "profile_completed_with_fallback_or_unknown"
    )
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2))
    return 0 if placement["strict_npu_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
