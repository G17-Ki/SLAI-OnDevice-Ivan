#!/usr/bin/env python3
"""Resumably profile pinned Qualcomm Zipformer SM8850 context binaries.

This is deliberately profile-only.  The release archive contains compiled QNN
contexts but not the matching source model or local golden outputs, so a
successful run proves target compatibility, latency, memory, and placement; it
does not prove numerical equivalence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import qai_hub as hub


TARGET_DEVICE = "Snapdragon 8 Elite Gen 5 QRD"
TARGET_OS = "16"
QAIRT_VERSION = "2.45"
EXPECTED_SHA256 = {
    "decoder": "7cdee7614dfe625f558e746005012b56f1e3c3adca78b7d8f3c6ce9a6bc43a02",
    "joiner": "d8d99b58d5ecd4a5922ebbd92a7d3a7b8b2c8cc297340c79e58e7821821170e2",
    "encoder": "3764077ffd094bbfa6134695949b2dc83ed2964f832da79c1c2dc4105f0b883f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def classify_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    raw_detail = profile.get("execution_detail", []) or []
    detail = raw_detail if isinstance(raw_detail, list) else []
    counts: Counter[str] = Counter()
    operations: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(detail):
        operation = raw_operation if isinstance(raw_operation, dict) else {"raw": raw_operation}
        compute_unit = str(operation.get("compute_unit", "UNSPECIFIED"))
        normalized = compute_unit.upper()
        if any(token in normalized for token in ("NPU", "HTP", "HEXAGON")):
            classification = "hexagon_npu"
        elif "CPU" in normalized:
            classification = "cpu_fallback"
        else:
            classification = "unsupported_or_export_failure"
        counts[classification] += 1
        operations.append(
            {
                "index": index,
                "classification": classification,
                "compute_unit": compute_unit,
                "operation": operation,
            }
        )
    return {
        "profile_available": True,
        "operation_count": len(operations),
        "counts": {
            "hexagon_npu": counts["hexagon_npu"],
            "cpu_fallback": counts["cpu_fallback"],
            "unsupported_or_export_failure": counts[
                "unsupported_or_export_failure"
            ],
        },
        "strict_npu_pass": bool(operations)
        and counts["hexagon_npu"] == len(operations),
        "operations": operations,
    }


def profile_metrics(profile: Mapping[str, Any]) -> dict[str, Any]:
    raw_summary = profile.get("execution_summary", {}) or {}
    summary = raw_summary if isinstance(raw_summary, dict) else {"raw": raw_summary}
    latency_us = summary.get("estimated_inference_time")
    peak_bytes = summary.get("estimated_inference_peak_memory")
    return {
        "latency_ms": None if latency_us is None else float(latency_us) / 1000.0,
        "peak_memory_mib": None
        if peak_bytes is None
        else float(peak_bytes) / (1024**2),
        "raw_execution_summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--component", choices=tuple(EXPECTED_SHA256), required=True
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--wait-timeout", type=int, default=3600)
    parser.add_argument("--confirm-cloud", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_cloud:
        raise SystemExit("Cloud work requires --confirm-cloud")
    if not os.environ.get("TMUX"):
        raise SystemExit("Cloud work must run inside tmux")
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
            "target": {
                "device": TARGET_DEVICE,
                "os": TARGET_OS,
                "chipset": "sm8850",
                "htp": "v81",
                "qairt": "2.45.0.260326154327",
            },
            "method": {
                "profile_iterations": args.iterations,
                "numerical_gate": "not_available_without_matching_source_golden",
            },
            "components": {},
        }
    if state["method"]["profile_iterations"] != args.iterations:
        raise SystemExit("Existing state uses a different --iterations value")
    prerequisites = {
        "decoder": (),
        "joiner": ("decoder",),
        "encoder": ("decoder", "joiner"),
    }
    for prerequisite in prerequisites[args.component]:
        prerequisite_status = state["components"].get(prerequisite, {}).get(
            "status"
        )
        if prerequisite_status != "profile_passed_strict_npu":
            raise SystemExit(
                f"Refusing {args.component} before {prerequisite} passes strict NPU "
                "placement in this output directory"
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
    atomic_json(state_path, state)

    client = hub.Client()
    device = hub.Device(TARGET_DEVICE, TARGET_OS, ["framework:qnn"])
    if item["model_id"]:
        model = client.get_model(item["model_id"])
    else:
        model = client.upload_model(
            str(model_path), name=f"lfm-demo-zipformer-release-{args.component}-sm8850"
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
            name=f"lfm-demo-zipformer-release-{args.component}-sm8850-profile",
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
