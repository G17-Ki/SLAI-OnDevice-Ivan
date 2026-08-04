#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import qai_hub as hub

from path_config import REPO_ROOT, WORK_ROOT

# Backward-compatible import for submit_component_aihub.py.
WORKSPACE = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="Exact AI Hub device name")
    parser.add_argument(
        "--device-os",
        default="",
        help="Optional exact OS version from list_aihub_devices.py.",
    )
    parser.add_argument(
        "--device-attribute",
        action="append",
        default=[],
        help="Optional AI Hub device attribute; repeat to disambiguate the target.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/exports/fastconformer_adapter_mel80.pt2",
    )
    parser.add_argument(
        "--io",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/exports/fastconformer_adapter_mel80.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/aihub/fastconformer",
    )
    parser.add_argument(
        "--runtime",
        choices=("strict-npu", "ort-diagnostic"),
        default="strict-npu",
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=20,
        help="Maximum remote profiler iterations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local inputs and print the job plan without contacting AI Hub.",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def first_array(outputs: dict) -> np.ndarray:
    if not outputs:
        raise RuntimeError("AI Hub inference returned no tensors")
    first_value = next(iter(outputs.values()))
    if isinstance(first_value, list):
        return np.asarray(first_value[0])
    return np.asarray(first_value)


def runtime_options(runtime: str, profile_iterations: int) -> tuple[str, str]:
    if profile_iterations < 1:
        raise ValueError("--profile-iterations must be positive")
    if runtime == "strict-npu":
        # A QNN context binary is compiled for the HTP backend. Adding
        # --compute_unit to the compile job is unnecessary and is not a
        # documented compile option in qai-hub 0.52.0.
        compile_options = (
            "--target_runtime qnn_context_binary "
            "--qnn_options default_graph_htp_precision=FLOAT16"
        )
        profile_options = f"--max_profiler_iterations {profile_iterations}"
    elif runtime == "ort-diagnostic":
        # QNN EP attempts NPU placement. ONNX Runtime keeps its CPU EP as the
        # fallback for nodes that QNN cannot accept.
        compile_options = "--target_runtime onnx"
        profile_options = (
            "--onnx_execution_providers qnn "
            f"--max_profiler_iterations {profile_iterations}"
        )
    else:
        raise ValueError(f"Unknown runtime: {runtime}")
    return compile_options, profile_options


def classify_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Classify compiled runtime layers using qai-hub profile JSON.

    Qualcomm may fuse several source graph operations into one runtime layer,
    so these are authoritative placement counts but are not necessarily a
    one-to-one count of torch.export nodes.
    """

    layers = profile.get("execution_detail", []) or []
    counts = Counter(str(layer.get("compute_unit", "UNSPECIFIED")).upper() for layer in layers)
    classified_layers = {
        "hexagon_npu": [layer for layer in layers if str(layer.get("compute_unit", "")).upper() == "NPU"],
        "cpu_fallback": [layer for layer in layers if str(layer.get("compute_unit", "")).upper() == "CPU"],
        "other_or_unspecified": [
            layer
            for layer in layers
            if str(layer.get("compute_unit", "")).upper() not in {"NPU", "CPU"}
        ],
    }
    return {
        "unit_counts": dict(sorted(counts.items())),
        "npu_runtime_layers": len(classified_layers["hexagon_npu"]),
        "cpu_fallback_runtime_layers": len(classified_layers["cpu_fallback"]),
        "other_runtime_layers": len(classified_layers["other_or_unspecified"]),
        "layers": classified_layers,
        "counting_note": (
            "AI Hub execution_detail reports compiled runtime layers; compiler fusion means "
            "counts may differ from torch.export operation counts."
        ),
    }


def profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    execution = profile.get("execution_summary", {}) or {}
    latency_us = execution.get("estimated_inference_time")
    peak_bytes = execution.get("estimated_inference_peak_memory")
    return {
        "latency_ms": None if latency_us is None else float(latency_us) / 1_000.0,
        "peak_memory_mb": None if peak_bytes is None else float(peak_bytes) / (1024.0**2),
        "raw_execution_summary": execution,
    }


def job_record(job: Any) -> dict[str, Any]:
    status = job.get_status()
    return {
        "job_id": getattr(job, "job_id", None),
        "url": getattr(job, "url", None),
        "status": status.code,
        "message": status.message,
    }


def write_summary(output_dir: Path, runtime: str, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"summary-{runtime}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Exported model not found: {args.model}")
    if not args.io.is_file():
        raise FileNotFoundError(f"Golden I/O not found: {args.io}")

    with np.load(args.io) as io:
        mel = io["mel"].astype(np.float32, copy=False)
        golden = io["adapted"].astype(np.float32, copy=False)
    if mel.shape != (1, 128, 80):
        raise ValueError(f"Expected mel shape (1, 128, 80), got {mel.shape}")
    if golden.shape != (1, 10, 2048):
        raise ValueError(f"Expected golden shape (1, 10, 2048), got {golden.shape}")
    if not np.isfinite(mel).all():
        raise ValueError("Input mel contains NaN or Inf")
    if not np.isfinite(golden).all():
        raise ValueError("Golden output contains NaN or Inf")

    compile_options, profile_options = runtime_options(
        args.runtime, args.profile_iterations
    )
    plan = {
        "device": {
            "name": args.device,
            "os": args.device_os,
            "attributes": args.device_attribute,
        },
        "runtime": args.runtime,
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "io": str(args.io),
        "input_specs": {"mel": [[1, 128, 80], "float32"]},
        "compile_options": compile_options,
        "profile_options": profile_options,
        "output_mapping": {"#0": "adapted"},
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "classification": (
            "strict compile failure or runtime-layer NPU/CPU placement from profile execution_detail"
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = hub.Client()
    device = hub.Device(
        name=args.device,
        os=args.device_os,
        attributes=args.device_attribute,
    )
    summary: dict[str, Any] = {"plan": plan, "jobs": {}}

    compile_job = client.submit_compile_job(
        model=str(args.model),
        device=device,
        input_specs={"mel": ((1, 128, 80), "float32")},
        options=compile_options,
        name=f"lfm-fastconformer-{args.runtime}",
    )
    target_model = compile_job.get_target_model()
    summary["jobs"]["compile"] = job_record(compile_job)
    if target_model is None:
        summary.update(
            {
                "status": "unsupported_or_compile_failure",
                "unsupported_operations": (
                    "Inspect the compile job error details/log for the failing source operation."
                ),
            }
        )
        write_summary(args.output_dir, args.runtime, summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(2)

    profile_job = client.submit_profile_job(
        model=target_model,
        device=device,
        options=profile_options,
        name=f"lfm-fastconformer-profile-{args.runtime}",
    )
    profile = profile_job.download_profile()
    summary["jobs"]["profile"] = job_record(profile_job)
    if not isinstance(profile, dict):
        summary.update({"status": "profile_failure", "profile": profile})
        write_summary(args.output_dir, args.runtime, summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(3)

    inference_job = client.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"mel": [mel]},
        name=f"lfm-fastconformer-inference-{args.runtime}",
    )
    outputs = inference_job.download_output_data()
    summary["jobs"]["inference"] = job_record(inference_job)
    if not isinstance(outputs, dict):
        summary.update({"status": "inference_failure"})
        write_summary(args.output_dir, args.runtime, summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(4)
    actual = first_array(outputs).astype(np.float32, copy=False)
    if actual.shape != golden.shape:
        summary.update(
            {
                "status": "output_shape_mismatch",
                "output_shape": list(actual.shape),
                "golden_shape": list(golden.shape),
            }
        )
        write_summary(args.output_dir, args.runtime, summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(5)
    if not np.isfinite(actual).all():
        summary.update({"status": "non_finite_remote_output"})
        write_summary(args.output_dir, args.runtime, summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(6)
    delta = np.abs(actual - golden)
    tolerance_pass = bool(np.allclose(actual, golden, atol=args.atol, rtol=args.rtol))

    np.savez(args.output_dir / f"outputs-{args.runtime}.npz", actual=actual, golden=golden)
    placement = classify_profile(profile)
    (args.output_dir / f"placement-{args.runtime}.json").write_text(
        json.dumps(placement, indent=2, default=str) + "\n"
    )
    summary.update(
        {
            "status": "passed" if tolerance_pass else "numerical_mismatch",
            "output_shape": list(actual.shape),
            "golden_shape": list(golden.shape),
            "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
            "tolerance_pass": tolerance_pass,
            "tolerances": {"atol": args.atol, "rtol": args.rtol},
            "metrics": profile_metrics(profile),
            "placement": placement,
            "strict_npu_pass": (
                args.runtime == "strict-npu"
                and placement["npu_runtime_layers"] > 0
                and placement["cpu_fallback_runtime_layers"] == 0
                and placement["other_runtime_layers"] == 0
            ),
            "profile": profile,
        }
    )
    write_summary(args.output_dir, args.runtime, summary)
    print(json.dumps(summary, indent=2, default=str))
    if not tolerance_pass:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
