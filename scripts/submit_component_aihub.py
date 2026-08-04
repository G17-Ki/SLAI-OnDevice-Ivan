#!/usr/bin/env python3
"""Submit any fixed-shape LFM component and compare remote outputs to NPZ goldens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import qai_hub as hub

from submit_encoder_aihub import (
    classify_profile,
    job_record,
    profile_metrics,
    runtime_options,
)
from path_config import WORK_ROOT


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        return value, value
    remote_or_graph, npz_key = value.split("=", 1)
    if not remote_or_graph or not npz_key:
        raise argparse.ArgumentTypeError("mapping must be NAME or NAME=NPZ_KEY")
    return remote_or_graph, npz_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile/profile/infer a fixed-shape PT2 component. Inputs and golden "
            "outputs come from one NPZ file."
        )
    )
    parser.add_argument("--component", required=True, help="Short report/job name")
    parser.add_argument("--device", required=True, help="Exact AI Hub device name")
    parser.add_argument("--device-os", default="")
    parser.add_argument("--device-attribute", action="append", default=[])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--io", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        type=parse_mapping,
        required=True,
        metavar="GRAPH_NAME[=NPZ_KEY]",
        help="Graph input name and NPZ key; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--output",
        action="append",
        type=parse_mapping,
        required=True,
        metavar="REMOTE_NAME|#INDEX=NPZ_KEY",
        help=(
            "Explicit remote-to-golden mapping. Use #0, #1, ... when compiled "
            "output names are not known in advance."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/aihub/components",
    )
    parser.add_argument(
        "--runtime", choices=("strict-npu", "ort-diagnostic"), default="strict-npu"
    )
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def numpy_dtype_name(dtype: np.dtype[Any]) -> str:
    supported = {
        np.dtype("float32"): "float32",
        np.dtype("float16"): "float16",
        np.dtype("int64"): "int64",
        np.dtype("int32"): "int32",
        np.dtype("int16"): "int16",
        np.dtype("int8"): "int8",
        np.dtype("uint16"): "uint16",
        np.dtype("uint8"): "uint8",
        np.dtype("bool"): "bool",
    }
    try:
        return supported[np.dtype(dtype)]
    except KeyError as error:
        raise ValueError(f"Unsupported AI Hub input dtype: {dtype}") from error


def ordered_remote_outputs(outputs: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    ordered: list[tuple[str, np.ndarray]] = []
    for name, samples in outputs.items():
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError(
                f"Expected exactly one sample for remote output {name!r}; got {type(samples).__name__}"
            )
        ordered.append((name, np.asarray(samples[0])))
    return ordered


def select_remote_output(
    requested: str, outputs: dict[str, Any], ordered: list[tuple[str, np.ndarray]]
) -> tuple[str, np.ndarray]:
    if requested.startswith("#"):
        try:
            index = int(requested[1:])
            return ordered[index]
        except (ValueError, IndexError) as error:
            raise KeyError(f"Remote output index does not exist: {requested}") from error
    if requested not in outputs:
        raise KeyError(
            f"Remote output {requested!r} not found; available names: {list(outputs)}"
        )
    samples = outputs[requested]
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"Expected exactly one sample for remote output {requested!r}")
    return requested, np.asarray(samples[0])


def compiled_output_names(target_model: Any) -> list[str]:
    names: list[str] = []
    for _, tensors in target_model.output_spec.items():
        names.extend(str(tensor.name) for tensor in tensors)
    return names


def compare_output(
    actual: np.ndarray, golden: np.ndarray, atol: float, rtol: float
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "actual_shape": list(actual.shape),
        "golden_shape": list(golden.shape),
        "actual_dtype": str(actual.dtype),
        "golden_dtype": str(golden.dtype),
        "actual_finite": bool(np.isfinite(actual).all()),
        "golden_finite": bool(np.isfinite(golden).all()),
    }
    if actual.shape != golden.shape or not result["actual_finite"] or not result["golden_finite"]:
        result["passed"] = False
        return result

    if np.issubdtype(golden.dtype, np.integer) or np.issubdtype(golden.dtype, np.bool_):
        result.update(
            {
                "comparison": "exact",
                "passed": bool(np.array_equal(actual, golden)),
            }
        )
    else:
        actual64 = actual.astype(np.float64).ravel()
        golden64 = golden.astype(np.float64).ravel()
        signed_delta = actual64 - golden64
        delta = np.abs(signed_delta)
        rmse = float(np.sqrt(np.mean(signed_delta * signed_delta))) if delta.size else 0.0
        golden_rms = float(np.sqrt(np.mean(golden64 * golden64))) if delta.size else 0.0
        denominator = float(np.linalg.norm(actual64) * np.linalg.norm(golden64))
        cosine = float(np.dot(actual64, golden64) / denominator) if denominator else 1.0
        result.update(
            {
                "comparison": "allclose",
                "atol": atol,
                "rtol": rtol,
                "max_abs_error": float(delta.max()) if delta.size else 0.0,
                "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
                "p95_abs_error": float(np.percentile(delta, 95)) if delta.size else 0.0,
                "p99_abs_error": float(np.percentile(delta, 99)) if delta.size else 0.0,
                "rmse": rmse,
                "golden_rms": golden_rms,
                "normalized_rmse": rmse / golden_rms if golden_rms else 0.0,
                "cosine_similarity": cosine,
                "passed": bool(np.allclose(actual, golden, atol=atol, rtol=rtol)),
            }
        )
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.io.is_file():
        raise FileNotFoundError(args.io)
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Tolerances must be non-negative")

    with np.load(args.io) as archive:
        missing = {
            key for _, key in [*args.input, *args.output] if key not in archive.files
        }
        if missing:
            raise KeyError(f"NPZ keys not found: {sorted(missing)}")
        inputs = {graph: np.array(archive[key]) for graph, key in args.input}
        goldens = {key: np.array(archive[key]) for _, key in args.output}

    for name, value in inputs.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Input {name!r} contains NaN or Inf")
    for name, value in goldens.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Golden output {name!r} contains NaN or Inf")

    input_specs = {
        name: (tuple(value.shape), numpy_dtype_name(value.dtype))
        for name, value in inputs.items()
    }
    compile_options, profile_options = runtime_options(
        args.runtime, args.profile_iterations
    )
    truncate_64bit_io = args.runtime == "strict-npu" and any(
        np.issubdtype(value.dtype, np.integer) and value.dtype.itemsize == 8
        for value in [*inputs.values(), *goldens.values()]
    )
    if truncate_64bit_io:
        # QNN context binaries do not expose int64 tensor I/O. Token IDs are
        # far inside int32 range, so use AI Hub's explicit boundary truncation
        # while preserving exact-value comparison against the int64 golden.
        compile_options += " --truncate_64bit_io"
    remote_inputs = {
        name: value.astype(np.int32, copy=False)
        if truncate_64bit_io
        and np.issubdtype(value.dtype, np.integer)
        and value.dtype.itemsize == 8
        else value
        for name, value in inputs.items()
    }
    run_dir = args.output_dir / args.component
    plan = {
        "component": args.component,
        "device": {
            "name": args.device,
            "os": args.device_os,
            "attributes": args.device_attribute,
        },
        "runtime": args.runtime,
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "io": str(args.io),
        "input_mapping": dict(args.input),
        "output_mapping": dict(args.output),
        "input_specs": {
            name: [list(shape), dtype] for name, (shape, dtype) in input_specs.items()
        },
        "compile_options": compile_options,
        "profile_options": profile_options,
        "truncate_64bit_io": truncate_64bit_io,
        "remote_input_dtypes": {
            name: str(value.dtype) for name, value in remote_inputs.items()
        },
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    client = hub.Client()
    device = hub.Device(args.device, args.device_os, args.device_attribute)
    summary: dict[str, Any] = {"plan": plan, "jobs": {}}
    compile_job = client.submit_compile_job(
        model=str(args.model),
        device=device,
        input_specs=input_specs,
        options=compile_options,
        name=f"lfm-{args.component}-{args.runtime}",
    )
    target_model = compile_job.get_target_model()
    summary["jobs"]["compile"] = job_record(compile_job)
    if target_model is None:
        summary.update(
            {
                "status": "unsupported_or_compile_failure",
                "unsupported_operations": (
                    "Use the compile job error details/log to identify the failing source operation."
                ),
            }
        )
        write_json(run_dir / f"summary-{args.runtime}.json", summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(2)

    target_output_names = compiled_output_names(target_model)
    summary["compiled_output_names"] = target_output_names
    summary["compiled_output_spec"] = str(target_model.output_spec)
    profile_job = client.submit_profile_job(
        model=target_model,
        device=device,
        options=profile_options,
        name=f"lfm-{args.component}-profile-{args.runtime}",
    )
    profile = profile_job.download_profile()
    summary["jobs"]["profile"] = job_record(profile_job)
    if not isinstance(profile, dict):
        summary.update({"status": "profile_failure"})
        write_json(run_dir / f"summary-{args.runtime}.json", summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(3)

    placement = classify_profile(profile)
    profile_path = run_dir / f"profile-{args.runtime}.json"
    write_json(profile_path, profile)
    write_json(run_dir / f"placement-{args.runtime}.json", placement)
    inference_job = client.submit_inference_job(
        model=target_model,
        device=device,
        inputs={name: [value] for name, value in remote_inputs.items()},
        name=f"lfm-{args.component}-inference-{args.runtime}",
    )
    remote = inference_job.download_output_data()
    summary["jobs"]["inference"] = job_record(inference_job)
    if not isinstance(remote, dict):
        summary.update({"status": "inference_failure"})
        write_json(run_dir / f"summary-{args.runtime}.json", summary)
        print(json.dumps(summary, indent=2, default=str))
        raise SystemExit(4)

    ordered = ordered_remote_outputs(remote)
    comparisons: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {}
    resolved_mapping: dict[str, str] = {}
    for remote_selector, golden_key in args.output:
        resolved_selector = remote_selector
        if remote_selector.startswith("#"):
            index = int(remote_selector[1:])
            if index < len(target_output_names) and target_output_names[index] in remote:
                resolved_selector = target_output_names[index]
        remote_name, actual = select_remote_output(resolved_selector, remote, ordered)
        resolved_mapping[remote_name] = golden_key
        comparisons[golden_key] = compare_output(
            actual, goldens[golden_key], args.atol, args.rtol
        )
        saved[f"actual__{golden_key}"] = actual
        saved[f"golden__{golden_key}"] = goldens[golden_key]
    np.savez(run_dir / f"outputs-{args.runtime}.npz", **saved)

    all_outputs_pass = all(item["passed"] for item in comparisons.values())
    strict_npu_pass = (
        args.runtime == "strict-npu"
        and placement["npu_runtime_layers"] > 0
        and placement["cpu_fallback_runtime_layers"] == 0
        and placement["other_runtime_layers"] == 0
    )
    summary.update(
        {
            "status": "passed" if all_outputs_pass else "numerical_mismatch",
            "resolved_output_mapping": resolved_mapping,
            "comparisons": comparisons,
            "metrics": profile_metrics(profile),
            "placement": placement,
            "strict_npu_pass": strict_npu_pass,
            "profile_artifact": str(profile_path),
        }
    )
    write_json(run_dir / f"summary-{args.runtime}.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    if not all_outputs_pass:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
