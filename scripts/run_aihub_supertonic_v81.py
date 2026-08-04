#!/usr/bin/env python3
"""Compile, profile, and compare fixed-shape Supertonic 3 on SM8850 v81.

Run this script from a detached tmux session.  It submits every selected
component before polling, persists each job ID immediately, and keeps failures
as evidence so one unsupported graph does not wedge the batch.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import numpy as np
import qai_hub as hub


COMPONENT_MODELS = {
    "duration_predictor": "duration_predictor.int8.onnx",
    "text_encoder": "text_encoder.int8.onnx",
    "vector_estimator": "vector_estimator.int8.onnx",
    "vocoder": "vocoder.int8.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--io-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--component", action="append", choices=tuple(COMPONENT_MODELS)
    )
    parser.add_argument("--device", default="Snapdragon 8 Elite Gen 5 QRD")
    parser.add_argument("--device-os", default="16")
    parser.add_argument("--qairt-version", default="2.45")
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ceiling-seconds", type=int, default=10_800)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    args = parser.parse_args()
    if args.profile_iterations < 1 or args.poll_seconds < 1 or args.ceiling_seconds < 1:
        parser.error("iteration, poll, and ceiling values must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_name(dtype: np.dtype[Any]) -> str:
    names = {
        np.dtype("float32"): "float32",
        np.dtype("float16"): "float16",
        np.dtype("int64"): "int64",
        np.dtype("int32"): "int32",
        np.dtype("int16"): "int16",
        np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8",
        np.dtype("bool"): "bool",
    }
    if np.dtype(dtype) not in names:
        raise ValueError(f"Unsupported AI Hub dtype: {dtype}")
    return names[np.dtype(dtype)]


def status_record(job: Any) -> dict[str, Any]:
    status = job.get_status()
    return {
        "job_id": getattr(job, "job_id", None),
        "url": getattr(job, "url", None),
        "status": status.code,
        "message": status.message,
        "finished": bool(status.finished),
        "success": bool(status.success),
    }


def poll(
    jobs: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    stage: str,
    poll_seconds: int,
    ceiling_seconds: int,
) -> dict[str, dict[str, Any]]:
    started = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            snapshot = {name: status_record(job) for name, job in jobs.items()}
            consecutive_errors = 0
        except Exception as error:
            consecutive_errors += 1
            state.setdefault("poll_errors", []).append(
                {
                    "stage": stage,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "type": type(error).__name__,
                    "message": str(error),
                    "consecutive": consecutive_errors,
                }
            )
            write_json(state_path, state)
            if time.monotonic() - started >= ceiling_seconds:
                return {}
            time.sleep(min(60, poll_seconds * max(1, consecutive_errors)))
            continue
        state.setdefault("polls", []).append(
            {
                "stage": stage,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "jobs": snapshot,
            }
        )
        state.setdefault("jobs", {})[stage] = snapshot
        write_json(state_path, state)
        print(json.dumps({"stage": stage, "jobs": snapshot}, indent=2), flush=True)
        if all(value["finished"] for value in snapshot.values()):
            return snapshot
        if time.monotonic() - started >= ceiling_seconds:
            state.setdefault("pending_at_ceiling", {})[stage] = {
                name: value for name, value in snapshot.items() if not value["finished"]
            }
            write_json(state_path, state)
            return snapshot
        time.sleep(poll_seconds)


def compare(actual: np.ndarray, golden: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
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
    actual64 = actual.astype(np.float64).ravel()
    golden64 = golden.astype(np.float64).ravel()
    signed = actual64 - golden64
    absolute = np.abs(signed)
    rmse = float(np.sqrt(np.mean(signed * signed))) if signed.size else 0.0
    golden_rms = float(np.sqrt(np.mean(golden64 * golden64))) if signed.size else 0.0
    denominator = float(np.linalg.norm(actual64) * np.linalg.norm(golden64))
    result.update(
        {
            "comparison": "allclose",
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": float(absolute.max()) if absolute.size else 0.0,
            "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            "p95_abs_error": float(np.percentile(absolute, 95)) if absolute.size else 0.0,
            "rmse": rmse,
            "golden_rms": golden_rms,
            "normalized_rmse": rmse / golden_rms if golden_rms else 0.0,
            "cosine_similarity": (
                float(np.dot(actual64, golden64) / denominator) if denominator else 1.0
            ),
            "passed": bool(np.allclose(actual, golden, atol=atol, rtol=rtol)),
        }
    )
    return result


def classify_profile(profile: dict[str, Any]) -> dict[str, Any]:
    layers = profile.get("execution_detail", []) or []
    counts = Counter(str(layer.get("compute_unit", "UNSPECIFIED")).upper() for layer in layers)
    npu = [layer for layer in layers if "NPU" in str(layer.get("compute_unit", "")).upper()]
    cpu = [layer for layer in layers if "CPU" in str(layer.get("compute_unit", "")).upper()]
    other = [layer for layer in layers if layer not in npu and layer not in cpu]
    return {
        "unit_counts": dict(sorted(counts.items())),
        "npu_runtime_layers": len(npu),
        "cpu_fallback_runtime_layers": len(cpu),
        "other_runtime_layers": len(other),
        "layers": {"hexagon_npu": npu, "cpu_fallback": cpu, "other": other},
    }


def profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    execution = profile.get("execution_summary", {}) or {}
    latency_us = execution.get("estimated_inference_time")
    peak_bytes = execution.get("estimated_inference_peak_memory")
    return {
        "latency_ms": None if latency_us is None else float(latency_us) / 1000.0,
        "peak_memory_mib": None if peak_bytes is None else float(peak_bytes) / (1024**2),
        "raw_execution_summary": execution,
    }


def download_logs(job: Any, path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return {"status": "downloaded", "files": job.download_job_logs(str(path))}
    except Exception as error:
        return {"status": "failed", "type": type(error).__name__, "message": str(error)}


def upload_with_retry(path: Path, name: str, attempts: int = 5) -> Any:
    """Retry transient S3/TLS failures while keeping the batch unattended."""

    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return hub.upload_model(str(path), name=name)
        except Exception as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt == attempts:
                raise RuntimeError("AI Hub upload failed after retries: " + " | ".join(errors)) from error
            time.sleep(min(60, 5 * (2 ** (attempt - 1))))
    raise AssertionError("unreachable")


def load_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path) as archive:
        inputs = {
            key.removeprefix("input__"): np.array(archive[key])
            for key in archive.files
            if key.startswith("input__")
        }
        goldens = {
            key.removeprefix("golden__"): np.array(archive[key])
            for key in archive.files
            if key.startswith("golden__")
        }
    if not inputs or len(goldens) != 1:
        raise ValueError(f"Expected inputs and one golden in {path}")
    return inputs, goldens


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    io_dir = args.io_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    components = args.component or list(COMPONENT_MODELS)
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qai_hub": getattr(hub, "__version__", "unknown"),
        },
        "target": {
            "device": args.device,
            "os": args.device_os,
            "chipset": "sm8850",
            "htp": "v81",
            "qairt_version": args.qairt_version,
        },
        "components": components,
        "items": {},
        "jobs": {},
        "polls": [],
    }
    write_json(state_path, state)
    client = hub.Client()
    device = hub.Device(args.device, args.device_os, ["framework:qnn"])
    compile_options = (
        "--target_runtime qnn_context_binary "
        "--qnn_options default_graph_htp_precision=FLOAT16 "
        f"--qairt_version {args.qairt_version}"
    )
    profile_options = (
        f"--qairt_version {args.qairt_version} "
        f"--max_profiler_iterations {args.profile_iterations}"
    )

    compile_jobs: dict[str, Any] = {}
    prepared: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for name in components:
        item: dict[str, Any] = {"status": "preparing"}
        state["items"][name] = item
        try:
            model_path = model_dir / COMPONENT_MODELS[name]
            bundle_path = io_dir / f"{name}.npz"
            if not model_path.is_file() or not bundle_path.is_file():
                raise FileNotFoundError(model_path if not model_path.is_file() else bundle_path)
            inputs, goldens = load_bundle(bundle_path)
            has_int64 = any(value.dtype == np.int64 for value in inputs.values())
            options = compile_options + (" --truncate_64bit_io" if has_int64 else "")
            input_specs = {
                key: (tuple(value.shape), dtype_name(value.dtype))
                for key, value in inputs.items()
            }
            remote_inputs = {
                key: value.astype(np.int32, copy=False) if value.dtype == np.int64 else value
                for key, value in inputs.items()
            }
            item.update(
                {
                    "model": str(model_path),
                    "model_bytes": model_path.stat().st_size,
                    "model_sha256": sha256(model_path),
                    "bundle": str(bundle_path),
                    "input_specs": {
                        key: [list(shape), dtype] for key, (shape, dtype) in input_specs.items()
                    },
                    "remote_input_dtypes": {
                        key: str(value.dtype) for key, value in remote_inputs.items()
                    },
                    "golden_outputs": {
                        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in goldens.items()
                    },
                    "compile_options": options,
                    "profile_options": profile_options,
                }
            )
            source_model = upload_with_retry(
                model_path, name=f"lfm-demo-supertonic-{name}"
            )
            item["source_model_id"] = source_model.model_id
            job = client.submit_compile_job(
                model=source_model,
                device=device,
                input_specs=input_specs,
                options=options,
                name=f"lfm-demo-v81-supertonic-{name}",
                retry=False,
            )
            compile_jobs[name] = job
            prepared[name] = (remote_inputs, goldens)
            item["jobs"] = {"compile": status_record(job)}
            item["status"] = "compile_submitted"
            write_json(state_path, state)
        except Exception as error:
            item["status"] = "submission_error"
            item["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(state_path, state)

    compile_snapshot = poll(
        compile_jobs,
        state,
        state_path,
        "compile",
        args.poll_seconds,
        args.ceiling_seconds,
    ) if compile_jobs else {}

    profile_jobs: dict[str, Any] = {}
    inference_jobs: dict[str, Any] = {}
    target_models: dict[str, Any] = {}
    for name, job in compile_jobs.items():
        item = state["items"][name]
        snapshot = compile_snapshot.get(name, {})
        item["jobs"]["compile"] = snapshot or item["jobs"]["compile"]
        if not snapshot.get("success"):
            item["status"] = "compile_pending" if not snapshot.get("finished") else "compile_failed"
            item["job_logs"] = {"compile": download_logs(job, output_dir / name / "job_logs" / "compile")}
            write_json(state_path, state)
            continue
        target = job.get_target_model()
        if target is None:
            item["status"] = "compile_returned_no_model"
            write_json(state_path, state)
            continue
        target_models[name] = target
        remote_inputs, _ = prepared[name]
        profile = client.submit_profile_job(
            model=target,
            device=device,
            options=profile_options,
            name=f"lfm-demo-v81-supertonic-{name}-profile",
            retry=False,
        )
        inference = client.submit_inference_job(
            model=target,
            device=device,
            inputs={key: [value] for key, value in remote_inputs.items()},
            options=f"--qairt_version {args.qairt_version}",
            name=f"lfm-demo-v81-supertonic-{name}-inference",
            retry=False,
        )
        profile_jobs[name] = profile
        inference_jobs[name] = inference
        item["jobs"]["profile"] = status_record(profile)
        item["jobs"]["inference"] = status_record(inference)
        item["status"] = "profile_and_inference_submitted"
        write_json(state_path, state)

    dependent_jobs = {
        **{f"{name}:profile": job for name, job in profile_jobs.items()},
        **{f"{name}:inference": job for name, job in inference_jobs.items()},
    }
    dependent_snapshot = poll(
        dependent_jobs,
        state,
        state_path,
        "profile_and_inference",
        args.poll_seconds,
        args.ceiling_seconds,
    ) if dependent_jobs else {}

    for name in target_models:
        item = state["items"][name]
        profile_snapshot = dependent_snapshot.get(f"{name}:profile", {})
        inference_snapshot = dependent_snapshot.get(f"{name}:inference", {})
        item["jobs"]["profile"] = profile_snapshot or item["jobs"]["profile"]
        item["jobs"]["inference"] = inference_snapshot or item["jobs"]["inference"]
        if not profile_snapshot.get("success") or not inference_snapshot.get("success"):
            item["status"] = (
                "remote_jobs_pending"
                if not profile_snapshot.get("finished") or not inference_snapshot.get("finished")
                else "profile_or_inference_failed"
            )
            write_json(state_path, state)
            continue
        try:
            profile = profile_jobs[name].download_profile()
            remote = inference_jobs[name].download_output_data()
            if not isinstance(profile, dict) or not isinstance(remote, dict):
                raise TypeError("AI Hub profile or inference output is not a dictionary")
            item_dir = output_dir / name
            write_json(item_dir / "profile.json", profile)
            placement = classify_profile(profile)
            write_json(item_dir / "placement.json", placement)
            _, goldens = prepared[name]
            golden_name, golden = next(iter(goldens.items()))
            remote_name, samples = next(iter(remote.items()))
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError(f"Unexpected remote output structure for {remote_name}")
            actual = np.asarray(samples[0])
            comparison = compare(actual, golden, args.atol, args.rtol)
            np.savez(
                item_dir / "remote_outputs.npz",
                **{f"actual__{golden_name}": actual, f"golden__{golden_name}": golden},
            )
            strict_npu = (
                placement["npu_runtime_layers"] > 0
                and placement["cpu_fallback_runtime_layers"] == 0
                and placement["other_runtime_layers"] == 0
            )
            item["result"] = {
                "metrics": profile_metrics(profile),
                "placement": placement,
                "strict_npu_pass": strict_npu,
                "remote_output_name": remote_name,
                "golden_output_name": golden_name,
                "comparison": comparison,
            }
            item["status"] = (
                "passed" if strict_npu and comparison["passed"] else "completed_with_warning"
            )
            item["job_logs"] = {
                "compile": download_logs(compile_jobs[name], item_dir / "job_logs" / "compile"),
                "profile": download_logs(profile_jobs[name], item_dir / "job_logs" / "profile"),
                "inference": download_logs(inference_jobs[name], item_dir / "job_logs" / "inference"),
            }
            write_json(state_path, state)
        except Exception as error:
            item["status"] = "result_download_error"
            item["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(state_path, state)

    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    statuses = {name: value["status"] for name, value in state["items"].items()}
    state["summary"] = {
        "statuses": statuses,
        "passed": sum(value == "passed" for value in statuses.values()),
        "total": len(statuses),
    }
    write_json(state_path, state)
    print(json.dumps(state["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
