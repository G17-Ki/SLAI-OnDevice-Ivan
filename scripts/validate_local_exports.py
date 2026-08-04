#!/usr/bin/env python3
"""Reload every fixed-shape PT2 artifact and compare it with saved goldens."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from path_config import REPO_ROOT, WORK_ROOT

EXPORT_ROOT = WORK_ROOT / "lfm-feasibility/exports"
ARTIFACT_ROOT = WORK_ROOT / "lfm-feasibility/artifacts"
DEFAULT_REPORT = REPO_ROOT / "reports/local_export_validation.json"


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_equal": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = (actual.float() - expected.float()).abs()
    return {
        "shape_equal": True,
        "exact_equal": bool(torch.equal(actual, expected)),
        "max_abs_error": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean()) if delta.numel() else 0.0,
    }


def load_npz_case(
    npz_name: str,
    input_keys: tuple[str, ...],
    output_keys: tuple[str, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    with np.load(EXPORT_ROOT / npz_name) as arrays:
        inputs = tuple(torch.from_numpy(arrays[key]) for key in input_keys)
        outputs = tuple(torch.from_numpy(arrays[key]) for key in output_keys)
    return inputs, outputs


def load_pt_case(
    input_path: Path,
    output_path: Path,
    input_keys: tuple[str, ...],
    output_keys: tuple[str, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    saved_inputs = torch.load(input_path, map_location="cpu", weights_only=True)
    saved_outputs = torch.load(output_path, map_location="cpu", weights_only=True)
    return (
        tuple(saved_inputs[key] for key in input_keys),
        tuple(saved_outputs[key] for key in output_keys),
    )


def normalize_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in value):
        return tuple(value)
    raise TypeError(f"Unsupported exported output type: {type(value)!r}")


def validate_case(
    name: str,
    model_path: Path,
    load_io: Callable[[], tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    inputs, expected_outputs = load_io()
    load_started = time.perf_counter()
    exported = torch.export.load(model_path)
    module = exported.module()
    load_seconds = time.perf_counter() - load_started
    inference_started = time.perf_counter()
    with torch.inference_mode():
        actual_outputs = normalize_outputs(module(*inputs))
    inference_seconds = time.perf_counter() - inference_started

    if len(actual_outputs) != len(expected_outputs):
        errors: list[dict[str, Any]] = [{
            "output_count_equal": False,
            "actual_count": len(actual_outputs),
            "expected_count": len(expected_outputs),
        }]
        passed = False
    else:
        errors = [
            tensor_error(actual, expected)
            for actual, expected in zip(actual_outputs, expected_outputs, strict=True)
        ]
        passed = all(
            error.get("shape_equal") and error.get("max_abs_error", float("inf")) <= tolerance
            for error in errors
        )

    result = {
        "name": name,
        "status": "passed" if passed else "failed",
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "output_errors": errors,
        "absolute_tolerance": tolerance,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    del module, exported, inputs, expected_outputs, actual_outputs
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--case",
        action="append",
        help="Validate only the named case; repeat to select multiple cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases: dict[
        str,
        tuple[
            Path,
            Callable[[], tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]],
            float,
        ],
    ] = {
        "fastconformer": (
            EXPORT_ROOT / "fastconformer_adapter_mel80.pt2",
            lambda: load_npz_case("fastconformer_adapter_mel80.npz", ("mel",), ("adapted",)),
            0.0,
        ),
        "backbone_conv": (
            EXPORT_ROOT / "lfm2_layer0_conv_seq16.pt2",
            lambda: load_npz_case("lfm2_layer0_conv_seq16.npz", ("hidden_states",), ("output",)),
            0.0,
        ),
        "backbone_attention": (
            EXPORT_ROOT / "lfm2_layer2_attention_seq16.pt2",
            lambda: load_npz_case("lfm2_layer2_attention_seq16.npz", ("hidden_states",), ("output",)),
            0.0,
        ),
        "backbone_conv_cached_decode": (
            EXPORT_ROOT / "lfm2_layer0_conv_decode_cache.pt2",
            lambda: load_npz_case(
                "lfm2_layer0_conv_decode_cache.npz",
                ("hidden_states", "conv_cache"),
                ("output", "updated_conv_cache"),
            ),
            0.0,
        ),
        "backbone_attention_cached_decode": (
            EXPORT_ROOT / "lfm2_layer2_attention_decode_kv16_past8.pt2",
            lambda: load_npz_case(
                "lfm2_layer2_attention_decode_kv16_past8.npz",
                ("hidden_states", "key_cache", "value_cache"),
                ("output", "updated_key_cache", "updated_value_cache"),
            ),
            1e-7,
        ),
        "depth_decoder": (
            EXPORT_ROOT / "depth_decoder_hidden1x2048.pt2",
            lambda: load_pt_case(
                ARTIFACT_ROOT / "depth-probe-golden/inputs.pt",
                ARTIFACT_ROOT / "depth-probe-golden/outputs.pt",
                ("hidden",),
                ("tokens", "next_audio_embedding"),
            ),
            0.0,
        ),
        "detokenizer_t4": (
            EXPORT_ROOT / "detok_neural_probe_codes_t4.pt2",
            lambda: load_pt_case(
                ARTIFACT_ROOT / "detok-neural-probe-t4/inputs.pt",
                ARTIFACT_ROOT / "detok-neural-probe-t4/outputs.pt",
                ("codes",),
                ("log_abs", "angle"),
            ),
            0.0,
        ),
        "detokenizer_t8": (
            EXPORT_ROOT / "detok_neural_probe_codes_t8.pt2",
            lambda: load_pt_case(
                ARTIFACT_ROOT / "detok-neural-probe-t8/inputs.pt",
                ARTIFACT_ROOT / "detok-neural-probe-t8/outputs.pt",
                ("codes",),
                ("log_abs", "angle"),
            ),
            0.0,
        ),
    }

    selected = args.case or list(cases)
    unknown = sorted(set(selected) - set(cases))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}; choices={sorted(cases)}")

    results = []
    for name in selected:
        model_path, loader, tolerance = cases[name]
        if not model_path.is_file():
            results.append({"name": name, "status": "missing", "model": str(model_path)})
            continue
        result = validate_case(name, model_path, loader, tolerance)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    report = {
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "validation_scope": "serialized PT2 reload versus saved local golden tensors",
        "torch": torch.__version__,
        "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
