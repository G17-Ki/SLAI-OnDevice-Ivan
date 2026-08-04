#!/usr/bin/env python3
"""Create deterministic fixed-shape Supertonic 3 inputs and local goldens.

The public Sherpa-ONNX package keeps text preprocessing outside the neural
graphs.  This helper exercises the four exported INT8 ONNX components below
that boundary with representative fixed shapes suitable for Qualcomm AI Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import onnxruntime as ort


COMPONENT_MODELS = {
    "duration_predictor": "duration_predictor.int8.onnx",
    "text_encoder": "text_encoder.int8.onnx",
    "vector_estimator": "vector_estimator.int8.onnx",
    "vocoder": "vocoder.int8.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--latent-length", type=int, default=32)
    parser.add_argument("--speaker", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.text_length < 1 or args.latent_length < 1:
        parser.error("fixed lengths must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_voice_style(path: Path, speaker: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = path.read_bytes()
    header_bytes = 6 * 8
    if len(payload) < header_bytes:
        raise ValueError(f"Voice style header is truncated: {path}")
    dims = struct.unpack("<6q", payload[:header_bytes])
    if any(value <= 0 for value in dims):
        raise ValueError(f"Invalid voice style dimensions: {dims}")
    ttl_shape = tuple(int(value) for value in dims[:3])
    dp_shape = tuple(int(value) for value in dims[3:])
    if ttl_shape[0] != dp_shape[0]:
        raise ValueError("TTL and duration style speaker counts differ")
    if not 0 <= speaker < ttl_shape[0]:
        raise ValueError(f"speaker must be in [0, {ttl_shape[0] - 1}]")

    ttl_count = int(np.prod(ttl_shape))
    dp_count = int(np.prod(dp_shape))
    expected = header_bytes + (ttl_count + dp_count) * 4
    if len(payload) != expected:
        raise ValueError(f"Voice style size mismatch: got {len(payload)}, expected {expected}")
    values = np.frombuffer(payload, dtype="<f4", offset=header_bytes)
    ttl = values[:ttl_count].reshape(ttl_shape)[speaker : speaker + 1].copy()
    dp = values[ttl_count:].reshape(dp_shape)[speaker : speaker + 1].copy()
    return ttl, dp, {
        "speaker_count": ttl_shape[0],
        "selected_speaker": speaker,
        "ttl_full_shape": list(ttl_shape),
        "dp_full_shape": list(dp_shape),
    }


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_component(
    model_path: Path, inputs: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    runner = session(model_path)
    input_meta = [
        {"name": value.name, "shape": value.shape, "type": value.type}
        for value in runner.get_inputs()
    ]
    output_meta = [
        {"name": value.name, "shape": value.shape, "type": value.type}
        for value in runner.get_outputs()
    ]
    expected = [value.name for value in runner.get_inputs()]
    if set(expected) != set(inputs):
        raise ValueError(
            f"{model_path.name} input mismatch: expected {expected}, got {list(inputs)}"
        )
    output_values = runner.run(None, {name: inputs[name] for name in expected})
    outputs = {
        meta.name: np.asarray(value)
        for meta, value in zip(runner.get_outputs(), output_values, strict=True)
    }
    for name, value in {**inputs, **outputs}.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{model_path.name} tensor {name} contains NaN or Inf")
    return outputs, input_meta, output_meta


def describe_array(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()) if value.size else None,
        "max": float(value.max()) if value.size else None,
        "mean": float(value.astype(np.float64).mean()) if value.size else None,
    }


def write_bundle(
    output_dir: Path,
    name: str,
    model_path: Path,
    inputs: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    input_meta: list[dict[str, Any]],
    output_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    bundle_path = output_dir / f"{name}.npz"
    arrays = {f"input__{key}": value for key, value in inputs.items()}
    arrays.update({f"golden__{key}": value for key, value in outputs.items()})
    np.savez(bundle_path, **arrays)
    return {
        "model": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256(model_path),
        "bundle": str(bundle_path.resolve()),
        "bundle_bytes": bundle_path.stat().st_size,
        "inputs": {key: describe_array(value) for key, value in inputs.items()},
        "outputs": {key: describe_array(value) for key, value in outputs.items()},
        "onnx_input_metadata": input_meta,
        "onnx_output_metadata": output_meta,
    }


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [*COMPONENT_MODELS.values(), "voice.bin"]:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)

    style_ttl, style_dp, voice = parse_voice_style(
        model_dir / "voice.bin", args.speaker
    )
    rng = np.random.default_rng(args.seed)
    text_ids = (np.arange(args.text_length, dtype=np.int64) + 32)[None, :]
    text_mask = np.ones((1, 1, args.text_length), dtype=np.float32)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "fixed_shapes": {
            "batch": 1,
            "text_length": args.text_length,
            "latent_length": args.latent_length,
            "latent_channels": 144,
        },
        "seed": args.seed,
        "voice": voice,
        "components": {},
    }

    duration_inputs = {
        "text_ids": text_ids,
        "style_dp": style_dp.astype(np.float32, copy=False),
        "text_mask": text_mask,
    }
    duration_outputs, input_meta, output_meta = run_component(
        model_dir / COMPONENT_MODELS["duration_predictor"], duration_inputs
    )
    manifest["components"]["duration_predictor"] = write_bundle(
        output_dir,
        "duration_predictor",
        model_dir / COMPONENT_MODELS["duration_predictor"],
        duration_inputs,
        duration_outputs,
        input_meta,
        output_meta,
    )

    encoder_inputs = {
        "text_ids": text_ids,
        "style_ttl": style_ttl.astype(np.float32, copy=False),
        "text_mask": text_mask,
    }
    encoder_outputs, input_meta, output_meta = run_component(
        model_dir / COMPONENT_MODELS["text_encoder"], encoder_inputs
    )
    manifest["components"]["text_encoder"] = write_bundle(
        output_dir,
        "text_encoder",
        model_dir / COMPONENT_MODELS["text_encoder"],
        encoder_inputs,
        encoder_outputs,
        input_meta,
        output_meta,
    )

    text_emb = next(iter(encoder_outputs.values())).astype(np.float32, copy=False)
    vector_inputs = {
        "noisy_latent": rng.standard_normal(
            (1, 144, args.latent_length), dtype=np.float32
        ),
        "text_emb": text_emb,
        "style_ttl": style_ttl.astype(np.float32, copy=False),
        "latent_mask": np.ones((1, 1, args.latent_length), dtype=np.float32),
        "text_mask": text_mask,
        "current_step": np.asarray([0.0], dtype=np.float32),
        "total_step": np.asarray([5.0], dtype=np.float32),
    }
    vector_outputs, input_meta, output_meta = run_component(
        model_dir / COMPONENT_MODELS["vector_estimator"], vector_inputs
    )
    manifest["components"]["vector_estimator"] = write_bundle(
        output_dir,
        "vector_estimator",
        model_dir / COMPONENT_MODELS["vector_estimator"],
        vector_inputs,
        vector_outputs,
        input_meta,
        output_meta,
    )

    vocoder_inputs = {
        "latent": next(iter(vector_outputs.values())).astype(np.float32, copy=False)
    }
    vocoder_outputs, input_meta, output_meta = run_component(
        model_dir / COMPONENT_MODELS["vocoder"], vocoder_inputs
    )
    manifest["components"]["vocoder"] = write_bundle(
        output_dir,
        "vocoder",
        model_dir / COMPONENT_MODELS["vocoder"],
        vocoder_inputs,
        vocoder_outputs,
        input_meta,
        output_meta,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
