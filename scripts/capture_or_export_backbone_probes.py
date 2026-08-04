#!/usr/bin/env python3
"""Capture and export fixed-shape probes for real LFM2 decoder layers.

The probes deliberately stop at one decoder layer.  This keeps the first
backbone feasibility test small enough for a MacBook while preserving the
official layer implementation and official checkpoint weights, including the
layer's normalization and feed-forward sublayers.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer, Lfm2RotaryEmbedding

from path_config import MODEL_REVISION, MODEL_SNAPSHOT, REPO_ROOT, WORK_ROOT

CHECKPOINT = MODEL_SNAPSHOT / "model.safetensors"

PROBES = {
    "conv": {"layer_idx": 0, "expected_layer_type": "conv"},
    "attention": {"layer_idx": 2, "expected_layer_type": "full_attention"},
}


class FixedDecoderLayerProbe(nn.Module):
    """Return only the layer output, with all non-data inputs baked in."""

    def __init__(
        self,
        layer: Lfm2DecoderLayer,
        *,
        layer_type: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.layer_type = layer_type
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("causal_mask", causal_mask)
        self.register_buffer("position_ids", position_ids)
        self.register_buffer("cache_position", cache_position)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The convolution layer does not consume an attention mask.  Passing
        # None avoids presenting its padding-mask helper with a 4-D causal mask.
        attention_mask = self.causal_mask if self.layer_type == "full_attention" else None
        return self.layer(
            hidden_states,
            position_embeddings=(self.cos, self.sin),
            attention_mask=attention_mask,
            position_ids=self.position_ids,
            past_key_values=None,
            cache_position=self.cache_position,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["conv", "attention", "all"], default="all")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/artifacts/backbone-probes",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/exports",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPO_ROOT / "reports",
    )
    return parser.parse_args()


def load_lfm_config() -> Lfm2Config:
    raw = json.loads((MODEL_SNAPSHOT / "config.json").read_text())
    config = Lfm2Config(**raw["lfm"])
    # Explicit eager attention gives a stable, readable operator graph for the
    # first Qualcomm inspection instead of selecting a host-specific backend.
    config._attn_implementation = "eager"
    return config


def load_official_layer(
    config: Lfm2Config, layer_idx: int, *, dtype: torch.dtype
) -> tuple[Lfm2DecoderLayer, list[str]]:
    layer = Lfm2DecoderLayer(config, layer_idx).to(dtype=dtype)
    checkpoint_prefix = f"lfm.layers.{layer_idx}."
    selected: dict[str, torch.Tensor] = {}
    selected_checkpoint_keys: list[str] = []

    with safe_open(CHECKPOINT, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key.startswith(checkpoint_prefix):
                local_key = key.removeprefix(checkpoint_prefix)
                selected[local_key] = reader.get_tensor(key).to(dtype=dtype)
                selected_checkpoint_keys.append(key)

    missing, unexpected = layer.load_state_dict(selected, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Selective layer load mismatch: missing={missing}, unexpected={unexpected}"
        )
    return layer.eval(), selected_checkpoint_keys


def fixed_auxiliary_tensors(
    config: Lfm2Config, sequence_length: int, *, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(sequence_length, dtype=torch.long)
    dummy = torch.zeros(1, sequence_length, config.hidden_size, dtype=dtype)
    rotary = Lfm2RotaryEmbedding(config)
    with torch.inference_mode():
        cos, sin = rotary(dummy, position_ids)
    # Tensors created inside inference_mode carry a permanent inference flag.
    # torch.export's AOTAutograd pass rejects such buffers even for inference;
    # cloning after leaving the context preserves the values as normal tensors.
    cos = cos.clone()
    sin = sin.clone()

    # Additive mask: zero on/below the diagonal, -inf above it.
    causal_mask = torch.full(
        (1, 1, sequence_length, sequence_length),
        float("-inf"),
        dtype=dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    return {
        "cos": cos,
        "sin": sin,
        "causal_mask": causal_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }


def deterministic_input(config: Lfm2Config, sequence_length: int) -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=sequence_length * config.hidden_size,
        dtype=torch.float32,
    )
    # Avoid a highly regular identical pattern across tokens while remaining
    # deterministic and independent of global random state.
    return (values + 0.01 * torch.sin(values * 37.0)).reshape(
        1, sequence_length, config.hidden_size
    )


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value if platform.system() == "Darwin" else value * 1024)


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def benchmark(module: nn.Module, hidden_states: torch.Tensor, runs: int) -> dict[str, float]:
    with torch.inference_mode():
        module(hidden_states)
        elapsed_ms = []
        for _ in range(runs):
            started = time.perf_counter()
            module(hidden_states)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "runs": runs,
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def run_probe(component: str, args: argparse.Namespace) -> dict:
    spec = PROBES[component]
    layer_idx = int(spec["layer_idx"])
    expected_layer_type = str(spec["expected_layer_type"])
    dtype = torch.float32

    config = load_lfm_config()
    actual_layer_type = config.layer_types[layer_idx]
    if actual_layer_type != expected_layer_type:
        raise RuntimeError(
            f"Layer {layer_idx} changed type: expected {expected_layer_type}, got {actual_layer_type}"
        )

    rss_before_load = max_rss_bytes()
    layer, selected_checkpoint_keys = load_official_layer(config, layer_idx, dtype=dtype)
    auxiliaries = fixed_auxiliary_tensors(config, args.sequence_length, dtype=dtype)
    probe = FixedDecoderLayerProbe(
        layer,
        layer_type=actual_layer_type,
        **auxiliaries,
    ).eval()
    rss_after_load = max_rss_bytes()

    hidden_states = deterministic_input(config, args.sequence_length)
    artifact_dir = args.artifact_root / component
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.export_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        eager_output = probe(hidden_states)
    rss_after_eager = max_rss_bytes()

    golden_path = artifact_dir / "golden.pt"
    torch.save(
        {
            "hidden_states": hidden_states,
            "output": eager_output,
            "layer_idx": layer_idx,
            "layer_type": actual_layer_type,
            "model_revision": MODEL_REVISION,
        },
        golden_path,
    )
    golden = torch.load(golden_path, map_location="cpu", weights_only=True)

    eager_latency = benchmark(probe, hidden_states, args.benchmark_runs)

    export_path = args.export_root / f"lfm2_layer{layer_idx}_{component}_seq{args.sequence_length}.pt2"
    export_started = time.perf_counter()
    exported = torch.export.export(probe, (hidden_states,), strict=False)
    export_seconds = time.perf_counter() - export_started
    torch.export.save(exported, export_path)
    rss_after_export = max_rss_bytes()

    operator_counts = Counter(
        str(node.target)
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    )
    # Verify the serialized artifact, rather than only the in-memory object
    # returned by export.
    del exported
    gc.collect()
    saved_exported = torch.export.load(export_path)
    exported_module = saved_exported.module()
    with torch.inference_mode():
        exported_output = exported_module(hidden_states)
    exported_latency = benchmark(exported_module, hidden_states, args.benchmark_runs)

    numpy_path = args.export_root / f"lfm2_layer{layer_idx}_{component}_seq{args.sequence_length}.npz"
    np.savez(
        numpy_path,
        hidden_states=hidden_states.numpy(),
        output=golden["output"].numpy(),
    )

    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in probe.parameters()
    )
    golden_error = tensor_error(exported_output, golden["output"])
    status = "passed" if golden_error["max_abs"] == 0.0 else "failed"
    report = {
        "status": status,
        "stage": "local_fixed_shape_export",
        "component": component,
        "model": "LiquidAI/LFM2.5-Audio-1.5B",
        "model_revision": MODEL_REVISION,
        "transformers_layer_class": "Lfm2DecoderLayer",
        "layer_idx": layer_idx,
        "layer_type": actual_layer_type,
        "checkpoint": str(CHECKPOINT),
        "selective_checkpoint_key_count": len(selected_checkpoint_keys),
        "selective_checkpoint_keys": selected_checkpoint_keys,
        "dtype": str(dtype).removeprefix("torch."),
        "fixed_input_shapes": {"hidden_states": list(hidden_states.shape)},
        "flat_output_shapes": {"output": list(exported_output.shape)},
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "eager_latency": eager_latency,
        "exported_latency": exported_latency,
        "export_seconds": export_seconds,
        "memory": {
            "metric": "process_max_rss_high_water_mark",
            "rss_before_load_bytes": rss_before_load,
            "rss_after_load_bytes": rss_after_load,
            "rss_after_eager_bytes": rss_after_eager,
            "rss_after_export_bytes": rss_after_export,
        },
        "export": {
            "format": "torch.export pt2",
            "artifact": str(export_path),
            "artifact_bytes": export_path.stat().st_size,
            "numpy_io": str(numpy_path),
            "golden": str(golden_path),
        },
        "exported_vs_local_golden": golden_error,
        "exported_aten_operator_counts": dict(sorted(operator_counts.items())),
        "aihub": {
            "status": "not_submitted",
            "operator_placement": "pending AI Hub credentials and target device selection",
        },
        "scope_note": (
            "Prefill-style fixed sequence probe without KV cache. It includes the official "
            "operator, normalization, residual, and feed-forward sublayers."
        ),
    }
    report_path = args.report_root / f"backbone_{component}_probe.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    if args.benchmark_runs < 1:
        raise ValueError("--benchmark-runs must be positive")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {CHECKPOINT}")

    components = list(PROBES) if args.component == "all" else [args.component]
    summaries = []
    for component in components:
        summaries.append(run_probe(component, args))
        gc.collect()

    if any(summary["status"] != "passed" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
