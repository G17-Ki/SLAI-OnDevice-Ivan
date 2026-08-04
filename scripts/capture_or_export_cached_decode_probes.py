#!/usr/bin/env python3
"""Export one-token LFM2 decode probes with explicit fixed-shape cache I/O.

The official Transformers implementation owns cache tensors inside a mutable
``Lfm2HybridConvCache`` Python object. That interface cannot be a deployment
graph boundary, so these wrappers express the equivalent update with ordinary
input/output tensors while retaining the official layer weights and math.
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
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2DecoderLayer,
    Lfm2HybridConvCache,
    Lfm2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from path_config import MODEL_REVISION, MODEL_SNAPSHOT, REPO_ROOT, WORK_ROOT

CHECKPOINT = MODEL_SNAPSHOT / "model.safetensors"

PROBES = {
    "conv": {"layer_idx": 0, "expected_layer_type": "conv"},
    "attention": {"layer_idx": 2, "expected_layer_type": "full_attention"},
}


class ExplicitConvCacheDecode(nn.Module):
    """One official conv decoder layer with cache tensor in/out."""

    def __init__(self, layer: Lfm2DecoderLayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self, hidden_states: torch.Tensor, conv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized = self.layer.operator_norm(hidden_states)
        projected = self.layer.conv.in_proj(normalized).transpose(-1, -2)
        b_gate, c_gate, x = projected.chunk(3, dim=-2)
        bx = b_gate * x

        # Fixed one-token decode: drop the oldest state and append the current
        # projected token. This is equivalent to the official roll + final-slot
        # assignment path for cache_position > 0.
        updated_cache = torch.cat((conv_cache[:, :, 1:], bx), dim=-1)
        conv_out = torch.sum(
            updated_cache * self.layer.conv.conv.weight[:, 0, :], dim=-1
        )
        if self.layer.conv.bias:
            conv_out = conv_out + self.layer.conv.conv.bias
        operator_output = self.layer.conv.out_proj(
            (c_gate * conv_out.unsqueeze(-1)).transpose(-1, -2).contiguous()
        )
        output = residual + operator_output
        output = output + self.layer.feed_forward(self.layer.ffn_norm(output))
        return output, updated_cache


class ExplicitAttentionCacheDecode(nn.Module):
    """One official attention decoder layer with fixed-capacity KV tensor I/O."""

    def __init__(
        self,
        layer: Lfm2DecoderLayer,
        *,
        capacity: int,
        past_length: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.capacity = capacity
        self.past_length = past_length
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        mask = torch.full((1, 1, 1, capacity), float("-inf"), dtype=cos.dtype)
        mask[:, :, :, : past_length + 1] = 0.0
        self.register_buffer("attention_mask", mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized = self.layer.operator_norm(hidden_states)
        attention = self.layer.self_attn
        input_shape = normalized.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)

        query = attention.q_layernorm(
            attention.q_proj(normalized).view(*hidden_shape)
        ).transpose(1, 2)
        key = attention.k_layernorm(
            attention.k_proj(normalized).view(*hidden_shape)
        ).transpose(1, 2)
        value = attention.v_proj(normalized).view(*hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, self.cos, self.sin)

        # The write slot and valid length are constants in this minimal graph.
        # Tail slots remain explicit state and are masked out of attention.
        updated_key = torch.cat(
            (
                key_cache[:, :, : self.past_length, :],
                key,
                key_cache[:, :, self.past_length + 1 :, :],
            ),
            dim=2,
        )
        updated_value = torch.cat(
            (
                value_cache[:, :, : self.past_length, :],
                value,
                value_cache[:, :, self.past_length + 1 :, :],
            ),
            dim=2,
        )
        attention_output, _ = eager_attention_forward(
            attention,
            query,
            updated_key,
            updated_value,
            self.attention_mask,
            dropout=0.0,
            scaling=attention.scaling,
        )
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        output = residual + attention.out_proj(attention_output)
        output = output + self.layer.feed_forward(self.layer.ffn_norm(output))
        return output, updated_key, updated_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("conv", "attention", "all"), default="all")
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--past-length", type=int, default=8)
    parser.add_argument("--benchmark-runs", type=int, default=5)
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


def load_config() -> Lfm2Config:
    raw = json.loads((MODEL_SNAPSHOT / "config.json").read_text())
    config = Lfm2Config(**raw["lfm"])
    config._attn_implementation = "eager"
    return config


def load_layer(
    config: Lfm2Config, layer_idx: int
) -> tuple[Lfm2DecoderLayer, list[str]]:
    layer = Lfm2DecoderLayer(config, layer_idx).to(dtype=torch.float32)
    prefix = f"lfm.layers.{layer_idx}."
    selected: dict[str, torch.Tensor] = {}
    selected_keys: list[str] = []
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key.startswith(prefix):
                selected[key.removeprefix(prefix)] = reader.get_tensor(key).float()
                selected_keys.append(key)
    missing, unexpected = layer.load_state_dict(selected, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Layer load mismatch: missing={missing}, unexpected={unexpected}")
    return layer.eval(), selected_keys


def deterministic_hidden(length: int, hidden_size: int, *, phase: float) -> torch.Tensor:
    index = torch.arange(length * hidden_size, dtype=torch.float32)
    values = 0.25 * torch.sin(index / 37.0 + phase) + 0.1 * torch.cos(index / 101.0)
    return values.reshape(1, length, hidden_size).contiguous()


def rotary_for_positions(
    config: Lfm2Config, hidden_states: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Lfm2RotaryEmbedding(config)
    with torch.inference_mode():
        cos, sin = rotary(hidden_states, positions)
    return cos.clone(), sin.clone()


def causal_mask(length: int) -> torch.Tensor:
    mask = torch.full((1, 1, length, length), float("-inf"), dtype=torch.float32)
    return torch.triu(mask, diagonal=1)


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def benchmark(module: nn.Module, inputs: tuple[torch.Tensor, ...], runs: int) -> dict[str, Any]:
    samples: list[float] = []
    with torch.inference_mode():
        module(*inputs)
        for _ in range(runs):
            started = time.perf_counter()
            module(*inputs)
            samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "runs": runs,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def operator_counts(exported: torch.export.ExportedProgram) -> dict[str, int]:
    counts = Counter(
        str(node.target)
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    )
    return dict(sorted(counts.items()))


def run_conv(args: argparse.Namespace, config: Lfm2Config) -> dict[str, Any]:
    layer_idx = int(PROBES["conv"]["layer_idx"])
    layer, selected_keys = load_layer(config, layer_idx)
    past_hidden = deterministic_hidden(args.past_length, config.hidden_size, phase=0.0)
    current_hidden = deterministic_hidden(1, config.hidden_size, phase=1.0)
    cache = Lfm2HybridConvCache(config, max_batch_size=1, dtype=torch.float32)

    past_positions = torch.arange(args.past_length, dtype=torch.long).unsqueeze(0)
    past_cos_sin = rotary_for_positions(config, past_hidden, past_positions)
    with torch.inference_mode():
        layer(
            past_hidden,
            position_embeddings=past_cos_sin,
            past_key_values=cache,
            cache_position=torch.arange(args.past_length, dtype=torch.long),
        )
    input_cache = cache.conv_cache[layer_idx].clone()

    current_position = torch.tensor([[args.past_length]], dtype=torch.long)
    current_cos_sin = rotary_for_positions(config, current_hidden, current_position)
    with torch.inference_mode():
        official_output = layer(
            current_hidden,
            position_embeddings=current_cos_sin,
            past_key_values=cache,
            cache_position=torch.tensor([args.past_length], dtype=torch.long),
        )
    official_cache = cache.conv_cache[layer_idx].clone()

    wrapper = ExplicitConvCacheDecode(layer).eval()
    inputs = (current_hidden, input_cache)
    with torch.inference_mode():
        eager_output, eager_cache = wrapper(*inputs)
    eager_errors = {
        "output": tensor_error(eager_output, official_output),
        "conv_cache": tensor_error(eager_cache, official_cache),
    }

    export_path = args.export_root / "lfm2_layer0_conv_decode_cache.pt2"
    npz_path = export_path.with_suffix(".npz")
    started = time.perf_counter()
    exported = torch.export.export(wrapper, inputs, strict=False)
    export_seconds = time.perf_counter() - started
    counts = operator_counts(exported)
    torch.export.save(exported, export_path)
    del exported
    gc.collect()
    reloaded = torch.export.load(export_path).module()
    with torch.inference_mode():
        actual_output, actual_cache = reloaded(*inputs)
    reload_errors = {
        "output": tensor_error(actual_output, official_output),
        "conv_cache": tensor_error(actual_cache, official_cache),
    }
    np.savez(
        npz_path,
        hidden_states=current_hidden.numpy(),
        conv_cache=input_cache.numpy(),
        output=official_output.numpy(),
        updated_conv_cache=official_cache.numpy(),
    )
    max_error = max(v["max_abs"] for v in (*eager_errors.values(), *reload_errors.values()))
    return {
        "status": "passed" if max_error <= 1e-5 else "failed",
        "stage": "local_fixed_shape_cached_decode_export",
        "component": "conv",
        "model": "LiquidAI/LFM2.5-Audio-1.5B",
        "model_revision": MODEL_REVISION,
        "layer_idx": layer_idx,
        "layer_type": config.layer_types[layer_idx],
        "dtype": "float32",
        "decode_semantics": {
            "tokens_per_call": 1,
            "prefill_tokens_used_to_build_test_cache": args.past_length,
            "cache_update": "drop oldest conv state and append current projected B*x",
            "position": args.past_length,
        },
        "fixed_input_shapes": {
            "hidden_states": list(current_hidden.shape),
            "conv_cache": list(input_cache.shape),
        },
        "fixed_output_shapes": {
            "output": list(actual_output.shape),
            "updated_conv_cache": list(actual_cache.shape),
        },
        "official_eager_vs_explicit_eager": eager_errors,
        "official_eager_vs_serialized_reload": reload_errors,
        "benchmark": benchmark(wrapper, inputs, args.benchmark_runs),
        "memory": {"process_max_rss_bytes": max_rss_bytes()},
        "export": {
            "format": "torch.export pt2",
            "artifact": str(export_path),
            "artifact_bytes": export_path.stat().st_size,
            "numpy_io": str(npz_path),
            "export_seconds": export_seconds,
            "operator_counts": counts,
        },
        "selected_checkpoint_keys": selected_keys,
        "limitations": [
            "One decoder layer only; no multi-layer cache routing.",
            "One token per invocation and batch size 1 are fixed.",
            "The cache state is explicit, but the absolute position is baked into this graph.",
        ],
    }


def run_attention(args: argparse.Namespace, config: Lfm2Config) -> dict[str, Any]:
    layer_idx = int(PROBES["attention"]["layer_idx"])
    layer, selected_keys = load_layer(config, layer_idx)
    past_hidden = deterministic_hidden(args.past_length, config.hidden_size, phase=0.5)
    current_hidden = deterministic_hidden(1, config.hidden_size, phase=1.5)
    cache = Lfm2HybridConvCache(config, max_batch_size=1, dtype=torch.float32)

    past_positions = torch.arange(args.past_length, dtype=torch.long).unsqueeze(0)
    past_cos_sin = rotary_for_positions(config, past_hidden, past_positions)
    with torch.inference_mode():
        layer(
            past_hidden,
            position_embeddings=past_cos_sin,
            attention_mask=causal_mask(args.past_length),
            past_key_values=cache,
            cache_position=torch.arange(args.past_length, dtype=torch.long),
        )
    key_prefix = cache.key_cache[layer_idx].clone()
    value_prefix = cache.value_cache[layer_idx].clone()
    head_dim = layer.self_attn.head_dim
    key_cache = torch.zeros(
        1, config.num_key_value_heads, args.capacity, head_dim, dtype=torch.float32
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache[:, :, : args.past_length, :] = key_prefix
    value_cache[:, :, : args.past_length, :] = value_prefix

    current_position = torch.tensor([[args.past_length]], dtype=torch.long)
    current_cos_sin = rotary_for_positions(config, current_hidden, current_position)
    with torch.inference_mode():
        official_output = layer(
            current_hidden,
            position_embeddings=current_cos_sin,
            attention_mask=torch.zeros(
                1, 1, 1, args.past_length + 1, dtype=torch.float32
            ),
            past_key_values=cache,
            cache_position=torch.tensor([args.past_length], dtype=torch.long),
        )
    official_key = torch.cat(
        (cache.key_cache[layer_idx], key_cache[:, :, args.past_length + 1 :, :]), dim=2
    )
    official_value = torch.cat(
        (cache.value_cache[layer_idx], value_cache[:, :, args.past_length + 1 :, :]), dim=2
    )

    wrapper = ExplicitAttentionCacheDecode(
        layer,
        capacity=args.capacity,
        past_length=args.past_length,
        cos=current_cos_sin[0],
        sin=current_cos_sin[1],
    ).eval()
    inputs = (current_hidden, key_cache, value_cache)
    with torch.inference_mode():
        eager_output, eager_key, eager_value = wrapper(*inputs)
    eager_errors = {
        "output": tensor_error(eager_output, official_output),
        "key_cache": tensor_error(eager_key, official_key),
        "value_cache": tensor_error(eager_value, official_value),
    }

    export_path = args.export_root / "lfm2_layer2_attention_decode_kv16_past8.pt2"
    if args.capacity != 16 or args.past_length != 8:
        export_path = args.export_root / (
            f"lfm2_layer2_attention_decode_kv{args.capacity}_past{args.past_length}.pt2"
        )
    npz_path = export_path.with_suffix(".npz")
    started = time.perf_counter()
    exported = torch.export.export(wrapper, inputs, strict=False)
    export_seconds = time.perf_counter() - started
    counts = operator_counts(exported)
    torch.export.save(exported, export_path)
    del exported
    gc.collect()
    reloaded = torch.export.load(export_path).module()
    with torch.inference_mode():
        actual_output, actual_key, actual_value = reloaded(*inputs)
    reload_errors = {
        "output": tensor_error(actual_output, official_output),
        "key_cache": tensor_error(actual_key, official_key),
        "value_cache": tensor_error(actual_value, official_value),
    }
    np.savez(
        npz_path,
        hidden_states=current_hidden.numpy(),
        key_cache=key_cache.numpy(),
        value_cache=value_cache.numpy(),
        output=official_output.numpy(),
        updated_key_cache=official_key.numpy(),
        updated_value_cache=official_value.numpy(),
    )
    max_error = max(v["max_abs"] for v in (*eager_errors.values(), *reload_errors.values()))
    return {
        "status": "passed" if max_error <= 1e-5 else "failed",
        "stage": "local_fixed_shape_cached_decode_export",
        "component": "attention",
        "model": "LiquidAI/LFM2.5-Audio-1.5B",
        "model_revision": MODEL_REVISION,
        "layer_idx": layer_idx,
        "layer_type": config.layer_types[layer_idx],
        "dtype": "float32",
        "decode_semantics": {
            "tokens_per_call": 1,
            "cache_capacity_tokens": args.capacity,
            "valid_past_tokens": args.past_length,
            "write_slot": args.past_length,
            "valid_tokens_after_call": args.past_length + 1,
            "position": args.past_length,
            "unused_tail_is_preserved_and_attention_masked": True,
        },
        "fixed_input_shapes": {
            "hidden_states": list(current_hidden.shape),
            "key_cache": list(key_cache.shape),
            "value_cache": list(value_cache.shape),
        },
        "fixed_output_shapes": {
            "output": list(actual_output.shape),
            "updated_key_cache": list(actual_key.shape),
            "updated_value_cache": list(actual_value.shape),
        },
        "official_eager_vs_explicit_eager": eager_errors,
        "official_eager_vs_serialized_reload": reload_errors,
        "benchmark": benchmark(wrapper, inputs, args.benchmark_runs),
        "memory": {"process_max_rss_bytes": max_rss_bytes()},
        "export": {
            "format": "torch.export pt2",
            "artifact": str(export_path),
            "artifact_bytes": export_path.stat().st_size,
            "numpy_io": str(npz_path),
            "export_seconds": export_seconds,
            "operator_counts": counts,
        },
        "selected_checkpoint_keys": selected_keys,
        "limitations": [
            "One decoder layer only; no multi-layer cache routing.",
            "One token per invocation and batch size 1 are fixed.",
            "Past length, write slot, rotary position, and valid-attention mask are baked into this graph.",
            "A production persistent session needs a small family of position/length graphs or a backend-supported dynamic index/mask interface.",
            "The cache does not yet implement eviction once fixed capacity is full.",
        ],
    }


def main() -> None:
    args = parse_args()
    if args.capacity <= 1:
        raise ValueError("--capacity must be greater than one")
    if not 0 < args.past_length < args.capacity:
        raise ValueError("--past-length must be positive and less than --capacity")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be positive")
    args.export_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    config = load_config()
    requested = ("conv", "attention") if args.component == "all" else (args.component,)
    reports = []
    for component in requested:
        expected = str(PROBES[component]["expected_layer_type"])
        layer_idx = int(PROBES[component]["layer_idx"])
        if config.layer_types[layer_idx] != expected:
            raise RuntimeError(
                f"Layer {layer_idx} changed type: expected {expected}, got {config.layer_types[layer_idx]}"
            )
        report = run_conv(args, config) if component == "conv" else run_attention(args, config)
        report_path = args.report_root / f"backbone_{component}_cached_decode_probe.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        reports.append(report)
    if any(report["status"] != "passed" for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
