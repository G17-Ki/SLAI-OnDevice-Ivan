#!/usr/bin/env python3
"""Capture and export the real-valued neural portion of the LFM audio detokenizer.

The official detokenizer ends with ``polar(exp(log_abs), angle)`` and a custom
ISTFT.  Complex tensors and FFT are poor first targets for mobile graph
conversion, so this probe stops at the exact boundary immediately before those
operations.  It uses the official configuration and weights for embedding,
LFM2 backbone, and output projection.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
import traceback
from pathlib import Path
from typing import Any

from path_config import (
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    REPO_ROOT,
    WORK_ROOT,
    configure_hf_home,
)

DETOK_DIR = MODEL_SNAPSHOT / "audio_detokenizer"
DEFAULT_ARTIFACT_ROOT = WORK_ROOT / "lfm-feasibility/artifacts"
DEFAULT_EXPORT_ROOT = WORK_ROOT / "lfm-feasibility/exports"
DEFAULT_REPORT = REPO_ROOT / "reports/detok_probe.json"

configure_hf_home()

import psutil  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from torch import nn  # noqa: E402
from transformers import Lfm2Config, Lfm2Model  # noqa: E402

from liquid_audio.detokenizer import FusedEmbedding  # noqa: E402


class DetokenizerNeuralProbe(nn.Module):
    """Official detokenizer up to, but excluding, polar reconstruction/ISTFT."""

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.emb = FusedEmbedding(hidden_size, codeboooks=8, vocab_size=2048)
        self.lfm = Lfm2Model(config)
        self.lin = nn.Linear(hidden_size, int(getattr(config, "output_size", 1282)))
        self.sliding_window_size = int(getattr(config, "sliding_window", 30))

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(codes)
        upsample_size = 6 * x.shape[1]
        x = nn.functional.interpolate(x.mT, upsample_size, mode="nearest-exact").mT

        idx = torch.arange(x.shape[1], device=x.device)
        d_idx = idx - idx[:, None]
        mask = torch.logical_and(d_idx <= 0, d_idx > -self.sliding_window_size)[None, None, ...]

        x = self.lfm(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        x = self.lin(x)
        log_abs, angle = torch.chunk(x.mT.contiguous(), 2, dim=1)
        return log_abs, angle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--action", choices=("all", "capture", "export"), default="all")
    parser.add_argument("--benchmark-iterations", type=int, default=5)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if platform.system() == "Darwin" else value * 1024


def fixed_codes(length: int) -> torch.Tensor:
    """Deterministic, valid values covering all eight codebooks."""
    values = torch.arange(8 * length, dtype=torch.long).reshape(1, 8, length)
    return (values * 73 + 19) % 2048


def compatible_config() -> tuple[Lfm2Config, dict[str, Any]]:
    config_path = DETOK_DIR / "config.json"
    raw = json.loads(config_path.read_text())
    original_layer_types = list(raw["layer_types"])
    # Transformers represents local/sliding attention with the full-attention
    # module and applies the supplied sliding mask.  This is the same conversion
    # used by the official Liquid Audio processor, without invoking its .cuda().
    raw["layer_types"] = ["full_attention" if x == "sliding_attention" else x for x in original_layer_types]
    return Lfm2Config.from_dict(raw), {
        "config_path": str(config_path),
        "original_layer_types": original_layer_types,
        "transformers_layer_types": raw["layer_types"],
    }


def load_probe() -> tuple[DetokenizerNeuralProbe, dict[str, Any]]:
    config, config_info = compatible_config()
    probe = DetokenizerNeuralProbe(config).eval().to(dtype=torch.float32)

    checkpoint = DETOK_DIR / "model.safetensors"
    selected: dict[str, torch.Tensor] = {}
    selected_prefixes = ("emb.", "lfm.", "lin.")
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        all_keys = list(reader.keys())
        for key in all_keys:
            if key.startswith(selected_prefixes):
                selected[key] = reader.get_tensor(key).to(dtype=torch.float32)

    missing, unexpected = probe.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Selective state load mismatch: missing={missing}, unexpected={unexpected}")

    load_info = {
        **config_info,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_tensor_count": len(all_keys),
        "selected_tensor_count": len(selected),
        "excluded_tensors": sorted(set(all_keys) - set(selected)),
        "compute_dtype": "float32",
        "weight_source_dtype": "bfloat16",
        "device": "cpu",
        "parameter_count": sum(parameter.numel() for parameter in probe.parameters()),
    }
    return probe, load_info


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_fraction": float(finite.float().mean()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


def benchmark(probe: nn.Module, codes: torch.Tensor, iterations: int) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    samples: list[float] = []
    with torch.inference_mode():
        started = time.perf_counter()
        outputs = probe(codes)
        first_seconds = time.perf_counter() - started
        for _ in range(iterations):
            started = time.perf_counter()
            outputs = probe(codes)
            samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    return outputs, {
        "first_inference_seconds": first_seconds,
        "iterations": iterations,
        "samples_seconds": samples,
        "mean_seconds": sum(samples) / len(samples),
        "median_seconds": ordered[len(ordered) // 2],
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def capture_one(
    probe: DetokenizerNeuralProbe,
    length: int,
    artifact_root: Path,
    benchmark_iterations: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    codes = fixed_codes(length)
    memory_before = rss_bytes()
    peak_before = peak_rss_bytes()
    (log_abs, angle), timings = benchmark(probe, codes, benchmark_iterations)
    memory_after = rss_bytes()
    peak_after = peak_rss_bytes()

    artifact_dir = artifact_root / f"detok-neural-probe-t{length}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"codes": codes}, artifact_dir / "inputs.pt")
    torch.save({"log_abs": log_abs, "angle": angle}, artifact_dir / "outputs.pt")
    summary = {
        "length": length,
        "artifact_dir": str(artifact_dir),
        "input": {"shape": list(codes.shape), "dtype": str(codes.dtype), "min": int(codes.min()), "max": int(codes.max())},
        "outputs": {"log_abs": tensor_stats(log_abs), "angle": tensor_stats(angle)},
        "expected_output_shape": [1, 641, 6 * length],
        "timings": timings,
        "memory": {
            "rss_before_inference_bytes": memory_before,
            "rss_after_inference_bytes": memory_after,
            "rss_delta_bytes": memory_after - memory_before,
            "peak_rss_before_inference_bytes": peak_before,
            "peak_rss_after_inference_bytes": peak_after,
            "peak_rss_delta_bytes": peak_after - peak_before,
        },
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return codes, (log_abs, angle), summary


def export_one(
    probe: DetokenizerNeuralProbe,
    codes: torch.Tensor,
    golden: tuple[torch.Tensor, torch.Tensor],
    length: int,
    export_root: Path,
) -> dict[str, Any]:
    export_root.mkdir(parents=True, exist_ok=True)
    output_path = export_root / f"detok_neural_probe_codes_t{length}.pt2"
    started = time.perf_counter()
    memory_before = rss_bytes()
    peak_before = peak_rss_bytes()
    try:
        exported = torch.export.export(probe, (codes,), strict=True)
        export_seconds = time.perf_counter() - started
        memory_after_export = rss_bytes()
        peak_after_export = peak_rss_bytes()

        with torch.inference_mode():
            actual = exported.module()(codes)
        operator_counts: dict[str, int] = {}
        for node in exported.graph_module.graph.nodes:
            if node.op == "call_function":
                operator = str(node.target)
                operator_counts[operator] = operator_counts.get(operator, 0) + 1
        errors = []
        for expected_tensor, actual_tensor in zip(golden, actual, strict=True):
            errors.append(
                {
                    "max_abs_error": float((expected_tensor - actual_tensor).abs().max()),
                    "mean_abs_error": float((expected_tensor - actual_tensor).abs().mean()),
                }
            )

        torch.export.save(exported, output_path)
        numpy_io = output_path.with_suffix(".npz")
        np.savez(
            numpy_io,
            codes=codes.numpy(),
            log_abs=golden[0].numpy(),
            angle=golden[1].numpy(),
        )
        reloaded = torch.export.load(output_path)
        with torch.inference_mode():
            reloaded_actual = reloaded.module()(codes)
        reload_errors = []
        for expected_tensor, actual_tensor in zip(golden, reloaded_actual, strict=True):
            reload_errors.append(
                {
                    "max_abs_error": float((expected_tensor - actual_tensor).abs().max()),
                    "mean_abs_error": float((expected_tensor - actual_tensor).abs().mean()),
                }
            )
        return {
            "status": "success",
            "pt2_path": str(output_path),
            "pt2_bytes": output_path.stat().st_size,
            "numpy_io": str(numpy_io),
            "export_seconds": export_seconds,
            "rss_before_export_bytes": memory_before,
            "rss_after_export_bytes": memory_after_export,
            "rss_delta_bytes": memory_after_export - memory_before,
            "peak_rss_before_export_bytes": peak_before,
            "peak_rss_after_export_bytes": peak_after_export,
            "peak_rss_delta_bytes": peak_after_export - peak_before,
            "eager_vs_exported": {"log_abs": errors[0], "angle": errors[1]},
            "eager_vs_serialized_reload": {
                "log_abs": reload_errors[0],
                "angle": reload_errors[1],
            },
            "graph": {
                "node_count": sum(1 for _ in exported.graph_module.graph.nodes),
                "operator_counts": dict(sorted(operator_counts.items())),
                "contains_complex_or_fft": any(
                    "polar" in operator or "fft" in operator for operator in operator_counts
                ),
            },
        }
    except Exception as exc:
        return {
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
            "rss_before_export_bytes": memory_before,
            "rss_after_failure_bytes": rss_bytes(),
            "peak_rss_before_export_bytes": peak_before,
            "peak_rss_after_failure_bytes": peak_rss_bytes(),
        }


def main() -> None:
    args = parse_args()
    if any(length <= 0 for length in args.lengths):
        raise ValueError("All code lengths must be positive")
    if args.benchmark_iterations <= 0:
        raise ValueError("--benchmark-iterations must be positive")

    torch.manual_seed(0)
    process_memory_before_load = rss_bytes()
    process_peak_before_load = peak_rss_bytes()
    load_started = time.perf_counter()
    probe, load_info = load_probe()
    load_seconds = time.perf_counter() - load_started
    process_memory_after_load = rss_bytes()
    process_peak_after_load = peak_rss_bytes()

    report: dict[str, Any] = {
        "probe": "official audio detokenizer neural graph before polar/ISTFT",
        "model": "LiquidAI/LFM2.5-Audio-1.5B",
        "model_revision": MODEL_REVISION,
        "boundary": {
            "included": ["8-codebook fused embedding", "6x nearest upsample", "8-layer LFM2", "1282-bin linear projection", "split into log_abs and angle"],
            "excluded": ["torch.polar", "complex spectrogram", "inverse real FFT", "overlap-add ISTFT"],
            "reason": "Keep the first conversion probe entirely real-valued and isolate complex/FFT deployment risk.",
        },
        "blockers_and_limits": [
            {
                "scope": "local neural PT2 export",
                "status": "none after real-valued split",
            },
            {
                "scope": "full waveform detokenizer graph",
                "status": "not covered by this probe",
                "operations": ["torch.polar", "torch.fft.irfft", "torch.nn.functional.fold"],
                "next_action": "Keep complex reconstruction/ISTFT on host or replace it with a separately deployable real-valued synthesis implementation.",
            },
            {
                "scope": "Qualcomm operator placement, latency, and memory",
                "status": "not known from local PT2 export",
                "next_action": "Submit the fixed-shape PT2 artifact through the AI Hub component runner when credentials and an exact target device are available.",
            },
        ],
        "environment": {
            "python_platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "load": {
            **load_info,
            "seconds": load_seconds,
            "rss_before_bytes": process_memory_before_load,
            "rss_after_bytes": process_memory_after_load,
            "rss_delta_bytes": process_memory_after_load - process_memory_before_load,
            "peak_rss_before_bytes": process_peak_before_load,
            "peak_rss_after_bytes": process_peak_after_load,
            "peak_rss_delta_bytes": process_peak_after_load - process_peak_before_load,
        },
        "cases": {},
    }

    captured: dict[int, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = {}
    # Capture every eager case before exporting any graph. This keeps an earlier
    # export's serializer/caches from contaminating later inference RSS samples.
    for length in args.lengths:
        codes, golden, capture_summary = capture_one(
            probe, length, args.artifact_root, args.benchmark_iterations
        )
        captured[length] = (codes, golden)
        report["cases"][f"t{length}"] = {"capture": capture_summary}

    for length in args.lengths:
        codes, golden = captured[length]
        case = report["cases"][f"t{length}"]
        if args.action in ("all", "export"):
            case["export"] = export_one(probe, codes, golden, length, args.export_root)
        else:
            case["export"] = {"status": "not_requested"}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
