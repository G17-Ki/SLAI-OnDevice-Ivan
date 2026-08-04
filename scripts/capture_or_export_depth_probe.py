#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import psutil
import numpy as np

from path_config import (
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    REPO_ROOT,
    WORK_ROOT,
    configure_hf_home,
)

configure_hf_home()

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from torch import nn  # noqa: E402

from liquid_audio.model.transformer import (  # noqa: E402
    MHA,
    RawLMBackbone,
    SharedEmbedding,
    StandardBlock,
)
from liquid_audio.model.lfm2_audio import LFM2AudioModel  # noqa: E402


@dataclass(frozen=True)
class DepthProbeConfig:
    hidden_size: int
    codebooks: int
    audio_vocab_size: int
    depthformer_layers: int
    depthformer_dim: int
    depthformer_tie: bool
    tie_audio_embeddings: bool


class FixedShapeDepthDecoderProbe(nn.Module):
    """Greedily decode one fixed [1, hidden_size] LFM audio frame.

    The eight codec steps are intentionally unrolled. Each step attends to the
    previously decoded codebooks through the real Depthformer KV cache. The
    returned audio embedding is the exact summed, offset embedding that the
    upstream model feeds into the LFM backbone at the next audio frame.
    """

    def __init__(self, config: DepthProbeConfig) -> None:
        super().__init__()
        self.config = config

        scale = 1 / (2 * config.depthformer_layers) ** 0.5
        layers = [
            StandardBlock(
                MHA(config.depthformer_dim, out_init_scale=scale),
                out_init_scale=scale,
            )
            for _ in range(config.depthformer_layers)
        ]
        self.depthformer = RawLMBackbone(layers, has_embedding=False)
        self.depth_linear = nn.Linear(
            config.hidden_size,
            config.depthformer_dim * config.codebooks,
        )
        self.depth_embeddings = nn.ModuleList(
            [
                SharedEmbedding(
                    dim=config.depthformer_dim,
                    vocab_size=config.audio_vocab_size,
                    tie_embedding=config.depthformer_tie,
                )
                for _ in range(config.codebooks)
            ]
        )
        self.audio_embedding = SharedEmbedding(
            dim=config.hidden_size,
            vocab_size=config.audio_vocab_size * config.codebooks,
            embed_init_scale=1.0,
            norm_eps=0.00001,
            tie_embedding=config.tie_audio_embeddings,
        )
        self.register_buffer(
            "codebook_offsets",
            torch.arange(config.codebooks, dtype=torch.long) * config.audio_vocab_size,
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden is fixed to [1, 2048] for this feasibility probe.
        depth_inputs = self.depth_linear(hidden).view(
            1,
            self.config.codebooks,
            self.config.depthformer_dim,
        )
        previous_token_embedding = torch.zeros_like(depth_inputs[:, 0, :])
        cache = None
        token_list: list[torch.Tensor] = []

        # Static unroll: Qualcomm sees eight concrete greedy decode steps rather
        # than a data-dependent loop.
        for codebook_index in range(self.config.codebooks):
            step_input = depth_inputs[:, codebook_index, :] + previous_token_embedding
            step_output, cache = self.depthformer.forward_cached(
                step_input.unsqueeze(1),
                cache,
            )
            logits = self.depth_embeddings[codebook_index].get_logits(step_output[:, 0, :])
            token = logits.argmax(dim=-1)
            token_list.append(token)
            previous_token_embedding = self.depth_embeddings[codebook_index](token)

        tokens = torch.stack(token_list, dim=1)
        offset_tokens = tokens + self.codebook_offsets.unsqueeze(0)
        next_audio_embedding = self.audio_embedding(offset_tokens).sum(dim=1)
        return tokens, next_audio_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/artifacts/depth-probe-golden",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/exports/depth_decoder_hidden1x2048.pt2",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports/depth_export.json",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Capture the eager golden only.",
    )
    return parser.parse_args()


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def load_probe_config() -> DepthProbeConfig:
    raw = json.loads((MODEL_SNAPSHOT / "config.json").read_text())
    return DepthProbeConfig(
        hidden_size=int(raw["lfm"]["hidden_size"]),
        codebooks=int(raw["codebooks"]),
        audio_vocab_size=2049,
        depthformer_layers=int(raw["depthformer"]["layers"]),
        depthformer_dim=int(raw["depthformer"]["dim"]),
        depthformer_tie=bool(raw["depthformer"]["tie"]),
        tie_audio_embeddings=bool(raw["tie_audio_embeddings"]),
    )


def load_real_weight_probe(*, dtype: torch.dtype = torch.float32) -> FixedShapeDepthDecoderProbe:
    if not MODEL_SNAPSHOT.is_dir():
        raise FileNotFoundError(f"Official model snapshot not found: {MODEL_SNAPSHOT}")

    probe = FixedShapeDepthDecoderProbe(load_probe_config())
    prefixes = (
        "depth_linear.",
        "depthformer.",
        "depth_embeddings.",
        "audio_embedding.",
    )
    selected: dict[str, torch.Tensor] = {}
    checkpoint = MODEL_SNAPSHOT / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key.startswith(prefixes):
                selected[key] = reader.get_tensor(key).to(dtype=dtype)

    missing, unexpected = probe.load_state_dict(selected, strict=False)
    allowed_missing = {"codebook_offsets"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            "Selective depth state load mismatch: "
            f"missing={missing}, unexpected={unexpected}, loaded={len(selected)}"
        )
    return probe.eval().to(dtype=dtype)


def deterministic_hidden(hidden_size: int) -> torch.Tensor:
    """Stable synthetic LFM hidden state, independent of random generators."""
    index = torch.arange(hidden_size, dtype=torch.float32)
    hidden = torch.sin(index / 17.0) + 0.5 * torch.cos(index / 31.0)
    return hidden.unsqueeze(0).contiguous()


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def run_official_greedy_sampler(
    probe: FixedShapeDepthDecoderProbe,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Call the upstream sampler implementation against the selective modules."""
    official_view = SimpleNamespace(
        codebooks=probe.config.codebooks,
        depthformer_dim=probe.config.depthformer_dim,
        depth_linear=probe.depth_linear,
        depthformer=probe.depthformer,
        depth_embeddings=probe.depth_embeddings,
    )
    return LFM2AudioModel._sample_audio_frame(
        official_view,
        hidden.squeeze(0),
        temperature=None,
        top_k=None,
    )


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.export.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    config = load_probe_config()
    hidden = deterministic_hidden(config.hidden_size)

    rss_before_load = rss_bytes()
    load_started = time.perf_counter()
    probe = load_real_weight_probe(dtype=torch.float32)
    load_seconds = time.perf_counter() - load_started
    rss_after_load = rss_bytes()

    inference_started = time.perf_counter()
    with torch.inference_mode():
        golden_tokens, golden_audio_embedding = probe(hidden)
        official_tokens = run_official_greedy_sampler(probe, hidden).unsqueeze(0)
    inference_seconds = time.perf_counter() - inference_started
    rss_after_inference = rss_bytes()

    if not torch.equal(golden_tokens, official_tokens):
        raise RuntimeError(
            "Probe unroll does not match official _sample_audio_frame: "
            f"probe={golden_tokens.tolist()}, official={official_tokens.tolist()}"
        )

    torch.save({"hidden": hidden}, args.artifact_dir / "inputs.pt")
    torch.save(
        {
            "tokens": golden_tokens,
            "next_audio_embedding": golden_audio_embedding,
        },
        args.artifact_dir / "outputs.pt",
    )
    numpy_io = args.export.with_suffix(".npz")
    numpy_io.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        numpy_io,
        hidden=hidden.numpy(),
        tokens=golden_tokens.numpy(),
        next_audio_embedding=golden_audio_embedding.numpy(),
    )

    report: dict[str, object] = {
        "component": "LFM2.5-Audio real-weight RQ/depth decoder",
        "model_revision": MODEL_REVISION,
        "checkpoint": str(MODEL_SNAPSHOT / "model.safetensors"),
        "dtype": "float32",
        "fixed_input_shapes": {"hidden": list(hidden.shape)},
        "output_shapes": {
            "tokens": list(golden_tokens.shape),
            "next_audio_embedding": list(golden_audio_embedding.shape),
        },
        "golden": {
            "artifact_dir": str(args.artifact_dir),
            "numpy_io": str(numpy_io),
            "tokens": golden_tokens.tolist(),
            "official_sample_audio_frame_tokens_equal": True,
            "next_audio_embedding_statistics": {
                "min": float(golden_audio_embedding.min()),
                "max": float(golden_audio_embedding.max()),
                "mean": float(golden_audio_embedding.mean()),
                "std": float(golden_audio_embedding.std()),
            },
        },
        "timings": {
            "component_load_seconds": load_seconds,
            "eager_inference_seconds": inference_seconds,
        },
        "memory": {
            "rss_before_component_load_bytes": rss_before_load,
            "rss_after_component_load_bytes": rss_after_load,
            "rss_after_eager_inference_bytes": rss_after_inference,
        },
        "selective_weight_prefixes": [
            "depth_linear.",
            "depthformer.",
            "depth_embeddings.",
            "audio_embedding.",
        ],
        "export": {"status": "skipped" if args.skip_export else "not_started"},
        "status": "golden_captured",
    }

    if not args.skip_export:
        export_started = time.perf_counter()
        try:
            exported = torch.export.export(probe, (hidden,), strict=False)
            export_seconds = time.perf_counter() - export_started
            torch.export.save(exported, args.export)

            exported_module = exported.module()
            with torch.inference_mode():
                exported_tokens, exported_audio_embedding = exported_module(hidden)

            report["export"] = {
                "status": "passed",
                "format": "torch.export pt2",
                "artifact": str(args.export),
                "artifact_bytes": args.export.stat().st_size,
                "export_seconds": export_seconds,
                "exported_vs_eager": {
                    "tokens_equal": bool(torch.equal(exported_tokens, golden_tokens)),
                    "next_audio_embedding": tensor_error(
                        exported_audio_embedding,
                        golden_audio_embedding,
                    ),
                },
            }
            report["status"] = "passed"
        except Exception as error:  # Preserve the exact exporter blocker for triage.
            report["export"] = {
                "status": "failed",
                "export_seconds_before_failure": time.perf_counter() - export_started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            report["status"] = "export_failed"

    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if report["status"] == "export_failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
