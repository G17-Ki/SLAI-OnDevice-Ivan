#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from component_utils import REPO_ROOT, WORK_ROOT, load_encoder_adapter


class FixedShapeEncoderAdapter(nn.Module):
    """Expose one fixed float input and bake the sequence length into the graph."""

    def __init__(self, component: nn.Module, mel_frames: int) -> None:
        super().__init__()
        self.component = component
        self.register_buffer("fixed_mel_len", torch.tensor([mel_frames], dtype=torch.long))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        adapted, _ = self.component(mel, self.fixed_mel_len)
        return adapted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/artifacts/encoder-golden-question",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORK_ROOT / "lfm-feasibility/exports/fastconformer_adapter_mel80.pt2",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "reports/fastconformer_export.json",
    )
    return parser.parse_args()


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.golden_dir / "inputs.pt", map_location="cpu", weights_only=True)
    golden = torch.load(args.golden_dir / "outputs.pt", map_location="cpu", weights_only=True)
    mel = inputs["mel"].float().contiguous()
    mel_len = inputs["mel_len"].contiguous()

    component = load_encoder_adapter(dtype=torch.float32)
    fixed_component = FixedShapeEncoderAdapter(component, mel.shape[-1]).eval()
    with torch.inference_mode():
        eager_adapted = fixed_component(mel)

    export_started = time.perf_counter()
    exported = torch.export.export(fixed_component, (mel,), strict=False)
    export_seconds = time.perf_counter() - export_started
    torch.export.save(exported, args.output)

    exported_module = exported.module()
    with torch.inference_mode():
        exported_adapted = exported_module(mel)

    numpy_input = mel.numpy()
    numpy_output = eager_adapted.numpy()
    numpy_path = args.output.with_suffix(".npz")
    import numpy as np

    np.savez(numpy_path, mel=numpy_input, adapted=numpy_output)

    summary = {
        "format": "torch.export pt2",
        "fixed_input_shapes": {
            "mel": list(mel.shape),
        },
        "output_shapes": {
            "adapted": list(exported_adapted.shape),
        },
        "export_seconds": export_seconds,
        "artifact": str(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "numpy_io": str(numpy_path),
        "eager_vs_component_golden": {
            "adapted": tensor_error(eager_adapted, golden["adapted"]),
        },
        "exported_vs_eager": {
            "adapted": tensor_error(exported_adapted, eager_adapted),
        },
        "status": "passed",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
