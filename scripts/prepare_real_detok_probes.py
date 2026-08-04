#!/usr/bin/env python3
"""Build detokenizer fixed-shape goldens from real LFM-generated audio codes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--model-t4", type=Path, required=True)
    parser.add_argument("--model-t8", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.tokens) as archive:
        all_codes = np.asarray(archive["turn1_audio_tokens"], dtype=np.int64)
    valid = all_codes[:, ~(all_codes >= 2048).any(axis=0)]
    if valid.shape[1] < 8:
        raise ValueError(f"Need at least 8 valid frames, found {valid.shape[1]}")

    manifest = {"source": str(args.tokens), "valid_frames": int(valid.shape[1]), "probes": {}}
    for frames, model_path in ((4, args.model_t4), (8, args.model_t8)):
        codes = valid[:, :frames][None, ...]
        module = torch.export.load(model_path).module()
        with torch.inference_mode():
            log_abs, angle = module(torch.from_numpy(codes))
        output_path = args.output_dir / f"detok_real_turn1_t{frames}.npz"
        np.savez(
            output_path,
            codes=codes,
            log_abs=log_abs.detach().cpu().numpy(),
            angle=angle.detach().cpu().numpy(),
        )
        manifest["probes"][f"t{frames}"] = {
            "path": str(output_path),
            "codes_shape": list(codes.shape),
            "code_min": int(codes.min()),
            "code_max": int(codes.max()),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
