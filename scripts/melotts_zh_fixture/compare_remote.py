#!/usr/bin/env python3
"""Compare later remote component outputs against a local MeloTTS fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fixture_common import stitch_decoder_audio


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    delta = candidate - reference
    denominator = max(float(np.sqrt(np.mean(reference * reference))), 1e-12)
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    cosine_denominator = max(float(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat)), 1e-12)
    return {
        "finite": bool(np.isfinite(candidate).all()),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "nrmse": float(np.sqrt(np.mean(delta * delta)) / denominator),
        "cosine": float(np.dot(ref_flat, cand_flat) / cosine_denominator),
    }


def load_single(path: Path, key: str) -> np.ndarray:
    with np.load(path) as values:
        if key not in values:
            raise KeyError(f"{path} has no {key!r}; found {values.files}")
        return values[key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--remote-bert", type=Path)
    parser.add_argument("--remote-encoder", type=Path)
    parser.add_argument("--remote-flow", type=Path)
    parser.add_argument("--remote-decoder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture_dir.expanduser().resolve()
    manifest = json.loads((fixture / "manifest.json").read_text())
    result: dict[str, object] = {"schema_version": 1, "comparisons": {}}
    comparisons: dict[str, object] = result["comparisons"]  # type: ignore[assignment]
    if args.remote_bert:
        comparisons["bert.hidden_states"] = metrics(
            load_single(fixture / "bert_local_outputs.npz", "hidden_states"),
            load_single(args.remote_bert, "hidden_states"),
        )
    if args.remote_encoder:
        with np.load(fixture / "encoder_local_outputs.npz") as local, np.load(args.remote_encoder) as remote:
            comparisons["encoder"] = {name: metrics(local[name], remote[name]) for name in local.files}
    if args.remote_flow:
        comparisons["flow.z"] = metrics(
            load_single(fixture / "flow_local_outputs.npz", "z"),
            load_single(args.remote_flow, "z"),
        )
    if args.remote_decoder:
        with np.load(args.remote_decoder) as remote:
            chunk_keys = sorted(name for name in remote.files if name.startswith("audio_"))
            remote_audio = stitch_decoder_audio(
                [remote[name] for name in chunk_keys], int(manifest["y_length_frames"])
            )
        local_audio = load_single(fixture / "decoder_local_outputs.npz", "stitched_audio")
        comparisons["decoder.stitched_audio"] = metrics(local_audio, remote_audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
