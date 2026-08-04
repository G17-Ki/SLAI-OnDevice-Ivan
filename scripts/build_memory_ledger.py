#!/usr/bin/env python3
"""Create a transparent static-weight and recurrent-cache memory ledger."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from path_config import MODEL_REVISION, REPO_ROOT, WORK_ROOT, model_snapshot_for


def gib(value: int) -> float:
    return value / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Legacy checkout-root override; maps mutable state to WORKSPACE/work.",
    )
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    workspace = (args.workspace or REPO_ROOT).expanduser().resolve()
    work_root = (
        workspace / "work"
        if args.workspace
        else args.work_root.expanduser().resolve()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_snapshot:
        snapshot = args.model_snapshot.expanduser().resolve()
    elif args.workspace:
        snapshot = (
            work_root
            / "cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B/snapshots"
            / MODEL_REVISION
        )
    else:
        snapshot = model_snapshot_for(work_root=work_root)
    measured_files = {
        "BF16 main checkpoint": snapshot / "model.safetensors",
        "BF16 input audio tokenizer checkpoint": snapshot / "tokenizer-e351c8d8-checkpoint125.safetensors",
        "FP32 output audio detokenizer checkpoint": snapshot / "audio_detokenizer/model.safetensors",
        "Q4 backbone/main GGUF": work_root / "models/lfm25-audio-gguf/LFM2.5-Audio-1.5B-Q4_0.gguf",
        "Q4 audio projection/encoder GGUF": work_root / "models/lfm25-audio-gguf/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
        "Q4 vocoder GGUF": work_root / "models/lfm25-audio-gguf/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
        "Q4 tokenizer GGUF": work_root / "models/lfm25-audio-gguf/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
    }
    rows = []
    for label, path in measured_files.items():
        rows.append(
            {
                "category": "measured file",
                "item": label,
                "bytes": path.stat().st_size,
                "mib": path.stat().st_size / (1024**2),
                "source": str(path),
            }
        )

    # LFM2 config: 6 attention layers, 8 KV heads, head dim 64.
    # K and V together: 6 * 8 * 64 * 2 elements per cached position.
    kv_elements_per_position = 6 * 8 * 64 * 2
    conv_elements = 10 * 2048 * 3
    for dtype, bytes_per_element in (("BF16/FP16", 2), ("INT8", 1)):
        for positions in (4096, 8192, 16384, 32768, 128000):
            value = kv_elements_per_position * positions * bytes_per_element
            rows.append(
                {
                    "category": "derived cache",
                    "item": f"LFM backbone KV cache, {positions} positions, {dtype}",
                    "bytes": value,
                    "mib": value / (1024**2),
                    "source": "Derived from pinned config: 6 attention layers, 8 KV heads, head_dim 64, K+V",
                }
            )
        conv_value = conv_elements * bytes_per_element
        rows.append(
            {
                "category": "derived cache",
                "item": f"LFM fixed convolution cache, {dtype}",
                "bytes": conv_value,
                "mib": conv_value / (1024**2),
                "source": "Derived from pinned config: 10 conv layers x hidden 2048 x cache length 3",
            }
        )

    with (args.output_dir / "memory_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bf16_static = sum(row["bytes"] for row in rows[:3])
    q4_static = sum(row["bytes"] for row in rows[3:7])
    payload = {
        "totals": {
            "bf16_downloaded_runtime_checkpoints_bytes": bf16_static,
            "bf16_downloaded_runtime_checkpoints_gib": gib(bf16_static),
            "q4_gguf_bundle_bytes": q4_static,
            "q4_gguf_bundle_gib": gib(q4_static),
        },
        "architecture": {
            "attention_layers": 6,
            "convolution_layers": 10,
            "hidden_size": 2048,
            "attention_heads": 32,
            "kv_heads": 8,
            "head_dim": 64,
            "conv_cache_length": 3,
            "kv_elements_per_position": kv_elements_per_position,
        },
        "rows": rows,
        "limitations": [
            "File sizes are static storage, not total resident application memory.",
            "Cache estimates exclude allocator alignment, QNN context buffers, activations, VAD, audio I/O, and the application.",
            "The Q4 bundle is a macOS/llama.cpp packaging reference, not a QNN artifact or AR1 fit measurement.",
        ],
    }
    (args.output_dir / "memory_ledger.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], indent=2))


if __name__ == "__main__":
    main()
