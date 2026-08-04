#!/usr/bin/env python3
"""Download and verify the pinned English/Mandarin Zipformer model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_REPO = "pfluo/k2fsa-zipformer-chinese-english-mixed"
MODEL_REVISION = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"
MODEL_FILES = {
    "exp/encoder-epoch-99-avg-1.int8.onnx": (
        "e92df89d84ae5cb1571a2489c85d5d385c31fb8022892a240bd8c4c301b3ecc8"
    ),
    "exp/decoder-epoch-99-avg-1.onnx": (
        "d91b63c84f27f1d822244ced4cc2b9f7fb8a60a328f13b860408b8889dd0a0ed"
    ),
    "exp/joiner-epoch-99-avg-1.int8.onnx": (
        "16e901969495d5704a27a929e396a2ff9cc6fbfdeafdc6418e8c88b370113af6"
    ),
    "data/lang_char_bpe/tokens.txt": (
        "a8e0e4ec53810e433789b54a5c0134a7eaa2ffca595a6334d54c00da858841d3"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the exact Zipformer revision used by the benchmark."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for filename, expected_sha256 in MODEL_FILES.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                revision=MODEL_REVISION,
                filename=filename,
                local_dir=output_dir,
                force_download=args.force_download,
            )
        ).resolve()
        actual_sha256 = sha256(downloaded)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        records.append(
            {
                "path": filename,
                "bytes": downloaded.stat().st_size,
                "sha256": actual_sha256,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "files": records,
    }
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
