#!/usr/bin/env python3
"""Download the pinned Common Voice fixture and Faster-Whisper models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DATASET_REPO = "argmaxinc/common_voice_17_0-debug-zip"
DATASET_REVISION = "e77e410d41a77f733efbaf7307bd0ff7ebd6d0af"
MODEL_REVISIONS = {
    "tiny": (
        "Systran/faster-whisper-tiny",
        "d90ca5fe260221311c53c58e660288d3deb8d356",
    ),
    "base": (
        "Systran/faster-whisper-base",
        "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    ),
    "small": (
        "Systran/faster-whisper-small",
        "536b0662742c02347bc0e980a01041f333bce120",
    ),
}
MODEL_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    dataset_dir = work_dir / "downloads" / "common_voice_17_debug"
    model_root = work_dir / "models"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    dataset_paths = []
    for filename in ("debug.zip", "metadata.json", "README.md"):
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=filename,
            local_dir=dataset_dir,
        )
        dataset_paths.append(Path(downloaded).resolve())

    model_records: dict[str, object] = {}
    for short_name, (repo_id, revision) in MODEL_REVISIONS.items():
        destination = model_root / f"faster-whisper-{short_name}"
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=MODEL_FILES,
            local_dir=destination,
        )
        paths = [destination / filename for filename in MODEL_FILES]
        model_records[short_name] = {
            "repo_id": repo_id,
            "revision": revision,
            "files": [file_record(path, work_dir) for path in paths],
        }

    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": DATASET_REVISION,
            "license": "CC0-1.0",
            "files": [file_record(path, work_dir) for path in dataset_paths],
        },
        "models": model_records,
    }
    output = work_dir / "asset_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
