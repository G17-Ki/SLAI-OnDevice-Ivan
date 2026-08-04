#!/usr/bin/env python3
"""Download and prepare only the pinned SenseVoice-supported CC0 fixture."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASET_REPO = "argmaxinc/common_voice_17_0-debug-zip"
DATASET_REVISION = "e77e410d41a77f733efbaf7307bd0ff7ebd6d0af"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    download_dir = work_dir / "downloads" / "common_voice_17_debug"
    dataset_dir = work_dir / "dataset"
    download_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for filename in ("debug.zip", "metadata.json"):
        paths[filename] = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=DATASET_REVISION,
                filename=filename,
                local_dir=download_dir,
            )
        ).resolve()

    shared_prepare = (
        Path(__file__).resolve().parents[1]
        / "multilingual_asr_mac_m2"
        / "prepare_common_voice_subset.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(shared_prepare),
            "--archive",
            str(paths["debug.zip"]),
            "--metadata",
            str(paths["metadata.json"]),
            "--output-dir",
            str(dataset_dir),
            "--languages",
            "en",
            "ja",
            "ko",
            "zh",
        ],
        check=True,
    )
    print(dataset_dir / "dataset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
