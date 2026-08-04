#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys

from path_config import HF_HOME, MODEL_SNAPSHOT, REPO_ROOT, VENDOR_ROOT, WORK_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Report clean-clone path resolution without importing ML runtimes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repo_root": str(REPO_ROOT),
        "work_root": str(WORK_ROOT),
        "vendor_root": str(VENDOR_ROOT),
        "hf_home": str(HF_HOME),
        "model_snapshot": str(MODEL_SNAPSHOT),
        "vendor_repo_present": VENDOR_ROOT.is_dir(),
        "model_snapshot_present": MODEL_SNAPSHOT.is_dir(),
    }
    if not args.paths_only:
        import torch
        import torchaudio
        import transformers
        from accelerate import __version__ as accelerate_version

        result.update(
            {
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate_version,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
