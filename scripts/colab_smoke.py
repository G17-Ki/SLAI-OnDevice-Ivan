#!/usr/bin/env python3
"""Small Colab CUDA smoke test; writes one compact JSON result."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


payload: dict[str, object] = {
    "python": sys.version,
    "disk_content": shutil.disk_usage("/content")._asdict(),
}

try:
    process = subprocess.run(
        ["nvidia-smi"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload["nvidia_smi_returncode"] = process.returncode
    payload["nvidia_smi_head"] = process.stdout[:2000]
except Exception as exc:
    payload["nvidia_smi_error"] = repr(exc)

try:
    import torch

    payload["torch_version"] = torch.__version__
    payload["cuda_available"] = torch.cuda.is_available()
    payload["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    payload["torch_error"] = repr(exc)

result = Path("/content/colab_smoke_result.json")
result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
