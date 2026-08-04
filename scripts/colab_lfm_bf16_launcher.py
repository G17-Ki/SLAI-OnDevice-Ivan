#!/usr/bin/env python3
"""Launch the uploaded BF16 runner in a fresh Python process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


runner = Path("/content/colab_lfm_bf16_runner.py")
if not runner.is_file():
    raise FileNotFoundError(runner)

completed = subprocess.run([sys.executable, str(runner)], check=False)
raise SystemExit(completed.returncode)
