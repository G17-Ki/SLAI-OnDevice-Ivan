#!/usr/bin/env python3
"""Launch the candidate-quality runner in a clean remote process."""

from __future__ import annotations

import subprocess
import sys


raise SystemExit(
    subprocess.run(
        [sys.executable, "/content/colab_candidate_quality_runner.py"], check=False
    ).returncode
)
