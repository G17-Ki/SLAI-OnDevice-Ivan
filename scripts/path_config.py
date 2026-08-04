#!/usr/bin/env python3
"""Shared checkout-relative paths for the legacy feasibility scripts.

The repository checkout is authoritative. Large mutable state can be moved
with ``LFM_DEMO_WORK_ROOT``. ``HF_HOME`` remains the standard Hugging Face
override, and ``LFM_DEMO_MODEL_SNAPSHOT`` can pin an already-downloaded local
snapshot directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
MODEL_CACHE_NAME = "models--LiquidAI--LFM2.5-Audio-1.5B"


def _environment_path(
    name: str,
    default: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    raw = values.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = _environment_path("LFM_DEMO_WORK_ROOT", REPO_ROOT / "work")
VENDOR_ROOT = REPO_ROOT / "vendor" / "liquid-audio"
HF_HOME = _environment_path("HF_HOME", WORK_ROOT / "cache" / "huggingface")
MODEL_SNAPSHOT = _environment_path(
    "LFM_DEMO_MODEL_SNAPSHOT",
    HF_HOME / "hub" / MODEL_CACHE_NAME / "snapshots" / MODEL_REVISION,
)


def configure_hf_home() -> Path:
    """Set the default Hugging Face cache without overriding the caller."""

    os.environ.setdefault("HF_HOME", str(HF_HOME))
    return Path(os.environ["HF_HOME"]).expanduser().resolve()


def model_snapshot_for(
    revision: str = MODEL_REVISION,
    *,
    work_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a pinned snapshot, respecting standard and project overrides."""

    values = os.environ if environment is None else environment
    explicit = values.get("LFM_DEMO_MODEL_SNAPSHOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    selected_work = (work_root or WORK_ROOT).expanduser().resolve()
    hf_default = selected_work / "cache" / "huggingface"
    hf_root = _environment_path("HF_HOME", hf_default, environment=values)
    return hf_root / "hub" / MODEL_CACHE_NAME / "snapshots" / revision


def path_record() -> dict[str, str]:
    return {
        "repo_root": str(REPO_ROOT),
        "work_root": str(WORK_ROOT),
        "vendor_root": str(VENDOR_ROOT),
        "hf_home": str(HF_HOME),
        "model_snapshot": str(MODEL_SNAPSHOT),
    }


if __name__ == "__main__":
    print(json.dumps(path_record(), indent=2))
