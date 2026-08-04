#!/usr/bin/env python3
"""Fail-closed Colab host preflight for the QAIRT 2.42 Qwen lane."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil


MINIMUM_BYTES = 24_000_000_000
OUTPUT = Path("/content/qairt242_qwen25_preflight.json")


def memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal is unavailable")


def os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            result[key] = value.strip().strip('"')
    return result


release = os_release()
payload = {
    "schema_version": 1,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "ubuntu_version": release.get("VERSION_ID"),
    "memory_bytes": memory_bytes(),
    "content_free_bytes": shutil.disk_usage("/content").free,
    "cpu_count": os.cpu_count(),
}
payload["status"] = "pass" if (
    payload["machine"].lower() in {"x86_64", "amd64"}
    and payload["ubuntu_version"] == "22.04"
    and payload["memory_bytes"] >= MINIMUM_BYTES
    and payload["content_free_bytes"] >= MINIMUM_BYTES
) else "failed_closed"
OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pass":
    raise SystemExit(2)
