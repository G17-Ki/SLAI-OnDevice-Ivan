#!/usr/bin/env python3
"""Write a compact, public-safe status probe inside the Colab VM."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path("/content/zipformer-source-v058")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    worker = load_json(ROOT / "worker_status.json")
    export = load_json(ROOT / "export_status.json")
    checkpoint = load_json(ROOT / "checkpoint_meta.json")
    try:
        pid = int((ROOT / "worker.pid").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        pid = 0
    alive = process_alive(pid)
    worker_state = str(worker.get("state", "unknown"))
    if worker_state == "succeeded":
        state = "succeeded"
    elif worker_state == "failed":
        state = "failed"
    elif alive:
        state = "running"
    else:
        state = "stale"
    payload = {
        "format_version": 1,
        "state": state,
        "worker_alive": alive,
        "worker_state": worker_state,
        "export_state": export.get("state", "not_started"),
        "active_component": export.get("active_component"),
        "completed_components": export.get("completed_components", []),
        "checkpoint_generation": int(checkpoint.get("generation", 0)),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "checkpoint_size_bytes": int(checkpoint.get("size_bytes", 0)),
        "checked_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    if state == "failed":
        payload["error_type"] = worker.get("error_type", "unknown")
        payload["error"] = worker.get("error", "remote worker failed")
    atomic_write_json(ROOT / "probe.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
