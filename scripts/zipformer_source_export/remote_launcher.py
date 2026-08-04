#!/usr/bin/env python3
"""Start the Colab worker in a detached process and return immediately."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path("/content/zipformer-source-v058")
WORKER = Path("/content/remote_worker.py")
RESUME = Path("/content/zipformer-source-resume.tar")


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    pid_path = ROOT / "worker.pid"
    try:
        old_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        old_pid = 0
    if process_alive(old_pid):
        atomic_write_json(
            ROOT / "launcher_status.json",
            {"state": "already_running", "pid": old_pid},
        )
        return 0
    if not WORKER.is_file():
        atomic_write_json(
            ROOT / "launcher_status.json",
            {"state": "failed", "error": "uploaded worker is missing"},
        )
        return 2

    command = [
        sys.executable,
        str(WORKER),
        "--root",
        str(ROOT),
        "--components",
        "decoder,joiner,encoder",
        "--threads",
        "4",
    ]
    if RESUME.is_file():
        command.extend(["--resume", str(RESUME)])
    log_path = ROOT / "worker.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd="/content",
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    atomic_write_json(
        ROOT / "launcher_status.json",
        {"state": "started", "pid": process.pid, "resume_present": RESUME.is_file()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
