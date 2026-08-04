#!/usr/bin/env python3
"""Install the pinned Colab host prerequisites and run the QAIRT controller."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


CONTROL = Path("/content/qairt242_qwen25_control")
WORKSPACE = Path("/content/qairt242_qwen25_workspace")
SCRATCH = Path("/content/qairt242_qwen25_scratch")
STATUS = Path("/content/qairt242_qwen25_remote_status.json")
EXPECTED = {
    "qairt242_colab_qwen25_genie.py":
        "f5f602276289ec289d9b90595273739f82448e7fa1b45e6b259dc89963c8c5bd",
    "qairt242_qwen25_genie_spec.json":
        "8b25f22513a78aa9cea7898720ead903e41a77ef4475506450e17fa816c4565a",
    "qairt242_qwen25_15b_genie_source_lock.json":
        "cd8af4ced04ff562fbf9b2916a2d2065dfa092c3b4f8e78693e952eb1aa65e5e",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(status: str, stage: str, **extra: object) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "stage": stage,
        "updated_utc": now(),
        **extra,
    }
    temporary = STATUS.with_suffix(".json.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("COMMAND_JSON=" + json.dumps(command), flush=True)
    subprocess.run(command, check=True, env=env)


def verify_controls() -> None:
    for name, expected in EXPECTED.items():
        path = CONTROL / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"control identity mismatch: {name}")


def main() -> int:
    write_status("running", "verify_control")
    verify_controls()
    release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if release.get("VERSION_ID") != "22.04":
        raise RuntimeError(f"expected Ubuntu 22.04, got {release.get('VERSION_ID')}")
    if shutil.disk_usage("/content").free < 24_000_000_000:
        raise RuntimeError("less than 24 GB free under /content")

    write_status("running", "install_host_prerequisites")
    apt_env = dict(os.environ)
    apt_env["DEBIAN_FRONTEND"] = "noninteractive"
    run(["apt-get", "update", "-qq"], env=apt_env)
    run([
        "apt-get", "install", "-y", "--no-install-recommends",
        "python3.10", "python3.10-venv", "python3-distutils",
        "build-essential", "clang-14", "cmake", "ninja-build",
        "curl", "unzip", "ca-certificates",
    ], env=apt_env)
    python310 = shutil.which("python3.10")
    if python310 is None:
        raise RuntimeError("python3.10 was not installed")

    controller = CONTROL / "qairt242_colab_qwen25_genie.py"
    spec = CONTROL / "qairt242_qwen25_genie_spec.json"
    source_lock = CONTROL / "qairt242_qwen25_15b_genie_source_lock.json"
    write_status("running", "controller_plan")
    run([
        python310, str(controller), "plan",
        "--spec", str(spec), "--source-lock", str(source_lock),
    ])
    write_status("running", "controller_run")
    run([
        python310, str(controller), "run",
        "--spec", str(spec), "--source-lock", str(source_lock),
        "--workspace", str(WORKSPACE), "--scratch", str(SCRATCH),
    ])
    write_status("running", "controller_verify")
    run([
        python310, str(controller), "verify",
        "--spec", str(spec), "--source-lock", str(source_lock),
        "--workspace", str(WORKSPACE),
    ])
    manifest = WORKSPACE / "stages/deliverable/result_manifest.json"
    write_status(
        "complete",
        "complete",
        manifest=str(manifest),
        manifest_sha256=sha256(manifest),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        write_status(
            "failed_closed",
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
