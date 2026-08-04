#!/usr/bin/env python3
"""Bootstrap a pinned Colab environment and run the Zipformer exporter.

This process is launched detached inside the Colab VM.  It is intentionally
standard-library-only until the isolated environment has been installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
import zipfile

import export_sources as exporter


PIP_VERSION = "25.1.1"
MAX_RESUME_MEMBERS = 1000
MAX_RESUME_BYTES = 4 * 1024 * 1024 * 1024
HARNESS_FILES = (
    "/content/export_sources.py",
    "/content/remote_worker.py",
    "/content/remote_launcher.py",
    "/content/remote_probe.py",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def update_status(root: Path, **updates: Any) -> dict[str, Any]:
    path = root / "worker_status.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {}
    payload.update(updates)
    payload["format_version"] = exporter.FORMAT_VERSION
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)
    return payload


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        detail = exporter.sanitize_error(completed.stdout[-4000:], 1600)
        raise RuntimeError(
            f"subprocess failed with exit code {completed.returncode}: "
            f"{Path(command[0]).name}; {detail}"
        )


def verify_resume_archive(path: Path) -> list[tarfile.TarInfo]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    members: list[tarfile.TarInfo]
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
    if len(members) > MAX_RESUME_MEMBERS:
        raise RuntimeError("resume archive contains too many members")
    total_size = 0
    names: set[str] = set()
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.isfile():
            raise RuntimeError("resume archive contains an unsafe member")
        if member.name in names:
            raise RuntimeError("resume archive contains duplicate members")
        names.add(member.name)
        total_size += member.size
        if total_size > MAX_RESUME_BYTES:
            raise RuntimeError("resume archive expands beyond the safety limit")
    if "SHA256SUMS" not in names:
        raise RuntimeError("resume archive is missing SHA256SUMS")
    return members


def restore_resume_archive(root: Path, path: Path) -> None:
    members = verify_resume_archive(path)
    with tarfile.open(path, "r:*") as archive:
        for member in members:
            destination = root.joinpath(*PurePosixPath(member.name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("resume member could not be read")
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(source, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
    verify_restored_checksums(root)


def verify_restored_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("resume checksum manifest could not be read") from error
    if not lines:
        raise RuntimeError("resume checksum manifest is empty")
    for line in lines:
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError("resume checksum manifest is malformed") from error
        pure = PurePosixPath(relative)
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or pure.is_absolute()
            or ".." in pure.parts
        ):
            raise RuntimeError("resume checksum entry is invalid")
        path = root.joinpath(*pure.parts)
        if not path.is_file() or exporter.sha256_file(path) != expected:
            raise RuntimeError("resume artifact checksum mismatch")


def write_harness_provenance(root: Path) -> None:
    records: dict[str, Any] = {}
    for raw_path in HARNESS_FILES:
        path = Path(raw_path)
        if path.is_file():
            records[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": exporter.sha256_file(path),
            }
    atomic_write_json(
        root / "harness_provenance.json",
        {
            "format_version": exporter.FORMAT_VERSION,
            "files": records,
            "created_at": utc_now(),
        },
    )


def environment_matches(venv_python: Path) -> bool:
    marker = venv_python.parent.parent / ".environment_ready.json"
    if not venv_python.is_file() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        payload.get("qaihm_version") == exporter.QAIHM_VERSION
        and payload.get("qaihm_wheel_sha256") == exporter.QAIHM_WHEEL_SHA256
        and payload.get("torch_version") == exporter.TORCH_VERSION
    )


def create_environment(root: Path) -> Path:
    venv = root / "venv"
    venv_python = venv / "bin" / "python"
    if environment_matches(venv_python):
        update_status(root, state="checking_existing_environment")
        run([str(venv_python), "-m", "pip", "check"])
        return venv_python

    if venv.exists():
        shutil.rmtree(venv)
    update_status(root, state="creating_isolated_environment")
    run([sys.executable, "-m", "venv", str(venv)])
    run([str(venv_python), "-m", "pip", "install", f"pip=={PIP_VERSION}"])

    package_cache = root / "package_cache"
    package_cache.mkdir(parents=True, exist_ok=True)
    update_status(root, state="downloading_pinned_qaihm_wheel")
    run(
        [
            str(venv_python),
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--disable-pip-version-check",
            f"qai-hub-models=={exporter.QAIHM_VERSION}",
            "--dest",
            str(package_cache),
        ]
    )
    wheels = sorted(package_cache.glob("qai_hub_models-0.58.0-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected exactly one QAIHM 0.58.0 wheel")
    wheel = wheels[0]
    wheel_digest = exporter.sha256_file(wheel)
    if wheel_digest != exporter.QAIHM_WHEEL_SHA256:
        raise RuntimeError("QAIHM wheel SHA-256 mismatch")
    with zipfile.ZipFile(wheel) as archive:
        patch_bytes = archive.read(
            "qai_hub_models/models/zipformer/external_repos/icefall_patches.diff"
        )
    patch_digest = hashlib.sha256(patch_bytes).hexdigest()
    if patch_digest != exporter.ICEFALL_PATCH_SHA256:
        raise RuntimeError("QAIHM Zipformer Icefall patch SHA-256 mismatch")

    update_status(root, state="installing_pinned_cpu_runtime")
    run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            f"torch=={exporter.TORCH_VERSION}+cpu",
            f"torchvision=={exporter.TORCHVISION_VERSION}+cpu",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    )
    run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            f"{wheel}[zipformer]",
            f"qai-hub=={exporter.QAI_HUB_VERSION}",
            f"onnx=={exporter.ONNX_VERSION}",
            f"onnxruntime=={exporter.ONNXRUNTIME_VERSION}",
            f"huggingface-hub=={exporter.HUGGINGFACE_HUB_VERSION}",
            f"k2=={exporter.K2_VERSION}",
            "--find-links",
            "https://k2-fsa.github.io/k2/cpu.html",
        ]
    )
    run([str(venv_python), "-m", "pip", "check"])

    freeze_lines = subprocess.run(
        [str(venv_python), "-m", "pip", "list", "--format=freeze"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.splitlines()
    freeze = "\n".join(sorted(line for line in freeze_lines if line.strip())) + "\n"
    exporter.atomic_write_bytes(root / "requirements.lock.txt", freeze.encode("utf-8"))

    version_probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import importlib.metadata,json;"
                "names=['qai-hub-models','qai-hub','torch','torchvision','onnx',"
                "'onnxruntime','huggingface-hub','k2'];"
                "print(json.dumps({n:importlib.metadata.version(n) for n in names}))"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    versions = json.loads(version_probe.stdout)
    expected = {
        "qai-hub-models": exporter.QAIHM_VERSION,
        "qai-hub": exporter.QAI_HUB_VERSION,
        "torch": f"{exporter.TORCH_VERSION}+cpu",
        "torchvision": f"{exporter.TORCHVISION_VERSION}+cpu",
        "onnx": exporter.ONNX_VERSION,
        "onnxruntime": exporter.ONNXRUNTIME_VERSION,
        "huggingface-hub": exporter.HUGGINGFACE_HUB_VERSION,
        "k2": exporter.K2_VERSION,
    }
    if versions != expected:
        raise RuntimeError(
            "installed package versions do not match the pinned environment"
        )

    provenance = {
        "format_version": exporter.FORMAT_VERSION,
        "python": subprocess.run(
            [
                str(venv_python),
                "-c",
                "import platform;print(platform.python_version())",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip(),
        "packages": versions,
        "qaihm_wheel": {
            "filename": wheel.name,
            "size_bytes": wheel.stat().st_size,
            "sha256": wheel_digest,
        },
        "zipformer_icefall_patch_sha256": patch_digest,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "package_provenance.json", provenance)
    atomic_write_json(
        venv / ".environment_ready.json",
        {
            "qaihm_version": exporter.QAIHM_VERSION,
            "qaihm_wheel_sha256": exporter.QAIHM_WHEEL_SHA256,
            "torch_version": exporter.TORCH_VERSION,
        },
    )
    return venv_python


def run_worker(root: Path, components: str, threads: int, resume: Path | None) -> int:
    root.mkdir(parents=True, exist_ok=True)
    if resume is not None and resume.is_file():
        update_status(root, state="restoring_checkpoint")
        restore_resume_archive(root, resume)
    update_status(
        root,
        state="starting",
        pid=os.getpid(),
        requested_components=list(exporter.normalize_components(components)),
    )
    write_harness_provenance(root)

    try:
        venv_python = create_environment(root)
        update_status(root, state="running_exporter")
        environment = os.environ.copy()
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[variable] = str(threads)
        environment["ZIPFORMER_EXPORT_THREADS"] = str(threads)
        run(
            [
                str(venv_python),
                "/content/export_sources.py",
                "--root",
                str(root),
                "--components",
                components,
                "--threads",
                str(threads),
            ],
            env=environment,
        )
        update_status(root, state="succeeded", pid=os.getpid(), completed_at=utc_now())
        exporter.refresh_checkpoint(root, "remote worker succeeded")
        return 0
    except Exception as error:
        update_status(
            root,
            state="failed",
            pid=os.getpid(),
            error_type=type(error).__name__,
            error=exporter.sanitize_error(error),
            failed_at=utc_now(),
        )
        try:
            exporter.refresh_checkpoint(root, "remote worker failed")
        except Exception:
            pass
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/content/zipformer-source-v058")
    )
    parser.add_argument("--components", default=",".join(exporter.COMPONENT_ORDER))
    parser.add_argument("--threads", type=int, default=exporter.DEFAULT_THREADS)
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exporter.normalize_components(args.components)
    if not 1 <= args.threads <= 8:
        raise SystemExit("--threads must be between 1 and 8")
    return run_worker(args.root, args.components, args.threads, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
