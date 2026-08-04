#!/usr/bin/env python3
"""Prepare checksum-pinned SenseVoice/QNN assets outside the repository.

The script deliberately keeps Qualcomm runtime object files and model payloads
out of Git. It can reuse a trusted local seed directory, but every file is
verified against the checked-in manifest before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MANIFEST_PATH = SCRIPT_DIR / "assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def verify(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing asset: {path}")
    if path.stat().st_size != spec["bytes"]:
        raise RuntimeError(
            f"size mismatch for {path.name}: {path.stat().st_size} != {spec['bytes']}"
        )
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: {actual} != {spec['sha256']}"
        )


def fetch(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LMF-demo-reproducible-asset-fetcher/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            os.replace(temporary, output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def acquire(asset_id: str, spec: dict, downloads: Path, seeds: list[Path]) -> Path:
    output = downloads / spec["filename"]
    if output.exists():
        verify(output, spec)
        return output

    for seed in seeds:
        candidate = seed / spec["filename"]
        if candidate.is_file():
            verify(candidate, spec)
            shutil.copy2(candidate, output)
            verify(output, spec)
            return output

    print(f"Downloading {asset_id}: {spec['filename']}", file=sys.stderr)
    fetch(spec["url"], output)
    verify(output, spec)
    return output


def safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:bz2") as bundle:
        root = destination.resolve()
        for member in bundle.getmembers():
            member_path = (destination / member.name).resolve()
            if not is_within(member_path, root):
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"unsupported archive member type: {member.name}")
        bundle.extractall(destination)


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {filename} under {root}, found {len(matches)}"
        )
    return matches[0]


def copy_tree_contents(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def prepare(root: Path, seeds: list[Path], include_apk: bool) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if is_within(root, REPO_ROOT):
        raise RuntimeError(
            "refusing to place QNN runtime/model payloads inside the repository"
        )

    root.mkdir(parents=True, exist_ok=True)
    downloads = root / "downloads"
    downloads.mkdir(exist_ok=True)
    selected = ["qnn_headers", "qnn_runtime", "sensevoice_context"]
    if include_apk:
        selected.append("upstream_smoke_apk")

    acquired: dict[str, Path] = {}
    for asset_id in selected:
        acquired[asset_id] = acquire(
            asset_id, manifest["assets"][asset_id], downloads, seeds
        )

    extracted = root / "extracted"
    for asset_id in ("qnn_headers", "qnn_runtime", "sensevoice_context"):
        safe_extract(acquired[asset_id], extracted / asset_id)

    qnn_version = manifest["target"]["qnn_version"]
    include_source = find_one(extracted / "qnn_headers", "QnnInterface.h").parent
    while include_source.name != "include":
        if include_source == include_source.parent:
            raise RuntimeError("could not locate QNN include root")
        include_source = include_source.parent
    sdk_root = root / "qnn-sdk" / qnn_version
    copy_tree_contents(include_source, sdk_root / "include")

    runtime_source = find_one(extracted / "qnn_runtime", "libQnnHtp.so").parent
    runtime_root = root / "runtime" / qnn_version
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    runtime_names = [
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnSystem.so",
        "libQnnHtpV81Stub.so",
        "libQnnHtpV81Skel.so",
    ]
    for name in runtime_names:
        source = runtime_source / name
        if not source.is_file():
            raise RuntimeError(f"missing required QNN runtime member: {name}")
        shutil.copy2(source, runtime_root / name)

    model_source = find_one(extracted / "sensevoice_context", "model.bin").parent
    model_root = root / "model" / "sensevoice-sm8850-10s-int8"
    if model_root.exists():
        shutil.rmtree(model_root)
    shutil.copytree(model_source, model_root)

    context_spec = manifest["assets"]["sensevoice_context"]
    if sha256(model_root / "model.bin") != context_spec["model_bin_sha256"]:
        raise RuntimeError("extracted model.bin hash mismatch")
    if sha256(model_root / "tokens.txt") != context_spec["tokens_sha256"]:
        raise RuntimeError("extracted tokens.txt hash mismatch")

    result = {
        "schema_version": 1,
        "target": manifest["target"],
        "sherpa_onnx": manifest["sherpa_onnx"],
        "paths": {
            "qnn_sdk_root": str(sdk_root),
            "qnn_runtime_root": str(runtime_root),
            "sensevoice_model_root": str(model_root),
            "upstream_smoke_apk": (
                str(acquired["upstream_smoke_apk"])
                if "upstream_smoke_apk" in acquired
                else None
            ),
        },
        "files": {},
        "distribution": {
            "qnn_runtime": "external acquisition only; do not commit raw Qualcomm objects",
            "model": "external acquisition only; preserve upstream license",
            "upstream_apk": "upstream smoke-test artifact; do not republish from this repository",
        },
    }
    for path in [
        *(runtime_root / name for name in runtime_names),
        model_root / "model.bin",
        model_root / "tokens.txt",
    ]:
        result["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    output = root / "bundle-manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    default_root = Path(
        os.environ.get(
            "LMF_NPU_BUNDLE_ROOT",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "lmf-demo"
            / "npu-multilingual"
            / "bundle",
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--seed-dir",
        action="append",
        type=Path,
        default=[],
        help="Reuse matching downloaded assets after mandatory hash verification.",
    )
    parser.add_argument("--include-upstream-apk", action="store_true")
    args = parser.parse_args()

    try:
        result = prepare(args.root.expanduser(), args.seed_dir, args.include_upstream_apk)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
