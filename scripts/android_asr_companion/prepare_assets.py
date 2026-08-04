#!/usr/bin/env python3
"""Acquire and verify external Android ASR companion assets.

No model, Qualcomm runtime, or Android executable is copied into the
repository.  The checked-in files contain only URLs, hashes, and orchestration
logic. SenseVoice/QNN acquisition delegates to the existing pinned manifest
and preparer in scripts/npu_multilingual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ASSET_MANIFEST = SCRIPT_DIR / "assets.json"
SENSEVOICE_PREPARER = REPO_ROOT / "scripts/npu_multilingual/prepare_sensevoice_sm8850_bundle.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_external(path: Path, label: str) -> None:
    if is_within(path, REPO_ROOT):
        raise RuntimeError(f"{label} must remain outside the repository: {path}")


def verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing required file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"size mismatch for {path.name}: {actual_bytes} != {expected_bytes}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: {actual_sha256} != {expected_sha256}"
        )


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LMF-demo-android-asr-companion/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            os.replace(temporary, output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def acquire(
    spec: dict[str, Any],
    downloads: Path,
    seed_dirs: list[Path],
    offline: bool,
) -> Path:
    output = downloads / spec["filename"]
    if output.exists():
        verify_file(
            output,
            expected_bytes=spec["bytes"],
            expected_sha256=spec["sha256"],
        )
        return output
    for seed_dir in seed_dirs:
        candidate = seed_dir.expanduser() / spec["filename"]
        if candidate.is_file():
            verify_file(
                candidate,
                expected_bytes=spec["bytes"],
                expected_sha256=spec["sha256"],
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, output)
            verify_file(
                output,
                expected_bytes=spec["bytes"],
                expected_sha256=spec["sha256"],
            )
            return output
    if offline:
        raise RuntimeError(f"offline mode: missing pinned asset {spec['filename']}")
    print(f"Downloading {spec['url']}", file=sys.stderr)
    download(spec["url"], output)
    verify_file(
        output,
        expected_bytes=spec["bytes"],
        expected_sha256=spec["sha256"],
    )
    return output


def safe_extract_model(archive: Path, destination: Path, model_directory: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent)
    )
    try:
        with tarfile.open(archive, "r:bz2") as bundle:
            temporary_root = temporary.resolve()
            for member in bundle.getmembers():
                member_path = (temporary / member.name).resolve()
                if not is_within(member_path, temporary_root):
                    raise RuntimeError(f"unsafe archive member: {member.name}")
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"unsupported archive member: {member.name}")
            bundle.extractall(temporary, filter="data")
        extracted = temporary / model_directory
        if not extracted.is_dir():
            raise RuntimeError(f"archive lacks expected root {model_directory}")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(extracted, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def verify_sensevoice(
    bundle_root: Path,
    sensevoice_manifest: dict[str, Any],
    seed_dirs: list[Path],
    offline: bool,
) -> dict[str, Any]:
    require_external(bundle_root, "SenseVoice bundle")
    downloads = bundle_root / "downloads"
    required_archive_ids = ("qnn_headers", "qnn_runtime", "sensevoice_context")
    archives_ready = all(
        (downloads / sensevoice_manifest["assets"][asset_id]["filename"]).is_file()
        for asset_id in required_archive_ids
    )
    if not (bundle_root / "bundle-manifest.json").is_file() or not archives_ready:
        if offline:
            raise RuntimeError("offline mode: prepared SenseVoice/QNN bundle is incomplete")
        command = [sys.executable, str(SENSEVOICE_PREPARER), "--root", str(bundle_root)]
        for seed_dir in seed_dirs:
            command.extend(("--seed-dir", str(seed_dir.expanduser())))
        subprocess.run(command, check=True)

    for asset_id in required_archive_ids:
        spec = sensevoice_manifest["assets"][asset_id]
        verify_file(
            downloads / spec["filename"],
            expected_bytes=spec["bytes"],
            expected_sha256=spec["sha256"],
        )

    manifest_path = bundle_root / "bundle-manifest.json"
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundle.get("target") != sensevoice_manifest["target"]:
        raise RuntimeError("SenseVoice bundle target does not match the pinned manifest")
    paths = bundle.get("paths", {})
    runtime_root = Path(paths.get("qnn_runtime_root", "__missing__")).expanduser()
    model_root = Path(paths.get("sensevoice_model_root", "__missing__")).expanduser()
    for path, label in ((runtime_root, "QNN runtime"), (model_root, "SenseVoice model")):
        require_external(path, label)
        if not is_within(path, bundle_root):
            raise RuntimeError(f"{label} path escapes prepared bundle: {path}")

    expected_members = bundle.get("files", {})
    required_runtime = (
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnSystem.so",
        "libQnnHtpV81Stub.so",
        "libQnnHtpV81Skel.so",
    )
    for filename in required_runtime:
        spec = expected_members.get(filename)
        if not spec:
            raise RuntimeError(f"bundle manifest lacks {filename}")
        verify_file(
            runtime_root / filename,
            expected_bytes=spec["bytes"],
            expected_sha256=spec["sha256"],
        )
    context_spec = sensevoice_manifest["assets"]["sensevoice_context"]
    model_bin = model_root / "model.bin"
    tokens = model_root / "tokens.txt"
    if sha256(model_bin) != context_spec["model_bin_sha256"]:
        raise RuntimeError("SenseVoice model.bin hash mismatch")
    if sha256(tokens) != context_spec["tokens_sha256"]:
        raise RuntimeError("SenseVoice tokens.txt hash mismatch")
    return {
        "bundle_manifest": str(manifest_path),
        "runtime_root": str(runtime_root),
        "model_root": str(model_root),
    }


def verify_runner(runner_root: Path) -> dict[str, str]:
    require_external(runner_root, "Android runner")
    manifest_path = runner_root / "runner-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"missing runner manifest at {manifest_path}; run "
            "scripts/npu_multilingual/build_sherpa_qnn_android.sh first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_target = {"abi": "arm64-v8a", "provider": "qnn", "android_api": 27}
    if manifest.get("target") != expected_target:
        raise RuntimeError("runner target does not match pinned Android arm64 QNN build")
    if manifest.get("sherpa_onnx_commit") != "142807252687d81b40d6315f23470a1512a00de3":
        raise RuntimeError("runner Sherpa-ONNX commit is not pinned v1.13.4")
    for filename in ("sherpa-onnx-offline", "libonnxruntime.so"):
        spec = manifest.get("files", {}).get(filename)
        if not spec:
            raise RuntimeError(f"runner manifest lacks {filename}")
        verify_file(
            runner_root / filename,
            expected_bytes=spec["bytes"],
            expected_sha256=spec["sha256"],
        )
    return {
        "root": str(runner_root),
        "cli": str(runner_root / "sherpa-onnx-offline"),
        "onnxruntime": str(runner_root / "libonnxruntime.so"),
    }


def prepare_russian(
    external_root: Path,
    spec: dict[str, Any],
    seed_dirs: list[Path],
    offline: bool,
) -> dict[str, str]:
    require_external(external_root, "Russian model cache")
    archive = acquire(spec, external_root / "downloads", seed_dirs, offline)
    model_root = external_root / "models" / spec["model_directory"]
    marker = model_root / ".archive.sha256"
    reusable = marker.is_file() and marker.read_text(encoding="ascii").strip() == spec["sha256"]
    reusable = reusable and all((model_root / name).is_file() for name in spec["required_files"])
    if not reusable:
        safe_extract_model(archive, model_root, spec["model_directory"])
        marker.write_text(spec["sha256"] + "\n", encoding="ascii")
    for filename in spec["required_files"]:
        path = model_root / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Russian model member missing or empty: {filename}")
    return {"archive": str(archive), "model_root": str(model_root)}


def main() -> int:
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        type=Path,
        default=default_cache / "lmf-demo/android-asr-companion",
    )
    parser.add_argument(
        "--sensevoice-bundle-root",
        type=Path,
        default=default_cache / "lmf-demo/npu-multilingual/bundle",
    )
    parser.add_argument(
        "--runner-root",
        type=Path,
        default=default_cache
        / "lmf-demo/npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn",
    )
    parser.add_argument("--seed-dir", type=Path, action="append", default=[])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
        sensevoice_path = (SCRIPT_DIR / manifest["sensevoice_manifest"]).resolve()
        sensevoice_manifest = json.loads(sensevoice_path.read_text(encoding="utf-8"))
        external_root = args.external_root.expanduser().resolve()
        bundle_root = args.sensevoice_bundle_root.expanduser().resolve()
        runner_root = args.runner_root.expanduser().resolve()
        require_external(external_root, "Android ASR companion cache")
        external_root.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": 1,
            "repository_payloads": False,
            "sensevoice": verify_sensevoice(
                bundle_root, sensevoice_manifest, args.seed_dir, args.offline
            ),
            "russian": prepare_russian(
                external_root,
                manifest["russian_zipformer"],
                args.seed_dir,
                args.offline,
            ),
            "runner": verify_runner(runner_root),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
