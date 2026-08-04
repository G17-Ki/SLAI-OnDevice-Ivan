#!/usr/bin/env python3
"""Stage verified native executables/libraries for the app-owned runtime APK.

The output must stay outside the repository. Model weights are installed into
app-private phone storage separately. Native runners, shared libraries, and the
HTP DSP skeleton are immutable APK payloads so Android/FastRPC can load them
from the installed native-library directory without an adb-owned process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEXT_ASSETS = REPO_ROOT / "scripts/android_text_companion/assets.json"
EXPECTED_ASR_RUNNER_PATCH = {
    "name": "sherpa_onnx_v1.13.4_persistent_offline.patch",
    "sha256": "d048024b9ff24d044e6780400f61a1cba805e09c33d6b2925b5d530c6a71442c",
    "patched_source_sha256": (
        "e2ba4608199103815bd98f9ff354ded2988d2ef7ec0e039a9dac7a82f9600208"
    ),
}
EXPECTED_ASR_RUNNER_PROTOCOL = {
    "name": "stdin_wav_path_jsonl",
    "version": 1,
    "server_flag": "--persistent-server=true",
    "copies_flag": "--persistent-copies=",
    "request_framing": "utf8_wav_path_line",
    "response_framing": "utf8_json_line",
    "ready_payload": {"persistent_ready": True, "protocol_version": 1},
    "quit_command": "__SHERPA_ONNX_PERSISTENT_QUIT__",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing required native file: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise RuntimeError(f"size mismatch for {path.name}")
    if sha256(path) != spec["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}")


def require_external(path: Path) -> None:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return
    raise RuntimeError(f"native staging output must remain outside the repository: {path}")


def copy_verified(source: Path, destination: Path, spec: dict[str, Any]) -> None:
    verify(source, spec)
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    verify(destination, spec)


def main() -> int:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lmf-demo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asr-runner-root",
        type=Path,
        default=cache_root
        / "npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn",
    )
    parser.add_argument(
        "--qnn-bundle-root",
        type=Path,
        default=cache_root / "npu-multilingual/bundle",
    )
    parser.add_argument(
        "--text-runtime-root",
        type=Path,
        default=cache_root / "android-text-companion/runtime/llama-b10083",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    require_external(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    abi_output = output / "arm64-v8a"
    abi_output.mkdir()

    runner_root = args.asr_runner_root.expanduser().resolve()
    runner_manifest = json.loads(
        (runner_root / "runner-manifest.json").read_text(encoding="utf-8")
    )
    if runner_manifest.get("target") != {
        "abi": "arm64-v8a",
        "android_api": 27,
        "provider": "qnn",
    }:
        raise RuntimeError("ASR runner manifest does not target Android arm64 QNN")
    if runner_manifest.get("sherpa_onnx_patch") != EXPECTED_ASR_RUNNER_PATCH:
        raise RuntimeError("ASR runner manifest does not contain the pinned persistent patch")
    if runner_manifest.get("persistent_protocol") != EXPECTED_ASR_RUNNER_PROTOCOL:
        raise RuntimeError("ASR runner manifest does not provide persistent protocol v1")
    runner_files = runner_manifest["files"]
    copy_verified(
        runner_root / "sherpa-onnx-offline",
        abi_output / "libasrrunner.so",
        runner_files["sherpa-onnx-offline"],
    )
    copy_verified(
        runner_root / "libonnxruntime.so",
        abi_output / "libonnxruntime.so",
        runner_files["libonnxruntime.so"],
    )

    qnn_bundle = args.qnn_bundle_root.expanduser().resolve()
    qnn_manifest = json.loads(
        (qnn_bundle / "bundle-manifest.json").read_text(encoding="utf-8")
    )
    if qnn_manifest.get("target", {}).get("soc") != "SM8850":
        raise RuntimeError("QNN bundle is not the pinned SM8850 target")
    qnn_root = Path(qnn_manifest["paths"]["qnn_runtime_root"])
    qnn_files = qnn_manifest["files"]
    for name in (
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpV81Skel.so",
        "libQnnHtpV81Stub.so",
        "libQnnSystem.so",
    ):
        copy_verified(qnn_root / name, abi_output / name, qnn_files[name])

    text_manifest = json.loads(TEXT_ASSETS.read_text(encoding="utf-8"))["runner"]
    text_root = args.text_runtime_root.expanduser().resolve()
    for name, spec in text_manifest["files"].items():
        if name == "LICENSE":
            continue
        destination_name = "libqwenserver.so" if name == "llama-server" else name
        copy_verified(text_root / name, abi_output / destination_name, spec)

    staged = {}
    for path in sorted(abi_output.iterdir()):
        if path.is_file():
            staged[path.name] = {
                "apk_path": f"lib/arm64-v8a/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "repository_payloads": False,
        "target": {"soc": "SM8850", "abi": "arm64-v8a", "htp": "v81"},
        "asr_runner_commit": runner_manifest.get("sherpa_onnx_commit"),
        "asr_runner_patch": runner_manifest["sherpa_onnx_patch"],
        "asr_runner_protocol": runner_manifest["persistent_protocol"],
        "text_runtime_revision": text_manifest["source_revision"],
        "files": staged,
    }
    manifest_path = output.parent / (output.name + "-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"jni_dir": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
