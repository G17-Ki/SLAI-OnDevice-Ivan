#!/usr/bin/env python3
"""Inventory the official fixed-shape SM8850 SenseVoice QNN context package."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


ARCHIVE_NAME = (
    "sherpa-onnx-qnn-SM8850-binary-10-seconds-"
    "sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2"
)
ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models-qnn-binary/" + ARCHIVE_NAME
)
ARCHIVE_ASSET_ID = 326303090
ARCHIVE_BYTES = 162023574
ARCHIVE_SHA256 = (
    "ecbc1ffba39f8e23582b79a199d12e8455425a22ef7b6b18c535ce25fcff2d64"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    archive = args.archive.resolve()
    archive_hash = sha256(archive)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(
            f"archive size mismatch: expected {ARCHIVE_BYTES}, got {archive.stat().st_size}"
        )
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(
            f"archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {archive_hash}"
        )

    files = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(model_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    info = {}
    for line in (model_dir / "info.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = int(value) if value.isdigit() else value

    output = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": {
            "release_tag": "asr-models-qnn-binary",
            "asset_id": ARCHIVE_ASSET_ID,
            "name": ARCHIVE_NAME,
            "url": ARCHIVE_URL,
            "bytes": archive.stat().st_size,
            "sha256": archive_hash,
        },
        "target": {
            "soc": "Qualcomm SM8850",
            "htp_arch": "v81",
            "provider": "QNN HTP",
            "context_format": "pre-generated QNN context binary (model.bin)",
            "fixed_audio_limit_s": 10,
            "input_policy": "shorter audio is padded to 10 seconds; longer audio is truncated",
            "num_frames_from_info_txt": info.get("num_frames"),
        },
        "runtime": {
            "context_generated_with": "QNN SDK 2.40.0.251030",
            "required_application_libraries": [
                "libonnxruntime.so",
                "libQnnHtp.so",
                "libQnnHtpPrepare.so",
                "libQnnSystem.so",
                "libQnnHtpV81Stub.so",
            ],
            "required_dsp_library": "libQnnHtpV81Skel.so",
            "sdk_paths": {
                "android": "$QNN_SDK_ROOT/lib/aarch64-android",
                "dsp": "$QNN_SDK_ROOT/lib/hexagon-v81/unsigned",
            },
            "adsp_library_path_separator": ";",
        },
        "official_reference": {
            "device": "Xiaomi 17 Pro, SM8850",
            "ten_second_context": {
                "recognizer_init_s": 1.214,
                "decode_s": 0.290,
                "source_audio_s": 5.592,
                "effective_padded_context_s": 10,
            },
            "five_second_context": {
                "decode_s": 0.052,
                "source_audio_s": 5.592,
                "effective_truncated_context_s": 5,
            },
            "notes": (
                "Published Sherpa-ONNX log; not measured on the project's Nubia. "
                "The 10-second context is the recommended first package here because "
                "the diagnostic fixture contains clips slightly longer than 8 seconds."
            ),
        },
        "license": {
            "model_weights": "FunASR Model Open Source License Agreement v1.1 (custom)",
            "archive_license_file_sha256": sha256(model_dir / "LICENSE"),
            "runtime": "Sherpa-ONNX Apache-2.0; Qualcomm SDK has its own terms",
        },
        "files": files,
        "total_file_bytes": sum(record["bytes"] for record in files),
        "references": {
            "model_docs": "https://k2-fsa.github.io/sherpa/onnx/qnn/models.html",
            "phone_binary_guide": (
                "https://k2-fsa.github.io/sherpa/onnx/qnn/"
                "run-executables-on-your-phone-binary.html"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
