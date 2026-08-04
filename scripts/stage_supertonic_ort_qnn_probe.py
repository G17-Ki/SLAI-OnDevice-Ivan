#!/usr/bin/env python3
"""Stage the exact Supertonic duration raw-ONNX ORT-QNN phone canary.

This is packaging and deterministic fixture preparation only. It does not run
model conversion or inference on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(
    "/private/tmp/lmf-tts-bundle-20260729-1145/supertonic3-int8/"
    "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
)
DEFAULT_RUNTIME_INVENTORY = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_ort_qnn_runtime_inventory.json"
)
DEFAULT_ORT_AAR = Path.home() / (
    ".cache/lmf-demo/whisper-small-v0.58.0/"
    "onnxruntime-android-qnn-1.25.0.aar"
)

GATE_ID = "supertonic3_duration_raw_onnx_ort_qnn_sm8850_probe_v1"
PACKAGE = "ai.liquid.lfmdemo.supertonicprobe"
MODEL_NAME = "duration_predictor.int8.onnx"
MODEL_BYTES = 3_700_147
MODEL_SHA256 = "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"
VOICE_BYTES = 517_168
VOICE_SHA256 = "67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8"
ORT_BYTES = 7_720_525
ORT_SHA256 = "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198"
QAIRT_BUILD = "2.42.0.251225135753_193295"
SPEAKER = 6
EXPECTED_DURATION = np.float32(2.127588987350464)

TEXT_IDS = np.asarray(
    [
        29, 64, 73, 31, 41, 79, 2, 68, 78, 2, 48, 60, 77, 68, 78, 13,
        2, 68, 73, 2, 38, 77, 60, 73, 62, 64, 15, 29, 16, 64, 73, 31,
    ],
    dtype=np.int64,
).reshape(1, 32)

EXPECTED_ARRAY_IDENTITIES = {
    "text_ids": "2fd3fac41650673075085524f4b2e3a5aed442ddff672caeba14c9f7cb8d5890",
    "style_dp": "1340f291942c45d0ff4226da9c77ecc1e8cf4301379a2f1249a1efbf77ea4e21",
    "text_mask": "4c821d782d02e504c72b4f24521dd786a78f6e8af8f14aa9a4e85ab96089f746",
    "duration": "9790218b17df1d99d3dfcf02a11a7aafc59f3e8346ee0f4e55dbb2fda7df5cbe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--runtime-inventory", type=Path, default=DEFAULT_RUNTIME_INVENTORY
    )
    parser.add_argument("--ort-aar", type=Path, default=DEFAULT_ORT_AAR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")


def array_identity(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parse_duration_style(path: Path, speaker: int) -> np.ndarray:
    payload = path.read_bytes()
    dims = struct.unpack("<6q", payload[: 6 * 8])
    ttl_shape = tuple(int(value) for value in dims[:3])
    dp_shape = tuple(int(value) for value in dims[3:])
    if ttl_shape != (10, 50, 256) or dp_shape != (10, 8, 16):
        raise RuntimeError(f"unexpected Supertonic voice shapes: {dims}")
    if not 0 <= speaker < dp_shape[0]:
        raise ValueError("speaker is outside voice.bin bounds")
    ttl_count = int(np.prod(ttl_shape))
    dp_count = int(np.prod(dp_shape))
    expected_bytes = 6 * 8 + (ttl_count + dp_count) * 4
    if len(payload) != expected_bytes:
        raise RuntimeError("voice.bin payload length does not match its header")
    values = np.frombuffer(payload, dtype="<f4", offset=6 * 8)
    duration_values = values[ttl_count : ttl_count + dp_count].reshape(dp_shape)
    return duration_values[speaker : speaker + 1].astype(np.float32, copy=True)


def write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def copy_checked(
    source: Path,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    verify_file(source, expected_bytes, expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    verify_file(temporary, expected_bytes, expected_sha256)
    temporary.replace(destination)
    return {"bytes": expected_bytes, "sha256": expected_sha256}


def write_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes(path, encoded)


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output == Path(output.anchor):
        raise RuntimeError("refusing to stage at filesystem root")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir.expanduser().resolve()
    model_source = model_dir / MODEL_NAME
    voice_source = model_dir / "voice.bin"
    verify_file(model_source, MODEL_BYTES, MODEL_SHA256)
    verify_file(voice_source, VOICE_BYTES, VOICE_SHA256)
    verify_file(args.ort_aar.expanduser().resolve(), ORT_BYTES, ORT_SHA256)

    style_dp = parse_duration_style(voice_source, SPEAKER)
    text_mask = np.ones((1, 1, 32), dtype=np.float32)
    duration = np.asarray([EXPECTED_DURATION], dtype=np.float32)
    arrays = {
        "text_ids": TEXT_IDS,
        "style_dp": style_dp,
        "text_mask": text_mask,
        "duration": duration,
    }
    for name, value in arrays.items():
        actual = array_identity(value)
        if actual != EXPECTED_ARRAY_IDENTITIES[name]:
            raise RuntimeError(f"{name} fixture identity mismatch: {actual}")

    asset_root = output / "assets/supertonic_probe"
    model_destination = asset_root / "models" / MODEL_NAME
    model_record = copy_checked(
        model_source, model_destination, MODEL_BYTES, MODEL_SHA256
    )

    raw_assets = [
        ("fixtures/text_ids.i64le", TEXT_IDS.astype("<i8", copy=False)),
        ("fixtures/style_dp.f32le", style_dp.astype("<f4", copy=False)),
        ("fixtures/text_mask.f32le", text_mask.astype("<f4", copy=False)),
        ("fixtures/duration.f32le", duration.astype("<f4", copy=False)),
    ]
    asset_records: list[dict[str, Any]] = [
        {
            "path": f"supertonic_probe/models/{MODEL_NAME}",
            "role": "source_duration_predictor_int8_onnx",
            **model_record,
        }
    ]
    for relative_path, value in raw_assets:
        destination = asset_root / relative_path
        identity = write_bytes(destination, value.tobytes(order="C"))
        asset_records.append(
            {
                "path": f"supertonic_probe/{relative_path}",
                "role": relative_path.rsplit("/", 1)[-1],
                **identity,
            }
        )

    inventory_path = args.runtime_inventory.expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("release", {}).get("qairt_build") != QAIRT_BUILD:
        raise RuntimeError("runtime inventory is not the accepted QAIRT 2.42 build")
    runtime_root = Path(inventory["root"])
    native_records: list[dict[str, Any]] = []
    for item in inventory["files"]:
        source = runtime_root / item["path"]
        destination = output / "jniLibs/arm64-v8a" / item["name"]
        copied = copy_checked(source, destination, item["bytes"], item["sha256"])
        native_records.append(
            {
                "path": f"arm64-v8a/{item['name']}",
                "role": f"qairt_{item['name']}",
                "elf": "hexagon-v81" if "Skel" in item["name"] else "aarch64",
                **copied,
            }
        )

    aar_destination = output / "aar/onnxruntime-android-qnn-1.25.0.aar"
    copy_checked(
        args.ort_aar.expanduser().resolve(),
        aar_destination,
        ORT_BYTES,
        ORT_SHA256,
    )

    manifest = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "target": {
            "package": PACKAGE,
            "soc": "SM8850",
            "abi": "arm64-v8a",
            "htp_arch": "v81",
            "android_api": 28,
        },
        "runtime": {
            "onnx_runtime": "1.25.0",
            "qairt": QAIRT_BUILD,
            "qnn_provider": "QNNExecutionProvider",
            "backend": "libQnnHtp.so",
            "disable_cpu_ep_fallback": True,
        },
        "model": {
            "id": "supertonic-3-shared-int8",
            "component": "duration_predictor",
            "source_format": "raw_onnx",
            "source_sha256": MODEL_SHA256,
            "opset": 19,
            "symbolic_dimension_overrides": {
                "batch_size": 1,
                "text_length": 32,
            },
            "input_contract": {
                "text_ids": {"shape": [1, 32], "dtype": "int64"},
                "style_dp": {"shape": [1, 8, 16], "dtype": "float32"},
                "text_mask": {"shape": [1, 1, 32], "dtype": "float32"},
            },
            "output_contract": {
                "duration": {"shape": [1], "dtype": "float32"},
            },
        },
        "fixture": {
            "text": "It is Paris, in France.",
            "processed_text": "<en>It is Paris, in France.</en>",
            "language": "en",
            "speaker": SPEAKER,
            "expected_duration": float(EXPECTED_DURATION),
            "absolute_tolerance": 0.1,
            "array_identities": EXPECTED_ARRAY_IDENTITIES,
        },
        "capabilities": {
            "session_load": True,
            "single_duration_execution": True,
            "full_tts": False,
            "audio_output": False,
            "production_integration": False,
        },
        "assets": sorted(asset_records, key=lambda item: item["path"]),
        "native_libraries": sorted(native_records, key=lambda item: item["path"]),
        "claim_boundary": (
            "Raw-ONNX duration-component compatibility only. No full TTS, audio "
            "quality, latency, or production-runtime claim."
        ),
    }
    manifest_path = asset_root / "payload_manifest.json"
    write_json(manifest_path, manifest)
    stage_summary = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "output": str(output),
        "payload_manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "ort_aar": {
            "path": str(aar_destination),
            "bytes": ORT_BYTES,
            "sha256": ORT_SHA256,
        },
        "host_model_execution": False,
        "status": "staged",
    }
    write_json(output / "probe_payload_manifest.json", stage_summary)
    print(json.dumps(stage_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
