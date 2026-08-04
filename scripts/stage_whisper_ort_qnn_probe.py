#!/usr/bin/env python3
"""Stage the pinned Whisper Small ORT-QNN probe payload outside Git.

The staged directory is a build input for the isolated
``ai.liquid.lfmdemo.whisperprobe`` APK. It contains the exact ORT 1.25 QNN AAR,
QAIRT 2.42 SM8850/v81 runtime, external EPContext wrappers/binaries, and the
VoiceAI vocabulary/config provenance. No device interaction is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GATE_ID = "nubia_whisper_small_ort_qnn_epcontext_sm8850_probe_v1"
PACKAGE_ID = "ai.liquid.lfmdemo.whisperprobe"
MODEL_ARCHIVE_NAME = (
    "whisper_small_quantized-precompiled_qnn_onnx-w8a16-"
    "qualcomm_snapdragon_8_elite_gen5-v0.55.0.zip"
)
MODEL_ARCHIVE_ROOT = (
    "whisper_small_quantized-precompiled_qnn_onnx-w8a16-"
    "qualcomm_snapdragon_8_elite_gen5"
)
VOICE_ARCHIVE_NAME = (
    "whisper_small_quantized-voice_ai-w8a16-"
    "qualcomm_snapdragon_8_elite_gen5_for_galaxy.zip"
)
VOICE_ARCHIVE_ROOT = (
    "whisper_small_quantized-voice_ai-w8a16-"
    "qualcomm_snapdragon_8_elite_gen5_for_galaxy"
)
ORT_AAR_NAME = "onnxruntime-android-qnn-1.25.0.aar"


MODEL_ARCHIVE = {
    "bytes": 294_053_273,
    "sha256": "3cc2d238a167b7fc9094ae02a08d5cfdab0238fd8211ccde97bb7ff2897dd1be",
}
VOICE_ARCHIVE = {
    "bytes": 294_167_485,
    "sha256": "7472bd2bdc06d71dd95039ff2202e8bd8bf8e89dcd1a8cef02c8724dd91af69c",
}
ORT_AAR = {
    "bytes": 7_720_525,
    "sha256": "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198",
}

FEATURE_FIXTURE = {
    "bytes": 480_000,
    "sha256": "29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1",
    "source_wav_sha256": "b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69",
}


MODEL_MEMBERS: dict[str, dict[str, Any]] = {
    "encoder.onnx": {
        "bytes": 12_690,
        "sha256": "ef98bc252765daf43ce30bfc411799a7eca694b5d7afd41cc3aca04fd0ebc117",
    },
    "encoder_qairt_context.bin": {
        "bytes": 133_652_480,
        "sha256": "72b46ef12f7e33a2b1f4afc8712b5f85375a7cb52b8d8119095402825d5d62b7",
    },
    "decoder.onnx": {
        "bytes": 39_448,
        "sha256": "68d2df9598ef91ca259e4fd492371f128011df805bc837143e413a831bb7944b",
    },
    "decoder_qairt_context.bin": {
        "bytes": 225_492_992,
        "sha256": "0eafa59c9f921cf195a344d5004cf596f36f59cef96fe42b8b0b01827fd0716a",
    },
    "metadata.json": {
        "bytes": 43_677,
        "sha256": "451c583967ff6262c4afeee71ec32add18c54c49cefa5a4fc464b4388b31a9c0",
    },
}


VOICE_MEMBERS: dict[str, dict[str, Any]] = {
    "config.json": {
        "bytes": 874,
        "sha256": "b55a56acd03d62469e00566b9f48230485b4bd6919e85134f662a4a20f48f281",
    },
    "vocab.bin": {
        "bytes": 357_313,
        "sha256": "0ba87984671b92e03b56b84ce9b217020663f6a269b5a9800901391430b79c4b",
    },
}


QAIRT_FILES: dict[str, dict[str, Any]] = {
    "libQnnSystem.so": {
        "relative": "lib/aarch64-android/libQnnSystem.so",
        "bytes": 2_689_096,
        "sha256": "a69cfc4350c16b226bcaa19c657234820bb93e3ff6116cf3845ae77557f1f7c3",
        "elf": "aarch64",
    },
    "libQnnHtp.so": {
        "relative": "lib/aarch64-android/libQnnHtp.so",
        "bytes": 2_641_976,
        "sha256": "10eb1923b34ac9a726e4a446b395d5eee2ea5388487917778aaf954e8bfb2d92",
        "elf": "aarch64",
    },
    "libQnnHtpPrepare.so": {
        "relative": "lib/aarch64-android/libQnnHtpPrepare.so",
        "bytes": 83_685_856,
        "sha256": "2daafb376fe6904aa325e29810480e3d76dc392991d1367ae6ea3508db2b60ac",
        "elf": "aarch64",
    },
    "libQnnHtpV81Stub.so": {
        "relative": "lib/aarch64-android/libQnnHtpV81Stub.so",
        "bytes": 714_000,
        "sha256": "6e9e480223b248e70c43954e89dfbd9962600b59bdc9f3069bf4780fdcf98bba",
        "elf": "aarch64",
    },
    "libQnnHtpV81Skel.so": {
        "relative": "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so",
        "bytes": 11_271_968,
        "sha256": "a3e62af7044227804bc229437107cb54425e05f43d57cc26a876dd94d54a0cfc",
        "elf": "hexagon-v81",
    },
}


ORT_NATIVE_FILES: dict[str, dict[str, Any]] = {
    "libonnxruntime.so": {
        "bytes": 20_352_056,
        "sha256": "491f2fa4260cf8d379d0720795e211325b294db830ce7db2055bce9733c849a2",
        "elf": "aarch64",
    },
    "libonnxruntime4j_jni.so": {
        "bytes": 111_488,
        "sha256": "72496f82823056fd3d2195e668a7afbb7771892419befa36c5e6d1c9bd6d204c",
        "elf": "aarch64",
    },
}


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def verify_file(path: Path, spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing required file: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise RuntimeError(f"size mismatch for {path}")
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {path}")


def require_external_output(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return
    raise RuntimeError(f"probe payload must remain outside the repository: {resolved}")


def copy_verified(source: Path, destination: Path, spec: dict[str, Any]) -> None:
    verify_file(source, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    verify_file(destination, spec)


def extract_verified_member(
    archive: Path,
    member: str,
    destination: Path,
    spec: dict[str, Any],
) -> None:
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(member)
        except KeyError as error:
            raise RuntimeError(f"missing archive member: {member}") from error
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"refusing symlink archive member: {member}")
        if info.is_dir() or info.file_size != int(spec["bytes"]):
            raise RuntimeError(f"archive member size/type mismatch: {member}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(info) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    verify_file(destination, spec)


def file_record(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_manifest(
    asset_records: list[dict[str, Any]],
    native_records: list[dict[str, Any]],
    feature_fixture: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "target": {
            "package": PACKAGE_ID,
            "abi": "arm64-v8a",
            "soc": "SM8850",
            "htp_arch": "v81",
            "android_api": 28,
        },
        "capabilities": {
            "session_load": True,
            "contract_enumeration": True,
            "synthetic_quantized_encoder": True,
            "four_step_decode": True,
            "vocab_decode": True,
            "prequantized_feature_fixture": feature_fixture is not None,
            "real_audio_frontend": False,
            "real_audio_transcription": False,
            "microphone_input": False,
        },
        "decode_contract": {
            "initial_token_id": 50258,
            "eos_token_id": 50257,
            "steps": 4,
            "attention_mask_unfilled_uint16": 0,
            "attention_mask_filled_uint16": 65535,
            "attention_unmask_order": [199, 198, 197, 196],
            "self_kv_logical_zero_uint8": 128,
            "position_ids": [0, 1, 2, 3],
            "logits_argmax_dtype": "uint16",
        },
        "runtime": {
            "onnx_runtime": "1.25.0",
            "qairt": "2.42.0.251225135753_193295",
            "qnn_provider": "QNNExecutionProvider",
            "backend": "libQnnHtp.so",
            "disable_cpu_ep_fallback": True,
        },
        "model": {
            "id": "whisper_small_quantized",
            "runtime": "precompiled_qnn_onnx",
            "precision": "w8a16",
            "encoder_feature_shape": [1, 80, 3000],
            "encoder_feature_dtype": "uint16",
            "encoder_feature_zero_point": 32072,
            "tensor_contract_sha256": (
                "b2ecfbc6d7ffabfb933a32f04a4b20f03ee7d15a40001f631372f72220f1b2a3"
            ),
        },
        "vocabulary": {
            "base_entries": 50257,
            "format": "NUL-delimited raw token bytes",
            "raw_nul_token_id": 188,
            "empty_token_id": 50256,
            "first_special_token_id": 50257,
            "decode": "concatenate bytes, then decode UTF-8 once",
        },
        "feature_fixture": feature_fixture,
        "assets": sorted(asset_records, key=lambda item: item["path"]),
        "native_libraries": sorted(native_records, key=lambda item: item["path"]),
        "build_inputs": {
            "ort_aar": {
                "path": f"aar/{ORT_AAR_NAME}",
                **ORT_AAR,
            },
            "model_archive": {
                "filename": MODEL_ARCHIVE_NAME,
                **MODEL_ARCHIVE,
            },
            "voice_ai_archive": {
                "filename": VOICE_ARCHIVE_NAME,
                **VOICE_ARCHIVE,
                "config_runtime_note": (
                    "config.json requests QAIRT 2.45; only config/vocab provenance is reused. "
                    "The ORT EPContext route is pinned to its own QAIRT 2.42 build."
                ),
            },
        },
        "repository_payloads": False,
    }


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lmf-demo"
    whisper_root = cache_root / "whisper-small-v0.58.0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-archive",
        type=Path,
        default=whisper_root / MODEL_ARCHIVE_NAME,
    )
    parser.add_argument(
        "--voice-ai-archive",
        type=Path,
        default=whisper_root / VOICE_ARCHIVE_NAME,
    )
    parser.add_argument(
        "--ort-aar",
        type=Path,
        default=whisper_root / ORT_AAR_NAME,
    )
    parser.add_argument(
        "--qairt-root",
        type=Path,
        default=(
            cache_root
            / "qairt-2.42.0.251225-android-min/qairt/2.42.0.251225"
        ),
    )
    parser.add_argument(
        "--feature-fixture",
        type=Path,
        help="Optional exact 1x80x3000 little-endian UINT16 feature tensor (480000 bytes).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    require_external_output(output)
    if output.exists():
        raise RuntimeError(f"refusing existing output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    model_archive = args.model_archive.expanduser().resolve()
    voice_archive = args.voice_ai_archive.expanduser().resolve()
    ort_aar = args.ort_aar.expanduser().resolve()
    qairt_root = args.qairt_root.expanduser().resolve()
    verify_file(model_archive, MODEL_ARCHIVE)
    verify_file(voice_archive, VOICE_ARCHIVE)
    verify_file(ort_aar, ORT_AAR)

    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        asset_root = temp_root / "assets/whisper_probe"
        model_root = asset_root / "models"
        voice_root = asset_root / "voice_ai"
        native_root = temp_root / "jniLibs/arm64-v8a"
        aar_root = temp_root / "aar"

        asset_records: list[dict[str, Any]] = []
        for name, spec in MODEL_MEMBERS.items():
            destination = model_root / name
            extract_verified_member(
                model_archive,
                f"{MODEL_ARCHIVE_ROOT}/{name}",
                destination,
                spec,
            )
            asset_records.append(
                file_record(destination, temp_root / "assets", f"model_{name}")
            )

        for name, spec in VOICE_MEMBERS.items():
            destination = voice_root / name
            extract_verified_member(
                voice_archive,
                f"{VOICE_ARCHIVE_ROOT}/{name}",
                destination,
                spec,
            )
            asset_records.append(
                file_record(destination, temp_root / "assets", f"voice_ai_{name}")
            )

        feature_record = None
        if args.feature_fixture:
            feature_source = args.feature_fixture.expanduser().resolve()
            verify_file(feature_source, FEATURE_FIXTURE)
            feature_destination = model_root / "input_features.u16le"
            copy_verified(feature_source, feature_destination, FEATURE_FIXTURE)
            feature_record = file_record(
                feature_destination,
                temp_root / "assets",
                "prequantized_feature_fixture",
            )
            asset_records.append(feature_record)

        native_records: list[dict[str, Any]] = []
        for name, spec in QAIRT_FILES.items():
            source = qairt_root / spec["relative"]
            destination = native_root / name
            copy_verified(source, destination, spec)
            destination.chmod(0o755)
            record = file_record(destination, temp_root / "jniLibs", f"qairt_{name}")
            record["elf"] = spec["elf"]
            native_records.append(record)

        # These two are merged from the exact AAR by AGP, so record their
        # expected installed identities even though they are not duplicated.
        for name, spec in ORT_NATIVE_FILES.items():
            native_records.append(
                {
                    "role": f"ort_{name}",
                    "path": f"arm64-v8a/{name}",
                    "bytes": spec["bytes"],
                    "sha256": spec["sha256"],
                    "elf": spec["elf"],
                    "source": "ort_aar",
                }
            )

        copy_verified(ort_aar, aar_root / ORT_AAR_NAME, ORT_AAR)

        manifest = build_manifest(
            asset_records,
            native_records,
            feature_record,
        )
        manifest_path = asset_root / "payload_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_manifest = temp_root / "probe_payload_manifest.json"
        shutil.copy2(manifest_path, output_manifest)
        temp_root.rename(output)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    result = {
        "payload_dir": str(output),
        "gradle_env": f"WHISPER_PROBE_PAYLOAD_DIR={output}",
        "manifest": str(output / "probe_payload_manifest.json"),
        "real_audio_transcription": False,
        "device_mutated": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
