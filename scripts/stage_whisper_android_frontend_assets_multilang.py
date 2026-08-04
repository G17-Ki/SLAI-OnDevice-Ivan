#!/usr/bin/env python3
"""Stage immutable five-language assets for Whisper frontend profile v2.

This additive, local-only stager deliberately leaves the historical v1
stager, manifest, paths, and trust roots untouched. It performs no model
inference, network access, Android installation, or device communication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = REPO_ROOT / "artifacts/whisper_native_frontend_audit"
ASSET_PREFIX = Path("whisper_frontend/v2")
GATE_ID = "whisper_qai_hf_bundled_wav_frontend_multilang_v2"
FRONTEND_PROFILE = "v2"
FIXTURE_LANGUAGES = ("en", "zh", "ru", "ko", "ja")

ASSETS: tuple[dict[str, Any], ...] = (
    {
        "path": "whisper_frontend/v2/fixtures/question_en.wav",
        "source": "scripts/phone_acceptance_fixtures/question_en.wav",
        "bytes": 105_682,
        "sha256": "b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69",
        "role": "English mono PCM16 16 kHz bundled frontend fixture",
        "language": "en",
        "pcm_frames": 52_819,
    },
    {
        "path": "whisper_frontend/v2/fixtures/question_zh.wav",
        "source": "scripts/phone_acceptance_fixtures/question_zh.wav",
        "bytes": 83_478,
        "sha256": "bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63",
        "role": "Chinese mono PCM16 16 kHz bundled frontend fixture",
        "language": "zh",
        "pcm_frames": 41_717,
    },
    {
        "path": "whisper_frontend/v2/fixtures/question_ru.wav",
        "source": "scripts/phone_acceptance_fixtures/question_ru.wav",
        "bytes": 48_674,
        "sha256": "3545b8a20e3f320174a4e4dee57ff77268f30f79526eef91c86c778e3c180683",
        "role": "Russian mono PCM16 16 kHz bundled frontend fixture",
        "language": "ru",
        "pcm_frames": 24_315,
    },
    {
        "path": "whisper_frontend/v2/fixtures/question_ko.wav",
        "source": "scripts/phone_acceptance_fixtures/question_ko.wav",
        "bytes": 66_908,
        "sha256": "0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de",
        "role": "Korean mono PCM16 16 kHz bundled frontend fixture",
        "language": "ko",
        "pcm_frames": 33_432,
    },
    {
        "path": "whisper_frontend/v2/fixtures/question_ja.wav",
        "source": "scripts/phone_acceptance_fixtures/question_ja.wav",
        "bytes": 79_464,
        "sha256": "0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761",
        "role": "Japanese mono PCM16 16 kHz bundled frontend fixture",
        "language": "ja",
        "pcm_frames": 39_710,
    },
    {
        "path": "whisper_frontend/v2/tables/hann.f32le.bin",
        "source": "inputs/hann.f32le.bin",
        "bytes": 1_600,
        "sha256": "c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e",
        "role": "pinned Torch periodic Hann [400] float32 little-endian",
    },
    {
        "path": "whisper_frontend/v2/tables/mel_filters.f32le.bin",
        "source": "inputs/mel_filters.f32le.bin",
        "bytes": 64_320,
        "sha256": "d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9",
        "role": "pinned Hugging Face mel matrix [201,80] float32 little-endian",
    },
    {
        "path": "whisper_frontend/v2/goldens/en.features.u16le.bin",
        "source": "reference/en/features.u16le.bin",
        "bytes": 480_000,
        "sha256": "29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1",
        "role": "QAI float64-division English UINT16 [1,80,3000] golden",
        "language": "en",
    },
    {
        "path": "whisper_frontend/v2/goldens/zh.features.u16le.bin",
        "source": "reference/zh/features.u16le.bin",
        "bytes": 480_000,
        "sha256": "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263",
        "role": "QAI float64-division Chinese UINT16 [1,80,3000] golden",
        "language": "zh",
    },
    {
        "path": "whisper_frontend/v2/goldens/ru.features.u16le.bin",
        "source": "reference/ru/features.u16le.bin",
        "bytes": 480_000,
        "sha256": "0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4",
        "role": "QAI float64-division Russian UINT16 [1,80,3000] golden",
        "language": "ru",
    },
    {
        "path": "whisper_frontend/v2/goldens/ko.features.u16le.bin",
        "source": "reference/ko/features.u16le.bin",
        "bytes": 480_000,
        "sha256": "808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5",
        "role": "QAI float64-division Korean UINT16 [1,80,3000] golden",
        "language": "ko",
    },
    {
        "path": "whisper_frontend/v2/goldens/ja.features.u16le.bin",
        "source": "reference/ja/features.u16le.bin",
        "bytes": 480_000,
        "sha256": "8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80",
        "role": "corrected QAI float64-division Japanese UINT16 [1,80,3000] golden",
        "language": "ja",
    },
)


class StageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise StageError(f"missing pinned frontend input: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise StageError(
            f"frontend input byte count mismatch for {path}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise StageError(
            f"frontend input SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def source_path(record: dict[str, Any], audit_dir: Path) -> Path:
    source = Path(record["source"])
    if source.parts[0] == "scripts":
        return REPO_ROOT / source
    return audit_dir / source


def canonical_manifest() -> dict[str, Any]:
    records = [
        {key: value for key, value in source_record.items() if key != "source"}
        for source_record in ASSETS
    ]
    return {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "frontend_profile": FRONTEND_PROFILE,
        "fixture_languages": list(FIXTURE_LANGUAGES),
        "target": {
            "package": "ai.liquid.lfmdemo.whisperprobe",
            "abi": "arm64-v8a",
            "frontend_output": {
                "dtype": "uint16",
                "shape": [1, 80, 3000],
                "layout": "mel_major_frame_minor",
                "byte_order": "little_endian",
            },
        },
        "capabilities": {
            "bundled_wav_frontend_verification": True,
            "microphone_input": False,
            "real_audio_transcription": False,
        },
        "algorithm": {
            "pcm_normalization": "float32(int16 / 32768.0)",
            "padded_samples": 480_000,
            "fft_size": 400,
            "hop_size": 160,
            "frame_start": "frame*160-200",
            "padding": "torch_reflect_-1_to_1",
            "fft": "persistent_kissfft_400_real",
            "power": "hypotf(real,imag)^2",
            "mel_accumulation": "explicit_fmaf",
            "postprocess": "clamp_1e-10_log10_global_max_minus_8_(x+4)/4",
            "quantization": {
                "scale": 4.677007018472068e-05,
                "zero_point": 32072,
                "feature_cast_before_division": "float32_to_float64",
                "rounding": "ties_to_even",
            },
        },
        "verification_policy": {
            "strict_asset_sha256": True,
            "report_exact_output_sha256": True,
            "max_uint16_delta": 1,
            "max_differing_uint16_values": 240,
            "max_differing_fraction": 0.001,
            "target_acceptance_promoted": False,
        },
        "assets": records,
        "claim_boundary": (
            "Passing this five-language fixture gate verifies only the bundled-WAV "
            "frontend on the executing ABI. It does not enable microphone input or "
            "claim a semantic transcription."
        ),
    }


def manifest_bytes() -> bytes:
    return (json.dumps(canonical_manifest(), indent=2, sort_keys=True) + "\n").encode()


def verify_bundle(output: Path) -> dict[str, Any]:
    asset_root = output / "assets"
    manifest_path = asset_root / ASSET_PREFIX / "manifest.json"
    if not manifest_path.is_file():
        raise StageError(f"missing staged frontend manifest: {manifest_path}")
    expected_manifest = manifest_bytes()
    actual_manifest = manifest_path.read_bytes()
    if actual_manifest != expected_manifest:
        raise StageError("staged frontend manifest is not canonical")
    for record in ASSETS:
        require_file(asset_root / record["path"], record["bytes"], record["sha256"])
    return {
        "frontend_profile": FRONTEND_PROFILE,
        "fixture_languages": list(FIXTURE_LANGUAGES),
        "gate_id": GATE_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(actual_manifest).hexdigest(),
        "assets": len(ASSETS),
        "bytes": sum(record["bytes"] for record in ASSETS),
    }


def stage(output: Path, audit_dir: Path = DEFAULT_AUDIT_DIR) -> dict[str, Any]:
    output = output.resolve()
    repo = REPO_ROOT.resolve()
    if output == repo or repo in output.parents:
        raise StageError("frontend staging output must stay outside the repository")
    if output == Path(output.anchor) or len(output.parts) < 4:
        raise StageError(f"refusing broad frontend staging output: {output}")
    if output.exists():
        raise StageError(f"frontend staging output already exists: {output}")

    for record in ASSETS:
        require_file(source_path(record, audit_dir), record["bytes"], record["sha256"])

    asset_root = output / "assets"
    try:
        for record in ASSETS:
            destination = asset_root / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(record, audit_dir), destination)
        manifest_path = asset_root / ASSET_PREFIX / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_bytes())
        return verify_bundle(output)
    except BaseException:
        if output.is_dir():
            shutil.rmtree(output)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = stage(arguments.output, arguments.audit_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
