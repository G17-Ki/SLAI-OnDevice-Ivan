#!/usr/bin/env python3
"""Stage the isolated four-stage Supertonic raw-ONNX ORT-QNN probe.

This is a bounded packaging step. It copies pinned source models and the
phone-proven ORT 1.25 / QAIRT 2.42 runtime, then materializes deterministic
fixed-shape inputs and host reference tensors. It performs no conversion,
training, network access, Android installation, or device communication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(
    "/private/tmp/lmf-tts-bundle-20260729-1145/supertonic3-int8/"
    "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
)
DEFAULT_FIXTURE_DIR = Path("/private/tmp/lmf-supertonic-chain-fixture-v1")
DEFAULT_RUNTIME_INVENTORY = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_ort_qnn_runtime_inventory.json"
)
DEFAULT_ORT_AAR = Path.home() / (
    ".cache/lmf-demo/whisper-small-v0.58.0/"
    "onnxruntime-android-qnn-1.25.0.aar"
)

GATE_ID = "supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1"
PACKAGE = "ai.liquid.lfmdemo.supertonicchainprobe"
QAIRT_BUILD = "2.42.0.251225135753_193295"
ORT_BYTES = 7_720_525
ORT_SHA256 = "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198"

MODEL_FILES: dict[str, dict[str, Any]] = {
    "duration_predictor": {
        "filename": "duration_predictor.int8.onnx",
        "bytes": 3_700_147,
        "sha256": "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
        "inputs": {
            "text_ids": {"dtype": "int64", "shape": [1, 32]},
            "style_dp": {"dtype": "float32", "shape": [1, 8, 16]},
            "text_mask": {"dtype": "float32", "shape": [1, 1, 32]},
        },
        "outputs": {"duration": {"dtype": "float32", "shape": [1]}},
    },
    "text_encoder": {
        "filename": "text_encoder.int8.onnx",
        "bytes": 36_416_150,
        "sha256": "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
        "inputs": {
            "text_ids": {"dtype": "int64", "shape": [1, 32]},
            "style_ttl": {"dtype": "float32", "shape": [1, 50, 256]},
            "text_mask": {"dtype": "float32", "shape": [1, 1, 32]},
        },
        "outputs": {"text_emb": {"dtype": "float32", "shape": [1, 256, 32]}},
    },
    "vector_estimator": {
        "filename": "vector_estimator.int8.onnx",
        "bytes": 78_400_833,
        "sha256": "20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd",
        "inputs": {
            "noisy_latent": {"dtype": "float32", "shape": [1, 144, 32]},
            "text_emb": {"dtype": "float32", "shape": [1, 256, 32]},
            "style_ttl": {"dtype": "float32", "shape": [1, 50, 256]},
            "latent_mask": {"dtype": "float32", "shape": [1, 1, 32]},
            "text_mask": {"dtype": "float32", "shape": [1, 1, 32]},
            "current_step": {"dtype": "float32", "shape": [1]},
            "total_step": {"dtype": "float32", "shape": [1]},
        },
        "outputs": {
            "denoised_latent": {"dtype": "float32", "shape": [1, 144, 32]}
        },
    },
    "vocoder": {
        "filename": "vocoder.int8.onnx",
        "bytes": 25_991_073,
        "sha256": "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152",
        "inputs": {"latent": {"dtype": "float32", "shape": [1, 144, 32]}},
        "outputs": {"wav_tts": {"dtype": "float32", "shape": [1, 98_304]}},
    },
}

SUPPORT_FILES: dict[str, dict[str, Any]] = {
    "voice.bin": {
        "bytes": 517_168,
        "sha256": "67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8",
    },
    "unicode_indexer.bin": {
        "bytes": 262_144,
        "sha256": "8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30",
    },
    "tts.json": {
        "bytes": 8_253,
        "sha256": "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    },
}

ARRAYS: tuple[dict[str, Any], ...] = (
    {
        "source": "text_ids_local",
        "path": "fixtures/text_ids.i64le",
        "dtype": "int64",
        "shape": [1, 32],
        "array_sha256": "2fd3fac41650673075085524f4b2e3a5aed442ddff672caeba14c9f7cb8d5890",
        "role": "duration_and_text_encoder_input",
    },
    {
        "source": "text_mask",
        "path": "fixtures/text_mask.f32le",
        "dtype": "float32",
        "shape": [1, 1, 32],
        "array_sha256": "4c821d782d02e504c72b4f24521dd786a78f6e8af8f14aa9a4e85ab96089f746",
        "role": "all_text_stage_mask",
    },
    {
        "source": "style_dp",
        "path": "fixtures/style_dp.f32le",
        "dtype": "float32",
        "shape": [1, 8, 16],
        "array_sha256": "1340f291942c45d0ff4226da9c77ecc1e8cf4301379a2f1249a1efbf77ea4e21",
        "role": "duration_style_input",
    },
    {
        "source": "style_ttl",
        "path": "fixtures/style_ttl.f32le",
        "dtype": "float32",
        "shape": [1, 50, 256],
        "array_sha256": "fb92cc6a6f470d5570c2cce6a4ace633ea63087f064a4f0e096f8d434da416f6",
        "role": "text_and_vector_style_input",
    },
    {
        "source": "latent_mask",
        "path": "fixtures/latent_mask.f32le",
        "dtype": "float32",
        "shape": [1, 1, 32],
        "array_sha256": "31e29f29a1e668fa9ef0460e44bc9e0e2f5279b4c646c1415b5ee8ea16116865",
        "role": "fixed_latent_mask_31_of_32",
    },
    {
        "source": "latent_initial",
        "path": "fixtures/latent_initial.f32le",
        "dtype": "float32",
        "shape": [1, 144, 32],
        "array_sha256": "b4943d187105cd5efaecacac8e967219215ba8875248b17550182f17deb09ede",
        "role": "seed_42_initial_noise",
    },
    {
        "source": "duration_raw_local",
        "path": "goldens/duration.f32le",
        "dtype": "float32",
        "shape": [1],
        "array_sha256": "9790218b17df1d99d3dfcf02a11a7aafc59f3e8346ee0f4e55dbb2fda7df5cbe",
        "role": "host_duration_reference",
    },
    {
        "source": "text_embedding_local",
        "path": "goldens/text_embedding.f32le",
        "dtype": "float32",
        "shape": [1, 256, 32],
        "array_sha256": "e059e5281d8085604a861857f6d9c63a4a42b8f09c573aa458991378fffc1fe7",
        "role": "host_text_encoder_reference",
    },
    {
        "source": "latent_step_0_masked_local",
        "path": "goldens/latent_step_0.f32le",
        "dtype": "float32",
        "shape": [1, 144, 32],
        "array_sha256": "846ad1358af4037379a2415f5917b444dd9fd882e7edc2935f7124a2412cd063",
        "role": "host_vector_step_0_reference",
    },
    {
        "source": "latent_step_1_masked_local",
        "path": "goldens/latent_step_1.f32le",
        "dtype": "float32",
        "shape": [1, 144, 32],
        "array_sha256": "99ea75c8103b36b69f4e5dc6491bb1838948caa1dbc8b0dd92d0bb3452daea24",
        "role": "host_vector_step_1_reference",
    },
    {
        "source": "latent_step_2_masked_local",
        "path": "goldens/latent_step_2.f32le",
        "dtype": "float32",
        "shape": [1, 144, 32],
        "array_sha256": "12910d6b341b84e014f487ab34bea2a0ae85afe7da6ff92fbf0ccb1c81268aef",
        "role": "host_vector_step_2_reference",
    },
    {
        "source": "latent_step_3_masked_local",
        "path": "goldens/latent_step_3.f32le",
        "dtype": "float32",
        "shape": [1, 144, 32],
        "array_sha256": "095b2d9cfeecd92c1da39954029c945b769e5cce67f889d67a93df63dc450c6a",
        "role": "host_vector_step_3_reference",
    },
    {
        "source": "waveform_padded_local",
        "path": "goldens/waveform_padded.f32le",
        "dtype": "float32",
        "shape": [1, 98_304],
        "array_sha256": "179b32995063b8f3ef19c6c406d90fd5ab96712d12cf80974d09cadd6cee7880",
        "role": "host_vocoder_reference_diagnostic_only",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
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


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing pinned input: {path}")
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"pinned input byte count mismatch: {path}")
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"pinned input SHA-256 mismatch: {path}")


def atomic_write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial output exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def copy_checked(
    source: Path,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    require_file(source, expected_bytes, expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial output exists: {temporary}")
    try:
        shutil.copyfile(source, temporary)
        require_file(temporary, expected_bytes, expected_sha256)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"bytes": expected_bytes, "sha256": expected_sha256}


def validate_fixture(fixture_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = fixture_dir / "fixture_manifest.json"
    fixture_path = fixture_dir / "fixture.npz"
    if not manifest_path.is_file() or not fixture_path.is_file():
        raise RuntimeError("missing deterministic Supertonic chain fixture")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fixture = {
        "text": "It is Paris, in France.",
        "language": "en",
        "processed_text": "<en>It is Paris, in France.</en>",
        "fixed_text_length": 32,
        "speaker": 6,
        "steps": 4,
        "seed": 42,
        "speed": 1.0,
        "sample_rate": 44_100,
        "samples_per_latent": 3_072,
        "latent_channels": 144,
        "real_latent_length": 31,
        "fixed_latent_length": 32,
        "padded_positions": 1,
        "wav_length": 93_826,
    }
    for field, expected in expected_fixture.items():
        if manifest.get("fixture", {}).get(field) != expected:
            raise RuntimeError(f"fixture contract mismatch: {field}")
    host = manifest.get("host", {})
    if (
        host.get("numpy") != "1.26.4"
        or host.get("onnxruntime") != "1.28.0"
        or host.get("local_threads") != 1
    ):
        raise RuntimeError("fixture host-runtime identity mismatch")
    if manifest.get("local_checks", {}).get("final_padding_is_zero") is not True:
        raise RuntimeError("fixture final padding is not zero")
    arrays: dict[str, np.ndarray] = {}
    with np.load(fixture_path, allow_pickle=False) as bundle:
        for record in ARRAYS:
            source = record["source"]
            if source not in bundle.files:
                raise RuntimeError(f"fixture array is missing: {source}")
            value = np.asarray(bundle[source])
            if str(value.dtype) != record["dtype"] or list(value.shape) != record["shape"]:
                raise RuntimeError(f"fixture array contract mismatch: {source}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"fixture array is non-finite: {source}")
            if sha256_array(value) != record["array_sha256"]:
                raise RuntimeError(f"fixture array identity mismatch: {source}")
            manifest_record = manifest.get("arrays", {}).get(source, {})
            if manifest_record.get("sha256") != record["array_sha256"]:
                raise RuntimeError(f"fixture manifest array mismatch: {source}")
            arrays[source] = value.copy()
    return manifest, arrays


def require_external_output(output: Path) -> None:
    output = output.resolve()
    repo = ROOT.resolve()
    if output == Path(output.anchor) or len(output.parts) < 4:
        raise RuntimeError(f"refusing broad staging output: {output}")
    if output == repo or repo in output.parents:
        raise RuntimeError("staging output must stay outside the repository")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"staging output must be absent or empty: {output}")


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    require_external_output(output)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.expanduser().resolve()
    fixture_dir = args.fixture_dir.expanduser().resolve()
    fixture_manifest, arrays = validate_fixture(fixture_dir)
    ort_aar = args.ort_aar.expanduser().resolve()
    require_file(ort_aar, ORT_BYTES, ORT_SHA256)

    asset_root = output / "assets/supertonic_chain_probe"
    asset_records: list[dict[str, Any]] = []
    try:
        for component, record in MODEL_FILES.items():
            source = model_dir / record["filename"]
            destination = asset_root / "models" / record["filename"]
            identity = copy_checked(
                source, destination, record["bytes"], record["sha256"]
            )
            asset_records.append(
                {
                    "path": f"supertonic_chain_probe/models/{record['filename']}",
                    "role": f"raw_onnx_{component}",
                    **identity,
                }
            )
        for filename, record in SUPPORT_FILES.items():
            identity = copy_checked(
                model_dir / filename,
                asset_root / "support" / filename,
                record["bytes"],
                record["sha256"],
            )
            asset_records.append(
                {
                    "path": f"supertonic_chain_probe/support/{filename}",
                    "role": f"source_support_{filename}",
                    **identity,
                }
            )
        array_records: list[dict[str, Any]] = []
        for record in ARRAYS:
            value = arrays[record["source"]]
            little = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
            relative = Path(record["path"])
            identity = atomic_write(asset_root / relative, little.tobytes(order="C"))
            output_record = {
                "path": f"supertonic_chain_probe/{relative.as_posix()}",
                "role": record["role"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "array_sha256": record["array_sha256"],
                **identity,
            }
            asset_records.append(output_record)
            array_records.append(output_record)

        inventory_path = args.runtime_inventory.expanduser().resolve()
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory.get("release", {}).get("qairt_build") != QAIRT_BUILD:
            raise RuntimeError("runtime inventory is not the pinned QAIRT 2.42 build")
        runtime_root = Path(inventory["root"])
        native_records: list[dict[str, Any]] = []
        for item in inventory["files"]:
            identity = copy_checked(
                runtime_root / item["path"],
                output / "jniLibs/arm64-v8a" / item["name"],
                item["bytes"],
                item["sha256"],
            )
            native_records.append(
                {
                    "path": f"arm64-v8a/{item['name']}",
                    "role": f"qairt_{item['name']}",
                    "elf": "hexagon-v81" if "Skel" in item["name"] else "aarch64",
                    **identity,
                }
            )

        aar_destination = output / "aar/onnxruntime-android-qnn-1.25.0.aar"
        copy_checked(ort_aar, aar_destination, ORT_BYTES, ORT_SHA256)
        stable_fixture_contract = {
            "fixture": {
                key: fixture_manifest["fixture"][key]
                for key in (
                    "text",
                    "language",
                    "processed_text",
                    "speaker",
                    "steps",
                    "seed",
                    "sample_rate",
                    "wav_length",
                    "real_latent_length",
                    "fixed_text_length",
                    "fixed_latent_length",
                )
            },
            "arrays": {
                record["source"]: record["array_sha256"] for record in ARRAYS
            },
            "host_reference_runtime": {
                "numpy": "1.26.4",
                "onnxruntime": "1.28.0",
                "threads": 1,
            },
        }
        fixture_contract_bytes = json.dumps(
            stable_fixture_contract, sort_keys=True, separators=(",", ":")
        ).encode()
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
                "execution_mode": "sequential",
                "inter_op_threads": 1,
                "intra_op_threads": 1,
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": True,
            },
            "models": {
                component: {
                    key: value
                    for key, value in record.items()
                    if key in {"filename", "bytes", "sha256", "inputs", "outputs"}
                }
                for component, record in MODEL_FILES.items()
            },
            "fixture": {
                **stable_fixture_contract["fixture"],
                "speed": 1.0,
                "duration_reference": float(
                    arrays["duration_raw_local"].reshape(-1)[0]
                ),
                "duration_absolute_tolerance": 0.1,
                "samples_per_latent": 3_072,
                "latent_channels": 144,
                "padded_positions": 1,
                "fixture_contract_sha256": hashlib.sha256(
                    fixture_contract_bytes
                ).hexdigest(),
                "array_records": sorted(array_records, key=lambda item: item["path"]),
            },
            "verification_policy": {
                "require_all_four_sessions": True,
                "require_chained_outputs": True,
                "require_finite_tensors": True,
                "require_exact_shapes": True,
                "require_zero_masked_padding": True,
                "host_reference_comparisons_recorded": True,
                "host_reference_tolerances_calibrated_on_phone": False,
                "host_reference_comparisons_promoted": False,
                "require_non_silent_waveform": True,
                "audio_quality_promoted": False,
            },
            "capabilities": {
                "duration_execution": True,
                "text_encoder_execution": True,
                "four_step_vector_execution": True,
                "vocoder_execution": True,
                "chained_pcm_tensor": True,
                "audio_file_output": False,
                "audio_playback": False,
                "audio_quality_acceptance": False,
                "production_integration": False,
            },
            "assets": sorted(asset_records, key=lambda item: item["path"]),
            "native_libraries": sorted(native_records, key=lambda item: item["path"]),
            "claim_boundary": (
                "Isolated raw-ONNX four-component compatibility scaffold. A pass may "
                "prove QNN session/execution, chaining, finite tensors, shape safety, "
                "and a non-silent PCM tensor only. Host-reference tolerances, audio "
                "quality, language identity, playback, latency promotion, unplugged "
                "durability, and production integration remain unclaimed."
            ),
        }
        manifest_path = asset_root / "payload_manifest.json"
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        manifest_identity = atomic_write(manifest_path, manifest_bytes)
        summary = {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "output": str(output),
            "payload_manifest": {
                "path": str(manifest_path),
                **manifest_identity,
            },
            "ort_aar": {
                "path": str(aar_destination),
                "bytes": ORT_BYTES,
                "sha256": ORT_SHA256,
            },
            "model_bytes": sum(record["bytes"] for record in MODEL_FILES.values()),
            "host_model_execution_during_staging": False,
            "device_mutated": False,
            "status": "staged",
        }
        atomic_write(
            output / "probe_payload_manifest.json",
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException:
        if output.is_dir():
            shutil.rmtree(output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
