#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import zipfile

import verify_nubia_whisper_ort_qnn_gate as gate


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def fake_elf(machine: int, marker: bytes, *, elf_class: int = 2) -> bytes:
    value = bytearray(96 + len(marker))
    value[:4] = b"\x7fELF"
    value[4] = elf_class
    value[5] = 1
    value[18:20] = machine.to_bytes(2, "little")
    value[32 : 32 + len(marker)] = marker
    return bytes(value)


def zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, value in entries:
            bundle.writestr(name, value)
    return output.getvalue()


class OrtQnnGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.layer_count = 2
        self.metadata = self.make_metadata()
        self.metadata_bytes = json.dumps(
            self.metadata, separators=(",", ":"), sort_keys=True
        ).encode()
        self.contract = gate.extract_tensor_contract(self.metadata)
        self.contract_sha = digest(gate.canonical_json_bytes(self.contract))
        self.encoder_onnx = self.wrapper_bytes("encoder_qairt_context.bin")
        self.decoder_onnx = self.wrapper_bytes("decoder_qairt_context.bin")
        self.encoder_context = b"encoder-qairt-context"
        self.decoder_context = b"decoder-qairt-context"
        self.model_archive = self.root / "model.zip"
        self.write_model_archive()
        self.ort_aar = self.root / "onnxruntime-qnn.aar"
        self.write_ort_aar()
        self.voice_archive = self.root / "voice-ai.zip"
        self.write_voice_archive()
        self.runtime_files = self.make_runtime_files()
        self.spec = self.make_spec()
        self.runtime_manifest = self.make_runtime_manifest()
        self.runner_manifest = self.make_runner_manifest()
        self.app_uid_evidence = self.make_app_uid_evidence()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def descriptor(dtype: str, shape: list[int], *, zero_point: int | None = None) -> dict:
        value: dict = {"dtype": dtype, "shape": shape}
        if zero_point is not None:
            value["quantization_parameters"] = {
                "scale": 0.5,
                "zero_point": zero_point,
            }
        return value

    def make_metadata(self) -> dict:
        encoder_outputs = {}
        decoder_inputs = {
            "attention_mask": self.descriptor(
                "uint16", [1, 1, 1, 200], zero_point=65535
            ),
            "input_ids": self.descriptor("int32", [1, 1]),
            "position_ids": self.descriptor("int32", [1]),
        }
        decoder_outputs = {
            "logits": self.descriptor("uint16", [1, 51865, 1, 1], zero_point=10)
        }
        decoder_inputs["attention_mask"]["quantization_parameters"]["scale"] = (
            0.0015259021893143654
        )
        for layer in range(self.layer_count):
            encoder_outputs[f"k_cache_cross_{layer}"] = self.descriptor(
                "uint8", [2, 1, 1, 1], zero_point=128
            )
            encoder_outputs[f"v_cache_cross_{layer}"] = self.descriptor(
                "uint8", [2, 1, 1, 1], zero_point=128
            )
            decoder_inputs[f"k_cache_cross_{layer}"] = copy.deepcopy(
                encoder_outputs[f"k_cache_cross_{layer}"]
            )
            decoder_inputs[f"v_cache_cross_{layer}"] = copy.deepcopy(
                encoder_outputs[f"v_cache_cross_{layer}"]
            )
            decoder_inputs[f"k_cache_self_{layer}_in"] = self.descriptor(
                "uint8", [2, 1, 1, 199], zero_point=128
            )
            decoder_inputs[f"v_cache_self_{layer}_in"] = self.descriptor(
                "uint8", [2, 1, 199, 1], zero_point=128
            )
            decoder_outputs[f"k_cache_self_{layer}_out"] = copy.deepcopy(
                decoder_inputs[f"k_cache_self_{layer}_in"]
            )
            decoder_outputs[f"v_cache_self_{layer}_out"] = copy.deepcopy(
                decoder_inputs[f"v_cache_self_{layer}_in"]
            )
        return {
            "model_id": "whisper_small_quantized",
            "model_name": "Whisper-Small-Quantized",
            "runtime": "precompiled_qnn_onnx",
            "precision": "w8a16",
            "tool_versions": {
                "qairt": "2.42.0.test-build",
                "onnx_runtime": "1.25.0",
            },
            "chipset_attributes": {
                "aliases": ["qualcomm-snapdragon-8-elite-gen5", "sm8850"],
                "htp_version": 81,
            },
            "model_files": {
                "encoder.onnx": {
                    "inputs": {
                        "input_features": self.descriptor(
                            "uint16", [1, 80, 3000], zero_point=32072
                        )
                    },
                    "outputs": encoder_outputs,
                },
                "decoder.onnx": {
                    "inputs": decoder_inputs,
                    "outputs": decoder_outputs,
                },
            },
        }

    @staticmethod
    def wrapper_bytes(context_filename: str) -> bytes:
        return (
            b"protobuf "
            b"com.microsoft EPContext QNNContext embed_mode ep_cache_context "
            + f"./{context_filename}".encode()
            + b" source"
        )

    def write_model_archive(self) -> None:
        root = "model"
        self.model_entries = [
            (f"{root}/", b""),
            (f"{root}/encoder.onnx", self.encoder_onnx),
            (f"{root}/encoder_qairt_context.bin", self.encoder_context),
            (f"{root}/metadata.json", self.metadata_bytes),
            (f"{root}/decoder.onnx", self.decoder_onnx),
            (f"{root}/decoder_qairt_context.bin", self.decoder_context),
        ]
        self.model_archive.write_bytes(zip_bytes(self.model_entries))

    def write_ort_aar(self) -> None:
        class_entries = [
            ("ai/onnxruntime/OrtEnvironment.class", b"createSession"),
            ("ai/onnxruntime/OnnxTensor.class", b"createTensor"),
            (
                "ai/onnxruntime/OrtSession$SessionOptions.class",
                b"addQnn addConfigEntry",
            ),
        ]
        classes = zip_bytes(class_entries)
        native = fake_elf(
            gate.ELF_MACHINE["aarch64"],
            b"QNNExecutionProvider libQnnHtp.so 1.25.0",
        )
        jni = fake_elf(gate.ELF_MACHINE["aarch64"], b"jni")
        self.ort_entries = [
            ("META-INF/", b""),
            ("jni/", b""),
            ("jni/arm64-v8a/", b""),
            ("classes.jar", classes),
            ("jni/arm64-v8a/libonnxruntime.so", native),
            ("jni/arm64-v8a/libonnxruntime4j_jni.so", jni),
        ]
        self.ort_aar.write_bytes(zip_bytes(self.ort_entries))

    def write_voice_archive(self) -> None:
        self.voice_config = {
            "name": "whisper-small-quantized",
            "version": "1.0.0",
            "model_type": "whisper",
            "capabilities": {"language_detection": True},
            "parameters": {"sample_rates": [16000]},
            "assets": {"vocab_path": "vocab.bin"},
        }
        self.voice_config_bytes = json.dumps(
            self.voice_config, separators=(",", ":"), sort_keys=True
        ).encode()
        self.vocab_bytes = b"pinned-voice-ai-vocabulary"
        self.voice_entries = [
            ("voice/", b""),
            ("voice/config.json", self.voice_config_bytes),
            ("voice/vocab.bin", self.vocab_bytes),
        ]
        self.voice_archive.write_bytes(zip_bytes(self.voice_entries))

    def make_runtime_files(self) -> dict[str, Path]:
        root = self.root / "runtime"
        root.mkdir()
        values = {
            "libQnnSystem.so": fake_elf(
                gate.ELF_MACHINE["aarch64"], b"2.42.0.test-build"
            ),
            "libQnnHtp.so": fake_elf(
                gate.ELF_MACHINE["aarch64"], b"2.42.0.test-build"
            ),
            "libQnnHtpV81Stub.so": fake_elf(
                gate.ELF_MACHINE["aarch64"], b"2.42.0.test-build"
            ),
            "libQnnHtpV81Skel.so": fake_elf(
                gate.ELF_MACHINE["hexagon"],
                b"2.42.0.test-build",
                elf_class=1,
            ),
        }
        result = {}
        for name, value in values.items():
            path = root / name
            path.write_bytes(value)
            result[name] = path
        return result

    @staticmethod
    def release_spec(path: Path, entries: list[tuple[str, bytes]]) -> dict:
        members = []
        for name, value in entries:
            is_directory = name.endswith("/")
            item = {
                "bytes": len(value),
                "name": name,
                "type": "directory" if is_directory else "file",
            }
            if not is_directory:
                item["sha256"] = digest(value)
            members.append(item)
        return {
            "archive_filename": path.name,
            "archive_bytes": path.stat().st_size,
            "archive_sha256": gate.sha256_file(path),
            "members": members,
        }

    def make_spec(self) -> dict:
        tensor_names = {
            "encoder": {
                "inputs": ["input_features"],
                "outputs": ["k_cache_cross_{layer}", "v_cache_cross_{layer}"],
            },
            "decoder": {
                "inputs": [
                    "attention_mask",
                    "input_ids",
                    "position_ids",
                    "k_cache_cross_{layer}",
                    "v_cache_cross_{layer}",
                    "k_cache_self_{layer}_in",
                    "v_cache_self_{layer}_in",
                ],
                "outputs": [
                    "logits",
                    "k_cache_self_{layer}_out",
                    "v_cache_self_{layer}_out",
                ],
            },
        }
        runtime_requirements = []
        for name, path in self.runtime_files.items():
            runtime_requirements.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": gate.sha256_file(path),
                    "elf_machine": (
                        "hexagon" if name.endswith("Skel.so") else "aarch64"
                    ),
                    "required_ascii_markers": ["2.42.0.test-build"],
                }
            )
        return {
            "schema_version": 1,
            "gate_id": "test_ort_qnn_gate",
            "model_release": self.release_spec(
                self.model_archive, self.model_entries
            ),
            "metadata": {
                "sha256": digest(self.metadata_bytes),
                "tensor_contract_sha256": self.contract_sha,
                "layer_count": self.layer_count,
                "tensor_names": tensor_names,
                "required_values": [
                    {"path": "model_id", "equals": "whisper_small_quantized"},
                    {"path": "runtime", "equals": "precompiled_qnn_onnx"},
                    {"path": "precision", "equals": "w8a16"},
                    {"path": "tool_versions.qairt", "equals": "2.42.0.test-build"},
                    {"path": "tool_versions.onnx_runtime", "equals": "1.25.0"},
                    {"path": "chipset_attributes.htp_version", "equals": 81},
                ],
            },
            "ep_context": {
                "wrappers": [
                    {
                        "onnx_filename": "encoder.onnx",
                        "onnx_sha256": digest(self.encoder_onnx),
                        "context_filename": "encoder_qairt_context.bin",
                        "context_sha256": digest(self.encoder_context),
                        "external_reference": "./encoder_qairt_context.bin",
                    },
                    {
                        "onnx_filename": "decoder.onnx",
                        "onnx_sha256": digest(self.decoder_onnx),
                        "context_filename": "decoder_qairt_context.bin",
                        "context_sha256": digest(self.decoder_context),
                        "external_reference": "./decoder_qairt_context.bin",
                    },
                ]
            },
            "ort": {
                "version": "1.25.0",
                "release": self.release_spec(self.ort_aar, self.ort_entries),
                "native_library_sha256": digest(
                    dict(self.ort_entries)["jni/arm64-v8a/libonnxruntime.so"]
                ),
                "required_native_markers": [
                    "QNNExecutionProvider",
                    "libQnnHtp.so",
                    "1.25.0",
                ],
                "required_java_classes": [
                    "ai/onnxruntime/OrtEnvironment.class",
                    "ai/onnxruntime/OnnxTensor.class",
                    "ai/onnxruntime/OrtSession$SessionOptions.class",
                ],
                "required_java_markers": ["addQnn", "addConfigEntry"],
            },
            "voice_assets": {
                "release": self.release_spec(
                    self.voice_archive, self.voice_entries
                ),
                "config_sha256": digest(self.voice_config_bytes),
                "vocab_sha256": digest(self.vocab_bytes),
                "config_required_values": [
                    {"path": "name", "equals": "whisper-small-quantized"},
                    {"path": "version", "equals": "1.0.0"},
                    {"path": "model_type", "equals": "whisper"},
                    {"path": "capabilities.language_detection", "equals": True},
                    {"path": "parameters.sample_rates", "equals": [16000]},
                    {"path": "assets.vocab_path", "equals": "vocab.bin"},
                ],
            },
            "runtime": {
                "qairt_version": "2.42.0.251225",
                "exact_qairt_build": "2.42.0.test-build",
                "source_archive": {
                    "filename": "qairt.zip",
                    "bytes": 123,
                    "sha256": "a" * 64,
                },
                "target": {
                    "platform": "android",
                    "host_abi": "arm64-v8a",
                    "htp_arch": "v81",
                    "soc": "SM8850",
                },
                "required_libraries": runtime_requirements,
            },
            "runner": {
                "target": {
                    "platform": "android",
                    "abi": "arm64-v8a",
                    "htp_arch": "v81",
                    "package": "ai.liquid.lfmdemo.runtime",
                },
                "deployment": {
                    "model_storage": "app_private",
                    "runtime_storage": "app_private",
                    "requires_host_transport": False,
                },
                "provider": {
                    "name": "QNNExecutionProvider",
                    "backend": "libQnnHtp.so",
                    "context_mode": "external",
                },
                "frontend": {
                    "sample_rate_hz": 16000,
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 80,
                    "feature_frames": 3000,
                },
                "decode_loop": {
                    "max_decode_tokens": 200,
                    "initial_self_kv": "release_quantized_zero_point",
                    "attention_mask_unfilled_real": -100,
                    "attention_mask_unfilled_uint16": 0,
                    "attention_mask_filled_uint16": 65535,
                    "token_selection": "argmax",
                    "eos_token_id": 50257,
                    "task": "transcribe",
                    "native_language_mode": True,
                },
                "required_artifact_roles": [
                    "android_runner",
                    "feature_extractor",
                    "vocab_decoder",
                ],
            },
            "app_uid": {
                "target": {
                    "device_serial": "test-device",
                    "soc": "SM8850",
                    "abi": "arm64-v8a",
                    "htp_arch": "v81",
                    "package": "ai.liquid.lfmdemo.runtime",
                },
                "required_checks": {
                    "run_as_package_uid": True,
                    "host_transport_required": False,
                    "assets_loaded_from_app_private": True,
                    "qnn_backend_loaded": True,
                    "htp_v81_loaded": True,
                    "encoder_session_created": True,
                    "decoder_session_created": True,
                    "encoder_execute_passed": True,
                    "decoder_execute_passed": True,
                    "vocab_decode_passed": True,
                    "offline_after_reboot": True,
                },
            },
        }

    @staticmethod
    def artifact_entry(path: Path, key: str, value: str) -> dict:
        return {
            key: value,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": gate.sha256_file(path),
        }

    def make_runtime_manifest(self, *, qairt_build: str | None = None) -> Path:
        root = self.root / "runtime"
        manifest = {
            "schema_version": 1,
            "root": str(root),
            "target": copy.deepcopy(self.spec["runtime"]["target"]),
            "release": {
                "qairt_version": self.spec["runtime"]["qairt_version"],
                "qairt_build": qairt_build or self.spec["runtime"]["exact_qairt_build"],
                "source_archive_filename": "qairt.zip",
                "source_archive_bytes": 123,
                "source_archive_sha256": "a" * 64,
            },
            "files": [
                self.artifact_entry(path, "name", name)
                for name, path in self.runtime_files.items()
            ],
        }
        path = self.root / "runtime.json"
        write_json(path, manifest)
        return path

    def actual_tensor_names(self) -> dict:
        return gate.contract_tensor_names(self.contract)

    def make_runner_manifest(self, *, decode_loop: dict | None = None) -> Path:
        root = self.root / "runner"
        root.mkdir(exist_ok=True)
        artifacts = []
        for role in self.spec["runner"]["required_artifact_roles"]:
            path = root / f"{role}.bin"
            path.write_bytes(role.encode())
            artifacts.append(self.artifact_entry(path, "role", role))
        manifest = {
            "schema_version": 1,
            "root": str(root),
            "target": copy.deepcopy(self.spec["runner"]["target"]),
            "deployment": copy.deepcopy(self.spec["runner"]["deployment"]),
            "provider": copy.deepcopy(self.spec["runner"]["provider"]),
            "frontend": copy.deepcopy(self.spec["runner"]["frontend"]),
            "decode_loop": copy.deepcopy(
                decode_loop or self.spec["runner"]["decode_loop"]
            ),
            "bindings": {
                "model_archive_sha256": self.spec["model_release"]["archive_sha256"],
                "model_metadata_sha256": self.spec["metadata"]["sha256"],
                "tensor_contract_sha256": self.contract_sha,
                "ort_aar_sha256": self.spec["ort"]["release"]["archive_sha256"],
                "ort_native_library_sha256": self.spec["ort"]["native_library_sha256"],
                "voice_ai_archive_sha256": self.spec["voice_assets"]["release"]["archive_sha256"],
                "voice_config_sha256": self.spec["voice_assets"]["config_sha256"],
                "voice_vocab_sha256": self.spec["voice_assets"]["vocab_sha256"],
                "runtime_inventory_sha256": gate.sha256_file(self.runtime_manifest),
            },
            "tensor_contract": {
                "full_contract_sha256": self.contract_sha,
                "quantization_source": "release_metadata",
                "tensor_names": self.actual_tensor_names(),
            },
            "artifacts": artifacts,
        }
        path = self.root / "runner.json"
        write_json(path, manifest)
        return path

    def make_app_uid_evidence(self, *, checks: dict | None = None) -> Path:
        runner = json.loads(self.runner_manifest.read_text())
        artifact_hashes = {
            item["role"]: item["sha256"] for item in runner["artifacts"]
        }
        bindings = {
            "model_archive_sha256": self.spec["model_release"]["archive_sha256"],
            "model_metadata_sha256": self.spec["metadata"]["sha256"],
            "encoder_onnx_sha256": digest(self.encoder_onnx),
            "encoder_context_sha256": digest(self.encoder_context),
            "decoder_onnx_sha256": digest(self.decoder_onnx),
            "decoder_context_sha256": digest(self.decoder_context),
            "ort_aar_sha256": self.spec["ort"]["release"]["archive_sha256"],
            "ort_native_library_sha256": self.spec["ort"]["native_library_sha256"],
            "voice_config_sha256": self.spec["voice_assets"]["config_sha256"],
            "voice_vocab_sha256": self.spec["voice_assets"]["vocab_sha256"],
            "runtime_inventory_sha256": gate.sha256_file(self.runtime_manifest),
            "runner_manifest_sha256": gate.sha256_file(self.runner_manifest),
        }
        for role, value in artifact_hashes.items():
            bindings[f"runner_{role}_sha256"] = value
        evidence = {
            "schema_version": 1,
            "status": "pass",
            "target": copy.deepcopy(self.spec["app_uid"]["target"]),
            "package_uid": 10234,
            "process_uid": 10234,
            "qairt_build": self.spec["runtime"]["exact_qairt_build"],
            "bindings": bindings,
            "checks": copy.deepcopy(
                checks or self.spec["app_uid"]["required_checks"]
            ),
        }
        path = self.root / "app_uid.json"
        write_json(path, evidence)
        return path

    def evaluate(self, spec: dict | None = None, **overrides) -> dict:
        values = {
            "model_archive": self.model_archive,
            "ort_aar": self.ort_aar,
            "voice_ai_archive": self.voice_archive,
            "runtime_inventory": self.runtime_manifest,
            "runner_manifest": self.runner_manifest,
            "app_uid_evidence": self.app_uid_evidence,
        }
        values.update(overrides)
        return gate.evaluate_gate(spec or self.spec, **values)

    @staticmethod
    def blocker_codes(result: dict) -> set[str]:
        return {item["code"] for item in result["blockers"]}

    def test_complete_bound_route_is_ready(self) -> None:
        result = self.evaluate()
        self.assertTrue(result["ready"])
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["checks"]["ep_context"]["status"], "pass")
        self.assertEqual(result["checks"]["ort"]["status"], "pass")

    def test_missing_runtime_runner_and_app_uid_fail_closed(self) -> None:
        result = self.evaluate(
            runtime_inventory=None,
            runner_manifest=None,
            app_uid_evidence=None,
        )
        self.assertFalse(result["ready"])
        self.assertEqual(
            self.blocker_codes(result),
            {
                "runtime_inventory_missing",
                "runner_manifest_missing",
                "app_uid_evidence_missing",
            },
        )
        self.assertEqual(result["checks"]["model_release"]["status"], "pass")
        self.assertEqual(result["checks"]["voice_assets"]["status"], "pass")

    def test_model_archive_hash_mismatch_stops_semantic_checks(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["model_release"]["archive_sha256"] = "0" * 64
        result = self.evaluate(spec)
        self.assertIn(
            "model_release_archive_sha256_mismatch", self.blocker_codes(result)
        )
        self.assertEqual(result["checks"]["metadata"]["status"], "not_checked")
        self.assertEqual(result["checks"]["ep_context"]["status"], "not_checked")

    def test_external_context_reference_is_semantically_required(self) -> None:
        self.encoder_onnx = self.encoder_onnx.replace(
            b"./encoder_qairt_context.bin", b"encoder_qairt_context.bin/"
        )
        self.write_model_archive()
        spec = self.make_spec()
        spec["ep_context"]["wrappers"][0]["external_reference"] = (
            "./encoder_qairt_context.bin"
        )
        result = self.evaluate(spec)
        self.assertIn("ep_context_marker_mismatch", self.blocker_codes(result))

    def test_ort_native_library_must_include_qnn_provider(self) -> None:
        entries = []
        for name, value in self.ort_entries:
            if name.endswith("libonnxruntime.so"):
                value = fake_elf(
                    gate.ELF_MACHINE["aarch64"], b"CPUExecutionProvider 1.25.0"
                )
            entries.append((name, value))
        self.ort_entries = entries
        self.ort_aar.write_bytes(zip_bytes(entries))
        spec = self.make_spec()
        result = self.evaluate(spec)
        self.assertIn("ort_qnn_marker_missing", self.blocker_codes(result))

    def test_ort_native_library_must_be_arm64(self) -> None:
        entries = []
        for name, value in self.ort_entries:
            if name.endswith("libonnxruntime.so"):
                value = fake_elf(62, b"QNNExecutionProvider libQnnHtp.so 1.25.0")
            entries.append((name, value))
        self.ort_entries = entries
        self.ort_aar.write_bytes(zip_bytes(entries))
        spec = self.make_spec()
        result = self.evaluate(spec)
        self.assertIn("ort_semantic_validation_failed", self.blocker_codes(result))

    def test_voice_config_is_tokenizer_asset_but_must_match_model(self) -> None:
        self.voice_config["model_type"] = "not-whisper"
        self.voice_config_bytes = json.dumps(
            self.voice_config, separators=(",", ":"), sort_keys=True
        ).encode()
        self.voice_entries[1] = ("voice/config.json", self.voice_config_bytes)
        self.voice_archive.write_bytes(zip_bytes(self.voice_entries))
        spec = self.make_spec()
        result = self.evaluate(spec)
        self.assertIn("voice_config_value_mismatch", self.blocker_codes(result))

    def test_runtime_must_match_exact_context_compiler_build(self) -> None:
        self.runtime_manifest = self.make_runtime_manifest(
            qairt_build="2.42.0.different-build"
        )
        result = self.evaluate(app_uid_evidence=None)
        self.assertIn("runtime_inventory_invalid", self.blocker_codes(result))

    def test_runtime_libraries_are_spec_hash_pinned(self) -> None:
        manifest = json.loads(self.runtime_manifest.read_text())
        manifest["files"][0]["sha256"] = "f" * 64
        write_json(self.runtime_manifest, manifest)
        result = self.evaluate(runner_manifest=None, app_uid_evidence=None)
        self.assertIn("runtime_library_invalid", self.blocker_codes(result))

    def test_required_runtime_license_file_fails_closed_when_absent(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["runtime"]["required_supporting_files"] = [
            {"path": "NOTICE.txt", "bytes": 1, "sha256": digest(b"x")}
        ]
        result = self.evaluate(spec, runner_manifest=None, app_uid_evidence=None)
        self.assertIn("runtime_supporting_file_missing", self.blocker_codes(result))

    def test_runner_decode_quantization_contract_is_exact(self) -> None:
        wrong = copy.deepcopy(self.spec["runner"]["decode_loop"])
        wrong["attention_mask_unfilled_uint16"] = 65535
        self.runner_manifest = self.make_runner_manifest(decode_loop=wrong)
        result = self.evaluate(app_uid_evidence=None)
        self.assertIn("runner_manifest_invalid", self.blocker_codes(result))

    def test_specified_mask_values_must_derive_from_release_quantization(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["runner"]["decode_loop"]["attention_mask_unfilled_uint16"] = 1
        self.spec = spec
        self.runner_manifest = self.make_runner_manifest()
        result = self.evaluate(spec, app_uid_evidence=None)
        self.assertIn(
            "runner_quantized_decode_value_mismatch", self.blocker_codes(result)
        )

    def test_runner_tensor_names_must_match_metadata(self) -> None:
        manifest = json.loads(self.runner_manifest.read_text())
        manifest["tensor_contract"]["tensor_names"]["encoder"]["inputs"] = [
            "whisper_mel"
        ]
        write_json(self.runner_manifest, manifest)
        result = self.evaluate(app_uid_evidence=None)
        self.assertIn("runner_tensor_names_mismatch", self.blocker_codes(result))

    def test_failed_app_uid_htp_execution_is_blocked(self) -> None:
        checks = copy.deepcopy(self.spec["app_uid"]["required_checks"])
        checks["decoder_execute_passed"] = False
        self.app_uid_evidence = self.make_app_uid_evidence(checks=checks)
        result = self.evaluate()
        self.assertIn("app_uid_readiness_check_failed", self.blocker_codes(result))

    def test_app_uid_evidence_must_bind_exact_context(self) -> None:
        evidence = json.loads(self.app_uid_evidence.read_text())
        evidence["bindings"]["decoder_context_sha256"] = "0" * 64
        write_json(self.app_uid_evidence, evidence)
        result = self.evaluate()
        self.assertIn("app_uid_binding_mismatch", self.blocker_codes(result))

    def test_verifier_never_uses_zip_extraction_apis(self) -> None:
        with mock.patch.object(
            zipfile.ZipFile, "extract", side_effect=AssertionError("extract called")
        ), mock.patch.object(
            zipfile.ZipFile,
            "extractall",
            side_effect=AssertionError("extractall called"),
        ):
            result = self.evaluate()
        self.assertTrue(result["ready"])

    def test_report_is_deterministic_and_has_no_timestamp(self) -> None:
        first = self.evaluate()
        second = self.evaluate()
        self.assertEqual(first, second)
        payload = json.dumps(first, sort_keys=True)
        self.assertNotIn("timestamp", payload)
        self.assertNotIn("created_at", payload)


if __name__ == "__main__":
    unittest.main()
