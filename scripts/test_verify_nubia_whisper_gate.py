#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import zipfile

import verify_nubia_whisper_gate as gate


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def fake_elf(machine: int, marker: bytes, *, elf_class: int = 2) -> bytes:
    value = bytearray(64)
    value[:4] = b"\x7fELF"
    value[4] = elf_class
    value[5] = 1
    value[18:20] = machine.to_bytes(2, "little")
    value[20 : 20 + len(marker)] = marker
    return bytes(value)


class NubiaWhisperGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.layer_count = 2
        self.tensor_names = {
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
        self.metadata = self.make_metadata()
        self.metadata_bytes = json.dumps(
            self.metadata, sort_keys=True, separators=(",", ":")
        ).encode()
        self.encoder_bytes = b"encoder-context"
        self.decoder_bytes = b"decoder-context"
        self.archive_root = "release"
        self.archive_path = self.root / "release.zip"
        with zipfile.ZipFile(
            self.archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            bundle.writestr(f"{self.archive_root}/", b"")
            bundle.writestr(
                f"{self.archive_root}/metadata.json", self.metadata_bytes
            )
            bundle.writestr(f"{self.archive_root}/decoder.bin", self.decoder_bytes)
            bundle.writestr(f"{self.archive_root}/encoder.bin", self.encoder_bytes)
        contract = gate.extract_tensor_contract(self.metadata)
        self.contract_sha = digest(gate.canonical_json_bytes(contract))
        self.spec = self.make_spec()
        self.voice_ai_archive = self.make_voice_ai_archive()
        self.runtime_manifest = self.make_runtime_manifest()
        self.preprocessing_manifest = self.make_preprocessing_manifest()
        self.runner_manifest = self.make_runner_manifest()
        self.app_uid_evidence = self.make_app_uid_evidence()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def descriptor(dtype: str = "uint8") -> dict:
        return {"dtype": dtype, "shape": [1]}

    def make_metadata(self) -> dict:
        encoder_outputs = {}
        decoder_inputs = {
            "attention_mask": self.descriptor("uint16"),
            "input_ids": self.descriptor("int32"),
            "position_ids": self.descriptor("int32"),
        }
        decoder_outputs = {"logits": self.descriptor("uint16")}
        for layer in range(self.layer_count):
            encoder_outputs[f"k_cache_cross_{layer}"] = self.descriptor()
            encoder_outputs[f"v_cache_cross_{layer}"] = self.descriptor()
            decoder_inputs[f"k_cache_cross_{layer}"] = self.descriptor()
            decoder_inputs[f"v_cache_cross_{layer}"] = self.descriptor()
            decoder_inputs[f"k_cache_self_{layer}_in"] = self.descriptor()
            decoder_inputs[f"v_cache_self_{layer}_in"] = self.descriptor()
            decoder_outputs[f"k_cache_self_{layer}_out"] = self.descriptor()
            decoder_outputs[f"v_cache_self_{layer}_out"] = self.descriptor()
        return {
            "model_id": "whisper_small_quantized",
            "model_files": {
                "encoder.bin": {
                    "inputs": {"input_features": self.descriptor("uint16")},
                    "outputs": encoder_outputs,
                },
                "decoder.bin": {
                    "inputs": decoder_inputs,
                    "outputs": decoder_outputs,
                },
            },
        }

    def make_spec(self) -> dict:
        member_values = [
            (f"{self.archive_root}/", b"", "directory"),
            (
                f"{self.archive_root}/metadata.json",
                self.metadata_bytes,
                "file",
            ),
            (f"{self.archive_root}/decoder.bin", self.decoder_bytes, "file"),
            (f"{self.archive_root}/encoder.bin", self.encoder_bytes, "file"),
        ]
        members = []
        for name, value, member_type in member_values:
            item = {"bytes": len(value), "name": name, "type": member_type}
            if member_type == "file":
                item["sha256"] = digest(value)
            members.append(item)
        return {
            "schema_version": 1,
            "gate_id": "test_gate",
            "release": {
                "archive_filename": self.archive_path.name,
                "archive_root": self.archive_root,
                "archive_bytes": self.archive_path.stat().st_size,
                "archive_sha256": gate.sha256_file(self.archive_path),
                "members": members,
            },
            "metadata": {
                "sha256": digest(self.metadata_bytes),
                "required_values": [
                    {"path": "model_id", "equals": "whisper_small_quantized"}
                ],
                "layer_count": self.layer_count,
                "tensor_contract_sha256": self.contract_sha,
                "tensor_names": self.tensor_names,
            },
            "runtime": {
                "minimum_qairt": "2.45",
                "target": {
                    "platform": "android",
                    "host_abi": "arm64-v8a",
                    "htp_arch": "v81",
                    "soc": "SM8850",
                },
                "required_libraries": [
                    {"name": "libQnnHtp.so", "elf_machine": "aarch64"},
                    {"name": "libQnnHtpV81Skel.so", "elf_machine": "hexagon"},
                ],
            },
            "preprocessing": {
                "source_repository": "openai/whisper-small",
                "required_asset_roles": [
                    "tokenizer_model",
                    "feature_extractor_config",
                ],
                "required_language_codes": ["en", "ja", "ru", "zh"],
                "tokenizer": {
                    "vocab_size": 51865,
                    "multilingual": True,
                },
                "feature_extractor": {
                    "sample_rate_hz": 16000,
                    "n_mels": 80,
                    "feature_frames": 3000,
                },
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
                "required_artifact_roles": ["whisper_runner", "qnn_loader"],
                "decode_loop": {
                    "max_decode_tokens": 200,
                    "initial_self_kv": "zeros",
                    "attention_mask_unfilled_value": -100,
                    "token_selection": "argmax",
                    "termination_token": "eos",
                },
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
                    "qnn_system_loaded": True,
                    "encoder_execute_passed": True,
                    "host_transport_required": False,
                },
            },
        }

    def make_voice_ai_archive(self) -> Path:
        metadata = copy.deepcopy(self.metadata)
        metadata["runtime"] = "voice_ai"
        metadata_bytes = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode()
        config = {
            "model_type": "whisper",
            "assets": {
                "encoder_path": "encoder.bin",
                "decoder_path": "decoder.bin",
                "vocab_path": "vocab.bin",
            },
            "parameters": {"sample_rates": [16000]},
            "runtime": {"qnn_version": {"major": 2, "minor": 45, "patch": 0}},
        }
        config_bytes = json.dumps(config, sort_keys=True).encode()
        vocab_bytes = b"voice-ai-vocabulary"
        archive_root = "voice-ai-release"
        path = self.root / "voice-ai.zip"
        member_values = [
            (f"{archive_root}/", b"", "directory"),
            (f"{archive_root}/config.json", config_bytes, "file"),
            (f"{archive_root}/metadata.json", metadata_bytes, "file"),
            (f"{archive_root}/decoder.bin", self.decoder_bytes, "file"),
            (f"{archive_root}/encoder.bin", self.encoder_bytes, "file"),
            (f"{archive_root}/vocab.bin", vocab_bytes, "file"),
        ]
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, value, _ in member_values:
                bundle.writestr(name, value)
        members = []
        for name, value, member_type in member_values:
            item = {"bytes": len(value), "name": name, "type": member_type}
            if member_type == "file":
                item["sha256"] = digest(value)
            members.append(item)
        self.spec["voice_ai"] = {
            "release": {
                "archive_filename": path.name,
                "archive_root": archive_root,
                "archive_bytes": path.stat().st_size,
                "archive_sha256": gate.sha256_file(path),
                "members": members,
            },
            "metadata": {
                "sha256": digest(metadata_bytes),
                "required_values": [
                    {"path": "runtime", "equals": "voice_ai"}
                ],
                "layer_count": self.layer_count,
                "tensor_contract_sha256": self.contract_sha,
                "tensor_names": copy.deepcopy(self.tensor_names),
            },
            "config": {
                "sha256": digest(config_bytes),
                "required_values": [
                    {"path": "model_type", "equals": "whisper"},
                    {"path": "parameters.sample_rates", "equals": [16000]},
                    {"path": "runtime.qnn_version.major", "equals": 2},
                    {"path": "runtime.qnn_version.minor", "equals": 45},
                ],
            },
            "vocab_sha256": digest(vocab_bytes),
            "runner_compatibility": {
                "runtime_family": "qualcomm_voice_ai",
                "sdk_provenance": "qualcomm_voice_ai_asr_sdk_qpm",
            },
            "runner_requirement": "test VoiceAI runner",
        }
        return path

    def artifact_entry(self, path: Path, role_or_name: str, key: str) -> dict:
        return {
            key: role_or_name,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": gate.sha256_file(path),
        }

    def make_runtime_manifest(self, qairt_version: str = "2.45.0") -> Path:
        root = self.root / "runtime"
        root.mkdir(exist_ok=True)
        host = root / "libQnnHtp.so"
        dsp = root / "libQnnHtpV81Skel.so"
        host.write_bytes(fake_elf(gate.ELF_MACHINE["aarch64"], b"host"))
        dsp.write_bytes(
            fake_elf(gate.ELF_MACHINE["hexagon"], b"dsp", elf_class=1)
        )
        manifest = {
            "schema_version": 1,
            "root": str(root),
            "target": copy.deepcopy(self.spec["runtime"]["target"]),
            "release": {
                "qairt_version": qairt_version,
                "release_id": "test-qairt",
                "source_archive_sha256": "a" * 64,
            },
            "files": [
                self.artifact_entry(host, host.name, "name"),
                self.artifact_entry(dsp, dsp.name, "name"),
            ],
        }
        path = self.root / "runtime.json"
        write_json(path, manifest)
        return path

    def make_preprocessing_manifest(self, omit_role: str | None = None) -> Path:
        root = self.root / "preprocessing"
        root.mkdir(exist_ok=True)
        files = []
        for role in self.spec["preprocessing"]["required_asset_roles"]:
            path = root / f"{role}.json"
            path.write_bytes(role.encode())
            if role != omit_role:
                files.append(self.artifact_entry(path, role, "role"))
        manifest = {
            "schema_version": 1,
            "root": str(root),
            "source": {
                "repository": "openai/whisper-small",
                "revision": "b" * 40,
            },
            "release_metadata_sha256": self.spec["metadata"]["sha256"],
            "supported_language_codes": ["en", "ja", "ru", "zh"],
            "tokenizer": copy.deepcopy(self.spec["preprocessing"]["tokenizer"]),
            "feature_extractor": copy.deepcopy(
                self.spec["preprocessing"]["feature_extractor"]
            ),
            "files": files,
        }
        path = self.root / "preprocessing.json"
        write_json(path, manifest)
        return path

    def make_runner_manifest(
        self,
        tensor_names: dict | None = None,
        *,
        release_metadata_sha256: str | None = None,
        voice_ai_compatibility: dict | None = None,
    ) -> Path:
        root = self.root / "runner"
        root.mkdir(exist_ok=True)
        runner = root / "whisper-runner"
        loader = root / "libwhisper_qnn_loader.so"
        runner.write_bytes(fake_elf(gate.ELF_MACHINE["aarch64"], b"runner"))
        loader.write_bytes(fake_elf(gate.ELF_MACHINE["aarch64"], b"loader"))
        manifest = {
            "schema_version": 1,
            "root": str(root),
            "target": copy.deepcopy(self.spec["runner"]["target"]),
            "deployment": copy.deepcopy(self.spec["runner"]["deployment"]),
            "artifacts": [
                self.artifact_entry(runner, "whisper_runner", "role"),
                self.artifact_entry(loader, "qnn_loader", "role"),
            ],
            "tensor_contract": {
                "release_metadata_sha256": (
                    release_metadata_sha256 or self.spec["metadata"]["sha256"]
                ),
                "full_contract_sha256": self.contract_sha,
                "quantization_parameters_source": "release_metadata",
                "tensor_names": copy.deepcopy(tensor_names or self.tensor_names),
            },
            "decode_loop": copy.deepcopy(self.spec["runner"]["decode_loop"]),
        }
        if voice_ai_compatibility is not None:
            manifest["voice_ai_compatibility"] = voice_ai_compatibility
        path = self.root / "runner.json"
        write_json(path, manifest)
        return path

    def make_app_uid_evidence(self, checks: dict | None = None) -> Path:
        runner_manifest = json.loads(self.runner_manifest.read_text())
        artifacts = {
            item["role"]: item for item in runner_manifest["artifacts"]
        }
        evidence = {
            "schema_version": 1,
            "status": "pass",
            "target": copy.deepcopy(self.spec["app_uid"]["target"]),
            "package_uid": 10234,
            "process_uid": 10234,
            "qairt_version": "2.45.0",
            "bindings": {
                "release_archive_sha256": self.spec["release"]["archive_sha256"],
                "release_metadata_sha256": self.spec["metadata"]["sha256"],
                "runtime_inventory_sha256": gate.sha256_file(self.runtime_manifest),
                "preprocessing_manifest_sha256": gate.sha256_file(
                    self.preprocessing_manifest
                ),
                "runner_manifest_sha256": gate.sha256_file(self.runner_manifest),
                "runner_sha256": artifacts["whisper_runner"]["sha256"],
                "loader_sha256": artifacts["qnn_loader"]["sha256"],
            },
            "checks": copy.deepcopy(
                checks or self.spec["app_uid"]["required_checks"]
            ),
        }
        path = self.root / "app_uid.json"
        write_json(path, evidence)
        return path

    def evaluate(self, spec: dict | None = None, **overrides) -> dict:
        values = {
            "archive_path": self.archive_path,
            "runtime_inventory": self.runtime_manifest,
            "preprocessing_manifest": self.preprocessing_manifest,
            "runner_manifest": self.runner_manifest,
            "app_uid_evidence": self.app_uid_evidence,
        }
        values.update(overrides)
        return gate.evaluate_gate(spec or self.spec, **values)

    def blocker_codes(self, result: dict) -> set[str]:
        return {item["code"] for item in result["blockers"]}

    def test_complete_bound_inventory_is_ready(self) -> None:
        result = self.evaluate()
        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["blockers"], [])
        self.assertFalse(result["policy"]["archive_extracted"])

    def test_missing_integration_inputs_fail_closed(self) -> None:
        result = gate.evaluate_gate(self.spec, archive_path=self.archive_path)
        self.assertFalse(result["ready"])
        self.assertEqual(
            self.blocker_codes(result),
            {
                "runtime_inventory_missing",
                "preprocessing_manifest_missing",
                "runner_manifest_missing",
                "app_uid_evidence_missing",
            },
        )
        self.assertEqual(result["checks"]["archive"]["status"], "pass")
        self.assertEqual(result["checks"]["metadata"]["status"], "pass")

    def test_archive_hash_mismatch_stops_member_streaming(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["release"]["archive_sha256"] = "0" * 64
        result = self.evaluate(spec)
        self.assertIn("archive_sha256_mismatch", self.blocker_codes(result))
        self.assertEqual(result["checks"]["components"]["status"], "not_checked")

    def test_exact_ordered_archive_member_list_is_required(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["release"]["members"] = spec["release"]["members"][1:]
        result = self.evaluate(spec)
        self.assertIn("archive_member_list_mismatch", self.blocker_codes(result))
        self.assertEqual(result["checks"]["components"]["status"], "not_checked")

    def test_component_hash_mismatch_fails_closed(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["release"]["members"][2]["sha256"] = "f" * 64
        result = self.evaluate(spec)
        self.assertIn("component_sha256_mismatch", self.blocker_codes(result))
        self.assertEqual(result["checks"]["metadata"]["status"], "not_checked")

    def test_runtime_older_than_qairt_245_is_blocked(self) -> None:
        self.runtime_manifest = self.make_runtime_manifest("2.40.0.251030")
        result = self.evaluate()
        self.assertIn("runtime_qairt_too_old", self.blocker_codes(result))
        self.assertEqual(result["checks"]["runtime"]["status"], "blocked")

    def test_missing_tokenizer_asset_role_is_blocked(self) -> None:
        self.preprocessing_manifest = self.make_preprocessing_manifest(
            "tokenizer_model"
        )
        result = self.evaluate()
        self.assertIn(
            "preprocessing_asset_role_missing", self.blocker_codes(result)
        )

    def test_sherpa_style_tensor_names_are_rejected(self) -> None:
        incompatible = copy.deepcopy(self.tensor_names)
        incompatible["encoder"]["inputs"] = ["whisper_mel"]
        self.runner_manifest = self.make_runner_manifest(incompatible)
        result = self.evaluate()
        self.assertIn("runner_tensor_names_mismatch", self.blocker_codes(result))
        mismatch = result["checks"]["runner"]["contract_mismatches"][0]
        self.assertIn("input_features", mismatch["missing"])
        self.assertIn("whisper_mel", mismatch["unexpected"])

    def test_failed_app_uid_loader_check_is_blocked(self) -> None:
        checks = copy.deepcopy(self.spec["app_uid"]["required_checks"])
        checks["encoder_execute_passed"] = False
        self.app_uid_evidence = self.make_app_uid_evidence(checks)
        result = self.evaluate()
        self.assertIn("app_uid_readiness_check_failed", self.blocker_codes(result))

    def test_optional_extraction_root_is_read_only_and_exact(self) -> None:
        extracted_parent = self.root / "existing-extraction"
        extracted = extracted_parent / self.archive_root
        extracted.mkdir(parents=True)
        (extracted / "metadata.json").write_bytes(self.metadata_bytes)
        (extracted / "decoder.bin").write_bytes(self.decoder_bytes)
        (extracted / "encoder.bin").write_bytes(self.encoder_bytes)
        result = self.evaluate(extraction_root=extracted_parent)
        self.assertTrue(result["ready"])
        self.assertEqual(result["checks"]["extraction_root"]["status"], "pass")

    def test_verifier_never_calls_zip_extraction_apis(self) -> None:
        with mock.patch.object(
            zipfile.ZipFile, "extract", side_effect=AssertionError("extract called")
        ), mock.patch.object(
            zipfile.ZipFile,
            "extractall",
            side_effect=AssertionError("extractall called"),
        ):
            result = self.evaluate()
        self.assertTrue(result["ready"])

    def test_voice_ai_archive_supplies_tokenizer_but_not_runner_or_runtime(self) -> None:
        result = gate.evaluate_gate(
            self.spec,
            archive_path=self.archive_path,
            voice_ai_archive=self.voice_ai_archive,
        )
        codes = self.blocker_codes(result)
        self.assertFalse(result["ready"])
        self.assertEqual(result["checks"]["voice_ai"]["status"], "pass")
        self.assertTrue(result["checks"]["voice_ai"]["tokenizer_ready"])
        self.assertEqual(result["checks"]["preprocessing"]["status"], "pass")
        self.assertNotIn("preprocessing_manifest_missing", codes)
        self.assertIn("runtime_inventory_missing", codes)
        self.assertIn("runner_manifest_missing", codes)
        self.assertIn("app_uid_evidence_missing", codes)

    def test_voice_ai_route_requires_the_qpm_sdk_runner_declaration(self) -> None:
        self.runner_manifest = self.make_runner_manifest(
            release_metadata_sha256=self.spec["voice_ai"]["metadata"]["sha256"]
        )
        result = self.evaluate(
            voice_ai_archive=self.voice_ai_archive,
            preprocessing_manifest=None,
            app_uid_evidence=None,
        )
        self.assertIn(
            "runner_voice_ai_compatibility_missing", self.blocker_codes(result)
        )

    def test_report_is_deterministic_and_has_no_timestamp(self) -> None:
        first = self.evaluate()
        second = self.evaluate()
        self.assertEqual(first, second)
        payload = json.dumps(first, sort_keys=True)
        self.assertNotIn("timestamp", payload)
        self.assertNotIn("created_at", payload)


if __name__ == "__main__":
    unittest.main()
