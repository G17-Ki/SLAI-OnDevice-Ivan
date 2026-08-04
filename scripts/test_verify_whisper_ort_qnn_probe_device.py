#!/usr/bin/env python3
"""Tests for the fail-closed Whisper probe device-evidence verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_whisper_ort_qnn_probe_device.py"
LOADER = importlib.util.spec_from_file_location("verify_probe_device", SCRIPT)
assert LOADER and LOADER.loader
verify = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(verify)


def digest(value: dict) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


class ProbeDeviceVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {
            "schema_version": 2,
            "gate_id": "gate",
            "target": {"package": "pkg", "soc": "SM8850", "abi": "arm64-v8a"},
            "accepted_packages": {
                "main": {"sha256": "a" * 64},
                "runtime": {"sha256": "b" * 64},
            },
            "apk_policy": {"version_name": "0.1"},
            "required_assets": [
                {
                    "path": "whisper_probe/models/encoder.onnx",
                    "bytes": 10,
                }
            ],
            "embedded_manifest": {
                "runtime": {
                    "onnx_runtime": "1.25.0",
                    "qnn_provider": "QNNExecutionProvider",
                },
                "decode_contract": {
                    "steps": 4,
                    "attention_unmask_order": [199, 198, 197, 196],
                },
            },
        }
        self.local = {
            "gate_id": "gate",
            "package": "pkg",
            "status": "pass",
            "apk": {"sha256": "c" * 64, "bytes": 123},
        }
        self.report = {
            "gate_id": "gate",
            "package": "pkg",
            "uid": 10510,
            "ort_version": "1.25.0",
            "available_providers": ["CPUExecutionProvider", "QNNExecutionProvider"],
            "qnn_provider_requested": True,
            "cpu_ep_fallback_disabled": True,
            "backend_path": "/data/app/pkg/lib/arm64/libQnnHtp.so",
            "model_storage": "/data/user/0/pkg/files/models",
            "host_transport_required": False,
            "real_audio_transcription": False,
            "encoder_session_created": True,
            "decoder_session_created": True,
            "synthetic_decode": {
                "feature_source": "prequantized_fixture",
                "feature_dtype": "uint16",
                "encoder_execute_passed": True,
                "decoder_execute_passed": True,
                "decoder_steps": 4,
                "attention_unmasked_indices": [199, 198, 197, 196],
                "token_ids": [1, 2, 3, 4],
                "semantic_transcript_claimed": False,
            },
            "timing_ms": {
                "encoder_session_load": 1.0,
                "decoder_session_load": 1.0,
                "execute": 1.0,
                "total": 3.0,
            },
            "status": "compatibility_probe_pass",
        }
        self.local_sha = digest(self.local)
        self.report_sha = digest(self.report)
        self.evidence = {
            "schema_version": 1,
            "gate_id": "gate",
            "package": "pkg",
            "device": {
                "model": "NX809J",
                "soc": "SM8850",
                "abi": "arm64-v8a",
                "airplane_mode_on": 1,
            },
            "installed_package": {
                "package": "pkg",
                "version_code": 1,
                "version_name": "0.1",
                "uid": 10510,
                "pid": 1234,
                "apk_path": "/data/app/pkg/base.apk",
                "bytes": 123,
                "sha256": "c" * 64,
            },
            "accepted_packages": {
                "main": {"sha256": "a" * 64},
                "runtime": {"sha256": "b" * 64},
            },
            "process": {
                "pid": 1234,
                "uid_fields": [10510, 10510, 10510, 10510],
                "report_owner_uid": 10510,
                "loaded_libraries": sorted(verify.REQUIRED_PROCESS_LIBRARIES),
                "fastrpc_cdsp_fd_open": True,
                "qcom_dmabuf_count": 2,
            },
            "resident_payload": {
                "root": "/data/user/0/pkg/files/models",
                "kib": 1,
                "model_files": {"encoder.onnx": 10},
            },
            "local_verifier_report": {"sha256": self.local_sha, "bytes": 1},
            "device_report": {"sha256": self.report_sha, "bytes": 1},
        }

    def run_verify(self) -> dict:
        return verify.verify_documents(
            self.spec,
            self.local,
            self.report,
            self.evidence,
            local_verify_sha256=self.local_sha,
            device_report_sha256=self.report_sha,
        )

    def test_passes_complete_evidence(self) -> None:
        self.assertEqual(self.run_verify()["status"], "pass")

    def test_rejects_cpu_fallback(self) -> None:
        self.report["cpu_ep_fallback_disabled"] = False
        with self.assertRaisesRegex(RuntimeError, "CPU EP fallback"):
            self.run_verify()

    def test_rejects_missing_fastrpc_fd(self) -> None:
        self.evidence["process"]["fastrpc_cdsp_fd_open"] = False
        with self.assertRaisesRegex(RuntimeError, "FastRPC CDSP"):
            self.run_verify()

    def test_rejects_accepted_apk_drift(self) -> None:
        self.evidence["accepted_packages"]["main"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "accepted APK"):
            self.run_verify()

    def test_rejects_report_digest_mismatch(self) -> None:
        self.evidence["device_report"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "app-private report SHA-256"):
            self.run_verify()

    def test_rejects_installed_apk_mismatch(self) -> None:
        self.evidence["installed_package"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "installed APK hash"):
            self.run_verify()

    def test_rejects_missing_qnn_library(self) -> None:
        self.evidence["process"]["loaded_libraries"].remove("libQnnHtp.so")
        with self.assertRaisesRegex(RuntimeError, "not all mapped"):
            self.run_verify()

    def test_rejects_resident_model_drift(self) -> None:
        self.evidence["resident_payload"]["model_files"]["encoder.onnx"] = 11
        with self.assertRaisesRegex(RuntimeError, "resident model file inventory"):
            self.run_verify()


if __name__ == "__main__":
    unittest.main()
