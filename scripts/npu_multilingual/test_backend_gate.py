#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import select_android_asr_backend as gate


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BackendGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        runtime = root / "runtime"
        model = root / "model"
        runtime.mkdir()
        model.mkdir()
        for name in gate.RUNTIME_FILES:
            (runtime / name).write_bytes(name.encode())
        (model / "model.bin").write_bytes(b"model")
        (model / "tokens.txt").write_bytes(b"tokens")
        self.assets = {
            "target": {"qnn_version": "2.40.0.251030"},
            "assets": {
                "sensevoice_context": {
                    "model_bin_sha256": digest(b"model"),
                    "tokens_sha256": digest(b"tokens"),
                }
            },
        }
        self.bundle = {
            "target": {
                "soc": "SM8850",
                "htp_arch": "v81",
                "qnn_version": "2.40.0.251030",
            },
            "paths": {
                "qnn_runtime_root": str(runtime),
                "sensevoice_model_root": str(model),
            },
        }
        self.device = {
            "connected": True,
            "soc_model": "SM8850",
            "abi": "arm64-v8a",
            "qnn_libraries": ["/vendor/lib64/libQnnHtpV81Stub.so"],
        }
        self.smoke = {
            "status": "pass",
            "provider": "qnn",
            "soc": "SM8850",
            "htp_arch": "v81",
            "qnn_version": "2.40.0.251030",
            "model_bin_sha256": digest(b"model"),
            "runtime_load": "pass",
            "transcript_parity": {"en": "pass", "zh": "pass"},
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def choose(self, language: str, scope: str = "all") -> dict:
        return gate.choose_backend(
            language, scope, self.device, self.bundle, self.smoke, self.assets
        )

    def test_explicit_english_uses_qnn_after_all_gates(self) -> None:
        self.assertEqual(self.choose("en")["provider"], "qnn")

    def test_primary_auto_can_use_qnn(self) -> None:
        self.assertEqual(self.choose("auto", "zh_en")["provider"], "qnn")

    def test_broad_auto_stays_on_cpu(self) -> None:
        result = self.choose("auto", "all")
        self.assertEqual(result["provider"], "cpu")
        self.assertIn("broad_auto_requires_cpu_multilingual_asr", result["reason_codes"])

    def test_russian_stays_on_cpu(self) -> None:
        self.assertEqual(self.choose("ru")["provider"], "cpu")

    def test_missing_smoke_fails_closed(self) -> None:
        result = gate.choose_backend(
            "zh", "all", self.device, self.bundle, {}, self.assets
        )
        self.assertEqual(result["provider"], "cpu")
        self.assertIn("phone_qnn_smoke_not_passed", result["reason_codes"])

    def test_wrong_soc_fails_closed(self) -> None:
        self.device["soc_model"] = "SM8750"
        result = self.choose("en")
        self.assertEqual(result["provider"], "cpu")
        self.assertIn("soc_is_not_exact_sm8850", result["reason_codes"])

    def test_missing_runtime_fails_closed(self) -> None:
        runtime = Path(self.bundle["paths"]["qnn_runtime_root"])
        (runtime / "libQnnHtpV81Skel.so").unlink()
        result = self.choose("en")
        self.assertEqual(result["provider"], "cpu")
        self.assertTrue(
            any(reason.startswith("qnn_runtime_files_missing") for reason in result["reason_codes"])
        )


if __name__ == "__main__":
    unittest.main()
