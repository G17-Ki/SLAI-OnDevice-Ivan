#!/usr/bin/env python3
"""Unit tests for the offline Whisper ORT-QNN payload stager."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage_whisper_ort_qnn_probe.py"
SPEC = importlib.util.spec_from_file_location("stage_whisper_probe", SCRIPT)
assert SPEC and SPEC.loader
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class StageWhisperProbeTest(unittest.TestCase):
    def test_repository_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the repository"):
            stage.require_external_output(ROOT / "staging/whisper-probe")

    def test_verify_file_checks_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"exact")
            spec = {
                "bytes": 5,
                "sha256": hashlib.sha256(b"exact").hexdigest(),
            }
            stage.verify_file(path, spec)
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                stage.verify_file(path, spec)

    def test_extract_verified_member_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("root/model.onnx", b"payload")
            destination = root / "out/model.onnx"
            spec = {
                "bytes": 7,
                "sha256": hashlib.sha256(b"payload").hexdigest(),
            }
            stage.extract_verified_member(
                archive, "root/model.onnx", destination, spec
            )
            self.assertEqual(destination.read_bytes(), b"payload")

    def test_manifest_fails_closed_for_real_audio(self) -> None:
        manifest = stage.build_manifest([], [], None)
        self.assertTrue(manifest["capabilities"]["four_step_decode"])
        self.assertFalse(manifest["capabilities"]["real_audio_frontend"])
        self.assertFalse(manifest["capabilities"]["real_audio_transcription"])
        self.assertEqual(manifest["decode_contract"]["attention_unmask_order"], [199, 198, 197, 196])
        self.assertEqual(manifest["vocabulary"]["raw_nul_token_id"], 188)

    def test_manifest_is_path_independent_and_deterministic(self) -> None:
        first = stage.build_manifest([], [], None)
        second = stage.build_manifest([], [], None)
        self.assertEqual(first, second)
        self.assertNotIn("stage_root_name", first)
        self.assertNotIn("output", json.dumps(first, sort_keys=True))

    def test_feature_fixture_hash_is_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "features.u16le"
            fixture.write_bytes(b"\x00" * stage.FEATURE_FIXTURE["bytes"])
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                stage.verify_file(fixture, stage.FEATURE_FIXTURE)


if __name__ == "__main__":
    unittest.main()
