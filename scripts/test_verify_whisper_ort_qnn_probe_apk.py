#!/usr/bin/env python3
"""Unit tests for fail-closed Whisper probe APK verification."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_whisper_ort_qnn_probe_apk.py"
SPEC_LOADER = importlib.util.spec_from_file_location("verify_whisper_probe_apk", SCRIPT)
assert SPEC_LOADER and SPEC_LOADER.loader
verify = importlib.util.module_from_spec(SPEC_LOADER)
SPEC_LOADER.loader.exec_module(verify)
AUTHORITATIVE_SPEC = json.loads(verify.DEFAULT_SPEC.read_text(encoding="utf-8"))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class VerifyWhisperProbeApkTest(unittest.TestCase):
    def make_archive_spec(self) -> tuple[dict, bytes, bytes, bytes]:
        feature = b"features"
        native = b"qnn"
        feature_record = {
            "path": "whisper_probe/models/input_features.u16le",
            "bytes": len(feature),
            "sha256": digest(feature),
            "role": "prequantized_feature_fixture",
        }
        native_record = {
            "path": "arm64-v8a/libQnnHtp.so",
            "bytes": len(native),
            "sha256": digest(native),
            "role": "qairt_libQnnHtp.so",
            "elf": "aarch64",
        }
        embedded = {
            "schema_version": 1,
            "gate_id": verify.EXPECTED_GATE_ID,
            "target": {"package": verify.EXPECTED_PACKAGE},
            "capabilities": {
                "real_audio_frontend": False,
                "real_audio_transcription": False,
                "four_step_decode": True,
            },
            "feature_fixture": feature_record,
        }
        spec = {
            "embedded_manifest": embedded,
            "required_assets": [feature_record],
            "required_native_libraries": [native_record],
        }
        manifest = verify.expected_embedded_manifest(spec)
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        spec["payload_manifest_sha256"] = digest(manifest_bytes)
        return spec, manifest_bytes, feature, native

    def build_apk(
        self,
        path: Path,
        *,
        tamper_feature: bool = False,
        extra_asset: bool = False,
        duplicate_manifest: bool = False,
    ) -> dict:
        spec, manifest_bytes, feature, native = self.make_archive_spec()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr(verify.PAYLOAD_MANIFEST_MEMBER, manifest_bytes)
                if duplicate_manifest:
                    bundle.writestr(verify.PAYLOAD_MANIFEST_MEMBER, manifest_bytes)
                bundle.writestr(
                    "assets/whisper_probe/models/input_features.u16le",
                    b"tampered" if tamper_feature else feature,
                )
                bundle.writestr("lib/arm64-v8a/libQnnHtp.so", native)
                if extra_asset:
                    bundle.writestr("assets/unexpected.bin", b"x")
        return spec

    def test_exact_archive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "probe.apk"
            spec = self.build_apk(apk)
            report = verify.verify_archive(apk, spec)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["verified_member_count"], 2)

    def test_tampered_member_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "probe.apk"
            spec = self.build_apk(apk, tamper_feature=True)
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify.verify_archive(apk, spec)

    def test_unexpected_asset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "probe.apk"
            spec = self.build_apk(apk, extra_asset=True)
            with self.assertRaisesRegex(RuntimeError, "asset set mismatch"):
                verify.verify_archive(apk, spec)

    def test_duplicate_zip_entry_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "probe.apk"
            spec = self.build_apk(apk, duplicate_manifest=True)
            with self.assertRaisesRegex(RuntimeError, "duplicate ZIP entries"):
                verify.verify_archive(apk, spec)

    def test_empty_authoritative_lists_fail(self) -> None:
        spec = copy.deepcopy(AUTHORITATIVE_SPEC)
        spec["required_assets"] = []
        with self.assertRaisesRegex(RuntimeError, "asset list"):
            verify.validate_spec(spec)

    def test_wrong_feature_fixture_fails(self) -> None:
        spec = copy.deepcopy(AUTHORITATIVE_SPEC)
        spec["embedded_manifest"]["feature_fixture"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "wrong feature fixture"):
            verify.validate_spec(spec)

    def test_invalid_verification_tool_digest_fails(self) -> None:
        spec = copy.deepcopy(AUTHORITATIVE_SPEC)
        spec["verification_toolchain"]["aapt2"]["sha256"] += "a"
        with self.assertRaisesRegex(RuntimeError, "verification-tool SHA-256"):
            verify.validate_spec(spec)

    def test_indented_optional_feature_is_parsed(self) -> None:
        badging = """\
package: name='ai.liquid.lfmdemo.whisperprobe' versionCode='1' versionName='0.1.0-ort-qnn-probe' compileSdkVersion='35'
sdkVersion:'28'
targetSdkVersion:'35'
launchable-activity: name='ai.liquid.lfmdemo.whisperprobe.ProbeActivity'
native-code: 'arm64-v8a'
  uses-feature-not-required: name='android.hardware.microphone'
"""
        parsed = verify.parse_aapt_evidence(badging, "")
        self.assertEqual(
            parsed["optional_features"], ["android.hardware.microphone"]
        )

    def test_wrong_binary_package_fails(self) -> None:
        parsed = self.valid_parsed_manifest()
        parsed["package"] = "ai.liquid.lfmdemo"
        with self.assertRaisesRegex(RuntimeError, "package mismatch"):
            verify.verify_android_manifest(parsed, AUTHORITATIVE_SPEC)

    def test_forbidden_binary_permission_fails(self) -> None:
        parsed = self.valid_parsed_manifest()
        parsed["permissions"] = ["android.permission.INTERNET"]
        with self.assertRaisesRegex(RuntimeError, "permissions mismatch"):
            verify.verify_android_manifest(parsed, AUTHORITATIVE_SPEC)

    def test_unexpected_binary_native_library_fails(self) -> None:
        parsed = self.valid_parsed_manifest()
        parsed["optional_native_libraries"].append("libwrong.so")
        with self.assertRaisesRegex(RuntimeError, "optional_native_libraries mismatch"):
            verify.verify_android_manifest(parsed, AUTHORITATIVE_SPEC)

    def test_signing_mismatch_fails(self) -> None:
        signing = copy.deepcopy(AUTHORITATIVE_SPEC["apk_policy"]["signing"])
        signing["certificate_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "signing evidence mismatch"):
            verify.verify_signing(signing, AUTHORITATIVE_SPEC)

    def valid_parsed_manifest(self) -> dict:
        policy = AUTHORITATIVE_SPEC["apk_policy"]
        return {
            "package": policy["package"],
            "version_code": policy["version_code"],
            "version_name": policy["version_name"],
            "compile_sdk": policy["compile_sdk"],
            "min_sdk": policy["min_sdk"],
            "target_sdk": policy["target_sdk"],
            "launcher": policy["launcher"],
            "permissions": list(policy["permissions"]),
            "native_abis": list(policy["native_abis"]),
            "required_native_libraries": list(policy["required_native_libraries"]),
            "optional_native_libraries": list(policy["optional_native_libraries"]),
            "optional_features": list(policy["required_optional_features"]),
        }


if __name__ == "__main__":
    unittest.main()
