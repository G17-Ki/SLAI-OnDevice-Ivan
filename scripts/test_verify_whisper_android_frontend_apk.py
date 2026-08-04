#!/usr/bin/env python3
"""Host-only tests for the independent whole-APK frontend trust root."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_whisper_android_frontend_apk.py"
LOADER = importlib.util.spec_from_file_location("verify_frontend_apk", SCRIPT)
assert LOADER and LOADER.loader
verify = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(verify)


class FakeZip:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def __enter__(self) -> "FakeZip":
        return self

    def __exit__(self, *ignored: object) -> None:
        del ignored

    def namelist(self) -> list[str]:
        return self.names


class FrontendApkVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_spec = verify.load_object(verify.DEFAULT_BASE_SPEC)
        self.frontend_spec = verify.load_object(verify.DEFAULT_FRONTEND_SPEC)

    def write_frontend_spec(self, directory: Path, value: dict) -> Path:
        path = directory / "frontend_spec.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def test_committed_trust_root_validates(self) -> None:
        verify.validate_frontend_spec(
            self.frontend_spec,
            self.base_spec,
            frontend_spec_path=verify.DEFAULT_FRONTEND_SPEC,
            base_spec_path=verify.DEFAULT_BASE_SPEC,
        )

    def test_explicit_v2_profile_validates_against_additive_stager(self) -> None:
        frontend_spec = verify.load_object(verify.DEFAULT_FRONTEND_V2_SPEC)
        verify.validate_frontend_spec(
            frontend_spec,
            self.base_spec,
            frontend_spec_path=verify.DEFAULT_FRONTEND_V2_SPEC,
            base_spec_path=verify.DEFAULT_BASE_SPEC,
            frontend_profile="v2",
        )
        with self.assertRaisesRegex(RuntimeError, "field inventory|wrong profile"):
            verify.validate_frontend_spec(
                frontend_spec,
                self.base_spec,
                frontend_spec_path=verify.DEFAULT_FRONTEND_V2_SPEC,
                base_spec_path=verify.DEFAULT_BASE_SPEC,
            )

    def test_committed_v2_trust_roots_validate(self) -> None:
        report_root = verify.DEFAULT_FRONTEND_V2_SPEC.parent
        paths = [
            verify.DEFAULT_FRONTEND_V2_SPEC,
            report_root
            / "nubia_whisper_android_frontend_multilang_v2_semantic_zh_probe_spec.json",
            report_root
            / "nubia_whisper_android_frontend_multilang_v2_semantic_ko_probe_spec.json",
            report_root
            / "nubia_whisper_android_frontend_multilang_v2_semantic_ru_probe_spec.json",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                frontend_spec = verify.load_object(path)
                verify.validate_frontend_spec(
                    frontend_spec,
                    self.base_spec,
                    frontend_spec_path=path,
                    base_spec_path=verify.DEFAULT_BASE_SPEC,
                    frontend_profile="v2",
                )

    def test_rejects_unpinned_claim_and_policy_drift(self) -> None:
        report_root = verify.DEFAULT_FRONTEND_V2_SPEC.parent
        source_path = report_root / (
            "nubia_whisper_android_frontend_multilang_v2_semantic_ru_probe_spec.json"
        )
        source = verify.load_object(source_path)
        mutations = {
            "claim_boundary": lambda value: value["claim_boundary"].__setitem__(
                "real_audio_transcription", True
            ),
            "accepted_package": lambda value: value["accepted_packages"][
                "ai.liquid.lfmdemo"
            ].__setitem__("sha256", "0" * 64),
            "semantic_language": lambda value: value[
                "semantic_decode_overlay"
            ].__setitem__("language_code", "zh"),
            "semantic_threshold": lambda value: value[
                "semantic_decode_overlay"
            ].__setitem__("max_cer", 1.0),
            "semantic_reference": lambda value: value[
                "semantic_decode_overlay"
            ].__setitem__("reference_text", "easy"),
            "build_policy": lambda value: value["build_policy"].__setitem__(
                "semantic_decode_enabled", False
            ),
            "device_policy": lambda value: value["device_output_contract"][
                "frontend"
            ].__setitem__("max_differing_values", 240_000),
            "null_semantic_overlay": lambda value: value.__setitem__(
                "semantic_decode_overlay", None
            ),
            "extra_field": lambda value: value.__setitem__("unvalidated_claim", True),
        }
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    value = copy.deepcopy(source)
                    mutate(value)
                    path = temporary / f"{name}.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        verify.validate_frontend_spec(
                            value,
                            self.base_spec,
                            frontend_spec_path=path,
                            base_spec_path=verify.DEFAULT_BASE_SPEC,
                            frontend_profile="v2",
                        )

    def test_rejects_null_v1_semantic_overlay_claim(self) -> None:
        frontend_spec = copy.deepcopy(self.frontend_spec)
        frontend_spec["semantic_decode_overlay"] = None
        frontend_spec["claim_boundary"]["semantic_candidate_decode"] = True
        with self.assertRaisesRegex(RuntimeError, "overlay is not an object"):
            verify.validate_frontend_contract(
                frontend_spec,
                self.base_spec,
                frontend_profile="v1",
            )

    def test_wrong_apk_bytes_fail_before_archive_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            apk = Path(temporary) / "probe.apk"
            apk.write_bytes(b"not-the-pinned-apk")
            with mock.patch.object(verify, "verify_archive") as archive:
                with self.assertRaisesRegex(RuntimeError, "whole frontend APK byte"):
                    verify.verify_apk(
                        apk,
                        self.base_spec,
                        self.frontend_spec,
                        base_spec_path=verify.DEFAULT_BASE_SPEC,
                        frontend_spec_path=verify.DEFAULT_FRONTEND_SPEC,
                        build_tools=Path(temporary),
                        java_home=Path(temporary),
                    )
                archive.assert_not_called()

    def test_wrong_apk_hash_fails_before_archive_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            apk = temporary / "probe.apk"
            apk.write_bytes(b"same-size-wrong-hash")
            frontend_spec = copy.deepcopy(self.frontend_spec)
            frontend_spec["apk"] = {"bytes": apk.stat().st_size, "sha256": "0" * 64}
            frontend_spec_path = self.write_frontend_spec(temporary, frontend_spec)
            with mock.patch.object(verify, "verify_archive") as archive:
                with self.assertRaisesRegex(RuntimeError, "whole frontend APK SHA-256"):
                    verify.verify_apk(
                        apk,
                        self.base_spec,
                        frontend_spec,
                        base_spec_path=verify.DEFAULT_BASE_SPEC,
                        frontend_spec_path=frontend_spec_path,
                        build_tools=temporary,
                        java_home=temporary,
                    )
                archive.assert_not_called()

    def test_matching_whole_apk_identity_reaches_all_downstream_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            apk = temporary / "probe.apk"
            apk.write_bytes(b"synthetic-apk-for-ordering-test")
            frontend_spec = copy.deepcopy(self.frontend_spec)
            frontend_spec["apk"] = {
                "bytes": apk.stat().st_size,
                "sha256": hashlib.sha256(apk.read_bytes()).hexdigest(),
            }
            frontend_spec_path = self.write_frontend_spec(temporary, frontend_spec)
            with (
                mock.patch.object(
                    verify, "verify_archive", return_value={"status": "pass"}
                ) as archive,
                mock.patch.object(
                    verify.base,
                    "verify_toolchain",
                    return_value=({}, {"status": "pass"}),
                ),
                mock.patch.object(
                    verify.base, "collect_binary_evidence", return_value={}
                ),
                mock.patch.object(
                    verify.base,
                    "verify_binary_evidence",
                    return_value={"status": "pass"},
                ),
            ):
                result = verify.verify_apk(
                    apk,
                    self.base_spec,
                    frontend_spec,
                    base_spec_path=verify.DEFAULT_BASE_SPEC,
                    frontend_spec_path=frontend_spec_path,
                    build_tools=temporary,
                    java_home=temporary,
                )
            self.assertEqual(result["status"], "pass")
            archive.assert_called_once()

    def test_rejects_wrong_base_spec_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            frontend_spec = copy.deepcopy(self.frontend_spec)
            frontend_spec["base_probe"]["spec"]["sha256"] = "0" * 64
            path = self.write_frontend_spec(temporary, frontend_spec)
            with self.assertRaisesRegex(RuntimeError, "does not pin the base spec"):
                verify.validate_frontend_spec(
                    frontend_spec,
                    self.base_spec,
                    frontend_spec_path=path,
                    base_spec_path=verify.DEFAULT_BASE_SPEC,
                )

    def test_rejects_wrong_stager_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            frontend_spec = copy.deepcopy(self.frontend_spec)
            frontend_spec["stager"]["sha256"] = "0" * 64
            path = self.write_frontend_spec(temporary, frontend_spec)
            with self.assertRaisesRegex(RuntimeError, "stager identity"):
                verify.validate_frontend_spec(
                    frontend_spec,
                    self.base_spec,
                    frontend_spec_path=path,
                    base_spec_path=verify.DEFAULT_BASE_SPEC,
                )

    def test_rejects_wrong_frontend_jni_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            frontend_spec = copy.deepcopy(self.frontend_spec)
            frontend_spec["frontend_native_library"]["path"] = "arm64-v8a/libwrong.so"
            path = self.write_frontend_spec(temporary, frontend_spec)
            with self.assertRaisesRegex(RuntimeError, "JNI trust root"):
                verify.validate_frontend_spec(
                    frontend_spec,
                    self.base_spec,
                    frontend_spec_path=path,
                    base_spec_path=verify.DEFAULT_BASE_SPEC,
                )

    def test_rejects_non_payload_zip_member_drift(self) -> None:
        names = [
            *(
                "assets/" + record["path"]
                for record in self.base_spec["required_assets"]
            ),
            verify.base.PAYLOAD_MANIFEST_MEMBER,
            *(
                "lib/" + record["path"]
                for record in self.base_spec["required_native_libraries"]
            ),
            *(
                "assets/" + record["path"]
                for record in self.frontend_spec["frontend_assets"]
            ),
            "assets/" + self.frontend_spec["frontend_manifest"]["path"],
            verify.FRONTEND_LIBRARY_MEMBER,
            *self.frontend_spec["zip_policy"]["other_members"],
            "unexpected-member.bin",
        ]
        with mock.patch.object(verify.zipfile, "ZipFile", return_value=FakeZip(names)):
            with self.assertRaisesRegex(RuntimeError, "non-payload member set"):
                verify.verify_archive(
                    Path("unused.apk"), self.base_spec, self.frontend_spec
                )

    def test_strict_loader_rejects_duplicate_keys_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            duplicate = temporary / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                verify.load_object(duplicate)
            invalid = temporary / "nan.json"
            invalid.write_text('{"a": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid JSON numeric"):
                verify.load_object(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
