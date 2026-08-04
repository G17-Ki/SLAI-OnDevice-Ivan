#!/usr/bin/env python3
"""Regression tests for the chain-probe whole-APK verifier."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_supertonic_chain_ort_qnn_probe_apk.py"
SPEC_PATH = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_chain_ort_qnn_probe_spec.json"
)


def load_verifier():
    specification = importlib.util.spec_from_file_location(
        "verify_supertonic_chain_apk", VERIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import chain APK verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


verifier = load_verifier()


class FakeArchive:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def namelist(self) -> list[str]:
        return self.names


class ApkVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = verifier.load_object(SPEC_PATH)

    def test_authoritative_spec_passes_internal_validation(self) -> None:
        verifier.validate_spec(self.spec, SPEC_PATH.resolve())

    def test_claim_promotion_in_spec_is_rejected(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["capabilities"]["audio_quality_acceptance"] = True
        with self.assertRaisesRegex(RuntimeError, "capability boundary"):
            verifier.validate_spec(changed, SPEC_PATH.resolve())

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
            verifier.strict_json_bytes(b'{"a":1,"a":2}', "duplicate")

    def test_whole_apk_identity_precedes_zip_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            apk = Path(temporary) / "candidate.apk"
            apk.write_bytes(b"not the pinned APK")
            with mock.patch.object(verifier, "verify_archive") as archive:
                with self.assertRaisesRegex(RuntimeError, "whole chain APK byte"):
                    verifier.verify_apk(
                        apk,
                        self.spec,
                        spec_path=SPEC_PATH.resolve(),
                        build_tools=Path(temporary),
                        java_home=Path(temporary),
                    )
                archive.assert_not_called()

    def test_archive_rejects_extra_non_payload_member_before_hashing(self) -> None:
        names = ["assets/" + record["path"] for record in self.spec["required_assets"]]
        names.append(verifier.PAYLOAD_MANIFEST_MEMBER)
        names.extend(
            "lib/" + record["path"] for record in self.spec["required_native_libraries"]
        )
        names.extend(self.spec["zip_policy"]["other_members"])
        names.append("unexpected.txt")
        with mock.patch.object(
            verifier.zipfile, "ZipFile", return_value=FakeArchive(names)
        ):
            with self.assertRaisesRegex(RuntimeError, "non-payload set mismatch"):
                verifier.verify_archive(Path("unused.apk"), self.spec)


if __name__ == "__main__":
    unittest.main()
