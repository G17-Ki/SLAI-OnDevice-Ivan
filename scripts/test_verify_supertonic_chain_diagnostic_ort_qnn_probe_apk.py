#!/usr/bin/env python3
"""Focused trust-root tests for the Supertonic diagnostic APK verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    ROOT / "scripts/verify_supertonic_chain_diagnostic_ort_qnn_probe_apk.py"
)


def load_verifier():
    specification = importlib.util.spec_from_file_location(
        "verify_supertonic_chain_diagnostic_apk_test", VERIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import diagnostic APK verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


verifier = load_verifier()


class DiagnosticApkVerifierTests(unittest.TestCase):
    def test_committed_spec_and_embedded_manifest_match(self) -> None:
        spec_path = verifier.DEFAULT_SPEC.resolve()
        spec = verifier.load_object(spec_path)
        _, source = verifier.validate_spec(spec, spec_path)
        with zipfile.ZipFile(verifier.DEFAULT_APK) as bundle:
            manifest = verifier.verify_embedded_manifest(
                bundle.read(verifier.PAYLOAD_MANIFEST_MEMBER), spec, source
            )
        self.assertEqual(manifest["gate_id"], verifier.GATE_ID)
        self.assertTrue(manifest["runtime"]["disable_cpu_ep_fallback"])
        self.assertFalse(manifest["diagnostic_policy"]["promotion_allowed"])

    def test_whole_apk_identity_precedes_archive_inspection(self) -> None:
        spec_path = verifier.DEFAULT_SPEC.resolve()
        spec = verifier.load_object(spec_path)
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "wrong.apk"
            wrong.write_bytes(b"not the pinned APK")
            with mock.patch.object(verifier, "verify_archive") as archive:
                with self.assertRaisesRegex(RuntimeError, "whole APK byte mismatch"):
                    verifier.verify_apk(
                        wrong,
                        spec,
                        spec_path=spec_path,
                        build_tools=verifier.DEFAULT_BUILD_TOOLS,
                        java_home=verifier.DEFAULT_JAVA_HOME,
                    )
                archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
