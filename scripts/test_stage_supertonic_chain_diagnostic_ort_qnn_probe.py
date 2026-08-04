#!/usr/bin/env python3
"""Focused tests for the distinct Supertonic stage-diagnostic stager."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STAGER_PATH = ROOT / "scripts/stage_supertonic_chain_diagnostic_ort_qnn_probe.py"


def load_stager():
    specification = importlib.util.spec_from_file_location(
        "stage_supertonic_chain_diagnostic", STAGER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import Supertonic chain diagnostic stager")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


stager = load_stager()


class DiagnosticStagerTests(unittest.TestCase):
    def test_identity_is_distinct_and_claims_are_diagnostic_only(self) -> None:
        self.assertEqual(
            stager.GATE_ID,
            "supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2",
        )
        self.assertEqual(
            stager.PACKAGE, "ai.liquid.lfmdemo.supertonicchaindiagprobe"
        )
        self.assertNotEqual(stager.GATE_ID, stager.BASE_GATE_ID)
        self.assertNotEqual(stager.PACKAGE, stager.BASE_PACKAGE)
        self.assertEqual(stager.BASE_MANIFEST_BYTES, 19_378)
        self.assertEqual(
            stager.BASE_MANIFEST_SHA256,
            "735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f",
        )
        source = STAGER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("adb", source.lower())
        self.assertIn('"cpu_ep_fallback_remains_disabled": True', source)
        self.assertIn('"failure_stage_recorded": True', source)
        self.assertIn('"failure_phase_recorded": True', source)
        self.assertIn('"promotion_allowed": False', source)

    def test_atomic_write_is_exclusive_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deep/value.json"
            identity = stager.atomic_write(path, b"{}\n")
            self.assertEqual(path.read_bytes(), b"{}\n")
            self.assertEqual(identity["bytes"], 3)
            path.with_name(path.name + ".partial").write_bytes(b"stale")
            with self.assertRaisesRegex(RuntimeError, "stale partial"):
                stager.atomic_write(path, b"other")

    def test_external_output_rejects_repository_and_nonempty_directory(self) -> None:
        with self.assertRaises(RuntimeError):
            stager.require_external_output(ROOT / "diagnostic-staging")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "deep/output"
            stager.require_external_output(output)
            output.mkdir(parents=True)
            (output / "existing").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "absent or empty"):
                stager.require_external_output(output)

    @unittest.skipUnless(
        stager.DEFAULT_BASE_PAYLOAD.is_dir(), "pinned r1 payload is unavailable"
    )
    def test_staged_manifest_preserves_pinned_payload_and_policy(self) -> None:
        _, base = stager.load_base_manifest(stager.DEFAULT_BASE_PAYLOAD)
        temporary_root = Path(
            tempfile.mkdtemp(prefix="lmf-diag-stager-test-", dir="/private/tmp")
        )
        staged_path = temporary_root / "diagnostic-payload"
        try:
            import sys

            with mock.patch.object(
                sys,
                "argv",
                [
                    str(STAGER_PATH),
                    "--output",
                    str(staged_path),
                    "--base-payload",
                    str(stager.DEFAULT_BASE_PAYLOAD),
                ],
            ):
                self.assertEqual(stager.main(), 0)
            manifest_path = (
                staged_path
                / "assets/supertonic_chain_diagnostic_probe/payload_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["gate_id"], stager.GATE_ID)
            self.assertEqual(manifest["target"]["package"], stager.PACKAGE)
            self.assertTrue(manifest["runtime"]["disable_cpu_ep_fallback"])
            self.assertEqual(
                manifest["diagnostic_policy"],
                {
                    "completed_stages_recorded": True,
                    "cpu_ep_fallback_remains_disabled": True,
                    "failure_phase_recorded": True,
                    "failure_stage_recorded": True,
                    "promotion_allowed": False,
                },
            )
            self.assertEqual(
                manifest["source_payload_manifest"],
                {
                    "bytes": stager.BASE_MANIFEST_BYTES,
                    "sha256": stager.BASE_MANIFEST_SHA256,
                },
            )
            self.assertEqual(len(manifest["assets"]), len(base["assets"]))
            self.assertEqual(
                len(manifest["native_libraries"]), len(base["native_libraries"])
            )
            for diagnostic, original in zip(manifest["assets"], base["assets"]):
                self.assertEqual(diagnostic["bytes"], original["bytes"])
                self.assertEqual(diagnostic["sha256"], original["sha256"])
                self.assertTrue(
                    diagnostic["path"].startswith(stager.DIAGNOSTIC_PREFIX)
                )
        finally:
            shutil.rmtree(temporary_root)


if __name__ == "__main__":
    unittest.main()
