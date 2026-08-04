#!/usr/bin/env python3
"""Focused unit tests for the deterministic Supertonic chain stager."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGER_PATH = ROOT / "scripts/stage_supertonic_chain_ort_qnn_probe.py"


def load_stager():
    specification = importlib.util.spec_from_file_location(
        "stage_supertonic_chain", STAGER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import Supertonic chain stager")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


stager = load_stager()


class StagerTests(unittest.TestCase):
    def test_pinned_identity_and_claim_boundary(self) -> None:
        self.assertEqual(
            stager.GATE_ID,
            "supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1",
        )
        self.assertEqual(stager.PACKAGE, "ai.liquid.lfmdemo.supertonicchainprobe")
        self.assertEqual(stager.QAIRT_BUILD, "2.42.0.251225135753_193295")
        self.assertEqual(stager.ORT_BYTES, 7_720_525)
        self.assertEqual(len(stager.MODEL_FILES), 4)
        source = STAGER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("adb", source.lower())
        self.assertIn('"host_reference_comparisons_promoted": False', source)
        self.assertIn('"audio_quality_promoted": False', source)
        self.assertIn('"production_integration": False', source)

    def test_array_hash_binds_dtype_shape_and_values(self) -> None:
        value = np.asarray([[1.0, -2.0]], dtype=np.float32)
        baseline = stager.sha256_array(value)
        self.assertNotEqual(baseline, stager.sha256_array(value.reshape(2, 1)))
        self.assertNotEqual(baseline, stager.sha256_array(value.astype(np.float64)))
        changed = value.copy()
        changed[0, 0] = 1.5
        self.assertNotEqual(baseline, stager.sha256_array(changed))

    def test_atomic_write_is_exclusive_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/value.bin"
            identity = stager.atomic_write(path, b"payload")
            self.assertEqual(path.read_bytes(), b"payload")
            self.assertEqual(identity["bytes"], 7)
            partial = path.with_name(path.name + ".partial")
            partial.write_bytes(b"stale")
            with self.assertRaisesRegex(RuntimeError, "stale partial"):
                stager.atomic_write(path, b"other")

    def test_external_output_rejects_repo_and_nonempty_directory(self) -> None:
        with self.assertRaises(RuntimeError):
            stager.require_external_output(ROOT / "staging")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "deep/output"
            stager.require_external_output(output)
            output.mkdir(parents=True)
            (output / "existing").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "absent or empty"):
                stager.require_external_output(output)

    @unittest.skipUnless(
        stager.DEFAULT_FIXTURE_DIR.is_dir(),
        "external deterministic fixture is not present",
    )
    def test_external_fixture_still_matches_pinned_contract(self) -> None:
        manifest, arrays = stager.validate_fixture(stager.DEFAULT_FIXTURE_DIR)
        self.assertEqual(manifest["fixture"]["steps"], 4)
        self.assertEqual(manifest["fixture"]["real_latent_length"], 31)
        self.assertEqual(len(arrays), len(stager.ARRAYS))


if __name__ == "__main__":
    unittest.main()
