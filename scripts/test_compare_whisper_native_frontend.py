#!/usr/bin/env python3
"""Focused tests for the native Whisper frontend audit."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
AUDIT_PATH = SCRIPT_DIR / "compare_whisper_native_frontend.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "compare_whisper_native_frontend", AUDIT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {AUDIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = load_audit_module()


class NativeWhisperFrontendAuditTest(unittest.TestCase):
    def test_fixture_profiles_preserve_v1_and_add_five_language_v2(self) -> None:
        self.assertEqual(audit.FIXTURE_PROFILES["v1"], ("en", "ja"))
        self.assertEqual(audit.FIXTURE_PROFILES["v2"], ("en", "zh", "ru", "ko", "ja"))
        self.assertEqual(audit.resolve_fixture_names("v1", None), ["en", "ja"])
        self.assertEqual(
            audit.resolve_fixture_names("v2", None),
            ["en", "zh", "ru", "ko", "ja"],
        )
        self.assertEqual(audit.resolve_fixture_names("v2", ["ko"]), ["ko"])

    def test_multilingual_fixture_identities_are_pinned(self) -> None:
        expected = {
            "zh": (
                41_717,
                "bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63",
                "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263",
            ),
            "ko": (
                33_432,
                "0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de",
                "808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5",
            ),
            "ru": (
                24_315,
                "3545b8a20e3f320174a4e4dee57ff77268f30f79526eef91c86c778e3c180683",
                "0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4",
            ),
        }
        for language, (frames, wav_sha256, quantized_sha256) in expected.items():
            fixture = audit.FIXTURES[language]
            self.assertEqual(fixture.frames, frames)
            self.assertEqual(fixture.wav_sha256, wav_sha256)
            self.assertEqual(fixture.quantized_sha256, quantized_sha256)
            _, record = audit.read_fixture(fixture)
            self.assertEqual(record["sha256"], wav_sha256)

    def test_torch_policy_is_explicitly_one_thread(self) -> None:
        self.assertEqual(
            audit.torch_thread_policy(),
            {
                "intraop_threads": 1,
                "interop_threads": 1,
                "omp_num_threads": "1",
                "mkl_num_threads": "1",
            },
        )

    def test_quantized_comparison_reports_tensor_coordinates(self) -> None:
        reference = np.zeros((audit.N_MELS, audit.N_FRAMES), dtype="<u2")
        actual = reference.copy()
        actual[7, 91] = 1
        record = audit.compare_quantized(reference, actual)
        self.assertFalse(record["strict_byte_parity"])
        self.assertEqual(record["different_elements"], 1)
        self.assertEqual(record["maximum_absolute_difference"], 1)
        self.assertEqual(record["first_differences"][0]["index"], [0, 7, 91])

    def test_float_comparison_is_bitwise_not_tolerance_only(self) -> None:
        reference = np.asarray([0.0, 1.0], dtype=np.float32)
        actual = reference.copy()
        actual[1] = np.nextafter(actual[1], np.float32(2.0))
        record = audit.compare_float(reference, actual)
        self.assertEqual(record["bitwise_different_elements"], 1)
        self.assertGreater(record["maximum_absolute_error"], 0.0)

    def test_pinned_source_inventory_exists_when_cache_is_present(self) -> None:
        required = [
            audit.DEFAULT_KISSFFT_ROOT,
            audit.DEFAULT_KALDI_PARENT,
            audit.DEFAULT_KISSFFT_ARCHIVE,
            audit.DEFAULT_KALDI_ARCHIVE,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("pinned native source cache is not present")
        record = audit.verify_sources(
            audit.DEFAULT_KISSFFT_ROOT,
            audit.DEFAULT_KALDI_PARENT,
            audit.DEFAULT_KISSFFT_ARCHIVE,
            audit.DEFAULT_KALDI_ARCHIVE,
        )
        self.assertEqual(
            record["archives"]["kissfft"]["sha256"],
            audit.EXPECTED_ARCHIVE_SHA256["kissfft"],
        )
        self.assertEqual(
            record["archives"]["kaldi_native_fbank"]["sha256"],
            audit.EXPECTED_ARCHIVE_SHA256["kaldi_native_fbank"],
        )

    def test_host_gate_rejects_stock_and_fails_closed_on_kissfft_hash(self) -> None:
        required = [
            audit.DEFAULT_KISSFFT_ROOT,
            audit.DEFAULT_KALDI_PARENT,
            audit.DEFAULT_KISSFFT_ARCHIVE,
            audit.DEFAULT_KALDI_ARCHIVE,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("pinned native source cache is not present")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "audit"
            result_path = output_dir / "result.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_PATH),
                    "--output-dir",
                    str(output_dir),
                    "--result",
                    str(result_path),
                    "--fixtures",
                    "en",
                    "--build-flavors",
                    "portable",
                    "--modes",
                    "kissfft-direct-scalar",
                    "kaldi-stock",
                    "--require-adapted-u16-parity",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 1, msg=process.stderr)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            variants = result["fixtures"]["en"]["variants"]
            adapted = variants["portable/kissfft-direct-scalar"]
            stock = variants["portable/kaldi-stock"]
            self.assertEqual(adapted["frame0"]["bitwise_different_elements"], 0)
            self.assertGreater(adapted["quantized_features"]["different_elements"], 0)
            self.assertLess(adapted["quantized_features"]["different_elements"], 1_000)
            self.assertEqual(
                adapted["quantized_features"]["maximum_absolute_difference"], 1
            )
            self.assertGreater(
                stock["quantized_features"]["different_elements"], 200_000
            )
            self.assertGreater(
                stock["quantized_features"]["maximum_absolute_difference"],
                10_000,
            )
            self.assertFalse(result["verdict"]["kissfft_strict_byte_parity"])
            self.assertTrue(result["verdict"]["kissfft_bounded_equivalence"])
            self.assertTrue(result["verdict"]["stock_path_rejected"])

    def test_v2_profile_runs_all_five_fixtures_with_bounded_equivalence(self) -> None:
        required = [
            audit.DEFAULT_KISSFFT_ROOT,
            audit.DEFAULT_KALDI_PARENT,
            audit.DEFAULT_KISSFFT_ARCHIVE,
            audit.DEFAULT_KALDI_ARCHIVE,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("pinned native source cache is not present")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "audit"
            result_path = output_dir / "result.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_PATH),
                    "--output-dir",
                    str(output_dir),
                    "--result",
                    str(result_path),
                    "--fixture-profile",
                    "v2",
                    "--build-flavors",
                    "portable",
                    "--modes",
                    "kissfft-direct-scalar",
                    "--require-adapted-bounded-parity",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, msg=process.stderr)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["fixture_profile"], "v2")
            self.assertEqual(
                result["selected_fixtures"], ["en", "zh", "ru", "ko", "ja"]
            )
            self.assertEqual(
                result["host"]["torch_thread_policy"]["intraop_threads"], 1
            )
            self.assertEqual(
                result["host"]["torch_thread_policy"]["interop_threads"], 1
            )
            self.assertTrue(result["verdict"]["kissfft_bounded_equivalence"])
            for language in result["selected_fixtures"]:
                quantized = result["fixtures"][language]["variants"][
                    "portable/kissfft-direct-scalar"
                ]["quantized_features"]
                self.assertLessEqual(quantized["maximum_absolute_difference"], 1)
                self.assertLessEqual(quantized["different_fraction"], 0.001)


if __name__ == "__main__":
    unittest.main()
