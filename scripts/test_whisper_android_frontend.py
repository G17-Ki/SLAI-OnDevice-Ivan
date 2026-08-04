#!/usr/bin/env python3
"""Fail-closed host/unit gates for the Android bundled-WAV frontend."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "android-app/whisper-probe"
CPP = MODULE / "src/main/cpp"
KISS = CPP / "third_party/kissfft"
RUNNER_SOURCE = MODULE / "src/test/native/whisper_frontend_host_runner.c"
STAGER_PATH = ROOT / "scripts/stage_whisper_android_frontend_assets.py"


def load_stager() -> Any:
    specification = importlib.util.spec_from_file_location(
        "stage_whisper_android_frontend_assets", STAGER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_u16(actual_path: Path, golden_path: Path) -> dict[str, Any]:
    actual = actual_path.read_bytes()
    golden = golden_path.read_bytes()
    if len(actual) != 480_000 or len(golden) != 480_000:
        raise AssertionError("frontend tensors must each contain 480000 bytes")
    differing = 0
    max_delta = 0
    for offset in range(0, len(actual), 2):
        actual_value = struct.unpack_from("<H", actual, offset)[0]
        golden_value = struct.unpack_from("<H", golden, offset)[0]
        delta = abs(actual_value - golden_value)
        if delta:
            differing += 1
            max_delta = max(max_delta, delta)
    return {
        "produced_tensor_sha256": hashlib.sha256(actual).hexdigest(),
        "golden_tensor_sha256": hashlib.sha256(golden).hexdigest(),
        "differing_uint16_values": differing,
        "differing_fraction": differing / 240_000,
        "max_uint16_delta": max_delta,
    }


class WhisperAndroidFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stager = load_stager()
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="whisper-android-frontend-test-"
        )
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "bundle"
        cls.stager.stage(cls.bundle)
        cls.runner = cls.root / "whisper_frontend_host_runner"
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("host C compiler is unavailable")
        command = [
            compiler,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{CPP}",
            f"-I{KISS}",
            str(CPP / "whisper_frontend_core.c"),
            str(KISS / "kiss_fft.c"),
            str(KISS / "kiss_fftr.c"),
            str(RUNNER_SOURCE),
            "-lm",
            "-o",
            str(cls.runner),
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise AssertionError(
                "host frontend compile failed\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_asset_bundle_is_canonical_and_corrected(self) -> None:
        result = self.stager.verify_bundle(self.bundle)
        self.assertEqual(result["assets"], 6)
        self.assertEqual(
            result["manifest_sha256"],
            "aea54e5c751f9a2b489c709e5ca149db7c39cbdabea3ed4675483c00e236693f",
        )
        manifest = json.loads(
            (self.bundle / "assets/whisper_frontend/v1/manifest.json").read_text()
        )
        encoded = json.dumps(manifest, sort_keys=True)
        self.assertIn(
            "8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80",
            encoded,
        )
        self.assertNotIn(
            "05ec7e5f",
            encoded,
        )

    def test_native_core_matches_en_and_corrected_ja_bound(self) -> None:
        asset_root = self.bundle / "assets/whisper_frontend/v1"
        results: dict[str, Any] = {}
        for language, frames in (("en", 52_819), ("ja", 39_710)):
            output = self.root / f"{language}.features.u16le.bin"
            process = subprocess.run(
                [
                    str(self.runner),
                    str(asset_root / f"fixtures/question_{language}.wav"),
                    str(frames),
                    str(asset_root / "tables/hann.f32le.bin"),
                    str(asset_root / "tables/mel_filters.f32le.bin"),
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                process.returncode,
                0,
                f"{language} frontend failed: {process.stderr}",
            )
            result = compare_u16(
                output, asset_root / f"goldens/{language}.features.u16le.bin"
            )
            self.assertLessEqual(result["max_uint16_delta"], 1)
            self.assertLessEqual(result["differing_uint16_values"], 240)
            self.assertLessEqual(result["differing_fraction"], 0.001)
            results[language] = result
        print(json.dumps({"native_frontend": results}, sort_keys=True))

    def test_native_core_rejects_truncated_wav(self) -> None:
        asset_root = self.bundle / "assets/whisper_frontend/v1"
        truncated = self.root / "truncated.wav"
        truncated.write_bytes(
            (asset_root / "fixtures/question_en.wav").read_bytes()[:43]
        )
        process = subprocess.run(
            [
                str(self.runner),
                str(truncated),
                "52819",
                str(asset_root / "tables/hann.f32le.bin"),
                str(asset_root / "tables/mel_filters.f32le.bin"),
                str(self.root / "should-not-exist.bin"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("WAV is null or truncated", process.stderr)

    def test_bundle_verifier_rejects_tampered_golden(self) -> None:
        tampered = self.root / "tampered-bundle"
        shutil.copytree(self.bundle, tampered)
        golden = tampered / (
            "assets/whisper_frontend/v1/goldens/ja.features.u16le.bin"
        )
        value = bytearray(golden.read_bytes())
        value[0] ^= 1
        golden.write_bytes(value)
        with self.assertRaisesRegex(self.stager.StageError, "SHA-256 mismatch"):
            self.stager.verify_bundle(tampered)

    def test_android_source_is_fail_closed(self) -> None:
        gradle = (MODULE / "build.gradle").read_text()
        manifest = (MODULE / "src/main/AndroidManifest.xml").read_text()
        activity = (
            MODULE
            / "src/main/java/ai/liquid/lfmdemo/whisperprobe/ProbeActivity.java"
        ).read_text()
        frontend = (
            MODULE
            / "src/main/java/ai/liquid/lfmdemo/whisperprobe/WhisperFrontendProbe.java"
        ).read_text()
        core = (CPP / "whisper_frontend_core.c").read_text()
        jni = (CPP / "whisper_frontend_jni.c").read_text()
        self.assertNotIn("android.permission.RECORD_AUDIO", manifest)
        self.assertNotIn("android.permission.INTERNET", manifest)
        self.assertIn("-fno-fast-math", gradle)
        self.assertIn("-ffp-contract=off", gradle)
        self.assertIn("WHISPER_FRONTEND_FIXTURES_EMBEDDED", gradle)
        self.assertIn("bundled_audio_frontend", activity)
        self.assertIn('EXTRA_RUN_ID = "run_id"', activity)
        self.assertIn("StandardCopyOption.ATOMIC_MOVE", activity)
        self.assertIn('report.put("completion_marker"', activity)
        self.assertIn('report.put("run_id", run.runId)', activity)
        self.assertIn("differing_uint16_values", frontend)
        self.assertIn("produced_tensor_sha256", frontend)
        self.assertIn("max_uint16_delta", frontend)
        self.assertIn("boundedGatePassed ? english.output : null", frontend)
        self.assertIn("torch_reflect_index", core)
        self.assertIn("nearbyint((double)normalized", core)
        self.assertIn("pthread_once(&g_frontend_once", jni)
        self.assertNotIn("05ec7e5f", gradle + frontend + core + jni)
        expected_kiss_hashes = {
            "kiss_fft.c": "3e8af99313c7e94f952ac12db46d3d5c5dd4a9bf93e0c32d43efa5fe39724051",
            "kiss_fftr.c": "128b651120725e8d55f6b9edf5d44b06dc022b260894eac078c6ff8aa7159561",
            "kiss_fft.h": "9d6c7ddf520932c0aad669c86c064a24e571e594681be9f1491c14c9c9c75693",
            "kiss_fftr.h": "a0f72c1bddc3126d7a51ad80e965b58f6af77e4419aa882a0b11951e783632a8",
            "kiss_fft_log.h": "f954cb6890ec999f7fbc80cdd1c8c0194bbaeb0f1c27e11bd23450f53871cf8c",
            "_kiss_fft_guts.h": "dd1731891d0fff0e297198847de39216c5c8b9677ea4fe53160d1f6c74218d40",
            "COPYING": "a2840585f8411be8e6826a31ef15ae65c950bd74a2437a73b013398a934ad0c6",
        }
        for name, expected in expected_kiss_hashes.items():
            self.assertEqual(sha256(KISS / name), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
