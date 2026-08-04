#!/usr/bin/env python3
"""Host gates for the additive five-language Whisper frontend-v2 bundle."""

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
STAGER_PATH = ROOT / "scripts/stage_whisper_android_frontend_assets_multilang.py"
EXPECTED_MANIFEST_SHA256 = (
    "acb5c46f3bbbe3c11620bd662242157b30695500c1a0e8b6e7510acdc3157736"
)
LANGUAGE_FRAMES = {
    "en": 52_819,
    "zh": 41_717,
    "ru": 24_315,
    "ko": 33_432,
    "ja": 39_710,
}


def load_stager() -> Any:
    specification = importlib.util.spec_from_file_location(
        "stage_whisper_android_frontend_assets_multilang", STAGER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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


class WhisperAndroidFrontendMultilangTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stager = load_stager()
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="whisper-android-frontend-v2-test-"
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

    def test_v2_bundle_is_canonical_and_additive(self) -> None:
        result = self.stager.verify_bundle(self.bundle)
        self.assertEqual(result["assets"], 12)
        self.assertEqual(result["frontend_profile"], "v2")
        self.assertEqual(result["fixture_languages"], list(LANGUAGE_FRAMES))
        self.assertEqual(result["manifest_sha256"], EXPECTED_MANIFEST_SHA256)

        manifest_path = self.bundle / "assets/whisper_frontend/v2/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["frontend_profile"], "v2")
        self.assertEqual(manifest["fixture_languages"], list(LANGUAGE_FRAMES))
        self.assertEqual(
            manifest["gate_id"],
            "whisper_qai_hf_bundled_wav_frontend_multilang_v2",
        )
        self.assertEqual(
            manifest["verification_policy"]["max_differing_uint16_values"],
            240,
        )
        self.assertEqual(len(manifest["assets"]), 12)
        self.assertFalse((self.bundle / "assets/whisper_frontend/v1").exists())

    def test_native_core_matches_all_five_language_bounds(self) -> None:
        asset_root = self.bundle / "assets/whisper_frontend/v2"
        results: dict[str, Any] = {}
        for language, frames in LANGUAGE_FRAMES.items():
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
        print(json.dumps({"native_frontend_v2": results}, sort_keys=True))

    def test_bundle_verifier_rejects_tampered_multilingual_golden(self) -> None:
        tampered = self.root / "tampered-bundle"
        shutil.copytree(self.bundle, tampered)
        golden = tampered / "assets/whisper_frontend/v2/goldens/zh.features.u16le.bin"
        value = bytearray(golden.read_bytes())
        value[-1] ^= 1
        golden.write_bytes(value)
        with self.assertRaisesRegex(self.stager.StageError, "SHA-256 mismatch"):
            self.stager.verify_bundle(tampered)

    def test_bundle_verifier_rejects_noncanonical_manifest(self) -> None:
        tampered = self.root / "noncanonical-bundle"
        shutil.copytree(self.bundle, tampered)
        manifest_path = tampered / "assets/whisper_frontend/v2/manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(self.stager.StageError, "not canonical"):
            self.stager.verify_bundle(tampered)

    def test_android_profile_registry_is_additive_and_data_driven(self) -> None:
        gradle = (MODULE / "build.gradle").read_text(encoding="utf-8")
        activity = (
            MODULE / "src/main/java/ai/liquid/lfmdemo/whisperprobe/ProbeActivity.java"
        ).read_text(encoding="utf-8")
        frontend = (
            MODULE
            / "src/main/java/ai/liquid/lfmdemo/whisperprobe/WhisperFrontendProbe.java"
        ).read_text(encoding="utf-8")
        vocab = (
            MODULE / "src/main/java/ai/liquid/lfmdemo/whisperprobe/VocabDecoder.java"
        ).read_text(encoding="utf-8")

        self.assertIn('System.getenv("WHISPER_FRONTEND_PROFILE") ?: "v1"', gradle)
        self.assertIn('languages: ["en", "ja"]', gradle)
        self.assertIn('languages: ["en", "zh", "ru", "ko", "ja"]', gradle)
        self.assertIn(EXPECTED_MANIFEST_SHA256, gradle)
        self.assertIn('buildConfigField "String", "WHISPER_FRONTEND_PROFILE"', gradle)
        self.assertIn('frontend.featuresForLanguage("en")', activity)
        self.assertNotIn("frontend.englishFeatures", activity)
        self.assertIn("final Map<String, byte[]> featuresByLanguage", frontend)
        self.assertIn("addFixture(fixtures, fixtureZh(V2_ASSET_ROOT))", frontend)
        self.assertIn("addFixture(fixtures, fixtureRu(V2_ASSET_ROOT))", frontend)
        self.assertIn("addFixture(fixtures, fixtureKo(V2_ASSET_ROOT))", frontend)
        self.assertIn('if ("v1".equals(profile.id)', frontend)
        self.assertIn("languages.add(selectedLanguage)", frontend)
        self.assertIn("CodingErrorAction.REPORT", vocab)
        self.assertIn("Character.isISOControl(codePoint)", vocab)


if __name__ == "__main__":
    unittest.main(verbosity=2)
