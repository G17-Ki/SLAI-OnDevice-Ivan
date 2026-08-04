#!/usr/bin/env python3

import hashlib
import json
import pathlib
import unittest
import wave


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "scripts" / "phone_acceptance_fixtures"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
PLAN_ROOT = REPO_ROOT / "scripts" / "phone_acceptance_plans"


class PhoneAcceptanceFixtureTests(unittest.TestCase):
    def test_manifest_authenticates_canonical_wavs(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        fixtures = manifest["fixtures"]
        self.assertIn("question_ja", fixtures)
        self.assertIn("question_ko", fixtures)

        for fixture_id, spec in fixtures.items():
            with self.subTest(fixture=fixture_id):
                path = FIXTURE_ROOT / spec["file"]
                payload = path.read_bytes()
                self.assertEqual(len(payload), spec["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), spec["sha256"])
                self.assertEqual(payload[:4], b"RIFF")
                self.assertEqual(payload[8:12], b"WAVE")
                self.assertEqual(payload[12:16], b"fmt ")
                self.assertEqual(payload[36:40], b"data")
                with wave.open(str(path), "rb") as wav:
                    self.assertEqual(wav.getframerate(), spec["sample_rate_hz"])
                    self.assertEqual(wav.getnchannels(), spec["channels"])
                    self.assertEqual(wav.getsampwidth(), spec["sample_width_bytes"])
                    self.assertEqual(wav.getnframes(), spec["frames"])
                    self.assertAlmostEqual(
                        wav.getnframes() / wav.getframerate(),
                        spec["duration_seconds"],
                        places=7,
                    )

    def test_schema_three_plans_pin_manifest_hashes(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["fixtures"]
        by_file = {spec["file"]: spec for spec in manifest.values()}
        for plan_path in sorted(PLAN_ROOT.glob("*.json")):
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if int(plan.get("schema_version", 0)) < 3:
                continue
            for case in plan["cases"]:
                with self.subTest(plan=plan_path.name, case=case["name"]):
                    filename = pathlib.Path(case["audio"]).name
                    self.assertIn(filename, by_file)
                    self.assertEqual(
                        case["expectedAudioSha256"],
                        by_file[filename]["sha256"],
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
