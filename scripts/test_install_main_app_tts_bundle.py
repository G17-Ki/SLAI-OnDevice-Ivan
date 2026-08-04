#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import install_main_app_tts_bundle as installer


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class InstallerPlanTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        files = {
            "models/acoustic.onnx": b"acoustic-model",
            "models/tokens.txt": b"a 0\nb 1\n",
            "models/dict/words.bin": b"dictionary",
            "models/vocoder.onnx": b"vocoder",
        }
        for relative, data in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        candidates = {
            "schema_version": 1,
            "candidates": [
                {
                    "id": "fixture",
                    "language": "zh",
                    "engine": "matcha",
                    "files": {
                        "acoustic_model": "models/acoustic.onnx",
                        "tokens": "models/tokens.txt",
                        "dict_dir": "models/dict",
                        "lexicon": "models/tokens.txt,models/vocoder.onnx",
                    },
                }
            ],
        }
        candidates_bytes = json.dumps(candidates, ensure_ascii=False).encode()
        (root / "candidates.json").write_bytes(candidates_bytes)
        (root / "phrases.json").write_text('{"zh": []}\n', encoding="utf-8")
        manifest_files = {
            relative: {"bytes": len(data), "sha256": digest(data)}
            for relative, data in files.items()
            if not relative.endswith("tokens.txt")
        }
        manifest_files["candidates.json"] = {
            "bytes": len(candidates_bytes),
            "sha256": digest(candidates_bytes),
        }
        phrases = (root / "phrases.json").read_bytes()
        manifest_files["phrases.json"] = {
            "bytes": len(phrases),
            "sha256": digest(phrases),
        }
        manifest = {
            "schema_version": 1,
            "candidate_count": 1,
            "missing_sources": [],
            "files": manifest_files,
        }
        (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_full_plan_hashes_manifest_omissions_and_orders_metadata_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan = installer.build_bundle_plan(self.make_bundle(Path(temp)))
        self.assertEqual(plan.candidate_count, 1)
        self.assertEqual(
            [item.relative_path for item in plan.assets],
            [
                "models/acoustic.onnx",
                "models/dict/words.bin",
                "models/tokens.txt",
                "models/vocoder.onnx",
            ],
        )
        self.assertTrue(all(item.sha256 for item in plan.assets))
        self.assertEqual(
            [item.relative_path for item in plan.metadata],
            ["bundle_manifest.json", "phrases.json", "candidates.json"],
        )
        self.assertEqual(plan.existence_checks, ())

    def test_manifest_only_rejects_unauthenticated_resident_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_bundle(Path(temp))
            for path in sorted((root / "models").rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            (root / "models").rmdir()
            with self.assertRaisesRegex(installer.InstallError, "cannot authenticate resident asset"):
                installer.build_bundle_plan(root, manifest_only=True)

    def test_manifest_only_rejects_size_only_resident_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_bundle(Path(temp))
            manifest_path = root / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["models/acoustic.onnx"].pop("sha256")
            manifest_path.write_text(json.dumps(manifest))
            (root / "models/acoustic.onnx").unlink()
            with self.assertRaisesRegex(installer.InstallError, "requires a SHA-256"):
                installer.build_bundle_plan(root, manifest_only=True)

    def test_candidate_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_bundle(Path(temp))
            candidates_path = root / "candidates.json"
            candidates = json.loads(candidates_path.read_text())
            candidates["candidates"][0]["files"]["model"] = "../escape.onnx"
            data = json.dumps(candidates).encode()
            candidates_path.write_bytes(data)
            manifest_path = root / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["candidates.json"] = {
                "bytes": len(data),
                "sha256": digest(data),
            }
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(installer.InstallError, "escapes the bundle root"):
                installer.build_bundle_plan(root)

    def test_candidates_manifest_digest_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_bundle(Path(temp))
            with (root / "candidates.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(installer.InstallError, "source size mismatch"):
                installer.build_bundle_plan(root)

    def test_unknown_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_bundle(Path(temp))
            manifest_path = root / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(installer.InstallError, "unsupported bundle_manifest"):
                installer.build_bundle_plan(root)

    def test_dry_run_never_requires_adb(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_bundle(Path(temp))
            result = installer.main(
                ["--bundle", str(root), "--dry-run", "--adb", "/definitely/missing/adb"]
            )
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
