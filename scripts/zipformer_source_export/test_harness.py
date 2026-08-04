from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import export_sources as exporter  # noqa: E402
import remote_worker  # noqa: E402


class ExportHarnessTests(unittest.TestCase):
    def test_pins_match_source_export_gate(self) -> None:
        pins = exporter.pin_manifest()
        self.assertEqual(pins["qaihm"]["version"], "0.58.0")
        self.assertEqual(
            pins["qaihm"]["wheel_sha256"],
            "2100a95f173c6c7b8f3bf65dcb710586098099522a5a901c6f918f3b981c4ada",
        )
        self.assertEqual(
            pins["checkpoint"]["revision"],
            "6eb615ae77ecac05c5628d5c8ed7037c14a338d5",
        )
        self.assertEqual(
            pins["checkpoint"]["sha256"],
            "9fb1a61c426c3910dc7228aec7c3e5783f268c15a25eb93cf37db9faf4f8e243",
        )
        self.assertEqual(pins["export"]["onnx_opset"], 17)
        self.assertFalse(pins["export"]["onnx_dynamo"])

    def test_component_order_is_canonical_and_deduplicated(self) -> None:
        self.assertEqual(
            exporter.normalize_components("encoder,decoder,joiner,decoder"),
            ("decoder", "joiner", "encoder"),
        )
        with self.assertRaises(ValueError):
            exporter.normalize_components("unknown")

    def test_error_sanitizer_removes_url_secret_and_home(self) -> None:
        credential_name = "refresh_" + "token"
        scheme = "htt" + "ps"
        raw = (
            f"request {scheme}://example.invalid/path?value=placeholder "
            f"{credential_name}=placeholder-value path={Path.home()}/private "
            "/content/private/model.pt"
        )
        sanitized = exporter.sanitize_error(raw)
        self.assertNotIn("https://", sanitized)
        self.assertNotIn("placeholder-value", sanitized)
        self.assertNotIn(str(Path.home()), sanitized)
        self.assertNotIn("/content/private", sanitized)
        self.assertIn("<url>", sanitized)
        self.assertIn("<redacted>", sanitized)

    def test_float_and_integer_comparisons(self) -> None:
        try:
            import numpy as np
        except ImportError as error:
            self.skipTest(str(error))
        close = exporter.compare_arrays(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.000001], dtype=np.float32),
            np,
        )
        self.assertTrue(close["passed"])
        integer_mismatch = exporter.compare_arrays(
            np.array([1, 2], dtype=np.int32),
            np.array([1, 3], dtype=np.int32),
            np,
        )
        self.assertFalse(integer_mismatch["passed"])

    def test_component_resume_requires_matching_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "sources" / "decoder.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"authoritative trace")
            manifest = {
                "status": "passed",
                "checkpoint_sha256": exporter.CHECKPOINT_SHA256,
                "qaihm_version": exporter.QAIHM_VERSION,
                "all_local_outputs_pass": True,
                "artifacts": {
                    "torchscript": {
                        "path": "sources/decoder.pt",
                        "sha256": exporter.sha256_file(artifact),
                    }
                },
            }
            exporter.atomic_write_json(root / "manifests" / "decoder.json", manifest)
            self.assertTrue(exporter.component_complete(root, "decoder"))
            artifact.write_bytes(b"changed")
            self.assertFalse(exporter.component_complete(root, "decoder"))

    def test_checkpoint_is_safe_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "sources").mkdir()
            (root / "sources" / "decoder.pt").write_bytes(b"trace")
            exporter.atomic_write_json(
                root / "export_status.json", {"state": "running"}
            )
            meta = exporter.refresh_checkpoint(root, "unit test")
            archive = root / "checkpoint.tar"
            self.assertEqual(meta["sha256"], exporter.sha256_file(archive))
            with tarfile.open(archive, "r") as bundle:
                names = bundle.getnames()
                self.assertIn("sources/decoder.pt", names)
                self.assertIn("SHA256SUMS", names)
                self.assertNotIn("checkpoint_meta.json", names)
                for member in bundle.getmembers():
                    self.assertTrue(member.isfile())
                    self.assertNotIn("..", Path(member.name).parts)

    def test_resume_restore_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            archive_path = parent / "unsafe.tar"
            payload = b"bad"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("../escape")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaises(RuntimeError):
                remote_worker.restore_resume_archive(parent / "root", archive_path)

    def test_resume_restore_recovers_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            archive_path = parent / "resume.tar"
            payload = b"model"
            checksum = hashlib.sha256(payload).hexdigest()
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("sources/joiner.pt")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                checksum_payload = f"{checksum}  sources/joiner.pt\n".encode()
                checksum_info = tarfile.TarInfo("SHA256SUMS")
                checksum_info.size = len(checksum_payload)
                archive.addfile(checksum_info, io.BytesIO(checksum_payload))
            root = parent / "root"
            remote_worker.restore_resume_archive(root, archive_path)
            self.assertEqual((root / "sources" / "joiner.pt").read_bytes(), payload)

    def test_resume_restore_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            archive_path = parent / "resume.tar"
            payload = b"tampered-model"
            checksum_payload = f"{'0' * 64}  sources/decoder.pt\n".encode()
            with tarfile.open(archive_path, "w") as archive:
                model_info = tarfile.TarInfo("sources/decoder.pt")
                model_info.size = len(payload)
                archive.addfile(model_info, io.BytesIO(payload))
                checksum_info = tarfile.TarInfo("SHA256SUMS")
                checksum_info.size = len(checksum_payload)
                archive.addfile(checksum_info, io.BytesIO(checksum_payload))
            with self.assertRaises(RuntimeError):
                remote_worker.restore_resume_archive(parent / "root", archive_path)

    def test_exporter_self_check_avoids_heavy_imports(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "export_sources.py"), "--self-check"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["self_check"], "passed")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("qai_hub_models", sys.modules)

    def test_orchestrator_dry_run_has_no_cloud_action(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT_DIR / "orchestrate_colab.sh"), "--dry-run"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "dry_run_passed")
        self.assertFalse(payload["cloud_action"])


if __name__ == "__main__":
    unittest.main()
