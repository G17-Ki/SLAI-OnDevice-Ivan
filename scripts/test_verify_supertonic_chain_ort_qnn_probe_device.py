#!/usr/bin/env python3
"""Focused evidence-safety tests for the chain device verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_supertonic_chain_ort_qnn_probe_device.py"


def load_verifier():
    specification = importlib.util.spec_from_file_location(
        "verify_supertonic_chain_device", VERIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import chain device verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


verifier = load_verifier()


def stats(elements: int = 4) -> dict:
    return {
        "elements": elements,
        "finite": True,
        "min": -1.0,
        "max": 1.0,
        "mean": 0.0,
        "rms": 0.5,
        "max_abs": 1.0,
        "nonzero_values": elements,
        "values_over_unit_range": 0,
        "sha256_float32le": "a" * 64,
    }


def comparison(elements: int = 4) -> dict:
    return {
        "diagnostic_only": True,
        "promoted": False,
        "elements": elements,
        "max_abs_error": 0.1,
        "mean_abs_error": 0.01,
        "rmse": 0.02,
        "cosine_similarity": 0.99,
        "actual_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
    }


class DeviceVerifierTests(unittest.TestCase):
    def test_process_parser_binds_uid_app_owned_qnn_fastrpc_dma_and_smaps(self) -> None:
        root = "/data/app/~~token/ai.liquid.lfmdemo.supertonicchainprobe-token"
        status = "Name:\tprobe\nUid:\t10123\t10123\t10123\t10123\nVmRSS:\t12345 kB\n"
        maps = "\n".join(
            [
                f"1000-2000 r-xp 0 00:00 0 {root}/lib/arm64/libQnnHtp.so",
                f"2000-3000 r-xp 0 00:00 0 {root}/lib/arm64/libonnxruntime.so",
                "3000-4000 r-xp 0 00:00 0 /vendor/lib64/libcdsprpc.so",
            ]
        )
        fds = "lrwx------ 1 u u 64 0 -> /dev/fastrpc-cdsp\n" + (
            "lrwx------ 1 u u 64 1 -> /dmabuf:qcom,system\n"
        )
        smaps = (
            "1000-2000 r-xp 0 00:00 0 /x\n"
            "Rss: 100 kB\nPss: 50 kB\nPrivate_Dirty: 10 kB\nSwap: 2 kB\n"
            "2000-3000 r-xp 0 00:00 0 /y\n"
            "Rss: 200 kB\nPss: 75 kB\nPrivate_Dirty: 20 kB\nSwap: 3 kB\n"
        )
        parsed = verifier.parse_process_snapshot(status, maps, fds, smaps, root)
        self.assertEqual(parsed["uid_fields"], [10123] * 4)
        self.assertEqual(parsed["rss_kib"], 300)
        self.assertEqual(parsed["pss_kib"], 125)
        self.assertTrue(parsed["fastrpc_cdsp_fd_open"])
        self.assertEqual(parsed["qcom_dmabuf_count"], 1)
        self.assertIn("libQnnHtp.so", parsed["loaded_libraries"])
        self.assertTrue(
            any(path.endswith("/libQnnHtp.so") for path in parsed["app_owned_mappings"])
        )

    def test_reference_comparison_cannot_be_promoted(self) -> None:
        value = comparison()
        value["diagnostic_only"] = False
        value["promoted"] = True
        with self.assertRaisesRegex(RuntimeError, "diagnostic-only"):
            verifier.verify_comparison(value, 4, "comparison")

    def test_boolean_numeric_field_is_rejected(self) -> None:
        value = stats()
        value["elements"] = True
        with self.assertRaisesRegex(RuntimeError, "invalid integer"):
            verifier.verify_stats(value, 4, "stats")

    def test_timestamp_must_be_utc_and_millisecond_exact(self) -> None:
        self.assertEqual(
            verifier.utc_unix_ms("1970-01-01T00:00:00.123Z", "time"),
            123,
        )
        with self.assertRaisesRegex(RuntimeError, "millisecond-exact"):
            verifier.utc_unix_ms("2026-07-30T00:00:00.123456Z", "time")

    def test_raw_artifact_hash_binding_rejects_tampering(self) -> None:
        root = "/data/app/~~token/package-token"
        values = {
            "status": b"Name:\tp\nUid:\t10001\t10001\t10001\t10001\nVmRSS:\t1 kB\n",
            "maps": f"1-2 r-xp 0 00:00 0 {root}/lib/arm64/libQnnHtp.so\n".encode(),
            "fd": b"x -> /dev/fastrpc-cdsp\nx -> /dmabuf:qcom,a\n",
            "smaps": b"1-2 r-xp 0 00:00 0 /x\nRss: 1 kB\nPss: 1 kB\nPrivate_Dirty: 0 kB\nSwap: 0 kB\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = {}
            for label, payload in values.items():
                path = directory / f"{label}.txt"
                path.write_bytes(payload)
                records[label] = {
                    "path": path.name,
                    "bytes": len(payload),
                    "sha256": verifier.hashlib.sha256(payload).hexdigest(),
                }
            parsed = verifier.parse_process_snapshot(
                *(
                    values[label].decode()
                    for label in ("status", "maps", "fd", "smaps")
                ),
                root,
            )
            process = {
                "pid": 1,
                "report_owner_uid": 10001,
                **parsed,
                "raw_artifacts": records,
            }
            (directory / "maps.txt").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "byte count mismatch|SHA-256 mismatch"
            ):
                verifier.verify_raw_artifacts(process, directory, root)


if __name__ == "__main__":
    unittest.main()
