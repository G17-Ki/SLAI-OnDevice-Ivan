#!/usr/bin/env python3
"""Host-only regressions for the fail-closed frontend device collector."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_whisper_android_frontend_device_probe.py"
REPORT_ROOT = ROOT / "reports/android_phone/streaming_response/whisper_v81"
BASE_SPEC = REPORT_ROOT / "nubia_whisper_ort_qnn_probe_spec.json"
FRONTEND_V1_SPEC = REPORT_ROOT / "nubia_whisper_android_frontend_probe_spec.json"
FRONTEND_V2_RU_SPEC = (
    REPORT_ROOT
    / "nubia_whisper_android_frontend_multilang_v2_semantic_ru_probe_spec.json"
)
LOADER = importlib.util.spec_from_file_location("run_frontend_device_probe", SCRIPT)
assert LOADER and LOADER.loader
collector = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(collector)


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


class ReportAdb:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def run_as(
        self, arguments: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        del arguments, check
        return subprocess.CompletedProcess([], 0, self.raw, b"")


class PropertyAdb:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def text(
        self,
        arguments: list[str],
        *,
        check: bool = True,
        timeout: float = 60.0,
    ) -> str:
        del check, timeout
        key = arguments[-1]
        return self.values.get(key, "")


class ExistenceAdb:
    def __init__(self, marker: bytes) -> None:
        self.marker = marker

    def run_as(
        self, arguments: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        del arguments, check
        return subprocess.CompletedProcess([], 0, self.marker, b"")


class FrontendDeviceCollectorTest(unittest.TestCase):
    def test_profile_selection_accepts_exact_v1_and_v2_contracts(self) -> None:
        base = json.loads(BASE_SPEC.read_text())
        frontend_v1 = json.loads(FRONTEND_V1_SPEC.read_text())
        frontend_v2 = json.loads(FRONTEND_V2_RU_SPEC.read_text())
        self.assertEqual(
            collector.select_frontend_profile(frontend_v1, base, "auto"),
            "v1",
        )
        self.assertEqual(
            collector.select_frontend_profile(frontend_v2, base, "auto"),
            "v2",
        )

    def test_profile_selection_rejects_explicit_mismatch(self) -> None:
        base = json.loads(BASE_SPEC.read_text())
        frontend_v2 = json.loads(FRONTEND_V2_RU_SPEC.read_text())
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            collector.select_frontend_profile(frontend_v2, base, "v1")

    def test_profile_selection_rejects_counterfeit_declaration(self) -> None:
        base = json.loads(BASE_SPEC.read_text())
        counterfeit = json.loads(FRONTEND_V1_SPEC.read_text())
        counterfeit["frontend_profile"] = "v2"
        with self.assertRaises(RuntimeError):
            collector.select_frontend_profile(counterfeit, base, "auto")

    def test_profile_selection_rejects_relaxed_claim_contract(self) -> None:
        base = json.loads(BASE_SPEC.read_text())
        counterfeit = copy.deepcopy(json.loads(FRONTEND_V2_RU_SPEC.read_text()))
        counterfeit["claim_boundary"]["real_audio_transcription"] = True
        with self.assertRaisesRegex(RuntimeError, "claim boundary drift"):
            collector.select_frontend_profile(counterfeit, base, "auto")

    def test_app_private_existence_uses_explicit_marker(self) -> None:
        self.assertTrue(
            collector.app_private_path_exists(ExistenceAdb(b"PRESENT"), "x")
        )
        self.assertFalse(
            collector.app_private_path_exists(ExistenceAdb(b"ABSENT"), "x")
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected.*marker"):
            collector.app_private_path_exists(ExistenceAdb(b"stat: missing"), "x")

    def test_package_uid_accepts_android_16_app_id_and_cross_checks_listing(
        self,
    ) -> None:
        dump = "  Package [ai.liquid.lfmdemo.whisperprobe]\n    appId=10510\n"
        listing = "package:ai.liquid.lfmdemo.whisperprobe uid:10510\n"
        self.assertEqual(collector.parse_package_uid(dump, listing), 10510)

    def test_package_uid_accepts_legacy_user_id(self) -> None:
        dump = "    userId=10510\n"
        listing = "package:ai.liquid.lfmdemo.whisperprobe uid:10510\n"
        self.assertEqual(collector.parse_package_uid(dump, listing), 10510)

    def test_package_uid_rejects_disagreeing_sources(self) -> None:
        dump = "    appId=10510\n"
        listing = "package:ai.liquid.lfmdemo.whisperprobe uid:10511\n"
        with self.assertRaisesRegex(RuntimeError, "sources disagree"):
            collector.parse_package_uid(dump, listing)

    def test_failed_closed_report_is_archived_before_raise(self) -> None:
        run_id = "frontend-failed-1"
        report = {
            "run_id": run_id,
            "report_complete": True,
            "completion_marker": "WHISPER_PROBE_REPORT_COMPLETE_V1",
            "status": "failed_closed",
            "error": "fixture mismatch",
        }
        raw = encoded(report)
        polled_raw, polled = collector.poll_report(ReportAdb(raw), run_id, 1.0)
        self.assertEqual(polled_raw, raw)
        self.assertEqual(polled, report)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "after report archival"):
                collector.archive_report(output, run_id, polled_raw, polled)
            archived = output / f"device_report_{run_id}.json"
            self.assertEqual(archived.read_bytes(), raw)
            self.assertFalse(
                any(path.name.endswith("_current.json") for path in output.iterdir())
            )

    def test_success_report_archive_does_not_publish_current_alias(self) -> None:
        run_id = "frontend-pass-1"
        report = {"run_id": run_id, "status": "compatibility_probe_pass"}
        raw = encoded(report)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            archived = collector.archive_report(output, run_id, raw, report)
            self.assertEqual(archived.read_bytes(), raw)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [f"device_report_{run_id}.json"],
            )

    def test_soc_identity_falls_back_when_ro_soc_model_is_empty(self) -> None:
        adb = PropertyAdb(
            {
                "ro.soc.model": "",
                "ro.vendor.qti.soc_model": "Qualcomm Technologies SM8850",
                "/sys/devices/soc0/machine": "ignored",
                "ro.board.platform": "ignored",
            }
        )
        self.assertEqual(
            collector.resolve_soc(adb),
            {
                "value": "SM8850",
                "source": "ro.vendor.qti.soc_model",
                "raw": "Qualcomm Technologies SM8850",
            },
        )

    def test_raw_process_artifacts_are_atomic_and_hash_bound(self) -> None:
        raw = {
            "status": b"Name:\tprobe\nUid:\t10510\t10510\t10510\t10510\n",
            "maps": b"1000-2000 r-xp 0 00:00 0 /data/app/libQnnHtp.so\n",
            "fd": b"1 -> /dev/fastrpc-cdsp\n2 -> anon_inode:[dmabuf]\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = collector.archive_process_artifacts(output, "frontend-raw-1", raw)
            self.assertEqual(set(records), set(raw))
            for label, record in records.items():
                path = output / record["path"]
                self.assertEqual(path.read_bytes(), raw[label])
                self.assertEqual(record["bytes"], len(raw[label]))
                self.assertEqual(
                    record["sha256"], hashlib.sha256(raw[label]).hexdigest()
                )
                self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_process_parser_recognizes_dma_name_variants(self) -> None:
        parsed = collector.device_verifier.parse_process_snapshot(
            "Uid:\t10510\t10510\t10510\t10510\n",
            "1000-2000 r-xp 0 00:00 0 /data/app/libwhisper_frontend.so\n",
            "1 -> /dev/fastrpc-cdsp\n"
            "2 -> anon_inode:[dmabuf]\n"
            "3 -> anon_inode:[dma-buf]\n"
            "4 -> anon_inode:[dma_buf]\n",
        )
        self.assertEqual(parsed["uid_fields"], [10510] * 4)
        self.assertEqual(parsed["loaded_libraries"], ["libwhisper_frontend.so"])
        self.assertTrue(parsed["fastrpc_cdsp_fd_open"])
        self.assertEqual(parsed["qcom_dmabuf_count"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
