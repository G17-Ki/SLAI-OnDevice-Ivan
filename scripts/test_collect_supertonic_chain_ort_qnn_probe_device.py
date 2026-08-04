#!/usr/bin/env python3
"""Safety regressions for the read-only Supertonic chain collector."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "scripts/collect_supertonic_chain_ort_qnn_probe_device.py"


def load_collector():
    specification = importlib.util.spec_from_file_location(
        "collect_supertonic_chain_device", COLLECTOR_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import chain collector")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


collector = load_collector()


class FakeAdb:
    def __init__(self, marker: bytes) -> None:
        self.marker = marker

    def run_as(self, arguments, *, check=True):
        return types.SimpleNamespace(stdout=self.marker, returncode=0)


class CollectorTests(unittest.TestCase):
    def test_matching_failed_closed_report_is_archived_before_raise(self) -> None:
        raw = b'{"run_id":"run-1","status":"failed_closed","error":"boom"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "after report archival"):
                collector.archive_report(
                    output,
                    "run-1",
                    raw,
                    {"run_id": "run-1", "status": "failed_closed", "error": "boom"},
                )
            archived = output / "supertonic_chain_device_report_run-1.json"
            self.assertEqual(archived.read_bytes(), raw)

    def test_uid_sources_must_agree(self) -> None:
        uid, sources = collector.parse_package_uid(
            "  appId=10123\n",
            "package:ai.liquid.lfmdemo.supertonicchainprobe uid:10123",
        )
        self.assertEqual(uid, 10123)
        self.assertEqual(sources, {"dumpsys": 10123, "package_manager": 10123})
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            collector.parse_package_uid(
                "  userId=10123\n",
                "package:ai.liquid.lfmdemo.supertonicchainprobe uid:10124",
            )

    def test_explicit_existence_markers_avoid_run_as_exit_code_ambiguity(self) -> None:
        self.assertTrue(
            collector.app_private_path_exists(FakeAdb(b"PRESENT"), "files/x")
        )
        self.assertFalse(
            collector.app_private_path_exists(FakeAdb(b"ABSENT"), "files/x")
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            collector.app_private_path_exists(FakeAdb(b""), "files/x")

    def test_collector_has_no_device_mutation_commands_or_current_alias(self) -> None:
        source = COLLECTOR_PATH.read_text(encoding="utf-8")
        for forbidden in (
            '["install"',
            '["uninstall"',
            '"force-stop"',
            '"pm", "clear"',
            '"am", "start"',
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("_current.json", source)
        self.assertIn('"current_alias_published": False', source)
        self.assertIn('"verification_published": False', source)

    def test_atomic_write_rejects_stale_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            path.with_name("value.json.partial").write_bytes(b"stale")
            with self.assertRaisesRegex(RuntimeError, "stale host partial"):
                collector.atomic_write(path, b"{}\n")


if __name__ == "__main__":
    unittest.main()
