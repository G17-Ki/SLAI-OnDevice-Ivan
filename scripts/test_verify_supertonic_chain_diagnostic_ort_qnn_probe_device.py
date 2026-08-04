#!/usr/bin/env python3
"""Failure-localization tests for the no-ADB diagnostic verifier."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    ROOT / "scripts/verify_supertonic_chain_diagnostic_ort_qnn_probe_device.py"
)


def load_verifier():
    specification = importlib.util.spec_from_file_location(
        "verify_supertonic_chain_diagnostic_device_test", VERIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import diagnostic device verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


verifier = load_verifier()


def iso(unix_ms: int) -> str:
    return (
        datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def duration_session_failure(run_id: str) -> dict:
    started = 1_785_360_000_000
    completed = started + 1_234
    return {
        "schema_version": 1,
        "gate_id": verifier.GATE_ID,
        "status": "failed_closed_stage_diagnostic",
        "error_type": "ai.onnxruntime.OrtException",
        "error": "CPU fallback disabled",
        "cpu_ep_fallback_disabled": True,
        "full_neural_chain": False,
        "chained_pcm_tensor": False,
        "full_tts": False,
        "audio_file_output": False,
        "audio_playback": False,
        "audio_quality_accepted": False,
        "host_reference_comparisons_promoted": False,
        "production_integration": False,
        "accepted_packages_mutated": False,
        "diagnostic_schema_version": 2,
        "diagnostic_marker": verifier.DIAGNOSTIC_MARKER,
        "completed_stages": ["payload_install", "runtime_initialization"],
        "completed_stage_count": 2,
        "completed_stage_reports": {
            "payload_install": {
                "component": "payload_install",
                "manifest_verified": True,
                "assets_verified": True,
                "native_libraries_verified": True,
                "model_storage": (
                    "/data/user/0/ai.liquid.lfmdemo.supertonicchaindiagprobe/"
                    "files/supertonic_chain_diagnostic_probe/payload-v2"
                ),
                "runtime_storage": (
                    "/data/app/~~token/ai.liquid.lfmdemo."
                    "supertonicchaindiagprobe-token==/lib/arm64"
                ),
            },
            "runtime_initialization": {
                "component": "runtime_initialization",
                "native_libraries_loaded": True,
                "ort_version": "1.25.0",
                "available_providers": [
                    "CPUExecutionProvider",
                    "QNNExecutionProvider",
                ],
                "qnn_provider_available": True,
                "fixture_files_loaded": True,
            },
        },
        "sessions_created": {
            "duration_predictor": False,
            "text_encoder": False,
            "vector_estimator": False,
            "vocoder": False,
        },
        "completed_vector_steps": 0,
        "completed_vector_step_reports": [],
        "diagnostic_outcome": "failure_localized",
        "failure_stage": "duration_predictor",
        "failure_phase": "session_create",
        "evidence_schema_version": 1,
        "package": verifier.PACKAGE,
        "run_id": run_id,
        "run_trigger": "automation_intent",
        "started_at_unix_ms": started,
        "started_at_utc": iso(started),
        "completed_at_unix_ms": completed,
        "completed_at_utc": iso(completed),
        "elapsed_wall_ms": completed - started,
        "elapsed_monotonic_ms": 1_234.0,
        "report_complete": True,
        "completion_marker": verifier.COMPLETION_MARKER,
        "report_path": verifier.REPORT_PATH,
    }


class DiagnosticDeviceVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = verifier.load_object(verifier.DEFAULT_SPEC)
        _, cls.source = verifier.host_verifier.validate_spec(
            cls.spec, verifier.DEFAULT_SPEC.resolve()
        )

    def test_complete_failed_report_is_a_verified_diagnostic_outcome(self) -> None:
        result = verifier.verify_report(
            duration_session_failure("diag-test-r1"),
            self.spec,
            self.source,
            "diag-test-r1",
        )
        self.assertEqual(result["kind"], "failure")
        self.assertEqual(result["failure_stage"], "duration_predictor")
        self.assertEqual(result["failure_phase"], "session_create")

    def test_completed_stage_prefix_cannot_be_forged(self) -> None:
        report = duration_session_failure("diag-test-r2")
        report["completed_stages"] = ["payload_install", "text_encoder"]
        with self.assertRaisesRegex(RuntimeError, "ordered prefix"):
            verifier.verify_report(
                report, self.spec, self.source, "diag-test-r2"
            )

    def test_vector_step_index_is_rejected_on_non_vector_failure(self) -> None:
        report = duration_session_failure("diag-test-r3")
        report["failure_vector_step"] = 0
        with self.assertRaisesRegex(RuntimeError, "index presence"):
            verifier.verify_report(
                report, self.spec, self.source, "diag-test-r3"
            )


if __name__ == "__main__":
    unittest.main()
