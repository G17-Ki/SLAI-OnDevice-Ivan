#!/usr/bin/env python3
"""Tests for the resume-safe Supertonic QAIRT 2.42 compile probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/submit_supertonic_qairt242_recompile.py"
SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_qairt242_compile_spec.json"
)
LOADER = importlib.util.spec_from_file_location("submit_supertonic_242", SCRIPT)
assert LOADER and LOADER.loader
submit = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(submit)


class FakeStatus:
    finished = True
    success = True


class FakeModel:
    model_id = "model-id"
    name = "model-name"


class FakeDevice:
    name = "device"
    os = "16"
    attributes = ["framework:qnn", "chipset:sm8850"]


class FakeJob:
    job_id = "job-id"
    model = FakeModel()
    device = FakeDevice()
    options = "--qairt_version 2.45"
    shapes = {"input": ((1, 2), "float32")}

    @staticmethod
    def get_status() -> FakeStatus:
        return FakeStatus()


class FakeFramework:
    name = "QAIRT"
    api_version = "2.45"
    api_tags = ["default"]
    full_version = "2.45.0.260326154327"


class FakeClient:
    @staticmethod
    def get_frameworks() -> list[FakeFramework]:
        return [FakeFramework()]


class SupertonicCompileProbeTest(unittest.TestCase):
    def test_committed_spec_is_valid_and_duration_is_first(self) -> None:
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        submit.validate_spec(spec)
        self.assertEqual(spec["components"]["duration_predictor"]["pilot_order"], 1)
        for record in spec["components"].values():
            self.assertIn("--qairt_version 2.42", record["compile_options"])
            self.assertNotIn("--qairt_version 2.45", record["compile_options"])

    def test_rejects_qairt_245_fallback(self) -> None:
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        spec["components"]["duration_predictor"]["compile_options"] = (
            "--qairt_version 2.45"
        )
        with self.assertRaisesRegex(RuntimeError, "QAIRT 2.42"):
            submit.validate_spec(spec)

    def test_normalizes_specs(self) -> None:
        self.assertEqual(
            submit.normalize_specs({"z": ((1, 2), "float32")}),
            {"z": [[1, 2], "float32"]},
        )

    def test_prior_job_validation_passes_exact_contract(self) -> None:
        record = {
            "prior_compile_job": "job-id",
            "source_model": {"id": "model-id", "name": "model-name"},
            "prior_options": "--qairt_version 2.45",
            "input_specs": {"input": [[1, 2], "float32"]},
        }
        target = {
            "device": "device",
            "os": "16",
            "required_attributes": ["framework:qnn", "chipset:sm8850"],
        }
        submit.validate_prior_job(FakeJob(), record, target)

    def test_prior_job_validation_rejects_source_drift(self) -> None:
        record = {
            "prior_compile_job": "job-id",
            "source_model": {"id": "wrong", "name": "model-name"},
            "prior_options": "--qairt_version 2.45",
            "input_specs": {"input": [[1, 2], "float32"]},
        }
        target = {
            "device": "device",
            "os": "16",
            "required_attributes": ["framework:qnn", "chipset:sm8850"],
        }
        with self.assertRaisesRegex(RuntimeError, "source model ID"):
            submit.validate_prior_job(FakeJob(), record, target)

    def test_framework_inventory_is_machine_readable(self) -> None:
        self.assertEqual(
            submit.framework_inventory(FakeClient()),
            [
                {
                    "name": "QAIRT",
                    "api_version": "2.45",
                    "api_tags": ["default"],
                    "full_version": "2.45.0.260326154327",
                }
            ],
        )

    def test_state_rejects_changed_spec(self) -> None:
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_copy = root / "spec.json"
            spec_copy.write_text(json.dumps(spec), encoding="utf-8")
            state_path = root / "state.json"
            state = submit.load_or_initialize_state(state_path, spec_copy, spec)
            submit.atomic_write_json(state_path, state)
            spec_copy.write_text(json.dumps({**spec, "changed": True}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different compile spec"):
                submit.load_or_initialize_state(state_path, spec_copy, spec)


if __name__ == "__main__":
    unittest.main()
