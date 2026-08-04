#!/usr/bin/env python3
"""Lightweight tests for the QAIRT 2.42 Colab duration canary."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qairt242_colab_supertonic_duration_canary.py"
SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_duration_spec.json"
)
NOTEBOOK = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_duration_canary.ipynb"
)
MANUAL = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_duration_canary.md"
)
LOADER = importlib.util.spec_from_file_location("qairt242_colab_canary", SCRIPT)
assert LOADER and LOADER.loader
canary = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(canary)


class Qairt242ColabCanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC.read_text(encoding="utf-8"))

    def test_committed_spec_matches_controller_trust_root(self) -> None:
        self.assertEqual(canary.sha256_file(SPEC), canary.EXPECTED_SPEC_SHA256)
        canary.validate_spec(self.spec, SPEC)

    def test_plan_is_duration_only_and_has_no_host_execution(self) -> None:
        plan = canary.plan(self.spec)
        self.assertEqual(plan["component"], "duration_predictor")
        self.assertFalse(plan["host_execution"])
        self.assertEqual(plan["target"]["soc_model"], 87)
        self.assertEqual(plan["target"]["dsp_arch"], "v81")
        self.assertEqual(plan["fixed_inputs"]["text_ids"]["dtype"], "int32")
        self.assertIn("QAIRT 2.45 fallback", plan["forbidden"])
        self.assertNotIn("2.45", plan["archives"]["qairt"]["url"])

    def test_converter_command_locks_all_fixed_inputs(self) -> None:
        command = canary.converter_command(
            Path("/sdk"), Path("/work/fixed.onnx"), Path("/work/model.cpp"), self.spec
        )
        joined = " ".join(command)
        self.assertIn("--out_node duration", joined)
        self.assertIn("--input_dim text_ids 1,32", joined)
        self.assertIn("--input_dtype text_ids int32", joined)
        self.assertIn("--input_dim style_dp 1,8,16", joined)
        self.assertIn("--input_dim text_mask 1,1,32", joined)
        self.assertNotIn("--overwrite_model_prefix", command)
        self.assertNotIn("2.45", joined)

    def test_backend_and_context_commands_lock_sm8850_v81(self) -> None:
        htp, wrapper = canary.backend_configs(
            Path("/sdk"), "duration_graph", self.spec
        )
        self.assertEqual(htp["devices"], [{
            "dsp_arch": "v81",
            "soc_model": 87,
            "pd_session": "unsigned",
        }])
        self.assertEqual(htp["graphs"][0]["graph_names"], ["duration_graph"])
        self.assertEqual(htp["graphs"][0]["O"], 3)
        self.assertEqual(htp["graphs"][0]["fp16_relaxed_precision"], 1)
        self.assertEqual(wrapper["soc_configs"], {"soc_model": 87})
        command = canary.context_command(
            Path("/sdk"),
            Path("/work/model.so"),
            Path("/work/context"),
            Path("/work/backend.json"),
            self.spec,
        )
        self.assertEqual(command[command.index("--soc_model") + 1], "87")
        self.assertNotIn("2.45", " ".join(command))

    def test_duration_extractor_rejects_duplicates_and_accepts_one_pinned_member(self) -> None:
        payload = b"small-duration-model"
        local_spec = copy.deepcopy(self.spec)
        local_spec["component"].update(
            {
                "archive_basename": "duration_predictor.int8.onnx",
                "source_bytes": len(payload),
                "source_sha256": __import__("hashlib").sha256(payload).hexdigest(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.tar.bz2"
            with tarfile.open(archive, "w:bz2") as output:
                info = tarfile.TarInfo("model/duration_predictor.int8.onnx")
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))
                other = tarfile.TarInfo("model/text_encoder.int8.onnx")
                other.size = 1
                output.addfile(other, io.BytesIO(b"x"))
            target = root / "out/duration_predictor.int8.onnx"
            record = canary.extract_duration_model(archive, target, local_spec)
            self.assertEqual(record["sha256"], local_spec["component"]["source_sha256"])
            self.assertEqual(target.read_bytes(), payload)

            duplicate = root / "duplicate.tar.bz2"
            with tarfile.open(duplicate, "w:bz2") as output:
                for prefix in ("a", "b"):
                    info = tarfile.TarInfo(
                        f"{prefix}/duration_predictor.int8.onnx"
                    )
                    info.size = len(payload)
                    output.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(canary.CanaryError, "must contain one"):
                canary.extract_duration_model(duplicate, root / "duplicate.onnx", local_spec)

    def test_completed_stage_is_reused_only_while_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            state_path = workspace / "state.json"
            state: dict[str, object] = {"stages": {}}
            calls = []

            def action(partial: Path) -> dict[str, object]:
                calls.append(True)
                (partial / "result.bin").write_bytes(b"trusted")
                return {"kind": "test"}

            record = canary.run_stage(
                "unit",
                "fingerprint",
                workspace,
                state_path,
                state,
                action,
            )
            self.assertEqual(record["status"], "complete")
            canary.run_stage(
                "unit",
                "fingerprint",
                workspace,
                state_path,
                state,
                action,
            )
            self.assertEqual(len(calls), 1)
            (workspace / "stages/unit/result.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(canary.CanaryError, "drift"):
                canary.run_stage(
                    "unit",
                    "fingerprint",
                    workspace,
                    state_path,
                    state,
                    action,
                )

    def test_graph_name_parser_requires_exactly_one_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "model.cpp"
            cpp.write_text(
                'getQnnGraphConfigFromInfo("duration_graph", configs, count, out);',
                encoding="utf-8",
            )
            self.assertEqual(canary.parse_graph_name(cpp), "duration_graph")
            cpp.write_text("no graph here", encoding="utf-8")
            with self.assertRaisesRegex(canary.CanaryError, "determine one"):
                canary.parse_graph_name(cpp)

    def test_embedded_fixed_model_helper_is_valid_python(self) -> None:
        compile(canary.FIXED_MODEL_HELPER, "fix_model.py", "exec")

    def test_deterministic_bundle_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "payload"
            source.mkdir()
            (source / "b.txt").write_text("b\n", encoding="utf-8")
            (source / "a.txt").write_text("a\n", encoding="utf-8")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            canary.deterministic_tar_gz(source, first)
            canary.deterministic_tar_gz(source, second)
            self.assertEqual(canary.sha256_file(first), canary.sha256_file(second))

    def test_controller_contains_no_secret_or_qairt_fallback_contract(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("client.ini", text)
        self.assertNotIn("QAI_HUB_API_TOKEN", text)
        self.assertNotIn("--qairt_version 2.45", text)
        self.assertIn("phone_execution_proven\": False", text)

    def test_notebook_is_valid_and_pins_uploaded_controls(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn(canary.sha256_file(SCRIPT), source)
        self.assertIn(canary.sha256_file(SPEC), source)
        self.assertIn("python3.10", source)
        self.assertIn("clang-14", source)
        self.assertIn("'run'", source)
        self.assertIn("'verify'", source)
        self.assertNotIn("QAI_HUB_API_TOKEN", source)
        manual = MANUAL.read_text(encoding="utf-8")
        self.assertIn("optional ahead-of-time context experiment", manual)
        self.assertIn("context_generated_unexecuted", manual)
        self.assertIn(canary.sha256_file(NOTEBOOK), manual)


if __name__ == "__main__":
    unittest.main()
