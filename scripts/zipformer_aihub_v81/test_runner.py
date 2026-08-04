from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import runner


class RunnerTests(unittest.TestCase):
    def test_thread_cap_is_forced(self) -> None:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "ORT_NUM_THREADS",
        ):
            self.assertEqual(runner.os.environ[name], "2")

    def test_seeded_inputs_are_reproducible(self) -> None:
        first = runner.make_inputs(runner.COMPONENTS["joiner"], 42)
        second = runner.make_inputs(runner.COMPONENTS["joiner"], 42)
        self.assertEqual(list(first), ["encoder_out", "decoder_out"])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_integer_comparison_is_exact(self) -> None:
        golden = np.array([1, 2], dtype=np.int64)
        same_values = np.array([1, 2], dtype=np.int32)
        different = np.array([1, 3], dtype=np.int32)
        self.assertTrue(runner.compare_array(same_values, golden, 1.0, 1.0)["passed"])
        self.assertFalse(runner.compare_array(different, golden, 1.0, 1.0)["passed"])
        float_values = np.array([1.0, 2.0], dtype=np.float32)
        self.assertFalse(runner.compare_array(float_values, golden, 1.0, 1.0)["passed"])

    def test_profile_classifies_every_operation(self) -> None:
        profile = {
            "execution_detail": [
                {"name": "a", "compute_unit": "NPU"},
                {"name": "b", "compute_unit": "HTP"},
                {"name": "c", "compute_unit": "CPU"},
                {"name": "d", "compute_unit": "GPU"},
                {"name": "e", "compute_unit": "NPU (CPU fallback)"},
            ]
        }
        result = runner.classify_profile(profile)
        self.assertEqual(result["operation_count"], 5)
        self.assertEqual(result["counts"]["hexagon_npu"], 2)
        self.assertEqual(result["counts"]["cpu_fallback"], 2)
        self.assertEqual(result["counts"]["unsupported_or_export_failure"], 1)
        self.assertFalse(result["strict_npu_pass"])

    def test_canary_gate_blocks_encoder(self) -> None:
        state = {
            "components": {
                "decoder": {"status": "passed"},
                "joiner": {"status": "completed_with_warning"},
                "encoder": {"status": "not_started"},
            }
        }
        gate = runner.evaluate_canary_gate(state)
        self.assertEqual(gate["status"], "blocked")
        self.assertFalse(gate["encoder_submission_allowed"])

    def test_canary_gate_requires_all_recorded_evidence(self) -> None:
        accepted_result = {
            "accepted": True,
            "strict_npu_pass": True,
            "all_numerics_pass": True,
        }
        state = {
            "components": {
                "decoder": {"status": "passed", "result": dict(accepted_result)},
                "joiner": {"status": "passed", "result": dict(accepted_result)},
                "encoder": {"status": "not_started"},
            }
        }
        gate = runner.evaluate_canary_gate(state)
        self.assertEqual(gate["status"], "passed")
        self.assertTrue(gate["encoder_submission_allowed"])
        state["components"]["joiner"]["result"]["strict_npu_pass"] = False
        gate = runner.evaluate_canary_gate(state)
        self.assertEqual(gate["status"], "blocked")
        self.assertFalse(gate["encoder_submission_allowed"])

    def test_context_binary_output_names_are_flattened(self) -> None:
        model = SimpleNamespace(
            output_spec={
                "decoder_model": [
                    SimpleNamespace(name="decoder_out"),
                    SimpleNamespace(name="state_out"),
                ]
            }
        )
        self.assertEqual(
            runner.target_output_names(model), ["decoder_out", "state_out"]
        )

    def test_unexpected_remote_output_fails_numerical_gate(self) -> None:
        model = SimpleNamespace(
            output_spec={"decoder_model": [SimpleNamespace(name="decoder_out")]}
        )
        comparison, _ = runner.compare_remote_outputs(
            {
                "decoder_out": [np.zeros((1, 2), dtype=np.float32)],
                "unexpected": [np.zeros((1,), dtype=np.float32)],
            },
            {"decoder_out": np.zeros((1, 2), dtype=np.float32)},
            model,
            0.0,
            0.0,
        )
        self.assertFalse(comparison["all_numerics_pass"])
        self.assertEqual(comparison["unexpected_remote_outputs"], ["unexpected"])

    def test_torch_quantization_marker_fails_float_gate(self) -> None:
        class FakeModel:
            graph = "graph(%x): quantized::linear_dynamic(%x)"

            @staticmethod
            def named_parameters():
                return []

            @staticmethod
            def modules():
                return []

            @staticmethod
            def named_buffers():
                return []

        result = runner.inspect_torch_float_source(
            FakeModel(), SimpleNamespace(__version__="test")
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["graph_markers"], ["quantized::"])

    def test_quantized_filename_markers_cover_int16_and_generic_wxa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for filename in (
                "encoder-int16.pt",
                "encoder-uint16.pt",
                "decoder-w16a16.pt",
                "joiner-w123a456.pt",
            ):
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_bytes(b"floating point fixture")
                    result = runner.quantization_marker_scan(path)
                    self.assertTrue(result["filename_suggests_quantized"])
                    self.assertFalse(result["passed_float_source_gate"])

    def test_torch_int16_buffers_fail_float_gate(self) -> None:
        class FakeModel:
            graph = "graph(%x): aten::linear(%x)"

            @staticmethod
            def named_parameters():
                return []

            @staticmethod
            def modules():
                return []

            @staticmethod
            def named_buffers():
                return [
                    ("signed", SimpleNamespace(dtype="torch.int16")),
                    ("unsigned", SimpleNamespace(dtype="torch.uint16")),
                ]

        result = runner.inspect_torch_float_source(
            FakeModel(), SimpleNamespace(__version__="test")
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            [item["name"] for item in result["quantized_buffers"]],
            ["signed", "unsigned"],
        )

    def test_onnx_quantization_ops_fail_semantic_float_gate(self) -> None:
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError as error:
            self.skipTest(f"onnx is unavailable: {error}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for op_type in ("ConvInteger", "MatMulInteger", "DynamicQuantizeLinear"):
                with self.subTest(op_type=op_type):
                    path = root / f"{op_type}.onnx"
                    input_count = 1 if op_type == "DynamicQuantizeLinear" else 2
                    node = helper.make_node(
                        op_type,
                        ["x"] * input_count,
                        ["out"],
                        name="quantized_node",
                    )
                    graph = helper.make_graph(
                        [node],
                        "quantized_graph",
                        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
                    )
                    onnx.save_model(helper.make_model(graph), path)
                    result = runner.quantization_marker_scan(path)
                    inspection = result["onnx_semantic_inspection"]
                    self.assertTrue(inspection["inspection_complete"])
                    self.assertIn(
                        op_type,
                        [item["op_type"] for item in inspection["quantized_ops"]],
                    )
                    self.assertFalse(result["passed_float_source_gate"])

    def test_onnx_external_int16_initializer_is_detected_without_payload(self) -> None:
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError as error:
            self.skipTest(f"onnx is unavailable: {error}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "model.onnx"
            initializer = numpy_helper.from_array(
                np.array([1, 2], dtype=np.int16), name="weight"
            )
            graph = helper.make_graph(
                [helper.make_node("Identity", ["weight"], ["out"])],
                "external_int16_graph",
                [],
                [helper.make_tensor_value_info("out", TensorProto.INT16, [2])],
                [initializer],
            )
            onnx.save_model(
                helper.make_model(graph),
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.data",
                size_threshold=0,
            )
            (root / "weights.data").unlink()

            result = runner.quantization_marker_scan(path)
            inspection = result["onnx_semantic_inspection"]
            self.assertTrue(inspection["inspection_complete"])
            self.assertEqual(
                inspection["low_precision_integer_initializers"],
                [
                    {
                        "name": "weight",
                        "location": "graph",
                        "kind": "initializer",
                        "dtype": "INT16",
                        "external_data": True,
                    }
                ],
            )
            self.assertEqual(
                inspection["external_data_initializers"],
                inspection["low_precision_integer_initializers"],
            )
            self.assertFalse(result["passed_float_source_gate"])

    def test_float_onnx_passes_semantic_float_gate(self) -> None:
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError as error:
            self.skipTest(f"onnx is unavailable: {error}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.onnx"
            initializer = numpy_helper.from_array(
                np.array([1.0, 2.0], dtype=np.float32), name="weight"
            )
            graph = helper.make_graph(
                [helper.make_node("Identity", ["weight"], ["out"])],
                "float_graph",
                [],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
                [initializer],
            )
            onnx.save_model(helper.make_model(graph), path)
            result = runner.quantization_marker_scan(path)
            self.assertTrue(result["onnx_semantic_inspection"]["inspection_complete"])
            self.assertTrue(result["onnx_semantic_inspection"]["passed"])
            self.assertTrue(result["passed_float_source_gate"])

    def test_quoted_json_secret_is_redacted(self) -> None:
        value = '{"api_' + 'key": "dummy-value"}'
        redacted = runner._redact_string(value, ())
        self.assertNotIn("dummy-value", redacted)
        self.assertIn("<redacted>", redacted)

    def test_generic_android_absolute_path_is_redacted(self) -> None:
        redacted = runner._redact_string(
            "runtime file: /data/local/tmp/context.bin", ()
        )
        self.assertNotIn("/data/local/tmp", redacted)
        self.assertIn("<absolute_path>", redacted)

    def test_torch_fixture_requires_runtime_float_inspection(self) -> None:
        static_only = {"float_source_gate": {"passed_float_source_gate": True}}
        self.assertFalse(
            runner.fixture_float_gate_valid(static_only, Path("decoder.pt"))
        )
        inspected = {
            "float_source_gate": {
                "passed_float_source_gate": True,
                "torch_runtime_inspection": {"passed": True},
            }
        }
        self.assertTrue(runner.fixture_float_gate_valid(inspected, Path("decoder.pt")))

    def test_onnx_fixture_requires_semantic_float_inspection(self) -> None:
        static_only = {"float_source_gate": {"passed_float_source_gate": True}}
        self.assertFalse(
            runner.fixture_float_gate_valid(static_only, Path("decoder.onnx"))
        )
        inspected = {
            "float_source_gate": {
                "passed_float_source_gate": True,
                "onnx_semantic_inspection": {
                    "inspection_complete": True,
                    "passed": True,
                },
            }
        }
        self.assertTrue(
            runner.fixture_float_gate_valid(inspected, Path("decoder.onnx"))
        )

    def test_atomic_json_redacts_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "state.json"
            secret_path = root / "private" / "model.onnx"
            credential_label = "api_" + "key="
            runner.atomic_write_json(
                destination,
                {"path": str(secret_path), "message": credential_label + "dummy-value"},
                (root,),
            )
            text = destination.read_text(encoding="utf-8")
            parsed = json.loads(text)
            self.assertNotIn(str(root), text)
            self.assertNotIn("dummy-value", text)
            self.assertIn("<absolute_path>", parsed["path"])
            self.assertIn("<redacted>", parsed["message"])

    def test_offline_preflight_redacts_temp_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = argparse.Namespace(
                model_dir=root / "models",
                output_dir=root / "run",
                decoder_model="decoder.onnx",
                joiner_model="joiner.onnx",
                encoder_model="encoder.onnx",
                seed=20260723,
                confirm_cloud=False,
            )
            report = runner.offline_preflight(args)
            self.assertFalse(report["network_contacted"])
            self.assertFalse(report["readiness"]["cloud_run_ready"])
            text = (root / "run" / "preflight.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root), text)


if __name__ == "__main__":
    unittest.main()
