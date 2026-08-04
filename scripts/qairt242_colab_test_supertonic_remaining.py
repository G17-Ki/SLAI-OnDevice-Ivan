#!/usr/bin/env python3
"""Lightweight tests for the remaining-graph QAIRT 2.42 controller."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qairt242_colab_supertonic_remaining.py"
SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_remaining_spec.json"
)
NOTEBOOK = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_remaining.ipynb"
)
LOADER = importlib.util.spec_from_file_location("qairt242_remaining", SCRIPT)
assert LOADER and LOADER.loader
remaining = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(remaining)


class Qairt242SupertonicRemainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC.read_text(encoding="utf-8"))

    def test_committed_spec_and_base_controller_match_trust_roots(self) -> None:
        self.assertEqual(
            remaining.sha256_file(SPEC), remaining.EXPECTED_SPEC_SHA256
        )
        self.assertEqual(
            remaining.sha256_file(remaining.BASE_CONTROLLER),
            remaining.BASE_CONTROLLER_SHA256,
        )
        remaining.validate_spec(self.spec, SPEC)

    def test_plan_is_component_specific_and_host_inert(self) -> None:
        vector = remaining.plan(self.spec, "vector_estimator")
        vocoder = remaining.plan(self.spec, "vocoder")
        self.assertFalse(vector["host_execution"])
        self.assertEqual(vector["target"]["soc_model"], 87)
        self.assertEqual(vector["target"]["dsp_arch"], "v81")
        self.assertEqual(vector["fixed_inputs"]["noisy_latent"]["shape"], [1, 144, 32])
        self.assertEqual(vocoder["fixed_inputs"]["latent"]["shape"], [1, 144, 32])
        self.assertIn("CPU backend fallback", vector["forbidden"])
        self.assertIn("QAIRT 2.45 context", " ".join(vector["forbidden"]))

    def test_spec_rejects_all_fallbacks_and_component_drift(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["execution_policy"]["fallback_policy"]["allow_cpu_backend"] = True
        with self.assertRaisesRegex(remaining.RepairError, "fallback policy"):
            remaining.validate_spec(changed)
        changed = copy.deepcopy(self.spec)
        changed["components"]["vector_estimator"]["source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(remaining.RepairError, "vector_estimator hash"):
            remaining.validate_spec(changed)

    def test_converter_commands_pin_every_fixed_input_and_output(self) -> None:
        for name in remaining.ALLOWED_COMPONENTS:
            component = self.spec["components"][name]
            command = remaining.converter_command(
                Path("/sdk"),
                Path(f"/work/{name}.onnx"),
                Path(f"/work/{name}.cpp"),
                component,
                self.spec,
            )
            joined = " ".join(command)
            output_name = next(iter(component["onnx_contract"]["outputs"]))
            self.assertIn(f"--out_node {output_name}", joined)
            for tensor_name, tensor in component["onnx_contract"]["inputs"].items():
                dims = ",".join(map(str, tensor["fixed_shape"]))
                self.assertIn(f"--input_dim {tensor_name} {dims}", joined)
                self.assertIn(f"--input_dtype {tensor_name} float32", joined)
            self.assertNotIn("--qairt_version 2.45", joined)

    def test_backend_and_context_commands_pin_sm8850_v81(self) -> None:
        htp, wrapper = remaining.base.backend_configs(
            Path("/sdk"), "graph", self.spec
        )
        self.assertEqual(
            htp["devices"],
            [{"dsp_arch": "v81", "soc_model": 87, "pd_session": "unsigned"}],
        )
        self.assertEqual(htp["graphs"][0]["O"], 3)
        self.assertEqual(wrapper["soc_configs"], {"soc_model": 87})
        component = self.spec["components"]["vector_estimator"]
        command = remaining.context_command(
            Path("/sdk"),
            Path("/work/model.so"),
            Path("/work/context"),
            Path("/work/wrapper.json"),
            component,
            self.spec,
        )
        self.assertEqual(command[command.index("--soc_model") + 1], "87")
        self.assertIn(component["context_basename"], command)

    def test_component_extractor_rejects_duplicates(self) -> None:
        payload = b"pinned-model"
        component = copy.deepcopy(self.spec["components"]["vector_estimator"])
        component["source"] = {
            "archive_basename": "vector_estimator.int8.onnx",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.tar.bz2"
            with tarfile.open(archive, "w:bz2") as output:
                info = tarfile.TarInfo("model/vector_estimator.int8.onnx")
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))
            destination = root / "out/model.onnx"
            record = remaining.extract_component(archive, destination, component)
            self.assertEqual(record["sha256"], component["source"]["sha256"])

            duplicate = root / "duplicate.tar.bz2"
            with tarfile.open(duplicate, "w:bz2") as output:
                for prefix in ("a", "b"):
                    info = tarfile.TarInfo(
                        f"{prefix}/vector_estimator.int8.onnx"
                    )
                    info.size = len(payload)
                    output.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(remaining.RepairError, "must contain one"):
                remaining.extract_component(duplicate, root / "duplicate.onnx", component)

    def test_embedded_metadata_helper_is_valid_python(self) -> None:
        compile(remaining.FIXED_MODEL_HELPER, "fix_model.py", "exec")
        self.assertIn("node payload changed", remaining.FIXED_MODEL_HELPER)
        self.assertIn("metadata-only sanitized hash drift", remaining.FIXED_MODEL_HELPER)

    def test_flat_colab_control_directory_can_render_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory)
            shutil.copy2(SCRIPT, control / SCRIPT.name)
            shutil.copy2(remaining.BASE_CONTROLLER, control / remaining.BASE_CONTROLLER.name)
            shutil.copy2(SPEC, control / SPEC.name)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(control / SCRIPT.name),
                    "plan",
                    "--component",
                    "vector_estimator",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["component"], "vector_estimator")
            self.assertFalse(result["host_execution"])

    def test_notebook_is_valid_and_pins_all_controls(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn(remaining.sha256_file(SCRIPT), code)
        self.assertIn(remaining.BASE_CONTROLLER_SHA256, code)
        self.assertIn(remaining.EXPECTED_SPEC_SHA256, code)
        self.assertIn("('vector_estimator', 'vocoder')", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    def _counterfeit_workspace(
        self,
        workspace: Path,
        *,
        incomplete_manifest: bool = False,
        fake_manifest_source: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        spec = copy.deepcopy(self.spec)
        component = spec["components"]["vector_estimator"]
        source_payload = b"source-model"
        source_sha = hashlib.sha256(source_payload).hexdigest()
        component["source"].update(
            {"bytes": len(source_payload), "sha256": source_sha}
        )
        component["minimum_context_bytes"] = 1

        state = remaining.initial_state(SPEC, spec, "vector_estimator")
        timestamp = "2026-07-30T00:00:00Z"
        state.update(
            {
                "status": "complete",
                "updated_utc": timestamp,
                "completed_utc": timestamp,
                "archives": {
                    name: {
                        "status": "verified",
                        "filename": record["filename"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                        "reused": True,
                    }
                    for name, record in spec["archives"].items()
                },
                "host": {
                    "platform": "Linux-test",
                    "machine": "x86_64",
                    "ubuntu_version": "22.04",
                    "python": "3.10.12",
                    "clang": "Ubuntu clang version 14.0.0",
                    "cmake": "cmake version 3.22.1",
                    "workspace_free_bytes_at_preflight": 16 * 1024**3,
                },
            }
        )
        tool_paths = {
            *[
                spec["tool_contract"][key]
                for key in (
                    "converter",
                    "model_lib_generator",
                    "context_binary_generator",
                    "context_binary_utility",
                    "htp_backend",
                    "htp_extension",
                )
            ],
            *spec["tool_contract"]["android_runtime_inventory"],
            "bin/envsetup.sh",
            "bin/check-python-dependency",
        }
        critical = [
            {
                "path": path,
                "bytes": 1,
                "sha256": hashlib.sha256(path.encode()).hexdigest(),
            }
            for path in sorted(tool_paths)
        ]
        state["sdk"] = {
            "archive_sha256": remaining.EXPECTED_QAIRT_SHA256,
            "build_id": spec["archives"]["qairt"]["build_id"],
            "critical_files": critical,
        }
        all_packages = {"test": "1"}
        state["python_environment"] = {
            "python": "3.10.12",
            "packages": spec["host_dependencies"]["python_packages"],
            "all_packages": all_packages,
            "all_packages_sha256": remaining.base.canonical_sha256(all_packages),
        }
        critical_by_path = {item["path"]: item for item in critical}
        tool_inventory_paths = {
            *[
                spec["tool_contract"][key]
                for key in (
                    "converter",
                    "model_lib_generator",
                    "context_binary_generator",
                    "context_binary_utility",
                    "htp_backend",
                    "htp_extension",
                )
            ],
            *spec["tool_contract"]["android_runtime_inventory"],
        }
        tool_inventory = {
            path: {
                "bytes": critical_by_path[path]["bytes"],
                "sha256": critical_by_path[path]["sha256"],
            }
            for path in tool_inventory_paths
        }

        def finish_stage(
            name: str, fingerprint: str, details: dict[str, object]
        ) -> dict[str, object]:
            root = workspace / "stages" / name
            return {
                "status": "complete",
                "fingerprint": fingerprint,
                "started_utc": timestamp,
                "completed_utc": timestamp,
                "details": details,
                "outputs": remaining.base.inventory_files(root, workspace),
            }

        stages: dict[str, object] = {}
        source_root = workspace / "stages/source_model"
        source_root.mkdir(parents=True)
        source_path = source_root / component["source"]["archive_basename"]
        source_path.write_bytes(source_payload)
        source_details = {
            "archive_member": f"model/{source_path.name}",
            "filename": source_path.name,
            "bytes": len(source_payload),
            "sha256": source_sha,
        }
        stages["source_model"] = finish_stage(
            "source_model",
            remaining.base.stage_fingerprint(
                "source_model",
                remaining.EXPECTED_SPEC_SHA256,
                {
                    "component": "vector_estimator",
                    "archive_sha256": remaining.EXPECTED_SUPERTONIC_SHA256,
                    "source_sha256": source_sha,
                },
            ),
            source_details,
        )

        fixed_root = workspace / "stages/fixed_model"
        fixed_root.mkdir()
        derived = b"fixed-model"
        derived_sha = hashlib.sha256(derived).hexdigest()
        node_sha = hashlib.sha256(b"nodes").hexdigest()
        initializer_sha = hashlib.sha256(b"initializers").hexdigest()
        (fixed_root / "vector_estimator.fixed.onnx").write_bytes(derived)
        (fixed_root / "fix_model.py").write_text(
            remaining.FIXED_MODEL_HELPER, encoding="utf-8"
        )
        remaining.base.atomic_write_json(
            fixed_root / "onnx_contract.json", component["onnx_contract"]
        )
        (fixed_root / "fixed_model.log").write_text("ok\n", encoding="utf-8")
        fixed_metadata = {
            "source": {"sha256": source_sha},
            "derived": {"bytes": len(derived), "sha256": derived_sha},
            "node_payload_sha256": node_sha,
            "initializer_payload_sha256": initializer_sha,
            "removed_duplicate_io_value_info": [],
        }
        remaining.base.atomic_write_json(
            fixed_root / "fixed_model_metadata.json", fixed_metadata
        )
        fixed_details = {
            "allowed_mutation": "test",
            "source_sha256": source_sha,
            "derived_sha256": derived_sha,
            "derived_bytes": len(derived),
            "node_payload_sha256": node_sha,
            "initializer_payload_sha256": initializer_sha,
            "removed_duplicate_io_value_info": [],
        }
        stages["fixed_model"] = finish_stage(
            "fixed_model",
            remaining.base.stage_fingerprint(
                "fixed_model",
                remaining.EXPECTED_SPEC_SHA256,
                {"component": "vector_estimator", "source_sha256": source_sha},
            ),
            fixed_details,
        )

        conversion_root = workspace / "stages/qnn_conversion"
        conversion_root.mkdir()
        basename = component["model_library_basename"]
        cpp = conversion_root / f"{basename}.cpp"
        cpp.write_text(
            'getQnnGraphConfigFromInfo("test_graph", configs, count, out);\n',
            encoding="utf-8",
        )
        weights = conversion_root / f"{basename}.bin"
        weights.write_bytes(b"weights")
        (conversion_root / "qnn_converter.log").write_text("ok\n", encoding="utf-8")
        conversion_details = {
            "command": ["qnn-onnx-converter"],
            "qnn_graph_name": "test_graph",
            "cpp_sha256": remaining.sha256_file(cpp),
            "weights_sha256": remaining.sha256_file(weights),
        }
        stages["qnn_conversion"] = finish_stage(
            "qnn_conversion",
            remaining.base.stage_fingerprint(
                "qnn_conversion",
                remaining.EXPECTED_SPEC_SHA256,
                {
                    "component": "vector_estimator",
                    "fixed_sha256": derived_sha,
                },
            ),
            conversion_details,
        )

        library_root = workspace / "stages/model_library"
        library = library_root / "model_libs/x86_64-linux-clang" / f"lib{basename}.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"library")
        (library_root / "model_lib_generator.log").write_text("ok\n", encoding="utf-8")
        library_details = {
            "command": ["qnn-model-lib-generator"],
            "library_relative": library.relative_to(library_root).as_posix(),
            "library_bytes": library.stat().st_size,
            "library_sha256": remaining.sha256_file(library),
        }
        stages["model_library"] = finish_stage(
            "model_library",
            remaining.base.stage_fingerprint(
                "model_library",
                remaining.EXPECTED_SPEC_SHA256,
                {
                    "component": "vector_estimator",
                    "cpp_sha256": conversion_details["cpp_sha256"],
                    "weights_sha256": conversion_details["weights_sha256"],
                },
            ),
            library_details,
        )

        context_root = workspace / "stages/context_binary"
        context_path = context_root / "context" / f"{component['context_basename']}.bin"
        context_path.parent.mkdir(parents=True)
        context_path.write_bytes(b"context")
        for filename in (
            "backend_extensions.json",
            "htp_backend_config.json",
            "context_metadata.json",
        ):
            remaining.base.atomic_write_json(context_root / filename, {})
        for filename in ("context_binary_utility.log", "context_generator.log"):
            (context_root / filename).write_text("ok\n", encoding="utf-8")
        context_details = {
            "command": ["qnn-context-binary-generator"],
            "utility_command": ["qnn-context-binary-utility"],
            "qnn_graph_name": "test_graph",
            "target": spec["target"],
            "context_relative": context_path.relative_to(context_root).as_posix(),
            "context_bytes": context_path.stat().st_size,
            "context_sha256": remaining.sha256_file(context_path),
            "context_metadata_sha256": remaining.sha256_file(
                context_root / "context_metadata.json"
            ),
            "tool_inventory": tool_inventory,
        }
        stages["context_binary"] = finish_stage(
            "context_binary",
            remaining.base.stage_fingerprint(
                "context_binary",
                remaining.EXPECTED_SPEC_SHA256,
                {
                    "component": "vector_estimator",
                    "model_library_sha256": library_details["library_sha256"],
                    "qnn_graph_name": "test_graph",
                    "tool_inventory_sha256": remaining.base.canonical_sha256(
                        tool_inventory
                    ),
                    "target": spec["target"],
                },
            ),
            context_details,
        )

        deliverable_root = workspace / "stages/deliverable"
        deliverable_root.mkdir()
        manifest = {
            "schema_version": 1,
            "experiment_id": remaining.EXPECTED_EXPERIMENT_ID,
            "component": "vector_estimator",
            "status": "context_generated_unexecuted",
            "claim_boundary": spec["claim_boundary"],
            "created_utc": timestamp,
            "spec_sha256": remaining.EXPECTED_SPEC_SHA256,
            "controller_sha256": remaining.sha256_file(SCRIPT),
            "base_controller_sha256": remaining.BASE_CONTROLLER_SHA256,
            "host": state["host"],
            "python_environment": state["python_environment"],
            "archives": state["archives"],
            "source_model": copy.deepcopy(source_details),
            "fixed_model": fixed_details,
            "qnn_conversion": conversion_details,
            "model_library": library_details,
            "context": {
                key: value
                for key, value in context_details.items()
                if key != "tool_inventory"
            },
            "tool_inventory": tool_inventory,
            "runtime_binaries_bundled": False,
            "phone_execution_proven": False,
            "npu_execution_proven": False,
            "next_gate": "physical verification",
        }
        if fake_manifest_source:
            manifest["source_model"]["sha256"] = "0" * 64
        if incomplete_manifest:
            manifest = {"schema_version": 1}
        remaining.base.atomic_write_json(
            deliverable_root / "result_manifest.json", manifest
        )
        bundle_sha = hashlib.sha256(b"bundle").hexdigest()
        deliverable_details = {
            "manifest_status": "context_generated_unexecuted",
            "bundle_bytes": len(b"bundle"),
            "bundle_sha256": bundle_sha,
            "phone_execution_proven": False,
            "npu_execution_proven": False,
        }
        stages["deliverable"] = finish_stage(
            "deliverable",
            remaining.base.stage_fingerprint(
                "deliverable",
                remaining.EXPECTED_SPEC_SHA256,
                {
                    "component": "vector_estimator",
                    "context_sha256": context_details["context_sha256"],
                    "context_metadata_sha256": context_details[
                        "context_metadata_sha256"
                    ],
                    "controller_sha256": remaining.sha256_file(SCRIPT),
                    "base_controller_sha256": remaining.BASE_CONTROLLER_SHA256,
                },
            ),
            deliverable_details,
        )
        state["stages"] = stages
        state["result"] = deliverable_details
        return state, spec

    def test_verify_rejects_forged_complete_state_without_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            state = remaining.initial_state(SPEC, self.spec, "vector_estimator")
            state["status"] = "complete"
            with self.assertRaisesRegex(remaining.RepairError, "state field"):
                remaining.verify_completed(
                    workspace, state, self.spec, "vector_estimator"
                )

    def test_verify_rejects_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            state, spec = self._counterfeit_workspace(
                workspace, incomplete_manifest=True
            )
            with self.assertRaisesRegex(remaining.RepairError, "manifest keys"):
                remaining.verify_completed(
                    workspace, state, spec, "vector_estimator"
                )

    def test_verify_rejects_fake_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            state, spec = self._counterfeit_workspace(
                workspace, fake_manifest_source=True
            )
            with self.assertRaisesRegex(remaining.RepairError, "manifest source"):
                remaining.verify_completed(
                    workspace, state, spec, "vector_estimator"
                )

    def test_verify_rejects_fully_populated_state_missing_locked_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            state, spec = self._counterfeit_workspace(workspace)
            source = workspace / "stages/source_model/vector_estimator.int8.onnx"
            source.unlink()
            with self.assertRaisesRegex(remaining.RepairError, "missing output"):
                remaining.verify_completed(
                    workspace, state, spec, "vector_estimator"
                )

    def test_verify_rejects_forged_stage_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            state, spec = self._counterfeit_workspace(workspace)
            state["stages"]["source_model"]["fingerprint"] = "0" * 64
            with self.assertRaisesRegex(remaining.RepairError, "source_model fingerprint"):
                remaining.verify_completed(
                    workspace, state, spec, "vector_estimator"
                )

    def test_controller_has_no_245_or_secret_execution_path(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("--qairt_version 2.45", text)
        self.assertNotIn("QAI_HUB_API_TOKEN", text)
        self.assertNotIn("client.ini", text)
        self.assertIn('"npu_execution_proven": False', text)


if __name__ == "__main__":
    unittest.main()
