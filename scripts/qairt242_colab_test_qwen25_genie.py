#!/usr/bin/env python3
"""Focused host-only tests for the QAIRT 2.42 Qwen2.5 Genie lane."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qairt242_colab_qwen25_genie.py"
PREFLIGHT = ROOT / "scripts/qairt242_colab_qwen25_preflight.py"
REMOTE_WORKER = ROOT / "scripts/qairt242_colab_qwen25_remote_worker.py"
SPEC = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_qwen25_genie_spec.json"
PRIMARY_LOCK = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_qwen25_15b_genie_source_lock.json"
FOOTPRINT_LOCK = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_qwen25_05b_genie_source_lock.template.json"
NOTEBOOK = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_colab_qwen25_genie.ipynb"
LAUNCH_STATE = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_qwen25_colab_launch_state.json"
BOUNDARY = ROOT / "reports/android_phone/model_sweep_20260729/QAIRT242_QWEN25_GENIE_BOUNDARY.md"
LOCAL_GGUF = Path(
    os.environ.get(
        "LMF_QWEN25_GGUF",
        Path.home()
        / ".cache/lmf-demo/android-text-companion/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )
)
LOCAL_QAIRT = Path(
    os.environ.get(
        "LMF_QAIRT242_ARCHIVE",
        Path.home() / ".cache/lmf-demo/qairt-v2.42.0.251225.zip",
    )
)

loader = importlib.util.spec_from_file_location("qairt242_qwen25_genie", SCRIPT)
assert loader and loader.loader
lane = importlib.util.module_from_spec(loader)
loader.loader.exec_module(lane)

worker_loader = importlib.util.spec_from_file_location(
    "qairt242_colab_qwen25_remote_worker", REMOTE_WORKER
)
assert worker_loader and worker_loader.loader
worker = importlib.util.module_from_spec(worker_loader)
worker_loader.loader.exec_module(worker)


class Qairt242Qwen25GenieTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC.read_text(encoding="utf-8"))

    def test_spec_matches_controller_trust_root(self) -> None:
        self.assertEqual(lane.sha256_file(SPEC), lane.EXPECTED_SPEC_SHA256)
        lane.validate_spec(self.spec, SPEC)

    def test_primary_lock_is_approved_and_revision_pinned(self) -> None:
        lock = json.loads(PRIMARY_LOCK.read_text(encoding="utf-8"))
        identity = lane.validate_source_lock(lock, self.spec, True)
        self.assertTrue(identity["approved"])
        self.assertEqual(identity["sha256"], "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e")
        self.assertIn(identity["revision"], identity["url"])

    def test_footprint_template_can_plan_but_cannot_run(self) -> None:
        lock = json.loads(FOOTPRINT_LOCK.read_text(encoding="utf-8"))
        identity = lane.validate_source_lock(lock, self.spec, False)
        self.assertFalse(identity["approved"])
        with self.assertRaisesRegex(lane.LaneError, "not approved"):
            lane.validate_source_lock(lock, self.spec, True)

    def test_plan_is_htp_only_and_pins_mixed_tensor_gate(self) -> None:
        lock = json.loads(PRIMARY_LOCK.read_text(encoding="utf-8"))
        identity = lane.validate_source_lock(lock, self.spec, True)
        plan = lane.plan(self.spec, PRIMARY_LOCK, identity)
        self.assertFalse(plan["host_execution"])
        self.assertEqual(plan["backend"], "HTP")
        self.assertEqual(plan["target"], "chipset:SM8850;dsp_arch:v81;soc_model:87")
        self.assertEqual(plan["mixed_tensor_gate"]["actual_required_types"], {"F32": 141, "Q4_K": 169, "Q6_K": 29})
        self.assertIn("QAIRT 2.45 fallback", plan["forbidden"])

    def test_locked_local_gguf_identity_and_header_when_available(self) -> None:
        if not LOCAL_GGUF.is_file():
            self.skipTest("locked local GGUF is not present on this host")
        self.assertEqual(LOCAL_GGUF.stat().st_size, 1117320736)
        self.assertEqual(lane.sha256_file(LOCAL_GGUF), "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e")
        record = lane.inspect_gguf(LOCAL_GGUF)
        contract = self.spec["source_profiles"]["qwen2_5_1_5b_instruct_q4_k_m"]["gguf_contract"]
        lane.validate_gguf_contract(record, contract)
        self.assertEqual(record["tensor_types"], {"F32": 141, "Q4_K": 169, "Q6_K": 29})

    def test_pinned_sdk_sources_cover_q4_k_and_q6_k_when_available(self) -> None:
        if not LOCAL_QAIRT.is_file():
            self.skipTest("pinned QAIRT archive is not present on this host")
        base = "qairt/2.42.0.251225/"
        with zipfile.ZipFile(LOCAL_QAIRT) as archive:
            for item in self.spec["qairt"]["critical_members"]:
                data = archive.read(base + item["path"])
                self.assertEqual(len(data), item["bytes"])
                self.assertEqual(__import__("hashlib").sha256(data).hexdigest(), item["sha256"])
            ggml = archive.read(base + "lib/python/qti/aisw/converters/gguf_builder/ggml.py").decode()
            utils = archive.read(base + "lib/python/qti/aisw/converters/gguf_builder/utils.py").decode()
        for needle in ("def dequantize_q4_k", "def dequantize_q6_k", 'GGML_TYPES["Q4_K"]', 'GGML_TYPES["Q6_K"]'):
            self.assertIn(needle, ggml)
        self.assertIn('"Q4_K"', utils)
        self.assertIn('"Q6_K"', utils)

    def test_unknown_ggml_tensor_type_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.gguf"
            name = b"x"
            payload = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
            payload += struct.pack("<Q", len(name)) + name
            payload += struct.pack("<I", 1) + struct.pack("<Q", 1)
            payload += struct.pack("<I", 999) + struct.pack("<Q", 0)
            path.write_bytes(payload)
            with self.assertRaisesRegex(lane.LaneError, "unknown GGML tensor type"):
                lane.inspect_gguf(path)

    def trusted_complete_state(self) -> tuple[dict, dict]:
        lock = json.loads(PRIMARY_LOCK.read_text(encoding="utf-8"))
        identity = lane.validate_source_lock(lock, self.spec, True)
        result = {
            "manifest_status": self.spec["output_contract"]["required_manifest_status"],
            "phone_execution_proven": False,
            "npu_execution_proven": False,
        }
        state = {
            "schema_version": 1,
            "experiment_id": lane.EXPECTED_EXPERIMENT_ID,
            "spec_sha256": lane.EXPECTED_SPEC_SHA256,
            "controller_sha256": lane.sha256_file(SCRIPT),
            "source_lock_sha256": lane.sha256_file(PRIMARY_LOCK),
            "source_identity": identity,
            "created_utc": "2026-07-29T00:00:00Z",
            "updated_utc": "2026-07-29T00:00:00Z",
            "status": "complete",
            "stages": {},
            "host": {
                "platform": "Linux-test",
                "python": "3.10.0",
                "ram_bytes": 24_000_000_000,
                "workspace_free_bytes": 24_000_000_000,
                "scratch_free_bytes": 24_000_000_000,
            },
            "completed_utc": "2026-07-29T00:00:00Z",
            "result": result,
        }
        return state, identity

    def trusted_stage_records(self, identity: dict) -> dict:
        build_outputs: list[dict] = []
        result = {
            "manifest_status": self.spec["output_contract"]["required_manifest_status"],
            "phone_execution_proven": False,
            "npu_execution_proven": False,
        }
        return {
            "container_build": {
                "status": "complete",
                "fingerprint": lane.build_fingerprint(self.spec, identity),
                "started_utc": "2026-07-29T00:00:00Z",
                "completed_utc": "2026-07-29T00:00:00Z",
                "details": {},
                "outputs": build_outputs,
            },
            "deliverable": {
                "status": "complete",
                "fingerprint": lane.deliverable_fingerprint(
                    build_outputs, lane.sha256_file(SCRIPT)
                ),
                "completed_utc": "2026-07-29T00:00:00Z",
                "details": result,
                "outputs": [],
            },
        }

    def test_verify_rejects_zero_or_missing_required_stages(self) -> None:
        state, identity = self.trusted_complete_state()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            for stages in (
                {},
                {"container_build": {"status": "complete", "outputs": []}},
            ):
                with self.subTest(stages=sorted(stages)):
                    state["stages"] = stages
                    with self.assertRaisesRegex(lane.LaneError, "stage inventory"):
                        lane.verify_completed(
                            workspace, state, self.spec, PRIMARY_LOCK, identity
                        )

    def test_verify_rejects_incomplete_manifest(self) -> None:
        state, identity = self.trusted_complete_state()
        state["stages"] = self.trusted_stage_records(identity)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            manifest = workspace / "stages/deliverable/result_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({
                    "status": "context_generated_unexecuted",
                    "phone_execution_proven": False,
                    "npu_execution_proven": False,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(lane.LaneError, "manifest field inventory"):
                lane.verify_completed(
                    workspace, state, self.spec, PRIMARY_LOCK, identity
                )

    def test_verify_rejects_forged_stage_fingerprint(self) -> None:
        state, identity = self.trusted_complete_state()
        state["stages"] = self.trusted_stage_records(identity)
        state["stages"]["container_build"]["fingerprint"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(lane.LaneError, "build fingerprint drift"):
                lane.verify_completed(
                    Path(directory) / "workspace",
                    state,
                    self.spec,
                    PRIMARY_LOCK,
                    identity,
                )

    def test_verify_rejects_counterfeit_without_locked_inputs(self) -> None:
        state, identity = self.trusted_complete_state()
        state["stages"] = self.trusted_stage_records(identity)
        profile = self.spec["source_profiles"][identity["profile_id"]]
        contract = profile["gguf_contract"]
        manifest = dict.fromkeys(lane.RESULT_MANIFEST_FIELDS)
        manifest.update({
            "schema_version": lane.SCHEMA_VERSION,
            "experiment_id": lane.EXPECTED_EXPERIMENT_ID,
            "status": self.spec["output_contract"]["required_manifest_status"],
            "claim_boundary": self.spec["claim_boundary"],
            "created_utc": "2026-07-29T00:00:00Z",
            "spec_sha256": lane.EXPECTED_SPEC_SHA256,
            "controller_sha256": lane.sha256_file(SCRIPT),
            "source_lock_sha256": lane.sha256_file(PRIMARY_LOCK),
            "source": identity,
            "source_gguf": {
                "filename": profile["filename"],
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
                "inspection": {
                    "version": contract["version"],
                    "tensor_count": contract["tensor_count"],
                    "metadata_kv_count": contract["metadata_kv_count"],
                    "metadata": {},
                    "array_counts": {},
                    "tensor_types": contract["tensor_types"],
                },
            },
            "host": state["host"],
            "next_gate": lane.NEXT_GATE,
        })
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            manifest_path = workspace / "stages/deliverable/result_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(lane.LaneError, "locked source GGUF is missing"):
                lane.verify_completed(
                    workspace,
                    state,
                    self.spec,
                    PRIMARY_LOCK,
                    identity,
                )

    def test_verify_rejects_fake_source_identity(self) -> None:
        state, identity = self.trusted_complete_state()
        state["source_identity"] = {**identity, "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(lane.LaneError, "source identity drift"):
                lane.verify_completed(
                    Path(directory) / "workspace",
                    state,
                    self.spec,
                    PRIMARY_LOCK,
                    identity,
                )

    def test_build_helper_is_valid_python_and_has_no_device_action(self) -> None:
        compile(lane.BUILD_HELPER, "build_qwen_container.py", "exec")
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("adb ", text)
        self.assertNotIn("QAI_HUB_API_TOKEN", text)
        self.assertNotIn("2.45", lane.BUILD_HELPER)
        self.assertIn('phone_execution_proven": False', text)
        self.assertIn('npu_execution_proven": False', text)

    def test_delivery_control_hash_chain_is_coherent(self) -> None:
        control_hashes = {
            SCRIPT.name: lane.sha256_file(SCRIPT),
            SPEC.name: lane.sha256_file(SPEC),
            PRIMARY_LOCK.name: lane.sha256_file(PRIMARY_LOCK),
        }
        self.assertEqual(worker.EXPECTED, control_hashes)

        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        notebook_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for name, digest in control_hashes.items():
            self.assertIn(f"'{name}': '{digest}'", notebook_source)

        launch_state = json.loads(LAUNCH_STATE.read_text(encoding="utf-8"))
        launch_pins = {
            "controller": SCRIPT,
            "preflight": PREFLIGHT,
            "remote_worker": REMOTE_WORKER,
            "spec": SPEC,
            "source_lock": PRIMARY_LOCK,
        }
        for key, path in launch_pins.items():
            self.assertEqual(
                launch_state[key],
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": lane.sha256_file(path),
                },
            )

        boundary = BOUNDARY.read_text(encoding="utf-8")
        boundary_pins = {
            NOTEBOOK.name: NOTEBOOK,
            str(SCRIPT.relative_to(ROOT)): SCRIPT,
            SPEC.name: SPEC,
            PRIMARY_LOCK.name: PRIMARY_LOCK,
        }
        for label, path in boundary_pins.items():
            self.assertIn(
                f"| `{label}` | `{lane.sha256_file(path)}` |",
                boundary,
            )


if __name__ == "__main__":
    unittest.main()
