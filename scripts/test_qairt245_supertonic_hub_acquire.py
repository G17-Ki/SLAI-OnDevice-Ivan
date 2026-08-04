#!/usr/bin/env python3
"""Focused tests for the isolated QAIRT 2.45 Supertonic lane."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qairt245_supertonic_hub_acquire.py"
SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/qairt245_supertonic_isolated_spec.json"
)
HUB_STATE = ROOT / (
    "reports/android_phone/model_sweep_20260729/qairt245_supertonic_hub_state.json"
)
NOTEBOOK = ROOT / (
    "reports/android_phone/model_sweep_20260729/qairt245_supertonic_hub_acquire.ipynb"
)
REPORT = ROOT / (
    "reports/android_phone/model_sweep_20260729/QAIRT245_SUPERTONIC_ISOLATED_LANE.md"
)
LOADER = importlib.util.spec_from_file_location("qairt245_supertonic", SCRIPT)
assert LOADER and LOADER.loader
lane = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(lane)


def file_record(root: Path, path: str, payload: bytes) -> dict[str, object]:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class Qairt245SupertonicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(SPEC.read_text(encoding="utf-8"))
        self.hub_state = json.loads(HUB_STATE.read_text(encoding="utf-8"))

    def acquired_metadata(self) -> dict[str, object]:
        return {
            "observed_utc": self.hub_state["observed_utc"],
            "hub": self.hub_state["hub"],
            "runtime": self.hub_state["runtime"],
        }

    def valid_lock_fixture(
        self, workspace: Path
    ) -> tuple[dict[str, object], dict[str, object]]:
        controller_hash = lane.sha256_file(SCRIPT)
        completed = "2026-07-30T00:00:00Z"
        stages: dict[str, object] = {"metadata": {"status": "complete"}}
        for index, name in enumerate(lane.EXPECTED_COMPONENTS, start=1):
            target = self.spec["hub"]["components"][name]["target_model"]
            stages[f"context:{name}"] = {
                "status": "complete",
                "completed_utc": completed,
                "target_model_id": target["id"],
                "authenticated_object": lane._expected_context_object(target),
                "file": {
                    "path": self.spec["acquisition"]["context_output_names"][name],
                    "bytes": target["bytes"],
                    "sha256": f"{index:064x}",
                },
            }
        stages["runtime_archive"] = {
            "status": "complete",
            "completed_utc": completed,
            "authenticated_object": lane._expected_runtime_object(self.spec),
            "file": {
                "path": f"downloads/{self.spec['runtime']['filename']}",
                "bytes": self.spec["runtime"]["bytes"],
                "sha256": "a" * 64,
            },
        }
        runtime_members = [
            lane._expected_runtime_member_record(self.spec, member)
            for member in self.spec["runtime"]["required_members"]
        ]
        stages["runtime_members"] = {
            "status": "complete",
            "completed_utc": completed,
            "files": runtime_members,
        }
        state: dict[str, object] = {
            "schema_version": 1,
            "lane_id": self.spec["lane_id"],
            "status": "complete",
            "created_utc": completed,
            "spec_sha256": lane.EXPECTED_SPEC_SHA256,
            "controller_sha256": controller_hash,
            "stages": stages,
            "updated_utc": completed,
            "host": {
                "system": "Linux",
                "machine": "x86_64",
                "python": "3.11.0",
                "free_bytes_at_start": 10_000_000_000,
            },
            "metadata": self.acquired_metadata(),
            "completed_utc": completed,
        }
        shutil.copy2(SPEC, workspace / SPEC.name)
        locked = lane._write_artifact_lock(SPEC, self.spec, workspace, state)
        lock = locked["lock"]
        stages["artifact_lock"] = {
            "status": "complete",
            "path": self.spec["acquisition"]["artifact_lock"],
        }
        lane.atomic_write_json(workspace / lane.ACQUIRE_STATE_FILENAME, state)
        return state, lock

    def write_fixture(
        self, workspace: Path, state: dict[str, object], lock: dict[str, object]
    ) -> None:
        lane.atomic_write_json(workspace / lane.ACQUIRE_STATE_FILENAME, state)
        lane.atomic_write_json(
            workspace / self.spec["acquisition"]["artifact_lock"], lock
        )

    def current_lock_hash(self, workspace: Path) -> str:
        return lane.sha256_file(workspace / self.spec["acquisition"]["artifact_lock"])

    def test_committed_spec_matches_controller_trust_root(self) -> None:
        self.assertEqual(lane.sha256_file(SPEC), lane.EXPECTED_SPEC_SHA256)
        lane.validate_spec(self.spec, SPEC)

    def test_plan_is_offline_and_requires_context_hashes(self) -> None:
        result = lane.plan(self.spec)
        self.assertFalse(result["network_used"])
        self.assertEqual(result["qairt_build"], "2.45.0.260326154327")
        self.assertEqual(
            set(result["components"]),
            {"duration_predictor", "text_encoder", "vector_estimator", "vocoder"},
        )
        for component in result["components"].values():
            self.assertEqual(
                component["sha256_status"],
                "must_be_computed_after_authenticated_download_on_linux",
            )

    def test_rejects_protected_package_reuse(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["safety_policy"]["candidate_package_id"] = "ai.liquid.lfmdemo"
        with self.assertRaisesRegex(lane.LaneError, "candidate package identity"):
            lane.validate_spec(changed)

    def test_rejects_runtime_context_build_mismatch(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["runtime"]["exact_build"] = "2.42.0.251225135753_193295"
        with self.assertRaisesRegex(lane.LaneError, "runtime/contexts build mismatch"):
            lane.validate_spec(changed)

    def test_rejects_promoted_unverified_context_hash(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["hub"]["components"]["vocoder"]["target_model"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(lane.LaneError, "unverified Hub hash"):
            lane.validate_spec(changed)

    def test_parse_zip_index(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("qairt/2.45.0/test.bin", b"test-payload")
        payload = buffer.getvalue()
        eocd = lane.parse_eocd(payload[-lane.TAIL_BYTES :])
        start = eocd["central_directory_offset"]
        end = start + eocd["central_directory_bytes"]
        index = lane.parse_central_directory(payload[start:end], eocd["entries"])
        record = index["qairt/2.45.0/test.bin"]
        self.assertEqual(record["bytes"], len(b"test-payload"))
        self.assertEqual(record["crc32"], f"{zipfile.crc32(b'test-payload'):08x}")

    def test_validate_job_observation_rejects_object_drift(self) -> None:
        expected = self.spec["hub"]["components"]["duration_predictor"]
        target = expected["target_model"]
        observed = {
            "job": {
                **expected["compile_job"],
                "hub_version": self.spec["hub"]["hub_version"],
                "status": "SUCCESS",
            },
            "device": {
                "name": self.spec["hub"]["target"]["device"],
                "os": self.spec["hub"]["target"]["os"],
                "attributes": self.spec["hub"]["target"]["required_attributes"],
            },
            "source_model": expected["source_model"],
            "target_model": {
                key: target[key]
                for key in ("id", "name", "type", "metadata", "inputs", "outputs")
            },
            "download_authorization": {
                "transport": "scoped_s3_credentials",
                "filename": target["filename"],
                "bytes": target["bytes"],
                "opaque_etag": target["opaque_etag"],
                "object_version_id": target["object_version_id"],
                "last_modified": target["last_modified"],
                "checksum_sha256": None,
            },
        }
        lane._validate_job_observation(
            "duration_predictor", observed, expected, self.spec
        )
        observed["download_authorization"]["opaque_etag"] = "0" * 32
        with self.assertRaisesRegex(lane.LaneError, "download opaque_etag drift"):
            lane._validate_job_observation(
                "duration_predictor", observed, expected, self.spec
            )

    def test_hub_loader_rejects_non_qualcomm_profile_endpoint(self) -> None:
        fake_hub = mock.Mock(__version__="0.52.0")
        fake_hub.Client.return_value.config.api_url = "https://attacker.invalid"
        fake_hub.Client.return_value.config.web_url = lane.OFFICIAL_HUB_URL
        with (
            mock.patch.dict("sys.modules", {"qai_hub": fake_hub}),
            self.assertRaisesRegex(lane.LaneError, "pinned Qualcomm origin"),
        ):
            lane._load_hub("untrusted-profile")

    def test_runtime_range_metadata_rejects_https_downgrade(self) -> None:
        response = mock.Mock(
            status_code=206,
            content=b"bounded",
            headers={"content-range": "bytes 0-6/7"},
            url="http://softwarecenter.qualcomm.com/runtime.zip",
        )
        session = mock.Mock()
        session.get.return_value = response
        with self.assertRaisesRegex(lane.LaneError, "redirected away from HTTPS"):
            lane._range_get(
                session,
                "https://softwarecenter.qualcomm.com/runtime.zip",
                "bytes=0-6",
                7,
            )

    def test_acquire_rejects_macos(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(lane.platform, "system", return_value="Darwin"),
        ):
            with self.assertRaisesRegex(lane.LaneError, "Linux-only"):
                lane.preflight_acquire(Path(directory) / "workspace", self.spec)

    def test_metadata_requires_explicit_nonhistorical_state_path(self) -> None:
        with (
            mock.patch("sys.argv", [str(SCRIPT), "metadata", "--spec", str(SPEC)]),
            self.assertRaisesRegex(lane.LaneError, "--state is required"),
        ):
            lane.main()
        with (
            mock.patch(
                "sys.argv",
                [
                    str(SCRIPT),
                    "metadata",
                    "--spec",
                    str(SPEC),
                    "--state",
                    f"/tmp/{lane.PINNED_METADATA_STATE_FILENAME}",
                ],
            ),
            self.assertRaisesRegex(lane.LaneError, "may not reuse"),
        ):
            lane.main()

    def test_colab_notebook_is_hash_pinned_session_only_and_fail_closed(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for path in (SCRIPT, SPEC, HUB_STATE):
            self.assertIn(f"'{path.name}': '{lane.sha256_file(path)}'", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        self.assertIn("qai-hub==0.52.0", code)
        self.assertEqual(code.count("'pip', 'install'"), 1)
        self.assertIn("platform.system() != 'Linux'", code)
        self.assertIn("platform.machine() != 'x86_64'", code)
        self.assertIn("QAIHUB_CLIENT_INI", code)
        self.assertIn("getpass.getpass", code)
        self.assertIn("SESSION_ROOT = Path('/content/", code)
        self.assertIn("SESSION_CONFIG = SESSION_ROOT", code)
        self.assertIn("os.O_EXCL", code)
        self.assertIn("os.fdopen", code)
        self.assertIn("chmod(0o600)", code)
        self.assertIn("finally:", code)
        self.assertIn("cleanup_session_material()", code)
        for command in ("'metadata'", "'acquire'", "'verify'"):
            self.assertIn(command, code)
        self.assertIn("'--expected-lock-sha256'", code)
        self.assertIn("artifact_lock_sha256", code)
        self.assertIn("runtime_archive_sha256", code)
        self.assertIn("runtime_member_sha256", code)
        self.assertNotIn("QAI_HUB_API_TOKEN", all_source)
        self.assertNotIn("qairt242", all_source.lower())
        self.assertNotIn("'configure'", code)
        self.assertNotIn("print(SESSION_SECRET", code)
        self.assertNotIn("print(profile", code)
        self.assertNotIn("'adb'", code.lower())

    def test_lane_report_pins_current_offline_controls(self) -> None:
        report = REPORT.read_text(encoding="utf-8")
        labels = {
            SPEC: SPEC.name,
            HUB_STATE: HUB_STATE.name,
            SCRIPT: str(SCRIPT.relative_to(ROOT)),
            NOTEBOOK: NOTEBOOK.name,
            Path(__file__).resolve(): str(Path(__file__).resolve().relative_to(ROOT)),
        }
        for path, label in labels.items():
            self.assertIn(
                f"| `{label}` | `{lane.sha256_file(path)}` |",
                report,
            )

    def test_verify_lock_accepts_only_a_spec_and_state_joined_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, lock = self.valid_lock_fixture(workspace)
            trusted_hash = self.current_lock_hash(workspace)
            with (
                mock.patch.object(lane, "verify_file_record"),
                mock.patch.object(lane, "verify_runtime_archive_contract"),
            ):
                result = lane.verify_lock(workspace, self.spec, trusted_hash)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(tuple(result["context_sha256"]), lane.EXPECTED_COMPONENTS)
            self.assertEqual(
                result["artifact_lock_sha256"],
                lane.sha256_file(workspace / self.spec["acquisition"]["artifact_lock"]),
            )
            self.assertEqual(
                result["runtime_archive_sha256"],
                lock["runtime_archive"]["file"]["sha256"],
            )

    def test_verify_lock_detects_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state, lock = self.valid_lock_fixture(workspace)
            name = "duration_predictor"
            target = self.spec["hub"]["components"][name]["target_model"]
            path = workspace / self.spec["acquisition"]["context_output_names"][name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"d" * target["bytes"])
            actual = {
                "path": self.spec["acquisition"]["context_output_names"][name],
                "bytes": target["bytes"],
                "sha256": lane.sha256_file(path),
            }
            state["stages"][f"context:{name}"]["file"] = actual
            lock["contexts"][name]["file"] = copy.deepcopy(actual)
            self.write_fixture(workspace, state, lock)
            trusted_hash = self.current_lock_hash(workspace)

            original_verify = lane.verify_file_record

            def verify_duration(candidate: Path, record: dict[str, object]) -> None:
                if record["path"] == actual["path"]:
                    original_verify(candidate, record)

            with (
                mock.patch.object(
                    lane, "verify_file_record", side_effect=verify_duration
                ),
                mock.patch.object(lane, "verify_runtime_archive_contract"),
            ):
                self.assertEqual(
                    lane.verify_lock(workspace, self.spec, trusted_hash)["status"],
                    "verified",
                )
                with path.open("r+b") as stream:
                    stream.write(b"x")
                with self.assertRaisesRegex(lane.LaneError, "SHA-256 drift"):
                    lane.verify_lock(workspace, self.spec, trusted_hash)

    def test_verify_lock_rejects_self_hashed_tiny_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state, lock = self.valid_lock_fixture(workspace)
            for name in lane.EXPECTED_COMPONENTS:
                tiny = file_record(
                    workspace,
                    self.spec["acquisition"]["context_output_names"][name],
                    name.encode("utf-8"),
                )
                state["stages"][f"context:{name}"]["file"] = tiny
                lock["contexts"][name]["file"] = copy.deepcopy(tiny)
            tiny_archive = file_record(
                workspace,
                f"downloads/{self.spec['runtime']['filename']}",
                b"counterfeit archive",
            )
            state["stages"]["runtime_archive"]["file"] = tiny_archive
            lock["runtime_archive"]["file"] = copy.deepcopy(tiny_archive)
            tiny_members = []
            for expected in lock["runtime_members"]:
                tiny = file_record(
                    workspace, expected["path"], expected["role"].encode("utf-8")
                )
                tiny["role"] = expected["role"]
                tiny_members.append(tiny)
            state["stages"]["runtime_members"]["files"] = tiny_members
            lock["runtime_members"] = copy.deepcopy(tiny_members)
            self.write_fixture(workspace, state, lock)
            with (
                self.assertRaisesRegex(
                    lane.LaneError, "context file duration_predictor bytes drift"
                ),
                mock.patch.object(lane, "verify_runtime_archive_contract"),
            ):
                lane.verify_lock(
                    workspace, self.spec, self.current_lock_hash(workspace)
                )

    def test_verify_lock_rejects_missing_extra_or_duplicate_inventory(self) -> None:
        context_mutations = {
            "missing": lambda values: values.pop("vocoder"),
            "extra": lambda values: values.__setitem__(
                "unexpected", copy.deepcopy(values["vocoder"])
            ),
        }
        for label, mutate in context_mutations.items():
            with (
                self.subTest(inventory=f"context-{label}"),
                tempfile.TemporaryDirectory() as directory,
            ):
                workspace = Path(directory)
                _, lock = self.valid_lock_fixture(workspace)
                mutate(lock["contexts"])
                lane.atomic_write_json(
                    workspace / self.spec["acquisition"]["artifact_lock"], lock
                )
                with (
                    mock.patch.object(lane, "verify_runtime_archive_contract"),
                    self.assertRaisesRegex(lane.LaneError, "context order or set"),
                ):
                    lane.verify_lock(
                        workspace, self.spec, self.current_lock_hash(workspace)
                    )

        member_mutations = {
            "missing": lambda values: values.pop(),
            "extra": lambda values: values.append(copy.deepcopy(values[-1])),
            "duplicate": lambda values: values.__setitem__(
                -1, copy.deepcopy(values[0])
            ),
        }
        for label, mutate in member_mutations.items():
            with (
                self.subTest(inventory=f"member-{label}"),
                tempfile.TemporaryDirectory() as directory,
            ):
                workspace = Path(directory)
                _, lock = self.valid_lock_fixture(workspace)
                mutate(lock["runtime_members"])
                lane.atomic_write_json(
                    workspace / self.spec["acquisition"]["artifact_lock"], lock
                )
                expected = (
                    "runtime member count"
                    if label != "duplicate"
                    else "duplicate runtime member"
                )
                with (
                    mock.patch.object(lane, "verify_file_record"),
                    mock.patch.object(lane, "verify_runtime_archive_contract"),
                    self.assertRaisesRegex(lane.LaneError, expected),
                ):
                    lane.verify_lock(
                        workspace, self.spec, self.current_lock_hash(workspace)
                    )

    def test_verify_lock_rejects_swapped_context_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, lock = self.valid_lock_fixture(workspace)
            first, second = lane.EXPECTED_COMPONENTS[:2]
            lock["contexts"][first], lock["contexts"][second] = (
                lock["contexts"][second],
                lock["contexts"][first],
            )
            lane.atomic_write_json(
                workspace / self.spec["acquisition"]["artifact_lock"], lock
            )
            with (
                mock.patch.object(lane, "verify_runtime_archive_contract"),
                self.assertRaisesRegex(lane.LaneError, "context model ID drift"),
            ):
                lane.verify_lock(
                    workspace, self.spec, self.current_lock_hash(workspace)
                )

    def test_verify_lock_rejects_fake_protected_hashes_and_extra_keys(self) -> None:
        for label in ("protected", "extra-key"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                _, lock = self.valid_lock_fixture(workspace)
                if label == "protected":
                    lock["protected_artifacts"]["main_apk_sha256"] = "0" * 64
                    expected = "protected artifact identity drift"
                else:
                    lock["untrusted_extension"] = True
                    expected = "artifact lock keys drift"
                lane.atomic_write_json(
                    workspace / self.spec["acquisition"]["artifact_lock"], lock
                )
                with (
                    mock.patch.object(lane, "verify_runtime_archive_contract"),
                    self.assertRaisesRegex(lane.LaneError, expected),
                ):
                    lane.verify_lock(
                        workspace, self.spec, self.current_lock_hash(workspace)
                    )

    def test_verify_lock_rejects_counterfeit_state_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state, lock = self.valid_lock_fixture(workspace)
            state["controller_sha256"] = "0" * 64
            self.write_fixture(workspace, state, lock)
            with (
                mock.patch.object(lane, "verify_runtime_archive_contract"),
                self.assertRaisesRegex(
                    lane.LaneError, "workspace state controller drift"
                ),
            ):
                lane.verify_lock(
                    workspace, self.spec, self.current_lock_hash(workspace)
                )

    def test_exact_sized_forged_context_checkpoint_is_freshly_replaced(self) -> None:
        class FreshTarget:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload
                self.calls = 0

            def download(self, path: str) -> str:
                self.calls += 1
                Path(path).write_bytes(self.payload)
                return path

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            state, lock = self.valid_lock_fixture(workspace)
            name = "duration_predictor"
            target = self.spec["hub"]["components"][name]["target_model"]
            destination = (
                workspace / self.spec["acquisition"]["context_output_names"][name]
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            forged_payload = b"f" * target["bytes"]
            destination.write_bytes(forged_payload)
            forged_record = {
                "path": self.spec["acquisition"]["context_output_names"][name],
                "bytes": target["bytes"],
                "sha256": hashlib.sha256(forged_payload).hexdigest(),
            }
            state["stages"][f"context:{name}"]["file"] = forged_record
            lock["contexts"][name]["file"] = copy.deepcopy(forged_record)
            self.write_fixture(workspace, state, lock)

            fresh_payload = b"a" * target["bytes"]
            downloader = FreshTarget(fresh_payload)
            result = lane._download_context(
                name,
                downloader,
                target,
                destination,
                workspace,
                state["stages"][f"context:{name}"],
            )
            self.assertEqual(downloader.calls, 1)
            self.assertEqual(
                result["file"]["sha256"], hashlib.sha256(fresh_payload).hexdigest()
            )
            self.assertEqual(lane.sha256_file(destination), result["file"]["sha256"])
            self.assertNotEqual(result["file"]["sha256"], forged_record["sha256"])

    def test_forged_complete_runtime_archive_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            changed = copy.deepcopy(self.spec)
            payload = b"same-sized self-declared archive, but not a ZIP"
            changed["runtime"]["filename"] = "forged-runtime.zip"
            changed["runtime"]["bytes"] = len(payload)
            destination = workspace / "downloads/forged-runtime.zip"
            archive_record = file_record(
                workspace, "downloads/forged-runtime.zip", payload
            )
            stage = {
                "status": "complete",
                "completed_utc": "2026-07-30T00:00:00Z",
                "authenticated_object": lane._expected_runtime_object(changed),
                "file": archive_record,
            }
            with self.assertRaisesRegex(lane.LaneError, "ZIP EOCD"):
                lane._download_runtime_archive(changed, destination, workspace, stage)

    def test_standalone_verify_requires_correct_trusted_lock_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self.valid_lock_fixture(workspace)
            with self.assertRaisesRegex(
                lane.LaneError, "trusted artifact lock SHA-256 is required"
            ):
                lane.verify_lock(workspace, self.spec, None)
            with self.assertRaisesRegex(
                lane.LaneError, "does not match the trusted acquisition record"
            ):
                lane.verify_lock(workspace, self.spec, "0" * 64)

    def test_acquire_hands_fresh_lock_hash_to_internal_verify(self) -> None:
        completed = "2026-07-30T00:00:00Z"
        observed: list[str] = []

        def context_stage(
            name: str,
            _target: object,
            expected: dict[str, object],
            _destination: Path,
            _workspace: Path,
            _prior: dict[str, object] | None,
        ) -> dict[str, object]:
            return {
                "status": "complete",
                "completed_utc": completed,
                "target_model_id": expected["id"],
                "authenticated_object": lane._expected_context_object(expected),
                "file": {
                    "path": self.spec["acquisition"]["context_output_names"][name],
                    "bytes": expected["bytes"],
                    "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
                },
            }

        runtime_stage = {
            "status": "complete",
            "completed_utc": completed,
            "authenticated_object": lane._expected_runtime_object(self.spec),
            "file": {
                "path": f"downloads/{self.spec['runtime']['filename']}",
                "bytes": self.spec["runtime"]["bytes"],
                "sha256": "a" * 64,
            },
        }
        member_stage = {
            "status": "complete",
            "completed_utc": completed,
            "files": [
                lane._expected_runtime_member_record(self.spec, member)
                for member in self.spec["runtime"]["required_members"]
            ],
        }

        def verify_handoff(
            workspace: Path,
            _spec: dict[str, object],
            expected_lock_sha256: str | None,
        ) -> dict[str, object]:
            self.assertIsNotNone(expected_lock_sha256)
            actual = lane.sha256_file(
                workspace / self.spec["acquisition"]["artifact_lock"]
            )
            self.assertEqual(expected_lock_sha256, actual)
            observed.append(actual)
            return {"status": "verified", "artifact_lock_sha256": actual}

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            targets = {name: object() for name in lane.EXPECTED_COMPONENTS}
            with (
                mock.patch.object(
                    lane,
                    "preflight_acquire",
                    return_value={
                        "system": "Linux",
                        "machine": "x86_64",
                        "python": "3.11.0",
                        "free_bytes_at_start": 10_000_000_000,
                    },
                ),
                mock.patch.object(
                    lane,
                    "collect_hub_metadata",
                    return_value=(self.hub_state["hub"], targets),
                ),
                mock.patch.object(
                    lane,
                    "collect_runtime_metadata",
                    return_value=self.hub_state["runtime"],
                ),
                mock.patch.object(lane, "_download_context", side_effect=context_stage),
                mock.patch.object(
                    lane, "_download_runtime_archive", return_value=runtime_stage
                ),
                mock.patch.object(
                    lane, "_extract_runtime_members", return_value=member_stage
                ),
                mock.patch.object(lane, "verify_lock", side_effect=verify_handoff),
            ):
                result = lane.acquire(SPEC, self.spec, workspace, None)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(len(observed), 1)

    def test_external_failure_secret_is_not_persisted_in_acquire_state(self) -> None:
        secret = "X-Amz-Credential=DO-NOT-PERSIST"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            with (
                mock.patch.object(
                    lane,
                    "preflight_acquire",
                    return_value={
                        "system": "Linux",
                        "machine": "x86_64",
                        "python": "3.11.0",
                        "free_bytes_at_start": 10_000_000_000,
                    },
                ),
                mock.patch.object(
                    lane,
                    "collect_hub_metadata",
                    side_effect=RuntimeError(f"signed URL leaked {secret}"),
                ),
                self.assertRaisesRegex(RuntimeError, "signed URL leaked"),
            ):
                lane.acquire(SPEC, self.spec, workspace, None)
            state_text = (workspace / lane.ACQUIRE_STATE_FILENAME).read_text(
                encoding="utf-8"
            )
            self.assertNotIn(secret, state_text)
            self.assertIn("details intentionally omitted", state_text)

    def test_simple_yaml_identity_parser(self) -> None:
        values = lane._parse_simple_yaml_scalars(
            "# generated\nproduct: QAIRT\nversion: 2.45.0\n  nested: ignored\n"
        )
        self.assertEqual(values, {"product": "QAIRT", "version": "2.45.0"})


if __name__ == "__main__":
    unittest.main()
