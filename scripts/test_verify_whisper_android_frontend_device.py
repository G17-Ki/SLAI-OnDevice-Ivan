#!/usr/bin/env python3
"""Tests for the fail-closed bundled-WAV frontend device verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_whisper_android_frontend_device.py"
FRONTEND_SPEC = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_android_frontend_probe_spec.json"
)
BASE_SPEC = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_ort_qnn_probe_spec.json"
)
LOCAL_VERIFY = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "whisper_android_frontend_local_verify.json"
)
LOADER = importlib.util.spec_from_file_location("verify_frontend_device", SCRIPT)
assert LOADER and LOADER.loader
verify = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(verify)


def digest(value: dict) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


class FrontendDeviceVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frontend_spec = json.loads(FRONTEND_SPEC.read_text())
        self.base_spec = json.loads(BASE_SPEC.read_text())
        self.run_id = "frontend-20260730T010203Z-a1b2c3"
        package = self.frontend_spec["target"]["package"]
        base_gate = self.frontend_spec["base_probe"]["gate_id"]
        frontend_gate = self.frontend_spec["gate_id"]
        contract = self.frontend_spec["device_output_contract"]
        native_sha = self.frontend_spec["frontend_native_library"]["sha256"]
        self.local = json.loads(LOCAL_VERIFY.read_text())
        en = {
            **contract["frontend"]["fixtures"]["en"],
            "produced_tensor_sha256": "1" * 64,
            "exact_sha256_match": False,
            "differing_uint16_values": 105,
            "differing_fraction": 105 / 240_000,
            "max_uint16_delta": 1,
            "frontend_compute_ms": 12.0,
            "bounded_gate_passed": True,
        }
        ja = {
            **contract["frontend"]["fixtures"]["ja"],
            "produced_tensor_sha256": "2" * 64,
            "exact_sha256_match": False,
            "differing_uint16_values": 70,
            "differing_fraction": 70 / 240_000,
            "max_uint16_delta": 1,
            "frontend_compute_ms": 9.0,
            "bounded_gate_passed": True,
        }
        step_times = [1.0, 1.1, 1.2, 1.3]
        decoder_execute = round(sum(step_times), 3)
        self.report = {
            "schema_version": 1,
            "evidence_schema_version": 1,
            "gate_id": base_gate,
            "package": package,
            "uid": 10510,
            "run_id": self.run_id,
            "run_trigger": "automation_intent",
            "started_at_unix_ms": 1785354123000,
            "started_at_utc": "2026-07-29T19:42:03Z",
            "completed_at_unix_ms": 1785354124000,
            "completed_at_utc": "2026-07-29T19:42:04Z",
            "elapsed_wall_ms": 1000,
            "elapsed_monotonic_ms": 1000.0,
            "report_complete": True,
            "completion_marker": "WHISPER_PROBE_REPORT_COMPLETE_V1",
            "report_path": contract["report_path"],
            "accepted_packages_mutated": False,
            "ort_version": "1.25.0",
            "available_providers": ["CPUExecutionProvider", "QNNExecutionProvider"],
            "qnn_provider_requested": True,
            "cpu_ep_fallback_disabled": True,
            "backend_path": "/data/app/pkg/lib/arm64/libQnnHtp.so",
            "model_storage": f"/data/user/0/{package}/files/whisper_probe/payload-v1",
            "runtime_storage": "/data/app/pkg/lib/arm64",
            "host_transport_required": False,
            "real_audio_transcription": False,
            "encoder_session_created": True,
            "decoder_session_created": True,
            "synthetic_decode": {
                "feature_source": "bundled_question_en_native_frontend",
                "feature_dtype": "uint16",
                "feature_zero_point": 32072,
                "encoder_execute_passed": True,
                "decoder_execute_passed": True,
                "decoder_steps": 4,
                "token_ids": [50259, 10, 11, 12],
                "first_token_id": 50259,
                "attention_unmasked_indices": [199, 198, 197, 196],
                "decoded_base_text": "",
                "encoder_execute_ms": 10.0,
                "decoder_execute_ms": decoder_execute,
                "decoder_loop_ms": 6.0,
                "decoder_step_ms": step_times,
                "semantic_transcript_claimed": False,
            },
            "bundled_audio_frontend": {
                "schema_version": 1,
                "gate_id": frontend_gate,
                "manifest_sha256": self.frontend_spec["frontend_manifest"]["sha256"],
                "implementation": "jni_persistent_kissfft_400_real",
                "feature_dtype": "uint16",
                "feature_shape": [1, 80, 3000],
                "feature_layout": "mel_major_frame_minor",
                "fixtures": {"en": en, "ja": ja},
                "frontend_compute_ms": {"en": 12.0, "ja": 9.0, "aggregate": 21.0},
                "frontend_verification_total_ms": 25.0,
                "verification_policy": {
                    "max_uint16_delta": 1,
                    "max_differing_values": 240,
                    "max_differing_fraction": 0.001,
                    "target_acceptance_promoted": False,
                },
                "bounded_gate_passed": True,
                "real_audio_frontend": False,
                "real_audio_transcription": False,
                "microphone_input": False,
                "status": "bundled_wav_frontend_bound_pass",
                "claim_boundary": (
                    "This verifies only bundled WAV preprocessing on the executing ABI. "
                    "It does not enable microphone input or claim a semantic transcription."
                ),
                "native_library_sha256": native_sha,
            },
            "timing_ms": {
                "encoder_session_load": 20.0,
                "decoder_session_load": 30.0,
                "encoder_execute": 10.0,
                "decoder_execute": decoder_execute,
                "decoder_loop": 6.0,
                "decoder_step": step_times,
                "execute": 20.0,
                "total": 80.0,
            },
            "status": "compatibility_probe_pass",
            "claim_boundary": (
                "Session/run success is probe-package feasibility evidence only; "
                "it is not final ai.liquid.lfmdemo.runtime integration or real-audio acceptance."
            ),
        }
        self.local_sha = digest(self.local)
        self.report_sha = digest(self.report)
        expected_models = {
            item["path"].removeprefix("whisper_probe/models/"): item["bytes"]
            for item in self.base_spec["required_assets"]
            if item["path"].startswith("whisper_probe/models/")
        }
        accepted = copy.deepcopy(self.frontend_spec["accepted_packages"])
        self.evidence = {
            "schema_version": 2,
            "gate_id": frontend_gate,
            "package": package,
            "run_id": self.run_id,
            "run": {"requested_run_id": self.run_id, "trigger": "automation_intent"},
            "device": {
                "model": "NX809J",
                "soc": "SM8850",
                "abi": "arm64-v8a",
                "airplane_mode_on": 1,
            },
            "installed_package": {
                "package": package,
                "version_code": 1,
                "version_name": "0.1.0-ort-qnn-probe",
                "uid": 10510,
                "pid": 1234,
                "apk_path": "/data/app/pkg/base.apk",
                **self.frontend_spec["apk"],
            },
            "accepted_packages_before": accepted,
            "accepted_packages": copy.deepcopy(accepted),
            "process": {
                "pid": 1234,
                "uid_fields": [10510, 10510, 10510, 10510],
                "report_owner_uid": 10510,
                "loaded_libraries": sorted(verify.REQUIRED_PROCESS_LIBRARIES),
                "fastrpc_cdsp_fd_open": True,
                "qcom_dmabuf_count": 2,
                "raw_artifacts": {
                    "status": {
                        "path": "proc_status.txt",
                        "bytes": 1,
                        "sha256": "3" * 64,
                    },
                    "maps": {
                        "path": "proc_maps.txt",
                        "bytes": 1,
                        "sha256": "4" * 64,
                    },
                    "fd": {
                        "path": "proc_fd.txt",
                        "bytes": 1,
                        "sha256": "5" * 64,
                    },
                },
            },
            "resident_payload": {
                "root": self.report["model_storage"],
                "kib": 352027,
                "model_files": expected_models,
            },
            "installed_frontend_library": {
                "path": "/data/app/pkg/lib/arm64/libwhisper_frontend.so",
                "bytes": self.frontend_spec["frontend_native_library"]["bytes"],
                "sha256": native_sha,
            },
            "local_verifier_report": {"sha256": self.local_sha, "bytes": 1},
            "device_report": {
                "device_path": contract["report_path"],
                "partial_device_path": contract["report_path"] + ".partial",
                "partial_exists_after_capture": False,
                "sha256": self.report_sha,
                "bytes": 1,
            },
            "partial_report_absent": True,
        }

    def run_verify(self) -> dict:
        return verify.verify_documents(
            self.frontend_spec,
            self.base_spec,
            self.local,
            self.report,
            self.evidence,
            expected_run_id=self.run_id,
            local_verify_sha256=self.local_sha,
            device_report_sha256=self.report_sha,
            frontend_spec_path=FRONTEND_SPEC,
            base_spec_path=BASE_SPEC,
        )

    def stage_file_bundle(
        self,
        temporary: Path,
        *,
        frontend_spec: dict | None = None,
        local_verify: dict | None = None,
    ) -> dict[str, Path]:
        def write_json(name: str, value: dict) -> Path:
            path = temporary / name
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return path

        staged_frontend = copy.deepcopy(frontend_spec or self.frontend_spec)
        staged_base = copy.deepcopy(self.base_spec)
        staged_local = copy.deepcopy(local_verify or self.local)
        staged_report = copy.deepcopy(self.report)
        staged_evidence = copy.deepcopy(self.evidence)
        frontend_spec_path = write_json("frontend_spec.json", staged_frontend)
        base_spec_path = write_json("base_spec.json", staged_base)
        if "trust_root" in staged_local:
            staged_local["trust_root"].update(
                {
                    "path": str(frontend_spec_path.resolve()),
                    "bytes": frontend_spec_path.stat().st_size,
                    "sha256": verify.sha256(frontend_spec_path),
                    "base_spec_sha256": verify.sha256(base_spec_path),
                }
            )
        local_path = write_json("local.json", staged_local)
        report_path = write_json("report.json", staged_report)

        raw_values = {
            "status": "Uid:\t10510\t10510\t10510\t10510\n",
            "maps": "".join(
                f"1000-2000 r-xp 0 00:00 0 /data/app/{library}\n"
                for library in sorted(verify.REQUIRED_PROCESS_LIBRARIES)
            ),
            "fd": (
                "1 -> /dev/fastrpc-cdsp\n"
                "2 -> anon_inode:[dmabuf]\n"
                "3 -> anon_inode:[dma-buf]\n"
            ),
        }
        raw_paths: dict[str, Path] = {}
        for label, raw in raw_values.items():
            path = temporary / f"proc_{label}.txt"
            path.write_text(raw, encoding="utf-8")
            raw_paths[label] = path
            staged_evidence["process"]["raw_artifacts"][label] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": verify.sha256(path),
            }
        staged_evidence["local_verifier_report"].update(
            {
                "bytes": local_path.stat().st_size,
                "sha256": verify.sha256(local_path),
            }
        )
        staged_evidence["device_report"].update(
            {
                "bytes": report_path.stat().st_size,
                "sha256": verify.sha256(report_path),
            }
        )
        evidence_path = write_json("evidence.json", staged_evidence)
        return {
            "frontend_spec": frontend_spec_path,
            "base_spec": base_spec_path,
            "local": local_path,
            "report": report_path,
            "evidence": evidence_path,
            "maps": raw_paths["maps"],
        }

    def test_passes_complete_frontend_evidence(self) -> None:
        self.assertEqual(self.run_verify()["status"], "pass")

    def test_rejects_minimal_forged_local_verifier_report(self) -> None:
        self.local = {
            "status": "pass",
            "gate_id": self.frontend_spec["gate_id"],
            "base_gate_id": self.base_spec["gate_id"],
            "package": self.frontend_spec["target"]["package"],
            "apk": {
                "path": "/tmp/forged.apk",
                **self.frontend_spec["apk"],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "field inventory"):
            self.run_verify()

    def test_rejects_local_archive_status_forgery(self) -> None:
        self.local["archive"]["status"] = "pass-ish"
        with self.assertRaisesRegex(RuntimeError, "archive verification failed"):
            self.run_verify()

    def test_rejects_local_signing_forgery(self) -> None:
        self.local["binary"]["signing"]["v2"] = False
        with self.assertRaisesRegex(RuntimeError, "signing evidence mismatch"):
            self.run_verify()

    def test_rejects_local_device_mutation_forgery(self) -> None:
        self.local["device_mutated"] = True
        with self.assertRaisesRegex(RuntimeError, "mutated a device"):
            self.run_verify()

    def test_rejects_local_claim_boundary_forgery(self) -> None:
        self.local["claim_boundary"] = "Everything was verified on a phone."
        with self.assertRaisesRegex(RuntimeError, "claim boundary mismatch"):
            self.run_verify()

    def test_rejects_local_trust_root_forgery(self) -> None:
        self.local["trust_root"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "trust-root identity mismatch"):
            self.run_verify()

    def test_passes_data_driven_five_language_fixture_inventory(self) -> None:
        identities = {
            "zh": {
                "wav_sha256": "bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63",
                "wav_pcm_frames": 41_717,
                "golden_tensor_sha256": "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263",
                "differing": 136,
                "milliseconds": 10.0,
            },
            "ru": {
                "wav_sha256": "3545b8a20e3f320174a4e4dee57ff77268f30f79526eef91c86c778e3c180683",
                "wav_pcm_frames": 24_315,
                "golden_tensor_sha256": "0e0c25a24157edabe3c1d193e60e8b59886aa9b2ba9b1581fb5c90c2eaf3bbc4",
                "differing": 40,
                "milliseconds": 8.0,
            },
            "ko": {
                "wav_sha256": "0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de",
                "wav_pcm_frames": 33_432,
                "golden_tensor_sha256": "808b38d3c4198c15566116562eb7ad61fbafc01912a747ae439047fb9d7eb1c5",
                "differing": 93,
                "milliseconds": 9.0,
            },
        }
        contract_fixtures = self.frontend_spec["device_output_contract"]["frontend"][
            "fixtures"
        ]
        report_frontend = self.report["bundled_audio_frontend"]
        for index, (language, identity) in enumerate(identities.items(), start=3):
            contract_fixtures[language] = {
                "wav_sha256": identity["wav_sha256"],
                "wav_pcm_frames": identity["wav_pcm_frames"],
                "golden_tensor_sha256": identity["golden_tensor_sha256"],
            }
            report_frontend["fixtures"][language] = {
                "wav_sha256": identity["wav_sha256"],
                "wav_pcm_frames": identity["wav_pcm_frames"],
                "golden_tensor_sha256": identity["golden_tensor_sha256"],
                "produced_tensor_sha256": str(index) * 64,
                "exact_sha256_match": False,
                "differing_uint16_values": identity["differing"],
                "differing_fraction": identity["differing"] / 240_000,
                "max_uint16_delta": 1,
                "frontend_compute_ms": identity["milliseconds"],
                "bounded_gate_passed": True,
            }
            report_frontend["frontend_compute_ms"][language] = identity["milliseconds"]
        compute = report_frontend["frontend_compute_ms"]
        compute["aggregate"] = round(
            sum(value for key, value in compute.items() if key != "aggregate"), 3
        )
        report_frontend["frontend_verification_total_ms"] = compute["aggregate"] + 5.0
        result = self.run_verify()
        self.assertEqual(
            set(result["frontend_fixtures"]), {"en", "zh", "ru", "ko", "ja"}
        )

    def test_passes_data_driven_en_plus_selected_semantic_inventory(self) -> None:
        contract_fixtures = self.frontend_spec["device_output_contract"]["frontend"][
            "fixtures"
        ]
        del contract_fixtures["ja"]
        contract_fixtures["zh"] = {
            "wav_sha256": "bfc02160e9b86c487dc122a7e501e71d411ada2d7a5b94ad59b1bf0f8e149c63",
            "wav_pcm_frames": 41_717,
            "golden_tensor_sha256": "6d18da5e1ec42bbd8161881a598e835c365138377b3d42364ff360e4849d4263",
        }
        report_frontend = self.report["bundled_audio_frontend"]
        del report_frontend["fixtures"]["ja"]
        report_frontend["fixtures"]["zh"] = {
            "wav_sha256": contract_fixtures["zh"]["wav_sha256"],
            "wav_pcm_frames": 41_717,
            "golden_tensor_sha256": contract_fixtures["zh"]["golden_tensor_sha256"],
            "produced_tensor_sha256": "3" * 64,
            "exact_sha256_match": False,
            "differing_uint16_values": 136,
            "differing_fraction": 136 / 240_000,
            "max_uint16_delta": 1,
            "frontend_compute_ms": 10.0,
            "bounded_gate_passed": True,
        }
        report_frontend["frontend_compute_ms"] = {
            "en": 12.0,
            "zh": 10.0,
            "aggregate": 22.0,
        }
        report_frontend["frontend_verification_total_ms"] = 25.0
        result = self.run_verify()
        self.assertEqual(set(result["frontend_fixtures"]), {"en", "zh"})

    def test_rejects_stale_run_id(self) -> None:
        self.report["run_id"] = "older-run"
        with self.assertRaisesRegex(RuntimeError, "app report run_id"):
            self.run_verify()

    def test_rejects_incomplete_report(self) -> None:
        self.report["report_complete"] = False
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.run_verify()

    def test_rejects_wall_elapsed_mismatch(self) -> None:
        self.report["elapsed_wall_ms"] = 999
        with self.assertRaisesRegex(RuntimeError, "wall elapsed time"):
            self.run_verify()

    def test_rejects_boolean_monotonic_elapsed(self) -> None:
        self.report["elapsed_monotonic_ms"] = True
        with self.assertRaisesRegex(RuntimeError, "not numeric"):
            self.run_verify()

    def test_rejects_failed_closed_report(self) -> None:
        self.report["status"] = "failed_closed"
        with self.assertRaisesRegex(RuntimeError, "probe did not pass"):
            self.run_verify()

    def test_rejects_partial_file(self) -> None:
        self.evidence["partial_report_absent"] = False
        with self.assertRaisesRegex(RuntimeError, "partial report"):
            self.run_verify()

    def test_rejects_whole_apk_drift(self) -> None:
        self.evidence["installed_package"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "installed APK hash"):
            self.run_verify()

    def test_rejects_accepted_package_drift(self) -> None:
        self.evidence["accepted_packages"]["ai.liquid.lfmdemo"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "accepted APK hash"):
            self.run_verify()

    def test_rejects_wrong_corrected_ja_golden(self) -> None:
        self.report["bundled_audio_frontend"]["fixtures"]["ja"][
            "golden_tensor_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "ja golden tensor"):
            self.run_verify()

    def test_rejects_frontend_bound_excess(self) -> None:
        fixture = self.report["bundled_audio_frontend"]["fixtures"]["en"]
        fixture["differing_uint16_values"] = 241
        fixture["differing_fraction"] = 241 / 240_000
        with self.assertRaisesRegex(RuntimeError, "exceeds bound"):
            self.run_verify()

    def test_rejects_fraction_count_mismatch(self) -> None:
        self.report["bundled_audio_frontend"]["fixtures"]["en"][
            "differing_fraction"
        ] = 0.0
        with self.assertRaisesRegex(RuntimeError, "fraction/count"):
            self.run_verify()

    def test_rejects_boolean_differing_fraction(self) -> None:
        self.report["bundled_audio_frontend"]["fixtures"]["en"][
            "differing_fraction"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "differing fraction invalid"):
            self.run_verify()

    def test_rejects_extra_frontend_fixture(self) -> None:
        self.report["bundled_audio_frontend"]["fixtures"]["fr"] = copy.deepcopy(
            self.report["bundled_audio_frontend"]["fixtures"]["en"]
        )
        with self.assertRaisesRegex(RuntimeError, "fixture inventory"):
            self.run_verify()

    def test_rejects_disagreed_frontend_fixture_timing(self) -> None:
        self.report["bundled_audio_frontend"]["frontend_compute_ms"]["en"] = 11.0
        with self.assertRaisesRegex(RuntimeError, "en timing disagrees"):
            self.run_verify()

    def test_rejects_boolean_top_level_frontend_timing(self) -> None:
        self.report["bundled_audio_frontend"]["frontend_compute_ms"]["en"] = True
        with self.assertRaisesRegex(RuntimeError, "not numeric"):
            self.run_verify()

    def test_rejects_native_jni_drift(self) -> None:
        self.evidence["installed_frontend_library"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "frontend JNI hash"):
            self.run_verify()

    def test_rejects_prequantized_fallback(self) -> None:
        self.report["synthetic_decode"]["feature_source"] = "prequantized_fixture"
        with self.assertRaisesRegex(RuntimeError, "feature source"):
            self.run_verify()

    def test_rejects_cpu_fallback(self) -> None:
        self.report["cpu_ep_fallback_disabled"] = False
        with self.assertRaisesRegex(RuntimeError, "CPU EP fallback"):
            self.run_verify()

    def test_rejects_missing_frontend_mapping(self) -> None:
        self.evidence["process"]["loaded_libraries"].remove("libwhisper_frontend.so")
        with self.assertRaisesRegex(RuntimeError, "not all mapped"):
            self.run_verify()

    def test_rejects_missing_raw_process_artifact(self) -> None:
        del self.evidence["process"]["raw_artifacts"]["maps"]
        with self.assertRaisesRegex(RuntimeError, "raw process artifact inventory"):
            self.run_verify()

    def test_rejects_decoder_timing_count(self) -> None:
        self.report["synthetic_decode"]["decoder_step_ms"].pop()
        with self.assertRaisesRegex(RuntimeError, "timing count"):
            self.run_verify()

    def test_strict_loader_rejects_duplicate_keys_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            duplicate = Path(temporary) / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                verify.load_object(duplicate)
            invalid = Path(temporary) / "nan.json"
            invalid.write_text('{"a": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid JSON numeric"):
                verify.load_object(invalid)

    def test_verify_files_rejects_tampered_raw_process_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = self.stage_file_bundle(Path(temporary_name))
            self.assertEqual(
                verify.verify_files(
                    paths["frontend_spec"],
                    paths["base_spec"],
                    paths["local"],
                    paths["report"],
                    paths["evidence"],
                    expected_run_id=self.run_id,
                )["status"],
                "pass",
            )
            paths["maps"].write_text(
                paths["maps"].read_text(encoding="utf-8") + "tampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "raw process artifact byte"):
                verify.verify_files(
                    paths["frontend_spec"],
                    paths["base_spec"],
                    paths["local"],
                    paths["report"],
                    paths["evidence"],
                    expected_run_id=self.run_id,
                )

    def test_verify_files_rejects_tampered_claim_contract(self) -> None:
        forged_spec = copy.deepcopy(self.frontend_spec)
        forged_spec["claim_boundary"]["real_audio_transcription"] = True
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = self.stage_file_bundle(
                Path(temporary_name),
                frontend_spec=forged_spec,
            )
            with self.assertRaisesRegex(RuntimeError, "claim boundary drift"):
                verify.verify_files(
                    paths["frontend_spec"],
                    paths["base_spec"],
                    paths["local"],
                    paths["report"],
                    paths["evidence"],
                    expected_run_id=self.run_id,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
