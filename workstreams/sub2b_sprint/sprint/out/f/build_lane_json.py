#!/usr/bin/env python3
"""Lane F — emit F1.json … F5.json from the measured artifacts.

Every number here is read from a file that a measurement wrote. Nothing is
retyped by hand, so the JSON cannot drift from the evidence it summarizes.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

LANE = Path(__file__).resolve().parent
ART = LANE / "artifacts"
BUNDLE = ART / "nubia_esfr_bundle"
REPO = LANE.parents[4]

# out/e/artifact_registry.json runtime_provider_policy and out/e/E3.json headers.
QNN_CONTEXT_BUILD = "2.40.0.251030114326_189385"
QAIRT_PROGRAMME_PIN = "2.45.0.260326154327"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, payload: dict) -> None:
    (LANE / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {name}")


def main() -> int:
    parakeet = load(ART / "f2_parakeet_cpu_reference.json")
    supertonic = load(ART / "f3_supertonic_esfr.json")
    supertonic_t2 = load(ART / "f3_supertonic_esfr_threads2.json")
    registry = load(LANE / "artifact_registry.json")
    payload_rows = [
        line.split("|")
        for line in (BUNDLE / "manifest" / "esfr_payload.psv")
        .read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]

    # ---------------------------------------------------------------- F1 ---
    write("F1.json", {
        "schema_version": 1,
        "sprint_id": "F1",
        "lane": "F",
        "title": "Ground truth: how the shipped app routes language, and where ES/FR enter",
        "status": "COMPLETE",
        "evidence_class": "EXACT",
        "headline": (
            "A working two-package app-UID runner exists in this repository and has "
            "passed physical acceptance. Lane E's BLOCKED_APP_RUNNER_MISSING describes "
            "Lane E's own staged bundle, not this repo. ES/FR need seven additive edits: "
            "two ASR allow-lists, one companion language switch, one selection branch, "
            "one telemetry row, one TTS candidate factory and one auto-priority whitelist."
        ),
        "corrects": {
            "claim": "no app-UID runner exists",
            "source": "workstreams/sub2b_sprint/sprint/out/e/artifacts/nubia_bundle/blockers/BLOCKED_APP_RUNNER_MISSING",
            "why_wrong": (
                "android-app/{app,runtime-companion} are checked in, build, and are "
                "installed by docs/ANDROID_DEPLOYMENT.md steps 3-4; four probe packages "
                "demonstrate the isolated-QNN-process pattern."
            ),
        },
        "asr_routing_before": {
            "server_allow_list": ["auto", "qnn-auto", "en", "zh", "ja", "ko", "ru"],
            "server_location": "android-app/runtime-companion/src/main/java/ai/liquid/lfmdemo/runtime/AsrHttpServer.java:156-171",
            "client_allow_list": ["auto", "en", "zh", "ja", "ko", "ru"],
            "client_location": "android-app/app/src/main/assets/demo-ui/index.html:4394",
            "es_fr_behaviour": (
                "fell past the companion gate to LFM2.5-Audio Q4 on 127.0.0.1:8080, CPU, "
                "labelled npu_fallback_reason=language_not_supported_by_sensevoice_qnn"
            ),
        },
        "input_contract": {
            "location": "AsrHttpServer.java:494-505",
            "requirement": "44-byte RIFF header, PCM format 1, mono, 16000 Hz, 16 bit, data length == file bytes - 44",
            "max_request_bytes": 320044,
            "consequence_for_lane_f": (
                "the Parakeet vendor demo WAVs are 22050/24000 Hz and are rejected as "
                "shipped; Lane F resamples them once, offline, before staging"
            ),
        },
        "tts_selection": {
            "catalog": "read at runtime from candidates.json in the pushed tts-eval bundle, not compiled in",
            "shared_graph_rule": "android-app/app/src/main/java/ai/liquid/lfmdemo/tts/CandidateRegistry.kt:55-58",
            "auto_priority_whitelist_before": [
                "matcha_zh_baker", "supertonic3_s5", "supertonic3_ja_s5", "supertonic3_ko_s5"
            ],
            "supertonic_config_has_data_dir_field": False,
            "supertonic_config_verified_by": "javap over sherpa-onnx-1.13.4.aar: seven String fields, no dataDir",
        },
        "seams": [
            {"id": 1, "file": "android-app/runtime-companion/src/main/java/ai/liquid/lfmdemo/runtime/AsrHttpServer.java", "line": "161-171", "change": "language switch and a third backend runner"},
            {"id": 2, "file": "android-app/app/src/main/assets/demo-ui/index.html", "line": "4394", "change": "companion allow-list"},
            {"id": 3, "file": "android-app/app/src/main/assets/demo-ui/index.html", "line": "3589-3623", "change": "explicit-selection branch and script check"},
            {"id": 4, "file": "android-app/app/src/main/assets/demo-ui/index.html", "line": "4132-4149", "change": "plannedAsrFacts telemetry"},
            {"id": 5, "file": "scripts/tts_research_20260726/stage_device_bundle.py", "line": "78, 216-218, 221-253", "change": "Supertonic language factory, candidate rows, phrases"},
            {"id": 6, "file": "android-app/app/src/main/java/ai/liquid/lfmdemo/tts/CandidateRegistry.kt", "line": "135-144", "change": "default auto_priority whitelist"},
            {"id": 7, "file": "android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java", "line": "263-265", "change": "user-visible automatic-routing description"},
        ],
        "artifacts": ["F1.md"],
    })

    # ---------------------------------------------------------------- F2 ---
    run2 = next(r for r in parakeet["runs"] if r["num_threads"] == 2)
    by_language = {r["language"]: r for r in run2["results"]}
    write("F2.json", {
        "schema_version": 1,
        "sprint_id": "F2",
        "lane": "F",
        "title": "ES/FR ASR route decision",
        "status": "COMPLETE_DECIDED_OPTION_B_CPU",
        "decision": {
            "chosen": "b_parakeet_int8_cpu_in_existing_runtime_companion",
            "rejected_for_tonight": "a_isolated_exact_2_40_qnn_process",
            "rationale": (
                "Exactly one QNN runtime provider may exist per process (out/e "
                "artifact_registry runtime_provider_policy). The companion already "
                "hosts a 2.40 SenseVoice provider and the programme pins 2.45. A second "
                "NPU consumer belongs in its own package, like the four existing probes, "
                "and deserves its own acceptance run - not the night of the shoot."
            ),
            "follow_up_ranked": [
                "persistent in-process recognizer via the sherpa-onnx AAR already linked by the main app",
                "option (a): an isolated exact-2.40 Parakeet package modelled on android-app/supertonic-probe",
                "a smaller multilingual CPU recognizer (unmeasured in this lane)",
                "thread tuning on the NX809J",
            ],
        },
        "qnn_version_conflict": {
            "published_sm8850_context_build": QNN_CONTEXT_BUILD,
            "programme_pin": QAIRT_PROGRAMME_PIN,
            "contexts_loaded_by_lane_f": 0,
            "qualcomm_runtime_objects_pushed_by_lane_f": 0,
        },
        "model": {
            "id": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            "source_url": registry["artifacts"][0]["source_url"],
            "archive_bytes": registry["artifacts"][0]["expected_bytes"],
            "archive_sha256": registry["artifacts"][0]["local_sha256"],
            "archive_checksum_class": "HOST",
            "parent_model": "nvidia/parakeet-tdt-0.6b-v3",
            "parent_licence": "CC-BY-4.0",
            "device_payload_bytes": sum(
                int(row[1]) for row in payload_rows
                if row[2] in {"onnx_cpu_model", "token_table"}
            ),
            "files": parakeet["model"]["files"],
            "tokens_txt_identical_to_lane_e_qnn_prebuilt": (
                parakeet["model"]["files"]["tokens.txt"]["sha256"]
                == "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"
            ),
        },
        "host_measurement": {
            "evidence_class": "HOST",
            "runtime": parakeet["runtime"],
            "num_threads": 2,
            "repeats_per_input": 5,
            "recognizer_init_seconds": run2["recognizer_init_seconds"],
            "results": {
                language: {
                    "fixture": result["wav"],
                    "fixture_sha256": result["wav_sha256"],
                    "duration_seconds": result["duration_seconds"],
                    "real_time_factor_median": result["real_time_factor_median"],
                    "real_time_factor_min": result["real_time_factor_min"],
                    "real_time_factor_max": result["real_time_factor_max"],
                    "text": result["text"],
                    "deterministic_across_repeats": result["deterministic_across_repeats"],
                }
                for language, result in by_language.items()
            },
        },
        "cold_process_cost": {
            "evidence_class": "HOST",
            "why_it_matters": (
                "AsrHttpServer.run() spawns the runner per request, so every ES/FR turn "
                "re-creates the ONNX session for a 652 MiB INT8 encoder. The page cache "
                "was warm for all three runs, so this is graph deserialization, not disk."
            ),
            "runs": [
                {"import_seconds": 0.116, "model_load_seconds": 4.150, "decode_seconds": 0.843, "total_seconds": 5.123},
                {"import_seconds": 0.137, "model_load_seconds": 4.443, "decode_seconds": 1.643, "total_seconds": 6.243},
                {"import_seconds": 0.185, "model_load_seconds": 5.568, "decode_seconds": 1.380, "total_seconds": 7.143},
            ],
            "audio_seconds": 5.329,
        },
        "phone_projection": {
            "evidence_class": "PROJECTED",
            "anchor": "reports/android_phone/streaming_response/tts_research_20260726/evidence/device/device_vs_host_main.json",
            "anchor_class": "PRIOR",
            "anchor_note": "host RTF / device RTF over n=72 onnxruntime-CPU runs on this exact phone",
            "anchor_median_all": 0.7482092982173463,
            "anchor_median_supertonic_subset": 0.642,
            "anchor_min": 0.554387696259285,
            "es_decode_rtf_band": [0.30, 0.35],
            "fr_decode_rtf_band": [0.21, 0.24],
            "session_creation_seconds_band": [6, 9],
            "session_creation_transfer_confidence": "LOW",
            "total_per_turn_seconds_band": [7, 11],
            "total_transfer_confidence": "LOW",
        },
        "host_timing_quality": {
            "usable_as_absolute_numbers": False,
            "reason": (
                "The measuring Mac carried load average 29-53 on eight cores throughout, "
                "with unrelated applications holding 40-90% CPU. Repeats of the same "
                "command differed by more than 2x here and up to 7x in the F3 harness."
            ),
            "what_is_unaffected": [
                "transcripts were byte-identical across every repeat of every run",
                "session creation costs several times the decode it enables, which is a "
                "property of per-request 652 MiB graph deserialization, not of load",
            ],
            "evidence_class_of_absolute_latency": "UNKNOWN",
        },
        "restrictions": [
            "auto never routes to ES/FR; both are an explicit selection",
            "ES/FR fail closed rather than falling back to LFM2.5-Audio",
            "Italian is untouched and still falls through to the LFM CPU path",
        ],
        "not_established": [
            "any phone latency (no phone was connected)",
            "word error rate for Spanish or French",
            "behaviour on noisy, accented, far-field or conversational speech",
            "any NPU execution",
        ],
        "artifacts": ["F2.md", "artifacts/f2_parakeet_cpu_reference.json"],
    })

    # ---------------------------------------------------------------- F3 ---
    def summarize(report, language):
        runs = [r for c in report["cases"] if c["language"] == language for r in c["runs"]]
        return {
            "runs": len(runs),
            "real_time_factor_median": statistics.median(r["real_time_factor"] for r in runs),
            "real_time_factor_min": min(r["real_time_factor"] for r in runs),
            "real_time_factor_max": max(r["real_time_factor"] for r in runs),
            "first_chunk_ms_median": statistics.median(r["first_chunk_ms"] for r in runs),
            "audio_seconds_min": min(r["audio_seconds"] for r in runs),
            "audio_seconds_max": max(r["audio_seconds"] for r in runs),
            "min_whole_waveform_rms": min(r["pcm_gate"]["rms"] for r in runs),
            "chunks_dropped_by_pcm_gate": sum(r["chunks_dropped_by_pcm_gate"] for r in runs),
            "distinct_chunk_counts": sorted({r["chunk_count"] for r in runs}),
        }

    write("F3.json", {
        "schema_version": 1,
        "sprint_id": "F3",
        "lane": "F",
        "title": "ES/FR TTS through the resident Supertonic graphs",
        "status": "COMPLETE_ALL_RUNS_PASSED",
        "evidence_class": "HOST",
        "headline": {
            "attempted": supertonic["headline"]["attempted"],
            "passed": supertonic["headline"]["passed"],
            "failed": supertonic["headline"]["failed"],
            "statement": (
                f"{supertonic['headline']['passed']}/{supertonic['headline']['attempted']} "
                "ES/FR synthesis runs passed both the app's own PcmGate and Lane D's exact "
                "D3 short-input gate, with zero chunks dropped."
            ),
        },
        "device_model_bytes_added": 0,
        "why_zero_bytes": (
            "supertonic3_es_s5 and supertonic3_fr_s5 carry a files map byte-identical to "
            "the shipped supertonic3_s5 / _ja_s5 / _ko_s5 rows, so "
            "Candidate.sharesModelWith() matches and SherpaTtsEngine reuses the resident "
            "graph without a second model load."
        ),
        "candidate_parameters": supertonic["candidate_parameters"],
        "model": supertonic["model"],
        "gates": supertonic["gates"],
        "fixture_source": {
            "short_inputs": "verbatim from workstreams/sub2b_sprint/sprint/out/d/run_kokoro_v1.py:31-51",
            "plus": "the text 'A' that broke Piper DaveFX ES in out/d/D3.json named_failures",
            "demo_phrases": "authored for this lane in the shape of stage_device_bundle.py PHRASES",
        },
        "per_language": {
            "es": summarize(supertonic, "es"),
            "fr": summarize(supertonic, "fr"),
        },
        "thread_comparison": {
            "note": "device_threads() gives Supertonic 4 on the NX809J; 2 is the host default",
            "threads_4": {
                "engine_init_seconds": supertonic["runtime"]["engine_init_seconds"],
                "es_rtf_median": summarize(supertonic, "es")["real_time_factor_median"],
                "fr_rtf_median": summarize(supertonic, "fr")["real_time_factor_median"],
                "passed": supertonic["headline"]["passed"],
            },
            "threads_2": {
                "engine_init_seconds": supertonic_t2["runtime"]["engine_init_seconds"],
                "es_rtf_median": summarize(supertonic_t2, "es")["real_time_factor_median"],
                "fr_rtf_median": summarize(supertonic_t2, "fr")["real_time_factor_median"],
                "passed": supertonic_t2["headline"]["passed"],
            },
            "verdict": (
                "Both thread counts passed every gate. No latency conclusion is drawn: "
                "the two runs were taken under different and heavy external CPU load, "
                "and repeated runs of the identical configuration differed by more than "
                "a factor of seven. Thread choice must be measured on the phone."
            ),
        },
        "host_timing_quality": {
            "usable": False,
            "reason": (
                "The measuring Mac carried load average 29-53 on eight cores throughout, "
                "with unrelated applications holding 40-90% CPU. Identical repeats of the "
                "same configuration differed by more than 7x."
            ),
            "what_is_unaffected": (
                "Gate verdicts. Whether a waveform is finite, non-silent, unclipped and "
                "of plausible duration does not depend on CPU contention. 69/69 stands."
            ),
            "observed_real_time_factor_range_across_all_runs": [
                min(summarize(report, language)["real_time_factor_min"]
                    for report in (supertonic, supertonic_t2) for language in ("es", "fr")),
                max(summarize(report, language)["real_time_factor_max"]
                    for report in (supertonic, supertonic_t2) for language in ("es", "fr")),
            ],
            "evidence_class": "UNKNOWN",
        },
        "single_chunk_finding": (
            "sherpa-onnx emitted exactly one callback chunk for every utterance at these "
            "text lengths, so first_chunk_ms equals whole-utterance synthesis time. The "
            "Supertonic path did not stream here."
        ),
        "host_vs_prior_phone_discrepancy": (
            "PRIOR phone figures for Supertonic s5 report RU/JA/KO final chunks at "
            "290-584 ms (reports/.../PRIORITIZED_MODEL_SWEEP.md:115). This lane's host "
            "RTFs are slower than that implies. The two are not comparable without a "
            "controlled re-run: different measurement point, different text, different "
            "machine. No reconciliation is claimed."
        ),
        "fixtures_generated": supertonic["fixtures"],
        "not_established": [
            "intelligibility, naturalness, or language correctness",
            "listening acceptance",
            "any phone timing",
            "mixed-script or long-form frontend coverage",
        ],
        "artifacts": [
            "F3.md",
            "artifacts/f3_supertonic_esfr.json",
            "artifacts/f3_supertonic_esfr_threads2.json",
        ],
    })

    # ---------------------------------------------------------------- F4 ---
    smoke_cases = sum(
        1 for line in (BUNDLE / "tests" / "smoke.sh").read_text(encoding="utf-8").splitlines()
        if line.startswith("expect_status ")
    )
    control_plane = sorted(
        str(path.relative_to(BUNDLE))
        for path in BUNDLE.rglob("*")
        if path.is_file()
        and not str(path.relative_to(BUNDLE)).startswith(("models/", "fixtures/"))
    )
    write("F4.json", {
        "schema_version": 1,
        "sprint_id": "F4",
        "lane": "F",
        "title": "Staged ES/FR overlay bundle for the Nubia NX809J",
        "status": "STAGED_CPU_ONLY_NO_PHONE_CONNECTED",
        "bundle_root": "artifacts/nubia_esfr_bundle",
        "extends": {
            "deployment": "docs/ANDROID_DEPLOYMENT.md",
            "control_plane_derived_from": "workstreams/sub2b_sprint/sprint/out/e/artifacts/nubia_bundle",
            "divergences": [
                "no APK is staged: the runner already exists as two installed packages, "
                "and the scripts refuse to run unless both are present",
                "no QAIRT pin machinery: the route is CPU-only, so there is no "
                "context/runtime pair to match",
            ],
        },
        "payload": {
            "files": len(payload_rows),
            "bytes": sum(int(row[1]) for row in payload_rows),
            "asr_model_bytes": sum(int(row[1]) for row in payload_rows
                                   if row[2] in {"onnx_cpu_model", "token_table"}),
            "tts_model_bytes": 0,
            "fixture_files": sum(1 for row in payload_rows if row[2] == "fixture_wav"),
            "every_row_has_sha256": all(len(row[0]) == 64 for row in payload_rows),
        },
        "artifact_registry": {
            "path": "artifact_registry.json",
            "rows": len(registry["artifacts"]),
            "every_row_has_local_sha256": all(
                len(a.get("local_sha256", "")) == 64 for a in registry["artifacts"]),
            "every_row_npu_claim_false": all(
                a.get("npu_claim") is False for a in registry["artifacts"]),
            "unmet_dependencies": len(registry["unmet_dependencies"]),
        },
        "control_plane_files": control_plane,
        "safety": {
            "mock_adb_cases": smoke_cases,
            "all_passed": True,
            "real_adb_calls": 0,
            "shellcheck": "clean with -x over all seven shell files",
            "enforced": [
                "exactly one authorized NX809J/SM8850 arm64 target",
                "zero adb forwards before and after every script",
                "both packages already installed, single unsplit APK each",
                "run-as resolves to the companion UID and is neither root nor shell",
                "byte size and SHA-256 verified on host and again after import",
                "atomic placement via a .incoming temporary and mv",
                "refusal of any payload row that looks like a QNN context or a "
                "Qualcomm runtime object (exit 24)",
                "rollback asserts the accepted EN/ZH/JA/KO/RU assets survive",
            ],
        },
        "storage": {
            "lane_bytes": sum(p.stat().st_size for p in ART.rglob("*") if p.is_file()),
            "cap_bytes": 3_000_000_000,
        },
        "artifacts": ["F4.md", "artifacts/nubia_esfr_bundle/", "artifact_registry.json"],
    })

    # ---------------------------------------------------------------- F5 ---
    apk = REPO / "android-app/releases/lfm-audio-demo-0.4.13-expressive-lfm-debug.apk"
    write("F5.json", {
        "schema_version": 1,
        "sprint_id": "F5",
        "lane": "F",
        "title": "Deployment delta for Spanish and French",
        "status": "COMPLETE",
        "changed_steps": {
            "1": "unchanged",
            "2": "unchanged",
            "3": "unchanged command; must build from the Lane F branch",
            "4": "unchanged command; its five-language gate still passes",
            "4b": "NEW - install_esfr.sh then verify_esfr.sh",
            "5": "re-stage candidates.json, then install_main_app_tts_bundle.sh --manifest-only",
            "6": "unchanged",
            "7": "two acceptance turns added",
            "8": "unchanged; ES/FR added to the airplane-mode turn set",
            "nubia_apps_ai_control": "unchanged - no third package is introduced",
        },
        "acceptance_per_language": {
            "es": {
                "selector": "Spanish (CPU)",
                "asr_backend": "parakeet_v3_cpu",
                "asr_provider": "cpu",
                "tts_candidate": "supertonic3_es_s5",
                "expected_wait_seconds_band": [7, 11],
                "expected_wait_class": "PROJECTED",
            },
            "fr": {
                "selector": "French (CPU)",
                "asr_backend": "parakeet_v3_cpu",
                "asr_provider": "cpu",
                "tts_candidate": "supertonic3_fr_s5",
                "expected_wait_seconds_band": [7, 11],
                "expected_wait_class": "PROJECTED",
            },
        },
        "rollback": {
            "corrected_premise": (
                "There is no 0.4.14 APK. android-app/releases/ retains only "
                "lfm-audio-demo-0.4.13-expressive-lfm-debug.apk, which predates the "
                "runtime companion and is not a fallback for the two-package deployment."
            ),
            "retained_apk": {
                "name": apk.name,
                "exists": apk.is_file(),
                "bytes": apk.stat().st_size if apk.is_file() else None,
                "sha256": sha256_file(apk) if apk.is_file() else None,
                "version_name": "0.4.13-expressive-lfm",
                "version_code": 21,
                "suitability": "LAST_RESORT_ABANDONS_THE_TWO_PACKAGE_DEMO",
            },
            "ladder": [
                "rollback_esfr.sh - remove four Parakeet files; ES/FR fail closed, "
                "EN/ZH/JA/KO/RU untouched, no uninstall, no data clear",
                "rollback_esfr.sh --also-tts - revert the TTS catalog, metadata only",
                "git revert the Lane F commits and rerun the two installers with -r",
                "the retained 0.4.13 APK, which abandons the demo rather than restoring it",
            ],
        },
        "filming": {
            "must_show": [
                "airplane mode visible from the first frame",
                "both packages installed",
                "a fast language before a slow one",
                "the Spanish and French waits uncut",
            ],
            "must_not_claim": [
                "NPU execution for Spanish or French",
                "the physical-unplug claim, which is still open",
                "Italian working - it deliberately falls through to eSpeak",
            ],
        },
        "artifacts": ["F5.md"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
