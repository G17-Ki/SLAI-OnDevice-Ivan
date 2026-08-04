#!/usr/bin/env python3
"""Lane F F4 — build the ES/FR overlay bundle's manifests and artifact registry.

Every file the bundle can put on the phone gets one row in
`manifest/esfr_payload.psv` and one row in `artifact_registry.json`, both keyed
by a locally computed SHA-256. Nothing is invented: sizes and digests are read
from the files on disk, and provenance fields come from the pinned source URLs
recorded in this repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LANE_ROOT = Path(__file__).resolve().parent
BUNDLE = LANE_ROOT / "artifacts" / "nubia_esfr_bundle"
MAIN_PACKAGE = "ai.liquid.lfmdemo"
RUNTIME_PACKAGE = "ai.liquid.lfmdemo.runtime"

PARAKEET_DIR = "models/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
SUPERTONIC_DIR = "models/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"

PARAKEET_SOURCE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
)
PARAKEET_ARCHIVE_BYTES = 487_170_055
PARAKEET_ARCHIVE_SHA256 = (
    "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf"
)
SUPERTONIC_SOURCE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"
)
SUPERTONIC_ARCHIVE_BYTES = 128_774_318
SUPERTONIC_ARCHIVE_SHA256 = (
    "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427"
)

# The two Supertonic routes ES/FR add. Identical `files` map to the shipped
# RU/JA/KO rows, so CandidateRegistry.sharesModelWith() lets the app switch
# language without a second model load and the device gains zero bytes.
SUPERTONIC_FILES = {
    "duration_predictor": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx",
    "text_encoder": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx",
    "vector_estimator": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx",
    "vocoder": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx",
    "tts_json": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json",
    "unicode_indexer": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin",
    "voice_style": "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(relative: str, klass: str, package: str, app_path: str) -> dict:
    path = BUNDLE / relative
    if not path.is_file():
        raise SystemExit(f"missing bundle payload: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "class": klass,
        "package": package,
        "app_private_path": app_path,
    }


def supertonic_candidate(language: str, label: str, note: str) -> dict:
    return {
        "id": f"supertonic3_{language}_s5",
        "label": f"Supertonic 3 {label} — 5 steps",
        "language": language,
        "engine": "supertonic",
        "num_threads": 4,
        "sid": 6,
        "speed": 1.0,
        "num_steps": 5,
        "auto_priority": 10,
        "extra": {"lang": language, "seed": "0"},
        "files": dict(SUPERTONIC_FILES),
        "notes": note,
    }


def main() -> int:
    payload: list[dict] = []

    # --- ASR: the Parakeet CPU model into the runtime companion ------------
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"):
        payload.append(row(f"{PARAKEET_DIR}/{name}", "onnx_cpu_model",
                           RUNTIME_PACKAGE, f"files/asr/models/parakeet/{name}"))
    payload.append(row(f"{PARAKEET_DIR}/tokens.txt", "token_table",
                       RUNTIME_PACKAGE, "files/asr/models/parakeet/tokens.txt"))

    # --- ASR acceptance fixtures -------------------------------------------
    asr_fixtures = {
        "es": "asr_es_parakeet_vendor.wav",
        "fr": "asr_fr_parakeet_vendor.wav",
    }
    for language, name in asr_fixtures.items():
        payload.append(row(f"fixtures/asr/{name}", "fixture_wav", RUNTIME_PACKAGE,
                           f"files/asr/fixtures/{name}"))

    # --- TTS reference renders, for A/B against what the phone produces ----
    tts_fixtures = sorted(p.name for p in (BUNDLE / "fixtures" / "tts").glob("*.wav"))
    for name in tts_fixtures:
        payload.append(row(f"fixtures/tts/{name}", "fixture_wav", RUNTIME_PACKAGE,
                           f"files/asr/fixtures/tts_reference/{name}"))

    (BUNDLE / "manifest").mkdir(parents=True, exist_ok=True)
    lines = ["sha256|bytes|class|package|path|app_private_path"]
    lines += [
        f"{item['sha256']}|{item['bytes']}|{item['class']}|{item['package']}|"
        f"{item['path']}|{item['app_private_path']}"
        for item in payload
    ]
    (BUNDLE / "manifest" / "esfr_payload.psv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # --- expected transcripts ----------------------------------------------
    reference = json.loads(
        (LANE_ROOT / "artifacts" / "f2_parakeet_cpu_reference.json").read_text(encoding="utf-8")
    )
    by_language = {
        result["language"]: result
        for run in reference["runs"] if run["num_threads"] == 2
        for result in run["results"]
    }
    expected = {
        "schema_version": 1,
        "source": "workstreams/sub2b_sprint/sprint/out/f/artifacts/f2_parakeet_cpu_reference.json",
        "evidence_class_of_transcripts": "HOST",
        "note": (
            "These strings were produced by the same model files and the same "
            "canonical WAVs on an Apple M2 CPU. Requiring exact equality on the "
            "phone is a parity assertion, not an accuracy claim. A mismatch means "
            "the phone is not running what was measured."
        ),
        "languages": {
            language: {
                "backend": "parakeet_v3_cpu",
                "provider": "cpu",
                "fixture": f"fixtures/asr/{asr_fixtures[language]}",
                "fixture_sha256": next(
                    item["sha256"] for item in payload
                    if item["path"].endswith(asr_fixtures[language])
                ),
                "transcript": by_language[language]["text"],
                "host_real_time_factor_median": by_language[language]["real_time_factor_median"],
            }
            for language in ("es", "fr")
        },
    }
    (BUNDLE / "manifest" / "esfr_expected_transcripts.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- the two TTS candidate rows ----------------------------------------
    candidates = {
        "schema_version": 1,
        "purpose": (
            "Merge these two rows into the tts-eval bundle's candidates.json. "
            "Their files map is byte-identical to the shipped RU/JA/KO Supertonic "
            "rows, so no model file is added to the phone."
        ),
        "adds_model_bytes_to_device": 0,
        "generated_by": "scripts/tts_research_20260726/stage_device_bundle.py",
        "candidates": [
            supertonic_candidate(
                "es", "ES",
                "host: Lane D D3 6/6 short-input safe; Lane F F3 re-verified "
                "against PcmGate. Not phone-timed."),
            supertonic_candidate(
                "fr", "FR",
                "host: Lane D D3 6/6 short-input safe; Lane F F3 re-verified "
                "against PcmGate. Not phone-timed."),
        ],
    }
    (BUNDLE / "manifest" / "esfr_tts_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- bundle.sha256 ------------------------------------------------------
    (BUNDLE / "manifest" / "bundle.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in payload),
        encoding="utf-8")

    # --- artifact registry --------------------------------------------------
    supertonic_rows = []
    for name in sorted(p.name for p in (BUNDLE / SUPERTONIC_DIR).iterdir() if p.is_file()):
        path = BUNDLE / SUPERTONIC_DIR / name
        supertonic_rows.append({
            "artifact_id": f"supertonic3_int8_{name.replace('.', '_')}",
            "name": name,
            "source_url": SUPERTONIC_SOURCE,
            "revision": "release=tts-models;archive=sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
            "expected_bytes": path.stat().st_size,
            "local_sha256": sha256_file(path),
            "checksum_status": "LOCALLY_VERIFIED",
            "licence": "OpenRAIL-M (Supertone/supertonic-3); use restrictions apply",
            "licence_source_url": "https://huggingface.co/Supertone/supertonic-3/raw/3cadd1ee6394adea1bd021217a0e650ede09a323/LICENSE",
            "destination": "already resident on the phone under tts-eval/; not re-pushed by this bundle",
            "artifact_type": "onnx-cpu-tts-graph" if name.endswith(".onnx") else "tts-metadata",
            "required_runtime": "onnxruntime via sherpa-onnx AAR",
            "required_runtime_version": "sherpa-onnx 1.13.4",
            "npu_claim": False,
            "readiness": "RESIDENT_REUSED_BY_ES_FR_ROUTES",
        })

    parakeet_rows = [{
        "artifact_id": "sherpa_parakeet_v3_int8_cpu_archive",
        "name": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "source_url": PARAKEET_SOURCE,
        "revision": "release=asr-models",
        "expected_bytes": PARAKEET_ARCHIVE_BYTES,
        "local_sha256": PARAKEET_ARCHIVE_SHA256,
        "checksum_status": "LOCALLY_VERIFIED_BEFORE_RECOVERABLE_ARCHIVE_DELETION",
        "licence": "CC-BY-4.0 parent model nvidia/parakeet-tdt-0.6b-v3; the release "
                   "payload carries no licence file, so attribution must accompany "
                   "redistribution",
        "licence_source_url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        "destination": "artifacts/nubia_esfr_bundle/models/asr/",
        "artifact_type": "source-archive-onnx-cpu-transducer",
        "required_runtime": "onnxruntime CPU provider via sherpa-onnx",
        "required_runtime_version": "sherpa-onnx 1.13.4",
        "npu_claim": False,
        "readiness": "VERIFIED_AND_EXTRACTED_ARCHIVE_REMOVED",
    }]
    for item in payload:
        if not item["path"].startswith(PARAKEET_DIR):
            continue
        name = item["path"].rsplit("/", 1)[1]
        parakeet_rows.append({
            "artifact_id": f"parakeet_v3_int8_cpu_{name.replace('.', '_')}",
            "name": name,
            "source_url": PARAKEET_SOURCE,
            "revision": "release=asr-models;member of the verified archive above",
            "expected_bytes": item["bytes"],
            "local_sha256": item["sha256"],
            "checksum_status": "LOCALLY_VERIFIED",
            "licence": "CC-BY-4.0 parent model nvidia/parakeet-tdt-0.6b-v3",
            "licence_source_url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
            "destination": f"{RUNTIME_PACKAGE}:{item['app_private_path']}",
            "artifact_type": item["class"],
            "required_runtime": "onnxruntime CPU provider via sherpa-onnx",
            "required_runtime_version": "sherpa-onnx 1.13.4",
            "npu_claim": False,
            "readiness": "STAGED_NOT_INSTALLED_NO_PHONE_CONNECTED",
        })

    fixture_rows = [{
        "artifact_id": f"fixture_{item['path'].rsplit('/', 1)[1].replace('.', '_')}",
        "name": item["path"].rsplit("/", 1)[1],
        "source_url": (
            PARAKEET_SOURCE + "#test_wavs" if "parakeet_vendor" in item["path"]
            else "generated locally by out/f/run_supertonic_esfr.py from the "
                 "verified Supertonic 3 INT8 graphs"
        ),
        "revision": "resampled once to canonical 16 kHz mono PCM16 by out/f/make_canonical_wav.py",
        "expected_bytes": item["bytes"],
        "local_sha256": item["sha256"],
        "checksum_status": "LOCALLY_VERIFIED",
        "licence": (
            "CC-BY-4.0 parent model demo audio" if "parakeet_vendor" in item["path"]
            else "output of an OpenRAIL-M model; the model's use restrictions "
                 "follow the audio"
        ),
        "licence_source_url": (
            "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
            if "parakeet_vendor" in item["path"]
            else "https://huggingface.co/Supertone/supertonic-3"
        ),
        "destination": f"{item['package']}:{item['app_private_path']}",
        "artifact_type": "fixture_wav_canonical_16k_mono_pcm16",
        "required_runtime": "none",
        "required_runtime_version": "n/a",
        "npu_claim": False,
        "readiness": "STAGED_NOT_INSTALLED_NO_PHONE_CONNECTED",
    } for item in payload if item["class"] == "fixture_wav"]

    registry = {
        "schema_version": 1,
        "registry_id": "lane-f-esfr-artifact-registry-v1",
        "scope": (
            "Every artifact Lane F's ES/FR overlay can place on the Nubia NX809J, "
            "plus the resident Supertonic graphs the two new TTS routes reuse. "
            "This is not a complete five-language product manifest; lanes A-E own "
            "the rest."
        ),
        "status": "STAGED_CPU_ONLY_NO_PHONE_CONNECTED",
        "npu_policy": {
            "rule": "This overlay installs no QNN context and loads no Qualcomm "
                    "runtime object. install_esfr.sh refuses any payload row that "
                    "looks like one.",
            "why": "The published SM8850 Parakeet contexts embed QNN "
                   "2.40.0.251030114326_189385 (out/e/E3.json) while the programme "
                   "pins 2.45.0.260326154327 (out/e/E5.json), and exactly one QNN "
                   "runtime provider may exist per process.",
            "every_artifact_npu_claim": False,
        },
        "device_payload_bytes": sum(item["bytes"] for item in payload),
        "device_payload_files": len(payload),
        "artifacts": parakeet_rows + supertonic_rows + fixture_rows,
        "supertonic_archive_provenance": {
            "source_url": SUPERTONIC_SOURCE,
            "expected_bytes": SUPERTONIC_ARCHIVE_BYTES,
            "published_sha256": SUPERTONIC_ARCHIVE_SHA256,
            "local_sha256": SUPERTONIC_ARCHIVE_SHA256,
            "checksum_status": "PUBLISHED_PIN_MATCHED_LOCALLY_THEN_ARCHIVE_DELETED",
            "published_pin_source": "scripts/supertonic3/download_model.sh:12",
        },
        "unmet_dependencies": [
            {
                "dependency_id": "phone_execution_of_es_fr",
                "owner": "operator at the Nubia",
                "status": "UNMET",
                "blocker": "No phone was connected during Lane F. Every phone "
                           "latency figure in F2.md is PROJECTED.",
            },
            {
                "dependency_id": "es_fr_accuracy_measurement",
                "owner": "a later evaluation lane",
                "status": "UNMET",
                "blocker": "Four vendor demo clips decoded correctly. That is an "
                           "integration proof, not a word error rate.",
            },
            {
                "dependency_id": "es_fr_listening_acceptance",
                "owner": "a human listener",
                "status": "UNMET",
                "blocker": "F3 proves the ES/FR waveforms are finite, non-silent "
                           "and unclipped. It says nothing about intelligibility "
                           "or naturalness.",
            },
            {
                "dependency_id": "persistent_recognizer_process",
                "owner": "a follow-up app task",
                "status": "UNMET",
                "blocker": "The companion spawns the runner per request, so each "
                           "ES/FR turn re-creates a 652 MiB ONNX session. See "
                           "F2.md section 6.",
            },
        ],
        "acceptance_assertions": [
            "install_esfr.sh refuses to run unless both packages are already installed.",
            "install_esfr.sh refuses any pre-existing adb forward and re-asserts "
            "zero forwards before exiting.",
            "Every payload file is byte- and SHA-256-verified on the host and "
            "again after import into app-private storage.",
            "verify_esfr.sh requires exact ES and FR transcript parity, "
            "provider == cpu, and runner_owned_by_service == true.",
            "verify_esfr.sh fails if the accepted EN/ZH/JA/KO/RU stack has "
            "regressed.",
            "No artifact in this registry carries an NPU claim.",
        ],
        "evidence_sources": [
            "workstreams/sub2b_sprint/sprint/out/f/artifacts/f2_parakeet_cpu_reference.json",
            "workstreams/sub2b_sprint/sprint/out/f/artifacts/f3_supertonic_esfr.json",
            "workstreams/sub2b_sprint/sprint/out/e/E3.json",
            "workstreams/sub2b_sprint/sprint/out/e/E5.json",
            "workstreams/sub2b_sprint/sprint/out/d/D3.json",
            "reports/android_phone/streaming_response/tts_research_20260726/evidence/device/device_vs_host_main.json",
        ],
    }
    (LANE_ROOT / "artifact_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "bundle_id": "lane-f-nubia-nx809j-esfr-overlay-v1",
        "target": {"device": "Nubia NX809J", "soc": "SM8850", "abi": "arm64-v8a"},
        "extends": {
            "deployment": "docs/ANDROID_DEPLOYMENT.md",
            "packages": [MAIN_PACKAGE, RUNTIME_PACKAGE],
            "control_plane_derived_from":
                "workstreams/sub2b_sprint/sprint/out/e/artifacts/nubia_bundle",
        },
        "state": {
            "phone_contacted": False,
            "npu": "NOT_USED_BY_THIS_OVERLAY",
            "asr_es_fr": "CPU_PARAKEET_V3_INT8",
            "tts_es_fr": "CPU_SUPERTONIC_3_FIVE_STEP_SHARED_GRAPHS",
        },
        "payload": {
            "manifest": "manifest/esfr_payload.psv",
            "files": len(payload),
            "bytes": sum(item["bytes"] for item in payload),
            "new_model_bytes_for_asr": sum(
                item["bytes"] for item in payload
                if item["class"] in {"onnx_cpu_model", "token_table"}
            ),
            "new_model_bytes_for_tts": 0,
        },
        "registry": "../../artifact_registry.json",
    }
    (BUNDLE / "manifest" / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"payload rows: {len(payload)}")
    print(f"payload bytes: {sum(item['bytes'] for item in payload):,}")
    print(f"registry artifacts: {len(registry['artifacts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
