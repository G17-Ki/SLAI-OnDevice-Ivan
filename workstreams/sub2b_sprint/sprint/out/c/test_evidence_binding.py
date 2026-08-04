"""Tamper tests for Lane C record and archive evidence binding."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_results as aggregate  # noqa: E402
import colab_remote_runner as remote_runner  # noqa: E402


ZIPFORMER_MODEL = "pfluo/k2fsa-zipformer-chinese-english-mixed"
ZIPFORMER_REVISION = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"


def _fixture_sample(condition: str) -> dict[str, object]:
    reference = "fixture reference"
    sample: dict[str, object] = {
        "utterance_id": "fleurs:cmn_hans_cn:fixture",
        "language": "zh",
        "condition": condition,
        "audio_path": f"/fixture/fixture.{condition}.wav",
        "reference": reference,
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "duration_s": 1.0,
        "duration_seconds": 1.0,
        "config": "cmn_hans_cn",
        "audio_relpath": f"audio/zh/fixture.{condition}.wav",
        "reference_source": "raw_transcription",
        "raw_transcription": reference,
        "transcription": reference,
        "source_id": "fixture-source",
        "fleurs_id": "1",
        "source_path": "fixture.wav",
        "source_parquet_repo_path": "data/cmn_hans_cn/test/fixture.parquet",
        "audio_pcm_sha256": "1" * 64,
        "clean_pcm_sha256": "2" * 64,
        "raw_audio_sha256": "3" * 64,
        "audio_file_sha256": "4" * 64,
        "num_samples": 16_000,
        "sample_rate": 16_000,
        "channels": 1,
        "pcm_dtype": "float32",
        "source_sample_rate": 48_000,
        "source_channels": 2,
    }
    if condition == "musan_10db":
        sample.update(
            {
                "noise_repo_path": "noise/free-sound/example.wav",
                "noise_raw_audio_sha256": "5" * 64,
                "noise_pcm_sha256": "6" * 64,
                "noise_segment_pcm_sha256": "7" * 64,
                "mix_entry_sha256": "8" * 64,
                "mix_selector_sha256": "9" * 64,
                "noise_offset_samples": 123,
                "noise_wrapped": False,
                "target_snr_db": 10.0,
                "achieved_snr_db": 10.0,
            }
        )
    return sample


def _record(sample: dict[str, object]) -> dict[str, object]:
    return {
        **sample,
        "engine": "zipformer",
        "model": ZIPFORMER_MODEL,
        "bank": "L-fork",
        "revision": ZIPFORMER_REVISION,
        "hypothesis": "fixture hypothesis",
        "status": "ok",
        "evidence_class": "HOST",
    }


def _engine_spec() -> dict[str, object]:
    return {
        "models": {
            "zipformer": {
                "repo_id": ZIPFORMER_MODEL,
                "bank": "L-fork",
                "revision": ZIPFORMER_REVISION,
                "languages": ["zh"],
            }
        }
    }


def test_validate_records_binds_complete_fixture_provenance() -> None:
    samples = [_fixture_sample("clean"), _fixture_sample("musan_10db")]
    validation = remote_runner.validate_records(
        [_record(sample) for sample in samples],
        engine="zipformer",
        engine_spec=_engine_spec(),
        fixture_samples=samples,
        engine_metadata={},
        expected_per_condition=1,
    )

    assert validation["all_records_fixture_bound"] is True
    assert validation["fixture_binding"] == {
        "schema": "lane-c-record-fixture-binding/1.0",
        "valid": True,
        "strict_schema": True,
        "record_count": 2,
        "base_fields": list(remote_runner._FIXTURE_BASE_MIRROR_FIELDS),
        "noise_fields": list(remote_runner._FIXTURE_NOISE_MIRROR_FIELDS),
    }


@pytest.mark.parametrize(
    ("condition", "field", "replacement"),
    (
        ("clean", "reference", "tampered reference"),
        ("clean", "reference_sha256", "a" * 64),
        ("clean", "duration_s", 2.0),
        ("clean", "audio_pcm_sha256", "b" * 64),
        ("clean", "sample_rate", 8_000),
        ("musan_10db", "mix_entry_sha256", "c" * 64),
        ("musan_10db", "noise_offset_samples", 999),
    ),
)
def test_validate_records_rejects_tampered_fixture_fields(
    condition: str, field: str, replacement: object
) -> None:
    samples = [_fixture_sample("clean"), _fixture_sample("musan_10db")]
    records = [_record(sample) for sample in samples]
    selected = 0 if condition == "clean" else 1
    records[selected][field] = replacement

    with pytest.raises(ValueError, match="fixture field|duration"):
        remote_runner.validate_records(
            records,
            engine="zipformer",
            engine_spec=_engine_spec(),
            fixture_samples=samples,
            engine_metadata={},
            expected_per_condition=1,
        )


def test_validate_records_rejects_missing_mirrored_field() -> None:
    samples = [_fixture_sample("clean"), _fixture_sample("musan_10db")]
    records = [_record(sample) for sample in samples]
    del records[1]["noise_pcm_sha256"]

    with pytest.raises(KeyError, match="noise_pcm_sha256"):
        remote_runner.validate_records(
            records,
            engine="zipformer",
            engine_spec=_engine_spec(),
            fixture_samples=samples,
            engine_metadata={},
            expected_per_condition=1,
        )


def _omni_smoke_fixture_and_records() -> tuple[
    dict[str, dict[str, object]], list[dict[str, object]]
]:
    fixture_index: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    hypotheses = {
        "en": "hello",
        "zh": "你好",
        "es": "hola",
        "fr": "bonjour",
        "ru": "привет",
    }
    for language in aggregate.LANGUAGES:
        for index in range(aggregate.TARGET_N):
            utterance_id = f"{language}-{index:03d}"
            fixture_index[f"{utterance_id}\tclean"] = {
                "utterance_id": utterance_id,
                "language": language,
                "condition": "clean",
            }
            if index < aggregate.OMNI_SMOKE_N:
                records.append(
                    {
                        "utterance_id": utterance_id,
                        "language": language,
                        "condition": "clean",
                        "hypothesis": hypotheses[language],
                        "phase": "smoke",
                        "smoke_test": True,
                        "route": "ctc-utterance/fp16-cuda/smoke",
                    }
                )
    return fixture_index, records


def test_omni_gates_are_recomputed_from_exact_smoke_hypotheses() -> None:
    fixture_index, records = _omni_smoke_fixture_and_records()

    gates, reasons = aggregate._recomputed_omni_gates(records, fixture_index)

    assert reasons == []
    assert all(gate["collapsed"] is False for gate in gates.values())
    assert all(gate["remaining_samples_evaluated"] == 270 for gate in gates.values())

    tampered = copy.deepcopy(records)
    en_smoke = [record for record in tampered if record["language"] == "en"]
    for record in en_smoke[:7]:
        record["hypothesis"] = ""
    gates, reasons = aggregate._recomputed_omni_gates(tampered, fixture_index)

    assert reasons == []
    assert gates["en"]["empty_count"] == 7
    assert gates["en"]["collapsed"] is True
    assert gates["en"]["remaining_samples_skipped"] == 270


def test_omni_phase_tags_are_bound_to_first_30_clean_fixture_rows() -> None:
    fixture_index, records = _omni_smoke_fixture_and_records()
    metadata = {
        "smoke_execution": {
            "languages_in_order": list(aggregate.LANGUAGES),
            "clean_n_per_language": aggregate.OMNI_SMOKE_N,
            "all_language_smokes_before_expansion": True,
            "expansion_policy": "only languages whose smoke gate did not collapse",
            "resumability": "record identity checkpoints are reused in both phases",
        }
    }

    assert aggregate._omni_phase_reasons(
        records, fixture_index, set(), metadata
    ) == []

    records[0]["phase"] = "evaluation"
    reasons = aggregate._omni_phase_reasons(records, fixture_index, set(), metadata)
    assert any("tags mismatch" in reason for reason in reasons)


def test_fixture_schema_bump_rejects_old_cached_runtime_manifest(
    tmp_path: Path,
) -> None:
    runtime_manifest = tmp_path / "fixture_runtime.json"
    runtime_manifest.write_text(
        json.dumps({"schema": "lane-c-asr-fixture/1.0", "languages": ["zh"]}),
        encoding="utf-8",
    )

    assert remote_runner is not None
    import colab_common as common

    assert common._resume_fixture(runtime_manifest, ["zh"], tmp_path) is None


def _score_cell(
    condition: str, fixture_sha256: str, mix_manifest_sha256: str
) -> dict[str, object]:
    return {
        "model": ZIPFORMER_MODEL,
        "bank": "L-fork",
        "revision": ZIPFORMER_REVISION,
        "language": "zh",
        "condition": condition,
        "corpus": (
            f"google/fleurs@{aggregate.FLEURS_REVISION}/cmn_hans_cn/test"
        ),
        "n": aggregate.TARGET_N,
        "bootstrap_B": aggregate.BOOTSTRAP_B,
        "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
        "fixture_sha256": fixture_sha256,
        "mix_manifest_sha256": mix_manifest_sha256,
        "evidence_class": "HOST",
    }


def _complete_zipformer_payloads() -> dict[str, bytes]:
    provenance = {
        "model": ZIPFORMER_MODEL,
        "bank": "L-fork",
        "revision": ZIPFORMER_REVISION,
    }
    cells = [
        {
            **provenance,
            "language": "zh",
            "condition": condition,
            "expected_records": aggregate.TARGET_N,
            "records": aggregate.TARGET_N,
        }
        for condition in aggregate.CONDITIONS
    ]
    noise_files = [f"noise/free-sound/noise-{index}.wav" for index in range(738)]
    musan = {
        "source_corpus": "MUSAN",
        "subset_mirror": True,
        "repo_id": aggregate.MUSAN_REPO,
        "revision": aggregate.MUSAN_REVISION,
        "resolved_commit": aggregate.MUSAN_REVISION,
        "glob": aggregate.MUSAN_NOISE_GLOB,
        "expected_noise_file_count": aggregate.MUSAN_NOISE_WAV_COUNT,
        "provenance_files": [],
        "files": noise_files,
    }
    noise_bank_sha256 = aggregate.canonical_sha256(musan)
    sample_manifest: list[dict[str, object]] = []
    mix_entries: list[dict[str, object]] = []
    for index in range(aggregate.TARGET_N):
        utterance_id = f"fleurs:cmn_hans_cn:{index:04d}"
        reference = f"参考文本{index}"
        base = {
            "utterance_id": utterance_id,
            "language": "zh",
            "config": "cmn_hans_cn",
            "reference": reference,
            "reference_source": "raw_transcription",
            "raw_transcription": reference,
            "transcription": reference,
            "source_id": str(index),
            "fleurs_id": str(index),
            "source_path": f"source-{index}.wav",
            "source_parquet_repo_path": "data/cmn_hans_cn/test/part.parquet",
            "raw_audio_sha256": hashlib.sha256(f"raw-{index}".encode()).hexdigest(),
            "clean_pcm_sha256": hashlib.sha256(f"clean-{index}".encode()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            "num_samples": 16_000,
            "duration_seconds": 1.0,
            "duration_s": 1.0,
            "sample_rate": 16_000,
            "channels": 1,
            "pcm_dtype": "float32-le",
            "source_sample_rate": 48_000,
            "source_channels": 2,
        }
        clean_row = {
            **base,
            "condition": "clean",
            "audio_relpath": f"audio/zh/{index:04d}.clean.wav",
            "audio_pcm_sha256": base["clean_pcm_sha256"],
            "audio_file_sha256": hashlib.sha256(
                f"clean-file-{index}".encode()
            ).hexdigest(),
        }
        entry: dict[str, object] = {
            "utterance_id": utterance_id,
            "language": "zh",
            "condition": "musan_10db",
            "seed": aggregate.BOOTSTRAP_SEED,
            "target_snr_db": 10.0,
            "selector_sha256": hashlib.sha256(
                f"selector-{index}".encode()
            ).hexdigest(),
            "noise_repo_path": noise_files[index],
            "noise_file_index": index,
            "offset_samples": index,
            "noise_wrapped": False,
            "clean_num_samples": 16_000,
            "noise_source_num_samples": 32_000,
            "clean_raw_audio_sha256": base["raw_audio_sha256"],
            "clean_pcm_sha256": base["clean_pcm_sha256"],
            "noise_raw_audio_sha256": hashlib.sha256(
                f"noise-raw-{index}".encode()
            ).hexdigest(),
            "noise_pcm_sha256": hashlib.sha256(
                f"noise-pcm-{index}".encode()
            ).hexdigest(),
            "noise_segment_pcm_sha256": hashlib.sha256(
                f"noise-segment-{index}".encode()
            ).hexdigest(),
            "mixed_pcm_sha256": hashlib.sha256(
                f"mixed-{index}".encode()
            ).hexdigest(),
            "clean_power": 1.0,
            "noise_power": 1.0,
            "noise_scale": 1.0,
            "postmix_gain": 1.0,
            "achieved_snr_db": 10.0,
        }
        entry["entry_sha256"] = aggregate.canonical_sha256(entry)
        noisy_row = {
            **base,
            "condition": "musan_10db",
            "audio_relpath": f"audio/zh/{index:04d}.musan_10db.wav",
            "audio_pcm_sha256": entry["mixed_pcm_sha256"],
            "audio_file_sha256": hashlib.sha256(
                f"noisy-file-{index}".encode()
            ).hexdigest(),
            "noise_repo_path": entry["noise_repo_path"],
            "noise_raw_audio_sha256": entry["noise_raw_audio_sha256"],
            "noise_pcm_sha256": entry["noise_pcm_sha256"],
            "noise_segment_pcm_sha256": entry["noise_segment_pcm_sha256"],
            "mix_entry_sha256": entry["entry_sha256"],
            "mix_selector_sha256": entry["selector_sha256"],
            "noise_offset_samples": entry["offset_samples"],
            "noise_wrapped": entry["noise_wrapped"],
            "target_snr_db": entry["target_snr_db"],
            "achieved_snr_db": entry["achieved_snr_db"],
        }
        sample_manifest.extend((clean_row, noisy_row))
        mix_entries.append(entry)
    mix_manifest = {
        "schema": aggregate.MIX_MANIFEST_SCHEMA,
        "seed": aggregate.BOOTSTRAP_SEED,
        "target_snr_db": 10.0,
        "noise_bank_sha256": noise_bank_sha256,
        "entries": mix_entries,
    }
    mix_manifest_sha256 = aggregate.canonical_sha256(mix_manifest)
    fleurs = {
        "repo_id": "google/fleurs",
        "revision": aggregate.FLEURS_REVISION,
        "resolved_commit": aggregate.FLEURS_REVISION,
        "split": "test",
        "configs": {"zh": "cmn_hans_cn"},
        "test_row_counts": {"zh": aggregate.FLEURS_TEST_ROWS["zh"]},
    }
    audio = {"sample_rate": 16_000, "channels": 1, "pcm_dtype": "float32-le"}
    fixture_manifest = {
        "schema": aggregate.FIXTURE_SCHEMA,
        "fleurs": fleurs,
        "audio": audio,
        "musan": musan,
        "noise_bank_sha256": noise_bank_sha256,
        "mix_manifest_sha256": mix_manifest_sha256,
        "samples": sample_manifest,
    }
    fixture_manifest_sha256 = aggregate.canonical_sha256(fixture_manifest)
    fixture = {
        "schema": aggregate.FIXTURE_SCHEMA,
        "languages": ["zh"],
        "conditions": list(aggregate.CONDITIONS),
        "samples_per_language_per_condition": aggregate.TARGET_N,
        "sample_count": len(sample_manifest),
        "sample_manifest": sample_manifest,
        "sample_index": {
            f"{row['utterance_id']}\t{row['condition']}": row
            for row in sample_manifest
        },
        "fixture_hashes_by_language": {
            "zh": aggregate.canonical_sha256(sample_manifest)
        },
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "fixture_sha256": fixture_manifest_sha256,
        "mix_manifest_sha256": mix_manifest_sha256,
        "mix_manifest": mix_manifest,
        "noise_bank_sha256": noise_bank_sha256,
        "fleurs": fleurs,
        "musan": musan,
        "audio": audio,
    }
    records = [
        {
            **sample,
            "audio_path": f"/fixture/{sample['audio_relpath']}",
            "engine": "zipformer",
            **provenance,
            "route": "online-transducer/modified-beam-search",
            "hypothesis": f"hypothesis {sample['utterance_id']}",
            "status": "ok",
            "evidence_class": "HOST",
        }
        for sample in sample_manifest
    ]
    binding = {
        "schema": "lane-c-record-fixture-binding/1.0",
        "valid": True,
        "strict_schema": True,
        "record_count": len(records),
        "base_fields": list(aggregate.FIXTURE_BINDING_BASE_FIELDS),
        "noise_fields": list(aggregate.FIXTURE_BINDING_NOISE_FIELDS),
    }
    validation = {
        "schema": "lane-c-record-validation/1.1",
        "engine": "zipformer",
        "valid": True,
        "record_count": len(records),
        "expected_record_count": len(records),
        "all_inference_status_ok": True,
        "all_records_fixture_bound": True,
        "fixture_binding": binding,
        "omni_collapsed_languages": [],
        "expected_provenance": [provenance],
        "counts": [{**provenance, "records": len(records)}],
        "cell_counts": cells,
    }
    installs: dict[str, object] = {}
    asset_group = aggregate.MODEL_ASSET_GROUPS[ZIPFORMER_MODEL][0]
    metadata: dict[str, object] = {
        "artifacts": [
            {
                "repo_id": ZIPFORMER_MODEL,
                "revision": ZIPFORMER_REVISION,
                "files": [
                    {
                        "relative_path": item["file"],
                        "sha256_observed": item["sha256"],
                        "sha256_verified": True,
                        **(
                            {
                                "bytes_observed": item["bytes"],
                                "bytes_verified": True,
                            }
                            if "bytes" in item
                            else {}
                        ),
                    }
                    for item in asset_group
                ],
            }
        ]
    }
    environment: dict[str, object] = {}
    clean_records = [record for record in records if record["condition"] == "clean"]
    partial_traces: list[dict[str, object]] = []
    for record in clean_records[: aggregate.PARTIAL_TRACE_N]:
        trace: dict[str, object] = {
            "engine": "zipformer",
            **provenance,
            "route": "Zipformer online",
            "utterance_id": record["utterance_id"],
            "language": "zh",
            "condition": "clean",
            "audio_path": record["audio_path"],
            "duration_s": record["duration_s"],
            "first_partial_latency_s": 0.1,
            "final_hypothesis": record["hypothesis"],
            "partials": [
                {
                    "text": record["hypothesis"],
                    "audio_time_s": 1.0,
                    "wall_elapsed_s": 0.1,
                }
            ],
        }
        trace["trace_sha256"] = aggregate.canonical_sha256(trace)
        partial_traces.append(trace)
    partial_metrics: list[object] = [
        {
            "route": "Zipformer online",
            "language": "zh",
            "n_input": aggregate.PARTIAL_TRACE_N,
            "n_trace_events_input": aggregate.PARTIAL_TRACE_N,
            "n_valid": aggregate.PARTIAL_TRACE_N,
            "n_invalid": 0,
            "n_partials_emitted": aggregate.PARTIAL_TRACE_N,
            "n_no_partials": 0,
            "partials_emitted": True,
            "first_partial_latency_ms": 100.0,
            "partial_rewrite_rate_edits_per_audio_second": 0.0,
            "final_vs_last_partial_divergence": 0.0,
            "bootstrap_B": aggregate.BOOTSTRAP_B,
            "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
            "evidence_class": "HOST",
            "utterances": [
                {"utterance_id": trace["utterance_id"], "partials_emitted": True}
                for trace in partial_traces
            ],
        }
    ]
    scored_models = [
        {
            **provenance,
            "n_records": len(records),
            "bootstrap_B": aggregate.BOOTSTRAP_B,
            "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
            "corpus_repo_id": "google/fleurs",
            "corpus_revision": aggregate.FLEURS_REVISION,
            "corpus_split": "test",
            "fixture_manifest_sha256": fixture_manifest_sha256,
            "mix_manifest_sha256": mix_manifest_sha256,
            "normalisers_source_sha256": aggregate.NORMALISERS_SHA256,
            "evidence_class": "HOST",
            "cells": [
                _score_cell(
                    condition,
                    fixture["fixture_hashes_by_language"]["zh"],
                    mix_manifest_sha256,
                )
                for condition in aggregate.CONDITIONS
            ],
        }
    ]
    run_status = {
        "schema": "lane-c-colab-engine-result/1.0",
        "engine": "zipformer",
        "venue": "Google Colab",
        "evidence_class": "HOST",
        "training_performed": False,
        "status": "COMPLETE",
        "languages": ["zh"],
        "records": len(records),
        "partial_traces": len(partial_traces),
        "models": [provenance],
        "record_validation": validation,
    }
    engine_result = {
        **run_status,
        "fixture": fixture,
        "install_receipts": installs,
        "engine_metadata": metadata,
        "scored_models": scored_models,
        "partial_metrics": partial_metrics,
        "environment": environment,
    }

    def encoded(value: object) -> bytes:
        return (json.dumps(value, sort_keys=True) + "\n").encode()

    return {
        "engine_metadata.json": encoded(metadata),
        "engine_result.json": encoded(engine_result),
        "environment.json": encoded(environment),
        "fixture_receipt.json": encoded(fixture),
        "install_receipts.json": encoded(installs),
        "partial_metrics.json": encoded(partial_metrics),
        "partial_traces.jsonl": b"".join(
            encoded(trace) for trace in partial_traces
        ),
        "records.jsonl": b"".join(encoded(record) for record in records),
        "run_status.json": encoded(run_status),
        "scored_models.json": encoded(scored_models),
    }


def _write_archive(
    results_root: Path,
    payloads: dict[str, bytes],
    *,
    manifest_mutation=None,
) -> Path:
    archive_path = results_root / "zipformer" / "result.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": aggregate.ARCHIVE_SCHEMA,
        "policy": "exact text-evidence allowlist; audio and model artifacts excluded",
        "allowlist": sorted(aggregate.ARCHIVE_PAYLOAD_ALLOWLIST),
        "included": [
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for name, raw in sorted(payloads.items())
        ],
    }
    if manifest_mutation is not None:
        manifest_mutation(manifest)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payloads.items()):
            archive.writestr(name, raw)
        archive.writestr("archive_manifest.json", json.dumps(manifest, sort_keys=True))
    return archive_path


def test_read_engine_zip_accepts_fully_bound_complete_archive(tmp_path: Path) -> None:
    _write_archive(tmp_path, _complete_zipformer_payloads())

    result = aggregate.read_engine_zip(tmp_path, "zipformer")

    assert result is not None
    assert result["status"] == "COMPLETE"
    assert result["_archive_validation"]["valid"] is True
    assert aggregate.valid_sha256(result["_member_sha256"]["records.jsonl"])


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_sha",
        "manifest_size",
        "missing_records",
        "wrong_engine",
        "training",
        "validation_count",
        "record_and_receipt",
        "missing_partial",
        "artifact_sha",
    ),
)
def test_read_engine_zip_rejects_tampered_complete_archive(
    tmp_path: Path, mutation: str
) -> None:
    payloads = _complete_zipformer_payloads()
    manifest_mutation = None
    if mutation == "manifest_sha":
        manifest_mutation = lambda manifest: manifest["included"][0].update(
            sha256="0" * 64
        )
    elif mutation == "manifest_size":
        manifest_mutation = lambda manifest: manifest["included"][0].update(bytes=0)
    elif mutation == "missing_records":
        del payloads["records.jsonl"]
    elif mutation == "record_and_receipt":
        records = [
            json.loads(line)
            for line in payloads["records.jsonl"].decode().splitlines()
        ]
        records[0]["reference"] = "coordinated tamper"
        payloads["records.jsonl"] = b"".join(
            (json.dumps(record, sort_keys=True) + "\n").encode()
            for record in records
        )
        engine_result = json.loads(payloads["engine_result.json"])
        run_status = json.loads(payloads["run_status.json"])
        engine_result["record_validation"]["valid"] = True
        engine_result["record_validation"]["all_records_fixture_bound"] = True
        run_status["record_validation"] = engine_result["record_validation"]
        payloads["engine_result.json"] = (
            json.dumps(engine_result, sort_keys=True) + "\n"
        ).encode()
        payloads["run_status.json"] = (
            json.dumps(run_status, sort_keys=True) + "\n"
        ).encode()
    elif mutation == "missing_partial":
        traces = payloads["partial_traces.jsonl"].decode().splitlines()[:-1]
        payloads["partial_traces.jsonl"] = (
            ("\n".join(traces) + "\n").encode() if traces else b""
        )
        engine_result = json.loads(payloads["engine_result.json"])
        run_status = json.loads(payloads["run_status.json"])
        metric = engine_result["partial_metrics"][0]
        for key in ("n_input", "n_trace_events_input", "n_valid", "n_partials_emitted"):
            metric[key] -= 1
        metric["utterances"].pop()
        engine_result["partial_traces"] -= 1
        run_status["partial_traces"] -= 1
        payloads["partial_metrics.json"] = (
            json.dumps(engine_result["partial_metrics"], sort_keys=True) + "\n"
        ).encode()
        payloads["engine_result.json"] = (
            json.dumps(engine_result, sort_keys=True) + "\n"
        ).encode()
        payloads["run_status.json"] = (
            json.dumps(run_status, sort_keys=True) + "\n"
        ).encode()
    elif mutation == "artifact_sha":
        metadata = json.loads(payloads["engine_metadata.json"])
        metadata["artifacts"][0]["files"][0]["sha256_observed"] = "0" * 64
        engine_result = json.loads(payloads["engine_result.json"])
        engine_result["engine_metadata"] = metadata
        payloads["engine_metadata.json"] = (
            json.dumps(metadata, sort_keys=True) + "\n"
        ).encode()
        payloads["engine_result.json"] = (
            json.dumps(engine_result, sort_keys=True) + "\n"
        ).encode()
    else:
        engine_result = json.loads(payloads["engine_result.json"])
        run_status = json.loads(payloads["run_status.json"])
        if mutation == "wrong_engine":
            engine_result["engine"] = "omni"
            run_status["engine"] = "omni"
        elif mutation == "training":
            engine_result["training_performed"] = True
            run_status["training_performed"] = True
        elif mutation == "validation_count":
            engine_result["record_validation"]["cell_counts"][0]["records"] -= 1
            run_status["record_validation"]["cell_counts"][0]["records"] -= 1
        payloads["engine_result.json"] = (
            json.dumps(engine_result, sort_keys=True) + "\n"
        ).encode()
        payloads["run_status.json"] = (
            json.dumps(run_status, sort_keys=True) + "\n"
        ).encode()
    _write_archive(tmp_path, payloads, manifest_mutation=manifest_mutation)

    result = aggregate.read_engine_zip(tmp_path, "zipformer")

    assert result is not None
    assert result["status"] == "FAILED"
    assert result["evidence_class"] == "UNKNOWN"
    assert result["_archive_validation"]["valid"] is False


def test_invalid_archive_forces_table_and_c4_cells_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _complete_zipformer_payloads()
    _write_archive(
        tmp_path,
        payloads,
        manifest_mutation=lambda manifest: manifest["included"][0].update(
            sha256="0" * 64
        ),
    )
    result = aggregate.read_engine_zip(tmp_path, "zipformer")
    assert result is not None
    monkeypatch.setattr(aggregate, "fixture_pin_reasons", lambda *_: [])
    monkeypatch.setattr(aggregate, "artifact_pin_reasons", lambda *_: [])
    engines = {
        "sensevoice": None,
        "nemo": None,
        "zipformer": result,
        "gigaam": None,
        "omni": None,
    }

    table_row = next(
        row
        for row in aggregate.build_table(engines)
        if row["bank"] == "L-fork" and row["language"] == "zh"
    )
    c4_row = next(
        row
        for row in aggregate.build_c4(engines)["rows"]
        if row["route"] == "Zipformer online" and row["language"] == "zh"
    )

    assert table_row["clean_evidence_class"] == "UNKNOWN"
    assert table_row["noisy_bucket"]["evidence_class"] == "UNKNOWN"
    assert c4_row["evidence_class"] == "UNKNOWN"
    assert "archive evidence validation failed" in c4_row["blocker"]
