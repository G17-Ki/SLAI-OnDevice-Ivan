"""Focused fail-closed tests for Lane C result aggregation."""

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


OMNI_MODEL = "facebook/omniASR-CTC-300M"
OMNI_REVISION = "8e35f0cc28fa6099e0c14d56db85ce0423baa691"
ZIPFORMER_MODEL = "pfluo/k2fsa-zipformer-chinese-english-mixed"
ZIPFORMER_REVISION = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"
SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"
SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"


def _score_cell(
    *,
    model: str,
    bank: str,
    revision: str,
    language: str,
    condition: str = "clean",
    n: int = aggregate.TARGET_N,
) -> dict[str, object]:
    return {
        "model": model,
        "bank": bank,
        "revision": revision,
        "language": language,
        "condition": condition,
        "corpus": (
            f"google/fleurs@{aggregate.FLEURS_REVISION}/"
            f"{aggregate.FLEURS_CONFIGS[language]}/test"
        ),
        "n": n,
        "metric": "cer" if language == "zh" else "wer",
        "percent": 42.0,
        "ci95_percent": [40.0, 44.0],
        "normalizer_id": f"norm_{language}/test",
        "basic_normalizer_id": "basic/test",
        "basic_delta_pp": 0.5,
        "normalizer_delta_pp": 0.5,
        "ranking_inadmissible": False,
        "bootstrap_B": aggregate.BOOTSTRAP_B,
        "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
        "ci_method": "percentile utterance bootstrap, ratio of sums",
        "fixture_sha256": "a" * 64,
        "evidence_class": "HOST",
    }


def _omni_collapsed_result() -> dict[str, object]:
    cells = [
        _score_cell(
            model=OMNI_MODEL,
            bank="OmniASR",
            revision=OMNI_REVISION,
            language=language,
            n=aggregate.OMNI_SMOKE_N,
        )
        for language in aggregate.LANGUAGES
    ]
    gates = {
        language: {
            "language": language,
            "smoke_n_expected": aggregate.OMNI_SMOKE_N,
            "smoke_n_observed": aggregate.OMNI_SMOKE_N,
            "remaining_samples_skipped": (
                2 * aggregate.TARGET_N - aggregate.OMNI_SMOKE_N
            ),
            "collapsed": True,
            "reasons": ["empty fraction 1.000 > 0.20"],
        }
        for language in aggregate.LANGUAGES
    }
    return {
        "engine": "omni",
        "status": "COMPLETE",
        "evidence_class": "HOST",
        "record_validation": {
            "valid": True,
            "all_inference_status_ok": True,
            "omni_collapsed_languages": list(aggregate.LANGUAGES),
        },
        "engine_metadata": {"smoke_gate": gates},
        "fixture": {"fixture_manifest_sha256": "b" * 64},
        "scored_models": [
            {
                "model": OMNI_MODEL,
                "bank": "OmniASR",
                "revision": OMNI_REVISION,
                "evidence_class": "HOST",
                "fixture_manifest_sha256": "b" * 64,
                "normalisers_source_sha256": aggregate.NORMALISERS_SHA256,
                "bootstrap_B": aggregate.BOOTSTRAP_B,
                "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
                "cells": cells,
            }
        ],
        "_archive_path": "artifacts/results/omni/result.zip",
        "_archive_sha256": "c" * 64,
        "_archive_members": ["engine_result.json", "records.jsonl"],
        "_member_sha256": {
            "engine_result.json": "d" * 64,
            "records.jsonl": "e" * 64,
        },
    }


def _engine_results_with_omni(
    result: dict[str, object] | None,
) -> dict[str, dict[str, object] | None]:
    return {
        "sensevoice": None,
        "nemo": None,
        "zipformer": None,
        "gigaam": None,
        "omni": result,
    }


def _zipformer_result() -> dict[str, object]:
    traces: list[dict[str, object]] = []
    for index in range(aggregate.PARTIAL_TRACE_N):
        trace: dict[str, object] = {
            "route": "Zipformer online",
            "model": ZIPFORMER_MODEL,
            "language": "zh",
            "condition": "clean",
            "utterance_id": f"zh-{index}",
            "partials": [{"text": "test", "stream_s": 0.1}],
        }
        trace["trace_sha256"] = aggregate.canonical_sha256(trace)
        traces.append(trace)
    metric = {
        "route": "Zipformer online",
        "language": "zh",
        "partials_emitted": True,
        "first_partial_latency_ms": 100.0,
        "partial_rewrite_rate_edits_per_audio_second": 0.1,
        "final_vs_last_partial_divergence": 0.0,
        "normalizer_id": "norm_zh/test",
        "n_input": aggregate.PARTIAL_TRACE_N,
        "n_invalid": 0,
        "bootstrap_B": aggregate.BOOTSTRAP_B,
        "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
        "evidence_class": "HOST",
    }
    return {
        "engine": "zipformer",
        "status": "COMPLETE",
        "evidence_class": "HOST",
        "record_validation": {
            "valid": True,
            "all_inference_status_ok": True,
        },
        "engine_metadata": {
            "artifact_caveats": {
                "checkpoint_mismatch": True,
                "readme_score_checkpoint": "epoch-14.pt",
                "selected_export_checkpoint": "epoch-99-avg-1",
            },
            "context_policy": {"mode": "online"},
            "partial_capability": {"native": True},
            "settings": {"chunk_ms": 100},
        },
        "scored_models": [
            {
                "model": ZIPFORMER_MODEL,
                "bank": "L-fork",
                "revision": ZIPFORMER_REVISION,
                "evidence_class": "HOST",
                "cells": [
                    _score_cell(
                        model=ZIPFORMER_MODEL,
                        bank="L-fork",
                        revision=ZIPFORMER_REVISION,
                        language="zh",
                    )
                ],
            }
        ],
        "partial_metrics": [metric],
        "_partial_traces": traces,
        "_archive_path": "artifacts/results/zipformer/result.zip",
        "_archive_sha256": "f" * 64,
        "_member_sha256": {"partial_traces.jsonl": "1" * 64},
    }


def _zipformer_c4_row(result: dict[str, object]) -> dict[str, object]:
    payload = aggregate.build_c4(
        {
            "sensevoice": None,
            "nemo": None,
            "zipformer": result,
            "gigaam": None,
            "omni": None,
        }
    )
    return next(
        row
        for row in payload["rows"]
        if row["route"] == "Zipformer online" and row["language"] == "zh"
    )


def _sensevoice_probe_result() -> dict[str, object]:
    manifest: list[dict[str, object]] = []
    languages: dict[str, object] = {}
    for language in ("en", "zh"):
        entries: list[dict[str, object]] = []
        for index in range(aggregate.SENSEVOICE_FIXED_CONTEXT_PROBE_N):
            num_samples = 80_000 if index % 2 == 0 else 200_000
            sample = {
                "utterance_id": f"{language}-clean-{index:02d}",
                "language": language,
                "condition": "clean",
                "audio_relpath": f"audio/{language}-{index:02d}.wav",
                "audio_pcm_sha256": "1" * 64,
                "clean_pcm_sha256": "1" * 64,
                "audio_file_sha256": "2" * 64,
                "reference_sha256": "3" * 64,
                "num_samples": num_samples,
                "sample_rate": aggregate.SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
            }
            manifest.append(sample)
            entry: dict[str, object] = {
                "utterance_id": sample["utterance_id"],
                "language": language,
                "condition": "clean",
                "audio_relpath": sample["audio_relpath"],
                "source_audio_pcm_sha256": sample["audio_pcm_sha256"],
                "source_clean_pcm_sha256": sample["clean_pcm_sha256"],
                "source_audio_file_sha256": sample["audio_file_sha256"],
                "source_reference_sha256": sample["reference_sha256"],
                "source_num_samples": num_samples,
                "source_sample_rate": aggregate.SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
                "context_seconds": aggregate.SENSEVOICE_FIXED_CONTEXT_SECONDS,
                "derived_num_samples": aggregate.SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES,
                "derived_sample_rate": aggregate.SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
                "scoring_included": False,
                "reference_modified": False,
                "model_invoked": True,
                "status": "ok",
                "transformation": (
                    "right_zero_pad"
                    if num_samples < aggregate.SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES
                    else "truncate_tail"
                ),
                "copied_source_samples": min(
                    num_samples, aggregate.SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES
                ),
                "derived_pcm_sha256": "4" * 64,
                "derived_wav_sha256": "5" * 64,
                "final_text_sha256": "6" * 64,
                "final_text_empty": False,
            }
            entry["entry_sha256"] = aggregate.canonical_sha256(entry)
            entries.append(entry)
        languages[language] = {
            "expected": aggregate.SENSEVOICE_FIXED_CONTEXT_PROBE_N,
            "selected": aggregate.SENSEVOICE_FIXED_CONTEXT_PROBE_N,
            "invoked": aggregate.SENSEVOICE_FIXED_CONTEXT_PROBE_N,
            "succeeded": aggregate.SENSEVOICE_FIXED_CONTEXT_PROBE_N,
            "failed": 0,
            "samples": entries,
        }
    receipt: dict[str, object] = {
        "schema": "lane-c-sensevoice-fixed-context-probe/1.0",
        "route": "fixed-10s-context/final-only",
        "model": SENSEVOICE_MODEL,
        "bank": "D-fork",
        "revision": SENSEVOICE_REVISION,
        "context_seconds": aggregate.SENSEVOICE_FIXED_CONTEXT_SECONDS,
        "sample_rate": aggregate.SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
        "target_num_samples": aggregate.SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES,
        "probe_n_per_language": aggregate.SENSEVOICE_FIXED_CONTEXT_PROBE_N,
        "condition": "clean",
        "main_scoring_inputs_unchanged": True,
        "scoring_included": False,
        "reference_scoring_included": False,
        "partial_api": "utterance-only/no partial API",
        "partials_emitted": False,
        "languages": languages,
        "valid": True,
    }
    receipt["receipt_sha256"] = aggregate.canonical_sha256(receipt)
    return {
        "engine": "sensevoice",
        "status": "COMPLETE",
        "evidence_class": "HOST",
        "record_validation": {
            "valid": True,
            "all_inference_status_ok": True,
        },
        "fixture": {"sample_manifest": manifest},
        "engine_metadata": {"fixed_context_probe": receipt},
        "scored_models": [
            {
                "model": SENSEVOICE_MODEL,
                "bank": "D-fork",
                "revision": SENSEVOICE_REVISION,
                "evidence_class": "HOST",
                "cells": [
                    _score_cell(
                        model=SENSEVOICE_MODEL,
                        bank="D-fork",
                        revision=SENSEVOICE_REVISION,
                        language=language,
                    )
                    for language in ("en", "zh")
                ],
            }
        ],
    }


def test_c4_requires_valid_fixture_bound_sensevoice_10s_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "artifact_pin_reasons", lambda *_: [])
    result = _sensevoice_probe_result()
    payload = aggregate.build_c4(
        {
            "sensevoice": result,
            "nemo": None,
            "zipformer": None,
            "gigaam": None,
            "omni": None,
        }
    )
    sensevoice_rows = [
        row for row in payload["rows"] if row["engine"] == "sensevoice"
    ]
    assert len(sensevoice_rows) == 2
    assert all(row["evidence_class"] == "HOST" for row in sensevoice_rows)
    assert all(row["partials_emitted"] is False for row in sensevoice_rows)
    assert all(row["fixed_context_probe"]["n"] == 30 for row in sensevoice_rows)
    assert all(
        row["fixed_context_probe"]["scoring_included"] is False
        and row["fixed_context_probe"]["main_scoring_inputs_unchanged"] is True
        for row in sensevoice_rows
    )

    tampered = copy.deepcopy(result)
    probe = tampered["engine_metadata"]["fixed_context_probe"]
    probe["scoring_included"] = True
    probe_without_hash = dict(probe)
    probe_without_hash.pop("receipt_sha256")
    probe["receipt_sha256"] = aggregate.canonical_sha256(probe_without_hash)
    tampered_payload = aggregate.build_c4(
        {
            "sensevoice": tampered,
            "nemo": None,
            "zipformer": None,
            "gigaam": None,
            "omni": None,
        }
    )
    tampered_rows = [
        row
        for row in tampered_payload["rows"]
        if row["engine"] == "sensevoice"
    ]
    assert all(row["evidence_class"] == "UNKNOWN" for row in tampered_rows)
    assert all("protocol mismatch" in row["blocker"] for row in tampered_rows)


def test_collapsed_omni_is_host_smoke_but_never_comparable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(aggregate, "fixture_pin_reasons", lambda *_: [])
    monkeypatch.setattr(aggregate, "artifact_pin_reasons", lambda *_: [])
    result = _omni_collapsed_result()
    rows = aggregate.build_table(_engine_results_with_omni(result))
    omni_rows = [row for row in rows if row["bank"] == "OmniASR"]

    assert len(omni_rows) == len(aggregate.LANGUAGES)
    assert aggregate.c3_completion(omni_rows)["complete"] is True
    for row in omni_rows:
        assert row["clean_evidence_class"] == "HOST"
        assert row["clean_evaluation_scope"] == "collapsed_smoke"
        assert row["clean_smoke_evidence"] is True
        assert row["clean_comparable"] is False
        assert row["ranking_inadmissible"] is True
        assert row["evidence_class"] == "UNKNOWN"
        assert row["noisy_bucket"]["evidence_class"] == "UNKNOWN"
        assert row["noisy_bucket"]["n"] == 0
    assert aggregate.comparability(rows)["all_cells_comparable"] is False

    monkeypatch.setattr(
        aggregate,
        "read_engine_zip",
        lambda _root, engine: result if engine == "omni" else None,
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["aggregate_results.py", "--out-dir", str(out_dir)],
    )
    assert aggregate.main() == 0
    c3 = json.loads((out_dir / "C3.json").read_text(encoding="utf-8"))
    c5 = json.loads((out_dir / "C5.json").read_text(encoding="utf-8"))
    assert c3["status"] == "COMPLETE"
    assert set(c3["completion"]["language_modes"].values()) == {
        "collapsed_n30_clean_smoke"
    }
    assert c5["status"] == "PARTIAL"
    assert "MUSAN subset-mirror caveat" in (
        out_dir / "LANE_SUMMARY.md"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result: result.update(status="FAILED"),
        lambda result: result["record_validation"].update(valid=False),
        lambda result: result["engine_metadata"]["smoke_gate"]["en"].update(
            collapsed=False
        ),
    ),
)
def test_unvalidated_or_failed_omni_smoke_is_unknown(
    monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    monkeypatch.setattr(aggregate, "fixture_pin_reasons", lambda *_: [])
    monkeypatch.setattr(aggregate, "artifact_pin_reasons", lambda *_: [])
    result = _omni_collapsed_result()
    mutation(result)
    row = next(
        row
        for row in aggregate.build_table(_engine_results_with_omni(result))
        if row["bank"] == "OmniASR" and row["language"] == "en"
    )
    assert row["clean_evidence_class"] == "UNKNOWN"
    assert row["clean_smoke_evidence"] is False
    assert row["clean_evaluation_scope"] == "unknown"


def test_archive_paths_are_portable(tmp_path: Path) -> None:
    results_root = tmp_path / "detached-results"
    archive_path = results_root / "omni" / "result.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "engine_result.json",
            json.dumps(
                {
                    "engine": "omni",
                    "status": "COMPLETE",
                    "evidence_class": "HOST",
                }
            ),
        )
    receipt = aggregate.read_engine_zip(results_root, "omni")
    assert receipt is not None
    assert receipt["_archive_path"] == "artifacts/results/omni/result.zip"
    assert aggregate._relative_artifact_path(
        "/unrelated/build/out/c/artifacts/results/omni/result.zip", tmp_path
    ) == "artifacts/results/omni/result.zip"
    assert aggregate._relative_artifact_path(
        "/private/nonportable/result.zip", tmp_path
    ) is None

    compact = aggregate.compact_engine_receipts(
        {"zipformer": _zipformer_result()}, tmp_path
    )["zipformer"]
    assert compact["engine_metadata"]["context_policy"] == {"mode": "online"}
    assert compact["engine_metadata"]["artifact_caveats"] == {
        "checkpoint_mismatch": True,
        "readme_score_checkpoint": "epoch-14.pt",
        "selected_export_checkpoint": "epoch-99-avg-1",
    }
    assert compact["engine_metadata"]["partial_capability"] == {"native": True}
    assert compact["engine_metadata"]["settings"] == {"chunk_ms": 100}
    assert compact["record_validation"] == {
        "valid": True,
        "all_inference_status_ok": True,
    }


def test_full_host_cell_requires_validated_all_ok_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "fixture_pin_reasons", lambda *_: [])
    monkeypatch.setattr(aggregate, "artifact_pin_reasons", lambda *_: [])
    result = _zipformer_result()
    engines = {
        "sensevoice": None,
        "nemo": None,
        "zipformer": result,
        "gigaam": None,
        "omni": None,
    }
    row = next(
        row
        for row in aggregate.build_table(engines)
        if row["bank"] == "L-fork" and row["language"] == "zh"
    )
    assert row["clean_evidence_class"] == "HOST"

    result["record_validation"]["all_inference_status_ok"] = False
    row = next(
        row
        for row in aggregate.build_table(engines)
        if row["bank"] == "L-fork" and row["language"] == "zh"
    )
    assert row["clean_evidence_class"] == "UNKNOWN"
    assert "inference statuses did not validate as all-ok" in row["blocker"]["clean"]


def test_c4_accepts_only_host_zero_invalid_metrics_and_verified_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "artifact_pin_reasons", lambda *_: [])
    baseline = _zipformer_result()
    row = _zipformer_c4_row(baseline)
    assert row["evidence_class"] == "HOST"
    assert row["trace_sha256_verified_count"] == aggregate.PARTIAL_TRACE_N

    invalid_metric = copy.deepcopy(baseline)
    invalid_metric["partial_metrics"][0]["n_invalid"] = 1
    row = _zipformer_c4_row(invalid_metric)
    assert row["evidence_class"] == "UNKNOWN"
    assert "n_invalid=1" in row["blocker"]

    non_host_metric = copy.deepcopy(baseline)
    non_host_metric["partial_metrics"][0]["evidence_class"] = "UNKNOWN"
    row = _zipformer_c4_row(non_host_metric)
    assert row["evidence_class"] == "UNKNOWN"
    assert "metric evidence class is not HOST" in row["blocker"]

    invalid_record_receipt = copy.deepcopy(baseline)
    invalid_record_receipt["record_validation"]["valid"] = False
    row = _zipformer_c4_row(invalid_record_receipt)
    assert row["evidence_class"] == "UNKNOWN"
    assert "record validation is not valid" in row["blocker"]

    tampered_trace = copy.deepcopy(baseline)
    tampered_trace["_partial_traces"][0]["partials"][0]["text"] = "tampered"
    row = _zipformer_c4_row(tampered_trace)
    assert row["evidence_class"] == "UNKNOWN"
    assert row["trace_sha256_count"] == aggregate.PARTIAL_TRACE_N
    assert row["trace_sha256_verified_count"] == aggregate.PARTIAL_TRACE_N - 1
    assert "fail canonical verification" in row["blocker"]


def _portable_fixture() -> dict[str, object]:
    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    reference = "相同的参考文本"
    reference_sha = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    base: dict[str, object] = {
        "utterance_id": "fleurs:cmn_hans_cn:run-a",
        "language": "zh",
        "config": "cmn_hans_cn",
        "reference": reference,
        "reference_sha256": reference_sha,
        "duration_s": 1.0,
        "duration_seconds": 1.0,
        "reference_source": "raw_transcription",
        "raw_transcription": reference,
        "transcription": "相 同 的 参 考 文 本",
        "source_id": "17|path:/run-a/cache/audio.wav",
        "fleurs_id": "17",
        "source_path": "/run-a/cache/audio.wav",
        "source_parquet_repo_path": (
            "parquet-data/cmn_hans_cn/test-00000-of-00001.parquet"
        ),
        "raw_audio_sha256": digest("raw-audio"),
        "clean_pcm_sha256": digest("clean-pcm"),
        "num_samples": 16_000,
        "sample_rate": 16_000,
        "channels": 1,
        "pcm_dtype": "float32-le",
        "source_sample_rate": 48_000,
        "source_channels": 2,
    }
    clean = {
        **base,
        "condition": "clean",
        "audio_relpath": "audio/zh/run-a.clean.wav",
        "audio_pcm_sha256": base["clean_pcm_sha256"],
        "audio_file_sha256": digest("run-a-clean-container"),
    }
    noisy = {
        **base,
        "condition": "musan_10db",
        "audio_relpath": "audio/zh/run-a.musan_10db.wav",
        "audio_pcm_sha256": digest("mixed-pcm"),
        "audio_file_sha256": digest("run-a-noisy-container"),
        "noise_repo_path": "noise/free-sound/example.wav",
        "noise_raw_audio_sha256": digest("noise-raw"),
        "noise_pcm_sha256": digest("noise-pcm"),
        "noise_segment_pcm_sha256": digest("noise-segment"),
        "mix_entry_sha256": digest("run-a-mix-entry"),
        "mix_selector_sha256": digest("run-a-selector"),
        "noise_offset_samples": 73,
        "noise_wrapped": False,
        "target_snr_db": 10.0,
        "achieved_snr_db": 10.00000001,
    }
    rows = [clean, noisy]
    noise_bank_sha = digest("noise-bank")
    return {
        "sample_manifest": rows,
        "fixture_hashes_by_language": {
            "zh": aggregate.canonical_sha256(rows)
        },
        "noise_bank_sha256": noise_bank_sha,
        "fleurs": {
            "repo_id": "google/fleurs",
            "revision": aggregate.FLEURS_REVISION,
            "resolved_commit": aggregate.FLEURS_REVISION,
            "split": "test",
            "configs": {"zh": "cmn_hans_cn"},
            "test_row_counts": {
                "zh": aggregate.FLEURS_TEST_ROWS["zh"]
            },
        },
        "audio": {
            "sample_rate": 16_000,
            "channels": 1,
            "pcm_dtype": "float32-le",
        },
        "musan": {
            "repo_id": aggregate.MUSAN_REPO,
            "revision": aggregate.MUSAN_REVISION,
            "resolved_commit": aggregate.MUSAN_REVISION,
        },
        "mix_manifest": {
            "schema": aggregate.MIX_MANIFEST_SCHEMA,
            "seed": aggregate.BOOTSTRAP_SEED,
            "target_snr_db": 10.0,
            "noise_bank_sha256": noise_bank_sha,
            "entries": [],
        },
    }


def test_portable_fixture_identity_ignores_only_run_volatile_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "TARGET_N", 1)
    original = _portable_fixture()
    relocated = copy.deepcopy(original)
    rows = relocated["sample_manifest"]
    assert isinstance(rows, list)
    for index, row in enumerate(rows):
        assert isinstance(row, dict)
        row["utterance_id"] = "fleurs:cmn_hans_cn:run-b"
        row["source_id"] = "17|path:/run-b/cache/audio.wav"
        row["source_path"] = "/run-b/cache/audio.wav"
        row["audio_relpath"] = f"audio/zh/run-b-{index}.wav"
        row["audio_file_sha256"] = hashlib.sha256(
            f"run-b-container-{index}".encode("utf-8")
        ).hexdigest()
        if row["condition"] == "musan_10db":
            row["mix_entry_sha256"] = hashlib.sha256(
                b"run-b-mix-entry"
            ).hexdigest()
            row["mix_selector_sha256"] = hashlib.sha256(
                b"run-b-selector"
            ).hexdigest()
    rows.reverse()

    original_archive_hash = aggregate.canonical_sha256(
        original["sample_manifest"]
    )
    relocated_archive_hash = aggregate.canonical_sha256(rows)
    original_identity = aggregate.portable_fixture_identity_sha256(
        original, "zh"
    )
    relocated_identity = aggregate.portable_fixture_identity_sha256(
        relocated, "zh"
    )

    assert original_archive_hash != relocated_archive_hash
    assert aggregate.valid_sha256(original_identity)
    assert original_identity == relocated_identity


def test_cell_index_memoizes_fixture_identity_per_engine_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "TARGET_N", 1)
    fixture = _portable_fixture()
    engine_result = {
        "fixture": fixture,
        "scored_models": [
            {
                "model": "model-a",
                "cells": [
                    {
                        "model": "model-a",
                        "language": "zh",
                        "condition": condition,
                    }
                    for condition in aggregate.CONDITIONS
                ],
            },
            {
                "model": "model-b",
                "cells": [
                    {
                        "model": "model-b",
                        "language": "zh",
                        "condition": condition,
                    }
                    for condition in aggregate.CONDITIONS
                ],
            },
        ],
    }
    original = aggregate.portable_fixture_identity_sha256
    call_count = 0

    def counted(fixture_info, language):
        nonlocal call_count
        call_count += 1
        return original(fixture_info, language)

    monkeypatch.setattr(
        aggregate, "portable_fixture_identity_sha256", counted
    )
    indexed = aggregate.cell_index({"shared-engine": engine_result})
    cells = [cell for matches in indexed.values() for cell in matches]
    assert len(cells) == 4
    observed = {
        aggregate.fixture_identity_hash_for(cell, "zh") for cell in cells
    }
    observed.update(
        aggregate.fixture_identity_hash_for(cell, "zh") for cell in cells
    )

    assert len(observed) == 1
    assert aggregate.valid_sha256(next(iter(observed)))
    assert call_count == 1
    assert len({id(cell["_fixture_identity_cache"]) for cell in cells}) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "reference",
        "raw_audio",
        "clean_pcm",
        "mixed_pcm",
        "noise_segment",
        "noise_offset",
        "snr",
    ),
)
def test_portable_fixture_identity_changes_on_stable_evidence(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    monkeypatch.setattr(aggregate, "TARGET_N", 1)
    original = _portable_fixture()
    changed = copy.deepcopy(original)
    rows = changed["sample_manifest"]
    assert isinstance(rows, list)
    clean, noisy = rows
    assert isinstance(clean, dict) and isinstance(noisy, dict)
    replacement = hashlib.sha256(f"changed-{mutation}".encode()).hexdigest()
    if mutation == "reference":
        reference = "不同的参考文本"
        reference_sha = hashlib.sha256(reference.encode("utf-8")).hexdigest()
        for row in rows:
            row["reference"] = reference
            row["raw_transcription"] = reference
            row["reference_sha256"] = reference_sha
    elif mutation == "raw_audio":
        for row in rows:
            row["raw_audio_sha256"] = replacement
    elif mutation == "clean_pcm":
        for row in rows:
            row["clean_pcm_sha256"] = replacement
        clean["audio_pcm_sha256"] = replacement
    elif mutation == "mixed_pcm":
        noisy["audio_pcm_sha256"] = replacement
    elif mutation == "noise_segment":
        noisy["noise_segment_pcm_sha256"] = replacement
    elif mutation == "noise_offset":
        noisy["noise_offset_samples"] = 74
    elif mutation == "snr":
        noisy["target_snr_db"] = 9.0
        noisy["achieved_snr_db"] = 9.00000001
        mix_manifest = changed["mix_manifest"]
        assert isinstance(mix_manifest, dict)
        mix_manifest["target_snr_db"] = 9.0

    original_identity = aggregate.portable_fixture_identity_sha256(
        original, "zh"
    )
    changed_identity = aggregate.portable_fixture_identity_sha256(
        changed, "zh"
    )

    assert aggregate.valid_sha256(original_identity)
    assert aggregate.valid_sha256(changed_identity)
    assert original_identity != changed_identity


def test_comparability_uses_portable_identity_and_reports_archive_hashes() -> None:
    portable_hash = "a" * 64
    mix_hash = "b" * 64
    rows: list[dict[str, object]] = []
    for index, bank in enumerate(("D-fork", "L-fork", "OmniASR"), start=1):
        archive_hash = str(index) * 64
        rows.append(
            {
                "bank": bank,
                "language": "zh",
                "corpus_revision": aggregate.FLEURS_REVISION,
                "corpus_config": aggregate.FLEURS_CONFIGS["zh"],
                "split": "test",
                "n": aggregate.TARGET_N,
                "normaliser_id": "norm_zh/shared",
                "fixture_sha256": archive_hash,
                "fixture_identity_sha256": portable_hash,
                "clean_evidence_class": "HOST",
                "ranking_inadmissible": False,
                "noisy_bucket": {
                    "n": aggregate.TARGET_N,
                    "normaliser_id": "norm_zh/shared",
                    "fixture_sha256": archive_hash,
                    "fixture_identity_sha256": portable_hash,
                    "mix_manifest_language_sha256": mix_hash,
                    "evidence_class": "HOST",
                },
            }
        )

    check = next(
        item
        for item in aggregate.comparability(rows)["checks"]
        if item["language"] == "zh"
    )
    assert check["clean"]["comparable_across_all_three_banks"] is True
    assert check["musan_10db"]["comparable_across_all_three_banks"] is True
    assert check["clean"]["fixture_identity_hashes"] == [portable_hash]
    assert len(check["clean"]["archive_fixture_hashes"]) == 3

    rows[0]["fixture_identity_sha256"] = "c" * 64
    rows[0]["noisy_bucket"]["fixture_identity_sha256"] = "c" * 64
    changed_check = next(
        item
        for item in aggregate.comparability(rows)["checks"]
        if item["language"] == "zh"
    )
    assert changed_check["clean"]["comparable_across_all_three_banks"] is False
    assert (
        changed_check["musan_10db"]["comparable_across_all_three_banks"]
        is False
    )
