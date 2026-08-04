"""Offline tests for Lane C's shared Colab utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_common as common  # noqa: E402
import colab_remote_runner as remote_runner  # noqa: E402


def fake_lane_b_normalisers() -> SimpleNamespace:
    def score_corpus(pairs, language, *, normalizer):
        utterances = []
        for reference, hypothesis in pairs:
            reference_tokens = reference.split()
            hypothesis_tokens = hypothesis.split()
            if hypothesis_tokens == reference_tokens:
                substitutions = insertions = deletions = 0
            elif not hypothesis_tokens:
                substitutions = insertions = 0
                deletions = len(reference_tokens)
            elif len(hypothesis_tokens) == len(reference_tokens):
                substitutions = sum(
                    left != right
                    for left, right in zip(reference_tokens, hypothesis_tokens)
                )
                insertions = deletions = 0
            else:
                substitutions = 0
                insertions = max(0, len(hypothesis_tokens) - len(reference_tokens))
                deletions = max(0, len(reference_tokens) - len(hypothesis_tokens))
            errors = substitutions + insertions + deletions
            utterances.append(
                {
                    "language": language,
                    "metric": "CER" if language == "zh" else "WER",
                    "normalizer_id": f"{normalizer}-id",
                    "reference_raw": reference,
                    "hypothesis_raw": hypothesis,
                    "reference_normalized": reference.lower(),
                    "hypothesis_normalized": hypothesis.lower(),
                    "reference_tokens": reference_tokens,
                    "hypothesis_tokens": hypothesis_tokens,
                    "substitutions": substitutions,
                    "insertions": insertions,
                    "deletions": deletions,
                    "reference_token_count": len(reference_tokens),
                    "errors": errors,
                    "error_rate": errors / len(reference_tokens),
                }
            )
        return {
            "language": language,
            "metric": "CER" if language == "zh" else "WER",
            "normalizer_id": f"{normalizer}-id",
            "utterances": utterances,
        }

    def normalize(text, language):
        return text.lower(), ["test"]

    def tokenize_normalized(text, language):
        if language == "zh":
            return [character for character in text if not character.isspace()]
        return text.split()

    return SimpleNamespace(
        compare_corpus=lambda pairs, language: {},
        score_corpus=score_corpus,
        normalize=normalize,
        tokenize_normalized=tokenize_normalized,
        normalizer_id=lambda language: f"project-{language}-id",
        PACKAGE_SHA256="f" * 64,
    )


def validation_fixture_sample(
    index: int,
    condition: str,
    *,
    language: str = "en",
) -> dict[str, object]:
    return {
        "utterance_id": f"u-{index}",
        "language": language,
        "condition": condition,
        "audio_path": f"/fixture/{language}-{index}-{condition}.wav",
        "reference": f"reference {index}",
        "duration_s": 1.0,
    }


def validation_record(
    sample: dict[str, object],
    *,
    engine: str = "fixture",
    status: str = "ok",
) -> dict[str, object]:
    return {
        **sample,
        "engine": engine,
        "model": "example/model",
        "bank": "test-bank",
        "revision": "deadbeef",
        "hypothesis": "hypothesis",
        "status": status,
    }


def validation_engine_spec(*, omni: bool = False) -> dict[str, object]:
    spec: dict[str, object] = {
        "models": {
            "example": {
                "repo_id": "example/model",
                "bank": "test-bank",
                "revision": "deadbeef",
                "languages": ["en"],
            }
        }
    }
    if omni:
        spec["inference"] = {"smoke_clean_n_per_language": 2}
    return spec


def test_protocol_pins_are_exact() -> None:
    assert common.FLEURS_REPO_ID == "google/fleurs"
    assert (
        common.FLEURS_REVISION
        == "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    )
    assert common.FLEURS_CONFIGS == {
        "en": "en_us",
        "zh": "cmn_hans_cn",
        "es": "es_419",
        "fr": "fr_fr",
        "ru": "ru_ru",
    }
    assert common.FLEURS_SPLIT == "test"
    assert common.FIXTURE_UTTERANCES_PER_LANGUAGE == 150
    assert common.MAX_AUDIO_DURATION_SECONDS == 25.0
    assert common.MUSAN_REPO_ID == "alexwengg/musan_mini"
    assert common.MUSAN_REVISION == "54ce99dd44add93cec39c4d37bdcc59c7870a849"
    assert common.MUSAN_NOISE_FILE_COUNT == 738
    assert "noise/free-sound/LICENSE" in common.MUSAN_PROVENANCE_FILES
    assert common.MIX_SEED == 20260801
    assert common.MIX_SNR_DB == 10.0
    assert common.BOOTSTRAP_B == 10_000


def test_selection_is_duration_filtered_then_numeric_id_sorted() -> None:
    rows = [
        {"id": "10", "num_samples": 16_000},
        {"id": "2", "num_samples": 24_000},
        {"id": "1", "num_samples": 25 * 16_000},
        {"id": "3", "num_samples": 25 * 16_000 + 1},
        {"id": "11", "num_samples": 32_000},
    ]
    selected = common.select_fleurs_rows(rows, n=4)
    assert [row["id"] for row in selected] == ["1", "2", "10", "11"]
    assert common.row_duration_seconds(selected[0]) == 25.0


def test_selection_is_independent_of_input_order() -> None:
    rows = [
        {"id": identifier, "duration_seconds": duration}
        for identifier, duration in (
            ("7", 4.0),
            ("1", 2.0),
            ("9", 26.0),
            ("3", 1.0),
        )
    ]
    forward = common.select_fleurs_rows(rows, n=3)
    reverse = common.select_fleurs_rows(list(reversed(rows)), n=3)
    assert [row["id"] for row in forward] == ["1", "3", "7"]
    assert [row["id"] for row in reverse] == ["1", "3", "7"]


def test_repeated_sentence_ids_use_stable_utterance_discriminators() -> None:
    rows = [
        {"id": "7", "path": "speaker-b.wav", "duration_seconds": 1.0},
        {"id": "7", "path": "speaker-a.wav", "duration_seconds": 1.0},
        {"id": "8", "path": "speaker-c.wav", "duration_seconds": 1.0},
    ]
    selected = common.select_fleurs_rows(rows, n=3)
    assert [(row["id"], row["path"]) for row in selected] == [
        ("7", "speaker-a.wav"),
        ("7", "speaker-b.wav"),
        ("8", "speaker-c.wav"),
    ]


def test_parquet_selection_avoids_optional_pandas_bridge() -> None:
    class FakeTable:
        def __init__(self, rows):
            self.rows = [dict(row) for row in rows]

        def slice(self, offset, length):
            return FakeTable(self.rows[offset : offset + length])

        def take(self, _indices):
            raise AssertionError("Arrow take would probe the pandas bridge")

        def to_pylist(self):
            return [dict(row) for row in self.rows]

    class FakeParquetFile:
        def __init__(self, rows):
            self.rows = rows
            self.schema_arrow = SimpleNamespace(
                names=["id", "path", "duration_seconds", "payload"]
            )

        def read(self, columns=None):
            if columns is None:
                return FakeTable(self.rows)
            return FakeTable(
                [{key: row[key] for key in columns} for row in self.rows]
            )

    rows_by_path = {
        Path("part-0.parquet"): [
            {
                "id": str(index),
                "path": f"speaker-{index}.wav",
                "duration_seconds": 1.0,
                "payload": f"row-{index}",
            }
            for index in range(100)
        ],
        Path("part-1.parquet"): [
            {
                "id": str(index),
                "path": f"speaker-{index}.wav",
                "duration_seconds": 1.0,
                "payload": f"row-{index}",
            }
            for index in range(100, 200)
        ],
    }
    parquet = SimpleNamespace(
        ParquetFile=lambda path: FakeParquetFile(rows_by_path[path])
    )

    selected, observed = common._selected_parquet_rows(
        [
            ("data/part-0.parquet", Path("part-0.parquet")),
            ("data/part-1.parquet", Path("part-1.parquet")),
        ],
        parquet,
    )

    assert observed == 200
    assert [int(row["id"]) for row in selected] == list(range(150))
    assert selected[-1]["payload"] == "row-149"
    assert selected[-1]["_source_parquet_repo_path"] == "data/part-1.parquet"


def test_post_install_worker_uses_fresh_python_process(tmp_path: Path) -> None:
    result_path = tmp_path / "result.zip"
    probe_path = tmp_path / "worker_probe.txt"
    worker_path = tmp_path / "worker.py"
    worker_path.write_text(
        """\
import os
from pathlib import Path

Path(os.environ["LANE_C_TEST_PROBE"]).write_text(
    f"{os.getpid()}|{os.environ.get('LANE_C_POST_INSTALL_WORKER', '')}",
    encoding="utf-8",
)
Path(os.environ["LANE_C_TEST_RESULT"]).write_bytes(b"worker-result")
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["LANE_C_TEST_PROBE"] = str(probe_path)
    environment["LANE_C_TEST_RESULT"] = str(result_path)

    returncode = remote_runner.launch_post_install_worker(
        result_path,
        runner_path=worker_path,
        environment=environment,
    )

    worker_pid, marker = probe_path.read_text(encoding="utf-8").split("|")
    assert returncode == 0
    assert int(worker_pid) != os.getpid()
    assert marker == "1"
    assert result_path.read_bytes() == b"worker-result"


def test_record_validation_rejects_non_ok_inference() -> None:
    samples = [
        validation_fixture_sample(0, "clean"),
        validation_fixture_sample(0, "musan_10db"),
    ]
    records = [validation_record(sample) for sample in samples]
    records[1]["status"] = "error"

    with pytest.raises(ValueError, match="non-ok inference status"):
        remote_runner.validate_records(
            records,
            engine="fixture",
            engine_spec=validation_engine_spec(),
            fixture_samples=samples,
            engine_metadata={},
            expected_per_condition=1,
        )


def test_record_validation_requires_exact_cell_counts() -> None:
    samples = [
        validation_fixture_sample(0, "clean"),
        validation_fixture_sample(0, "musan_10db"),
    ]

    with pytest.raises(ValueError, match="record count mismatch"):
        remote_runner.validate_records(
            [validation_record(samples[0])],
            engine="fixture",
            engine_spec=validation_engine_spec(),
            fixture_samples=samples,
            engine_metadata={},
            expected_per_condition=1,
        )


def test_record_validation_accepts_explicit_omni_smoke_collapse() -> None:
    clean = [validation_fixture_sample(index, "clean") for index in range(3)]
    noisy = [
        validation_fixture_sample(index, "musan_10db") for index in range(3)
    ]
    samples = clean + noisy
    records = [validation_record(sample, engine="omni") for sample in clean[:2]]

    result = remote_runner.validate_records(
        records,
        engine="omni",
        engine_spec=validation_engine_spec(omni=True),
        fixture_samples=samples,
        engine_metadata={
            "smoke_gate": {
                "en": {
                    "collapsed": True,
                    "smoke_n_expected": 2,
                    "smoke_n_observed": 2,
                    "remaining_samples_skipped": 4,
                }
            }
        },
        expected_per_condition=3,
    )

    assert result["valid"] is True
    assert result["expected_record_count"] == 2
    assert result["omni_collapsed_languages"] == ["en"]
    assert result["cell_counts"] == [
        {
            "model": "example/model",
            "bank": "test-bank",
            "revision": "deadbeef",
            "language": "en",
            "condition": "clean",
            "expected_records": 2,
            "records": 2,
        }
    ]


def test_noise_assignment_is_hash_derived_and_path_order_independent() -> None:
    paths = [
        "noise/free-sound/z.wav",
        "noise/sound-bible/a.wav",
        "noise/free-sound/m.wav",
    ]
    first = common.derive_noise_assignment(
        "fleurs:en_us:42",
        paths,
        clean_length_samples=100,
        noise_length_samples=1000,
    )
    second = common.derive_noise_assignment(
        "fleurs:en_us:42",
        list(reversed(paths)),
        clean_length_samples=100,
        noise_length_samples=1000,
    )
    assert first == second
    assert first["noise_repo_path"] in paths
    assert 0 <= first["offset_samples"] <= 900
    assert first["selector_sha256"] == common.sha256_bytes(
        b"lane-c-musan-v1\x0020260801\x00fleurs:en_us:42"
    )
    changed = common.derive_noise_assignment(
        "fleurs:en_us:43",
        paths,
        clean_length_samples=100,
        noise_length_samples=1000,
    )
    assert changed["selector_sha256"] != first["selector_sha256"]


def test_short_noise_wrap_is_deterministic() -> None:
    noise = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    segment = common.extract_noise_segment(noise, 8, 1)
    assert segment.tolist() == [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_mix_math_reaches_ten_db_and_prevents_clipping() -> None:
    time = np.arange(16_000, dtype=np.float64) / 16_000.0
    clean = (0.95 * np.sin(2.0 * math.pi * 440.0 * time)).astype(np.float32)
    noise = np.sign(np.sin(2.0 * math.pi * 137.0 * time)).astype(np.float32)
    mixed, receipt = common.mix_at_snr(clean, noise)
    assert mixed.dtype == np.float32
    assert mixed.shape == clean.shape
    assert receipt["achieved_snr_db"] == pytest.approx(10.0, abs=2e-6)
    assert receipt["mixed_peak"] <= common.MIX_PEAK_LIMIT + 1e-6
    assert receipt["postmix_gain"] < 1.0
    assert common.pcm_sha256(mixed) == common.pcm_sha256(mixed.copy())


def test_canonical_json_and_atomic_checkpoint_are_stable(tmp_path: Path) -> None:
    left = {"z": 1, "text": "中文", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "text": "中文", "z": 1}
    assert common.canonical_json_bytes(left) == common.canonical_json_bytes(
        right
    )
    assert common.canonical_sha256(left) == common.canonical_sha256(right)
    path = tmp_path / "checkpoint.json"
    common.atomic_checkpoint_write(path, left)
    assert json.loads(path.read_text(encoding="utf-8")) == left
    assert not list(tmp_path.glob(".*.tmp"))


def test_score_records_retains_sid_basic_control_and_silent_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    records = [
        {
            "utterance_id": "u1",
            "language": "en",
            "condition": "clean",
            "reference": "alpha beta",
            "hypothesis": "alpha delta",
        },
        {
            "utterance_id": "u2",
            "language": "en",
            "condition": "clean",
            "reference": "hello",
            "hypothesis": "",
        },
    ]
    result = common.score_records(
        records,
        model="fixture-model",
        bank="fixture-bank",
        revision="deadbeef",
        fixture_info={"fixture_hashes_by_language": {"en": "f" * 64}},
    )
    assert result["bootstrap_B"] == 10_000
    assert len(result["cells"]) == 1
    cell = result["cells"][0]
    assert cell["n"] == 2
    assert cell["bootstrap_B"] == 10_000
    assert cell["normalizer_id"]
    assert cell["basic_normalizer_id"]
    assert cell["utterances"][0]["S"] == 1
    assert cell["utterances"][0]["I"] == 0
    assert cell["utterances"][0]["D"] == 0
    assert cell["utterances"][1]["D"] == 1
    assert (
        cell["utterances"][1]["silent_failure_flags"]["empty_output"] is True
    )


def test_lane_b_pair_list_api_is_canonicalized_and_bootstrapped() -> None:
    lane_b = fake_lane_b_normalisers()
    result = common._score_cell_with_normalisers(
        lane_b,
        ["alpha beta", "hello"],
        ["alpha delta", "hello"],
        "en",
    )
    assert result["normaliser_api"].startswith("lane-b")
    assert result["pinned"]["bootstrap_B"] == 10_000
    assert result["pinned"]["rate"] == pytest.approx(1 / 3)
    assert result["pinned"]["utterances"][0]["S"] == 1
    assert result["pinned"]["utterances"][0]["reference_length"] == 2
    assert result["pinned"]["normalizer_id"] == "project-id"
    assert result["basic"]["normalizer_id"] == "basic-id"


def test_partial_trace_metrics_use_audio_delta_and_preserve_no_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    traces = [
        {
            "utterance_id": "u1",
            "language": "en",
            "t2_s": 1.0,
            "partials": [
                {
                    "t3p_s": 1.2,
                    "text": "hello",
                    "audio_consumed_ms": 100,
                },
                {
                    "t3p_s": 1.4,
                    "text": "hello world",
                    "audio_consumed_ms": 600,
                },
            ],
            "final_hypothesis": "hello world",
        },
        {
            "utterance_id": "u2",
            "language": "en",
            "partials": [],
            "structural_reason": "utterance-level decoder",
            "final_hypothesis": "not used",
        },
    ]
    result = common.partial_trace_metrics(traces, "en")
    assert result["n_input"] == 2
    assert result["n_partials_emitted"] == 1
    assert result["n_no_partials"] == 1
    assert result["first_partial_latency"]["p50_s"] == pytest.approx(0.2)
    assert result["partial_rewrite_rate"]["value"] == pytest.approx(5.0 / 3.0)
    assert result["final_partial_divergence"]["value"] == 0.0
    assert result["final_partial_differs_fraction"] == 0.0
    assert result["utterances"][1]["partials_emitted"] is False
    assert result["utterances"][0]["trace"][0]["text"] == "hello"


def test_flat_zipformer_callback_rows_are_grouped_by_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    result = common.partial_trace_metrics(
        [
            {
                "utterance_id": "zh-1",
                "language": "zh",
                "condition": "clean",
                "engine": "zipformer",
                "stream_time_s": 0.1,
                "audio_time_s": 0.1,
                "wall_elapsed_s": 0.04,
                "text": "",
                "is_final": False,
            },
            {
                "utterance_id": "zh-1",
                "language": "zh",
                "condition": "clean",
                "engine": "zipformer",
                "stream_time_s": 0.2,
                "audio_time_s": 0.2,
                "wall_elapsed_s": 0.06,
                "text": "你",
                "is_final": False,
            },
            {
                "utterance_id": "zh-1",
                "language": "zh",
                "condition": "clean",
                "engine": "zipformer",
                "stream_time_s": 0.7,
                "audio_time_s": 0.5,
                "wall_elapsed_s": 0.09,
                "text": "你好",
                "is_final": False,
            },
            {
                "utterance_id": "zh-1",
                "language": "zh",
                "condition": "clean",
                "engine": "zipformer",
                "stream_time_s": 0.8,
                "audio_time_s": 0.5,
                "wall_elapsed_s": 0.1,
                "text": "你好",
                "is_final": True,
            },
        ],
        "zh",
    )
    assert result["n_input"] == 1
    assert result["n_trace_events_input"] == 4
    assert result["n_partials_emitted"] == 1
    assert result["first_partial_latency_ms"] == pytest.approx(60.0)
    assert (
        result["partial_rewrite_rate_edits_per_audio_second"]
        == pytest.approx(2.0)
    )
    assert result["final_vs_last_partial_divergence"] == 0.0
    assert len(result["utterances"][0]["source_events"]) == 4


def test_nested_zipformer_trace_filters_empty_callbacks_and_tail_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    result = common.partial_trace_metrics(
        [
            {
                "utterance_id": "zh-nested",
                "language": "zh",
                "condition": "clean",
                "first_partial_latency_s": 0.06,
                "final_hypothesis": "你好吗",
                "partials": [
                    {
                        "stream_time_s": 0.1,
                        "audio_time_s": 0.1,
                        "wall_elapsed_s": 0.04,
                        "text": "",
                        "phase": "audio",
                    },
                    {
                        "stream_time_s": 0.2,
                        "audio_time_s": 0.2,
                        "wall_elapsed_s": 0.06,
                        "text": "你",
                        "phase": "audio",
                    },
                    {
                        "stream_time_s": 0.8,
                        "audio_time_s": 0.5,
                        "wall_elapsed_s": 0.09,
                        "text": "你好",
                        "phase": "tail_silence",
                    },
                    {
                        "stream_time_s": 0.9,
                        "audio_time_s": 0.5,
                        "wall_elapsed_s": 0.10,
                        "text": "你好吗",
                        "phase": "tail_silence",
                    },
                ],
            }
        ],
        "zh",
    )

    utterance = result["utterances"][0]
    assert result["n_partials_emitted"] == 1
    assert [item["text"] for item in utterance["trace"]] == [
        "你",
        "你好",
        "你好吗",
    ]
    assert utterance["trace"][0]["index"] == 1
    assert utterance["rewrite_edits"] == 2
    assert utterance["rewrite_audio_seconds"] == pytest.approx(0.5)
    assert utterance["source_audio_exposure_s"] == pytest.approx(0.5)
    assert utterance["positive_partial_transition_audio_seconds"] == pytest.approx(
        0.3
    )
    assert utterance["partial_rewrite_rate"] == pytest.approx(4.0)
    assert utterance["partial_rewrite_rate_status"] == "defined"
    assert utterance["zero_source_audio_rewrite_transition_count"] == 1
    assert utterance["zero_source_audio_rewrite_edits"] == 1
    transition_audio = [
        transition["audio_seconds"] for transition in utterance["transitions"]
    ]
    assert transition_audio == pytest.approx([0.3, 0.0])
    assert result["partial_rewrite_rate_edits_per_audio_second"] == pytest.approx(
        4.0
    )
    assert result["partial_rewrite_rate_status"] == "defined"
    assert "synthetic tail silence excluded" in result[
        "partial_rewrite_rate_denominator_semantics"
    ]
    assert utterance["last_partial"] == "你好吗"
    assert utterance["final_partial_divergence"] == 0.0


def test_nested_zipformer_trace_with_only_empty_callbacks_has_no_partials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    traces = [
        {
            "utterance_id": f"zh-empty-{index:02d}",
            "language": "zh",
            "condition": "clean",
            "final_hypothesis": "最终结果",
            "partials": [
                {
                    "stream_time_s": 0.1,
                    "audio_time_s": 0.1,
                    "wall_elapsed_s": 0.03,
                    "text": "",
                },
                {
                    "stream_time_s": 0.8,
                    "audio_time_s": 0.5,
                    "wall_elapsed_s": 0.08,
                    "text": "   ",
                    "phase": "tail_silence",
                },
            ],
        }
        for index in range(30)
    ]
    result = common.partial_trace_metrics(traces, "zh")

    assert result["n_partials_emitted"] == 0
    assert result["n_valid"] == 30
    assert result["n_no_partials"] == 30
    assert result["n_invalid"] == 0
    assert result["partials_emitted"] is False
    assert result["execution_observed"] is True
    assert result["evidence_class"] == "HOST"
    assert result["utterances"][0]["partials_emitted"] is False
    assert result["utterances"][0]["source_audio_exposure_s"] == pytest.approx(
        0.5
    )
    assert result["utterances"][0]["trace"] == []


def test_zero_delta_rewrite_is_counted_against_source_audio_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    result = common.partial_trace_metrics(
        [
            {
                "utterance_id": "defined",
                "language": "en",
                "t2_s": 0.0,
                "partials": [
                    {"t3p_s": 0.1, "text": "a", "audio_consumed_ms": 100},
                    {
                        "t3p_s": 0.2,
                        "text": "a b",
                        "audio_consumed_ms": 600,
                    },
                ],
                "final_hypothesis": "a b",
            },
            {
                "utterance_id": "undefined",
                "language": "en",
                "t2_s": 0.0,
                "partials": [
                    {"t3p_s": 0.1, "text": "x", "audio_consumed_ms": 400},
                    {
                        "t3p_s": 0.2,
                        "text": "x y",
                        "audio_consumed_ms": 400,
                    },
                ],
                "final_hypothesis": "x y",
            },
        ],
        "en",
    )

    summary = result["partial_rewrite_rate"]
    assert summary["value"] == pytest.approx(2.0)
    assert summary["numerator"] == pytest.approx(2.0)
    assert summary["denominator"] == pytest.approx(1.0)
    assert result["partial_rewrite_rate_edits_per_audio_second"] == pytest.approx(
        2.0
    )
    assert result["zero_source_audio_rewrite_utterance_count"] == 1
    assert result["zero_source_audio_rewrite_transition_count"] == 1
    undefined_row = next(
        row for row in result["utterances"] if row["utterance_id"] == "undefined"
    )
    assert undefined_row["partial_rewrite_rate"] == pytest.approx(2.5)
    assert undefined_row["rewrite_audio_seconds"] == pytest.approx(0.4)
    assert undefined_row["transitions"][0]["audio_seconds"] == 0.0
    assert undefined_row["transitions"][0]["zero_source_audio_rewrite"] is True


def test_rewrite_with_zero_total_source_exposure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    result = common.partial_trace_metrics(
        [
            {
                "utterance_id": "zero-exposure",
                "language": "en",
                "t2_s": 0.0,
                "partials": [
                    {"t3p_s": 0.1, "text": "a", "audio_consumed_ms": 0},
                    {"t3p_s": 0.2, "text": "a b", "audio_consumed_ms": 0},
                ],
                "final_hypothesis": "a b",
            }
        ],
        "en",
    )

    summary = result["partial_rewrite_rate"]
    assert summary["value"] is None
    assert summary["status"] == "undefined_zero_source_audio_exposure"
    assert summary["zero_source_audio_exposure_utterance_count"] == 1
    assert summary["zero_source_audio_exposure_rewrite_edits"] == 1
    assert summary["affected_utterance_ids"] == ["zero-exposure"]
    assert result["utterances"][0]["partial_rewrite_rate"] is None


def test_nonmonotonic_partial_trace_is_rejected_not_silently_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "_load_sibling_normalisers",
        fake_lane_b_normalisers,
    )
    result = common.partial_trace_metrics(
        [
            {
                "utterance_id": "bad",
                "language": "en",
                "t2_s": 0.0,
                "partials": [
                    {
                        "t3p_s": 0.2,
                        "text": "a",
                        "audio_consumed_ms": 500,
                    },
                    {
                        "t3p_s": 0.3,
                        "text": "b",
                        "audio_consumed_ms": 400,
                    },
                ],
                "final_hypothesis": "b",
            }
        ],
        "en",
    )
    assert result["n_invalid"] == 1
    assert result["n_partials_emitted"] == 0
    assert result["evidence_class"] == "UNKNOWN"
    assert "not monotonic" in result["invalid_traces"][0]["error"]
