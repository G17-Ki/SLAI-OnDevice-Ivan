"""Focused offline tests for Lane C engine scheduling and metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_engines as engines  # noqa: E402


def test_omni_install_bypasses_only_malformed_python_minor_cap() -> None:
    commands = engines.ENGINE_SPECS["omni"]["install_commands"]
    package_command = next(
        command for command in commands if "omnilingual-asr==0.2.0" in command
    )

    assert "--ignore-requires-python" in package_command
    assert "fairseq2==0.6" in package_command
    assert "huggingface_hub==0.32.6" in package_command
    assert all(
        "--ignore-requires-python" not in command
        for command in commands
        if command is not package_command
    )


def test_gigaam_pins_minimal_pyannote_namespace_dependency() -> None:
    commands = engines.ENGINE_SPECS["gigaam"]["install_commands"]
    packages = [value for command in commands for value in command]

    assert "pyannote.core==5.0.0" in packages
    assert not any(value.startswith("pyannote.audio") for value in packages)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        assert duration >= 0.0
        self.sleep_calls.append(duration)
        self.now += duration


class FakeOnlineRecognizer:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.accept_times: list[float] = []
        self.decode_count = 0

    def create_stream(self) -> "FakeOnlineStream":
        return FakeOnlineStream(self)

    def is_ready(self, stream: "FakeOnlineStream") -> bool:
        return stream.ready

    def decode_stream(self, stream: "FakeOnlineStream") -> None:
        stream.ready = False
        self.decode_count += 1
        self.clock.now += 0.005

    def get_result(self, stream: "FakeOnlineStream") -> str:
        return "你好" if self.decode_count else ""


class FakeFinalOnlyRecognizer(FakeOnlineRecognizer):
    def get_result(self, stream: "FakeOnlineStream") -> str:
        return "最终结果" if stream.finished else ""


class FakeOnlineStream:
    def __init__(self, recognizer: FakeOnlineRecognizer) -> None:
        self.recognizer = recognizer
        self.ready = False
        self.finished = False

    def accept_waveform(self, sample_rate: int, chunk: list[float]) -> None:
        assert sample_rate == 16_000
        assert chunk
        self.recognizer.accept_times.append(self.recognizer.clock.now)
        self.ready = True

    def input_finished(self) -> None:
        self.finished = True


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def zeros(size: int, dtype: Any) -> list[float]:
        assert dtype == FakeNumpy.float32
        return [0.0] * size


def zipformer_sample() -> dict[str, Any]:
    return {
        "utterance_id": "zh-clean-0",
        "language": "zh",
        "condition": "clean",
        "audio_path": "/fixture/zh-clean-0.wav",
        "reference": "你好",
        "duration_s": 0.2,
        "_input_index": 0,
    }


def test_retained_zipformer_trace_is_fed_on_real_time_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    recognizer = FakeOnlineRecognizer(clock)
    monkeypatch.setattr(
        engines,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep),
    )

    hypothesis, traces, first_partial_s = engines._decode_zipformer_stream(
        recognizer=recognizer,
        waveform=[0.0] * 3_200,
        sample=zipformer_sample(),
        retain_trace=True,
        np=FakeNumpy,
        started=clock.perf_counter(),
    )

    assert hypothesis == "你好"
    assert first_partial_s == pytest.approx(0.105)
    assert first_partial_s >= 0.1
    assert recognizer.accept_times == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.86]
    )
    assert len(clock.sleep_calls) == len(recognizer.accept_times)

    trace = traces[0]
    assert trace["feed_pacing"] == {
        "applied": True,
        "mode": "real_time_cumulative_deadlines",
        "nominal_chunk_s": 0.1,
        "scope": "audio plus synthetic tail silence",
        "sleep_before_accept": True,
        "deadline_basis": "cumulative samples / 16000 Hz",
        "decoder_compute_consumes_deadline_budget": True,
    }
    assert "real-time audio arrival" in trace["first_partial_latency_semantics"]
    assert trace["partials"][0]["feed_deadline_s"] == pytest.approx(0.1)
    assert trace["partials"][0]["wall_elapsed_s"] == pytest.approx(0.105)
    assert trace["partials"][0]["feed_pacing_applied"] is True
    first_tail = next(
        event for event in trace["partials"] if event["phase"] == "tail_silence"
    )
    assert first_tail["stream_time_s"] == pytest.approx(0.3)
    assert first_tail["source_audio_consumed_s"] == pytest.approx(0.2)
    assert first_tail["audio_consumed_s"] == pytest.approx(0.2)


def test_non_retained_zipformer_evaluation_stays_accelerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    recognizer = FakeOnlineRecognizer(clock)
    monkeypatch.setattr(
        engines,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep),
    )

    hypothesis, traces, first_partial_s = engines._decode_zipformer_stream(
        recognizer=recognizer,
        waveform=[0.0] * 3_200,
        sample=zipformer_sample(),
        retain_trace=False,
        np=FakeNumpy,
        started=clock.perf_counter(),
    )

    assert hypothesis == "你好"
    assert traces == []
    assert first_partial_s == pytest.approx(0.005)
    assert clock.sleep_calls == []
    assert recognizer.accept_times[0] == pytest.approx(0.0)


def test_zipformer_final_result_is_not_mislabeled_as_a_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    recognizer = FakeFinalOnlyRecognizer(clock)
    monkeypatch.setattr(
        engines,
        "time",
        SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep),
    )

    hypothesis, traces, first_partial_s = engines._decode_zipformer_stream(
        recognizer=recognizer,
        waveform=[0.0] * 3_200,
        sample=zipformer_sample(),
        retain_trace=True,
        np=FakeNumpy,
        started=clock.perf_counter(),
    )

    assert hypothesis == "最终结果"
    assert first_partial_s is None
    assert traces[0]["first_partial_latency_s"] is None
    assert not any(event["text"] for event in traces[0]["partials"])


def omni_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    input_index = 0
    for language in engines._LANGUAGES:
        for clean_index in range(31):
            samples.append(
                {
                    "utterance_id": f"{language}-clean-{clean_index:02d}",
                    "language": language,
                    "condition": "clean",
                    "audio_path": f"/fixture/{language}-clean-{clean_index:02d}.wav",
                    "reference": "reference",
                    "duration_s": 1.0,
                    "_input_index": input_index,
                    "clean_index": clean_index,
                }
            )
            input_index += 1
        for noisy_index in range(2):
            samples.append(
                {
                    "utterance_id": f"{language}-noise-{noisy_index:02d}",
                    "language": language,
                    "condition": "musan_10db",
                    "audio_path": f"/fixture/{language}-noise-{noisy_index:02d}.wav",
                    "reference": "reference",
                    "duration_s": 1.0,
                    "_input_index": input_index,
                }
            )
            input_index += 1
    return samples


def omni_infer_recorder(
    calls: list[list[dict[str, Any]]], *, collapsed_language: str = "zh"
):
    transcripts = {
        "en": "hello",
        "zh": "你好",
        "es": "hola",
        "fr": "bonjour",
        "ru": "привет",
    }

    def infer(batch: list[dict[str, Any]]) -> list[str]:
        calls.append([dict(sample) for sample in batch])
        return [
            "" if sample["language"] == collapsed_language else transcripts[sample["language"]]
            for sample in batch
        ]

    return infer


def run_omni_schedule(
    tmp_path: Path,
    samples: list[dict[str, Any]],
    calls: list[list[dict[str, Any]]],
    *,
    records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    existing = list(records or [])
    completed = engines._successful_record_identities(existing)
    result_records, _, gates = engines._run_omni_two_phase_samples(
        selected=samples,
        records=existing,
        completed=completed,
        model_id=engines.ENGINE_SPECS["omni"]["models"]["omni"]["repo_id"],
        infer_batch=omni_infer_recorder(calls),
        run_id="test-omni-two-phase",
        progress_path=tmp_path / "omni-progress.jsonl",
        batch_index=0,
        smoke_n=30,
    )
    return result_records, gates


def test_omni_smokes_every_language_before_expansion_and_skips_collapse(
    tmp_path: Path,
) -> None:
    samples = omni_samples()
    calls: list[list[dict[str, Any]]] = []

    records, gates = run_omni_schedule(tmp_path, samples, calls)

    first_expansion = next(
        index
        for index, batch in enumerate(calls)
        if any(
            sample["condition"] != "clean" or sample.get("clean_index", 0) >= 30
            for sample in batch
        )
    )
    smoke_seen = {language: set() for language in engines._LANGUAGES}
    for batch in calls[:first_expansion]:
        for sample in batch:
            smoke_seen[sample["language"]].add(sample["utterance_id"])
    assert {language: len(ids) for language, ids in smoke_seen.items()} == {
        language: 30 for language in engines._LANGUAGES
    }

    counts = {language: 0 for language in engines._LANGUAGES}
    for record in records:
        counts[record["language"]] += 1
    assert counts == {"en": 33, "zh": 30, "es": 33, "fr": 33, "ru": 33}
    assert not any(
        record["language"] == "zh" and record["phase"] == "evaluation"
        for record in records
    )
    assert gates["zh"]["collapsed"] is True
    assert gates["zh"]["remaining_samples_total"] == 3
    assert gates["zh"]["remaining_samples_evaluated"] == 0
    assert gates["zh"]["remaining_samples_skipped"] == 3
    for language in ("en", "es", "fr", "ru"):
        assert gates[language]["collapsed"] is False
        assert gates[language]["remaining_samples_evaluated"] == 3
        assert gates[language]["remaining_samples_skipped"] == 0


def test_omni_two_phase_schedule_resumes_without_duplicates(tmp_path: Path) -> None:
    samples = omni_samples()
    initial_calls: list[list[dict[str, Any]]] = []
    initial_records, _ = run_omni_schedule(tmp_path / "initial", samples, initial_calls)
    checkpointed = [
        record
        for record in initial_records
        if record["phase"] == "smoke" and int(record["utterance_id"].rsplit("-", 1)[1]) < 5
    ]

    resumed_calls: list[list[dict[str, Any]]] = []
    resumed_records, gates = run_omni_schedule(
        tmp_path / "resumed",
        samples,
        resumed_calls,
        records=checkpointed,
    )

    identities = [engines._record_identity_from_record(r) for r in resumed_records]
    assert len(identities) == len(set(identities))
    assert len(resumed_records) == 162
    assert gates["zh"]["smoke_n_observed"] == 30
    assert gates["zh"]["remaining_samples_skipped"] == 3

    first_expansion = next(
        index
        for index, batch in enumerate(resumed_calls)
        if any(
            sample["condition"] != "clean" or sample.get("clean_index", 0) >= 30
            for sample in batch
        )
    )
    pending_smoke_seen = {language: set() for language in engines._LANGUAGES}
    for batch in resumed_calls[:first_expansion]:
        for sample in batch:
            pending_smoke_seen[sample["language"]].add(sample["utterance_id"])
    assert {
        language: len(ids) for language, ids in pending_smoke_seen.items()
    } == {language: 25 for language in engines._LANGUAGES}


def test_error_checkpoint_is_retried_and_replaced_by_success(tmp_path: Path) -> None:
    sample = zipformer_sample()
    model_id = engines.ENGINE_SPECS["zipformer"]["models"]["zipformer"][
        "repo_id"
    ]
    progress_path = tmp_path / "retry-progress.jsonl"
    completed: set[tuple[str, str, str, str]] = set()

    def fail(_batch: list[dict[str, Any]]) -> list[str]:
        raise RuntimeError("transient failure")

    error_records, next_batch = engines._run_batched_samples(
        engine="zipformer",
        model_id=model_id,
        route="test-route",
        samples=[sample],
        batch_size=1,
        infer_batch=fail,
        completed=completed,
        run_id="retry-test",
        progress_path=progress_path,
        batch_index=0,
    )
    identity = engines._record_identity(model_id, sample)
    assert error_records[0]["status"] == "error"
    assert completed == set()
    assert engines._successful_record_identities(error_records) == set()

    success_records, _ = engines._run_batched_samples(
        engine="zipformer",
        model_id=model_id,
        route="test-route",
        samples=[sample],
        batch_size=1,
        infer_batch=lambda _batch: ["recovered"],
        completed=completed,
        run_id="retry-test",
        progress_path=progress_path,
        batch_index=next_batch,
    )
    assert success_records[0]["status"] == "ok"
    assert completed == {identity}

    merged = engines._merge_records(
        [*error_records, *success_records],
        error_records,
    )
    assert len(merged) == 1
    assert merged[0]["status"] == "ok"
    assert merged[0]["hypothesis"] == "recovered"

    resumed = engines._load_progress(progress_path, "retry-test")
    assert len(resumed["records"]) == 1
    assert resumed["records"][0]["status"] == "ok"
    assert engines._successful_record_identities(resumed["records"]) == {
        identity
    }


def test_run_id_binds_protocol_and_full_sample_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = {
        **zipformer_sample(),
        "audio_pcm_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "source_parquet_repo_path": "data/zh/test.parquet",
    }
    validated = engines._validate_samples([sample])
    current = engines._run_id("zipformer", validated)

    changed = [dict(validated[0], audio_pcm_sha256="c" * 64)]
    assert engines._run_id("zipformer", changed) != current

    monkeypatch.setattr(
        engines,
        "_RUN_PROTOCOL_VERSION",
        "lane-c-asr-inference-checkpoint-legacy",
    )
    assert engines._run_id("zipformer", validated) != current


def test_load_progress_ignores_old_protocol_even_with_matching_run_id(
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "old-protocol-progress.jsonl"
    run_id = "same-run-id"
    record = {
        "model": "example/model",
        "utterance_id": "u1",
        "condition": "clean",
        "audio_path": "/fixture/u1.wav",
        "status": "ok",
    }
    engines._append_progress(
        progress_path,
        {
            "schema_version": engines._SCHEMA_VERSION,
            "event": "batch",
            "run_id": run_id,
            "engine": "zipformer",
            "batch_index": 7,
            "records": [record],
            "partial_traces": [],
        },
    )

    ignored = engines._load_progress(progress_path, run_id)
    assert ignored["records"] == []
    assert ignored["next_batch_index"] == 0

    engines._checkpoint_batch(
        progress_path,
        run_id,
        "zipformer",
        0,
        [record],
        [],
    )
    resumed = engines._load_progress(progress_path, run_id)
    assert resumed["records"] == [record]
    assert resumed["next_batch_index"] == 1


def test_sensevoice_fixed_context_probe_invokes_exact_10s_without_scoring(
    tmp_path: Path,
) -> None:
    import numpy as np

    class FakeSoundFile:
        @staticmethod
        def read(
            path: str | Path, *, dtype: str, always_2d: bool
        ) -> tuple[Any, int]:
            assert dtype == "float32"
            assert always_2d is False
            return (
                np.frombuffer(Path(path).read_bytes(), dtype="<f4").copy(),
                16_000,
            )

        @staticmethod
        def write(
            path: str | Path,
            waveform: Any,
            sample_rate: int,
            *,
            subtype: str,
            format: str,
        ) -> None:
            assert sample_rate == 16_000
            assert subtype == "FLOAT"
            assert format == "WAV"
            Path(path).write_bytes(
                np.asarray(waveform, dtype="<f4").tobytes(order="C")
            )

        @staticmethod
        def info(path: str | Path) -> Any:
            return SimpleNamespace(
                frames=len(Path(path).read_bytes()) // 4,
                samplerate=16_000,
            )

    sf = FakeSoundFile()

    samples = []
    for input_index, (language, num_samples) in enumerate(
        (("en", 16_000), ("zh", 192_000))
    ):
        path = tmp_path / f"{language}.wav"
        waveform = np.linspace(-0.25, 0.25, num_samples, dtype=np.float32)
        sf.write(path, waveform, 16_000, subtype="FLOAT", format="WAV")
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        pcm_sha256 = hashlib.sha256(
            np.asarray(waveform, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        reference = f"{language} reference"
        samples.append(
            {
                "utterance_id": f"{language}-clean-0",
                "language": language,
                "condition": "clean",
                "audio_path": str(path),
                "audio_relpath": f"audio/{language}.wav",
                "audio_pcm_sha256": pcm_sha256,
                "clean_pcm_sha256": pcm_sha256,
                "audio_file_sha256": engines._sha256_file(path),
                "reference": reference,
                "reference_sha256": hashlib.sha256(
                    reference.encode("utf-8")
                ).hexdigest(),
                "num_samples": int(waveform.size),
                "sample_rate": int(sample_rate),
                "duration_s": waveform.size / sample_rate,
                "_input_index": input_index,
            }
        )

    class FakeSenseVoice:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            for path in kwargs["input"]:
                info = sf.info(path)
                assert info.frames == 160_000
                assert info.samplerate == 16_000
            self.calls.append(kwargs)
            return [
                {"text": f"{kwargs['language']}-final"}
                for _path in kwargs["input"]
            ]

    model = FakeSenseVoice()
    receipt = engines._run_sensevoice_fixed_context_probe(
        model=model,
        samples=samples,
        output_dir=tmp_path / "probe-work",
        postprocess=lambda text: text,
        np=np,
        sf=sf,
        probe_n_per_language=1,
    )

    assert receipt["valid"] is True
    assert receipt["scoring_included"] is False
    assert receipt["main_scoring_inputs_unchanged"] is True
    assert receipt["partials_emitted"] is False
    assert receipt["receipt_sha256"] == engines._canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert [call["language"] for call in model.calls] == ["en", "zh"]
    assert receipt["languages"]["en"]["samples"][0]["transformation"] == (
        "right_zero_pad"
    )
    assert receipt["languages"]["zh"]["samples"][0]["transformation"] == (
        "truncate_tail"
    )
    assert all(
        entry["derived_num_samples"] == 160_000
        and entry["model_invoked"] is True
        and entry["status"] == "ok"
        and entry["scoring_included"] is False
        for language in ("en", "zh")
        for entry in receipt["languages"][language]["samples"]
    )
    assert not list((tmp_path / "probe-work").rglob("*.wav"))
    assert [Path(sample["audio_path"]).stat().st_size for sample in samples]


def test_sensevoice_context_policy_separates_whole_utterance_scoring_from_probe() -> None:
    probe = {"valid": True, "receipt_sha256": "a" * 64}
    policy = engines._sensevoice_context_policy(
        [{"duration_s": 2.5}, {"duration_s": 11.75}, {"duration_s": 4.0}],
        probe,
    )

    assert policy["evaluation_input"] == "whole utterance"
    assert policy["main_scoring_inputs_unchanged"] is True
    assert policy["maximum_selected_fixture_duration_s"] == pytest.approx(11.75)
    assert policy["maximum_context_s"] == pytest.approx(11.75)
    assert policy["product_fixed_context_s"] == 10.0
    assert policy["product_fixed_context_separately_exercised"] is True
    assert policy["fixed_context_probe_scoring_included"] is False
    assert policy["fixed_context_probe_receipt_sha256"] == "a" * 64
    assert "C1 scoring keeps each fixture utterance" in policy["note"]
    assert engines._sensevoice_context_policy([], None)[
        "product_fixed_context_separately_exercised"
    ] is False
