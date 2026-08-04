#!/usr/bin/env python3
"""Execute one isolated Lane C ASR engine inside Google Colab.

The local orchestrator uploads this file plus ``normalisers.py``,
``colab_common.py`` and ``colab_engines.py``.  Raw audio and model weights never
enter the returned archive; only manifests, hashes, hypotheses, scoring
receipts, partial traces, and environment metadata are packaged.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence
import zipfile


CONFIG_PATH = Path("/content/lane_c_engine_config.json")
REMOTE_RUNNER_PATH = Path("/content/lane_c_remote_runner.py")
POST_INSTALL_WORKER_ENV = "LANE_C_POST_INSTALL_WORKER"
POST_INSTALL_STRATEGY = "fresh-python-subprocess-after-install"
EXPECTED_CONDITIONS = ("clean", "musan_10db")
EXPECTED_SAMPLES_PER_CONDITION = 150

_ARCHIVE_ALLOWLIST = frozenset(
    {
        "engine_metadata.json",
        "engine_result.json",
        "environment.json",
        "fixture_receipt.json",
        "install_receipts.json",
        "partial_metrics.json",
        "partial_traces.jsonl",
        "records.jsonl",
        "run_status.json",
        "scored_models.json",
        "traceback.txt",
    }
)
_REQUIRED_RECORD_FIELDS = (
    "engine",
    "model",
    "bank",
    "revision",
    "utterance_id",
    "language",
    "condition",
    "audio_path",
    "reference",
    "hypothesis",
    "duration_s",
    "status",
)
_STRICT_ENGINE_MODEL_REPOS = {
    "sensevoice": frozenset({"FunAudioLLM/SenseVoiceSmall"}),
    "nemo": frozenset(
        {"nvidia/parakeet-tdt-0.6b-v3", "nvidia/canary-180m-flash"}
    ),
    "zipformer": frozenset({"pfluo/k2fsa-zipformer-chinese-english-mixed"}),
    "gigaam": frozenset({"ai-sage/GigaAM-Multilingual"}),
    "omni": frozenset({"facebook/omniASR-CTC-300M"}),
}
_FIXTURE_BASE_MIRROR_FIELDS = (
    "reference",
    "reference_sha256",
    "duration_s",
    "duration_seconds",
    "config",
    "audio_relpath",
    "reference_source",
    "raw_transcription",
    "transcription",
    "source_id",
    "fleurs_id",
    "source_path",
    "source_parquet_repo_path",
    "audio_pcm_sha256",
    "clean_pcm_sha256",
    "raw_audio_sha256",
    "audio_file_sha256",
    "num_samples",
    "sample_rate",
    "channels",
    "pcm_dtype",
    "source_sample_rate",
    "source_channels",
)
_FIXTURE_NOISE_MIRROR_FIELDS = (
    "noise_repo_path",
    "noise_raw_audio_sha256",
    "noise_pcm_sha256",
    "noise_segment_pcm_sha256",
    "mix_entry_sha256",
    "mix_selector_sha256",
    "noise_offset_samples",
    "noise_wrapped",
    "target_snr_db",
    "achieved_snr_db",
)
_FIXTURE_HASH_FIELDS = frozenset(
    {
        "reference_sha256",
        "audio_pcm_sha256",
        "clean_pcm_sha256",
        "raw_audio_sha256",
        "audio_file_sha256",
        "noise_raw_audio_sha256",
        "noise_pcm_sha256",
        "noise_segment_pcm_sha256",
        "mix_entry_sha256",
        "mix_selector_sha256",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_record_fixture_binding(
    record: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    record_index: int,
    strict_schema: bool,
) -> None:
    """Require every provenance-bearing record field to mirror its fixture."""

    condition = str(sample["condition"])
    if strict_schema:
        missing_fixture_fields = [
            field for field in _FIXTURE_BASE_MIRROR_FIELDS if field not in sample
        ]
        if missing_fixture_fields:
            raise KeyError(
                f"fixture sample for record {record_index} lacks binding fields: "
                f"{missing_fixture_fields}"
            )
        active_fields = list(_FIXTURE_BASE_MIRROR_FIELDS)
    else:
        # Unit fixtures and third-party callers may be deliberately minimal;
        # every provenance field they do provide is still fail-closed.
        active_fields = ["reference", "duration_s"]
        active_fields.extend(
            field
            for field in _FIXTURE_BASE_MIRROR_FIELDS
            if field in sample and field not in active_fields
        )

    if condition == "musan_10db":
        missing_noise_fields = [
            field for field in _FIXTURE_NOISE_MIRROR_FIELDS if field not in sample
        ]
        if strict_schema and missing_noise_fields:
            raise KeyError(
                f"noisy fixture sample for record {record_index} lacks binding fields: "
                f"{missing_noise_fields}"
            )
        active_fields.extend(
            field
            for field in _FIXTURE_NOISE_MIRROR_FIELDS
            if strict_schema or field in sample
        )
    else:
        unexpected_noise_fields = [
            field
            for field in _FIXTURE_NOISE_MIRROR_FIELDS
            if field in sample or field in record
        ]
        if unexpected_noise_fields:
            raise ValueError(
                f"clean record {record_index} carries noise/mix fields: "
                f"{unexpected_noise_fields}"
            )

    for field in active_fields:
        if field not in record:
            raise KeyError(
                f"record {record_index} lacks mirrored fixture field {field!r}"
            )
        if record[field] != sample[field]:
            raise ValueError(
                f"record {record_index} fixture field {field!r} mismatch: "
                f"expected {sample[field]!r}, observed {record[field]!r}"
            )
        if field in _FIXTURE_HASH_FIELDS and not _valid_sha256(sample[field]):
            raise ValueError(
                f"fixture field {field!r} for record {record_index} is not SHA-256"
            )

    reference = sample["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"fixture reference for record {record_index} is not text")
    if "reference_sha256" in active_fields:
        computed_reference_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest()
        if sample["reference_sha256"] != computed_reference_hash:
            raise ValueError(
                f"fixture reference SHA mismatch for record {record_index}"
            )

    duration_s = record["duration_s"]
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise TypeError(f"record {record_index} duration_s is not numeric")
    if not math.isfinite(float(duration_s)) or float(duration_s) <= 0.0:
        raise ValueError(f"record {record_index} duration_s is invalid")
    if "duration_seconds" in active_fields and record["duration_seconds"] != duration_s:
        raise ValueError(
            f"record {record_index} duration_s/duration_seconds disagree"
        )
    if strict_schema:
        num_samples = record["num_samples"]
        sample_rate = record["sample_rate"]
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples <= 0
            or isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError(
                f"record {record_index} has invalid sample-count/rate metadata"
            )
        expected_duration = num_samples / sample_rate
        if not math.isclose(float(duration_s), expected_duration, abs_tol=1e-9):
            raise ValueError(
                f"record {record_index} duration does not match num_samples/sample_rate"
            )


def installed_distribution_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def launch_post_install_worker(
    output_zip: Path,
    *,
    runner_path: Path = REMOTE_RUNNER_PATH,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run fixture construction and inference in a clean Python process.

    Colab kernels commonly preload NumPy.  Replacing NumPy 2.x on disk with a
    pinned 1.26.x wheel cannot make already-loaded extension state coherent;
    imports such as ``numpy.random`` can then fail much later in scoring.  A
    fresh interpreter is the only reliable boundary after the pip installs.
    """

    worker_environment = dict(os.environ if environment is None else environment)
    worker_environment[POST_INSTALL_WORKER_ENV] = "1"
    command = [sys.executable, str(runner_path)]
    completed = subprocess.run(
        command,
        check=False,
        env=worker_environment,
    )
    if not output_zip.is_file():
        raise RuntimeError(
            "post-install worker exited without a result archive "
            f"(returncode={completed.returncode})"
        )
    return completed.returncode


def package_directory(source: Path, destination: Path) -> None:
    """Package only the compact, explicitly approved evidence files.

    This is an allowlist instead of a suffix filter: an accidental fixture,
    checkpoint, cache, or nested archive can never enter the returned zip.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    included: list[dict[str, object]] = []
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(_ARCHIVE_ALLOWLIST):
            path = source / name
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"refusing non-regular archive member: {path}")
            archive.write(path, name)
            included.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        archive.writestr(
            "archive_manifest.json",
            json.dumps(
                {
                    "schema": "lane-c-colab-archive/1.0",
                    "created_at": utc_now(),
                    "policy": "exact text-evidence allowlist; audio and model artifacts excluded",
                    "included": included,
                    "allowlist": sorted(_ARCHIVE_ALLOWLIST),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    temporary.replace(destination)


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    engine: str,
    engine_spec: Mapping[str, Any],
    fixture_samples: Sequence[Mapping[str, Any]],
    engine_metadata: Mapping[str, Any],
    expected_per_condition: int = EXPECTED_SAMPLES_PER_CONDITION,
) -> dict[str, Any]:
    """Fail closed unless every expected fixture inference succeeded exactly once."""

    if expected_per_condition <= 0:
        raise ValueError("expected_per_condition must be positive")

    models = engine_spec.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError(f"ENGINE_SPECS[{engine!r}] has no model provenance")
    expected: set[tuple[str, str, str]] = set()
    model_specs: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for model_key, model_spec in models.items():
        if not isinstance(model_spec, Mapping):
            raise TypeError(f"model spec {model_key!r} is not a mapping")
        try:
            provenance = (
                str(model_spec["repo_id"]),
                str(model_spec["bank"]),
                str(model_spec["revision"]),
            )
        except KeyError as exc:
            raise KeyError(
                f"model spec {model_key!r} lacks exact repo/bank/revision provenance"
            ) from exc
        if provenance in expected:
            raise ValueError(f"duplicate model provenance in engine spec: {provenance}")
        expected.add(provenance)
        model_specs[provenance] = model_spec

    fixture_by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fixture_identities: set[tuple[str, str, str]] = set()
    for index, sample in enumerate(fixture_samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"fixture sample {index} is not a mapping")
        missing = [
            field
            for field in ("utterance_id", "language", "condition", "audio_path")
            if field not in sample
        ]
        if missing:
            raise KeyError(f"fixture sample {index} lacks required fields: {missing}")
        fixture_identity = (
            str(sample["utterance_id"]),
            str(sample["condition"]),
            str(sample["audio_path"]),
        )
        if fixture_identity in fixture_identities:
            raise ValueError(f"duplicate fixture sample identity: {fixture_identity}")
        fixture_identities.add(fixture_identity)
        fixture_by_language[str(sample["language"])].append(sample)

    omni_gates: Mapping[str, Any] = {}
    smoke_n = 0
    if engine == "omni":
        gates = engine_metadata.get("smoke_gate")
        if not isinstance(gates, Mapping):
            raise ValueError("Omni metadata lacks smoke_gate receipts")
        omni_gates = gates
        inference = engine_spec.get("inference")
        if not isinstance(inference, Mapping):
            raise ValueError("Omni engine spec lacks inference settings")
        smoke_n = int(inference.get("smoke_clean_n_per_language", 0))
        if smoke_n <= 0:
            raise ValueError("Omni smoke sample count must be positive")

    strict_fixture_binding = (
        frozenset(provenance[0] for provenance in expected)
        == _STRICT_ENGINE_MODEL_REPOS.get(engine)
    )
    expected_identities: set[tuple[str, str, str, str, str, str, str]] = set()
    expected_samples_by_identity: dict[
        tuple[str, str, str, str, str, str, str], Mapping[str, Any]
    ] = {}
    expected_cell_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    collapsed_languages: list[str] = []
    for provenance, model_spec in model_specs.items():
        model_languages = model_spec.get("languages")
        if not isinstance(model_languages, Sequence) or isinstance(
            model_languages, (str, bytes)
        ):
            raise TypeError(f"model {provenance[0]!r} has no language sequence")
        for language_value in model_languages:
            language = str(language_value)
            language_samples = fixture_by_language.get(language, [])
            fixture_counts: dict[str, int] = defaultdict(int)
            for sample in language_samples:
                fixture_counts[str(sample["condition"])] += 1
            exact_fixture_counts = {
                condition: expected_per_condition
                for condition in EXPECTED_CONDITIONS
            }
            if dict(fixture_counts) != exact_fixture_counts:
                raise ValueError(
                    f"fixture count mismatch for {provenance[0]} {language}: "
                    f"expected {exact_fixture_counts}, observed {dict(fixture_counts)}"
                )

            expected_samples = language_samples
            if engine == "omni":
                gate = omni_gates.get(language)
                if not isinstance(gate, Mapping):
                    raise ValueError(f"Omni smoke gate is missing for {language}")
                collapsed = gate.get("collapsed")
                if type(collapsed) is not bool:
                    raise TypeError(f"Omni smoke gate collapsed flag is invalid for {language}")
                clean_samples = [
                    sample
                    for sample in language_samples
                    if str(sample["condition"]) == "clean"
                ]
                smoke_samples = clean_samples[:smoke_n]
                expected_skipped = (
                    len(language_samples) - len(smoke_samples) if collapsed else 0
                )
                expected_gate_values = {
                    "smoke_n_expected": len(smoke_samples),
                    "smoke_n_observed": len(smoke_samples),
                    "remaining_samples_skipped": expected_skipped,
                }
                observed_gate_values = {
                    key: gate.get(key) for key in expected_gate_values
                }
                if observed_gate_values != expected_gate_values:
                    raise ValueError(
                        f"Omni smoke gate count mismatch for {language}: expected "
                        f"{expected_gate_values}, observed {observed_gate_values}"
                    )
                if collapsed:
                    collapsed_languages.append(language)
                    expected_samples = smoke_samples

            for sample in expected_samples:
                identity = (
                    provenance[0],
                    provenance[1],
                    provenance[2],
                    str(sample["utterance_id"]),
                    language,
                    str(sample["condition"]),
                    str(sample["audio_path"]),
                )
                if identity in expected_identities:
                    raise ValueError(f"duplicate expected record identity: {identity}")
                expected_identities.add(identity)
                expected_samples_by_identity[identity] = sample
                expected_cell_counts[
                    (*provenance, language, str(sample["condition"]))
                ] += 1

    if not records:
        raise ValueError(f"{engine} returned no records for a non-empty fixture")
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    cell_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} is not a mapping")
        missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise KeyError(f"record {index} lacks required fields: {missing}")
        provenance = (
            str(record["model"]),
            str(record["bank"]),
            str(record["revision"]),
        )
        if provenance not in expected:
            raise ValueError(
                f"record {index} has unregistered model/bank/revision {provenance}; "
                f"expected one of {sorted(expected)}"
            )
        if str(record["engine"]) != engine:
            raise ValueError(
                f"record {index} engine {record['engine']!r} does not match {engine!r}"
            )
        if record["status"] != "ok":
            raise ValueError(
                f"record {index} has non-ok inference status {record['status']!r}"
            )
        identity = (
            *provenance,
            str(record["utterance_id"]),
            str(record["language"]),
            str(record["condition"]),
            str(record["audio_path"]),
        )
        if identity in seen:
            raise ValueError(f"duplicate engine record identity: {identity}")
        fixture_sample = expected_samples_by_identity.get(identity)
        if fixture_sample is None:
            raise ValueError(
                f"record {index} does not identify an expected fixture sample: {identity}"
            )
        _validate_record_fixture_binding(
            record,
            fixture_sample,
            record_index=index,
            strict_schema=strict_fixture_binding,
        )
        if record.get("evidence_class") != "HOST" and strict_fixture_binding:
            raise ValueError(
                f"record {index} does not assert HOST execution evidence"
            )
        seen.add(identity)
        counts[provenance] += 1
        cell_counts[
            (*provenance, str(record["language"]), str(record["condition"]))
        ] += 1

    if dict(cell_counts) != dict(expected_cell_counts):
        expected_rendered = {
            "|".join(key): value for key, value in sorted(expected_cell_counts.items())
        }
        observed_rendered = {
            "|".join(key): value for key, value in sorted(cell_counts.items())
        }
        raise ValueError(
            "record count mismatch by model/language/condition: "
            f"expected {expected_rendered}, observed {observed_rendered}"
        )
    if seen != expected_identities:
        missing_identities = sorted(expected_identities - seen)
        unexpected_identities = sorted(seen - expected_identities)
        raise ValueError(
            "record fixture identities do not match exactly: "
            f"missing={missing_identities[:3]}, unexpected={unexpected_identities[:3]}"
        )
    return {
        "schema": "lane-c-record-validation/1.1",
        "engine": engine,
        "valid": True,
        "record_count": len(records),
        "expected_record_count": len(expected_identities),
        "all_inference_status_ok": True,
        "all_records_fixture_bound": True,
        "fixture_binding": {
            "schema": "lane-c-record-fixture-binding/1.0",
            "valid": True,
            "strict_schema": strict_fixture_binding,
            "record_count": len(records),
            "base_fields": list(_FIXTURE_BASE_MIRROR_FIELDS),
            "noise_fields": list(_FIXTURE_NOISE_MIRROR_FIELDS),
        },
        "omni_collapsed_languages": sorted(set(collapsed_languages)),
        "expected_provenance": [
            {"model": model, "bank": bank, "revision": revision}
            for model, bank, revision in sorted(expected)
        ],
        "counts": [
            {
                "model": model,
                "bank": bank,
                "revision": revision,
                "records": count,
            }
            for (model, bank, revision), count in sorted(counts.items())
        ],
        "cell_counts": [
            {
                "model": model,
                "bank": bank,
                "revision": revision,
                "language": language,
                "condition": condition,
                "expected_records": expected_cell_counts[
                    (model, bank, revision, language, condition)
                ],
                "records": count,
            }
            for (model, bank, revision, language, condition), count in sorted(
                cell_counts.items()
            )
        ],
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    engine = str(config["engine"])
    remote_root = Path(config["remote_root"])
    output_zip = Path(config["output_zip"])
    result_dir = remote_root / "result"
    work_dir = remote_root / "work"
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    post_install_worker = os.environ.get(POST_INSTALL_WORKER_ENV) == "1"

    receipt: dict[str, object] = {
        "schema": "lane-c-colab-engine-result/1.0",
        "engine": engine,
        "started_at": utc_now(),
        "venue": "Google Colab",
        "evidence_class": "HOST",
        "training_performed": False,
        "status": "RUNNING",
    }
    if not post_install_worker:
        write_json(result_dir / "run_status.json", receipt)

    phase = "load_post_install_handoff" if post_install_worker else "import_modules"
    install_receipts: dict[str, object] = {
        "schema": "lane-c-colab-installs/1.0",
        "common": None,
        "engine": None,
    }
    fixture_receipt: dict[str, object] = {}
    engine_metadata: dict[str, object] = {}
    records: list[dict[str, object]] = []
    partial_traces: list[dict[str, object]] = []
    scored_models: list[dict[str, object]] = []
    partial_metrics: list[dict[str, object]] = []
    environment: dict[str, object] = {}
    record_validation: dict[str, object] | None = None

    try:
        if post_install_worker:
            loaded_receipt = json.loads(
                (result_dir / "run_status.json").read_text(encoding="utf-8")
            )
            loaded_installs = json.loads(
                (result_dir / "install_receipts.json").read_text(encoding="utf-8")
            )
            if not isinstance(loaded_receipt, dict) or not isinstance(
                loaded_installs, dict
            ):
                raise TypeError("post-install handoff receipts must be JSON objects")
            if loaded_receipt.get("engine") != engine:
                raise ValueError("post-install handoff engine does not match config")
            execution = loaded_installs.get("execution")
            if not isinstance(execution, dict) or execution.get(
                "strategy"
            ) != POST_INSTALL_STRATEGY:
                raise ValueError("post-install handoff lacks the fresh-process receipt")
            if loaded_installs.get("common") is None or loaded_installs.get(
                "engine"
            ) is None:
                raise ValueError("post-install handoff has incomplete dependency receipts")
            receipt = loaded_receipt
            install_receipts = loaded_installs
            phase = "import_modules"

        # Both modules are deliberately standard-library-only at import time.
        # Every pinned artifact is public.  Disable implicit token discovery so
        # Hugging Face does not query Colab's UI-only secret vault and stall a
        # headless run before falling back to anonymous access.
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        sys.path.insert(0, "/content")
        import colab_common
        import colab_engines

        if engine not in colab_engines.ENGINE_SPECS:
            raise ValueError(f"unknown engine: {engine}")

        if not post_install_worker:
            phase = "install_common_dependencies"
            install_receipts["common"] = colab_common.install_common_dependencies()
            write_json(result_dir / "install_receipts.json", install_receipts)
            phase = "install_engine_dependencies"
            install_receipts["engine"] = colab_engines.install_engine(engine)
            install_receipts["completed_at"] = utc_now()
            install_receipts["execution"] = {
                "strategy": POST_INSTALL_STRATEGY,
                "python_executable": sys.executable,
                "runner_path": str(REMOTE_RUNNER_PATH),
                "installed_distributions": installed_distribution_versions(
                    ("numpy", "pandas", "pyarrow")
                ),
                "reason": (
                    "avoid mixed in-memory/on-disk binary extension state after "
                    "pinned installs in a preloaded Colab kernel"
                ),
            }
            receipt["post_install_execution"] = POST_INSTALL_STRATEGY
            write_json(result_dir / "run_status.json", receipt)
            write_json(result_dir / "install_receipts.json", install_receipts)

            phase = "launch_post_install_worker"
            return launch_post_install_worker(output_zip)

        phase = "validate_post_install_runtime"
        numpy = importlib.import_module("numpy")
        importlib.import_module("numpy.random")
        pyarrow = importlib.import_module("pyarrow")
        execution = install_receipts["execution"]
        if not isinstance(execution, dict):
            raise TypeError("post-install execution receipt is not a mapping")
        installed_distributions = execution.get("installed_distributions")
        if not isinstance(installed_distributions, Mapping):
            raise TypeError("installed distribution versions are missing")
        loaded_versions = {
            "numpy": str(numpy.__version__),
            "pyarrow": str(pyarrow.__version__),
        }
        expected_loaded_versions = {
            name: installed_distributions.get(name) for name in loaded_versions
        }
        if loaded_versions != expected_loaded_versions:
            raise RuntimeError(
                "fresh worker loaded unexpected binary dependency versions: "
                f"expected {expected_loaded_versions}, observed {loaded_versions}"
            )
        execution["worker_validation"] = {
            "captured_at": utc_now(),
            "loaded_versions": loaded_versions,
            "numpy_random_import": "ok",
        }
        write_json(result_dir / "install_receipts.json", install_receipts)

        spec = colab_engines.ENGINE_SPECS[engine]
        languages = list(spec["languages"])
        phase = "build_fixture"
        fixture = colab_common.build_fixture(languages, work_dir / "fixture")
        fixture_samples = list(fixture["samples"])
        fixture_receipt = {key: value for key, value in fixture.items() if key != "samples"}
        write_json(result_dir / "fixture_receipt.json", fixture_receipt)

        phase = "run_engine"
        engine_output = colab_engines.run_engine(
            engine,
            fixture_samples,
            work_dir / "engine",
        )
        records = list(engine_output.get("records", []))
        partial_traces = list(engine_output.get("partial_traces", []))
        engine_metadata = dict(engine_output.get("metadata", {}))
        write_jsonl(result_dir / "records.jsonl", records)
        write_jsonl(result_dir / "partial_traces.jsonl", partial_traces)
        write_json(result_dir / "engine_metadata.json", engine_metadata)

        phase = "validate_records"
        record_validation = validate_records(
            records,
            engine=engine,
            engine_spec=spec,
            fixture_samples=fixture_samples,
            engine_metadata=engine_metadata,
        )

        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for record in records:
            grouped[
                (
                    str(record["model"]),
                    str(record["bank"]),
                    str(record["revision"]),
                )
            ].append(record)

        phase = "score_records"
        for (model, bank, revision), selected in sorted(grouped.items()):
            scored_models.append(
                colab_common.score_records(
                    selected,
                    model=model,
                    bank=bank,
                    revision=revision,
                    fixture_info=fixture_receipt,
                )
            )
        write_json(result_dir / "scored_models.json", scored_models)

        phase = "score_partials"
        partial_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for trace in partial_traces:
            partial_groups[(str(trace.get("route", engine)), str(trace["language"]))].append(trace)
        for (route, language), traces in sorted(partial_groups.items()):
            metric = colab_common.partial_trace_metrics(traces, language)
            metric["route"] = route
            partial_metrics.append(metric)
        write_json(result_dir / "partial_metrics.json", partial_metrics)

        phase = "capture_environment"
        environment = colab_common.environment_receipt()
        write_json(result_dir / "environment.json", environment)

        phase = "write_result"
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_at": utc_now(),
                "languages": languages,
                "records": len(records),
                "partial_traces": len(partial_traces),
                "models": [
                    {"model": model, "bank": bank, "revision": revision}
                    for model, bank, revision in sorted(grouped)
                ],
                "fixture_hashes_by_language": fixture_receipt.get(
                    "fixture_hashes_by_language", {}
                ),
                "mix_manifest_sha256": fixture_receipt.get("mix_manifest_sha256"),
                "record_validation": record_validation,
            }
        )
        write_json(result_dir / "run_status.json", receipt)
        write_json(
            result_dir / "engine_result.json",
            {
                **receipt,
                "fixture": fixture_receipt,
                "install_receipts": install_receipts,
                "engine_metadata": engine_metadata,
                "record_validation": record_validation,
                "scored_models": scored_models,
                "partial_metrics": partial_metrics,
                "environment": environment,
            },
        )
        package_directory(result_dir, output_zip)
        return 0
    except Exception as exc:
        receipt.update(
            {
                "status": "FAILED",
                "completed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "phase": phase,
            }
        )
        write_json(result_dir / "run_status.json", receipt)
        (result_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(result_dir / "install_receipts.json", install_receipts)
        write_json(
            result_dir / "engine_result.json",
            {
                **receipt,
                "fixture": fixture_receipt,
                "install_receipts": install_receipts,
                "engine_metadata": engine_metadata,
                "record_validation": record_validation,
                "records": len(records),
                "partial_traces": len(partial_traces),
                "scored_models": scored_models,
                "partial_metrics": partial_metrics,
                "environment": environment,
                "traceback_file": "traceback.txt",
            },
        )
        package_directory(result_dir, output_zip)
        raise
    finally:
        # The VM is stopped by the local orchestrator.  Explicitly avoid ever
        # copying its checkpoint cache or fixture audio into the result archive.
        if (result_dir / "raw_audio").exists():
            shutil.rmtree(result_dir / "raw_audio")


if __name__ == "__main__":
    raise SystemExit(main())
