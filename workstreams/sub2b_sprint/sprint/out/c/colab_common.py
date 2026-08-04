"""Deterministic shared data, scoring, and receipt utilities for Lane C.

This module is intentionally importable before Colab dependencies are installed:
all module-level imports are from the Python standard library.  NumPy, SciPy,
PyArrow, SoundFile, and huggingface_hub are imported only inside the functions
that need them.

The fixture is deliberately small and frozen:

* google/fleurs at one commit, test split, 150 rows per language;
* the first rows by numeric/string id after excluding audio over 25 seconds;
* raw_transcription is the scoring reference and transcription is retained;
* canonical audio is mono, 16 kHz, little-endian float32 PCM;
* a paired MUSAN 10 dB condition is derived from a per-utterance SHA-256 recipe.

The scoring path delegates text normalization and edit accounting to the sibling
normalisers.py module.  Corpus rates are ratios of sums and all bootstrap
resampling uses whole utterances.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import subprocess
import sys
import tempfile
from typing import Any
import unicodedata


SCHEMA_VERSION = "lane-c-colab-common/1.0"
FIXTURE_SCHEMA = "lane-c-asr-fixture/1.1"
MIX_MANIFEST_SCHEMA = "lane-c-musan-mix-manifest/1.0"
SCORE_SCHEMA = "lane-c-asr-score/1.0"
PARTIAL_SCHEMA = "lane-c-partial-metrics/1.0"

FLEURS_REPO_ID = "google/fleurs"
FLEURS_DATASET_ID = FLEURS_REPO_ID
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_REPO_TYPE = "dataset"
FLEURS_SPLIT = "test"
FLEURS_CONFIGS: dict[str, str] = {
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "es": "es_419",
    "fr": "fr_fr",
    "ru": "ru_ru",
}
FLEURS_TEST_ROW_COUNTS: dict[str, int] = {
    "en": 647,
    "zh": 945,
    "es": 908,
    "fr": 676,
    "ru": 775,
}
LANGUAGE_ORDER = ("en", "zh", "es", "fr", "ru")
FIXTURE_UTTERANCES_PER_LANGUAGE = 150
FLEURS_N = FIXTURE_UTTERANCES_PER_LANGUAGE
MAX_AUDIO_DURATION_SECONDS = 25.0
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_PCM_DTYPE = "float32-le"

MUSAN_REPO_ID = "alexwengg/musan_mini"
MUSAN_DATASET_ID = MUSAN_REPO_ID
MUSAN_REVISION = "54ce99dd44add93cec39c4d37bdcc59c7870a849"
MUSAN_REPO_TYPE = "dataset"
MUSAN_NOISE_GLOB = "noise/**/*.wav"
MUSAN_NOISE_FILE_COUNT = 738
MUSAN_PROVENANCE_FILES = (
    "README",
    "noise/README",
    "noise/free-sound/LICENSE",
    "noise/free-sound/ANNOTATIONS",
    "noise/sound-bible/LICENSE",
)
MIX_CONDITION = "musan_10db"
CLEAN_CONDITION = "clean"
MIX_SNR_DB = 10.0
MIX_SEED = 20_260_801
MIX_PEAK_LIMIT = 0.999

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20_260_801
NORMALISER_DELTA_THRESHOLD_PP = 2.0

COMMON_DEPENDENCIES = (
    "huggingface_hub==0.36.0",
    "numpy==1.26.4",
    "opencc-python-reimplemented==0.1.7",
    "pyarrow==18.1.0",
    "scipy==1.16.1",
    "soundfile==0.13.1",
    "whisper-normalizer==0.1.15",
)
RECEIPT_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "pyarrow",
    "scipy",
    "soundfile",
    "torch",
    "transformers",
    "datasets",
    "nemo-toolkit",
    "opencc-python-reimplemented",
    "whisper-normalizer",
)

_LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "en_us": "en",
    "en-us": "en",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "chinese": "zh",
    "cmn_hans_cn": "zh",
    "cmn-hans-cn": "zh",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "es_419": "es",
    "es-419": "es",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "fr_fr": "fr",
    "fr-fr": "fr",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "ru_ru": "ru",
    "ru-ru": "ru",
}


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON in the single canonical form used by every manifest hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_json_default,
    )


def _canonical_json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_dumps(value).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return sha256(bytes(data)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Durably replace path with data without exposing a partial checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    pretty: bool = True,
) -> Path:
    if pretty:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
                default=_canonical_json_default,
            )
            + "\n"
        ).encode("utf-8")
    else:
        encoded = canonical_json_bytes(payload) + b"\n"
    return atomic_write_bytes(path, encoded)


def atomic_checkpoint_write(
    path: str | os.PathLike[str],
    payload: Any,
) -> Path:
    """Alias with an explicit name for engine-loop checkpoint writes."""

    return atomic_write_json(path, payload, pretty=False)


def install_common_dependencies(
    packages: Sequence[str] = COMMON_DEPENDENCIES,
) -> dict[str, Any]:
    """Install the lightweight fixture/scoring dependency closure in Colab."""

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        *packages,
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "schema": f"{SCHEMA_VERSION}/dependency-install",
        "command": command,
        "returncode": completed.returncode,
        "packages": list(packages),
        "versions": _package_versions(RECEIPT_PACKAGES),
    }


def environment_receipt() -> dict[str, Any]:
    """Capture enough host state to make a Colab result auditable."""

    receipt: dict[str, Any] = {
        "schema": f"{SCHEMA_VERSION}/environment",
        "captured_at": utc_now(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "colab": {
            "detected": bool(
                os.environ.get("COLAB_RELEASE_TAG")
                or os.environ.get("COLAB_BACKEND_VERSION")
                or Path("/content").is_dir()
            ),
            "release_tag": os.environ.get("COLAB_RELEASE_TAG"),
            "backend_version": os.environ.get("COLAB_BACKEND_VERSION"),
        },
        "packages": _package_versions(RECEIPT_PACKAGES),
        "protocol": {
            "fleurs_repo_id": FLEURS_REPO_ID,
            "fleurs_revision": FLEURS_REVISION,
            "musan_repo_id": MUSAN_REPO_ID,
            "musan_revision": MUSAN_REVISION,
            "mix_seed": MIX_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_B": BOOTSTRAP_B,
        },
        "module_sha256": _module_sha256(),
    }
    nvidia = _run_receipt_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia is not None:
        receipt["nvidia_smi"] = nvidia
    git = _run_receipt_command(["git", "rev-parse", "HEAD"])
    if git is not None:
        receipt["git_head"] = git["stdout"].strip()
    return receipt


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:
            versions[name] = f"unavailable:{type(exc).__name__}"
    return versions


def _module_sha256() -> str:
    try:
        return sha256_file(__file__)
    except OSError:
        return "unavailable"


def _run_receipt_command(command: Sequence[str]) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def canonical_language(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    key = language.strip().lower()
    try:
        return _LANGUAGE_ALIASES[key]
    except KeyError as exc:
        key_with_underscore = key.replace("-", "_")
        try:
            return _LANGUAGE_ALIASES[key_with_underscore]
        except KeyError:
            raise ValueError(f"unsupported Lane C language: {language!r}") from exc


def _canonical_languages(languages: Iterable[str]) -> list[str]:
    selected = {canonical_language(language) for language in languages}
    if not selected:
        raise ValueError("at least one language is required")
    return [language for language in LANGUAGE_ORDER if language in selected]


def _id_sort_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, bool):
        return (1, str(value))
    if isinstance(value, int):
        return (0, value, str(value))
    if isinstance(value, float) and value.is_integer():
        return (0, int(value), str(value))
    text = str(value)
    if re.fullmatch(r"[+-]?\d+", text):
        return (0, int(text), text)
    return (1, text)


def _fleurs_row_discriminator(row: Mapping[str, Any]) -> str:
    """Return a stable within-language discriminator for repeated FLEURS ids."""

    source_path = row.get("path")
    if source_path is None and isinstance(row.get("audio"), Mapping):
        source_path = row["audio"].get("path")
    if source_path:
        return f"path:{source_path}"
    parquet_path = row.get("_source_parquet_repo_path")
    parquet_index = row.get("_source_parquet_row_index")
    if parquet_path is not None and parquet_index is not None:
        return f"parquet:{parquet_path}#{int(parquet_index)}"
    stable_scalars = {
        str(key): value
        for key, value in row.items()
        if key not in {"audio", "audio_bytes", "bytes"}
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    return f"metadata:{canonical_sha256(stable_scalars)}"


def _fleurs_row_identity(row: Mapping[str, Any]) -> str:
    return f"{row['id']}|{_fleurs_row_discriminator(row)}"


def row_duration_seconds(
    row: Mapping[str, Any],
    *,
    default_sample_rate: int = TARGET_SAMPLE_RATE,
) -> float:
    """Read duration without decoding when FLEURS metadata permits it."""

    for key in ("duration", "duration_s", "duration_seconds"):
        value = row.get(key)
        if value is not None:
            duration = float(value)
            if math.isfinite(duration) and duration >= 0.0:
                return duration
    samples = row.get("num_samples")
    if samples is None:
        samples = row.get("n_samples")
    if samples is not None:
        sample_rate = (
            row.get("sampling_rate")
            or row.get("sample_rate")
            or (
                row.get("audio", {}).get("sampling_rate")
                if isinstance(row.get("audio"), Mapping)
                else None
            )
            or default_sample_rate
        )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"invalid sample rate in row {row.get('id')!r}")
        return int(samples) / sample_rate
    raw = _audio_bytes_from_row(row)
    soundfile = importlib.import_module("soundfile")
    info = soundfile.info(io.BytesIO(raw))
    if info.samplerate <= 0:
        raise ValueError(f"invalid encoded sample rate in row {row.get('id')!r}")
    return info.frames / info.samplerate


def select_fleurs_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    n: int = FIXTURE_UTTERANCES_PER_LANGUAGE,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> list[Mapping[str, Any]]:
    """Select eligible utterances by id, then stable path/source-row order.

    FLEURS ``id`` is a sentence id and is intentionally repeated across
    speakers, so it cannot be used as the utterance primary key by itself.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if max_duration_seconds <= 0.0:
        raise ValueError("max_duration_seconds must be positive")
    eligible: list[Mapping[str, Any]] = []
    seen_rows: set[str] = set()
    for row in rows:
        if "id" not in row:
            raise KeyError("every FLEURS row must contain id")
        row_identity = _fleurs_row_identity(row)
        if row_identity in seen_rows:
            raise ValueError(f"duplicate FLEURS utterance identity: {row_identity}")
        seen_rows.add(row_identity)
        duration = row_duration_seconds(row)
        if duration <= max_duration_seconds:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            _id_sort_key(row["id"]),
            _fleurs_row_discriminator(row),
        )
    )
    if len(eligible) < n:
        raise ValueError(
            f"only {len(eligible)} rows satisfy duration <= "
            f"{max_duration_seconds:.1f}s; need {n}"
        )
    return eligible[:n]


def derive_noise_assignment(
    utterance_id: str,
    noise_paths: Sequence[str],
    *,
    clean_length_samples: int | None = None,
    noise_length_samples: int | None = None,
    seed: int = MIX_SEED,
) -> dict[str, Any]:
    """Derive the MUSAN file and offset from a stable per-utterance hash."""

    if not utterance_id:
        raise ValueError("utterance_id must be non-empty")
    paths = sorted({str(path) for path in noise_paths})
    if not paths:
        raise ValueError("at least one noise path is required")
    material = f"lane-c-musan-v1\0{seed}\0{utterance_id}".encode("utf-8")
    digest = sha256(material).digest()
    digest_hex = digest.hex()
    file_index = int.from_bytes(digest[:8], "big") % len(paths)
    selected_path = paths[file_index]
    assignment: dict[str, Any] = {
        "selector_sha256": digest_hex,
        "seed": seed,
        "noise_file_index": file_index,
        "noise_repo_path": selected_path,
    }
    if clean_length_samples is not None or noise_length_samples is not None:
        if clean_length_samples is None or noise_length_samples is None:
            raise ValueError("clean and noise lengths must be supplied together")
        if clean_length_samples <= 0 or noise_length_samples <= 0:
            raise ValueError("clean and noise lengths must be positive")
        offset_word = int.from_bytes(digest[8:16], "big")
        if noise_length_samples >= clean_length_samples:
            maximum_offset = noise_length_samples - clean_length_samples
            offset = offset_word % (maximum_offset + 1)
            wrapped = False
        else:
            offset = offset_word % noise_length_samples
            maximum_offset = noise_length_samples - 1
            wrapped = True
        assignment.update(
            {
                "offset_samples": offset,
                "maximum_offset_samples": maximum_offset,
                "noise_wrapped": wrapped,
            }
        )
    return assignment


def extract_noise_segment(
    noise: Any,
    length_samples: int,
    offset_samples: int,
) -> Any:
    """Take a deterministic segment, wrapping only when the source is short."""

    numpy = importlib.import_module("numpy")
    source = numpy.asarray(noise, dtype=numpy.float32).reshape(-1)
    if source.size == 0:
        raise ValueError("noise audio is empty")
    if length_samples <= 0:
        raise ValueError("length_samples must be positive")
    if offset_samples < 0:
        raise ValueError("offset_samples must be non-negative")
    if source.size >= length_samples:
        if offset_samples + length_samples > source.size:
            raise ValueError("non-wrapping noise offset exceeds source length")
        return numpy.ascontiguousarray(
            source[offset_samples : offset_samples + length_samples],
            dtype=numpy.float32,
        )
    indices = (
        numpy.arange(length_samples, dtype=numpy.int64) + int(offset_samples)
    ) % source.size
    return numpy.ascontiguousarray(source[indices], dtype=numpy.float32)


def pcm_sha256(audio: Any) -> str:
    """Hash canonical little-endian float32 mono PCM, excluding the WAV header."""

    numpy = importlib.import_module("numpy")
    pcm = numpy.asarray(audio, dtype="<f4").reshape(-1)
    return sha256_bytes(pcm.tobytes(order="C"))


def mix_at_snr(
    clean: Any,
    noise: Any,
    *,
    snr_db: float = MIX_SNR_DB,
    peak_limit: float = MIX_PEAK_LIMIT,
) -> tuple[Any, dict[str, Any]]:
    """Mix equal-length mono vectors at an exact power-ratio target."""

    numpy = importlib.import_module("numpy")
    clean64 = numpy.asarray(clean, dtype=numpy.float64).reshape(-1)
    noise64 = numpy.asarray(noise, dtype=numpy.float64).reshape(-1)
    if clean64.size == 0:
        raise ValueError("clean audio is empty")
    if clean64.size != noise64.size:
        raise ValueError("clean and noise vectors must have equal length")
    if not numpy.isfinite(clean64).all() or not numpy.isfinite(noise64).all():
        raise ValueError("mix inputs must be finite")
    if not math.isfinite(float(snr_db)):
        raise ValueError("snr_db must be finite")
    if not 0.0 < peak_limit <= 1.0:
        raise ValueError("peak_limit must be in (0, 1]")

    clean_power = float(numpy.mean(clean64 * clean64))
    noise_power = float(numpy.mean(noise64 * noise64))
    if clean_power <= 0.0:
        raise ValueError("clean audio has zero power")
    if noise_power <= 0.0:
        raise ValueError("noise audio has zero power")
    target_ratio = 10.0 ** (float(snr_db) / 10.0)
    noise_scale = math.sqrt(clean_power / (noise_power * target_ratio))
    scaled_noise64 = noise64 * noise_scale
    unbounded = clean64 + scaled_noise64
    maximum = float(numpy.max(numpy.abs(unbounded)))
    postmix_gain = 1.0 if maximum <= peak_limit else peak_limit / maximum

    clean_component = numpy.asarray(clean64 * postmix_gain, dtype=numpy.float32)
    noise_component = numpy.asarray(
        scaled_noise64 * postmix_gain, dtype=numpy.float32
    )
    mixed = numpy.asarray(
        clean_component.astype(numpy.float64)
        + noise_component.astype(numpy.float64),
        dtype=numpy.float32,
    )
    mixed = numpy.ascontiguousarray(mixed)
    achieved_clean_power = float(
        numpy.mean(clean_component.astype(numpy.float64) ** 2)
    )
    achieved_noise_power = float(
        numpy.mean(noise_component.astype(numpy.float64) ** 2)
    )
    achieved_snr_db = 10.0 * math.log10(
        achieved_clean_power / achieved_noise_power
    )
    return mixed, {
        "target_snr_db": float(snr_db),
        "achieved_snr_db": achieved_snr_db,
        "clean_power": clean_power,
        "noise_power": noise_power,
        "noise_scale": noise_scale,
        "postmix_gain": postmix_gain,
        "peak_limit": peak_limit,
        "mixed_peak": float(numpy.max(numpy.abs(mixed))),
    }


def build_fixture(
    languages: Iterable[str],
    root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build or resume the pinned clean and MUSAN-10dB FLEURS fixture."""

    selected_languages = _canonical_languages(languages)
    work_root = Path(root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest_path = work_root / "manifests" / "fixture_runtime.json"
    resumed = _resume_fixture(runtime_manifest_path, selected_languages, work_root)
    if resumed is not None:
        return resumed

    numpy = importlib.import_module("numpy")
    pyarrow_parquet = importlib.import_module("pyarrow.parquet")
    huggingface_hub = importlib.import_module("huggingface_hub")

    hf_cache = work_root / "hf_cache"
    audio_root = work_root / "audio"
    manifest_root = work_root / "manifests"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    api = huggingface_hub.HfApi()
    fleurs_info = _pinned_repo_info(
        api,
        FLEURS_REPO_ID,
        FLEURS_REVISION,
        FLEURS_REPO_TYPE,
    )
    fleurs_files = sorted(
        api.list_repo_files(
            repo_id=FLEURS_REPO_ID,
            revision=FLEURS_REVISION,
            repo_type=FLEURS_REPO_TYPE,
        )
    )

    clean_rows: list[dict[str, Any]] = []
    parquet_receipts: dict[str, list[dict[str, Any]]] = {}
    selected_ids_by_language: dict[str, list[str]] = {}
    for language in selected_languages:
        config = FLEURS_CONFIGS[language]
        repo_paths = _fleurs_parquet_paths(fleurs_files, config, FLEURS_SPLIT)
        if not repo_paths:
            raise RuntimeError(
                f"no exact {config}/{FLEURS_SPLIT} parquet found at "
                f"{FLEURS_REPO_ID}@{FLEURS_REVISION}"
            )
        local_parquets: list[tuple[str, Path]] = []
        parquet_receipts[language] = []
        for repo_path in repo_paths:
            local_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=FLEURS_REPO_ID,
                    filename=repo_path,
                    revision=FLEURS_REVISION,
                    repo_type=FLEURS_REPO_TYPE,
                    cache_dir=str(hf_cache),
                )
            )
            local_parquets.append((repo_path, local_path))
            parquet_receipts[language].append(
                {
                    "repo_path": repo_path,
                    "sha256": sha256_file(local_path),
                    "bytes": local_path.stat().st_size,
                }
            )
        selected_rows, observed_count = _selected_parquet_rows(
            local_parquets,
            pyarrow_parquet,
        )
        expected_count = FLEURS_TEST_ROW_COUNTS[language]
        if observed_count != expected_count:
            raise RuntimeError(
                f"FLEURS {config} test row-count mismatch: "
                f"expected {expected_count}, observed {observed_count}"
            )
        selected_rows = select_fleurs_rows(selected_rows)
        selected_ids_by_language[language] = [
            _fleurs_row_identity(row) for row in selected_rows
        ]
        for row in selected_rows:
            raw_audio = _audio_bytes_from_row(row)
            canonical_audio, source_sample_rate, source_channels = (
                _decode_canonical_audio(raw_audio)
            )
            duration_seconds = canonical_audio.size / TARGET_SAMPLE_RATE
            if duration_seconds > MAX_AUDIO_DURATION_SECONDS + 1e-9:
                raise RuntimeError(
                    f"decoded duration exceeds selection cap for {row['id']}: "
                    f"{duration_seconds:.6f}s"
                )
            raw_reference = row.get("raw_transcription")
            transcription = row.get("transcription")
            if not isinstance(raw_reference, str) or not raw_reference.strip():
                raise RuntimeError(
                    f"FLEURS row {row['id']} has no usable raw_transcription"
                )
            if not isinstance(transcription, str):
                transcription = "" if transcription is None else str(transcription)
            fleurs_id = str(row["id"])
            source_id = _fleurs_row_identity(row)
            utterance_key = sha256_bytes(source_id.encode("utf-8"))[:20]
            utterance_id = f"fleurs:{config}:{utterance_key}"
            file_token = sha256_bytes(utterance_id.encode("utf-8"))[:20]
            clean_relpath = (
                Path("audio") / language / f"{file_token}.{CLEAN_CONDITION}.wav"
            )
            clean_path = work_root / clean_relpath
            _atomic_write_wav(clean_path, canonical_audio)
            clean_rows.append(
                {
                    "utterance_id": utterance_id,
                    "language": language,
                    "config": config,
                    "condition": CLEAN_CONDITION,
                    "audio_path": str(clean_path),
                    "audio_relpath": clean_relpath.as_posix(),
                    "reference": raw_reference,
                    "reference_source": "raw_transcription",
                    "raw_transcription": raw_reference,
                    "transcription": transcription,
                    "source_id": source_id,
                    "fleurs_id": fleurs_id,
                    "source_path": row.get("path")
                    or (
                        row.get("audio", {}).get("path")
                        if isinstance(row.get("audio"), Mapping)
                        else None
                    ),
                    "source_parquet_repo_path": row["_source_parquet_repo_path"],
                    "raw_audio_sha256": sha256_bytes(raw_audio),
                    "audio_pcm_sha256": pcm_sha256(canonical_audio),
                    "clean_pcm_sha256": pcm_sha256(canonical_audio),
                    "audio_file_sha256": sha256_file(clean_path),
                    "reference_sha256": sha256_bytes(
                        raw_reference.encode("utf-8")
                    ),
                    "num_samples": int(canonical_audio.size),
                    "duration_seconds": duration_seconds,
                    "duration_s": duration_seconds,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "channels": TARGET_CHANNELS,
                    "pcm_dtype": TARGET_PCM_DTYPE,
                    "source_sample_rate": source_sample_rate,
                    "source_channels": source_channels,
                }
            )
            del canonical_audio

    musan_info = _pinned_repo_info(
        api,
        MUSAN_REPO_ID,
        MUSAN_REVISION,
        MUSAN_REPO_TYPE,
    )
    musan_files = sorted(
        api.list_repo_files(
            repo_id=MUSAN_REPO_ID,
            revision=MUSAN_REVISION,
            repo_type=MUSAN_REPO_TYPE,
        )
    )
    missing_provenance = sorted(set(MUSAN_PROVENANCE_FILES) - set(musan_files))
    if missing_provenance:
        raise RuntimeError(
            "pinned MUSAN subset is missing provenance files: "
            + ", ".join(missing_provenance)
        )
    noise_repo_paths = _musan_noise_paths(musan_files)
    if not noise_repo_paths:
        raise RuntimeError(
            f"no {MUSAN_NOISE_GLOB} files found at "
            f"{MUSAN_REPO_ID}@{MUSAN_REVISION}"
        )
    if len(noise_repo_paths) != MUSAN_NOISE_FILE_COUNT:
        raise RuntimeError(
            "pinned MUSAN subset noise-file count mismatch: "
            f"expected {MUSAN_NOISE_FILE_COUNT}, observed {len(noise_repo_paths)}"
        )
    provenance_receipts = []
    for repo_path in MUSAN_PROVENANCE_FILES:
        local_path = Path(
            huggingface_hub.hf_hub_download(
                repo_id=MUSAN_REPO_ID,
                filename=repo_path,
                revision=MUSAN_REVISION,
                repo_type=MUSAN_REPO_TYPE,
                cache_dir=str(hf_cache),
            )
        )
        provenance_receipts.append(
            {
                "repo_path": repo_path,
                "sha256": sha256_file(local_path),
                "bytes": local_path.stat().st_size,
            }
        )
    noise_bank_manifest = {
        "source_corpus": "MUSAN",
        "subset_mirror": True,
        "repo_id": MUSAN_REPO_ID,
        "revision": MUSAN_REVISION,
        "resolved_commit": musan_info["resolved_commit"],
        "glob": MUSAN_NOISE_GLOB,
        "expected_noise_file_count": MUSAN_NOISE_FILE_COUNT,
        "provenance_files": provenance_receipts,
        "files": noise_repo_paths,
    }
    noise_bank_sha256 = canonical_sha256(noise_bank_manifest)

    rows_by_noise: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clean_row in clean_rows:
        assignment = derive_noise_assignment(
            clean_row["utterance_id"],
            noise_repo_paths,
        )
        clean_row["_noise_selector"] = assignment
        rows_by_noise[assignment["noise_repo_path"]].append(clean_row)

    noisy_rows: list[dict[str, Any]] = []
    mix_entries: list[dict[str, Any]] = []
    for noise_repo_path in sorted(rows_by_noise):
        noise_local_path = Path(
            huggingface_hub.hf_hub_download(
                repo_id=MUSAN_REPO_ID,
                filename=noise_repo_path,
                revision=MUSAN_REVISION,
                repo_type=MUSAN_REPO_TYPE,
                cache_dir=str(hf_cache),
            )
        )
        noise_raw = noise_local_path.read_bytes()
        canonical_noise, noise_source_rate, noise_source_channels = (
            _decode_canonical_audio(noise_raw)
        )
        noise_pcm_hash = pcm_sha256(canonical_noise)
        noise_raw_hash = sha256_bytes(noise_raw)
        for clean_row in rows_by_noise[noise_repo_path]:
            clean_audio = _read_canonical_wav(clean_row["audio_path"])
            assignment = derive_noise_assignment(
                clean_row["utterance_id"],
                noise_repo_paths,
                clean_length_samples=int(clean_audio.size),
                noise_length_samples=int(canonical_noise.size),
            )
            segment = extract_noise_segment(
                canonical_noise,
                int(clean_audio.size),
                int(assignment["offset_samples"]),
            )
            mixed, mix_math = mix_at_snr(
                clean_audio,
                segment,
                snr_db=MIX_SNR_DB,
            )
            file_token = sha256_bytes(
                clean_row["utterance_id"].encode("utf-8")
            )[:20]
            noisy_relpath = (
                Path("audio") / clean_row["language"] / f"{file_token}.{MIX_CONDITION}.wav"
            )
            noisy_path = work_root / noisy_relpath
            _atomic_write_wav(noisy_path, mixed)
            entry = {
                "utterance_id": clean_row["utterance_id"],
                "language": clean_row["language"],
                "condition": MIX_CONDITION,
                "seed": MIX_SEED,
                "target_snr_db": MIX_SNR_DB,
                "selector_sha256": assignment["selector_sha256"],
                "noise_repo_path": noise_repo_path,
                "noise_file_index": assignment["noise_file_index"],
                "offset_samples": assignment["offset_samples"],
                "noise_wrapped": assignment["noise_wrapped"],
                "clean_num_samples": int(clean_audio.size),
                "noise_source_num_samples": int(canonical_noise.size),
                "clean_raw_audio_sha256": clean_row["raw_audio_sha256"],
                "clean_pcm_sha256": clean_row["clean_pcm_sha256"],
                "noise_raw_audio_sha256": noise_raw_hash,
                "noise_pcm_sha256": noise_pcm_hash,
                "noise_segment_pcm_sha256": pcm_sha256(segment),
                "mixed_pcm_sha256": pcm_sha256(mixed),
                **mix_math,
            }
            entry["entry_sha256"] = canonical_sha256(entry)
            mix_entries.append(entry)
            noisy_rows.append(
                {
                    **{
                        key: value
                        for key, value in clean_row.items()
                        if not key.startswith("_")
                    },
                    "condition": MIX_CONDITION,
                    "audio_path": str(noisy_path),
                    "audio_relpath": noisy_relpath.as_posix(),
                    "audio_pcm_sha256": entry["mixed_pcm_sha256"],
                    "audio_file_sha256": sha256_file(noisy_path),
                    "noise_repo_path": noise_repo_path,
                    "noise_raw_audio_sha256": noise_raw_hash,
                    "noise_pcm_sha256": noise_pcm_hash,
                    "noise_segment_pcm_sha256": entry[
                        "noise_segment_pcm_sha256"
                    ],
                    "mix_entry_sha256": entry["entry_sha256"],
                    "mix_selector_sha256": assignment["selector_sha256"],
                    "noise_offset_samples": assignment["offset_samples"],
                    "noise_wrapped": assignment["noise_wrapped"],
                    "target_snr_db": MIX_SNR_DB,
                    "achieved_snr_db": mix_math["achieved_snr_db"],
                }
            )
            del clean_audio, segment, mixed
        del canonical_noise

    condition_order = {CLEAN_CONDITION: 0, MIX_CONDITION: 1}
    samples = [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in clean_rows + noisy_rows
    ]
    samples.sort(
        key=lambda row: (
            LANGUAGE_ORDER.index(row["language"]),
            _id_sort_key(row["source_id"]),
            condition_order[row["condition"]],
        )
    )
    mix_entries.sort(
        key=lambda entry: (
            LANGUAGE_ORDER.index(entry["language"]),
            _id_sort_key(entry["utterance_id"].rsplit(":", 1)[-1]),
        )
    )
    mix_manifest = {
        "schema": MIX_MANIFEST_SCHEMA,
        "seed": MIX_SEED,
        "target_snr_db": MIX_SNR_DB,
        "noise_bank_sha256": noise_bank_sha256,
        "entries": mix_entries,
    }
    mix_manifest_sha256 = canonical_sha256(mix_manifest)
    atomic_write_json(
        manifest_root / "musan_10db_mix_manifest.json",
        {
            **mix_manifest,
            "mix_manifest_sha256": mix_manifest_sha256,
        },
    )

    canonical_sample_rows = [
        _canonical_sample_manifest_row(sample) for sample in samples
    ]
    fixture_hashes_by_language = {
        language: canonical_sha256(
            [
                row
                for row in canonical_sample_rows
                if row["language"] == language
            ]
        )
        for language in selected_languages
    }
    fixture_manifest = {
        "schema": FIXTURE_SCHEMA,
        "fleurs": {
            "repo_id": FLEURS_REPO_ID,
            "revision": FLEURS_REVISION,
            "resolved_commit": fleurs_info["resolved_commit"],
            "split": FLEURS_SPLIT,
            "configs": {
                language: FLEURS_CONFIGS[language]
                for language in selected_languages
            },
            "test_row_counts": {
                language: FLEURS_TEST_ROW_COUNTS[language]
                for language in selected_languages
            },
            "parquet_files": parquet_receipts,
            "selection": {
                "n_per_language": FIXTURE_UTTERANCES_PER_LANGUAGE,
                "sort": "sentence id ascending (numeric where integral), then source path/parquet row",
                "maximum_duration_seconds": MAX_AUDIO_DURATION_SECONDS,
                "reference": "raw_transcription",
                "retained_secondary_reference": "transcription",
            },
        },
        "audio": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
            "pcm_dtype": TARGET_PCM_DTYPE,
        },
        "musan": noise_bank_manifest,
        "noise_bank_sha256": noise_bank_sha256,
        "mix_manifest_sha256": mix_manifest_sha256,
        "samples": canonical_sample_rows,
    }
    fixture_manifest_sha256 = canonical_sha256(fixture_manifest)
    atomic_write_json(
        manifest_root / "fixture_manifest.json",
        {
            **fixture_manifest,
            "fixture_manifest_sha256": fixture_manifest_sha256,
        },
    )

    sample_index = {
        _sample_index_key(sample["utterance_id"], sample["condition"]): (
            _canonical_sample_manifest_row(sample)
        )
        for sample in samples
    }
    result: dict[str, Any] = {
        "schema": FIXTURE_SCHEMA,
        "created_at": utc_now(),
        "resumed": False,
        "remote_work_root": str(work_root),
        "languages": selected_languages,
        "conditions": [CLEAN_CONDITION, MIX_CONDITION],
        "samples_per_language_per_condition": FIXTURE_UTTERANCES_PER_LANGUAGE,
        "sample_count": len(samples),
        "samples": samples,
        "sample_manifest": canonical_sample_rows,
        "sample_index": sample_index,
        "selected_ids_by_language": selected_ids_by_language,
        "fixture_hashes_by_language": fixture_hashes_by_language,
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "fixture_sha256": fixture_manifest_sha256,
        "mix_manifest_sha256": mix_manifest_sha256,
        "mix_manifest": mix_manifest,
        "noise_bank_sha256": noise_bank_sha256,
        "fleurs": fixture_manifest["fleurs"],
        "musan": noise_bank_manifest,
        "audio": fixture_manifest["audio"],
        "environment": environment_receipt(),
    }
    atomic_write_json(runtime_manifest_path, result)
    return result


def _resume_fixture(
    runtime_manifest_path: Path,
    languages: Sequence[str],
    work_root: Path,
) -> dict[str, Any] | None:
    if not runtime_manifest_path.is_file():
        return None
    try:
        result = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("schema") != FIXTURE_SCHEMA:
        return None
    if result.get("languages") != list(languages):
        return None
    fleurs = result.get("fleurs", {})
    musan = result.get("musan", {})
    if fleurs.get("revision") != FLEURS_REVISION:
        return None
    if musan.get("revision") != MUSAN_REVISION:
        return None
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    for sample in samples:
        relpath = sample.get("audio_relpath")
        expected = sample.get("audio_file_sha256")
        if not relpath or not expected:
            return None
        audio_path = (work_root / relpath).resolve()
        try:
            audio_path.relative_to(work_root)
        except ValueError:
            return None
        if not audio_path.is_file() or sha256_file(audio_path) != expected:
            return None
        sample["audio_path"] = str(audio_path)
    result["remote_work_root"] = str(work_root)
    result["resumed"] = True
    return result


def _pinned_repo_info(
    api: Any,
    repo_id: str,
    revision: str,
    repo_type: str,
) -> dict[str, Any]:
    info = api.repo_info(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
    )
    resolved_commit = str(getattr(info, "sha", ""))
    if resolved_commit != revision:
        raise RuntimeError(
            f"{repo_id}@{revision} resolved to {resolved_commit or 'UNKNOWN'}"
        )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "resolved_commit": resolved_commit,
        "repo_type": repo_type,
    }


def _fleurs_parquet_paths(
    repo_files: Sequence[str],
    config: str,
    split: str,
) -> list[str]:
    selected: list[str] = []
    config_pattern = re.compile(
        rf"(^|[/_.-]){re.escape(config.lower())}($|[/_.-])"
    )
    split_pattern = re.compile(
        rf"(^|[/_.-]){re.escape(split.lower())}($|[/_.-])"
    )
    for repo_path in repo_files:
        lowered = repo_path.lower()
        if not lowered.endswith(".parquet"):
            continue
        if not config_pattern.search(lowered):
            continue
        if not split_pattern.search(lowered):
            continue
        selected.append(repo_path)
    return sorted(selected)


def _musan_noise_paths(repo_files: Sequence[str]) -> list[str]:
    selected = []
    for repo_path in repo_files:
        parts = PurePosixPath(repo_path).parts
        if (
            len(parts) >= 3
            and parts[0] == "noise"
            and parts[-1].lower().endswith(".wav")
        ):
            selected.append(repo_path)
    return sorted(selected)


def _selected_parquet_rows(
    local_parquets: Sequence[tuple[str, Path]],
    pyarrow_parquet: Any,
) -> tuple[list[dict[str, Any]], int]:
    metadata_rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for repo_path, local_path in local_parquets:
        parquet_file = pyarrow_parquet.ParquetFile(local_path)
        files[repo_path] = parquet_file
        names = set(parquet_file.schema_arrow.names)
        metadata_columns = [
            name
            for name in (
                "id",
                "path",
                "num_samples",
                "n_samples",
                "duration",
                "duration_s",
                "duration_seconds",
                "sampling_rate",
                "sample_rate",
            )
            if name in names
        ]
        if "id" not in metadata_columns:
            raise RuntimeError(f"FLEURS parquet lacks id: {repo_path}")
        if not any(
            name
            in metadata_columns
            for name in (
                "num_samples",
                "n_samples",
                "duration",
                "duration_s",
                "duration_seconds",
            )
        ):
            metadata_columns.append("audio")
        rows = parquet_file.read(columns=metadata_columns).to_pylist()
        for index, row in enumerate(rows):
            row["_source_parquet_repo_path"] = repo_path
            row["_source_parquet_row_index"] = index
            metadata_rows.append(row)
    observed_count = len(metadata_rows)
    selected_metadata = select_fleurs_rows(metadata_rows)
    selected_by_file: dict[str, list[int]] = defaultdict(list)
    for row in selected_metadata:
        selected_by_file[row["_source_parquet_repo_path"]].append(
            int(row["_source_parquet_row_index"])
        )
    full_rows: list[dict[str, Any]] = []
    for repo_path in sorted(selected_by_file):
        indices = selected_by_file[repo_path]
        table = files[repo_path].read()
        for source_index in indices:
            # ``pyarrow.array(indices)`` probes PyArrow's optional pandas
            # bridge for generic Python inputs.  In a long-lived Colab kernel,
            # a pinned NumPy install can leave the preloaded pandas extension
            # ABI-incompatible even though this path does not need pandas.
            # Integer slicing stays Arrow-native and returns the same row while
            # avoiding that optional import entirely.
            selected_rows = table.slice(source_index, 1).to_pylist()
            if len(selected_rows) != 1:
                raise IndexError(
                    f"selected FLEURS row {source_index} is outside {repo_path}"
                )
            row = selected_rows[0]
            row["_source_parquet_repo_path"] = repo_path
            row["_source_parquet_row_index"] = source_index
            full_rows.append(row)
    full_rows.sort(
        key=lambda row: (
            _id_sort_key(row["id"]),
            _fleurs_row_discriminator(row),
        )
    )
    return full_rows, observed_count


def _audio_bytes_from_row(row: Mapping[str, Any]) -> bytes:
    audio = row.get("audio")
    candidates: list[Any] = []
    if isinstance(audio, Mapping):
        candidates.extend(
            [
                audio.get("bytes"),
                audio.get("audio_bytes"),
                audio.get("data"),
            ]
        )
    else:
        candidates.append(audio)
    candidates.extend([row.get("audio_bytes"), row.get("bytes")])
    for candidate in candidates:
        if isinstance(candidate, bytes):
            return candidate
        if isinstance(candidate, (bytearray, memoryview)):
            return bytes(candidate)
    raise RuntimeError(
        f"FLEURS row {row.get('id')!r} does not contain encoded audio bytes"
    )


def _decode_canonical_audio(encoded: bytes) -> tuple[Any, int, int]:
    numpy = importlib.import_module("numpy")
    scipy_signal = importlib.import_module("scipy.signal")
    soundfile = importlib.import_module("soundfile")
    audio, sample_rate = soundfile.read(
        io.BytesIO(encoded),
        dtype="float32",
        always_2d=True,
    )
    source_channels = int(audio.shape[1])
    if source_channels <= 0 or int(audio.shape[0]) <= 0:
        raise ValueError("decoded audio is empty")
    mono = numpy.mean(audio.astype(numpy.float64), axis=1)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        mono = scipy_signal.resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )
    mono = numpy.asarray(mono, dtype=numpy.float32).reshape(-1)
    if not numpy.isfinite(mono).all():
        raise ValueError("decoded audio contains non-finite samples")
    mono = numpy.clip(mono, -1.0, 1.0)
    return numpy.ascontiguousarray(mono), int(sample_rate), source_channels


def _atomic_write_wav(path: Path, audio: Any) -> None:
    numpy = importlib.import_module("numpy")
    soundfile = importlib.import_module("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".wav",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        soundfile.write(
            str(temporary),
            numpy.asarray(audio, dtype=numpy.float32).reshape(-1),
            TARGET_SAMPLE_RATE,
            format="WAV",
            subtype="FLOAT",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_canonical_wav(path: str | os.PathLike[str]) -> Any:
    numpy = importlib.import_module("numpy")
    soundfile = importlib.import_module("soundfile")
    audio, sample_rate = soundfile.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    if int(sample_rate) != TARGET_SAMPLE_RATE or audio.shape[1] != 1:
        raise RuntimeError(f"fixture WAV is not canonical mono 16 kHz: {path}")
    return numpy.ascontiguousarray(audio[:, 0], dtype=numpy.float32)


def _canonical_sample_manifest_row(sample: Mapping[str, Any]) -> dict[str, Any]:
    retained_keys = (
        "utterance_id",
        "language",
        "config",
        "condition",
        "audio_relpath",
        "reference",
        "reference_source",
        "raw_transcription",
        "transcription",
        "source_id",
        "fleurs_id",
        "source_path",
        "source_parquet_repo_path",
        "raw_audio_sha256",
        "audio_pcm_sha256",
        "clean_pcm_sha256",
        "audio_file_sha256",
        "reference_sha256",
        "num_samples",
        "duration_seconds",
        "duration_s",
        "sample_rate",
        "channels",
        "pcm_dtype",
        "source_sample_rate",
        "source_channels",
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
    return {key: sample.get(key) for key in retained_keys if key in sample}


def score_records(
    records: Sequence[Mapping[str, Any]],
    model: str,
    bank: str,
    revision: str,
    fixture_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Score model records by language and condition under both normalisers."""

    if not records:
        raise ValueError("at least one ASR record is required")
    normalisers = _load_sibling_normalisers()
    fixture_index = _fixture_sample_index(fixture_info)
    hydrated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for original in records:
        record = dict(original)
        utterance_id = str(
            record.get("utterance_id")
            or record.get("id")
            or record.get("fleurs_id")
            or ""
        )
        if not utterance_id:
            raise KeyError("record lacks utterance_id")
        condition = str(record.get("condition") or CLEAN_CONDITION)
        key = (utterance_id, condition)
        fixture_sample = fixture_index.get(key, {})
        language_value = record.get("language") or fixture_sample.get("language")
        if language_value is None:
            raise KeyError(f"record {utterance_id} lacks language")
        language = canonical_language(str(language_value))
        reference = record.get("reference")
        if reference is None:
            reference = fixture_sample.get("reference")
        hypothesis = record.get("hypothesis")
        if hypothesis is None:
            hypothesis = record.get("text")
        if not isinstance(reference, str):
            raise TypeError(f"record {utterance_id} reference must be text")
        if hypothesis is None:
            hypothesis = ""
        if not isinstance(hypothesis, str):
            hypothesis = str(hypothesis)
        if key in seen:
            raise ValueError(f"duplicate record for {utterance_id}/{condition}")
        seen.add(key)
        hydrated.append(
            {
                **record,
                "utterance_id": utterance_id,
                "language": language,
                "condition": condition,
                "reference": reference,
                "hypothesis": hypothesis,
                "audio_pcm_sha256": record.get("audio_pcm_sha256")
                or fixture_sample.get("audio_pcm_sha256"),
                "raw_audio_sha256": record.get("raw_audio_sha256")
                or fixture_sample.get("raw_audio_sha256"),
                "duration_seconds": record.get("duration_seconds")
                or record.get("duration_s")
                or fixture_sample.get("duration_seconds")
                or fixture_sample.get("duration_s"),
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in hydrated:
        grouped[(record["language"], record["condition"])].append(record)

    cells: list[dict[str, Any]] = []
    utterance_rows: list[dict[str, Any]] = []
    for (language, condition), selected in sorted(
        grouped.items(),
        key=lambda item: (
            LANGUAGE_ORDER.index(item[0][0]),
            0 if item[0][1] == CLEAN_CONDITION else 1,
            item[0][1],
        ),
    ):
        selected.sort(key=lambda row: _id_sort_key(row["utterance_id"]))
        references = [row["reference"] for row in selected]
        hypotheses = [row["hypothesis"] for row in selected]
        control = _score_cell_with_normalisers(
            normalisers,
            references,
            hypotheses,
            language,
        )
        pinned_utterances = control["pinned"]["utterances"]
        basic_utterances = control["basic"]["utterances"]
        cell_utterances: list[dict[str, Any]] = []
        for source, pinned, basic in zip(
            selected,
            pinned_utterances,
            basic_utterances,
        ):
            flags = _silent_failure_flags(
                normalisers,
                source["reference"],
                source["hypothesis"],
                language,
            )
            row = {
                "utterance_id": source["utterance_id"],
                "language": language,
                "condition": condition,
                "model": model,
                "bank": bank,
                "revision": revision,
                "reference": source["reference"],
                "hypothesis": source["hypothesis"],
                "normalized_reference": pinned.get("normalized_reference"),
                "normalized_hypothesis": pinned.get("normalized_hypothesis"),
                "normalizer_id": pinned.get(
                    "normalizer_id", control["pinned"].get("normalizer_id")
                ),
                "metric": pinned.get("metric", control["pinned"].get("metric")),
                "S": int(pinned.get("S", pinned.get("substitutions", 0))),
                "I": int(pinned.get("I", pinned.get("insertions", 0))),
                "D": int(pinned.get("D", pinned.get("deletions", 0))),
                "errors": int(
                    pinned.get(
                        "errors",
                        int(pinned.get("S", 0))
                        + int(pinned.get("I", 0))
                        + int(pinned.get("D", 0)),
                    )
                ),
                "reference_length": int(pinned["reference_length"]),
                "hypothesis_length": int(
                    pinned.get("hypothesis_length", 0)
                ),
                "rate": float(pinned["rate"]),
                "basic_normalized_reference": basic.get(
                    "normalized_reference"
                ),
                "basic_normalized_hypothesis": basic.get(
                    "normalized_hypothesis"
                ),
                "basic_normalizer_id": basic.get(
                    "normalizer_id", control["basic"].get("normalizer_id")
                ),
                "basic_S": int(
                    basic.get("S", basic.get("substitutions", 0))
                ),
                "basic_I": int(
                    basic.get("I", basic.get("insertions", 0))
                ),
                "basic_D": int(
                    basic.get("D", basic.get("deletions", 0))
                ),
                "basic_reference_length": int(basic["reference_length"]),
                "basic_rate": float(basic["rate"]),
                "silent_failure_flags": flags,
                "audio_pcm_sha256": source.get("audio_pcm_sha256"),
                "raw_audio_sha256": source.get("raw_audio_sha256"),
                "duration_seconds": source.get("duration_seconds"),
            }
            cell_utterances.append(row)
            utterance_rows.append(row)

        fixture_hash = (
            fixture_info.get("fixture_hashes_by_language", {}).get(language)
            if isinstance(
                fixture_info.get("fixture_hashes_by_language"), Mapping
            )
            else fixture_info.get("fixture_manifest_sha256")
            or fixture_info.get("fixture_sha256")
        )
        cell = {
            "model": model,
            "bank": bank,
            "revision": revision,
            "language": language,
            "condition": condition,
            "corpus": (
                f"{FLEURS_REPO_ID}@{FLEURS_REVISION}"
                f"/{FLEURS_CONFIGS[language]}/{FLEURS_SPLIT}"
            ),
            "n": len(selected),
            "metric": control["pinned"]["metric"],
            "wer_or_cer": control["pinned"]["rate"],
            "percent": control["pinned"]["percent"],
            "ci95": control["pinned"]["ci95"],
            "ci95_percent": control["pinned"]["ci95_percent"],
            "normalizer_id": control["pinned"]["normalizer_id"],
            "basic_wer_or_cer": control["basic"]["rate"],
            "basic_percent": control["basic"]["percent"],
            "basic_ci95": control["basic"]["ci95"],
            "basic_normalizer_id": control["basic"]["normalizer_id"],
            "basic_delta_pp": control["basic_delta_pp"],
            "normalizer_delta_pp": control["normalizer_delta_pp"],
            "ranking_inadmissible": control["ranking_inadmissible"],
            "normalizer_delta_threshold_pp": NORMALISER_DELTA_THRESHOLD_PP,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_method": "percentile utterance bootstrap, ratio of sums",
            "cluster_unit": "utterance",
            "fixture_sha256": fixture_hash,
            "mix_manifest_sha256": fixture_info.get(
                "mix_manifest_sha256"
            ),
            "evidence_class": "HOST",
            "counts": control["pinned"].get("counts"),
            "basic_counts": control["basic"].get("counts"),
            "silent_failure_counts": _sum_silent_flags(cell_utterances),
            "utterances": cell_utterances,
        }
        cells.append(cell)

    return {
        "schema": SCORE_SCHEMA,
        "created_at": utc_now(),
        "model": model,
        "bank": bank,
        "revision": revision,
        "corpus_repo_id": FLEURS_REPO_ID,
        "corpus_revision": FLEURS_REVISION,
        "corpus_split": FLEURS_SPLIT,
        "n_records": len(hydrated),
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "evidence_class": "HOST",
        "fixture_manifest_sha256": fixture_info.get(
            "fixture_manifest_sha256"
        )
        or fixture_info.get("fixture_sha256"),
        "mix_manifest_sha256": fixture_info.get("mix_manifest_sha256"),
        "normalisers_source_sha256": getattr(
            normalisers,
            "SOURCE_SHA256",
            getattr(
                normalisers,
                "PACKAGE_SHA256",
                sha256_file(Path(__file__).with_name("normalisers.py")),
            ),
        ),
        "cells": cells,
        "utterances": utterance_rows,
    }


def _fixture_sample_index(
    fixture_info: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    compact = fixture_info.get("sample_index")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    if isinstance(compact, Mapping):
        for key, sample in compact.items():
            if not isinstance(sample, Mapping):
                continue
            utterance_id = sample.get("utterance_id")
            condition = sample.get("condition")
            if utterance_id is None or condition is None:
                try:
                    utterance_id, condition = _split_sample_index_key(str(key))
                except ValueError:
                    continue
            index[(str(utterance_id), str(condition))] = sample
    samples = fixture_info.get("samples")
    if not isinstance(samples, list):
        return index
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        utterance_id = sample.get("utterance_id")
        condition = sample.get("condition")
        if utterance_id is not None and condition is not None:
            index[(str(utterance_id), str(condition))] = sample
    return index


def _sample_index_key(utterance_id: str, condition: str) -> str:
    return f"{utterance_id}\t{condition}"


def _split_sample_index_key(value: str) -> tuple[str, str]:
    if "\t" not in value:
        raise ValueError("invalid sample index key")
    return tuple(value.rsplit("\t", 1))  # type: ignore[return-value]


def _load_sibling_normalisers() -> Any:
    path = Path(__file__).resolve().with_name("normalisers.py")
    if not path.is_file():
        raise RuntimeError(f"sibling normalisers.py is missing: {path}")
    module_name = f"_lane_c_normalisers_{sha256_file(path)[:12]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import sibling normalisers.py: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _score_cell_with_normalisers(
    normalisers: Any,
    references: Sequence[str],
    hypotheses: Sequence[str],
    language: str,
) -> dict[str, Any]:
    # Lane B's frozen package is authoritative.  Its exact API scores a list
    # of (reference, hypothesis) pairs and retains per-utterance edit counts.
    if hasattr(normalisers, "compare_corpus") and hasattr(
        normalisers, "score_corpus"
    ):
        pairs = list(zip(references, hypotheses))
        project_raw = _call_lane_b_score_corpus(
            normalisers,
            pairs,
            language,
            basic=False,
        )
        basic_raw = _call_lane_b_score_corpus(
            normalisers,
            pairs,
            language,
            basic=True,
        )
        project = _canonicalize_normalizer_corpus(project_raw, language)
        basic = _canonicalize_normalizer_corpus(basic_raw, language)
        basic_delta_pp = basic["percent"] - project["percent"]
        absolute_delta_pp = abs(basic_delta_pp)
        return {
            "pinned": project,
            "basic": basic,
            "basic_delta_pp": basic_delta_pp,
            "normalizer_delta_pp": absolute_delta_pp,
            "ranking_inadmissible": (
                absolute_delta_pp > NORMALISER_DELTA_THRESHOLD_PP
            ),
            "inadmissibility_threshold_pp": NORMALISER_DELTA_THRESHOLD_PP,
            "normaliser_api": "lane-b score_corpus(pair-list)",
        }

    # Keep the self-contained fallback usable until the Lane B source replaces
    # it in out/c for the Colab upload.
    if hasattr(normalisers, "score_with_basic_control"):
        result = normalisers.score_with_basic_control(
            references,
            hypotheses,
            language,
            B=BOOTSTRAP_B,
            seed=BOOTSTRAP_SEED,
        )
        result.setdefault(
            "ranking_inadmissible",
            abs(float(result["basic_delta_pp"]))
            > NORMALISER_DELTA_THRESHOLD_PP,
        )
        result.setdefault(
            "normalizer_delta_pp", abs(float(result["basic_delta_pp"]))
        )
        return result
    raise RuntimeError(
        "sibling normalisers.py exports neither Lane B score_corpus nor "
        "score_with_basic_control"
    )


def _call_lane_b_score_corpus(
    normalisers: Any,
    pairs: Sequence[tuple[str, str]],
    language: str,
    *,
    basic: bool,
) -> Mapping[str, Any]:
    inspect = importlib.import_module("inspect")
    signature = inspect.signature(normalisers.score_corpus)
    parameters = signature.parameters
    kwargs: dict[str, Any] = {}
    if "normalizer" in parameters:
        kwargs["normalizer"] = "basic" if basic else "project"
    elif "basic" in parameters:
        kwargs["basic"] = basic
    if "B" in parameters:
        kwargs["B"] = BOOTSTRAP_B
    if "seed" in parameters:
        kwargs["seed"] = BOOTSTRAP_SEED
    result = normalisers.score_corpus(pairs, language, **kwargs)
    if not isinstance(result, Mapping):
        raise TypeError("Lane B score_corpus must return a mapping")
    if isinstance(result.get("utterances"), list):
        return result

    # A future compact API may omit utterances.  Rehydrate them through the
    # sibling package's own score_pair function so S/I/D remains auditable.
    if not hasattr(normalisers, "score_pair"):
        raise RuntimeError("Lane B score_corpus omitted utterances and score_pair")
    pair_signature = inspect.signature(normalisers.score_pair)
    pair_parameters = pair_signature.parameters
    utterances = []
    for reference, hypothesis in pairs:
        pair_kwargs: dict[str, Any] = {}
        if "normalizer" in pair_parameters:
            pair_kwargs["normalizer"] = "basic" if basic else "project"
        elif "basic" in pair_parameters:
            pair_kwargs["basic"] = basic
        utterances.append(
            normalisers.score_pair(
                reference,
                hypothesis,
                language,
                **pair_kwargs,
            )
        )
    return {**dict(result), "utterances": utterances}


def _canonicalize_normalizer_corpus(
    raw: Mapping[str, Any],
    language: str,
) -> dict[str, Any]:
    raw_utterances = raw.get("utterances")
    if not isinstance(raw_utterances, list) or not raw_utterances:
        raise ValueError("normaliser corpus result contains no utterances")
    utterances = [
        _canonicalize_normalizer_utterance(row, language)
        for row in raw_utterances
    ]
    bootstrap = _bootstrap_ratio_pairs(
        [
            (float(row["errors"]), float(row["reference_length"]))
            for row in utterances
        ],
        B=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
    )
    counts = {
        "S": sum(row["S"] for row in utterances),
        "I": sum(row["I"] for row in utterances),
        "D": sum(row["D"] for row in utterances),
        "substitutions": sum(row["S"] for row in utterances),
        "insertions": sum(row["I"] for row in utterances),
        "deletions": sum(row["D"] for row in utterances),
        "errors": sum(row["errors"] for row in utterances),
        "reference_length": sum(
            row["reference_length"] for row in utterances
        ),
        "hypothesis_length": sum(
            row["hypothesis_length"] for row in utterances
        ),
    }
    metric = str(raw.get("metric") or ("CER" if language == "zh" else "WER"))
    rate = float(bootstrap["value"])
    ci95 = list(bootstrap["ci95"])
    return {
        "language": language,
        "metric": metric.lower(),
        "normalizer_id": str(raw.get("normalizer_id") or ""),
        "n": len(utterances),
        "rate": rate,
        "percent": rate * 100.0,
        "wer_or_cer": rate,
        "ci95": ci95,
        "ci95_percent": [value * 100.0 for value in ci95],
        "bootstrap_B": BOOTSTRAP_B,
        "seed": BOOTSTRAP_SEED,
        "ci_method": "percentile",
        "cluster_unit": "utterance",
        "counts": counts,
        "utterances": utterances,
    }


def _canonicalize_normalizer_utterance(
    raw: Mapping[str, Any],
    language: str,
) -> dict[str, Any]:
    substitutions = int(raw.get("S", raw.get("substitutions", 0)))
    insertions = int(raw.get("I", raw.get("insertions", 0)))
    deletions = int(raw.get("D", raw.get("deletions", 0)))
    reference_length = int(
        raw.get(
            "reference_length",
            raw.get("reference_token_count", 0),
        )
    )
    hypothesis_tokens = raw.get("hypothesis_tokens")
    hypothesis_length = int(
        raw.get(
            "hypothesis_length",
            len(hypothesis_tokens)
            if isinstance(hypothesis_tokens, Sequence)
            and not isinstance(hypothesis_tokens, (str, bytes))
            else 0,
        )
    )
    errors = int(
        raw.get(
            "errors",
            substitutions + insertions + deletions,
        )
    )
    if reference_length <= 0:
        raise ValueError(
            "each bootstrapped utterance must have positive reference length"
        )
    return {
        **dict(raw),
        "language": language,
        "metric": str(
            raw.get("metric") or ("CER" if language == "zh" else "WER")
        ).lower(),
        "normalized_reference": raw.get(
            "normalized_reference",
            raw.get("reference_normalized"),
        ),
        "normalized_hypothesis": raw.get(
            "normalized_hypothesis",
            raw.get("hypothesis_normalized"),
        ),
        "S": substitutions,
        "I": insertions,
        "D": deletions,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "errors": errors,
        "reference_length": reference_length,
        "hypothesis_length": hypothesis_length,
        "rate": errors / reference_length,
    }


def _silent_failure_flags(
    normalisers: Any,
    reference: str,
    hypothesis: str,
    language: str,
) -> dict[str, Any]:
    if hasattr(normalisers, "silent_failure_flags"):
        return dict(
            normalisers.silent_failure_flags(
                reference,
                hypothesis,
                language,
            )
        )
    expected_script = (
        "han" if language == "zh" else "cyrillic" if language == "ru" else "latin"
    )
    reference_script = _dominant_script(reference)
    hypothesis_script = _dominant_script(hypothesis)
    empty_output = not hypothesis.strip()
    traditional_output = False
    if language == "zh" and hypothesis.strip() and hasattr(
        normalisers, "_opencc_t2s"
    ):
        try:
            hypothesis_nfkc = unicodedata.normalize("NFKC", hypothesis)
            reference_nfkc = unicodedata.normalize("NFKC", reference)
            traditional_output = bool(
                normalisers._opencc_t2s(hypothesis_nfkc) != hypothesis_nfkc
                and normalisers._opencc_t2s(reference_nfkc) == reference_nfkc
            )
        except Exception:
            traditional_output = False
    script_mismatch = bool(
        not empty_output
        and (
            traditional_output
            or (
                reference_script is not None
                and hypothesis_script is not None
                and reference_script != hypothesis_script
            )
        )
    )
    language_swap = bool(
        not empty_output
        and hypothesis_script is not None
        and hypothesis_script != expected_script
    )
    return {
        "empty_output": empty_output,
        "script_family_mismatch": script_mismatch,
        "language_swap": language_swap,
        "zh_traditional_output_against_simplified_reference": traditional_output,
        "expected_script": expected_script,
        "reference_dominant_script": reference_script,
        "hypothesis_dominant_script": hypothesis_script,
    }


def _dominant_script(text: str) -> str | None:
    counts = {"han": 0, "cyrillic": 0, "latin": 0}
    for character in text:
        codepoint = ord(character)
        name = unicodedata.name(character, "")
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x323AF
        ):
            counts["han"] += 1
        elif "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
    maximum = max(counts.values())
    if maximum == 0:
        return None
    winners = [name for name, count in counts.items() if count == maximum]
    return winners[0] if len(winners) == 1 else None


def _sum_silent_flags(
    utterances: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for utterance in utterances:
        flags = utterance.get("silent_failure_flags", {})
        if not isinstance(flags, Mapping):
            continue
        for name, value in flags.items():
            if value is True:
                counts[str(name)] += 1
    return dict(sorted(counts.items()))


def partial_trace_metrics(
    traces: Sequence[Mapping[str, Any]],
    language: str,
) -> dict[str, Any]:
    """Compute auditable partial-stability metrics for one route/language."""

    canonical_lang = canonical_language(language)
    normalisers = _load_sibling_normalisers()
    prepared_traces = _prepare_partial_traces(traces)
    per_utterance: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    no_partial_count = 0
    for trace_index, trace in enumerate(prepared_traces):
        utterance_id = str(
            trace.get("utterance_id")
            or trace.get("id")
            or f"trace-{trace_index}"
        )
        trace_language = trace.get("language")
        if trace_language is not None and canonical_language(
            str(trace_language)
        ) != canonical_lang:
            raise ValueError(
                f"trace {utterance_id} language does not match {canonical_lang}"
            )
        try:
            partials = _coerce_partials(trace)
            source_audio_exposure_s = _partial_source_audio_exposure_s(
                trace, partials
            )
            if not partials:
                no_partial_count += 1
                per_utterance.append(
                    {
                        "utterance_id": utterance_id,
                        "partials_emitted": False,
                        "structural_reason": trace.get("structural_reason")
                        or trace.get("no_partials_reason")
                        or "no non-empty partial callback was emitted",
                        "first_partial_latency_s": None,
                        "rewrite_edits": 0,
                        "rewrite_audio_seconds": 0.0,
                        "source_audio_exposure_s": source_audio_exposure_s,
                        "rewrite_audio_denominator_semantics": (
                            "not applicable: no non-empty partial hypotheses"
                        ),
                        "partial_rewrite_rate": None,
                        "partial_rewrite_rate_status": "not_applicable_no_partials",
                        "zero_source_audio_rewrite_transition_count": 0,
                        "zero_source_audio_rewrite_edits": 0,
                        "raw_rewrite_edits": 0,
                        "raw_partial_rewrite_rate": None,
                        "raw_partial_rewrite_rate_status": (
                            "not_applicable_no_partials"
                        ),
                        "raw_zero_source_audio_rewrite_transition_count": 0,
                        "raw_zero_source_audio_rewrite_edits": 0,
                        "final_partial_divergence": None,
                        "final_partial_differs": None,
                        "trace": [],
                        "source_events": trace.get("_source_events"),
                    }
                )
                continue
            _validate_partial_monotonicity(partials)
            first_latency = _first_partial_latency(trace, partials[0])
            normalized_partials = [
                _normalize_for_partial(normalisers, item["text"], canonical_lang)
                for item in partials
            ]
            normalized_tokens = [
                _metric_tokens(normalisers, text, canonical_lang)
                for text in normalized_partials
            ]
            raw_tokens = [
                _raw_metric_tokens(item["text"], canonical_lang)
                for item in partials
            ]
            rewrite_edits = 0
            raw_rewrite_edits = 0
            positive_transition_audio_seconds = 0.0
            zero_audio_rewrite_edits = 0
            raw_zero_audio_rewrite_edits = 0
            zero_audio_rewrite_transition_count = 0
            raw_zero_audio_rewrite_transition_count = 0
            transitions: list[dict[str, Any]] = []
            for index in range(1, len(partials)):
                elapsed = (
                    partials[index]["audio_consumed_s"]
                    - partials[index - 1]["audio_consumed_s"]
                )
                normalized_distance = _edit_distance(
                    normalized_tokens[index - 1],
                    normalized_tokens[index],
                )
                raw_distance = _edit_distance(
                    raw_tokens[index - 1],
                    raw_tokens[index],
                )
                if elapsed > 0.0:
                    positive_transition_audio_seconds += elapsed
                zero_audio_rewrite = elapsed == 0.0 and normalized_distance > 0
                raw_zero_audio_rewrite = elapsed == 0.0 and raw_distance > 0
                if zero_audio_rewrite:
                    zero_audio_rewrite_edits += normalized_distance
                    zero_audio_rewrite_transition_count += 1
                if raw_zero_audio_rewrite:
                    raw_zero_audio_rewrite_edits += raw_distance
                    raw_zero_audio_rewrite_transition_count += 1
                rewrite_edits += normalized_distance
                raw_rewrite_edits += raw_distance
                transitions.append(
                    {
                        "from_index": index - 1,
                        "to_index": index,
                        "audio_seconds": elapsed,
                        "normalized_edit_distance": normalized_distance,
                        "raw_edit_distance": raw_distance,
                        "zero_source_audio_rewrite": zero_audio_rewrite,
                        "raw_zero_source_audio_rewrite": raw_zero_audio_rewrite,
                    }
                )
            rewrite_rate = (
                rewrite_edits / source_audio_exposure_s
                if source_audio_exposure_s > 0.0
                else (0.0 if rewrite_edits == 0 else None)
            )
            raw_rewrite_rate = (
                raw_rewrite_edits / source_audio_exposure_s
                if source_audio_exposure_s > 0.0
                else (0.0 if raw_rewrite_edits == 0 else None)
            )
            final_hypothesis = _final_hypothesis(trace)
            normalized_final = _normalize_for_partial(
                normalisers,
                final_hypothesis,
                canonical_lang,
            )
            normalized_last = normalized_partials[-1]
            final_characters = list(normalized_final)
            last_characters = list(normalized_last)
            final_distance = _edit_distance(
                final_characters,
                last_characters,
            )
            if final_characters:
                divergence = final_distance / len(final_characters)
            else:
                divergence = 0.0 if not last_characters else math.inf
            per_utterance.append(
                {
                    "utterance_id": utterance_id,
                    "partials_emitted": True,
                    "first_partial_latency_s": first_latency,
                    "rewrite_edits": rewrite_edits,
                    "rewrite_audio_seconds": source_audio_exposure_s,
                    "source_audio_exposure_s": source_audio_exposure_s,
                    "positive_partial_transition_audio_seconds": (
                        positive_transition_audio_seconds
                    ),
                    "rewrite_audio_denominator_semantics": (
                        "maximum actual source audio consumed across callback "
                        "events; synthetic tail silence excluded"
                    ),
                    "partial_rewrite_rate": rewrite_rate,
                    "partial_rewrite_rate_status": (
                        "undefined_zero_source_audio_exposure"
                        if source_audio_exposure_s == 0.0 and rewrite_edits > 0
                        else "defined"
                    ),
                    "zero_source_audio_rewrite_transition_count": (
                        zero_audio_rewrite_transition_count
                    ),
                    "zero_source_audio_rewrite_edits": zero_audio_rewrite_edits,
                    "raw_rewrite_edits": raw_rewrite_edits,
                    "raw_partial_rewrite_rate": raw_rewrite_rate,
                    "raw_partial_rewrite_rate_status": (
                        "undefined_zero_source_audio_exposure"
                        if source_audio_exposure_s == 0.0
                        and raw_rewrite_edits > 0
                        else "defined"
                    ),
                    "raw_zero_source_audio_rewrite_transition_count": (
                        raw_zero_audio_rewrite_transition_count
                    ),
                    "raw_zero_source_audio_rewrite_edits": (
                        raw_zero_audio_rewrite_edits
                    ),
                    "final_hypothesis": final_hypothesis,
                    "last_partial": partials[-1]["text"],
                    "normalized_final_hypothesis": normalized_final,
                    "normalized_last_partial": normalized_last,
                    "final_partial_edit_distance": final_distance,
                    "final_reference_characters": len(final_characters),
                    "final_partial_divergence": divergence,
                    "final_partial_differs": final_distance > 0,
                    "transitions": transitions,
                    "trace": [
                        {
                            **item,
                            "normalized_text": normalized,
                        }
                        for item, normalized in zip(
                            partials, normalized_partials
                        )
                    ],
                    "source_events": trace.get("_source_events"),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(
                {
                    "utterance_id": utterance_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    valid_emitted = [
        row
        for row in per_utterance
        if row["partials_emitted"]
        and math.isfinite(float(row["final_partial_divergence"]))
    ]
    latencies = [
        float(row["first_partial_latency_s"])
        for row in valid_emitted
        if row["first_partial_latency_s"] is not None
    ]
    zero_audio_rewrite_transition_rows = [
        row
        for row in valid_emitted
        if int(row["zero_source_audio_rewrite_transition_count"]) > 0
    ]
    raw_zero_audio_rewrite_transition_rows = [
        row
        for row in valid_emitted
        if int(row["raw_zero_source_audio_rewrite_transition_count"]) > 0
    ]
    zero_audio_exposure_rewrite_rows = [
        row
        for row in valid_emitted
        if float(row["rewrite_audio_seconds"]) == 0.0
        and int(row["rewrite_edits"]) > 0
    ]
    raw_zero_audio_exposure_rewrite_rows = [
        row
        for row in valid_emitted
        if float(row["rewrite_audio_seconds"]) == 0.0
        and int(row["raw_rewrite_edits"]) > 0
    ]
    rewrite_pairs = [
        (
            float(row["rewrite_edits"]),
            float(row["rewrite_audio_seconds"]),
        )
        for row in valid_emitted
        if float(row["rewrite_audio_seconds"]) > 0.0
    ]
    raw_rewrite_pairs = [
        (
            float(row["raw_rewrite_edits"]),
            float(row["rewrite_audio_seconds"]),
        )
        for row in valid_emitted
        if float(row["rewrite_audio_seconds"]) > 0.0
    ]
    divergences = [
        float(row["final_partial_divergence"]) for row in valid_emitted
    ]
    first_partial_summary = _median_order_statistic_summary(latencies)
    defined_rewrite_summary = _bootstrap_ratio_pairs(
        rewrite_pairs,
        B=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
    )
    rewrite_summary = (
        _fail_closed_rewrite_ratio_summary(
            defined_rewrite_summary,
            valid_n=len(valid_emitted),
            affected_rows=zero_audio_exposure_rewrite_rows,
            edit_count_key="rewrite_edits",
        )
        if zero_audio_exposure_rewrite_rows
        else defined_rewrite_summary
    )
    defined_raw_rewrite_summary = _bootstrap_ratio_pairs(
        raw_rewrite_pairs,
        B=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
    )
    raw_rewrite_summary = (
        _fail_closed_rewrite_ratio_summary(
            defined_raw_rewrite_summary,
            valid_n=len(valid_emitted),
            affected_rows=raw_zero_audio_exposure_rewrite_rows,
            edit_count_key="raw_rewrite_edits",
        )
        if raw_zero_audio_exposure_rewrite_rows
        else defined_raw_rewrite_summary
    )
    rewrite_denominator_semantics = (
        "per utterance: maximum actual source audio consumed across callback "
        "events; aggregate: ratio of summed edits to summed per-utterance "
        "source-audio exposure; synthetic tail silence excluded"
    )
    rewrite_summary["denominator_semantics"] = rewrite_denominator_semantics
    raw_rewrite_summary["denominator_semantics"] = rewrite_denominator_semantics
    divergence_summary = _bootstrap_mean(
        divergences,
        B=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
    )
    divergence_ratio = _bootstrap_ratio_pairs(
        [
            (
                float(row["final_partial_edit_distance"]),
                float(row["final_reference_characters"]),
            )
            for row in valid_emitted
            if int(row["final_reference_characters"]) > 0
        ],
        B=BOOTSTRAP_B,
        seed=BOOTSTRAP_SEED,
    )
    differing = sum(
        bool(row["final_partial_differs"]) for row in valid_emitted
    )
    return {
        "schema": PARTIAL_SCHEMA,
        "language": canonical_lang,
        "normalizer_id": _normalizer_id(normalisers, canonical_lang),
        "n_input": len(prepared_traces),
        "n_trace_events_input": len(traces),
        "n_valid": len(per_utterance),
        "n_partials_emitted": len(valid_emitted),
        "n_no_partials": no_partial_count,
        "n_invalid": len(invalid),
        "partials_emitted": bool(valid_emitted),
        "first_partial_latency": first_partial_summary,
        "first_partial_latency_ms": (
            first_partial_summary["p50_s"] * 1000.0
            if first_partial_summary["p50_s"] is not None
            else None
        ),
        "partial_rewrite_rate": rewrite_summary,
        "partial_rewrite_rate_denominator_semantics": (
            rewrite_denominator_semantics
        ),
        "partial_rewrite_rate_edits_per_audio_second": rewrite_summary[
            "value"
        ],
        "partial_rewrite_rate_status": rewrite_summary.get(
            "status",
            (
                "defined"
                if rewrite_summary["value"] is not None
                else "not_observed"
            ),
        ),
        "zero_source_audio_rewrite_utterance_count": len(
            zero_audio_rewrite_transition_rows
        ),
        "zero_source_audio_rewrite_transition_count": sum(
            int(row["zero_source_audio_rewrite_transition_count"])
            for row in zero_audio_rewrite_transition_rows
        ),
        "zero_source_audio_rewrite_edits": sum(
            int(row["zero_source_audio_rewrite_edits"])
            for row in zero_audio_rewrite_transition_rows
        ),
        "raw_partial_rewrite_rate": raw_rewrite_summary,
        "raw_partial_rewrite_rate_status": raw_rewrite_summary.get(
            "status",
            (
                "defined"
                if raw_rewrite_summary["value"] is not None
                else "not_observed"
            ),
        ),
        "raw_zero_source_audio_rewrite_utterance_count": len(
            raw_zero_audio_rewrite_transition_rows
        ),
        "raw_zero_source_audio_rewrite_transition_count": sum(
            int(row["raw_zero_source_audio_rewrite_transition_count"])
            for row in raw_zero_audio_rewrite_transition_rows
        ),
        "raw_zero_source_audio_rewrite_edits": sum(
            int(row["raw_zero_source_audio_rewrite_edits"])
            for row in raw_zero_audio_rewrite_transition_rows
        ),
        "final_partial_divergence": divergence_summary,
        "final_vs_last_partial_divergence": divergence_summary["value"],
        "final_partial_divergence_ratio_of_sums": divergence_ratio,
        "final_partial_differs_fraction": (
            differing / len(valid_emitted) if valid_emitted else None
        ),
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_unit": "utterance",
        "execution_observed": bool(per_utterance),
        "evidence_class": "HOST" if per_utterance else "UNKNOWN",
        "invalid_traces": invalid,
        "utterances": per_utterance,
    }


def _prepare_partial_traces(
    traces: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Accept either per-utterance traces or flat callback event rows."""

    if not traces:
        return []
    nested_keys = ("partials", "partial_trace", "trace")
    nested = [
        trace
        for trace in traces
        if any(key in trace for key in nested_keys)
    ]
    if nested:
        if len(nested) != len(traces):
            raise ValueError("cannot mix nested traces with flat callback rows")
        return list(traces)

    if not all("text" in trace for trace in traces):
        return list(traces)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, event in enumerate(traces):
        utterance_id = str(
            event.get("utterance_id")
            or event.get("id")
            or f"trace-{index}"
        )
        condition = str(event.get("condition") or CLEAN_CONDITION)
        grouped[(utterance_id, condition)].append(event)

    prepared: list[Mapping[str, Any]] = []
    for (utterance_id, condition), events in sorted(grouped.items()):
        ordered_events = sorted(
            events,
            key=lambda event: (
                float(
                    event.get(
                        "stream_time_s",
                        event.get(
                            "audio_time_s",
                            event.get("wall_elapsed_s", 0.0),
                        ),
                    )
                ),
                float(event.get("wall_elapsed_s", 0.0)),
                bool(event.get("is_final", False)),
            ),
        )
        final_events = [
            event for event in ordered_events if bool(event.get("is_final"))
        ]
        nonfinal_events = [
            event
            for event in ordered_events
            if not bool(event.get("is_final"))
            and str(event.get("text", "")).strip()
        ]
        final_hypothesis = (
            str(final_events[-1].get("text", ""))
            if final_events
            else (
                str(ordered_events[-1].get("text", ""))
                if ordered_events
                else ""
            )
        )
        partials = [
            {
                "t3p_s": float(
                    event.get(
                        "wall_elapsed_s",
                        event.get("timestamp_s", 0.0),
                    )
                ),
                "text": str(event.get("text", "")),
                # Rewrite stability is normalized by source audio, not by the
                # synthetic tail silence deliberately fed to finalize decoding.
                "audio_consumed_s": float(
                    _first_present(
                        event,
                        (
                            "source_audio_consumed_s",
                            "audio_time_s",
                            "audio_consumed_s",
                            "stream_time_s",
                        ),
                    )
                    or 0.0
                ),
            }
            for event in nonfinal_events
        ]
        first_latency = (
            float(
                nonfinal_events[0].get(
                    "wall_elapsed_s",
                    nonfinal_events[0].get("timestamp_s", 0.0),
                )
            )
            if nonfinal_events
            else None
        )
        first = ordered_events[0]
        prepared.append(
            {
                "utterance_id": utterance_id,
                "language": first.get("language"),
                "condition": condition,
                "route": first.get("route") or first.get("engine"),
                "partials": partials,
                "first_partial_latency_s": first_latency,
                "final_hypothesis": final_hypothesis,
                "structural_reason": (
                    None
                    if partials
                    else "no non-final non-empty callback was emitted"
                ),
                "_source_events": [dict(event) for event in ordered_events],
            }
        )
    return prepared


def _coerce_partials(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_partials = trace.get("partials")
    if raw_partials is None:
        raw_partials = trace.get("partial_trace")
    if raw_partials is None:
        raw_partials = trace.get("trace")
    if raw_partials is None:
        return []
    if not isinstance(raw_partials, Sequence) or isinstance(
        raw_partials, (str, bytes)
    ):
        raise TypeError("partials must be a sequence")
    partials: list[dict[str, Any]] = []
    for index, item in enumerate(raw_partials):
        if isinstance(item, Mapping):
            text = item.get("text")
            if text is None:
                text = item.get("partial")
            if text is None:
                text = item.get("hypothesis")
            timestamp = _first_present(
                item,
                ("t3p_s", "timestamp_s", "time_s", "emitted_at_s"),
            )
            if timestamp is None:
                timestamp_ms = _first_present(
                    item,
                    ("t3p_ms", "timestamp_ms", "time_ms", "emitted_at_ms"),
                )
                timestamp = (
                    float(timestamp_ms) / 1000.0
                    if timestamp_ms is not None
                    else None
                )
            consumed = _first_present(
                item,
                (
                    "source_audio_consumed_s",
                    "audio_time_s",
                    "audio_consumed_s",
                    "audio_seconds",
                    "consumed_s",
                ),
            )
            if consumed is None:
                consumed_ms = _first_present(
                    item,
                    (
                        "audio_consumed_ms",
                        "audio_ms",
                        "consumed_ms",
                    ),
                )
                consumed = (
                    float(consumed_ms) / 1000.0
                    if consumed_ms is not None
                    else None
                )
            latency = _first_present(
                item,
                ("latency_s", "first_partial_latency_s"),
            )
            if latency is None:
                latency_ms = _first_present(
                    item,
                    ("latency_ms", "first_partial_latency_ms"),
                )
                latency = (
                    float(latency_ms) / 1000.0
                    if latency_ms is not None
                    else None
                )
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes)
        ):
            if len(item) != 3:
                raise ValueError(
                    "tuple partials must be (t3p_s, text, audio_consumed_ms)"
                )
            timestamp, text, consumed_ms = item
            consumed = float(consumed_ms) / 1000.0
            latency = None
        else:
            raise TypeError(f"partial {index} has unsupported type")
        if text is None:
            raise KeyError(f"partial {index} lacks text")
        # Empty callbacks are decoder timing events, not partial hypotheses.
        # Filtering them also prevents the first empty -> text callback from
        # being counted as a rewrite.
        if not str(text).strip():
            continue
        if consumed is None:
            raise KeyError(f"partial {index} lacks audio_consumed_ms/s")
        partials.append(
            {
                "index": index,
                "text": str(text),
                "t3p_s": float(timestamp) if timestamp is not None else None,
                "audio_consumed_s": float(consumed),
                "latency_s": float(latency) if latency is not None else None,
            }
        )
    return partials


def _partial_source_audio_exposure_s(
    trace: Mapping[str, Any],
    partials: Sequence[Mapping[str, Any]],
) -> float:
    """Return actual source-audio exposure represented by callback events."""

    raw_events = trace.get("_source_events")
    if raw_events is None:
        raw_events = trace.get("partials")
    if raw_events is None:
        raw_events = trace.get("partial_trace")
    if raw_events is None:
        raw_events = trace.get("trace")
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, Sequence) or isinstance(
        raw_events, (str, bytes)
    ):
        raise TypeError("partial callback events must be a sequence")

    exposures: list[float] = []
    for index, item in enumerate(raw_events):
        consumed: Any = None
        if isinstance(item, Mapping):
            consumed = _first_present(
                item,
                (
                    "source_audio_consumed_s",
                    "audio_time_s",
                    "audio_consumed_s",
                    "audio_seconds",
                    "consumed_s",
                ),
            )
            if consumed is None:
                consumed_ms = _first_present(
                    item,
                    ("audio_consumed_ms", "audio_ms", "consumed_ms"),
                )
                if consumed_ms is not None:
                    consumed = float(consumed_ms) / 1000.0
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) == 3:
                consumed = float(item[2]) / 1000.0
        if consumed is None:
            continue
        exposure = float(consumed)
        if not math.isfinite(exposure) or exposure < 0.0:
            raise ValueError(
                f"partial callback {index} has invalid source-audio exposure"
            )
        exposures.append(exposure)

    if not exposures:
        exposures.extend(float(item["audio_consumed_s"]) for item in partials)
    return max(exposures, default=0.0)


def _first_present(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _validate_partial_monotonicity(partials: Sequence[Mapping[str, Any]]) -> None:
    last_timestamp: float | None = None
    last_audio = -math.inf
    for item in partials:
        consumed = float(item["audio_consumed_s"])
        if not math.isfinite(consumed) or consumed < 0.0:
            raise ValueError("audio-consumed timestamps must be finite and non-negative")
        if consumed < last_audio:
            raise ValueError("audio-consumed timestamps are not monotonic")
        last_audio = consumed
        timestamp = item.get("t3p_s")
        if timestamp is not None:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("partial timestamps must be finite")
            if last_timestamp is not None and timestamp < last_timestamp:
                raise ValueError("partial timestamps are not monotonic")
            last_timestamp = timestamp


def _first_partial_latency(
    trace: Mapping[str, Any],
    first_partial: Mapping[str, Any],
) -> float:
    explicit = _first_present(
        trace,
        ("first_partial_latency_s", "l_partial_first_s"),
    )
    if explicit is not None:
        latency = float(explicit)
    else:
        explicit_ms = _first_present(
            trace,
            ("first_partial_latency_ms", "l_partial_first_ms"),
        )
        if explicit_ms is not None:
            latency = float(explicit_ms) / 1000.0
        elif first_partial.get("latency_s") is not None:
            latency = float(first_partial["latency_s"])
        else:
            timestamp = first_partial.get("t3p_s")
            endpoint = _first_present(
                trace,
                ("t2_s", "endpoint_time_s", "endpoint_s"),
            )
            if endpoint is None:
                endpoint_ms = _first_present(
                    trace,
                    ("t2_ms", "endpoint_time_ms", "endpoint_ms"),
                )
                endpoint = (
                    float(endpoint_ms) / 1000.0
                    if endpoint_ms is not None
                    else None
                )
            if timestamp is None or endpoint is None:
                raise KeyError(
                    "first-partial latency needs an explicit latency or t3p/t2"
                )
            latency = float(timestamp) - float(endpoint)
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("first-partial latency must be finite and non-negative")
    return latency


def _final_hypothesis(trace: Mapping[str, Any]) -> str:
    value = _first_present(
        trace,
        ("final_hypothesis", "final", "hypothesis", "final_text"),
    )
    if value is None:
        raise KeyError("partial trace lacks final_hypothesis")
    return str(value)


def _normalize_for_partial(
    normalisers: Any,
    text: str,
    language: str,
) -> str:
    if hasattr(normalisers, "normalize_text"):
        normalized = normalisers.normalize_text(text, language)
        if isinstance(normalized, tuple):
            return str(normalized[0])
        return str(normalized)
    if hasattr(normalisers, "normalize"):
        normalized = normalisers.normalize(text, language)
        return str(normalized[0] if isinstance(normalized, tuple) else normalized)
    function = getattr(normalisers, f"norm_{language}", None)
    if function is None:
        raise RuntimeError(
            f"sibling normalisers.py cannot normalize {language}"
        )
    normalized = function(text)
    return str(normalized[0] if isinstance(normalized, tuple) else normalized)


def _metric_tokens(
    normalisers: Any,
    normalized_text: str,
    language: str,
) -> list[str]:
    if hasattr(normalisers, "metric_tokens"):
        return list(normalisers.metric_tokens(normalized_text, language))
    if hasattr(normalisers, "tokenize_normalized"):
        return list(
            normalisers.tokenize_normalized(normalized_text, language)
        )
    if language == "zh":
        return [character for character in normalized_text if not character.isspace()]
    return normalized_text.split()


def _raw_metric_tokens(text: str, language: str) -> list[str]:
    if language == "zh":
        return [character for character in text if not character.isspace()]
    return text.split()


def _normalizer_id(normalisers: Any, language: str) -> str | None:
    if hasattr(normalisers, "get_normalizer_id"):
        return str(normalisers.get_normalizer_id(language))
    if hasattr(normalisers, "normalizer_id"):
        return str(normalisers.normalizer_id(language))
    identifiers = getattr(
        normalisers,
        "NORMALIZER_IDS",
        getattr(normalisers, "NORMALISER_IDS", {}),
    )
    if isinstance(identifiers, Mapping):
        value = identifiers.get(f"norm_{language}")
        return str(value) if value is not None else None
    return None


def _edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (0 if left_item == right_item else 1),
                )
            )
        previous = current
    return previous[-1]


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of no values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fail_closed_rewrite_ratio_summary(
    defined_subset: Mapping[str, Any],
    *,
    valid_n: int,
    affected_rows: Sequence[Mapping[str, Any]],
    edit_count_key: str,
) -> dict[str, Any]:
    """Refuse a finite rewrite rate when a rewrite has zero source duration."""

    return {
        "value": None,
        "ci95": None,
        "n": valid_n,
        "n_defined_subset": int(defined_subset.get("n", 0)),
        "n_undefined": len(affected_rows),
        "B": int(defined_subset["B"]),
        "seed": int(defined_subset["seed"]),
        "ci_method": "not computed: zero total source-audio exposure",
        "status": "undefined_zero_source_audio_exposure",
        "reason": (
            "At least one utterance rewrote a hypothesis while its callbacks "
            "represented zero total source-audio exposure; a finite rate is "
            "undefined."
        ),
        "zero_source_audio_exposure_utterance_count": len(affected_rows),
        "zero_source_audio_exposure_rewrite_edits": sum(
            int(row[edit_count_key]) for row in affected_rows
        ),
        "affected_utterance_ids": [
            str(row["utterance_id"]) for row in affected_rows
        ],
        "defined_subset": dict(defined_subset),
    }


def _bootstrap_ratio_pairs(
    pairs: Sequence[tuple[float, float]],
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    if not pairs:
        return {
            "value": None,
            "ci95": None,
            "n": 0,
            "B": B,
            "seed": seed,
            "ci_method": "utterance percentile bootstrap, ratio of sums",
        }
    if any(denominator <= 0.0 for _, denominator in pairs):
        raise ValueError("bootstrap ratio denominators must be positive")
    numerator = sum(item[0] for item in pairs)
    denominator = sum(item[1] for item in pairs)
    rng = random.Random(seed)
    n = len(pairs)
    samples: list[float] = []
    for _ in range(B):
        sampled_numerator = 0.0
        sampled_denominator = 0.0
        for _ in range(n):
            selected = pairs[rng.randrange(n)]
            sampled_numerator += selected[0]
            sampled_denominator += selected[1]
        samples.append(sampled_numerator / sampled_denominator)
    return {
        "value": numerator / denominator,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "n": n,
        "B": B,
        "seed": seed,
        "ci_method": "utterance percentile bootstrap, ratio of sums",
        "numerator": numerator,
        "denominator": denominator,
        "per_utterance_p50": _percentile(
            [a / b for a, b in pairs], 0.5
        ),
        "per_utterance_p95": _percentile(
            [a / b for a, b in pairs], 0.95
        ),
    }


def _bootstrap_mean(
    values: Sequence[float],
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    if not values:
        return {
            "value": None,
            "ci95": None,
            "n": 0,
            "B": B,
            "seed": seed,
            "ci_method": "utterance percentile bootstrap of mean",
        }
    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(B):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        samples.append(total / n)
    return {
        "value": sum(values) / n,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "n": n,
        "B": B,
        "seed": seed,
        "ci_method": "utterance percentile bootstrap of mean",
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


def _median_order_statistic_summary(
    values: Sequence[float],
) -> dict[str, Any]:
    if not values:
        return {
            "p50_s": None,
            "ci95_s": None,
            "n": 0,
            "ci_method": "exact binomial order-statistic interval",
        }
    ordered = sorted(values)
    n = len(ordered)
    lower_rank = _median_lower_rank(n, 0.025)
    upper_rank = n - lower_rank + 1
    return {
        "p50_s": _percentile(ordered, 0.5),
        "ci95_s": [
            ordered[lower_rank - 1],
            ordered[upper_rank - 1],
        ],
        "n": n,
        "ci_method": "exact binomial order-statistic interval",
        "lower_order_statistic": lower_rank,
        "upper_order_statistic": upper_rank,
    }


def _median_lower_rank(n: int, lower_tail: float) -> int:
    if n <= 1:
        return 1
    cumulative = 0.0
    best = 1
    denominator = 2.0**n
    for successes in range(0, n + 1):
        cumulative += math.comb(n, successes) / denominator
        candidate_rank = successes + 1
        if cumulative <= lower_tail and candidate_rank <= n:
            best = candidate_rank
        else:
            break
    return best


__all__ = [
    "BOOTSTRAP_B",
    "BOOTSTRAP_SEED",
    "CLEAN_CONDITION",
    "COMMON_DEPENDENCIES",
    "FIXTURE_UTTERANCES_PER_LANGUAGE",
    "FLEURS_CONFIGS",
    "FLEURS_DATASET_ID",
    "FLEURS_N",
    "FLEURS_REPO_ID",
    "FLEURS_REVISION",
    "FLEURS_SPLIT",
    "FLEURS_TEST_ROW_COUNTS",
    "LANGUAGE_ORDER",
    "MAX_AUDIO_DURATION_SECONDS",
    "MIX_CONDITION",
    "MIX_SEED",
    "MIX_SNR_DB",
    "MUSAN_DATASET_ID",
    "MUSAN_NOISE_GLOB",
    "MUSAN_NOISE_FILE_COUNT",
    "MUSAN_PROVENANCE_FILES",
    "MUSAN_REPO_ID",
    "MUSAN_REVISION",
    "TARGET_SAMPLE_RATE",
    "atomic_checkpoint_write",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_fixture",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "canonical_language",
    "canonical_sha256",
    "derive_noise_assignment",
    "environment_receipt",
    "extract_noise_segment",
    "install_common_dependencies",
    "mix_at_snr",
    "partial_trace_metrics",
    "pcm_sha256",
    "row_duration_seconds",
    "score_records",
    "select_fleurs_rows",
    "sha256_bytes",
    "sha256_file",
]
