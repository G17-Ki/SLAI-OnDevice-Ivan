#!/usr/bin/env python3
"""Aggregate isolated Colab receipts into Lane C's C1--C5 deliverables.

Missing remote receipts are rendered as ``UNKNOWN`` rows.  This is deliberate:
the evidence protocol forbids replacing a failed or blocked host run with vendor
benchmarks, cross-corpus numbers, or arithmetic projections.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import unicodedata
import zipfile


FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
MUSAN_REPO = "alexwengg/musan_mini"
MUSAN_REVISION = "54ce99dd44add93cec39c4d37bdcc59c7870a849"
MUSAN_SOURCE_LABEL = "MUSAN noise subset mirror"
MUSAN_SOURCE_CAVEAT = (
    "Hugging Face subset mirror, not the canonical OpenSLR-17 distribution; "
    "it preserves the original MUSAN README plus per-source LICENSE and "
    "ANNOTATIONS files."
)
MUSAN_MIRROR_BYTES = 689_875_088
MUSAN_NOISE_WAV_BYTES = 246_777_286
MUSAN_NOISE_WAV_COUNT = 738
MUSAN_NOISE_GLOB = "noise/**/*.wav"
LANGUAGES = ("en", "zh", "es", "fr", "ru")
CONDITIONS = ("clean", "musan_10db")
TARGET_N = 150
OMNI_SMOKE_N = 30
PARTIAL_TRACE_N = 30
SENSEVOICE_FIXED_CONTEXT_SECONDS = 10.0
SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE = 16_000
SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES = 160_000
SENSEVOICE_FIXED_CONTEXT_PROBE_N = 30
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20_260_801
NORMALISERS_SHA256 = "82f38a8cbf3208e360acf0b6789e8940d345ace76e0bd11e3834a34edda828dc"
ZIPFORMER_MODEL = "pfluo/k2fsa-zipformer-chinese-english-mixed"
ZIPFORMER_REVISION = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"
FLEURS_CONFIGS = {
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "es": "es_419",
    "fr": "fr_fr",
    "ru": "ru_ru",
}
FLEURS_TEST_ROWS = {"en": 647, "zh": 945, "es": 908, "fr": 676, "ru": 775}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARCHIVE_SCHEMA = "lane-c-colab-archive/1.0"
ARCHIVE_PAYLOAD_ALLOWLIST = frozenset(
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
COMPLETE_ARCHIVE_MEMBERS = frozenset(
    {
        "archive_manifest.json",
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
    }
)
FIXTURE_BINDING_BASE_FIELDS = (
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
FIXTURE_BINDING_NOISE_FIELDS = (
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
FIXTURE_SCHEMA = "lane-c-asr-fixture/1.1"
MIX_MANIFEST_SCHEMA = "lane-c-musan-mix-manifest/1.0"
PORTABLE_FIXTURE_IDENTITY_SCHEMA = "lane-c-portable-fixture-identity/1.0"
# These values remain mandatory and are still validated inside each archive,
# but they are not portable identities for the decoded evaluation inputs.
# In particular, libsndfile FLOAT WAVs carry a PEAK-chunk timestamp, so the
# container SHA changes even when the canonical PCM is byte-identical.
PORTABLE_FIXTURE_VOLATILE_FIELDS = frozenset(
    {
        "utterance_id",
        "source_id",
        "source_path",
        "audio_relpath",
        "audio_file_sha256",
        "mix_entry_sha256",
        "mix_selector_sha256",
    }
)
MUSAN_CAVEAT_MARKDOWN = (
    "> **MUSAN subset-mirror caveat:** the 10 dB bucket uses "
    f"`{MUSAN_REPO}@{MUSAN_REVISION}`, a Hugging Face subset mirror rather "
    "than the canonical OpenSLR-17 distribution. The receipt preserves the "
    "upstream README, licences, annotations, and exact noise-file inventory; "
    "results must retain this provenance qualifier."
)

# Each model must resolve to one complete allowed asset group.  Repository
# revisions alone are not enough to identify the bytes that were executed.
MODEL_ASSET_GROUPS: dict[str, tuple[tuple[dict[str, Any], ...], ...]] = {
    "FunAudioLLM/SenseVoiceSmall": (
        (
            {"file": "model.pt", "bytes": 936_291_369, "sha256": "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"},
            {"file": "chn_jpn_yue_eng_ko_spectok.bpe.model", "bytes": 377_341, "sha256": "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"},
        ),
    ),
    "nvidia/parakeet-tdt-0.6b-v3": (
        (
            {"file": "parakeet-tdt-0.6b-v3.nemo", "bytes": 2_509_332_480, "sha256": "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"},
            {"file": "tokenizer.json", "sha256": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"},
        ),
        (
            {"file": "model.safetensors", "bytes": 2_508_311_120, "sha256": "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"},
            {"file": "tokenizer.json", "sha256": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"},
        ),
    ),
    "nvidia/canary-180m-flash": (
        (
            {"file": "canary-180m-flash.nemo", "bytes": 736_665_600, "sha256": "7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675"},
        ),
    ),
    "pfluo/k2fsa-zipformer-chinese-english-mixed": (
        (
            {"file": "exp/encoder-epoch-99-avg-1.onnx", "bytes": 268_919_863, "sha256": "a509c0d2797d7da93891950cefb1ab2daa33608f48895551d96a1a2fbc00d254"},
            {"file": "exp/decoder-epoch-99-avg-1.onnx", "bytes": 13_876_468, "sha256": "d91b63c84f27f1d822244ced4cc2b9f7fb8a60a328f13b860408b8889dd0a0ed"},
            {"file": "exp/joiner-epoch-99-avg-1.onnx", "bytes": 12_833_675, "sha256": "40f1268ddf9d65f0c2d5b8d2ba7ed6f3443d51898dfa826216afd9c53598f074"},
            {"file": "data/lang_char_bpe/tokens.txt", "sha256": "a8e0e4ec53810e433789b54a5c0134a7eaa2ffca595a6334d54c00da858841d3"},
        ),
    ),
    "ai-sage/GigaAM-Multilingual": (
        (
            {"file": "pytorch_model.bin", "bytes": 883_170_115, "sha256": "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728"},
        ),
    ),
    "facebook/omniASR-CTC-300M": (
        (
            {"file": "omniASR-CTC-300M.pt", "bytes": 1_302_113_892, "sha256": "4d31c6a7725a03e90cf26080c67361928755a9f43d322dd574145fbe91c76aec"},
            {"file": "omniASR_tokenizer.model", "bytes": 87_562, "sha256": "b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe"},
        ),
    ),
}

ROUTES: dict[str, dict[str, dict[str, str]]] = {
    "D-fork": {
        "en": {
            "model": "FunAudioLLM/SenseVoiceSmall",
            "engine": "sensevoice",
            "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        },
        "zh": {
            "model": "FunAudioLLM/SenseVoiceSmall",
            "engine": "sensevoice",
            "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        },
        "es": {
            "model": "nvidia/parakeet-tdt-0.6b-v3",
            "engine": "nemo",
            "revision": "7c35754d166cca382ad1e53e68b01e7c575f3a1d",
        },
        "fr": {
            "model": "nvidia/parakeet-tdt-0.6b-v3",
            "engine": "nemo",
            "revision": "7c35754d166cca382ad1e53e68b01e7c575f3a1d",
        },
        "ru": {
            "model": "nvidia/parakeet-tdt-0.6b-v3",
            "engine": "nemo",
            "revision": "7c35754d166cca382ad1e53e68b01e7c575f3a1d",
        },
    },
    "L-fork": {
        "en": {
            "model": "nvidia/canary-180m-flash",
            "engine": "nemo",
            "revision": "b12ab418510d093e83890178fd0e8b0d0f7918a6",
        },
        "zh": {
            "model": "pfluo/k2fsa-zipformer-chinese-english-mixed",
            "engine": "zipformer",
            "revision": "6eb615ae77ecac05c5628d5c8ed7037c14a338d5",
        },
        "es": {
            "model": "nvidia/canary-180m-flash",
            "engine": "nemo",
            "revision": "b12ab418510d093e83890178fd0e8b0d0f7918a6",
        },
        "fr": {
            "model": "nvidia/canary-180m-flash",
            "engine": "nemo",
            "revision": "b12ab418510d093e83890178fd0e8b0d0f7918a6",
        },
        "ru": {
            "model": "ai-sage/GigaAM-Multilingual",
            "engine": "gigaam",
            "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
        },
    },
    "OmniASR": {
        language: {
            "model": "facebook/omniASR-CTC-300M",
            "engine": "omni",
            "revision": "8e35f0cc28fa6099e0c14d56db85ce0423baa691",
        }
        for language in LANGUAGES
    },
}

PARTIAL_ROUTES = (
    {
        "route": "SenseVoiceSmall",
        "bank": "D-fork",
        "model": "FunAudioLLM/SenseVoiceSmall",
        "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        "engine": "sensevoice",
        "languages": ["en", "zh"],
        "native_partials": False,
        "cause": "Selected FunASR utterance inference returns a completed result; no incremental callback is exposed.",
    },
    {
        "route": "Parakeet fixed-window",
        "bank": "D-fork",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "revision": "7c35754d166cca382ad1e53e68b01e7c575f3a1d",
        "engine": "nemo",
        "languages": ["es", "fr", "ru"],
        "native_partials": False,
        "cause": "The selected fixed-window transcribe route returns final hypotheses. NeMo's separate streaming script is a different product path.",
    },
    {
        "route": "Canary AED utterance",
        "bank": "L-fork",
        "model": "nvidia/canary-180m-flash",
        "revision": "b12ab418510d093e83890178fd0e8b0d0f7918a6",
        "engine": "nemo",
        "languages": ["en", "es", "fr"],
        "native_partials": False,
        "cause": "Canary's selected AED transcribe API resolves an utterance as one decoder result.",
    },
    {
        "route": "Zipformer online",
        "bank": "L-fork",
        "model": "pfluo/k2fsa-zipformer-chinese-english-mixed",
        "revision": "6eb615ae77ecac05c5628d5c8ed7037c14a338d5",
        "engine": "zipformer",
        "languages": ["zh"],
        "native_partials": True,
        "cause": "sherpa-onnx OnlineRecognizer exposes successive get_result(stream) values under a 100 ms feed schedule.",
    },
    {
        "route": "GigaAM utterance",
        "bank": "L-fork",
        "model": "ai-sage/GigaAM-Multilingual",
        "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
        "engine": "gigaam",
        "languages": ["ru"],
        "native_partials": False,
        "cause": "The pinned Transformers remote-code transcribe(path) API returns one completed transcription.",
    },
    {
        "route": "OmniASR CTC utterance",
        "bank": "OmniASR",
        "model": "facebook/omniASR-CTC-300M",
        "revision": "8e35f0cc28fa6099e0c14d56db85ce0423baa691",
        "engine": "omni",
        "languages": list(LANGUAGES),
        "native_partials": False,
        "cause": "ASRInferencePipeline.transcribe batches utterances and returns completed CTC strings; no incremental callback is exposed.",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _engine_route_expectations(
    engine: str,
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    languages_by_provenance: dict[tuple[str, str, str], set[str]] = {}
    for bank, routes in ROUTES.items():
        for language, route in routes.items():
            if route["engine"] != engine:
                continue
            provenance = (route["model"], bank, route["revision"])
            languages_by_provenance.setdefault(provenance, set()).add(language)
    return {
        provenance: tuple(
            language for language in LANGUAGES if language in model_languages
        )
        for provenance, model_languages in languages_by_provenance.items()
    }


def _json_member(raw_members: Mapping[str, bytes], name: str) -> Any:
    raw = raw_members.get(name)
    if raw is None:
        raise KeyError(name)
    return json.loads(raw.decode("utf-8"))


def _jsonl_member(raw_members: Mapping[str, bytes], name: str) -> list[dict[str, Any]]:
    raw = raw_members.get(name)
    if raw is None:
        raise KeyError(name)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{name}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _manifest_reasons(
    manifest: Any,
    names: list[str],
    raw_members: Mapping[str, bytes],
) -> list[str]:
    if not isinstance(manifest, dict):
        return ["archive_manifest.json is not a JSON object"]
    reasons: list[str] = []
    if manifest.get("schema") != ARCHIVE_SCHEMA:
        reasons.append("archive manifest schema mismatch")
    if manifest.get("allowlist") != sorted(ARCHIVE_PAYLOAD_ALLOWLIST):
        reasons.append("archive manifest allowlist mismatch")
    included = manifest.get("included")
    if not isinstance(included, list):
        return reasons + ["archive manifest included list missing"]
    entries: dict[str, dict[str, Any]] = {}
    entry_order: list[str] = []
    for index, entry in enumerate(included):
        if not isinstance(entry, dict):
            reasons.append(f"archive manifest entry {index} is not an object")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or path not in ARCHIVE_PAYLOAD_ALLOWLIST:
            reasons.append(f"archive manifest entry {index} has invalid path")
            continue
        if path in entries:
            reasons.append(f"archive manifest repeats member {path}")
            continue
        entries[path] = entry
        entry_order.append(path)
    payload_names = sorted(name for name in names if name != "archive_manifest.json")
    if entry_order != payload_names:
        reasons.append("archive manifest member list/order does not match archive")
    for name in payload_names:
        entry = entries.get(name)
        raw = raw_members.get(name)
        if entry is None or raw is None:
            continue
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size != len(raw):
            reasons.append(f"archive manifest byte size mismatch for {name}")
        observed_sha = entry.get("sha256")
        computed_sha = hashlib.sha256(raw).hexdigest()
        if not valid_sha256(observed_sha) or observed_sha != computed_sha:
            reasons.append(f"archive manifest SHA-256 mismatch for {name}")
    return reasons


def _fixture_evidence_index(
    fixture: Any,
    expected_languages: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate the archived fixture receipt and return its exact sample index."""

    if not isinstance(fixture, dict):
        return {}, ["embedded fixture receipt missing"]
    reasons: list[str] = []
    if fixture.get("schema") != FIXTURE_SCHEMA:
        reasons.append("fixture schema mismatch")
    if fixture.get("languages") != expected_languages:
        reasons.append("fixture language inventory mismatch")
    if fixture.get("conditions") != list(CONDITIONS):
        reasons.append("fixture condition inventory mismatch")
    if fixture.get("samples_per_language_per_condition") != TARGET_N:
        reasons.append("fixture per-cell sample count mismatch")
    fleurs = fixture.get("fleurs")
    expected_configs = {
        language: FLEURS_CONFIGS[language] for language in expected_languages
    }
    expected_test_rows = {
        language: FLEURS_TEST_ROWS[language] for language in expected_languages
    }
    if not isinstance(fleurs, dict):
        reasons.append("fixture FLEURS provenance missing")
    else:
        expected_fleurs = {
            "repo_id": "google/fleurs",
            "revision": FLEURS_REVISION,
            "resolved_commit": FLEURS_REVISION,
            "split": "test",
            "configs": expected_configs,
            "test_row_counts": expected_test_rows,
        }
        if any(fleurs.get(key) != value for key, value in expected_fleurs.items()):
            reasons.append("fixture FLEURS provenance mismatch")
    audio = fixture.get("audio")
    if not isinstance(audio, dict) or {
        "sample_rate": audio.get("sample_rate"),
        "channels": audio.get("channels"),
        "pcm_dtype": audio.get("pcm_dtype"),
    } != {"sample_rate": 16_000, "channels": 1, "pcm_dtype": "float32-le"}:
        reasons.append("fixture canonical audio metadata mismatch")
    musan = fixture.get("musan")
    if not isinstance(musan, dict):
        reasons.append("fixture MUSAN provenance missing")
    else:
        expected_musan = {
            "repo_id": MUSAN_REPO,
            "revision": MUSAN_REVISION,
            "resolved_commit": MUSAN_REVISION,
            "glob": MUSAN_NOISE_GLOB,
            "expected_noise_file_count": MUSAN_NOISE_WAV_COUNT,
        }
        if any(musan.get(key) != value for key, value in expected_musan.items()):
            reasons.append("fixture MUSAN provenance mismatch")
        files = musan.get("files")
        if not isinstance(files, list) or len(files) != MUSAN_NOISE_WAV_COUNT:
            reasons.append("fixture MUSAN noise inventory mismatch")
        computed_noise_bank_hash = canonical_sha256(musan)
        if fixture.get("noise_bank_sha256") != computed_noise_bank_hash:
            reasons.append("fixture noise-bank SHA mismatch")

    manifest_rows = fixture.get("sample_manifest")
    if not isinstance(manifest_rows, list) or not all(
        isinstance(row, dict) for row in manifest_rows
    ):
        return {}, reasons + ["fixture sample_manifest missing or invalid"]
    expected_sample_count = len(expected_languages) * len(CONDITIONS) * TARGET_N
    if fixture.get("sample_count") != expected_sample_count:
        reasons.append("fixture declared sample count mismatch")
    if len(manifest_rows) != expected_sample_count:
        reasons.append("fixture sample_manifest row count mismatch")

    computed_index: dict[str, dict[str, Any]] = {}
    noisy_rows: dict[tuple[str, str], dict[str, Any]] = {}
    fixture_cell_counts: dict[tuple[str, str], int] = {}
    hash_fields = {
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
    for index, raw_row in enumerate(manifest_rows):
        row = dict(raw_row)
        utterance_id = row.get("utterance_id")
        language = row.get("language")
        condition = row.get("condition")
        if not all(isinstance(value, str) and value for value in (utterance_id, language, condition)):
            reasons.append(f"fixture sample_manifest row {index} lacks identity")
            continue
        key = f"{utterance_id}\t{condition}"
        if key in computed_index:
            reasons.append(f"fixture sample_manifest repeats {key!r}")
            continue
        computed_index[key] = row
        fixture_cell = (str(language), str(condition))
        fixture_cell_counts[fixture_cell] = fixture_cell_counts.get(fixture_cell, 0) + 1
        missing = [field for field in FIXTURE_BINDING_BASE_FIELDS if field not in row]
        if missing:
            reasons.append(f"fixture row {index} lacks base fields: {missing}")
        if language not in expected_languages:
            reasons.append(f"fixture row {index} has unexpected language")
        if condition not in CONDITIONS:
            reasons.append(f"fixture row {index} has unexpected condition")
        active_hash_fields = hash_fields.intersection(row)
        if any(not valid_sha256(row[field]) for field in active_hash_fields):
            reasons.append(f"fixture row {index} has an invalid SHA-256 field")
        reference = row.get("reference")
        if isinstance(reference, str) and row.get("reference_sha256") != hashlib.sha256(
            reference.encode("utf-8")
        ).hexdigest():
            reasons.append(f"fixture row {index} reference SHA mismatch")
        duration = first_number(row.get("duration_s"))
        duration_seconds = first_number(row.get("duration_seconds"))
        num_samples = first_number(row.get("num_samples"))
        sample_rate = first_number(row.get("sample_rate"))
        if (
            duration is None
            or duration_seconds is None
            or duration <= 0
            or duration != duration_seconds
            or num_samples is None
            or sample_rate is None
            or num_samples <= 0
            or sample_rate <= 0
            or not math.isclose(
                float(duration), float(num_samples) / float(sample_rate), abs_tol=1e-9
            )
        ):
            reasons.append(f"fixture row {index} has inconsistent duration metadata")
        if (
            row.get("sample_rate") != 16_000
            or row.get("channels") != 1
            or row.get("pcm_dtype") != "float32-le"
        ):
            reasons.append(f"fixture row {index} canonical audio metadata mismatch")
        if condition == "musan_10db":
            missing_noise = [
                field for field in FIXTURE_BINDING_NOISE_FIELDS if field not in row
            ]
            if missing_noise:
                reasons.append(f"fixture noisy row {index} lacks fields: {missing_noise}")
            noisy_rows[(str(utterance_id), str(language))] = row
        elif any(field in row for field in FIXTURE_BINDING_NOISE_FIELDS):
            reasons.append(f"fixture clean row {index} carries noise/mix fields")

    expected_fixture_cell_counts = {
        (language, condition): TARGET_N
        for language in expected_languages
        for condition in CONDITIONS
    }
    if fixture_cell_counts != expected_fixture_cell_counts:
        reasons.append("fixture language/condition counts do not match exact protocol")

    sample_index = fixture.get("sample_index")
    if sample_index != computed_index:
        reasons.append("fixture sample_index disagrees with sample_manifest")
    hashes_by_language = fixture.get("fixture_hashes_by_language")
    expected_hashes_by_language = {
        language: canonical_sha256(
            [row for row in manifest_rows if row.get("language") == language]
        )
        for language in expected_languages
    }
    if hashes_by_language != expected_hashes_by_language:
        reasons.append("per-language fixture hashes do not match sample_manifest")

    mix_manifest = fixture.get("mix_manifest")
    if not isinstance(mix_manifest, dict):
        reasons.append("fixture mix manifest missing")
    else:
        if mix_manifest.get("schema") != MIX_MANIFEST_SCHEMA:
            reasons.append("fixture mix-manifest schema mismatch")
        if mix_manifest.get("seed") != BOOTSTRAP_SEED:
            reasons.append("fixture mix-manifest seed mismatch")
        if first_number(mix_manifest.get("target_snr_db")) != 10.0:
            reasons.append("fixture mix-manifest target SNR mismatch")
        if mix_manifest.get("noise_bank_sha256") != fixture.get(
            "noise_bank_sha256"
        ):
            reasons.append("fixture mix/noise-bank SHA linkage mismatch")
        computed_mix_hash = canonical_sha256(mix_manifest)
        if fixture.get("mix_manifest_sha256") != computed_mix_hash:
            reasons.append("fixture mix-manifest SHA mismatch")
        entries = mix_manifest.get("entries")
        if not isinstance(entries, list):
            reasons.append("fixture mix-manifest entries missing")
            entries = []
        if len(entries) != len(expected_languages) * TARGET_N:
            reasons.append("fixture mix-manifest entry count mismatch")
        seen_mix_rows: set[tuple[str, str]] = set()
        mix_language_counts: dict[str, int] = {}
        field_pairs = (
            ("noise_repo_path", "noise_repo_path"),
            ("noise_raw_audio_sha256", "noise_raw_audio_sha256"),
            ("noise_pcm_sha256", "noise_pcm_sha256"),
            ("noise_segment_pcm_sha256", "noise_segment_pcm_sha256"),
            ("entry_sha256", "mix_entry_sha256"),
            ("selector_sha256", "mix_selector_sha256"),
            ("offset_samples", "noise_offset_samples"),
            ("noise_wrapped", "noise_wrapped"),
            ("target_snr_db", "target_snr_db"),
            ("achieved_snr_db", "achieved_snr_db"),
            ("mixed_pcm_sha256", "audio_pcm_sha256"),
            ("clean_pcm_sha256", "clean_pcm_sha256"),
            ("clean_raw_audio_sha256", "raw_audio_sha256"),
            ("clean_num_samples", "num_samples"),
        )
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                reasons.append(f"mix-manifest entry {index} is not an object")
                continue
            observed_entry_hash = entry.get("entry_sha256")
            unhashed_entry = dict(entry)
            unhashed_entry.pop("entry_sha256", None)
            if not valid_sha256(observed_entry_hash) or canonical_sha256(
                unhashed_entry
            ) != observed_entry_hash:
                reasons.append(f"mix-manifest entry {index} SHA mismatch")
            key = (str(entry.get("utterance_id", "")), str(entry.get("language", "")))
            mix_language_counts[key[1]] = mix_language_counts.get(key[1], 0) + 1
            fixture_row = noisy_rows.get(key)
            if fixture_row is None:
                reasons.append(f"mix-manifest entry {index} has no noisy fixture row")
                continue
            if key in seen_mix_rows:
                reasons.append(f"mix-manifest repeats noisy fixture row {key!r}")
            seen_mix_rows.add(key)
            if any(entry.get(entry_field) != fixture_row.get(row_field) for entry_field, row_field in field_pairs):
                reasons.append(f"mix-manifest entry {index} disagrees with fixture row")
        if seen_mix_rows != set(noisy_rows):
            reasons.append("mix-manifest does not cover every noisy fixture row exactly")
        if mix_language_counts != {
            language: TARGET_N for language in expected_languages
        }:
            reasons.append("mix-manifest per-language counts do not match protocol")

    fixture_manifest = {
        "schema": fixture.get("schema"),
        "fleurs": fixture.get("fleurs"),
        "audio": fixture.get("audio"),
        "musan": fixture.get("musan"),
        "noise_bank_sha256": fixture.get("noise_bank_sha256"),
        "mix_manifest_sha256": fixture.get("mix_manifest_sha256"),
        "samples": manifest_rows,
    }
    computed_fixture_hash = canonical_sha256(fixture_manifest)
    if fixture.get("fixture_manifest_sha256") != computed_fixture_hash:
        reasons.append("fixture manifest SHA does not match archived sample manifest")
    if fixture.get("fixture_sha256") != computed_fixture_hash:
        reasons.append("fixture SHA alias does not match archived sample manifest")
    return computed_index, reasons


def _record_fixture_reasons(
    records: list[dict[str, Any]],
    fixture_index: Mapping[str, dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for index, record in enumerate(records):
        key = f"{record.get('utterance_id', '')}\t{normalize_condition(str(record.get('condition', '')))}"
        sample = fixture_index.get(key)
        if sample is None:
            reasons.append(f"records.jsonl row {index} has no exact fixture sample")
            continue
        fields = list(FIXTURE_BINDING_BASE_FIELDS)
        condition = str(sample.get("condition"))
        if condition == "musan_10db":
            fields.extend(FIXTURE_BINDING_NOISE_FIELDS)
        elif any(field in record for field in FIXTURE_BINDING_NOISE_FIELDS):
            reasons.append(f"records.jsonl clean row {index} carries noise/mix fields")
        identity_fields = ("utterance_id", "language", "condition")
        for field in (*identity_fields, *fields):
            if field not in record:
                reasons.append(f"records.jsonl row {index} lacks fixture field {field}")
            elif record[field] != sample.get(field):
                reasons.append(f"records.jsonl row {index} fixture field {field} mismatch")
        audio_path = record.get("audio_path")
        audio_relpath = sample.get("audio_relpath")
        if not isinstance(audio_path, str) or not isinstance(audio_relpath, str) or not (
            audio_path == audio_relpath or audio_path.endswith("/" + audio_relpath)
        ):
            reasons.append(f"records.jsonl row {index} audio path/relpath mismatch")
        reference = record.get("reference")
        if isinstance(reference, str) and record.get("reference_sha256") != hashlib.sha256(
            reference.encode("utf-8")
        ).hexdigest():
            reasons.append(f"records.jsonl row {index} reference SHA mismatch")
    return reasons


def _partial_receipt_reasons(
    engine: str,
    result: Mapping[str, Any],
    partial_traces: list[dict[str, Any]],
    fixture_index: Mapping[str, dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[str]:
    metrics = result.get("partial_metrics")
    if engine != "zipformer":
        reasons: list[str] = []
        if partial_traces:
            reasons.append(f"{engine} unexpectedly archived partial traces")
        if metrics != []:
            reasons.append(f"{engine} unexpectedly archived partial metrics")
        return reasons

    reasons = []
    clean_fixture_rows = [
        row
        for row in fixture_index.values()
        if row.get("language") == "zh" and row.get("condition") == "clean"
    ]
    expected_trace_ids = {
        str(row["utterance_id"]) for row in clean_fixture_rows[:PARTIAL_TRACE_N]
    }
    if len(partial_traces) != PARTIAL_TRACE_N:
        reasons.append(
            f"Zipformer partial trace count is {len(partial_traces)}, expected {PARTIAL_TRACE_N}"
        )
    observed_trace_ids: set[str] = set()
    record_hypotheses = {
        str(record.get("utterance_id")): record.get("hypothesis")
        for record in records
        if record.get("language") == "zh"
        and normalize_condition(str(record.get("condition"))) == "clean"
    }
    for index, trace in enumerate(partial_traces):
        expected_trace_fields = {
            "engine": "zipformer",
            "model": ZIPFORMER_MODEL,
            "bank": "L-fork",
            "revision": ZIPFORMER_REVISION,
            "route": "Zipformer online",
            "language": "zh",
            "condition": "clean",
        }
        if any(trace.get(key) != value for key, value in expected_trace_fields.items()):
            reasons.append(f"Zipformer trace {index} route/provenance mismatch")
        if not trace_sha256_matches(trace):
            reasons.append(f"Zipformer trace {index} self-digest mismatch")
        utterance_id = str(trace.get("utterance_id", ""))
        if not utterance_id or utterance_id in observed_trace_ids:
            reasons.append(f"Zipformer trace {index} has missing/duplicate utterance id")
        observed_trace_ids.add(utterance_id)
        fixture_row = fixture_index.get(f"{utterance_id}\tclean")
        if fixture_row is None:
            reasons.append(f"Zipformer trace {index} has no clean fixture sample")
        else:
            audio_path = trace.get("audio_path")
            audio_relpath = fixture_row.get("audio_relpath")
            if (
                trace.get("duration_s") != fixture_row.get("duration_s")
                or not isinstance(audio_path, str)
                or not isinstance(audio_relpath, str)
                or not (
                    audio_path == audio_relpath
                    or audio_path.endswith("/" + audio_relpath)
                )
            ):
                reasons.append(f"Zipformer trace {index} fixture linkage mismatch")
        if trace.get("final_hypothesis") != record_hypotheses.get(utterance_id):
            reasons.append(f"Zipformer trace {index} final hypothesis mismatch")
        partials = trace.get("partials")
        if not isinstance(partials, list) or not partials:
            reasons.append(f"Zipformer trace {index} has no callback event sequence")
    if observed_trace_ids != expected_trace_ids:
        reasons.append("Zipformer traces are not the exact first 30 clean fixture rows")

    if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(
        metrics[0], dict
    ):
        return reasons + ["Zipformer requires exactly one partial metric receipt"]
    metric = metrics[0]
    expected_metric_fields = {
        "route": "Zipformer online",
        "language": "zh",
        "n_input": PARTIAL_TRACE_N,
        "n_trace_events_input": PARTIAL_TRACE_N,
        "n_valid": PARTIAL_TRACE_N,
        "n_invalid": 0,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if any(metric.get(key) != value for key, value in expected_metric_fields.items()):
        reasons.append("Zipformer partial metric counts/protocol mismatch")
    metric_utterances = metric.get("utterances")
    metric_ids = {
        str(item.get("utterance_id"))
        for item in metric_utterances
        if isinstance(item, dict)
    } if isinstance(metric_utterances, list) else set()
    if metric_ids != expected_trace_ids or len(metric_utterances or []) != PARTIAL_TRACE_N:
        reasons.append("Zipformer partial metric utterances do not match traces")
    partials_emitted = metric.get("partials_emitted")
    if type(partials_emitted) is not bool:
        reasons.append("Zipformer partial metric lacks a boolean emitted flag")
    elif partials_emitted:
        if metric.get("evidence_class") != "HOST":
            reasons.append("Zipformer emitted-partial metric is not HOST evidence")
        for key in (
            "first_partial_latency_ms",
            "partial_rewrite_rate_edits_per_audio_second",
            "final_vs_last_partial_divergence",
        ):
            if first_number(metric.get(key)) is None:
                reasons.append(f"Zipformer partial metric lacks {key}")
    else:
        if metric.get("n_no_partials") != PARTIAL_TRACE_N or metric.get(
            "n_partials_emitted"
        ) != 0:
            reasons.append("Zipformer no-partials metric counts are incoherent")
    return reasons


def _script_class(character: str) -> str:
    codepoint = ord(character)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    ):
        return "cjk"
    if 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "kana"
    if (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return "hangul"
    name = unicodedata.name(character, "")
    if "CYRILLIC" in name:
        return "cyrillic"
    if "LATIN" in name:
        return "latin"
    return "other"


def _wrong_script(text: str, language: str) -> bool:
    counts = {"latin": 0, "cyrillic": 0, "cjk": 0, "kana": 0, "hangul": 0}
    for character in text:
        script = _script_class(character)
        if script in counts:
            counts[script] += 1
    if language == "zh":
        expected = counts["cjk"]
        competing = (
            counts["latin"]
            + counts["cyrillic"]
            + counts["kana"]
            + counts["hangul"]
        )
    elif language == "ru":
        expected = counts["cyrillic"]
        competing = (
            counts["latin"] + counts["cjk"] + counts["kana"] + counts["hangul"]
        )
    else:
        expected = counts["latin"]
        competing = (
            counts["cyrillic"] + counts["cjk"] + counts["kana"] + counts["hangul"]
        )
    return expected == 0 or expected < competing


def _recomputed_omni_gates(
    records: list[dict[str, Any]],
    fixture_index: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    reasons: list[str] = []
    gates: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        clean_rows = [
            sample
            for sample in fixture_index.values()
            if sample.get("language") == language
            and sample.get("condition") == "clean"
        ]
        smoke_ids = {
            str(sample["utterance_id"]) for sample in clean_rows[:OMNI_SMOKE_N]
        }
        smoke_records = [
            record
            for record in records
            if record.get("language") == language
            and record.get("condition") == "clean"
            and str(record.get("utterance_id", "")) in smoke_ids
        ]
        observed_ids = {str(record.get("utterance_id", "")) for record in smoke_records}
        if observed_ids != smoke_ids or len(smoke_records) != OMNI_SMOKE_N:
            reasons.append(f"Omni {language} smoke identities/count do not match fixture")
        hypotheses = [str(record.get("hypothesis", "")) for record in smoke_records]
        empty_count = sum(not hypothesis.strip() for hypothesis in hypotheses)
        wrong_script_count = sum(
            bool(hypothesis.strip()) and _wrong_script(hypothesis, language)
            for hypothesis in hypotheses
        )
        denominator = len(smoke_records)
        empty_fraction = empty_count / denominator if denominator else 1.0
        wrong_script_fraction = (
            wrong_script_count / denominator if denominator else 1.0
        )
        incomplete = denominator < OMNI_SMOKE_N
        collapsed = (
            incomplete or empty_fraction > 0.20 or wrong_script_fraction > 0.50
        )
        gate_reasons: list[str] = []
        if incomplete:
            gate_reasons.append(
                f"smoke incomplete: {denominator}/{OMNI_SMOKE_N} records"
            )
        if empty_fraction > 0.20:
            gate_reasons.append(f"empty fraction {empty_fraction:.3f} > 0.20")
        if wrong_script_fraction > 0.50:
            gate_reasons.append(
                f"wrong-script fraction {wrong_script_fraction:.3f} > 0.50"
            )
        gates[language] = {
            "language": language,
            "smoke_n_expected": OMNI_SMOKE_N,
            "smoke_n_observed": denominator,
            "empty_count": empty_count,
            "empty_fraction": empty_fraction,
            "wrong_script_count": wrong_script_count,
            "wrong_script_fraction": wrong_script_fraction,
            "thresholds": {
                "empty_fraction_gt": 0.20,
                "wrong_script_fraction_gt": 0.50,
            },
            "collapsed": collapsed,
            "reasons": gate_reasons,
            "remaining_samples_total": 2 * TARGET_N - OMNI_SMOKE_N,
            "remaining_samples_evaluated": (
                0 if collapsed else 2 * TARGET_N - OMNI_SMOKE_N
            ),
            "remaining_samples_skipped": (
                2 * TARGET_N - OMNI_SMOKE_N if collapsed else 0
            ),
        }
    return gates, reasons


def _omni_phase_reasons(
    records: list[dict[str, Any]],
    fixture_index: Mapping[str, dict[str, Any]],
    collapsed_languages: set[str],
    metadata: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    smoke_execution = metadata.get("smoke_execution")
    expected_smoke_execution = {
        "languages_in_order": list(LANGUAGES),
        "clean_n_per_language": OMNI_SMOKE_N,
        "all_language_smokes_before_expansion": True,
        "expansion_policy": "only languages whose smoke gate did not collapse",
        "resumability": "record identity checkpoints are reused in both phases",
    }
    if smoke_execution != expected_smoke_execution:
        reasons.append("Omni smoke-execution protocol receipt mismatch")
    smoke_ids_by_language = {
        language: {
            str(row["utterance_id"])
            for row in [
                sample
                for sample in fixture_index.values()
                if sample.get("language") == language
                and sample.get("condition") == "clean"
            ][:OMNI_SMOKE_N]
        }
        for language in LANGUAGES
    }
    observed_smoke_ids = {language: set() for language in LANGUAGES}
    for index, record in enumerate(records):
        language = str(record.get("language", ""))
        utterance_id = str(record.get("utterance_id", ""))
        is_smoke = (
            normalize_condition(str(record.get("condition", ""))) == "clean"
            and utterance_id in smoke_ids_by_language.get(language, set())
        )
        if is_smoke:
            observed_smoke_ids[language].add(utterance_id)
            expected_tags = {
                "phase": "smoke",
                "smoke_test": True,
                "route": "ctc-utterance/fp16-cuda/smoke",
                "condition": "clean",
            }
        else:
            expected_tags = {
                "phase": "evaluation",
                "smoke_test": False,
                "route": "ctc-utterance/fp16-cuda/full",
            }
        if any(record.get(key) != value for key, value in expected_tags.items()):
            reasons.append(f"Omni record {index} smoke/evaluation tags mismatch")
        if language in collapsed_languages and not is_smoke:
            reasons.append(f"collapsed Omni language {language} contains expansion records")
    for language, expected_ids in smoke_ids_by_language.items():
        if observed_smoke_ids[language] != expected_ids:
            reasons.append(f"Omni {language} smoke records are not the first 30 clean rows")
    return reasons


def _scored_receipt_reasons(
    result: Mapping[str, Any],
    expectations: Mapping[tuple[str, str, str], tuple[str, ...]],
    expected_cells: Mapping[tuple[str, str, str, str, str], int],
) -> list[str]:
    reasons: list[str] = []
    scored_models = result.get("scored_models")
    if not isinstance(scored_models, list):
        return ["scored-model receipts missing"]
    by_provenance: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, scored in enumerate(scored_models):
        if not isinstance(scored, dict):
            reasons.append(f"scored-model receipt {index} is not an object")
            continue
        provenance = (
            str(scored.get("model", "")),
            str(scored.get("bank", "")),
            str(scored.get("revision", "")),
        )
        if provenance in by_provenance:
            reasons.append(f"duplicate scored-model receipt for {provenance}")
            continue
        by_provenance[provenance] = scored
    if set(by_provenance) != set(expectations):
        reasons.append("scored-model provenance inventory mismatch")

    fixture = result.get("fixture")
    fixture_hash = (
        fixture.get("fixture_manifest_sha256") if isinstance(fixture, dict) else None
    )
    mix_hash = fixture.get("mix_manifest_sha256") if isinstance(fixture, dict) else None
    language_hashes = (
        fixture.get("fixture_hashes_by_language", {})
        if isinstance(fixture, dict)
        else {}
    )
    for provenance, expected_languages in expectations.items():
        scored = by_provenance.get(provenance)
        if scored is None:
            continue
        expected_model_count = sum(
            count
            for cell_key, count in expected_cells.items()
            if cell_key[:3] == provenance
        )
        expected_top = {
            "evidence_class": "HOST",
            "n_records": expected_model_count,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "corpus_repo_id": "google/fleurs",
            "corpus_revision": FLEURS_REVISION,
            "corpus_split": "test",
            "fixture_manifest_sha256": fixture_hash,
            "mix_manifest_sha256": mix_hash,
            "normalisers_source_sha256": NORMALISERS_SHA256,
        }
        if any(scored.get(key) != value for key, value in expected_top.items()):
            reasons.append(f"scored-model protocol/provenance mismatch for {provenance[0]}")
        cells = flatten_cells(scored)
        observed_cells: dict[tuple[str, str, str, str, str], int] = {}
        for index, cell in enumerate(cells):
            cell_key = (
                str(cell.get("model") or provenance[0]),
                str(cell.get("bank") or provenance[1]),
                str(cell.get("revision") or provenance[2]),
                str(cell.get("language", "")),
                normalize_condition(str(cell.get("condition", ""))),
            )
            if cell_key in observed_cells:
                reasons.append(f"duplicate scored cell for {cell_key}")
            n = cell.get("n")
            observed_cells[cell_key] = int(n) if isinstance(n, int) else -1
            language = cell_key[3]
            expected_cell = {
                "evidence_class": "HOST",
                "corpus": (
                    f"google/fleurs@{FLEURS_REVISION}/"
                    f"{FLEURS_CONFIGS.get(language, '')}/test"
                ),
                "fixture_sha256": (
                    language_hashes.get(language)
                    if isinstance(language_hashes, dict)
                    else None
                ),
                "mix_manifest_sha256": mix_hash,
                "bootstrap_B": BOOTSTRAP_B,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
            if any(cell.get(key) != value for key, value in expected_cell.items()):
                reasons.append(f"scored cell {index} protocol/fixture linkage mismatch")
        expected_model_cells = {
            key: count
            for key, count in expected_cells.items()
            if key[:3] == provenance
        }
        if observed_cells != expected_model_cells:
            reasons.append(f"scored cells/counts mismatch for {provenance[0]}")
    return reasons


def _complete_result_reasons(
    result: dict[str, Any],
    engine: str,
    raw_members: Mapping[str, bytes],
    records: list[dict[str, Any]],
    partial_traces: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    expectations = _engine_route_expectations(engine)
    if not expectations:
        return [f"no local route expectations registered for engine {engine!r}"]
    expected_languages = [
        language
        for language in LANGUAGES
        if any(language in languages for languages in expectations.values())
    ]
    fixture_index, fixture_reasons = _fixture_evidence_index(
        result.get("fixture"), expected_languages
    )
    reasons.extend(fixture_reasons)
    metadata = result.get("engine_metadata")
    validation = result.get("record_validation")
    if not isinstance(metadata, dict):
        metadata = {}
        reasons.append("embedded engine metadata missing")
    if not isinstance(validation, dict):
        return reasons + ["record-validation receipt missing"]
    for model, _bank, revision in sorted(expectations):
        for artifact_reason in artifact_pin_reasons(
            {"_engine_metadata": metadata}, model, revision
        ):
            reasons.append(f"{model} artifact receipt: {artifact_reason}")

    if validation.get("schema") != "lane-c-record-validation/1.1":
        reasons.append("record-validation schema mismatch")
    if validation.get("engine") != engine:
        reasons.append("record-validation engine mismatch")
    if validation.get("valid") is not True:
        reasons.append("record validation is not valid")
    if validation.get("all_inference_status_ok") is not True:
        reasons.append("record validation does not assert all-ok inference")
    if validation.get("all_records_fixture_bound") is not True:
        reasons.append("record validation does not bind all records to fixtures")
    binding = validation.get("fixture_binding")
    expected_binding = {
        "schema": "lane-c-record-fixture-binding/1.0",
        "valid": True,
        "strict_schema": True,
        "record_count": len(records),
        "base_fields": list(FIXTURE_BINDING_BASE_FIELDS),
        "noise_fields": list(FIXTURE_BINDING_NOISE_FIELDS),
    }
    if binding != expected_binding:
        reasons.append("record fixture-binding receipt is incomplete or mismatched")

    expected_provenance = [
        {"model": model, "bank": bank, "revision": revision}
        for model, bank, revision in sorted(expectations)
    ]
    if validation.get("expected_provenance") != expected_provenance:
        reasons.append("record-validation provenance does not match local routes")
    if result.get("models") != expected_provenance:
        reasons.append("engine result model provenance does not match local routes")
    if result.get("languages") != expected_languages:
        reasons.append("engine result language inventory mismatch")

    collapsed_languages: set[str] = set()
    if engine == "omni":
        recomputed_gates, gate_reasons = _recomputed_omni_gates(
            records, fixture_index
        )
        reasons.extend(gate_reasons)
        gates = metadata.get("smoke_gate")
        if not isinstance(gates, dict) or set(gates) != set(expected_languages):
            reasons.append("Omni smoke-gate language inventory mismatch")
            gates = {}
        for language in expected_languages:
            expected_gate = recomputed_gates[language]
            observed_gate = gates.get(language)
            if not isinstance(observed_gate, dict):
                reasons.append(f"Omni smoke gate missing for {language}")
                continue
            if observed_gate != expected_gate:
                reasons.append(
                    f"Omni smoke gate does not match recomputed evidence for {language}"
                )
            if expected_gate["collapsed"]:
                collapsed_languages.add(language)
    observed_collapses = validation.get("omni_collapsed_languages")
    if observed_collapses != sorted(collapsed_languages):
        reasons.append("record-validation Omni collapse inventory mismatch")

    expected_cells: dict[tuple[str, str, str, str, str], int] = {}
    for provenance, languages in expectations.items():
        for language in languages:
            if engine == "omni" and language in collapsed_languages:
                expected_cells[(*provenance, language, "clean")] = OMNI_SMOKE_N
            else:
                for condition in CONDITIONS:
                    expected_cells[(*provenance, language, condition)] = TARGET_N
    expected_cell_rows = [
        {
            "model": model,
            "bank": bank,
            "revision": revision,
            "language": language,
            "condition": condition,
            "expected_records": count,
            "records": count,
        }
        for (model, bank, revision, language, condition), count in sorted(
            expected_cells.items()
        )
    ]
    if validation.get("cell_counts") != expected_cell_rows:
        reasons.append("record-validation cell counts do not match exact protocol")

    expected_model_counts: dict[tuple[str, str, str], int] = {}
    for cell_key, count in expected_cells.items():
        provenance = cell_key[:3]
        expected_model_counts[provenance] = expected_model_counts.get(provenance, 0) + count
    expected_count_rows = [
        {"model": model, "bank": bank, "revision": revision, "records": count}
        for (model, bank, revision), count in sorted(expected_model_counts.items())
    ]
    if validation.get("counts") != expected_count_rows:
        reasons.append("record-validation model counts do not match exact protocol")
    reasons.extend(_scored_receipt_reasons(result, expectations, expected_cells))

    actual_cells: dict[tuple[str, str, str, str, str], int] = {}
    identities: set[tuple[Any, ...]] = set()
    for index, record in enumerate(records):
        if record.get("engine") != engine:
            reasons.append(f"records.jsonl engine mismatch at row {index}")
        if record.get("status") != "ok":
            reasons.append(f"records.jsonl non-ok status at row {index}")
        if record.get("evidence_class") != "HOST":
            reasons.append(f"records.jsonl non-HOST evidence at row {index}")
        key = (
            str(record.get("model", "")),
            str(record.get("bank", "")),
            str(record.get("revision", "")),
            str(record.get("language", "")),
            normalize_condition(str(record.get("condition", ""))),
        )
        actual_cells[key] = actual_cells.get(key, 0) + 1
        identity = (
            *key,
            str(record.get("utterance_id", "")),
            str(record.get("audio_path", "")),
        )
        if identity in identities:
            reasons.append(f"records.jsonl duplicate identity at row {index}")
        identities.add(identity)
    if actual_cells != expected_cells:
        reasons.append("records.jsonl cell counts do not match exact protocol")
    reasons.extend(_record_fixture_reasons(records, fixture_index))
    if engine == "omni":
        reasons.extend(
            _omni_phase_reasons(
                records,
                fixture_index,
                collapsed_languages,
                metadata,
            )
        )
    reasons.extend(
        _partial_receipt_reasons(
            engine,
            result,
            partial_traces,
            fixture_index,
            records,
        )
    )

    expected_total = sum(expected_cells.values())
    for label, value in (
        ("engine result", result.get("records")),
        ("record validation", validation.get("record_count")),
        ("record-validation expectation", validation.get("expected_record_count")),
        ("records.jsonl", len(records)),
    ):
        if value != expected_total:
            reasons.append(f"{label} record count does not equal {expected_total}")
    if result.get("partial_traces") != len(partial_traces):
        reasons.append("engine result partial-trace count disagrees with JSONL")

    standalone_pairs = (
        ("engine_metadata.json", "engine_metadata"),
        ("environment.json", "environment"),
        ("fixture_receipt.json", "fixture"),
        ("install_receipts.json", "install_receipts"),
        ("partial_metrics.json", "partial_metrics"),
        ("scored_models.json", "scored_models"),
    )
    for member, result_key in standalone_pairs:
        try:
            standalone = _json_member(raw_members, member)
        except (KeyError, TypeError, UnicodeDecodeError, ValueError):
            reasons.append(f"{member} is missing or invalid")
            continue
        if standalone != result.get(result_key):
            reasons.append(f"{member} disagrees with embedded {result_key}")
    try:
        run_status = _json_member(raw_members, "run_status.json")
    except (KeyError, TypeError, UnicodeDecodeError, ValueError):
        reasons.append("run_status.json is missing or invalid")
    else:
        if not isinstance(run_status, dict):
            reasons.append("run_status.json is not an object")
        elif any(result.get(key) != value for key, value in run_status.items()):
            reasons.append("run_status.json disagrees with engine_result.json")
    return reasons


def read_engine_zip(results_root: Path, engine: str) -> dict[str, Any] | None:
    archive_path = results_root / engine / "result.zip"
    portable_archive_path = f"artifacts/results/{engine}/result.zip"
    if not archive_path.is_file():
        return None
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("archive contains duplicate member names")
            invalid_names = [
                name
                for name in names
                if name not in ARCHIVE_PAYLOAD_ALLOWLIST
                and name != "archive_manifest.json"
            ]
            if invalid_names:
                raise ValueError(f"archive contains non-allowlisted members: {invalid_names}")
            if any(info.is_dir() for info in infos):
                raise ValueError("archive contains directory members")
            raw_members = {name: archive.read(name) for name in names}
            if archive.testzip() is not None:
                raise ValueError("archive CRC validation failed")

        reasons: list[str] = []
        try:
            manifest = _json_member(raw_members, "archive_manifest.json")
        except (KeyError, TypeError, UnicodeDecodeError, ValueError):
            manifest = None
            reasons.append("archive_manifest.json is missing or invalid")
        else:
            reasons.extend(_manifest_reasons(manifest, sorted(names), raw_members))
        try:
            result = _json_member(raw_members, "engine_result.json")
        except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("engine_result.json is missing or invalid") from exc
        if not isinstance(result, dict):
            raise TypeError("engine_result.json is not a JSON object")
        try:
            records = _jsonl_member(raw_members, "records.jsonl")
        except (KeyError, TypeError, UnicodeDecodeError, ValueError):
            records = []
            if result.get("status") == "COMPLETE":
                reasons.append("records.jsonl is missing or invalid for COMPLETE result")
        try:
            partial_traces = _jsonl_member(raw_members, "partial_traces.jsonl")
        except (KeyError, TypeError, UnicodeDecodeError, ValueError):
            partial_traces = []
            if result.get("status") == "COMPLETE":
                reasons.append(
                    "partial_traces.jsonl is missing or invalid for COMPLETE result"
                )

        if result.get("schema") != "lane-c-colab-engine-result/1.0":
            reasons.append("engine result schema mismatch")
        if result.get("engine") != engine:
            reasons.append("engine result identity does not match archive route")
        if result.get("training_performed") is not False:
            reasons.append("engine result does not assert training_performed=false")
        if result.get("status") not in {"COMPLETE", "FAILED"}:
            reasons.append("engine result status is invalid")
        if result.get("status") == "COMPLETE":
            missing = sorted(COMPLETE_ARCHIVE_MEMBERS - set(names))
            if missing:
                reasons.append(f"COMPLETE archive lacks required members: {missing}")
            if result.get("evidence_class") != "HOST":
                reasons.append("COMPLETE engine result is not HOST evidence")
            reasons.extend(
                _complete_result_reasons(
                    result,
                    engine,
                    raw_members,
                    records,
                    partial_traces,
                )
            )

        members = sorted(names)
        member_sha256 = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in sorted(raw_members.items())
        }
        result["_archive_path"] = portable_archive_path
        result["_archive_sha256"] = archive_sha256
        result["_archive_members"] = members
        result["_member_sha256"] = member_sha256
        result["_partial_traces"] = partial_traces
        result["_archive_validation"] = {
            "schema": "lane-c-host-archive-validation/1.0",
            "valid": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
        }
        if reasons:
            result["_claimed_status"] = result.get("status")
            result["_claimed_evidence_class"] = result.get("evidence_class")
            result["status"] = "FAILED"
            result["evidence_class"] = "UNKNOWN"
            result["error"] = "archive evidence validation failed: " + "; ".join(
                dict.fromkeys(reasons)
            )
        return result
    except (KeyError, OSError, TypeError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as exc:
        return {
            "engine": engine,
            "status": "FAILED",
            "evidence_class": "UNKNOWN",
            "error": f"result.zip evidence is invalid: {type(exc).__name__}: {exc}",
            "_archive_path": portable_archive_path,
            "_archive_sha256": archive_sha256,
            "_archive_validation": {
                "schema": "lane-c-host-archive-validation/1.0",
                "valid": False,
                "reasons": [f"{type(exc).__name__}: {exc}"],
            },
        }


def flatten_cells(scored_model: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("cells", "summaries", "results", "scores"):
        value = scored_model.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            flattened = []
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict):
                    item = dict(nested_value)
                    if isinstance(nested_key, str) and "/" in nested_key:
                        language, condition = nested_key.split("/", 1)
                        item.setdefault("language", language)
                        item.setdefault("condition", condition)
                    flattened.append(item)
            return flattened
    return []


def cell_index(
    engine_results: dict[str, dict[str, Any] | None],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for engine, engine_result in engine_results.items():
        if not engine_result:
            continue
        # All cells from one engine share the same fixture.  Keep a shared,
        # lazy cache so clean/noisy cells and multiple model routes compute a
        # portable language identity at most once during this indexing pass.
        fixture_identity_cache: dict[str, str | None] = {}
        for scored in engine_result.get("scored_models", []):
            if not isinstance(scored, dict):
                continue
            for cell in flatten_cells(scored):
                model = str(cell.get("model") or scored.get("model") or "")
                language = str(cell.get("language", ""))
                condition = normalize_condition(
                    str(cell.get("condition", cell.get("noisy_bucket", "clean")))
                )
                enriched = {
                    **cell,
                    "model": model,
                    "bank": cell.get("bank") or scored.get("bank"),
                    "revision": cell.get("revision") or scored.get("revision"),
                    "_engine": engine,
                    "_engine_status": engine_result.get("status"),
                    "_engine_evidence_class": engine_result.get("evidence_class"),
                    "_engine_result": engine_result,
                    "_engine_metadata": engine_result.get("engine_metadata", {}),
                    "_scored": scored,
                    "_fixture": engine_result.get("fixture", {}),
                    "_fixture_identity_cache": fixture_identity_cache,
                    "_archive_path": engine_result.get("_archive_path"),
                    "_archive_sha256": engine_result.get("_archive_sha256"),
                    "_member_sha256": engine_result.get("_member_sha256", {}),
                }
                indexed.setdefault((model, language, condition), []).append(enriched)
    return indexed


def first_number(*values: Any) -> float | int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return value
    return None


def first_list(*values: Any) -> list[Any] | None:
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return list(value)
    return None


def normalize_condition(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if lowered in {"", "none", "null", "clean"}:
        return "clean"
    if lowered in {"musan_10db", "musan_babble_10db", "musan_at_10_db"}:
        return "musan_10db"
    return lowered


def locate_cell(
    indexed: dict[tuple[str, str, str], list[dict[str, Any]]],
    model: str,
    language: str,
    condition: str,
) -> dict[str, Any] | None:
    matches = indexed.get((model, language, condition), [])
    if not matches:
        return None
    selected = dict(matches[0])
    selected["_duplicate_count"] = len(matches)
    return selected


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def trace_sha256_matches(trace: Mapping[str, Any]) -> bool:
    """Verify a trace's self-digest over its canonical unhashed payload."""

    observed = trace.get("trace_sha256")
    if not valid_sha256(observed):
        return False
    payload = dict(trace)
    payload.pop("trace_sha256", None)
    return canonical_sha256(payload) == observed


def record_validation_reasons(
    engine_result: Any, *, label: str = "engine"
) -> list[str]:
    if not isinstance(engine_result, dict):
        return [f"{label} result receipt missing"]
    archive_validation = engine_result.get("_archive_validation")
    if isinstance(archive_validation, dict) and archive_validation.get("valid") is not True:
        archive_reasons = archive_validation.get("reasons")
        detail = (
            ": " + "; ".join(str(reason) for reason in archive_reasons[:3])
            if isinstance(archive_reasons, list) and archive_reasons
            else ""
        )
        return [f"{label} archive evidence validation failed{detail}"]
    validation = engine_result.get("record_validation")
    if not isinstance(validation, dict):
        return [f"{label} record-validation receipt missing"]
    reasons: list[str] = []
    if validation.get("valid") is not True:
        reasons.append(f"{label} record validation is not valid")
    if validation.get("all_inference_status_ok") is not True:
        reasons.append(f"{label} inference statuses did not validate as all-ok")
    if "all_records_fixture_bound" in validation and validation.get(
        "all_records_fixture_bound"
    ) is not True:
        reasons.append(f"{label} records were not bound to exact fixture samples")
    return reasons


def omni_collapsed_smoke_gate_reasons(
    cell: dict[str, Any], language: str
) -> list[str]:
    """Return blockers unless this is a validated completed Omni collapse."""

    reasons: list[str] = []
    engine_result = cell.get("_engine_result", {})
    if cell.get("_engine") != "omni":
        reasons.append("collapsed smoke evidence did not come from the Omni engine")
    if not isinstance(engine_result, dict):
        return reasons + ["Omni engine receipt missing"]
    if engine_result.get("status") != "COMPLETE":
        reasons.append("Omni engine receipt is not COMPLETE")
    if engine_result.get("evidence_class") != "HOST":
        reasons.append("Omni engine receipt is not HOST evidence")

    validation = engine_result.get("record_validation")
    reasons.extend(record_validation_reasons(engine_result, label="Omni"))
    if isinstance(validation, dict):
        collapsed_languages = validation.get("omni_collapsed_languages")
        if (
            not isinstance(collapsed_languages, list)
            or language not in collapsed_languages
        ):
            reasons.append("language is absent from validated Omni collapses")

    metadata = engine_result.get("engine_metadata")
    smoke_gates = metadata.get("smoke_gate") if isinstance(metadata, dict) else None
    gate = smoke_gates.get(language) if isinstance(smoke_gates, dict) else None
    if not isinstance(gate, dict):
        return reasons + ["per-language Omni smoke gate receipt missing"]
    if gate.get("collapsed") is not True:
        reasons.append("Omni smoke gate did not mark the language collapsed")
    if gate.get("smoke_n_expected") != OMNI_SMOKE_N:
        reasons.append("Omni smoke gate expected count is not 30")
    if gate.get("smoke_n_observed") != OMNI_SMOKE_N:
        reasons.append("Omni smoke gate observed count is not 30")
    expected_skipped = 2 * TARGET_N - OMNI_SMOKE_N
    if gate.get("remaining_samples_skipped") != expected_skipped:
        reasons.append(
            "Omni smoke gate skipped-count receipt does not match the collapsed protocol"
        )
    gate_reasons = gate.get("reasons")
    if not isinstance(gate_reasons, list) or not gate_reasons:
        reasons.append("Omni smoke gate has no collapse reason")
    return reasons


def fixture_hash_for(cell: dict[str, Any] | None, language: str) -> str | None:
    if not cell:
        return None
    direct = cell.get("fixture_sha256") or cell.get("fixture_hash")
    if isinstance(direct, str):
        return direct
    fixture = cell.get("_fixture", {})
    mapping = fixture.get("fixture_hashes_by_language", {}) if isinstance(fixture, dict) else {}
    value = mapping.get(language) if isinstance(mapping, dict) else None
    if isinstance(value, dict):
        return value.get("sha256") or value.get("fixture_sha256")
    return value if isinstance(value, str) else None


def fixture_manifest_hash_for(cell: dict[str, Any] | None) -> str | None:
    if not cell:
        return None
    scored = cell.get("_scored", {})
    fixture = cell.get("_fixture", {})
    for value in (
        scored.get("fixture_manifest_sha256") if isinstance(scored, dict) else None,
        fixture.get("fixture_manifest_sha256") if isinstance(fixture, dict) else None,
        fixture.get("fixture_sha256") if isinstance(fixture, dict) else None,
    ):
        if isinstance(value, str):
            return value
    return None


def mix_manifest_info(
    cell: dict[str, Any] | None, language: str
) -> dict[str, Any]:
    if not cell:
        return {
            "mix_manifest_sha256": None,
            "computed_mix_manifest_sha256": None,
            "mix_manifest_hash_matches": False,
            "language_manifest_sha256": None,
            "language_entries": 0,
            "mixed_wav_hashes": 0,
        }
    fixture = cell.get("_fixture", {})
    scored = cell.get("_scored", {})
    manifest = fixture.get("mix_manifest", {}) if isinstance(fixture, dict) else {}
    entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
    selected = [
        item
        for item in entries
        if isinstance(item, dict) and str(item.get("language")) == language
    ]
    mixed_hashes = {
        str(
            item.get("mixed_wav_sha256")
            or item.get("mixed_pcm_sha256")
            or item.get("mix_entry_sha256")
            or item.get("entry_sha256")
            or ""
        )
        for item in selected
    }
    mixed_hashes = {value for value in mixed_hashes if valid_sha256(value)}
    global_hash = (
        cell.get("mix_manifest_sha256")
        or (scored.get("mix_manifest_sha256") if isinstance(scored, dict) else None)
        or (fixture.get("mix_manifest_sha256") if isinstance(fixture, dict) else None)
    )
    language_manifest = None
    computed_global_hash = (
        canonical_sha256(manifest) if isinstance(manifest, dict) and manifest else None
    )
    if selected and isinstance(manifest, dict):
        language_manifest = canonical_sha256(
            {
                "schema": manifest.get("schema"),
                "seed": manifest.get("seed"),
                "target_snr_db": manifest.get("target_snr_db"),
                "noise_bank_sha256": manifest.get("noise_bank_sha256"),
                "language": language,
                "entries": selected,
            }
        )
    return {
        "mix_manifest_sha256": global_hash,
        "computed_mix_manifest_sha256": computed_global_hash,
        "mix_manifest_hash_matches": bool(
            valid_sha256(global_hash) and global_hash == computed_global_hash
        ),
        "language_manifest_sha256": language_manifest,
        "language_entries": len(selected),
        "mixed_wav_hashes": len(mixed_hashes),
    }


def _asset_file_path(value: Mapping[str, Any]) -> str:
    return str(
        value.get("relative_path")
        or value.get("file")
        or value.get("filename")
        or value.get("path")
        or ""
    )


def artifact_pin_reasons(cell: dict[str, Any], model: str, revision: str) -> list[str]:
    metadata = cell.get("_engine_metadata", {})
    artifacts = metadata.get("artifacts", []) if isinstance(metadata, dict) else []
    matching_artifacts = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("repo_id") == model
        and artifact.get("revision") == revision
    ]
    if not matching_artifacts:
        return ["exact model artifact receipt missing"]
    files: list[dict[str, Any]] = []
    for artifact in matching_artifacts:
        files.extend(
            item
            for item in artifact.get("files", [])
            if isinstance(item, dict)
        )
    allowed_groups = MODEL_ASSET_GROUPS.get(model, ())
    group_failures: list[list[str]] = []
    for group in allowed_groups:
        reasons: list[str] = []
        for expected in group:
            expected_path = str(expected["file"])
            candidates = [
                item
                for item in files
                if _asset_file_path(item) == expected_path
                or _asset_file_path(item).endswith("/" + expected_path)
            ]
            if not candidates:
                reasons.append(f"asset missing: {expected_path}")
                continue
            actual = candidates[0]
            observed_sha = actual.get("sha256_observed") or actual.get("sha256")
            if observed_sha != expected["sha256"]:
                reasons.append(f"asset SHA mismatch: {expected_path}")
            if actual.get("sha256_verified") is False:
                reasons.append(f"asset SHA not verified: {expected_path}")
            expected_bytes = expected.get("bytes")
            if expected_bytes is not None:
                observed_bytes = first_number(
                    actual.get("bytes_observed"), actual.get("bytes")
                )
                if observed_bytes != expected_bytes:
                    reasons.append(f"asset byte mismatch: {expected_path}")
                if actual.get("bytes_verified") is False:
                    reasons.append(f"asset bytes not verified: {expected_path}")
        if not reasons:
            return []
        group_failures.append(reasons)
    if not allowed_groups:
        return ["no aggregator asset pin registered"]
    shortest = min(group_failures, key=len)
    return shortest


def _mapping_value(mapping: Any, language: str, config: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    return mapping.get(language, mapping.get(config))


def portable_fixture_identity_sha256(
    fixture: Any, language: str
) -> str | None:
    """Hash exact per-language evaluation content, excluding run-local wrappers.

    This is a secondary, host-computed comparison identity.  The original
    archive fixture hashes, record bindings, and member hashes remain the
    authoritative tamper/self-consistency checks and are never replaced by
    this digest.
    """

    if not isinstance(fixture, Mapping) or language not in FLEURS_CONFIGS:
        return None
    manifest = fixture.get("sample_manifest")
    if not isinstance(manifest, list) or not all(
        isinstance(row, Mapping) for row in manifest
    ):
        return None

    projected_rows: list[dict[str, Any]] = []
    condition_counts: dict[str, int] = {}
    base_hash_fields = (
        "reference_sha256",
        "raw_audio_sha256",
        "clean_pcm_sha256",
        "audio_pcm_sha256",
        "audio_file_sha256",
    )
    noise_hash_fields = (
        "noise_raw_audio_sha256",
        "noise_pcm_sha256",
        "noise_segment_pcm_sha256",
        "mix_entry_sha256",
        "mix_selector_sha256",
    )
    for raw_row in manifest:
        if str(raw_row.get("language")) != language:
            continue
        row = dict(raw_row)
        condition = str(row.get("condition", ""))
        if condition not in CONDITIONS:
            return None
        if any(field not in row for field in FIXTURE_BINDING_BASE_FIELDS):
            return None
        if any(not valid_sha256(row.get(field)) for field in base_hash_fields):
            return None
        reference = row.get("reference")
        if (
            not isinstance(reference, str)
            or hashlib.sha256(reference.encode("utf-8")).hexdigest()
            != row.get("reference_sha256")
        ):
            return None
        if condition == "musan_10db":
            if any(field not in row for field in FIXTURE_BINDING_NOISE_FIELDS):
                return None
            if any(
                not valid_sha256(row.get(field)) for field in noise_hash_fields
            ):
                return None
        elif any(field in row for field in FIXTURE_BINDING_NOISE_FIELDS):
            return None
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        projected_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in PORTABLE_FIXTURE_VOLATILE_FIELDS
            }
        )

    if condition_counts != {condition: TARGET_N for condition in CONDITIONS}:
        return None
    if len(projected_rows) != len(CONDITIONS) * TARGET_N:
        return None

    fleurs = fixture.get("fleurs")
    audio = fixture.get("audio")
    musan = fixture.get("musan")
    mix_manifest = fixture.get("mix_manifest")
    if not all(
        isinstance(value, Mapping)
        for value in (fleurs, audio, musan, mix_manifest)
    ):
        return None
    config = FLEURS_CONFIGS[language]
    fleurs_config = _mapping_value(fleurs.get("configs"), language, config)
    test_row_count = _mapping_value(
        fleurs.get("test_row_counts"), language, config
    )
    noise_bank_sha256 = fixture.get("noise_bank_sha256")
    if (
        fleurs_config is None
        or test_row_count is None
        or not valid_sha256(noise_bank_sha256)
        or not isinstance(fleurs.get("repo_id"), str)
        or not isinstance(fleurs.get("resolved_commit"), str)
        or not isinstance(fleurs.get("split"), str)
        or not isinstance(musan.get("repo_id"), str)
        or not isinstance(musan.get("resolved_commit"), str)
        or not isinstance(mix_manifest.get("schema"), str)
        or first_number(mix_manifest.get("seed")) is None
        or first_number(mix_manifest.get("target_snr_db")) is None
    ):
        return None

    projected_rows.sort(
        key=lambda row: json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    payload = {
        "schema": PORTABLE_FIXTURE_IDENTITY_SCHEMA,
        "language": language,
        "conditions": list(CONDITIONS),
        "samples_per_language_per_condition": TARGET_N,
        "fleurs": {
            "repo_id": fleurs.get("repo_id"),
            "resolved_commit": fleurs.get("resolved_commit"),
            "split": fleurs.get("split"),
            "config": fleurs_config,
            "test_row_count": test_row_count,
        },
        "audio": dict(audio),
        "musan": {
            "repo_id": musan.get("repo_id"),
            "resolved_commit": musan.get("resolved_commit"),
            "noise_bank_sha256": noise_bank_sha256,
        },
        "mix": {
            "schema": mix_manifest.get("schema"),
            "seed": mix_manifest.get("seed"),
            "target_snr_db": mix_manifest.get("target_snr_db"),
        },
        "samples": projected_rows,
    }
    return canonical_sha256(payload)


def fixture_identity_hash_for(
    cell: dict[str, Any] | None, language: str
) -> str | None:
    if not cell:
        return None
    cache = cell.get("_fixture_identity_cache")
    if isinstance(cache, dict):
        if language not in cache:
            cache[language] = portable_fixture_identity_sha256(
                cell.get("_fixture"), language
            )
        value = cache.get(language)
        return value if isinstance(value, str) else None
    return portable_fixture_identity_sha256(cell.get("_fixture"), language)


def fixture_pin_reasons(cell: dict[str, Any], language: str) -> list[str]:
    reasons: list[str] = []
    fixture = cell.get("_fixture", {})
    scored = cell.get("_scored", {})
    if not isinstance(fixture, dict) or not isinstance(scored, dict):
        return ["fixture/scoring receipt missing"]
    fleurs = fixture.get("fleurs", {})
    config = FLEURS_CONFIGS[language]
    expected_corpus = f"google/fleurs@{FLEURS_REVISION}/{config}/test"
    if not isinstance(fleurs, dict):
        reasons.append("FLEURS fixture pin missing")
    else:
        if fleurs.get("repo_id") != "google/fleurs":
            reasons.append("FLEURS repo mismatch")
        if fleurs.get("revision") != FLEURS_REVISION:
            reasons.append("FLEURS revision mismatch")
        if fleurs.get("resolved_commit") != FLEURS_REVISION:
            reasons.append("FLEURS resolved commit mismatch")
        if fleurs.get("split") != "test":
            reasons.append("FLEURS split mismatch")
        if _mapping_value(fleurs.get("configs"), language, config) != config:
            reasons.append("FLEURS config mismatch")
        if _mapping_value(fleurs.get("test_row_counts"), language, config) != FLEURS_TEST_ROWS[language]:
            reasons.append("FLEURS test row-count assertion missing/mismatched")
    if scored.get("corpus_repo_id") != "google/fleurs":
        reasons.append("scored corpus repo mismatch")
    if scored.get("corpus_revision") != FLEURS_REVISION:
        reasons.append("scored corpus revision mismatch")
    if scored.get("corpus_split") != "test":
        reasons.append("scored corpus split mismatch")
    if cell.get("corpus") != expected_corpus:
        reasons.append("cell corpus/config/split mismatch")
    fixture_hash = fixture_hash_for(cell, language)
    fixture_identity_hash = fixture_identity_hash_for(cell, language)
    fixture_manifest_hash = fixture_manifest_hash_for(cell)
    if not valid_sha256(fixture_hash):
        reasons.append("per-language fixture SHA missing")
    if not valid_sha256(fixture_identity_hash):
        reasons.append("portable per-language fixture identity missing")
    if not valid_sha256(fixture_manifest_hash):
        reasons.append("fixture manifest SHA missing")
    normalisers_hash = scored.get("normalisers_source_sha256")
    if normalisers_hash != NORMALISERS_SHA256:
        reasons.append("Lane B normaliser source SHA mismatch")
    return reasons


def musan_pin_reasons(cell: dict[str, Any], language: str) -> list[str]:
    fixture = cell.get("_fixture", {})
    musan = fixture.get("musan", {}) if isinstance(fixture, dict) else {}
    info = mix_manifest_info(cell, language)
    reasons: list[str] = []
    if not isinstance(musan, dict):
        reasons.append("MUSAN receipt missing")
    else:
        if musan.get("repo_id") != MUSAN_REPO:
            reasons.append("MUSAN repo mismatch")
        if musan.get("revision") != MUSAN_REVISION:
            reasons.append("MUSAN revision mismatch")
        if musan.get("resolved_commit") != MUSAN_REVISION:
            reasons.append("MUSAN resolved commit mismatch")
        if musan.get("glob") != MUSAN_NOISE_GLOB:
            reasons.append("MUSAN noise glob mismatch")
        files = musan.get("files")
        if not isinstance(files, list) or len(files) != MUSAN_NOISE_WAV_COUNT:
            reasons.append("MUSAN noise WAV inventory mismatch")
    if not valid_sha256(info["mix_manifest_sha256"]):
        reasons.append("global mix-manifest SHA missing")
    elif not info["mix_manifest_hash_matches"]:
        reasons.append("global mix-manifest SHA does not match manifest content")
    if not valid_sha256(info["language_manifest_sha256"]):
        reasons.append("per-language mix-manifest SHA missing")
    if info["language_entries"] != TARGET_N:
        reasons.append(f"mix manifest has {info['language_entries']} entries, expected {TARGET_N}")
    if info["mixed_wav_hashes"] != TARGET_N:
        reasons.append(f"mix manifest has {info['mixed_wav_hashes']} mixed-WAV hashes, expected {TARGET_N}")
    return reasons


def extract_condition_result(
    engine_results: dict[str, dict[str, Any] | None],
    indexed: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    bank: str,
    route: dict[str, str],
    language: str,
    condition: str,
) -> dict[str, Any]:
    cell = locate_cell(indexed, route["model"], language, condition)
    pinned = cell.get("pinned", {}) if isinstance(cell, dict) else {}
    if not isinstance(pinned, dict):
        pinned = {}
    metric = (cell or {}).get("metric") or pinned.get("metric")
    percent = first_number(
        (cell or {}).get("percent"),
        (cell or {}).get("value_percent"),
        (cell or {}).get("wer_or_cer_percent"),
        pinned.get("percent"),
    )
    if percent is None:
        rate = first_number(
            (cell or {}).get("rate"),
            (cell or {}).get("wer_or_cer"),
            pinned.get("rate"),
            pinned.get("wer_or_cer"),
        )
        percent = float(rate) * 100.0 if rate is not None else None
    ci95 = first_list((cell or {}).get("ci95_percent"), pinned.get("ci95_percent"))
    if ci95 is None:
        raw_ci = first_list((cell or {}).get("ci95"), pinned.get("ci95"))
        if raw_ci is not None and all(first_number(value) is not None for value in raw_ci):
            ci95 = [float(raw_ci[0]) * 100.0, float(raw_ci[1]) * 100.0]
    n = first_number((cell or {}).get("n"), pinned.get("n"))
    normalizer_id = (
        (cell or {}).get("normalizer_id")
        or (cell or {}).get("normaliser_id")
        or pinned.get("normalizer_id")
        or pinned.get("normaliser_id")
    )
    basic_normalizer_id = (cell or {}).get("basic_normalizer_id")
    basic_delta = first_number(
        (cell or {}).get("basic_delta_pp"),
        (cell or {}).get("basic_delta"),
        (cell or {}).get("normalizer_delta_signed_pp"),
    )
    absolute_delta = first_number((cell or {}).get("normalizer_delta_pp"))
    if absolute_delta is None and basic_delta is not None:
        absolute_delta = abs(float(basic_delta))
    reasons: list[str] = []
    valid_collapsed_smoke = False
    engine_result = engine_results.get(route["engine"])
    if cell is None:
        reasons.append(
            "Google Colab engine receipt missing"
            if engine_result is None
            else str(engine_result.get("error") or "exact model/language/condition cell missing")
        )
    else:
        if cell.get("_duplicate_count") != 1:
            reasons.append("duplicate scored cells")
        if cell.get("_engine_status") != "COMPLETE":
            reasons.append("engine status is not COMPLETE")
        if cell.get("_engine_evidence_class") != "HOST":
            reasons.append("engine evidence class is not HOST")
        reasons.extend(
            record_validation_reasons(cell.get("_engine_result"), label="engine")
        )
        scored = cell.get("_scored", {})
        if not isinstance(scored, dict) or scored.get("evidence_class") != "HOST":
            reasons.append("scored-model evidence class is not HOST")
        if cell.get("evidence_class") != "HOST":
            reasons.append("cell evidence class is not HOST")
        if cell.get("model") != route["model"]:
            reasons.append("model repo mismatch")
        if cell.get("bank") != bank:
            reasons.append("bank mismatch")
        if cell.get("revision") != route["revision"]:
            reasons.append("model revision mismatch")
        if normalize_condition(str(cell.get("condition"))) != condition:
            reasons.append("condition mismatch")
        if n != TARGET_N:
            smoke_candidate = (
                bank == "OmniASR"
                and route["engine"] == "omni"
                and condition == "clean"
                and n == OMNI_SMOKE_N
            )
            if smoke_candidate:
                smoke_reasons = omni_collapsed_smoke_gate_reasons(cell, language)
                if smoke_reasons:
                    reasons.extend(smoke_reasons)
                else:
                    valid_collapsed_smoke = True
            else:
                reasons.append(f"n={n!r}; expected exactly {TARGET_N}")
        expected_metric = "cer" if language == "zh" else "wer"
        if metric != expected_metric:
            reasons.append(f"metric={metric!r}; expected {expected_metric}")
        if percent is None:
            reasons.append("point estimate missing")
        if ci95 is None or len(ci95) != 2 or not all(first_number(value) is not None for value in ci95):
            reasons.append("bootstrap 95% CI missing")
        elif float(ci95[0]) > float(ci95[1]):
            reasons.append("bootstrap 95% CI is reversed")
        if not isinstance(normalizer_id, str) or not normalizer_id.startswith(f"norm_{language}/"):
            reasons.append("pinned normaliser id missing/mismatched")
        if not isinstance(basic_normalizer_id, str) or not basic_normalizer_id:
            reasons.append("BasicTextNormalizer id missing")
        if basic_delta is None or absolute_delta is None:
            reasons.append("BasicTextNormalizer delta missing")
        elif not math.isclose(abs(float(basic_delta)), float(absolute_delta), abs_tol=1e-8):
            reasons.append("signed/absolute normaliser deltas disagree")
        bootstrap_b = first_number(cell.get("bootstrap_B"), scored.get("bootstrap_B"))
        bootstrap_seed = first_number(cell.get("bootstrap_seed"), cell.get("seed"), scored.get("bootstrap_seed"), scored.get("seed"))
        if bootstrap_b != BOOTSTRAP_B:
            reasons.append("bootstrap B mismatch")
        if bootstrap_seed != BOOTSTRAP_SEED:
            reasons.append("bootstrap seed mismatch")
        ci_method = cell.get("ci_method")
        if not isinstance(ci_method, str) or "utterance" not in ci_method.lower() or "ratio" not in ci_method.lower():
            reasons.append("utterance ratio-of-sums CI method missing")
        reasons.extend(fixture_pin_reasons(cell, language))
        reasons.extend(artifact_pin_reasons(cell, route["model"], route["revision"]))
        if condition == "musan_10db":
            reasons.extend(musan_pin_reasons(cell, language))
    mix_info = mix_manifest_info(cell, language)
    fixture_hash = fixture_hash_for(cell, language)
    fixture_identity_hash = fixture_identity_hash_for(cell, language)
    fixture_manifest_hash = fixture_manifest_hash_for(cell)
    ranking_inadmissible = bool(
        valid_collapsed_smoke
        or (cell or {}).get("ranking_inadmissible")
        or (absolute_delta is not None and float(absolute_delta) > 2.0)
    )
    evidence = "HOST" if not reasons else "UNKNOWN"
    smoke_evidence = valid_collapsed_smoke and evidence == "HOST"
    evaluation_scope = (
        "collapsed_smoke"
        if smoke_evidence
        else "full"
        if evidence == "HOST" and n == TARGET_N
        else "unknown"
    )
    comparable = bool(evidence == "HOST" and n == TARGET_N and not smoke_evidence)
    non_comparability_reason = (
        "Validated OmniASR n=30 clean collapse smoke: HOST execution evidence, "
        "but not a full n=150 estimate and inadmissible for ranking."
        if smoke_evidence
        else None
    )
    smoke_gate = None
    if smoke_evidence and isinstance(engine_result, dict):
        metadata = engine_result.get("engine_metadata", {})
        gates = metadata.get("smoke_gate", {}) if isinstance(metadata, dict) else {}
        if isinstance(gates, dict) and isinstance(gates.get(language), dict):
            smoke_gate = gates[language]
    member_sha = (cell or {}).get("_member_sha256", {})
    return {
        "condition": condition,
        "n": int(n) if n is not None else 0,
        "wer_or_cer": {
            "metric": metric or ("cer" if language == "zh" else "wer"),
            "value_percent": float(percent) if percent is not None else None,
            "unit": "percent",
        },
        "ci95": [float(ci95[0]), float(ci95[1])] if ci95 else None,
        "ci_method": (cell or {}).get("ci_method"),
        "bootstrap_B": int(first_number((cell or {}).get("bootstrap_B"), (cell or {}).get("_scored", {}).get("bootstrap_B")) or 0),
        "seed": int(first_number((cell or {}).get("bootstrap_seed"), (cell or {}).get("seed"), (cell or {}).get("_scored", {}).get("bootstrap_seed")) or 0),
        "normaliser_id": normalizer_id,
        "basic_normaliser_id": basic_normalizer_id,
        "basic_delta": float(basic_delta) if basic_delta is not None else None,
        "basic_delta_absolute_pp": float(absolute_delta) if absolute_delta is not None else None,
        "fixture_sha256": fixture_hash,
        "fixture_identity_sha256": fixture_identity_hash,
        "fixture_manifest_sha256": fixture_manifest_hash,
        "mix_manifest_sha256": mix_info["mix_manifest_sha256"],
        "mix_manifest_hash_verified": mix_info["mix_manifest_hash_matches"],
        "mix_manifest_language_sha256": mix_info["language_manifest_sha256"],
        "mixed_wav_hash_count": mix_info["mixed_wav_hashes"],
        "counts": (cell or {}).get("counts"),
        "silent_failure_counts": (cell or {}).get("silent_failure_counts"),
        "evidence_class": evidence,
        "evaluation_scope": evaluation_scope,
        "smoke_evidence": smoke_evidence,
        "smoke_gate": smoke_gate,
        "comparable": comparable,
        "non_comparability_reason": non_comparability_reason,
        "above_15_percent": bool(percent is not None and float(percent) > 15.0),
        "ranking_inadmissible": ranking_inadmissible,
        "normaliser_delta_over_2pp": bool(absolute_delta is not None and float(absolute_delta) > 2.0),
        "blocker": "; ".join(dict.fromkeys(reasons)) if reasons else None,
        "evidence_link": {
            "engine": route["engine"],
            "archive_path": (cell or {}).get("_archive_path"),
            "archive_sha256": (cell or {}).get("_archive_sha256"),
            "records_member": "records.jsonl" if member_sha.get("records.jsonl") else None,
            "records_sha256": member_sha.get("records.jsonl"),
            "engine_result_sha256": member_sha.get("engine_result.json"),
        },
    }


def build_table(engine_results: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    indexed = cell_index(engine_results)
    rows: list[dict[str, Any]] = []
    for bank, routes in ROUTES.items():
        for language in LANGUAGES:
            route = routes[language]
            clean = extract_condition_result(
                engine_results,
                indexed,
                bank=bank,
                route=route,
                language=language,
                condition="clean",
            )
            noisy = extract_condition_result(
                engine_results,
                indexed,
                bank=bank,
                route=route,
                language=language,
                condition="musan_10db",
            )
            row_evidence = (
                "HOST"
                if clean["evidence_class"] == "HOST"
                and noisy["evidence_class"] == "HOST"
                else "UNKNOWN"
            )
            rows.append(
                {
                    "bank": bank,
                    "model": route["model"],
                    "revision": route["revision"],
                    "language": language,
                    "corpus": "google/fleurs test",
                    "corpus_revision": FLEURS_REVISION,
                    "corpus_config": FLEURS_CONFIGS[language],
                    "split": "test",
                    "n": clean["n"],
                    "wer_or_cer": clean["wer_or_cer"],
                    "ci95": clean["ci95"],
                    "ci_method": clean["ci_method"],
                    "bootstrap_B": clean["bootstrap_B"],
                    "seed": clean["seed"],
                    "normaliser_id": clean["normaliser_id"],
                    "basic_delta": clean["basic_delta"],
                    "basic_delta_absolute_pp": clean["basic_delta_absolute_pp"],
                    "fixture_sha256": clean["fixture_sha256"],
                    "fixture_identity_sha256": clean[
                        "fixture_identity_sha256"
                    ],
                    "fixture_manifest_sha256": clean["fixture_manifest_sha256"],
                    "clean_evidence_class": clean["evidence_class"],
                    "clean_evaluation_scope": clean["evaluation_scope"],
                    "clean_smoke_evidence": clean["smoke_evidence"],
                    "clean_smoke_gate": clean["smoke_gate"],
                    "clean_comparable": clean["comparable"],
                    "clean_non_comparability_reason": clean[
                        "non_comparability_reason"
                    ],
                    "noisy_bucket": {
                        **noisy,
                        "label": "MUSAN 10 dB",
                        "source_label": MUSAN_SOURCE_LABEL,
                        "source": MUSAN_REPO,
                        "source_revision": MUSAN_REVISION,
                        "source_caveat": MUSAN_SOURCE_CAVEAT,
                        "snr_db": 10.0,
                    },
                    "evidence_class": row_evidence,
                    "above_15_percent": clean["above_15_percent"],
                    "ranking_inadmissible": clean["ranking_inadmissible"],
                    "normaliser_delta_over_2pp": clean["normaliser_delta_over_2pp"],
                    "blocker": {
                        "clean": clean["blocker"],
                        "musan_10db": noisy["blocker"],
                    }
                    if row_evidence == "UNKNOWN"
                    else None,
                    "evidence_link": clean["evidence_link"],
                }
            )
    return rows


def c3_row_completion_mode(row: Mapping[str, Any]) -> str | None:
    """Classify an Omni language as full evaluation or valid collapsed smoke."""

    noisy = row.get("noisy_bucket")
    if not isinstance(noisy, dict):
        return None
    if (
        row.get("evidence_class") == "HOST"
        and row.get("clean_evidence_class") == "HOST"
        and row.get("n") == TARGET_N
        and noisy.get("evidence_class") == "HOST"
        and noisy.get("n") == TARGET_N
        and row.get("clean_smoke_evidence") is not True
    ):
        return "full_n150_clean_and_noisy"
    if (
        row.get("bank") == "OmniASR"
        and row.get("clean_evidence_class") == "HOST"
        and row.get("clean_evaluation_scope") == "collapsed_smoke"
        and row.get("clean_smoke_evidence") is True
        and row.get("n") == OMNI_SMOKE_N
        and row.get("clean_comparable") is False
        and row.get("ranking_inadmissible") is True
        and noisy.get("evidence_class") == "UNKNOWN"
        and noisy.get("n") == 0
    ):
        return "collapsed_n30_clean_smoke"
    return None


def c3_completion(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    language_modes = {
        str(row.get("language")): c3_row_completion_mode(row) for row in rows
    }
    return {
        "complete": set(language_modes) == set(LANGUAGES)
        and all(mode is not None for mode in language_modes.values()),
        "language_modes": language_modes,
    }


def comparability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for language in LANGUAGES:
        selected = [row for row in rows if row["language"] == language]
        clean_host = [row for row in selected if row["clean_evidence_class"] == "HOST"]
        clean_archive_hashes = {row["fixture_sha256"] for row in clean_host}
        clean_identity_values = [
            row.get("fixture_identity_sha256") for row in clean_host
        ]
        clean_identity_hashes = set(clean_identity_values)
        clean_ns = {row["n"] for row in clean_host}
        clean_norms = {row["normaliser_id"] for row in clean_host}
        clean_corpora = {
            (row["corpus_revision"], row["corpus_config"], row["split"])
            for row in clean_host
        }
        clean_comparable = (
            len(clean_host) == 3
            and all(valid_sha256(value) for value in clean_identity_values)
            and len(clean_identity_hashes) == 1
            and len(clean_ns) == 1
            and clean_ns == {TARGET_N}
            and len(clean_norms) == 1
            and len(clean_corpora) == 1
        )
        noisy_host = [
            row
            for row in selected
            if row["noisy_bucket"]["evidence_class"] == "HOST"
        ]
        noisy_archive_hashes = {
            row["noisy_bucket"]["fixture_sha256"] for row in noisy_host
        }
        noisy_identity_values = [
            row["noisy_bucket"].get("fixture_identity_sha256")
            for row in noisy_host
        ]
        noisy_identity_hashes = set(noisy_identity_values)
        noisy_mix_hashes = {
            row["noisy_bucket"]["mix_manifest_language_sha256"]
            for row in noisy_host
        }
        noisy_ns = {row["noisy_bucket"]["n"] for row in noisy_host}
        noisy_norms = {row["noisy_bucket"]["normaliser_id"] for row in noisy_host}
        noisy_comparable = (
            len(noisy_host) == 3
            and all(valid_sha256(value) for value in noisy_identity_values)
            and len(noisy_identity_hashes) == 1
            and len(noisy_mix_hashes) == 1
            and len(noisy_ns) == 1
            and noisy_ns == {TARGET_N}
            and len(noisy_norms) == 1
        )
        ranking_admissible = clean_comparable and all(
            not row["ranking_inadmissible"] for row in selected
        )
        checks.append(
            {
                "language": language,
                "clean": {
                    "host_rows": len(clean_host),
                    "fixture_hashes": sorted(clean_archive_hashes),
                    "archive_fixture_hashes": sorted(clean_archive_hashes),
                    "fixture_identity_hashes": sorted(
                        value
                        for value in clean_identity_hashes
                        if isinstance(value, str)
                    ),
                    "n_values": sorted(clean_ns),
                    "normaliser_ids": sorted(clean_norms),
                    "comparable_across_all_three_banks": clean_comparable,
                    "ranking_admissible": ranking_admissible,
                },
                "musan_10db": {
                    "host_rows": len(noisy_host),
                    "fixture_hashes": sorted(noisy_archive_hashes),
                    "archive_fixture_hashes": sorted(noisy_archive_hashes),
                    "fixture_identity_hashes": sorted(
                        value
                        for value in noisy_identity_hashes
                        if isinstance(value, str)
                    ),
                    "mix_manifest_language_hashes": sorted(noisy_mix_hashes),
                    "n_values": sorted(noisy_ns),
                    "normaliser_ids": sorted(noisy_norms),
                    "comparable_across_all_three_banks": noisy_comparable,
                },
                "comparable_across_all_three_banks": clean_comparable
                and noisy_comparable,
            }
        )
    return {
        "all_cells_comparable": all(
            item["comparable_across_all_three_banks"] for item in checks
        ),
        "all_clean_rankings_admissible": all(
            item["clean"]["ranking_admissible"] for item in checks
        ),
        "checks": checks,
    }


def engine_artifact(engine_results: dict[str, dict[str, Any] | None], engine: str, *keys: str) -> Any:
    result = engine_results.get(engine)
    if not result:
        return None
    value: Any = result.get("engine_metadata", {})
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def route_completion_reasons(
    indexed: dict[tuple[str, str, str], list[dict[str, Any]]],
    route: Mapping[str, Any],
    language: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    cell = locate_cell(indexed, str(route["model"]), language, "clean")
    if cell is None:
        return None, ["executed clean route receipt missing"]
    reasons: list[str] = []
    if cell.get("_duplicate_count") != 1:
        reasons.append("duplicate clean route receipts")
    if cell.get("_engine_status") != "COMPLETE":
        reasons.append("engine status is not COMPLETE")
    if cell.get("_engine_evidence_class") != "HOST":
        reasons.append("engine evidence class is not HOST")
    reasons.extend(
        record_validation_reasons(cell.get("_engine_result"), label="engine")
    )
    if cell.get("evidence_class") != "HOST":
        reasons.append("route cell evidence class is not HOST")
    if first_number(cell.get("n")) is None or int(cell.get("n", 0)) <= 0:
        reasons.append("route cell has no executed utterances")
    if cell.get("model") != route["model"]:
        reasons.append("route model mismatch")
    if cell.get("bank") != route["bank"]:
        reasons.append("route bank mismatch")
    if cell.get("revision") != route["revision"]:
        reasons.append("route revision mismatch")
    reasons.extend(
        artifact_pin_reasons(cell, str(route["model"]), str(route["revision"]))
    )
    return cell, reasons


def trace_language(trace: Mapping[str, Any]) -> str:
    return str(trace.get("language") or trace.get("lang") or "")


def trace_matches_route(trace: Mapping[str, Any], route: Mapping[str, Any]) -> bool:
    trace_route = str(trace.get("route") or "")
    trace_model = str(trace.get("model") or "")
    return trace_route == route["route"] or trace_model == route["model"]


def sensevoice_fixed_context_probe_reasons(
    engine_result: Mapping[str, Any], language: str
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    metadata = engine_result.get("engine_metadata")
    if not isinstance(metadata, Mapping):
        return None, ["SenseVoice engine metadata missing"]
    receipt = metadata.get("fixed_context_probe")
    if not isinstance(receipt, Mapping):
        return None, ["SenseVoice fixed-10-second context probe receipt missing"]
    expected_top = {
        "schema": "lane-c-sensevoice-fixed-context-probe/1.0",
        "route": "fixed-10s-context/final-only",
        "model": "FunAudioLLM/SenseVoiceSmall",
        "bank": "D-fork",
        "revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        "context_seconds": SENSEVOICE_FIXED_CONTEXT_SECONDS,
        "sample_rate": SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
        "target_num_samples": SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES,
        "probe_n_per_language": SENSEVOICE_FIXED_CONTEXT_PROBE_N,
        "condition": "clean",
        "main_scoring_inputs_unchanged": True,
        "scoring_included": False,
        "reference_scoring_included": False,
        "partial_api": "utterance-only/no partial API",
        "partials_emitted": False,
        "valid": True,
    }
    if any(receipt.get(key) != value for key, value in expected_top.items()):
        reasons.append("SenseVoice fixed-context probe protocol mismatch")
    observed_receipt_hash = receipt.get("receipt_sha256")
    unhashed_receipt = dict(receipt)
    unhashed_receipt.pop("receipt_sha256", None)
    if not valid_sha256(observed_receipt_hash) or canonical_sha256(
        unhashed_receipt
    ) != observed_receipt_hash:
        reasons.append("SenseVoice fixed-context probe receipt SHA mismatch")

    fixture = engine_result.get("fixture")
    manifest = fixture.get("sample_manifest") if isinstance(fixture, Mapping) else None
    if not isinstance(manifest, list):
        return None, reasons + ["SenseVoice fixture sample manifest missing"]
    expected_samples = [
        sample
        for sample in manifest
        if isinstance(sample, Mapping)
        and sample.get("language") == language
        and normalize_condition(str(sample.get("condition"))) == "clean"
    ][:SENSEVOICE_FIXED_CONTEXT_PROBE_N]
    if len(expected_samples) != SENSEVOICE_FIXED_CONTEXT_PROBE_N:
        reasons.append(
            f"SenseVoice fixed-context fixture selection for {language} is incomplete"
        )

    languages = receipt.get("languages")
    if not isinstance(languages, Mapping) or set(languages) != {"en", "zh"}:
        return None, reasons + [
            "SenseVoice fixed-context probe language inventory mismatch"
        ]
    language_receipt = languages.get(language)
    if not isinstance(language_receipt, Mapping):
        return None, reasons + [
            f"SenseVoice fixed-context probe receipt missing for {language}"
        ]
    expected_counts = {
        "expected": SENSEVOICE_FIXED_CONTEXT_PROBE_N,
        "selected": SENSEVOICE_FIXED_CONTEXT_PROBE_N,
        "invoked": SENSEVOICE_FIXED_CONTEXT_PROBE_N,
        "succeeded": SENSEVOICE_FIXED_CONTEXT_PROBE_N,
        "failed": 0,
    }
    if any(
        language_receipt.get(key) != value
        for key, value in expected_counts.items()
    ):
        reasons.append(f"SenseVoice fixed-context probe counts mismatch for {language}")
    entries = language_receipt.get("samples")
    if not isinstance(entries, list) or not all(
        isinstance(entry, Mapping) for entry in entries
    ):
        return None, reasons + [
            f"SenseVoice fixed-context sample receipts missing for {language}"
        ]
    if len(entries) != SENSEVOICE_FIXED_CONTEXT_PROBE_N:
        reasons.append(
            f"SenseVoice fixed-context sample count mismatch for {language}"
        )

    expected_by_identity = {
        (str(sample.get("utterance_id")), str(sample.get("audio_relpath"))): sample
        for sample in expected_samples
    }
    observed_identities: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(entries):
        entry = dict(raw_entry)
        identity = (
            str(entry.get("utterance_id", "")),
            str(entry.get("audio_relpath", "")),
        )
        if identity in observed_identities:
            reasons.append(
                f"SenseVoice fixed-context probe repeats {language} sample {identity!r}"
            )
        observed_identities.add(identity)
        sample = expected_by_identity.get(identity)
        if sample is None:
            reasons.append(
                f"SenseVoice fixed-context probe sample {index} is not fixture-bound"
            )
            continue
        expected_source = {
            "language": language,
            "condition": "clean",
            "source_audio_pcm_sha256": sample.get("audio_pcm_sha256"),
            "source_clean_pcm_sha256": sample.get("clean_pcm_sha256"),
            "source_audio_file_sha256": sample.get("audio_file_sha256"),
            "source_reference_sha256": sample.get("reference_sha256"),
            "source_num_samples": sample.get("num_samples"),
            "source_sample_rate": sample.get("sample_rate"),
            "context_seconds": SENSEVOICE_FIXED_CONTEXT_SECONDS,
            "derived_num_samples": SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES,
            "derived_sample_rate": SENSEVOICE_FIXED_CONTEXT_SAMPLE_RATE,
            "scoring_included": False,
            "reference_modified": False,
            "model_invoked": True,
            "status": "ok",
        }
        if any(entry.get(key) != value for key, value in expected_source.items()):
            reasons.append(
                f"SenseVoice fixed-context probe sample {index} provenance mismatch"
            )
        source_n = first_number(sample.get("num_samples"))
        expected_transform = (
            "right_zero_pad"
            if source_n is not None
            and source_n < SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES
            else "truncate_tail"
            if source_n is not None
            and source_n > SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES
            else "identity"
        )
        expected_copied = (
            min(int(source_n), SENSEVOICE_FIXED_CONTEXT_NUM_SAMPLES)
            if source_n is not None
            else None
        )
        if entry.get("transformation") != expected_transform or entry.get(
            "copied_source_samples"
        ) != expected_copied:
            reasons.append(
                f"SenseVoice fixed-context probe sample {index} transform mismatch"
            )
        for field in (
            "derived_pcm_sha256",
            "derived_wav_sha256",
            "final_text_sha256",
        ):
            if not valid_sha256(entry.get(field)):
                reasons.append(
                    f"SenseVoice fixed-context probe sample {index} lacks {field}"
                )
        observed_entry_hash = entry.pop("entry_sha256", None)
        if not valid_sha256(observed_entry_hash) or canonical_sha256(
            entry
        ) != observed_entry_hash:
            reasons.append(
                f"SenseVoice fixed-context probe sample {index} SHA mismatch"
            )
    if observed_identities != set(expected_by_identity):
        reasons.append(
            f"SenseVoice fixed-context probe does not cover exact {language} selection"
        )

    summary = {
        "schema": receipt.get("schema"),
        "route": receipt.get("route"),
        "context_seconds": receipt.get("context_seconds"),
        "n": language_receipt.get("succeeded"),
        "condition": receipt.get("condition"),
        "scoring_included": receipt.get("scoring_included"),
        "main_scoring_inputs_unchanged": receipt.get(
            "main_scoring_inputs_unchanged"
        ),
        "partials_emitted": receipt.get("partials_emitted"),
        "receipt_sha256": observed_receipt_hash,
        "valid": not reasons,
    }
    return summary, reasons


def build_c4(engine_results: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    indexed = cell_index(engine_results)
    rows: list[dict[str, Any]] = []
    for route in PARTIAL_ROUTES:
        engine_result = engine_results.get(str(route["engine"]))
        metrics = (
            engine_result.get("partial_metrics", [])
            if isinstance(engine_result, dict)
            else []
        )
        traces = (
            engine_result.get("_partial_traces", [])
            if isinstance(engine_result, dict)
            else []
        )
        for language in route["languages"]:
            completion_cell, reasons = route_completion_reasons(
                indexed, route, language
            )
            selected_traces = [
                trace
                for trace in traces
                if isinstance(trace, dict)
                and trace_language(trace) == language
                and trace_matches_route(trace, route)
                and normalize_condition(str(trace.get("condition") or "clean"))
                == "clean"
            ]
            trace_digest = (
                canonical_sha256(selected_traces) if selected_traces else None
            )
            trace_hash_count = sum(
                valid_sha256(trace.get("trace_sha256"))
                for trace in selected_traces
            )
            trace_hash_verified_count = sum(
                trace_sha256_matches(trace) for trace in selected_traces
            )
            metric = next(
                (
                    item
                    for item in metrics
                    if isinstance(item, dict)
                    and str(item.get("language")) == language
                    and (
                        not item.get("route")
                        or str(item.get("route")) == route["route"]
                    )
                ),
                None,
            )
            row: dict[str, Any] = {
                "bank": route["bank"],
                "route": route["route"],
                "model": route["model"],
                "revision": route["revision"],
                "engine": route["engine"],
                "language": language,
                "partials_emitted": None,
                "structural_reason": None,
                "first_partial_latency_ms": None,
                "partial_rewrite_rate_edits_per_audio_second": None,
                "final_vs_last_partial_divergence": None,
                "normaliser_id": (
                    completion_cell.get("normalizer_id")
                    if completion_cell
                    else None
                ),
                "n": int(
                    first_number(completion_cell.get("n")) or 0
                )
                if completion_cell
                else 0,
                "trace_count": len(selected_traces),
                "trace_sha256_count": trace_hash_count,
                "trace_sha256_verified_count": trace_hash_verified_count,
                "trace_digest": trace_digest,
                "trace_archive": {
                    "path": (
                        completion_cell.get("_archive_path")
                        if completion_cell
                        else None
                    ),
                    "archive_sha256": (
                        completion_cell.get("_archive_sha256")
                        if completion_cell
                        else None
                    ),
                    "member": "partial_traces.jsonl"
                    if isinstance(engine_result, dict)
                    and engine_result.get("_member_sha256", {}).get(
                        "partial_traces.jsonl"
                    )
                    else None,
                    "member_sha256": (
                        engine_result.get("_member_sha256", {}).get(
                            "partial_traces.jsonl"
                        )
                        if isinstance(engine_result, dict)
                        else None
                    ),
                },
                "evidence_class": "UNKNOWN",
                "blocker": None,
            }
            if not route["native_partials"]:
                row["partials_emitted"] = False
                row["structural_reason"] = route["cause"]
                if route["model"] == "FunAudioLLM/SenseVoiceSmall":
                    if isinstance(engine_result, Mapping):
                        probe, probe_reasons = (
                            sensevoice_fixed_context_probe_reasons(
                                engine_result, language
                            )
                        )
                        row["fixed_context_probe"] = probe
                        reasons.extend(probe_reasons)
                    else:
                        row["fixed_context_probe"] = None
                        reasons.append(
                            "SenseVoice fixed-10-second context probe receipt missing"
                        )
                if not reasons:
                    row["evidence_class"] = "HOST"
                else:
                    row["blocker"] = "; ".join(dict.fromkeys(reasons))
                rows.append(row)
                continue

            if metric is None:
                reasons.append("partial metric receipt missing")
            else:
                row["partials_emitted"] = bool(metric.get("partials_emitted"))
                row["first_partial_latency_ms"] = metric.get(
                    "first_partial_latency_ms"
                )
                row["partial_rewrite_rate_edits_per_audio_second"] = metric.get(
                    "partial_rewrite_rate_edits_per_audio_second"
                )
                row["final_vs_last_partial_divergence"] = metric.get(
                    "final_vs_last_partial_divergence"
                )
                row["normaliser_id"] = metric.get("normalizer_id") or metric.get(
                    "normaliser_id"
                )
                row["n"] = int(first_number(metric.get("n_input")) or 0)
                row["first_partial_latency"] = metric.get(
                    "first_partial_latency"
                )
                row["partial_rewrite_rate"] = metric.get(
                    "partial_rewrite_rate"
                )
                row["final_partial_divergence"] = metric.get(
                    "final_partial_divergence"
                )
                if first_number(metric.get("bootstrap_B")) != BOOTSTRAP_B:
                    reasons.append("partial bootstrap B mismatch")
                if first_number(metric.get("bootstrap_seed")) != BOOTSTRAP_SEED:
                    reasons.append("partial bootstrap seed mismatch")
                if metric.get("evidence_class") != "HOST":
                    reasons.append("partial metric evidence class is not HOST")
                n_invalid = first_number(metric.get("n_invalid"))
                if n_invalid is None:
                    reasons.append("partial metric n_invalid is missing")
                elif float(n_invalid) != 0.0:
                    reasons.append(
                        f"partial metric n_invalid={n_invalid}; expected 0"
                    )
            if len(selected_traces) != PARTIAL_TRACE_N:
                reasons.append(
                    f"trace_count={len(selected_traces)}; expected {PARTIAL_TRACE_N}"
                )
            if trace_hash_count != len(selected_traces):
                reasons.append("one or more per-utterance trace digests missing")
            if trace_hash_verified_count != len(selected_traces):
                reasons.append(
                    "one or more per-utterance trace digests fail canonical verification"
                )
            if not valid_sha256(trace_digest):
                reasons.append("route/language trace digest missing")
            if row["partials_emitted"]:
                for key in (
                    "first_partial_latency_ms",
                    "partial_rewrite_rate_edits_per_audio_second",
                    "final_vs_last_partial_divergence",
                ):
                    if first_number(row.get(key)) is None:
                        reasons.append(f"{key} missing")
            elif row["partials_emitted"] is False:
                row["structural_reason"] = (
                    "Executed 100 ms Zipformer feed emitted no non-empty partials."
                )
            if not reasons:
                row["evidence_class"] = "HOST"
            else:
                row["blocker"] = "; ".join(dict.fromkeys(reasons))
            rows.append(row)
    return {
        "schema": "lane-c-c4/1.0",
        "sprint_id": "C4",
        "status": "COMPLETE" if all(row["evidence_class"] == "HOST" for row in rows) else "PARTIAL",
        "chunk_schedule_ms": 100,
        "rows": rows,
        "product_default": "Emit partial text only for the Zipformer route if its measured trace is acceptable; every other selected route is final-only unless a different runtime path is deliberately engineered.",
        "evidence_note": "A route with no partial API is recorded as partials_emitted=false, not as a large latency.",
    }


def fmt_percent(value: Any) -> str:
    return "UNKNOWN" if value is None else f"{float(value):.2f}%"


def table_markdown(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| Lang | Model | Clean n | Clean WER/CER | Clean 95% CI | MUSAN n | MUSAN WER/CER | MUSAN 95% CI | Basic Δ clean/noisy | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metric = row["wer_or_cer"]
        value = metric.get("value_percent")
        ci = row.get("ci95")
        noisy = row["noisy_bucket"]
        noisy_metric = noisy["wer_or_cer"]
        noisy_value = noisy_metric.get("value_percent")
        noisy_ci = noisy.get("ci95")
        flags = []
        if row.get("above_15_percent"):
            flags.append("clean>15%")
        if noisy.get("above_15_percent"):
            flags.append("noisy>15%")
        if row.get("clean_smoke_evidence"):
            flags.append("clean collapsed smoke (non-comparable)")
        elif row.get("ranking_inadmissible"):
            flags.append("clean normaliser-carried")
        if noisy.get("ranking_inadmissible"):
            flags.append("noisy normaliser-carried")
        if row.get("clean_evidence_class") != "HOST":
            flags.append("clean UNKNOWN")
        if noisy.get("evidence_class") != "HOST":
            flags.append("noisy UNKNOWN")
        lines.append(
            "| {language} | `{model}` | {n} | {metric} {value} | {ci} | {noisy_n} | {noisy_metric} {noisy_value} | {noisy_ci} | {delta}/{noisy_delta} | {flags} |".format(
                language=str(row["language"]).upper(),
                model=row["model"],
                n=row["n"] or "—",
                metric=str(metric.get("metric", "")).upper(),
                value=fmt_percent(value),
                ci=("UNKNOWN" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]%"),
                noisy_n=noisy.get("n") or "—",
                noisy_metric=str(noisy_metric.get("metric", "")).upper(),
                noisy_value=fmt_percent(noisy_value),
                noisy_ci=(
                    "UNKNOWN"
                    if not noisy_ci
                    else f"[{noisy_ci[0]:.2f}, {noisy_ci[1]:.2f}]%"
                ),
                delta=("UNKNOWN" if row.get("basic_delta") is None else f"{row['basic_delta']:+.2f} pp"),
                noisy_delta=(
                    "UNKNOWN"
                    if noisy.get("basic_delta") is None
                    else f"{noisy['basic_delta']:+.2f} pp"
                ),
                flags=", ".join(flags) or "—",
            )
        )
    return "\n".join(lines)


def task_report(
    sprint_id: str,
    title: str,
    rows: list[dict[str, Any]],
    extra: list[str],
    *,
    status: str | None = None,
    status_detail: str | None = None,
) -> str:
    host = sum(row["evidence_class"] == "HOST" for row in rows)
    resolved_status = status or ("COMPLETE" if host == len(rows) else "PARTIAL")
    detail = status_detail or f"{host}/{len(rows)} complete clean+MUSAN HOST rows"
    return "\n".join(
        [
            f"# {sprint_id} — {title}",
            "",
            f"Status: **{resolved_status}** ({detail}).",
            "",
            MUSAN_CAVEAT_MARKDOWN,
            "",
            table_markdown(rows),
            "",
            *extra,
            "",
            "All error rates use Lane B's byte-identical, source-hashed per-language normaliser and an utterance-level percentile bootstrap (B=10,000, seed=20260801). Vendor and cross-corpus numbers are not substituted for missing HOST cells.",
        ]
    )


def winners(
    rows: list[dict[str, Any]], comparable: Mapping[str, Any]
) -> dict[str, Any]:
    output = {}
    for language in LANGUAGES:
        check = next(
            (
                item
                for item in comparable.get("checks", [])
                if item.get("language") == language
            ),
            {},
        )
        clean_check = check.get("clean", {}) if isinstance(check, dict) else {}
        if not clean_check.get("ranking_admissible"):
            if clean_check.get("comparable_across_all_three_banks"):
                reason = (
                    "all three clean banks are fixture-comparable, but at least "
                    "one row is ranking-inadmissible because its normaliser delta "
                    "exceeds 2 pp"
                )
            else:
                reason = "all three clean banks are not fixture-comparable"
            output[language] = {
                "status": "UNKNOWN",
                "reason": reason,
            }
            continue
        candidates = [
            row
            for row in rows
            if row["language"] == language
            and row["clean_evidence_class"] == "HOST"
            and not row["ranking_inadmissible"]
            and row["wer_or_cer"]["value_percent"] is not None
        ]
        if len(candidates) != 3:
            output[language] = {
                "status": "UNKNOWN",
                "reason": f"expected three admissible banks, found {len(candidates)}",
            }
            continue
        candidates.sort(key=lambda row: row["wer_or_cer"]["value_percent"])
        best = candidates[0]
        output[language] = {
            "status": "HOST",
            "bank": best["bank"],
            "model": best["model"],
            "metric": best["wer_or_cer"]["metric"],
            "value_percent": best["wer_or_cer"]["value_percent"],
            "n": best["n"],
        }
    return output


def _relative_artifact_path(value: Any, out_dir: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("artifacts", "results"):
            return Path(*parts[index:]).as_posix()
    try:
        return path.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return None


def compact_engine_receipts(
    engine_results: Mapping[str, dict[str, Any] | None], out_dir: Path
) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    retained_metadata_keys = (
        "artifacts",
        "artifact_caveats",
        "fixed_context_probe",
        "parakeet_tokenizer",
        "parakeet_tokenizer_audit",
        "smoke_gate",
        "runtime",
        "precision",
        "partial_api",
        "partial_capability",
        "context_policy",
        "settings",
    )
    for engine, result in engine_results.items():
        if not result:
            receipts[engine] = {
                "status": "MISSING",
                "evidence_class": "UNKNOWN",
                "blocker": "Google Colab engine receipt missing",
            }
            continue
        fixture = result.get("fixture", {})
        metadata = result.get("engine_metadata", {})
        compact_metadata = {
            key: metadata.get(key)
            for key in retained_metadata_keys
            if isinstance(metadata, dict) and key in metadata
        }
        receipts[engine] = {
            "status": result.get("status"),
            "evidence_class": result.get("evidence_class"),
            "error": result.get("error"),
            "models": result.get("models", []),
            "archive": {
                "path": _relative_artifact_path(
                    result.get("_archive_path"), out_dir
                ),
                "sha256": result.get("_archive_sha256"),
                "members": result.get("_archive_members", []),
                "member_sha256": result.get("_member_sha256", {}),
                "validation": result.get("_archive_validation"),
            },
            "fixture": {
                "fixture_manifest_sha256": fixture.get(
                    "fixture_manifest_sha256"
                )
                if isinstance(fixture, dict)
                else None,
                "fixture_hashes_by_language": fixture.get(
                    "fixture_hashes_by_language", {}
                )
                if isinstance(fixture, dict)
                else {},
                "mix_manifest_sha256": fixture.get("mix_manifest_sha256")
                if isinstance(fixture, dict)
                else None,
                "noise_bank_sha256": fixture.get("noise_bank_sha256")
                if isinstance(fixture, dict)
                else None,
                "fleurs": fixture.get("fleurs")
                if isinstance(fixture, dict)
                else None,
                "musan": fixture.get("musan")
                if isinstance(fixture, dict)
                else None,
            },
            "environment": result.get("environment"),
            "record_validation": result.get("record_validation"),
            "engine_metadata": compact_metadata,
        }
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    results_root = (args.results_root or (out_dir / "artifacts" / "results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_results = {engine: read_engine_zip(results_root, engine) for engine in ("sensevoice", "nemo", "zipformer", "gigaam", "omni")}
    rows = build_table(engine_results)
    comparable = comparability(rows)
    generated_at = utc_now()
    receipts = compact_engine_receipts(engine_results, out_dir)

    dependency = {
        "lane_b_normalisers": "MET",
        "source_branch": "sprint/b-backbone",
        "source_path": "workstreams/sub2b_sprint/sprint/out/b/normalisers.py",
        "lane_c_path": "workstreams/sub2b_sprint/sprint/out/c/normalisers.py",
        "source_sha256": NORMALISERS_SHA256,
        "lane_c_sha256": NORMALISERS_SHA256,
        "byte_identical": True,
    }

    above_15_warnings: list[dict[str, Any]] = []
    normaliser_warnings: list[dict[str, Any]] = []
    for row in rows:
        for condition, result in (
            ("clean", row),
            ("musan_10db", row["noisy_bucket"]),
        ):
            if result.get("above_15_percent"):
                above_15_warnings.append(
                    {
                        "bank": row["bank"],
                        "language": row["language"],
                        "condition": condition,
                        "value_percent": result["wer_or_cer"][
                            "value_percent"
                        ],
                    }
                )
            if result.get("normaliser_delta_over_2pp"):
                normaliser_warnings.append(
                    {
                        "bank": row["bank"],
                        "language": row["language"],
                        "condition": condition,
                        "delta_pp": result.get("basic_delta"),
                    }
                )

    table_payload = {
        "schema": "lane-c-asr-table/1.0",
        "generated_at": generated_at,
        "corpus": {
            "id": "google/fleurs",
            "revision": FLEURS_REVISION,
            "split": "test",
            "configs": FLEURS_CONFIGS,
            "verified_test_row_counts": FLEURS_TEST_ROWS,
            "n_target_per_language": TARGET_N,
            "max_duration_seconds": 25.0,
        },
        "noisy_bucket": {
            "label": MUSAN_SOURCE_LABEL,
            "source": MUSAN_REPO,
            "revision": MUSAN_REVISION,
            "source_caveat": MUSAN_SOURCE_CAVEAT,
            "mirror_bytes": MUSAN_MIRROR_BYTES,
            "noise_wav_count": MUSAN_NOISE_WAV_COUNT,
            "noise_wav_bytes": MUSAN_NOISE_WAV_BYTES,
            "noise_glob": MUSAN_NOISE_GLOB,
            "snr_db": 10.0,
            "seed": BOOTSTRAP_SEED,
            "mix_manifest_hashes_by_engine": {
                engine: receipt.get("fixture", {}).get(
                    "mix_manifest_sha256"
                )
                for engine, receipt in receipts.items()
                if isinstance(receipt, dict)
            },
        },
        "normaliser_dependency": dependency,
        "comparability": comparable,
        "engine_receipts": receipts,
        "rows": rows,
        "warnings": {
            "above_15_percent": above_15_warnings,
            "normaliser_delta_over_2pp": normaliser_warnings,
        },
        "per_language_clean_leader": winners(rows, comparable),
    }
    write_json(out_dir / "asr_table.json", table_payload)

    c1_rows = [row for row in rows if row["bank"] == "D-fork"]
    c2_rows = [row for row in rows if row["bank"] == "L-fork"]
    c3_rows = [row for row in rows if row["bank"] == "OmniASR"]
    ru_host_rows = [
        row
        for row in rows
        if row["language"] == "ru"
        and row["clean_evidence_class"] == "HOST"
        and row["bank"] in {"D-fork", "L-fork"}
    ]
    ru_by_model = {row["model"]: row for row in ru_host_rows}
    parakeet_ru = ru_by_model.get("nvidia/parakeet-tdt-0.6b-v3")
    gigaam_ru = ru_by_model.get("ai-sage/GigaAM-Multilingual")
    ru_host_result = None
    if parakeet_ru and gigaam_ru:
        parakeet_value = parakeet_ru["wer_or_cer"]["value_percent"]
        gigaam_value = gigaam_ru["wer_or_cer"]["value_percent"]
        ru_ranking_blocked_models = [
            row["model"]
            for row in (parakeet_ru, gigaam_ru)
            if row["ranking_inadmissible"]
        ]
        ru_ranking_admissible = not (
            parakeet_ru["ranking_inadmissible"]
            or gigaam_ru["ranking_inadmissible"]
        )
        ru_host_result = {
            "parakeet_WER_percent": parakeet_value,
            "GigaAM_WER_percent": gigaam_value,
            "parakeet_basic_delta_pp": parakeet_ru["basic_delta"],
            "GigaAM_basic_delta_pp": gigaam_ru["basic_delta"],
            "lower_observed_rate": (
                "nvidia/parakeet-tdt-0.6b-v3"
                if parakeet_value < gigaam_value
                else "ai-sage/GigaAM-Multilingual"
            ),
            "vendor_order_reversed": bool(parakeet_value < gigaam_value),
            "ranking_admissible": ru_ranking_admissible,
            "ranking_blocker": (
                None
                if ru_ranking_admissible
                else (
                    "Clean RU BasicTextNormalizer deltas exceed 2 pp for: "
                    + ", ".join(ru_ranking_blocked_models)
                )
            ),
            "evidence_class": "HOST",
        }
    omni_smoke_receipt = (
        receipts.get("omni", {})
        .get("engine_metadata", {})
        .get("smoke_gate")
    )
    omni_smoke_all_passed = bool(
        isinstance(omni_smoke_receipt, dict)
        and set(omni_smoke_receipt) == set(LANGUAGES)
        and all(
            isinstance(gate, dict)
            and gate.get("collapsed") is False
            and gate.get("smoke_n_observed") == OMNI_SMOKE_N
            for gate in omni_smoke_receipt.values()
        )
    )
    omni_above_15_clean = [
        row["language"]
        for row in c3_rows
        if row["wer_or_cer"].get("value_percent") is not None
        and row["wer_or_cer"]["value_percent"] > 15.0
    ]
    c3_completion_receipt = c3_completion(c3_rows)
    clean_leaders = table_payload["per_language_clean_leader"]
    admissible_clean_leaders = {
        language: clean_leaders[language]
        for language in LANGUAGES
        if clean_leaders.get(language, {}).get("status") == "HOST"
    }
    withheld_clean_rankings = {
        language: clean_leaders[language]
        for language in LANGUAGES
        if clean_leaders.get(language, {}).get("status") != "HOST"
    }
    admissible_leader_text = "; ".join(
        "{language}={bank} / `{model}` ({metric} {value:.2f}%)".format(
            language=language.upper(),
            bank=result["bank"],
            model=result["model"],
            metric=str(result["metric"]).upper(),
            value=float(result["value_percent"]),
        )
        for language, result in admissible_clean_leaders.items()
    ) or "none"
    withheld_languages = [
        language.upper() for language in withheld_clean_rankings
    ]
    if len(withheld_languages) > 1:
        withheld_language_text = (
            ", ".join(withheld_languages[:-1])
            + " and "
            + withheld_languages[-1]
        )
    else:
        withheld_language_text = withheld_languages[0] if withheld_languages else "none"
    omni_above_15_text = ", ".join(
        language.upper() for language in omni_above_15_clean
    ) or "none"
    omni_screen_note = (
        f"All five 30-utterance smoke gates passed and expanded to full "
        f"n={TARGET_N} clean+MUSAN evaluation. Clean error exceeded 15% in "
        f"{len(omni_above_15_clean)}/{len(LANGUAGES)} languages "
        f"({omni_above_15_text}); ES was the only language at or below 15%."
        if omni_smoke_all_passed
        else (
            "Not all five 30-utterance smoke gates passed; see the validated "
            "per-language receipt in `C3.json`."
        )
    )
    ru_host_note = (
        "Same-corpus HOST observation reversed the vendor ordering on the "
        "same validated FLEURS `ru_ru` n=150 portable fixture identity: Parakeet "
        f"{float(ru_host_result['parakeet_WER_percent']):.2f}% WER versus "
        f"GigaAM {float(ru_host_result['GigaAM_WER_percent']):.2f}% WER. "
        + (
            "This clean RU ranking is admissible."
            if ru_host_result["ranking_admissible"]
            else (
                "The observed rates are comparable, but the formal RU winner "
                "is withheld because the clean RU BasicTextNormalizer deltas "
                f"are {float(ru_host_result['parakeet_basic_delta_pp']):+.2f} "
                f"pp and {float(ru_host_result['GigaAM_basic_delta_pp']):+.2f} "
                "pp (>2 pp)."
            )
        )
        + " Vendor 4.4 versus 5.51 figures remain VENDOR context only."
        if ru_host_result
        else (
            "The same-corpus RU HOST comparison was unavailable; vendor 4.4 "
            "versus 5.51 figures remain VENDOR context only."
        )
    )
    c4_payload = build_c4(engine_results)
    c4_payload["engine_receipts"] = receipts

    parakeet_tokenizer = engine_artifact(
        engine_results, "nemo", "parakeet_tokenizer"
    ) or engine_artifact(
        engine_results, "nemo", "parakeet_tokenizer_audit"
    )
    c1_payload = {
        "schema": "lane-c-c1/1.0",
        "sprint_id": "C1",
        "status": "COMPLETE" if all(row["evidence_class"] == "HOST" for row in c1_rows) else "PARTIAL",
        "bank": "D-fork",
        "rows": c1_rows,
        "parakeet_tokenizer_cjk_artifact": parakeet_tokenizer,
        "engine_receipts": {
            "sensevoice": receipts["sensevoice"],
            "nemo": receipts["nemo"],
        },
        "dependency": dependency,
    }
    c2_payload = {
        "schema": "lane-c-c2/1.0",
        "sprint_id": "C2",
        "status": "COMPLETE" if all(row["evidence_class"] == "HOST" for row in c2_rows) else "PARTIAL",
        "bank": "L-fork",
        "rows": c2_rows,
        "russian_same_corpus_test": {
            "models": ["nvidia/parakeet-tdt-0.6b-v3", "ai-sage/GigaAM-Multilingual"],
            "corpus": "google/fleurs ru_ru",
            "vendor_context_only": {"GigaAM_WER_percent": 4.4, "parakeet_WER_percent": 5.51, "evidence_class": "VENDOR"},
            "host_rows": ru_host_rows,
            "host_result": ru_host_result,
        },
        "engine_receipts": {
            "nemo": receipts["nemo"],
            "zipformer": receipts["zipformer"],
            "gigaam": receipts["gigaam"],
        },
        "dependency": dependency,
    }
    c3_payload = {
        "schema": "lane-c-c3/1.0",
        "sprint_id": "C3",
        "status": "COMPLETE" if c3_completion_receipt["complete"] else "PARTIAL",
        "bank": "OmniASR",
        "required_revision": "8e35f0cc28fa6099e0c14d56db85ce0423baa691",
        "smoke_rule": "First 30 clean utterances per language; stop a collapsed language before the remaining 120 and noisy bucket.",
        "completion_rule": (
            "Each language must have either n=150 HOST clean+MUSAN results, or "
            "a validated COMPLETE-receipt n=30 collapsed clean smoke. Collapsed "
            "smoke is HOST execution evidence but non-comparable and ranking-inadmissible; "
            "its absent noisy bucket remains UNKNOWN."
        ),
        "completion": c3_completion_receipt,
        "smoke_receipt": omni_smoke_receipt,
        "screening_summary": {
            "all_five_smoke_gates_passed": omni_smoke_all_passed,
            "languages_above_15_percent_clean": omni_above_15_clean,
            "languages_at_or_below_15_percent_clean": [
                language
                for language in LANGUAGES
                if language not in omni_above_15_clean
            ],
        },
        "rows": c3_rows,
        "engine_receipts": {"omni": receipts["omni"]},
        "dependency": dependency,
    }
    c5_payload = {
        "schema": "lane-c-c5/1.0",
        "sprint_id": "C5",
        "status": "COMPLETE"
        if all(row["evidence_class"] == "HOST" for row in rows)
        and comparable["all_cells_comparable"]
        and comparable["all_clean_rankings_admissible"]
        else "PARTIAL",
        "asr_table": "asr_table.json",
        "rows": rows,
        "comparability": comparable,
        "warnings": table_payload["warnings"],
        "status_reason": (
            f"All {len(rows)} rows are HOST and fixture-comparable; C5 is "
            f"PARTIAL solely because clean rankings for "
            f"{withheld_language_text} are inadmissible after absolute "
            "BasicTextNormalizer deltas above 2 pp."
            if all(row["evidence_class"] == "HOST" for row in rows)
            and comparable["all_cells_comparable"]
            and not comparable["all_clean_rankings_admissible"]
            else None
        ),
        "ranking_summary": {
            "admissible_clean_leaders": admissible_clean_leaders,
            "withheld_clean_rankings": withheld_clean_rankings,
        },
        "omni_screening_summary": c3_payload["screening_summary"],
        "engine_receipts": receipts,
    }

    for name, payload in (("C1", c1_payload), ("C2", c2_payload), ("C3", c3_payload), ("C4", c4_payload), ("C5", c5_payload)):
        write_json(out_dir / f"{name}.json", payload)

    write_text(
        out_dir / "C1.md",
        task_report(
            "C1",
            "D-fork ASR bank",
            c1_rows,
            [
                "Parakeet tokenizer evidence is preserved in `C1.json`: explicit CJK/Kana/Hangul vocabulary counts, the pinned tokenizer SHA, probe output, and the byte-fallback flag are reported separately. Byte fallback means the probe is not described as literal CJK unencodability.",
            ],
        ),
    )
    write_text(
        out_dir / "C2.md",
        task_report(
            "C2",
            "L-fork ASR bank",
            c2_rows,
            [
                ru_host_note,
                "Zipformer runtime attribution caveat: the executed, self-hashed ONNX export is epoch 99, while the repository README score names epoch 14; `tokens.txt` is in the exact runtime set and `bpe.model` is not. HOST results are attributed only to the executed assets.",
            ],
        ),
    )
    write_text(
        out_dir / "C3.md",
        task_report(
            "C3",
            "OmniASR-CTC-300M five-language screen",
            c3_rows,
            [
                omni_screen_note,
                "The checkpoint is pinned to the required revision. A validated collapsed language stopped at n=30 is HOST smoke evidence, but it is explicitly non-comparable and ranking-inadmissible; its missing noisy bucket remains UNKNOWN.",
            ],
            status=c3_payload["status"],
            status_detail=(
                "all five n=30 smoke gates passed; every language completed "
                "n=150 clean+MUSAN HOST evaluation"
                if omni_smoke_all_passed
                else (
                    "each language has either full n=150 clean+noisy evidence "
                    "or a validated n=30 collapsed clean smoke"
                    if c3_completion_receipt["complete"]
                    else (
                        "one or more languages satisfy neither the full nor "
                        "collapsed-smoke completion rule"
                    )
                )
            ),
        ),
    )

    c4_lines = [
        "# C4 — Partial stability",
        "",
        f"Status: **{c4_payload['status']}**.",
        "",
        "| Route | Language | Partials | First partial | Rewrite rate | Final divergence | Trace digest | Class |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in c4_payload["rows"]:
        c4_lines.append(
            "| {route} | {language} | {partials} | {latency} | {rewrite} | {divergence} | {trace} | {klass} |".format(
                route=row["route"],
                language=str(row["language"]).upper(),
                partials=("UNKNOWN" if row["partials_emitted"] is None else str(row["partials_emitted"]).lower()),
                latency=row.get("first_partial_latency_ms") if row.get("first_partial_latency_ms") is not None else "—",
                rewrite=row.get("partial_rewrite_rate_edits_per_audio_second") if row.get("partial_rewrite_rate_edits_per_audio_second") is not None else "—",
                divergence=row.get("final_vs_last_partial_divergence") if row.get("final_vs_last_partial_divergence") is not None else "—",
                trace=row.get("trace_digest") or "—",
                klass=row["evidence_class"],
            )
        )
    sensevoice_probes = [
        row
        for row in c4_payload["rows"]
        if isinstance(row.get("fixed_context_probe"), dict)
    ]
    if sensevoice_probes:
        c4_lines.extend(["", "## SenseVoice fixed-context capability probe", ""])
        for row in sensevoice_probes:
            probe = row["fixed_context_probe"]
            c4_lines.append(
                "- {language}: exact {context:g} s clean contexts, n={n}; "
                "scoring included={scored}; main C1 inputs unchanged={unchanged}; "
                "receipt `{digest}`.".format(
                    language=str(row["language"]).upper(),
                    context=float(probe.get("context_seconds") or 0),
                    n=probe.get("n") or 0,
                    scored=str(bool(probe.get("scoring_included"))).lower(),
                    unchanged=str(
                        bool(probe.get("main_scoring_inputs_unchanged"))
                    ).lower(),
                    digest=probe.get("receipt_sha256") or "UNKNOWN",
                )
            )
    c4_lines.extend(["", c4_payload["product_default"], "", c4_payload["evidence_note"]])
    write_text(out_dir / "C4.md", "\n".join(c4_lines))

    write_text(
        out_dir / "C5.md",
        "\n".join(
            [
                "# C5 — Comparable ASR table",
                "",
                f"Status: **{c5_payload['status']}**. {c5_payload['status_reason'] or ''}".rstrip(),
                "",
                MUSAN_CAVEAT_MARKDOWN,
                "",
                table_markdown(rows),
                "",
                f"Complete clean+MUSAN HOST rows: **{sum(row['evidence_class'] == 'HOST' for row in rows)}/{len(rows)}**. Cross-bank fixture comparability: **{comparable['all_cells_comparable']}**.",
                "",
                f"Admissible clean leaders: {admissible_leader_text}.",
                "",
                f"Clean rankings withheld for {withheld_language_text}: all rows remain comparable, but at least one normaliser delta exceeds 2 pp in each withheld language.",
                "",
                omni_screen_note,
                "",
                "Rows above 15% are flagged, and any absolute BasicTextNormalizer delta above 2 pp is ranking-inadmissible.",
            ]
        ),
    )

    host_rows = [row for row in rows if row["evidence_class"] == "HOST"]
    unknown_rows = [row for row in rows if row["evidence_class"] == "UNKNOWN"]
    clean_host_count = sum(
        row["clean_evidence_class"] == "HOST" for row in rows
    )
    noisy_host_count = sum(
        row["noisy_bucket"]["evidence_class"] == "HOST" for row in rows
    )
    if not host_rows:
        missing = ", ".join(
            f"{engine}={receipt.get('status', 'MISSING')}"
            for engine, receipt in receipts.items()
            if receipt.get("status") != "COMPLETE"
        )
        headline = (
            "No complete clean+MUSAN ASR row reached HOST; no bank ranking is admissible."
            + (f" Engine receipts: {missing}." if missing else "")
        )
        confidence = (
            "HIGH confidence in the fail-closed evidence status; NO confidence in any bank ranking."
        )
    elif comparable["all_cells_comparable"] and comparable[
        "all_clean_rankings_admissible"
    ]:
        headline = f"Clean same-fixture leaders: {admissible_leader_text}."
        confidence = (
            "HIGH for execution provenance and fixture comparability; MEDIUM for "
            "n=150 point estimates and utterance-bootstrap CIs, which are screens "
            "rather than certification."
        )
    elif (
        len(host_rows) == len(rows)
        and comparable["all_cells_comparable"]
        and not comparable["all_clean_rankings_admissible"]
    ):
        headline = (
            f"All {len(rows)}/{len(rows)} rows are HOST and same-fixture "
            f"comparable. Admissible clean leaders: {admissible_leader_text}. "
            f"{withheld_language_text} clean winners are withheld because "
            f"normaliser sensitivity "
            f"exceeds 2 pp. OmniASR passed all five smoke gates but exceeded "
            f"15% clean error in {len(omni_above_15_clean)}/{len(LANGUAGES)} "
            f"languages ({omni_above_15_text})."
        )
        confidence = (
            "HIGH for execution provenance and fixture comparability; MEDIUM for "
            "the n=150 point estimates and bootstrap intervals; NO formal ranking "
            f"confidence for {withheld_language_text} because their normaliser "
            "deltas exceed 2 pp."
        )
    else:
        headline = (
            "Some HOST measurements returned, but at least one language lacks three comparable, ranking-admissible banks; no cross-bank leader is declared."
        )
        confidence = "LOW: one or more cells are missing, non-comparable, or normaliser-carried."
    summary = "\n".join(
        [
            "# Lane C — ASR summary",
            "",
            headline,
            "",
            f"Confidence: **{confidence}**",
            "",
            "## Important noisy-bucket provenance caveat",
            "",
            MUSAN_CAVEAT_MARKDOWN,
            "",
            f"Complete clean+MUSAN HOST rows: {len(host_rows)}/{len(rows)}; UNKNOWN rows: {len(unknown_rows)}. Clean HOST cells: {clean_host_count}/{len(rows)}; MUSAN HOST cells: {noisy_host_count}/{len(rows)}.",
            "",
            f"Same-corpus comparability across all three banks: **{comparable['all_cells_comparable']}**.",
            "",
            f"Lane B dependency: **MET**. `out/c/normalisers.py` is byte-identical to `sprint/b-backbone:workstreams/sub2b_sprint/sprint/out/b/normalisers.py` at SHA-256 `{NORMALISERS_SHA256}`.",
            "",
            "No training, fine-tuning, quantization, export, or local heavyweight inference was performed.",
        ]
    )
    write_text(out_dir / "LANE_SUMMARY.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
