#!/usr/bin/env python3
"""Validate Lane D's committed reports against the retained artifact bundle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


OUT_D = Path(__file__).resolve().parent
MAX_BYTES = 3_000_000_000
ALLOWED_EVIDENCE_CLASSES = {
    "EXACT",
    "UB",
    "VENDOR",
    "BENCHMARK",
    "HOST",
    "AIHUB-PROFILED",
    "PRIOR",
    "PROJECTED",
    "UNKNOWN",
}


def load(name: str) -> dict:
    return json.loads((OUT_D / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_evidence_classes(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.endswith("evidence_class"):
                assert value in ALLOWED_EVIDENCE_CLASSES, (key, value)
            elif key == "evidence_classes":
                assert isinstance(value, list) and value
                assert set(value).issubset(ALLOWED_EVIDENCE_CLASSES), value
            assert_evidence_classes(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_evidence_classes(value)


def main() -> int:
    d1 = load("D1.json")
    d2 = load("D2.json")
    d3 = load("D3.json")
    d4 = load("D4.json")
    d5 = load("D5.json")
    table = load("tts_table.json")
    prompts = load("D5_prompts.json")
    listen = (OUT_D / "LISTEN.md").read_text(encoding="utf-8")

    assert d1["headline"]["complete_engine_language_comparisons"] == 12
    assert not d1["headline"]["th13_gate_cleared_by_any_cell"]
    assert all(row["n"] == 40 for row in d1["comparisons"])
    assert d1["synthesis_inventory"]["valid_wavs_written"] == 513
    assert d1["synthesis_inventory"]["hard_failures"] == 7
    assert d1["protocol"]["corpus"]["dataset_source_ref"] == "refs/convert/parquet"
    assert (
        d1["protocol"]["corpus"]["dataset_source_ref_status"]
        == "MUTABLE_REF_NOT_RESOLVED_OR_HASHED_AT_FETCH"
    )
    assert not d1["protocol"]["corpus"]["source_parquet_hashes_retained"]
    matcha_probe_meta = d1["artifacts"]["matcha_frontend_probe"]
    matcha_probe_path = OUT_D / matcha_probe_meta["path"]
    matcha_probe = json.loads(matcha_probe_path.read_text(encoding="utf-8"))
    assert matcha_probe_path.stat().st_size == matcha_probe_meta["bytes"]
    assert sha256_file(matcha_probe_path) == matcha_probe_meta["sha256"]
    assert len(matcha_probe["cases"]) == matcha_probe_meta["case_count"] == 14
    assert all(row["passed_waveform_assertions"] for row in matcha_probe["cases"])
    assert {
        "Wild",
        "Card",
        "SANParks",
        "CPU",
        "RAM",
        "GB",
        "API",
        "JSON",
        "VPN",
    }.issubset(set(matcha_probe["ignored_oov_tokens_in_first_seen_order"]))

    assert d2["headline"]["silero_v5_cis_base_parameters"]["value"] == 20_926_952
    assert not d2["temporary_cache_lifecycle"]["present_at_finalization"]
    assert d4["headline"]["canonical_first_chunk_ms"]["evidence_class"] == "HOST"

    assert d3["headline"]["attempted"] == 96
    assert d3["headline"]["passed"] == 90
    assert d3["headline"]["failed"] == 6
    assert len(d3["routes"]) == 15

    inventory = d5["inventory"]
    assert inventory["expected_wav_count"] == 280
    assert inventory["existing_wav_count"] == 278
    assert inventory["missing_expected_count"] == 2
    assert inventory["extra_unexpected_count"] == 0
    assert inventory["invalid_filename_count"] == 0
    assert inventory["all_existing_wavs_readable"]
    assert inventory["all_existing_wavs_pass_waveform_gate"]

    listening_dir = OUT_D / "artifacts" / "listening"
    wavs = sorted(listening_dir.glob("*.wav"))
    assert len(wavs) == 278
    assert sum(path.stat().st_size for path in wavs) == inventory["total_wav_bytes"]
    lines = "".join(
        f"{path.name}\t{sha256_file(path)}\t{path.stat().st_size}\n" for path in wavs
    )
    assert hashlib.sha256(lines.encode("utf-8")).hexdigest() == inventory[
        "clip_hash_manifest_sha256"
    ]

    expected = []
    matrix = {
        "en": ("kokoro", "piper", "supertonic"),
        "zh": ("kokoro", "matcha"),
        "es": ("kokoro", "piper", "supertonic"),
        "fr": ("kokoro", "piper", "supertonic"),
        "ru": ("piper", "silero", "supertonic"),
    }
    for language, engines in matrix.items():
        assert len(prompts["languages"][language]) == 20
        for engine in engines:
            expected.extend(f"{language}_{engine}_{index:02d}.wav" for index in range(1, 21))
    present = {path.name for path in wavs}
    assert set(expected) - present == {"zh_kokoro_09.wav", "zh_kokoro_14.wav"}
    assert present - set(expected) == set()

    linked = re.findall(r"\]\((artifacts/listening/[^)]+\.wav)\)", listen)
    assert len(linked) == 278
    assert len(set(linked)) == 278
    assert {Path(link).name for link in linked} == present
    assert all((OUT_D / link).is_file() for link in linked)
    assert "zh_kokoro_09.wav" not in linked
    assert "zh_kokoro_14.wav" not in linked
    assert listen.count("non-finite output; no WAV") == 2

    assert table["status"] == "NO_SINGLE_SAFE_COMMERCIALLY_UNBLOCKED_FIVE_LANGUAGE_ENGINE"
    assert not table["headline"]["single_engine_safe_all_five"]
    assert not table["headline"]["single_engine_commercially_unblocked_all_five"]
    assert table["headline"]["best_supported_sprint_routes"]["zh"] == []

    for payload in (d1, d2, d3, d4, d5, table):
        assert_evidence_classes(payload)

    total_bytes = sum(path.stat().st_size for path in OUT_D.rglob("*") if path.is_file())
    assert total_bytes < MAX_BYTES, (total_bytes, MAX_BYTES)
    print(
        json.dumps(
            {
                "status": "PASS",
                "d1_comparisons": 12,
                "d3": {"attempted": 96, "passed": 90, "failed": 6},
                "d5_wavs": 278,
                "d5_bytes": inventory["total_wav_bytes"],
                "out_d_total_bytes": total_bytes,
                "storage_cap_bytes": MAX_BYTES,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
