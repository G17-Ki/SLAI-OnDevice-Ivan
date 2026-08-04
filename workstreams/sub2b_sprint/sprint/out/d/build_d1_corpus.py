#!/usr/bin/env python3
"""Build Lane D's deterministic 40-item matched-human FLEURS corpus.

The five FLEURS configurations are parallel translations.  We select a shared set
of sentence IDs from the held-out test split, then retain one natural recording
for each selected ID and language.  Large Parquet files are read one language at
a time; only the selected WAVs and a compact manifest are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq
import soundfile as sf


DATASET = "google/fleurs"
PARQUET_SOURCE_REF = "refs/convert/parquet"
PARQUET_URL_REF = "refs%2Fconvert%2Fparquet"
CONFIGS = {
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "es": "es_419",
    "fr": "fr_fr",
    "ru": "ru_ru",
}
TARGETS = {
    "question": 1,
    "numbers": 8,
    "dates": 6,
    "names": 8,
    "acronyms": 6,
    "long_clause": 10,
}


def parquet_url(config: str) -> str:
    return (
        "https://huggingface.co/datasets/google/fleurs/resolve/"
        f"{PARQUET_URL_REF}/{config}/test/0000.parquet"
    )


def load_metadata(lang: str, config: str) -> dict[int, dict[str, Any]]:
    url = parquet_url(config)
    with fsspec.open(url, "rb", block_size=8 << 20, cache_type="readahead").open() as fh:
        rows = pq.read_table(
            fh,
            columns=["id", "path", "transcription", "raw_transcription"],
        ).to_pylist()

    # FLEURS can contain multiple recordings of one semantic sentence.  The
    # first row is retained deterministically; row_index is needed for audio.
    by_id: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        sentence_id = int(row["id"])
        if sentence_id not in by_id:
            by_id[sentence_id] = {
                "row_index": row_index,
                "path": row["path"],
                "transcription": row["transcription"],
                "raw_transcription": row["raw_transcription"],
            }
    print(f"metadata {lang}: {len(rows)} rows, {len(by_id)} unique IDs", flush=True)
    return by_id


def classify(texts: dict[str, str]) -> list[str]:
    joined = "\n".join(texts.values())
    en = texts["en"]
    categories: list[str] = []

    if "?" in joined or "？" in joined:
        categories.append("question")
    if re.search(r"\d", joined):
        categories.append("numbers")
    named_month = re.search(
        r"\b(?:January|February|March|April|June|July|August|September|"
        r"October|November|December)\b",
        en,
    ) or re.search(r"\b(?:in|on|late|early|by|until|from)\s+May\b", en)
    if re.search(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)", joined) or named_month:
        categories.append("dates")
    if re.search(r"(?<!\w)(?:[A-ZА-ЯЁ]{2,}|(?:[A-ZА-ЯЁ]\.){2,})(?!\w)", joined):
        categories.append("acronyms")

    title_words = []
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", en):
        prefix = en[: match.start()].rstrip()
        # A capital at the start of a sentence is grammar, not name evidence.
        if not prefix or prefix[-1] in ".!?":
            continue
        title_words.append(match.group())
    if title_words:
        categories.append("names")

    punctuation = sum(en.count(mark) for mark in [",", ";", ":", "(", ")"])
    clause_words = re.search(
        r"\b(?:although|because|which|while|whose|whereas|after|before|despite|"
        r"unless|when|that)\b",
        en,
        re.IGNORECASE,
    )
    if (len(en) >= 155 and punctuation >= 2) or (len(en) >= 135 and clause_words):
        categories.append("long_clause")
    return categories


def stable_tiebreak(sentence_id: int) -> int:
    digest = hashlib.sha256(f"lane-d-fleurs-v1:{sentence_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_ids(all_meta: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    common = set.intersection(*(set(rows) for rows in all_meta.values()))
    candidates: list[dict[str, Any]] = []
    for sentence_id in sorted(common):
        texts = {
            lang: all_meta[lang][sentence_id]["raw_transcription"]
            for lang in CONFIGS
        }
        en_len = len(texts["en"])
        # Avoid tiny fragments and clips likely to breach the 30 s safety bound.
        if not 38 <= en_len <= 270:
            continue
        candidates.append(
            {
                "sentence_id": sentence_id,
                "categories": classify(texts),
                "texts": texts,
                "en_length": en_len,
                "stable_tiebreak": stable_tiebreak(sentence_id),
            }
        )

    selected: list[dict[str, Any]] = []
    chosen: set[int] = set()
    counts: Counter[str] = Counter()

    # Meet each requested coverage target.  Prefer items that close several
    # deficits at once and are near a comfortable listening length.
    while any(counts[name] < target for name, target in TARGETS.items()):
        unmet = {name for name, target in TARGETS.items() if counts[name] < target}
        pool = [c for c in candidates if c["sentence_id"] not in chosen and unmet.intersection(c["categories"])]
        if not pool:
            break
        pool.sort(
            key=lambda c: (
                -len(unmet.intersection(c["categories"])),
                abs(c["en_length"] - 145),
                c["stable_tiebreak"],
            )
        )
        item = pool[0]
        selected.append(item)
        chosen.add(item["sentence_id"])
        counts.update(item["categories"])

    # Fill to 40 with a deterministic mix, favouring moderate-length clips and
    # category-bearing items without duplicating semantic sentence IDs.
    remainder = [c for c in candidates if c["sentence_id"] not in chosen]
    remainder.sort(
        key=lambda c: (
            0 if c["categories"] else 1,
            abs(c["en_length"] - 120),
            c["stable_tiebreak"],
        )
    )
    for item in remainder:
        if len(selected) >= 40:
            break
        selected.append(item)
        chosen.add(item["sentence_id"])
        counts.update(item["categories"])

    if len(selected) != 40:
        raise RuntimeError(f"selected {len(selected)} items, expected 40")
    for name, target in TARGETS.items():
        if counts[name] < target:
            raise RuntimeError(f"coverage target not met: {name}={counts[name]} < {target}")

    # Stable public IDs are independent of selection order internals.
    selected.sort(key=lambda item: item["sentence_id"])
    for index, item in enumerate(selected, 1):
        item["item_id"] = f"fleurs_{index:02d}"
        item.pop("stable_tiebreak", None)
    return selected


def extract_audio(
    lang: str,
    config: str,
    selected: list[dict[str, Any]],
    meta: dict[int, dict[str, Any]],
    audio_root: Path,
) -> list[dict[str, Any]]:
    wanted = {meta[item["sentence_id"]]["row_index"]: item for item in selected}
    lang_dir = audio_root / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    existing = [lang_dir / f"{item['item_id']}_human.wav" for item in selected]
    if all(path.is_file() and path.stat().st_size > 44 for path in existing):
        print(f"audio {lang}: reusing {len(existing)} retained WAVs", flush=True)
        rows = None
    else:
        url = parquet_url(config)
        temp_dir = audio_root.parent / "_parquet_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_parquet = temp_dir / f"{config}-test.parquet"
        print(f"audio {lang}: resumable download {url}", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "8",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--silent",
                "--show-error",
                "--output",
                str(local_parquet),
                url,
            ],
            check=True,
        )
        try:
            rows = pq.read_table(local_parquet, columns=["audio"]).column("audio").to_pylist()
        finally:
            local_parquet.unlink(missing_ok=True)

    records: list[dict[str, Any]] = []
    for row_index, item in sorted(wanted.items(), key=lambda pair: pair[1]["item_id"]):
        out = lang_dir / f"{item['item_id']}_human.wav"
        if rows is not None:
            audio = rows[row_index]
            payload = audio["bytes"]
            if not payload:
                raise RuntimeError(f"empty embedded audio for {lang} row {row_index}")
            out.write_bytes(payload)
        else:
            payload = out.read_bytes()
        info = sf.info(out)
        records.append(
            {
                "item_id": item["item_id"],
                "sentence_id": item["sentence_id"],
                "relative_path": str(out.relative_to(audio_root.parents[2])),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sample_rate_hz": info.samplerate,
                "frames": info.frames,
                "duration_s": round(info.duration, 6),
                "channels": info.channels,
                "source_row_index": row_index,
            }
        )
    print(f"audio {lang}: retained {len(records)} WAVs", flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Tracked corpus manifest JSON")
    parser.add_argument("--audio-root", type=Path, required=True, help="Ignored human-audio directory")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    all_meta = {lang: load_metadata(lang, config) for lang, config in CONFIGS.items()}
    selected = select_ids(all_meta)
    coverage = Counter(cat for item in selected for cat in item["categories"])

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Lane D D1 matched-human ASR backcheck corpus",
        "dataset": DATASET,
        "dataset_source_ref": PARQUET_SOURCE_REF,
        "dataset_source_ref_status": "MUTABLE_REF_NOT_RESOLVED_OR_HASHED_AT_FETCH",
        "source_parquet_hashes_retained": False,
        "local_evidence_integrity": (
            "The committed manifest and every retained human WAV are SHA-256 pinned; "
            "the source Parquet blobs were deleted without retained hashes."
        ),
        "split": "test",
        "selection_method": "deterministic shared-ID stratification; first recording per semantic sentence ID",
        "target_count_per_language": 40,
        "coverage_targets": TARGETS,
        "coverage_achieved": dict(sorted(coverage.items())),
        "languages": CONFIGS,
        "items": selected,
        "human_audio": {},
    }

    if not args.metadata_only:
        for lang, config in CONFIGS.items():
            manifest["human_audio"][lang] = extract_audio(
                lang,
                config,
                selected,
                all_meta[lang],
                args.audio_root,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": len(selected), "coverage": coverage}, default=dict), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
