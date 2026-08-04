#!/usr/bin/env python3
"""Extract a deterministic seven-language Common Voice ASR fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path


DATASET_REPO = "argmaxinc/common_voice_17_0-debug-zip"
DATASET_REVISION = "e77e410d41a77f733efbaf7307bd0ff7ebd6d0af"
EXPECTED_ARCHIVE_SHA256 = (
    "4dc5cd3f2f65f0ede599dc778b893d25baa9efb97570d114bf6085ea86208c57"
)
EXPECTED_METADATA_SHA256 = (
    "826476a68f5f90d238bc72e894ff1dfe4a0233bc2c869d4573db3f5325627d70"
)
DEFAULT_LANGUAGES = ("en", "es", "de", "fr", "ja", "ko", "zh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--allow-checksum-mismatch", action="store_true")
    args = parser.parse_args()

    archive = args.archive.resolve()
    metadata_path = args.metadata.resolve()
    output_dir = args.output_dir.resolve()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    archive_sha = sha256(archive)
    metadata_sha = sha256(metadata_path)
    if not args.allow_checksum_mismatch:
        if archive_sha != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(
                f"archive SHA-256 mismatch: expected {EXPECTED_ARCHIVE_SHA256}, "
                f"got {archive_sha}"
            )
        if metadata_sha != EXPECTED_METADATA_SHA256:
            raise ValueError(
                f"metadata SHA-256 mismatch: expected {EXPECTED_METADATA_SHA256}, "
                f"got {metadata_sha}"
            )

    metadata = json.loads(metadata_path.read_text())
    selected = [row for row in metadata if row["language"] in args.languages]
    by_language = {language: [] for language in args.languages}
    for row in selected:
        by_language[row["language"]].append(row)
    wrong_counts = {
        language: len(rows)
        for language, rows in by_language.items()
        if len(rows) != 3
    }
    if wrong_counts:
        raise ValueError(f"expected exactly three clips per language: {wrong_counts}")

    sample_records = []
    with zipfile.ZipFile(archive) as source_zip:
        members = set(source_zip.namelist())
        for language in args.languages:
            for row in by_language[language]:
                member = row["audio"]
                if member not in members:
                    raise FileNotFoundError(f"archive member is missing: {member}")
                destination = samples_dir / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source_zip.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                sample_records.append(
                    {
                        "sample_id": Path(member).stem,
                        "language": language,
                        "common_voice_locale": row["cv_17_0_language"],
                        "reference": row["text"],
                        "duration_metadata_s": row["duration"],
                        "archive_member": member,
                        "audio_path": destination.relative_to(output_dir).as_posix(),
                        "audio_bytes": destination.stat().st_size,
                        "audio_sha256": sha256(destination),
                    }
                )

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": DATASET_REVISION,
            "source_release": "Mozilla Common Voice Corpus 17.0",
            "license": "CC0-1.0",
            "archive_sha256": archive_sha,
            "metadata_sha256": metadata_sha,
        },
        "selection": {
            "method": "all three debug clips for each requested language",
            "languages": args.languages,
            "sample_count": len(sample_records),
        },
        "samples": sample_records,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
