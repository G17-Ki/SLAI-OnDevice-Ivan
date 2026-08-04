#!/usr/bin/env python3
"""Build Lane B's frozen 80-item-per-language Global-MMLU screen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb


DATASET_ID = "CohereLabs/Global-MMLU"
DATASET_REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
SEED = 1729
LANGUAGES = ("en", "zh", "es", "fr", "ru")
QUOTAS = {
    ("Business", "CA"): 8,
    ("Business", "CS"): 2,
    ("Humanities", "CA"): 10,
    ("Humanities", "CS"): 10,
    ("Medical", "CA"): 9,
    ("Medical", "CS"): 1,
    ("Other", "CA"): 5,
    ("Other", "CS"): 5,
    ("STEM", "CA"): 14,
    ("STEM", "CS"): 1,
    ("Social Sciences", "CA"): 8,
    ("Social Sciences", "CS"): 7,
}


def stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{sample_id}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    conn = duckdb.connect()
    columns = [
        "sample_id",
        "subject",
        "subject_category",
        "question",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "answer",
        "required_knowledge",
        "time_sensitive",
        "reference",
        "culture",
        "region",
        "country",
        "cultural_sensitivity_label",
        "is_annotated",
    ]
    relation = conn.execute(
        f"SELECT {', '.join(columns)} FROM read_parquet(?)", [str(path)]
    )
    return [dict(zip(columns, row, strict=True)) for row in relation.fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows_by_language = {
        language: load_rows(args.parquet_dir / f"{language}.parquet")
        for language in LANGUAGES
    }
    english = rows_by_language["en"]
    eligible: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in english:
        key = (row["subject_category"], row["cultural_sensitivity_label"])
        if (
            row["is_annotated"]
            and key in QUOTAS
            and "'Yes'" not in row["time_sensitive"]
        ):
            eligible[key].append(row)

    selected_ids: list[str] = []
    quota_receipt: list[dict] = []
    for key, quota in QUOTAS.items():
        candidates = sorted(eligible[key], key=lambda row: stable_key(row["sample_id"]))
        if len(candidates) < quota:
            raise RuntimeError(f"insufficient eligible rows for {key}: {len(candidates)} < {quota}")
        chosen = candidates[:quota]
        selected_ids.extend(row["sample_id"] for row in chosen)
        quota_receipt.append(
            {
                "subject_category": key[0],
                "cultural_sensitivity_label": key[1],
                "eligible": len(candidates),
                "selected": quota,
            }
        )

    if len(selected_ids) != 80 or len(set(selected_ids)) != 80:
        raise RuntimeError("selection is not exactly 80 unique sample ids")
    selected_ids = sorted(selected_ids, key=stable_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "global_mmlu_80x5.jsonl"
    language_counts: dict[str, int] = {}
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as output:
        for language in LANGUAGES:
            index = {row["sample_id"]: row for row in rows_by_language[language]}
            missing = [sample_id for sample_id in selected_ids if sample_id not in index]
            if missing:
                raise RuntimeError(f"{language} is missing selected ids: {missing[:3]}")
            language_counts[language] = len(selected_ids)
            for sample_id in selected_ids:
                row = index[sample_id]
                record = {
                    "item_id": f"gmmlu/{language}/{sample_id}",
                    "language": language,
                    "dataset_id": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "split": "test",
                    **row,
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    answer_mismatches = []
    for sample_id in selected_ids:
        answers = {
            language: next(
                row["answer"]
                for row in rows_by_language[language]
                if row["sample_id"] == sample_id
            )
            for language in LANGUAGES
        }
        if len(set(answers.values())) != 1:
            answer_mismatches.append({"sample_id": sample_id, "answers": answers})
    if answer_mismatches:
        raise RuntimeError(f"cross-language answer mismatches: {answer_mismatches[:3]}")

    receipt = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "test",
        "seed": SEED,
        "selection_algorithm": (
            "Filter is_annotated=true, cultural_sensitivity_label in {CA,CS}, and "
            "time_sensitive contains no 'Yes'; group by the frozen category/label quotas; "
            "sort each group by sha256('1729\\0' + sample_id); take the first quota rows."
        ),
        "paired_across_languages": True,
        "languages": list(LANGUAGES),
        "language_counts": language_counts,
        "quota_receipt": quota_receipt,
        "selected_sample_ids_in_stable_order": selected_ids,
        "manifest_path": str(jsonl_path.name),
        "manifest_sha256": sha256_file(jsonl_path),
        "total_records": sum(language_counts.values()),
        "answer_mismatch_count": 0,
    }
    (args.output_dir / "global_mmlu_selection.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
