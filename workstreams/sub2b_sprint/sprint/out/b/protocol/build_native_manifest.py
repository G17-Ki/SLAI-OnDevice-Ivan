#!/usr/bin/env python3
"""Combine and validate Lane B's frozen 40-item native prompt sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


LANGUAGE_ORDER = ("en", "zh", "es", "fr", "ru")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(en_path: Path, es_fr_path: Path, ru_zh_path: Path) -> dict[str, list[dict]]:
    en = json.loads(en_path.read_text(encoding="utf-8"))
    es_fr = json.loads(es_fr_path.read_text(encoding="utf-8"))
    ru_zh = json.loads(ru_zh_path.read_text(encoding="utf-8"))
    return {
        "en": en["items"],
        "zh": ru_zh["zh_hans"],
        "es": es_fr["es"],
        "fr": es_fr["fr"],
        "ru": ru_zh["ru"],
    }


def iter_regexes(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_regexes(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if "regex" in key or key in {
                "required_all",
                "required_any",
                "forbidden_any",
                "required_any_groups",
            }:
                yield from iter_regexes(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument("--es-fr", type=Path, required=True)
    parser.add_argument("--ru-zh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    by_language = load_inputs(args.en, args.es_fr, args.ru_zh)
    all_ids: set[str] = set()
    validation: dict[str, dict] = {}
    records: list[dict] = []

    for language in LANGUAGE_ORDER:
        items = by_language[language]
        if len(items) != 40:
            raise RuntimeError(f"{language}: expected 40 items, got {len(items)}")
        categories = Counter()
        for ordinal, item in enumerate(items, start=1):
            item = dict(item)
            item_id = item["id"]
            if item_id in all_ids:
                raise RuntimeError(f"duplicate item id: {item_id}")
            all_ids.add(item_id)
            source_language = item.get("language")
            if language == "zh":
                if source_language not in {"zh", "zh-Hans"}:
                    raise RuntimeError(f"{item_id}: unexpected language {source_language}")
                item["language"] = "zh"
            elif source_language != language:
                raise RuntimeError(f"{item_id}: unexpected language {source_language}")

            rubric = item.pop("deterministic_rubric", item.get("rubric"))
            if rubric is None:
                raise RuntimeError(f"{item_id}: missing rubric")
            item.pop("rubric", None)
            for pattern in iter_regexes(rubric):
                try:
                    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
                except re.error as exc:
                    raise RuntimeError(f"{item_id}: invalid regex {pattern!r}: {exc}") from exc

            category = item["category"]
            categories[category] += 1
            record = {
                "item_id": item_id,
                "ordinal_within_language": ordinal,
                "language": language,
                "category": category,
                "prompt": item["prompt"],
                "rubric": rubric,
                "max_words": item.get("max_words"),
                "max_chars": item.get("max_chars"),
                "source": "frozen_codex_authored_native_language_screen",
            }
            records.append(record)
        validation[language] = {
            "count": len(items),
            "category_counts_as_authored": dict(sorted(categories.items())),
            "max_words": 60 if language != "zh" else None,
            "max_chars": 90 if language == "zh" else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "native_generation_40x5.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    receipt = {
        "schema_version": "lane-b-native-prompts/1.0",
        "languages": list(LANGUAGE_ORDER),
        "items_per_language": 40,
        "total_items": len(records),
        "authoring_note": (
            "Prompts were authored directly in each target language by Codex subagents to a "
            "shared category blueprint. They were not exposed to candidate-model outputs. "
            "No native-human authorship or validation is claimed."
        ),
        "validation": validation,
        "manifest_path": output_path.name,
        "manifest_sha256": sha256_file(output_path),
        "source_file_sha256": {
            "en": sha256_file(args.en),
            "es_fr": sha256_file(args.es_fr),
            "ru_zh": sha256_file(args.ru_zh),
        },
        "regex_engine_for_run": "regex==2026.7.19, VERSION1 Unicode mode",
    }
    (args.output_dir / "native_generation_selection.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
