#!/usr/bin/env python3
"""Frozen five-language normalisers shared by sprint lanes B and C.

Public normalisers return ``(normalised_text, applied_rules)``.  Scoring helpers
retain substitutions, insertions, and deletions instead of publishing only a
corpus mean.  The project normaliser is the ranking path; Whisper's
``BasicTextNormalizer`` is exposed only as the mandatory delta control.

Runtime dependencies for the exact path:

    pip install opencc-python-reimplemented==0.1.7 whisper-normalizer==0.1.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence


SEMVER = "1.0.0"
LANGUAGES = ("en", "zh", "es", "fr", "ru")
_SPACE_RE = re.compile(r"\s+")
_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
    (0x30000, 0x323AF),
)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


PACKAGE_SHA256 = _source_sha256()


def normalizer_id(language: str) -> str:
    language = _canonical_language(language)
    return f"norm_{language}/{SEMVER}+{PACKAGE_SHA256[:8]}"


def basic_normalizer_id() -> str:
    return "whisper_basic/0.1.15/default"


def _canonical_language(language: str) -> str:
    language = language.lower().replace("_", "-")
    aliases = {"cmn": "zh", "zh-cn": "zh", "zh-hans": "zh", "eng": "en", "spa": "es", "fra": "fr", "fre": "fr", "rus": "ru"}
    language = aliases.get(language, language)
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language {language!r}; expected one of {LANGUAGES}")
    return language


def _ensure_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"transcript must be str, got {type(text).__name__}")
    return text


@lru_cache(maxsize=None)
def _require_distribution(distribution: str, expected_version: str) -> None:
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"missing frozen dependency {distribution}=={expected_version}; "
            "install the exact version rather than using a fallback"
        ) from exc
    if installed != expected_version:
        raise RuntimeError(
            f"frozen dependency mismatch for {distribution}: expected {expected_version}, got {installed}"
        )


def _nfkc(text: str, rules: list[str]) -> str:
    result = unicodedata.normalize("NFKC", text)
    if result != text:
        rules.append("unicode_nfkc")
    return result


def _lower(text: str, rules: list[str]) -> str:
    result = text.lower()
    if result != text:
        rules.append("lowercase")
    return result


def _collapse_spaces(text: str, rules: list[str], *, remove_all: bool = False) -> str:
    if remove_all:
        result = "".join(text.split())
        if result != text:
            rules.append("strip_all_whitespace")
        return result
    result = _SPACE_RE.sub(" ", text).strip()
    if result != text:
        rules.append("collapse_whitespace")
    return result


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _strip_word_punctuation(
    text: str,
    rules: list[str],
    *,
    keep_apostrophe: bool = False,
    keep_hyphen: bool = False,
) -> str:
    output: list[str] = []
    punctuation_changed = False
    symbols_separated = False
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        previous_is_letter = index > 0 and _is_letter(text[index - 1])
        next_is_letter = index + 1 < len(text) and _is_letter(text[index + 1])
        if char == "'" and keep_apostrophe and previous_is_letter and next_is_letter:
            output.append(char)
        elif char in {"-", "‐", "‑"} and keep_hyphen and previous_is_letter and next_is_letter:
            output.append("-")
            punctuation_changed = punctuation_changed or char != "-"
        elif category.startswith("P"):
            output.append(" ")
            punctuation_changed = True
        elif category.startswith("S"):
            output.extend((" ", char, " "))
            symbols_separated = True
        else:
            output.append(char)
    if punctuation_changed:
        rules.append("strip_punctuation")
    if symbols_separated:
        rules.append("preserve_symbols_as_tokens")
    return "".join(output)


def _strip_zh_punctuation(text: str, rules: list[str]) -> str:
    result = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    if result != text:
        rules.append("strip_cjk_ascii_punctuation")
    return result


def _opencc_t2s(text: str) -> str:
    _require_distribution("opencc-python-reimplemented", "0.1.7")
    try:
        from opencc import OpenCC
    except ImportError as exc:  # pragma: no cover - exercised only in dependency-poor callers
        raise RuntimeError(
            "ZH normalization requires opencc-python-reimplemented==0.1.7; "
            "install it rather than silently skipping t2s"
        ) from exc
    converter = getattr(_opencc_t2s, "_converter", None)
    if converter is None:
        converter = OpenCC("t2s")
        setattr(_opencc_t2s, "_converter", converter)
    return converter.convert(text)


def norm_zh(text: str) -> tuple[str, list[str]]:
    rules: list[str] = []
    result = _nfkc(_ensure_text(text), rules)
    converted = _opencc_t2s(result)
    if converted != result:
        rules.append("opencc_t2s")
    result = _strip_zh_punctuation(converted, rules)
    result = _collapse_spaces(result, rules, remove_all=True)
    return result, rules


def norm_ru(text: str, *, fold_yo: bool = True) -> tuple[str, list[str]]:
    rules: list[str] = []
    result = _nfkc(_ensure_text(text), rules)
    result = _lower(result, rules)
    if fold_yo:
        folded = result.replace("ё", "е")
        if folded != result:
            rules.append("fold_yo_to_e")
        result = folded
    without_stress = result.replace("\u0301", "")
    if without_stress != result:
        rules.append("strip_combining_acute_u0301")
    result = without_stress
    result = _strip_word_punctuation(result, rules, keep_hyphen=True)
    result = _collapse_spaces(result, rules)
    return result, rules


def norm_fr(text: str) -> tuple[str, list[str]]:
    rules: list[str] = []
    result = _nfkc(_ensure_text(text), rules)
    result = _lower(result, rules)
    unified = result.replace("’", "'")
    if unified != result:
        rules.append("unify_curly_apostrophe")
    result = _strip_word_punctuation(unified, rules, keep_apostrophe=True, keep_hyphen=True)
    result = _collapse_spaces(result, rules)
    return result, rules


def norm_es(text: str) -> tuple[str, list[str]]:
    rules: list[str] = []
    result = _nfkc(_ensure_text(text), rules)
    result = _lower(result, rules)
    without_inverted = result.replace("¿", "").replace("¡", "")
    if without_inverted != result:
        rules.append("remove_inverted_question_exclamation")
    result = _strip_word_punctuation(without_inverted, rules)
    result = _collapse_spaces(result, rules)
    return result, rules


def norm_en(text: str) -> tuple[str, list[str]]:
    original = _ensure_text(text)
    _require_distribution("whisper-normalizer", "0.1.15")
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as exc:  # pragma: no cover - distribution check normally catches this
        raise RuntimeError("whisper-normalizer==0.1.15 is installed but cannot be imported") from exc
    normalizer = getattr(norm_en, "_normalizer", None)
    if normalizer is None:
        normalizer = EnglishTextNormalizer()
        setattr(norm_en, "_normalizer", normalizer)
    result = normalizer(original)
    rules = ["whisper_english_text_normalizer"] if result != original else []
    return result, rules


NORMALIZERS: dict[str, Callable[[str], tuple[str, list[str]]]] = {
    "en": norm_en,
    "zh": norm_zh,
    "es": norm_es,
    "fr": norm_fr,
    "ru": norm_ru,
}


def normalize(text: str, language: str) -> tuple[str, list[str]]:
    return NORMALIZERS[_canonical_language(language)](text)


def normalize_basic(text: str) -> tuple[str, list[str]]:
    original = _ensure_text(text)
    _require_distribution("whisper-normalizer", "0.1.15")
    try:
        from whisper_normalizer.basic import BasicTextNormalizer
    except ImportError as exc:  # pragma: no cover - distribution check normally catches this
        raise RuntimeError("whisper-normalizer==0.1.15 is installed but cannot be imported") from exc
    normalizer = getattr(normalize_basic, "_normalizer", None)
    if normalizer is None:
        normalizer = BasicTextNormalizer()
        setattr(normalize_basic, "_normalizer", normalizer)
    result = normalizer(original).strip()
    rules = ["whisper_basic_text_normalizer_0.1.15"] if result != original else []
    return result, rules


def _is_han(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _HAN_RANGES)


def _is_latin(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(char, "")


def tokenize_normalized(text: str, language: str) -> list[str]:
    language = _canonical_language(language)
    if language != "zh":
        return text.split() if text else []
    tokens: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for char in text:
        if _is_han(char):
            flush()
            tokens.append(char)
        elif _is_latin(char) or char.isdigit() or (buffer and unicodedata.category(char).startswith("M")):
            buffer.append(char)
        elif char.isalnum() or unicodedata.category(char).startswith("S"):
            flush()
            tokens.append(char)
        else:
            flush()
    flush()
    return tokens


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int
    reference_token_count: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def error_rate(self) -> float | None:
        if self.reference_token_count == 0:
            return 0.0 if self.errors == 0 else None
        return self.errors / self.reference_token_count


def levenshtein_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    # Cells carry (total cost, substitutions, insertions, deletions).  The
    # secondary tuple makes ties deterministic and stable for audit output.
    previous = [(index, 0, index, 0) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [(ref_index, 0, 0, ref_index)]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            if ref_token == hyp_token:
                current.append(previous[hyp_index - 1])
                continue
            sub = previous[hyp_index - 1]
            delete = previous[hyp_index]
            insert = current[hyp_index - 1]
            candidates = (
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2], delete[3] + 1),
                (insert[0] + 1, insert[1], insert[2] + 1, insert[3]),
            )
            current.append(min(candidates))
        previous = current
    cost, substitutions, insertions, deletions = previous[-1]
    assert cost == substitutions + insertions + deletions
    return EditCounts(substitutions, insertions, deletions, len(reference))


def score_pair(
    reference: str,
    hypothesis: str,
    language: str,
    *,
    normalizer: Literal["project", "basic"] = "project",
) -> dict:
    language = _canonical_language(language)
    if normalizer not in {"project", "basic"}:
        raise ValueError("normalizer must be exactly 'project' or 'basic'")
    normalizer_fn = normalize if normalizer == "project" else lambda text, _language: normalize_basic(text)
    reference_normalized, reference_rules = normalizer_fn(reference, language)
    hypothesis_normalized, hypothesis_rules = normalizer_fn(hypothesis, language)
    reference_tokens = tokenize_normalized(reference_normalized, language)
    hypothesis_tokens = tokenize_normalized(hypothesis_normalized, language)
    counts = levenshtein_counts(reference_tokens, hypothesis_tokens)
    return {
        "language": language,
        "metric": "CER" if language == "zh" else "WER",
        "normalizer": normalizer,
        "normalizer_id": normalizer_id(language) if normalizer == "project" else basic_normalizer_id(),
        "reference_raw": reference,
        "hypothesis_raw": hypothesis,
        "reference_normalized": reference_normalized,
        "hypothesis_normalized": hypothesis_normalized,
        "reference_rules": reference_rules,
        "hypothesis_rules": hypothesis_rules,
        "reference_tokens": reference_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        **asdict(counts),
        "errors": counts.errors,
        "error_rate": counts.error_rate,
    }


def score_corpus(
    pairs: Iterable[tuple[str, str]],
    language: str,
    *,
    normalizer: Literal["project", "basic"] = "project",
) -> dict:
    utterances = [score_pair(ref, hyp, language, normalizer=normalizer) for ref, hyp in pairs]
    totals = {
        key: sum(int(row[key]) for row in utterances)
        for key in ("substitutions", "insertions", "deletions", "reference_token_count", "errors")
    }
    rate = None if totals["reference_token_count"] == 0 and totals["errors"] else (
        0.0 if totals["reference_token_count"] == 0 else totals["errors"] / totals["reference_token_count"]
    )
    return {
        "language": _canonical_language(language),
        "metric": "CER" if _canonical_language(language) == "zh" else "WER",
        "normalizer": normalizer,
        "normalizer_id": normalizer_id(language) if normalizer == "project" else basic_normalizer_id(),
        "n": len(utterances),
        **totals,
        "error_rate": rate,
        "utterances": utterances,
    }


def compare_corpus(pairs: Iterable[tuple[str, str]], language: str) -> dict:
    pairs = list(pairs)
    project = score_corpus(pairs, language, normalizer="project")
    basic = score_corpus(pairs, language, normalizer="basic")
    if project["error_rate"] is None or basic["error_rate"] is None:
        delta_pp = None
    else:
        delta_pp = 100.0 * (basic["error_rate"] - project["error_rate"])
    return {
        "language": _canonical_language(language),
        "metric": project["metric"],
        "n": len(pairs),
        "project": {key: value for key, value in project.items() if key != "utterances"},
        "basic": {key: value for key, value in basic.items() if key != "utterances"},
        "delta_pp_basic_minus_project": delta_pp,
        "ranking_inadmissible": delta_pp is None or abs(delta_pp) > 2.0,
    }


def _load_jsonl_pairs(path: Path) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {language: [] for language in LANGUAGES}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                language = _canonical_language(row["language"])
                grouped[language].append((row["reference"], row["hypothesis"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed fixture row") from exc
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-fixtures", type=Path, help="JSONL rows with language/reference/hypothesis")
    parser.add_argument("--output", type=Path, help="write JSON result here instead of stdout")
    args = parser.parse_args()
    if not args.delta_fixtures:
        parser.error("--delta-fixtures is required")
    grouped = _load_jsonl_pairs(args.delta_fixtures)
    result = {
        "normalizer_package_sha256": PACKAGE_SHA256,
        "normalizer_ids": {language: normalizer_id(language) for language in LANGUAGES},
        "basic_normalizer_id": basic_normalizer_id(),
        "cells": [compare_corpus(grouped[language], language) for language in LANGUAGES],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
