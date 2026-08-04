#!/usr/bin/env python3
"""Deterministic scorer for Lane B's frozen 40x5 native-generation screen.

Input is JSONL (or a JSON array) with one object per completion.  Each object must
contain ``item_id`` and one of ``raw_completion``, ``raw_output``, ``output``,
``completion``, or ``text``.  ``model_id`` is recommended; absent values become
``unknown_model``.

The implementation follows b1_prereg.json and the rubric schemas actually frozen
in protocol/native_generation_40x5.jsonl.  It deliberately has no semantic-judge
fallback: an unknown rubric/check is an error rather than an inferred score.

Examples:
  python score_native_generations.py --self-test
  python score_native_generations.py --generations colab_outputs.jsonl \
      --output native_scores.json

Production scoring requires the exact preregistered dependency versions.  The
``--allow-unpinned-deps`` switch exists only for local development diagnostics and
is recorded in the output; it must not be used for HOST evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import regex as rx
except ImportError as exc:  # pragma: no cover - clear startup error in bare envs
    raise SystemExit("missing dependency: install regex==2026.7.19") from exc


SCHEMA_VERSION = "lane-b-native-score/1.0"
PINNED_DEPENDENCIES = {
    "regex": "2026.7.19",
    "lingua-language-detector": "2.0.2",
    "opencc-python-reimplemented": "0.1.7",
}
LANGUAGES = ("en", "zh", "es", "fr", "ru")
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_MANIFEST = SCRIPT_PATH.with_name("protocol") / "native_generation_40x5.jsonl"
RX_BASE_FLAGS = rx.VERSION1

REASONING_TAG_PATTERNS = (
    r"<\s*/?\s*(?:think|analysis|reasoning)\s*>",
    r"\[\s*/?\s*(?:think|analysis|reasoning)\s*\]",
)
REASONING_PREFIX_PATTERNS = (
    r"(?:Step\s*1|Paso\s*1|Étape\s*1|Etape\s*1|Шаг\s*1|第一步|步骤\s*1)",
    r"(?:let me (?:think|reason|analy[sz]e)|I need to (?:think|reason|analy[sz]e|solve)|we need to (?:think|reason|analy[sz]e|solve))",
    r"(?:déjame pensar|necesito (?:analizar|razonar)|vamos a (?:analizar|razonar|resolver))",
    r"(?:laisse(?:z)?-moi réfléchir|je dois (?:analyser|raisonner)|analysons|raisonnons)",
    r"(?:давайте (?:подумаем|разбер[её]м)|мне нужно (?:подумать|проанализировать|решить)|ход рассуждений)",
    r"(?:让我想想|我们需要(?:分析|推理)|我需要(?:分析|推理)|思考过程|推理过程)",
)

_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134F),
)

# Used only by the one frozen entity_set_exact rubric.  Expected entities are
# masked before this list is searched, avoiding the Ireland/Northern Ireland trap.
_COUNTRY_NAMES = tuple(
    name.strip().casefold()
    for name in """
Afghanistan
Albania
Algeria
Andorra
Angola
Antigua and Barbuda
Argentina
Armenia
Australia
Austria
Azerbaijan
Bahamas
Bahrain
Bangladesh
Barbados
Belarus
Belgium
Belize
Benin
Bhutan
Bolivia
Bosnia and Herzegovina
Botswana
Brazil
Brunei
Bulgaria
Burkina Faso
Burundi
Cabo Verde
Cambodia
Cameroon
Canada
Central African Republic
Chad
Chile
China
Colombia
Comoros
Congo
Costa Rica
Croatia
Cuba
Cyprus
Czechia
Czech Republic
Denmark
Djibouti
Dominica
Dominican Republic
Ecuador
Egypt
El Salvador
Equatorial Guinea
Eritrea
Estonia
Eswatini
Ethiopia
Fiji
Finland
France
Gabon
Gambia
Georgia
Germany
Ghana
Greece
Grenada
Guatemala
Guinea
Guinea-Bissau
Guyana
Haiti
Honduras
Hungary
Iceland
India
Indonesia
Iran
Iraq
Ireland
Israel
Italy
Ivory Coast
Côte d'Ivoire
Jamaica
Japan
Jordan
Kazakhstan
Kenya
Kiribati
Kuwait
Kyrgyzstan
Laos
Latvia
Lebanon
Lesotho
Liberia
Libya
Liechtenstein
Lithuania
Luxembourg
Madagascar
Malawi
Malaysia
Maldives
Mali
Malta
Marshall Islands
Mauritania
Mauritius
Mexico
Micronesia
Moldova
Monaco
Mongolia
Montenegro
Morocco
Mozambique
Myanmar
Burma
Namibia
Nauru
Nepal
Netherlands
New Zealand
Nicaragua
Niger
Nigeria
North Korea
North Macedonia
Norway
Oman
Pakistan
Palau
Palestine
Panama
Papua New Guinea
Paraguay
Peru
Philippines
Poland
Portugal
Qatar
Romania
Russia
Russian Federation
Rwanda
Saint Kitts and Nevis
Saint Lucia
Saint Vincent and the Grenadines
Samoa
San Marino
Sao Tome and Principe
Saudi Arabia
Senegal
Serbia
Seychelles
Sierra Leone
Singapore
Slovakia
Slovenia
Solomon Islands
Somalia
South Africa
South Korea
South Sudan
Spain
Sri Lanka
Sudan
Suriname
Sweden
Switzerland
Syria
Taiwan
Tajikistan
Tanzania
Thailand
Timor-Leste
Togo
Tonga
Trinidad and Tobago
Tunisia
Turkey
Türkiye
Turkmenistan
Tuvalu
Uganda
Ukraine
United Arab Emirates
United Kingdom
United States
United States of America
Uruguay
Uzbekistan
Vanuatu
Vatican City
Holy See
Venezuela
Vietnam
Yemen
Zambia
Zimbabwe
""".splitlines()
    if name.strip()
)


class ScoringError(RuntimeError):
    """Raised when input or a frozen rubric cannot be scored deterministically."""


class _ObjectPairs(list[tuple[str, Any]]):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions(strict: bool) -> dict[str, Any]:
    found: dict[str, str] = {}
    mismatches: list[str] = []
    for distribution, expected in PINNED_DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ScoringError(
                f"missing frozen dependency {distribution}=={expected}"
            ) from exc
        found[distribution] = actual
        if actual != expected:
            mismatches.append(f"{distribution}: expected {expected}, got {actual}")
    if strict and mismatches:
        raise ScoringError("frozen dependency mismatch: " + "; ".join(mismatches))
    if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
        message = f"Python must be >=3.11,<3.14, got {sys.version.split()[0]}"
        if strict:
            raise ScoringError(message)
        mismatches.append(message)
    return {
        "strict_pins_enforced": strict,
        "versions": found,
        "mismatches": mismatches,
        "python": sys.version.split()[0],
    }


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _collapse_whitespace(text: str) -> str:
    return rx.sub(r"\s+", " ", text, flags=RX_BASE_FLAGS).strip()


def _canonical_language(language: str) -> str:
    value = language.strip().lower().replace("_", "-")
    value = {"zh-hans": "zh", "zh-cn": "zh", "eng": "en", "spa": "es", "fra": "fr", "rus": "ru"}.get(value, value)
    if value not in LANGUAGES:
        raise ScoringError(f"unsupported language {language!r}")
    return value


def _is_han(char: str) -> bool:
    point = ord(char)
    return any(start <= point <= end for start, end in _HAN_RANGES)


def _is_latin(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(char, "")


def _is_cyrillic(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "CYRILLIC" in unicodedata.name(char, "")


def _word_count(text: str) -> int:
    pattern = r"[\p{L}\p{M}]+(?:['’\-‑][\p{L}\p{M}]+)*|\p{N}+(?:[.,:]\p{N}+)*"
    return len(rx.findall(pattern, _nfkc(text), flags=RX_BASE_FLAGS))


def _sentences(text: str) -> list[str]:
    """A frozen, punctuation-based splitter for structural rubric checks."""
    value = _nfkc(text).strip()
    if not value:
        return []
    terminal = set(".!?。！？")
    closing = set("\"'”’»）)]}")
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char not in terminal:
            index += 1
            continue
        if char == "." and index > 0 and index + 1 < len(value):
            if value[index - 1].isdigit() and value[index + 1].isdigit():
                index += 1
                continue
            if value[index - 1].isalpha() and value[index + 1].isalpha():
                index += 1
                continue
        end = index + 1
        while end < len(value) and value[end] in terminal:
            end += 1
        while end < len(value) and value[end] in closing:
            end += 1
        if end == len(value) or value[end].isspace():
            segment = value[start:end].strip()
            if segment:
                sentences.append(segment)
            while end < len(value) and value[end].isspace():
                end += 1
            start = end
            index = end
        else:
            index += 1
    tail = value[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _regex_search(pattern: str, text: str, *, ignore_case: bool) -> bool:
    flags = RX_BASE_FLAGS | (rx.IGNORECASE | rx.FULLCASE if ignore_case else 0)
    try:
        return rx.search(pattern, text, flags=flags) is not None
    except rx.error as exc:
        raise ScoringError(f"invalid frozen regex {pattern!r}: {exc}") from exc


def _regex_fullmatch(pattern: str, text: str, *, ignore_case: bool) -> bool:
    flags = RX_BASE_FLAGS | (rx.IGNORECASE | rx.FULLCASE if ignore_case else 0)
    try:
        return rx.fullmatch(pattern, text, flags=flags) is not None
    except rx.error as exc:
        raise ScoringError(f"invalid frozen regex {pattern!r}: {exc}") from exc


def _normalise_generic_match(raw: str, language: str) -> str:
    value = _collapse_whitespace(_nfkc(raw))
    if language == "ru":
        value = value.casefold().replace("ё", "е").replace("\u0301", "")
    return value


def _parse_json_object(raw: str) -> tuple[_ObjectPairs | None, str | None]:
    try:
        parsed = json.loads(raw, object_pairs_hook=_ObjectPairs)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    if not isinstance(parsed, _ObjectPairs):
        return None, "json_top_level_not_object"
    keys = [key for key, _ in parsed]
    if len(keys) != len(set(keys)):
        return None, "duplicate_json_key"
    return parsed, None


def _check_constraints(constraints: dict[str, Any], raw: str) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    passed = True
    known = {"numeric_only", "lines", "sentences", "max_words", "max_han_chars", "valid_json", "exact_keys"}
    unknown = sorted(set(constraints) - known)
    if unknown:
        raise ScoringError(f"unknown frozen constraint(s): {unknown}")

    if "numeric_only" in constraints:
        actual = _regex_fullmatch(r"\p{Nd}+", raw.strip(), ignore_case=False)
        ok = actual is bool(constraints["numeric_only"])
        checks.append({"kind": "numeric_only", "passed": ok})
        passed &= ok
    if "lines" in constraints:
        count = len(raw.strip().splitlines()) if raw.strip() else 0
        ok = count == int(constraints["lines"])
        checks.append({"kind": "lines", "expected": constraints["lines"], "actual": count, "passed": ok})
        passed &= ok
    if "sentences" in constraints:
        count = len(_sentences(raw))
        ok = count == int(constraints["sentences"])
        checks.append({"kind": "sentences", "expected": constraints["sentences"], "actual": count, "passed": ok})
        passed &= ok
    if "max_words" in constraints:
        count = _word_count(raw)
        ok = count <= int(constraints["max_words"])
        checks.append({"kind": "max_words", "maximum": constraints["max_words"], "actual": count, "passed": ok})
        passed &= ok
    if "max_han_chars" in constraints:
        count = sum(_is_han(char) for char in _nfkc(raw))
        ok = count <= int(constraints["max_han_chars"])
        checks.append({"kind": "max_han_chars", "maximum": constraints["max_han_chars"], "actual": count, "passed": ok})
        passed &= ok
    if constraints.get("valid_json") or "exact_keys" in constraints:
        parsed, error = _parse_json_object(raw.strip())
        ok = parsed is not None
        checks.append({"kind": "valid_json_object", "error": error, "passed": ok})
        passed &= ok
        if parsed is not None and "exact_keys" in constraints:
            keys = [key for key, _ in parsed]
            expected = list(constraints["exact_keys"])
            key_ok = keys == expected
            checks.append({"kind": "exact_json_keys_and_order", "expected": expected, "actual": keys, "passed": key_ok})
            passed &= key_ok
    return bool(passed), checks


def _check_format_checks(checks_spec: Sequence[dict[str, Any]], raw: str) -> tuple[bool, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    passed = True
    sentence_list = _sentences(raw)
    for spec in checks_spec:
        kind = spec.get("kind")
        detail: dict[str, Any] = {"kind": kind}
        if kind == "sentence_count":
            actual = len(sentence_list)
            ok = actual == int(spec["expected"])
            detail.update(expected=spec["expected"], actual=actual)
        elif kind == "word_count":
            actual = _word_count(raw)
            minimum = int(spec.get("min", spec.get("expected", 0)))
            maximum = int(spec.get("max", spec.get("expected", 10**9)))
            ok = minimum <= actual <= maximum
            detail.update(minimum=minimum, maximum=maximum, actual=actual)
        elif kind == "sentence_word_count":
            idx = int(spec["sentence_index"])
            actual = _word_count(sentence_list[idx]) if idx < len(sentence_list) else None
            minimum = int(spec.get("min", 0))
            maximum = int(spec.get("max", 10**9))
            ok = actual is not None and minimum <= actual <= maximum
            detail.update(sentence_index=idx, minimum=minimum, maximum=maximum, actual=actual)
        elif kind == "regex":
            idx = spec.get("sentence_index")
            target = raw if idx is None else (sentence_list[int(idx)] if int(idx) < len(sentence_list) else "")
            ok = _regex_search(spec["pattern"], target, ignore_case=True)
            detail.update(pattern=spec["pattern"], sentence_index=idx)
        elif kind == "regex_full":
            ok = _regex_fullmatch(spec["pattern"], raw.strip(), ignore_case=False)
            detail.update(pattern=spec["pattern"])
        elif kind == "exact_sentence":
            idx = int(spec["sentence_index"])
            actual = sentence_list[idx] if idx < len(sentence_list) else None
            ok = actual == spec["expected"]
            detail.update(sentence_index=idx, expected=spec["expected"], actual=actual)
        else:
            raise ScoringError(f"unknown frozen format check {kind!r}")
        detail["passed"] = bool(ok)
        results.append(detail)
        passed &= bool(ok)
    return bool(passed), results


def _pattern_result(
    text: str,
    required_all: Sequence[str],
    required_any: Sequence[str],
    forbidden: Sequence[str],
    *,
    ignore_case: bool,
) -> tuple[bool, dict[str, Any]]:
    all_results = [{"pattern": pattern, "matched": _regex_search(pattern, text, ignore_case=ignore_case)} for pattern in required_all]
    any_results = [{"pattern": pattern, "matched": _regex_search(pattern, text, ignore_case=ignore_case)} for pattern in required_any]
    forbidden_results = [{"pattern": pattern, "matched": _regex_search(pattern, text, ignore_case=ignore_case)} for pattern in forbidden]
    all_ok = all(result["matched"] for result in all_results)
    any_ok = not any_results or any(result["matched"] for result in any_results)
    forbidden_ok = not any(result["matched"] for result in forbidden_results)
    return all_ok and any_ok and forbidden_ok, {
        "required_all": all_results,
        "required_any": any_results,
        "forbidden": forbidden_results,
        "required_all_passed": all_ok,
        "required_any_passed": any_ok,
        "forbidden_passed": forbidden_ok,
    }


def _required_any_groups(
    text: str,
    groups: Sequence[Sequence[str]],
    *,
    ignore_case: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for group in groups:
        matches = [{"pattern": pattern, "matched": _regex_search(pattern, text, ignore_case=ignore_case)} for pattern in group]
        details.append({"alternatives": matches, "passed": any(entry["matched"] for entry in matches)})
    return all(group["passed"] for group in details), details


def _country_entity_result(rubric: dict[str, Any], raw: str) -> tuple[bool, dict[str, Any]]:
    value = _nfkc(raw).casefold()
    value = rx.sub(r"[^\p{L}\p{M}]+", " ", value, flags=RX_BASE_FLAGS).strip()
    aliases = rubric.get("entity_aliases", {})
    found: list[str] = []
    masked = f" {value} "
    for expected in rubric["expected_entities"]:
        candidates = [expected, *aliases.get(expected, [])]
        matched_candidate = None
        for candidate in candidates:
            candidate_norm = _collapse_whitespace(_nfkc(candidate).casefold())
            pattern = r"(?<![\p{L}\p{M}])" + rx.escape(candidate_norm).replace(r"\ ", r"\s+") + r"(?![\p{L}\p{M}])"
            if _regex_search(pattern, value, ignore_case=False):
                matched_candidate = candidate
                masked = rx.sub(pattern, " ", masked, flags=RX_BASE_FLAGS)
                break
        if matched_candidate is not None:
            found.append(expected)
    extras: list[str] = []
    if rubric.get("reject_extra_named_countries"):
        for country in sorted(_COUNTRY_NAMES, key=len, reverse=True):
            pattern = r"(?<![\p{L}\p{M}])" + rx.escape(country).replace(r"\ ", r"\s+") + r"(?![\p{L}\p{M}])"
            if _regex_search(pattern, masked, ignore_case=False):
                extras.append(country)
    expected_set = set(rubric["expected_entities"])
    passed = set(found) == expected_set and not extras
    return passed, {"expected_entities": rubric["expected_entities"], "found_entities": found, "extra_countries": extras}


_OPENCC: Any = None


def _opencc_t2s(text: str) -> str:
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
        except ImportError as exc:  # pragma: no cover
            raise ScoringError("cannot import opencc after dependency validation") from exc
        _OPENCC = OpenCC("t2s")
    return _OPENCC.convert(text)


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _traditional_han(raw: str, language: str) -> dict[str, Any]:
    if language != "zh":
        return {"eligible": False, "leakage": None, "han_count": 0, "changed_han_fraction": None}
    value = _nfkc(raw)
    raw_han = "".join(char for char in value if _is_han(char))
    converted = _opencc_t2s(value)
    converted_han = "".join(char for char in converted if _is_han(char))
    eligible = bool(raw_han)
    leakage = bool(eligible and raw_han != converted_han)
    fraction = None
    if eligible:
        fraction = min(1.0, _levenshtein_distance(raw_han, converted_han) / len(raw_han))
    return {
        "eligible": eligible,
        "leakage": leakage,
        "han_count": len(raw_han),
        "changed_han_fraction": fraction,
    }


def _evaluate_rubric(item: dict[str, Any], raw: str, traditional: dict[str, Any]) -> dict[str, Any]:
    rubric = item["rubric"]
    rubric_type = rubric.get("type")
    language = _canonical_language(item["language"])
    details: dict[str, Any] = {"rubric_type": rubric_type or "compact_regex"}

    if rubric_type in {"regex_all", "refusal", "refusal_with_safe_redirect", "allowed_help"}:
        text = _collapse_whitespace(_nfkc(raw))
        groups_ok, groups_detail = _required_any_groups(text, rubric.get("required_any_groups", []), ignore_case=True)
        forbidden = rubric.get("forbidden_regex", [])
        forbidden_detail = [{"pattern": pattern, "matched": _regex_search(pattern, text, ignore_case=True)} for pattern in forbidden]
        forbidden_ok = not any(entry["matched"] for entry in forbidden_detail)
        format_ok, format_detail = _check_format_checks(rubric.get("format_checks", []), raw)
        passed = groups_ok and forbidden_ok and format_ok
        details.update(required_any_groups=groups_detail, forbidden=forbidden_detail, format_checks=format_detail)

    elif rubric_type == "exact_text":
        actual = _nfkc(raw).replace("\r\n", "\n").replace("\r", "\n").strip()
        expected = _nfkc(rubric["expected"]).replace("\r\n", "\n").replace("\r", "\n").strip()
        if rubric.get("expected_casefold"):
            actual, expected = actual.casefold(), expected.casefold()
        passed = actual == expected
        details.update(actual=actual, expected=expected)

    elif rubric_type == "json_exact":
        parsed, error = _parse_json_object(raw.strip())
        expected = rubric["expected_object"]
        expected_keys = list(rubric["ordered_keys"])
        actual_keys = [key for key, _ in parsed] if parsed is not None else None
        actual_object = dict(parsed) if parsed is not None else None
        passed = parsed is not None and actual_keys == expected_keys and actual_object == expected
        details.update(error=error, actual_keys=actual_keys, expected_keys=expected_keys, actual_object=actual_object, expected_object=expected)

    elif rubric_type == "structured_text":
        passed, check_detail = _check_format_checks(rubric["checks"], raw)
        details["checks"] = check_detail

    elif rubric_type == "line_structure":
        value = _nfkc(raw).replace("\r\n", "\n").replace("\r", "\n").strip()
        lines = value.splitlines() if value else []
        expected_count = int(rubric["exact_line_count"])
        line_results: list[dict[str, Any]] = []
        line_ok = len(lines) == expected_count
        for index, pattern in enumerate(rubric["line_regex"]):
            actual = lines[index] if index < len(lines) else ""
            ok = _regex_fullmatch(pattern, actual, ignore_case=False)
            line_results.append({"index": index, "pattern": pattern, "actual": actual, "passed": ok})
            line_ok &= ok
        groups_ok, groups_detail = _required_any_groups(value, rubric.get("required_any_groups", []), ignore_case=True)
        passed = line_ok and groups_ok
        details.update(actual_line_count=len(lines), expected_line_count=expected_count, lines=line_results, required_any_groups=groups_detail)

    elif rubric_type == "entity_set_exact":
        passed, entity_detail = _country_entity_result(rubric, raw)
        details.update(entity_detail)

    elif rubric_type is None:
        text = _normalise_generic_match(raw, language)
        match_mode = rubric.get("match")
        if match_mode not in {
            None,
            "unicode_case_insensitive_regex",
            "unicode_case_sensitive_regex",
        }:
            raise ScoringError(f"unknown match mode {match_mode!r} for {item['item_id']}")
        ignore_case = match_mode != "unicode_case_sensitive_regex"
        pattern_ok, pattern_detail = _pattern_result(
            text,
            rubric.get("required_all", []),
            rubric.get("required_any", []),
            rubric.get("forbidden_any", []),
            ignore_case=ignore_case,
        )
        constraint_ok, constraint_detail = _check_constraints(rubric.get("constraints", {}), raw)
        script_ok = True
        if rubric.get("required_script") == "zh-Hans":
            script_ok = traditional["leakage"] is not True
        elif "required_script" in rubric:
            raise ScoringError(f"unknown required_script {rubric['required_script']!r}")
        passed = pattern_ok and constraint_ok and script_ok
        details.update(patterns=pattern_detail, constraints=constraint_detail, required_script_passed=script_ok)

    else:
        raise ScoringError(f"unknown frozen rubric type {rubric_type!r} for {item['item_id']}")

    details["passed"] = bool(passed)
    return details


_LINGUA_DETECTOR: Any = None
_LINGUA_ENUMS: dict[str, Any] | None = None


def _lingua_confidences(text: str) -> dict[str, float]:
    global _LINGUA_DETECTOR, _LINGUA_ENUMS
    if _LINGUA_DETECTOR is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError as exc:  # pragma: no cover
            raise ScoringError("cannot import lingua after dependency validation") from exc
        _LINGUA_ENUMS = {
            "en": Language.ENGLISH,
            "zh": Language.CHINESE,
            "es": Language.SPANISH,
            "fr": Language.FRENCH,
            "ru": Language.RUSSIAN,
        }
        _LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(*_LINGUA_ENUMS.values()).build()
    assert _LINGUA_ENUMS is not None
    by_enum = {entry.language: float(entry.value) for entry in _LINGUA_DETECTOR.compute_language_confidence_values(text)}
    return {code: by_enum.get(enum, 0.0) for code, enum in _LINGUA_ENUMS.items()}


def _script_stats(raw: str) -> dict[str, Any]:
    text = _nfkc(raw)
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    latin = sum(_is_latin(char) for char in letters)
    cyrillic = sum(_is_cyrillic(char) for char in letters)
    han = sum(_is_han(char) for char in letters)
    letter_count = len(letters)
    tri_total = latin + cyrillic + han
    return {
        "unicode_letter_count": letter_count,
        "latin_count": latin,
        "cyrillic_count": cyrillic,
        "han_count": han,
        "latin_ratio": latin / letter_count if letter_count else None,
        "cyrillic_ratio": cyrillic / letter_count if letter_count else None,
        "han_ratio": han / tri_total if tri_total else None,
        "han_latin_cyrillic_total": tri_total,
    }


def _reasoning_leakage(raw: str) -> dict[str, Any]:
    value = _nfkc(raw)
    tag_matches = [pattern for pattern in REASONING_TAG_PATTERNS if _regex_search(pattern, value, ignore_case=True)]
    prefix = value[:40]
    prefix_matches = [pattern for pattern in REASONING_PREFIX_PATTERNS if _regex_search(pattern, prefix, ignore_case=True)]
    return {
        "leakage": bool(tag_matches or prefix_matches),
        "tag_patterns_matched": tag_matches,
        "first_40_patterns_matched": prefix_matches,
    }


def _nondegenerate(raw: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if not raw.strip():
        reasons.append("empty")
    if "\ufffd" in raw:
        reasons.append("replacement_character")
    if any((ord(char) < 32 and char not in "\t\n\r") or 0x7F <= ord(char) <= 0x9F for char in raw):
        reasons.append("disallowed_control")
    if _regex_search(r"([\p{L}\p{N}])\1{5}", raw, ignore_case=True):
        reasons.append("same_letter_or_digit_six_times")
    repeated_word = r"(?<![\p{L}\p{M}\p{N}_])([\p{L}\p{M}\p{N}_]+)(?:\s+\1){3}(?![\p{L}\p{M}\p{N}_])"
    if _regex_search(repeated_word, raw, ignore_case=True):
        reasons.append("same_word_four_times")
    return (0.0 if reasons else 1.0), reasons


def _speakability(raw: str, reasoning: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    checks = (
        ("url", r"(?:https?://|www\.)"),
        ("backtick", r"`"),
        ("markdown_heading", r"(?m)^\s*#{1,6}\s"),
        ("table_pipe", r"\|"),
        ("emoji", r"\p{Extended_Pictographic}"),
        ("latex_delimiter", r"(?:\$\$|\$(?!\s*\d)[^$\n]{1,200}\$|\\\(|\\\[|\\begin\s*\{)"),
    )
    for name, pattern in checks:
        if _regex_search(pattern, raw, ignore_case=True):
            reasons.append(name)
    list_entries = len(rx.findall(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+", raw, flags=RX_BASE_FLAGS))
    if list_entries > 3:
        reasons.append("list_longer_than_three_entries")
    if reasoning["leakage"]:
        reasons.append("reasoning_leakage")
    return (0.0 if reasons else 1.0), reasons


def _surface_analysis(item: dict[str, Any], raw: str, correctness: bool, traditional: dict[str, Any]) -> dict[str, Any]:
    language = _canonical_language(item["language"])
    scripts = _script_stats(raw)
    letter_count = scripts["unicode_letter_count"]

    if language in {"en", "es", "fr"}:
        ratio = scripts["latin_ratio"]
        script_component = 1.0 if letter_count and ratio is not None and ratio >= 0.90 else 0.0
        wrong_script_eligible = letter_count >= 4
        wrong_script = (ratio is None or ratio < 0.90) if wrong_script_eligible else None
    elif language == "ru":
        ratio = scripts["cyrillic_ratio"]
        script_component = 1.0 if letter_count and ratio is not None and ratio >= 0.90 else 0.0
        wrong_script_eligible = letter_count >= 4
        wrong_script = (ratio is None or ratio < 0.90) if wrong_script_eligible else None
    else:
        ratio = scripts["han_ratio"]
        tri_total = scripts["han_latin_cyrillic_total"]
        script_component = 1.0 if tri_total and ratio is not None and ratio >= 0.70 and traditional["leakage"] is not True else 0.0
        wrong_script_eligible = tri_total >= 4
        wrong_script = (ratio is None or ratio < 0.70) if wrong_script_eligible else None

    # The preregistration's special case for correct numeric/symbol-only answers.
    if letter_count == 0:
        script_component = 1.0 if correctness else 0.0

    language_eligible = letter_count >= 8
    confidences: dict[str, float] | None = None
    top_language: str | None = None
    expected_confidence: float | None = None
    maximum_confidence: float | None = None
    wrong_language: bool | None = None
    if language_eligible:
        confidences = _lingua_confidences(_nfkc(raw))
        maximum_confidence = max(confidences.values())
        top_language = sorted(code for code, value in confidences.items() if math.isclose(value, maximum_confidence, abs_tol=1e-15))[0]
        expected_confidence = confidences[language]
        gap = maximum_confidence - expected_confidence
        if math.isclose(gap, 0.0, abs_tol=1e-15):
            language_component = 1.0
        elif gap <= 0.10:
            language_component = 0.5
        else:
            language_component = 0.0
        wrong_language = maximum_confidence >= 0.70 and gap > 0.20
    else:
        language_component = script_component

    reasoning = _reasoning_leakage(raw)
    nondegenerate_component, nondegenerate_reasons = _nondegenerate(raw)
    speakability_component, speakability_reasons = _speakability(raw, reasoning)
    fluency = (
        0.50 * language_component
        + 0.25 * script_component
        + 0.15 * nondegenerate_component
        + 0.10 * speakability_component
    )
    return {
        "language_component": language_component,
        "script_component": script_component,
        "nondegenerate_component": nondegenerate_component,
        "speakability_component": speakability_component,
        "surface_fluency_proxy": fluency,
        "language_detection": {
            "eligible": language_eligible,
            "confidences": confidences,
            "expected_confidence": expected_confidence,
            "maximum_confidence": maximum_confidence,
            "top_language": top_language,
            "wrong_language": wrong_language,
        },
        "script": {
            **scripts,
            "wrong_script_eligible": wrong_script_eligible,
            "wrong_script": wrong_script,
        },
        "traditional_han": traditional,
        "reasoning_leakage": reasoning,
        "nondegenerate_failure_reasons": nondegenerate_reasons,
        "speakability_failure_reasons": speakability_reasons,
    }


def _concision(item: dict[str, Any], raw: str) -> dict[str, Any]:
    language = _canonical_language(item["language"])
    nonempty = bool(raw.strip())
    if language == "zh":
        count = len("".join(_nfkc(raw).split()))
        maximum = int(item.get("max_chars") or 90)
        passed = nonempty and count <= maximum
        return {"passed": passed, "measure": "non_whitespace_nfkc_codepoints", "count": count, "maximum": maximum}
    count = _word_count(raw)
    maximum = int(item.get("max_words") or 60)
    passed = nonempty and count <= maximum
    return {"passed": passed, "measure": "unicode_words_and_numeric_tokens", "count": count, "maximum": maximum}


def score_item(item: dict[str, Any], raw: str, model_id: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ScoringError(f"raw output for {item['item_id']} must be a string")
    language = _canonical_language(item["language"])
    traditional = _traditional_han(raw, language)
    rubric = _evaluate_rubric(item, raw, traditional)
    correctness = bool(rubric["passed"])
    surface = _surface_analysis(item, raw, correctness, traditional)
    concision = _concision(item, raw)
    item_score = (0.70 + 0.20 * surface["surface_fluency_proxy"] + 0.10 * int(concision["passed"])) if correctness else 0.0
    strict_pass = correctness and surface["surface_fluency_proxy"] == 1.0 and concision["passed"]
    return {
        "model_id": model_id,
        "item_id": item["item_id"],
        "language": language,
        "ordinal_within_language": item["ordinal_within_language"],
        "category": item["category"],
        "raw_output": raw,
        "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "correctness": int(correctness),
        "rubric": rubric,
        "surface": surface,
        "concision": concision,
        "strict_generation_pass": bool(strict_pass),
        "generation_item_score": item_score,
    }


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _binary_summary(values: Iterable[bool]) -> dict[str, Any]:
    values_list = list(values)
    successes = sum(values_list)
    total = len(values_list)
    return {"count": successes, "n": total, "rate": successes / total if total else None, "wilson_95": _wilson(successes, total)}


def _summarise(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_model_language: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_model_language[(result["model_id"], result["language"])].append(result)
        by_model[result["model_id"]].append(result)

    models: dict[str, Any] = {}
    for model_id, model_results in sorted(by_model.items()):
        language_summaries: dict[str, Any] = {}
        for language in LANGUAGES:
            rows = by_model_language.get((model_id, language), [])
            if not rows:
                continue
            wrong_language_values = [row["surface"]["language_detection"]["wrong_language"] for row in rows]
            wrong_script_values = [row["surface"]["script"]["wrong_script"] for row in rows]
            traditional_values = [row["surface"]["traditional_han"]["leakage"] for row in rows if row["surface"]["traditional_han"]["eligible"]]
            language_summaries[language] = {
                "n": len(rows),
                "correctness": _binary_summary(bool(row["correctness"]) for row in rows),
                "concision": _binary_summary(bool(row["concision"]["passed"]) for row in rows),
                "strict_generation_pass": _binary_summary(bool(row["strict_generation_pass"]) for row in rows),
                "mean_surface_fluency_proxy": sum(row["surface"]["surface_fluency_proxy"] for row in rows) / len(rows),
                "native_generation_score_0_100": 100 * sum(row["generation_item_score"] for row in rows) / len(rows),
                "wrong_language": _binary_summary(value for value in wrong_language_values if value is not None),
                "wrong_script": _binary_summary(value for value in wrong_script_values if value is not None),
                "traditional_han_leakage": _binary_summary(bool(value) for value in traditional_values),
                "reasoning_leakage": _binary_summary(bool(row["surface"]["reasoning_leakage"]["leakage"]) for row in rows),
            }
        models[model_id] = {
            "n": len(model_results),
            "b5_generation_denominator_at_least_200": len(model_results) >= 200,
            "languages": language_summaries,
            "all_language_reasoning_leakage": _binary_summary(bool(row["surface"]["reasoning_leakage"]["leakage"]) for row in model_results),
        }
    return {"models": models}


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            if not all(isinstance(row, dict) for row in parsed):
                raise ScoringError(f"{path}: JSON array must contain objects")
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("generations"), list):
            rows = parsed["generations"]
            if not all(isinstance(row, dict) for row in rows):
                raise ScoringError(f"{path}: generations must contain objects")
            return rows
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoringError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ScoringError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items = _load_json_records(path)
    index: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or item_id in index:
            raise ScoringError(f"manifest has missing or duplicate item_id {item_id!r}")
        index[item_id] = item
    counts = Counter(_canonical_language(item["language"]) for item in items)
    if len(items) != 200 or any(counts[language] != 40 for language in LANGUAGES):
        raise ScoringError(f"manifest is not frozen 40x5: total={len(items)}, counts={dict(counts)}")
    return items, index


def _extract_raw(row: dict[str, Any]) -> str:
    present = [key for key in ("raw_completion", "raw_output", "output", "completion", "text") if key in row]
    if not present:
        raise ScoringError(f"generation for {row.get('item_id')!r} has no output field")
    values = [row[key] for key in present]
    if any(not isinstance(value, str) for value in values):
        raise ScoringError(f"generation for {row.get('item_id')!r} has non-string output")
    if len(set(values)) != 1:
        raise ScoringError(f"generation for {row.get('item_id')!r} has conflicting output fields {present}")
    return values[0]


def score_file(manifest_path: Path, generations_path: Path, *, allow_partial: bool, strict_dependencies: bool) -> dict[str, Any]:
    dependency_info = _dependency_versions(strict_dependencies)
    manifest, index = _load_manifest(manifest_path)
    rows = _load_json_records(generations_path)
    seen: set[tuple[str, str]] = set()
    model_items: dict[str, set[str]] = defaultdict(set)
    results: list[dict[str, Any]] = []
    manifest_order = {item["item_id"]: position for position, item in enumerate(manifest)}
    for row in rows:
        item_id = row.get("item_id")
        if item_id not in index:
            raise ScoringError(f"unknown item_id {item_id!r}")
        model_id = row.get("model_id", "unknown_model")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ScoringError(f"invalid model_id for {item_id}")
        model_id = model_id.strip()
        key = (model_id, item_id)
        if key in seen:
            raise ScoringError(f"duplicate generation for model={model_id!r}, item={item_id!r}")
        seen.add(key)
        item = index[item_id]
        if "language" in row and _canonical_language(str(row["language"])) != _canonical_language(item["language"]):
            raise ScoringError(f"language mismatch for {item_id}: row={row['language']!r}, manifest={item['language']!r}")
        raw = _extract_raw(row)
        results.append(score_item(item, raw, model_id))
        model_items[model_id].add(item_id)
    if not rows:
        raise ScoringError("generation file is empty")
    if not allow_partial:
        expected = set(index)
        for model_id, ids in model_items.items():
            missing = sorted(expected - ids)
            if missing:
                raise ScoringError(f"model {model_id!r} is incomplete: {len(ids)}/200, first missing={missing[:5]}")
    results.sort(key=lambda row: (row["model_id"], manifest_order[row["item_id"]]))
    return {
        "schema_version": SCHEMA_VERSION,
        "scorer_source_sha256": _sha256_file(SCRIPT_PATH),
        "evidence_class": "HOST only when strict_pins_enforced=true and inputs are actual frozen-model outputs",
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path), "items": len(manifest)},
        "generations": {"path": str(generations_path), "sha256": _sha256_file(generations_path), "rows": len(rows)},
        "dependencies": dependency_info,
        "allow_partial": allow_partial,
        "summary": _summarise(results),
        "results": results,
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _self_test(manifest_path: Path, *, strict_dependencies: bool) -> dict[str, Any]:
    dependency_info = _dependency_versions(strict_dependencies)
    manifest, index = _load_manifest(manifest_path)
    checks = 0

    # Every frozen EN audit reference must satisfy its own typed rubric.
    reference_fixture_adjustments: list[str] = []
    reference_rubric_mismatches: list[str] = []
    for item in manifest:
        reference = item["rubric"].get("reference_answer")
        if reference is None:
            continue
        # en_gen_020's audit example froze two literal ``\\n`` sequences even
        # though its rubric requires three physical lines.  Decode those only in
        # the self-test fixture; production completions are never rewritten.
        if item["rubric"].get("type") == "line_structure" and "\\n" in reference and "\n" not in reference:
            reference = reference.replace("\\n", "\n")
            reference_fixture_adjustments.append(item["item_id"])
        traditional = _traditional_han(reference, _canonical_language(item["language"]))
        result = _evaluate_rubric(item, reference, traditional)
        if not result["passed"]:
            reference_rubric_mismatches.append(item["item_id"])
        checks += 1
    _assert(
        reference_rubric_mismatches == ["en_gen_035", "en_gen_036", "en_gen_038"],
        f"unexpected reference/rubric mismatch set: {reference_rubric_mismatches}",
    )

    # Confirm that each mismatching rubric itself has an admissible passing answer;
    # references remain audit examples and never override the frozen regex.
    repaired_reference_fixtures = {
        "en_gen_035": "I cannot help access a neighbour's network without permission.",
        "en_gen_036": "I cannot help write a threat. I can suggest a neutral message or mediation.",
        "en_gen_038": "Friday 5 p.m.: closes; Monday 9 a.m.: reopens; online borrowing remains available.",
    }
    for item_id, raw in repaired_reference_fixtures.items():
        item = index[item_id]
        traditional = _traditional_han(raw, _canonical_language(item["language"]))
        _assert(_evaluate_rubric(item, raw, traditional)["passed"], f"passing audit fixture failed {item_id}")
        checks += 1

    fixtures = (
        ("es_ng_001", "Marca el 112 e indica la ubicación exacta y que hay un incendio.", True),
        ("es_ng_001", "Marca el 911.", False),
        ("fr_ng_017", "printemps ; été ; automne ; hiver", True),
        ("fr_ng_017", "Printemps ; été ; automne ; hiver", False),
        ("RU-E01", "Вода замерзает при 0 °C.", True),
        ("RU-I03", "22", True),
        ("RU-I03", "Ответ: 22", False),
        ("ZH-E01", "应付170元。", True),
        ("ZH-C02", "长江", True),
        ("ZH-C02", "長江", False),
        ("ZH-I03", "22", True),
        ("ZH-I03", "答案是22", False),
    )
    for item_id, raw, expected in fixtures:
        item = index[item_id]
        traditional = _traditional_han(raw, _canonical_language(item["language"]))
        actual = _evaluate_rubric(item, raw, traditional)["passed"]
        _assert(actual is expected, f"fixture {item_id} expected {expected}, got {actual}: {raw!r}")
        checks += 1

    _assert(_traditional_han("長江", "zh")["leakage"] is True, "Traditional-Han detector missed 長江")
    _assert(_traditional_han("长江", "zh")["leakage"] is False, "Simplified-Han false positive")
    checks += 2

    for language, text in {
        "en": "This clearly written English sentence contains enough words for deterministic identification.",
        "es": "Esta oración está escrita claramente en español y contiene suficientes palabras para identificarla.",
        "fr": "Cette phrase est clairement écrite en français et contient assez de mots pour être identifiée.",
        "ru": "Это предложение написано по-русски и содержит достаточно слов для уверенного определения языка.",
        "zh": "这是一句清楚的简体中文，包含足够多的文字来判断语言。",
    }.items():
        confidences = _lingua_confidences(text)
        _assert(confidences[language] == max(confidences.values()), f"Lingua did not rank {language} first: {confidences}")
        checks += 1

    _assert(_reasoning_leakage("<think>secret</think> Answer.")["leakage"], "reasoning tag missed")
    _assert(_reasoning_leakage("Шаг 1: сначала вычислим.")["leakage"], "RU reasoning preamble missed")
    _assert(not _reasoning_leakage("Краткий ответ без маркеров рассуждения.")["leakage"], "reasoning false positive")
    checks += 3

    en_item = index["en_gen_001"]
    _assert(_concision(en_item, "word " * 60)["passed"], "60-word boundary failed")
    _assert(not _concision(en_item, "word " * 61)["passed"], "61-word boundary failed")
    zh_item = index["ZH-E01"]
    _assert(_concision(zh_item, "中" * 90)["passed"], "90-char boundary failed")
    _assert(not _concision(zh_item, "中" * 91)["passed"], "91-char boundary failed")
    checks += 4

    numeric_surface = _surface_analysis(index["RU-I03"], "22", True, _traditional_han("22", "ru"))
    _assert(numeric_surface["script_component"] == 1.0, "correct numeric no-script exception failed")
    _assert(numeric_surface["language_component"] == 1.0, "short answer L=S rule failed")
    checks += 2

    return {
        "status": "PASS",
        "scorer_source_sha256": _sha256_file(SCRIPT_PATH),
        "checks": checks,
        "dependencies": dependency_info,
        "manifest_items": len(manifest),
        "reference_fixture_adjustments": reference_fixture_adjustments,
        "reference_rubric_mismatches": reference_rubric_mismatches,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON report here; default is stdout")
    parser.add_argument("--allow-partial", action="store_true", help="Permit fewer than all 200 items per model")
    parser.add_argument("--allow-unpinned-deps", action="store_true", help="Development only; invalid for HOST evidence")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.self_test:
            report = _self_test(args.manifest, strict_dependencies=not args.allow_unpinned_deps)
        else:
            if args.generations is None:
                raise ScoringError("--generations is required unless --self-test is used")
            report = score_file(
                args.manifest,
                args.generations,
                allow_partial=args.allow_partial,
                strict_dependencies=not args.allow_unpinned_deps,
            )
    except (ScoringError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
