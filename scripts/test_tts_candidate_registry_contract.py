#!/usr/bin/env python3
"""Focused source/metadata contracts for Android neural-TTS routing."""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/tts/CandidateRegistry.kt"
SHERPA = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/tts/SherpaTtsEngine.kt"
BRIDGE = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java"
STAGER = ROOT / "scripts/tts_research_20260726/stage_device_bundle.py"

CANONICAL_LANGUAGE = re.compile(r"^[a-z]{2,3}$")
SUPERTONIC_REQUIRED_FILES = {
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
    "tts_json",
    "unicode_indexer",
    "voice_style",
}


def load_staged_candidates() -> list[dict[str, object]]:
    spec = importlib.util.spec_from_file_location("stage_device_bundle", STAGER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.CANDIDATES)


def validate_catalog(candidates: list[dict[str, object]]) -> None:
    """Executable mirror of the runtime metadata contract."""
    assert candidates, "TTS candidates array is empty"
    ids: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate["id"]
        language = candidate["language"]
        engine = candidate["engine"]
        files = candidate["files"]
        assert isinstance(candidate_id, str) and candidate_id and candidate_id == candidate_id.strip()
        assert candidate_id not in ids, f"duplicate id: {candidate_id}"
        ids.add(candidate_id)
        assert isinstance(language, str) and CANONICAL_LANGUAGE.fullmatch(language)
        assert isinstance(engine, str) and engine
        assert isinstance(files, dict) and files
        assert all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in files.items()
        )
        extra = candidate.get("extra", {})
        assert isinstance(extra, dict)
        assert extra.get("lang", language) == language
        if engine == "supertonic":
            assert SUPERTONIC_REQUIRED_FILES.issubset(files)
            assert int(candidate.get("num_steps", 0)) > 0
            assert extra.get("lang") == language


def assert_rejected(candidates: list[dict[str, object]]) -> None:
    try:
        validate_catalog(candidates)
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    raise AssertionError("malformed catalog was accepted")


def ranked_auto_candidates(
    candidates: list[dict[str, object]],
    language: str,
) -> list[dict[str, object]]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["language"] == language and int(candidate.get("auto_priority", 0)) > 0
    ]
    return sorted(eligible, key=lambda candidate: (-int(candidate["auto_priority"]), candidate["id"]))


def first_available(
    ranked: list[dict[str, object]],
    available_ids: set[str],
) -> dict[str, object] | None:
    return next((candidate for candidate in ranked if candidate["id"] in available_ids), None)


def next_available_after_failure(
    ranked: list[dict[str, object]],
    failed_id: str,
    available_ids: set[str],
) -> dict[str, object] | None:
    after_failed = False
    for candidate in ranked:
        if not after_failed:
            after_failed = candidate["id"] == failed_id
            continue
        if candidate["id"] in available_ids:
            return candidate
    return None


def model_signature(candidate: dict[str, object]) -> tuple[object, object, object]:
    """Executable mirror of Candidate.sharesModelWith()."""
    files = candidate["files"]
    assert isinstance(files, dict)
    return candidate["engine"], candidate["num_threads"], tuple(sorted(files.items()))


def system_locale_available(
    package_installed: bool,
    capabilities_known: bool,
    offline_languages: set[str],
    locale_tag: str,
) -> bool:
    """Executable mirror of the synchronous eSpeak locale gate."""
    language = locale_tag.strip().split("-", 1)[0].lower()
    return (
        package_installed
        and capabilities_known
        and bool(language)
        and language in offline_languages
    )


def candidate_supports_locale(candidate: dict[str, object], locale_tag: str) -> bool:
    language = locale_tag.strip().split("-", 1)[0].lower()
    return bool(language) and candidate["language"] == language


def main() -> None:
    registry_source = REGISTRY.read_text(encoding="utf-8")
    sherpa_source = SHERPA.read_text(encoding="utf-8")
    bridge_source = BRIDGE.read_text(encoding="utf-8")

    for marker in (
        "canonicalLanguageCode.matches(language)",
        "Duplicate TTS candidate id",
        "supertonicRequiredFiles",
        'extra["lang"] == null || extra["lang"] == language',
        "must declare extra.lang",
        "must use a positive num_steps",
        "fun sharesModelWith(other: Candidate)",
        "val generationLanguage: String",
    ):
        assert marker in registry_source, f"runtime registry contract is missing: {marker}"
    for marker in (
        "Integer.compare(right.getAutoPriority(), left.getAutoPriority())",
        "firstAvailableAutoCandidate(autoCandidatesForLocale(localeTag))",
        "candidate = firstAvailableAutoCandidate(ranked)",
        "nextAvailableAutoCandidateForLocale",
        "sherpaTts.canReuseModel(candidate)",
        "sherpaTts.selectCandidate(candidate)",
        "candidateSupportsLocale(candidate, localeTag)",
        "candidateSupportsLocale(candidate, resolvedLocaleTag)",
        "return hasSystemTtsEngineForLocale(localeTag)",
        "canInitializeSystemTtsForLocale(resolvedLocaleTag)",
        "canInitializeSystemTtsForLocale(localeTag)",
        'systemFallbackAvailable ? "system" : "none"',
        "systemTtsCapabilitiesKnown",
        "systemTtsOfflineLanguages.contains(language)",
        "warmSystemTtsCapabilitiesOnUi",
        "isUsableOfflineVoice(voice)",
        "KEY_FEATURE_NOT_INSTALLED",
    ):
        assert marker in bridge_source, f"runtime auto-routing contract is missing: {marker}"
    assert '"generation_language" to c.generationLanguage' in sherpa_source
    assert "val candidate: Candidate" in sherpa_source and "task.candidate" in sherpa_source
    assert "autoCandidateForLocale(localeTag) != null || hasSystemTtsEngine()" not in bridge_source

    candidates = load_staged_candidates()
    validate_catalog(candidates)
    assert_rejected([])

    for required_field in ("id", "language", "engine", "files"):
        malformed = copy.deepcopy(candidates)
        del malformed[0][required_field]
        assert_rejected(malformed)

    duplicate = copy.deepcopy(candidates)
    duplicate[1]["id"] = duplicate[0]["id"]
    assert_rejected(duplicate)

    for invalid_language in ("JA", "ja-JP", "ja_JP", " ja", ""):
        malformed = copy.deepcopy(candidates)
        target = next(item for item in malformed if item["id"] == "supertonic3_ja_s5")
        target["language"] = invalid_language
        target["extra"]["lang"] = invalid_language
        assert_rejected(malformed)

    mismatch = copy.deepcopy(candidates)
    target = next(item for item in mismatch if item["id"] == "supertonic3_ja_s5")
    target["extra"]["lang"] = "ko"
    assert_rejected(mismatch)

    missing_extra = copy.deepcopy(candidates)
    target = next(item for item in missing_extra if item["id"] == "supertonic3_ja_s5")
    del target["extra"]["lang"]
    assert_rejected(missing_extra)

    non_supertonic_mismatch = copy.deepcopy(candidates)
    target = next(item for item in non_supertonic_mismatch if item["engine"] != "supertonic")
    target["extra"] = {"lang": "ja" if target["language"] != "ja" else "ko"}
    assert_rejected(non_supertonic_mismatch)

    for required_file in SUPERTONIC_REQUIRED_FILES:
        malformed = copy.deepcopy(candidates)
        target = next(item for item in malformed if item["id"] == "supertonic3_ko_s5")
        del target["files"][required_file]
        assert_rejected(malformed)

    empty_file_path = copy.deepcopy(candidates)
    empty_file_path[0]["files"][next(iter(empty_file_path[0]["files"]))] = ""
    assert_rejected(empty_file_path)

    routing_fixture = [
        {"id": "winner", "language": "ja", "auto_priority": 20},
        {"id": "fallback_a", "language": "ja", "auto_priority": 10},
        {"id": "fallback_b", "language": "ja", "auto_priority": 10},
        {"id": "manual_only", "language": "ja", "auto_priority": 0},
        {"id": "other_language", "language": "ko", "auto_priority": 100},
    ]
    ranked = ranked_auto_candidates(routing_fixture, "ja")
    assert [candidate["id"] for candidate in ranked] == [
        "winner",
        "fallback_a",
        "fallback_b",
    ]
    selected = first_available(ranked, {"fallback_b", "fallback_a"})
    assert selected and selected["id"] == "fallback_a"
    assert first_available(ranked, set()) is None

    after_winner_failure = next_available_after_failure(
        ranked,
        "winner",
        {"fallback_a", "fallback_b"},
    )
    assert after_winner_failure and after_winner_failure["id"] == "fallback_a"
    after_secondary_failure = next_available_after_failure(
        ranked,
        "fallback_a",
        {"fallback_b"},
    )
    assert after_secondary_failure and after_secondary_failure["id"] == "fallback_b"

    # Every Supertonic language route must share one graph signature. This is
    # the whole reason ES and FR cost zero extra bytes on the phone and switch
    # without a model reload: SherpaTtsEngine.canReuseModel() compares exactly
    # this signature.
    shared_ids = {
        "supertonic3_en_s5",
        "supertonic3_s5",
        "supertonic3_ja_s5",
        "supertonic3_ko_s5",
        "supertonic3_es_s5",
        "supertonic3_fr_s5",
    }
    shared_supertonic = {
        candidate["id"]: candidate
        for candidate in candidates
        if candidate["id"] in shared_ids
    }
    assert set(shared_supertonic) == shared_ids
    assert len({model_signature(candidate) for candidate in shared_supertonic.values()}) == 1
    assert {
        candidate["extra"]["lang"] for candidate in shared_supertonic.values()
    } == {"en", "ru", "ja", "ko", "es", "fr"}
    for candidate in shared_supertonic.values():
        assert int(candidate["auto_priority"]) > 0, candidate["id"]
        assert int(candidate["num_steps"]) == 5, candidate["id"]
    different_threads = copy.deepcopy(shared_supertonic["supertonic3_ja_s5"])
    different_threads["num_threads"] = int(different_threads["num_threads"]) + 1
    assert model_signature(different_threads) != model_signature(
        shared_supertonic["supertonic3_ja_s5"]
    )

    # Installation alone must never become a locale-support claim. The gate
    # opens only after the initialized engine reports a matching offline voice.
    assert not system_locale_available(True, False, {"fr"}, "fr-FR")
    assert not system_locale_available(True, True, {"fr"}, "de-DE")
    assert not system_locale_available(False, True, {"fr"}, "fr-FR")
    assert system_locale_available(True, True, {"fr", "ru"}, "fr-FR")

    japanese = shared_supertonic["supertonic3_ja_s5"]
    assert candidate_supports_locale(japanese, "ja-JP")
    assert not candidate_supports_locale(japanese, "ko-KR")

    print("TTS candidate registry and auto-routing contracts passed")


if __name__ == "__main__":
    main()
