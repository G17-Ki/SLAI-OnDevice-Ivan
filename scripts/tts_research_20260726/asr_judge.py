#!/usr/bin/env python3
"""Fixed-locale ASR backcheck for synthesized speech.

This is an intelligibility screen, not a naturalness measurement. A low CER
proves the words survived synthesis; it says nothing about whether the voice
sounds human. Rankings must not be built on this number alone.

The judge is deliberately locked to one language per invocation. Letting
Whisper auto-detect would let a candidate "pass" by producing the wrong script,
which is one of the hard failures this run is trying to catch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


# Punctuation and whitespace are removed on both sides; nothing else is edited.
_PUNCT = re.compile(
    r"[\s"
    r"　-〿"     # CJK punctuation
    r"＀-￯"     # fullwidth forms
    r"!-/:-@\[-`{-~"     # ASCII punctuation
    r"‐-‧‰-⁞"  # general punctuation incl. em dash
    r"]+"
)


_DIGIT_RUN = re.compile(r"\d")


def normalize(text: str, locale: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    if locale == "zh":
        # Whisper freely emits traditional Han for simplified input. That is a
        # rendering choice, not a synthesis defect, so both sides are folded to
        # simplified before scoring.
        try:
            import zhconv

            text = zhconv.convert(text, "zh-cn")
        except ImportError:
            pass
    text = _PUNCT.sub("" if locale == "zh" else " ", text)
    if locale == "ru":
        # Standard Russian ASR scoring folds ё to е on both sides.
        text = text.replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


# Whisper was trained on subtitle corpora and hallucinates their credit lines
# onto trailing silence. On this run it appended "редактор субтитров н
# закомолдина корректор а егорова" to the end of well-synthesized Russian
# paragraphs, which inflated CER from ~0.02 to ~0.18 and would have made higher
# Supertonic step counts look like a regression. The text is not in the audio,
# so scoring it as a synthesis error is simply wrong.
#
# Only a *trailing* span starting at one of these markers is removed, and the
# raw hypothesis is always kept, so the edit is auditable and cannot silently
# delete real recognized speech from the middle of an utterance.
_HALLUCINATION_MARKERS = (
    "редактор субтитров",
    "корректор",
    "субтитры сделал",
    "субтитры создавал",
    "субтитры и перевод",
    "продолжение следует",
    "amara.org",
    "字幕由",
    "字幕志愿者",
    "请不吝点赞",
    "明镜与点点栏目",
    "thanks for watching",
    "subscribe to",
)


def strip_trailing_hallucination(
    normalized: str, min_start: int | None = None
) -> tuple[str, str | None]:
    """Remove a trailing subtitle-credit hallucination. Returns (kept, removed).

    ``min_start`` is the earliest index a marker may occupy and still count as
    trailing. Callers that know the reference pass a point just inside it, so
    the marker must appear after the utterance the engine was asked to speak.
    Standalone it defaults to the midpoint, which keeps a marker word occurring
    in genuine content from being treated as an artifact.

    When several markers match, the *earliest* qualifying one wins: Whisper
    emits a whole credit block ("редактор субтитров … корректор …"), and
    stripping from the later marker would leave half of it scored as speech.
    """
    lowered = normalized.lower()
    floor = len(lowered) // 2 if min_start is None else max(min_start, 1)
    best = None
    for marker in _HALLUCINATION_MARKERS:
        index = lowered.find(marker)
        while index != -1:
            if index >= floor:
                best = index if best is None else min(best, index)
                break
            index = lowered.find(marker, index + 1)
    if best is None:
        return normalized, None
    return normalized[:best].strip(), normalized[best:].strip()


def digit_rendering_mismatch(reference: str, hypothesis: str) -> bool:
    """True when the judge wrote Arabic digits for spelled-out numbers.

    Whisper transcribes spoken numerals as digits ("24") while these fixtures
    spell them out ("двадцати четырёх", "二十四"). Both are correct renderings
    of the same audio, so a case in this state measures the judge's inverse text
    normalization rather than the synthesizer, and its error rate is reported
    separately instead of being folded into the headline micro rate.
    """
    return bool(_DIGIT_RUN.search(hypothesis)) and not _DIGIT_RUN.search(reference)


def _levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str, locale: str) -> dict:
    ref = normalize(reference, locale)
    hyp_raw = normalize(hypothesis, locale)
    # A credit block can only be trailing if it starts at or beyond the text
    # the engine was actually asked to speak.
    hyp, removed = strip_trailing_hallucination(hyp_raw, int(len(ref) * 0.8))
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    cer_edits = _levenshtein(ref_chars, hyp_chars)
    out = {
        "reference_normalized": ref,
        "hypothesis_normalized": hyp,
        "ref_char_count": len(ref_chars),
        "cer_edits": cer_edits,
        "cer": cer_edits / len(ref_chars) if ref_chars else None,
    }
    if removed:
        out["hypothesis_before_hallucination_strip"] = hyp_raw
        out["stripped_trailing_hallucination"] = removed
    if locale != "zh":
        ref_words = ref.split()
        hyp_words = hyp.split()
        wer_edits = _levenshtein(ref_words, hyp_words)
        out.update(
            {
                "ref_word_count": len(ref_words),
                "wer_edits": wer_edits,
                "wer": wer_edits / len(ref_words) if ref_words else None,
            }
        )
    return out


def dominant_script(text: str) -> str:
    counts = {"han": 0, "cyrillic": 0, "latin": 0, "other": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            counts["other"] += 1
            continue
        if "CJK" in name:
            counts["han"] += 1
        elif "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values())
    if total == 0:
        return "empty"
    return max(counts, key=counts.get)


EXPECTED_SCRIPT = {"zh": "han", "ru": "cyrillic", "en": "latin", "fr": "latin", "it": "latin", "es": "latin"}


def rescore(source: Path, output: Path) -> int:
    """Recompute error rates from stored hypotheses without re-running ASR.

    The raw transcript is the expensive, immutable evidence; normalization is a
    scoring decision that may legitimately be revised. Keeping them separable
    means a normalization fix never silently reruns (or worse, changes) the
    recognition itself.
    """
    data = json.loads(source.read_text(encoding="utf-8"))
    locale = data["locale"]
    data.setdefault("rescored_from", source.name)

    for entry in data.get("calibration", []):
        if "hypothesis" in entry and "reference" in entry:
            entry.update(error_rates(entry["reference"], entry["hypothesis"], locale))

    for engine in data["engines"].values():
        for case in engine["cases"]:
            if "hypothesis" not in case:
                continue
            reference = case.get("reference")
            if reference is None:
                # Older payloads stored only the normalized reference.
                reference = case.get("reference_normalized", "")
            case.update(error_rates(reference, case["hypothesis"], locale))
            case["digit_rendering_mismatch"] = digit_rendering_mismatch(
                case["reference_normalized"], case["hypothesis_normalized"]
            )

    # Whether the judge writes digits for a given case depends on that engine's
    # audio, so a per-engine exclusion set would score candidates over different
    # cases. Exclude the union instead, so every engine is compared on exactly
    # the same material.
    excluded = sorted({
        case["case_id"]
        for engine in data["engines"].values()
        for case in engine["cases"]
        if case.get("digit_rendering_mismatch")
    })
    data["digit_rendering_cases_excluded"] = excluded

    for key, engine in data["engines"].items():
        cer_edits = ref_chars = wer_edits = ref_words = 0
        all_cer_edits = all_ref_chars = 0
        for case in engine["cases"]:
            if "cer_edits" not in case:
                continue
            all_cer_edits += case["cer_edits"]
            all_ref_chars += case["ref_char_count"]
            if case["case_id"] in excluded:
                continue
            cer_edits += case["cer_edits"]
            ref_chars += case["ref_char_count"]
            if "wer_edits" in case:
                wer_edits += case["wer_edits"]
                ref_words += case["ref_word_count"]
        engine["micro_cer"] = cer_edits / ref_chars if ref_chars else None
        engine["micro_wer"] = wer_edits / ref_words if ref_words else None
        engine["micro_cer_including_digit_cases"] = (
            all_cer_edits / all_ref_chars if all_ref_chars else None
        )
        engine["digit_rendering_cases_excluded"] = excluded
        engine["scored_case_count"] = sum(
            1 for c in engine["cases"] if "cer" in c and c["case_id"] not in excluded
        )
        print(f"  {key}: micro_cer={engine['micro_cer']} micro_wer={engine['micro_wer']}")
    print(f"  (cases excluded for all engines: {excluded})")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore-input", type=Path,
                    help="recompute rates from an existing judge output; no ASR is run")
    ap.add_argument("--bench-json", type=Path, action="append",
                    help="bench.py output; may be repeated")
    ap.add_argument("--locale", choices=sorted(EXPECTED_SCRIPT))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--extra-wav", type=Path, action="append", default=[],
                    help="path=reference pairs are given via --calibration instead")
    ap.add_argument("--calibration", action="append", default=[],
                    help="wav_path::reference_text, used to measure the judge's own floor")
    args = ap.parse_args()

    if args.rescore_input:
        return rescore(args.rescore_input, args.output)
    if not args.bench_json or not args.locale:
        ap.error("--bench-json and --locale are required unless --rescore-input is used")

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type, cpu_threads=2)

    def transcribe(path: Path) -> str:
        segments, _info = model.transcribe(
            str(path),
            language=args.locale,
            beam_size=args.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return "".join(seg.text for seg in segments)

    results = {"schema_version": 1, "locale": args.locale,
               "judge": {"model": args.model, "compute_type": args.compute_type,
                         "beam_size": args.beam_size, "language_forced": args.locale},
               "calibration": [], "engines": {}}

    for spec in args.calibration:
        wav, _, ref = spec.partition("::")
        p = Path(wav)
        if not p.exists():
            results["calibration"].append({"wav": p.name, "error": "missing"})
            continue
        hyp = transcribe(p)
        entry = {"wav": p.name, "reference": ref, "hypothesis": hyp,
                 "dominant_script": dominant_script(hyp)}
        entry.update(error_rates(ref, hyp, args.locale))
        results["calibration"].append(entry)
        print(f"[calibration] {p.name}: cer={entry['cer']}", flush=True)

    for bench_path in args.bench_json:
        if not bench_path.is_file():
            # A candidate whose bench run failed must not take the whole judge
            # down with it; record the absence instead.
            results.setdefault("missing_bench_json", []).append(bench_path.name)
            print(f"[skip] missing bench json: {bench_path.name}", flush=True)
            continue
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        audio_dir = bench_path.parent / "audio"
        key = f"{bench['engine_id']}{bench.get('tag', '')}"
        cases = []
        total_cer_edits = total_ref_chars = 0
        total_wer_edits = total_ref_words = 0
        for case_id, summary in bench["summary"].items():
            wav_name = summary.get("wav_path")
            reference = summary.get("reference")
            if not wav_name or not reference:
                cases.append({"case_id": case_id, "skipped": "no_wav_or_reference",
                              "gate_failures": summary.get("hard_failures", [])})
                continue
            wav = audio_dir / wav_name
            if not wav.exists():
                cases.append({"case_id": case_id, "skipped": "wav_missing"})
                continue
            hyp = transcribe(wav)
            script = dominant_script(hyp)
            entry = {"case_id": case_id, "wav": wav_name, "reference": reference,
                     "hypothesis": hyp, "dominant_script": script,
                     "script_ok": script == EXPECTED_SCRIPT[args.locale]}
            entry.update(error_rates(reference, hyp, args.locale))
            entry["digit_rendering_mismatch"] = digit_rendering_mismatch(
                entry["reference_normalized"], entry["hypothesis_normalized"]
            )
            if not hyp.strip():
                entry["hard_failure"] = "empty_asr_output"
            elif not entry["script_ok"]:
                entry["hard_failure"] = f"wrong_dominant_script:{script}"
            cases.append(entry)
            if not entry["digit_rendering_mismatch"]:
                total_cer_edits += entry["cer_edits"]
                total_ref_chars += entry["ref_char_count"]
                if "wer_edits" in entry:
                    total_wer_edits += entry["wer_edits"]
                    total_ref_words += entry["ref_word_count"]
            print(f"[{key}] {case_id}: cer={entry['cer']:.4f}"
                  + (f" wer={entry['wer']:.4f}" if "wer" in entry else "")
                  + (" [digit-rendering, excluded from micro]" if entry["digit_rendering_mismatch"] else ""),
                  flush=True)

        excluded = [c["case_id"] for c in cases if c.get("digit_rendering_mismatch")]
        all_cer_edits = sum(c["cer_edits"] for c in cases if "cer_edits" in c)
        all_ref_chars = sum(c["ref_char_count"] for c in cases if "ref_char_count" in c)
        results["engines"][key] = {
            "engine_id": bench["engine_id"],
            "tag": bench.get("tag", ""),
            "micro_cer": total_cer_edits / total_ref_chars if total_ref_chars else None,
            "micro_wer": total_wer_edits / total_ref_words if total_ref_words else None,
            "micro_cer_including_digit_cases": all_cer_edits / all_ref_chars if all_ref_chars else None,
            "digit_rendering_cases_excluded": excluded,
            "scored_case_count": sum(1 for c in cases if "cer" in c and not c.get("digit_rendering_mismatch")),
            "hard_failures": sorted({c["hard_failure"] for c in cases if "hard_failure" in c}),
            "cases": cases,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    for key, val in results["engines"].items():
        print(f"  {key}: micro_cer={val['micro_cer']} micro_wer={val['micro_wer']} "
              f"failures={val['hard_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
