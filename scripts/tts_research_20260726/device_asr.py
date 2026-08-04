#!/usr/bin/env python3
"""Run the fixed-locale ASR judge over device-rendered audio.

Same judge, same normalization, same hallucination and digit-rendering
handling as the host run, so a device CER can be compared with a host CER
directly instead of being a separate incomparable number.

The point is narrow but important: the host proved these engines produce
intelligible audio *on the host*. Nothing so far proves the phone build of the
same engine does. Different runtime, different ABI, different thread count.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asr_judge import (  # noqa: E402
    digit_rendering_mismatch,
    dominant_script,
    error_rates,
    EXPECTED_SCRIPT,
)

# Device phrase id -> the reference text it was asked to speak. Kept here rather
# than read from phrases.json so a mismatch between what was pushed and what is
# scored cannot pass silently.
REFERENCES = {
    "zh": {
        "micro": "好的。",
        "clause": "北京是中国的首都。",
        "answer": "北京是中国的首都，也是全国的政治和文化中心。",
        "weather": "今天下午天气晴朗，气温大约二十四度，出门不需要带伞。",
        "question": "你想让我把这段话翻译成英文吗？",
        "long": "光合作用是植物把光能转换成化学能的过程。叶绿体吸收阳光，并利用水和二氧化碳合成葡萄糖。"
                "这个过程同时释放氧气，供其他生物呼吸使用。因此光合作用是地球上大多数生态系统的能量基础。",
        "floor1": "好",
        "floor2": "好的",
    },
    "ru": {
        "micro": "Конечно.",
        "clause": "Москва — столица России.",
        "answer": "Москва — столица России и её крупнейший город.",
        "weather": "Сегодня днём будет ясно, около двадцати четырёх градусов, зонт брать не нужно.",
        "question": "Хочешь, я переведу этот текст на английский?",
        "long": "Фотосинтез — это процесс, в котором растения превращают энергию света в химическую энергию. "
                "Хлоропласты поглощают солнечный свет и используют воду и углекислый газ для синтеза глюкозы. "
                "При этом выделяется кислород, которым дышат другие организмы.",
    },
}

RU_CANDIDATES = ("supertonic3", "piper_ru")


def language_of(candidate_id: str) -> str:
    return "ru" if candidate_id.startswith(RU_CANDIDATES) else "zh"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--host-asr", type=Path, action="append", default=[],
                    help="host judge output(s) to compare against")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=2)

    host_cer = {}
    for path in args.host_asr:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, engine in data["engines"].items():
            host_cer[key] = engine.get("micro_cer")

    per_engine = defaultdict(lambda: {"cer_edits": 0, "ref_chars": 0,
                                      "wer_edits": 0, "ref_words": 0, "cases": []})

    for wav in sorted(args.wav_dir.glob("*.wav")):
        stem = wav.stem
        if "__" not in stem:
            continue
        candidate_id, phrase_id = stem.split("__", 1)
        lang = language_of(candidate_id)
        reference = REFERENCES.get(lang, {}).get(phrase_id)
        if reference is None:
            continue

        segments, _ = model.transcribe(
            str(wav), language=lang, beam_size=5,
            condition_on_previous_text=False, vad_filter=False,
        )
        hypothesis = "".join(s.text for s in segments)
        script = dominant_script(hypothesis)
        entry = {
            "phrase_id": phrase_id,
            "wav": wav.name,
            "reference": reference,
            "hypothesis": hypothesis,
            "dominant_script": script,
            "script_ok": script == EXPECTED_SCRIPT[lang],
        }
        entry.update(error_rates(reference, hypothesis, lang))
        entry["digit_rendering_mismatch"] = digit_rendering_mismatch(
            entry["reference_normalized"], entry["hypothesis_normalized"]
        )
        bucket = per_engine[candidate_id]
        bucket["language"] = lang
        bucket["cases"].append(entry)
        print(f"{candidate_id:<18} {phrase_id:<9} cer={entry['cer']:.4f} "
              f"script={script}", flush=True)

    # Exclusions must match the host exactly or the comparison is meaningless.
    # The host micro rate drops (a) degenerate floor phrases and (b) any case
    # where the judge rendered digits for spelled-out numerals, unioned across
    # engines. Scoring the device over a different case set made every candidate
    # look worse on device than on host, which was an artifact.
    excluded_per_lang = defaultdict(set)
    for cid, b in per_engine.items():
        for case in b["cases"]:
            if case["phrase_id"].startswith("floor") or case.get("digit_rendering_mismatch"):
                excluded_per_lang[b["language"]].add(case["phrase_id"])

    for cid, b in per_engine.items():
        excluded = excluded_per_lang[b["language"]]
        for case in b["cases"]:
            if case["phrase_id"] in excluded:
                continue
            b["cer_edits"] += case["cer_edits"]
            b["ref_chars"] += case["ref_char_count"]
            if "wer_edits" in case:
                b["wer_edits"] += case["wer_edits"]
                b["ref_words"] += case["ref_word_count"]

    engines = {}
    for cid, b in per_engine.items():
        micro_cer = b["cer_edits"] / b["ref_chars"] if b["ref_chars"] else None
        micro_wer = b["wer_edits"] / b["ref_words"] if b["ref_words"] else None
        host_key = "kokoro_zh" if cid == "kokoro_zh_int8" else cid
        engines[cid] = {
            "language": b.get("language"),
            "device_micro_cer": micro_cer,
            "device_micro_wer": micro_wer,
            "host_micro_cer": host_cer.get(host_key),
            "cer_delta_device_minus_host": (
                micro_cer - host_cer[host_key]
                if micro_cer is not None and host_cer.get(host_key) is not None else None
            ),
            "excluded_phrases": sorted(excluded_per_lang[b["language"]]),
            "cases": b["cases"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": 1, "judge": args.model.split("/")[-1],
                    "note": "floor phrases and digit-rendering cases excluded from micro rates, unioned per language, matching the host methodology",
                    "engines": engines}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== device vs host micro-CER ===")
    for cid, e in sorted(engines.items()):
        h = e["host_micro_cer"]
        d = e["device_micro_cer"]
        delta = e["cer_delta_device_minus_host"]
        print(f"  {cid:<18} device={d if d is None else f'{d:.4f}':<8} "
              f"host={'-' if h is None else f'{h:.4f}':<8} "
              f"delta={'-' if delta is None else f'{delta:+.4f}'}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
