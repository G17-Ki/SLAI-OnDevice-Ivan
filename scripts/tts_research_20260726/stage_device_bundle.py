#!/usr/bin/env python3
"""Assemble the on-device model bundle for the TTS bench app.

Copies exactly the files each candidate needs into one push-ready tree, dedupes
assets shared between candidates (eSpeak NG data, the Vocos vocoders), writes
the `candidates.json` the app reads, a phrase set, and a manifest with sizes and
SHA-256 digests.

The app reads `candidates.json` at runtime, so a candidate can be added,
removed, or re-tuned by editing the bundle and re-pushing it — no rebuild.

Nothing here touches a device; `tools/push_tts_bundle.sh` does the transfer.
"""

from __future__ import annotations

import argparse
import copy as copy_module
import hashlib
import json
import re
import shutil
from pathlib import Path

# Shared assets are stored once under shared/ and referenced by every candidate
# that needs them. eSpeak NG data alone is 17 MiB and is required by five
# candidates; copying it per candidate would add ~70 MiB for nothing.
SHARED = {
    "espeak-ng-data": ("vits-piper-ru_RU-dmitri-medium/espeak-ng-data", "shared/espeak-ng-data"),
    "vocos-22khz": ("vocos-22khz-univ.onnx", "shared/vocos-22khz-univ.onnx"),
    "vocos-24khz": ("vocos_24khz.onnx", "shared/vocos-24khz.onnx"),
}

KOKORO_INT8 = "kokoro-int8-v1_1-zh/kokoro-int8-multi-lang-v1_1"
KOKORO_FP32 = "kokoro-multi-lang-v1_1"
KOKORO_V1 = "kokoro-multi-lang-v1_0"
SUPERTONIC = "supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
ZIPVOICE = "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


# Thread counts measured on NX809J/SM8850, not guessed: see the thread-scaling
# table in DEVICE_RESULTS.md. The per-candidate `num_threads` literals below are
# the host default of 2 and are overwritten by device_threads() before the
# candidates file is written. Three behaviours drive this: MeloTTS and Piper
# scale to 8, Supertonic stops scaling past 2 (4 is its measured floor), and
# Matcha is *fastest at one thread* because thread setup costs more than it
# saves at that model size.
DEVICE_THREADS = {
    "matcha_zh_baker": 1,
    "melotts_zh": 8,
    "melotts_zh_int8": 8,
    "piper_zh_huayan": 8,
    "piper_ru_dmitri": 8,
    "piper_ru_irina": 8,
    "piper_ru_ruslan": 8,
}


def device_threads(candidate_id: str) -> int:
    """Device-optimal thread count; 4 for everything not individually measured."""
    return DEVICE_THREADS.get(candidate_id, 4)


def piper(
    candidate_id,
    label,
    voice_dir,
    stem,
    language,
    notes,
    known=None,
    auto_priority=0,
):
    return {
        "id": candidate_id, "label": label, "language": language, "engine": "vits",
        "num_threads": 2, "sid": 0, "speed": 1.0,
        "auto_priority": auto_priority,
        "files": {
            "model": f"{voice_dir}/{stem}.onnx",
            "tokens": f"{voice_dir}/tokens.txt",
            "data_dir": "shared/espeak-ng-data",
        },
        "notes": notes, **({"known_issue": known} if known else {}),
    }


def supertonic(steps, language="ru", speaker=6):
    language_labels = {
        "en": "EN",
        "ru": "RU",
        "ja": "JA",
        "ko": "KO",
        "es": "ES",
        "fr": "FR",
    }
    if language not in language_labels:
        raise ValueError(f"Unsupported Supertonic language: {language}")
    if not 0 <= speaker <= 9:
        raise ValueError("Supertonic speaker must be in the published range 0..9")
    if speaker == 6:
        candidate_id = (
            f"supertonic3_s{steps}"
            if language == "ru"
            else f"supertonic3_{language}_s{steps}"
        )
    else:
        candidate_id = f"supertonic3_{language}_v{speaker}_s{steps}"
    host_notes = {
        "en": {
            5: "host: Lane D D1 point estimate -8.9 pp WER; D3 6/6 short-input safe. Phone timing required.",
        },
        "ru": {
            4: "host: first chunk 0.437 s P95, WER 8.3%",
            5: "host: first chunk 0.529 s P95, WER 1.7% — recommended",
            6: "host: first chunk 0.704 s P95",
            8: "host: first chunk 1.316 s P95, RTF P95 0.679 — near the RTF gate",
        },
        "ja": {
            5: "host: first chunk 0.467 s P95; broader Japanese ASR backchecks remain mixed",
        },
        "ko": {
            5: "host: first chunk 0.628 s P95; short Korean sample retranscribed exactly",
        },
        # Lane D D3 measured Supertonic 24/24 short-input waveform passes across
        # EN/ES/FR/RU; Lane F F3 re-ran the ES/FR half through this exact
        # candidate configuration and the app's own PcmGate. Neither result is a
        # phone measurement and neither establishes intelligibility.
        "es": {
            5: "host: Lane D D3 6/6 short-input safe; Lane F F3 re-verified against PcmGate. Not phone-timed.",
        },
        "fr": {
            5: "host: Lane D D3 6/6 short-input safe; Lane F F3 re-verified against PcmGate. Not phone-timed.",
        },
    }
    if speaker == 6:
        label = f"Supertonic 3 {language_labels[language]} — {steps} steps"
        notes = host_notes[language][steps]
    else:
        label = f"Supertonic 3 {language_labels[language]} — voice {speaker}"
        notes = (
            f"Bundled voice-bank SID {speaker}; same Supertonic graph and 5-step "
            "setting as the current voice. Intelligibility and naturalness are "
            "unmeasured; user audition required."
        )
    auto_priority = 0
    if steps == 5 and speaker == 6:
        auto_priority = 10
    if steps == 5 and language == "en" and speaker == 4:
        auto_priority = 20
    return {
        "id": candidate_id,
        "label": label,
        "language": language, "engine": "supertonic",
        "num_threads": 2, "sid": speaker, "speed": 1.0, "num_steps": steps,
        "auto_priority": auto_priority,
        "extra": {"lang": language, "seed": "0"},
        "files": {
            "duration_predictor": f"{SUPERTONIC}/duration_predictor.int8.onnx",
            "text_encoder": f"{SUPERTONIC}/text_encoder.int8.onnx",
            "vector_estimator": f"{SUPERTONIC}/vector_estimator.int8.onnx",
            "vocoder": f"{SUPERTONIC}/vocoder.int8.onnx",
            "tts_json": f"{SUPERTONIC}/tts.json",
            "unicode_indexer": f"{SUPERTONIC}/unicode_indexer.bin",
            "voice_style": f"{SUPERTONIC}/voice.bin",
        },
        "notes": notes,
    }


def supertonic_voice_bank(language):
    """Expose every bundled Supertonic speaker without duplicating model bytes."""
    return [supertonic(5, language, speaker) for speaker in range(10)]


CANDIDATES = [
    # ---- Mandarin ----
    {
        "id": "matcha_zh_baker", "label": "Matcha ZH (baker)", "language": "zh",
        "engine": "matcha", "num_threads": 2, "sid": 0, "speed": 1.0,
        "auto_priority": 10,
        "files": {
            "acoustic_model": "matcha-icefall-zh-baker/model-steps-3.onnx",
            "vocoder": "shared/vocos-22khz-univ.onnx",
            "lexicon": "matcha-icefall-zh-baker/lexicon.txt",
            "tokens": "matcha-icefall-zh-baker/tokens.txt",
            "dict_dir": "matcha-icefall-zh-baker/dict",
        },
        "notes": "Host winner on every measured gate. Corpus is non-commercial only.",
    },
    {
        "id": "melotts_zh", "label": "MeloTTS ZH (float32)", "language": "zh",
        "engine": "vits", "num_threads": 2, "sid": 0, "speed": 1.0,
        "files": {
            "model": "vits-melo-tts-zh_en/model.onnx",
            "lexicon": "vits-melo-tts-zh_en/lexicon.txt",
            "tokens": "vits-melo-tts-zh_en/tokens.txt",
            "dict_dir": "vits-melo-tts-zh_en/dict",
        },
        "notes": "MIT, no eSpeak dependency, exact-QRD NPU chain passes. Host RTF 0.715 — too slow on CPU.",
    },
    {
        "id": "melotts_zh_int8", "label": "MeloTTS ZH (INT8)", "language": "zh",
        "engine": "vits", "num_threads": 2, "sid": 0, "speed": 1.0,
        "files": {
            "model": "vits-melo-tts-zh_en/model.int8.onnx",
            "lexicon": "vits-melo-tts-zh_en/lexicon.txt",
            "tokens": "vits-melo-tts-zh_en/tokens.txt",
            "dict_dir": "vits-melo-tts-zh_en/dict",
        },
        "notes": "Included to confirm the host finding on device.",
        "known_issue": "Host RTF 2.0-2.9, three times slower than float32. Expect it to be unusable.",
    },
    piper("piper_zh_huayan", "Piper ZH (huayan)", "vits-piper-zh_CN-huayan-medium",
          "zh_CN-huayan-medium", "zh",
          "Fast and clean on host. Corpus licence is Unknown, so it cannot be recommended."),
    {
        "id": "kokoro_zh_fp32", "label": "Kokoro ZH (float32)", "language": "zh",
        "engine": "kokoro", "num_threads": 2, "sid": 3, "speed": 1.0,
        "files": {
            "model": f"{KOKORO_FP32}/model.onnx",
            "voices": f"{KOKORO_FP32}/voices.bin",
            "tokens": f"{KOKORO_FP32}/tokens.txt",
            "lexicon": f"{KOKORO_FP32}/lexicon-us-en.txt,{KOKORO_FP32}/lexicon-zh.txt",
            "data_dir": "shared/espeak-ng-data",
            "dict_dir": f"{KOKORO_FP32}/dict",
        },
        "notes": "Correct at short inputs, unlike the INT8 build. Host RTF ~0.97 — barely faster than real time.",
    },
    {
        "id": "kokoro_zh_int8", "label": "Kokoro ZH (INT8)", "language": "zh",
        "engine": "kokoro", "num_threads": 2, "sid": 3, "speed": 1.0,
        "files": {
            "model": f"{KOKORO_INT8}/model.int8.onnx",
            "voices": f"{KOKORO_INT8}/voices.bin",
            "tokens": f"{KOKORO_INT8}/tokens.txt",
            "lexicon": f"{KOKORO_INT8}/lexicon-us-en.txt,{KOKORO_INT8}/lexicon-zh.txt",
            "data_dir": "shared/espeak-ng-data",
            "dict_dir": f"{KOKORO_INT8}/dict",
        },
        "notes": "Included only so the INT8 defect can be reproduced on device.",
        "known_issue": "Returns all-non-finite PCM for inputs under 3 Han characters. "
                       "The app's gate drops those chunks; you will hear silence, not noise.",
    },
    {
        "id": "zipvoice_zh", "label": "ZipVoice-distill ZH (ceiling)", "language": "zh",
        "engine": "zipvoice", "num_threads": 2, "sid": 0, "speed": 1.0,
        "files": {
            "encoder": f"{ZIPVOICE}/encoder.int8.onnx",
            "decoder": f"{ZIPVOICE}/decoder.int8.onnx",
            "vocoder": "shared/vocos-24khz.onnx",
            "tokens": f"{ZIPVOICE}/tokens.txt",
            "lexicon": f"{ZIPVOICE}/lexicon.txt",
            "data_dir": "shared/espeak-ng-data",
        },
        "reference_wav": f"{ZIPVOICE}/test_wavs/news-female.wav",
        "reference_text": "各位村民, 大家新年好! 近期, 湖北省武汉市等多个地区",
        "notes": "Zero-shot: clones the prompt clip. Quality ceiling reference, not a phone candidate.",
        "known_issue": "Host RTF 0.8-3.4 and 1 GiB peak RSS. Expect multi-second first audio.",
    },
    # ---- English ----
    {
        "id": "kokoro_en_sarah_v1",
        "label": "Kokoro EN — Sarah",
        "language": "en",
        "engine": "kokoro",
        "num_threads": 2,
        "sid": 9,
        "speed": 1.0,
        "auto_priority": 0,
        "files": {
            "model": f"{KOKORO_V1}/model.onnx",
            "voices": f"{KOKORO_V1}/voices.bin",
            "tokens": f"{KOKORO_V1}/tokens.txt",
            "lexicon": f"{KOKORO_V1}/lexicon-us-en.txt,{KOKORO_V1}/lexicon-zh.txt",
            "data_dir": "shared/espeak-ng-data",
            "dict_dir": f"{KOKORO_V1}/dict",
        },
        "notes": "Kokoro-82M v1.0 af_sarah (speaker id 9); user listening approval pending.",
    },
    # All ten styles reuse the same multilingual Supertonic graphs and voice bank.
    *supertonic_voice_bank("en"),
    # ---- Russian ----
    supertonic(4), *supertonic_voice_bank("ru"), supertonic(6), supertonic(8),
    piper("piper_ru_dmitri", "Piper RU (dmitri, CC0)", "vits-piper-ru_RU-dmitri-medium",
          "ru_RU-dmitri-medium", "ru",
          "Only 'clear' verdict on host and the only CC0 Russian corpus found."),
    piper("piper_ru_irina", "Piper RU (irina)", "vits-piper-ru_RU-irina-medium",
          "ru_RU-irina-medium", "ru",
          "Best measured Russian intelligibility. Corpus licence Unknown — reference only."),
    piper("piper_ru_ruslan", "Piper RU (ruslan)", "vits-piper-ru_RU-ruslan-medium",
          "ru_RU-ruslan-medium", "ru",
          "Highest proxy score of the Russian voices. Corpus is CC BY-NC-SA 4.0.",
          auto_priority=20),
    # ---- Japanese and Korean (reuse the same multilingual Supertonic graphs) ----
    supertonic(5, "ja"),
    supertonic(5, "ko"),
    # ---- Spanish and French (same graphs again; zero extra bytes on device) ----
    supertonic(5, "es"),
    supertonic(5, "fr"),
]

PHRASES = {
    "zh": [
        ("micro", "好的。"),
        ("clause", "北京是中国的首都。"),
        ("answer", "北京是中国的首都，也是全国的政治和文化中心。"),
        ("weather", "今天下午天气晴朗，气温大约二十四度，出门不需要带伞。"),
        ("question", "你想让我把这段话翻译成英文吗？"),
        ("long", "光合作用是植物把光能转换成化学能的过程。叶绿体吸收阳光，并利用水和二氧化碳合成葡萄糖。"
                 "这个过程同时释放氧气，供其他生物呼吸使用。因此光合作用是地球上大多数生态系统的能量基础。"),
        ("floor1", "好"),
        ("floor2", "好的"),
    ],
    "ru": [
        ("micro", "Конечно."),
        ("clause", "Москва — столица России."),
        ("answer", "Москва — столица России и её крупнейший город."),
        ("weather", "Сегодня днём будет ясно, около двадцати четырёх градусов, зонт брать не нужно."),
        ("question", "Хочешь, я переведу этот текст на английский?"),
        ("long", "Фотосинтез — это процесс, в котором растения превращают энергию света в химическую энергию. "
                 "Хлоропласты поглощают солнечный свет и используют воду и углекислый газ для синтеза глюкозы. "
                 "При этом выделяется кислород, которым дышат другие организмы."),
    ],
    "ja": [
        ("micro", "もちろんです。"),
        ("clause", "答えは三十二です。"),
        ("answer", "十三足す十九の答えは三十二です。"),
    ],
    "ko": [
        ("micro", "물론입니다."),
        ("clause", "답은 32입니다."),
        ("answer", "13 더하기 19의 답은 32입니다."),
    ],
    "es": [
        ("micro", "Claro."),
        ("clause", "Madrid es la capital de España."),
        ("answer", "Madrid es la capital de España y su ciudad más grande."),
    ],
    "fr": [
        ("micro", "Bien sûr."),
        ("clause", "Paris est la capitale de la France."),
        ("answer", "Paris est la capitale de la France et sa plus grande ville."),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    shutil.copy2(src, dst)
    return dst.stat().st_size


def candidate_references(candidate):
    refs = {
        part.strip()
        for rel in candidate["files"].values()
        for part in rel.split(",")
        if part.strip()
    }
    if candidate.get("reference_wav"):
        refs.add(candidate["reference_wav"])
    return refs


def referenced_manifest_files(reference, files):
    if reference in files:
        return {reference}
    prefix = reference.rstrip("/") + "/"
    return {path for path in files if path.startswith(prefix)}


def write_catalog(out, selected):
    (out / "candidates.json").write_text(
        json.dumps({"schema_version": 1, "candidates": selected}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "phrases.json").write_text(
        json.dumps(
            {lang: [{"id": i, "text": t} for i, t in items] for lang, items in PHRASES.items()},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def stage_resident_metadata(resident_manifest, out, selected):
    """Write a metadata-only tree that reuses checksum-pinned resident assets."""
    source = json.loads(resident_manifest.read_text(encoding="utf-8"))
    if source.get("schema_version") != 1 or not isinstance(source.get("files"), dict):
        raise ValueError("resident manifest must be a schema-version-1 bundle manifest")
    if source.get("missing_sources"):
        raise ValueError("resident manifest records missing sources")

    asset_files = {
        path: spec
        for path, spec in source["files"].items()
        if path not in {"candidates.json", "phrases.json", "bundle_manifest.json"}
    }
    for path, spec in asset_files.items():
        if not isinstance(spec, dict):
            raise ValueError(f"resident manifest entry for {path} must be an object")
        size = spec.get("bytes")
        digest = spec.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"resident manifest entry for {path} has invalid bytes")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"resident manifest entry for {path} requires SHA-256")
        spec["sha256"] = digest.lower()
    missing = []
    per_candidate = {}
    for candidate in selected:
        resolved = set()
        for reference in candidate_references(candidate):
            matched = referenced_manifest_files(reference, asset_files)
            if not matched:
                missing.append(f"{candidate['id']}:{reference}")
            resolved.update(matched)
        size = sum(asset_files[path]["bytes"] for path in resolved)
        per_candidate[candidate["id"]] = {
            "bytes": size,
            "mib": round(size / 1048576, 1),
            "language": candidate["language"],
            "engine": candidate["engine"],
        }
    if missing:
        raise ValueError("resident manifest is missing candidate assets: " + ", ".join(missing))

    write_catalog(out, selected)
    files = dict(asset_files)
    for name in ("candidates.json", "phrases.json"):
        path = out / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "candidate_count": len(selected),
        "total_bytes": sum(spec["bytes"] for spec in asset_files.values()),
        "total_mib": round(sum(spec["bytes"] for spec in asset_files.values()) / 1048576, 1),
        "per_candidate_mib_including_shared": per_candidate,
        "missing_sources": [],
        "files": files,
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-root", type=Path)
    source.add_argument(
        "--resident-manifest",
        type=Path,
        help="write metadata only while reusing assets authenticated by this existing manifest",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="candidate id to leave out of the bundle")
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    selected = [
        copy_module.deepcopy(candidate)
        for candidate in CANDIDATES
        if candidate["id"] not in args.exclude
    ]
    for candidate in selected:
        candidate["num_threads"] = device_threads(candidate["id"])

    if args.resident_manifest:
        resident_manifest = args.resident_manifest.expanduser().resolve()
        try:
            manifest = stage_resident_metadata(resident_manifest, out, selected)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            print(f"Could not stage resident metadata: {error}")
            return 1
        print(f"metadata-only bundle: {out}")
        print(f"candidates: {len(selected)}   resident assets: {manifest['total_mib']} MiB")
        for cid, info in sorted(
            manifest["per_candidate_mib_including_shared"].items(),
            key=lambda item: -item[1]["mib"],
        ):
            print(
                f"  {cid:<22} {info['language']}  {info['mib']:>7.1f} MiB  "
                f"({info['engine']})"
            )
        return 0

    root = args.model_root.expanduser().resolve()

    copied: dict[str, int] = {}
    missing: list[str] = []

    for _, (src_rel, dst_rel) in SHARED.items():
        src = root / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        copied[dst_rel] = copy(src, out / dst_rel)

    for cand in selected:
        for key, rel in cand["files"].items():
            for part in rel.split(","):
                part = part.strip()
                if part.startswith("shared/") or part in copied:
                    continue
                src = root / part
                if not src.exists():
                    missing.append(f"{cand['id']}:{part}")
                    continue
                copied[part] = copy(src, out / part)
        ref = cand.get("reference_wav")
        if ref and ref not in copied:
            src = root / ref
            if src.exists():
                copied[ref] = copy(src, out / ref)
            else:
                missing.append(f"{cand['id']}:{ref}")

    write_catalog(out, selected)

    total = sum(copied.values())
    per_candidate = {}
    for cand in selected:
        files = {p.strip() for rel in cand["files"].values() for p in rel.split(",")}
        if cand.get("reference_wav"):
            files.add(cand["reference_wav"])
        per_candidate[cand["id"]] = {
            "bytes": sum(copied.get(f, 0) for f in files),
            "mib": round(sum(copied.get(f, 0) for f in files) / 1048576, 1),
            "language": cand["language"],
            "engine": cand["engine"],
        }

    manifest = {
        "schema_version": 1,
        "candidate_count": len(selected),
        "total_bytes": total,
        "total_mib": round(total / 1048576, 1),
        "per_candidate_mib_including_shared": per_candidate,
        "missing_sources": sorted(set(missing)),
        "files": {
            str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"bundle: {out}")
    print(f"candidates: {len(selected)}   total: {manifest['total_mib']} MiB")
    for cid, info in sorted(per_candidate.items(), key=lambda kv: -kv[1]["mib"]):
        print(f"  {cid:<22} {info['language']}  {info['mib']:>7.1f} MiB  ({info['engine']})")
    if missing:
        print("\nMISSING SOURCES (bundle is incomplete):")
        for m in sorted(set(missing)):
            print(f"  {m}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
