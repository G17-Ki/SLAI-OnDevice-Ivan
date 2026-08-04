#!/usr/bin/env python3
"""Run Kokoro v1.0 q8f16 for Lane D's EN/ES/FR D1, D3, and D5 cells."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
from kokoro_onnx import Kokoro


LANGUAGE_CONFIG = {
    "en": {"lang_code": "en-us", "voice": "af_sarah"},
    "es": {"lang_code": "es", "voice": "ef_dora"},
    "fr": {"lang_code": "fr-fr", "voice": "ff_siwis"},
}

SHORT_INPUTS = {
    "en": [
        ("char_1", "I"),
        ("char_2", "OK"),
        ("char_3", "yes"),
        ("word_1", "yes"),
        ("word_2", "yes now"),
        ("word_3", "yes right now"),
    ],
    "es": [
        ("char_1", "a"),
        ("char_2", "sí"),
        ("char_3", "sol"),
        ("word_1", "sí"),
        ("word_2", "sí claro"),
        ("word_3", "sí claro ahora"),
    ],
    "fr": [
        ("char_1", "à"),
        ("char_2", "or"),
        ("char_3", "oui"),
        ("word_1", "oui"),
        ("word_2", "oui merci"),
        ("word_3", "oui merci beaucoup"),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility_create_audio(self: Kokoro, phonemes: str, voice: np.ndarray, speed: float):
    """Use float speed for onnx-community's input_ids export.

    kokoro-onnx 0.5.0 sends int32 speed to every input_ids graph, while this
    pinned q8f16 graph declares tensor(float).  No weights or graph are changed.
    """

    tokens = np.asarray(self.tokenizer.tokenize(phonemes[:510]), dtype=np.int64)
    style = np.asarray(voice[len(tokens)], dtype=np.float32)
    inputs = {
        "input_ids": np.asarray([[0, *tokens.tolist(), 0]], dtype=np.int64),
        "style": style,
        "speed": np.asarray([speed], dtype=np.float32),
    }
    audio = self.sess.run(None, inputs)[0].reshape(-1).astype(np.float32, copy=False)
    return audio, 24000


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def inspect_waveform(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    samples = np.asarray(samples).reshape(-1)
    finite = bool(samples.size and np.isfinite(samples).all())
    duration = samples.size / sample_rate if sample_rate else 0.0
    if finite and samples.size:
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples)))
        clip_level = 10 ** (-0.1 / 20)
        longest_clip_samples = longest_true_run(np.abs(samples) >= clip_level)
    else:
        rms = math.nan
        peak = math.nan
        longest_clip_samples = 0
    max_clip_samples = int(math.floor(sample_rate * 0.005))
    checks = {
        "finite": finite,
        "nonempty": bool(samples.size),
        "nonsilent": bool(finite and duration >= 0.1 and rms > 10 ** (-50 / 20)),
        "nonclipped": bool(finite and longest_clip_samples <= max_clip_samples),
        "plausible_duration": bool(0.15 <= duration <= 30.0),
    }
    return {
        "sample_count": int(samples.size),
        "sample_rate_hz": sample_rate,
        "duration_s": round(duration, 6),
        "rms": None if not math.isfinite(rms) else rms,
        "peak_abs": None if not math.isfinite(peak) else peak,
        "clip_level_abs": 10 ** (-0.1 / 20),
        "longest_clip_run_samples": longest_clip_samples,
        "max_allowed_clip_run_samples": max_clip_samples,
        "checks": checks,
        "pass": all(checks.values()),
        "evidence_class": "HOST",
    }


def synthesize(
    engine: Kokoro,
    text: str,
    lang_code: str,
    voice: str,
    out_path: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        samples, sample_rate = engine.create(
            text,
            voice=voice,
            speed=1.0,
            lang=lang_code,
            trim=True,
        )
        elapsed = time.perf_counter() - started
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        waveform = inspect_waveform(samples, sample_rate)
        if out_path is not None and waveform["checks"]["finite"] and waveform["checks"]["nonempty"]:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, samples, sample_rate, subtype="PCM_16")
            artifact = {
                "relative_path": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
            }
        else:
            artifact = None
        return {
            "status": "PASS" if waveform["pass"] else "FAIL_WAVEFORM_ASSERTION",
            "synthesis_s": elapsed,
            "rtf": elapsed / waveform["duration_s"] if waveform["duration_s"] else None,
            "waveform": waveform,
            "artifact": artifact,
        }
    except Exception as exc:  # named failure mode is part of D3's contract
        return {
            "status": "FAIL_EXCEPTION",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "synthesis_s": time.perf_counter() - started,
            "waveform": None,
            "artifact": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--d1-corpus", type=Path, required=True)
    parser.add_argument("--d5-prompts", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    d1 = json.loads(args.d1_corpus.read_text())
    d5 = json.loads(args.d5_prompts.read_text())
    load_started = time.perf_counter()
    engine = Kokoro(str(args.model), str(args.voices))
    engine._create_audio = types.MethodType(compatibility_create_audio, engine)
    load_s = time.perf_counter() - load_started

    result: dict[str, Any] = {
        "schema_version": 1,
        "engine": "Kokoro-82M-v1.0-ONNX q8f16",
        "coverage": ["en", "es", "fr"],
        "unsupported": ["ru"],
        "evidence_class": "HOST",
        "model": {
            "repo": "onnx-community/Kokoro-82M-v1.0-ONNX",
            "revision": "1939ad2a8e416c0acfeecc08a694d14ef25f2231",
            "file": "onnx/model_q8f16.onnx",
            "prebuilt_quantization": "q8f16",
            "bytes": args.model.stat().st_size,
            "sha256": sha256_file(args.model),
        },
        "voices": {
            "file": args.voices.name,
            "bytes": args.voices.stat().st_size,
            "sha256": sha256_file(args.voices),
        },
        "runtime": {
            "kokoro_onnx": importlib.metadata.version("kokoro-onnx"),
            "onnxruntime": ort.__version__,
            "providers": engine.sess.get_providers(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model_load_s": load_s,
            "compatibility_note": "Harness sends float32 speed because pinned graph declares tensor(float); graph and weights unchanged.",
        },
        "d1": {},
        "d3": {},
        "d5": {},
    }

    for lang, cfg in LANGUAGE_CONFIG.items():
        print(f"D3 {lang}", flush=True)
        safety_rows = []
        for case_id, text in SHORT_INPUTS[lang]:
            row = synthesize(engine, text, cfg["lang_code"], cfg["voice"], None)
            row.update({"case_id": case_id, "text": text})
            safety_rows.append(row)
            print(f"  {case_id}: {row['status']}", flush=True)
        result["d3"][lang] = safety_rows

        print(f"D1 {lang}", flush=True)
        d1_rows = []
        for item in d1["items"]:
            item_id = item["item_id"]
            text = item["texts"][lang]
            out = args.out_root / "audio" / "d1" / lang / "kokoro" / f"{item_id}.wav"
            row = synthesize(engine, text, cfg["lang_code"], cfg["voice"], out)
            row.update({"item_id": item_id, "sentence_id": item["sentence_id"], "text": text})
            d1_rows.append(row)
            print(f"  {item_id}: {row['status']}", flush=True)
        result["d1"][lang] = d1_rows

        print(f"D5 {lang}", flush=True)
        d5_rows = []
        for prompt in d5["languages"][lang]:
            index = int(prompt["listen_index"])
            out = args.out_root / "listening" / f"{lang}_kokoro_{index:02d}.wav"
            row = synthesize(engine, prompt["text"], cfg["lang_code"], cfg["voice"], out)
            row.update(
                {
                    "listen_index": index,
                    "case_id": prompt["case_id"],
                    "category": prompt["primary_category"],
                    "text": prompt["text"],
                }
            )
            d5_rows.append(row)
            print(f"  {index:02d}: {row['status']}", flush=True)
        result["d5"][lang] = d5_rows

    result_path = args.out_root / "kokoro" / "kokoro_v1_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
