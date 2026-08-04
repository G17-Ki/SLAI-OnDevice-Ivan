#!/usr/bin/env python3
"""Lane F F3 — ES/FR Supertonic route verification and fixture generation.

Mirrors the Android synthesis path exactly:

  android-app/app/src/main/java/ai/liquid/lfmdemo/tts/SherpaTtsEngine.kt:171-179
      OfflineTtsSupertonicModelConfig(duration_predictor, text_encoder,
      vector_estimator, vocoder, tts_json, unicode_indexer, voice_style)
  :139-143  provider = "cpu"
  :253-261  GenerationConfig(silence_scale=1.0, speed, sid, num_steps, extra)
  :263-292  every generated chunk passes PcmGate before AudioTrack

and applies two waveform gates to what comes back:

  * PCM_GATE  — a faithful port of tts/PcmGate.kt:20-89, applied per chunk
                exactly where the app applies it, and to the joined waveform.
  * D3_GATE   — Lane D's exact short-input gate (out/d/D3.json "gate"):
                finite, non-empty, RMS > -50 dBFS, duration in [0.15, 30] s,
                no contiguous run above -0.1 dBFS longer than 5 ms.

An utterance passes only if every chunk passes PCM_GATE (the app drops failing
chunks, and drops them all -> "all_chunks_rejected_invalid_pcm") and the joined
waveform passes both gates.

HOST measurement on the machine that runs it. Not phone performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

# --- tts/PcmGate.kt:22-26 ---------------------------------------------------
CLIP_LEVEL = 0.999
MIN_RMS = 0.005
WARN_ABS_DC = 0.01
MAX_ABS_DC = 0.10
MAX_CLIP_FRACTION = 0.001

# --- Lane D D3 gate (out/d/D3.json "gate") ----------------------------------
D3_MIN_RMS_LINEAR = 0.0031622776601683794      # -50 dBFS, exclusive
D3_DURATION_RANGE = (0.15, 30.0)               # inclusive
D3_CLIP_LINEAR = 0.9885530946569389            # -0.1 dBFS
D3_MAX_SUSTAINED_CLIP_MS = 5.0

# --- Lane D short-input cases, verbatim from out/d/run_kokoro_v1.py:31-51 ----
# plus the one text that broke Piper DaveFX ES in out/d/D3.json named_failures.
SHORT_INPUTS = {
    "es": [
        ("char_1", "a"),
        ("char_2", "sí"),
        ("char_3", "sol"),
        ("word_1", "sí"),
        ("word_2", "sí claro"),
        ("word_3", "sí claro ahora"),
        ("d3_named_failure", "A"),
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

# Demo phrases in the shape of scripts/tts_research_20260726/stage_device_bundle.py
# PHRASES (:221-253): micro / clause / answer / weather / question.
DEMO_PHRASES = {
    "es": [
        ("micro", "Claro."),
        ("clause", "Madrid es la capital de España."),
        ("answer", "Madrid es la capital de España y su ciudad más grande."),
        ("weather", "Esta tarde hará sol, unos veinticuatro grados; no hace falta paraguas."),
        ("question", "¿Quieres que traduzca este texto al inglés?"),
    ],
    "fr": [
        ("micro", "Bien sûr."),
        ("clause", "Paris est la capitale de la France."),
        ("answer", "Paris est la capitale de la France et sa plus grande ville."),
        ("weather", "Cet après-midi il fera beau, environ vingt-quatre degrés; inutile de prendre un parapluie."),
        ("question", "Veux-tu que je traduise ce texte en anglais ?"),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_gate(samples: np.ndarray) -> dict:
    """Faithful port of tts/PcmGate.kt:44-89."""
    if samples.size == 0:
        return {"passed": False, "failures": ["empty"], "warnings": [],
                "samples": 0, "non_finite": 0, "peak": 0.0, "rms": 0.0,
                "dc": 0.0, "clip_count": 0}
    finite_mask = np.isfinite(samples)
    non_finite = int(samples.size - finite_mask.sum())
    finite = samples[finite_mask].astype(np.float64)
    if finite.size == 0:
        return {"passed": False, "failures": ["all_samples_non_finite"],
                "warnings": [], "samples": int(samples.size),
                "non_finite": non_finite, "peak": 0.0, "rms": 0.0, "dc": 0.0,
                "clip_count": 0}
    magnitude = np.abs(finite)
    peak = float(magnitude.max())
    rms = float(np.sqrt(np.mean(finite * finite)))
    dc = float(np.mean(finite))
    clip_count = int((magnitude >= CLIP_LEVEL).sum())
    clip_fraction = clip_count / finite.size

    failures, warnings = [], []
    if non_finite > 0:
        failures.append(f"non_finite_pcm:{non_finite}")
    if rms < MIN_RMS:
        failures.append(f"near_silent_rms:{rms:.6f}")
    if clip_fraction > MAX_CLIP_FRACTION:
        failures.append(f"clipping:{clip_fraction:.6f}")
    if abs(dc) > MAX_ABS_DC:
        failures.append(f"severe_dc_offset:{dc:.6f}")
    elif abs(dc) > WARN_ABS_DC:
        warnings.append(f"dc_offset_needs_blocker:{dc:.6f}")
    if peak > 0.99:
        warnings.append(f"peak_near_full_scale:{peak:.4f}")
    return {"passed": not failures, "failures": failures, "warnings": warnings,
            "samples": int(samples.size), "non_finite": non_finite,
            "peak": peak, "rms": rms, "dc": dc, "clip_count": clip_count}


def d3_gate(samples: np.ndarray, rate: int) -> dict:
    failures = []
    if samples.size == 0:
        return {"passed": False, "failures": ["empty"], "duration_seconds": 0.0,
                "rms": 0.0, "peak": 0.0, "max_sustained_clip_ms": 0.0}
    if not np.isfinite(samples).all():
        failures.append("non_finite")
    finite = samples[np.isfinite(samples)].astype(np.float64)
    duration = samples.size / rate
    rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0
    peak = float(np.abs(finite).max()) if finite.size else 0.0
    if not (rms > D3_MIN_RMS_LINEAR):
        failures.append(f"rms_at_or_below_-50dBFS:{rms:.6f}")
    if not (D3_DURATION_RANGE[0] <= duration <= D3_DURATION_RANGE[1]):
        failures.append(f"duration_out_of_range:{duration:.6f}")

    over = np.abs(samples) > D3_CLIP_LINEAR
    longest = 0
    current = 0
    for value in over:
        current = current + 1 if value else 0
        longest = max(longest, current)
    sustained_ms = longest * 1000.0 / rate
    if sustained_ms > D3_MAX_SUSTAINED_CLIP_MS:
        failures.append(f"sustained_clipping_ms:{sustained_ms:.3f}")
    return {"passed": not failures, "failures": failures,
            "duration_seconds": duration, "rms": rms, "peak": peak,
            "max_sustained_clip_ms": sustained_ms}


def write_canonical_16k_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Downmix/resample to the AsrHttpServer canonical contract and write it."""
    import soxr
    if rate != 16000:
        samples = soxr.resample(samples.astype(np.float32), rate, 16000,
                                quality="VHQ")
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    path.write_bytes(header + pcm)


def build_tts(model_dir: Path, threads: int) -> sherpa_onnx.OfflineTts:
    model = sherpa_onnx.OfflineTtsModelConfig(
        supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=str(model_dir / "duration_predictor.int8.onnx"),
            text_encoder=str(model_dir / "text_encoder.int8.onnx"),
            vector_estimator=str(model_dir / "vector_estimator.int8.onnx"),
            vocoder=str(model_dir / "vocoder.int8.onnx"),
            tts_json=str(model_dir / "tts.json"),
            unicode_indexer=str(model_dir / "unicode_indexer.bin"),
            voice_style=str(model_dir / "voice.bin"),
        ),
        num_threads=threads,
        provider="cpu",
        debug=False,
    )
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model))


def synthesize(tts, text: str, language: str, steps: int, sid: int,
               speed: float) -> dict:
    """One utterance, gating each chunk exactly where SherpaTtsEngine does."""
    config = sherpa_onnx.GenerationConfig()
    config.silence_scale = 1.0
    config.speed = speed
    config.sid = sid
    config.num_steps = steps
    config.extra = {"lang": language, "seed": "0"}

    chunks: list[dict] = []
    kept: list[np.ndarray] = []
    started = time.perf_counter()
    first_chunk_ms = [-1.0]

    def on_chunk(samples, _progress) -> int:
        block = np.asarray(samples, dtype=np.float32)
        if first_chunk_ms[0] < 0:
            first_chunk_ms[0] = (time.perf_counter() - started) * 1000.0
        gate = pcm_gate(block)
        chunks.append(gate)
        if gate["passed"]:
            kept.append(block.copy())
        return 1

    audio = tts.generate(text, config, on_chunk)
    elapsed = time.perf_counter() - started
    rate = int(audio.sample_rate)
    full = np.asarray(audio.samples, dtype=np.float32)

    dropped = sum(1 for c in chunks if not c["passed"])
    joined = np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)
    whole_pcm = pcm_gate(full)
    whole_d3 = d3_gate(full, rate)
    app_error = None
    if not chunks:
        app_error = "no_chunks_emitted"
    elif dropped == len(chunks):
        app_error = "all_chunks_rejected_invalid_pcm"

    return {
        "sample_rate_hz": rate,
        "sample_count": int(full.size),
        "audio_seconds": float(full.size) / rate if rate else 0.0,
        "synthesis_seconds": elapsed,
        "real_time_factor": (elapsed / (full.size / rate)) if full.size else -1.0,
        "first_chunk_ms": first_chunk_ms[0],
        "chunk_count": len(chunks),
        "chunks_dropped_by_pcm_gate": dropped,
        "chunk_failures": sorted({f for c in chunks for f in c["failures"]}),
        "kept_sample_count": int(joined.size),
        "app_reported_error": app_error,
        "pcm_gate": whole_pcm,
        "d3_gate": whole_d3,
        "passed": (app_error is None and dropped == 0
                   and whole_pcm["passed"] and whole_d3["passed"]),
        "_samples": full,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--languages", default="es,fr")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sid", type=int, default=6)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    languages = [x.strip() for x in arguments.languages.split(",") if x.strip()]
    arguments.fixture_dir.mkdir(parents=True, exist_ok=True)

    model_dir: Path = arguments.model_dir
    model_files = {
        name: {"bytes": (model_dir / name).stat().st_size,
               "sha256": sha256_file(model_dir / name)}
        for name in ("duration_predictor.int8.onnx", "text_encoder.int8.onnx",
                     "vector_estimator.int8.onnx", "vocoder.int8.onnx",
                     "tts.json", "unicode_indexer.bin", "voice.bin")
    }

    init_started = time.perf_counter()
    tts = build_tts(model_dir, arguments.threads)
    init_seconds = time.perf_counter() - init_started

    cases, fixtures = [], []
    attempted = passed = 0
    for language in languages:
        for kind, table in (("short_input", SHORT_INPUTS),
                            ("demo_phrase", DEMO_PHRASES)):
            for case_id, text in table.get(language, []):
                runs = []
                for _ in range(max(1, arguments.repeats)):
                    runs.append(synthesize(tts, text, language,
                                           arguments.steps, arguments.sid,
                                           arguments.speed))
                attempted += len(runs)
                run_passed = sum(1 for r in runs if r["passed"])
                passed += run_passed
                cases.append({
                    "language": language,
                    "kind": kind,
                    "case_id": case_id,
                    "text": text,
                    "repeats": len(runs),
                    "passed": run_passed,
                    "failed": len(runs) - run_passed,
                    "verdict": "PASS" if run_passed == len(runs) else "FAIL",
                    "runs": [{k: v for k, v in r.items()
                              if not k.startswith("_")} for r in runs],
                })
                status = "PASS" if run_passed == len(runs) else "FAIL"
                print(f"{language}/{kind}/{case_id:<18} {status} "
                      f"{run_passed}/{len(runs)} "
                      f"dur={runs[0]['audio_seconds']:.3f}s "
                      f"rms={runs[0]['pcm_gate']['rms']:.4f} "
                      f"rtf={runs[0]['real_time_factor']:.3f} :: {text}")

                if kind == "demo_phrase" and runs[0]["passed"]:
                    fixture_id = f"tts_{language}_supertonic3_s{arguments.steps}_{case_id}"
                    destination = arguments.fixture_dir / f"{fixture_id}.wav"
                    write_canonical_16k_wav(destination, runs[0]["_samples"],
                                            runs[0]["sample_rate_hz"])
                    payload = destination.read_bytes()
                    fixtures.append({
                        "fixture_id": fixture_id,
                        "language": language,
                        "case_id": case_id,
                        "text": text,
                        "path": destination.name,
                        "bytes": len(payload),
                        "sha256": sha256_file(destination),
                        "native_sample_rate_hz": runs[0]["sample_rate_hz"],
                        "stored_sample_rate_hz": 16000,
                        "canonical_contract": "riff44_pcm16_mono_16000hz",
                        "fits_fixed_10s_window": len(payload) <= 320_044,
                    })

    report = {
        "schema_version": 1,
        "sprint_id": "F3",
        "purpose": "ES/FR Supertonic route verification against the app's own "
                   "PcmGate plus Lane D's exact D3 short-input gate",
        "evidence_class": "HOST",
        "not_a_measurement_of": [
            "Nubia NX809J / SM8850 phone latency or audio quality",
            "intelligibility, naturalness, or language correctness",
            "NPU execution (provider is cpu)",
        ],
        "runtime": {
            "host": f"{platform.platform()} {platform.machine()}",
            "python": platform.python_version(),
            "sherpa_onnx": sherpa_onnx.__version__,
            "provider": "cpu",
            "num_threads": arguments.threads,
            "engine_init_seconds": init_seconds,
        },
        "candidate_parameters": {
            "engine": "supertonic",
            "num_steps": arguments.steps,
            "sid": arguments.sid,
            "speed": arguments.speed,
            "extra": {"lang": "<per case>", "seed": "0"},
            "mirrors": "SherpaTtsEngine.kt:171-179 and :253-261",
        },
        "model": {"directory": model_dir.name, "files": model_files},
        "gates": {
            "pcm_gate": {
                "source": "android-app/.../tts/PcmGate.kt:20-89",
                "min_rms": MIN_RMS, "max_abs_dc": MAX_ABS_DC,
                "max_clip_fraction": MAX_CLIP_FRACTION,
                "clip_level": CLIP_LEVEL,
                "applied": "per generated chunk and to the whole waveform",
            },
            "d3_gate": {
                "source": "workstreams/sub2b_sprint/sprint/out/d/D3.json gate",
                "min_rms_linear_exclusive": D3_MIN_RMS_LINEAR,
                "duration_seconds_inclusive": list(D3_DURATION_RANGE),
                "sustained_clipping_linear": D3_CLIP_LINEAR,
                "max_sustained_clipping_ms": D3_MAX_SUSTAINED_CLIP_MS,
            },
        },
        "headline": {"attempted": attempted, "passed": passed,
                     "failed": attempted - passed},
        "cases": cases,
        "fixtures": fixtures,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\n{passed}/{attempted} synthesis runs passed both gates")
    print(f"wrote {arguments.out}")
    return 0 if passed == attempted else 1


if __name__ == "__main__":
    raise SystemExit(main())
