#!/usr/bin/env python3
"""Profile audio -> ASR -> streaming text LLM -> standalone TTS first PCM.

The ASR and LLM endpoints are OpenAI-compatible HTTP servers. Supertonic 3 is
kept warm in-process and synthesis begins on the first safe text boundary while
the LLM response continues streaming.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="16 kHz mono WAV input")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8082/v1/chat/completions")
    parser.add_argument("--llm-model", default="Qwen3-0.6B-Q4_0")
    parser.add_argument(
        "--system-prompt",
        default=(
            "Answer accurately and concisely in the same language as the user. "
            "Start with the answer. Do not show reasoning."
        ),
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Supertonic 3 model directory")
    parser.add_argument("--language", default="en")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--speaker", type=int, default=6)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=64)
    parser.add_argument(
        "--boundary-policy",
        choices=("sentence", "clause", "bounded"),
        default="bounded",
        help="When to launch TTS: terminal sentence, clause punctuation, or a hard character bound",
    )
    parser.add_argument("--playback-cushion-ms", type=float, default=140.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-wav", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups cannot be negative")
    if args.min_chars < 1 or args.max_chars < args.min_chars:
        parser.error("require 1 <= --min-chars <= --max-chars")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def post_stream(url: str, body: dict[str, Any], timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def iter_sse(response):
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        yield json.loads(data)


def visible_response(text: str) -> str:
    if "<think>" in text and "</think>" not in text:
        return ""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def first_speakable_prefix(
    text: str, min_chars: int, max_chars: int, boundary_policy: str
) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_chars:
        return None

    for match in re.finditer(r"[.!?。！？](?:[\"'”’）)]|\s|$)", text):
        end = match.end()
        if end >= min_chars:
            return text[:end].strip()

    if boundary_policy == "sentence":
        return None

    soft = None
    for match in re.finditer(r"[,;:，、；：](?:\s|$)", text):
        if match.end() >= max(min_chars, max_chars // 2):
            soft = match.end()
            break
    if soft is not None:
        return text[:soft].strip()

    if boundary_policy == "clause":
        return None

    if len(text) < max_chars:
        return None
    boundary = text.rfind(" ", min_chars, max_chars + 1)
    if boundary < min_chars:
        boundary = max_chars
    return text[:boundary].strip()


def create_tts(model_dir: Path, threads: int):
    supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=str(model_dir / "duration_predictor.int8.onnx"),
        text_encoder=str(model_dir / "text_encoder.int8.onnx"),
        vector_estimator=str(model_dir / "vector_estimator.int8.onnx"),
        vocoder=str(model_dir / "vocoder.int8.onnx"),
        tts_json=str(model_dir / "tts.json"),
        unicode_indexer=str(model_dir / "unicode_indexer.bin"),
        voice_style=str(model_dir / "voice.bin"),
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=supertonic,
            num_threads=threads,
            provider="cpu",
            debug=False,
        )
    )
    if not config.validate():
        raise ValueError("Invalid Supertonic model configuration")
    return sherpa_onnx.OfflineTts(config)


def asr_request(audio: Path, url: str, timeout: float, started: float) -> dict[str, Any]:
    body = {
        "model": "LFM2.5-Audio-1.5B",
        "stream": True,
        "modalities": ["text"],
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "Perform ASR."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
    }
    chunks: list[str] = []
    first_at = None
    with post_stream(url, body, timeout) as response:
        for event in iter_sse(response):
            content = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                if first_at is None:
                    first_at = time.perf_counter()
                chunks.append(content)
    ended = time.perf_counter()
    return {
        "text": "".join(chunks).strip(),
        "first_text_ms": None if first_at is None else (first_at - started) * 1000,
        "complete_ms": (ended - started) * 1000,
    }


def profile_once(args: argparse.Namespace, tts, run_index: int, measured: bool) -> dict[str, Any]:
    started = time.perf_counter()
    asr = asr_request(args.audio, args.asr_url, args.timeout, started)

    body = {
        "model": args.llm_model,
        "stream": True,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": args.system_prompt,
            },
            {"role": "user", "content": asr["text"] + " /no_think"},
        ],
    }

    llm_parts: list[str] = []
    first_llm_token_at = None
    first_visible_at = None
    chunk_ready_at = None
    tts_started_at = None
    first_pcm_at = None
    tts_complete_at = None
    first_chunk_text = None
    first_chunk_samples: np.ndarray | None = None
    first_chunk_rate = None
    generated_audio_seconds = None
    worker_error: BaseException | None = None
    worker: threading.Thread | None = None
    worker_lock = threading.Lock()

    def launch_tts(text: str) -> None:
        nonlocal worker, first_chunk_text, tts_started_at
        nonlocal first_pcm_at, tts_complete_at, first_chunk_samples
        nonlocal first_chunk_rate, generated_audio_seconds, worker_error
        if worker is not None:
            return
        first_chunk_text = text
        tts_started_at = time.perf_counter()

        def run() -> None:
            nonlocal first_pcm_at, tts_complete_at, first_chunk_samples
            nonlocal first_chunk_rate, generated_audio_seconds, worker_error
            try:
                config = sherpa_onnx.GenerationConfig()
                config.sid = args.speaker
                config.num_steps = args.steps
                config.speed = args.speed
                config.extra = {"lang": args.language, "seed": "0", "max_len": str(args.max_chars)}

                def callback(samples: np.ndarray, _progress: float) -> int:
                    nonlocal first_pcm_at, first_chunk_samples
                    with worker_lock:
                        if first_pcm_at is None:
                            first_pcm_at = time.perf_counter()
                            first_chunk_samples = np.asarray(samples, dtype=np.float32).copy()
                    return 1

                audio = tts.generate(text, config, callback)
                generated_audio_seconds = len(audio.samples) / audio.sample_rate
                first_chunk_rate = int(audio.sample_rate)
                tts_complete_at = time.perf_counter()
            except BaseException as error:  # Preserve remote-style failure evidence in the JSON.
                worker_error = error
                tts_complete_at = time.perf_counter()

        worker = threading.Thread(target=run, name="first-tts-chunk", daemon=True)
        worker.start()

    with post_stream(args.llm_url, body, args.timeout) as response:
        for event in iter_sse(response):
            content = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if not content:
                continue
            now = time.perf_counter()
            if first_llm_token_at is None:
                first_llm_token_at = now
            llm_parts.append(content)
            visible = visible_response("".join(llm_parts))
            if visible and first_visible_at is None:
                first_visible_at = now
            if worker is None:
                prefix = first_speakable_prefix(
                    visible, args.min_chars, args.max_chars, args.boundary_policy
                )
                if prefix:
                    chunk_ready_at = now
                    launch_tts(prefix)
    llm_complete_at = time.perf_counter()
    visible = visible_response("".join(llm_parts))
    if worker is None and visible:
        chunk_ready_at = llm_complete_at
        launch_tts(visible)
    if worker is not None:
        worker.join(args.timeout)
        if worker.is_alive():
            raise TimeoutError("TTS worker did not finish before timeout")
    if worker_error is not None:
        raise RuntimeError(f"TTS failed: {type(worker_error).__name__}: {worker_error}")
    ended = time.perf_counter()

    if args.sample_wav and measured and run_index == args.warmups and first_chunk_samples is not None:
        args.sample_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.sample_wav, first_chunk_samples, first_chunk_rate, subtype="PCM_16")

    def elapsed(moment: float | None) -> float | None:
        return None if moment is None else (moment - started) * 1000

    first_pcm_ms = elapsed(first_pcm_at)
    return {
        "run_index": run_index,
        "measured": measured,
        "audio": args.audio.name,
        "language": args.language,
        "asr_transcript": asr["text"],
        "asr_first_text_ms": asr["first_text_ms"],
        "asr_complete_ms": asr["complete_ms"],
        "llm_first_token_ms": elapsed(first_llm_token_at),
        "llm_first_visible_token_ms": elapsed(first_visible_at),
        "first_tts_chunk_ready_ms": elapsed(chunk_ready_at),
        "tts_started_ms": elapsed(tts_started_at),
        "first_pcm_ms": first_pcm_ms,
        "first_playback_scheduled_ms": (
            None if first_pcm_ms is None else first_pcm_ms + args.playback_cushion_ms
        ),
        "llm_complete_ms": elapsed(llm_complete_at),
        "tts_complete_ms": elapsed(tts_complete_at),
        "pipeline_complete_ms": (ended - started) * 1000,
        "llm_response": visible,
        "first_tts_chunk": first_chunk_text,
        "first_chunk_sample_rate": first_chunk_rate,
        "first_chunk_samples": None if first_chunk_samples is None else int(first_chunk_samples.size),
        "first_chunk_audio_seconds": (
            None
            if first_chunk_samples is None or not first_chunk_rate
            else float(first_chunk_samples.size / first_chunk_rate)
        ),
        "generated_audio_seconds": generated_audio_seconds,
    }


def main() -> int:
    args = parse_args()
    load_started = time.perf_counter()
    tts = create_tts(args.model_dir, args.threads)
    load_ms = (time.perf_counter() - load_started) * 1000

    runs = []
    for index in range(args.warmups + args.runs):
        measured = index >= args.warmups
        result = profile_once(args, tts, index, measured)
        runs.append(result)
        phase = "measure" if measured else "warmup"
        print(
            f"{phase} {index + 1}/{args.warmups + args.runs}: "
            f"asr={result['asr_complete_ms']:.1f} ms "
            f"chunk={result['first_tts_chunk_ready_ms']!s} ms "
            f"pcm={result['first_pcm_ms']!s} ms "
            f"play={result['first_playback_scheduled_ms']!s} ms",
            flush=True,
        )

    measured = [run for run in runs if run["measured"]]
    timing_keys = (
        "asr_first_text_ms",
        "asr_complete_ms",
        "llm_first_token_ms",
        "llm_first_visible_token_ms",
        "first_tts_chunk_ready_ms",
        "first_pcm_ms",
        "first_playback_scheduled_ms",
        "llm_complete_ms",
        "tts_complete_ms",
        "pipeline_complete_ms",
    )
    timing = {}
    for key in timing_keys:
        values = [float(run[key]) for run in measured if run[key] is not None]
        if values:
            timing[key] = describe(values)

    output = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_class": (
            "Steady-state Apple Silicon pipeline timing. It is not phone, NPU, AR1, or glasses evidence."
        ),
        "configuration": {
            "audio": str(args.audio),
            "asr_url": args.asr_url,
            "llm_url": args.llm_url,
            "llm_model": args.llm_model,
            "system_prompt": args.system_prompt,
            "tts_model_dir": str(args.model_dir),
            "language": args.language,
            "threads": args.threads,
            "steps": args.steps,
            "speed": args.speed,
            "speaker": args.speaker,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "boundary_policy": args.boundary_policy,
            "playback_cushion_ms": args.playback_cushion_ms,
            "warmups": args.warmups,
            "runs": args.runs,
            "tts_load_ms": load_ms,
        },
        "summary": {"measured_runs": len(measured), "timing": timing},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
