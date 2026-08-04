#!/usr/bin/env python3
"""Profile ASR -> streamed text LLM -> serialized chunked Supertonic.

This is an architecture experiment rather than an Android result.  It measures
the exact hand-off points needed for a multilingual fallback: finish ASR, start
a text model, flush the first speakable clause, start its synthesis without
waiting for the rest of the answer, and serialize every later chunk.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
from threading import BoundedSemaphore
import time
from typing import Any
import urllib.request

import numpy as np
import sherpa_onnx
import soundfile as sf

from streaming_text_chunks import ChunkPolicy, IncrementalTextChunker


DEFAULT_SYSTEM_PROMPT = (
    "You are a low-latency voice assistant. Answer in the user's language. "
    "Your first words must be the answer itself: never acknowledge, preface, "
    "explain, or restate the request. Interpret 'Can you help me X?' as an "
    "instruction to do X instead of answering yes. For a slogan request, "
    "return exactly one quoted slogan and no other text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Complete mono WAV after VAD closes")
    parser.add_argument(
        "--asr-backend",
        choices=("lfm-http", "faster-whisper"),
        default="lfm-http",
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--whisper-compute-type", default="int8")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8082/v1/chat/completions")
    parser.add_argument("--llm-model", default="Qwen3-0.6B-Q4_0")
    parser.add_argument("--tts-model-dir", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--speaker", type=int, default=6)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--flush-chars", type=int, default=64)
    parser.add_argument(
        "--max-queued-tts-chunks",
        type=int,
        default=4,
        help="Bound running plus waiting standalone-TTS chunks; applies backpressure to the SSE reader.",
    )
    parser.add_argument("--playback-cushion-ms", type=float, default=140.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups cannot be negative")
    if (
        args.threads < 1
        or args.steps < 1
        or args.flush_chars < 1
        or args.max_queued_tts_chunks < 1
    ):
        parser.error("thread, step, flush, and queue values must be positive")
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


def sse_events(url: str, payload: dict[str, Any], timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            yield json.loads(data)


def visible_response(response: str) -> str:
    if "<think>" in response and "</think>" not in response:
        return ""
    return re.sub(r"<think>[\s\S]*?</think>", "", response).strip()


def transcribe(args: argparse.Namespace, local_asr=None) -> dict[str, Any]:
    if args.asr_backend == "faster-whisper":
        if local_asr is None:
            raise RuntimeError("Faster-Whisper backend was selected but not initialized")
        started = time.perf_counter()
        segments, info = local_asr.transcribe(
            str(args.audio),
            language=None if args.language == "auto" else args.language,
            task="transcribe",
            beam_size=1,
            vad_filter=False,
        )
        parts: list[str] = []
        first_text_at: float | None = None
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if first_text_at is None:
                first_text_at = time.perf_counter()
            parts.append(text)
        ended = time.perf_counter()
        return {
            "transcript": " ".join(parts).strip(),
            "first_text_ms": None if first_text_at is None else (first_text_at - started) * 1000,
            "complete_ms": (ended - started) * 1000,
            "detected_language": info.language,
            "language_probability": float(info.language_probability),
        }

    payload = {
        "model": "LFM2.5-Audio-1.5B",
        "stream": True,
        "modalities": ["text"],
        "temperature": 0,
        "max_tokens": 192,
        "messages": [
            {"role": "system", "content": "Perform ASR."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(args.audio.read_bytes()).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
    }
    started = time.perf_counter()
    first_text_at: float | None = None
    parts: list[str] = []
    for event in sse_events(args.asr_url, payload, args.timeout):
        content = event.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            if first_text_at is None:
                first_text_at = time.perf_counter()
            parts.append(content)
    ended = time.perf_counter()
    return {
        "transcript": "".join(parts).strip(),
        "first_text_ms": None if first_text_at is None else (first_text_at - started) * 1000,
        "complete_ms": (ended - started) * 1000,
        "detected_language": None,
        "language_probability": None,
    }


def create_tts(args: argparse.Namespace):
    root = args.tts_model_dir.expanduser().resolve()
    model = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=str(root / "duration_predictor.int8.onnx"),
        text_encoder=str(root / "text_encoder.int8.onnx"),
        vector_estimator=str(root / "vector_estimator.int8.onnx"),
        vocoder=str(root / "vocoder.int8.onnx"),
        tts_json=str(root / "tts.json"),
        unicode_indexer=str(root / "unicode_indexer.bin"),
        voice_style=str(root / "voice.bin"),
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=model,
            num_threads=args.threads,
            provider="cpu",
            debug=False,
        )
    )
    if not config.validate():
        raise ValueError("Invalid Supertonic configuration")
    return sherpa_onnx.OfflineTts(config)


def synthesize(
    tts,
    args: argparse.Namespace,
    text: str,
    capture_audio: bool,
) -> dict[str, Any]:
    config = sherpa_onnx.GenerationConfig()
    config.sid = args.speaker
    config.num_steps = args.steps
    config.speed = args.speed
    config.extra = {"lang": args.language, "seed": "0", "max_len": str(args.flush_chars)}
    started = time.perf_counter()
    first_at: float | None = None
    first_chunk: np.ndarray | None = None

    def callback(samples: np.ndarray, _progress: float) -> int:
        nonlocal first_at, first_chunk
        if first_at is None:
            first_at = time.perf_counter()
            if capture_audio:
                first_chunk = np.asarray(samples, dtype=np.float32).copy()
        return 1

    audio = tts.generate(text, config, callback)
    ended = time.perf_counter()
    samples = np.asarray(audio.samples, dtype=np.float32)
    return {
        "started_at": started,
        "first_at": first_at,
        "ended_at": ended,
        "first_callback_ms": None if first_at is None else (first_at - started) * 1000,
        "complete_ms": (ended - started) * 1000,
        "samples": samples if capture_audio else None,
        "first_chunk": first_chunk,
        "sample_rate": int(audio.sample_rate),
        "audio_seconds": float(samples.size / audio.sample_rate),
    }


def run_once(
    args: argparse.Namespace,
    tts,
    executor: ThreadPoolExecutor,
    local_asr,
    run_index: int,
    measured: bool,
) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    asr = transcribe(args, local_asr)
    asr_complete_at = time.perf_counter()
    llm_started = asr_complete_at
    payload = {
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
            {
                "role": "user",
                "content": asr["transcript"]
                + (" /no_think" if "qwen" in args.llm_model.casefold() else ""),
            },
        ],
    }
    llm_parts: list[str] = []
    first_token_at: float | None = None
    first_visible_at: float | None = None
    chunk_ready_at: float | None = None
    first_chunk_text: str | None = None
    tts_jobs: list[dict[str, Any]] = []
    tts_slots = BoundedSemaphore(args.max_queued_tts_chunks)
    capture_audio = bool(measured and run_index == args.warmups and args.sample_dir)
    tts_backpressure_seconds = 0.0
    max_tts_backpressure_seconds = 0.0
    visible_queued = ""
    chunker = IncrementalTextChunker(
        ChunkPolicy(
            min_chars=min(8, args.flush_chars),
            clause_chars=min(32, args.flush_chars),
            max_chars=args.flush_chars,
            boundary_policy="bounded",
        )
    )

    def enqueue_tts(text: str, ready_at: float) -> None:
        nonlocal chunk_ready_at, first_chunk_text
        nonlocal tts_backpressure_seconds, max_tts_backpressure_seconds
        if not tts_jobs:
            chunk_ready_at = ready_at
            first_chunk_text = text
        # A slow standalone TTS must not create an unbounded text-job backlog.
        # Blocking this SSE reader provides deliberate backpressure after the
        # configured number of running/waiting chunks.
        wait_started = time.perf_counter()
        tts_slots.acquire()
        waited = time.perf_counter() - wait_started
        tts_backpressure_seconds += waited
        max_tts_backpressure_seconds = max(max_tts_backpressure_seconds, waited)
        try:
            future = executor.submit(synthesize, tts, args, text, capture_audio)
        except BaseException:
            tts_slots.release()
            raise
        future.add_done_callback(lambda _future: tts_slots.release())
        tts_jobs.append(
            {
                "text": text,
                "ready_at": ready_at,
                # The executor has one worker: chunks synthesize in response
                # order, while the first job overlaps continued LLM decoding.
                "future": future,
            }
        )

    for event in sse_events(args.llm_url, payload, args.timeout):
        content = event.get("choices", [{}])[0].get("delta", {}).get("content")
        if not content:
            continue
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        llm_parts.append(content)
        visible = visible_response("".join(llm_parts))
        if visible and first_visible_at is None:
            first_visible_at = now
        if not visible.startswith(visible_queued):
            raise RuntimeError("Visible LLM text changed after it was queued for speech")
        new_visible = visible[len(visible_queued):]
        visible_queued = visible
        for chunk in chunker.push(new_visible):
            enqueue_tts(chunk, now)
    llm_ended = time.perf_counter()
    visible_text = visible_response("".join(llm_parts))
    if not visible_text.startswith(visible_queued):
        raise RuntimeError("Final visible LLM text changed after it was queued for speech")
    final_visible = visible_text[len(visible_queued):]
    for chunk in chunker.push(final_visible):
        enqueue_tts(chunk, llm_ended)
    for chunk in chunker.finish():
        enqueue_tts(chunk, llm_ended)
    if not tts_jobs:
        raise RuntimeError("Text LLM produced no speakable response")

    tts_results: list[dict[str, Any]] = []
    deadline = time.monotonic() + args.timeout
    for job in tts_jobs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Standalone TTS queue did not finish before timeout")
        result = job["future"].result(timeout=remaining)
        result["text"] = job["text"]
        result["ready_at"] = job["ready_at"]
        tts_results.append(result)

    pipeline_ended = time.perf_counter()
    first_tts_result = tts_results[0]
    first_pcm_at = first_tts_result["first_at"]
    first_audible_at = (
        None
        if first_pcm_at is None
        else first_pcm_at + args.playback_cushion_ms / 1000.0
    )
    sample_rates = {int(result["sample_rate"]) for result in tts_results}
    if len(sample_rates) != 1:
        raise RuntimeError(f"Standalone TTS changed sample rate across chunks: {sample_rates}")
    sample_rate = sample_rates.pop()
    captured_samples = [result["samples"] for result in tts_results if result["samples"] is not None]
    all_samples = np.concatenate(captured_samples) if captured_samples else None

    sample_path = None
    first_chunk_path = None
    if capture_audio:
        args.sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = args.sample_dir / "cascade_response.wav"
        if all_samples is None:
            raise RuntimeError("Sample capture was requested but standalone TTS returned no samples")
        sf.write(sample_path, all_samples, sample_rate, subtype="PCM_16")
        if first_tts_result["first_chunk"] is not None:
            first_chunk_path = args.sample_dir / "cascade_first_chunk.wav"
            sf.write(
                first_chunk_path,
                first_tts_result["first_chunk"],
                sample_rate,
                subtype="PCM_16",
            )

    return {
        "run_index": run_index,
        "measured": measured,
        "transcript": asr["transcript"],
        "asr_detected_language": asr["detected_language"],
        "asr_language_probability": asr["language_probability"],
        "response": visible_text,
        "first_tts_chunk": first_chunk_text,
        "tts_chunks": [
            {
                "text": result["text"],
                "ready_from_llm_start_ms": (result["ready_at"] - llm_started) * 1000,
                "start_from_pipeline_ms": (result["started_at"] - pipeline_started) * 1000,
                "first_callback_ms": result["first_callback_ms"],
                "complete_ms": result["complete_ms"],
                "audio_seconds": result["audio_seconds"],
            }
            for result in tts_results
        ],
        "timing_ms": {
            "asr_first_text": asr["first_text_ms"],
            "asr_complete": asr["complete_ms"],
            "llm_first_token_from_llm_start": (
                None if first_token_at is None else (first_token_at - llm_started) * 1000
            ),
            "llm_first_visible_from_llm_start": (
                None if first_visible_at is None else (first_visible_at - llm_started) * 1000
            ),
            "first_chunk_ready_from_llm_start": (
                None if chunk_ready_at is None else (chunk_ready_at - llm_started) * 1000
            ),
            "llm_complete_from_llm_start": (llm_ended - llm_started) * 1000,
            "tts_first_callback_from_tts_start": first_tts_result["first_callback_ms"],
            "tts_complete_from_tts_start": (
                tts_results[-1]["ended_at"] - first_tts_result["started_at"]
            )
            * 1000,
            "tts_queue_backpressure_total": tts_backpressure_seconds * 1000,
            "tts_queue_backpressure_max": max_tts_backpressure_seconds * 1000,
            "pipeline_first_pcm": (
                None if first_pcm_at is None else (first_pcm_at - pipeline_started) * 1000
            ),
            "pipeline_first_audible_scheduled": (
                None if first_audible_at is None else (first_audible_at - pipeline_started) * 1000
            ),
            "pipeline_complete": (pipeline_ended - pipeline_started) * 1000,
        },
        "tts_audio_seconds": sum(result["audio_seconds"] for result in tts_results),
        "tts_chunk_count": len(tts_results),
        "sample_rate": sample_rate,
        "sample_path": str(sample_path) if sample_path else None,
        "first_chunk_path": str(first_chunk_path) if first_chunk_path else None,
    }


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    local_asr = None
    asr_load_ms = 0.0
    if args.asr_backend == "faster-whisper":
        from faster_whisper import WhisperModel

        asr_load_started = time.perf_counter()
        local_asr = WhisperModel(
            args.whisper_model,
            device="cpu",
            compute_type=args.whisper_compute_type,
        )
        asr_load_ms = (time.perf_counter() - asr_load_started) * 1000

    tts_load_started = time.perf_counter()
    tts = create_tts(args)
    tts_load_ms = (time.perf_counter() - tts_load_started) * 1000
    warm_config = sherpa_onnx.GenerationConfig()
    warm_config.sid = args.speaker
    warm_config.num_steps = args.steps
    warm_config.speed = args.speed
    warm_config.extra = {"lang": args.language, "seed": "0"}
    tts.generate("Hello.", warm_config)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for index in range(args.warmups + args.runs):
            measured = index >= args.warmups
            row = run_once(args, tts, executor, local_asr, index, measured)
            rows.append(row)
            timing = row["timing_ms"]
            phase = "measure" if measured else "warmup"
            print(
                f"{phase} {index + 1}/{args.warmups + args.runs}: "
                f"asr={timing['asr_complete']:.1f} ms "
                f"chunk={timing['first_chunk_ready_from_llm_start']:.1f} ms "
                f"first_pcm={timing['pipeline_first_pcm']:.1f} ms "
                f"audible={timing['pipeline_first_audible_scheduled']:.1f} ms",
                flush=True,
            )

    measured_rows = [row for row in rows if row["measured"]]
    metric_names = tuple(measured_rows[0]["timing_ms"])
    summary = {
        name: describe(
            [
                float(row["timing_ms"][name])
                for row in measured_rows
                if row["timing_ms"][name] is not None
            ]
        )
        for name in metric_names
    }
    document = {
        "schema_version": 2,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_class": "Apple Silicon architecture experiment; not phone/NPU/glasses",
        "settings": {
            "audio": args.audio.name,
            "asr_backend": args.asr_backend,
            "asr_url": args.asr_url,
            "whisper_model": args.whisper_model if local_asr is not None else None,
            "whisper_compute_type": args.whisper_compute_type if local_asr is not None else None,
            "asr_load_ms": asr_load_ms,
            "llm_url": args.llm_url,
            "llm_model": args.llm_model,
            "system_prompt": args.system_prompt,
            "tts_model": args.tts_model_dir.name,
            "language": args.language,
            "threads": args.threads,
            "steps": args.steps,
            "speaker": args.speaker,
            "speed": args.speed,
            "flush_chars": args.flush_chars,
            "tts_chunk_policy": {
                "boundary_policy": "bounded",
                "min_chars": min(8, args.flush_chars),
                "clause_chars": min(32, args.flush_chars),
                "max_chars": args.flush_chars,
                "ascii_sentence_lookahead": True,
            },
            "max_queued_tts_chunks": args.max_queued_tts_chunks,
            "playback_cushion_ms": args.playback_cushion_ms,
            "tts_load_ms": tts_load_ms,
            "warmups": args.warmups,
            "runs": args.runs,
        },
        "summary": summary,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
