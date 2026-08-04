#!/usr/bin/env python3
"""Measure streamed Japanese LFM text/audio latency through the local HTTP server."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import statistics
import struct
import time
import unicodedata
import urllib.request
import wave
from pathlib import Path
from typing import Any


SYSTEM_PROMPTS = {
    "interleaved": "Respond with interleaved text and audio.",
    "asr": "Perform ASR in japanese.",
    "tts": "Perform TTS in japanese.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1/chat/completions")
    parser.add_argument("--mode", choices=sorted(SYSTEM_PROMPTS), default="interleaved")
    parser.add_argument("--model", default="LFM2.5-Audio-1.5B-JP")
    parser.add_argument(
        "--system-prompt",
        help="Override the mode's official prompt (useful for checkpoint controls)",
    )
    parser.add_argument("--input-audio", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--reference-text")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def normalize_for_cer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
    ).lower()


def character_error_rate(reference: str, prediction: str) -> float:
    expected = normalize_for_cer(reference)
    actual = normalize_for_cer(prediction)
    previous = list(range(len(actual) + 1))
    for row, expected_character in enumerate(expected, 1):
        current = [row]
        for column, actual_character in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_character != actual_character),
                )
            )
        previous = current
    return 100.0 * previous[-1] / max(1, len(expected))


def audio_content(path: Path) -> list[dict[str, Any]]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [{"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}]


def make_payload(args: argparse.Namespace) -> bytes:
    if args.mode in {"interleaved", "asr"}:
        if args.input_audio is None:
            raise ValueError(f"--input-audio is required for {args.mode}")
        user_content: str | list[dict[str, Any]] = audio_content(args.input_audio)
    else:
        if not args.text:
            raise ValueError("--text is required for tts")
        user_content = args.text

    modalities = {
        "interleaved": ["text", "audio"],
        "asr": ["text"],
        "tts": ["audio"],
    }[args.mode]
    payload = {
        "model": args.model,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "reset_context": True,
        "modalities": modalities,
        "messages": [
            {
                "role": "system",
                "content": args.system_prompt or SYSTEM_PROMPTS[args.mode],
            },
            {"role": "user", "content": user_content},
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def float32_to_pcm16(raw: bytes) -> bytes:
    if len(raw) % 4:
        raise ValueError(f"Float32 PCM chunk has invalid byte length: {len(raw)}")
    output = bytearray(len(raw) // 2)
    for index, (sample,) in enumerate(struct.iter_unpack("<f", raw)):
        value = max(-1.0, min(1.0, sample))
        integer = round(value * (32767 if value >= 0 else 32768))
        struct.pack_into("<h", output, index * 2, integer)
    return bytes(output)


def decode_audio_delta(delta: dict[str, Any]) -> tuple[bytes, int, str] | None:
    if isinstance(delta.get("audio"), dict):
        audio = delta["audio"]
        raw = base64.b64decode(audio["data"])
        return raw, int(audio.get("sample_rate", 24000)), str(audio.get("format", "pcm16"))
    if isinstance(delta.get("audio_chunk"), dict):
        audio = delta["audio_chunk"]
        raw = base64.b64decode(audio["data"])
        return float32_to_pcm16(raw), int(audio.get("sample_rate", 24000)), "float32"
    return None


def pcm_stats(pcm16: bytes) -> dict[str, float | int] | None:
    if not pcm16:
        return None
    count = len(pcm16) // 2
    sum_squares = 0.0
    peak = 0
    for (sample,) in struct.iter_unpack("<h", pcm16):
        peak = max(peak, abs(sample))
        sum_squares += float(sample) * sample
    return {
        "samples": count,
        "peak_normalized": peak / 32768.0,
        "rms_normalized": math.sqrt(sum_squares / count) / 32768.0,
    }


def write_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_once(args: argparse.Namespace, run_index: int, measured: bool) -> dict[str, Any]:
    request = urllib.request.Request(
        args.base_url,
        data=make_payload(args),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_text: float | None = None
    first_audio: float | None = None
    text_parts: list[str] = []
    pcm_parts: list[bytes] = []
    sample_rate: int | None = None
    audio_formats: set[str] = set()
    audio_arrivals: list[float] = []
    audio_chunk_samples: list[int] = []
    event_count = 0
    audio_chunk_count = 0

    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(json.dumps(event["error"], ensure_ascii=False))
            event_count += 1
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if isinstance(text, str) and text:
                if first_text is None:
                    first_text = time.perf_counter()
                text_parts.append(text)
            audio_delta = decode_audio_delta(delta)
            if audio_delta is not None:
                pcm16, chunk_rate, audio_format = audio_delta
                if first_audio is None:
                    first_audio = time.perf_counter()
                if sample_rate is not None and sample_rate != chunk_rate:
                    raise ValueError(f"Audio sample rate changed from {sample_rate} to {chunk_rate}")
                sample_rate = chunk_rate
                audio_formats.add(audio_format)
                pcm_parts.append(pcm16)
                audio_arrivals.append(time.perf_counter())
                audio_chunk_samples.append(len(pcm16) // 2)
                audio_chunk_count += 1

    ended = time.perf_counter()
    transcript = "".join(text_parts).strip()
    pcm16 = b"".join(pcm_parts)
    if pcm16 and sample_rate:
        wav_path = (
            args.output.parent
            / "audio"
            / args.output.stem
            / f"{args.mode}-{run_index:02d}.wav"
        )
        write_wav(wav_path, pcm16, sample_rate)
    else:
        wav_path = None

    result: dict[str, Any] = {
        "run_index": run_index,
        "measured": measured,
        "first_text_ms": None if first_text is None else (first_text - started) * 1000,
        "first_pcm_ms": None if first_audio is None else (first_audio - started) * 1000,
        "total_ms": (ended - started) * 1000,
        "transcript": transcript,
        "sse_events": event_count,
        "audio_chunks": audio_chunk_count,
        "audio_formats": sorted(audio_formats),
        "audio_sample_rate": sample_rate,
        "audio_seconds": None if not pcm16 or not sample_rate else len(pcm16) / 2.0 / sample_rate,
        "first_chunk_ms": (
            None
            if not audio_chunk_samples or not sample_rate
            else audio_chunk_samples[0] * 1000.0 / sample_rate
        ),
        "audio_chunk_ms": (
            None
            if not audio_chunk_samples or not sample_rate
            else summarize([samples * 1000.0 / sample_rate for samples in audio_chunk_samples])
        ),
        "largest_pcm_arrival_gap_ms": (
            None
            if len(audio_arrivals) < 2
            else max(
                (current - previous) * 1000.0
                for previous, current in zip(audio_arrivals, audio_arrivals[1:])
            )
        ),
        "audio_wav": None if wav_path is None else str(wav_path),
        "pcm": pcm_stats(pcm16),
        "pcm_sha256": hashlib.sha256(pcm16).hexdigest() if pcm16 else None,
    }
    if result["audio_seconds"] and result["first_pcm_ms"] is not None:
        result["stream_generation_rtf"] = (
            (result["total_ms"] - result["first_pcm_ms"]) / 1000.0 / result["audio_seconds"]
        )
    else:
        result["stream_generation_rtf"] = None
    if args.reference_text is not None:
        result["reference_text"] = args.reference_text
        result["character_error_rate_percent"] = character_error_rate(args.reference_text, transcript)
        result["exact_normalized"] = (
            normalize_for_cer(args.reference_text) == normalize_for_cer(transcript)
        )
    return result


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise ValueError("--runs must be at least 1 and --warmups cannot be negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total_runs = args.warmups + args.runs
    for index in range(total_runs):
        measured = index >= args.warmups
        result = request_once(args, index + 1, measured)
        results.append(result)
        print(
            json.dumps(
                {
                    "run": index + 1,
                    "measured": measured,
                    "first_text_ms": result["first_text_ms"],
                    "first_pcm_ms": result["first_pcm_ms"],
                    "total_ms": result["total_ms"],
                    "transcript": result["transcript"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    measured_results = [result for result in results if result["measured"]]
    metrics = {}
    for key in (
        "first_text_ms",
        "first_pcm_ms",
        "total_ms",
        "audio_seconds",
        "largest_pcm_arrival_gap_ms",
        "stream_generation_rtf",
        "character_error_rate_percent",
    ):
        values = [
            float(value)
            for result in measured_results
            if (value := result.get(key)) is not None
        ]
        metrics[key] = summarize(values)
    output = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "mode": args.mode,
        "model": args.model,
        "system_prompt": args.system_prompt or SYSTEM_PROMPTS[args.mode],
        "input_audio": None if args.input_audio is None else str(args.input_audio.resolve()),
        "input_audio_sha256": (
            None if args.input_audio is None else sha256_file(args.input_audio)
        ),
        "input_text": args.text,
        "request_parameters": {
            "stream": True,
            "temperature": 0,
            "reset_context": True,
            "modalities": {
                "interleaved": ["text", "audio"],
                "asr": ["text"],
                "tts": ["audio"],
            }[args.mode],
        },
        "warmups": args.warmups,
        "measured_runs": args.runs,
        "max_tokens": args.max_tokens,
        "summary": metrics,
        "runs": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
