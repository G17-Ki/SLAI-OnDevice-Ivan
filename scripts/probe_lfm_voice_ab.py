#!/usr/bin/env python3
"""Capture LFM TTS SSE, WAV, latency, and signal metrics without changing the phone."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import struct
import time
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SAMPLE_RATE_DEFAULT = 24_000


@dataclass
class AudioChunk:
    received_ms: float
    sample_rate_hz: int
    encoding: str
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--case", action="append", dest="selected_cases")
    return parser.parse_args()


def cases() -> list[dict[str, Any]]:
    text = (
        "Good morning. I found a quiet table by the window, and the sunlight makes "
        "the whole room feel warm and calm. Would you like me to read the next message?"
    )
    prompts = {
        "us_female": "Perform TTS. Use the US female voice.",
        "us_male": "Perform TTS. Use the US male voice.",
        "uk_male": "Perform TTS. Use the UK male voice.",
        "uk_female": "Perform TTS. Use the UK female voice.",
    }
    return [
        {
            "name": "current_us_female_temp0",
            "text": text,
            "system_prompt": prompts["us_female"],
            "sampling": {"temperature": 0},
            "purpose": "Exact currently deployed app TTS payload sampling.",
        },
        {
            "name": "us_male_temp0",
            "text": text,
            "system_prompt": prompts["us_male"],
            "sampling": {"temperature": 0},
            "purpose": "Official predefined voice, current greedy sampling.",
        },
        {
            "name": "uk_male_temp0",
            "text": text,
            "system_prompt": prompts["uk_male"],
            "sampling": {"temperature": 0},
            "purpose": "Official predefined voice, current greedy sampling.",
        },
        {
            "name": "uk_female_temp0",
            "text": text,
            "system_prompt": prompts["uk_female"],
            "sampling": {"temperature": 0},
            "purpose": "Official predefined voice, current greedy sampling.",
        },
        {
            "name": "us_female_temp04_topk32",
            "text": text,
            "system_prompt": prompts["us_female"],
            "sampling": {"temperature": 0.4, "top_k": 32},
            "purpose": "Conservative non-greedy prosody probe.",
        },
        {
            "name": "us_female_temp08_topk64",
            "text": text,
            "system_prompt": prompts["us_female"],
            "sampling": {"temperature": 0.8, "top_k": 64},
            "purpose": "Sampling recommended by the upstream LFM TTS example.",
        },
        {
            "name": "interleaved_exact_sentence",
            "text": (
                "Say this exact sentence verbatim: Good morning. I found a quiet table by the window, "
                "and the sunlight makes the whole room feel warm and calm. Would you like me to read "
                "the next message?"
            ),
            "system_prompt": "Respond with interleaved text and audio.",
            "sampling": {"temperature": 0},
            "modalities": ["text", "audio"],
            "purpose": "Experimental conversational-tone path requested after the fixed-voice capture.",
        },
    ]


def payload(case: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": "LFM2.5-Audio-1.5B",
        "stream": True,
        "reset_context": True,
        "modalities": case.get("modalities", ["audio"]),
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": case["system_prompt"]},
            {"role": "user", "content": case["text"]},
        ],
    }
    result.update(case["sampling"])
    return result


def decode_audio_delta(delta: dict[str, Any]) -> tuple[array, int, str] | None:
    legacy = delta.get("audio_chunk")
    current = delta.get("audio")
    item = legacy or current
    if not item:
        return None
    if isinstance(item, str):
        encoded = item
        encoding = "pcm16" if current else "float32"
        sample_rate = SAMPLE_RATE_DEFAULT
    else:
        encoded = item.get("data")
        if not encoded:
            return None
        encoding = str(item.get("format") or item.get("encoding") or ("pcm16" if current else "float32"))
        sample_rate = int(item.get("sample_rate") or SAMPLE_RATE_DEFAULT)
    raw = base64.b64decode(encoded, validate=True)
    normalized = encoding.lower()
    if any(key in normalized for key in ("pcm16", "s16", "16le", "int16")) or (current and "float" not in normalized):
        values = array("h")
        values.frombytes(raw)
        if struct.pack("=h", 1) != struct.pack("<h", 1):
            values.byteswap()
        samples = array("f", (value / 32768.0 for value in values))
        return samples, sample_rate, encoding
    values = array("f")
    values.frombytes(raw)
    if struct.pack("=f", 1.0) != struct.pack("<f", 1.0):
        values.byteswap()
    return values, sample_rate, encoding


def audio_metrics(samples: array, sample_rate: int) -> dict[str, Any]:
    count = len(samples)
    if not count:
        return {"sample_count": 0, "duration_s": 0.0}
    finite = [float(value) for value in samples if math.isfinite(value)]
    if len(finite) != count:
        raise ValueError("non-finite PCM sample")
    peak = max(abs(value) for value in finite)
    rms = math.sqrt(sum(value * value for value in finite) / count)
    clipping_count = sum(abs(value) >= 0.999 for value in finite)
    crossings = sum(
        1
        for left, right in zip(finite, finite[1:])
        if (left < 0 <= right) or (left > 0 >= right)
    )

    frame_samples = max(1, round(sample_rate * 0.02))
    silence_threshold = 10 ** (-45 / 20)
    silent_frames: list[bool] = []
    for start in range(0, count, frame_samples):
        frame = finite[start : start + frame_samples]
        frame_rms = math.sqrt(sum(value * value for value in frame) / len(frame))
        silent_frames.append(frame_rms < silence_threshold)
    runs: list[dict[str, Any]] = []
    run_start: int | None = None
    for index, silent in enumerate(silent_frames + [False]):
        if silent and run_start is None:
            run_start = index
        elif not silent and run_start is not None:
            duration_ms = (index - run_start) * 20.0
            if duration_ms >= 200.0:
                runs.append(
                    {
                        "start_ms": run_start * 20.0,
                        "duration_ms": duration_ms,
                        "is_leading": run_start == 0,
                        "is_trailing": index == len(silent_frames),
                    }
                )
            run_start = None
    internal = [run for run in runs if not run["is_leading"] and not run["is_trailing"]]
    return {
        "sample_rate_hz": sample_rate,
        "sample_count": count,
        "duration_s": count / sample_rate,
        "peak": peak,
        "peak_dbfs": 20 * math.log10(max(peak, 1e-12)),
        "rms": rms,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "clipping_count_ge_0_999": clipping_count,
        "clipping_fraction": clipping_count / count,
        "zero_crossings": crossings,
        "zero_crossing_rate": crossings / max(1, count - 1),
        "silence_definition": "20 ms frame RMS below -45 dBFS; report runs >=200 ms",
        "silence_runs": runs,
        "internal_silence_gap_count": len(internal),
        "longest_internal_silence_gap_ms": max((run["duration_ms"] for run in internal), default=0.0),
    }


def write_wav(path: Path, samples: array, sample_rate: int) -> None:
    pcm = array(
        "h",
        (
            max(-32768, min(32767, round(max(-1.0, min(1.0, float(value))) * 32767)))
            for value in samples
        ),
    )
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        pcm.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    case_payload = payload(case)
    prefix = args.out / case["name"]
    request_path = prefix.with_suffix(".request.json")
    raw_path = prefix.with_suffix(".sse")
    wav_path = prefix.with_suffix(".wav")
    metadata_path = prefix.with_suffix(".json")
    request_path.write_text(json.dumps(case_payload, indent=2, ensure_ascii=False) + "\n")

    body = json.dumps(case_payload, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    response = connection.getresponse()
    headers_ms = (time.perf_counter() - started) * 1000
    response_headers = dict(response.getheaders())
    raw = bytearray()
    event_buffer: list[bytes] = []
    chunks: list[AudioChunk] = []
    samples = array("f")
    sample_rate: int | None = None
    finish_reason: str | None = None
    first_audio_ms: float | None = None
    parse_errors: list[str] = []
    text_deltas: list[str] = []

    while True:
        line = response.readline()
        if not line:
            if event_buffer:
                parse_errors.append("unterminated final SSE event")
            break
        raw.extend(line)
        if line not in (b"\n", b"\r\n"):
            event_buffer.append(line)
            continue
        event_raw = b"".join(event_buffer)
        event_buffer.clear()
        data_lines = [part[5:].lstrip() for part in event_raw.splitlines() if part.startswith(b"data:")]
        if not data_lines:
            continue
        data = b"\n".join(data_lines)
        if data == b"[DONE]":
            continue
        try:
            event = json.loads(data)
            if event.get("error"):
                parse_errors.append(f"server error: {event['error']}")
                continue
            choice = (event.get("choices") or [{}])[0]
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                text_deltas.append(content)
            decoded = decode_audio_delta(delta)
            if decoded is None:
                continue
            current_samples, current_rate, encoding = decoded
            received_ms = (time.perf_counter() - started) * 1000
            if first_audio_ms is None:
                first_audio_ms = received_ms
            if sample_rate is None:
                sample_rate = current_rate
            elif sample_rate != current_rate:
                raise ValueError(f"sample-rate change {sample_rate} -> {current_rate}")
            samples.extend(current_samples)
            chunks.append(AudioChunk(received_ms, current_rate, encoding, len(current_samples)))
        except Exception as error:  # preserve the raw evidence even if one event is malformed
            parse_errors.append(f"{type(error).__name__}: {error}")

    total_ms = (time.perf_counter() - started) * 1000
    connection.close()
    raw_path.write_bytes(raw)
    if sample_rate and samples:
        write_wav(wav_path, samples, sample_rate)

    result: dict[str, Any] = {
        "name": case["name"],
        "purpose": case["purpose"],
        "text": case["text"],
        "system_prompt": case["system_prompt"],
        "sampling": case["sampling"],
        "http": {
            "status": response.status,
            "reason": response.reason,
            "headers": response_headers,
            "response_headers_ms": headers_ms,
        },
        "timing": {
            "first_audio_ms": first_audio_ms,
            "total_response_ms": total_ms,
            "generation_minus_audio_duration_ms": None,
        },
        "finish_reason": finish_reason,
        "generated_text": "".join(text_deltas),
        "audio_chunks": [asdict(chunk) for chunk in chunks],
        "parse_errors": parse_errors,
        "files": {
            "request": request_path.name,
            "raw_sse": raw_path.name,
            "wav": wav_path.name if wav_path.exists() else None,
        },
    }
    if samples and sample_rate:
        result["signal"] = audio_metrics(samples, sample_rate)
        result["timing"]["generation_minus_audio_duration_ms"] = total_ms - result["signal"]["duration_s"] * 1000
        result["files"]["wav_sha256"] = sha256(wav_path)
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    selected = set(args.selected_cases or [])
    selected_cases = [case for case in cases() if not selected or case["name"] in selected]
    if selected and len(selected_cases) != len(selected):
        missing = sorted(selected - {case["name"] for case in selected_cases})
        raise SystemExit(f"unknown case(s): {', '.join(missing)}")
    for case in selected_cases:
        print(f"running {case['name']}", flush=True)
        try:
            result = run_case(args, case)
        except Exception as error:
            result = {
                "name": case["name"],
                "system_prompt": case["system_prompt"],
                "sampling": case["sampling"],
                "fatal_error": f"{type(error).__name__}: {error}",
            }
            (args.out / f"{case['name']}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
        results.append(result)
        print(json.dumps(result.get("timing", result), ensure_ascii=False), flush=True)
    summary = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": f"http://{args.host}:{args.port}/v1/chat/completions",
        "case_count": len(results),
        "results": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0 if all("fatal_error" not in result for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
