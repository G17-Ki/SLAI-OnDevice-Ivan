#!/usr/bin/env python3
"""Persistent client for sherpa-onnx-offline-websocket-server.

The wire format is the upstream Sherpa protocol: little-endian int32 sample
rate, little-endian int32 float payload size, then normalized float32 samples.
One WebSocket remains open for every requested utterance.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import struct
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable, Protocol


class WebSocketLike(Protocol):
    async def send(self, message: bytes | str) -> None: ...

    async def recv(self) -> bytes | str: ...


def encode_pcm16_wave(path: Path) -> tuple[int, bytes]:
    """Return sample rate and normalized little-endian float32 PCM bytes."""

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono WAV, got {wav.getnchannels()} channels")
        if wav.getsampwidth() != 2:
            raise ValueError(
                f"{path}: expected PCM16 WAV, got {wav.getsampwidth() * 8}-bit samples"
            )
        if wav.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed WAV is unsupported: {wav.getcomptype()}")
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    pcm16 = array.array("h")
    pcm16.frombytes(raw)
    if sys.byteorder != "little":
        pcm16.byteswap()
    samples = array.array("f", (sample / 32768.0 for sample in pcm16))
    if sys.byteorder != "little":
        samples.byteswap()
    return sample_rate, samples.tobytes()


def frame_payload(path: Path) -> bytes:
    sample_rate, samples = encode_pcm16_wave(path)
    if len(samples) > 0x7FFFFFFF:
        raise ValueError(f"{path}: float payload exceeds the protocol int32 limit")
    return struct.pack("<ii", sample_rate, len(samples)) + samples


async def send_utterance(
    websocket: WebSocketLike,
    path: Path,
    *,
    chunk_bytes: int = 10240,
    timeout_seconds: float = 30.0,
    require_nonempty: bool = False,
) -> dict[str, Any]:
    if chunk_bytes < 8:
        raise ValueError("chunk_bytes must be at least 8")

    payload = frame_payload(path)
    started = time.perf_counter()
    for offset in range(0, len(payload), chunk_bytes):
        await websocket.send(payload[offset : offset + chunk_bytes])

    raw_result = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if isinstance(raw_result, bytes):
        raw_result = raw_result.decode("utf-8")
    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Server returned invalid JSON for {path}: {raw_result!r}") from exc

    text = result.get("text")
    if require_nonempty and (not isinstance(text, str) or not text.strip()):
        raise RuntimeError(f"Server returned an empty transcript for {path}: {result!r}")

    return {
        "wave": str(path),
        "latency_ms": round(latency_ms, 3),
        "result": result,
    }


async def transcribe_on_connection(
    websocket: WebSocketLike,
    paths: Iterable[Path],
    *,
    repeat: int = 1,
    chunk_bytes: int = 10240,
    timeout_seconds: float = 30.0,
    require_nonempty: bool = False,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for _ in range(repeat):
        for path in paths:
            outputs.append(
                await send_utterance(
                    websocket,
                    path,
                    chunk_bytes=chunk_bytes,
                    timeout_seconds=timeout_seconds,
                    require_nonempty=require_nonempty,
                )
            )
    await websocket.send("Done")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode WAV files sequentially over one persistent Sherpa WebSocket"
    )
    parser.add_argument("wav", nargs="+", type=Path, help="mono PCM16 WAV file")
    parser.add_argument("--server-addr", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6007)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--chunk-bytes", type=int, default=10240)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--require-nonempty", action="store_true")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    for path in args.wav:
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install the host client dependency: pip install websockets") from exc

    uri = f"ws://{args.server_addr}:{args.server_port}"
    async with websockets.connect(
        uri,
        open_timeout=args.timeout_seconds,
        close_timeout=2,
        ping_interval=None,
        max_size=None,
    ) as websocket:
        outputs = await transcribe_on_connection(
            websocket,
            args.wav,
            repeat=args.repeat,
            chunk_bytes=args.chunk_bytes,
            timeout_seconds=args.timeout_seconds,
            require_nonempty=args.require_nonempty,
        )

    for output in outputs:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
