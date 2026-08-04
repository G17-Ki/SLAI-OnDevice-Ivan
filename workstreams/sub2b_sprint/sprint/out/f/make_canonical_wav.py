#!/usr/bin/env python3
"""Lane F — produce canonical phone-contract WAVs.

`AsrHttpServer.canonicalWav()` (android-app/runtime-companion/.../AsrHttpServer.java:494)
accepts only a 44-byte-header RIFF/WAVE file that is PCM format 1, mono,
16,000 Hz, 16 bits per sample, with a `data` chunk length equal to
`file_bytes - 44`. Anything else is rejected with `invalid_wav` before a
backend runs.

This converts arbitrary mono source WAVs into exactly that layout so the
fixture that is measured on the host is byte-identical to the fixture that is
pushed to the phone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import wave
from pathlib import Path

import numpy as np
import soxr

TARGET_RATE = 16000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mono_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        channels = source.getnchannels()
        rate = source.getframerate()
        raw = source.readframes(source.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def write_canonical(path: Path, samples: np.ndarray) -> None:
    """Write a 44-byte-header canonical WAV, no extra chunks."""
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, TARGET_RATE,
                                    TARGET_RATE * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    path.write_bytes(header + pcm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True,
                        help="id=path pairs")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in arguments.source:
        fixture_id, _, raw_path = pair.partition("=")
        source_path = Path(raw_path)
        samples, rate = read_mono_pcm16(source_path)
        if rate != TARGET_RATE:
            samples = soxr.resample(samples, rate, TARGET_RATE, quality="VHQ")
        destination = arguments.out_dir / f"{fixture_id}.wav"
        write_canonical(destination, samples)

        payload = destination.read_bytes()
        assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"
        assert struct.unpack("<H", payload[20:22])[0] == 1
        assert struct.unpack("<H", payload[22:24])[0] == 1
        assert struct.unpack("<I", payload[24:28])[0] == TARGET_RATE
        assert struct.unpack("<H", payload[34:36])[0] == 16
        assert struct.unpack("<I", payload[40:44])[0] == len(payload) - 44

        rows.append(
            {
                "fixture_id": fixture_id,
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "source_sample_rate_hz": rate,
                "path": destination.name,
                "bytes": len(payload),
                "sha256": sha256_file(destination),
                "sample_count": (len(payload) - 44) // 2,
                "duration_seconds": ((len(payload) - 44) // 2) / TARGET_RATE,
                "canonical_contract": "riff44_pcm16_mono_16000hz",
                "fits_fixed_10s_window": len(payload) <= 320_044,
            }
        )
        print(f"{fixture_id}: {rows[-1]['bytes']} bytes "
              f"{rows[-1]['duration_seconds']:.3f} s {rows[-1]['sha256']}")

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps({"schema_version": 1, "target_sample_rate_hz": TARGET_RATE,
                    "resampler": f"soxr {soxr.__version__} VHQ",
                    "fixtures": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
