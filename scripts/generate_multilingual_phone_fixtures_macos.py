#!/usr/bin/env python3
"""Generate canonical 16 kHz mono PCM16 phone-acceptance WAVs on macOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


FIXTURES = {
    "question_en": {
        "voice": "Samantha",
        "text": "What is the capital of France? Answer in one short sentence.",
    },
    "context_store_en": {
        "voice": "Samantha",
        "text": "Remember the secret phrase cobalt tulip.",
    },
    "context_recall_en": {
        "voice": "Samantha",
        "text": "What secret phrase did I ask you to remember?",
    },
    "long_response_en": {
        "voice": "Samantha",
        "text": "Explain how photosynthesis works in four detailed sentences.",
    },
    "question_zh": {
        "voice": "Tingting",
        "text": "请用中文简单介绍一下北京。",
    },
    "question_ru": {
        "voice": "Milena",
        "text": "Расскажи коротко о Москве.",
    },
    "question_fr": {
        "voice": "Thomas",
        "text": "Dis-moi brièvement quelle est la capitale de la France.",
    },
    "question_it": {
        "voice": "Alice",
        "text": "Dimmi brevemente qual è la capitale d'Italia.",
    },
    "question_es": {
        "voice": "Paulina",
        "text": "Dime brevemente cuál es la capital de España.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "phone_acceptance_fixtures",
    )
    return parser.parse_args()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required tool is unavailable: {name}")
    return path


def canonicalize(source: Path, destination: Path) -> dict[str, int | float]:
    with wave.open(str(source), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
        frame_count = reader.getnframes()
    if (channels, sample_width, sample_rate) != (1, 2, 16_000):
        raise SystemExit(
            f"Unexpected intermediate format for {source}: "
            f"channels={channels} width={sample_width} rate={sample_rate}"
        )
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(frames)
    size = destination.stat().st_size
    if size > 320_044:
        raise SystemExit(f"Fixture exceeds the ten-second companion bound: {destination}")
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frame_count,
        "duration_seconds": frame_count / sample_rate,
        "bytes": size,
    }


def main() -> None:
    args = parse_args()
    say = require_tool("say")
    ffmpeg = require_tool("ffmpeg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "format": "canonical pcm_s16le WAV, 16 kHz, mono",
        "fixtures": {},
    }
    with tempfile.TemporaryDirectory(prefix="lfm-phone-fixtures-") as temporary:
        work = Path(temporary)
        for name, spec in FIXTURES.items():
            aiff = work / f"{name}.aiff"
            intermediate = work / f"{name}.wav"
            destination = args.output_dir / f"{name}.wav"
            subprocess.run(
                [say, "-v", spec["voice"], "-r", "180", "-o", str(aiff), spec["text"]],
                check=True,
            )
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(aiff),
                    "-map_metadata",
                    "-1",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(intermediate),
                ],
                check=True,
            )
            facts = canonicalize(intermediate, destination)
            facts.update(
                {
                    "file": destination.name,
                    "voice": spec["voice"],
                    "text": spec["text"],
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
            manifest["fixtures"][name] = facts
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
