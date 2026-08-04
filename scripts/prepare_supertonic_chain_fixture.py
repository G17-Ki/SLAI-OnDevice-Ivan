#!/usr/bin/env python3
"""Prepare a real, deterministic fixed-shape Supertonic chain fixture.

This mirrors the public Sherpa-ONNX preprocessing and inference order. It runs
only small local ONNX Runtime inference and never contacts Qualcomm AI Hub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import struct
import time
from typing import Any
import unicodedata

import numpy as np
import onnxruntime as ort

from supertonic_chain_common import (
    audio_metrics,
    compare_arrays,
    describe_array,
    sha256_file,
    write_json,
    write_pcm16_wav,
)


MODEL_FILES = {
    "duration_predictor": "duration_predictor.int8.onnx",
    "text_encoder": "text_encoder.int8.onnx",
    "vector_estimator": "vector_estimator.int8.onnx",
    "vocoder": "vocoder.int8.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default="It is Paris, in France.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fixed-text-length", type=int, default=32)
    parser.add_argument("--fixed-latent-length", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-dynamic-reference", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.fixed_text_length < 1 or args.fixed_latent_length < 1:
        parser.error("steps and fixed lengths must be positive")
    if args.speed <= 0 or args.threads < 1:
        parser.error("speed and threads must be positive")
    return args


def preprocess_text(text: str, language: str) -> str:
    replacements = (
        ("–", "-"),
        ("‑", "-"),
        ("—", "-"),
        ("_", " "),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
        ("´", "'"),
        ("`", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("→", " "),
        ("←", " "),
        ("♥", ""),
        ("☆", ""),
        ("♡", ""),
        ("©", ""),
        ("\\", ""),
        ("@", " at "),
        ("e.g.,", "for example, "),
        ("i.e.,", "that is, "),
    )
    result = text
    for source, replacement in replacements:
        result = result.replace(source, replacement)
    result = "".join(
        character for character in result if not 0x1F000 <= ord(character) <= 0x1FFFF
    )
    result = re.sub(r"\s+([,.!?;:'])", r"\1", result)
    result = result.replace('""', '"').replace("''", "'").replace("`", "")
    result = " ".join(result.split())
    if result and result[-1] not in ".!?;:,'\")]}>…。」』】〉》›»“”‘’":
        result += "."
    return f"<{language}>{result}</{language}>"


def text_tensors(
    processed: str, indexer_path: Path, fixed_length: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    indexer = np.fromfile(indexer_path, dtype="<i4")
    normalized = unicodedata.normalize("NFKD", processed)
    ids = [
        int(indexer[ord(character)]) if ord(character) < indexer.size else 0
        for character in normalized
    ]
    if len(ids) > fixed_length:
        raise ValueError(
            f"Processed text has {len(ids)} IDs, exceeding fixed length {fixed_length}"
        )
    padded = np.zeros((1, fixed_length), dtype=np.int64)
    padded[0, : len(ids)] = ids
    mask = np.zeros((1, 1, fixed_length), dtype=np.float32)
    mask[0, 0, : len(ids)] = 1.0
    return padded, mask, ids


def parse_voice_style(
    path: Path, speaker: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = path.read_bytes()
    if len(payload) < 48:
        raise ValueError(f"Voice style header is truncated: {path}")
    dimensions = struct.unpack("<6q", payload[:48])
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"Invalid voice dimensions: {dimensions}")
    ttl_shape = tuple(int(value) for value in dimensions[:3])
    dp_shape = tuple(int(value) for value in dimensions[3:])
    if ttl_shape[0] != dp_shape[0] or not 0 <= speaker < ttl_shape[0]:
        raise ValueError(f"Invalid speaker {speaker} for voice shapes {dimensions}")
    ttl_count = int(np.prod(ttl_shape))
    dp_count = int(np.prod(dp_shape))
    expected = 48 + (ttl_count + dp_count) * 4
    if len(payload) != expected:
        raise ValueError(f"Voice style size {len(payload)} does not match {expected}")
    values = np.frombuffer(payload, dtype="<f4", offset=48)
    style_ttl = values[:ttl_count].reshape(ttl_shape)[speaker : speaker + 1].copy()
    style_dp = values[ttl_count:].reshape(dp_shape)[speaker : speaker + 1].copy()
    return (
        style_ttl,
        style_dp,
        {
            "speaker_count": ttl_shape[0],
            "speaker": speaker,
            "ttl_shape": list(style_ttl.shape),
            "dp_shape": list(style_dp.shape),
        },
    )


def create_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_vector_chain(
    session: ort.InferenceSession,
    initial: np.ndarray,
    text_embedding: np.ndarray,
    style_ttl: np.ndarray,
    latent_mask: np.ndarray,
    text_mask: np.ndarray,
    steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    latent = initial.astype(np.float32, copy=True)
    raw_outputs: list[np.ndarray] = []
    masked_outputs: list[np.ndarray] = []
    for step in range(steps):
        raw = np.asarray(
            session.run(
                None,
                {
                    "noisy_latent": latent,
                    "text_emb": text_embedding,
                    "style_ttl": style_ttl,
                    "latent_mask": latent_mask,
                    "text_mask": text_mask,
                    "current_step": np.asarray([step], dtype=np.float32),
                    "total_step": np.asarray([steps], dtype=np.float32),
                },
            )[0],
            dtype=np.float32,
        )
        masked = raw * latent_mask
        raw_outputs.append(raw.copy())
        masked_outputs.append(masked.copy())
        latent = masked
    return raw_outputs, masked_outputs


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        *MODEL_FILES.values(),
        "voice.bin",
        "unicode_indexer.bin",
        "tts.json",
    ]
    for filename in required:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)

    started = time.perf_counter()
    config = json.loads((model_dir / "tts.json").read_text())
    sample_rate = int(config["ae"]["sample_rate"])
    samples_per_latent = int(config["ae"]["base_chunk_size"]) * int(
        config["ttl"]["chunk_compress_factor"]
    )
    latent_channels = int(config["ttl"]["latent_dim"]) * int(
        config["ttl"]["chunk_compress_factor"]
    )

    processed = preprocess_text(args.text, args.language)
    text_ids, text_mask, unpadded_ids = text_tensors(
        processed, model_dir / "unicode_indexer.bin", args.fixed_text_length
    )
    style_ttl, style_dp, voice_record = parse_voice_style(
        model_dir / "voice.bin", args.speaker
    )
    sessions = {
        name: create_session(model_dir / filename, args.threads)
        for name, filename in MODEL_FILES.items()
    }

    duration_raw = np.asarray(
        sessions["duration_predictor"].run(
            None,
            {
                "text_ids": text_ids,
                "style_dp": style_dp,
                "text_mask": text_mask,
            },
        )[0],
        dtype=np.float32,
    )
    duration = duration_raw / np.float32(args.speed)
    if args.speed != 1.0:
        duration = np.maximum(duration, np.float32(0.1))
    text_embedding = np.asarray(
        sessions["text_encoder"].run(
            None,
            {
                "text_ids": text_ids,
                "style_ttl": style_ttl,
                "text_mask": text_mask,
            },
        )[0],
        dtype=np.float32,
    )

    wav_length = max(1, int(float(duration.reshape(-1)[0]) * sample_rate))
    real_latent_length = (wav_length + samples_per_latent - 1) // samples_per_latent
    if real_latent_length > args.fixed_latent_length:
        raise ValueError(
            f"Duration needs {real_latent_length} latent positions, exceeding "
            f"fixed length {args.fixed_latent_length}"
        )
    latent_mask = np.zeros((1, 1, args.fixed_latent_length), dtype=np.float32)
    latent_mask[:, :, :real_latent_length] = 1.0
    rng = np.random.default_rng(args.seed)
    latent_initial = rng.standard_normal(
        (1, latent_channels, args.fixed_latent_length), dtype=np.float32
    )
    latent_initial *= latent_mask
    step_raw, step_masked = run_vector_chain(
        sessions["vector_estimator"],
        latent_initial,
        text_embedding,
        style_ttl,
        latent_mask,
        text_mask,
        args.steps,
    )
    waveform_padded = np.asarray(
        sessions["vocoder"].run(None, {"latent": step_masked[-1]})[0],
        dtype=np.float32,
    )
    waveform_trimmed = waveform_padded[:, :wav_length].copy()

    arrays: dict[str, np.ndarray] = {
        "text_ids_local": text_ids,
        "text_ids_remote": text_ids.astype(np.int32),
        "text_mask": text_mask,
        "style_ttl": style_ttl,
        "style_dp": style_dp,
        "duration_raw_local": duration_raw,
        "duration_local": duration,
        "text_embedding_local": text_embedding,
        "latent_mask": latent_mask,
        "latent_initial": latent_initial,
        "waveform_padded_local": waveform_padded,
        "waveform_trimmed_local": waveform_trimmed,
    }
    for step, (raw, masked) in enumerate(zip(step_raw, step_masked, strict=True)):
        arrays[f"latent_step_{step}_raw_local"] = raw
        arrays[f"latent_step_{step}_masked_local"] = masked

    dynamic_record: dict[str, Any] = {"status": "skipped"}
    if not args.skip_dynamic_reference:
        dynamic_mask = np.ones((1, 1, real_latent_length), dtype=np.float32)
        dynamic_initial = latent_initial[:, :, :real_latent_length].copy()
        dynamic_raw, dynamic_masked = run_vector_chain(
            sessions["vector_estimator"],
            dynamic_initial,
            text_embedding,
            style_ttl,
            dynamic_mask,
            text_mask,
            args.steps,
        )
        dynamic_waveform = np.asarray(
            sessions["vocoder"].run(None, {"latent": dynamic_masked[-1]})[0],
            dtype=np.float32,
        )
        dynamic_trimmed = dynamic_waveform[:, :wav_length].copy()
        arrays.update(
            {
                "latent_final_dynamic_local": dynamic_masked[-1],
                "waveform_dynamic_local": dynamic_waveform,
                "waveform_dynamic_trimmed_local": dynamic_trimmed,
            }
        )
        dynamic_record = {
            "status": "completed",
            "latent_length": real_latent_length,
            "waveform": audio_metrics(
                waveform_trimmed, dynamic_trimmed, sample_rate=sample_rate
            ),
            "note": (
                "This compares fixed-length padded inference with a dynamic-length "
                "run using the same real-position initial noise. Flow trajectories "
                "are stochastic, so ASR/listening evidence is more meaningful than "
                "sample-aligned waveform correlation."
            ),
        }

    fixture_path = output_dir / "fixture.npz"
    np.savez(fixture_path, **arrays)
    write_pcm16_wav(
        output_dir / "local_fixed_trimmed.wav", waveform_trimmed, sample_rate
    )
    if "waveform_dynamic_trimmed_local" in arrays:
        write_pcm16_wav(
            output_dir / "local_dynamic_trimmed.wav",
            arrays["waveform_dynamic_trimmed_local"],
            sample_rate,
        )

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "local_threads": args.threads,
        },
        "fixture": {
            "text": args.text,
            "language": args.language,
            "processed_text": processed,
            "unpadded_text_length": len(unpadded_ids),
            "text_ids": unpadded_ids,
            "fixed_text_length": args.fixed_text_length,
            "speaker": args.speaker,
            "steps": args.steps,
            "seed": args.seed,
            "speed": args.speed,
            "duration_raw_s": float(duration_raw.reshape(-1)[0]),
            "duration_s": float(duration.reshape(-1)[0]),
            "sample_rate": sample_rate,
            "wav_length": wav_length,
            "samples_per_latent": samples_per_latent,
            "latent_channels": latent_channels,
            "real_latent_length": real_latent_length,
            "fixed_latent_length": args.fixed_latent_length,
            "padded_positions": args.fixed_latent_length - real_latent_length,
            "latent_mask_last": float(latent_mask[0, 0, -1]),
        },
        "voice": voice_record,
        "models": {
            name: {
                "filename": filename,
                "bytes": (model_dir / filename).stat().st_size,
                "sha256": sha256_file(model_dir / filename),
            }
            for name, filename in MODEL_FILES.items()
        },
        "support_files": {
            filename: {
                "bytes": (model_dir / filename).stat().st_size,
                "sha256": sha256_file(model_dir / filename),
            }
            for filename in ("voice.bin", "unicode_indexer.bin", "tts.json")
        },
        "fixture_file": {
            "filename": fixture_path.name,
            "bytes": fixture_path.stat().st_size,
            "sha256": sha256_file(fixture_path),
        },
        "arrays": {name: describe_array(value) for name, value in arrays.items()},
        "local_checks": {
            "final_padding_is_zero": bool(
                np.count_nonzero(step_masked[-1][:, :, real_latent_length:]) == 0
            ),
            "raw_and_masked_final": compare_arrays(
                step_raw[-1], step_masked[-1], atol=0.0, rtol=0.0
            ),
            "dynamic_reference": dynamic_record,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output_dir / "fixture_manifest.json", manifest)
    print(
        json.dumps(
            {
                "fixture": str(fixture_path),
                "processed_text": processed,
                "duration_s": manifest["fixture"]["duration_s"],
                "real_latent_length": real_latent_length,
                "fixed_latent_length": args.fixed_latent_length,
                "elapsed_s": manifest["elapsed_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
