#!/usr/bin/env python3
"""Classify a completed accumulated Piper chain without contacting AI Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submit_component_aihub import compare_output
from supertonic_chain_common import (
    audio_metrics,
    describe_array,
    sha256_file,
    write_json,
    write_pcm16_wav,
)


SAMPLE_RATE = 22_050
FIRST_CHUNK_FRAMES = 40
UPSAMPLE_FACTOR = 256
DECODER_STAGE_PATTERN = re.compile(r"^decoder_(\d+)$")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def all_comparison_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "passed" in value:
        return [value]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for child in value.values():
            rows.extend(all_comparison_rows(child))
        return rows
    return []


def finite_archive(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    descriptions = {name: describe_array(value) for name, value in arrays.items()}
    return {
        "all_finite": all(item["finite"] for item in descriptions.values()),
        "arrays": descriptions,
    }


def stage_job_summary(state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, stage in state.get("stages", {}).items():
        job = stage.get("job", {})
        result[name] = {
            "job_id": job.get("job_id"),
            "job_status": job.get("status"),
            "stage_status": stage.get("status"),
            "output_sha256": stage.get("output_sha256"),
        }
    return result


def decoder_stage_names(state: dict[str, Any]) -> list[str]:
    """Return decoder stages in numeric order and reject incomplete numbering."""

    indexed: list[tuple[int, str]] = []
    for name in state.get("stages", {}):
        match = DECODER_STAGE_PATTERN.fullmatch(name)
        if match:
            indexed.append((int(match.group(1)), name))
    indexed.sort()
    if not indexed:
        raise ValueError("No decoder_N stage exists in accumulated-chain state")
    indices = [index for index, _ in indexed]
    expected = list(range(len(indexed)))
    if indices != expected:
        raise ValueError(
            f"Decoder stages are not contiguous from decoder_0: {indices}"
        )
    return [name for _, name in indexed]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--first-chunk-dir",
        type=Path,
        help="Optional directory for deterministic local/remote first-chunk WAVs",
    )
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    args = parser.parse_args()

    chain_dir = args.chain_dir.expanduser().resolve()
    state_path = chain_dir / "state.json"
    state = json.loads(state_path.read_text())
    local_path = chain_dir / "local_chain.npz"
    remote_path = chain_dir / "remote_chain.npz"
    local = load_npz(local_path)
    remote = load_npz(remote_path)

    decoder_names = decoder_stage_names(state)
    remote_stage_names = ["encoder", "sdp", "flow", *decoder_names]
    remote_stage_paths = {
        name: chain_dir / "remote" / f"{name}.npz"
        for name in remote_stage_names
    }
    remote_stages = {name: load_npz(path) for name, path in remote_stage_paths.items()}

    required_stages = {"encoder", "sdp", "flow", *decoder_names}
    stage_names = set(state.get("stages", {}))
    execution_complete = stage_names.issuperset(required_stages) and all(
        state["stages"][name].get("job", {}).get("success") is True
        and state["stages"][name].get("status") == "passed"
        for name in required_stages
    )

    archive_checks = {
        "local_chain": finite_archive(local),
        "remote_chain": finite_archive(remote),
        "remote_stages": {
            name: finite_archive(arrays) for name, arrays in remote_stages.items()
        },
    }
    all_finite = (
        archive_checks["local_chain"]["all_finite"]
        and archive_checks["remote_chain"]["all_finite"]
        and all(
            item["all_finite"]
            for item in archive_checks["remote_stages"].values()
        )
    )

    runner_comparisons = state.get("result", {}).get("comparisons", {})
    comparison_rows = all_comparison_rows(runner_comparisons)
    runner_numerical_pass = bool(comparison_rows) and all(
        row.get("passed") is True for row in comparison_rows
    )

    decoder_comparisons = {}
    for name in decoder_names:
        index = int(DECODER_STAGE_PATTERN.fullmatch(name).group(1))
        remote_audio = remote_stages[name]["audio"]
        local_audio = local[f"decoder_output_{index}"]
        decoder_comparisons[name] = compare_output(
            remote_audio, local_audio, args.atol, args.rtol
        )
    decoder_numerical_pass = all(
        value.get("passed") is True for value in decoder_comparisons.values()
    )

    local_audio = np.asarray(local["audio"], dtype=np.float32).reshape(-1)
    remote_audio = np.asarray(remote["audio"], dtype=np.float32).reshape(-1)
    first_chunk_samples = FIRST_CHUNK_FRAMES * UPSAMPLE_FACTOR
    first_chunk_metrics = audio_metrics(
        remote_audio[:first_chunk_samples],
        local_audio[:first_chunk_samples],
        sample_rate=SAMPLE_RATE,
    )
    full_audio_metrics = audio_metrics(
        remote_audio, local_audio, sample_rate=SAMPLE_RATE
    )

    first_chunk_files = None
    if args.first_chunk_dir:
        chunk_dir = args.first_chunk_dir.expanduser().resolve()
        local_chunk = chunk_dir / "local_first_chunk.wav"
        remote_chunk = chunk_dir / "remote_first_chunk.wav"
        write_pcm16_wav(local_chunk, local_audio[:first_chunk_samples], SAMPLE_RATE)
        write_pcm16_wav(remote_chunk, remote_audio[:first_chunk_samples], SAMPLE_RATE)
        first_chunk_files = {
            "local": {"filename": local_chunk.name, "sha256": sha256_file(local_chunk)},
            "remote": {
                "filename": remote_chunk.name,
                "sha256": sha256_file(remote_chunk),
            },
        }

    numerical_pass = runner_numerical_pass and decoder_numerical_pass and all_finite
    if not execution_complete:
        status = "incomplete"
    elif numerical_pass:
        status = "execution_complete_numerical_pass"
    else:
        status = "execution_complete_numerical_warning"

    result = {
        "schema_version": 1,
        "status": status,
        "runner_raw_status": state.get("status"),
        "target": state.get("target"),
        "fixture": state.get("fixture"),
        "execution_complete": execution_complete,
        "all_remote_and_local_arrays_finite": all_finite,
        "runner_numerical_pass": runner_numerical_pass,
        "decoder_chunk_numerical_pass": decoder_numerical_pass,
        "overall_numerical_pass": numerical_pass,
        "jobs": stage_job_summary(state),
        "runner_comparisons": runner_comparisons,
        "decoder_chunk_comparisons": decoder_comparisons,
        "audio_metrics": {
            "full": full_audio_metrics,
            "first_chunk": first_chunk_metrics,
        },
        "streaming_boundary": {
            "first_chunk_frames": FIRST_CHUNK_FRAMES,
            "upsample_factor": UPSAMPLE_FACTOR,
            "first_chunk_samples": first_chunk_samples,
            "first_chunk_seconds": first_chunk_samples / SAMPLE_RATE,
            "complete_audio_samples": int(remote_audio.size),
            "complete_audio_seconds": remote_audio.size / SAMPLE_RATE,
            "decoder_calls": len(decoder_names),
        },
        "artifact_hashes": {
            "state_json": sha256_file(state_path),
            "local_chain_npz": sha256_file(local_path),
            "remote_chain_npz": sha256_file(remote_path),
            "local_reference_wav": sha256_file(chain_dir / "local_reference.wav"),
            "remote_chain_wav": sha256_file(chain_dir / "remote_chain.wav"),
            "remote_stage_npz": {
                name: sha256_file(path) for name, path in remote_stage_paths.items()
            },
        },
        "first_chunk_files": first_chunk_files,
        "archive_checks": archive_checks,
        "interpretation": (
            "Numerical comparison and ASR intelligibility are separate gates. "
            "This result does not measure phonemization, host orchestration, "
            "context loading, phone playback, or acoustic onset."
        ),
    }
    write_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if execution_complete and all_finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
