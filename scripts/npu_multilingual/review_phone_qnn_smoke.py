#!/usr/bin/env python3
"""Promote raw SM8850 SenseVoice smoke only after exact CPU parity passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-smoke", type=Path, required=True)
    parser.add_argument("--cpu-parity", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--runner-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    args = parser.parse_args()

    raw = load(args.raw_smoke)
    parity = load(args.cpu_parity)
    bundle = load(args.bundle_manifest)
    runner = load(args.runner_manifest)

    assert raw["status"] == "raw_pass"
    assert raw["provider"] == "qnn"
    assert raw["soc"] == bundle["target"]["soc"] == "SM8850"
    assert raw["htp_arch"] == bundle["target"]["htp_arch"] == "v81"
    assert raw["qnn_version"] == bundle["target"]["qnn_version"] == "2.40.0.251030"
    assert raw["model_bin_sha256"] == bundle["files"]["model.bin"]["sha256"]
    assert parity["parity"] == {"en": "pass", "zh": "pass"}
    assert parity["tokens_sha256"] == bundle["files"]["tokens.txt"]["sha256"]
    assert runner["target"] == {"abi": "arm64-v8a", "android_api": 27, "provider": "qnn"}

    qnn_runs = {}
    cpu_runs = {run["wav"]: run for run in parity["runs"]}
    for run in raw["runs"]:
        assert run["returncode"] == 0
        assert "Initializing HtpProvider" in run["stdout"]
        record = json.loads(next(line for line in run["stdout"].splitlines() if line.startswith("{")))
        cpu = cpu_runs[run["wav"]]
        assert cpu["qnn"]["text"] == record["text"]
        assert cpu["qnn"]["language"] == record["lang"]
        assert cpu["exact_text_parity_fixed_10s"] is True
        assert cpu["language_parity_fixed_10s"] is True
        qnn_runs[run["wav"][:2]] = {
            "wav_sha256": cpu["wav_sha256"],
            "duration_seconds": cpu["duration_seconds"],
            "language_tag": record["lang"],
            "transcript": record["text"],
            "qnn_decode_ms": float(run["stderr"].split("Elapsed seconds: ", 1)[1].split(" s", 1)[0]) * 1000,
            "whole_command_wall_ms": run["wall_seconds"] * 1000,
            "cpu_fixed_10s_decode_ms": cpu["cpu_fixed_10s"]["decode_ms"],
            "exact_fixed_10s_cpu_parity": True,
            "unpadded_cpu_transcript": cpu["cpu_variable"]["text"],
            "unpadded_matches_qnn": cpu["exact_text_parity_variable"],
        }

    runtime_files = {}
    runtime_root = Path(bundle["paths"]["qnn_runtime_root"])
    for name in ("libQnnHtp.so", "libQnnHtpPrepare.so", "libQnnHtpV81Skel.so", "libQnnHtpV81Stub.so", "libQnnSystem.so"):
        path = runtime_root / name
        expected = bundle["files"][name]
        assert path.stat().st_size == expected["bytes"]
        assert sha256(path) == expected["sha256"]
        runtime_files[name] = expected

    reviewed = {
        "schema_version": 1,
        "status": "pass",
        "runtime_load": "pass",
        "provider": "qnn",
        "soc": "SM8850",
        "htp_arch": "v81",
        "qnn_version": "2.40.0.251030",
        "model_bin_sha256": raw["model_bin_sha256"],
        "tokens_sha256": parity["tokens_sha256"],
        "transcript_parity": parity["parity"],
        "input_contract": "fixed_10_seconds_zero_padded",
        "runs": qnn_runs,
        "artifacts": {
            "runtime": runtime_files,
            "runner": runner["files"],
            "cpu_model_sha256": parity["model_sha256"],
        },
        "limitations": [
            "single physical QNN run per language",
            "no APK/JNI integration yet",
            "no sustained latency, RSS, thermal, or microphone/VAD QNN run",
            "unpadded CPU inference differs by one word/character; exact parity requires the fixed 10-second contract",
        ],
    }
    args.output.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.device_output.write_text(json.dumps(raw["device"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reviewed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
