#!/usr/bin/env python3
"""Stage a distinct, host-only Supertonic chain stage-diagnostic payload."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_PAYLOAD = Path(
    "/private/tmp/lmf-supertonic-chain-ort-qnn-probe-payload-v1"
)
GATE_ID = (
    "supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2"
)
PACKAGE = "ai.liquid.lfmdemo.supertonicchaindiagprobe"
BASE_GATE_ID = "supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1"
BASE_PACKAGE = "ai.liquid.lfmdemo.supertonicchainprobe"
BASE_PREFIX = "supertonic_chain_probe/"
DIAGNOSTIC_PREFIX = "supertonic_chain_diagnostic_probe/"
BASE_MANIFEST_BYTES = 19_378
BASE_MANIFEST_SHA256 = (
    "735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f"
)
ORT_BYTES = 7_720_525
ORT_SHA256 = "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198"
CLAIM_BOUNDARY = (
    "Isolated stage-diagnostic raw-ONNX four-component scaffold. It preserves "
    "CPU EP fallback disabled and records the exact failing component/phase plus "
    "completed stages. A pass or failure is diagnostic only and cannot promote "
    "full TTS, PCM quality, language identity, latency, unplugged durability, or "
    "production integration."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-payload", type=Path, default=DEFAULT_BASE_PAYLOAD
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing pinned base payload file: {path}")
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise RuntimeError(f"base payload identity mismatch: {path}")


def require_external_output(output: Path) -> None:
    output = output.resolve()
    repository = ROOT.resolve()
    if output == Path(output.anchor) or len(output.parts) < 4:
        raise RuntimeError(f"refusing broad staging output: {output}")
    if output == repository or repository in output.parents:
        raise RuntimeError("diagnostic staging output must be outside the repository")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"diagnostic staging output must be absent or empty: {output}")


def atomic_write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial output exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def link_checked(
    source: Path,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    require_file(source, expected_bytes, expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"diagnostic destination already exists: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    require_file(destination, expected_bytes, expected_sha256)


def load_base_manifest(base_payload: Path) -> tuple[Path, dict[str, Any]]:
    path = base_payload / "assets/supertonic_chain_probe/payload_manifest.json"
    require_file(path, BASE_MANIFEST_BYTES, BASE_MANIFEST_SHA256)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("gate_id") != BASE_GATE_ID
        or manifest.get("target", {}).get("package") != BASE_PACKAGE
        or manifest.get("runtime", {}).get("disable_cpu_ep_fallback") is not True
        or manifest.get("verification_policy", {}).get(
            "host_reference_comparisons_promoted"
        )
        is not False
        or manifest.get("capabilities", {}).get("production_integration")
        is not False
    ):
        raise RuntimeError("base payload semantic identity mismatch")
    return path, manifest


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    base_payload = arguments.base_payload.expanduser().resolve()
    require_external_output(output)
    output.mkdir(parents=True, exist_ok=True)
    base_manifest_path, base = load_base_manifest(base_payload)
    try:
        diagnostic = copy.deepcopy(base)
        diagnostic["gate_id"] = GATE_ID
        diagnostic["target"]["package"] = PACKAGE
        diagnostic["assets"] = []
        for record in base["assets"]:
            source_path = record["path"]
            if not source_path.startswith(BASE_PREFIX):
                raise RuntimeError(f"unexpected base asset path: {source_path}")
            diagnostic_path = DIAGNOSTIC_PREFIX + source_path.removeprefix(
                BASE_PREFIX
            )
            updated = copy.deepcopy(record)
            updated["path"] = diagnostic_path
            diagnostic["assets"].append(updated)
            link_checked(
                base_payload / "assets" / source_path,
                output / "assets" / diagnostic_path,
                record["bytes"],
                record["sha256"],
            )

        for record in base["native_libraries"]:
            link_checked(
                base_payload / "jniLibs" / record["path"],
                output / "jniLibs" / record["path"],
                record["bytes"],
                record["sha256"],
            )
        link_checked(
            base_payload / "aar/onnxruntime-android-qnn-1.25.0.aar",
            output / "aar/onnxruntime-android-qnn-1.25.0.aar",
            ORT_BYTES,
            ORT_SHA256,
        )

        diagnostic["diagnostic_policy"] = {
            "completed_stages_recorded": True,
            "cpu_ep_fallback_remains_disabled": True,
            "failure_phase_recorded": True,
            "failure_stage_recorded": True,
            "promotion_allowed": False,
        }
        diagnostic["source_payload_manifest"] = {
            "bytes": BASE_MANIFEST_BYTES,
            "sha256": BASE_MANIFEST_SHA256,
        }
        diagnostic["claim_boundary"] = CLAIM_BOUNDARY
        manifest_bytes = (
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
        ).encode()
        manifest_path = (
            output
            / "assets/supertonic_chain_diagnostic_probe/payload_manifest.json"
        )
        manifest_identity = atomic_write(manifest_path, manifest_bytes)
        summary = {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "package": PACKAGE,
            "base_payload_manifest": {
                "path": str(base_manifest_path),
                "bytes": BASE_MANIFEST_BYTES,
                "sha256": BASE_MANIFEST_SHA256,
            },
            "payload_manifest": {
                "path": str(manifest_path),
                **manifest_identity,
            },
            "host_model_execution_during_staging": False,
            "device_mutated": False,
            "status": "staged",
        }
        atomic_write(
            output / "diagnostic_probe_payload_manifest.json",
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException:
        if output.is_dir():
            shutil.rmtree(output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
