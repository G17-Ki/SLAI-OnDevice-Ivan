#!/usr/bin/env python3
"""Verify the immutable Supertonic stage-diagnostic APK without device access.

The diagnostic spec pins the complete APK before ZIP inspection and pins the
accepted r1 chain spec as the source of the unchanged model/runtime inventory.
"""

from __future__ import annotations

import argparse
import collections
import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/model_sweep_20260729"
DEFAULT_APK = ROOT / (
    "android-app/supertonic-chain-diagnostic-probe/build/outputs/apk/debug/"
    "supertonic-chain-diagnostic-probe-debug.apk"
)
DEFAULT_SPEC = REPORT_ROOT / "supertonic_chain_diagnostic_ort_qnn_probe_spec.json"
DEFAULT_BUILD_TOOLS = Path.home() / ".cache/lmf-demo/android-sdk/build-tools/34.0.0"
DEFAULT_JAVA_HOME = Path(
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
)
BASE_VERIFIER_PATH = ROOT / "scripts/verify_supertonic_chain_ort_qnn_probe_apk.py"
BASE_SPEC_RELATIVE = (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_chain_ort_qnn_probe_spec.json"
)
STAGER_RELATIVE = "scripts/stage_supertonic_chain_diagnostic_ort_qnn_probe.py"
GATE_ID = (
    "supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2"
)
PACKAGE = "ai.liquid.lfmdemo.supertonicchaindiagprobe"
VERSION_NAME = "0.2.0-stage-diagnostic-raw-onnx-qnn-probe"
LAUNCHER = PACKAGE + ".ProbeActivity"
PAYLOAD_MANIFEST_MEMBER = (
    "assets/supertonic_chain_diagnostic_probe/payload_manifest.json"
)
BASE_ASSET_PREFIX = "supertonic_chain_probe/"
DIAGNOSTIC_ASSET_PREFIX = "supertonic_chain_diagnostic_probe/"
SIGNING_CERTIFICATE_SHA256 = (
    "9ad7b2846b7c18853b99e444d4d05323321b3a9118c13c5a8fc242e58b5859e4"
)
ACCEPTED_PACKAGE_HASHES = {
    "ai.liquid.lfmdemo": (
        "a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a"
    ),
    "ai.liquid.lfmdemo.runtime": (
        "33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244"
    ),
}
EXPECTED_DIAGNOSTIC_POLICY = {
    "completed_stages_recorded": True,
    "cpu_ep_fallback_remains_disabled": True,
    "failure_phase_recorded": True,
    "failure_stage_recorded": True,
    "promotion_allowed": False,
}
EXPECTED_CLAIM_BOUNDARY = (
    "Isolated stage-diagnostic raw-ONNX four-component scaffold. It preserves "
    "CPU EP fallback disabled and records the exact failing component/phase plus "
    "completed stages. A pass or failure is diagnostic only and cannot promote "
    "full TTS, PCM quality, language identity, latency, unplugged durability, or "
    "production integration."
)


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import verifier helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


base_verifier = load_module("verify_supertonic_chain_apk_base", BASE_VERIFIER_PATH)


def sha256(path: Path) -> str:
    return base_verifier.sha256(path)


def load_object(path: Path) -> dict[str, Any]:
    return base_verifier.load_object(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(f"{label} field inventory mismatch: {actual!r}")
    return value


def expected_apk_policy() -> dict[str, Any]:
    return {
        "package": PACKAGE,
        "version_code": 1,
        "version_name": VERSION_NAME,
        "compile_sdk": 35,
        "min_sdk": 28,
        "target_sdk": 35,
        "launcher": LAUNCHER,
        "application_label": "Supertonic chain diagnostic",
        "permissions": [],
        "native_abis": ["arm64-v8a"],
        "required_native_libraries": [],
        "optional_native_libraries": ["libcdsprpc.so"],
        "optional_features": [],
        "debuggable": True,
        "allow_backup": False,
        "supports_rtl": True,
        "extract_native_libs": True,
        "uses_cleartext_traffic": False,
        "launcher_exported": True,
        "signing": {
            "v1": False,
            "v2": True,
            "v3": False,
            "v3_1": False,
            "v4": False,
            "source_stamp": False,
            "signer_count": 1,
            "certificate_sha256": SIGNING_CERTIFICATE_SHA256,
        },
    }


def source_spec(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = exact_keys(
        spec["source_chain_spec"], {"path", "bytes", "sha256"}, "source_chain_spec"
    )
    base_verifier.validate_identity(record, "source_chain_spec")
    require(record["path"] == BASE_SPEC_RELATIVE, "source spec path mismatch")
    path = (ROOT / record["path"]).resolve()
    base_verifier.verify_file(path, record)
    source = base_verifier.load_object(path)
    base_verifier.validate_spec(source, path)
    return path, source


def derived_assets(source: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in source["required_assets"]:
        path = record["path"]
        require(path.startswith(BASE_ASSET_PREFIX), "source asset prefix mismatch")
        updated = copy.deepcopy(record)
        updated["path"] = DIAGNOSTIC_ASSET_PREFIX + path.removeprefix(
            BASE_ASSET_PREFIX
        )
        output.append(updated)
    return output


def validate_spec(
    spec: dict[str, Any], spec_path: Path
) -> tuple[Path, dict[str, Any]]:
    exact_keys(
        spec,
        {
            "schema_version",
            "gate_id",
            "target",
            "apk",
            "payload_manifest",
            "stager",
            "source_chain_spec",
            "source_payload_manifest",
            "apk_policy",
            "accepted_packages",
            "diagnostic_policy",
            "zip_policy",
            "device_output_contract",
            "inheritance",
            "claim_boundary",
        },
        "diagnostic spec",
    )
    require(spec.get("schema_version") == 3, "diagnostic spec schema mismatch")
    require(spec.get("gate_id") == GATE_ID, "diagnostic gate mismatch")
    require(
        spec.get("target")
        == {
            "package": PACKAGE,
            "soc": "SM8850",
            "abi": "arm64-v8a",
            "htp_arch": "v81",
            "android_api": 28,
        },
        "diagnostic target mismatch",
    )
    for label in ("apk", "payload_manifest", "stager", "source_payload_manifest"):
        base_verifier.validate_identity(spec.get(label), label)
    require(
        spec["payload_manifest"].get("path")
        == "supertonic_chain_diagnostic_probe/payload_manifest.json",
        "diagnostic payload-manifest path mismatch",
    )
    require(spec["stager"].get("path") == STAGER_RELATIVE, "stager path mismatch")
    base_verifier.verify_file(ROOT / STAGER_RELATIVE, spec["stager"])
    require(
        spec["source_payload_manifest"]
        == {
            "bytes": 19_378,
            "sha256": (
                "735e61b070c7ad199a67bc18829036f4575551b5533a49e9ead5b36392f0819f"
            ),
        },
        "source payload-manifest identity mismatch",
    )
    require(spec["apk_policy"] == expected_apk_policy(), "APK policy mismatch")
    require(
        spec["accepted_packages"]
        == {
            package: {"sha256": digest}
            for package, digest in ACCEPTED_PACKAGE_HASHES.items()
        },
        "accepted-package trust roots mismatch",
    )
    require(
        spec["diagnostic_policy"]
        == {
            **EXPECTED_DIAGNOSTIC_POLICY,
            "diagnostic_schema_version": 2,
            "diagnostic_marker": "SUPERTONIC_CHAIN_STAGE_DIAGNOSTIC_V2",
            "ordered_success_stages": [
                "payload_install",
                "runtime_initialization",
                "duration_predictor",
                "text_encoder",
                "vector_estimator",
                "vocoder",
            ],
            "component_phases": [
                "session_create",
                "contract_validation",
                "execute",
                "output_validation",
            ],
            "vector_step_index_recorded": True,
            "completed_stage_reports_recorded": True,
            "completed_vector_step_reports_recorded": True,
        },
        "diagnostic evidence policy mismatch",
    )
    require(
        spec["zip_policy"]
        == {
            "alignment_bytes": 4,
            "all_assets_stored": True,
            "other_members": [
                "AndroidManifest.xml",
                "META-INF/com/android/build/gradle/app-metadata.properties",
                "classes.dex",
                "classes2.dex",
                "classes3.dex",
                "resources.arsc",
            ],
        },
        "diagnostic ZIP policy mismatch",
    )
    require(
        spec["device_output_contract"]
        == {
            "evidence_schema_version": 1,
            "diagnostic_schema_version": 2,
            "report_path": (
                "/data/user/0/ai.liquid.lfmdemo.supertonicchaindiagprobe/files/"
                "supertonic_chain_diagnostic_probe/reports/latest.json"
            ),
            "partial_report_path": (
                "/data/user/0/ai.liquid.lfmdemo.supertonicchaindiagprobe/files/"
                "supertonic_chain_diagnostic_probe/reports/latest.json.partial"
            ),
            "completion_marker": (
                "SUPERTONIC_CHAIN_DIAGNOSTIC_REPORT_COMPLETE_V2"
            ),
            "diagnostic_marker": "SUPERTONIC_CHAIN_STAGE_DIAGNOSTIC_V2",
            "run_trigger": "automation_intent",
        },
        "diagnostic device-output contract mismatch",
    )
    require(
        spec["inheritance"]
        == {
            "from_source_chain_spec": [
                "runtime",
                "models",
                "fixture",
                "fixture_source",
                "verification_policy",
                "capabilities",
                "ort_aar",
                "payload_native_libraries",
                "required_native_libraries",
                "device_policy",
                "verification_toolchain",
            ],
            "asset_path_prefix_from": BASE_ASSET_PREFIX,
            "asset_path_prefix_to": DIAGNOSTIC_ASSET_PREFIX,
        },
        "diagnostic inheritance contract mismatch",
    )
    require(spec["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "claim mismatch")
    require(
        spec_path == spec_path.resolve() and spec_path.is_file(),
        "diagnostic spec path must be a resolved file",
    )
    return source_spec(spec)


def verify_embedded_manifest(
    raw: bytes, spec: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    record = spec["payload_manifest"]
    require(len(raw) == record["bytes"], "embedded manifest byte count mismatch")
    require(
        base_verifier.hashlib.sha256(raw).hexdigest() == record["sha256"],
        "embedded manifest SHA-256 mismatch",
    )
    manifest = base_verifier.strict_json_bytes(raw, PAYLOAD_MANIFEST_MEMBER)
    exact_keys(
        manifest,
        {
            "schema_version",
            "gate_id",
            "target",
            "runtime",
            "models",
            "fixture",
            "verification_policy",
            "capabilities",
            "assets",
            "native_libraries",
            "diagnostic_policy",
            "source_payload_manifest",
            "claim_boundary",
        },
        "embedded diagnostic manifest",
    )
    require(manifest["schema_version"] == 1, "embedded manifest schema mismatch")
    require(manifest["gate_id"] == GATE_ID, "embedded manifest gate mismatch")
    target = copy.deepcopy(source["target"])
    target["package"] = PACKAGE
    require(manifest["target"] == target, "embedded manifest target mismatch")
    for field in (
        "runtime",
        "models",
        "fixture",
        "verification_policy",
        "capabilities",
    ):
        require(
            manifest[field] == source[field],
            f"embedded inherited field mismatch: {field}",
        )
    require(manifest["assets"] == derived_assets(source), "asset inventory mismatch")
    require(
        manifest["native_libraries"] == source["payload_native_libraries"],
        "native inventory mismatch",
    )
    require(
        manifest["diagnostic_policy"] == EXPECTED_DIAGNOSTIC_POLICY,
        "embedded diagnostic policy mismatch",
    )
    require(
        manifest["source_payload_manifest"] == spec["source_payload_manifest"],
        "embedded source provenance mismatch",
    )
    require(manifest["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "claim mismatch")
    return manifest


def verify_archive(
    apk: Path, spec: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    expected_assets = {
        "assets/" + record["path"]: record for record in derived_assets(source)
    }
    expected_native = {
        "lib/" + record["path"]: record
        for record in source["required_native_libraries"]
    }
    expected_asset_names = set(expected_assets) | {PAYLOAD_MANIFEST_MEMBER}
    expected_native_names = set(expected_native)
    expected_other = set(spec["zip_policy"]["other_members"])
    with zipfile.ZipFile(apk) as bundle:
        names = bundle.namelist()
        duplicates = sorted(
            name for name, count in collections.Counter(names).items() if count != 1
        )
        require(not duplicates, f"APK contains duplicate ZIP entries: {duplicates}")
        actual_assets = {name for name in names if name.startswith("assets/")}
        actual_native = {name for name in names if name.startswith("lib/")}
        actual_other = {
            name for name in names if not name.startswith(("assets/", "lib/"))
        }
        require(actual_assets == expected_asset_names, "APK asset set mismatch")
        require(actual_native == expected_native_names, "APK native set mismatch")
        require(actual_other == expected_other, "APK non-payload set mismatch")
        manifest_info = bundle.getinfo(PAYLOAD_MANIFEST_MEMBER)
        require(
            manifest_info.compress_type == zipfile.ZIP_STORED,
            "embedded payload manifest is compressed",
        )
        manifest = verify_embedded_manifest(
            bundle.read(PAYLOAD_MANIFEST_MEMBER), spec, source
        )
        verified_assets = [
            base_verifier.verify_member(bundle, name, record, require_stored=True)
            for name, record in sorted(expected_assets.items())
        ]
        verified_native = [
            base_verifier.verify_member(bundle, name, record)
            for name, record in sorted(expected_native.items())
        ]
    return {
        "status": "pass",
        "manifest_gate_id": manifest["gate_id"],
        "payload_manifest_sha256": spec["payload_manifest"]["sha256"],
        "verified_assets": verified_assets,
        "verified_native_libraries": verified_native,
        "other_members": sorted(expected_other),
    }


def verify_apk(
    apk: Path,
    spec: dict[str, Any],
    *,
    spec_path: Path,
    build_tools: Path,
    java_home: Path,
) -> dict[str, Any]:
    source_path, source = validate_spec(spec, spec_path)
    require(apk.is_file(), f"missing diagnostic APK: {apk}")
    expected = spec["apk"]
    require(apk.stat().st_size == expected["bytes"], "whole APK byte mismatch")
    actual_sha256 = sha256(apk)
    require(actual_sha256 == expected["sha256"], "whole APK SHA-256 mismatch")

    # ZIP data is intentionally untrusted until the complete APK identity passes.
    archive = verify_archive(apk, spec, source)
    tool_spec = {
        "verification_toolchain": source["verification_toolchain"],
        "apk_policy": spec["apk_policy"],
    }
    tools, toolchain = base_verifier.verify_toolchain(
        build_tools, java_home, tool_spec
    )
    binary = base_verifier.verify_binary_metadata(
        apk, tools, java_home, tool_spec
    )
    return {
        "schema_version": 3,
        "gate_id": GATE_ID,
        "package": PACKAGE,
        "status": "pass",
        "apk": {
            "path": str(apk),
            "bytes": apk.stat().st_size,
            "sha256": actual_sha256,
        },
        "authoritative_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": sha256(spec_path),
            "whole_apk_identity_required_before_zip_inspection": True,
        },
        "source_chain_spec": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256(source_path),
        },
        "stager": base_verifier.verify_file(ROOT / STAGER_RELATIVE, spec["stager"]),
        "archive": archive,
        "binary": binary,
        "verification_toolchain": toolchain,
        "device_mutated": False,
        "claim_boundary": (
            "Local immutable diagnostic-package integrity only. No Android/QNN "
            "execution, stage compatibility, PCM quality, latency, or production claim."
        ),
    }


def atomic_write(path: Path, payload: bytes) -> None:
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, default=DEFAULT_APK)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--build-tools-dir", type=Path, default=DEFAULT_BUILD_TOOLS)
    parser.add_argument("--java-home", type=Path, default=DEFAULT_JAVA_HOME)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    apk = arguments.apk.expanduser().resolve()
    spec_path = arguments.spec.expanduser().resolve()
    report = verify_apk(
        apk,
        load_object(spec_path),
        spec_path=spec_path,
        build_tools=arguments.build_tools_dir.expanduser().resolve(),
        java_home=arguments.java_home.expanduser().resolve(),
    )
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if arguments.report is not None:
        atomic_write(arguments.report.expanduser().resolve(), encoded)
    print(encoded.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
