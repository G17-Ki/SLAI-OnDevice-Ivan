#!/usr/bin/env python3
"""Fail-closed whole-APK verification for the Supertonic four-stage QNN probe.

The committed specification is the trust root.  The APK and its embedded
manifest are evidence checked against it; neither is allowed to define the
expected payload dynamically.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, BinaryIO
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APK = ROOT / (
    "android-app/supertonic-chain-probe/build/outputs/apk/debug/"
    "supertonic-chain-probe-debug.apk"
)
DEFAULT_SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_chain_ort_qnn_probe_spec.json"
)
DEFAULT_BUILD_TOOLS = Path.home() / ".cache/lmf-demo/android-sdk/build-tools/34.0.0"
DEFAULT_JAVA_HOME = Path(
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
)
GATE_ID = "supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1"
PACKAGE = "ai.liquid.lfmdemo.supertonicchainprobe"
VERSION_NAME = "0.1.0-full-chain-raw-onnx-qnn-probe"
LAUNCHER = PACKAGE + ".ProbeActivity"
PAYLOAD_MANIFEST_MEMBER = "assets/supertonic_chain_probe/payload_manifest.json"
STAGER_RELATIVE = "scripts/stage_supertonic_chain_ort_qnn_probe.py"
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
PINNED_MODELS = {
    "duration_predictor": (
        "duration_predictor.int8.onnx",
        3_700_147,
        "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    ),
    "text_encoder": (
        "text_encoder.int8.onnx",
        36_416_150,
        "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    ),
    "vector_estimator": (
        "vector_estimator.int8.onnx",
        78_400_833,
        "20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd",
    ),
    "vocoder": (
        "vocoder.int8.onnx",
        25_991_073,
        "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152",
    ),
}
EXPECTED_OTHER_MEMBERS = {
    "META-INF/com/android/build/gradle/app-metadata.properties",
    "classes.dex",
    "classes2.dex",
    "classes3.dex",
    "AndroidManifest.xml",
    "resources.arsc",
}
EXPECTED_CLAIM_BOUNDARY = (
    "Isolated raw-ONNX four-component compatibility scaffold. A pass may prove "
    "QNN session/execution, chaining, finite tensors, shape safety, and a "
    "non-silent PCM tensor only. Host-reference tolerances, audio quality, "
    "language identity, playback, latency promotion, unplugged durability, and "
    "production integration remain unclaimed."
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    def invalid_constant(item: str) -> Any:
        raise RuntimeError(f"invalid JSON numeric constant in {label}: {item}")

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON in {label}") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"expected JSON object in {label}")
    return decoded


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return strict_json_bytes(path.read_bytes(), str(path))


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError(f"invalid integer for {label}: {value!r}")
    return value


def validate_identity(record: Any, label: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"invalid identity record: {label}")
    require_int(record.get("bytes"), label + ".bytes", minimum=1)
    if (
        not isinstance(record.get("sha256"), str)
        or SHA256_RE.fullmatch(record["sha256"]) is None
    ):
        raise RuntimeError(f"invalid SHA-256 for {label}")


def verify_file(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing pinned file: {path}")
    if path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"pinned file byte count mismatch: {path}")
    actual_sha256 = sha256(path)
    if actual_sha256 != record["sha256"]:
        raise RuntimeError(f"pinned file SHA-256 mismatch: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
    }


def unique_records(records: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{label} must be a non-empty array")
    output: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"invalid {label} record at index {index}")
        validate_identity(record, f"{label}[{index}]")
        output.append(record)
        paths.append(record["path"])
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"duplicate paths in {label}")
    return output


def validate_spec(spec: dict[str, Any], spec_path: Path) -> None:
    if spec.get("schema_version") != 2 or spec.get("gate_id") != GATE_ID:
        raise RuntimeError("authoritative chain spec identity mismatch")
    if spec.get("target") != {
        "package": PACKAGE,
        "soc": "SM8850",
        "abi": "arm64-v8a",
        "htp_arch": "v81",
        "android_api": 28,
    }:
        raise RuntimeError("authoritative chain target mismatch")
    validate_identity(spec.get("apk"), "apk")
    validate_identity(spec.get("payload_manifest"), "payload_manifest")
    validate_identity(spec.get("stager"), "stager")
    if spec["payload_manifest"].get("path") != (
        "supertonic_chain_probe/payload_manifest.json"
    ):
        raise RuntimeError("payload-manifest path mismatch")
    if spec["stager"].get("path") != STAGER_RELATIVE:
        raise RuntimeError("stager path mismatch")
    verify_file(ROOT / STAGER_RELATIVE, spec["stager"])

    policy = spec.get("apk_policy", {})
    expected_policy = {
        "package": PACKAGE,
        "version_code": 1,
        "version_name": VERSION_NAME,
        "compile_sdk": 35,
        "min_sdk": 28,
        "target_sdk": 35,
        "launcher": LAUNCHER,
        "application_label": "Supertonic chain probe",
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
    if policy != expected_policy:
        raise RuntimeError("authoritative APK policy mismatch")

    accepted = spec.get("accepted_packages")
    expected_accepted = {
        package: {"sha256": digest}
        for package, digest in ACCEPTED_PACKAGE_HASHES.items()
    }
    if accepted != expected_accepted:
        raise RuntimeError("accepted-package trust roots mismatch")

    models = spec.get("models")
    if not isinstance(models, dict) or set(models) != set(PINNED_MODELS):
        raise RuntimeError("authoritative model inventory mismatch")
    for component, (filename, size, digest) in PINNED_MODELS.items():
        record = models[component]
        if (
            record.get("filename") != filename
            or record.get("bytes") != size
            or record.get("sha256") != digest
        ):
            raise RuntimeError(f"authoritative model identity mismatch: {component}")

    assets = unique_records(spec.get("required_assets"), "required_assets")
    payload_native = unique_records(
        spec.get("payload_native_libraries"), "payload_native_libraries"
    )
    native = unique_records(
        spec.get("required_native_libraries"), "required_native_libraries"
    )
    asset_paths = {record["path"] for record in assets}
    for component, (filename, size, digest) in PINNED_MODELS.items():
        path = "supertonic_chain_probe/models/" + filename
        matches = [record for record in assets if record["path"] == path]
        if (
            len(matches) != 1
            or matches[0]["bytes"] != size
            or matches[0]["sha256"] != digest
        ):
            raise RuntimeError(f"required model asset mismatch: {component}")
    if spec["payload_manifest"]["path"] in asset_paths:
        raise RuntimeError("payload manifest must be pinned separately from assets")
    payload_paths = {record["path"] for record in payload_native}
    native_paths = {record["path"] for record in native}
    if not payload_paths.issubset(native_paths):
        raise RuntimeError("payload-native inventory is not packaged exactly")
    if native_paths - payload_paths != {
        "arm64-v8a/libonnxruntime.so",
        "arm64-v8a/libonnxruntime4j_jni.so",
    }:
        raise RuntimeError("unexpected ORT/native inventory")

    zip_policy = spec.get("zip_policy", {})
    other_members = zip_policy.get("other_members")
    if (
        not isinstance(other_members, list)
        or len(other_members) != len(set(other_members))
        or set(other_members) != EXPECTED_OTHER_MEMBERS
        or zip_policy.get("all_assets_stored") is not True
        or zip_policy.get("alignment_bytes") != 4
    ):
        raise RuntimeError("authoritative ZIP policy mismatch")
    if spec.get("runtime") != {
        "backend": "libQnnHtp.so",
        "disable_cpu_ep_fallback": True,
        "enable_htp_fp16_precision": True,
        "execution_mode": "sequential",
        "htp_performance_mode": "burst",
        "inter_op_threads": 1,
        "intra_op_threads": 1,
        "onnx_runtime": "1.25.0",
        "qairt": "2.42.0.251225135753_193295",
        "qnn_provider": "QNNExecutionProvider",
    }:
        raise RuntimeError("authoritative runtime contract mismatch")
    if spec.get("verification_policy") != {
        "audio_quality_promoted": False,
        "host_reference_comparisons_promoted": False,
        "host_reference_comparisons_recorded": True,
        "host_reference_tolerances_calibrated_on_phone": False,
        "require_all_four_sessions": True,
        "require_chained_outputs": True,
        "require_exact_shapes": True,
        "require_finite_tensors": True,
        "require_non_silent_waveform": True,
        "require_zero_masked_padding": True,
    }:
        raise RuntimeError("authoritative verification policy mismatch")
    if spec.get("capabilities") != {
        "audio_file_output": False,
        "audio_playback": False,
        "audio_quality_acceptance": False,
        "chained_pcm_tensor": True,
        "duration_execution": True,
        "four_step_vector_execution": True,
        "production_integration": False,
        "text_encoder_execution": True,
        "vocoder_execution": True,
    }:
        raise RuntimeError("authoritative capability boundary mismatch")
    if spec.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        raise RuntimeError("authoritative claim boundary mismatch")
    fixture = spec.get("fixture", {})
    expected_fixture_fields = {
        "text": "It is Paris, in France.",
        "language": "en",
        "processed_text": "<en>It is Paris, in France.</en>",
        "fixed_text_length": 32,
        "speaker": 6,
        "steps": 4,
        "seed": 42,
        "speed": 1,
        "sample_rate": 44_100,
        "samples_per_latent": 3_072,
        "latent_channels": 144,
        "real_latent_length": 31,
        "fixed_latent_length": 32,
        "padded_positions": 1,
        "wav_length": 93_826,
        "duration_reference": 2.127588987350464,
        "duration_absolute_tolerance": 0.1,
        "fixture_contract_sha256": (
            "660d62daa0570c1d18ca8a6ce4df3ea6d12bacbee019ffb63f77d61b299ba714"
        ),
    }
    for field, expected in expected_fixture_fields.items():
        if fixture.get(field) != expected:
            raise RuntimeError(f"authoritative fixture mismatch: {field}")
    if fixture.get("array_records") != [
        record for record in spec["required_assets"] if record.get("dtype") is not None
    ]:
        raise RuntimeError("authoritative fixture array inventory mismatch")
    if spec.get("fixture_source") != {
        "fixture_npz": {
            "bytes": 1_024_940,
            "sha256": "1a9f30f1e31c85777e335d9c68cde1ff35344fcf8153292bebd23121b56868e4",
        },
        "fixture_manifest": {
            "bytes": 10_138,
            "sha256": "277bc5b31320a06bbcff587df33a2338c8d080520ee0356d476cc9fc50649163",
        },
        "generator_runtime": {
            "numpy": "1.26.4",
            "onnxruntime": "1.28.0",
            "local_threads": 1,
            "omp_num_threads": 1,
        },
    }:
        raise RuntimeError("authoritative fixture-source identity mismatch")
    if spec.get("ort_aar") != {
        "bytes": 7_720_525,
        "sha256": "397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198",
    }:
        raise RuntimeError("authoritative ORT AAR identity mismatch")
    if spec.get("device_output_contract") != {
        "evidence_schema_version": 1,
        "report_path": (
            "/data/user/0/ai.liquid.lfmdemo.supertonicchainprobe/files/"
            "supertonic_chain_probe/reports/latest.json"
        ),
        "partial_report_path": (
            "/data/user/0/ai.liquid.lfmdemo.supertonicchainprobe/files/"
            "supertonic_chain_probe/reports/latest.json.partial"
        ),
        "completion_marker": "SUPERTONIC_CHAIN_PROBE_REPORT_COMPLETE_V1",
        "run_trigger": "automation_intent",
    }:
        raise RuntimeError("authoritative device-output contract mismatch")
    if spec.get("device_policy") != {
        "model": "NX809J",
        "soc": "SM8850",
        "abi": "arm64-v8a",
        "airplane_mode_on": 1,
        "max_process_rss_kib": 1_228_800,
        "max_resident_payload_kib": 358_400,
        "required_loaded_libraries": [
            "libQnnHtp.so",
            "libQnnHtpV81Stub.so",
            "libQnnSystem.so",
            "libcdsprpc.so",
            "libonnxruntime.so",
            "libonnxruntime4j_jni.so",
        ],
        "required_app_owned_libraries": [
            "libQnnHtp.so",
            "libQnnHtpV81Stub.so",
            "libQnnSystem.so",
            "libonnxruntime.so",
            "libonnxruntime4j_jni.so",
        ],
        "required_fastrpc_fd": "/dev/fastrpc-cdsp",
        "minimum_qcom_dmabuf_count": 1,
    }:
        raise RuntimeError("authoritative device policy mismatch")
    if spec_path != spec_path.resolve() or not spec_path.is_file():
        raise RuntimeError("authoritative spec path must be a resolved file")


def verify_member(
    bundle: zipfile.ZipFile,
    member: str,
    record: dict[str, Any],
    *,
    require_stored: bool = False,
) -> dict[str, Any]:
    try:
        info = bundle.getinfo(member)
    except KeyError as error:
        raise RuntimeError(f"APK is missing required member: {member}") from error
    if info.is_dir() or info.file_size != record["bytes"]:
        raise RuntimeError(f"APK member size/type mismatch: {member}")
    if require_stored and info.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"APK asset is unexpectedly compressed: {member}")
    with bundle.open(info) as stream:
        actual_sha256 = sha256_stream(stream)
    if actual_sha256 != record["sha256"]:
        raise RuntimeError(f"APK member SHA-256 mismatch: {member}")
    return {
        "path": member,
        "bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "sha256": actual_sha256,
        "stored": info.compress_type == zipfile.ZIP_STORED,
    }


def verify_embedded_manifest(
    raw: bytes,
    spec: dict[str, Any],
) -> dict[str, Any]:
    record = spec["payload_manifest"]
    if (
        len(raw) != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise RuntimeError("embedded payload-manifest identity mismatch")
    manifest = strict_json_bytes(raw, PAYLOAD_MANIFEST_MEMBER)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("embedded payload-manifest schema_version mismatch")
    exact_fields = (
        "gate_id",
        "target",
        "runtime",
        "models",
        "fixture",
        "verification_policy",
        "capabilities",
        "claim_boundary",
    )
    for field in exact_fields:
        if manifest.get(field) != spec.get(field):
            raise RuntimeError(f"embedded payload-manifest {field} mismatch")
    if manifest.get("assets") != spec["required_assets"]:
        raise RuntimeError("embedded payload-manifest asset inventory mismatch")
    if manifest.get("native_libraries") != spec["payload_native_libraries"]:
        raise RuntimeError("embedded payload-manifest native inventory mismatch")
    if set(manifest) != {
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
        "claim_boundary",
    }:
        raise RuntimeError("embedded payload-manifest field inventory mismatch")
    return manifest


def verify_archive(apk: Path, spec: dict[str, Any]) -> dict[str, Any]:
    expected_assets = {
        "assets/" + record["path"]: record for record in spec["required_assets"]
    }
    expected_native = {
        "lib/" + record["path"]: record for record in spec["required_native_libraries"]
    }
    expected_asset_names = set(expected_assets) | {PAYLOAD_MANIFEST_MEMBER}
    expected_native_names = set(expected_native)
    expected_other = set(spec["zip_policy"]["other_members"])
    with zipfile.ZipFile(apk) as bundle:
        names = bundle.namelist()
        duplicates = sorted(
            name for name, count in collections.Counter(names).items() if count != 1
        )
        if duplicates:
            raise RuntimeError(f"APK contains duplicate ZIP entries: {duplicates}")
        actual_assets = {name for name in names if name.startswith("assets/")}
        actual_native = {name for name in names if name.startswith("lib/")}
        actual_other = {
            name for name in names if not name.startswith(("assets/", "lib/"))
        }
        if actual_assets != expected_asset_names:
            raise RuntimeError(
                "APK asset set mismatch; "
                f"missing={sorted(expected_asset_names - actual_assets)}, "
                f"extra={sorted(actual_assets - expected_asset_names)}"
            )
        if actual_native != expected_native_names:
            raise RuntimeError(
                "APK native set mismatch; "
                f"missing={sorted(expected_native_names - actual_native)}, "
                f"extra={sorted(actual_native - expected_native_names)}"
            )
        if actual_other != expected_other:
            raise RuntimeError(
                "APK non-payload set mismatch; "
                f"missing={sorted(expected_other - actual_other)}, "
                f"extra={sorted(actual_other - expected_other)}"
            )
        manifest_info = bundle.getinfo(PAYLOAD_MANIFEST_MEMBER)
        if manifest_info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("embedded payload manifest is unexpectedly compressed")
        manifest = verify_embedded_manifest(bundle.read(PAYLOAD_MANIFEST_MEMBER), spec)
        verified_assets = [
            verify_member(bundle, name, record, require_stored=True)
            for name, record in sorted(expected_assets.items())
        ]
        verified_native = [
            verify_member(bundle, name, record)
            for name, record in sorted(expected_native.items())
        ]
    return {
        "status": "pass",
        "payload_manifest_sha256": spec["payload_manifest"]["sha256"],
        "manifest_gate_id": manifest["gate_id"],
        "verified_assets": verified_assets,
        "verified_native_libraries": verified_native,
        "other_members": sorted(expected_other),
    }


def run_checked(args: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"verification command failed ({result.returncode}): {args[0]}\n{output}"
        )
    return output


def verify_toolchain(
    build_tools: Path,
    java_home: Path,
    spec: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    tools: dict[str, Path] = {}
    evidence: dict[str, Any] = {}
    records = spec.get("verification_toolchain")
    if not isinstance(records, dict) or set(records) != {
        "aapt2",
        "apksigner",
        "zipalign",
    }:
        raise RuntimeError("verification-tool inventory mismatch")
    for name, record in records.items():
        validate_identity(record, "verification_toolchain." + name)
        if record.get("relative_path") != name:
            raise RuntimeError(f"verification-tool relative path mismatch: {name}")
        path = build_tools / name
        evidence[name] = verify_file(path, record)
        tools[name] = path
    java = java_home / "bin/java"
    if not java.is_file():
        raise RuntimeError(f"missing JDK launcher: {java}")
    version = run_checked([str(java), "-version"])
    if re.search(r'\bversion "17(?:\.|\")', version) is None:
        raise RuntimeError("verification JDK is not Java 17")
    evidence["java_home"] = str(java_home)
    evidence["java_version_sha256"] = hashlib.sha256(version.encode()).hexdigest()
    return tools, evidence


def parse_aapt_evidence(badging: str, permissions: str) -> dict[str, Any]:
    package = re.search(
        r"^package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'"
        r".*compileSdkVersion='([^']+)'",
        badging,
        re.MULTILINE,
    )
    if package is None:
        raise RuntimeError("aapt2 badging did not expose package identity")

    def single(pattern: str, label: str) -> str:
        match = re.search(pattern, badging, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"aapt2 badging did not expose {label}")
        return match.group(1)

    return {
        "package": package.group(1),
        "version_code": int(package.group(2)),
        "version_name": package.group(3),
        "compile_sdk": int(package.group(4)),
        "min_sdk": int(single(r"^sdkVersion:'([^']+)'", "minimum SDK")),
        "target_sdk": int(single(r"^targetSdkVersion:'([^']+)'", "target SDK")),
        "launcher": single(r"^launchable-activity: name='([^']+)'", "launcher"),
        "application_label": single(
            r"^application-label:'([^']+)'", "application label"
        ),
        "permissions": sorted(
            set(
                re.findall(
                    r"^uses-permission(?:-[^:]*)?: name='([^']+)'",
                    permissions,
                    re.MULTILINE,
                )
            )
        ),
        "native_abis": re.findall(
            r"'([^']+)'", single(r"^native-code:(.*)$", "native ABI")
        ),
        "required_native_libraries": sorted(
            re.findall(r"^uses-native-library:'([^']+)'", badging, re.MULTILINE)
        ),
        "optional_native_libraries": sorted(
            re.findall(
                r"^uses-native-library-not-required:'([^']+)'",
                badging,
                re.MULTILINE,
            )
        ),
        "optional_features": sorted(
            re.findall(
                r"^\s*uses-feature-not-required: name='([^']+)'",
                badging,
                re.MULTILINE,
            )
        ),
        "debuggable": "\napplication-debuggable\n" in "\n" + badging,
    }


def parse_tree_booleans(xmltree: str) -> dict[str, bool]:
    def single(name: str) -> bool:
        matches = re.findall(
            rf":{re.escape(name)}\(0x[0-9a-f]+\)=(true|false)$",
            xmltree,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise RuntimeError(f"binary manifest did not expose one {name}: {matches}")
        return matches[0] == "true"

    return {
        "allow_backup": single("allowBackup"),
        "supports_rtl": single("supportsRtl"),
        "extract_native_libs": single("extractNativeLibs"),
        "uses_cleartext_traffic": single("usesCleartextTraffic"),
        "launcher_exported": single("exported"),
    }


def parse_signing_evidence(output: str) -> dict[str, Any]:
    def boolean(label: str) -> bool:
        match = re.search(rf"^{re.escape(label)}: (true|false)$", output, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"apksigner output is missing {label}")
        return match.group(1) == "true"

    signers = re.search(r"^Number of signers: (\d+)$", output, re.MULTILINE)
    certificate = re.search(
        r"^Signer #1 certificate SHA-256 digest: ([0-9a-fA-F]+)$",
        output,
        re.MULTILINE,
    )
    if signers is None or certificate is None:
        raise RuntimeError("apksigner output is missing signer identity")
    return {
        "v1": boolean("Verified using v1 scheme (JAR signing)"),
        "v2": boolean("Verified using v2 scheme (APK Signature Scheme v2)"),
        "v3": boolean("Verified using v3 scheme (APK Signature Scheme v3)"),
        "v3_1": boolean("Verified using v3.1 scheme (APK Signature Scheme v3.1)"),
        "v4": boolean("Verified using v4 scheme (APK Signature Scheme v4)"),
        "source_stamp": boolean("Verified for SourceStamp"),
        "signer_count": int(signers.group(1)),
        "certificate_sha256": certificate.group(1).lower(),
    }


def verify_binary_metadata(
    apk: Path,
    tools: dict[str, Path],
    java_home: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    badging = run_checked([str(tools["aapt2"]), "dump", "badging", str(apk)])
    permissions = run_checked([str(tools["aapt2"]), "dump", "permissions", str(apk)])
    xmltree = run_checked(
        [
            str(tools["aapt2"]),
            "dump",
            "xmltree",
            str(apk),
            "--file",
            "AndroidManifest.xml",
        ]
    )
    parsed = {
        **parse_aapt_evidence(badging, permissions),
        **parse_tree_booleans(xmltree),
    }
    policy = dict(spec["apk_policy"])
    signing_policy = policy.pop("signing")
    if parsed != policy:
        raise RuntimeError(
            f"binary Android manifest mismatch; actual={parsed!r}, expected={policy!r}"
        )
    signing_output = run_checked(
        [
            str(tools["apksigner"]),
            "verify",
            "--verbose",
            "--print-certs",
            str(apk),
        ],
        env=env,
    )
    signing = parse_signing_evidence(signing_output)
    if signing != signing_policy:
        raise RuntimeError("APK signing evidence mismatch")
    alignment_output = run_checked([str(tools["zipalign"]), "-c", "-v", "4", str(apk)])
    if "Verification succesful" not in alignment_output:
        raise RuntimeError("APK alignment verification failed")
    return {
        "android_manifest": {"status": "pass", **parsed},
        "signing": {"status": "pass", **signing},
        "alignment": {
            "status": "pass",
            "alignment_bytes": 4,
            "output_sha256": hashlib.sha256(alignment_output.encode()).hexdigest(),
        },
    }


def verify_apk(
    apk: Path,
    spec: dict[str, Any],
    *,
    spec_path: Path,
    build_tools: Path,
    java_home: Path,
) -> dict[str, Any]:
    validate_spec(spec, spec_path)
    if not apk.is_file():
        raise RuntimeError(f"missing chain probe APK: {apk}")
    expected_apk = spec["apk"]
    if apk.stat().st_size != expected_apk["bytes"]:
        raise RuntimeError("whole chain APK byte count mismatch")
    actual_sha256 = sha256(apk)
    if actual_sha256 != expected_apk["sha256"]:
        raise RuntimeError("whole chain APK SHA-256 mismatch")

    # No ZIP member or embedded-manifest inspection is allowed before the
    # complete file has matched the independent whole-APK trust root above.
    archive = verify_archive(apk, spec)
    tools, toolchain = verify_toolchain(build_tools, java_home, spec)
    binary = verify_binary_metadata(apk, tools, java_home, spec)
    return {
        "schema_version": 2,
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
        "stager": verify_file(ROOT / STAGER_RELATIVE, spec["stager"]),
        "archive": archive,
        "binary": binary,
        "verification_toolchain": toolchain,
        "device_mutated": False,
        "claim_boundary": (
            "Local immutable-package integrity only. No Android execution, QNN/HTP "
            "execution, PCM quality, latency, unplugged durability, or production claim."
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
    spec = load_object(spec_path)
    report = verify_apk(
        apk,
        spec,
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
