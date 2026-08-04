#!/usr/bin/env python3
"""Fail-closed verification of the isolated Whisper ORT-QNN probe APK.

The committed probe specification is the trust root. The APK-embedded payload
manifest is evidence to compare against that specification, never authority.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any, BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_ort_qnn_probe_spec.json"
)
DEFAULT_BUILD_TOOLS = (
    Path.home() / ".cache/lmf-demo/android-sdk/build-tools/34.0.0"
)
DEFAULT_JAVA_HOME = Path(
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
)
EXPECTED_GATE_ID = "nubia_whisper_small_ort_qnn_epcontext_sm8850_probe_v1"
EXPECTED_PACKAGE = "ai.liquid.lfmdemo.whisperprobe"
EXPECTED_FEATURE_FIXTURE_SHA256 = (
    "29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1"
)
PAYLOAD_MANIFEST_MEMBER = "assets/whisper_probe/payload_manifest.json"


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def verify_file(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing pinned verification tool: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise RuntimeError(f"verification tool size mismatch: {path}")
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"verification tool SHA-256 mismatch: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def expected_embedded_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(spec["embedded_manifest"])
    manifest["assets"] = copy.deepcopy(spec["required_assets"])
    manifest["native_libraries"] = copy.deepcopy(spec["required_native_libraries"])
    return manifest


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 2:
        raise RuntimeError("unsupported authoritative probe spec schema")
    if spec.get("gate_id") != EXPECTED_GATE_ID:
        raise RuntimeError("authoritative spec has the wrong gate_id")
    if spec.get("target", {}).get("package") != EXPECTED_PACKAGE:
        raise RuntimeError("authoritative spec has the wrong package")
    fixture = spec.get("embedded_manifest", {}).get("feature_fixture", {})
    if fixture.get("sha256") != EXPECTED_FEATURE_FIXTURE_SHA256:
        raise RuntimeError("authoritative spec has the wrong feature fixture")
    if spec.get("payload_manifest_sha256") is None:
        raise RuntimeError("authoritative spec does not pin payload manifest SHA-256")

    for name, record in spec.get("verification_toolchain", {}).items():
        tool_digest = record.get("sha256", "")
        if re.fullmatch(r"[0-9a-f]{64}", tool_digest) is None:
            raise RuntimeError(
                f"authoritative spec has an invalid {name} verification-tool SHA-256"
            )

    asset_paths = [item["path"] for item in spec.get("required_assets", [])]
    native_paths = [item["path"] for item in spec.get("required_native_libraries", [])]
    if not asset_paths or len(asset_paths) != len(set(asset_paths)):
        raise RuntimeError("authoritative spec asset list is empty or duplicated")
    if not native_paths or len(native_paths) != len(set(native_paths)):
        raise RuntimeError("authoritative spec native-library list is empty or duplicated")
    fixture_assets = [
        item
        for item in spec["required_assets"]
        if item["path"] == fixture.get("path")
    ]
    if fixture_assets != [fixture]:
        raise RuntimeError("authoritative fixture record is not an exact required asset")


def verify_member(
    bundle: zipfile.ZipFile,
    member: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    try:
        info = bundle.getinfo(member)
    except KeyError as error:
        raise RuntimeError(f"APK is missing required member: {member}") from error
    if info.is_dir() or info.file_size != int(record["bytes"]):
        raise RuntimeError(f"APK member size/type mismatch: {member}")
    with bundle.open(info) as stream:
        actual_sha256 = sha256_stream(stream)
    if actual_sha256 != record["sha256"]:
        raise RuntimeError(f"APK member SHA-256 mismatch: {member}")
    return {
        "path": member,
        "bytes": info.file_size,
        "sha256": actual_sha256,
        "compressed_bytes": info.compress_size,
    }


def verify_archive(apk: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not apk.is_file():
        raise RuntimeError(f"missing APK: {apk}")
    expected_assets = {
        "assets/" + record["path"]: record for record in spec["required_assets"]
    }
    expected_native = {
        "lib/" + record["path"]: record
        for record in spec["required_native_libraries"]
    }
    expected_asset_names = set(expected_assets) | {PAYLOAD_MANIFEST_MEMBER}
    expected_native_names = set(expected_native)

    with zipfile.ZipFile(apk) as bundle:
        names = bundle.namelist()
        duplicates = sorted(
            name for name, count in collections.Counter(names).items() if count != 1
        )
        if duplicates:
            raise RuntimeError(f"APK contains duplicate ZIP entries: {duplicates}")

        actual_asset_names = {name for name in names if name.startswith("assets/")}
        actual_native_names = {name for name in names if name.startswith("lib/")}
        if actual_asset_names != expected_asset_names:
            missing = sorted(expected_asset_names - actual_asset_names)
            extra = sorted(actual_asset_names - expected_asset_names)
            raise RuntimeError(f"APK asset set mismatch; missing={missing}, extra={extra}")
        if actual_native_names != expected_native_names:
            missing = sorted(expected_native_names - actual_native_names)
            extra = sorted(actual_native_names - expected_native_names)
            raise RuntimeError(
                f"APK native-library set mismatch; missing={missing}, extra={extra}"
            )

        raw_manifest = bundle.read(PAYLOAD_MANIFEST_MEMBER)
        manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        if manifest_sha256 != spec["payload_manifest_sha256"]:
            raise RuntimeError("embedded payload manifest SHA-256 mismatch")
        try:
            embedded_manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise RuntimeError("embedded payload manifest is invalid JSON") from error
        expected_manifest = expected_embedded_manifest(spec)
        if embedded_manifest != expected_manifest:
            raise RuntimeError(
                "embedded payload manifest does not exactly match authoritative spec"
            )

        verified = [
            verify_member(bundle, name, record)
            for name, record in sorted({**expected_assets, **expected_native}.items())
        ]
    return {
        "status": "pass",
        "payload_manifest_sha256": manifest_sha256,
        "verified_member_count": len(verified),
        "verified_members": verified,
    }


def parse_aapt_evidence(badging: str, permissions: str) -> dict[str, Any]:
    package_match = re.search(
        r"^package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'"
        r".*compileSdkVersion='([^']+)'",
        badging,
        re.MULTILINE,
    )
    if package_match is None:
        raise RuntimeError("aapt2 badging did not expose package identity")

    def single(pattern: str, label: str) -> str:
        match = re.search(pattern, badging, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"aapt2 badging did not expose {label}")
        return match.group(1)

    uses_permissions = sorted(
        set(
            re.findall(
                r"^uses-permission(?:-[^:]*)?: name='([^']+)'",
                permissions,
                re.MULTILINE,
            )
        )
    )
    return {
        "package": package_match.group(1),
        "version_code": int(package_match.group(2)),
        "version_name": package_match.group(3),
        "compile_sdk": int(package_match.group(4)),
        "min_sdk": int(single(r"^sdkVersion:'([^']+)'", "minimum SDK")),
        "target_sdk": int(single(r"^targetSdkVersion:'([^']+)'", "target SDK")),
        "launcher": single(
            r"^launchable-activity: name='([^']+)'", "launchable activity"
        ),
        "permissions": uses_permissions,
        "native_abis": re.findall(r"'([^']+)'", single(r"^native-code:(.*)$", "native ABI")),
        "required_native_libraries": sorted(
            re.findall(r"^uses-native-library:'([^']+)'", badging, re.MULTILINE)
        ),
        "optional_native_libraries": sorted(
            re.findall(
                r"^uses-native-library-not-required:'([^']+)'", badging, re.MULTILINE
            )
        ),
        "optional_features": sorted(
            re.findall(
                r"^\s*uses-feature-not-required: name='([^']+)'",
                badging,
                re.MULTILINE,
            )
        ),
    }


def verify_android_manifest(
    parsed: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    policy = spec["apk_policy"]
    exact_fields = {
        "package": policy["package"],
        "version_code": policy["version_code"],
        "version_name": policy["version_name"],
        "compile_sdk": policy["compile_sdk"],
        "min_sdk": policy["min_sdk"],
        "target_sdk": policy["target_sdk"],
        "launcher": policy["launcher"],
        "permissions": policy["permissions"],
        "native_abis": policy["native_abis"],
        "required_native_libraries": policy["required_native_libraries"],
        "optional_native_libraries": policy["optional_native_libraries"],
    }
    for key, expected in exact_fields.items():
        if parsed.get(key) != expected:
            raise RuntimeError(
                f"binary AndroidManifest {key} mismatch; "
                f"actual={parsed.get(key)!r}, expected={expected!r}"
            )
    required_optional_features = set(policy["required_optional_features"])
    if not required_optional_features.issubset(parsed["optional_features"]):
        raise RuntimeError("binary AndroidManifest is missing required optional features")
    return {"status": "pass", **parsed}


def parse_signing_evidence(output: str) -> dict[str, Any]:
    def boolean(label: str) -> bool:
        match = re.search(rf"^{re.escape(label)}: (true|false)$", output, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"apksigner output is missing {label}")
        return match.group(1) == "true"

    signer_count = re.search(r"^Number of signers: (\d+)$", output, re.MULTILINE)
    certificate = re.search(
        r"^Signer #1 certificate SHA-256 digest: ([0-9a-fA-F]+)$",
        output,
        re.MULTILINE,
    )
    if signer_count is None or certificate is None:
        raise RuntimeError("apksigner output is missing signer identity")
    return {
        "v1": boolean("Verified using v1 scheme (JAR signing)"),
        "v2": boolean("Verified using v2 scheme (APK Signature Scheme v2)"),
        "v3": boolean("Verified using v3 scheme (APK Signature Scheme v3)"),
        "v3_1": boolean("Verified using v3.1 scheme (APK Signature Scheme v3.1)"),
        "v4": boolean("Verified using v4 scheme (APK Signature Scheme v4)"),
        "source_stamp": boolean("Verified for SourceStamp"),
        "signer_count": int(signer_count.group(1)),
        "certificate_sha256": certificate.group(1).lower(),
    }


def verify_signing(parsed: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    expected = spec["apk_policy"]["signing"]
    if parsed != expected:
        raise RuntimeError(
            f"APK signing evidence mismatch; actual={parsed!r}, expected={expected!r}"
        )
    return {"status": "pass", **parsed}


def verify_alignment(output: str) -> dict[str, Any]:
    if "Verification succesful" not in output:
        raise RuntimeError("zipalign output did not report successful verification")
    return {
        "status": "pass",
        "alignment_bytes": 4,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def verify_toolchain(
    build_tools: Path, java_home: Path, spec: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    tools: dict[str, Path] = {}
    evidence: dict[str, Any] = {}
    for name, record in spec["verification_toolchain"].items():
        path = build_tools / record["relative_path"]
        evidence[name] = verify_file(path, record)
        tools[name] = path
    java = java_home / "bin/java"
    if not java.is_file():
        raise RuntimeError(f"missing pinned JDK 17 java launcher: {java}")
    evidence["java_home"] = str(java_home)
    return tools, evidence


def run_checked(
    args: list[str], env: dict[str, str] | None = None
) -> str:
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


def collect_binary_evidence(
    apk: Path,
    tools: dict[str, Path],
    java_home: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    return {
        "badging": run_checked([str(tools["aapt2"]), "dump", "badging", str(apk)]),
        "permissions": run_checked(
            [str(tools["aapt2"]), "dump", "permissions", str(apk)]
        ),
        "signing": run_checked(
            [str(tools["apksigner"]), "verify", "--verbose", "--print-certs", str(apk)],
            env=env,
        ),
        "alignment": run_checked(
            [str(tools["zipalign"]), "-c", "-v", "4", str(apk)]
        ),
    }


def verify_binary_evidence(
    evidence: dict[str, str], spec: dict[str, Any]
) -> dict[str, Any]:
    manifest = verify_android_manifest(
        parse_aapt_evidence(evidence["badging"], evidence["permissions"]), spec
    )
    signing = verify_signing(parse_signing_evidence(evidence["signing"]), spec)
    alignment = verify_alignment(evidence["alignment"])
    return {
        "android_manifest": manifest,
        "signing": signing,
        "alignment": alignment,
    }


def verify_apk(
    apk: Path,
    spec: dict[str, Any],
    build_tools: Path,
    java_home: Path,
) -> dict[str, Any]:
    validate_spec(spec)
    archive = verify_archive(apk, spec)
    tools, toolchain = verify_toolchain(build_tools, java_home, spec)
    binary = verify_binary_evidence(
        collect_binary_evidence(apk, tools, java_home), spec
    )
    return {
        "schema_version": 2,
        "gate_id": EXPECTED_GATE_ID,
        "package": EXPECTED_PACKAGE,
        "status": "pass",
        "apk": {
            "path": str(apk),
            "bytes": apk.stat().st_size,
            "sha256": sha256(apk),
        },
        "archive": archive,
        "binary": binary,
        "verification_toolchain": toolchain,
        "device_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--build-tools-dir", type=Path, default=DEFAULT_BUILD_TOOLS)
    parser.add_argument("--java-home", type=Path, default=DEFAULT_JAVA_HOME)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report = verify_apk(
        args.apk.expanduser().resolve(),
        spec,
        args.build_tools_dir.expanduser().resolve(),
        args.java_home.expanduser().resolve(),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
