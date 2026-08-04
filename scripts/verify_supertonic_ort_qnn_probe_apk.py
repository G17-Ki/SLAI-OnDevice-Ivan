#!/usr/bin/env python3
"""Fail-closed whole-APK verification for the Supertonic duration QNN probe."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APK = ROOT / (
    "android-app/supertonic-probe/build/outputs/apk/debug/"
    "supertonic-probe-debug.apk"
)
DEFAULT_SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "supertonic_duration_ort_qnn_probe_spec.json"
)
DEFAULT_BUILD_TOOLS = Path.home() / ".cache/lmf-demo/android-sdk/build-tools/34.0.0"
DEFAULT_JAVA_HOME = Path(
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
)
GATE_ID = "supertonic3_duration_raw_onnx_ort_qnn_sm8850_probe_v1"
PACKAGE = "ai.liquid.lfmdemo.supertonicprobe"
VERSION_NAME = "0.1.0-duration-raw-onnx-qnn-probe"
LAUNCHER = "ai.liquid.lfmdemo.supertonicprobe.ProbeActivity"
PAYLOAD_MANIFEST = "assets/supertonic_probe/payload_manifest.json"
PAYLOAD_MANIFEST_SHA256 = (
    "93bace8607936706851fe432338fb1c1a22fe38552784be3e5ed890ed52c688d"
)
SIGNING_CERT_SHA256 = (
    "9ad7b2846b7c18853b99e444d4d05323321b3a9118c13c5a8fc242e58b5859e4"
)

EXPECTED_MEMBERS = {
    "assets/supertonic_probe/fixtures/duration.f32le": (
        4,
        "7c55c050f781a693e2501d6ad4b7b69bff4d3ec504754a5206f43b69b30c3060",
    ),
    "assets/supertonic_probe/fixtures/style_dp.f32le": (
        512,
        "a7b917508369dff804c60307661ddcbcc6c80daf090cbe811dae95a3df792f8a",
    ),
    "assets/supertonic_probe/fixtures/text_ids.i64le": (
        256,
        "130c7ce81e66b732020423e4b83265a67da1cecdb4e9ec33b00d1e158f15068c",
    ),
    "assets/supertonic_probe/fixtures/text_mask.f32le": (
        128,
        "b638277a8690e175a9137feff1e43c067f9faf4e2f600caf468fb05b0403b717",
    ),
    "assets/supertonic_probe/models/duration_predictor.int8.onnx": (
        3_700_147,
        "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    ),
    "lib/arm64-v8a/libQnnHtp.so": (
        2_641_976,
        "10eb1923b34ac9a726e4a446b395d5eee2ea5388487917778aaf954e8bfb2d92",
    ),
    "lib/arm64-v8a/libQnnHtpPrepare.so": (
        83_685_856,
        "2daafb376fe6904aa325e29810480e3d76dc392991d1367ae6ea3508db2b60ac",
    ),
    "lib/arm64-v8a/libQnnHtpV81Skel.so": (
        11_271_968,
        "a3e62af7044227804bc229437107cb54425e05f43d57cc26a876dd94d54a0cfc",
    ),
    "lib/arm64-v8a/libQnnHtpV81Stub.so": (
        714_000,
        "6e9e480223b248e70c43954e89dfbd9962600b59bdc9f3069bf4780fdcf98bba",
    ),
    "lib/arm64-v8a/libQnnSystem.so": (
        2_689_096,
        "a69cfc4350c16b226bcaa19c657234820bb93e3ff6116cf3845ae77557f1f7c3",
    ),
    "lib/arm64-v8a/libonnxruntime.so": (
        20_352_056,
        "491f2fa4260cf8d379d0720795e211325b294db830ce7db2055bce9733c849a2",
    ),
    "lib/arm64-v8a/libonnxruntime4j_jni.so": (
        111_488,
        "72496f82823056fd3d2195e668a7afbb7771892419befa36c5e6d1c9bd6d204c",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(args: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, capture_output=True, text=True, env=env, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {args[0]}\n{output}")
    return output


def verify_archive(apk: Path, spec: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(apk) as bundle:
        names = bundle.namelist()
        duplicates = [
            name for name, count in collections.Counter(names).items() if count != 1
        ]
        if duplicates:
            raise RuntimeError(f"duplicate APK members: {sorted(duplicates)}")
        actual_assets = {name for name in names if name.startswith("assets/")}
        actual_native = {name for name in names if name.startswith("lib/")}
        expected_assets = {
            name for name in EXPECTED_MEMBERS if name.startswith("assets/")
        } | {PAYLOAD_MANIFEST}
        expected_native = {
            name for name in EXPECTED_MEMBERS if name.startswith("lib/")
        }
        if actual_assets != expected_assets:
            raise RuntimeError(
                "APK asset set mismatch: "
                f"missing={sorted(expected_assets - actual_assets)}, "
                f"extra={sorted(actual_assets - expected_assets)}"
            )
        if actual_native != expected_native:
            raise RuntimeError(
                "APK native set mismatch: "
                f"missing={sorted(expected_native - actual_native)}, "
                f"extra={sorted(actual_native - expected_native)}"
            )

        verified = []
        for name, (expected_bytes, expected_sha) in sorted(EXPECTED_MEMBERS.items()):
            info = bundle.getinfo(name)
            if info.file_size != expected_bytes:
                raise RuntimeError(f"APK member size mismatch: {name}")
            with bundle.open(info) as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual_sha = digest.hexdigest()
            if actual_sha != expected_sha:
                raise RuntimeError(f"APK member SHA-256 mismatch: {name}")
            verified.append(
                {
                    "path": name,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "sha256": actual_sha,
                }
            )

        raw_manifest = bundle.read(PAYLOAD_MANIFEST)
        manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
        expected_manifest = spec["payload_manifest"]
        if (
            manifest_sha != PAYLOAD_MANIFEST_SHA256
            or manifest_sha != expected_manifest["sha256"]
            or len(raw_manifest) != expected_manifest["bytes"]
        ):
            raise RuntimeError("embedded payload manifest identity mismatch")
        manifest = json.loads(raw_manifest)
        if manifest.get("gate_id") != GATE_ID:
            raise RuntimeError("embedded gate_id mismatch")
        if manifest.get("target", {}).get("package") != PACKAGE:
            raise RuntimeError("embedded package mismatch")
        runtime = manifest.get("runtime", {})
        if runtime != {
            "backend": "libQnnHtp.so",
            "disable_cpu_ep_fallback": True,
            "onnx_runtime": "1.25.0",
            "qairt": "2.42.0.251225135753_193295",
            "qnn_provider": "QNNExecutionProvider",
        }:
            raise RuntimeError("embedded runtime contract mismatch")
    return {
        "status": "pass",
        "payload_manifest_sha256": manifest_sha,
        "verified_member_count": len(verified),
        "verified_members": verified,
    }


def verify_binary_metadata(
    apk: Path,
    build_tools: Path,
    java_home: Path,
    spec: dict[str, object],
) -> dict[str, object]:
    tools: dict[str, Path] = {}
    for name, record in spec["verification_toolchain"].items():
        tool = build_tools / record["relative_path"]
        if not tool.is_file():
            raise FileNotFoundError(tool)
        if tool.stat().st_size != record["bytes"] or sha256(tool) != record["sha256"]:
            raise RuntimeError(f"verification-tool identity mismatch: {tool}")
        tools[name] = tool
    aapt2 = tools["aapt2"]
    apksigner = tools["apksigner"]
    zipalign = tools["zipalign"]

    badging = run_checked([str(aapt2), "dump", "badging", str(apk)])
    permissions = run_checked([str(aapt2), "dump", "permissions", str(apk)])
    package_match = re.search(
        r"^package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'",
        badging,
        re.MULTILINE,
    )
    if package_match is None:
        raise RuntimeError("aapt2 package identity missing")
    actual = {
        "package": package_match.group(1),
        "version_code": int(package_match.group(2)),
        "version_name": package_match.group(3),
        "launcher": re.search(
            r"^launchable-activity: name='([^']+)'", badging, re.MULTILINE
        ).group(1),
        "min_sdk": int(re.search(r"^sdkVersion:'([^']+)'", badging, re.MULTILINE).group(1)),
        "target_sdk": int(
            re.search(r"^targetSdkVersion:'([^']+)'", badging, re.MULTILINE).group(1)
        ),
        "permissions": sorted(
            set(re.findall(r"^uses-permission[^:]*: name='([^']+)'", permissions, re.MULTILINE))
        ),
        "native_abis": re.findall(
            r"'([^']+)'",
            re.search(r"^native-code:(.*)$", badging, re.MULTILINE).group(1),
        ),
        "optional_native_libraries": re.findall(
            r"^uses-native-library-not-required:'([^']+)'", badging, re.MULTILINE
        ),
    }
    policy = spec["apk_policy"]
    expected = {
        "package": policy["package"],
        "version_code": policy["version_code"],
        "version_name": policy["version_name"],
        "launcher": policy["launcher"],
        "min_sdk": policy["min_sdk"],
        "target_sdk": policy["target_sdk"],
        "permissions": policy["permissions"],
        "native_abis": policy["native_abis"],
        "optional_native_libraries": policy["optional_native_libraries"],
    }
    if actual != expected:
        raise RuntimeError(f"binary Android manifest mismatch: {actual!r}")

    env = dict(__import__("os").environ)
    env["JAVA_HOME"] = str(java_home)
    signing = run_checked(
        [str(apksigner), "verify", "--verbose", "--print-certs", str(apk)], env=env
    )
    if "Verified using v2 scheme (APK Signature Scheme v2): true" not in signing:
        raise RuntimeError("APK v2 signing verification failed")
    certificate = re.search(
        r"^Signer #1 certificate SHA-256 digest: ([0-9a-fA-F]+)$",
        signing,
        re.MULTILINE,
    )
    expected_certificate = policy["signing_certificate_sha256"]
    if (
        certificate is None
        or certificate.group(1).lower() != SIGNING_CERT_SHA256
        or certificate.group(1).lower() != expected_certificate
    ):
        raise RuntimeError("APK signer identity mismatch")
    alignment = run_checked([str(zipalign), "-c", "-v", "4", str(apk)])
    if "Verification succesful" not in alignment:
        raise RuntimeError("APK alignment verification failed")
    return {
        "status": "pass",
        **actual,
        "signing_certificate_sha256": expected_certificate,
        "zipalign": 4,
    }


def load_and_validate_spec(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("gate_id") != GATE_ID:
        raise RuntimeError("authoritative Supertonic spec identity mismatch")
    if value.get("target", {}).get("package") != PACKAGE:
        raise RuntimeError("authoritative Supertonic spec package mismatch")
    if value.get("payload_manifest", {}).get("sha256") != PAYLOAD_MANIFEST_SHA256:
        raise RuntimeError("authoritative payload-manifest hash mismatch")
    if value.get("apk_policy", {}).get("version_name") != VERSION_NAME:
        raise RuntimeError("authoritative APK version policy mismatch")
    if value.get("apk_policy", {}).get("launcher") != LAUNCHER:
        raise RuntimeError("authoritative launcher policy mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, default=DEFAULT_APK)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--build-tools", type=Path, default=DEFAULT_BUILD_TOOLS)
    parser.add_argument("--java-home", type=Path, default=DEFAULT_JAVA_HOME)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    apk = args.apk.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()
    if not apk.is_file():
        raise FileNotFoundError(apk)
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    spec = load_and_validate_spec(spec_path)
    actual_apk_sha256 = sha256(apk)
    if (
        apk.stat().st_size != spec["apk"]["bytes"]
        or actual_apk_sha256 != spec["apk"]["sha256"]
    ):
        raise RuntimeError("whole-APK size/SHA-256 does not match authoritative spec")
    report = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "package": PACKAGE,
        "apk": {
            "path": str(apk),
            "bytes": apk.stat().st_size,
            "sha256": actual_apk_sha256,
        },
        "authoritative_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": sha256(spec_path),
        },
        "archive": verify_archive(apk, spec),
        "binary_metadata": verify_binary_metadata(
            apk,
            args.build_tools.expanduser().resolve(),
            args.java_home.expanduser().resolve(),
            spec,
        ),
        "status": "pass",
        "claim_boundary": "Local APK integrity only; no phone or NPU-execution claim.",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
