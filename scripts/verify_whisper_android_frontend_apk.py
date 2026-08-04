#!/usr/bin/env python3
"""Verify the schema-2 ORT-QNN probe plus the isolated frontend overlay.

The authoritative ORT-QNN specification remains unchanged. This verifier
requires that exact base payload and permits only the separately pinned
bundled-WAV assets and JNI library added by the frontend milestone.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import struct
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_VERIFIER_PATH = ROOT / "scripts/verify_whisper_ort_qnn_probe_apk.py"
FRONTEND_STAGER_PATH = ROOT / "scripts/stage_whisper_android_frontend_assets.py"
FRONTEND_V2_STAGER_PATH = (
    ROOT / "scripts/stage_whisper_android_frontend_assets_multilang.py"
)
DEFAULT_BASE_SPEC = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_ort_qnn_probe_spec.json"
)
DEFAULT_FRONTEND_SPEC = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_android_frontend_probe_spec.json"
)
DEFAULT_FRONTEND_V2_SPEC = ROOT / (
    "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_android_frontend_multilang_v2_probe_spec.json"
)
DEFAULT_BUILD_TOOLS = Path.home() / ".cache/lmf-demo/android-sdk/build-tools/34.0.0"
DEFAULT_JAVA_HOME = Path(
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
)
FRONTEND_LIBRARY_MEMBER = "lib/arm64-v8a/libwhisper_frontend.so"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_FRONTEND_LIBRARY = {
    "path": "arm64-v8a/libwhisper_frontend.so",
    "bytes": 23048,
    "sha256": "475cad57685c04f3027c3c07a5dbfc9b67a33e8f0669fd4f2da36eefd48e9a6d",
    "elf": "aarch64",
}
EXPECTED_OTHER_MEMBERS = [
    "AndroidManifest.xml",
    "META-INF/com/android/build/gradle/app-metadata.properties",
    "classes.dex",
    "classes2.dex",
    "classes3.dex",
    "classes4.dex",
    "resources.arsc",
]
SEMANTIC_POLICIES = {
    "en": {
        "language_token_id": 50259,
        "reference_text": "What is the capital of France? Answer in one short sentence.",
        "max_wer": 0.25,
        "max_cer": 0.2,
    },
    "zh": {
        "language_token_id": 50260,
        "reference_text": "请用中文简单介绍一下北京。",
        "max_wer": 1.0,
        "max_cer": 0.2,
    },
    "ru": {
        "language_token_id": 50263,
        "reference_text": "Расскажи коротко о Москве.",
        "max_wer": 0.25,
        "max_cer": 0.1,
    },
    "ko": {
        "language_token_id": 50264,
        "reference_text": "프랑스의 수도는 어디인가요?",
        "max_wer": 0.34,
        "max_cer": 0.2,
    },
    "ja": {
        "language_token_id": 50266,
        "reference_text": "フランスの首都は、どこですか？",
        "max_wer": 1,
        "max_cer": 0.2,
    },
}


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import verification helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


base = load_module("verify_whisper_ort_qnn_probe_apk", BASE_VERIFIER_PATH)
frontend = load_module("stage_whisper_android_frontend_assets", FRONTEND_STAGER_PATH)
frontend_v2 = load_module(
    "stage_whisper_android_frontend_assets_multilang", FRONTEND_V2_STAGER_PATH
)
FRONTEND_PROFILES = {
    "v1": (frontend, FRONTEND_STAGER_PATH),
    "v2": (frontend_v2, FRONTEND_V2_STAGER_PATH),
}


def frontend_profile_contract(profile: str) -> tuple[Any, Path]:
    try:
        return FRONTEND_PROFILES[profile]
    except KeyError as error:
        raise RuntimeError(f"unsupported frontend profile: {profile}") from error


def fixture_contract(frontend_module: Any) -> dict[str, dict[str, Any]]:
    wavs: dict[str, dict[str, Any]] = {}
    goldens: dict[str, str] = {}
    for record in frontend_module.ASSETS:
        wav_match = re.fullmatch(
            r"whisper_frontend/v[12]/fixtures/question_([a-z]{2})\.wav",
            record["path"],
        )
        if wav_match:
            language = wav_match.group(1)
            wavs[language] = {
                "wav_sha256": record["sha256"],
                "wav_pcm_frames": record["pcm_frames"],
            }
            continue
        golden_match = re.fullmatch(
            r"whisper_frontend/v[12]/goldens/([a-z]{2})\.features\.u16le\.bin",
            record["path"],
        )
        if golden_match:
            goldens[golden_match.group(1)] = record["sha256"]
    if not wavs or set(wavs) != set(goldens):
        raise RuntimeError("frontend stager fixture inventory is inconsistent")
    return {
        language: {
            **wavs[language],
            "golden_tensor_sha256": goldens[language],
        }
        for language in wavs
    }


def expected_semantic_contract(
    frontend_spec: dict[str, Any],
    fixtures: dict[str, dict[str, Any]],
    frontend_profile: str,
) -> dict[str, Any] | None:
    if "semantic_decode_overlay" not in frontend_spec:
        return None
    semantic = frontend_spec["semantic_decode_overlay"]
    if not isinstance(semantic, dict):
        raise RuntimeError("semantic decode overlay is not an object")
    language = semantic.get("language_code")
    try:
        policy = SEMANTIC_POLICIES[language]
        fixture = fixtures[language]
    except (KeyError, TypeError) as error:
        raise RuntimeError("unsupported semantic language") from error
    if frontend_profile == "v1" and language not in {"en", "ja"}:
        raise RuntimeError("semantic language is unavailable in frontend v1")
    build_flag = ""
    if frontend_profile == "v2":
        build_flag = "WHISPER_FRONTEND_PROFILE=v2;"
    build_flag += "WHISPER_SEMANTIC_DECODE_ENABLED=true"
    if language != "en":
        build_flag += f";WHISPER_SEMANTIC_LANGUAGE={language}"
    language_token_id = policy["language_token_id"]
    return {
        "gate_id": f"whisper_small_bundled_{language}_semantic_candidate_v1",
        "build_flag": build_flag,
        "language_code": language,
        "language_token_id": language_token_id,
        "prompt_token_ids": [50258, language_token_id, 50359, 50363],
        "feature_source": f"bundled_question_{language}_native_frontend",
        "max_decoder_calls": 199,
        "maximum_generated_tokens": 196,
        "eot_token_id": 50257,
        "allowed_generated_token_min": 0,
        "allowed_generated_token_max": 50257,
        "suppressed_generated_token_min": 50258,
        "suppressed_generated_token_max": 51864,
        "attention_mask_mode": "cumulative_suffix_uint16",
        "reference_fixture": language,
        "reference_wav_sha256": fixture["wav_sha256"],
        "reference_text": policy["reference_text"],
        "require_eot": True,
        "max_wer": policy["max_wer"],
        "max_cer": policy["max_cer"],
        "max_semantic_compute_ms": 2500,
    }


def validate_frontend_contract(
    frontend_spec: dict[str, Any],
    base_spec: dict[str, Any],
    *,
    frontend_profile: str = "v1",
) -> None:
    frontend_module, frontend_stager_path = frontend_profile_contract(frontend_profile)
    if not isinstance(frontend_spec, dict):
        raise RuntimeError("frontend trust root is not an object")
    base.validate_spec(base_spec)
    if base_spec != load_object(DEFAULT_BASE_SPEC):
        raise RuntimeError("base spec differs from the committed trust root")

    semantic_present = "semantic_decode_overlay" in frontend_spec
    if semantic_present and not isinstance(
        frontend_spec["semantic_decode_overlay"], dict
    ):
        raise RuntimeError("semantic decode overlay is not an object")
    expected_top_level = {
        "schema_version",
        "gate_id",
        "target",
        "base_probe",
        "accepted_packages",
        "claim_boundary",
        "build_policy",
        "apk_policy",
        "stager",
        "frontend_assets",
        "frontend_manifest",
        "frontend_native_library",
        "device_output_contract",
        "zip_policy",
        "apk",
    }
    if frontend_profile == "v2":
        expected_top_level.add("frontend_profile")
    if semantic_present:
        expected_top_level.add("semantic_decode_overlay")
    if set(frontend_spec) != expected_top_level:
        raise RuntimeError("frontend trust-root field inventory drift")
    if frontend_spec["schema_version"] != 1:
        raise RuntimeError("unsupported frontend trust-root schema")
    if frontend_spec.get("frontend_profile", "v1") != frontend_profile:
        raise RuntimeError("frontend trust root has the wrong profile")
    if frontend_spec["gate_id"] != frontend_module.GATE_ID:
        raise RuntimeError("frontend trust root has the wrong gate_id")

    expected_target = {**base_spec["target"], "model": "NX809J"}
    if frontend_spec["target"] != expected_target:
        raise RuntimeError("frontend target contract drift")
    if frontend_spec["accepted_packages"] != base_spec["accepted_packages"]:
        raise RuntimeError("accepted-package trust roots drift")
    if frontend_spec["apk_policy"] != base_spec["apk_policy"]:
        raise RuntimeError("frontend/base APK policy mismatch")

    expected_claim = {
        "final_runtime_integration": False,
        "microphone_input": False,
        "probe_package_feasibility_only": True,
        "real_audio_acceptance": False,
        "real_audio_transcription": False,
    }
    if semantic_present and frontend_profile == "v1":
        expected_claim["semantic_candidate_decode"] = True
    if frontend_spec["claim_boundary"] != expected_claim:
        raise RuntimeError("frontend claim boundary drift")

    expected_build = {
        "android_ndk": "27.0.12077973",
        "frontend_compile_flags": ["-O2", "-fno-fast-math", "-ffp-contract=off"],
        "java": 17,
        "max_gradle_workers": 2,
    }
    if frontend_profile == "v2":
        semantic_language = frontend_spec.get("semantic_decode_overlay", {}).get(
            "language_code"
        )
        expected_build.update(
            {
                "frontend_profile": "v2",
                "semantic_decode_enabled": semantic_present,
            }
        )
        if semantic_present:
            expected_build["semantic_language"] = semantic_language
    if frontend_spec["build_policy"] != expected_build:
        raise RuntimeError("frontend build policy drift")

    expected_base_probe = {
        "gate_id": base.EXPECTED_GATE_ID,
        "spec": {
            "path": str(DEFAULT_BASE_SPEC.relative_to(ROOT)),
            "bytes": DEFAULT_BASE_SPEC.stat().st_size,
            "sha256": base.sha256(DEFAULT_BASE_SPEC),
        },
    }
    if frontend_spec["base_probe"] != expected_base_probe:
        raise RuntimeError("frontend trust root does not pin the base spec")

    expected_stager = {
        "path": str(frontend_stager_path.relative_to(ROOT)),
        "bytes": frontend_stager_path.stat().st_size,
        "sha256": base.sha256(frontend_stager_path),
    }
    if frontend_spec["stager"] != expected_stager:
        raise RuntimeError("frontend stager identity mismatch")
    expected_assets = sorted(
        (
            {key: record[key] for key in ("path", "bytes", "sha256")}
            for record in frontend_module.ASSETS
        ),
        key=lambda record: record["path"],
    )
    actual_assets = frontend_spec["frontend_assets"]
    if (
        not isinstance(actual_assets, list)
        or sorted(actual_assets, key=lambda record: str(record.get("path")))
        != expected_assets
    ):
        raise RuntimeError("frontend trust-root asset list differs from the stager")
    canonical_manifest = frontend_module.manifest_bytes()
    expected_manifest = {
        "path": str(frontend_module.ASSET_PREFIX / "manifest.json"),
        "bytes": len(canonical_manifest),
        "sha256": hashlib.sha256(canonical_manifest).hexdigest(),
    }
    if frontend_spec["frontend_manifest"] != expected_manifest:
        raise RuntimeError("frontend manifest trust root mismatch")
    if frontend_spec["frontend_native_library"] != EXPECTED_FRONTEND_LIBRARY:
        raise RuntimeError("frontend JNI trust root drift")

    fixtures = fixture_contract(frontend_module)
    expected_semantic = expected_semantic_contract(
        frontend_spec, fixtures, frontend_profile
    )
    if frontend_spec.get("semantic_decode_overlay") != expected_semantic:
        raise RuntimeError("semantic decode contract drift")
    executed_fixtures = fixtures
    if expected_semantic is not None and frontend_profile == "v2":
        executed_fixtures = {
            language: fixtures[language]
            for language in ("en", expected_semantic["language_code"])
        }
    expected_device_contract = {
        "evidence_schema_version": 1,
        "run_trigger": "automation_intent",
        "run_id_pattern": "[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        "report_path": (
            "/data/user/0/ai.liquid.lfmdemo.whisperprobe/files/"
            "whisper_probe/reports/latest.json"
        ),
        "completion_marker": "WHISPER_PROBE_REPORT_COMPLETE_V1",
        "feature_source": "bundled_question_en_native_frontend",
        "native_frontend_sha256": EXPECTED_FRONTEND_LIBRARY["sha256"],
        "frontend": {
            "implementation": "jni_persistent_kissfft_400_real",
            "feature_dtype": "uint16",
            "feature_shape": [1, 80, 3000],
            "feature_layout": "mel_major_frame_minor",
            "fixtures": executed_fixtures,
            "max_uint16_delta": 1,
            "max_differing_values": 240,
            "max_differing_fraction": 0.001,
        },
        "decoder": {"steps": 4, "attention_unmask_order": [199, 198, 197, 196]},
    }
    if frontend_spec["device_output_contract"] != expected_device_contract:
        raise RuntimeError("frontend device-output contract drift")
    if frontend_spec["zip_policy"] != {
        "duplicate_entries_forbidden": True,
        "other_members": EXPECTED_OTHER_MEMBERS,
    }:
        raise RuntimeError("frontend ZIP policy drift")
    apk = frontend_spec["apk"]
    if (
        not isinstance(apk, dict)
        or set(apk) != {"bytes", "sha256"}
        or not isinstance(apk["bytes"], int)
        or isinstance(apk["bytes"], bool)
        or apk["bytes"] <= 0
        or not isinstance(apk["sha256"], str)
        or SHA256_RE.fullmatch(apk["sha256"]) is None
    ):
        raise RuntimeError("frontend whole-APK identity is invalid")


def load_object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise RuntimeError(f"invalid JSON numeric constant in {path}: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def verify_frontend_elf(value: bytes, record: dict[str, Any]) -> dict[str, Any]:
    if len(value) != int(record["bytes"]):
        raise RuntimeError("frontend JNI library byte count mismatch")
    actual_sha256 = hashlib.sha256(value).hexdigest()
    if actual_sha256 != record["sha256"]:
        raise RuntimeError("frontend JNI library SHA-256 mismatch")
    if value[:6] != b"\x7fELF\x02\x01":
        raise RuntimeError("frontend JNI library is not ELF64 little-endian")
    machine = struct.unpack_from("<H", value, 18)[0]
    if machine != 183:
        raise RuntimeError("frontend JNI library is not AArch64")
    return {
        "path": FRONTEND_LIBRARY_MEMBER,
        "bytes": len(value),
        "sha256": actual_sha256,
        "elf_class": 64,
        "byte_order": "little_endian",
        "machine": "aarch64",
    }


def validate_frontend_spec(
    frontend_spec: dict[str, Any],
    base_spec: dict[str, Any],
    *,
    frontend_spec_path: Path,
    base_spec_path: Path,
    frontend_profile: str = "v1",
) -> None:
    frontend_module, frontend_stager_path = frontend_profile_contract(frontend_profile)
    if load_object(frontend_spec_path) != frontend_spec:
        raise RuntimeError("frontend trust-root object/path mismatch")
    if load_object(base_spec_path) != base_spec:
        raise RuntimeError("base spec object/path mismatch")
    validate_frontend_contract(
        frontend_spec,
        base_spec,
        frontend_profile=frontend_profile,
    )
    if frontend_spec.get("schema_version") != 1:
        raise RuntimeError("unsupported frontend trust-root schema")
    if frontend_spec.get("frontend_profile", "v1") != frontend_profile:
        raise RuntimeError("frontend trust root has the wrong profile")
    if frontend_spec.get("gate_id") != frontend_module.GATE_ID:
        raise RuntimeError("frontend trust root has the wrong gate_id")
    if frontend_spec.get("target", {}).get("package") != base.EXPECTED_PACKAGE:
        raise RuntimeError("frontend trust root has the wrong package")
    base.validate_spec(base_spec)
    base_record = frontend_spec.get("base_probe", {}).get("spec", {})
    if (
        base_record.get("bytes") != base_spec_path.stat().st_size
        or base_record.get("sha256") != base.sha256(base_spec_path)
        or frontend_spec["base_probe"].get("gate_id") != base_spec["gate_id"]
    ):
        raise RuntimeError("frontend trust root does not pin the base spec")
    if frontend_spec.get("apk_policy") != base_spec.get("apk_policy"):
        raise RuntimeError("frontend/base APK policy mismatch")
    stager_record = frontend_spec.get("stager", {})
    if stager_record.get(
        "bytes"
    ) != frontend_stager_path.stat().st_size or stager_record.get(
        "sha256"
    ) != base.sha256(frontend_stager_path):
        raise RuntimeError("frontend stager identity mismatch")
    identity_keys = ("path", "bytes", "sha256")
    expected_assets = sorted(
        (
            {key: record[key] for key in identity_keys}
            for record in frontend_module.ASSETS
        ),
        key=lambda record: record["path"],
    )
    actual_assets = frontend_spec.get("frontend_assets")
    if not isinstance(actual_assets, list):
        raise RuntimeError("frontend trust root asset list is not an array")
    actual_asset_identities = sorted(
        (
            {key: record.get(key) for key in identity_keys}
            for record in actual_assets
            if isinstance(record, dict)
        ),
        key=lambda record: str(record["path"]),
    )
    if (
        len(actual_asset_identities) != len(actual_assets)
        or actual_asset_identities != expected_assets
    ):
        raise RuntimeError("frontend trust-root asset list differs from the stager")
    manifest = frontend_spec.get("frontend_manifest", {})
    canonical_manifest = frontend_module.manifest_bytes()
    expected_manifest_path = str(frontend_module.ASSET_PREFIX / "manifest.json")
    if (
        manifest.get("path") != expected_manifest_path
        or manifest.get("bytes") != len(canonical_manifest)
        or manifest.get("sha256") != hashlib.sha256(canonical_manifest).hexdigest()
    ):
        raise RuntimeError("frontend manifest trust root mismatch")
    frontend_library = frontend_spec.get("frontend_native_library", {})
    if (
        frontend_library.get("path") != "arm64-v8a/libwhisper_frontend.so"
        or not isinstance(frontend_library.get("bytes"), int)
        or isinstance(frontend_library.get("bytes"), bool)
        or frontend_library["bytes"] <= 0
        or not isinstance(frontend_library.get("sha256"), str)
        or SHA256_RE.fullmatch(frontend_library["sha256"]) is None
        or frontend_library.get("elf") != "aarch64"
    ):
        raise RuntimeError("frontend JNI trust root is invalid")
    apk_record = frontend_spec.get("apk", {})
    if not isinstance(apk_record.get("bytes"), int) or apk_record["bytes"] <= 0:
        raise RuntimeError("frontend trust root has invalid APK bytes")
    if not isinstance(apk_record.get("sha256"), str) or len(apk_record["sha256"]) != 64:
        raise RuntimeError("frontend trust root has invalid APK SHA-256")
    if not frontend_spec_path.is_file():
        raise RuntimeError("frontend trust-root path is not a file")


def verify_archive(
    apk: Path,
    base_spec: dict[str, Any],
    frontend_spec: dict[str, Any],
    *,
    frontend_profile: str = "v1",
) -> dict[str, Any]:
    frontend_module, _ = frontend_profile_contract(frontend_profile)
    expected_base_assets = {
        "assets/" + record["path"]: record for record in base_spec["required_assets"]
    }
    expected_base_native = {
        "lib/" + record["path"]: record
        for record in base_spec["required_native_libraries"]
    }
    expected_frontend_assets = {
        "assets/" + record["path"]: record
        for record in frontend_spec["frontend_assets"]
    }
    frontend_manifest_record = frontend_spec["frontend_manifest"]
    frontend_manifest_member = "assets/" + frontend_manifest_record["path"]
    expected_asset_names = (
        set(expected_base_assets)
        | {base.PAYLOAD_MANIFEST_MEMBER}
        | set(expected_frontend_assets)
        | {frontend_manifest_member}
    )
    expected_native_names = set(expected_base_native) | {FRONTEND_LIBRARY_MEMBER}

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
                "frontend APK asset set mismatch; "
                f"missing={sorted(expected_asset_names - actual_assets)}, "
                f"extra={sorted(actual_assets - expected_asset_names)}"
            )
        if actual_native != expected_native_names:
            raise RuntimeError(
                "frontend APK native-library set mismatch; "
                f"missing={sorted(expected_native_names - actual_native)}, "
                f"extra={sorted(actual_native - expected_native_names)}"
            )
        expected_other = set(frontend_spec["zip_policy"]["other_members"])
        if actual_other != expected_other:
            raise RuntimeError(
                "frontend APK non-payload member set mismatch; "
                f"missing={sorted(expected_other - actual_other)}, "
                f"extra={sorted(actual_other - expected_other)}"
            )

        payload_manifest = bundle.read(base.PAYLOAD_MANIFEST_MEMBER)
        if (
            hashlib.sha256(payload_manifest).hexdigest()
            != base_spec["payload_manifest_sha256"]
        ):
            raise RuntimeError("base payload manifest SHA-256 mismatch")
        if json.loads(payload_manifest) != base.expected_embedded_manifest(base_spec):
            raise RuntimeError("base payload manifest no longer matches schema-2 spec")

        verified_base = [
            base.verify_member(bundle, name, record)
            for name, record in sorted(
                {**expected_base_assets, **expected_base_native}.items()
            )
        ]
        expected_manifest = frontend_module.manifest_bytes()
        actual_manifest = bundle.read(frontend_manifest_member)
        if (
            actual_manifest != expected_manifest
            or len(actual_manifest) != frontend_manifest_record["bytes"]
            or hashlib.sha256(actual_manifest).hexdigest()
            != frontend_manifest_record["sha256"]
        ):
            raise RuntimeError("frontend manifest is not canonical")
        verified_frontend = [
            base.verify_member(bundle, name, record)
            for name, record in sorted(expected_frontend_assets.items())
        ]
        frontend_library = verify_frontend_elf(
            bundle.read(FRONTEND_LIBRARY_MEMBER),
            frontend_spec["frontend_native_library"],
        )

    return {
        "status": "pass",
        "base_payload_manifest_sha256": base_spec["payload_manifest_sha256"],
        "frontend_manifest_sha256": hashlib.sha256(expected_manifest).hexdigest(),
        "base_verified_members": verified_base,
        "frontend_verified_members": verified_frontend,
        "frontend_native_library": frontend_library,
    }


def verify_apk(
    apk: Path,
    base_spec: dict[str, Any],
    frontend_spec: dict[str, Any],
    *,
    base_spec_path: Path,
    frontend_spec_path: Path,
    build_tools: Path,
    java_home: Path,
    frontend_profile: str = "v1",
) -> dict[str, Any]:
    validate_frontend_spec(
        frontend_spec,
        base_spec,
        frontend_spec_path=frontend_spec_path,
        base_spec_path=base_spec_path,
        frontend_profile=frontend_profile,
    )
    expected_apk = frontend_spec["apk"]
    if not apk.is_file():
        raise RuntimeError(f"missing frontend APK: {apk}")
    if apk.stat().st_size != expected_apk["bytes"]:
        raise RuntimeError("whole frontend APK byte count mismatch")
    actual_apk_sha256 = base.sha256(apk)
    if actual_apk_sha256 != expected_apk["sha256"]:
        raise RuntimeError("whole frontend APK SHA-256 mismatch")
    archive = verify_archive(
        apk,
        base_spec,
        frontend_spec,
        frontend_profile=frontend_profile,
    )
    tools, toolchain = base.verify_toolchain(build_tools, java_home, base_spec)
    binary = base.verify_binary_evidence(
        base.collect_binary_evidence(apk, tools, java_home), base_spec
    )
    return {
        "schema_version": 1,
        "gate_id": frontend_spec["gate_id"],
        "frontend_profile": frontend_profile,
        "base_gate_id": base.EXPECTED_GATE_ID,
        "package": base.EXPECTED_PACKAGE,
        "status": "pass",
        "apk": {
            "path": str(apk),
            "bytes": apk.stat().st_size,
            "sha256": actual_apk_sha256,
        },
        "trust_root": {
            "path": str(frontend_spec_path),
            "bytes": frontend_spec_path.stat().st_size,
            "sha256": base.sha256(frontend_spec_path),
            "base_spec_sha256": base.sha256(base_spec_path),
            "whole_apk_identity_required": True,
        },
        "archive": archive,
        "binary": binary,
        "verification_toolchain": toolchain,
        "device_mutated": False,
        "claim_boundary": (
            "Local APK identity and frontend-overlay packaging only; no Android "
            "execution, target numerical result, QNN execution, or device mutation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", required=True, type=Path)
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--frontend-spec", type=Path, default=DEFAULT_FRONTEND_SPEC)
    parser.add_argument(
        "--frontend-profile", choices=sorted(FRONTEND_PROFILES), default="v1"
    )
    parser.add_argument("--build-tools-dir", type=Path, default=DEFAULT_BUILD_TOOLS)
    parser.add_argument("--java-home", type=Path, default=DEFAULT_JAVA_HOME)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    base_spec_path = arguments.base_spec.expanduser().resolve()
    frontend_spec_path = arguments.frontend_spec.expanduser().resolve()
    base_spec = load_object(base_spec_path)
    frontend_spec = load_object(frontend_spec_path)
    report = verify_apk(
        arguments.apk.expanduser().resolve(),
        base_spec,
        frontend_spec,
        base_spec_path=base_spec_path,
        frontend_spec_path=frontend_spec_path,
        build_tools=arguments.build_tools_dir.expanduser().resolve(),
        java_home=arguments.java_home.expanduser().resolve(),
        frontend_profile=arguments.frontend_profile,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
