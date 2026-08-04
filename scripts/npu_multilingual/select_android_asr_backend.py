#!/usr/bin/env python3
"""Fail-closed multilingual Android ASR backend selector.

Production QNN selection requires all of the following:

* exact SM8850 and arm64-v8a device facts;
* visible HTP-v81 vendor support;
* a checksum-valid external QNN 2.40/SenseVoice bundle;
* a phone smoke-result for the same model/runtime/device class.

Any missing or failed gate returns the multilingual CPU fallback. Broad Auto
(English, Mandarin, Russian, French, Italian, Spanish) also remains on CPU;
SenseVoice Auto is safe only when the caller explicitly limits Auto to the
English/Mandarin scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_PATH = SCRIPT_DIR / "assets.json"
QNN_LANGUAGES = {"zh", "en", "ja", "ko", "yue"}
CPU_TARGET_LANGUAGES = {"zh", "en", "ru", "fr", "it", "es"}
RUNTIME_FILES = {
    "libQnnHtp.so",
    "libQnnHtpPrepare.so",
    "libQnnSystem.so",
    "libQnnHtpV81Stub.so",
    "libQnnHtpV81Skel.so",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_adb(adb: str, *args: str) -> str:
    result = subprocess.run(
        [adb, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.replace("\r", "")


def probe_adb(adb: str) -> dict[str, Any]:
    lines = run_adb(adb, "devices").splitlines()[1:]
    devices = [line.split()[0] for line in lines if line.strip().endswith("device")]
    if len(devices) != 1:
        return {
            "connected": False,
            "probe_status": "no_device" if not devices else "multiple_devices",
        }

    serial = devices[0]

    def prop(name: str) -> str:
        return run_adb(adb, "-s", serial, "shell", "getprop", name).strip()

    library_probe = subprocess.run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "find /vendor /system /system_ext /odm -maxdepth 8 -type f "
            "\\( -iname 'libQnnHtpV81*.so' -o -iname 'libQnnHtp.so' "
            "-o -iname 'libQnnSystem.so' \\) 2>/dev/null",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    libraries = library_probe.stdout.replace("\r", "").splitlines()
    if library_probe.returncode != 0 and not libraries:
        raise subprocess.CalledProcessError(
            library_probe.returncode,
            library_probe.args,
            output=library_probe.stdout,
            stderr=library_probe.stderr,
        )
    return {
        "connected": True,
        "probe_status": "ok",
        "soc_model": prop("ro.soc.model"),
        "board_platform": prop("ro.board.platform"),
        "hardware": prop("ro.hardware"),
        "product": prop("ro.product.name"),
        "abi": prop("ro.product.cpu.abi"),
        "abilist64": prop("ro.product.cpu.abilist64"),
        "android_release": prop("ro.build.version.release"),
        "qnn_libraries": sorted(set(libraries)),
    }


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_device(facts: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not facts.get("connected", True):
        reasons.append("phone_disconnected")
        return False, reasons
    soc = str(facts.get("soc_model", ""))
    if re.fullmatch(r"SM8850(?:-\w+)?", soc, flags=re.IGNORECASE) is None:
        reasons.append("soc_is_not_exact_sm8850")
    abi_values = f"{facts.get('abi', '')},{facts.get('abilist64', '')}"
    if "arm64-v8a" not in abi_values:
        reasons.append("arm64_v8a_missing")
    libraries = "\n".join(map(str, facts.get("qnn_libraries", [])))
    if "libQnnHtpV81" not in libraries:
        reasons.append("htp_v81_vendor_evidence_missing")
    return not reasons, reasons


def validate_bundle(bundle: dict[str, Any], assets: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    target = bundle.get("target", {})
    if target.get("soc") != "SM8850":
        reasons.append("bundle_soc_mismatch")
    if target.get("htp_arch") != "v81":
        reasons.append("bundle_htp_mismatch")
    if target.get("qnn_version") != assets["target"]["qnn_version"]:
        reasons.append("bundle_qnn_version_mismatch")

    paths = bundle.get("paths", {})
    runtime_root = Path(paths.get("qnn_runtime_root", "__missing__"))
    model_root = Path(paths.get("sensevoice_model_root", "__missing__"))
    if not runtime_root.is_dir():
        reasons.append("qnn_runtime_directory_missing")
    else:
        missing = sorted(name for name in RUNTIME_FILES if not (runtime_root / name).is_file())
        if missing:
            reasons.append("qnn_runtime_files_missing:" + ",".join(missing))

    model = model_root / "model.bin"
    tokens = model_root / "tokens.txt"
    context = assets["assets"]["sensevoice_context"]
    if not model.is_file():
        reasons.append("sensevoice_model_bin_missing")
    elif sha256(model) != context["model_bin_sha256"]:
        reasons.append("sensevoice_model_bin_hash_mismatch")
    if not tokens.is_file():
        reasons.append("sensevoice_tokens_missing")
    elif sha256(tokens) != context["tokens_sha256"]:
        reasons.append("sensevoice_tokens_hash_mismatch")
    return not reasons, reasons


def validate_smoke(smoke: dict[str, Any], assets: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if smoke.get("status") != "pass":
        reasons.append("phone_qnn_smoke_not_passed")
        return False, reasons
    if smoke.get("provider") != "qnn":
        reasons.append("smoke_provider_is_not_qnn")
    if smoke.get("soc") != "SM8850" or smoke.get("htp_arch") != "v81":
        reasons.append("smoke_target_mismatch")
    if smoke.get("qnn_version") != assets["target"]["qnn_version"]:
        reasons.append("smoke_qnn_version_mismatch")
    expected_model = assets["assets"]["sensevoice_context"]["model_bin_sha256"]
    if smoke.get("model_bin_sha256") != expected_model:
        reasons.append("smoke_model_hash_mismatch")
    if smoke.get("runtime_load") != "pass":
        reasons.append("smoke_runtime_load_not_passed")
    parity = smoke.get("transcript_parity", {})
    if parity.get("en") != "pass" or parity.get("zh") != "pass":
        reasons.append("smoke_en_zh_transcript_parity_not_passed")
    return not reasons, reasons


def choose_backend(
    language: str,
    auto_scope: str,
    device: dict[str, Any],
    bundle: dict[str, Any],
    smoke: dict[str, Any],
    assets: dict[str, Any],
) -> dict[str, Any]:
    language = language.lower()
    if language == "auto":
        qnn_language_eligible = auto_scope == "zh_en"
        resolved_request = "auto_zh_en" if qnn_language_eligible else "auto_all"
    else:
        qnn_language_eligible = language in QNN_LANGUAGES
        resolved_request = language

    device_ok, device_reasons = validate_device(device)
    bundle_ok, bundle_reasons = validate_bundle(bundle, assets)
    smoke_ok, smoke_reasons = validate_smoke(smoke, assets)
    reasons = device_reasons + bundle_reasons + smoke_reasons
    if not qnn_language_eligible:
        if language == "auto":
            reasons.insert(0, "broad_auto_requires_cpu_multilingual_asr")
        else:
            reasons.insert(0, "language_not_supported_by_sensevoice_qnn")

    use_qnn = qnn_language_eligible and device_ok and bundle_ok and smoke_ok
    selection = {
        "backend": "sensevoice_qnn_sm8850" if use_qnn else "cpu_multilingual",
        "provider": "qnn" if use_qnn else "cpu",
        "requested_language": language,
        "resolved_request": resolved_request,
        "auto_scope": auto_scope,
        "qnn_activated": use_qnn,
        "fallback_required": not use_qnn,
        "reason_codes": [] if use_qnn else reasons,
        "contract": {
            "qnn": "fixed 10-second SenseVoice context; pad short input and segment input longer than 10 seconds",
            "cpu": "multilingual CPU backend must support zh/en/ru/fr/it/es and preserve the requested or detected language",
        },
    }
    if language != "auto" and language not in QNN_LANGUAGES | CPU_TARGET_LANGUAGES:
        selection["reason_codes"].insert(0, "language_outside_declared_milestone")
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="auto")
    parser.add_argument("--auto-scope", choices=("all", "zh_en"), default="all")
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--device-facts", type=Path)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        assets = load_json(ASSETS_PATH)
        bundle = load_json(args.bundle_manifest)
        smoke = load_json(args.smoke_result)
        device = load_json(args.device_facts) if args.device_facts else probe_adb(args.adb)
        result = {
            "schema_version": 1,
            "device": device,
            "selection": choose_backend(
                args.language, args.auto_scope, device, bundle, smoke, assets
            ),
        }
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
