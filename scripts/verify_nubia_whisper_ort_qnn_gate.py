#!/usr/bin/env python3
"""Fail-closed gate for Whisper Small through ORT QNN EPContext on Nubia.

This route is intentionally separate from the raw QNN/Voice AI gate.  The
verifier is host-only and read-only except for an explicitly requested JSON
report.  It never extracts an archive, loads a model, invokes ADB, or uses the
network.  Readiness requires exact release archives, the QNN-enabled ORT AAR,
an exact-build QAIRT inventory, a bound Android runner, and app-UID execution
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SPEC = (
    REPO_ROOT
    / "reports/android_phone/streaming_response/whisper_v81/"
    "nubia_whisper_ort_qnn_gate_spec.json"
)
GATE_ORDER = {
    "spec": 0,
    "model_release": 1,
    "metadata": 2,
    "ep_context": 3,
    "ort": 4,
    "voice_assets": 5,
    "runtime": 6,
    "runner": 7,
    "app_uid": 8,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ELF_MACHINE = {"aarch64": 183, "hexagon": 164}


class GateInputError(ValueError):
    """Raised only for malformed verifier inputs or gate declarations."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateInputError(f"invalid JSON in {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise GateInputError(f"expected a JSON object in {path.name}")
    return value, sha256_bytes(raw)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def add_blocker(
    blockers: list[dict[str, Any]],
    gate: str,
    code: str,
    detail: str,
    *,
    expected: Any | None = None,
    actual: Any | None = None,
) -> None:
    item: dict[str, Any] = {"code": code, "detail": detail, "gate": gate}
    if expected is not None:
        item["expected"] = expected
    if actual is not None:
        item["actual"] = actual
    blockers.append(item)


def sorted_blockers(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        blockers,
        key=lambda item: (
            GATE_ORDER.get(str(item.get("gate")), 99),
            str(item.get("code")),
            str(item.get("detail")),
            canonical_json_bytes(item).decode("utf-8"),
        ),
    )


def safe_zip_name(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return False
    pure = PurePosixPath(name)
    if any(part in ("", ".", "..") for part in pure.parts):
        return False
    return not (pure.parts and ":" in pure.parts[0])


def validate_release_spec(release: Any, label: str) -> None:
    if not isinstance(release, dict):
        raise GateInputError(f"{label} release must be an object")
    if not isinstance(release.get("archive_filename"), str):
        raise GateInputError(f"{label} release has no archive filename")
    if not isinstance(release.get("archive_bytes"), int):
        raise GateInputError(f"{label} release has an invalid byte count")
    if not valid_sha256(release.get("archive_sha256")):
        raise GateInputError(f"{label} release has an invalid SHA-256")
    members = release.get("members")
    if not isinstance(members, list) or not members:
        raise GateInputError(f"{label} release has no exact member list")
    seen: set[str] = set()
    seen_file_basenames: set[str] = set()
    for member in members:
        if not isinstance(member, dict):
            raise GateInputError(f"{label} release contains a non-object member")
        name = member.get("name")
        if not isinstance(name, str) or not safe_zip_name(name):
            raise GateInputError(f"{label} release has an unsafe member path")
        if name in seen:
            raise GateInputError(f"{label} release repeats member {name}")
        seen.add(name)
        member_type = member.get("type")
        if member_type not in ("file", "directory"):
            raise GateInputError(f"{label} release has bad type for {name}")
        if not isinstance(member.get("bytes"), int) or member["bytes"] < 0:
            raise GateInputError(f"{label} release has bad size for {name}")
        if member_type == "file" and not valid_sha256(member.get("sha256")):
            raise GateInputError(f"{label} release has bad hash for {name}")
        if member_type == "file":
            basename = PurePosixPath(name).name
            if basename in seen_file_basenames:
                raise GateInputError(
                    f"{label} release repeats file basename {basename}"
                )
            seen_file_basenames.add(basename)


def member_by_basename(release: dict[str, Any], basename: str) -> dict[str, Any]:
    matches = [
        item
        for item in release["members"]
        if PurePosixPath(item["name"]).name == basename
    ]
    if len(matches) != 1:
        raise GateInputError(
            f"expected exactly one {basename} member, found {len(matches)}"
        )
    return matches[0]


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise GateInputError("ORT-QNN gate spec schema_version must be 1")
    for key in (
        "gate_id",
        "model_release",
        "metadata",
        "ep_context",
        "ort",
        "voice_assets",
        "runtime",
        "runner",
        "app_uid",
    ):
        if key not in spec:
            raise GateInputError(f"ORT-QNN gate spec is missing {key}")
        if key != "gate_id" and not isinstance(spec[key], dict):
            raise GateInputError(f"ORT-QNN gate spec {key} must be an object")
    if not isinstance(spec["gate_id"], str) or not spec["gate_id"]:
        raise GateInputError("ORT-QNN gate_id must be a non-empty string")
    validate_release_spec(spec["model_release"], "model")
    validate_release_spec(spec["ort"]["release"], "ORT AAR")
    validate_release_spec(spec["voice_assets"]["release"], "Voice AI")

    metadata_member = member_by_basename(spec["model_release"], "metadata.json")
    metadata = spec["metadata"]
    if metadata_member.get("sha256") != metadata.get("sha256"):
        raise GateInputError("metadata pin differs from the model archive member")
    if not valid_sha256(metadata.get("tensor_contract_sha256")):
        raise GateInputError("metadata tensor-contract SHA-256 is invalid")
    if not isinstance(metadata.get("layer_count"), int) or metadata["layer_count"] < 1:
        raise GateInputError("metadata layer_count is invalid")

    wrappers = spec["ep_context"].get("wrappers")
    if not isinstance(wrappers, list) or not wrappers:
        raise GateInputError("EPContext wrappers must be a non-empty list")
    for wrapper in wrappers:
        if not isinstance(wrapper, dict):
            raise GateInputError("EPContext wrapper declaration must be an object")
        onnx_member = member_by_basename(
            spec["model_release"], wrapper.get("onnx_filename", "")
        )
        context_member = member_by_basename(
            spec["model_release"], wrapper.get("context_filename", "")
        )
        if onnx_member.get("sha256") != wrapper.get("onnx_sha256"):
            raise GateInputError("EPContext ONNX pin differs from the archive member")
        if context_member.get("sha256") != wrapper.get("context_sha256"):
            raise GateInputError("EPContext binary pin differs from the archive member")
        reference = wrapper.get("external_reference")
        if reference != f"./{wrapper['context_filename']}":
            raise GateInputError("EPContext external reference must be same-directory")

    ort_native = member_by_basename(
        spec["ort"]["release"], "libonnxruntime.so"
    )
    if ort_native.get("sha256") != spec["ort"].get("native_library_sha256"):
        raise GateInputError("ORT native-library pin differs from the AAR member")
    config_member = member_by_basename(
        spec["voice_assets"]["release"], "config.json"
    )
    vocab_member = member_by_basename(
        spec["voice_assets"]["release"], "vocab.bin"
    )
    if config_member.get("sha256") != spec["voice_assets"].get("config_sha256"):
        raise GateInputError("Voice AI config pin differs from its archive member")
    if vocab_member.get("sha256") != spec["voice_assets"].get("vocab_sha256"):
        raise GateInputError("Voice AI vocab pin differs from its archive member")

    runtime = spec["runtime"]
    if runtime.get("exact_qairt_build") != nested_required_value(
        metadata.get("required_values"), "tool_versions.qairt"
    ):
        raise GateInputError("runtime build pin differs from release metadata")
    source = runtime.get("source_archive")
    if not isinstance(source, dict) or not valid_sha256(source.get("sha256")):
        raise GateInputError("runtime source archive is not exactly hash-pinned")
    if not isinstance(source.get("filename"), str) or not source["filename"]:
        raise GateInputError("runtime source archive filename is invalid")
    if not isinstance(source.get("bytes"), int) or source["bytes"] < 1:
        raise GateInputError("runtime source archive byte pin is invalid")
    required_libraries = runtime.get("required_libraries")
    if not isinstance(required_libraries, list) or not required_libraries:
        raise GateInputError("runtime required_libraries must be non-empty")
    for item in required_libraries:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise GateInputError("runtime library declaration is invalid")
        if item.get("elf_machine") not in ELF_MACHINE:
            raise GateInputError(f"runtime ELF machine is invalid for {item['name']}")
        if not isinstance(item.get("bytes"), int) or item["bytes"] < 1:
            raise GateInputError(f"runtime byte pin is invalid for {item['name']}")
        if not valid_sha256(item.get("sha256")):
            raise GateInputError(f"runtime hash pin is invalid for {item['name']}")
    for item in runtime.get("required_supporting_files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise GateInputError("runtime supporting-file declaration is invalid")
        if not isinstance(item.get("bytes"), int) or item["bytes"] < 1:
            raise GateInputError(
                f"runtime supporting-file byte pin is invalid for {item.get('path')}"
            )
        if not valid_sha256(item.get("sha256")):
            raise GateInputError(
                f"runtime supporting-file hash is invalid for {item.get('path')}"
            )


def nested_required_value(requirements: Any, path: str) -> Any:
    if not isinstance(requirements, list):
        return None
    matches = [
        item.get("equals")
        for item in requirements
        if isinstance(item, dict) and item.get("path") == path
    ]
    return matches[0] if len(matches) == 1 else None


def hash_bounded_stream(
    stream: BinaryIO, expected_bytes: int, *, capture: bool
) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    total = 0
    captured = bytearray() if capture else None
    while True:
        block = stream.read(min(1024 * 1024, expected_bytes - total + 1))
        if not block:
            break
        total += len(block)
        if total > expected_bytes:
            raise GateInputError("ZIP stream exceeded its pinned size")
        digest.update(block)
        if captured is not None:
            captured.extend(block)
    if total != expected_bytes:
        raise GateInputError(
            f"ZIP stream size mismatch: expected {expected_bytes}, got {total}"
        )
    return digest.hexdigest(), bytes(captured) if captured is not None else None


def verify_zip_release(
    release: dict[str, Any],
    archive_path: Path,
    *,
    gate: str,
    capture_basenames: set[str],
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    report: dict[str, Any] = {"filename": archive_path.name, "status": "blocked"}
    captured: dict[str, bytes] = {}
    if not archive_path.exists():
        add_blocker(
            blockers,
            gate,
            f"{gate}_archive_missing",
            "the pinned archive was not found",
            expected=release["archive_filename"],
            actual=archive_path.name,
        )
        return report, captured
    if archive_path.is_symlink() or not archive_path.is_file():
        add_blocker(
            blockers,
            gate,
            f"{gate}_archive_unsafe",
            "archive must be a regular non-symlink file",
        )
        return report, captured
    actual_bytes = archive_path.stat().st_size
    report["bytes"] = actual_bytes
    if actual_bytes != release["archive_bytes"]:
        add_blocker(
            blockers,
            gate,
            f"{gate}_archive_size_mismatch",
            "archive byte count differs from the exact pin",
            expected=release["archive_bytes"],
            actual=actual_bytes,
        )
        return report, captured
    actual_sha = sha256_file(archive_path)
    report["sha256"] = actual_sha
    if actual_sha != release["archive_sha256"]:
        add_blocker(
            blockers,
            gate,
            f"{gate}_archive_sha256_mismatch",
            "archive SHA-256 differs from the exact pin",
            expected=release["archive_sha256"],
            actual=actual_sha,
        )
        return report, captured

    expected_members = release["members"]
    expected_by_name = {item["name"]: item for item in expected_members}
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            infos = bundle.infolist()
            names = [item.filename for item in infos]
            expected_names = [item["name"] for item in expected_members]
            report["member_names"] = names
            if len(names) != len(set(names)):
                add_blocker(
                    blockers,
                    gate,
                    f"{gate}_archive_duplicate_members",
                    "archive contains duplicate member names",
                )
            if names != expected_names:
                add_blocker(
                    blockers,
                    gate,
                    f"{gate}_archive_member_list_mismatch",
                    "ordered archive member list differs from the exact pin",
                    expected=expected_names,
                    actual=names,
                )
            for info in infos:
                if not safe_zip_name(info.filename):
                    add_blocker(
                        blockers,
                        gate,
                        f"{gate}_archive_member_path_unsafe",
                        f"unsafe archive member path: {info.filename}",
                    )
                if info.flag_bits & 0x1:
                    add_blocker(
                        blockers,
                        gate,
                        f"{gate}_archive_member_encrypted",
                        f"encrypted member is forbidden: {info.filename}",
                    )
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                    add_blocker(
                        blockers,
                        gate,
                        f"{gate}_archive_member_symlink",
                        f"symlink member is forbidden: {info.filename}",
                    )
                expected = expected_by_name.get(info.filename)
                if expected is None:
                    continue
                actual_type = "directory" if info.is_dir() else "file"
                if actual_type != expected["type"]:
                    add_blocker(
                        blockers,
                        gate,
                        f"{gate}_archive_member_type_mismatch",
                        f"member type differs for {info.filename}",
                        expected=expected["type"],
                        actual=actual_type,
                    )
                if info.file_size != expected["bytes"]:
                    add_blocker(
                        blockers,
                        gate,
                        f"{gate}_archive_member_size_mismatch",
                        f"member byte count differs for {info.filename}",
                        expected=expected["bytes"],
                        actual=info.file_size,
                    )
            if any(item["gate"] == gate for item in blockers):
                return report, captured

            verified: dict[str, dict[str, Any]] = {}
            for info in infos:
                expected = expected_by_name[info.filename]
                if expected["type"] == "directory":
                    continue
                basename = PurePosixPath(info.filename).name
                with bundle.open(info, "r") as stream:
                    actual_member_sha, value = hash_bounded_stream(
                        stream,
                        expected["bytes"],
                        capture=basename in capture_basenames,
                    )
                verified[basename] = {
                    "bytes": info.file_size,
                    "sha256": actual_member_sha,
                }
                if actual_member_sha != expected["sha256"]:
                    add_blocker(
                        blockers,
                        gate,
                        f"{gate}_component_sha256_mismatch",
                        f"streamed member hash differs for {basename}",
                        expected=expected["sha256"],
                        actual=actual_member_sha,
                    )
                if value is not None:
                    captured[basename] = value
            report["verified_members"] = verified
            if not any(item["gate"] == gate for item in blockers):
                report["status"] = "pass"
    except (OSError, GateInputError, RuntimeError, zipfile.BadZipFile) as error:
        add_blocker(
            blockers,
            gate,
            f"{gate}_archive_validation_failed",
            str(error),
        )
    return report, captured


def nested_value(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def normalize_tensor_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateInputError("tensor descriptor must be an object")
    result = {
        key: value[key]
        for key in ("dtype", "shape", "quantization_parameters")
        if key in value
    }
    if "dtype" not in result or "shape" not in result:
        raise GateInputError("tensor descriptor lacks dtype or shape")
    return result


def extract_tensor_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    model_files = metadata.get("model_files")
    if not isinstance(model_files, dict):
        raise GateInputError("metadata is missing model_files")
    contract: dict[str, Any] = {}
    for filename, declaration in sorted(model_files.items()):
        if filename not in ("encoder.onnx", "decoder.onnx"):
            raise GateInputError(f"unexpected metadata model file: {filename}")
        if not isinstance(declaration, dict):
            raise GateInputError(f"bad tensor declaration for {filename}")
        component: dict[str, Any] = {}
        for side in ("inputs", "outputs"):
            tensors = declaration.get(side)
            if not isinstance(tensors, dict):
                raise GateInputError(f"{filename} is missing {side}")
            component[side] = {
                name: normalize_tensor_descriptor(descriptor)
                for name, descriptor in sorted(tensors.items())
            }
        contract[filename[:-5]] = component
    if set(contract) != {"encoder", "decoder"}:
        raise GateInputError("metadata must declare encoder and decoder ONNX models")
    return contract


def expand_names(values: Any, layer_count: int) -> list[str]:
    if not isinstance(values, list):
        raise GateInputError("tensor names must be a list")
    expanded: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise GateInputError("tensor names must be non-empty strings")
        if "{layer}" in value:
            if value.count("{layer}") != 1:
                raise GateInputError(f"bad tensor template: {value}")
            expanded.extend(
                value.replace("{layer}", str(index))
                for index in range(layer_count)
            )
        elif "{" in value or "}" in value:
            raise GateInputError(f"unsupported tensor template: {value}")
        else:
            expanded.append(value)
    if len(expanded) != len(set(expanded)):
        raise GateInputError("tensor names expand to duplicates")
    return sorted(expanded)


def declared_tensor_names(spec: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    layer_count = spec["metadata"]["layer_count"]
    declaration = spec["metadata"].get("tensor_names")
    if not isinstance(declaration, dict):
        raise GateInputError("metadata tensor_names must be an object")
    return {
        component: {
            side: expand_names(declaration[component][side], layer_count)
            for side in ("inputs", "outputs")
        }
        for component in ("encoder", "decoder")
    }


def contract_tensor_names(
    contract: dict[str, Any]
) -> dict[str, dict[str, list[str]]]:
    return {
        component: {
            side: sorted(contract[component][side])
            for side in ("inputs", "outputs")
        }
        for component in ("encoder", "decoder")
    }


def contract_dtype_counts(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for component in ("encoder", "decoder"):
        result[component] = {}
        for side in ("inputs", "outputs"):
            counts: dict[str, int] = {}
            for descriptor in contract[component][side].values():
                dtype = str(descriptor.get("dtype"))
                counts[dtype] = counts.get(dtype, 0) + 1
            result[component][side] = dict(sorted(counts.items()))
    return result


def contract_name_mismatches(
    expected: dict[str, dict[str, list[str]]],
    actual: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for component in ("encoder", "decoder"):
        for side in ("inputs", "outputs"):
            expected_set = set(expected[component][side])
            actual_set = set(actual[component][side])
            if expected_set != actual_set:
                result.append(
                    {
                        "component": component,
                        "missing": sorted(expected_set - actual_set),
                        "side": side,
                        "unexpected": sorted(actual_set - expected_set),
                    }
                )
    return result


def metadata_check(
    spec: dict[str, Any],
    raw: bytes | None,
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {"status": "not_checked"}
    internal: dict[str, Any] = {}
    if raw is None:
        report["reason"] = "model_release_gate_not_passed"
        return report, internal
    report["status"] = "blocked"
    try:
        metadata = json.loads(raw)
        if not isinstance(metadata, dict):
            raise GateInputError("metadata.json must be an object")
        for requirement in spec["metadata"]["required_values"]:
            path = requirement["path"]
            expected = requirement["equals"]
            try:
                actual = nested_value(metadata, path)
            except KeyError:
                add_blocker(
                    blockers,
                    "metadata",
                    "metadata_required_value_missing",
                    f"metadata is missing {path}",
                    expected=expected,
                )
                continue
            if actual != expected:
                add_blocker(
                    blockers,
                    "metadata",
                    "metadata_required_value_mismatch",
                    f"metadata value differs for {path}",
                    expected=expected,
                    actual=actual,
                )
        contract = extract_tensor_contract(metadata)
        contract_sha = sha256_bytes(canonical_json_bytes(contract))
        if contract_sha != spec["metadata"]["tensor_contract_sha256"]:
            add_blocker(
                blockers,
                "metadata",
                "metadata_tensor_contract_sha256_mismatch",
                "normalized tensor descriptors differ from the exact release",
                expected=spec["metadata"]["tensor_contract_sha256"],
                actual=contract_sha,
            )
        actual_names = contract_tensor_names(contract)
        mismatches = contract_name_mismatches(
            declared_tensor_names(spec), actual_names
        )
        for mismatch in mismatches:
            add_blocker(
                blockers,
                "metadata",
                "metadata_tensor_names_mismatch",
                f"tensor names differ for {mismatch['component']}.{mismatch['side']}",
                expected={"missing_from_metadata": mismatch["missing"]},
                actual={"unexpected_in_metadata": mismatch["unexpected"]},
            )
        report.update(
            {
                "boundary_dtype_counts": contract_dtype_counts(contract),
                "model_id": metadata.get("model_id"),
                "onnx_runtime": metadata.get("tool_versions", {}).get(
                    "onnx_runtime"
                ),
                "qairt_build": metadata.get("tool_versions", {}).get("qairt"),
                "tensor_contract_sha256": contract_sha,
                "tensor_names": actual_names,
            }
        )
        internal.update(
            {
                "contract": contract,
                "contract_sha256": contract_sha,
                "tensor_names": actual_names,
            }
        )
        if not any(item["gate"] == "metadata" for item in blockers):
            report["status"] = "pass"
    except (GateInputError, json.JSONDecodeError) as error:
        add_blocker(
            blockers,
            "metadata",
            "metadata_validation_failed",
            str(error),
        )
    return report, internal


def quantize_real_value(real_value: float, descriptor: dict[str, Any]) -> int:
    dtype_ranges = {
        "uint8": (0, 255),
        "uint16": (0, 65535),
    }
    dtype = descriptor.get("dtype")
    if dtype not in dtype_ranges:
        raise GateInputError(f"unsupported quantized dtype: {dtype!r}")
    parameters = descriptor.get("quantization_parameters")
    if not isinstance(parameters, dict):
        raise GateInputError("quantized tensor lacks quantization_parameters")
    scale = parameters.get("scale")
    zero_point = parameters.get("zero_point")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise GateInputError("quantized tensor scale is invalid")
    if not isinstance(zero_point, int):
        raise GateInputError("quantized tensor zero_point is invalid")
    lower, upper = dtype_ranges[dtype]
    return max(lower, min(upper, round(real_value / scale) + zero_point))


def ep_context_check(
    spec: dict[str, Any],
    captured: dict[str, bytes],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "not_checked", "wrappers": {}}
    if not all(
        item["onnx_filename"] in captured
        for item in spec["ep_context"]["wrappers"]
    ):
        report["reason"] = "model_release_gate_not_passed"
        return report
    report["status"] = "blocked"
    for wrapper in spec["ep_context"]["wrappers"]:
        filename = wrapper["onnx_filename"]
        value = captured[filename]
        marker_counts: dict[str, int] = {}
        expected_markers = {
            "domain": ("com.microsoft", 1),
            "operator": ("EPContext", 1),
            "node_name": ("QNNContext", 1),
            "embed_mode_attribute": ("embed_mode", 1),
            "context_attribute": ("ep_cache_context", 1),
            "external_reference": (wrapper["external_reference"], 1),
            "source_attribute": ("source", 1),
        }
        for marker_name, (marker, expected_count) in expected_markers.items():
            count = value.count(marker.encode("utf-8"))
            marker_counts[marker_name] = count
            if count != expected_count:
                add_blocker(
                    blockers,
                    "ep_context",
                    "ep_context_marker_mismatch",
                    f"{filename} marker count differs for {marker_name}",
                    expected=expected_count,
                    actual=count,
                )
        context_member = member_by_basename(
            spec["model_release"], wrapper["context_filename"]
        )
        report["wrappers"][filename] = {
            "context_filename": wrapper["context_filename"],
            "context_sha256": context_member["sha256"],
            "external_reference": wrapper["external_reference"],
            "marker_counts": marker_counts,
        }
    if not any(item["gate"] == "ep_context" for item in blockers):
        report["status"] = "pass"
    return report


def elf_identity_bytes(value: bytes, label: str) -> tuple[int, int]:
    if len(value) < 20 or value[:4] != b"\x7fELF":
        raise GateInputError(f"artifact is not ELF: {label}")
    if value[4] == 1:
        bits = 32
    elif value[4] == 2:
        bits = 64
    else:
        raise GateInputError(f"invalid ELF class: {label}")
    if value[5] == 1:
        endian = "<"
    elif value[5] == 2:
        endian = ">"
    else:
        raise GateInputError(f"invalid ELF byte order: {label}")
    machine = int(struct.unpack(f"{endian}H", value[18:20])[0])
    return machine, bits


def require_elf_bytes(value: bytes, label: str, machine_name: str) -> dict[str, Any]:
    machine, bits = elf_identity_bytes(value, label)
    expected_machine = ELF_MACHINE[machine_name]
    expected_bits = 64 if machine_name == "aarch64" else 32
    if machine != expected_machine or bits != expected_bits:
        raise GateInputError(
            f"ELF identity mismatch for {label}: expected {machine_name}/ELF"
            f"{expected_bits}, got machine {machine}/ELF{bits}"
        )
    return {"elf_class_bits": bits, "elf_machine": machine_name}


def ort_semantic_check(
    spec: dict[str, Any],
    captured: dict[str, bytes],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "not_checked"}
    required = {"classes.jar", "libonnxruntime.so", "libonnxruntime4j_jni.so"}
    if not required.issubset(captured):
        report["reason"] = "ort_archive_gate_not_passed"
        return report
    report["status"] = "blocked"
    try:
        native = captured["libonnxruntime.so"]
        native_identity = require_elf_bytes(
            native, "libonnxruntime.so", "aarch64"
        )
        jni_identity = require_elf_bytes(
            captured["libonnxruntime4j_jni.so"],
            "libonnxruntime4j_jni.so",
            "aarch64",
        )
        marker_counts: dict[str, int] = {}
        for marker in spec["ort"]["required_native_markers"]:
            count = native.count(marker.encode("utf-8"))
            marker_counts[marker] = count
            if count < 1:
                add_blocker(
                    blockers,
                    "ort",
                    "ort_qnn_marker_missing",
                    f"libonnxruntime.so lacks required marker {marker}",
                )
        with zipfile.ZipFile(io.BytesIO(captured["classes.jar"])) as classes:
            names = classes.namelist()
            for name in spec["ort"]["required_java_classes"]:
                if name not in names:
                    add_blocker(
                        blockers,
                        "ort",
                        "ort_java_class_missing",
                        f"classes.jar lacks required API class {name}",
                    )
            session_options = classes.read(
                "ai/onnxruntime/OrtSession$SessionOptions.class"
            )
            for marker in spec["ort"]["required_java_markers"]:
                if marker.encode("utf-8") not in session_options:
                    add_blocker(
                        blockers,
                        "ort",
                        "ort_java_api_marker_missing",
                        f"SessionOptions lacks required API marker {marker}",
                    )
        report.update(
            {
                "java_qnn_api": "addQnn",
                "jni_identity": jni_identity,
                "native_identity": native_identity,
                "native_marker_counts": marker_counts,
                "version": spec["ort"]["version"],
            }
        )
        if not any(item["gate"] == "ort" for item in blockers):
            report["status"] = "pass"
    except (GateInputError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        add_blocker(blockers, "ort", "ort_semantic_validation_failed", str(error))
    return report


def voice_assets_check(
    spec: dict[str, Any],
    captured: dict[str, bytes],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "not_checked"}
    if "config.json" not in captured or "vocab.bin" not in captured:
        report["reason"] = "voice_asset_archive_gate_not_passed"
        return report
    report["status"] = "blocked"
    try:
        config = json.loads(captured["config.json"])
        if not isinstance(config, dict):
            raise GateInputError("Voice AI config.json must be an object")
        for requirement in spec["voice_assets"]["config_required_values"]:
            path = requirement["path"]
            expected = requirement["equals"]
            try:
                actual = nested_value(config, path)
            except KeyError:
                add_blocker(
                    blockers,
                    "voice_assets",
                    "voice_config_value_missing",
                    f"Voice AI config is missing {path}",
                    expected=expected,
                )
                continue
            if actual != expected:
                add_blocker(
                    blockers,
                    "voice_assets",
                    "voice_config_value_mismatch",
                    f"Voice AI config differs for {path}",
                    expected=expected,
                    actual=actual,
                )
        actual_vocab_sha = sha256_bytes(captured["vocab.bin"])
        if actual_vocab_sha != spec["voice_assets"]["vocab_sha256"]:
            add_blocker(
                blockers,
                "voice_assets",
                "voice_vocab_sha256_mismatch",
                "Voice AI vocabulary differs from the exact asset pin",
                expected=spec["voice_assets"]["vocab_sha256"],
                actual=actual_vocab_sha,
            )
        report.update(
            {
                "asset_scope": "tokenizer_config_and_vocabulary_only",
                "config_sha256": sha256_bytes(captured["config.json"]),
                "feature_extractor_source": "bound_runner_implementation",
                "vocab_bytes": len(captured["vocab.bin"]),
                "vocab_sha256": actual_vocab_sha,
            }
        )
        if not any(item["gate"] == "voice_assets" for item in blockers):
            report["status"] = "pass"
    except (GateInputError, json.JSONDecodeError) as error:
        add_blocker(
            blockers,
            "voice_assets",
            "voice_assets_validation_failed",
            str(error),
        )
    return report


def path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_manifest_root(manifest: dict[str, Any], manifest_path: Path) -> Path:
    root_value = manifest.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise GateInputError("manifest root must be a non-empty path")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = manifest_path.parent / root
    if root.is_symlink():
        raise GateInputError("manifest root symlinks are forbidden")
    return root.resolve(strict=True)


def resolve_manifest_file(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise GateInputError("artifact path must be non-empty")
    if "\\" in relative_path:
        raise GateInputError(f"backslashes are forbidden in {relative_path}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise GateInputError(f"unsafe relative artifact path: {relative_path}")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GateInputError(f"artifact symlinks are forbidden: {relative_path}")
    resolved = cursor.resolve(strict=True)
    if not path_is_within(resolved, root) or not resolved.is_file():
        raise GateInputError(f"artifact is not a regular in-root file: {relative_path}")
    return resolved


def verify_manifest_file(
    root: Path,
    entry: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise GateInputError("artifact inventory entry must be an object")
    path = resolve_manifest_file(root, entry.get("path"))
    expected_bytes = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise GateInputError(f"invalid byte pin for {path.name}")
    if not valid_sha256(expected_sha):
        raise GateInputError(f"invalid hash pin for {path.name}")
    if expected is not None:
        if expected_bytes != expected["bytes"] or expected_sha != expected["sha256"]:
            raise GateInputError(f"manifest pin differs from spec for {path.name}")
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise GateInputError(f"artifact bytes/hash mismatch for {path.name}")
    evidence: dict[str, Any] = {
        "bytes": actual_bytes,
        "filename": path.name,
        "sha256": actual_sha,
    }
    machine_name = expected.get("elf_machine") if expected is not None else None
    if machine_name is not None:
        with path.open("rb") as stream:
            identity = require_elf_bytes(stream.read(20), path.name, machine_name)
        evidence.update(identity)
    if expected is not None:
        with path.open("rb") as stream:
            value = stream.read()
        for marker in expected.get("required_ascii_markers", []):
            if marker.encode("utf-8") not in value:
                raise GateInputError(
                    f"artifact {path.name} lacks required build marker {marker}"
                )
    return evidence


def load_optional_json(
    path: Path | None,
    *,
    gate: str,
    missing_code: str,
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    if path is None:
        add_blocker(
            blockers,
            gate,
            missing_code,
            f"{gate.replace('_', ' ')} manifest/evidence was not supplied",
        )
        return None, None, {"status": "blocked"}
    report: dict[str, Any] = {"filename": path.name, "status": "blocked"}
    if path.is_symlink() or not path.is_file():
        add_blocker(
            blockers,
            gate,
            f"{gate}_input_missing_or_unsafe",
            f"input must be a regular non-symlink JSON file: {path.name}",
        )
        return None, None, report
    try:
        value, digest = load_json_object(path)
        report["manifest_sha256"] = digest
        return value, digest, report
    except (OSError, GateInputError) as error:
        add_blocker(blockers, gate, f"{gate}_input_invalid", str(error))
        return None, None, report


def assert_exact_mapping(
    actual: Any,
    expected: dict[str, Any],
    label: str,
) -> None:
    if not isinstance(actual, dict):
        raise GateInputError(f"{label} must be an object")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise GateInputError(
                f"{label}.{key} mismatch: expected {expected_value!r}, "
                f"got {actual.get(key)!r}"
            )


def runtime_check(
    spec: dict[str, Any],
    manifest_path: Path | None,
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_sha, report = load_optional_json(
        manifest_path,
        gate="runtime",
        missing_code="runtime_inventory_missing",
        blockers=blockers,
    )
    internal: dict[str, Any] = {"manifest_sha256": manifest_sha}
    if manifest is None or manifest_path is None:
        return report, internal
    try:
        if manifest.get("schema_version") != 1:
            raise GateInputError("runtime inventory schema_version must be 1")
        assert_exact_mapping(
            manifest.get("target"), spec["runtime"]["target"], "runtime target"
        )
        release = manifest.get("release")
        expected_release = {
            "qairt_version": spec["runtime"]["qairt_version"],
            "qairt_build": spec["runtime"]["exact_qairt_build"],
            "source_archive_filename": spec["runtime"]["source_archive"][
                "filename"
            ],
            "source_archive_bytes": spec["runtime"]["source_archive"]["bytes"],
            "source_archive_sha256": spec["runtime"]["source_archive"]["sha256"],
        }
        assert_exact_mapping(release, expected_release, "runtime release")
        root = resolve_manifest_root(manifest, manifest_path)
        files = manifest.get("files")
        if not isinstance(files, list):
            raise GateInputError("runtime files must be a list")
        by_name: dict[str, dict[str, Any]] = {}
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise GateInputError("runtime file entry is invalid")
            name = entry["name"]
            if name in by_name:
                raise GateInputError(f"duplicate runtime library: {name}")
            if PurePosixPath(str(entry.get("path", ""))).name != name:
                raise GateInputError(f"runtime name/path mismatch for {name}")
            by_name[name] = entry
        verified: dict[str, Any] = {}
        for requirement in spec["runtime"]["required_libraries"]:
            name = requirement["name"]
            entry = by_name.get(name)
            if entry is None:
                add_blocker(
                    blockers,
                    "runtime",
                    "runtime_library_missing",
                    f"required exact-build QAIRT library is absent: {name}",
                )
                continue
            try:
                verified[name] = verify_manifest_file(
                    root, entry, expected=requirement
                )
            except (OSError, GateInputError) as error:
                add_blocker(
                    blockers,
                    "runtime",
                    "runtime_library_invalid",
                    str(error),
                )
        supporting_entries = manifest.get("supporting_files", [])
        if not isinstance(supporting_entries, list):
            raise GateInputError("runtime supporting_files must be a list")
        supporting_by_path = {
            entry.get("path"): entry
            for entry in supporting_entries
            if isinstance(entry, dict)
        }
        verified_supporting: dict[str, Any] = {}
        for requirement in spec["runtime"].get("required_supporting_files", []):
            relative_path = requirement["path"]
            entry = supporting_by_path.get(relative_path)
            if entry is None:
                add_blocker(
                    blockers,
                    "runtime",
                    "runtime_supporting_file_missing",
                    f"required QAIRT supporting file is absent: {relative_path}",
                )
                continue
            try:
                verified_supporting[relative_path] = verify_manifest_file(
                    root, entry, expected=requirement
                )
            except (OSError, GateInputError) as error:
                add_blocker(
                    blockers,
                    "runtime",
                    "runtime_supporting_file_invalid",
                    str(error),
                )
        report.update(
            {
                "qairt_build": release.get("qairt_build")
                if isinstance(release, dict)
                else None,
                "source_archive_sha256": expected_release[
                    "source_archive_sha256"
                ],
                "verified_libraries": verified,
                "verified_supporting_files": verified_supporting,
            }
        )
        internal["verified_libraries"] = verified
        if not any(item["gate"] == "runtime" for item in blockers):
            report["status"] = "pass"
    except (OSError, GateInputError) as error:
        add_blocker(blockers, "runtime", "runtime_inventory_invalid", str(error))
    return report, internal


def runner_check(
    spec: dict[str, Any],
    manifest_path: Path | None,
    metadata_internal: dict[str, Any],
    runtime_internal: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_sha, report = load_optional_json(
        manifest_path,
        gate="runner",
        missing_code="runner_manifest_missing",
        blockers=blockers,
    )
    internal: dict[str, Any] = {"manifest_sha256": manifest_sha}
    if manifest is None or manifest_path is None:
        return report, internal
    try:
        if manifest.get("schema_version") != 1:
            raise GateInputError("runner manifest schema_version must be 1")
        assert_exact_mapping(
            manifest.get("target"), spec["runner"]["target"], "runner target"
        )
        assert_exact_mapping(
            manifest.get("deployment"),
            spec["runner"]["deployment"],
            "runner deployment",
        )
        assert_exact_mapping(
            manifest.get("provider"), spec["runner"]["provider"], "runner provider"
        )
        assert_exact_mapping(
            manifest.get("frontend"), spec["runner"]["frontend"], "runner frontend"
        )
        assert_exact_mapping(
            manifest.get("decode_loop"),
            spec["runner"]["decode_loop"],
            "runner decode_loop",
        )
        decode_loop = manifest["decode_loop"]
        expected_bindings = {
            "model_archive_sha256": spec["model_release"]["archive_sha256"],
            "model_metadata_sha256": spec["metadata"]["sha256"],
            "tensor_contract_sha256": spec["metadata"]["tensor_contract_sha256"],
            "ort_aar_sha256": spec["ort"]["release"]["archive_sha256"],
            "ort_native_library_sha256": spec["ort"]["native_library_sha256"],
            "voice_ai_archive_sha256": spec["voice_assets"]["release"][
                "archive_sha256"
            ],
            "voice_config_sha256": spec["voice_assets"]["config_sha256"],
            "voice_vocab_sha256": spec["voice_assets"]["vocab_sha256"],
            "runtime_inventory_sha256": runtime_internal.get("manifest_sha256"),
        }
        missing_dependencies = sorted(
            key for key, value in expected_bindings.items() if value is None
        )
        if missing_dependencies:
            add_blocker(
                blockers,
                "runner",
                "runner_dependency_gate_not_passed",
                "runner cannot bind missing or invalid route inputs",
                actual=missing_dependencies,
            )
        bindings = manifest.get("bindings")
        if not isinstance(bindings, dict):
            raise GateInputError("runner bindings must be an object")
        for key, expected in expected_bindings.items():
            if expected is not None and bindings.get(key) != expected:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_binding_mismatch",
                    f"runner binding differs for {key}",
                    expected=expected,
                    actual=bindings.get(key),
                )
        declared_contract = manifest.get("tensor_contract")
        if not isinstance(declared_contract, dict):
            raise GateInputError("runner tensor_contract must be an object")
        if declared_contract.get("full_contract_sha256") != metadata_internal.get(
            "contract_sha256"
        ):
            add_blocker(
                blockers,
                "runner",
                "runner_tensor_contract_mismatch",
                "runner tensor descriptor hash differs from trusted metadata",
                expected=metadata_internal.get("contract_sha256"),
                actual=declared_contract.get("full_contract_sha256"),
            )
        if declared_contract.get("quantization_source") != "release_metadata":
            raise GateInputError("runner quantization_source must be release_metadata")
        declared_names = declared_contract.get("tensor_names")
        if declared_names != metadata_internal.get("tensor_names"):
            add_blocker(
                blockers,
                "runner",
                "runner_tensor_names_mismatch",
                "runner tensor names differ from trusted metadata",
            )
        trusted_contract = metadata_internal.get("contract")
        quantized_decode_values: dict[str, Any] = {}
        if isinstance(trusted_contract, dict):
            decoder_inputs = trusted_contract["decoder"]["inputs"]
            attention = decoder_inputs["attention_mask"]
            unfilled = quantize_real_value(
                float(decode_loop["attention_mask_unfilled_real"]), attention
            )
            filled = quantize_real_value(0.0, attention)
            quantized_decode_values.update(
                {
                    "attention_mask_filled": filled,
                    "attention_mask_unfilled": unfilled,
                }
            )
            declared_quantized_values = {
                "attention_mask_filled": decode_loop[
                    "attention_mask_filled_uint16"
                ],
                "attention_mask_unfilled": decode_loop[
                    "attention_mask_unfilled_uint16"
                ],
            }
            if declared_quantized_values != quantized_decode_values:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_quantized_decode_value_mismatch",
                    "declared decoder mask values differ from release quantization",
                    expected=quantized_decode_values,
                    actual=declared_quantized_values,
                )
            self_kv_zero_points_set: set[int] = set()
            for name, descriptor in decoder_inputs.items():
                if not name.startswith(("k_cache_self_", "v_cache_self_")):
                    continue
                parameters = descriptor.get("quantization_parameters")
                if not isinstance(parameters, dict) or not isinstance(
                    parameters.get("zero_point"), int
                ):
                    raise GateInputError(
                        f"self-KV tensor lacks an integer zero point: {name}"
                    )
                self_kv_zero_points_set.add(parameters["zero_point"])
            self_kv_zero_points = sorted(self_kv_zero_points_set)
            quantized_decode_values["self_kv_zero_points"] = self_kv_zero_points
            if decode_loop.get("initial_self_kv") != "release_quantized_zero_point":
                add_blocker(
                    blockers,
                    "runner",
                    "runner_self_kv_initialization_mismatch",
                    "logical-zero self KV must use release zero points",
                )
        root = resolve_manifest_root(manifest, manifest_path)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise GateInputError("runner artifacts must be a list")
        by_role: dict[str, dict[str, Any]] = {}
        for entry in artifacts:
            if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
                raise GateInputError("runner artifact entry is invalid")
            role = entry["role"]
            if role in by_role:
                raise GateInputError(f"duplicate runner artifact role: {role}")
            by_role[role] = entry
        verified: dict[str, Any] = {}
        for role in spec["runner"]["required_artifact_roles"]:
            entry = by_role.get(role)
            if entry is None:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_artifact_role_missing",
                    f"required runner artifact role is absent: {role}",
                )
                continue
            try:
                verified[role] = verify_manifest_file(root, entry)
            except (OSError, GateInputError) as error:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_artifact_invalid",
                    str(error),
                )
        report.update(
            {
                "quantized_decode_values": quantized_decode_values,
                "tensor_contract_sha256": declared_contract.get(
                    "full_contract_sha256"
                ),
                "verified_artifacts": verified,
            }
        )
        internal["verified_artifacts"] = verified
        if not any(item["gate"] == "runner" for item in blockers):
            report["status"] = "pass"
    except (OSError, GateInputError) as error:
        add_blocker(blockers, "runner", "runner_manifest_invalid", str(error))
    return report, internal


def app_uid_check(
    spec: dict[str, Any],
    evidence_path: Path | None,
    runtime_internal: dict[str, Any],
    runner_internal: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence, evidence_sha, report = load_optional_json(
        evidence_path,
        gate="app_uid",
        missing_code="app_uid_evidence_missing",
        blockers=blockers,
    )
    if evidence is None:
        return report
    try:
        if evidence.get("schema_version") != 1:
            raise GateInputError("app-UID evidence schema_version must be 1")
        if evidence.get("status") != "pass":
            add_blocker(
                blockers,
                "app_uid",
                "app_uid_probe_not_passed",
                "app-UID probe status is not pass",
                expected="pass",
                actual=evidence.get("status"),
            )
        assert_exact_mapping(
            evidence.get("target"), spec["app_uid"]["target"], "app-UID target"
        )
        package_uid = evidence.get("package_uid")
        process_uid = evidence.get("process_uid")
        if (
            not isinstance(package_uid, int)
            or package_uid < 10000
            or process_uid != package_uid
        ):
            add_blocker(
                blockers,
                "app_uid",
                "app_uid_identity_mismatch",
                "process UID must equal a non-root Android package UID",
                expected=package_uid,
                actual=process_uid,
            )
        if evidence.get("qairt_build") != spec["runtime"]["exact_qairt_build"]:
            add_blocker(
                blockers,
                "app_uid",
                "app_uid_qairt_build_mismatch",
                "loaded QAIRT build differs from the EPContext compiler build",
                expected=spec["runtime"]["exact_qairt_build"],
                actual=evidence.get("qairt_build"),
            )
        runner_artifacts = runner_internal.get("verified_artifacts", {})
        expected_bindings = {
            "model_archive_sha256": spec["model_release"]["archive_sha256"],
            "model_metadata_sha256": spec["metadata"]["sha256"],
            "encoder_onnx_sha256": member_by_basename(
                spec["model_release"], "encoder.onnx"
            )["sha256"],
            "encoder_context_sha256": member_by_basename(
                spec["model_release"], "encoder_qairt_context.bin"
            )["sha256"],
            "decoder_onnx_sha256": member_by_basename(
                spec["model_release"], "decoder.onnx"
            )["sha256"],
            "decoder_context_sha256": member_by_basename(
                spec["model_release"], "decoder_qairt_context.bin"
            )["sha256"],
            "ort_aar_sha256": spec["ort"]["release"]["archive_sha256"],
            "ort_native_library_sha256": spec["ort"]["native_library_sha256"],
            "voice_config_sha256": spec["voice_assets"]["config_sha256"],
            "voice_vocab_sha256": spec["voice_assets"]["vocab_sha256"],
            "runtime_inventory_sha256": runtime_internal.get("manifest_sha256"),
            "runner_manifest_sha256": runner_internal.get("manifest_sha256"),
        }
        for role in spec["runner"]["required_artifact_roles"]:
            expected_bindings[f"runner_{role}_sha256"] = runner_artifacts.get(
                role, {}
            ).get("sha256")
        missing_dependencies = sorted(
            key for key, value in expected_bindings.items() if value is None
        )
        if missing_dependencies:
            add_blocker(
                blockers,
                "app_uid",
                "app_uid_dependency_gate_not_passed",
                "app-UID evidence cannot bind missing route inputs",
                actual=missing_dependencies,
            )
        bindings = evidence.get("bindings")
        if not isinstance(bindings, dict):
            raise GateInputError("app-UID bindings must be an object")
        for key, expected in expected_bindings.items():
            if expected is not None and bindings.get(key) != expected:
                add_blocker(
                    blockers,
                    "app_uid",
                    "app_uid_binding_mismatch",
                    f"app-UID evidence binding differs for {key}",
                    expected=expected,
                    actual=bindings.get(key),
                )
        checks = evidence.get("checks")
        if not isinstance(checks, dict):
            raise GateInputError("app-UID checks must be an object")
        for key, expected in spec["app_uid"]["required_checks"].items():
            if checks.get(key) != expected:
                add_blocker(
                    blockers,
                    "app_uid",
                    "app_uid_readiness_check_failed",
                    f"app-UID readiness check differs for {key}",
                    expected=expected,
                    actual=checks.get(key),
                )
        report.update(
            {
                "evidence_sha256": evidence_sha,
                "package_uid": package_uid,
                "process_uid": process_uid,
                "qairt_build": evidence.get("qairt_build"),
            }
        )
        if not any(item["gate"] == "app_uid" for item in blockers):
            report["status"] = "pass"
    except GateInputError as error:
        add_blocker(blockers, "app_uid", "app_uid_evidence_invalid", str(error))
    return report


def evaluate_gate(
    spec: dict[str, Any],
    *,
    model_archive: Path,
    ort_aar: Path,
    voice_ai_archive: Path,
    runtime_inventory: Path | None = None,
    runner_manifest: Path | None = None,
    app_uid_evidence: Path | None = None,
) -> dict[str, Any]:
    validate_spec(spec)
    blockers: list[dict[str, Any]] = []
    wrapper_names = {
        item["onnx_filename"] for item in spec["ep_context"]["wrappers"]
    }
    model_report, model_captured = verify_zip_release(
        spec["model_release"],
        model_archive,
        gate="model_release",
        capture_basenames=wrapper_names | {"metadata.json"},
        blockers=blockers,
    )
    metadata_report, metadata_internal = metadata_check(
        spec, model_captured.get("metadata.json"), blockers
    )
    ep_context_report = ep_context_check(spec, model_captured, blockers)

    ort_archive_report, ort_captured = verify_zip_release(
        spec["ort"]["release"],
        ort_aar,
        gate="ort",
        capture_basenames={
            "classes.jar",
            "libonnxruntime.so",
            "libonnxruntime4j_jni.so",
        },
        blockers=blockers,
    )
    ort_semantics = ort_semantic_check(spec, ort_captured, blockers)
    ort_report = {"archive": ort_archive_report, "semantics": ort_semantics}
    ort_report["status"] = (
        "pass"
        if ort_archive_report.get("status") == "pass"
        and ort_semantics.get("status") == "pass"
        else "blocked"
    )

    voice_archive_report, voice_captured = verify_zip_release(
        spec["voice_assets"]["release"],
        voice_ai_archive,
        gate="voice_assets",
        capture_basenames={"config.json", "vocab.bin"},
        blockers=blockers,
    )
    voice_semantics = voice_assets_check(spec, voice_captured, blockers)
    voice_report = {
        "archive": voice_archive_report,
        "semantics": voice_semantics,
        "status": (
            "pass"
            if voice_archive_report.get("status") == "pass"
            and voice_semantics.get("status") == "pass"
            else "blocked"
        ),
    }

    runtime_report, runtime_internal = runtime_check(
        spec, runtime_inventory, blockers
    )
    runner_report, runner_internal = runner_check(
        spec,
        runner_manifest,
        metadata_internal,
        runtime_internal,
        blockers,
    )
    app_uid_report = app_uid_check(
        spec,
        app_uid_evidence,
        runtime_internal,
        runner_internal,
        blockers,
    )
    ordered = sorted_blockers(blockers)
    ready = not ordered
    return {
        "blockers": ordered,
        "checks": {
            "app_uid": app_uid_report,
            "ep_context": ep_context_report,
            "metadata": metadata_report,
            "model_release": model_report,
            "ort": ort_report,
            "runner": runner_report,
            "runtime": runtime_report,
            "voice_assets": voice_report,
        },
        "gate_id": spec["gate_id"],
        "policy": {
            "adb_invoked": False,
            "archives_extracted": False,
            "model_loaded": False,
            "network_used": False,
            "phone_mutated": False,
        },
        "ready": ready,
        "route": "onnxruntime_qnn_external_epcontext",
        "schema_version": 1,
        "spec_canonical_sha256": sha256_bytes(canonical_json_bytes(spec)),
        "status": "ready" if ready else "blocked",
    }


def error_report(error: Exception) -> dict[str, Any]:
    return {
        "blockers": [
            {
                "code": "gate_input_invalid",
                "detail": str(error),
                "gate": "spec",
            }
        ],
        "checks": {},
        "gate_id": "invalid",
        "policy": {
            "adb_invoked": False,
            "archives_extracted": False,
            "model_loaded": False,
            "network_used": False,
            "phone_mutated": False,
        },
        "ready": False,
        "route": "onnxruntime_qnn_external_epcontext",
        "schema_version": 1,
        "spec_canonical_sha256": None,
        "status": "invalid",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--ort-aar", type=Path, required=True)
    parser.add_argument("--voice-ai-archive", type=Path, required=True)
    parser.add_argument("--runtime-inventory", type=Path)
    parser.add_argument("--runner-manifest", type=Path)
    parser.add_argument("--app-uid-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec, _ = load_json_object(args.spec)
        report = evaluate_gate(
            spec,
            model_archive=args.model_archive,
            ort_aar=args.ort_aar,
            voice_ai_archive=args.voice_ai_archive,
            runtime_inventory=args.runtime_inventory,
            runner_manifest=args.runner_manifest,
            app_uid_evidence=args.app_uid_evidence,
        )
        exit_code = 0 if report["ready"] else 1
    except (OSError, GateInputError) as error:
        report = error_report(error)
        exit_code = 2
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    ) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
