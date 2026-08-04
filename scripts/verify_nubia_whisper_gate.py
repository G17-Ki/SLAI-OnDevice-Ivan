#!/usr/bin/env python3
"""Fail-closed host gate for Nubia Whisper Small W8A16 deployment.

The verifier is deliberately read-only except for an explicitly requested JSON
report.  It never extracts a ZIP, loads a QNN context, invokes ADB, or performs
inference.  A deployment is ready only when the pinned release, Android QAIRT
runtime, preprocessing assets, runner contract, and app-UID evidence all pass.
"""

from __future__ import annotations

import argparse
import hashlib
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
    "nubia_whisper_gate_spec.json"
)
GATE_ORDER = {
    "spec": 0,
    "archive": 1,
    "components": 2,
    "extraction_root": 3,
    "metadata": 4,
    "voice_ai": 5,
    "runtime": 6,
    "preprocessing": 7,
    "runner": 8,
    "app_uid": 9,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
ELF_MACHINE = {"aarch64": 183, "hexagon": 164}


class GateInputError(ValueError):
    """Raised for malformed verifier inputs, never for an unmet deployment gate."""


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


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise GateInputError("gate spec schema_version must be 1")
    for key in (
        "gate_id",
        "release",
        "metadata",
        "runtime",
        "preprocessing",
        "runner",
        "app_uid",
    ):
        if key not in spec:
            raise GateInputError(f"gate spec is missing {key}")
    release = spec["release"]
    if not valid_sha256(release.get("archive_sha256")):
        raise GateInputError("gate spec has an invalid archive SHA-256")
    if not isinstance(release.get("archive_bytes"), int):
        raise GateInputError("gate spec has an invalid archive byte count")
    members = release.get("members")
    if not isinstance(members, list) or not members:
        raise GateInputError("gate spec must declare a non-empty member list")
    metadata_spec = spec["metadata"]
    if not valid_sha256(metadata_spec.get("sha256")):
        raise GateInputError("gate spec has an invalid metadata SHA-256")
    if not valid_sha256(metadata_spec.get("tensor_contract_sha256")):
        raise GateInputError("gate spec has an invalid tensor-contract SHA-256")
    if (
        not isinstance(metadata_spec.get("layer_count"), int)
        or metadata_spec["layer_count"] < 1
    ):
        raise GateInputError("gate spec has an invalid transformer layer count")
    metadata_member_sha: str | None = None
    for member in members:
        if not isinstance(member, dict):
            raise GateInputError("gate spec contains a non-object archive member")
        name = member.get("name")
        member_type = member.get("type")
        member_bytes = member.get("bytes")
        if not isinstance(name, str) or not safe_zip_name(name):
            raise GateInputError("gate spec contains an unsafe archive member path")
        if member_type not in ("file", "directory"):
            raise GateInputError(f"gate spec has an invalid member type for {name}")
        if not isinstance(member_bytes, int) or member_bytes < 0:
            raise GateInputError(f"gate spec has an invalid byte count for {name}")
        if member_type == "file" and not valid_sha256(member.get("sha256")):
            raise GateInputError(f"gate spec has an invalid SHA-256 for {name}")
        if name.endswith("/metadata.json"):
            metadata_member_sha = member.get("sha256")
    if metadata_member_sha != metadata_spec["sha256"]:
        raise GateInputError(
            "gate spec metadata SHA-256 differs from its archive member pin"
        )
    parse_version(spec["runtime"].get("minimum_qairt"))
    voice_ai = spec.get("voice_ai")
    if voice_ai is not None:
        if not isinstance(voice_ai, dict):
            raise GateInputError("gate spec voice_ai must be an object")
        voice_release = voice_ai.get("release", {})
        voice_metadata = voice_ai.get("metadata", {})
        voice_config = voice_ai.get("config", {})
        if not valid_sha256(voice_release.get("archive_sha256")):
            raise GateInputError("Voice AI spec has an invalid archive SHA-256")
        if not isinstance(voice_release.get("archive_bytes"), int):
            raise GateInputError("Voice AI spec has an invalid archive byte count")
        voice_members = voice_release.get("members")
        if not isinstance(voice_members, list) or not voice_members:
            raise GateInputError("Voice AI spec has no exact archive member list")
        voice_hashes: dict[str, str] = {}
        for member in voice_members:
            if not isinstance(member, dict):
                raise GateInputError("Voice AI spec contains an invalid member")
            name = member.get("name")
            if not isinstance(name, str) or not safe_zip_name(name):
                raise GateInputError("Voice AI spec contains an unsafe member path")
            if member.get("type") == "file":
                member_sha = member.get("sha256")
                if not valid_sha256(member_sha):
                    raise GateInputError(
                        f"Voice AI spec has an invalid member SHA-256 for {name}"
                    )
                voice_hashes[PurePosixPath(name).name] = member_sha
        expected_voice_hashes = {
            "metadata.json": voice_metadata.get("sha256"),
            "config.json": voice_config.get("sha256"),
            "vocab.bin": voice_ai.get("vocab_sha256"),
        }
        for name, expected_sha in expected_voice_hashes.items():
            if not valid_sha256(expected_sha) or voice_hashes.get(name) != expected_sha:
                raise GateInputError(
                    f"Voice AI {name} pin differs from its archive member SHA-256"
                )
        if voice_metadata.get("tensor_contract_sha256") != metadata_spec.get(
            "tensor_contract_sha256"
        ):
            raise GateInputError(
                "Voice AI and raw QNN packages must pin the same tensor contract"
            )


def safe_zip_name(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return False
    pure = PurePosixPath(name)
    if any(part in ("", ".", "..") for part in pure.parts):
        return False
    return not (pure.parts and ":" in pure.parts[0])


def hash_bounded_stream(
    stream: BinaryIO, expected_bytes: int, *, capture: bool = False
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
            raise GateInputError("stream exceeded its pinned uncompressed size")
        digest.update(block)
        if captured is not None:
            captured.extend(block)
    if total != expected_bytes:
        raise GateInputError(
            f"stream size mismatch: expected {expected_bytes}, read {total}"
        )
    return digest.hexdigest(), bytes(captured) if captured is not None else None


def expected_member_map(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    members = spec["release"]["members"]
    result: dict[str, dict[str, Any]] = {}
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("name"), str):
            raise GateInputError("gate spec contains an invalid archive member")
        name = member["name"]
        if name in result:
            raise GateInputError(f"gate spec repeats archive member {name}")
        result[name] = member
    return result


def archive_checks(
    spec: dict[str, Any], archive_path: Path, blockers: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    release = spec["release"]
    archive_report: dict[str, Any] = {
        "filename": archive_path.name,
        "status": "blocked",
    }
    component_report: dict[str, Any] = {
        "items": {},
        "source": "zip_stream",
        "status": "not_checked",
    }
    internal: dict[str, Any] = {}
    if not archive_path.exists():
        add_blocker(
            blockers,
            "archive",
            "archive_missing",
            "the pinned Whisper release archive was not found",
            expected=release["archive_filename"],
            actual=archive_path.name,
        )
        return archive_report, component_report, internal
    if archive_path.is_symlink():
        add_blocker(
            blockers,
            "archive",
            "archive_symlink_rejected",
            "the release archive must be a regular non-symlink file",
        )
        return archive_report, component_report, internal
    if not archive_path.is_file():
        add_blocker(
            blockers,
            "archive",
            "archive_not_regular_file",
            "the release archive path is not a regular file",
        )
        return archive_report, component_report, internal

    actual_bytes = archive_path.stat().st_size
    archive_report["bytes"] = actual_bytes
    if actual_bytes != release["archive_bytes"]:
        add_blocker(
            blockers,
            "archive",
            "archive_size_mismatch",
            "release archive byte count does not match the pin",
            expected=release["archive_bytes"],
            actual=actual_bytes,
        )
        return archive_report, component_report, internal

    actual_sha256 = sha256_file(archive_path)
    archive_report["sha256"] = actual_sha256
    internal["archive_sha256"] = actual_sha256
    if actual_sha256 != release["archive_sha256"]:
        add_blocker(
            blockers,
            "archive",
            "archive_sha256_mismatch",
            "release archive SHA-256 does not match the pin",
            expected=release["archive_sha256"],
            actual=actual_sha256,
        )
        return archive_report, component_report, internal

    try:
        expected_by_name = expected_member_map(spec)
        expected_names = [item["name"] for item in release["members"]]
        with zipfile.ZipFile(archive_path) as bundle:
            infos = bundle.infolist()
            actual_names = [info.filename for info in infos]
            archive_report["member_names"] = actual_names
            if len(actual_names) != len(set(actual_names)):
                add_blocker(
                    blockers,
                    "archive",
                    "archive_duplicate_members",
                    "the ZIP contains duplicate member names",
                )
            if actual_names != expected_names:
                add_blocker(
                    blockers,
                    "archive",
                    "archive_member_list_mismatch",
                    "the ordered ZIP member list differs from the exact pin",
                    expected=expected_names,
                    actual=actual_names,
                )

            archive_unsafe = False
            for info in infos:
                if not safe_zip_name(info.filename):
                    archive_unsafe = True
                    add_blocker(
                        blockers,
                        "archive",
                        "archive_member_path_unsafe",
                        f"unsafe ZIP member path: {info.filename}",
                    )
                if info.flag_bits & 0x1:
                    archive_unsafe = True
                    add_blocker(
                        blockers,
                        "archive",
                        "archive_member_encrypted",
                        f"encrypted ZIP member is not allowed: {info.filename}",
                    )
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                    archive_unsafe = True
                    add_blocker(
                        blockers,
                        "archive",
                        "archive_member_symlink",
                        f"symlink ZIP member is not allowed: {info.filename}",
                    )
                expected = expected_by_name.get(info.filename)
                if expected is None:
                    continue
                expected_type = expected.get("type")
                actual_type = "directory" if info.is_dir() else "file"
                if actual_type != expected_type:
                    add_blocker(
                        blockers,
                        "archive",
                        "archive_member_type_mismatch",
                        f"ZIP member type differs for {info.filename}",
                        expected=expected_type,
                        actual=actual_type,
                    )
                if info.file_size != expected.get("bytes"):
                    add_blocker(
                        blockers,
                        "archive",
                        "archive_member_size_mismatch",
                        f"ZIP member byte count differs for {info.filename}",
                        expected=expected.get("bytes"),
                        actual=info.file_size,
                    )

            archive_gate_blocked = any(
                item["gate"] == "archive" for item in blockers
            )
            if archive_gate_blocked or archive_unsafe:
                return archive_report, component_report, internal

            component_report["status"] = "blocked"
            captured_files: dict[str, bytes] = {}
            for info in infos:
                expected = expected_by_name[info.filename]
                if expected["type"] == "directory":
                    continue
                short_name = PurePosixPath(info.filename).name
                capture = short_name in {"config.json", "metadata.json"}
                with bundle.open(info, "r") as stream:
                    actual_member_sha, captured = hash_bounded_stream(
                        stream, expected["bytes"], capture=capture
                    )
                component_report["items"][short_name] = {
                    "bytes": info.file_size,
                    "sha256": actual_member_sha,
                }
                if actual_member_sha != expected.get("sha256"):
                    add_blocker(
                        blockers,
                        "components",
                        "component_sha256_mismatch",
                        f"streamed component hash differs for {short_name}",
                        expected=expected.get("sha256"),
                        actual=actual_member_sha,
                    )
                if capture:
                    if captured is None:
                        raise GateInputError(f"failed to capture {short_name}")
                    captured_files[short_name] = captured
            if not any(item["gate"] == "components" for item in blockers):
                component_report["status"] = "pass"
                internal["captured_files"] = captured_files
                internal["metadata_bytes"] = captured_files.get("metadata.json")
                archive_report["status"] = "pass"
    except (OSError, GateInputError, zipfile.BadZipFile, RuntimeError) as error:
        add_blocker(
            blockers,
            "archive",
            "archive_stream_validation_failed",
            str(error),
        )
    return archive_report, component_report, internal


def path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_manifest_root(manifest: dict[str, Any], manifest_path: Path) -> Path:
    root_value = manifest.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise GateInputError("manifest root must be a non-empty path string")
    candidate = Path(root_value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.is_symlink():
        raise GateInputError("manifest root symlinks are not allowed")
    return candidate.resolve(strict=True)


def resolve_manifest_file(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise GateInputError("artifact path must be a non-empty string")
    if "\\" in relative_path:
        raise GateInputError(f"backslashes are not allowed in {relative_path}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise GateInputError(f"unsafe relative artifact path: {relative_path}")
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GateInputError(f"artifact symlinks are not allowed: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not path_is_within(resolved, root):
        raise GateInputError(f"artifact escapes its manifest root: {relative_path}")
    if not resolved.is_file():
        raise GateInputError(f"artifact is not a regular file: {relative_path}")
    return resolved


def elf_identity(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise GateInputError(f"artifact is not an ELF file: {path.name}")
    if header[4] == 1:
        elf_class_bits = 32
    elif header[4] == 2:
        elf_class_bits = 64
    else:
        raise GateInputError(f"artifact has invalid ELF class: {path.name}")
    if header[5] == 1:
        endian = "<"
    elif header[5] == 2:
        endian = ">"
    else:
        raise GateInputError(f"artifact has invalid ELF byte order: {path.name}")
    return int(struct.unpack(f"{endian}H", header[18:20])[0]), elf_class_bits


def verify_manifest_artifact(
    root: Path,
    entry: dict[str, Any],
    *,
    expected_machine: str | None = None,
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise GateInputError("artifact inventory entry must be an object")
    path = resolve_manifest_file(root, entry.get("path"))
    expected_bytes = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise GateInputError(f"invalid pinned byte count for {path.name}")
    if not valid_sha256(expected_sha):
        raise GateInputError(f"invalid pinned SHA-256 for {path.name}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise GateInputError(
            f"artifact byte count mismatch for {path.name}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise GateInputError(
            f"artifact SHA-256 mismatch for {path.name}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    evidence: dict[str, Any] = {
        "bytes": actual_bytes,
        "filename": path.name,
        "sha256": actual_sha,
    }
    if expected_machine is not None:
        expected_number = ELF_MACHINE.get(expected_machine)
        if expected_number is None:
            raise GateInputError(f"unsupported ELF machine pin {expected_machine}")
        actual_machine, actual_class_bits = elf_identity(path)
        if actual_machine != expected_number:
            raise GateInputError(
                f"ELF machine mismatch for {path.name}: expected "
                f"{expected_machine} ({expected_number}), got {actual_machine}"
            )
        expected_class_bits = 64 if expected_machine == "aarch64" else 32
        if actual_class_bits != expected_class_bits:
            raise GateInputError(
                f"ELF class mismatch for {path.name}: expected "
                f"ELF{expected_class_bits}, got ELF{actual_class_bits}"
            )
        evidence["elf_machine"] = expected_machine
        evidence["elf_class_bits"] = actual_class_bits
    return evidence


def extraction_root_check(
    spec: dict[str, Any],
    extraction_root: Path | None,
    metadata_bytes: bytes | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    if extraction_root is None:
        return {"status": "not_requested"}
    report: dict[str, Any] = {
        "name": extraction_root.name,
        "status": "blocked",
    }
    try:
        if extraction_root.is_symlink():
            raise GateInputError("extraction root symlinks are not allowed")
        resolved = extraction_root.resolve(strict=True)
        archive_root_name = spec["release"]["archive_root"]
        candidate = resolved / archive_root_name
        model_root = candidate if candidate.is_dir() else resolved
        if model_root.is_symlink() or not model_root.is_dir():
            raise GateInputError("optional extraction root is not a regular directory")
        expected_files = sorted(
            PurePosixPath(item["name"]).name
            for item in spec["release"]["members"]
            if item["type"] == "file"
        )
        actual_files: list[str] = []
        for item in model_root.iterdir():
            if item.is_symlink() or not item.is_file():
                raise GateInputError(
                    f"unexpected non-regular item in extraction root: {item.name}"
                )
            actual_files.append(item.name)
        actual_files.sort()
        if actual_files != expected_files:
            raise GateInputError(
                "extraction root file list mismatch: "
                f"expected {expected_files}, got {actual_files}"
            )
        expected_by_short_name = {
            PurePosixPath(item["name"]).name: item
            for item in spec["release"]["members"]
            if item["type"] == "file"
        }
        items: dict[str, Any] = {}
        for name in expected_files:
            expected = expected_by_short_name[name]
            evidence = verify_manifest_artifact(
                model_root,
                {
                    "path": name,
                    "bytes": expected["bytes"],
                    "sha256": expected["sha256"],
                },
            )
            items[name] = {"bytes": evidence["bytes"], "sha256": evidence["sha256"]}
        if metadata_bytes is not None:
            extracted_metadata = (model_root / "metadata.json").read_bytes()
            if extracted_metadata != metadata_bytes:
                raise GateInputError(
                    "extracted metadata bytes differ from the streamed archive metadata"
                )
        report.update({"items": items, "status": "pass"})
    except (OSError, GateInputError) as error:
        add_blocker(
            blockers,
            "extraction_root",
            "extraction_root_validation_failed",
            str(error),
        )
    return report


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
    normalized: dict[str, Any] = {}
    for key in ("dtype", "shape", "quantization_parameters"):
        if key in value:
            normalized[key] = value[key]
    if "dtype" not in normalized or "shape" not in normalized:
        raise GateInputError("tensor descriptor is missing dtype or shape")
    return normalized


def extract_tensor_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    model_files = metadata.get("model_files")
    if not isinstance(model_files, dict):
        raise GateInputError("release metadata is missing model_files")
    contract: dict[str, Any] = {}
    for filename, io_contract in sorted(model_files.items()):
        if not isinstance(filename, str) or not filename.endswith(".bin"):
            raise GateInputError("release metadata has an invalid model filename")
        if not isinstance(io_contract, dict):
            raise GateInputError(f"invalid tensor contract for {filename}")
        component: dict[str, Any] = {}
        for side in ("inputs", "outputs"):
            tensors = io_contract.get(side)
            if not isinstance(tensors, dict):
                raise GateInputError(f"{filename} is missing {side}")
            component[side] = {
                name: normalize_tensor_descriptor(descriptor)
                for name, descriptor in sorted(tensors.items())
            }
        contract[filename[:-4]] = component
    return contract


def expand_tensor_names(declaration: Any, layer_count: int) -> list[str]:
    if not isinstance(declaration, list):
        raise GateInputError("tensor name declaration must be a list")
    expanded: list[str] = []
    for item in declaration:
        if not isinstance(item, str) or not item:
            raise GateInputError("tensor name declarations must be non-empty strings")
        if "{layer}" in item:
            if item.count("{layer}") != 1:
                raise GateInputError(f"invalid tensor template: {item}")
            expanded.extend(item.replace("{layer}", str(index)) for index in range(layer_count))
        elif "{" in item or "}" in item:
            raise GateInputError(f"unsupported tensor template: {item}")
        else:
            expanded.append(item)
    if len(expanded) != len(set(expanded)):
        raise GateInputError("tensor name declaration expands to duplicate names")
    return sorted(expanded)


def declared_contract_names(
    declaration: Any, layer_count: int
) -> dict[str, dict[str, list[str]]]:
    if not isinstance(declaration, dict):
        raise GateInputError("tensor contract declaration must be an object")
    result: dict[str, dict[str, list[str]]] = {}
    for component in ("encoder", "decoder"):
        item = declaration.get(component)
        if not isinstance(item, dict):
            raise GateInputError(f"tensor contract is missing {component}")
        result[component] = {
            side: expand_tensor_names(item.get(side), layer_count)
            for side in ("inputs", "outputs")
        }
    return result


def actual_contract_names(
    contract: dict[str, Any]
) -> dict[str, dict[str, list[str]]]:
    return {
        component: {
            side: sorted(contract[component][side]) for side in ("inputs", "outputs")
        }
        for component in ("encoder", "decoder")
    }


def compare_contract_names(
    expected: dict[str, dict[str, list[str]]],
    actual: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for component in ("encoder", "decoder"):
        for side in ("inputs", "outputs"):
            expected_set = set(expected[component][side])
            actual_set = set(actual[component][side])
            if expected_set != actual_set:
                mismatches.append(
                    {
                        "component": component,
                        "missing": sorted(expected_set - actual_set),
                        "side": side,
                        "unexpected": sorted(actual_set - expected_set),
                    }
                )
    return mismatches


def metadata_check(
    spec: dict[str, Any],
    metadata_bytes: bytes | None,
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {"status": "not_checked"}
    internal: dict[str, Any] = {}
    if metadata_bytes is None:
        report["reason"] = "archive_component_gate_not_passed"
        return report, internal
    report["status"] = "blocked"
    try:
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise GateInputError("metadata.json is not a JSON object")
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
                    f"release metadata is missing {path}",
                    expected=expected,
                )
                continue
            if actual != expected:
                add_blocker(
                    blockers,
                    "metadata",
                    "metadata_required_value_mismatch",
                    f"release metadata value differs for {path}",
                    expected=expected,
                    actual=actual,
                )
        contract = extract_tensor_contract(metadata)
        full_contract_sha = sha256_bytes(canonical_json_bytes(contract))
        expected_contract_sha = spec["metadata"]["tensor_contract_sha256"]
        if full_contract_sha != expected_contract_sha:
            add_blocker(
                blockers,
                "metadata",
                "metadata_tensor_contract_sha256_mismatch",
                "normalized tensor descriptors differ from the pinned Qualcomm contract",
                expected=expected_contract_sha,
                actual=full_contract_sha,
            )
        layer_count = spec["metadata"]["layer_count"]
        expected_names = declared_contract_names(
            spec["metadata"]["tensor_names"], layer_count
        )
        actual_names = actual_contract_names(contract)
        mismatches = compare_contract_names(expected_names, actual_names)
        for mismatch in mismatches:
            add_blocker(
                blockers,
                "metadata",
                "metadata_tensor_names_mismatch",
                f"Qualcomm metadata contract mismatch for "
                f"{mismatch['component']}.{mismatch['side']}",
                expected={"missing_from_metadata": mismatch["missing"]},
                actual={"unexpected_in_metadata": mismatch["unexpected"]},
            )
        report.update(
            {
                "model_id": metadata.get("model_id"),
                "qairt_build": metadata.get("tool_versions", {}).get("qairt"),
                "tensor_contract_sha256": full_contract_sha,
                "tensor_names": actual_names,
            }
        )
        internal.update(
            {
                "contract": contract,
                "contract_names": actual_names,
                "contract_sha256": full_contract_sha,
                "metadata": metadata,
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


def remap_voice_ai_blockers(
    local_blockers: list[dict[str, Any]], blockers: list[dict[str, Any]]
) -> None:
    for item in local_blockers:
        mapped = dict(item)
        source_gate = str(mapped.get("gate"))
        mapped["gate"] = "voice_ai"
        if source_gate == "archive":
            mapped["code"] = f"voice_ai_{mapped['code']}"
        elif source_gate == "components":
            mapped["code"] = f"voice_ai_{mapped['code']}"
        elif source_gate == "metadata":
            mapped["code"] = f"voice_ai_{mapped['code']}"
        blockers.append(mapped)


def voice_ai_config_check(
    spec: dict[str, Any],
    config_bytes: bytes | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "not_checked"}
    if config_bytes is None:
        report["reason"] = "voice_ai_component_gate_not_passed"
        return report
    report["status"] = "blocked"
    try:
        config = json.loads(config_bytes)
        if not isinstance(config, dict):
            raise GateInputError("Voice AI config.json is not a JSON object")
        for requirement in spec["voice_ai"]["config"]["required_values"]:
            path = requirement["path"]
            expected = requirement["equals"]
            try:
                actual = nested_value(config, path)
            except KeyError:
                add_blocker(
                    blockers,
                    "voice_ai",
                    "voice_ai_config_required_value_missing",
                    f"Voice AI config is missing {path}",
                    expected=expected,
                )
                continue
            if actual != expected:
                add_blocker(
                    blockers,
                    "voice_ai",
                    "voice_ai_config_required_value_mismatch",
                    f"Voice AI config value differs for {path}",
                    expected=expected,
                    actual=actual,
                )
        report.update(
            {
                "model_type": config.get("model_type"),
                "qnn_version": config.get("runtime", {}).get("qnn_version"),
                "sample_rates": config.get("parameters", {}).get("sample_rates"),
            }
        )
        if not any(item["gate"] == "voice_ai" for item in blockers):
            report["status"] = "pass"
    except (GateInputError, json.JSONDecodeError) as error:
        add_blocker(
            blockers,
            "voice_ai",
            "voice_ai_config_validation_failed",
            str(error),
        )
    return report


def voice_ai_check(
    spec: dict[str, Any],
    archive_path: Path | None,
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if archive_path is None:
        return {"status": "not_requested"}, {"ready": False}
    voice_spec = spec.get("voice_ai")
    if not isinstance(voice_spec, dict):
        add_blocker(
            blockers,
            "voice_ai",
            "voice_ai_spec_missing",
            "a Voice AI archive was supplied but the gate spec has no Voice AI pin",
        )
        return {"status": "blocked"}, {"ready": False}

    local_blockers: list[dict[str, Any]] = []
    archive_report, component_report, archive_internal = archive_checks(
        {"release": voice_spec["release"]}, archive_path, local_blockers
    )
    metadata_report, metadata_internal = metadata_check(
        {"metadata": voice_spec["metadata"]},
        archive_internal.get("metadata_bytes"),
        local_blockers,
    )
    remap_voice_ai_blockers(local_blockers, blockers)
    config_report = voice_ai_config_check(
        spec,
        archive_internal.get("captured_files", {}).get("config.json"),
        blockers,
    )
    component_items = component_report.get("items", {})
    vocab = component_items.get("vocab.bin")
    tokenizer_ready = (
        component_report.get("status") == "pass"
        and config_report.get("status") == "pass"
        and isinstance(vocab, dict)
        and vocab.get("sha256") == voice_spec["vocab_sha256"]
    )
    if component_report.get("status") == "pass" and not tokenizer_ready:
        add_blocker(
            blockers,
            "voice_ai",
            "voice_ai_tokenizer_bundle_invalid",
            "Voice AI config/vocab did not satisfy the pinned tokenizer bundle",
        )
    voice_blocked = any(item["gate"] == "voice_ai" for item in blockers)
    ready = (
        not voice_blocked
        and archive_report.get("status") == "pass"
        and component_report.get("status") == "pass"
        and metadata_report.get("status") == "pass"
        and config_report.get("status") == "pass"
        and tokenizer_ready
    )
    report = {
        "archive": archive_report,
        "components": component_report,
        "config": config_report,
        "metadata": metadata_report,
        "runner_requirement": voice_spec["runner_requirement"],
        "status": "pass" if ready else "blocked",
        "tokenizer_ready": tokenizer_ready,
    }
    internal = {
        "archive_sha256": archive_internal.get("archive_sha256"),
        "contract_names": metadata_internal.get("contract_names"),
        "contract_sha256": metadata_internal.get("contract_sha256"),
        "ready": ready,
        "tokenizer_ready": tokenizer_ready,
    }
    return report, internal


def parse_version(value: Any) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise GateInputError("QAIRT version must be a string")
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise GateInputError(f"invalid QAIRT version: {value!r}")
    return tuple(int(part) for part in match.groups(default="0"))


def version_at_least(actual: Any, minimum: Any) -> bool:
    return parse_version(actual) >= parse_version(minimum)


def load_optional_manifest(
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
    if not path.is_file() or path.is_symlink():
        add_blocker(
            blockers,
            gate,
            f"{gate}_input_missing_or_unsafe",
            f"input must be a regular non-symlink JSON file: {path.name}",
        )
        return None, None, report
    try:
        value, manifest_sha = load_json_object(path)
        report["manifest_sha256"] = manifest_sha
        return value, manifest_sha, report
    except (OSError, GateInputError) as error:
        add_blocker(
            blockers,
            gate,
            f"{gate}_input_invalid",
            str(error),
        )
        return None, None, report


def runtime_check(
    spec: dict[str, Any],
    manifest_path: Path | None,
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_sha, report = load_optional_manifest(
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
        target = manifest.get("target", {})
        for key, expected in spec["runtime"]["target"].items():
            if target.get(key) != expected:
                raise GateInputError(
                    f"runtime target {key} mismatch: expected {expected!r}, "
                    f"got {target.get(key)!r}"
                )
        release = manifest.get("release", {})
        qairt_version = release.get("qairt_version")
        minimum = spec["runtime"]["minimum_qairt"]
        if not version_at_least(qairt_version, minimum):
            add_blocker(
                blockers,
                "runtime",
                "runtime_qairt_too_old",
                "Android runtime is older than the release context requirement",
                expected=f">={minimum}",
                actual=qairt_version,
            )
        if not valid_sha256(release.get("source_archive_sha256")):
            raise GateInputError(
                "runtime inventory must pin its source archive SHA-256"
            )
        if not isinstance(release.get("release_id"), str) or not release["release_id"]:
            raise GateInputError("runtime inventory must declare release_id")
        root = resolve_manifest_root(manifest, manifest_path)
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise GateInputError("runtime inventory files must be a list")
        by_name: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise GateInputError("invalid runtime file inventory entry")
            name = entry["name"]
            if name in by_name:
                raise GateInputError(f"duplicate runtime library entry: {name}")
            if PurePosixPath(str(entry.get("path", ""))).name != name:
                raise GateInputError(f"runtime entry name/path mismatch for {name}")
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
                    f"required QAIRT Android library is absent: {name}",
                )
                continue
            try:
                verified[name] = verify_manifest_artifact(
                    root, entry, expected_machine=requirement["elf_machine"]
                )
            except (OSError, GateInputError) as error:
                add_blocker(
                    blockers,
                    "runtime",
                    "runtime_library_invalid",
                    str(error),
                )
        report.update(
            {
                "qairt_version": qairt_version,
                "source_archive_sha256": release.get("source_archive_sha256"),
                "verified_libraries": verified,
            }
        )
        internal.update(
            {
                "qairt_version": qairt_version,
                "verified_libraries": verified,
            }
        )
        if not any(item["gate"] == "runtime" for item in blockers):
            report["status"] = "pass"
    except (OSError, GateInputError) as error:
        add_blocker(
            blockers,
            "runtime",
            "runtime_inventory_invalid",
            str(error),
        )
    return report, internal


def preprocessing_check(
    spec: dict[str, Any],
    manifest_path: Path | None,
    voice_ai_internal: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest_path is None and voice_ai_internal.get("ready"):
        archive_sha = voice_ai_internal.get("archive_sha256")
        return (
            {
                "feature_extractor_contract_ready": True,
                "source": "official_voice_ai_archive",
                "status": "pass",
                "tokenizer_ready": True,
                "voice_ai_archive_sha256": archive_sha,
            },
            {
                "manifest_sha256": None,
                "source": "official_voice_ai_archive",
                "voice_ai_archive_sha256": archive_sha,
            },
        )
    manifest, manifest_sha, report = load_optional_manifest(
        manifest_path,
        gate="preprocessing",
        missing_code="preprocessing_manifest_missing",
        blockers=blockers,
    )
    internal: dict[str, Any] = {
        "manifest_sha256": manifest_sha,
        "source": "standalone_manifest",
    }
    if manifest is None or manifest_path is None:
        return report, internal
    try:
        if manifest.get("schema_version") != 1:
            raise GateInputError("preprocessing manifest schema_version must be 1")
        source = manifest.get("source", {})
        expected_repository = spec["preprocessing"]["source_repository"]
        if source.get("repository") != expected_repository:
            raise GateInputError(
                f"preprocessing source repository mismatch: expected "
                f"{expected_repository!r}, got {source.get('repository')!r}"
            )
        revision = source.get("revision")
        if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
            raise GateInputError(
                "preprocessing source revision must be a pinned 40-64 digit hex commit"
            )
        if manifest.get("release_metadata_sha256") != spec["metadata"]["sha256"]:
            raise GateInputError(
                "preprocessing assets are not bound to the pinned release metadata"
            )
        for section in ("tokenizer", "feature_extractor"):
            actual_section = manifest.get(section)
            expected_section = spec["preprocessing"][section]
            if not isinstance(actual_section, dict):
                raise GateInputError(f"preprocessing manifest is missing {section}")
            for key, expected in expected_section.items():
                if actual_section.get(key) != expected:
                    raise GateInputError(
                        f"{section}.{key} mismatch: expected {expected!r}, "
                        f"got {actual_section.get(key)!r}"
                    )
        supported = manifest.get("supported_language_codes")
        if not isinstance(supported, list) or any(
            not isinstance(item, str) for item in supported
        ):
            raise GateInputError("supported_language_codes must be a string list")
        missing_languages = sorted(
            set(spec["preprocessing"]["required_language_codes"]) - set(supported)
        )
        if missing_languages:
            add_blocker(
                blockers,
                "preprocessing",
                "preprocessing_languages_missing",
                "preprocessing assets omit required native language tokens",
                expected=spec["preprocessing"]["required_language_codes"],
                actual=sorted(set(supported)),
            )
        root = resolve_manifest_root(manifest, manifest_path)
        files = manifest.get("files")
        if not isinstance(files, list):
            raise GateInputError("preprocessing manifest files must be a list")
        by_role: dict[str, dict[str, Any]] = {}
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
                raise GateInputError("invalid preprocessing file entry")
            role = entry["role"]
            if role in by_role:
                raise GateInputError(f"duplicate preprocessing asset role: {role}")
            by_role[role] = entry
        verified: dict[str, Any] = {}
        for role in spec["preprocessing"]["required_asset_roles"]:
            entry = by_role.get(role)
            if entry is None:
                add_blocker(
                    blockers,
                    "preprocessing",
                    "preprocessing_asset_role_missing",
                    f"required preprocessing asset role is absent: {role}",
                )
                continue
            try:
                verified[role] = verify_manifest_artifact(root, entry)
            except (OSError, GateInputError) as error:
                add_blocker(
                    blockers,
                    "preprocessing",
                    "preprocessing_asset_invalid",
                    str(error),
                )
        report.update(
            {
                "source_revision": revision,
                "supported_language_codes": sorted(set(supported)),
                "verified_assets": verified,
            }
        )
        internal["verified_assets"] = verified
        if not any(item["gate"] == "preprocessing" for item in blockers):
            report["status"] = "pass"
    except (OSError, GateInputError) as error:
        add_blocker(
            blockers,
            "preprocessing",
            "preprocessing_manifest_invalid",
            str(error),
        )
    return report, internal


def runner_check(
    spec: dict[str, Any],
    manifest_path: Path | None,
    metadata_internal: dict[str, Any],
    voice_ai_internal: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_sha, report = load_optional_manifest(
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
        target = manifest.get("target", {})
        for key, expected in spec["runner"]["target"].items():
            if target.get(key) != expected:
                raise GateInputError(
                    f"runner target {key} mismatch: expected {expected!r}, "
                    f"got {target.get(key)!r}"
                )
        deployment = manifest.get("deployment", {})
        for key, expected in spec["runner"]["deployment"].items():
            if deployment.get(key) != expected:
                raise GateInputError(
                    f"runner deployment {key} mismatch: expected {expected!r}, "
                    f"got {deployment.get(key)!r}"
                )
        root = resolve_manifest_root(manifest, manifest_path)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise GateInputError("runner artifacts must be a list")
        by_role: dict[str, dict[str, Any]] = {}
        for entry in artifacts:
            if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
                raise GateInputError("invalid runner artifact entry")
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
                verified[role] = verify_manifest_artifact(
                    root, entry, expected_machine="aarch64"
                )
            except (OSError, GateInputError) as error:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_artifact_invalid",
                    str(error),
                )

        declared = manifest.get("tensor_contract")
        if not isinstance(declared, dict):
            raise GateInputError("runner manifest is missing tensor_contract")
        expected_metadata_sha = (
            spec["voice_ai"]["metadata"]["sha256"]
            if voice_ai_internal.get("ready")
            else spec["metadata"]["sha256"]
        )
        if declared.get("release_metadata_sha256") != expected_metadata_sha:
            add_blocker(
                blockers,
                "runner",
                "runner_metadata_binding_mismatch",
                "runner is not bound to the pinned release metadata",
                expected=expected_metadata_sha,
                actual=declared.get("release_metadata_sha256"),
            )
        if voice_ai_internal.get("ready"):
            compatibility = manifest.get("voice_ai_compatibility", {})
            required_compatibility = spec["voice_ai"]["runner_compatibility"]
            for key, expected in required_compatibility.items():
                if compatibility.get(key) != expected:
                    add_blocker(
                        blockers,
                        "runner",
                        "runner_voice_ai_compatibility_missing",
                        f"runner Voice AI compatibility differs for {key}",
                        expected=expected,
                        actual=compatibility.get(key),
                    )
            expected_voice_archive = voice_ai_internal.get("archive_sha256")
            if compatibility.get("voice_ai_archive_sha256") != expected_voice_archive:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_voice_ai_archive_binding_mismatch",
                    "runner is not bound to the pinned Voice AI package",
                    expected=expected_voice_archive,
                    actual=compatibility.get("voice_ai_archive_sha256"),
                )
        expected_contract_sha = metadata_internal.get("contract_sha256")
        declared_contract_sha = declared.get("full_contract_sha256")
        if expected_contract_sha is None:
            add_blocker(
                blockers,
                "runner",
                "runner_release_contract_unavailable",
                "release metadata gate did not produce a trusted tensor contract",
            )
        elif declared_contract_sha != expected_contract_sha:
            add_blocker(
                blockers,
                "runner",
                "runner_tensor_descriptor_binding_mismatch",
                "runner tensor descriptor hash differs from the Qualcomm release",
                expected=expected_contract_sha,
                actual=declared_contract_sha,
            )
        runner_names = declared_contract_names(
            declared.get("tensor_names"), spec["metadata"]["layer_count"]
        )
        release_names = metadata_internal.get("contract_names")
        contract_mismatches: list[dict[str, Any]] = []
        if release_names is not None:
            contract_mismatches = compare_contract_names(release_names, runner_names)
            for mismatch in contract_mismatches:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_tensor_names_mismatch",
                    f"runner contract differs for "
                    f"{mismatch['component']}.{mismatch['side']}",
                    expected={"missing_from_runner": mismatch["missing"]},
                    actual={"unexpected_in_runner": mismatch["unexpected"]},
                )
        if declared.get("quantization_parameters_source") != "release_metadata":
            raise GateInputError(
                "runner must declare release_metadata as its quantization source"
            )
        decode_loop = manifest.get("decode_loop", {})
        for key, expected in spec["runner"]["decode_loop"].items():
            if decode_loop.get(key) != expected:
                add_blocker(
                    blockers,
                    "runner",
                    "runner_decode_loop_mismatch",
                    f"runner decode behavior differs for {key}",
                    expected=expected,
                    actual=decode_loop.get(key),
                )
        report.update(
            {
                "contract_mismatches": contract_mismatches,
                "tensor_contract_sha256": declared_contract_sha,
                "verified_artifacts": verified,
            }
        )
        internal["verified_artifacts"] = verified
        if not any(item["gate"] == "runner" for item in blockers):
            report["status"] = "pass"
    except (OSError, GateInputError) as error:
        add_blocker(
            blockers,
            "runner",
            "runner_manifest_invalid",
            str(error),
        )
    return report, internal


def app_uid_check(
    spec: dict[str, Any],
    evidence_path: Path | None,
    runtime_internal: dict[str, Any],
    preprocessing_internal: dict[str, Any],
    runner_internal: dict[str, Any],
    voice_ai_internal: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence, evidence_sha, report = load_optional_manifest(
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
                "app-UID loader/runner probe status is not pass",
                expected="pass",
                actual=evidence.get("status"),
            )
        target = evidence.get("target", {})
        for key, expected in spec["app_uid"]["target"].items():
            if target.get(key) != expected:
                raise GateInputError(
                    f"app-UID target {key} mismatch: expected {expected!r}, "
                    f"got {target.get(key)!r}"
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
                "runner process UID must equal the non-root Android package UID",
                expected=package_uid,
                actual=process_uid,
            )
        qairt_version = evidence.get("qairt_version")
        if not version_at_least(qairt_version, spec["runtime"]["minimum_qairt"]):
            add_blocker(
                blockers,
                "app_uid",
                "app_uid_qairt_too_old",
                "loaded phone runtime is older than the release requirement",
                expected=f">={spec['runtime']['minimum_qairt']}",
                actual=qairt_version,
            )
        artifacts = runner_internal.get("verified_artifacts", {})
        expected_bindings = {
            "release_archive_sha256": spec["release"]["archive_sha256"],
            "release_metadata_sha256": spec["metadata"]["sha256"],
            "runtime_inventory_sha256": runtime_internal.get("manifest_sha256"),
            "runner_manifest_sha256": runner_internal.get("manifest_sha256"),
            "runner_sha256": artifacts.get("whisper_runner", {}).get("sha256"),
            "loader_sha256": artifacts.get("qnn_loader", {}).get("sha256"),
        }
        if preprocessing_internal.get("source") == "official_voice_ai_archive":
            expected_bindings.update(
                {
                    "voice_ai_archive_sha256": voice_ai_internal.get(
                        "archive_sha256"
                    ),
                    "voice_ai_metadata_sha256": spec["voice_ai"]["metadata"][
                        "sha256"
                    ],
                }
            )
        else:
            expected_bindings["preprocessing_manifest_sha256"] = (
                preprocessing_internal.get("manifest_sha256")
            )
        missing_dependencies = sorted(
            key for key, value in expected_bindings.items() if value is None
        )
        if missing_dependencies:
            add_blocker(
                blockers,
                "app_uid",
                "app_uid_dependency_gate_not_passed",
                "app-UID evidence cannot bind missing or invalid gate inputs",
                actual=missing_dependencies,
            )
        bindings = evidence.get("bindings", {})
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
        checks = evidence.get("checks", {})
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
                "qairt_version": qairt_version,
            }
        )
        if not any(item["gate"] == "app_uid" for item in blockers):
            report["status"] = "pass"
    except GateInputError as error:
        add_blocker(
            blockers,
            "app_uid",
            "app_uid_evidence_invalid",
            str(error),
        )
    return report


def evaluate_gate(
    spec: dict[str, Any],
    *,
    archive_path: Path,
    voice_ai_archive: Path | None = None,
    extraction_root: Path | None = None,
    runtime_inventory: Path | None = None,
    preprocessing_manifest: Path | None = None,
    runner_manifest: Path | None = None,
    app_uid_evidence: Path | None = None,
) -> dict[str, Any]:
    validate_spec(spec)
    blockers: list[dict[str, Any]] = []
    archive_report, component_report, archive_internal = archive_checks(
        spec, archive_path, blockers
    )
    extraction_report = extraction_root_check(
        spec, extraction_root, archive_internal.get("metadata_bytes"), blockers
    )
    metadata_report, metadata_internal = metadata_check(
        spec, archive_internal.get("metadata_bytes"), blockers
    )
    voice_ai_report, voice_ai_internal = voice_ai_check(
        spec, voice_ai_archive, blockers
    )
    runtime_report, runtime_internal = runtime_check(
        spec, runtime_inventory, blockers
    )
    preprocessing_report, preprocessing_internal = preprocessing_check(
        spec, preprocessing_manifest, voice_ai_internal, blockers
    )
    runner_report, runner_internal = runner_check(
        spec, runner_manifest, metadata_internal, voice_ai_internal, blockers
    )
    app_uid_report = app_uid_check(
        spec,
        app_uid_evidence,
        runtime_internal,
        preprocessing_internal,
        runner_internal,
        voice_ai_internal,
        blockers,
    )
    ordered_blockers = sorted_blockers(blockers)
    ready = not ordered_blockers
    return {
        "blockers": ordered_blockers,
        "checks": {
            "app_uid": app_uid_report,
            "archive": archive_report,
            "components": component_report,
            "extraction_root": extraction_report,
            "metadata": metadata_report,
            "preprocessing": preprocessing_report,
            "runner": runner_report,
            "runtime": runtime_report,
            "voice_ai": voice_ai_report,
        },
        "gate_id": spec["gate_id"],
        "policy": {
            "adb_invoked": False,
            "archive_extracted": False,
            "model_loaded": False,
            "network_used": False,
            "phone_mutated": False,
        },
        "ready": ready,
        "schema_version": 1,
        "status": "ready" if ready else "blocked",
    }


def error_report(error: Exception) -> dict[str, Any]:
    return {
        "blockers": [
            {
                "code": "gate_spec_or_invocation_invalid",
                "detail": str(error),
                "gate": "spec",
            }
        ],
        "checks": {},
        "gate_id": "nubia_whisper_small_w8a16_app_uid",
        "policy": {
            "adb_invoked": False,
            "archive_extracted": False,
            "model_loaded": False,
            "network_used": False,
            "phone_mutated": False,
        },
        "ready": False,
        "schema_version": 1,
        "status": "invalid",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--voice-ai-archive",
        type=Path,
        help="optional pinned Voice AI package supplying config.json and vocab.bin",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--extraction-root",
        type=Path,
        help="inspect an existing extraction in addition to the ZIP; never extracts",
    )
    parser.add_argument("--runtime-inventory", type=Path)
    parser.add_argument("--preprocessing-manifest", type=Path)
    parser.add_argument("--runner-manifest", type=Path)
    parser.add_argument("--app-uid-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec, _ = load_json_object(args.spec)
        result = evaluate_gate(
            spec,
            archive_path=args.archive.expanduser(),
            voice_ai_archive=(
                args.voice_ai_archive.expanduser() if args.voice_ai_archive else None
            ),
            extraction_root=(
                args.extraction_root.expanduser() if args.extraction_root else None
            ),
            runtime_inventory=(
                args.runtime_inventory.expanduser() if args.runtime_inventory else None
            ),
            preprocessing_manifest=(
                args.preprocessing_manifest.expanduser()
                if args.preprocessing_manifest
                else None
            ),
            runner_manifest=(
                args.runner_manifest.expanduser() if args.runner_manifest else None
            ),
            app_uid_evidence=(
                args.app_uid_evidence.expanduser() if args.app_uid_evidence else None
            ),
        )
        exit_code = 0 if result["ready"] else 1
    except (OSError, GateInputError, KeyError, TypeError, ValueError) as error:
        result = error_report(error)
        exit_code = 2
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
