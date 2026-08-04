#!/usr/bin/env python3
"""Resume-safe QAIRT 2.42 Qwen2.5 GGUF-to-Genie HTP Colab lane."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_qwen25_genie_spec.json"
DEFAULT_SOURCE_LOCK = ROOT / "reports/android_phone/model_sweep_20260729/qairt242_qwen25_15b_genie_source_lock.json"
EXPECTED_SPEC_SHA256 = "8b25f22513a78aa9cea7898720ead903e41a77ef4475506450e17fa816c4565a"
EXPECTED_EXPERIMENT_ID = "qairt242_colab_qwen25_genie_sm8850_htp_v81_v1"
EXPECTED_QAIRT_SHA256 = "1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321"
SCHEMA_VERSION = 1
NEXT_GATE = (
    "Stage the exact container and pinned QAIRT 2.42 Android HTP/Genie runtime "
    "in an isolated app-owned phone directory; disable CPU fallback and capture "
    "app-UID HTP/FastRPC/DMA plus latency and multilingual quality evidence."
)
RESULT_MANIFEST_FIELDS = frozenset({
    "schema_version", "experiment_id", "status", "claim_boundary",
    "created_utc", "spec_sha256", "controller_sha256",
    "source_lock_sha256", "source", "source_gguf", "host",
    "python_packages", "target", "builder", "container_root",
    "container_artifacts", "qairt_archive_sha256",
    "qairt_verified_members", "android_runtime_inventory",
    "runtime_binaries_bundled", "phone_execution_proven",
    "npu_execution_proven", "next_gate",
})


class LaneError(RuntimeError):
    """A fail-closed validation or execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaneError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_sha256(value: Any, label: str) -> str:
    text = str(value)
    require(bool(re.fullmatch(r"[0-9a-f]{64}", text)), f"invalid {label} SHA-256")
    return text


def validate_spec(spec: dict[str, Any], path: Path | None = None) -> None:
    require(spec.get("schema_version") == 1, "unexpected spec schema")
    require(spec.get("experiment_id") == EXPECTED_EXPERIMENT_ID, "unexpected experiment identity")
    if path is not None:
        require(sha256_file(path) == EXPECTED_SPEC_SHA256, "spec SHA-256 does not match controller trust root")
    policy = spec["execution_policy"]
    require(policy["mac_execution_allowed"] is False, "macOS execution must stay blocked")
    require(policy["credentials_required"] is False, "credentials are forbidden")
    require(all(value is False for value in policy["fallback_policy"].values()), "a fallback was enabled")
    qairt = spec["qairt"]
    require(qairt["version"] == "2.42.0.251225", "QAIRT version changed")
    require(qairt["sha256"] == EXPECTED_QAIRT_SHA256, "QAIRT identity changed")
    require(qairt["bytes"] == 1543955191, "QAIRT size changed")
    primary = spec["source_profiles"]["qwen2_5_1_5b_instruct_q4_k_m"]
    require(primary["role"] == "runnable_primary", "primary source role changed")
    contract = primary["gguf_contract"]
    require(contract["tensor_types"] == {"F32": 141, "Q4_K": 169, "Q6_K": 29}, "mixed GGUF tensor contract changed")
    builder = spec["builder_contract"]
    require(builder["backend"] == "HTP", "backend changed")
    require(builder["target_string"] == "chipset:SM8850;dsp_arch:v81;soc_model:87", "target changed")
    require(spec["target"] == {"chipset": "SM8850", "soc_model": 87, "dsp_arch": "v81", "backend": "HTP", "target_abi": "aarch64-android"}, "target contract changed")
    output = spec["output_contract"]
    require(output == {"required_manifest_status": "context_generated_unexecuted", "runtime_binaries_bundled": False, "phone_execution_proven": False, "npu_execution_proven": False}, "claim boundary changed")


def load_spec(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    spec = read_json(resolved)
    require(isinstance(spec, dict), "spec root must be an object")
    validate_spec(spec, resolved)
    return resolved, spec


def validate_source_lock(lock: dict[str, Any], spec: dict[str, Any], require_approved: bool) -> dict[str, Any]:
    require(lock.get("schema_version") == 1, "unexpected source-lock schema")
    profile_id = str(lock.get("profile_id", ""))
    require(profile_id in spec["source_profiles"], "unknown source profile")
    profile = spec["source_profiles"][profile_id]
    source = lock.get("source", {})
    require(source.get("repository") == profile["repository"], "source repository changed")
    require(source.get("filename") == profile["filename"], "source filename changed")
    approval = lock.get("approval", {})
    approved = approval.get("approved_for_conversion") is True
    if not approved:
        require(not require_approved, "source lock is not approved for conversion")
        return {"profile_id": profile_id, "approved": False, "missing_identity": [key for key in ("revision", "url", "bytes", "sha256") if not source.get(key)]}
    require(approval.get("status") != "unapproved_template", "template cannot self-promote")
    require(bool(approval.get("approved_by")), "approved lock lacks reviewer/evidence")
    validate_sha256(approval.get("evidence_sha256"), "approval evidence")
    revision = str(source.get("revision", ""))
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "source revision must be an immutable 40-hex commit")
    size = source.get("bytes")
    require(isinstance(size, int) and size > 100_000_000, "source byte size is missing or implausible")
    digest = validate_sha256(source.get("sha256"), "source")
    expected_url = f"https://huggingface.co/{profile['repository']}/resolve/{revision}/{profile['filename']}?download=true"
    require(source.get("url") == expected_url, "source URL is not revision-pinned")
    expected_identity = profile.get("expected_identity")
    if expected_identity is not None:
        require({"revision": revision, "bytes": size, "sha256": digest} == expected_identity, "approved source identity differs from pinned profile")
    return {"profile_id": profile_id, "approved": True, "revision": revision, "bytes": size, "sha256": digest, "url": expected_url}


def load_source_lock(path: Path, spec: dict[str, Any], require_approved: bool) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    lock = read_json(resolved)
    require(isinstance(lock, dict), "source-lock root must be an object")
    identity = validate_source_lock(lock, spec, require_approved)
    return resolved, lock, identity


GGUF_VALUE_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K"}


def inspect_gguf(path: Path) -> dict[str, Any]:
    wanted_scalars = {
        "general.architecture", "general.name", "general.file_type", "general.quantization_version",
        "qwen2.context_length", "qwen2.embedding_length", "qwen2.feed_forward_length", "qwen2.block_count",
        "qwen2.attention.head_count", "qwen2.attention.head_count_kv", "qwen2.attention.layer_norm_rms_epsilon",
        "qwen2.rope.freq_base", "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
    }
    wanted_arrays = {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}
    with path.open("rb") as stream:
        require(stream.read(4) == b"GGUF", "invalid GGUF magic")
        version = struct.unpack("<I", stream.read(4))[0]
        tensor_count, kv_count = struct.unpack("<QQ", stream.read(16))

        def read_u64() -> int:
            data = stream.read(8)
            require(len(data) == 8, "truncated GGUF")
            return struct.unpack("<Q", data)[0]

        def read_string(capture: bool) -> str | None:
            size = read_u64()
            require(size < 64 * 1024 * 1024, "implausible GGUF string")
            data = stream.read(size)
            require(len(data) == size, "truncated GGUF string")
            return data.decode("utf-8") if capture else None

        def read_value(value_type: int, capture: bool) -> Any:
            if value_type in GGUF_VALUE_FORMATS:
                fmt = GGUF_VALUE_FORMATS[value_type]
                size = struct.calcsize(fmt)
                data = stream.read(size)
                require(len(data) == size, "truncated GGUF scalar")
                value = struct.unpack("<" + fmt, data)[0]
                return value if capture else None
            if value_type == 8:
                return read_string(capture)
            if value_type == 9:
                element_type = struct.unpack("<I", stream.read(4))[0]
                count = read_u64()
                require(count < 10_000_000, "implausible GGUF array")
                for _ in range(count):
                    read_value(element_type, False)
                return {"element_type": element_type, "count": count} if capture else None
            raise LaneError(f"unsupported GGUF metadata value type: {value_type}")

        metadata: dict[str, Any] = {}
        arrays: dict[str, int] = {}
        for _ in range(kv_count):
            key = read_string(True)
            require(isinstance(key, str), "invalid GGUF key")
            raw_type = stream.read(4)
            require(len(raw_type) == 4, "truncated GGUF metadata type")
            value_type = struct.unpack("<I", raw_type)[0]
            capture = key in wanted_scalars or key in wanted_arrays
            value = read_value(value_type, capture)
            if key in wanted_scalars:
                metadata[key] = value
            elif key in wanted_arrays:
                arrays[key] = int(value["count"])

        tensor_types: Counter[str] = Counter()
        tensor_names: list[str] = []
        for _ in range(tensor_count):
            name = read_string(True)
            require(isinstance(name, str), "invalid GGUF tensor name")
            raw_dims = stream.read(4)
            require(len(raw_dims) == 4, "truncated GGUF tensor rank")
            dimensions = struct.unpack("<I", raw_dims)[0]
            require(0 < dimensions <= 8, "invalid GGUF tensor rank")
            require(len(stream.read(8 * dimensions)) == 8 * dimensions, "truncated GGUF tensor shape")
            raw_type = stream.read(4)
            require(len(raw_type) == 4, "truncated GGUF tensor type")
            type_id = struct.unpack("<I", raw_type)[0]
            require(type_id in GGML_TYPES, f"unknown GGML tensor type id: {type_id}")
            require(len(stream.read(8)) == 8, "truncated GGUF tensor offset")
            tensor_types[GGML_TYPES[type_id]] += 1
            tensor_names.append(name)
    return {"version": version, "tensor_count": tensor_count, "metadata_kv_count": kv_count, "metadata": metadata, "array_counts": arrays, "tensor_types": dict(sorted(tensor_types.items())), "tensor_names": tensor_names}


def expected_qwen_tensor_names(layers: int) -> set[str]:
    names = {"output.weight", "token_embd.weight", "output_norm.weight"}
    suffixes = (
        "attn_norm.weight", "ffn_down.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_norm.weight",
        "attn_k.bias", "attn_k.weight", "attn_output.weight", "attn_q.bias", "attn_q.weight", "attn_v.bias", "attn_v.weight",
    )
    for layer in range(layers):
        names.update(f"blk.{layer}.{suffix}" for suffix in suffixes)
    return names


def validate_gguf_contract(record: dict[str, Any], contract: dict[str, Any]) -> None:
    for key in ("version", "tensor_count", "metadata_kv_count", "tensor_types"):
        require(record[key] == contract[key], f"GGUF {key} drift")
    metadata = record["metadata"]
    checks = {
        "general.architecture": contract["architecture"], "general.name": contract["model_name"],
        "qwen2.context_length": contract["context_length"], "qwen2.embedding_length": contract["hidden_size"],
        "qwen2.feed_forward_length": contract["intermediate_size"], "qwen2.block_count": contract["num_hidden_layers"],
        "qwen2.attention.head_count": contract["num_attention_heads"], "qwen2.attention.head_count_kv": contract["num_key_value_heads"],
        "qwen2.rope.freq_base": contract["rope_theta"], "tokenizer.ggml.model": contract["tokenizer_model"],
        "tokenizer.ggml.pre": contract["tokenizer_pre"], "tokenizer.ggml.bos_token_id": contract["bos_token_id"],
        "tokenizer.ggml.eos_token_id": contract["eos_token_id"], "tokenizer.ggml.padding_token_id": contract["padding_token_id"],
    }
    for key, expected in checks.items():
        require(metadata.get(key) == expected, f"GGUF metadata drift: {key}")
    require(abs(float(metadata["qwen2.attention.layer_norm_rms_epsilon"]) - contract["rms_norm_eps"]) < 1e-12, "GGUF RMS epsilon drift")
    require(contract["hidden_size"] // contract["num_attention_heads"] == contract["head_dim"], "head dimension contract is inconsistent")
    arrays = record["array_counts"]
    require(arrays == {"tokenizer.ggml.merges": contract["merge_count"], "tokenizer.ggml.token_type": contract["token_type_count"], "tokenizer.ggml.tokens": contract["token_count"]}, "tokenizer array counts drift")
    require(set(record["tensor_names"]) == expected_qwen_tensor_names(contract["num_hidden_layers"]), "GGUF tensor-name contract drift")


def safe_workspace(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    require(resolved != Path("/") and len(resolved.parts) >= 3, "workspace path is too broad")
    return resolved


def relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise LaneError(f"path escapes workspace: {path}") from error


def inventory_tree(root: Path, workspace: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append({"path": relative_to_workspace(path, workspace), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    require(records, f"no files produced under {root}")
    return records


def verify_inventory(records: Any, workspace: Path) -> None:
    require(isinstance(records, list) and bool(records), "empty output inventory")
    seen: set[str] = set()
    for record in records:
        require(
            isinstance(record, dict)
            and set(record) == {"path", "bytes", "sha256"},
            "output inventory record drift",
        )
        require(isinstance(record["path"], str) and record["path"], "invalid inventory path")
        relative = Path(record["path"])
        require(not relative.is_absolute(), "absolute inventory path")
        path = (workspace / relative).resolve()
        require(
            relative_to_workspace(path, workspace) == relative.as_posix(),
            f"non-canonical inventory path: {relative}",
        )
        require(relative.as_posix() not in seen, f"duplicate inventory path: {relative}")
        seen.add(relative.as_posix())
        require(
            isinstance(record["bytes"], int)
            and not isinstance(record["bytes"], bool)
            and record["bytes"] >= 0,
            f"invalid inventory size: {relative}",
        )
        validate_sha256(record["sha256"], f"inventory {relative}")
        require(path.is_file(), f"missing output: {relative}")
        require(path.stat().st_size == record["bytes"], f"size drift: {relative}")
        require(sha256_file(path) == record["sha256"], f"hash drift: {relative}")


def quarantine(path: Path, workspace: Path, reason: str) -> Path:
    relative_to_workspace(path, workspace)
    target_root = workspace / "quarantine"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{path.name}.{reason}.{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{os.getpid()}"
    path.replace(target)
    return target


@contextmanager
def workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / "qairt242_qwen25.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaneError("another process holds the workspace lock") from error
        yield
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND_JSON=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise LaneError(f"command failed ({result.returncode}): {command[0]}\n{log.read_text(encoding='utf-8', errors='replace')[-8000:]}")


def verified_artifact(name: str, record: dict[str, Any], workspace: Path, override: Path | None = None) -> Path:
    downloads = workspace / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    final = downloads / record["filename"]
    expected_hash = validate_sha256(record["sha256"], name)
    expected_bytes = int(record["bytes"])
    if final.is_file():
        require(final.stat().st_size == expected_bytes and sha256_file(final) == expected_hash, f"existing {name} identity drift")
        return final
    if override is not None:
        source = override.expanduser().resolve()
        require(source.is_file(), f"missing {name} override: {source}")
        require(source.stat().st_size == expected_bytes and sha256_file(source) == expected_hash, f"{name} override identity mismatch")
        shutil.copyfile(source, final)
        return final
    partial = final.with_suffix(final.suffix + ".part")
    command = ["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors", "--continue-at", "-", "--output", str(partial), record["url"]]
    run_logged(command, downloads, dict(os.environ), downloads / f"{final.name}.download.log")
    require(partial.stat().st_size == expected_bytes and sha256_file(partial) == expected_hash, f"downloaded {name} identity mismatch")
    partial.replace(final)
    return final


def validate_sdk_semantics(sdk_root: Path) -> None:
    factory = (sdk_root / "lib/python/qairt/gen_ai_api/gen_ai_builder_factory.py").read_text(encoding="utf-8")
    utils = (sdk_root / "lib/python/qti/aisw/converters/gguf_builder/utils.py").read_text(encoding="utf-8")
    ggml = (sdk_root / "lib/python/qti/aisw/converters/gguf_builder/ggml.py").read_text(encoding="utf-8")
    qwen = (sdk_root / "lib/python/qairt/gen_ai_api/builders/qwen/builder.py").read_text(encoding="utf-8")
    for needle in ("GGUFBuilder(pretrained_model_path).build_from_gguf()", "QwenBuilderHTP.from_pretrained", 'QWEN = "Qwen2ForCausalLM"'):
        require(needle in factory, f"factory evidence missing: {needle}")
    for needle in ('"qwen2": "Qwen2ForCausalLM"', '"Q4_K"', '"Q6_K"'):
        require(needle in utils, f"GGUF support evidence missing: {needle}")
    for needle in ("def dequantize_q4_k", "def dequantize_q6_k", 'ggml_type == GGML_TYPES["Q4_K"]', 'ggml_type == GGML_TYPES["Q6_K"]'):
        require(needle in ggml, f"mixed-tensor dispatch evidence missing: {needle}")
    require("class QwenBuilderHTP(GenAIBuilderHTP)" in qwen, "Qwen HTP builder evidence missing")


def prepare_sdk(archive: Path, scratch: Path, spec: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    container = scratch / "qairt242-sdk"
    sdk_root = container / spec["qairt"]["sdk_root_member"]
    marker_path = container / ".verified.json"
    expected = spec["qairt"]["critical_members"] + spec["qairt"]["android_runtime_inventory"]
    if marker_path.is_file() and sdk_root.is_dir():
        marker = read_json(marker_path)
        if marker.get("archive_sha256") == EXPECTED_QAIRT_SHA256 and marker.get("members") == expected:
            for item in expected:
                path = sdk_root / item["path"]
                require(path.is_file() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"], f"SDK member drift: {item['path']}")
            validate_sdk_semantics(sdk_root)
            return sdk_root, expected
        container.replace(scratch / f"qairt242-sdk.invalid.{os.getpid()}")
    partial = scratch / f".qairt242-sdk.partial.{os.getpid()}"
    if partial.exists():
        partial.replace(scratch / f"qairt242-sdk.interrupted.{os.getpid()}")
    partial.mkdir(parents=True)
    result = subprocess.run(["unzip", "-q", str(archive), "-d", str(partial)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    require(result.returncode == 0, f"QAIRT extraction failed: {result.stdout[-4000:]}")
    extracted = partial / spec["qairt"]["sdk_root_member"]
    for item in expected:
        path = extracted / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"], f"SDK member identity mismatch: {item['path']}")
    validate_sdk_semantics(extracted)
    atomic_write_json(partial / ".verified.json", {"archive_sha256": EXPECTED_QAIRT_SHA256, "members": expected})
    partial.replace(container)
    return container / spec["qairt"]["sdk_root_member"], expected


def sdk_env(sdk_root: Path, venv: Path, qairt_tmp: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "QAIRT_SDK_ROOT": str(sdk_root), "QNN_SDK_ROOT": str(sdk_root), "SNPE_ROOT": str(sdk_root),
        "QAIRT_TMP_DIR": str(qairt_tmp), "VIRTUAL_ENV": str(venv),
        "PATH": os.pathsep.join([str(venv / "bin"), str(sdk_root / "bin/x86_64-linux-clang"), env.get("PATH", "")]),
        "PYTHONPATH": os.pathsep.join([str(sdk_root / "lib/python"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        "LD_LIBRARY_PATH": os.pathsep.join([str(sdk_root / "lib/x86_64-linux-clang"), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep),
        "CC": shutil.which("clang-14") or "clang-14", "CXX": shutil.which("clang++-14") or "clang++-14",
    })
    return env


def package_inventory(python: Path, env: dict[str, str], pins: dict[str, str]) -> dict[str, str]:
    code = "import importlib.metadata as m,json,sys; print(json.dumps({n:m.version(n) for n in json.loads(sys.argv[1])},sort_keys=True))"
    result = subprocess.run([str(python), "-c", code, json.dumps(sorted(pins))], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    require(result.returncode == 0, f"could not inventory packages: {result.stdout}")
    installed = json.loads(result.stdout)
    for name, expected in pins.items():
        actual = installed.get(name)
        require(actual == expected or (name == "torch" and str(actual).startswith(expected + "+")), f"package drift: {name}={actual}")
    return installed


def prepare_venv(scratch: Path, sdk_root: Path, spec: dict[str, Any]) -> tuple[Path, dict[str, str], dict[str, str]]:
    venv = scratch / "qairt242-qwen-python310"
    qairt_tmp = scratch / "qairt-tmp"
    qairt_tmp.mkdir(parents=True, exist_ok=True)
    pins = spec["host_dependencies"]["python_packages"]
    marker_path = venv / ".verified.json"
    expected_marker = {"qairt_sha256": EXPECTED_QAIRT_SHA256, "pins": pins, "overrides": spec["host_dependencies"]["qairt_checker_compatibility_overrides"]}
    if marker_path.is_file() and (venv / "bin/python").is_file() and read_json(marker_path) == expected_marker:
        env = sdk_env(sdk_root, venv, qairt_tmp)
        return venv, env, package_inventory(venv / "bin/python", env, pins)
    if venv.exists():
        venv.replace(scratch / f"qairt242-qwen-python310.invalid.{os.getpid()}")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    env = sdk_env(sdk_root, venv, qairt_tmp)
    python = venv / "bin/python"
    run_logged([str(python), "-m", "pip", "install", "--upgrade", "pip==24.0", "setuptools==69.5.1", "wheel==0.43.0"], scratch, env, scratch / "pip-bootstrap.log")
    overrides = spec["host_dependencies"]["qairt_checker_compatibility_overrides"]
    run_logged([str(python), "-m", "pip", "install", "--only-binary=:all:", *[f"{name}=={version}" for name, version in sorted(overrides.items())]], scratch, env, scratch / "pip-qairt-overrides.log")
    run_logged([str(python), str(sdk_root / "bin/check-python-dependency")], scratch, env, scratch / "qairt-dependencies.log")
    run_logged([str(python), "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", f"torch=={pins['torch']}"], scratch, env, scratch / "pip-torch.log")
    remaining = [f"{name}=={version}" for name, version in sorted(pins.items()) if name != "torch"]
    run_logged([str(python), "-m", "pip", "install", "--upgrade", *remaining], scratch, env, scratch / "pip-genai.log")
    packages = package_inventory(python, env, pins)
    atomic_write_json(marker_path, expected_marker)
    return venv, env, packages


def parse_os_release() -> dict[str, str]:
    values = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def total_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise LaneError("could not determine total RAM")


def host_preflight(workspace: Path, scratch: Path, spec: dict[str, Any]) -> dict[str, Any]:
    require(platform.system() == "Linux", "run is allowed only on Linux/Colab")
    require(platform.machine().lower() in {"x86_64", "amd64"}, "QAIRT host tools require x86_64 Linux")
    require(parse_os_release().get("VERSION_ID") == "22.04", "QAIRT 2.42 lane requires Ubuntu 22.04")
    require(sys.version_info[:2] == (3, 10), "invoke this controller with Python 3.10")
    missing = [name for name in ("curl", "unzip", "clang-14", "clang++-14", "cmake") if shutil.which(name) is None]
    require(not missing, f"missing host commands: {missing}")
    workspace.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    policy = spec["execution_policy"]
    memory = total_memory_bytes()
    require(memory >= policy["minimum_ram_bytes"], f"high-RAM Colab required; found {memory} bytes")
    require(shutil.disk_usage(workspace).free >= policy["minimum_workspace_free_bytes"], "insufficient durable workspace space")
    require(shutil.disk_usage(scratch).free >= policy["minimum_scratch_free_bytes"], "insufficient scratch space")
    return {"platform": platform.platform(), "python": platform.python_version(), "ram_bytes": memory, "workspace_free_bytes": shutil.disk_usage(workspace).free, "scratch_free_bytes": shutil.disk_usage(scratch).free}


BUILD_HELPER = r'''
import json
import math
import os
from pathlib import Path
import sys

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gguf_path = Path(sys.argv[2])
cache_root = Path(sys.argv[3])
output = Path(sys.argv[4])
result_path = Path(sys.argv[5])

from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
from qairt.gen_ai_api.builders.qwen.builder import QwenBuilderHTP
from qairt.gen_ai_api.containers.llm_container import LLMContainer

builder = GenAIBuilderFactory.create(str(gguf_path), "HTP", cache_root=str(cache_root))
if not isinstance(builder, QwenBuilderHTP):
    raise RuntimeError(f"expected QwenBuilderHTP, got {type(builder)!r}")
expected = cfg["model"]
actual = {
    "n_vocab": builder.config.n_vocab,
    "n_heads": builder.config.n_heads,
    "n_layer": builder.config.n_layer,
    "n_embd": builder.config.n_embd,
    "bos_token": builder.config.bos_token,
    "eos_token": builder.config.eos_token,
    "rope_theta": builder.config.rope_theta,
}
required = {
    "n_vocab": expected["vocab_size"], "n_heads": expected["num_attention_heads"],
    "n_layer": expected["num_hidden_layers"], "n_embd": expected["hidden_size"],
    "bos_token": expected["bos_token_id"], "eos_token": expected["eos_token_id"],
    "rope_theta": expected["rope_theta"],
}
for key, value in required.items():
    if isinstance(value, float):
        if not math.isclose(float(actual[key]), value, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"builder config drift: {key}={actual[key]}")
    elif actual[key] != value:
        raise RuntimeError(f"builder config drift: {key}={actual[key]}")
builder.set_targets([cfg["target_string"]])
container = builder.build()
if not isinstance(container, LLMContainer):
    raise RuntimeError(f"expected LLMContainer, got {type(container)!r}")
container.save(str(output), exist_ok=False)
LLMContainer.load(str(output))
result_path.write_text(json.dumps({
    "builder_class": f"{type(builder).__module__}.{type(builder).__name__}",
    "container_class": f"{type(container).__module__}.{type(container).__name__}",
    "target_string": cfg["target_string"], "builder_config": actual,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'''


def validate_container(container: Path, spec: dict[str, Any]) -> list[Path]:
    for relative in spec["builder_contract"]["required_container_files"]:
        require((container / relative).is_file(), f"container lacks {relative}")
    contexts = sorted(container.glob(spec["builder_contract"]["required_context_glob"]))
    require(contexts, "container has no HTP context binaries")
    require(all(path.stat().st_size > 1024 for path in contexts), "an HTP context binary is unexpectedly small")
    metadata = read_json(container / "metadata.json")
    require(str(metadata.get("backend", "")).upper() == "HTP", "container metadata is not HTP")
    return contexts


def initial_state(spec_path: Path, source_lock_path: Path, source_identity: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "experiment_id": EXPECTED_EXPERIMENT_ID, "spec_sha256": EXPECTED_SPEC_SHA256, "controller_sha256": sha256_file(Path(__file__).resolve()), "source_lock_sha256": sha256_file(source_lock_path), "source_identity": source_identity, "created_utc": utc_now(), "updated_utc": utc_now(), "status": "initialized", "stages": {}}


def load_or_create_state(path: Path, spec_path: Path, source_lock_path: Path, source_identity: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        state = read_json(path)
        require(state.get("experiment_id") == EXPECTED_EXPERIMENT_ID, "state belongs to another experiment")
        require(state.get("spec_sha256") == EXPECTED_SPEC_SHA256, "state spec drift")
        require(state.get("controller_sha256") == sha256_file(Path(__file__).resolve()), "controller changed; use a new workspace")
        require(state.get("source_lock_sha256") == sha256_file(source_lock_path), "source lock changed; use a new workspace")
        require(state.get("source_identity") == source_identity, "state source identity drift")
        return state
    state = initial_state(spec_path, source_lock_path, source_identity)
    atomic_write_json(path, state)
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    atomic_write_json(path, state)


def build_fingerprint(spec: dict[str, Any], source_identity: dict[str, Any]) -> str:
    return canonical_sha256({
        "spec": EXPECTED_SPEC_SHA256,
        "source": source_identity,
        "qairt": EXPECTED_QAIRT_SHA256,
        "target": spec["builder_contract"]["target_string"],
    })


def deliverable_fingerprint(
    build_outputs: list[dict[str, Any]], controller_sha256: str
) -> str:
    return canonical_sha256({
        "build_outputs": build_outputs,
        "controller": controller_sha256,
    })


def expected_builder_config(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_vocab": contract["vocab_size"],
        "n_heads": contract["num_attention_heads"],
        "n_layer": contract["num_hidden_layers"],
        "n_embd": contract["hidden_size"],
        "bos_token": contract["bos_token_id"],
        "eos_token": contract["eos_token_id"],
        "rope_theta": contract["rope_theta"],
    }


def build_stage(workspace: Path, state_path: Path, state: dict[str, Any], gguf: Path, sdk_root: Path, venv: Path, env: dict[str, str], spec: dict[str, Any], source_identity: dict[str, Any]) -> dict[str, Any]:
    fingerprint = build_fingerprint(spec, source_identity)
    stages = state.setdefault("stages", {})
    existing = stages.get("container_build")
    if existing and existing.get("status") == "complete":
        require(existing.get("fingerprint") == fingerprint, "completed build fingerprint drift")
        verify_inventory(existing["outputs"], workspace)
        return existing
    if existing:
        require(existing.get("fingerprint") == fingerprint, "incomplete build fingerprint drift")
    final = workspace / "stages/container_build"
    if final.exists():
        quarantine(final, workspace, "orphaned")
    stages_root = workspace / "stages"
    stages_root.mkdir(parents=True, exist_ok=True)
    for stale in stages_root.glob(".container_build.partial.*"):
        quarantine(stale, workspace, "interrupted")
    partial = stages_root / f".container_build.partial.{os.getpid()}"
    partial.mkdir()
    stages["container_build"] = {"status": "in_progress", "fingerprint": fingerprint, "started_utc": utc_now()}
    write_state(state_path, state)
    helper = partial / "build_qwen_container.py"
    atomic_write_text(helper, BUILD_HELPER)
    helper_config = partial / "build_config.json"
    profile = spec["source_profiles"][source_identity["profile_id"]]
    atomic_write_json(helper_config, {"model": profile["gguf_contract"], "target_string": spec["builder_contract"]["target_string"]})
    cache_root = workspace / "qairt_cache"
    cache_root.mkdir(exist_ok=True)
    output = partial / "container"
    result_path = partial / "builder_result.json"
    command = [str(venv / "bin/python"), str(helper), str(helper_config), str(gguf), str(cache_root), str(output), str(result_path)]
    try:
        run_logged(command, partial, env, partial / "build.log")
        contexts = validate_container(output, spec)
        partial.replace(final)
        outputs = inventory_tree(final, workspace)
        gguf_artifacts = gguf.parent / f"gguf_artifacts_{gguf.stem}"
        require(gguf_artifacts.is_dir(), "QAIRT GGUFBuilder produced no durable GGUF artifacts")
        outputs.extend(inventory_tree(gguf_artifacts, workspace))
        record = {"status": "complete", "fingerprint": fingerprint, "started_utc": stages["container_build"]["started_utc"], "completed_utc": utc_now(), "details": {"builder": read_json(final / "builder_result.json"), "context_count": len(contexts), "gguf_artifacts": relative_to_workspace(gguf_artifacts, workspace)}, "outputs": outputs}
        stages["container_build"] = record
        write_state(state_path, state)
        return record
    except Exception as error:
        stages["container_build"] = {"status": "failed", "fingerprint": fingerprint, "started_utc": stages["container_build"]["started_utc"], "failed_utc": utc_now(), "error": {"type": type(error).__name__, "message": str(error)}, "resume_cache": relative_to_workspace(cache_root, workspace)}
        write_state(state_path, state)
        raise


def create_deliverable(workspace: Path, state_path: Path, state: dict[str, Any], spec_path: Path, source_lock_path: Path, spec: dict[str, Any], build: dict[str, Any], host: dict[str, Any], packages: dict[str, str], gguf_record: dict[str, Any], sdk_members: list[dict[str, Any]]) -> dict[str, Any]:
    fingerprint = deliverable_fingerprint(
        build["outputs"], state["controller_sha256"]
    )
    existing = state["stages"].get("deliverable")
    if existing and existing.get("status") == "complete":
        require(existing.get("fingerprint") == fingerprint, "deliverable fingerprint drift")
        verify_inventory(existing["outputs"], workspace)
        return existing
    final = workspace / "stages/deliverable"
    if final.exists():
        quarantine(final, workspace, "orphaned")
    partial = workspace / "stages" / f".deliverable.partial.{os.getpid()}"
    partial.mkdir(parents=True)
    container_inventory = [item for item in build["outputs"] if "/container/" in item["path"]]
    manifest = {
        "schema_version": 1, "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": spec["output_contract"]["required_manifest_status"], "claim_boundary": spec["claim_boundary"],
        "created_utc": utc_now(), "spec_sha256": EXPECTED_SPEC_SHA256, "controller_sha256": state["controller_sha256"],
        "source_lock_sha256": state["source_lock_sha256"], "source": state["source_identity"], "source_gguf": gguf_record,
        "host": host, "python_packages": packages, "target": spec["target"], "builder": build["details"],
        "container_root": "stages/container_build/container", "container_artifacts": container_inventory,
        "qairt_archive_sha256": EXPECTED_QAIRT_SHA256, "qairt_verified_members": sdk_members,
        "android_runtime_inventory": spec["qairt"]["android_runtime_inventory"], "runtime_binaries_bundled": False,
        "phone_execution_proven": False, "npu_execution_proven": False,
        "next_gate": NEXT_GATE,
    }
    atomic_write_json(partial / "result_manifest.json", manifest)
    shutil.copy2(spec_path, partial / spec_path.name)
    shutil.copy2(source_lock_path, partial / source_lock_path.name)
    sums = "".join(f"{item['sha256']}  {item['path']}\n" for item in container_inventory)
    atomic_write_text(partial / "container.SHA256SUMS", sums)
    partial.replace(final)
    record = {"status": "complete", "fingerprint": fingerprint, "completed_utc": utc_now(), "details": {"manifest_status": manifest["status"], "phone_execution_proven": False, "npu_execution_proven": False}, "outputs": inventory_tree(final, workspace)}
    state["stages"]["deliverable"] = record
    write_state(state_path, state)
    return record


def verify_completed(
    workspace: Path,
    state: dict[str, Any],
    spec: dict[str, Any],
    source_lock_path: Path,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    require(isinstance(state, dict), "lane state root must be an object")
    require(
        set(state) == {
            "schema_version", "experiment_id", "spec_sha256",
            "controller_sha256", "source_lock_sha256", "source_identity",
            "created_utc", "updated_utc", "status", "stages", "host",
            "completed_utc", "result",
        },
        "lane state field inventory drift",
    )
    require(state.get("schema_version") == SCHEMA_VERSION, "lane state schema drift")
    require(state.get("experiment_id") == EXPECTED_EXPERIMENT_ID, "lane state experiment drift")
    require(state.get("spec_sha256") == EXPECTED_SPEC_SHA256, "lane state spec drift")
    require(
        state.get("controller_sha256") == sha256_file(Path(__file__).resolve()),
        "lane state controller drift",
    )
    require(
        state.get("source_lock_sha256") == sha256_file(source_lock_path),
        "lane state source-lock drift",
    )
    require(state.get("source_identity") == source_identity, "lane state source identity drift")
    require(state.get("status") == "complete", "lane state is not complete")
    for key in ("created_utc", "updated_utc", "completed_utc"):
        require(isinstance(state[key], str) and state[key], f"lane state {key} is missing")

    host = state["host"]
    require(
        isinstance(host, dict)
        and set(host) == {
            "platform", "python", "ram_bytes",
            "workspace_free_bytes", "scratch_free_bytes",
        },
        "lane host evidence drift",
    )
    policy = spec["execution_policy"]
    require(isinstance(host["platform"], str) and host["platform"], "lane host platform drift")
    require(str(host["python"]).startswith("3.10."), "lane host Python drift")
    for key, minimum, label in (
        ("ram_bytes", policy["minimum_ram_bytes"], "RAM"),
        (
            "workspace_free_bytes",
            policy["minimum_workspace_free_bytes"],
            "workspace-space",
        ),
        (
            "scratch_free_bytes",
            policy["minimum_scratch_free_bytes"],
            "scratch-space",
        ),
    ):
        require(
            isinstance(host[key], int)
            and not isinstance(host[key], bool)
            and host[key] >= minimum,
            f"lane host {label} evidence drift",
        )

    stages = state.get("stages")
    require(isinstance(stages, dict), "lane state stages are missing")
    require(set(stages) == {"container_build", "deliverable"}, "required stage inventory drift")
    build = stages["container_build"]
    deliverable = stages["deliverable"]
    require(
        isinstance(build, dict)
        and set(build) == {
            "status", "fingerprint", "started_utc", "completed_utc",
            "details", "outputs",
        },
        "container build stage field inventory drift",
    )
    require(
        isinstance(deliverable, dict)
        and set(deliverable) == {
            "status", "fingerprint", "completed_utc", "details", "outputs",
        },
        "deliverable stage field inventory drift",
    )
    require(build.get("status") == "complete", "container build stage is not complete")
    require(deliverable.get("status") == "complete", "deliverable stage is not complete")
    for label, record, keys in (
        ("container build", build, ("started_utc", "completed_utc")),
        ("deliverable", deliverable, ("completed_utc",)),
    ):
        for key in keys:
            require(
                isinstance(record[key], str) and record[key],
                f"{label} stage {key} is missing",
            )
    require(
        build["fingerprint"] == build_fingerprint(spec, source_identity),
        "container build fingerprint drift",
    )
    require(
        deliverable["fingerprint"]
        == deliverable_fingerprint(build["outputs"], state["controller_sha256"]),
        "deliverable fingerprint drift",
    )
    expected_result = {
        "manifest_status": spec["output_contract"]["required_manifest_status"],
        "phone_execution_proven": False,
        "npu_execution_proven": False,
    }
    require(deliverable["details"] == expected_result, "deliverable result summary drift")
    require(state["result"] == expected_result, "lane result summary drift")

    manifest_path = workspace / "stages/deliverable/result_manifest.json"
    require(manifest_path.is_file(), "result manifest is missing")
    manifest = read_json(manifest_path)
    require(
        isinstance(manifest, dict) and set(manifest) == RESULT_MANIFEST_FIELDS,
        "result manifest field inventory drift",
    )
    require(manifest["schema_version"] == SCHEMA_VERSION, "manifest schema drift")
    require(manifest["experiment_id"] == EXPECTED_EXPERIMENT_ID, "manifest experiment drift")
    require(
        isinstance(manifest["created_utc"], str) and manifest["created_utc"],
        "manifest creation time drift",
    )
    require(
        manifest["status"] == spec["output_contract"]["required_manifest_status"],
        "manifest status drift",
    )
    require(manifest["claim_boundary"] == spec["claim_boundary"], "manifest claim boundary drift")
    require(manifest["spec_sha256"] == EXPECTED_SPEC_SHA256, "manifest spec drift")
    require(
        manifest["controller_sha256"] == state["controller_sha256"],
        "manifest controller drift",
    )
    require(
        manifest["source_lock_sha256"] == state["source_lock_sha256"],
        "manifest source-lock drift",
    )
    require(manifest["source"] == source_identity, "manifest source identity drift")
    require(manifest["host"] == host, "manifest host evidence drift")
    require(manifest["next_gate"] == NEXT_GATE, "manifest next-gate drift")
    source_gguf = manifest["source_gguf"]
    require(
        isinstance(source_gguf, dict)
        and set(source_gguf) == {"filename", "bytes", "sha256", "inspection"},
        "manifest source GGUF inventory drift",
    )
    profile = spec["source_profiles"][source_identity["profile_id"]]
    require(source_gguf["filename"] == profile["filename"], "manifest source filename drift")
    require(source_gguf["bytes"] == source_identity["bytes"], "manifest source size drift")
    require(source_gguf["sha256"] == source_identity["sha256"], "manifest source SHA-256 drift")
    inspection = source_gguf["inspection"]
    contract = profile["gguf_contract"]
    require(
        isinstance(inspection, dict)
        and set(inspection) == {
            "version", "tensor_count", "metadata_kv_count", "metadata",
            "array_counts", "tensor_types",
        },
        "manifest GGUF inspection inventory drift",
    )

    downloads = workspace / "downloads"
    source_path = downloads / profile["filename"]
    require(source_path.is_file(), "locked source GGUF is missing")
    require(source_path.stat().st_size == source_identity["bytes"], "locked source GGUF size drift")
    require(sha256_file(source_path) == source_identity["sha256"], "locked source GGUF SHA-256 drift")
    actual_inspection = inspect_gguf(source_path)
    validate_gguf_contract(actual_inspection, contract)
    require(
        inspection
        == {key: value for key, value in actual_inspection.items() if key != "tensor_names"},
        "manifest GGUF inspection drift",
    )
    qairt_record = spec["qairt"]
    qairt_path = downloads / qairt_record["filename"]
    require(qairt_path.is_file(), "locked QAIRT archive is missing")
    require(qairt_path.stat().st_size == qairt_record["bytes"], "locked QAIRT archive size drift")
    require(sha256_file(qairt_path) == EXPECTED_QAIRT_SHA256, "locked QAIRT archive SHA-256 drift")

    require(manifest["target"] == spec["target"], "manifest target drift")
    details = build["details"]
    require(
        isinstance(details, dict)
        and set(details) == {"builder", "context_count", "gguf_artifacts"},
        "container build detail inventory drift",
    )
    builder = details["builder"]
    require(
        isinstance(builder, dict)
        and set(builder) == {
            "builder_class", "container_class", "target_string", "builder_config",
        },
        "builder evidence inventory drift",
    )
    builder_contract = spec["builder_contract"]
    require(builder["builder_class"] == builder_contract["expected_builder"], "builder class drift")
    require(builder["container_class"] == builder_contract["container_class"], "container class drift")
    require(builder["target_string"] == builder_contract["target_string"], "builder target drift")
    require(builder["builder_config"] == expected_builder_config(contract), "builder config drift")
    require(manifest["builder"] == details, "manifest builder evidence drift")
    require(manifest["container_root"] == "stages/container_build/container", "container root drift")
    require(manifest["qairt_archive_sha256"] == EXPECTED_QAIRT_SHA256, "manifest QAIRT drift")
    expected_members = spec["qairt"]["critical_members"] + spec["qairt"]["android_runtime_inventory"]
    require(manifest["qairt_verified_members"] == expected_members, "verified QAIRT member drift")
    require(
        manifest["android_runtime_inventory"] == spec["qairt"]["android_runtime_inventory"],
        "Android runtime inventory drift",
    )
    output_contract = spec["output_contract"]
    require(
        manifest["runtime_binaries_bundled"] is output_contract["runtime_binaries_bundled"],
        "runtime-bundling claim drift",
    )
    require(
        manifest["phone_execution_proven"] is output_contract["phone_execution_proven"]
        and manifest["npu_execution_proven"] is output_contract["npu_execution_proven"],
        "invalid execution claim",
    )
    packages = manifest["python_packages"]
    expected_packages = spec["host_dependencies"]["python_packages"]
    require(
        isinstance(packages, dict) and set(packages) == set(expected_packages),
        "Python package inventory drift",
    )
    for name, expected in expected_packages.items():
        actual = packages.get(name)
        require(
            actual == expected or (name == "torch" and str(actual).startswith(expected + "+")),
            f"manifest package drift: {name}={actual}",
        )

    verify_inventory(build["outputs"], workspace)
    verify_inventory(deliverable["outputs"], workspace)
    verify_inventory(manifest["container_artifacts"], workspace)
    build_root = workspace / "stages/container_build"
    expected_artifact_relative = (
        f"downloads/gguf_artifacts_{Path(profile['filename']).stem}"
    )
    require(
        details["gguf_artifacts"] == expected_artifact_relative,
        "GGUF artifact root drift",
    )
    artifact_root = workspace / expected_artifact_relative
    expected_build_outputs = inventory_tree(build_root, workspace)
    expected_build_outputs.extend(inventory_tree(artifact_root, workspace))
    require(build["outputs"] == expected_build_outputs, "container build output inventory drift")

    container_root = workspace / manifest["container_root"]
    expected_container = inventory_tree(container_root, workspace)
    require(manifest["container_artifacts"] == expected_container, "container artifact inventory drift")
    contexts = validate_container(container_root, spec)
    require(
        isinstance(details["context_count"], int)
        and not isinstance(details["context_count"], bool)
        and details["context_count"] == len(contexts),
        "HTP context count drift",
    )
    require(
        (build_root / "build_qwen_container.py").read_text(encoding="utf-8")
        == BUILD_HELPER,
        "build helper drift",
    )
    require(
        read_json(build_root / "build_config.json")
        == {"model": contract, "target_string": builder_contract["target_string"]},
        "build config drift",
    )
    require(read_json(build_root / "builder_result.json") == builder, "builder result drift")

    deliverable_root = workspace / "stages/deliverable"
    require(
        deliverable["outputs"] == inventory_tree(deliverable_root, workspace),
        "deliverable output inventory drift",
    )
    expected_deliverable_paths = {
        "stages/deliverable/result_manifest.json",
        "stages/deliverable/container.SHA256SUMS",
        f"stages/deliverable/{DEFAULT_SPEC.name}",
        f"stages/deliverable/{source_lock_path.name}",
    }
    require(
        {item["path"] for item in deliverable["outputs"]} == expected_deliverable_paths,
        "deliverable file inventory drift",
    )
    require(
        sha256_file(workspace / "stages/deliverable" / DEFAULT_SPEC.name)
        == EXPECTED_SPEC_SHA256,
        "deliverable spec copy drift",
    )
    require(
        sha256_file(workspace / "stages/deliverable" / source_lock_path.name)
        == state["source_lock_sha256"],
        "deliverable source-lock copy drift",
    )
    expected_sums = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in expected_container
    )
    require(
        (workspace / "stages/deliverable/container.SHA256SUMS").read_text(
            encoding="utf-8"
        ) == expected_sums,
        "container checksum file drift",
    )
    return {"status": "verified_complete", "manifest": relative_to_workspace(manifest_path, workspace), "container": "stages/container_build/container", "source_sha256": state["source_identity"]["sha256"], "phone_execution_proven": False, "npu_execution_proven": False}


def execute(spec_path: Path, source_lock_path: Path, spec: dict[str, Any], source_identity: dict[str, Any], workspace: Path, scratch: Path, qairt_override: Path | None, source_override: Path | None) -> dict[str, Any]:
    workspace = safe_workspace(workspace)
    scratch = safe_workspace(scratch)
    host = host_preflight(workspace, scratch, spec)
    state_path = workspace / "qairt242_qwen25_state.json"
    with workspace_lock(workspace):
        state = load_or_create_state(state_path, spec_path, source_lock_path, source_identity)
        if state.get("status") == "complete":
            return verify_completed(
                workspace, state, spec, source_lock_path, source_identity
            )
        state.pop("last_failure", None)
        state["status"] = "running"
        state["host"] = host
        write_state(state_path, state)
        try:
            qairt_record = spec["qairt"]
            qairt_archive = verified_artifact("QAIRT", qairt_record, workspace, qairt_override)
            source_record = {"filename": spec["source_profiles"][source_identity["profile_id"]]["filename"], "url": source_identity["url"], "bytes": source_identity["bytes"], "sha256": source_identity["sha256"]}
            gguf = verified_artifact("Qwen GGUF", source_record, workspace, source_override)
            gguf_inspection = inspect_gguf(gguf)
            validate_gguf_contract(gguf_inspection, spec["source_profiles"][source_identity["profile_id"]]["gguf_contract"])
            gguf_record = {"filename": gguf.name, "bytes": gguf.stat().st_size, "sha256": sha256_file(gguf), "inspection": {key: value for key, value in gguf_inspection.items() if key != "tensor_names"}}
            sdk_root, sdk_members = prepare_sdk(qairt_archive, scratch, spec)
            venv, env, packages = prepare_venv(scratch, sdk_root, spec)
            build = build_stage(workspace, state_path, state, gguf, sdk_root, venv, env, spec, source_identity)
            deliverable = create_deliverable(workspace, state_path, state, spec_path, source_lock_path, spec, build, host, packages, gguf_record, sdk_members)
            state["status"] = "complete"
            state["completed_utc"] = utc_now()
            state["result"] = deliverable["details"]
            write_state(state_path, state)
            return verify_completed(
                workspace, state, spec, source_lock_path, source_identity
            )
        except Exception as error:
            state["status"] = "failed"
            state["last_failure"] = {"time_utc": utc_now(), "type": type(error).__name__, "message": str(error)}
            write_state(state_path, state)
            atomic_write_json(workspace / "qairt242_qwen25_failure_report.json", {"schema_version": 1, "status": "failed_closed", "error": state["last_failure"], "instruction": "Preserve the workspace and QAIRT cache, correct only the documented precondition, and rerun. Never substitute QAIRT 2.45, CPU, or an unapproved model."})
            raise


def plan(spec: dict[str, Any], source_lock_path: Path, source_identity: dict[str, Any]) -> dict[str, Any]:
    profile = spec["source_profiles"][source_identity["profile_id"]]
    return {
        "experiment_id": EXPECTED_EXPERIMENT_ID, "host_execution": False,
        "source_lock": source_lock_path.name, "source_profile": source_identity["profile_id"], "source_role": profile["role"],
        "source_approved": source_identity["approved"], "source_missing_identity": source_identity.get("missing_identity", []),
        "route": "GGUF -> QAIRT GGUFBuilder -> QwenBuilderHTP -> LLMContainer.save",
        "target": spec["builder_contract"]["target_string"], "backend": "HTP",
        "mixed_tensor_gate": {"actual_required_types": profile.get("gguf_contract", {}).get("tensor_types"), "qairt_242_support": {"Q4_K": "pinned dequantizer+dispatch", "Q6_K": "pinned dequantizer+dispatch"}},
        "resume": "durable GGUF artifacts plus QAIRT cache_root; completed outputs are hash-verified",
        "result_claim": "context_generated_unexecuted",
        "forbidden": ["macOS conversion", "QAIRT 2.45 fallback", "CPU backend fallback", "unapproved source", "phone/NPU execution claim without device evidence"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "verify"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/content/qairt242_qwen25_scratch"))
    parser.add_argument("--qairt-archive", type=Path)
    parser.add_argument("--source-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path, spec = load_spec(args.spec)
    source_lock_path, _, source_identity = load_source_lock(args.source_lock, spec, require_approved=args.command == "run")
    if args.command == "plan":
        print(json.dumps(plan(spec, source_lock_path, source_identity), indent=2, sort_keys=True))
        return 0
    require(args.workspace is not None, "--workspace is required for run and verify")
    workspace = safe_workspace(args.workspace)
    if args.command == "verify":
        state_path = workspace / "qairt242_qwen25_state.json"
        require(state_path.is_file(), f"missing state: {state_path}")
        result = verify_completed(
            workspace,
            read_json(state_path),
            spec,
            source_lock_path,
            source_identity,
        )
    else:
        result = execute(spec_path, source_lock_path, spec, source_identity, workspace, args.scratch, args.qairt_archive, args.source_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaneError as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
