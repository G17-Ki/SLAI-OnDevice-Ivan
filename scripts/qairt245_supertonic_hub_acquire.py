#!/usr/bin/env python3
"""Fail-closed QAIRT 2.45 Supertonic artifact acquisition lane.

``plan`` is offline. ``metadata`` performs authenticated, read-only Qualcomm
AI Hub lookups plus bounded HTTP range reads of the public QAIRT ZIP index.
``acquire`` is Linux/x86_64-only and is intended for Google Colab with a
persistent Drive workspace. It freshly downloads the four already-compiled
context binaries, validates the matching QAIRT archive against its pinned ZIP
and member contract, computes hashes, extracts only the pinned Android runtime
members, and writes an atomic artifact lock. Standalone ``verify`` requires an
artifact-lock SHA-256 obtained from that trusted acquisition boundary or a
separately pinned external record; a hash copied from the workspace under
verification is not a trust root.

This controller does not build an APK, call ADB, or mutate a phone. In
particular, it never treats the accepted QAIRT 2.42 runtime as compatible with
the QAIRT 2.45 contexts.
"""

from __future__ import annotations

import argparse
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
from urllib.parse import urlsplit
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/qairt245_supertonic_isolated_spec.json"
)
EXPECTED_LANE_ID = "qairt245_supertonic_isolated_sm8850_htp_v81_v1"
EXPECTED_SPEC_SHA256 = (
    "79cf01d8bbc9ba783d41196edce0f830a49c1d3e50549a222fdf09aacb3cf8ce"
)
EXPECTED_COMPONENTS = (
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
)
OFFICIAL_HUB_URL = "https://workbench.aihub.qualcomm.com"
TAIL_BYTES = 131072
MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
SCHEMA_VERSION = 1
ACQUIRE_STATE_FILENAME = "qairt245_supertonic_acquire_state.json"
PINNED_METADATA_STATE_FILENAME = "qairt245_supertonic_hub_state.json"


class LaneError(RuntimeError):
    """A validation, authentication, or acquisition invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaneError(message)


def require_exact_keys(
    value: Any, expected: set[str] | frozenset[str], label: str
) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    required = set(expected)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    require(
        not missing and not extra,
        f"{label} keys drift: missing={missing}, extra={extra}",
    )


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise LaneError(f"atomic temporary already exists: {temporary}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def validate_spec(spec: dict[str, Any], spec_path: Path | None = None) -> None:
    require(spec.get("schema_version") == SCHEMA_VERSION, "unexpected spec schema")
    require(spec.get("lane_id") == EXPECTED_LANE_ID, "unexpected lane identity")
    if spec_path is not None:
        require(spec_path.is_file(), f"missing spec: {spec_path}")
        require(
            sha256_file(spec_path) == EXPECTED_SPEC_SHA256,
            "spec SHA-256 does not match controller trust root",
        )

    policy = spec.get("safety_policy", {})
    for key, expected in {
        "host_metadata_allowed": True,
        "host_artifact_download_allowed": False,
        "macos_acquisition_allowed": False,
        "phone_mutation_allowed": False,
        "adb_allowed": False,
        "accepted_package_mutation_allowed": False,
        "accepted_runtime_mixing_allowed": False,
    }.items():
        require(policy.get(key) is expected, f"safety policy changed: {key}")
    require(
        policy.get("candidate_package_id") == "ai.liquid.lfmdemo.ttsprobe.qairt245",
        "candidate package identity changed",
    )
    protected_ids = {
        policy["protected_main_app"]["package_id"],
        policy["protected_runtime"]["package_id"],
    }
    require(
        policy["candidate_package_id"] not in protected_ids,
        "candidate reuses a protected package ID",
    )
    validate_sha256(policy["protected_main_app"]["apk_sha256"], "main APK")
    validate_sha256(policy["protected_runtime"]["apk_sha256"], "runtime APK")
    require(
        policy["protected_runtime"]["qairt_build"] == "2.42.0.251225135753_193295",
        "protected runtime identity changed",
    )

    hub = spec.get("hub", {})
    require(hub.get("client_version") == "0.52.0", "AI Hub client pin changed")
    require(hub.get("api_qairt_version") == "2.45", "Hub API version changed")
    require(
        hub.get("exact_qairt_build") == "2.45.0.260326154327",
        "Hub QAIRT build changed",
    )
    require(hub.get("hub_version") == "aihub-2026.07.17.0", "Hub build changed")
    target = hub.get("target", {})
    require(
        {
            "device": target.get("device"),
            "os": target.get("os"),
            "chipset": target.get("chipset"),
            "soc_model": target.get("soc_model"),
            "htp_arch": target.get("htp_arch"),
            "backend": target.get("backend"),
        }
        == {
            "device": "Snapdragon 8 Elite Gen 5 QRD",
            "os": "16",
            "chipset": "sm8850",
            "soc_model": "87",
            "htp_arch": "v81",
            "backend": "HTP",
        },
        "Hub target changed",
    )
    require(
        set(target.get("required_attributes", []))
        == {
            "abi:aarch64-android",
            "chipset:sm8850",
            "framework:qnn",
            "hexagon:v81",
            "soc-model:87",
        },
        "Hub target attributes changed",
    )
    components = hub.get("components", {})
    require(tuple(components) == EXPECTED_COMPONENTS, "component order or set changed")
    total_bytes = 0
    seen_jobs: set[str] = set()
    seen_models: set[str] = set()
    for name, component in components.items():
        compile_job = component.get("compile_job", {})
        source_model = component.get("source_model", {})
        target_model = component.get("target_model", {})
        job_id = str(compile_job.get("id", ""))
        model_id = str(target_model.get("id", ""))
        require(bool(re.fullmatch(r"j[a-z0-9]+", job_id)), f"invalid job ID: {name}")
        require(
            bool(re.fullmatch(r"m[a-z0-9]+", model_id)), f"invalid model ID: {name}"
        )
        require(
            job_id not in seen_jobs and model_id not in seen_models,
            "duplicate Hub identity",
        )
        seen_jobs.add(job_id)
        seen_models.add(model_id)
        options = str(compile_job.get("options", ""))
        require(
            "--target_runtime qnn_context_binary" in options, f"runtime changed: {name}"
        )
        require("--qairt_version 2.45" in options, f"QAIRT changed: {name}")
        require(
            "--qairt_version 2.42" not in options,
            f"2.42 leaked into 2.45 compile: {name}",
        )
        require(source_model.get("type") == "ONNX", f"source type changed: {name}")
        require(
            target_model.get("type") == "QNN_CONTEXT_BINARY",
            f"target type changed: {name}",
        )
        require(
            target_model.get("sha256") is None,
            f"unverified Hub hash was promoted: {name}",
        )
        require(
            target_model.get("sha256_status")
            == "must_be_computed_after_authenticated_download_on_linux",
            f"Hub hash policy changed: {name}",
        )
        size = target_model.get("bytes")
        require(isinstance(size, int) and size > 0, f"invalid target size: {name}")
        total_bytes += size
        require(
            bool(
                re.fullmatch(r"[0-9a-f]{32}", str(target_model.get("opaque_etag", "")))
            ),
            f"invalid opaque ETag: {name}",
        )
        require(
            bool(target_model.get("object_version_id")),
            f"missing object version: {name}",
        )
        metadata = target_model.get("metadata", {})
        require(
            metadata.get("QAIRT_SDK_VERSION") == hub["exact_qairt_build"],
            f"build mismatch: {name}",
        )
        require(
            metadata.get("QNN_CONTEXT_BIN_BACKEND") == "HTP",
            f"backend mismatch: {name}",
        )
        require(
            metadata.get("QNN_CONTEXT_BIN_HEXAGON_VERSION") == "v81",
            f"HTP mismatch: {name}",
        )
        require(
            metadata.get("QNN_CONTEXT_BIN_SOC_MODEL") == "87", f"SoC mismatch: {name}"
        )
        require(bool(target_model.get("inputs")), f"missing inputs: {name}")
        require(bool(target_model.get("outputs")), f"missing outputs: {name}")
    require(
        total_bytes == hub.get("total_target_model_bytes"), "Hub byte total changed"
    )

    runtime = spec.get("runtime", {})
    require(
        runtime.get("exact_build") == hub["exact_qairt_build"],
        "runtime/contexts build mismatch",
    )
    require(
        runtime.get("archive_version") == "2.45.0.260326", "runtime version changed"
    )
    require(runtime.get("bytes") == 1657471677, "runtime archive size changed")
    require(runtime.get("sha256") is None, "unverified archive hash was promoted")
    require(
        runtime.get("sha256_status") == "must_be_computed_after_full_download_on_linux",
        "runtime archive hash policy changed",
    )
    require(
        runtime.get("sdk_root_member") == "qairt/2.45.0.260326",
        "runtime SDK root changed",
    )
    require(
        bool(re.fullmatch(r"[0-9a-f]{32}", str(runtime.get("opaque_etag", "")))),
        "invalid runtime opaque ETag",
    )
    require(bool(runtime.get("object_version_id")), "missing runtime object version")
    central = runtime.get("central_directory", {})
    require(central.get("entries") == 10660, "central entry count changed")
    require(central.get("offset") == 1655805354, "central offset changed")
    require(central.get("bytes") == 1666301, "central size changed")
    validate_sha256(central.get("sha256"), "central directory")
    sdk_yaml = runtime.get("sdk_yaml_contract", {})
    validate_sha256(sdk_yaml.get("sha256"), "sdk.yaml")
    require(
        sdk_yaml.get("required_values")
        == {
            "product": "QAIRT",
            "version": "2.45.0",
            "build_id": "260326154327",
            "qnn_backend_api_version": "2.18.0",
            "android-ndk": "r26c",
        },
        "sdk.yaml identity changed",
    )
    members = runtime.get("required_members", [])
    require(len(members) == 11, "runtime member set changed")
    seen_paths: set[str] = set()
    for member in members:
        path = str(member.get("path", ""))
        require(
            path.startswith(runtime["sdk_root_member"] + "/"),
            "runtime member escaped SDK root",
        )
        require(path not in seen_paths, "duplicate runtime member")
        seen_paths.add(path)
        require(
            isinstance(member.get("bytes"), int) and member["bytes"] > 0,
            f"invalid member size: {path}",
        )
        require(
            isinstance(member.get("compressed_bytes"), int)
            and member["compressed_bytes"] > 0,
            f"invalid compressed size: {path}",
        )
        require(
            bool(re.fullmatch(r"[0-9a-f]{8}", str(member.get("crc32", "")))),
            f"invalid CRC32: {path}",
        )
        validate_sha256(member.get("sha256"), path)
    require(sdk_yaml["path"] in seen_paths, "sdk.yaml is not in runtime inventory")

    acquisition = spec.get("acquisition", {})
    require(
        acquisition.get("allowed_host", {}).get("system") == "Linux"
        and acquisition.get("allowed_host", {}).get("machine") == "x86_64",
        "acquisition host policy changed",
    )
    require(
        acquisition.get("minimum_free_bytes") == 4294967296, "free-space gate changed"
    )
    require(
        acquisition.get("required_result_status")
        == "artifacts_locked_unpackaged_unexecuted",
        "claim boundary changed",
    )


def load_spec(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    spec = read_json(resolved)
    require(isinstance(spec, dict), "spec root must be an object")
    validate_spec(spec, resolved)
    return resolved, spec


def plan(spec: dict[str, Any]) -> dict[str, Any]:
    components = spec["hub"]["components"]
    return {
        "lane_id": spec["lane_id"],
        "status": "ready_for_authenticated_metadata_then_linux_acquisition",
        "network_used": False,
        "target": spec["hub"]["target"],
        "qairt_build": spec["hub"]["exact_qairt_build"],
        "candidate_package_id": spec["safety_policy"]["candidate_package_id"],
        "components": {
            name: {
                "compile_job_id": value["compile_job"]["id"],
                "target_model_id": value["target_model"]["id"],
                "format": value["target_model"]["type"],
                "bytes": value["target_model"]["bytes"],
                "sha256_status": value["target_model"]["sha256_status"],
            }
            for name, value in components.items()
        },
        "runtime": {
            "url": spec["runtime"]["url"],
            "bytes": spec["runtime"]["bytes"],
            "exact_build": spec["runtime"]["exact_build"],
            "pinned_member_count": len(spec["runtime"]["required_members"]),
            "archive_sha256_status": spec["runtime"]["sha256_status"],
        },
        "next": [
            "metadata: authenticate all Hub identities and recheck the bounded runtime index",
            "acquire on Linux/Colab: download, hash, extract, and atomically lock artifacts",
            "record the returned artifact-lock SHA-256 outside the workspace trust boundary",
            "build a separate candidate APK only after the artifact lock verifies",
        ],
        "forbidden": [
            "artifact download on macOS",
            "ADB or phone mutation",
            "loading a 2.45 context in the protected 2.42 runtime",
            "reusing either accepted package ID or UID",
            "using a lock hash copied from an untrusted workspace as standalone verification authority",
            "phone/NPU/latency/audio claims from cloud metadata",
        ],
    }


def _load_hub(profile: str | None = None) -> tuple[Any, Any]:
    try:
        import qai_hub as hub
    except ImportError as error:
        raise LaneError("qai-hub==0.52.0 is required for metadata/acquire") from error
    require(str(getattr(hub, "__version__", "")) == "0.52.0", "qai-hub version drift")
    client = hub.Client(profile=profile) if profile else hub.Client()
    require(
        str(getattr(client.config, "api_url", "")) == OFFICIAL_HUB_URL,
        "AI Hub API endpoint is not the pinned Qualcomm origin",
    )
    require(
        str(getattr(client.config, "web_url", "")) == OFFICIAL_HUB_URL,
        "AI Hub web endpoint is not the pinned Qualcomm origin",
    )
    return hub, client


def _tensor_spec(value: Any) -> list[dict[str, Any]]:
    require(isinstance(value, dict) and len(value) == 1, "expected one QNN graph")
    tensors = next(iter(value.values()))
    return [
        {
            "name": str(tensor.name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for tensor in tensors
    ]


def _download_head(client: Any, model_id: str) -> dict[str, Any]:
    try:
        from qai_hub import public_rest_api as hub_api
        from qai_hub import s3_transfer
    except ImportError as error:
        raise LaneError("pinned qai-hub transfer helpers are unavailable") from error
    response = hub_api._get_model_download_response(
        client.config, model_id, use_acceleration=False
    )
    if response.HasField("credentials"):
        credentials = response.credentials
        s3_client = s3_transfer._build_client(credentials)
        try:
            head = s3_client.head_object(Bucket=credentials.bucket, Key=credentials.key)
        finally:
            close = getattr(s3_client, "close", None)
            if close:
                close()
        return {
            "transport": "scoped_s3_credentials",
            "filename": str(credentials.filename),
            "bytes": int(head["ContentLength"]),
            "opaque_etag": str(head.get("ETag", "")).strip('"'),
            "object_version_id": head.get("VersionId"),
            "last_modified": (
                head["LastModified"].isoformat() if head.get("LastModified") else None
            ),
            "checksum_sha256": head.get("ChecksumSHA256"),
        }
    raise LaneError(f"Hub model {model_id} did not return scoped S3 credentials")


def _validate_job_observation(
    name: str, observed: dict[str, Any], expected: dict[str, Any], spec: dict[str, Any]
) -> None:
    compile_job = expected["compile_job"]
    source_model = expected["source_model"]
    target_model = expected["target_model"]
    require(observed["job"]["id"] == compile_job["id"], f"job ID drift: {name}")
    require(observed["job"]["name"] == compile_job["name"], f"job name drift: {name}")
    require(observed["job"]["date"] == compile_job["date"], f"job date drift: {name}")
    require(
        observed["job"]["options"] == compile_job["options"],
        f"job options drift: {name}",
    )
    require(
        observed["job"]["hub_version"] == spec["hub"]["hub_version"],
        f"Hub version drift: {name}",
    )
    require(observed["job"]["status"] == "SUCCESS", f"job is not successful: {name}")
    require(observed["source_model"] == source_model, f"source model drift: {name}")
    for key in ("id", "name", "type", "metadata", "inputs", "outputs"):
        require(
            observed["target_model"][key] == target_model[key],
            f"target {key} drift: {name}",
        )
    expected_target = spec["hub"]["target"]
    require(
        observed["device"]["name"] == expected_target["device"], f"device drift: {name}"
    )
    require(
        observed["device"]["os"] == expected_target["os"], f"device OS drift: {name}"
    )
    require(
        set(expected_target["required_attributes"]).issubset(
            observed["device"]["attributes"]
        ),
        f"device attributes drift: {name}",
    )
    download = observed["download_authorization"]
    require(
        download["transport"] == "scoped_s3_credentials",
        f"download route drift: {name}",
    )
    for key in (
        "filename",
        "bytes",
        "opaque_etag",
        "object_version_id",
        "last_modified",
    ):
        require(download[key] == target_model[key], f"download {key} drift: {name}")
    require(
        download["checksum_sha256"] is None,
        f"unexpected server checksum semantics: {name}",
    )


def collect_hub_metadata(
    spec: dict[str, Any], profile: str | None = None, retain_targets: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, client = _load_hub(profile)
    frameworks = [
        {
            "name": framework.name,
            "api_version": framework.api_version,
            "api_tags": list(framework.api_tags),
            "full_version": framework.full_version,
        }
        for framework in client.get_frameworks()
    ]
    exact_frameworks = [
        item
        for item in frameworks
        if item["name"] == "QAIRT"
        and item["api_version"] == spec["hub"]["api_qairt_version"]
        and item["full_version"] == spec["hub"]["exact_qairt_build"]
    ]
    require(len(exact_frameworks) == 1, "exact QAIRT 2.45 framework is unavailable")
    observations: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for name, expected in spec["hub"]["components"].items():
        job = client.get_job(expected["compile_job"]["id"])
        status = job.get_status()
        require(
            status.finished and status.success, f"compile job failed or pending: {name}"
        )
        target = job.get_target_model()
        require(target is not None, f"compile returned no target model: {name}")
        metadata = {key.name: value for key, value in target.metadata.items()}
        observed = {
            "job": {
                "id": str(job.job_id),
                "name": str(job.name),
                "date": job.date.isoformat(),
                "hub_version": str(job.hub_version),
                "options": str(job.options),
                "status": str(status.code),
            },
            "device": {
                "name": str(job.device.name),
                "os": str(job.device.os),
                "attributes": list(job.device.attributes),
            },
            "source_model": {
                "id": str(job.model.model_id),
                "name": str(job.model.name),
                "type": str(job.model.model_type.name),
            },
            "target_model": {
                "id": str(target.model_id),
                "name": str(target.name),
                "type": str(target.model_type.name),
                "metadata": metadata,
                "inputs": _tensor_spec(target.input_spec),
                "outputs": _tensor_spec(target.output_spec),
            },
            "download_authorization": _download_head(client, str(target.model_id)),
        }
        _validate_job_observation(name, observed, expected, spec)
        observations[name] = observed
        if retain_targets:
            targets[name] = target
    return {
        "client_version": "0.52.0",
        "framework": exact_frameworks[0],
        "components": observations,
    }, targets


def parse_eocd(tail: bytes) -> dict[str, int]:
    offset = tail.rfind(b"PK\x05\x06")
    require(offset >= 0, "ZIP EOCD was not found")
    require(len(tail) - offset >= 22, "truncated ZIP EOCD")
    fields = struct.unpack_from("<IHHHHIIH", tail, offset)
    require(fields[0] == 0x06054B50, "invalid ZIP EOCD signature")
    require(fields[1] == 0 and fields[2] == 0, "multi-disk ZIP is forbidden")
    require(fields[3] == fields[4], "ZIP entry counts disagree")
    require(fields[7] == 0, "ZIP comments are forbidden for this trust root")
    return {
        "entries": int(fields[4]),
        "central_directory_bytes": int(fields[5]),
        "central_directory_offset": int(fields[6]),
    }


def parse_central_directory(data: bytes, entries: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    cursor = 0
    for _ in range(entries):
        require(cursor + 46 <= len(data), "truncated central directory")
        values = struct.unpack_from("<IHHHHHHIIIHHHHHII", data, cursor)
        require(values[0] == 0x02014B50, f"bad central signature at {cursor}")
        flags = int(values[3])
        name_bytes = int(values[10])
        extra_bytes = int(values[11])
        comment_bytes = int(values[12])
        end = cursor + 46 + name_bytes + extra_bytes + comment_bytes
        require(end <= len(data), "truncated central entry")
        raw_name = data[cursor + 46 : cursor + 46 + name_bytes]
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        require(name not in result, f"duplicate ZIP member: {name}")
        result[name] = {
            "path": name,
            "compression_method": int(values[4]),
            "crc32": f"{int(values[7]):08x}",
            "compressed_bytes": int(values[8]),
            "bytes": int(values[9]),
            "local_header_offset": int(values[16]),
        }
        cursor = end
    require(cursor == len(data), "central directory has trailing bytes")
    return result


def _range_get(
    session: Any, url: str, value: str, maximum: int
) -> tuple[bytes, dict[str, str]]:
    response = session.get(url, headers={"Range": value}, timeout=120)
    response.raise_for_status()
    require(
        urlsplit(str(response.url)).scheme == "https",
        "runtime metadata request was redirected away from HTTPS",
    )
    require(
        response.status_code == 206,
        f"HTTP range was not honored: {response.status_code}",
    )
    payload = response.content
    require(len(payload) <= maximum, "HTTP range response exceeded bound")
    return payload, {key.lower(): value for key, value in response.headers.items()}


def _remote_member(session: Any, url: str, entry: dict[str, Any]) -> bytes:
    offset = entry["local_header_offset"]
    probe, _ = _range_get(session, url, f"bytes={offset}-{offset + 4095}", 4096)
    require(len(probe) == 4096, "short local-header probe")
    fields = struct.unpack_from("<IHHHHHIIIHH", probe, 0)
    require(fields[0] == 0x04034B50, "invalid local ZIP header")
    require(int(fields[3]) == entry["compression_method"], "ZIP method mismatch")
    name_bytes = int(fields[9])
    extra_bytes = int(fields[10])
    name = probe[30 : 30 + name_bytes].decode("utf-8")
    require(name == entry["path"], "ZIP local member name mismatch")
    data_offset = offset + 30 + name_bytes + extra_bytes
    compressed, _ = _range_get(
        session,
        url,
        f"bytes={data_offset}-{data_offset + entry['compressed_bytes'] - 1}",
        entry["compressed_bytes"],
    )
    require(len(compressed) == entry["compressed_bytes"], "short compressed member")
    if entry["compression_method"] == 8:
        raw = zlib.decompress(compressed, -15)
    elif entry["compression_method"] == 0:
        raw = compressed
    else:
        raise LaneError(
            f"unsupported ZIP compression method: {entry['compression_method']}"
        )
    require(len(raw) == entry["bytes"], "remote member size mismatch")
    require(
        f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}" == entry["crc32"],
        "remote member CRC mismatch",
    )
    return raw


def _parse_simple_yaml_scalars(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        if key and raw.strip() and not line.startswith(" "):
            result[key.strip()] = raw.strip()
    return result


def collect_runtime_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        import requests
    except ImportError as error:
        raise LaneError("requests is required for runtime metadata") from error
    runtime = spec["runtime"]
    session = requests.Session()
    try:
        tail, headers = _range_get(
            session, runtime["url"], f"bytes=-{TAIL_BYTES}", TAIL_BYTES
        )
        content_range = headers.get("content-range", "")
        match = re.fullmatch(r"bytes \d+-\d+/(\d+)", content_range)
        require(match is not None, "runtime Content-Range is missing")
        require(int(match.group(1)) == runtime["bytes"], "runtime HTTP size drift")
        require(
            headers.get("etag", "").strip('"') == runtime["opaque_etag"],
            "runtime ETag drift",
        )
        require(
            headers.get("x-amz-version-id") == runtime["object_version_id"],
            "runtime object version drift",
        )
        require(
            headers.get("last-modified") == runtime["last_modified"],
            "runtime Last-Modified drift",
        )
        eocd = parse_eocd(tail)
        central_expected = runtime["central_directory"]
        require(
            eocd["entries"] == central_expected["entries"], "runtime entry-count drift"
        )
        require(
            eocd["central_directory_bytes"] == central_expected["bytes"],
            "runtime central size drift",
        )
        require(
            eocd["central_directory_offset"] == central_expected["offset"],
            "runtime central offset drift",
        )
        require(
            eocd["central_directory_bytes"] <= MAX_CENTRAL_DIRECTORY_BYTES,
            "runtime central directory is too large",
        )
        start = eocd["central_directory_offset"]
        end = start + eocd["central_directory_bytes"] - 1
        central, _ = _range_get(
            session,
            runtime["url"],
            f"bytes={start}-{end}",
            MAX_CENTRAL_DIRECTORY_BYTES,
        )
        require(
            len(central) == eocd["central_directory_bytes"], "short central directory"
        )
        require(
            sha256_file_bytes(central) == central_expected["sha256"],
            "central directory hash drift",
        )
        index = parse_central_directory(central, eocd["entries"])
        for member in runtime["required_members"]:
            observed = index.get(member["path"])
            require(observed is not None, f"runtime member missing: {member['path']}")
            for key in ("bytes", "compressed_bytes", "crc32"):
                require(
                    observed[key] == member[key],
                    f"runtime member {key} drift: {member['path']}",
                )
        sdk_contract = runtime["sdk_yaml_contract"]
        sdk_bytes = _remote_member(session, runtime["url"], index[sdk_contract["path"]])
        require(
            sha256_file_bytes(sdk_bytes) == sdk_contract["sha256"],
            "sdk.yaml hash drift",
        )
        values = _parse_simple_yaml_scalars(sdk_bytes.decode("utf-8"))
        for key, expected in sdk_contract["required_values"].items():
            require(values.get(key) == expected, f"sdk.yaml value drift: {key}")
        return {
            "url": runtime["url"],
            "bytes": runtime["bytes"],
            "opaque_etag": runtime["opaque_etag"],
            "object_version_id": runtime["object_version_id"],
            "last_modified": runtime["last_modified"],
            "central_directory": {
                "entries": eocd["entries"],
                "offset": eocd["central_directory_offset"],
                "bytes": eocd["central_directory_bytes"],
                "sha256": sha256_file_bytes(central),
            },
            "sdk_yaml": {
                "bytes": len(sdk_bytes),
                "sha256": sha256_file_bytes(sdk_bytes),
                "values": {key: values[key] for key in sdk_contract["required_values"]},
            },
            "required_members_indexed": len(runtime["required_members"]),
            "member_sha256_source": "pinned bounded range/decompression observation; reverified after full Linux acquisition",
        }
    finally:
        session.close()


def sha256_file_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def metadata_state(
    spec_path: Path,
    spec: dict[str, Any],
    hub_metadata: dict[str, Any],
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "lane_id": spec["lane_id"],
        "status": "metadata_authenticated_artifacts_not_downloaded",
        "observed_utc": utc_now(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "controller_sha256": sha256_file(Path(__file__).resolve()),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "hub": hub_metadata,
        "runtime": runtime_metadata,
        "unresolved": [
            "Hub context SHA-256 values require authenticated Linux/Colab downloads.",
            "The full QAIRT archive SHA-256 requires the Linux/Colab download.",
            "No candidate APK or JNI frontend has been built.",
            "No phone, NPU, latency, audio, or accepted-runtime compatibility claim exists.",
        ],
        "next_command": (
            "python scripts/qairt245_supertonic_hub_acquire.py acquire "
            "--workspace /content/drive/MyDrive/lmf-demo/qairt245-supertonic"
        ),
    }


def safe_workspace(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    require(resolved != Path("/"), "workspace cannot be filesystem root")
    require(len(resolved.parts) >= 3, "workspace path is too broad")
    return resolved


def relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise LaneError(f"path escapes workspace: {path}") from error


@contextmanager
def workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / "qairt245_supertonic.lock"
    relative_to_workspace(lock_path, workspace)
    require(not lock_path.is_symlink(), "workspace lock path is a symlink")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaneError("another process holds the acquisition lock") from error
        yield
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_acquire_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    atomic_write_json(path, state)


def load_or_create_acquire_state(
    path: Path, spec_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    spec_hash = sha256_file(spec_path)
    controller_hash = sha256_file(Path(__file__).resolve())
    if path.is_file():
        state = read_json(path)
        require(
            state.get("lane_id") == spec["lane_id"], "workspace lane identity drift"
        )
        require(state.get("spec_sha256") == spec_hash, "workspace spec identity drift")
        require(
            state.get("controller_sha256") == controller_hash,
            "workspace controller identity drift; use a fresh workspace",
        )
        require(isinstance(state.get("stages"), dict), "workspace stages are invalid")
        return state
    return {
        "schema_version": SCHEMA_VERSION,
        "lane_id": spec["lane_id"],
        "status": "initialized",
        "created_utc": utc_now(),
        "spec_sha256": spec_hash,
        "controller_sha256": controller_hash,
        "stages": {},
    }


def preflight_acquire(workspace: Path, spec: dict[str, Any]) -> dict[str, Any]:
    allowed = spec["acquisition"]["allowed_host"]
    require(platform.system() == allowed["system"], "acquire is Linux-only")
    require(platform.machine() == allowed["machine"], "acquire requires x86_64")
    workspace.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workspace).free
    require(
        free >= spec["acquisition"]["minimum_free_bytes"], "insufficient free space"
    )
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "free_bytes_at_start": free,
    }


def quarantine(path: Path, workspace: Path, reason: str) -> Path:
    relative_to_workspace(path, workspace)
    root = workspace / "quarantine"
    relative_to_workspace(root, workspace)
    require(not root.is_symlink(), "quarantine root is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    target = root / (
        f"{path.name}.{reason}."
        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{os.getpid()}"
    )
    path.replace(target)
    return target


def verify_file_record(path: Path, record: dict[str, Any]) -> None:
    require(path.is_file(), f"missing file: {path}")
    require(path.stat().st_size == record["bytes"], f"byte-size drift: {path}")
    require(sha256_file(path) == record["sha256"], f"SHA-256 drift: {path}")


def _download_context(
    name: str,
    target: Any,
    expected: dict[str, Any],
    destination: Path,
    workspace: Path,
    stage: dict[str, Any] | None,
) -> dict[str, Any]:
    # Context SHA-256 values are intentionally unknown before acquisition.
    # Therefore a persisted stage can never authenticate its own context hash:
    # an exact-sized counterfeit could rewrite both the file record and state.
    # Always perform a fresh authenticated Hub download in the current run.
    del stage
    relative_to_workspace(destination, workspace)
    partial = destination.with_name(f".{destination.stem}.part-{os.getpid()}.bin")
    relative_to_workspace(partial, workspace)
    require(
        not destination.is_symlink(), f"context destination is a symlink: {destination}"
    )
    require(not partial.is_symlink(), f"context partial is a symlink: {partial}")
    if partial.exists():
        quarantine(partial, workspace, "stale-partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    returned = Path(target.download(str(partial))).resolve()
    relative_to_workspace(returned, workspace)
    require(returned.is_file(), f"Hub download produced no file: {name}")
    require(
        returned.stat().st_size == expected["bytes"],
        f"Hub download size mismatch: {name}",
    )
    digest = sha256_file(returned)
    if destination.exists():
        if (
            destination.is_file()
            and destination.stat().st_size == expected["bytes"]
            and sha256_file(destination) == digest
        ):
            returned.unlink()
        else:
            quarantine(destination, workspace, "superseded-context")
            returned.replace(destination)
    else:
        returned.replace(destination)
    record = {
        "status": "complete",
        "completed_utc": utc_now(),
        "target_model_id": expected["id"],
        "authenticated_object": {
            "filename": expected["filename"],
            "bytes": expected["bytes"],
            "opaque_etag": expected["opaque_etag"],
            "object_version_id": expected["object_version_id"],
            "last_modified": expected["last_modified"],
        },
        "file": {
            "path": relative_to_workspace(destination, workspace),
            "bytes": destination.stat().st_size,
            "sha256": digest,
        },
    }
    verify_file_record(destination, record["file"])
    return record


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND_JSON=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise LaneError(f"download command failed ({result.returncode})\n{tail}")


def _download_runtime_archive(
    spec: dict[str, Any],
    destination: Path,
    workspace: Path,
    stage: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = spec["runtime"]
    relative_to_workspace(destination, workspace)
    require(
        not destination.is_symlink(),
        f"runtime archive destination is a symlink: {destination}",
    )
    if stage and stage.get("status") == "complete":
        require_exact_keys(
            stage,
            {"status", "completed_utc", "authenticated_object", "file"},
            "runtime archive checkpoint",
        )
        require(
            stage["authenticated_object"] == _expected_runtime_object(spec),
            "runtime archive checkpoint object drift",
        )
        _validate_lock_file_record(
            stage["file"],
            label="runtime archive checkpoint file",
            expected_path=f"downloads/{runtime['filename']}",
            expected_bytes=runtime["bytes"],
        )
        verify_file_record(destination, stage["file"])
        verify_runtime_archive_contract(destination, spec)
        return stage
    if destination.exists():
        quarantine(destination, workspace, "unlocked-runtime-archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    require(
        not partial.is_symlink(),
        f"runtime archive partial is a symlink: {partial}",
    )
    if partial.exists() and partial.stat().st_size > runtime["bytes"]:
        quarantine(partial, workspace, "oversize-partial")
    command = [
        "curl",
        "--fail",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--retry",
        "5",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--header",
        f'If-Match: "{runtime["opaque_etag"]}"',
        "--output",
        str(partial),
        runtime["url"],
    ]
    download_log = destination.parent / "runtime.download.log"
    require(not download_log.is_symlink(), "runtime download log is a symlink")
    run_logged(command, destination.parent, download_log)
    require(partial.is_file(), "runtime download produced no file")
    require(partial.stat().st_size == runtime["bytes"], "runtime archive size mismatch")
    digest = sha256_file(partial)
    verify_runtime_archive_contract(partial, spec)
    partial.replace(destination)
    record = {
        "status": "complete",
        "completed_utc": utc_now(),
        "authenticated_object": {
            "url": runtime["url"],
            "filename": runtime["filename"],
            "bytes": runtime["bytes"],
            "opaque_etag": runtime["opaque_etag"],
            "object_version_id": runtime["object_version_id"],
            "last_modified": runtime["last_modified"],
        },
        "file": {
            "path": relative_to_workspace(destination, workspace),
            "bytes": destination.stat().st_size,
            "sha256": digest,
        },
    }
    verify_file_record(destination, record["file"])
    return record


def verify_runtime_archive_contract(archive: Path, spec: dict[str, Any]) -> None:
    """Authenticate every runtime packaging input against the pinned ZIP index.

    The full official archive SHA-256 is unknown until acquisition, so a
    self-declared archive hash alone is not sufficient for resume or offline
    verification. The central-directory hash and every extracted member hash
    are pre-pinned in the repository spec and provide the content trust root
    for the only archive members this lane is allowed to package.
    """

    runtime = spec["runtime"]
    require(archive.is_file(), f"missing runtime archive: {archive}")
    require(archive.stat().st_size == runtime["bytes"], "runtime archive size drift")
    tail_bytes = min(TAIL_BYTES, runtime["bytes"])
    with archive.open("rb") as stream:
        stream.seek(-tail_bytes, os.SEEK_END)
        tail = stream.read(tail_bytes)
    eocd = parse_eocd(tail)
    expected_central = runtime["central_directory"]
    require(
        eocd["entries"] == expected_central["entries"], "runtime ZIP entry-count drift"
    )
    require(
        eocd["central_directory_offset"] == expected_central["offset"],
        "runtime ZIP central offset drift",
    )
    require(
        eocd["central_directory_bytes"] == expected_central["bytes"],
        "runtime ZIP central size drift",
    )
    require(
        eocd["central_directory_offset"] + eocd["central_directory_bytes"] + 22
        == runtime["bytes"],
        "runtime ZIP has trailing or missing bytes",
    )
    with archive.open("rb") as stream:
        stream.seek(eocd["central_directory_offset"])
        central = stream.read(eocd["central_directory_bytes"])
    require(
        len(central) == eocd["central_directory_bytes"],
        "runtime ZIP central directory is truncated",
    )
    require(
        sha256_file_bytes(central) == expected_central["sha256"],
        "runtime ZIP central-directory SHA-256 drift",
    )
    index = parse_central_directory(central, eocd["entries"])
    with zipfile.ZipFile(archive) as package:
        for member in runtime["required_members"]:
            observed = index.get(member["path"])
            require(
                observed is not None, f"runtime ZIP member missing: {member['path']}"
            )
            for key in ("bytes", "compressed_bytes", "crc32"):
                require(
                    observed[key] == member[key],
                    f"runtime ZIP member {key} drift: {member['path']}",
                )
            info = package.getinfo(member["path"])
            digest = hashlib.sha256()
            count = 0
            with package.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
            require(
                count == member["bytes"],
                f"runtime ZIP member size drift: {member['path']}",
            )
            require(
                digest.hexdigest() == member["sha256"],
                f"runtime ZIP member SHA-256 drift: {member['path']}",
            )


def _extract_runtime_members(
    spec: dict[str, Any],
    archive: Path,
    output_root: Path,
    workspace: Path,
    prior: dict[str, Any] | None,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    if prior and prior.get("status") == "complete":
        for record in prior["files"]:
            verify_file_record(workspace / record["path"], record)
        return prior
    sdk_root = spec["runtime"]["sdk_root_member"] + "/"
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as package:
        for member in spec["runtime"]["required_members"]:
            info = package.getinfo(member["path"])
            require(
                info.file_size == member["bytes"], f"ZIP size drift: {member['path']}"
            )
            require(
                info.compress_size == member["compressed_bytes"],
                f"ZIP compressed-size drift: {member['path']}",
            )
            require(
                f"{info.CRC:08x}" == member["crc32"], f"ZIP CRC drift: {member['path']}"
            )
            relative = member["path"].removeprefix(sdk_root)
            require(
                relative != member["path"] and not Path(relative).is_absolute(),
                "invalid runtime member path",
            )
            destination = (output_root / relative).resolve()
            relative_to_workspace(destination, workspace)
            existing = next(
                (
                    item
                    for item in records
                    if item["path"] == relative_to_workspace(destination, workspace)
                ),
                None,
            )
            if destination.is_file():
                candidate = {
                    "path": relative_to_workspace(destination, workspace),
                    "bytes": member["bytes"],
                    "sha256": member["sha256"],
                    "role": member["role"],
                }
                try:
                    verify_file_record(destination, candidate)
                    records.append(candidate)
                    continue
                except LaneError:
                    quarantine(destination, workspace, "runtime-member-drift")
            require(existing is None, "duplicate extracted runtime path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(f".{destination.name}.part-{os.getpid()}")
            digest = hashlib.sha256()
            count = 0
            with package.open(info) as source, partial.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    count += len(block)
                target.flush()
                os.fsync(target.fileno())
            require(
                count == member["bytes"], f"extracted size mismatch: {member['path']}"
            )
            require(
                digest.hexdigest() == member["sha256"],
                f"extracted hash mismatch: {member['path']}",
            )
            partial.replace(destination)
            if member["role"] == "engineering_canary_frontend":
                destination.chmod(0o755)
            record = {
                "path": relative_to_workspace(destination, workspace),
                "bytes": count,
                "sha256": digest.hexdigest(),
                "role": member["role"],
            }
            records.append(record)
            state["stages"]["runtime_members"] = {
                "status": "in_progress",
                "files": records,
            }
            write_acquire_state(state_path, state)
    result = {"status": "complete", "completed_utc": utc_now(), "files": records}
    for record in records:
        verify_file_record(workspace / record["path"], record)
    return result


def _expected_context_object(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": target["filename"],
        "bytes": target["bytes"],
        "opaque_etag": target["opaque_etag"],
        "object_version_id": target["object_version_id"],
        "last_modified": target["last_modified"],
    }


def _expected_runtime_object(spec: dict[str, Any]) -> dict[str, Any]:
    runtime = spec["runtime"]
    return {
        "url": runtime["url"],
        "filename": runtime["filename"],
        "bytes": runtime["bytes"],
        "opaque_etag": runtime["opaque_etag"],
        "object_version_id": runtime["object_version_id"],
        "last_modified": runtime["last_modified"],
    }


def _expected_runtime_member_record(
    spec: dict[str, Any], member: dict[str, Any]
) -> dict[str, Any]:
    sdk_prefix = spec["runtime"]["sdk_root_member"] + "/"
    relative = member["path"].removeprefix(sdk_prefix)
    require(relative != member["path"], "runtime member is outside the SDK root")
    path = (Path(spec["acquisition"]["runtime_output_root"]) / relative).as_posix()
    return {
        "path": path,
        "bytes": member["bytes"],
        "sha256": member["sha256"],
        "role": member["role"],
    }


def _validate_lock_file_record(
    record: Any,
    *,
    label: str,
    expected_path: str,
    expected_bytes: int,
    expected_sha256: str | None = None,
) -> None:
    require_exact_keys(record, {"path", "bytes", "sha256"}, label)
    require(record["path"] == expected_path, f"{label} path drift")
    require(record["bytes"] == expected_bytes, f"{label} bytes drift")
    digest = validate_sha256(record["sha256"], label)
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"{label} SHA-256 drift")


def _validate_acquired_metadata(metadata: Any, spec: dict[str, Any]) -> None:
    require_exact_keys(
        metadata, {"observed_utc", "hub", "runtime"}, "acquired metadata"
    )
    require(isinstance(metadata["observed_utc"], str), "metadata timestamp is invalid")
    hub_metadata = metadata["hub"]
    require_exact_keys(
        hub_metadata,
        {"client_version", "framework", "components"},
        "acquired Hub metadata",
    )
    require(hub_metadata["client_version"] == "0.52.0", "metadata client drift")
    framework = hub_metadata["framework"]
    require_exact_keys(
        framework,
        {"name", "api_version", "api_tags", "full_version"},
        "acquired framework",
    )
    require(framework["name"] == "QAIRT", "metadata framework drift")
    require(
        framework["api_version"] == spec["hub"]["api_qairt_version"],
        "metadata QAIRT API drift",
    )
    require(
        framework["full_version"] == spec["hub"]["exact_qairt_build"],
        "metadata QAIRT build drift",
    )
    require(
        isinstance(framework["api_tags"], list) and "default" in framework["api_tags"],
        "metadata QAIRT tags drift",
    )
    observations = hub_metadata["components"]
    require(isinstance(observations, dict), "metadata components must be an object")
    require(
        tuple(observations) == EXPECTED_COMPONENTS,
        "metadata component order or set drift",
    )
    for name in EXPECTED_COMPONENTS:
        observed = observations[name]
        require_exact_keys(
            observed,
            {"job", "device", "source_model", "target_model", "download_authorization"},
            f"metadata component {name}",
        )
        require_exact_keys(
            observed["job"],
            {"id", "name", "date", "hub_version", "options", "status"},
            f"metadata job {name}",
        )
        require_exact_keys(
            observed["device"], {"name", "os", "attributes"}, f"metadata device {name}"
        )
        require_exact_keys(
            observed["source_model"], {"id", "name", "type"}, f"metadata source {name}"
        )
        require_exact_keys(
            observed["target_model"],
            {"id", "name", "type", "metadata", "inputs", "outputs"},
            f"metadata target {name}",
        )
        require_exact_keys(
            observed["download_authorization"],
            {
                "transport",
                "filename",
                "bytes",
                "opaque_etag",
                "object_version_id",
                "last_modified",
                "checksum_sha256",
            },
            f"metadata download {name}",
        )
        _validate_job_observation(name, observed, spec["hub"]["components"][name], spec)

    runtime = metadata["runtime"]
    require_exact_keys(
        runtime,
        {
            "url",
            "bytes",
            "opaque_etag",
            "object_version_id",
            "last_modified",
            "central_directory",
            "sdk_yaml",
            "required_members_indexed",
            "member_sha256_source",
        },
        "acquired runtime metadata",
    )
    expected_runtime = spec["runtime"]
    for key in ("url", "bytes", "opaque_etag", "object_version_id", "last_modified"):
        require(runtime[key] == expected_runtime[key], f"metadata runtime {key} drift")
    require(
        runtime["central_directory"] == expected_runtime["central_directory"],
        "metadata runtime central directory drift",
    )
    sdk_contract = expected_runtime["sdk_yaml_contract"]
    expected_sdk = {
        "bytes": next(
            member["bytes"]
            for member in expected_runtime["required_members"]
            if member["path"] == sdk_contract["path"]
        ),
        "sha256": sdk_contract["sha256"],
        "values": sdk_contract["required_values"],
    }
    require(runtime["sdk_yaml"] == expected_sdk, "metadata sdk.yaml drift")
    require(
        runtime["required_members_indexed"]
        == len(expected_runtime["required_members"]),
        "metadata runtime member count drift",
    )
    require(
        runtime["member_sha256_source"]
        == "pinned bounded range/decompression observation; reverified after full Linux acquisition",
        "metadata runtime hash-source drift",
    )


def _write_artifact_lock(
    spec_path: Path,
    spec: dict[str, Any],
    workspace: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    context_records = {}
    for name in EXPECTED_COMPONENTS:
        stage = state["stages"][f"context:{name}"]
        context_records[name] = {
            "target_model_id": stage["target_model_id"],
            "authenticated_object": stage["authenticated_object"],
            "file": stage["file"],
        }
    runtime_stage = state["stages"]["runtime_archive"]
    runtime_archive = {
        "authenticated_object": runtime_stage["authenticated_object"],
        "file": runtime_stage["file"],
    }
    runtime_members = state["stages"]["runtime_members"]["files"]
    lock = {
        "schema_version": SCHEMA_VERSION,
        "lane_id": spec["lane_id"],
        "status": spec["acquisition"]["required_result_status"],
        "created_utc": utc_now(),
        "claim_boundary": spec["claim_boundary"],
        "spec_sha256": sha256_file(spec_path),
        "controller_sha256": state["controller_sha256"],
        "acquire_state": {
            "path": ACQUIRE_STATE_FILENAME,
            "schema_version": state["schema_version"],
            "created_utc": state["created_utc"],
            "spec_sha256": state["spec_sha256"],
            "controller_sha256": state["controller_sha256"],
        },
        "metadata_fingerprint": canonical_sha256(state["metadata"]),
        "qairt_build": spec["hub"]["exact_qairt_build"],
        "target": spec["hub"]["target"],
        "candidate_package_id": spec["safety_policy"]["candidate_package_id"],
        "contexts": context_records,
        "runtime_archive": runtime_archive,
        "runtime_members": runtime_members,
        "protected_artifacts": {
            "main_package_id": spec["safety_policy"]["protected_main_app"][
                "package_id"
            ],
            "main_apk_sha256": spec["safety_policy"]["protected_main_app"][
                "apk_sha256"
            ],
            "runtime_package_id": spec["safety_policy"]["protected_runtime"][
                "package_id"
            ],
            "runtime_apk_sha256": spec["safety_policy"]["protected_runtime"][
                "apk_sha256"
            ],
            "protected_qairt_build": spec["safety_policy"]["protected_runtime"][
                "qairt_build"
            ],
        },
        "phone_execution_proven": False,
        "npu_execution_proven": False,
        "apk_built": False,
    }
    path = workspace / spec["acquisition"]["artifact_lock"]
    atomic_write_json(path, lock)
    return {"path": relative_to_workspace(path, workspace), "lock": lock}


def verify_lock(
    workspace: Path,
    spec: dict[str, Any],
    expected_lock_sha256: str | None,
) -> dict[str, Any]:
    workspace = safe_workspace(workspace)
    lock_path = workspace / spec["acquisition"]["artifact_lock"]
    require(lock_path.is_file(), f"missing artifact lock: {lock_path}")
    require(
        expected_lock_sha256 is not None,
        "trusted artifact lock SHA-256 is required",
    )
    trusted_lock_hash = validate_sha256(expected_lock_sha256, "trusted artifact lock")
    actual_lock_hash = sha256_file(lock_path)
    require(
        actual_lock_hash == trusted_lock_hash,
        "artifact lock SHA-256 does not match the trusted acquisition record",
    )
    lock = read_json(lock_path)
    require_exact_keys(
        lock,
        {
            "schema_version",
            "lane_id",
            "status",
            "created_utc",
            "claim_boundary",
            "spec_sha256",
            "controller_sha256",
            "acquire_state",
            "metadata_fingerprint",
            "qairt_build",
            "target",
            "candidate_package_id",
            "contexts",
            "runtime_archive",
            "runtime_members",
            "protected_artifacts",
            "phone_execution_proven",
            "npu_execution_proven",
            "apk_built",
        },
        "artifact lock",
    )
    controller_hash = sha256_file(Path(__file__).resolve())
    require(lock["schema_version"] == SCHEMA_VERSION, "artifact lock schema drift")
    require(lock["lane_id"] == spec["lane_id"], "artifact lock lane drift")
    require(
        lock["status"] == spec["acquisition"]["required_result_status"],
        "artifact lock status drift",
    )
    require(
        lock["claim_boundary"] == spec["claim_boundary"], "artifact lock claim drift"
    )
    require(lock["spec_sha256"] == EXPECTED_SPEC_SHA256, "artifact lock spec drift")
    require(
        lock["controller_sha256"] == controller_hash, "artifact lock controller drift"
    )
    validate_sha256(lock["metadata_fingerprint"], "metadata fingerprint")
    require(
        lock["qairt_build"] == spec["hub"]["exact_qairt_build"],
        "artifact lock QAIRT drift",
    )
    require(lock["target"] == spec["hub"]["target"], "artifact lock target drift")
    require(
        lock["candidate_package_id"] == spec["safety_policy"]["candidate_package_id"],
        "artifact lock package drift",
    )
    require(lock["apk_built"] is False, "artifact lane made an APK claim")
    require(lock["phone_execution_proven"] is False, "artifact lane made a phone claim")
    require(lock["npu_execution_proven"] is False, "artifact lane made an NPU claim")

    protected = spec["safety_policy"]
    expected_protected = {
        "main_package_id": protected["protected_main_app"]["package_id"],
        "main_apk_sha256": protected["protected_main_app"]["apk_sha256"],
        "runtime_package_id": protected["protected_runtime"]["package_id"],
        "runtime_apk_sha256": protected["protected_runtime"]["apk_sha256"],
        "protected_qairt_build": protected["protected_runtime"]["qairt_build"],
    }
    require_exact_keys(
        lock["protected_artifacts"], set(expected_protected), "protected artifacts"
    )
    require(
        lock["protected_artifacts"] == expected_protected,
        "protected artifact identity drift",
    )

    state_identity = lock["acquire_state"]
    require_exact_keys(
        state_identity,
        {"path", "schema_version", "created_utc", "spec_sha256", "controller_sha256"},
        "acquisition state identity",
    )
    require(
        state_identity["path"] == ACQUIRE_STATE_FILENAME, "acquisition state path drift"
    )
    require(
        state_identity["schema_version"] == SCHEMA_VERSION,
        "acquisition state schema drift",
    )
    require(
        state_identity["spec_sha256"] == EXPECTED_SPEC_SHA256,
        "acquisition state spec drift",
    )
    require(
        state_identity["controller_sha256"] == controller_hash,
        "acquisition state controller drift",
    )
    state_path = (workspace / state_identity["path"]).resolve()
    relative_to_workspace(state_path, workspace)
    require(state_path.is_file(), f"missing acquisition state: {state_path}")
    state = read_json(state_path)
    required_state_keys = {
        "schema_version",
        "lane_id",
        "status",
        "created_utc",
        "spec_sha256",
        "controller_sha256",
        "stages",
        "updated_utc",
        "host",
        "metadata",
        "completed_utc",
    }
    allowed_state_keys = required_state_keys | {"last_failure"}
    require(isinstance(state, dict), "acquisition state must be an object")
    require(required_state_keys.issubset(state), "acquisition state is incomplete")
    require(
        set(state).issubset(allowed_state_keys), "acquisition state has unexpected keys"
    )
    require(state["schema_version"] == SCHEMA_VERSION, "workspace state schema drift")
    require(state["lane_id"] == spec["lane_id"], "workspace state lane drift")
    require(state["status"] == "complete", "workspace state is not complete")
    require(
        state["created_utc"] == state_identity["created_utc"],
        "workspace state creation drift",
    )
    require(state["spec_sha256"] == EXPECTED_SPEC_SHA256, "workspace state spec drift")
    require(
        state["controller_sha256"] == controller_hash,
        "workspace state controller drift",
    )
    require_exact_keys(
        state["host"],
        {"system", "machine", "python", "free_bytes_at_start"},
        "acquisition host",
    )
    require(state["host"]["system"] == "Linux", "workspace state host is not Linux")
    require(state["host"]["machine"] == "x86_64", "workspace state host is not x86_64")
    _validate_acquired_metadata(state["metadata"], spec)
    require(
        lock["metadata_fingerprint"] == canonical_sha256(state["metadata"]),
        "metadata fingerprint/state drift",
    )

    spec_copy = workspace / DEFAULT_SPEC.name
    require(spec_copy.is_file(), f"missing workspace spec copy: {spec_copy}")
    require(sha256_file(spec_copy) == EXPECTED_SPEC_SHA256, "workspace spec copy drift")

    expected_stage_keys = {
        "metadata",
        *(f"context:{name}" for name in EXPECTED_COMPONENTS),
        "runtime_archive",
        "runtime_members",
        "artifact_lock",
    }
    require_exact_keys(state["stages"], expected_stage_keys, "acquisition stages")
    require(
        state["stages"]["metadata"] == {"status": "complete"}, "metadata stage drift"
    )
    require(
        state["stages"]["artifact_lock"]
        == {"status": "complete", "path": spec["acquisition"]["artifact_lock"]},
        "artifact-lock stage drift",
    )

    contexts = lock["contexts"]
    require(isinstance(contexts, dict), "lock contexts must be an object")
    require(tuple(contexts) == EXPECTED_COMPONENTS, "lock context order or set drift")
    context_hashes: dict[str, str] = {}
    for name in EXPECTED_COMPONENTS:
        record = contexts[name]
        require_exact_keys(
            record,
            {"target_model_id", "authenticated_object", "file"},
            f"lock context {name}",
        )
        target = spec["hub"]["components"][name]["target_model"]
        require(
            record["target_model_id"] == target["id"], f"context model ID drift: {name}"
        )
        expected_object = _expected_context_object(target)
        require_exact_keys(
            record["authenticated_object"],
            set(expected_object),
            f"context object {name}",
        )
        require(
            record["authenticated_object"] == expected_object,
            f"context object identity drift: {name}",
        )
        expected_path = spec["acquisition"]["context_output_names"][name]
        _validate_lock_file_record(
            record["file"],
            label=f"context file {name}",
            expected_path=expected_path,
            expected_bytes=target["bytes"],
        )
        stage = state["stages"][f"context:{name}"]
        require_exact_keys(
            stage,
            {
                "status",
                "completed_utc",
                "target_model_id",
                "authenticated_object",
                "file",
            },
            f"context stage {name}",
        )
        require(stage["status"] == "complete", f"context stage is incomplete: {name}")
        require(
            record
            == {
                "target_model_id": stage["target_model_id"],
                "authenticated_object": stage["authenticated_object"],
                "file": stage["file"],
            },
            f"context lock/state drift: {name}",
        )
        path = (workspace / record["file"]["path"]).resolve()
        relative_to_workspace(path, workspace)
        verify_file_record(path, record["file"])
        context_hashes[name] = record["file"]["sha256"]

    runtime_archive = lock["runtime_archive"]
    require_exact_keys(
        runtime_archive, {"authenticated_object", "file"}, "runtime archive"
    )
    expected_runtime_object = _expected_runtime_object(spec)
    require_exact_keys(
        runtime_archive["authenticated_object"],
        set(expected_runtime_object),
        "runtime archive object",
    )
    require(
        runtime_archive["authenticated_object"] == expected_runtime_object,
        "runtime archive object identity drift",
    )
    expected_archive_path = f"downloads/{spec['runtime']['filename']}"
    _validate_lock_file_record(
        runtime_archive["file"],
        label="runtime archive file",
        expected_path=expected_archive_path,
        expected_bytes=spec["runtime"]["bytes"],
    )
    runtime_stage = state["stages"]["runtime_archive"]
    require_exact_keys(
        runtime_stage,
        {"status", "completed_utc", "authenticated_object", "file"},
        "runtime archive stage",
    )
    require(
        runtime_stage["status"] == "complete", "runtime archive stage is incomplete"
    )
    require(
        runtime_archive
        == {
            "authenticated_object": runtime_stage["authenticated_object"],
            "file": runtime_stage["file"],
        },
        "runtime archive lock/state drift",
    )
    archive_path = (workspace / runtime_archive["file"]["path"]).resolve()
    relative_to_workspace(archive_path, workspace)
    verify_file_record(archive_path, runtime_archive["file"])
    verify_runtime_archive_contract(archive_path, spec)

    runtime_members = lock["runtime_members"]
    require(isinstance(runtime_members, list), "runtime members must be a list")
    expected_members = [
        _expected_runtime_member_record(spec, member)
        for member in spec["runtime"]["required_members"]
    ]
    require(len(runtime_members) == len(expected_members), "runtime member count drift")
    require(
        len({item.get("path") for item in runtime_members if isinstance(item, dict)})
        == len(runtime_members),
        "duplicate runtime member path",
    )
    for index, (record, expected) in enumerate(zip(runtime_members, expected_members)):
        require_exact_keys(
            record, {"path", "bytes", "sha256", "role"}, f"runtime member {index}"
        )
        require(
            record == expected, f"runtime member identity drift: {expected['path']}"
        )
        path = (workspace / record["path"]).resolve()
        relative_to_workspace(path, workspace)
        verify_file_record(path, record)
    member_stage = state["stages"]["runtime_members"]
    require_exact_keys(
        member_stage, {"status", "completed_utc", "files"}, "runtime member stage"
    )
    require(member_stage["status"] == "complete", "runtime member stage is incomplete")
    require(member_stage["files"] == runtime_members, "runtime member lock/state drift")

    return {
        "status": "verified",
        "artifact_lock": relative_to_workspace(lock_path, workspace),
        "artifact_lock_sha256": actual_lock_hash,
        "artifact_lock_trust": "caller-supplied trusted acquisition SHA-256",
        "controller_sha256": controller_hash,
        "qairt_build": lock["qairt_build"],
        "context_count": len(contexts),
        "context_sha256": context_hashes,
        "runtime_archive_sha256": runtime_archive["file"]["sha256"],
        "runtime_member_count": len(runtime_members),
        "apk_built": False,
        "phone_execution_proven": False,
        "npu_execution_proven": False,
    }


def acquire(
    spec_path: Path,
    spec: dict[str, Any],
    workspace: Path,
    profile: str | None,
) -> dict[str, Any]:
    workspace = safe_workspace(workspace)
    host = preflight_acquire(workspace, spec)
    state_path = workspace / ACQUIRE_STATE_FILENAME
    with workspace_lock(workspace):
        state = load_or_create_acquire_state(state_path, spec_path, spec)
        state["status"] = "running"
        state["host"] = host
        write_acquire_state(state_path, state)
        try:
            hub_metadata, targets = collect_hub_metadata(
                spec, profile=profile, retain_targets=True
            )
            runtime_metadata = collect_runtime_metadata(spec)
            state["metadata"] = {
                "observed_utc": utc_now(),
                "hub": hub_metadata,
                "runtime": runtime_metadata,
            }
            state["stages"]["metadata"] = {"status": "complete"}
            write_acquire_state(state_path, state)

            for name in EXPECTED_COMPONENTS:
                destination = (
                    workspace / spec["acquisition"]["context_output_names"][name]
                )
                stage_key = f"context:{name}"
                stage = _download_context(
                    name,
                    targets[name],
                    spec["hub"]["components"][name]["target_model"],
                    destination,
                    workspace,
                    state["stages"].get(stage_key),
                )
                state["stages"][stage_key] = stage
                write_acquire_state(state_path, state)

            runtime_archive_path = workspace / "downloads" / spec["runtime"]["filename"]
            runtime_archive = _download_runtime_archive(
                spec,
                runtime_archive_path,
                workspace,
                state["stages"].get("runtime_archive"),
            )
            state["stages"]["runtime_archive"] = runtime_archive
            write_acquire_state(state_path, state)

            output_root = workspace / spec["acquisition"]["runtime_output_root"]
            runtime_members = _extract_runtime_members(
                spec,
                runtime_archive_path,
                output_root,
                workspace,
                state["stages"].get("runtime_members"),
                state_path,
                state,
            )
            state["stages"]["runtime_members"] = runtime_members
            write_acquire_state(state_path, state)

            spec_copy = workspace / spec_path.name
            relative_to_workspace(spec_copy, workspace)
            require(not spec_copy.is_symlink(), "workspace spec copy is a symlink")
            if spec_copy.is_file():
                require(
                    sha256_file(spec_copy) == EXPECTED_SPEC_SHA256,
                    "workspace spec copy drift",
                )
            else:
                shutil.copy2(spec_path, spec_copy)
            locked = _write_artifact_lock(spec_path, spec, workspace, state)
            state["stages"]["artifact_lock"] = {
                "status": "complete",
                "path": locked["path"],
            }
            state["status"] = "complete"
            state["completed_utc"] = utc_now()
            write_acquire_state(state_path, state)
            trusted_lock_hash = sha256_file(workspace / locked["path"])
            return verify_lock(workspace, spec, trusted_lock_hash)
        except Exception as error:
            state["status"] = "failed_closed"
            state["last_failure"] = {
                "time_utc": utc_now(),
                "type": type(error).__name__,
                "message": (
                    str(error)
                    if isinstance(error, LaneError)
                    else "External operation failed; details intentionally omitted from "
                    "persistent state and remain only in the ephemeral session log."
                ),
            }
            write_acquire_state(state_path, state)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "metadata", "acquire", "verify"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--expected-lock-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path, spec = load_spec(args.spec)
    if args.command == "plan":
        print(json.dumps(plan(spec), indent=2, sort_keys=True))
        return 0
    if args.command == "metadata":
        require(
            args.state is not None,
            "--state is required for metadata; use an ephemeral path and do not "
            "overwrite the pinned historical Hub observation",
        )
        require(
            args.state.name != PINNED_METADATA_STATE_FILENAME,
            "metadata output may not reuse the pinned historical Hub state filename",
        )
        hub_metadata, _ = collect_hub_metadata(spec, profile=args.profile)
        runtime_metadata = collect_runtime_metadata(spec)
        state = metadata_state(spec_path, spec, hub_metadata, runtime_metadata)
        atomic_write_json(args.state.expanduser().resolve(), state)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    require(args.workspace is not None, "--workspace is required for acquire/verify")
    workspace = safe_workspace(args.workspace)
    if args.command == "verify":
        require(
            args.expected_lock_sha256 is not None,
            "--expected-lock-sha256 is required for standalone verify and must come "
            "from the trusted acquisition output or a separately pinned record",
        )
        result = verify_lock(workspace, spec, args.expected_lock_sha256)
    else:
        result = acquire(spec_path, spec, workspace, args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaneError as error:
        print(
            json.dumps({"status": "failed_closed", "error": str(error)}, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(2)
