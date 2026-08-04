#!/usr/bin/env python3
"""Resume-safe Google Colab QAIRT 2.42 Supertonic duration canary.

This controller deliberately supports one experiment: convert the exact
3.7 MB Supertonic duration-predictor ONNX graph, build its x86_64 QNN model
library, and generate an offline SM8850/HTP-v81 context with the exact QAIRT
2.42 community SDK.  It is designed for a persistent Google Drive workspace
on an Ubuntu 22.04 Colab runtime.  It must not be used for conversion on macOS.

The controller uses no credentials.  Downloads are accepted only after their
pinned SHA-256 checks pass.  Every durable stage is content-addressed and is
reused only while all recorded output hashes still match.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / (
    "reports/android_phone/model_sweep_20260729/"
    "qairt242_colab_supertonic_duration_spec.json"
)
EXPECTED_SPEC_SHA256 = (
    "08e1425f1daba9626827dfce3b072f1191ff840c1e0f555743f39614546702ec"
)
EXPECTED_EXPERIMENT_ID = (
    "qairt242_colab_supertonic_duration_sm8850_htp_v81_v1"
)
EXPECTED_QAIRT_SHA256 = (
    "1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321"
)
EXPECTED_SUPERTONIC_SHA256 = (
    "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427"
)
EXPECTED_MODEL_SHA256 = (
    "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"
)
CONTROLLER_SCHEMA_VERSION = 1
MIN_FREE_BYTES = 8 * 1024**3


class CanaryError(RuntimeError):
    """A fail-closed canary validation or execution error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def validate_sha256(value: Any, label: str) -> str:
    text = str(value)
    require(bool(re.fullmatch(r"[0-9a-f]{64}", text)), f"invalid {label} SHA-256")
    return text


def validate_spec(spec: dict[str, Any], spec_path: Path | None = None) -> None:
    require(spec.get("schema_version") == 1, "unexpected spec schema")
    require(
        spec.get("experiment_id") == EXPECTED_EXPERIMENT_ID,
        "unexpected experiment identity",
    )
    if spec_path is not None:
        require(spec_path.is_file(), f"missing spec: {spec_path}")
        require(
            sha256_file(spec_path) == EXPECTED_SPEC_SHA256,
            "spec SHA-256 does not match the controller trust root",
        )

    policy = spec.get("execution_policy", {})
    require(policy.get("mac_execution_allowed") is False, "macOS must remain blocked")
    require(policy.get("credentials_required") is False, "credentials are forbidden")
    require(
        policy.get("network_hosts")
        == ["softwarecenter.qualcomm.com", "github.com"],
        "network host allowlist changed",
    )
    fallback = policy.get("fallback_policy", {})
    for key in (
        "allow_other_qairt_version",
        "allow_qairt_245_context_in_qairt_242_phone_runtime",
        "allow_other_supertonic_graph",
        "allow_unverified_download",
    ):
        require(fallback.get(key) is False, f"fallback must remain disabled: {key}")

    dependencies = spec.get("host_dependencies", {})
    require(dependencies.get("ubuntu_version") == "22.04", "Ubuntu pin changed")
    require(
        dependencies.get("python_version_prefix") == "3.10.",
        "Python pin changed",
    )
    require(dependencies.get("clang_major") == 14, "clang pin changed")
    require(
        dependencies.get("python_packages")
        == {
            "matplotlib": "3.8.4",
            "numpy": "1.26.4",
            "onnx": "1.16.1",
            "onnxruntime": "1.18.0",
            "onnxsim": "0.4.36",
            "protobuf": "3.20.3",
            "pyyaml": "6.0.2",
        },
        "Python package pins changed",
    )
    require(
        dependencies.get("compatibility_note")
        == (
            "QAIRT's dependency checker pins older matplotlib/PyYAML/protobuf "
            "builds that are not a reliable Python 3.10 wheel set or conflict "
            "with ONNX 1.16.1 metadata. Install these exact Python-3.10-compatible "
            "overrides before running the SDK checker; the converter/framework "
            "pins remain fixed."
        ),
        "Python compatibility policy changed",
    )

    archives = spec.get("archives", {})
    qairt = archives.get("qairt", {})
    supertonic = archives.get("supertonic", {})
    require(
        qairt.get("url")
        == "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        "Qualcomm_AI_Runtime_Community/All/2.42.0.251225/"
        "v2.42.0.251225.zip",
        "QAIRT URL changed",
    )
    require(
        validate_sha256(qairt.get("sha256"), "QAIRT") == EXPECTED_QAIRT_SHA256,
        "QAIRT hash changed",
    )
    require(qairt.get("bytes") == 1543955191, "QAIRT byte size changed")
    require(
        qairt.get("sdk_root_member") == "qairt/2.42.0.251225",
        "QAIRT SDK root changed",
    )
    require(
        qairt.get("build_id") == "2.42.0.251225135753_193295",
        "QAIRT build ID changed",
    )
    require(
        supertonic.get("url")
        == "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        "Supertonic URL changed",
    )
    require(
        validate_sha256(supertonic.get("sha256"), "Supertonic")
        == EXPECTED_SUPERTONIC_SHA256,
        "Supertonic archive hash changed",
    )

    component = spec.get("component", {})
    require(component.get("name") == "duration_predictor", "component changed")
    require(
        component.get("archive_basename") == "duration_predictor.int8.onnx",
        "model member changed",
    )
    require(component.get("source_bytes") == 3700147, "model size changed")
    require(
        validate_sha256(component.get("source_sha256"), "model")
        == EXPECTED_MODEL_SHA256,
        "model hash changed",
    )
    contract = component.get("onnx_contract", {})
    require(contract.get("ir_version") == 9, "ONNX IR version changed")
    require(contract.get("opset") == 19, "ONNX opset changed")
    require(contract.get("graph_name") == "main_graph", "ONNX graph changed")
    require(
        contract.get("inputs")
        == {
            "style_dp": {
                "source_dtype": "float32",
                "qnn_boundary_dtype": "float32",
                "shape": [1, 8, 16],
            },
            "text_ids": {
                "source_dtype": "int64",
                "qnn_boundary_dtype": "int32",
                "shape": [1, 32],
                "boundary_note": (
                    "Matches the existing --truncate_64bit_io compile contract; "
                    "Gather accepts int32 indices. No weight or node rewrite is allowed."
                ),
            },
            "text_mask": {
                "source_dtype": "float32",
                "qnn_boundary_dtype": "float32",
                "shape": [1, 1, 32],
            },
        },
        "fixed input contract changed",
    )
    require(
        contract.get("outputs") == {"duration": {"dtype": "float32", "shape": [1]}},
        "output contract changed",
    )

    target = spec.get("target", {})
    require(target.get("chipset") == "SM8850", "target chipset changed")
    require(target.get("soc_model") == 87, "SM8850 SoC model must be 87")
    require(target.get("dsp_arch") == "v81", "HTP architecture must be v81")
    require(target.get("target_abi") == "aarch64-android", "target ABI changed")
    require(target.get("backend") == "HTP", "backend changed")
    require(target.get("pd_session") == "unsigned", "PD session changed")
    require(target.get("graph_optimization_level") == 3, "optimization changed")
    require(target.get("vtcm_mb") == 8, "VTCM setting changed")
    require(target.get("qnn_precision") == "FLOAT16", "precision changed")

    tools = spec.get("tool_contract", {})
    expected_tools = {
        "converter": "bin/x86_64-linux-clang/qnn-onnx-converter",
        "model_lib_generator": "bin/x86_64-linux-clang/qnn-model-lib-generator",
        "context_binary_generator": (
            "bin/x86_64-linux-clang/qnn-context-binary-generator"
        ),
        "context_binary_utility": (
            "bin/x86_64-linux-clang/qnn-context-binary-utility"
        ),
        "htp_backend": "lib/x86_64-linux-clang/libQnnHtp.so",
        "htp_extension": (
            "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
        ),
        "android_runtime_inventory": [
            "lib/aarch64-android/libQnnHtp.so",
            "lib/aarch64-android/libQnnHtpV81Stub.so",
            "lib/aarch64-android/libQnnSystem.so",
            "lib/hexagon-v81/unsigned/libQnnHtpV81.so",
            "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so",
        ],
    }
    require(tools == expected_tools, "QAIRT tool/runtime contract changed")
    output = spec.get("output_contract", {})
    require(
        output.get("context_basename")
        == "supertonic_duration_sm8850_htp_v81_qairt242",
        "context basename changed",
    )
    require(
        output.get("required_manifest_status") == "context_generated_unexecuted",
        "manifest status changed",
    )
    require(output.get("minimum_context_bytes") == 1024, "context floor changed")
    require(
        output.get("include_runtime_binaries_in_bundle") is False,
        "runtime binaries must not be bundled",
    )


def load_spec(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    spec = read_json(resolved)
    require(isinstance(spec, dict), "spec root must be an object")
    validate_spec(spec, resolved)
    return resolved, spec


def safe_workspace(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    require(resolved != Path("/"), "workspace cannot be filesystem root")
    require(len(resolved.parts) >= 3, "workspace path is too broad")
    return resolved


def relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise CanaryError(f"path escapes workspace: {path}") from error


def inventory_files(root: Path, workspace: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": relative_to_workspace(path, workspace),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    require(records, f"stage produced no files: {root.name}")
    return records


def verify_inventory(records: list[dict[str, Any]], workspace: Path) -> None:
    require(bool(records), "empty output inventory")
    for record in records:
        relative = Path(str(record.get("path", "")))
        require(not relative.is_absolute(), "inventory contains an absolute path")
        path = (workspace / relative).resolve()
        relative_to_workspace(path, workspace)
        require(path.is_file(), f"missing recorded output: {relative}")
        require(path.stat().st_size == record.get("bytes"), f"size drift: {relative}")
        require(
            sha256_file(path) == record.get("sha256"), f"hash drift: {relative}"
        )


def initial_state(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "spec_path_name": spec_path.name,
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "controller_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": spec["claim_boundary"],
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "initialized",
        "archives": {},
        "stages": {},
    }


def load_or_create_state(
    path: Path, spec_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    if path.is_file():
        state = read_json(path)
        require(isinstance(state, dict), "state root must be an object")
        require(state.get("schema_version") == 1, "unexpected state schema")
        require(
            state.get("experiment_id") == EXPECTED_EXPERIMENT_ID,
            "state belongs to another experiment",
        )
        require(
            state.get("spec_sha256") == EXPECTED_SPEC_SHA256,
            "state belongs to another spec",
        )
        require(
            state.get("controller_sha256") == sha256_file(Path(__file__).resolve()),
            "controller changed after this state was created; use a new workspace",
        )
        return state
    state = initial_state(spec_path, spec)
    atomic_write_json(path, state)
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    atomic_write_json(path, state)


@contextmanager
def workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / "qairt242_colab.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CanaryError("another canary process holds the workspace lock") from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        stream.flush()
        yield
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def quarantine(path: Path, workspace: Path, reason: str) -> Path:
    relative_to_workspace(path, workspace)
    target_root = workspace / "quarantine"
    target_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = target_root / f"{path.name}.{reason}.{stamp}.{os.getpid()}"
    counter = 0
    while target.exists():
        counter += 1
        target = target_root / f"{path.name}.{reason}.{stamp}.{os.getpid()}.{counter}"
    path.replace(target)
    return target


def stage_fingerprint(
    name: str, spec_sha256: str, inputs: dict[str, Any]
) -> str:
    return canonical_sha256(
        {
            "controller_schema": CONTROLLER_SCHEMA_VERSION,
            "name": name,
            "spec_sha256": spec_sha256,
            "inputs": inputs,
        }
    )


StageAction = Callable[[Path], dict[str, Any]]


def run_stage(
    name: str,
    fingerprint: str,
    workspace: Path,
    state_path: Path,
    state: dict[str, Any],
    action: StageAction,
) -> dict[str, Any]:
    stages = state.setdefault("stages", {})
    existing = stages.get(name)
    final_dir = workspace / "stages" / name
    if existing and existing.get("status") == "complete":
        require(
            existing.get("fingerprint") == fingerprint,
            f"completed stage fingerprint changed: {name}",
        )
        verify_inventory(existing.get("outputs", []), workspace)
        return existing

    if final_dir.exists():
        quarantine(final_dir, workspace, "orphaned")
    partial_root = workspace / "stages"
    partial_root.mkdir(parents=True, exist_ok=True)
    for stale in sorted(partial_root.glob(f".{name}.partial.*")):
        quarantine(stale, workspace, "interrupted")
    partial = partial_root / f".{name}.partial.{os.getpid()}"
    partial.mkdir(parents=True)
    stages[name] = {
        "status": "in_progress",
        "fingerprint": fingerprint,
        "started_utc": utc_now(),
    }
    write_state(state_path, state)
    try:
        details = action(partial)
        partial.replace(final_dir)
        outputs = inventory_files(final_dir, workspace)
        record = {
            "status": "complete",
            "fingerprint": fingerprint,
            "started_utc": stages[name]["started_utc"],
            "completed_utc": utc_now(),
            "details": details,
            "outputs": outputs,
        }
        stages[name] = record
        write_state(state_path, state)
        return record
    except Exception as error:
        stages[name] = {
            "status": "failed",
            "fingerprint": fingerprint,
            "started_utc": stages[name]["started_utc"],
            "failed_utc": utc_now(),
            "error": {"type": type(error).__name__, "message": str(error)},
            "partial_path": relative_to_workspace(partial, workspace),
        }
        write_state(state_path, state)
        raise


def run_logged(
    command: list[str], cwd: Path, env: dict[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND_JSON=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise CanaryError(
            f"command failed with exit {result.returncode}: {command[0]}\n{tail}"
        )


def command_output(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    require(result.returncode == 0, f"command failed: {command}")
    return result.stdout.strip()


def parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_major_version(text: str, label: str) -> int:
    match = re.search(r"\b(\d+)(?:\.\d+)+", text)
    require(match is not None, f"could not parse {label} version")
    return int(match.group(1))


def parse_version_tuple(text: str, label: str) -> tuple[int, ...]:
    match = re.search(r"\b(\d+(?:\.\d+)+)", text)
    require(match is not None, f"could not parse {label} version")
    return tuple(int(item) for item in match.group(1).split("."))


def host_preflight(workspace: Path, spec: dict[str, Any]) -> dict[str, Any]:
    require(platform.system() == "Linux", "run is allowed only on Linux/Colab")
    require(
        platform.machine().lower() in {"x86_64", "amd64"},
        "QAIRT host tools require x86_64 Linux",
    )
    release = parse_os_release()
    require(
        release.get("VERSION_ID") == spec["host_dependencies"]["ubuntu_version"],
        "QAIRT 2.42 canary requires Ubuntu 22.04",
    )
    require(sys.version_info[:2] == (3, 10), "invoke the controller with Python 3.10")
    required_commands = [
        "curl",
        "unzip",
        "tar",
        "make",
        "cmake",
        "clang-14",
        "clang++-14",
    ]
    missing = [name for name in required_commands if shutil.which(name) is None]
    require(not missing, f"missing host commands: {missing}")
    clang = command_output(["clang-14", "--version"])
    require(parse_major_version(clang, "clang") == 14, "clang must be version 14")
    cmake = command_output(["cmake", "--version"])
    require(
        parse_version_tuple(cmake, "cmake") >= (3, 22),
        "cmake must be version 3.22 or newer",
    )
    workspace.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workspace).free
    require(
        free >= MIN_FREE_BYTES,
        f"workspace needs at least {MIN_FREE_BYTES} free bytes; found {free}",
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ubuntu_version": release.get("VERSION_ID"),
        "python": platform.python_version(),
        "clang": clang.splitlines()[0],
        "cmake": cmake.splitlines()[0],
        "workspace_free_bytes_at_preflight": free,
    }


def verified_download(
    name: str,
    record: dict[str, Any],
    workspace: Path,
    state_path: Path,
    state: dict[str, Any],
) -> Path:
    downloads = workspace / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    final = downloads / str(record["filename"])
    expected_hash = validate_sha256(record["sha256"], name)
    expected_bytes = record.get("bytes")
    if final.is_file():
        actual_hash = sha256_file(final)
        require(actual_hash == expected_hash, f"existing {name} archive hash drift")
        if expected_bytes is not None:
            require(final.stat().st_size == expected_bytes, f"{name} archive size drift")
        state.setdefault("archives", {})[name] = {
            "status": "verified",
            "filename": final.name,
            "bytes": final.stat().st_size,
            "sha256": actual_hash,
            "reused": True,
        }
        write_state(state_path, state)
        return final

    partial = final.with_suffix(final.suffix + ".part")
    log = downloads / f"{final.name}.download.log"
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        str(record["url"]),
    ]
    run_logged(command, downloads, dict(os.environ), log)
    actual_hash = sha256_file(partial)
    actual_bytes = partial.stat().st_size
    if actual_hash != expected_hash or (
        expected_bytes is not None and actual_bytes != expected_bytes
    ):
        quarantine(partial, workspace, "download-hash-mismatch")
        raise CanaryError(f"downloaded {name} archive failed its exact identity gate")
    partial.replace(final)
    state.setdefault("archives", {})[name] = {
        "status": "verified",
        "filename": final.name,
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "reused": False,
        "verified_utc": utc_now(),
    }
    write_state(state_path, state)
    return final


def prepare_sdk(
    archive: Path,
    scratch: Path,
    workspace: Path,
    spec: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    scratch = scratch.expanduser().resolve()
    require(scratch != Path("/"), "scratch cannot be filesystem root")
    scratch.mkdir(parents=True, exist_ok=True)
    sdk_container = scratch / "qairt242-sdk"
    marker_path = sdk_container / ".qairt242_colab_extraction.json"
    sdk_root = sdk_container / spec["archives"]["qairt"]["sdk_root_member"]

    critical = [
        *[
            spec["tool_contract"][key]
            for key in (
                "converter",
                "model_lib_generator",
                "context_binary_generator",
                "context_binary_utility",
                "htp_backend",
                "htp_extension",
            )
        ],
        *spec["tool_contract"]["android_runtime_inventory"],
        "bin/envsetup.sh",
        "bin/check-python-dependency",
    ]
    if marker_path.is_file() and sdk_root.is_dir():
        marker = read_json(marker_path)
        if marker.get("archive_sha256") == EXPECTED_QAIRT_SHA256:
            try:
                for record in marker.get("critical_files", []):
                    path = sdk_root / record["path"]
                    require(path.is_file(), f"missing SDK file: {record['path']}")
                    require(path.stat().st_size == record["bytes"], "SDK size drift")
                    require(sha256_file(path) == record["sha256"], "SDK hash drift")
                require(
                    {item["path"] for item in marker.get("critical_files", [])}
                    == set(critical),
                    "SDK critical inventory changed",
                )
                return sdk_root, marker
            except CanaryError:
                pass
        quarantine(sdk_container, workspace, "sdk-cache-invalid") if (
            workspace in sdk_container.parents
        ) else sdk_container.replace(
            scratch / f"qairt242-sdk.invalid.{os.getpid()}"
        )

    partial = scratch / f".qairt242-sdk.partial.{os.getpid()}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    result = subprocess.run(
        ["unzip", "-q", str(archive), "-d", str(partial)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CanaryError(f"QAIRT extraction failed: {result.stdout[-8000:]}")
    extracted_root = partial / spec["archives"]["qairt"]["sdk_root_member"]
    require(extracted_root.is_dir(), "pinned QAIRT SDK root was not extracted")
    records = []
    for relative in critical:
        path = extracted_root / relative
        require(path.is_file(), f"QAIRT archive lacks required file: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    marker = {
        "schema_version": 1,
        "archive_sha256": EXPECTED_QAIRT_SHA256,
        "sdk_build_id": spec["archives"]["qairt"]["build_id"],
        "critical_files": records,
        "created_utc": utc_now(),
    }
    atomic_write_json(partial / ".qairt242_colab_extraction.json", marker)
    if sdk_container.exists():
        raise CanaryError("SDK extraction target appeared concurrently")
    partial.replace(sdk_container)
    return sdk_container / spec["archives"]["qairt"]["sdk_root_member"], marker


def sdk_environment(sdk_root: Path, venv: Path) -> dict[str, str]:
    env = dict(os.environ)
    bin_dir = sdk_root / "bin/x86_64-linux-clang"
    python_dir = sdk_root / "lib/python"
    lib_dir = sdk_root / "lib/x86_64-linux-clang"
    env.update(
        {
            "QAIRT_SDK_ROOT": str(sdk_root),
            "QNN_SDK_ROOT": str(sdk_root),
            "SNPE_ROOT": str(sdk_root),
            "VIRTUAL_ENV": str(venv),
            "PATH": os.pathsep.join(
                [str(venv / "bin"), str(bin_dir), env.get("PATH", "")]
            ),
            "PYTHONPATH": os.pathsep.join(
                [str(python_dir), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            "LD_LIBRARY_PATH": os.pathsep.join(
                [str(lib_dir), env.get("LD_LIBRARY_PATH", "")]
            ).rstrip(os.pathsep),
            "CC": shutil.which("clang-14") or "clang-14",
            "CXX": shutil.which("clang++-14") or "clang++-14",
        }
    )
    return env


def prepare_venv(
    scratch: Path,
    sdk_root: Path,
    workspace: Path,
    spec: dict[str, Any],
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    venv = scratch / "qairt242-python310"
    marker_path = venv / ".qairt242_colab_venv.json"
    pins = spec["host_dependencies"]["python_packages"]
    expected = {
        "python": "3.10",
        "sdk_archive_sha256": EXPECTED_QAIRT_SHA256,
        "pins": pins,
    }
    if marker_path.is_file():
        marker = read_json(marker_path)
        if marker == expected and (venv / "bin/python").is_file():
            ensure_compiler_shims(venv)
            env = sdk_environment(sdk_root, venv)
            return venv, env, package_inventory(venv / "bin/python", env, pins)
        if workspace in venv.parents:
            quarantine(venv, workspace, "venv-invalid")
        else:
            venv.replace(scratch / f"qairt242-python310.invalid.{os.getpid()}")

    subprocess.run(
        [sys.executable, "-m", "venv", "--clear", str(venv)], check=True
    )
    ensure_compiler_shims(venv)
    env = sdk_environment(sdk_root, venv)
    python = venv / "bin/python"
    bootstrap_log = scratch / "qairt242-python-bootstrap.log"
    run_logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip==24.0",
            "setuptools==69.5.1",
            "wheel==0.43.0",
        ],
        scratch,
        env,
        bootstrap_log,
    )
    compatibility_log = scratch / "qairt242-python-compatibility-overrides.log"
    compatibility_packages = [
        f"{name}=={pins[name]}" for name in ("matplotlib", "protobuf", "pyyaml")
    ]
    run_logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            *compatibility_packages,
        ],
        scratch,
        env,
        compatibility_log,
    )
    sdk_dependency_log = scratch / "qairt242-sdk-python-dependencies.log"
    run_logged(
        [str(python), str(sdk_root / "bin/check-python-dependency")],
        scratch,
        env,
        sdk_dependency_log,
    )
    framework_log = scratch / "qairt242-framework-dependencies.log"
    packages = [f"{name}=={version}" for name, version in sorted(pins.items())]
    run_logged(
        [str(python), "-m", "pip", "install", "--upgrade", *packages],
        scratch,
        env,
        framework_log,
    )
    inventory = package_inventory(python, env, pins)
    atomic_write_json(marker_path, expected)
    return venv, env, inventory


def ensure_compiler_shims(venv: Path) -> None:
    for name, versioned in (("clang", "clang-14"), ("clang++", "clang++-14")):
        target_text = shutil.which(versioned)
        require(target_text is not None, f"missing compiler: {versioned}")
        target = Path(target_text).resolve()
        link = venv / "bin" / name
        if link.exists() or link.is_symlink():
            require(link.resolve() == target, f"compiler shim drift: {link}")
        else:
            link.symlink_to(target)


def package_inventory(
    python: Path, env: dict[str, str], pins: dict[str, str]
) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,platform; "
        "names=json.loads(__import__('sys').argv[1]); "
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{n:m.version(n) for n in names}},sort_keys=True))"
    )
    output = command_output([str(python), "-c", code, json.dumps(sorted(pins))], env)
    record = json.loads(output)
    require(record.get("python", "").startswith("3.10."), "venv Python drift")
    require(record.get("packages") == pins, "pinned Python package drift")
    installed = json.loads(
        command_output([str(python), "-m", "pip", "list", "--format=json"], env)
    )
    record["all_packages"] = dict(
        sorted((str(item["name"]).lower(), str(item["version"])) for item in installed)
    )
    record["all_packages_sha256"] = canonical_sha256(record["all_packages"])
    return record


def extract_duration_model(
    archive: Path,
    destination: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    component = spec["component"]
    basename = component["archive_basename"]
    with tarfile.open(archive, "r:bz2") as source:
        matches = [
            member
            for member in source.getmembers()
            if Path(member.name).name == basename
        ]
        require(len(matches) == 1, "Supertonic archive must contain one duration graph")
        member = matches[0]
        require(member.isfile(), "duration graph member is not a regular file")
        require(not member.name.startswith("/"), "absolute archive member is forbidden")
        require(".." not in Path(member.name).parts, "unsafe archive member path")
        require(member.size == component["source_bytes"], "duration member size drift")
        stream = source.extractfile(member)
        require(stream is not None, "could not read duration graph member")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
    require(destination.stat().st_size == component["source_bytes"], "model size drift")
    require(
        sha256_file(destination) == component["source_sha256"], "model hash drift"
    )
    return {
        "archive_member": member.name,
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


FIXED_MODEL_HELPER = r'''
import hashlib
import json
from pathlib import Path
import sys
import onnx
from onnx import TensorProto

source = Path(sys.argv[1])
target = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
contract = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def dtype_name(value):
    return {
        TensorProto.FLOAT: "float32",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
    }.get(value, f"onnx:{value}")

def dims(value):
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(str(dim.dim_param))
    return result

model = onnx.load(str(source), load_external_data=False)
if model.ir_version != contract["ir_version"]:
    raise RuntimeError("IR version drift")
opsets = {item.domain: item.version for item in model.opset_import}
if opsets.get("") != contract["opset"]:
    raise RuntimeError("ONNX opset drift")
if model.graph.name != contract["graph_name"]:
    raise RuntimeError("graph-name drift")
inputs = {item.name: item for item in model.graph.input}
outputs = {item.name: item for item in model.graph.output}
if set(inputs) != set(contract["inputs"]):
    raise RuntimeError("input-name drift")
if set(outputs) != set(contract["outputs"]):
    raise RuntimeError("output-name drift")

node_digest_before = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.node)
).hexdigest()
initializer_digest_before = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.initializer)
).hexdigest()

dtype_values = {
    "float32": TensorProto.FLOAT,
    "int64": TensorProto.INT64,
    "int32": TensorProto.INT32,
}
source_inputs = {}
expected_source_shapes = {
    "text_ids": ["batch_size", "text_length"],
    "style_dp": ["batch_size", 8, 16],
    "text_mask": ["batch_size", 1, "text_length"],
}
for name, expected in contract["inputs"].items():
    value = inputs[name]
    actual_dtype = dtype_name(value.type.tensor_type.elem_type)
    if actual_dtype != expected["source_dtype"]:
        raise RuntimeError(f"source dtype drift for {name}: {actual_dtype}")
    actual_shape = dims(value)
    if actual_shape != expected_source_shapes[name]:
        raise RuntimeError(f"source shape drift for {name}: {actual_shape}")
    source_inputs[name] = {"dtype": actual_dtype, "shape": actual_shape}
    value.type.tensor_type.elem_type = dtype_values[expected["qnn_boundary_dtype"]]
    shape = value.type.tensor_type.shape
    del shape.dim[:]
    for size in expected["shape"]:
        shape.dim.add().dim_value = int(size)

for name, expected in contract["outputs"].items():
    value = outputs[name]
    actual_dtype = dtype_name(value.type.tensor_type.elem_type)
    if actual_dtype != expected["dtype"]:
        raise RuntimeError(f"output dtype drift for {name}: {actual_dtype}")
    shape = value.type.tensor_type.shape
    del shape.dim[:]
    for size in expected["shape"]:
        shape.dim.add().dim_value = int(size)

onnx.checker.check_model(model, full_check=True)
node_digest_after = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.node)
).hexdigest()
initializer_digest_after = hashlib.sha256(
    b"".join(item.SerializeToString() for item in model.graph.initializer)
).hexdigest()
if node_digest_before != node_digest_after:
    raise RuntimeError("node payload changed while fixing I/O")
if initializer_digest_before != initializer_digest_after:
    raise RuntimeError("initializer payload changed while fixing I/O")

target.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(target))
reloaded = onnx.load(str(target), load_external_data=False)
onnx.checker.check_model(reloaded, full_check=True)
metadata = {
    "schema_version": 1,
    "source": {"bytes": source.stat().st_size, "sha256": sha256(source)},
    "derived": {"bytes": target.stat().st_size, "sha256": sha256(target)},
    "ir_version": reloaded.ir_version,
    "opsets": {item.domain: item.version for item in reloaded.opset_import},
    "graph_name": reloaded.graph.name,
    "node_count": len(reloaded.graph.node),
    "initializer_count": len(reloaded.graph.initializer),
    "node_payload_sha256": node_digest_after,
    "initializer_payload_sha256": initializer_digest_after,
    "source_inputs": source_inputs,
    "fixed_inputs": {
        item.name: {
            "dtype": dtype_name(item.type.tensor_type.elem_type),
            "shape": dims(item),
        }
        for item in reloaded.graph.input
    },
    "fixed_outputs": {
        item.name: {
            "dtype": dtype_name(item.type.tensor_type.elem_type),
            "shape": dims(item),
        }
        for item in reloaded.graph.output
    },
    "allowed_mutation": "Concrete I/O shapes plus text_ids int64-to-int32 boundary only",
}
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
'''


def converter_command(
    sdk_root: Path, fixed_model: Path, output_cpp: Path, spec: dict[str, Any]
) -> list[str]:
    contract = spec["component"]["onnx_contract"]
    command = [
        str(sdk_root / spec["tool_contract"]["converter"]),
        "--input_network",
        str(fixed_model),
        "--output_path",
        str(output_cpp),
        "--out_node",
        "duration",
        "--preserve_io",
        "datatype",
    ]
    for name in ("text_ids", "style_dp", "text_mask"):
        record = contract["inputs"][name]
        command.extend(["--input_dim", name, ",".join(map(str, record["shape"]))])
        command.extend(["--input_dtype", name, record["qnn_boundary_dtype"]])
    return command


def model_lib_command(
    sdk_root: Path, cpp: Path, weights: Path, output: Path, spec: dict[str, Any]
) -> list[str]:
    return [
        str(sdk_root / spec["tool_contract"]["model_lib_generator"]),
        "-c",
        str(cpp),
        "-b",
        str(weights),
        "-t",
        "x86_64-linux-clang",
        "-l",
        "supertonic_duration_qairt242",
        "-o",
        str(output),
    ]


def backend_configs(
    sdk_root: Path, graph_name: str, spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = spec["target"]
    htp = {
        "graphs": [
            {
                "graph_names": [graph_name],
                "vtcm_mb": target["vtcm_mb"],
                "O": target["graph_optimization_level"],
                "fp16_relaxed_precision": 1,
            }
        ],
        "devices": [
            {
                "dsp_arch": target["dsp_arch"],
                "soc_model": target["soc_model"],
                "pd_session": target["pd_session"],
            }
        ],
    }
    wrapper = {
        "backend_extensions": {
            "shared_library_path": str(
                sdk_root / spec["tool_contract"]["htp_extension"]
            ),
            "config_file_path": "HTP_CONFIG_PATH_PLACEHOLDER",
        },
        "soc_configs": {"soc_model": target["soc_model"]},
    }
    return htp, wrapper


def context_command(
    sdk_root: Path,
    model_library: Path,
    output_dir: Path,
    wrapper_config: Path,
    spec: dict[str, Any],
) -> list[str]:
    return [
        str(sdk_root / spec["tool_contract"]["context_binary_generator"]),
        "--backend",
        str(sdk_root / spec["tool_contract"]["htp_backend"]),
        "--model",
        str(model_library),
        "--binary_file",
        spec["output_contract"]["context_basename"],
        "--output_dir",
        str(output_dir),
        "--config_file",
        str(wrapper_config),
        "--soc_model",
        str(spec["target"]["soc_model"]),
        "--profiling_level",
        "basic",
        "--log_level",
        "info",
    ]


def parse_graph_name(cpp: Path) -> str:
    text = cpp.read_text(encoding="utf-8", errors="replace")
    names = set(
        re.findall(r'getQnnGraphConfigFromInfo\s*\(\s*"([^"\\]+)"', text)
    )
    require(len(names) == 1, f"could not determine one QNN graph name: {sorted(names)}")
    return next(iter(names))


def find_unique(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    require(len(matches) == 1, f"expected one {label}, found {len(matches)}")
    return matches[0]


def tool_inventory(sdk_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    records = {}
    relative_paths = [
        *[
            spec["tool_contract"][key]
            for key in (
                "converter",
                "model_lib_generator",
                "context_binary_generator",
                "context_binary_utility",
                "htp_backend",
                "htp_extension",
            )
        ],
        *spec["tool_contract"]["android_runtime_inventory"],
    ]
    for relative in relative_paths:
        path = sdk_root / relative
        require(path.is_file(), f"missing SDK inventory file: {relative}")
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def deterministic_tar_gz(source_dir: Path, output: Path) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
                    relative = path.relative_to(source_dir).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)


def plan(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": "plan_only",
        "host_execution": False,
        "component": spec["component"]["name"],
        "source": {
            "bytes": spec["component"]["source_bytes"],
            "sha256": spec["component"]["source_sha256"],
        },
        "fixed_inputs": {
            name: {
                "shape": record["shape"],
                "dtype": record["qnn_boundary_dtype"],
            }
            for name, record in spec["component"]["onnx_contract"]["inputs"].items()
        },
        "target": copy.deepcopy(spec["target"]),
        "archives": {
            name: {
                "url": record["url"],
                "sha256": record["sha256"],
                "bytes": record.get("bytes"),
            }
            for name, record in spec["archives"].items()
        },
        "stages": [
            "verified downloads",
            "duration graph extraction only",
            "fixed-shape/int32-boundary ONNX derivation",
            "QNN ONNX conversion",
            "x86_64 QNN model-library generation",
            "SM8850/HTP-v81 offline context generation",
            "context metadata and deterministic result bundle",
        ],
        "forbidden": [
            "macOS conversion",
            "another QAIRT version",
            "another Supertonic graph",
            "unverified download",
            "QAIRT 2.45 fallback",
            "phone execution claim",
        ],
    }


def verify_completed(workspace: Path, state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    require(state.get("status") == "complete", "canary state is not complete")
    for name, stage in state.get("stages", {}).items():
        if stage.get("status") == "complete":
            verify_inventory(stage.get("outputs", []), workspace)
    require(
        all(
            state.get("stages", {}).get(name, {}).get("status") == "complete"
            for name in (
                "source_model",
                "fixed_model",
                "qnn_conversion",
                "model_library",
                "context_binary",
                "deliverable",
            )
        ),
        "one or more required stages are incomplete",
    )
    manifest_path = workspace / "stages/deliverable/result_manifest.json"
    manifest = read_json(manifest_path)
    require(
        manifest.get("status")
        == spec["output_contract"]["required_manifest_status"],
        "result manifest status drift",
    )
    require(manifest.get("spec_sha256") == EXPECTED_SPEC_SHA256, "manifest spec drift")
    require(manifest.get("phone_execution_proven") is False, "invalid phone claim")
    bundle = workspace / "stages/deliverable/qairt242_colab_result.tar.gz"
    digest_text = (
        workspace / "stages/deliverable/qairt242_colab_result.tar.gz.sha256"
    ).read_text(encoding="utf-8")
    expected = digest_text.split()[0]
    require(sha256_file(bundle) == expected, "deliverable bundle hash drift")
    return {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": "verified_complete",
        "manifest": relative_to_workspace(manifest_path, workspace),
        "bundle": relative_to_workspace(bundle, workspace),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_sha256": expected,
    }


def execute(
    spec_path: Path,
    spec: dict[str, Any],
    workspace: Path,
    scratch: Path,
) -> dict[str, Any]:
    workspace = safe_workspace(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / "qairt242_colab_state.json"
    with workspace_lock(workspace):
        state = load_or_create_state(state_path, spec_path, spec)
        if state.get("status") == "complete":
            return verify_completed(workspace, state, spec)
        try:
            state["status"] = "running"
            state["host"] = host_preflight(workspace, spec)
            write_state(state_path, state)

            qairt_archive = verified_download(
                "qairt", spec["archives"]["qairt"], workspace, state_path, state
            )
            supertonic_archive = verified_download(
                "supertonic",
                spec["archives"]["supertonic"],
                workspace,
                state_path,
                state,
            )

            source_fp = stage_fingerprint(
                "source_model",
                EXPECTED_SPEC_SHA256,
                {"archive_sha256": EXPECTED_SUPERTONIC_SHA256},
            )

            def source_action(partial: Path) -> dict[str, Any]:
                return extract_duration_model(
                    supertonic_archive,
                    partial / spec["component"]["archive_basename"],
                    spec,
                )

            source_stage = run_stage(
                "source_model",
                source_fp,
                workspace,
                state_path,
                state,
                source_action,
            )
            source_model = (
                workspace
                / "stages/source_model"
                / spec["component"]["archive_basename"]
            )

            sdk_root, extraction_inventory = prepare_sdk(
                qairt_archive, scratch, workspace, spec
            )
            venv, env, packages = prepare_venv(
                scratch, sdk_root, workspace, spec
            )
            state["sdk"] = {
                "archive_sha256": EXPECTED_QAIRT_SHA256,
                "build_id": spec["archives"]["qairt"]["build_id"],
                "critical_files": extraction_inventory["critical_files"],
            }
            state["python_environment"] = packages
            write_state(state_path, state)

            fixed_fp = stage_fingerprint(
                "fixed_model",
                EXPECTED_SPEC_SHA256,
                {"source_sha256": source_stage["details"]["sha256"]},
            )

            def fixed_action(partial: Path) -> dict[str, Any]:
                helper = partial / "fix_model.py"
                atomic_write_text(helper, FIXED_MODEL_HELPER)
                contract = partial / "onnx_contract.json"
                atomic_write_json(contract, spec["component"]["onnx_contract"])
                output = partial / "duration_predictor.fixed32.onnx"
                metadata = partial / "fixed_model_metadata.json"
                log = partial / "fixed_model.log"
                command = [
                    str(venv / "bin/python"),
                    str(helper),
                    str(source_model),
                    str(output),
                    str(metadata),
                    str(contract),
                ]
                run_logged(command, partial, env, log)
                record = read_json(metadata)
                require(
                    record["source"]["sha256"] == EXPECTED_MODEL_SHA256,
                    "fixed-model helper source drift",
                )
                return {
                    "allowed_mutation": record["allowed_mutation"],
                    "source_sha256": record["source"]["sha256"],
                    "derived_sha256": record["derived"]["sha256"],
                    "derived_bytes": record["derived"]["bytes"],
                    "node_payload_sha256": record["node_payload_sha256"],
                    "initializer_payload_sha256": record[
                        "initializer_payload_sha256"
                    ],
                }

            fixed_stage = run_stage(
                "fixed_model",
                fixed_fp,
                workspace,
                state_path,
                state,
                fixed_action,
            )
            fixed_model = workspace / "stages/fixed_model/duration_predictor.fixed32.onnx"

            conversion_fp = stage_fingerprint(
                "qnn_conversion",
                EXPECTED_SPEC_SHA256,
                {"fixed_model_sha256": fixed_stage["details"]["derived_sha256"]},
            )

            def conversion_action(partial: Path) -> dict[str, Any]:
                cpp = partial / "supertonic_duration_qairt242.cpp"
                command = converter_command(sdk_root, fixed_model, cpp, spec)
                run_logged(command, partial, env, partial / "qnn_converter.log")
                require(cpp.is_file(), "QNN converter did not create C++ output")
                weights = cpp.with_suffix(".bin")
                require(weights.is_file(), "QNN converter did not create weight binary")
                graph_name = parse_graph_name(cpp)
                return {
                    "command": command,
                    "qnn_graph_name": graph_name,
                    "cpp_sha256": sha256_file(cpp),
                    "weights_sha256": sha256_file(weights),
                }

            conversion_stage = run_stage(
                "qnn_conversion",
                conversion_fp,
                workspace,
                state_path,
                state,
                conversion_action,
            )
            conversion_dir = workspace / "stages/qnn_conversion"
            cpp = conversion_dir / "supertonic_duration_qairt242.cpp"
            weights = conversion_dir / "supertonic_duration_qairt242.bin"

            model_lib_fp = stage_fingerprint(
                "model_library",
                EXPECTED_SPEC_SHA256,
                {
                    "cpp_sha256": conversion_stage["details"]["cpp_sha256"],
                    "weights_sha256": conversion_stage["details"]["weights_sha256"],
                },
            )

            def model_lib_action(partial: Path) -> dict[str, Any]:
                output = partial / "model_libs"
                command = model_lib_command(sdk_root, cpp, weights, output, spec)
                run_logged(command, partial, env, partial / "model_lib_generator.log")
                library = find_unique(
                    output, "libsupertonic_duration_qairt242.so", "QNN model library"
                )
                return {
                    "command": command,
                    "library_relative": library.relative_to(partial).as_posix(),
                    "library_bytes": library.stat().st_size,
                    "library_sha256": sha256_file(library),
                }

            model_lib_stage = run_stage(
                "model_library",
                model_lib_fp,
                workspace,
                state_path,
                state,
                model_lib_action,
            )
            model_library = (
                workspace
                / "stages/model_library"
                / model_lib_stage["details"]["library_relative"]
            )

            tools = tool_inventory(sdk_root, spec)
            context_fp = stage_fingerprint(
                "context_binary",
                EXPECTED_SPEC_SHA256,
                {
                    "model_library_sha256": model_lib_stage["details"][
                        "library_sha256"
                    ],
                    "qnn_graph_name": conversion_stage["details"]["qnn_graph_name"],
                    "tool_inventory_sha256": canonical_sha256(tools),
                    "target": spec["target"],
                },
            )

            def context_action(partial: Path) -> dict[str, Any]:
                graph_name = conversion_stage["details"]["qnn_graph_name"]
                htp, wrapper = backend_configs(sdk_root, graph_name, spec)
                htp_path = partial / "htp_backend_config.json"
                wrapper_path = partial / "backend_extensions.json"
                atomic_write_json(htp_path, htp)
                wrapper["backend_extensions"]["config_file_path"] = str(htp_path)
                atomic_write_json(wrapper_path, wrapper)
                output_dir = partial / "context"
                output_dir.mkdir()
                command = context_command(
                    sdk_root, model_library, output_dir, wrapper_path, spec
                )
                run_logged(command, partial, env, partial / "context_generator.log")
                expected = (
                    output_dir
                    / f"{spec['output_contract']['context_basename']}.bin"
                )
                require(expected.is_file(), "context generator did not create expected binary")
                require(
                    expected.stat().st_size
                    >= spec["output_contract"]["minimum_context_bytes"],
                    "context binary is unexpectedly small",
                )
                metadata = partial / "context_metadata.json"
                utility_command = [
                    str(
                        sdk_root
                        / spec["tool_contract"]["context_binary_utility"]
                    ),
                    "--context_binary",
                    str(expected),
                    "--json_file",
                    str(metadata),
                ]
                run_logged(
                    utility_command,
                    partial,
                    env,
                    partial / "context_binary_utility.log",
                )
                require(metadata.is_file(), "context metadata utility produced no JSON")
                read_json(metadata)
                return {
                    "command": command,
                    "utility_command": utility_command,
                    "qnn_graph_name": graph_name,
                    "target": copy.deepcopy(spec["target"]),
                    "context_relative": expected.relative_to(partial).as_posix(),
                    "context_bytes": expected.stat().st_size,
                    "context_sha256": sha256_file(expected),
                    "context_metadata_sha256": sha256_file(metadata),
                    "tool_inventory": tools,
                }

            context_stage = run_stage(
                "context_binary",
                context_fp,
                workspace,
                state_path,
                state,
                context_action,
            )

            deliverable_fp = stage_fingerprint(
                "deliverable",
                EXPECTED_SPEC_SHA256,
                {
                    "context_sha256": context_stage["details"]["context_sha256"],
                    "context_metadata_sha256": context_stage["details"][
                        "context_metadata_sha256"
                    ],
                    "controller_sha256": state["controller_sha256"],
                },
            )

            def deliverable_action(partial: Path) -> dict[str, Any]:
                payload = partial / "payload"
                payload.mkdir()
                context_dir = workspace / "stages/context_binary"
                context_file = context_dir / context_stage["details"]["context_relative"]
                copies = {
                    context_file: payload / context_file.name,
                    context_dir / "context_metadata.json": payload / "context_metadata.json",
                    context_dir / "htp_backend_config.json": payload / "htp_backend_config.json",
                    context_dir / "backend_extensions.json": payload / "backend_extensions.json",
                    context_dir / "context_generator.log": payload / "context_generator.log",
                    context_dir / "context_binary_utility.log": payload
                    / "context_binary_utility.log",
                    workspace / "stages/fixed_model/fixed_model_metadata.json": payload
                    / "fixed_model_metadata.json",
                }
                for source, target in copies.items():
                    require(source.is_file(), f"missing deliverable source: {source.name}")
                    shutil.copy2(source, target)
                shutil.copy2(spec_path, payload / spec_path.name)
                result_manifest = {
                    "schema_version": 1,
                    "experiment_id": EXPECTED_EXPERIMENT_ID,
                    "status": spec["output_contract"]["required_manifest_status"],
                    "claim_boundary": spec["claim_boundary"],
                    "created_utc": utc_now(),
                    "spec_sha256": EXPECTED_SPEC_SHA256,
                    "controller_sha256": state["controller_sha256"],
                    "host": copy.deepcopy(state["host"]),
                    "python_environment": packages,
                    "archives": copy.deepcopy(state["archives"]),
                    "source_model": copy.deepcopy(source_stage["details"]),
                    "fixed_model": copy.deepcopy(fixed_stage["details"]),
                    "qnn_conversion": copy.deepcopy(conversion_stage["details"]),
                    "model_library": copy.deepcopy(model_lib_stage["details"]),
                    "context": {
                        key: copy.deepcopy(value)
                        for key, value in context_stage["details"].items()
                        if key != "tool_inventory"
                    },
                    "tool_inventory": tools,
                    "runtime_binaries_bundled": False,
                    "phone_execution_proven": False,
                    "npu_execution_proven": False,
                    "next_gate": (
                        "Load this exact context with the phone-proven QAIRT 2.42 "
                        "runtime in an isolated candidate app, disable CPU fallback, "
                        "and capture app-UID HTP/FastRPC/DMA evidence."
                    ),
                }
                atomic_write_json(payload / "result_manifest.json", result_manifest)
                shutil.copy2(payload / "result_manifest.json", partial / "result_manifest.json")
                bundle = partial / "qairt242_colab_result.tar.gz"
                deterministic_tar_gz(payload, bundle)
                bundle_hash = sha256_file(bundle)
                atomic_write_text(
                    partial / "qairt242_colab_result.tar.gz.sha256",
                    f"{bundle_hash}  {bundle.name}\n",
                )
                return {
                    "manifest_status": result_manifest["status"],
                    "bundle_bytes": bundle.stat().st_size,
                    "bundle_sha256": bundle_hash,
                    "phone_execution_proven": False,
                }

            deliverable_stage = run_stage(
                "deliverable",
                deliverable_fp,
                workspace,
                state_path,
                state,
                deliverable_action,
            )
            state["status"] = "complete"
            state["completed_utc"] = utc_now()
            state["result"] = copy.deepcopy(deliverable_stage["details"])
            write_state(state_path, state)
            return verify_completed(workspace, state, spec)
        except Exception as error:
            state["status"] = "failed"
            state["last_failure"] = {
                "time_utc": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            }
            write_state(state_path, state)
            atomic_write_json(
                workspace / "qairt242_colab_failure_report.json",
                {
                    "schema_version": 1,
                    "experiment_id": EXPECTED_EXPERIMENT_ID,
                    "status": "failed_closed",
                    "spec_sha256": EXPECTED_SPEC_SHA256,
                    "time_utc": utc_now(),
                    "error": state["last_failure"],
                    "state_file": state_path.name,
                    "instruction": (
                        "Preserve this workspace. Fix only the documented host/tool "
                        "precondition, then rerun the same command to resume. Do not "
                        "substitute QAIRT 2.45 or another model."
                    ),
                },
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "verify"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path("/content/qairt242_colab_scratch"),
        help="Colab-local scratch; durable outputs stay in --workspace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path, spec = load_spec(args.spec)
    if args.command == "plan":
        print(json.dumps(plan(spec), indent=2, sort_keys=True))
        return 0
    if args.workspace is None:
        raise CanaryError("--workspace is required for run and verify")
    workspace = safe_workspace(args.workspace)
    if args.command == "run":
        result = execute(spec_path, spec, workspace, args.scratch)
    else:
        state_path = workspace / "qairt242_colab_state.json"
        require(state_path.is_file(), f"missing state: {state_path}")
        state = read_json(state_path)
        result = verify_completed(workspace, state, spec)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CanaryError as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
