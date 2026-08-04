#!/usr/bin/env python3
"""Collect immutable device evidence for an already-run Supertonic QNN probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/model_sweep_20260729"
DEFAULT_SPEC = REPORT_ROOT / "supertonic_duration_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = REPORT_ROOT / "supertonic_duration_ort_qnn_apk_local_verify.json"
PACKAGE = "ai.liquid.lfmdemo.supertonicprobe"
REPORT_RELATIVE = "files/supertonic_probe/reports/latest.json"
PAYLOAD_RELATIVE = "files/supertonic_probe/payload-v1"
REQUIRED_LIBRARIES = {
    "libQnnHtp.so",
    "libQnnHtpV81Stub.so",
    "libQnnSystem.so",
    "libcdsprpc.so",
    "libonnxruntime.so",
    "libonnxruntime4j_jni.so",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


class Adb:
    def __init__(self, serial: str) -> None:
        self.prefix = ["adb", "-s", serial]

    def run(self, arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            [*self.prefix, *arguments], capture_output=True, check=False, timeout=120
        )
        if check and result.returncode != 0:
            output = (result.stdout + result.stderr).decode(errors="replace")
            raise RuntimeError(f"ADB command failed: {' '.join(arguments)}\n{output}")
        return result

    def text(self, arguments: list[str], *, check: bool = True) -> str:
        return self.run(arguments, check=check).stdout.decode("utf-8").replace("\r", "").strip()

    def run_as(self, arguments: list[str]) -> bytes:
        return self.run(["exec-out", "run-as", PACKAGE, *arguments]).stdout

    def hash_remote(self, path: str, *, run_as: bool = False) -> dict[str, Any]:
        arguments = [*self.prefix, "exec-out"]
        if run_as:
            arguments.extend(["run-as", PACKAGE])
        arguments.extend(["cat", path])
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("cannot stream remote file")
        digest = hashlib.sha256()
        count = 0
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
        stderr = process.stderr.read().decode(errors="replace")
        if process.wait(timeout=120) != 0:
            raise RuntimeError(f"cannot hash remote file {path}: {stderr}")
        return {"path": path, "bytes": count, "sha256": digest.hexdigest()}


def package_path(adb: Adb, package: str) -> str:
    lines = adb.text(["shell", "pm", "path", package]).splitlines()
    paths = [line.removeprefix("package:") for line in lines if line.startswith("package:")]
    if len(paths) != 1 or not paths[0].endswith("/base.apk"):
        raise RuntimeError(f"unexpected installed APK path for {package}: {paths}")
    return paths[0]


def hash_package(adb: Adb, package: str) -> dict[str, Any]:
    return adb.hash_remote(package_path(adb, package))


def parse_status(status: str) -> tuple[list[int], int]:
    uid_match = re.search(r"^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", status, re.MULTILINE)
    rss_match = re.search(r"^VmRSS:\s+(\d+) kB", status, re.MULTILINE)
    if uid_match is None or rss_match is None:
        raise RuntimeError("process status is missing UID or VmRSS")
    return [int(value) for value in uid_match.groups()], int(rss_match.group(1))


def parse_rollup(rollup: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in ("Rss", "Pss", "Private_Dirty", "Swap"):
        match = re.search(rf"^{key}:\s+(\d+) kB", rollup, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"smaps_rollup is missing {key}")
        result[key.lower() + "_kib"] = int(match.group(1))
    return result


def process_snapshot(adb: Adb, pid: int, output_dir: Path, run_id: str) -> dict[str, Any]:
    raw = {
        "status": adb.run_as(["cat", f"/proc/{pid}/status"]),
        "maps": adb.run_as(["cat", f"/proc/{pid}/maps"]),
        "fd": adb.run_as(["ls", "-l", f"/proc/{pid}/fd"]),
        "smaps_rollup": adb.run_as(["cat", f"/proc/{pid}/smaps_rollup"]),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for label, payload in raw.items():
        path = output_dir / f"supertonic_{run_id}_proc_{label}.txt"
        atomic_write(path, payload)
        artifacts[label] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    status = raw["status"].decode()
    maps = raw["maps"].decode()
    fds = raw["fd"].decode()
    rollup = raw["smaps_rollup"].decode()
    uid_fields, status_rss_kib = parse_status(status)
    libraries = sorted(
        {
            line.rsplit("/", 1)[-1]
            for line in maps.splitlines()
            if "/" in line and line.rsplit("/", 1)[-1].startswith("lib")
        }
    )
    missing = REQUIRED_LIBRARIES - set(libraries)
    if missing:
        raise RuntimeError(f"required process libraries are missing: {sorted(missing)}")
    qcom_dmabuf_count = sum("/dmabuf:qcom," in line for line in fds.splitlines())
    if "/dev/fastrpc-cdsp" not in fds or qcom_dmabuf_count < 1:
        raise RuntimeError("FastRPC or QCOM DMA evidence is absent")
    return {
        "pid": pid,
        "uid_fields": uid_fields,
        "loaded_libraries": libraries,
        "fastrpc_cdsp_fd_open": True,
        "qcom_dmabuf_count": qcom_dmabuf_count,
        "status_rss_kib": status_rss_kib,
        **parse_rollup(rollup),
        "raw_artifacts": artifacts,
    }


def package_details(adb: Adb, spec: dict[str, Any]) -> dict[str, Any]:
    dump = adb.text(["shell", "dumpsys", "package", PACKAGE])
    def single(pattern: str, label: str) -> str:
        match = re.search(pattern, dump, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"dumpsys package is missing {label}")
        return match.group(1)
    record = hash_package(adb, PACKAGE)
    if record["bytes"] != spec["apk"]["bytes"] or record["sha256"] != spec["apk"]["sha256"]:
        raise RuntimeError("installed Supertonic APK does not match the trust root")
    return {
        "package": PACKAGE,
        "apk_path": package_path(adb, PACKAGE),
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "version_code": int(single(r"^\s*versionCode=(\d+)", "versionCode")),
        "version_name": single(r"^\s*versionName=([^\s]+)", "versionName"),
        "uid": int(single(r"^\s*(?:userId|appId)=(\d+)", "appId/userId")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    local_verify_path = args.local_verify.resolve()
    output_dir = args.output_dir.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    local_verify = json.loads(local_verify_path.read_text(encoding="utf-8"))
    if local_verify.get("status") != "pass":
        raise RuntimeError("local whole-APK verification did not pass")

    adb = Adb(args.serial)
    installed = package_details(adb, spec)
    pid_text = adb.text(["shell", "pidof", PACKAGE])
    if not re.fullmatch(r"\d+", pid_text):
        raise RuntimeError(f"unexpected probe PID: {pid_text!r}")
    pid = int(pid_text)

    report_bytes = adb.run_as(["cat", REPORT_RELATIVE])
    report = json.loads(report_bytes)
    if report.get("run_id") != args.run_id or report.get("report_complete") is not True:
        raise RuntimeError("app-private report does not match the requested complete run")
    report_path = output_dir / f"supertonic_duration_device_report_{args.run_id}.json"
    atomic_write(report_path, report_bytes)
    owner_uid = int(adb.run_as(["toybox", "stat", "-c", "%u", REPORT_RELATIVE]).decode().strip())

    process = process_snapshot(adb, pid, output_dir, args.run_id)
    process["report_owner_uid"] = owner_uid
    if process["uid_fields"] != [installed["uid"]] * 4 or owner_uid != installed["uid"]:
        raise RuntimeError("process/report ownership does not match installed app UID")

    model_remote = f"{PAYLOAD_RELATIVE}/models/duration_predictor.int8.onnx"
    model = adb.hash_remote(model_remote, run_as=True)
    if model["bytes"] != spec["model"]["bytes"] or model["sha256"] != spec["model"]["sha256"]:
        raise RuntimeError("resident Supertonic model identity mismatch")
    payload_kib = int(
        adb.run_as(["toybox", "du", "-sk", PAYLOAD_RELATIVE]).decode().split()[0]
    )

    accepted: dict[str, Any] = {}
    for package, expected in spec["accepted_packages"].items():
        record = hash_package(adb, package)
        if record["sha256"] != expected["sha256"]:
            raise RuntimeError(f"accepted APK changed: {package}")
        accepted[package] = record

    evidence = {
        "schema_version": 1,
        "gate_id": spec["gate_id"],
        "package": PACKAGE,
        "run_id": args.run_id,
        "device": {
            "serial": args.serial,
            "model": adb.text(["shell", "getprop", "ro.product.model"]),
            "soc": adb.text(["shell", "getprop", "ro.soc.model"]),
            "abi": adb.text(["shell", "getprop", "ro.product.cpu.abi"]),
            "airplane_mode_on": int(adb.text(["shell", "settings", "get", "global", "airplane_mode_on"])),
        },
        "installed_package": installed,
        "accepted_packages": accepted,
        "app_private_report": {
            "path": report_path.name,
            "bytes": len(report_bytes),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
        "authoritative_spec": {
            "path": spec_path.name,
            "bytes": spec_path.stat().st_size,
            "sha256": sha256(spec_path),
        },
        "local_verifier_report": {
            "path": local_verify_path.name,
            "bytes": local_verify_path.stat().st_size,
            "sha256": sha256(local_verify_path),
        },
        "process": process,
        "resident_payload": {
            "root": f"/data/user/0/{PACKAGE}/{PAYLOAD_RELATIVE}",
            "kib": payload_kib,
            "model": model,
        },
        "status": "collected",
        "claim_boundary": "Evidence collection only; promotion is decided by the independent verifier.",
    }
    evidence_path = output_dir / f"supertonic_duration_device_evidence_{args.run_id}.json"
    atomic_write(
        evidence_path,
        (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({
        "device_report": str(report_path),
        "device_evidence": str(evidence_path),
        "pid": pid,
        "rss_kib": process["rss_kib"],
        "qcom_dmabuf_count": process["qcom_dmabuf_count"],
        "status": "collected",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
