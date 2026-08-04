#!/usr/bin/env python3
"""Collect and independently verify a previously launched chain-probe run.

The collector is read-only with respect to the phone: it never installs,
uninstalls, clears, force-stops, or launches a package.  A matching failed-closed
app report is archived before the collector raises.  No "current" alias is
published before or after verification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/model_sweep_20260729"
DEFAULT_SPEC = REPORT_ROOT / "supertonic_chain_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = REPORT_ROOT / "supertonic_chain_ort_qnn_apk_local_verify.json"
DEVICE_VERIFIER_PATH = ROOT / "scripts/verify_supertonic_chain_ort_qnn_probe_device.py"
PACKAGE = "ai.liquid.lfmdemo.supertonicchainprobe"
GATE_ID = "supertonic3_full_chain_raw_onnx_ort_qnn_sm8850_probe_v1"
REPORT_RELATIVE = "files/supertonic_chain_probe/reports/latest.json"
PARTIAL_RELATIVE = REPORT_RELATIVE + ".partial"
PAYLOAD_RELATIVE = "files/supertonic_chain_probe/payload-v1"
LOCK_PATH = Path("/private/tmp/lmf-supertonic-chain-device-collector.lock")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


device_verifier = load_module("verify_supertonic_chain_device", DEVICE_VERIFIER_PATH)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale host partial exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def encoded_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


class Adb:
    def __init__(self, executable: str, serial: str) -> None:
        self.executable = executable
        self.serial = serial
        self.prefix = [executable, "-s", serial]

    def run(
        self,
        arguments: list[str],
        *,
        check: bool = True,
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess[bytes]:
        process = subprocess.run(
            [*self.prefix, *arguments],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
        if check and process.returncode != 0:
            output = (process.stdout + process.stderr).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ADB command failed ({process.returncode}): "
                f"{' '.join(arguments)}\n{output}"
            )
        return process

    def text(
        self,
        arguments: list[str],
        *,
        check: bool = True,
        timeout: float = 120.0,
    ) -> str:
        return (
            self.run(arguments, check=check, timeout=timeout)
            .stdout.decode("utf-8", errors="strict")
            .replace("\r", "")
            .strip()
        )

    def run_as(
        self, arguments: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        return self.run(["exec-out", "run-as", PACKAGE, *arguments], check=check)

    def hash_remote(self, path: str, *, run_as: bool = False) -> dict[str, Any]:
        arguments = [*self.prefix, "exec-out"]
        if run_as:
            arguments.extend(["run-as", PACKAGE])
        arguments.extend(["cat", path])
        process = subprocess.Popen(
            arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("cannot stream remote file")
        digest = hashlib.sha256()
        count = 0
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait(timeout=300) != 0:
            raise RuntimeError(f"cannot hash remote file {path}: {stderr}")
        return {"path": path, "bytes": count, "sha256": digest.hexdigest()}


def package_apk_path(adb: Adb, package: str) -> str:
    output = adb.text(["shell", "pm", "path", package])
    paths = [
        line.removeprefix("package:")
        for line in output.splitlines()
        if line.startswith("package:")
    ]
    if (
        len(paths) != 1
        or not paths[0].startswith("/data/app/")
        or not paths[0].endswith("/base.apk")
    ):
        raise RuntimeError(f"unexpected APK path inventory for {package}: {paths}")
    return paths[0]


def parse_package_uid(dump: str, listing: str) -> tuple[int, dict[str, int]]:
    dump_match = re.search(r"^\s*(?:userId|appId)=(\d+)\s*$", dump, re.MULTILINE)
    listing_matches = re.findall(
        rf"^package:{re.escape(PACKAGE)} uid:(\d+)$",
        listing,
        re.MULTILINE,
    )
    if dump_match is None or len(listing_matches) != 1:
        raise RuntimeError("package UID sources are incomplete")
    dump_uid = int(dump_match.group(1))
    listed_uid = int(listing_matches[0])
    if dump_uid != listed_uid or dump_uid < 10_000:
        raise RuntimeError("package UID sources disagree or are invalid")
    return dump_uid, {"dumpsys": dump_uid, "package_manager": listed_uid}


def package_details(adb: Adb, spec: dict[str, Any]) -> dict[str, Any]:
    dump = adb.text(["shell", "dumpsys", "package", PACKAGE])
    listing = adb.text(["shell", "cmd", "package", "list", "packages", "-U", PACKAGE])

    def single(pattern: str, label: str) -> str:
        match = re.search(pattern, dump, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"dumpsys package did not expose {label}")
        return match.group(1)

    uid, sources = parse_package_uid(dump, listing)
    apk_path = package_apk_path(adb, PACKAGE)
    installed = adb.hash_remote(apk_path)
    if (
        installed["bytes"] != spec["apk"]["bytes"]
        or installed["sha256"] != spec["apk"]["sha256"]
    ):
        raise RuntimeError("installed chain APK does not match the trust root")
    pid_text = adb.text(["shell", "pidof", PACKAGE])
    pids = [int(value) for value in pid_text.split() if value]
    if len(pids) != 1:
        raise RuntimeError(f"expected one chain-probe PID, got {pids}")
    return {
        "package": PACKAGE,
        "apk_path": apk_path,
        "bytes": installed["bytes"],
        "sha256": installed["sha256"],
        "version_code": int(single(r"^\s*versionCode=(\d+)", "versionCode")),
        "version_name": single(r"^\s*versionName=([^\s]+)", "versionName"),
        "uid": uid,
        "uid_sources": sources,
        "pid": pids[0],
    }


def accepted_packages(adb: Adb, spec: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for package, expected in spec["accepted_packages"].items():
        path = package_apk_path(adb, package)
        record = adb.hash_remote(path)
        if record["sha256"] != expected["sha256"]:
            raise RuntimeError(f"accepted APK changed: {package}")
        output[package] = record
    return output


def app_private_path_exists(adb: Adb, relative_path: str) -> bool:
    if not relative_path or relative_path.startswith("/") or "\x00" in relative_path:
        raise RuntimeError("invalid app-private relative path")
    result = adb.run_as(
        [
            "sh",
            "-c",
            'if [ -e "$1" ]; then printf PRESENT; else printf ABSENT; fi',
            "sh",
            relative_path,
        ]
    )
    marker = result.stdout.decode("utf-8", errors="strict")
    if marker == "PRESENT":
        return True
    if marker == "ABSENT":
        return False
    raise RuntimeError(f"unexpected app-private existence marker: {marker!r}")


def poll_report(
    adb: Adb, run_id: str, timeout_seconds: float
) -> tuple[bytes, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = adb.run_as(["cat", REPORT_RELATIVE], check=False)
        if result.returncode == 0 and result.stdout:
            try:
                report = device_verifier.strict_json_bytes(
                    result.stdout, "app-private latest.json"
                )
            except RuntimeError:
                time.sleep(0.5)
                continue
            if report.get("run_id") != run_id:
                time.sleep(0.5)
                continue
            if report.get("report_complete") is not True:
                raise RuntimeError("matching app report was published incomplete")
            if report.get("completion_marker") != device_verifier.COMPLETION_MARKER:
                raise RuntimeError("matching app report completion marker mismatch")
            return result.stdout, report
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for complete run_id={run_id}")


def archive_report(
    output_dir: Path,
    run_id: str,
    raw_report: bytes,
    report: dict[str, Any],
) -> Path:
    path = output_dir / f"supertonic_chain_device_report_{run_id}.json"
    atomic_write(path, raw_report)
    if report.get("status") == "failed_closed":
        raise RuntimeError(
            "matching chain probe failed closed after report archival: "
            + str(report.get("error"))
        )
    return path


def archive_process_artifacts(
    output_dir: Path, run_id: str, values: dict[str, bytes]
) -> dict[str, dict[str, Any]]:
    if set(values) != {"status", "maps", "fd", "smaps"}:
        raise RuntimeError("raw process artifact inventory mismatch")
    output: dict[str, dict[str, Any]] = {}
    for label in ("status", "maps", "fd", "smaps"):
        payload = values[label]
        if not payload:
            raise RuntimeError(f"raw process artifact is empty: {label}")
        name = f"supertonic_chain_{run_id}_proc_{label}.txt"
        atomic_write(output_dir / name, payload)
        output[label] = {
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return output


def collect_process(
    adb: Adb,
    installed: dict[str, Any],
    report_owner_uid: int,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    pid = installed["pid"]
    raw = {
        "status": adb.run_as(["cat", f"/proc/{pid}/status"]).stdout,
        "maps": adb.run_as(["cat", f"/proc/{pid}/maps"]).stdout,
        "fd": adb.run_as(["ls", "-l", f"/proc/{pid}/fd"]).stdout,
        "smaps": adb.run_as(["cat", f"/proc/{pid}/smaps"]).stdout,
    }
    decoded = {
        key: value.decode("utf-8", errors="strict") for key, value in raw.items()
    }
    app_root = installed["apk_path"].removesuffix("/base.apk")
    summary = device_verifier.parse_process_snapshot(
        decoded["status"],
        decoded["maps"],
        decoded["fd"],
        decoded["smaps"],
        app_root,
    )
    return {
        "pid": pid,
        "report_owner_uid": report_owner_uid,
        **summary,
        "raw_artifacts": archive_process_artifacts(output_dir, run_id, raw),
    }


def collect_resident_payload(
    adb: Adb, spec: dict[str, Any], model_root: str
) -> dict[str, Any]:
    prefix = "/data/user/0/" + PACKAGE + "/"
    if model_root != device_verifier.MODEL_ROOT or not model_root.startswith(prefix):
        raise RuntimeError("refusing unexpected app-private payload root")
    relative = model_root.removeprefix(prefix)
    manifest = adb.hash_remote(relative + "/payload_manifest.json", run_as=True)
    manifest["path"] = model_root + "/payload_manifest.json"
    models: dict[str, Any] = {}
    for component, expected in spec["models"].items():
        remote = relative + "/models/" + expected["filename"]
        record = adb.hash_remote(remote, run_as=True)
        record["path"] = model_root + "/models/" + expected["filename"]
        if (
            record["bytes"] != expected["bytes"]
            or record["sha256"] != expected["sha256"]
        ):
            raise RuntimeError(f"resident model identity mismatch: {component}")
        models[component] = record
    kib = int(
        adb.run_as(["toybox", "du", "-sk", relative])
        .stdout.decode("utf-8", errors="strict")
        .split()[0]
    )
    return {
        "root": model_root,
        "kib": kib,
        "payload_manifest": manifest,
        "models": models,
        "all_four_models_resident": True,
    }


def resolve_soc(adb: Adb) -> dict[str, str]:
    candidates = [
        ("ro.soc.model", adb.text(["shell", "getprop", "ro.soc.model"])),
        (
            "ro.vendor.qti.soc_model",
            adb.text(["shell", "getprop", "ro.vendor.qti.soc_model"]),
        ),
        (
            "/sys/devices/soc0/machine",
            adb.text(["shell", "cat", "/sys/devices/soc0/machine"], check=False),
        ),
        (
            "ro.board.platform",
            adb.text(["shell", "getprop", "ro.board.platform"]),
        ),
    ]
    for source, raw in candidates:
        value = raw.strip()
        if not value:
            continue
        match = re.search(r"\bSM\d{4}\b", value.upper())
        return {
            "value": match.group(0) if match is not None else value,
            "source": source,
            "raw": value,
        }
    raise RuntimeError("device exposed no non-empty SoC identity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--adb", default="adb")
    arguments = parser.parse_args()
    if RUN_ID_RE.fullmatch(arguments.run_id) is None:
        raise RuntimeError("invalid run_id")
    if not 1.0 <= arguments.timeout_seconds <= 600.0:
        raise RuntimeError("timeout must be between 1 and 600 seconds")
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = arguments.spec.expanduser().resolve()
    local_path = arguments.local_verify.expanduser().resolve()
    spec = device_verifier.load_object(spec_path)
    local = device_verifier.load_object(local_path)
    if local.get("status") != "pass":
        raise RuntimeError("local whole-APK verifier did not pass")

    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another chain collector is running") from error
        adb = Adb(arguments.adb, arguments.serial)
        if adb.text(["get-state"]) != "device":
            raise RuntimeError("ADB target is not ready")
        installed = package_details(adb, spec)
        raw_report, report = poll_report(
            adb, arguments.run_id, arguments.timeout_seconds
        )
        report_path = archive_report(output_dir, arguments.run_id, raw_report, report)
        if app_private_path_exists(adb, PARTIAL_RELATIVE):
            raise RuntimeError("app-private partial report still exists")
        owner_uid = int(
            adb.run_as(["toybox", "stat", "-c", "%u", REPORT_RELATIVE])
            .stdout.decode("utf-8", errors="strict")
            .strip()
        )
        process = collect_process(
            adb,
            installed,
            owner_uid,
            output_dir,
            arguments.run_id,
        )
        resident = collect_resident_payload(adb, spec, report.get("model_storage", ""))
        accepted = accepted_packages(adb, spec)
        soc = resolve_soc(adb)
        collected_ms = int(time.time() * 1000)
        evidence = {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "package": PACKAGE,
            "run_id": arguments.run_id,
            "run": {
                "requested_run_id": arguments.run_id,
                "trigger": spec["device_output_contract"]["run_trigger"],
            },
            "device": {
                "serial": arguments.serial,
                "model": adb.text(["shell", "getprop", "ro.product.model"]),
                "soc": soc["value"],
                "soc_source": soc["source"],
                "soc_raw": soc["raw"],
                "abi": adb.text(["shell", "getprop", "ro.product.cpu.abi"]),
                "airplane_mode_on": int(
                    adb.text(
                        [
                            "shell",
                            "settings",
                            "get",
                            "global",
                            "airplane_mode_on",
                        ]
                    )
                ),
            },
            "installed_package": installed,
            "accepted_packages": accepted,
            "process": process,
            "resident_payload": resident,
            "authoritative_spec": {
                "path": str(spec_path),
                "bytes": spec_path.stat().st_size,
                "sha256": device_verifier.sha256(spec_path),
            },
            "local_verifier_report": {
                "path": str(local_path),
                "bytes": local_path.stat().st_size,
                "sha256": device_verifier.sha256(local_path),
            },
            "device_report": {
                "path": str(report_path),
                "device_path": spec["device_output_contract"]["report_path"],
                "partial_device_path": spec["device_output_contract"][
                    "partial_report_path"
                ],
                "partial_exists_after_capture": False,
                "bytes": report_path.stat().st_size,
                "sha256": device_verifier.sha256(report_path),
            },
            "collection": {
                "collected_at_unix_ms": collected_ms,
                "collected_at_utc": datetime.fromtimestamp(
                    collected_ms / 1000, tz=timezone.utc
                )
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "matching_failed_closed_report_archived_before_error": True,
                "partial_report_absent": True,
            },
            "publication": {
                "current_alias_published": False,
                "verification_published": False,
            },
            "status": "collected_unverified",
            "claim_boundary": (
                "Evidence collection only; independent verification decides promotion."
            ),
        }
        evidence_path = output_dir / (
            f"supertonic_chain_device_evidence_{arguments.run_id}.json"
        )
        atomic_write(evidence_path, encoded_json(evidence))
        verified = device_verifier.verify_documents(
            spec,
            local,
            report,
            evidence,
            expected_run_id=arguments.run_id,
            spec_path=spec_path,
            local_path=local_path,
            report_path=report_path,
            evidence_path=evidence_path,
        )
        verify_path = output_dir / (
            f"supertonic_chain_device_verify_{arguments.run_id}.json"
        )
        atomic_write(verify_path, encoded_json(verified))
        print(encoded_json(verified).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
