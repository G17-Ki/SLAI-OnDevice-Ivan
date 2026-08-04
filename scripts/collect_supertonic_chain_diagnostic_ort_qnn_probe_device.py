#!/usr/bin/env python3
"""Launch, collect, and verify one installed Supertonic diagnostic run.

The collector never installs, uninstalls, clears, or updates any package. It
deletes only this diagnostic package's stale latest/partial report, confirms
both are absent, launches a run-id-bound intent, archives the matching atomic
report, and independently verifies either a pass or localized failure.
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
DEFAULT_SPEC = REPORT_ROOT / "supertonic_chain_diagnostic_ort_qnn_probe_spec.json"
DEFAULT_LOCAL_VERIFY = (
    REPORT_ROOT / "supertonic_chain_diagnostic_ort_qnn_apk_local_verify.json"
)
DEVICE_VERIFIER_PATH = (
    ROOT / "scripts/verify_supertonic_chain_diagnostic_ort_qnn_probe_device.py"
)
PACKAGE = "ai.liquid.lfmdemo.supertonicchaindiagprobe"
COMPONENT = PACKAGE + "/.ProbeActivity"
GATE_ID = (
    "supertonic3_full_chain_stage_diagnostic_raw_onnx_ort_qnn_sm8850_probe_v2"
)
REPORT_RELATIVE = "files/supertonic_chain_diagnostic_probe/reports/latest.json"
PARTIAL_RELATIVE = REPORT_RELATIVE + ".partial"
LOCK_PATH = Path("/private/tmp/lmf-supertonic-chain-diagnostic-device-collector.lock")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


device_verifier = load_module(
    "verify_supertonic_chain_diagnostic_device", DEVICE_VERIFIER_PATH
)


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


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": device_verifier.sha256(path),
    }


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
            output = (process.stdout + process.stderr).decode(
                "utf-8", errors="replace"
            )
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


def package_details(adb: Adb, spec: dict[str, Any]) -> dict[str, Any]:
    dump = adb.text(["shell", "dumpsys", "package", PACKAGE])

    def single(pattern: str, label: str) -> str:
        match = re.search(pattern, dump, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"dumpsys package did not expose {label}")
        return match.group(1)

    apk_path = package_apk_path(adb, PACKAGE)
    installed = adb.hash_remote(apk_path)
    if (
        installed["bytes"] != spec["apk"]["bytes"]
        or installed["sha256"] != spec["apk"]["sha256"]
    ):
        raise RuntimeError("installed diagnostic APK does not match trust root")
    return {
        "package": PACKAGE,
        "apk_path": apk_path,
        "bytes": installed["bytes"],
        "sha256": installed["sha256"],
        "version_code": int(single(r"^\s*versionCode=(\d+)", "versionCode")),
        "version_name": single(r"^\s*versionName=([^\s]+)", "versionName"),
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


def prepare_prelaunch(adb: Adb) -> dict[str, Any]:
    adb.text(["shell", "am", "force-stop", PACKAGE])
    adb.run_as(["rm", "-f", REPORT_RELATIVE, PARTIAL_RELATIVE])
    latest_absent = not app_private_path_exists(adb, REPORT_RELATIVE)
    partial_absent = not app_private_path_exists(adb, PARTIAL_RELATIVE)
    if not latest_absent or not partial_absent:
        raise RuntimeError("stale diagnostic report survived prelaunch cleanup")
    return {
        "package_force_stopped": True,
        "latest_report_deleted": True,
        "partial_report_deleted": True,
        "latest_report_absent_before_launch": latest_absent,
        "partial_report_absent_before_launch": partial_absent,
        "cleanup_completed_at_unix_ms": int(time.time() * 1000),
    }


def launch(adb: Adb, run_id: str) -> dict[str, Any]:
    requested_ms = int(time.time() * 1000)
    result = adb.run(
        [
            "shell",
            "am",
            "start",
            "-W",
            "-n",
            COMPONENT,
            "--ez",
            "run_probe",
            "true",
            "--es",
            "run_id",
            run_id,
        ],
        timeout=30.0,
    )
    output = result.stdout + result.stderr
    decoded = output.decode("utf-8", errors="replace")
    if "Status: ok" not in decoded and "Starting: Intent" not in decoded:
        raise RuntimeError(f"activity launch did not report success: {decoded}")
    return {
        "component": COMPONENT,
        "run_id": run_id,
        "run_probe_extra": True,
        "requested_at_unix_ms": requested_ms,
        "am_start_output_sha256": hashlib.sha256(output).hexdigest(),
    }


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
                time.sleep(0.25)
                continue
            if report.get("run_id") != run_id:
                raise RuntimeError("atomic report run_id differs from launched run")
            if report.get("report_complete") is not True:
                raise RuntimeError("matching app report was published incomplete")
            if report.get("completion_marker") != device_verifier.COMPLETION_MARKER:
                raise RuntimeError("matching app report completion marker mismatch")
            return result.stdout, report
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for complete run_id={run_id}")


def resolve_soc(adb: Adb) -> str:
    candidates = [
        adb.text(["shell", "getprop", "ro.soc.model"]),
        adb.text(["shell", "getprop", "ro.vendor.qti.soc_model"]),
        adb.text(["shell", "cat", "/sys/devices/soc0/machine"], check=False),
        adb.text(["shell", "getprop", "ro.board.platform"]),
    ]
    for raw in candidates:
        match = re.search(r"\bSM\d{4}\b", raw.upper())
        if match is not None:
            return match.group(0)
    raise RuntimeError("device did not expose an SM-series SoC identity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--local-verify", type=Path, default=DEFAULT_LOCAL_VERIFY)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
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
    _, _ = device_verifier.host_verifier.validate_spec(spec, spec_path)
    device_verifier.verify_local_report(local, spec, spec_path)

    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another diagnostic collector is running") from error
        adb = Adb(arguments.adb, arguments.serial)
        if adb.text(["get-state"]) != "device":
            raise RuntimeError("ADB target is not ready")
        installed = package_details(adb, spec)
        accepted_before = accepted_packages(adb, spec)
        prelaunch = prepare_prelaunch(adb)
        launch_evidence = launch(adb, arguments.run_id)
        raw_report, report = poll_report(
            adb, arguments.run_id, arguments.timeout_seconds
        )
        report_path = output_dir / (
            f"supertonic_chain_diagnostic_device_report_{arguments.run_id}.json"
        )
        atomic_write(report_path, raw_report)
        partial_absent = not app_private_path_exists(adb, PARTIAL_RELATIVE)
        if not partial_absent:
            raise RuntimeError("app-private partial report remains after capture")
        accepted_after = accepted_packages(adb, spec)
        collected_ms = int(time.time() * 1000)
        evidence = {
            "schema_version": 2,
            "gate_id": GATE_ID,
            "package": PACKAGE,
            "run_id": arguments.run_id,
            "prelaunch": prelaunch,
            "launch": launch_evidence,
            "device": {
                "serial": arguments.serial,
                "model": adb.text(["shell", "getprop", "ro.product.model"]),
                "soc": resolve_soc(adb),
                "abi": adb.text(["shell", "getprop", "ro.product.cpu.abi"]),
                "airplane_mode_on": int(
                    adb.text(
                        ["shell", "settings", "get", "global", "airplane_mode_on"]
                    )
                ),
            },
            "installed_package": installed,
            "accepted_packages_before": accepted_before,
            "accepted_packages_after": accepted_after,
            "authoritative_spec": identity(spec_path),
            "local_verifier_report": identity(local_path),
            "device_report": identity(report_path),
            "collection": {
                "collected_at_unix_ms": collected_ms,
                "collected_at_utc": datetime.fromtimestamp(
                    collected_ms / 1000, tz=timezone.utc
                )
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "matching_run_id": report.get("run_id") == arguments.run_id,
                "matching_completion_marker": (
                    report.get("completion_marker")
                    == device_verifier.COMPLETION_MARKER
                ),
                "partial_report_absent_after_capture": partial_absent,
                "diagnostic_status": report.get("status"),
                "failure_stage": report.get("failure_stage"),
                "failure_phase": report.get("failure_phase"),
                "failure_vector_step": report.get("failure_vector_step"),
            },
            "publication": {
                "current_alias_published": False,
                "verification_published": False,
            },
            "status": "collected_unverified",
            "claim_boundary": (
                "Stage-diagnostic collection only; independent verification controls "
                "the diagnostic result and no production claim is published."
            ),
        }
        evidence_path = output_dir / (
            f"supertonic_chain_diagnostic_device_evidence_{arguments.run_id}.json"
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
            f"supertonic_chain_diagnostic_device_verify_{arguments.run_id}.json"
        )
        atomic_write(verify_path, encoded_json(verified))
        print(encoded_json(verified).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
