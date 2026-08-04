#!/usr/bin/env python3
"""Install, run, collect, and verify the isolated frontend probe on one device.

This script is intentionally fail closed. It only installs the exact APK pinned
by the independent frontend spec, never uninstalls or clears app data, ignores
stale reports by run_id, and publishes host evidence atomically only after the
matching app-private report is complete.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/android_phone/streaming_response/whisper_v81"
DEFAULT_APK = (
    ROOT / "android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk"
)
DEFAULT_FRONTEND_SPEC = REPORT_ROOT / "nubia_whisper_android_frontend_probe_spec.json"
DEFAULT_BASE_SPEC = REPORT_ROOT / "nubia_whisper_ort_qnn_probe_spec.json"
LOCAL_VERIFIER_PATH = ROOT / "scripts/verify_whisper_android_frontend_apk.py"
DEVICE_VERIFIER_PATH = ROOT / "scripts/verify_whisper_android_frontend_device.py"
PACKAGE = "ai.liquid.lfmdemo.whisperprobe"
ACTIVITY = f"{PACKAGE}/.ProbeActivity"
REPORT_RELATIVE = "files/whisper_probe/reports/latest.json"
PARTIAL_RELATIVE = REPORT_RELATIVE + ".partial"
LOCK_PATH = Path("/private/tmp/lmf-whisper-frontend-device-probe.lock")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


local_verifier = load_module("verify_frontend_apk", LOCAL_VERIFIER_PATH)
device_verifier = load_module("verify_frontend_device", DEVICE_VERIFIER_PATH)


def select_frontend_profile(
    frontend_spec: dict[str, Any],
    base_spec: dict[str, Any],
    requested_profile: str,
) -> str:
    declared = frontend_spec.get("frontend_profile")
    if declared is None:
        inferred = "v1"
    elif declared == "v2":
        inferred = "v2"
    else:
        raise RuntimeError(f"unsupported declared frontend profile: {declared!r}")
    if requested_profile != "auto" and requested_profile != inferred:
        raise RuntimeError(
            "requested frontend profile does not match the trust root; "
            f"requested={requested_profile}, inferred={inferred}"
        )
    local_verifier.validate_frontend_contract(
        frontend_spec,
        base_spec,
        frontend_profile=inferred,
    )
    return inferred


def strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    def invalid_constant(item: str) -> Any:
        raise RuntimeError(f"invalid JSON number in {label}: {item}")

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON in {label}") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"expected JSON object in {label}")
    return decoded


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale host partial output exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
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
        self.prefix = [executable, "-s", serial]

    def run(
        self,
        arguments: list[str],
        *,
        check: bool = True,
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess[bytes]:
        process = subprocess.run(
            [*self.prefix, *arguments],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if check and process.returncode != 0:
            output = (process.stdout + process.stderr).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ADB command failed ({process.returncode}): {' '.join(arguments)}\n{output}"
            )
        return process

    def text(
        self, arguments: list[str], *, check: bool = True, timeout: float = 60.0
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
        while True:
            block = process.stdout.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            count += len(block)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(f"cannot hash remote file {path}: {stderr}")
        return {"path": path, "bytes": count, "sha256": digest.hexdigest()}


def package_apk_path(adb: Adb, package: str) -> str:
    output = adb.text(["shell", "pm", "path", package])
    paths = [line.removeprefix("package:") for line in output.splitlines() if line]
    if len(paths) != 1 or not paths[0].endswith("/base.apk"):
        raise RuntimeError(f"unexpected APK path inventory for {package}: {paths}")
    return paths[0]


def hash_installed_package(adb: Adb, package: str) -> dict[str, Any]:
    return adb.hash_remote(package_apk_path(adb, package))


def accepted_hashes(adb: Adb, spec: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for package, expected in spec["accepted_packages"].items():
        record = hash_installed_package(adb, package)
        if record["sha256"] != expected["sha256"]:
            raise RuntimeError(f"accepted APK hash mismatch: {package}")
        result[package] = {"sha256": record["sha256"], "bytes": record["bytes"]}
    return result


def parse_package_uid(dump: str, uid_listing: str) -> int:
    dump_match = re.search(
        r"^\s*(?:userId|appId)=(\d+)$",
        dump,
        re.MULTILINE,
    )
    if dump_match is None:
        raise RuntimeError("dumpsys did not expose userId/appId")
    listing_matches = re.findall(
        rf"^package:{re.escape(PACKAGE)} uid:(\d+)$",
        uid_listing,
        re.MULTILINE,
    )
    if len(listing_matches) != 1:
        raise RuntimeError(
            f"package manager did not expose one exact package UID: {listing_matches}"
        )
    dump_uid = int(dump_match.group(1))
    listed_uid = int(listing_matches[0])
    if dump_uid != listed_uid:
        raise RuntimeError(
            f"package UID sources disagree: dumpsys={dump_uid}, list={listed_uid}"
        )
    return dump_uid


def package_details(adb: Adb, apk_record: dict[str, Any]) -> dict[str, Any]:
    dump = adb.text(["shell", "dumpsys", "package", PACKAGE])
    uid_listing = adb.text(
        ["shell", "cmd", "package", "list", "packages", "-U", PACKAGE]
    )

    def single(pattern: str, label: str) -> str:
        match = re.search(pattern, dump, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"dumpsys did not expose {label}")
        return match.group(1)

    apk_path = package_apk_path(adb, PACKAGE)
    installed = adb.hash_remote(apk_path)
    if (
        installed["bytes"] != apk_record["bytes"]
        or installed["sha256"] != apk_record["sha256"]
    ):
        raise RuntimeError("installed APK does not match the whole-APK trust root")
    return {
        "package": PACKAGE,
        "version_code": int(single(r"^\s*versionCode=(\d+)", "versionCode")),
        "version_name": single(r"^\s*versionName=([^\s]+)", "versionName"),
        "uid": parse_package_uid(dump, uid_listing),
        "apk_path": apk_path,
        "bytes": installed["bytes"],
        "sha256": installed["sha256"],
    }


def app_private_path_exists(adb: Adb, relative_path: str) -> bool:
    if not relative_path or relative_path.startswith("/") or "\x00" in relative_path:
        raise RuntimeError("invalid app-private relative path")
    process = adb.run_as(
        [
            "sh",
            "-c",
            'if [ -e "$1" ]; then printf PRESENT; else printf ABSENT; fi',
            "sh",
            relative_path,
        ]
    )
    marker = process.stdout.decode("utf-8", errors="strict")
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
        process = adb.run_as(["cat", REPORT_RELATIVE], check=False)
        if process.returncode == 0 and process.stdout:
            try:
                report = strict_json_bytes(process.stdout, "app-private latest.json")
            except RuntimeError:
                time.sleep(0.5)
                continue
            if report.get("run_id") != run_id:
                time.sleep(0.5)
                continue
            if report.get("report_complete") is not True:
                raise RuntimeError("matching app report was published incomplete")
            if report.get("completion_marker") != "WHISPER_PROBE_REPORT_COMPLETE_V1":
                raise RuntimeError(
                    "matching app report has the wrong completion marker"
                )
            return process.stdout, report
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for completed run_id={run_id}")


def archive_report(
    output_dir: Path,
    run_id: str,
    raw_report: bytes,
    report: dict[str, Any],
) -> Path:
    report_path = output_dir / f"device_report_{run_id}.json"
    atomic_write(report_path, raw_report)
    if report.get("status") == "failed_closed":
        raise RuntimeError(
            "matching frontend probe failed closed after report archival: "
            + str(report.get("error"))
        )
    return report_path


def collect_process(
    adb: Adb,
    pid: int,
    report_uid: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    status = adb.run_as(["cat", f"/proc/{pid}/status"]).stdout.decode()
    maps = adb.run_as(["cat", f"/proc/{pid}/maps"]).stdout.decode()
    fds = adb.run_as(["ls", "-l", f"/proc/{pid}/fd"]).stdout.decode()
    parsed = device_verifier.parse_process_snapshot(status, maps, fds)
    process = {
        "pid": pid,
        "report_owner_uid": report_uid,
        **parsed,
    }
    return process, {
        "status": status.encode("utf-8"),
        "maps": maps.encode("utf-8"),
        "fd": fds.encode("utf-8"),
    }


def archive_process_artifacts(
    output_dir: Path,
    run_id: str,
    raw_artifacts: dict[str, bytes],
) -> dict[str, dict[str, Any]]:
    suffixes = {"status": "status.txt", "maps": "maps.txt", "fd": "fd.txt"}
    if set(raw_artifacts) != set(suffixes):
        raise RuntimeError("raw process artifact inventory mismatch")
    result: dict[str, dict[str, Any]] = {}
    for label, suffix in suffixes.items():
        name = f"proc_{run_id}_{suffix}"
        path = output_dir / name
        value = raw_artifacts[label]
        if not value:
            raise RuntimeError(f"raw process artifact is empty: {label}")
        atomic_write(path, value)
        result[label] = {
            "path": name,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    return result


def resolve_soc(adb: Adb) -> dict[str, str]:
    candidates = [
        ("ro.soc.model", adb.text(["shell", "getprop", "ro.soc.model"])),
        (
            "ro.vendor.qti.soc_model",
            adb.text(["shell", "getprop", "ro.vendor.qti.soc_model"]),
        ),
        (
            "/sys/devices/soc0/machine",
            adb.text(
                ["shell", "cat", "/sys/devices/soc0/machine"],
                check=False,
            ),
        ),
        ("ro.board.platform", adb.text(["shell", "getprop", "ro.board.platform"])),
    ]
    for source, raw in candidates:
        value = raw.strip()
        if not value:
            continue
        model = re.search(r"\bSM\d{4}\b", value.upper())
        return {
            "value": model.group(0) if model is not None else value,
            "source": source,
            "raw": value,
        }
    raise RuntimeError("device did not expose a non-empty SoC identity")


def collect_resident_payload(
    adb: Adb, base_spec: dict[str, Any], root: str
) -> dict[str, Any]:
    prefix = "/data/user/0/" + PACKAGE + "/"
    if not root.startswith(prefix):
        raise RuntimeError("refusing non-app-private resident payload path")
    relative_root = root.removeprefix(prefix)
    model_files: dict[str, int] = {}
    for record in base_spec["required_assets"]:
        model_prefix = "whisper_probe/models/"
        if record["path"].startswith(model_prefix):
            name = record["path"].removeprefix(model_prefix)
            remote = f"{relative_root}/models/{name}"
            size = int(
                adb.run_as(["toybox", "stat", "-c", "%s", remote])
                .stdout.decode()
                .strip()
            )
            model_files[name] = size
    kib = int(
        adb.run_as(["toybox", "du", "-sk", relative_root]).stdout.decode().split()[0]
    )
    return {"root": root, "kib": kib, "model_files": model_files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--apk", type=Path, default=DEFAULT_APK)
    parser.add_argument("--frontend-spec", type=Path, default=DEFAULT_FRONTEND_SPEC)
    parser.add_argument(
        "--frontend-profile",
        choices=("auto", "v1", "v2"),
        default="auto",
        help="validate an explicit profile or infer v1/v2 from the exact trust root",
    )
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--adb", default="adb")
    arguments = parser.parse_args()
    if not (10.0 <= arguments.timeout_seconds <= 600.0):
        raise RuntimeError("timeout must be between 10 and 600 seconds")
    run_id = arguments.run_id or (
        time.strftime("frontend-%Y%m%dT%H%M%SZ-", time.gmtime()) + uuid.uuid4().hex[:12]
    )
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise RuntimeError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another frontend device probe is already running"
            ) from error

        apk = arguments.apk.expanduser().resolve()
        frontend_spec_path = arguments.frontend_spec.expanduser().resolve()
        base_spec_path = arguments.base_spec.expanduser().resolve()
        frontend_spec = device_verifier.load_object(frontend_spec_path)
        base_spec = device_verifier.load_object(base_spec_path)
        frontend_profile = select_frontend_profile(
            frontend_spec,
            base_spec,
            arguments.frontend_profile,
        )
        local_report = local_verifier.verify_apk(
            apk,
            base_spec,
            frontend_spec,
            base_spec_path=base_spec_path,
            frontend_spec_path=frontend_spec_path,
            build_tools=local_verifier.DEFAULT_BUILD_TOOLS,
            java_home=local_verifier.DEFAULT_JAVA_HOME,
            frontend_profile=frontend_profile,
        )
        local_path = output_dir / "local_apk_verify.json"
        atomic_write(local_path, encoded_json(local_report))

        adb = Adb(arguments.adb, arguments.serial)
        require_state = adb.text(["get-state"])
        if require_state != "device":
            raise RuntimeError(f"ADB target is not ready: {require_state}")
        accepted_before = accepted_hashes(adb, frontend_spec)
        install = adb.run(["install", "-r", str(apk)], timeout=300.0)
        if b"Success" not in install.stdout + install.stderr:
            raise RuntimeError("ADB install did not report Success")
        installed = package_details(adb, frontend_spec["apk"])
        if installed["version_code"] != frontend_spec["apk_policy"]["version_code"]:
            raise RuntimeError("installed version code mismatch")
        if installed["version_name"] != frontend_spec["apk_policy"]["version_name"]:
            raise RuntimeError("installed version name mismatch")
        accepted_after_install = accepted_hashes(adb, frontend_spec)
        if accepted_after_install != accepted_before:
            raise RuntimeError("accepted packages changed during probe install")

        adb.run(["shell", "am", "force-stop", PACKAGE])
        adb.run(
            [
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                ACTIVITY,
                "--ez",
                "run_probe",
                "true",
                "--es",
                "run_id",
                run_id,
            ]
        )
        raw_report, report = poll_report(adb, run_id, arguments.timeout_seconds)
        report_path = archive_report(output_dir, run_id, raw_report, report)
        if app_private_path_exists(adb, PARTIAL_RELATIVE):
            raise RuntimeError(
                "app-private partial report still exists after completion"
            )

        pid_text = adb.text(["shell", "pidof", PACKAGE])
        pids = [int(value) for value in pid_text.split()]
        if len(pids) != 1:
            raise RuntimeError(f"expected one probe PID, got {pids}")
        installed["pid"] = pids[0]
        report_uid = int(
            adb.run_as(["toybox", "stat", "-c", "%u", REPORT_RELATIVE])
            .stdout.decode()
            .strip()
        )
        process, raw_process = collect_process(adb, pids[0], report_uid)
        process["raw_artifacts"] = archive_process_artifacts(
            output_dir, run_id, raw_process
        )
        resident = collect_resident_payload(adb, base_spec, report["model_storage"])
        native_path = report["runtime_storage"] + "/libwhisper_frontend.so"
        installed_frontend = adb.hash_remote(native_path)
        accepted_after = accepted_hashes(adb, frontend_spec)
        if accepted_after != accepted_before:
            raise RuntimeError("accepted packages changed during probe execution")

        soc = resolve_soc(adb)
        evidence = {
            "schema_version": 2,
            "gate_id": frontend_spec["gate_id"],
            "package": PACKAGE,
            "run_id": run_id,
            "run": {"requested_run_id": run_id, "trigger": "automation_intent"},
            "device": {
                "model": adb.text(["shell", "getprop", "ro.product.model"]),
                "soc": soc["value"],
                "soc_source": soc["source"],
                "soc_raw": soc["raw"],
                "abi": adb.text(["shell", "getprop", "ro.product.cpu.abi"]),
                "airplane_mode_on": int(
                    adb.text(["shell", "settings", "get", "global", "airplane_mode_on"])
                ),
            },
            "installed_package": installed,
            "accepted_packages_before": accepted_before,
            "accepted_packages": accepted_after,
            "process": process,
            "resident_payload": resident,
            "installed_frontend_library": installed_frontend,
            "local_verifier_report": {
                "path": str(local_path),
                "bytes": local_path.stat().st_size,
                "sha256": device_verifier.sha256(local_path),
            },
            "device_report": {
                "path": str(report_path),
                "device_path": frontend_spec["device_output_contract"]["report_path"],
                "partial_device_path": frontend_spec["device_output_contract"][
                    "report_path"
                ]
                + ".partial",
                "partial_exists_after_capture": False,
                "bytes": report_path.stat().st_size,
                "sha256": device_verifier.sha256(report_path),
            },
            "partial_report_absent": True,
        }
        evidence_path = output_dir / f"device_evidence_{run_id}.json"
        atomic_write(evidence_path, encoded_json(evidence))
        verified = device_verifier.verify_files(
            frontend_spec_path,
            base_spec_path,
            local_path,
            report_path,
            evidence_path,
            expected_run_id=run_id,
        )
        verified_path = output_dir / f"device_verify_{run_id}.json"
        atomic_write(verified_path, encoded_json(verified))
        print(encoded_json(verified).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
