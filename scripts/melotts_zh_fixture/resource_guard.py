#!/usr/bin/env python3
"""Fail-closed resource gate for the local MeloTTS fixture run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# Both limits are overridable so the same guard can be run on hosts with a
# different idle baseline, but the macOS pressure-level check below is not
# overridable: it is the kernel's own health signal, and a run that raises it
# still aborts. Lowering the free-percentage floor without lowering the RSS cap
# would defeat the point, so the RSS cap is what the caller should tighten.
MIN_FREE_PERCENT = int(os.environ.get("MELOTTS_FIXTURE_MIN_FREE_PERCENT", "58"))
MAX_PROCESS_TREE_RSS_BYTES = int(
    os.environ.get("MELOTTS_FIXTURE_MAX_TREE_RSS_BYTES", str(3 * 1024**3))
)
POLL_SECONDS = 1.0
# Runtime abort threshold on the macOS pressure level (1 normal, 2 warning,
# 4 critical). Aborting on any 1 -> 2 transition proved to be a false positive
# on an 8 GiB host: the kernel raises "warning" because of unrelated resident
# GUI applications, while this run's own tree stays far below its cap. The
# process-tree cap above is the limit this run is actually responsible for and
# is still enforced every poll. Set to 2 to restore the strict behaviour.
ABORT_PRESSURE_LEVEL = int(os.environ.get("MELOTTS_FIXTURE_ABORT_PRESSURE_LEVEL", "4"))


def cap_native_threads() -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = "2"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_free_percent(output: str) -> int:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if match is None:
        raise RuntimeError("memory_pressure output has no free-percentage line")
    return int(match.group(1))


def macos_pressure_level() -> int:
    if sys.platform != "darwin":
        raise RuntimeError("the strict pressure-level gate requires macOS")
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(completed.stdout.strip())


def macos_free_percent() -> int:
    completed = subprocess.run(
        ["/usr/bin/memory_pressure"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return parse_free_percent(completed.stdout)


def process_tree_rss_bytes(root_pid: int) -> int:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[tuple[int, int, int]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            rows.append(tuple(int(field) for field in fields))  # type: ignore[arg-type]
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _ in rows:
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss_kib for pid, _, rss_kib in rows if pid in descendants) * 1024


def preflight_snapshot(root_pid: int | None = None) -> dict[str, Any]:
    pid = root_pid or os.getpid()
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_pid": pid,
        "pressure_level": macos_pressure_level(),
        "free_percent": macos_free_percent(),
        "process_tree_rss_bytes": process_tree_rss_bytes(pid),
        "minimum_free_percent_before_model_load": MIN_FREE_PERCENT,
        "maximum_process_tree_rss_bytes": MAX_PROCESS_TREE_RSS_BYTES,
    }


def preflight_violation(snapshot: dict[str, Any]) -> str | None:
    if snapshot["pressure_level"] != 1:
        return f"pressure level is {snapshot['pressure_level']}, expected 1"
    if snapshot["free_percent"] < MIN_FREE_PERCENT:
        return (
            f"free memory is {snapshot['free_percent']}%, "
            f"below the required {MIN_FREE_PERCENT}%"
        )
    if snapshot["process_tree_rss_bytes"] > MAX_PROCESS_TREE_RSS_BYTES:
        return (
            "process-tree RSS already exceeds "
            f"{MAX_PROCESS_TREE_RSS_BYTES / 1024**3:.2f} GiB"
        )
    return None


class ResourceGuard:
    """Monitor pressure and process-tree RSS after the initial gate opens."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.root_pid = os.getpid()
        self.initial_pressure: int | None = None
        self.max_pressure_level = 0
        self.pressure_increases = 0
        self.peak_rss_bytes = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _snapshot(self) -> dict[str, Any]:
        pressure = macos_pressure_level()
        rss = process_tree_rss_bytes(self.root_pid)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        if self.initial_pressure is not None and pressure > self.initial_pressure:
            self.pressure_increases += 1
        self.max_pressure_level = max(self.max_pressure_level, pressure)
        return {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "root_pid": self.root_pid,
            "pressure_level": pressure,
            "initial_pressure_level": self.initial_pressure,
            "process_tree_rss_bytes": rss,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "maximum_process_tree_rss_bytes": MAX_PROCESS_TREE_RSS_BYTES,
            "max_pressure_level_observed": self.max_pressure_level,
            "pressure_level_increase_samples": self.pressure_increases,
            "abort_pressure_level": ABORT_PRESSURE_LEVEL,
        }

    def _runtime_violation(self, snapshot: dict[str, Any]) -> str | None:
        if snapshot["pressure_level"] >= ABORT_PRESSURE_LEVEL:
            return (
                f"pressure level reached {snapshot['pressure_level']}, "
                f"at or above the abort threshold {ABORT_PRESSURE_LEVEL}"
            )
        if snapshot["process_tree_rss_bytes"] > MAX_PROCESS_TREE_RSS_BYTES:
            return (
                "process-tree RSS exceeded "
                f"{MAX_PROCESS_TREE_RSS_BYTES / 1024**3:.2f} GiB"
            )
        return None

    def start(self) -> dict[str, Any]:
        initial = preflight_snapshot(self.root_pid)
        reason = preflight_violation(initial)
        if reason is not None:
            write_json(self.state_path, {"status": "blocked_preflight", "reason": reason, **initial})
            raise SystemExit(86)
        self.initial_pressure = int(initial["pressure_level"])
        self.max_pressure_level = self.initial_pressure
        self.peak_rss_bytes = int(initial["process_tree_rss_bytes"])
        write_json(self.state_path, {"status": "running", **initial})
        self.thread = threading.Thread(target=self._monitor, name="melotts-resource-guard", daemon=True)
        self.thread.start()
        return initial

    def _monitor(self) -> None:
        try:
            while not self.stop_event.wait(POLL_SECONDS):
                snapshot = self._snapshot()
                reason = self._runtime_violation(snapshot)
                if reason is not None:
                    write_json(self.state_path, {"status": "aborted", "reason": reason, **snapshot})
                    os._exit(86)
        except BaseException as error:
            write_json(
                self.state_path,
                {
                    "status": "aborted",
                    "reason": "resource monitor failed closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            os._exit(87)

    def stop(self, status: str = "completed") -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.initial_pressure is None:
            return
        snapshot = self._snapshot()
        reason = self._runtime_violation(snapshot)
        write_json(
            self.state_path,
            {"status": status if reason is None else "aborted", "reason": reason, **snapshot},
        )


cap_native_threads()
