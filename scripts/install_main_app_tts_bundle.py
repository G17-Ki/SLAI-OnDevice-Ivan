#!/usr/bin/env python3
"""Install a staged neural-TTS bundle into the main Android demo app.

The installer is deliberately file-by-file and non-destructive.  It never
clears app data or removes files from the destination.  Existing files that
match the expected size and SHA-256 are skipped, so an interrupted transfer can
be resumed by running the same command again.

The source bundle must have been produced by
``scripts/tts_research_20260726/stage_device_bundle.py`` and must contain both
``candidates.json`` and ``bundle_manifest.json``.  The main app must already be
installed, and adb must have an authorised connection while provisioning.
Neither adb nor the host is needed after the files have been installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


PACKAGE = "ai.liquid.lfmdemo.dev"
ACTIVITY = f"{PACKAGE}/.MainActivity"
REMOTE_ROOT = f"/sdcard/Android/data/{PACKAGE}/files/tts-eval"
PARTIAL_SUFFIX = ".lmf-install.partial"
METADATA_ORDER = ("bundle_manifest.json", "phrases.json", "candidates.json")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class InstallError(RuntimeError):
    """An actionable bundle or device provisioning error."""


@dataclass(frozen=True)
class FileSpec:
    relative_path: str
    local_path: Path | None
    bytes: int | None
    sha256: str | None
    role: str

    @property
    def remotely_authenticatable(self) -> bool:
        return self.bytes is not None or self.sha256 is not None


@dataclass(frozen=True)
class ExistenceSpec:
    relative_path: str
    require_directory: bool


@dataclass(frozen=True)
class BundlePlan:
    bundle_root: Path
    candidate_count: int
    assets: tuple[FileSpec, ...]
    metadata: tuple[FileSpec, ...]
    existence_checks: tuple[ExistenceSpec, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RemoteState:
    exists: bool
    is_file: bool
    bytes: int | None = None
    sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise InstallError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InstallError(f"{label} is not readable JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise InstallError(f"{label} must contain a JSON object: {path}")
    return value


def normalise_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InstallError(f"{label} must be a non-empty relative path")
    if value != value.strip():
        raise InstallError(f"{label} has leading or trailing whitespace: {value!r}")
    if "\\" in value or any(ord(char) < 32 for char in value):
        raise InstallError(f"{label} contains unsupported characters: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise InstallError(f"{label} escapes the bundle root: {value!r}")
    normalised = path.as_posix()
    if normalised != value.rstrip("/"):
        raise InstallError(f"{label} is not canonical: {value!r}")
    return normalised


def manifest_file_specs(manifest: dict[str, Any]) -> dict[str, tuple[int | None, str | None]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise InstallError("bundle_manifest.json must contain a files object")
    parsed: dict[str, tuple[int | None, str | None]] = {}
    for raw_path, raw_spec in raw_files.items():
        relative = normalise_relative_path(raw_path, "manifest file path")
        if relative in parsed:
            raise InstallError(f"duplicate manifest path: {relative}")
        if not isinstance(raw_spec, dict):
            raise InstallError(f"manifest entry for {relative} must be an object")
        expected_bytes = raw_spec.get("bytes")
        expected_sha = raw_spec.get("sha256")
        if expected_bytes is not None:
            if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
                raise InstallError(f"manifest bytes for {relative} must be a non-negative integer")
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
                raise InstallError(f"manifest SHA-256 for {relative} is invalid")
            expected_sha = expected_sha.lower()
        if expected_bytes is None and expected_sha is None:
            raise InstallError(f"manifest entry for {relative} has neither bytes nor SHA-256")
        parsed[relative] = expected_bytes, expected_sha
    return parsed


def candidate_references(candidates: dict[str, Any]) -> tuple[int, tuple[str, ...]]:
    raw_candidates = candidates.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise InstallError("candidates.json must contain a non-empty candidates array")
    seen_ids: set[str] = set()
    references: set[str] = set()
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise InstallError(f"candidate {index} must be an object")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise InstallError(f"candidate {index} has no valid id")
        if candidate_id in seen_ids:
            raise InstallError(f"duplicate candidate id: {candidate_id}")
        seen_ids.add(candidate_id)
        raw_files = candidate.get("files")
        if not isinstance(raw_files, dict) or not raw_files:
            raise InstallError(f"candidate {candidate_id} has no files object")
        for key, raw_value in raw_files.items():
            if not isinstance(raw_value, str):
                raise InstallError(f"candidate {candidate_id} file {key!r} must be a string")
            parts = [part.strip() for part in raw_value.split(",")]
            if not parts or any(not part for part in parts):
                raise InstallError(f"candidate {candidate_id} file {key!r} has an empty path")
            for part in parts:
                references.add(
                    normalise_relative_path(part, f"candidate {candidate_id} file {key!r}")
                )
        reference_wav = candidate.get("reference_wav")
        if reference_wav:
            references.add(
                normalise_relative_path(reference_wav, f"candidate {candidate_id} reference_wav")
            )
    return len(raw_candidates), tuple(sorted(references))


def ensure_path_within_bundle(bundle_root: Path, path: Path, label: str) -> None:
    try:
        path.resolve(strict=True).relative_to(bundle_root)
    except (OSError, ValueError) as error:
        raise InstallError(f"{label} resolves outside the bundle: {path}") from error


def make_local_spec(
    bundle_root: Path,
    relative: str,
    manifest_specs: dict[str, tuple[int | None, str | None]],
    role: str,
    digest_cache: dict[Path, str],
) -> FileSpec:
    path = bundle_root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file():
        raise InstallError(f"referenced bundle file is missing or is a symlink: {relative}")
    ensure_path_within_bundle(bundle_root, path, "bundle file")
    actual_bytes = path.stat().st_size
    actual_sha = digest_cache.setdefault(path, sha256_file(path))
    manifest_spec = manifest_specs.get(relative)
    if manifest_spec is not None:
        expected_bytes, expected_sha = manifest_spec
        if expected_bytes is not None and actual_bytes != expected_bytes:
            raise InstallError(
                f"source size mismatch for {relative}: {actual_bytes} != {expected_bytes}"
            )
        if expected_sha is not None and actual_sha != expected_sha:
            raise InstallError(
                f"source SHA-256 mismatch for {relative}: {actual_sha} != {expected_sha}"
            )
    return FileSpec(relative, path, actual_bytes, actual_sha, role)


def expand_local_reference(bundle_root: Path, relative: str) -> tuple[bool, tuple[str, ...]]:
    path = bundle_root.joinpath(*PurePosixPath(relative).parts)
    if not path.exists():
        return False, ()
    if path.is_symlink():
        raise InstallError(f"candidate reference is a symlink: {relative}")
    ensure_path_within_bundle(bundle_root, path, "candidate reference")
    if path.is_file():
        return False, (relative,)
    if not path.is_dir():
        raise InstallError(f"candidate reference is neither a file nor a directory: {relative}")
    files: list[str] = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise InstallError(f"candidate directory contains a symlink: {child}")
        if child.is_file():
            ensure_path_within_bundle(bundle_root, child, "candidate directory member")
            files.append(child.relative_to(bundle_root).as_posix())
    if not files:
        raise InstallError(f"candidate directory is empty: {relative}")
    return True, tuple(files)


def build_bundle_plan(bundle: Path, manifest_only: bool = False) -> BundlePlan:
    try:
        bundle_root = bundle.expanduser().resolve(strict=True)
    except OSError as error:
        raise InstallError(f"bundle directory is missing: {bundle}") from error
    if not bundle_root.is_dir():
        raise InstallError(f"bundle is not a directory: {bundle_root}")

    candidates_path = bundle_root / "candidates.json"
    manifest_path = bundle_root / "bundle_manifest.json"
    candidates = load_json(candidates_path, "candidates.json")
    manifest = load_json(manifest_path, "bundle_manifest.json")
    if candidates.get("schema_version") != 1:
        raise InstallError("unsupported candidates.json schema_version (expected 1)")
    if manifest.get("schema_version") != 1:
        raise InstallError("unsupported bundle_manifest.json schema_version (expected 1)")
    manifest_specs = manifest_file_specs(manifest)
    candidate_count, references = candidate_references(candidates)

    declared_count = manifest.get("candidate_count")
    if declared_count is not None and declared_count != candidate_count:
        raise InstallError(
            f"manifest candidate_count is {declared_count}, but candidates.json contains {candidate_count}"
        )
    missing_sources = manifest.get("missing_sources", [])
    if missing_sources:
        raise InstallError(f"bundle manifest records missing sources: {missing_sources!r}")
    if "candidates.json" not in manifest_specs:
        raise InstallError("bundle manifest must authenticate candidates.json")

    digest_cache: dict[Path, str] = {}
    asset_paths: set[str] = set()
    remote_only_specs: dict[str, tuple[int | None, str | None]] = {}
    existence_checks: set[tuple[str, bool]] = set()
    warnings: list[str] = []

    for reference in references:
        local_path = bundle_root.joinpath(*PurePosixPath(reference).parts)
        if local_path.exists():
            is_directory, expanded = expand_local_reference(bundle_root, reference)
            asset_paths.update(expanded)
            if manifest_only and is_directory:
                existence_checks.add((reference, True))
            continue
        if not manifest_only:
            raise InstallError(f"candidate reference is missing from the source bundle: {reference}")

        exact = manifest_specs.get(reference)
        descendants = {
            path: spec
            for path, spec in manifest_specs.items()
            if path.startswith(reference.rstrip("/") + "/")
        }
        if exact is not None:
            if exact[1] is None:
                raise InstallError(
                    f"manifest-only mode requires a SHA-256 for resident asset: {reference}"
                )
            remote_only_specs[reference] = exact
        elif descendants:
            missing_digests = sorted(
                path for path, (_, expected_sha) in descendants.items()
                if expected_sha is None
            )
            if missing_digests:
                raise InstallError(
                    "manifest-only mode requires SHA-256 for every resident asset under "
                    f"{reference}: {missing_digests[0]}"
                )
            remote_only_specs.update(descendants)
            existence_checks.add((reference, True))
        else:
            raise InstallError(
                "manifest-only mode cannot authenticate resident asset missing from the "
                f"source bundle and manifest: {reference}"
            )

    assets: dict[str, FileSpec] = {}
    for relative in sorted(asset_paths):
        assets[relative] = make_local_spec(
            bundle_root, relative, manifest_specs, "candidate asset", digest_cache
        )
    for relative, (expected_bytes, expected_sha) in sorted(remote_only_specs.items()):
        assets.setdefault(
            relative,
            FileSpec(relative, None, expected_bytes, expected_sha, "resident candidate asset"),
        )

    metadata: list[FileSpec] = []
    for relative in METADATA_ORDER:
        path = bundle_root / relative
        if path.exists():
            metadata.append(
                make_local_spec(bundle_root, relative, manifest_specs, "bundle metadata", digest_cache)
            )
        elif relative in {"candidates.json", "bundle_manifest.json"}:
            raise InstallError(f"required bundle metadata is missing: {relative}")

    return BundlePlan(
        bundle_root=bundle_root,
        candidate_count=candidate_count,
        assets=tuple(assets[path] for path in sorted(assets)),
        metadata=tuple(metadata),
        existence_checks=tuple(
            ExistenceSpec(path, require_directory)
            for path, require_directory in sorted(existence_checks)
        ),
        warnings=tuple(warnings),
    )


class Adb:
    def __init__(self, executable: str, serial: str | None):
        resolved = shutil.which(executable)
        if resolved is None and Path(executable).is_file():
            resolved = str(Path(executable).resolve())
        if resolved is None:
            raise InstallError(f"adb executable not found: {executable}")
        self.executable = resolved
        self.serial = self._select_serial(serial)

    def _base(self) -> list[str]:
        command = [self.executable]
        if getattr(self, "serial", None):
            command.extend(("-s", self.serial))
        return command

    def run(
        self,
        arguments: Sequence[str],
        *,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [*self._base(), *arguments],
            check=False,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise InstallError(
                f"adb command failed ({result.returncode}): {' '.join(arguments)}"
                + (f": {detail}" if detail else "")
            )
        return result

    def _select_serial(self, requested: str | None) -> str:
        result = subprocess.run(
            [self.executable, "devices"],
            check=False,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise InstallError(f"adb devices failed: {result.stderr.strip()}")
        devices: dict[str, str] = {}
        for line in result.stdout.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 2:
                devices[fields[0]] = fields[1]
        if requested:
            state = devices.get(requested)
            if state != "device":
                raise InstallError(
                    f"adb device {requested!r} is not authorised and online (state={state or 'absent'})"
                )
            return requested
        online = sorted(serial for serial, state in devices.items() if state == "device")
        if len(online) != 1:
            raise InstallError(
                f"expected exactly one authorised adb device, found {len(online)}; use --serial"
            )
        return online[0]

    def shell(self, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.run(("shell", command), check=check)

    def push(self, local: Path, remote: str) -> subprocess.CompletedProcess[str]:
        return self.run(("push", str(local), remote), check=False)


def remote_path(relative: str) -> str:
    relative = normalise_relative_path(relative, "remote relative path")
    result = f"{REMOTE_ROOT}/{relative}"
    if not result.startswith(REMOTE_ROOT + "/"):
        raise InstallError(f"refusing out-of-scope remote path: {result}")
    return result


def quote_remote(path: str) -> str:
    return shlex.quote(path)


def require_installed_app(adb: Adb) -> None:
    result = adb.shell(f"pm path {shlex.quote(PACKAGE)}", check=False)
    if result.returncode != 0 or not result.stdout.strip().startswith("package:"):
        raise InstallError(
            f"{PACKAGE} is not installed. Install the main demo APK with adb install -r first."
        )


def ensure_remote_root(adb: Adb) -> None:
    quoted = quote_remote(REMOTE_ROOT)
    first = adb.shell(f"mkdir -p {quoted} && test -d {quoted}", check=False)
    if first.returncode == 0:
        return
    # Launching once lets Android create getExternalFilesDir() on builds where
    # shell cannot create the package-owned directory before first launch.
    adb.shell(f"am start -W -n {shlex.quote(ACTIVITY)} >/dev/null", check=False)
    time.sleep(1)
    second = adb.shell(f"mkdir -p {quoted} && test -d {quoted}", check=False)
    if second.returncode == 0:
        return
    third = adb.shell(
        f"run-as {shlex.quote(PACKAGE)} mkdir -p {quoted}", check=False
    )
    if third.returncode != 0:
        detail = (third.stderr or second.stderr or first.stderr or "").strip()
        raise InstallError(
            f"cannot create the app external-files directory {REMOTE_ROOT}"
            + (f": {detail}" if detail else "")
        )


def query_remote_file(adb: Adb, relative: str, need_hash: bool) -> RemoteState:
    path = remote_path(relative)
    quoted = quote_remote(path)
    hash_command = f"; toybox sha256sum {quoted}" if need_hash else ""
    command = (
        f"if [ ! -e {quoted} ]; then exit 44; fi; "
        f"if [ ! -f {quoted} ]; then printf 'not-file\\n'; exit 45; fi; "
        f"toybox stat -c %s {quoted}{hash_command}"
    )
    result = adb.shell(command, check=False)
    if result.returncode == 44:
        return RemoteState(False, False)
    if result.returncode == 45:
        return RemoteState(True, False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise InstallError(f"could not inspect {path}: {detail or 'adb shell failed'}")
    lines = [line.strip() for line in result.stdout.replace("\r", "").splitlines() if line.strip()]
    try:
        actual_bytes = int(lines[0])
    except (IndexError, ValueError) as error:
        raise InstallError(f"unexpected remote stat output for {path}: {result.stdout!r}") from error
    actual_sha = None
    if need_hash:
        if len(lines) < 2 or not SHA256_RE.fullmatch(lines[1].split()[0]):
            raise InstallError(f"unexpected remote SHA-256 output for {path}: {result.stdout!r}")
        actual_sha = lines[1].split()[0].lower()
    return RemoteState(True, True, actual_bytes, actual_sha)


def query_remote_existence(adb: Adb, spec: ExistenceSpec) -> bool:
    path = remote_path(spec.relative_path)
    quoted = quote_remote(path)
    predicate = "-d" if spec.require_directory else "-e"
    result = adb.shell(f"test {predicate} {quoted}", check=False)
    if result.returncode in {0, 1}:
        return result.returncode == 0
    detail = (result.stderr or result.stdout or "").strip()
    raise InstallError(f"could not inspect {path}: {detail or 'adb shell failed'}")


def state_matches(spec: FileSpec, state: RemoteState) -> bool:
    if not state.exists or not state.is_file:
        return False
    if spec.bytes is not None and state.bytes != spec.bytes:
        return False
    if spec.sha256 is not None and state.sha256 != spec.sha256:
        return False
    return True


def inspect_remote_spec(adb: Adb, spec: FileSpec) -> RemoteState:
    return query_remote_file(adb, spec.relative_path, need_hash=spec.sha256 is not None)


def wait_for_device(adb: Adb, seconds: int = 30) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = adb.run(("get-state",), check=False)
        if result.returncode == 0 and result.stdout.strip() == "device":
            return True
        time.sleep(1)
    return False


def ensure_remote_parent(adb: Adb, destination: str) -> None:
    parent = str(PurePosixPath(destination).parent)
    result = adb.shell(f"mkdir -p {quote_remote(parent)}", check=False)
    if result.returncode == 0:
        return
    fallback = adb.shell(
        f"run-as {shlex.quote(PACKAGE)} mkdir -p {quote_remote(parent)}", check=False
    )
    if fallback.returncode != 0:
        detail = (fallback.stderr or result.stderr or "").strip()
        raise InstallError(f"cannot create remote directory {parent}: {detail}")


def install_file(adb: Adb, spec: FileSpec, max_retries: int) -> None:
    if spec.local_path is None or spec.bytes is None or spec.sha256 is None:
        raise InstallError(f"no authenticated local source is available for {spec.relative_path}")
    destination = remote_path(spec.relative_path)
    partial = destination + PARTIAL_SUFFIX
    ensure_remote_parent(adb, destination)

    partial_relative = spec.relative_path + PARTIAL_SUFFIX
    partial_state = query_remote_file(adb, partial_relative, need_hash=True)
    if not state_matches(spec, partial_state):
        last_detail = ""
        for attempt in range(1, max_retries + 1):
            result = adb.push(spec.local_path, partial)
            if result.returncode == 0:
                partial_state = query_remote_file(adb, partial_relative, need_hash=True)
                if state_matches(spec, partial_state):
                    break
                last_detail = "remote partial failed size/SHA-256 verification"
            else:
                last_detail = (result.stderr or result.stdout or "adb push failed").strip()
            if attempt < max_retries:
                print(f"  retry {attempt}/{max_retries}: {spec.relative_path}: {last_detail}")
                wait_for_device(adb)
        else:
            raise InstallError(
                f"failed to transfer {spec.relative_path} after {max_retries} attempts: {last_detail}"
            )
    else:
        print(f"  promoting verified partial: {spec.relative_path}")

    quoted_partial = quote_remote(partial)
    quoted_destination = quote_remote(destination)
    moved = adb.shell(
        f"chmod 644 {quoted_partial} && mv -f {quoted_partial} {quoted_destination}",
        check=False,
    )
    if moved.returncode != 0:
        moved = adb.shell(
            f"run-as {shlex.quote(PACKAGE)} mv -f {quoted_partial} {quoted_destination}",
            check=False,
        )
    if moved.returncode != 0:
        detail = (moved.stderr or moved.stdout or "").strip()
        raise InstallError(f"could not atomically install {spec.relative_path}: {detail}")
    final_state = inspect_remote_spec(adb, spec)
    if not state_matches(spec, final_state):
        raise InstallError(f"post-install verification failed for {spec.relative_path}")


def free_bytes_on_shared_storage(adb: Adb) -> int | None:
    result = adb.shell("df -Pk /sdcard | tail -n 1", check=False)
    if result.returncode != 0:
        return None
    fields = result.stdout.replace("\r", "").split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[3]) * 1024
    except ValueError:
        return None


def peak_additional_bytes(
    transfers: Iterable[tuple[FileSpec, RemoteState]],
) -> int:
    retained_growth = 0
    peak = 0
    for spec, old_state in transfers:
        assert spec.bytes is not None
        # The old destination and the new partial coexist until the atomic mv.
        peak = max(peak, retained_growth + spec.bytes)
        old_bytes = old_state.bytes if old_state.exists and old_state.is_file else 0
        retained_growth += spec.bytes - (old_bytes or 0)
    return max(0, peak)


def print_plan(plan: BundlePlan, manifest_only: bool) -> None:
    known_bytes = sum(spec.bytes or 0 for spec in plan.assets)
    authenticated = sum(1 for spec in plan.assets if spec.sha256 is not None)
    print(f"bundle: {plan.bundle_root}")
    print(f"target: {REMOTE_ROOT}")
    print(f"candidates: {plan.candidate_count}")
    print(
        f"referenced files: {len(plan.assets)} "
        f"({known_bytes / 1048576:.1f} MiB known; {authenticated} SHA-256 authenticated)"
    )
    print(f"metadata files: {len(plan.metadata)}")
    if manifest_only:
        print("mode: manifest-only (resident model files are verified, never transferred)")
    for warning in plan.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


def provision(
    plan: BundlePlan,
    adb: Adb,
    *,
    manifest_only: bool,
    verify_only: bool,
    max_retries: int,
) -> None:
    require_installed_app(adb)
    model = adb.shell("getprop ro.product.model", check=False).stdout.strip().replace("\r", "")
    print(f"device: {adb.serial}" + (f" ({model})" if model else ""))
    if not verify_only:
        ensure_remote_root(adb)

    missing_existence = [
        spec.relative_path for spec in plan.existence_checks if not query_remote_existence(adb, spec)
    ]
    if missing_existence:
        raise InstallError(
            "resident candidate paths are missing: " + ", ".join(missing_existence[:10])
        )

    asset_transfers: list[tuple[FileSpec, RemoteState]] = []
    mismatched_resident: list[str] = []
    verified_assets = 0
    for index, spec in enumerate(plan.assets, start=1):
        state = inspect_remote_spec(adb, spec)
        if index % 50 == 0:
            print(f"  inspected {index}/{len(plan.assets)} referenced files", flush=True)
        if state_matches(spec, state):
            verified_assets += 1
            continue
        if manifest_only or verify_only or spec.local_path is None:
            mismatched_resident.append(spec.relative_path)
        else:
            asset_transfers.append((spec, state))

    if mismatched_resident:
        prefix = "verification failed" if verify_only else "manifest-only preflight failed"
        sample = ", ".join(mismatched_resident[:10])
        remainder = len(mismatched_resident) - min(10, len(mismatched_resident))
        if remainder:
            sample += f" (and {remainder} more)"
        raise InstallError(f"{prefix}; resident files are absent or do not match: {sample}")

    metadata_transfers: list[tuple[FileSpec, RemoteState]] = []
    verified_metadata = 0
    for spec in plan.metadata:
        state = inspect_remote_spec(adb, spec)
        if state_matches(spec, state):
            verified_metadata += 1
        elif verify_only:
            raise InstallError(f"metadata verification failed: {spec.relative_path}")
        else:
            metadata_transfers.append((spec, state))

    if verify_only:
        print(
            f"verified: {verified_assets} referenced files, "
            f"{verified_metadata} metadata files; no changes made"
        )
        return

    all_transfers = [*asset_transfers, *metadata_transfers]
    needed = peak_additional_bytes(all_transfers)
    free = free_bytes_on_shared_storage(adb)
    headroom = 64 * 1024 * 1024
    if free is not None:
        print(
            f"shared storage free: {free / 1048576:.1f} MiB; "
            f"estimated peak additional use: {needed / 1048576:.1f} MiB"
        )
        if free < needed + headroom:
            raise InstallError(
                "insufficient shared storage for atomic installation "
                f"({free / 1048576:.1f} MiB free; need about "
                f"{(needed + headroom) / 1048576:.1f} MiB including 64 MiB headroom)"
            )

    print(
        f"assets: {verified_assets} already verified, {len(asset_transfers)} to install"
    )
    for index, (spec, _) in enumerate(asset_transfers, start=1):
        print(f"  asset {index}/{len(asset_transfers)}: {spec.relative_path}")
        install_file(adb, spec, max_retries)

    # Metadata is deliberately installed after every referenced model file;
    # candidates.json is last in METADATA_ORDER, so the app never observes a
    # new candidate registry pointing at incomplete assets.
    print(
        f"metadata: {verified_metadata} already verified, {len(metadata_transfers)} to install"
    )
    for spec, _ in metadata_transfers:
        print(f"  metadata: {spec.relative_path}")
        install_file(adb, spec, max_retries)

    # A final metadata check catches a concurrent or externally interrupted
    # replacement without re-hashing the entire model bundle a second time.
    for spec in plan.metadata:
        if not state_matches(spec, inspect_remote_spec(adb, spec)):
            raise InstallError(f"final metadata verification failed: {spec.relative_path}")
    print(
        f"installation complete: {len(plan.assets)} referenced files and "
        f"{len(plan.metadata)} metadata files verified"
    )
    print("No unrelated files were removed. Restart the app before selecting the updated voices.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install a staged neural-TTS bundle into the main LFM demo app without "
            "clearing app data or deleting unrelated files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Prerequisites:
  * stage the bundle with scripts/tts_research_20260726/stage_device_bundle.py
  * install {PACKAGE} before running this installer
  * connect and authorise adb only for provisioning
  * keep at least one largest changed file plus 64 MiB free for atomic replacement

The fixed destination is:
  {REMOTE_ROOT}

Examples:
  scripts/install_main_app_tts_bundle.sh --bundle /path/to/device-bundle
  scripts/install_main_app_tts_bundle.sh --bundle /path/to/device-bundle --manifest-only
  scripts/install_main_app_tts_bundle.sh --bundle /path/to/device-bundle --verify-only
  scripts/install_main_app_tts_bundle.sh --bundle /path/to/device-bundle --dry-run
""",
    )
    parser.add_argument("--bundle", type=Path, required=True, help="staged bundle directory")
    parser.add_argument("--serial", help="adb device serial (required with multiple devices)")
    parser.add_argument("--adb", default=os.environ.get("ADB", "adb"), help="adb executable")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="verify resident candidate assets and update only bundle metadata",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the expected device bundle without changing it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the local bundle and print the plan without using adb",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="per-file adb push attempts (default: 4)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    try:
        plan = build_bundle_plan(args.bundle, manifest_only=args.manifest_only)
        print_plan(plan, args.manifest_only)
        if args.dry_run:
            print("dry run complete: source validated; adb was not invoked")
            return 0
        adb = Adb(args.adb, args.serial)
        provision(
            plan,
            adb,
            manifest_only=args.manifest_only,
            verify_only=args.verify_only,
            max_retries=args.max_retries,
        )
        return 0
    except InstallError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
