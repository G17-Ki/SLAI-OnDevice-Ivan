#!/usr/bin/env python3
"""Acquire, extract, and verify the pinned llama.cpp Android runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import prepare_model


SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_PATH = SCRIPT_DIR / "assets.json"
DEFAULT_CACHE_ROOT = prepare_model.DEFAULT_CACHE_ROOT


def runtime_ready(directory: Path, files: dict[str, dict[str, Any]]) -> bool:
    try:
        for name, spec in files.items():
            prepare_model.verify_file(directory / name, spec)
    except RuntimeError:
        return False
    return True


def extract_verified(
    archive: Path,
    destination: Path,
    *,
    extracted_directory: str,
    files: dict[str, dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent)
    )
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = {member.name: member for member in bundle.getmembers()}
            for name, spec in files.items():
                member_name = f"{extracted_directory}/{name}"
                member = members.get(member_name)
                if member is None or not member.isfile():
                    raise RuntimeError(f"runtime archive is missing {member_name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read {member_name}")
                target = temporary / name
                with source, target.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
                prepare_model.verify_file(target, spec)
                target.chmod(0o755 if spec.get("executable") else 0o644)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare(
    *,
    external_root: Path,
    source: Path | None,
    seed_dirs: list[Path],
    offline: bool,
) -> dict[str, Any]:
    manifest = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    spec = manifest["runner"]
    archive_spec = {
        "bytes": spec["archive_bytes"],
        "sha256": spec["archive_sha256"],
    }
    external_root = external_root.expanduser().resolve()
    prepare_model.require_external(external_root, "Runtime cache")
    archive = external_root / "archives" / spec["archive_filename"]
    destination = external_root / "runtime" / spec["extracted_directory"]

    if archive.is_file():
        prepare_model.verify_file(archive, archive_spec)
    else:
        candidates: list[Path] = []
        if source is not None:
            candidates.append(source.expanduser())
        candidates.extend(
            seed.expanduser() / spec["archive_filename"] for seed in seed_dirs
        )
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is not None:
            prepare_model.require_external(selected, "Runtime archive source")
            prepare_model.copy_verified(selected, archive, archive_spec)
        elif offline:
            raise RuntimeError(
                "offline mode: supply --source or --seed-dir containing the pinned runtime archive"
            )
        else:
            print(f"Downloading {spec['archive_url']}", file=sys.stderr)
            prepare_model.download_verified(spec["archive_url"], archive, archive_spec)

    prepare_model.verify_file(archive, archive_spec)
    if not runtime_ready(destination, spec["files"]):
        extract_verified(
            archive,
            destination,
            extracted_directory=spec["extracted_directory"],
            files=spec["files"],
        )
    if not runtime_ready(destination, spec["files"]):
        raise RuntimeError("extracted runtime failed its final integrity check")
    return {
        "schema_version": 1,
        "runtime": {
            "version": spec["version"],
            "revision": spec["source_revision"],
            "path": str(destination),
            "archive": str(archive),
            "archive_sha256": spec["archive_sha256"],
            "files": spec["files"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing pinned runtime archive")
    parser.add_argument(
        "--seed-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory containing the pinned archive filename (repeatable)",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"External cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument("--offline", action="store_true", help="Never download")
    parser.add_argument("--output", type=Path, help="Write verified JSON metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = prepare(
            external_root=args.external_root,
            source=args.source,
            seed_dirs=args.seed_dir,
            offline=args.offline,
        )
    except (OSError, RuntimeError, ValueError, tarfile.TarError) as error:
        print(f"prepare_runtime.py: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
