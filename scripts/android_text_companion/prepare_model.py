#!/usr/bin/env python3
"""Acquire and verify the external Qwen model used by the Android companion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ASSETS_PATH = SCRIPT_DIR / "assets.json"
DEFAULT_CACHE_ROOT = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "lmf-demo"
    / "android-text-companion"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_external(path: Path, label: str) -> None:
    if is_within(path, REPO_ROOT):
        raise RuntimeError(f"{label} must remain outside the repository: {path}")


def verify_file(path: Path, spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing pinned model: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != spec["bytes"]:
        raise RuntimeError(
            f"size mismatch for {path.name}: {actual_bytes} != {spec['bytes']}"
        )
    actual_hash = sha256(path)
    if actual_hash != spec["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: {actual_hash} != {spec['sha256']}"
        )


def copy_verified(source: Path, destination: Path, spec: dict[str, Any]) -> None:
    verify_file(source, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        verify_file(temporary, spec)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_verified(url: str, destination: Path, spec: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LMF-demo-android-text-companion/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        verify_file(temporary, spec)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare(
    *,
    external_root: Path,
    source: Path | None,
    seed_dirs: list[Path],
    offline: bool,
) -> dict[str, Any]:
    manifest = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    spec = manifest["model"]
    external_root = external_root.expanduser().resolve()
    require_external(external_root, "Model cache")
    destination = external_root / "models" / spec["filename"]

    if destination.is_file():
        verify_file(destination, spec)
    else:
        candidates: list[Path] = []
        if source is not None:
            candidates.append(source.expanduser())
        candidates.extend(seed.expanduser() / spec["filename"] for seed in seed_dirs)
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is not None:
            require_external(selected, "Model source")
            copy_verified(selected, destination, spec)
        elif offline:
            raise RuntimeError(
                "offline mode: supply --source or --seed-dir containing the pinned model"
            )
        else:
            print(f"Downloading {spec['url']}", file=sys.stderr)
            download_verified(spec["url"], destination, spec)

    verify_file(destination, spec)
    return {
        "schema_version": 1,
        "model": {
            "name": spec["name"],
            "path": str(destination),
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
            "repository": spec["repository"],
            "revision": spec["revision"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing pinned GGUF to verify")
    parser.add_argument(
        "--seed-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory containing the pinned filename (repeatable)",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"External cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument("--offline", action="store_true", help="Never download")
    parser.add_argument("--output", type=Path, help="Write the verified JSON manifest")
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
    except (OSError, RuntimeError, ValueError) as error:
        print(f"prepare_model.py: {error}", file=sys.stderr)
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
