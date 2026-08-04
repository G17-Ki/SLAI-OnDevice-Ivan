#!/usr/bin/env python3
"""Redact local macOS absolute-path prefixes in JSON without changing data."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = (
    (re.compile(re.escape(REPO_ROOT.as_posix()) + r"/"), "<repo>/"),
    (re.compile(r"/Users/[^/]+/"), "<home>/"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite files in place. The default is a read-only check.",
    )
    return parser.parse_args()


def json_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(item for item in path.rglob("*.json") if item.is_file())
        elif path.suffix.casefold() == ".json":
            files.add(path)
        else:
            raise SystemExit(f"Not a JSON file or directory: {path}")
    return sorted(files)


def scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return type(left) is type(right) and left == right


def assert_only_strings_changed(before: Any, after: Any, pointer: str = "$") -> None:
    if isinstance(before, dict):
        if not isinstance(after, dict) or list(before) != list(after):
            raise ValueError(f"JSON object structure changed at {pointer}")
        for key in before:
            assert_only_strings_changed(before[key], after[key], f"{pointer}.{key}")
        return
    if isinstance(before, list):
        if not isinstance(after, list) or len(before) != len(after):
            raise ValueError(f"JSON array structure changed at {pointer}")
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            assert_only_strings_changed(left, right, f"{pointer}[{index}]")
        return
    if isinstance(before, str):
        if not isinstance(after, str):
            raise ValueError(f"JSON string type changed at {pointer}")
        return
    if not scalar_equal(before, after):
        raise ValueError(f"Non-string JSON value changed at {pointer}: {before!r} -> {after!r}")


def redact_text(text: str) -> tuple[str, int]:
    total = 0
    for pattern, replacement in REPLACEMENTS:
        text, count = pattern.subn(replacement, text)
        total += count
    return text, total


def main() -> int:
    args = parse_args()
    changed = 0
    replacements = 0
    for path in json_files(args.paths):
        original_text = path.read_text(encoding="utf-8")
        if "/Users/" not in original_text:
            continue
        redacted_text, count = redact_text(original_text)
        if count == 0:
            print(f"UNHANDLED {path}")
            continue
        before = json.loads(original_text)
        after = json.loads(redacted_text)
        assert_only_strings_changed(before, after)
        remaining = redacted_text.count("/Users/")
        if remaining:
            raise ValueError(f"{path}: {remaining} unredacted /Users/ occurrence(s)")
        changed += 1
        replacements += count
        action = "WRITE" if args.write else "WOULD_WRITE"
        print(f"{action} {path} ({count} replacement(s))")
        if args.write:
            path.write_text(redacted_text, encoding="utf-8")
    print(
        f"files_changed={changed} replacements={replacements} mode={'write' if args.write else 'check'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
