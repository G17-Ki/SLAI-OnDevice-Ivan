#!/usr/bin/env python3
"""Find or redact credential-bearing query strings in text artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlsplit


URL_RE = re.compile(r"https?://[^\s\"'<>]+")
SENSITIVE_QUERY_KEYS = {
    "awsaccesskeyid",
    "signature",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-signature",
}
TRAILING_PUNCTUATION = ").,;]"


def redact_text(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        raw = match.group(0)
        url = raw.rstrip(TRAILING_PUNCTUATION)
        suffix = raw[len(url) :]
        query_keys = {
            key.lower()
            for key, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True)
        }
        if not query_keys.intersection(SENSITIVE_QUERY_KEYS):
            return raw
        count += 1
        return f"{url.split('?', 1)[0]}?[REDACTED_SIGNED_QUERY]{suffix}"

    return URL_RE.sub(replace, text), count


def iter_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (candidate for candidate in path.rglob("*") if candidate.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite affected files; without this flag, only report findings",
    )
    args = parser.parse_args()

    affected_files = 0
    affected_urls = 0
    for path in iter_files(args.paths):
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        redacted, count = redact_text(original)
        if not count:
            continue
        affected_files += 1
        affected_urls += count
        print(f"{path}: {count}")
        if args.write:
            path.write_text(redacted, encoding="utf-8")

    action = "redacted" if args.write else "found"
    print(f"{action}_files={affected_files} {action}_urls={affected_urls}")
    return 0 if args.write or affected_urls == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
