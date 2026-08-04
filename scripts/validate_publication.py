#!/usr/bin/env python3
"""Run the repository's lightweight public-release integrity gate.

The validator deliberately uses only the Python standard library plus local
Git and Bash executables. It reads tracked files and non-ignored untracked
files, never changes the working tree, and exits non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SECRET_PATTERNS = {
    "oauth_authorization_code": re.compile(r"\b4/0A[A-Za-z0-9_-]{20,}\b"),
    "google_access_token": re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    "bearer_token": re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_signed_url": re.compile(r"(?i)[?&]X-Amz-(?:Credential|Signature|Security-Token)="),
    "google_signed_url": re.compile(r"(?i)[?&]X-Goog-(?:Credential|Signature)="),
    "azure_signed_url": re.compile(r"(?i)[?&](?:sig|se|sp|sv)=([^&\s]{12,})"),
}
PERSONAL_PATH = re.compile(r"(?:file://)?/Users/[A-Za-z0-9._-]+")
FORMATTED_TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".gradle",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".properties",
    ".py",
    ".sh",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}
VERBATIM_PATH_PARTS = {
    "device",
    "memory_attribution",
    "memory_matrix",
    "raw",
    "third_party",
    "vendor",
}


@dataclass
class Result:
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def bump(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount

    def fail(self, message: str) -> None:
        self.errors.append(message)


def run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def repository_root() -> Path:
    completed = run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())
    return Path(completed.stdout.decode("utf-8").strip()).resolve()


def candidate_files(root: Path) -> list[Path]:
    completed = run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    paths = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        path = root / raw.decode("utf-8", errors="strict")
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def strict_json(text: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def read_utf8(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check_structured_files(root: Path, files: list[Path], result: Result) -> None:
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json":
            result.bump("json_files")
            try:
                strict_json(path.read_text(encoding="utf-8"))
            except Exception as error:  # noqa: BLE001 - report every parse failure
                result.fail(f"strict JSON: {relative(root, path)}: {error}")
        elif suffix == ".py":
            result.bump("python_files")
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except Exception as error:  # noqa: BLE001
                result.fail(f"Python AST: {relative(root, path)}: {error}")
        elif suffix == ".sh":
            result.bump("shell_files")
            completed = run(["bash", "-n", str(path)], cwd=root, check=False)
            if completed.returncode:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                result.fail(f"shell syntax: {relative(root, path)}: {detail}")


def split_markdown_destination(raw: str) -> str:
    destination = raw.strip()
    if destination.startswith("<"):
        closing = destination.find(">")
        if closing >= 0:
            return destination[1:closing]
    # Unquoted spaces delimit an optional Markdown title. Public paths that
    # contain spaces are expected to use the angle-bracket form.
    return destination.split(maxsplit=1)[0]


def is_ignored(root: Path, path: Path) -> bool:
    completed = run(
        ["git", "check-ignore", "-q", "--", str(path)],
        cwd=root,
        check=False,
    )
    return completed.returncode == 0


def check_markdown_links(root: Path, files: list[Path], result: Result) -> None:
    for source in files:
        if source.suffix.lower() not in {".md", ".markdown"}:
            continue
        result.bump("markdown_files")
        text = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            destination = unquote(split_markdown_destination(match.group(1)))
            if not destination or destination.startswith("#"):
                continue
            lowered = destination.lower()
            if lowered.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            destination = destination.split("#", 1)[0].split("?", 1)[0]
            if not destination:
                continue
            result.bump("markdown_local_links")
            target = Path(destination)
            if target.is_absolute():
                result.fail(
                    f"absolute local Markdown link: {relative(root, source)} -> {destination}"
                )
                continue
            resolved = (source.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                result.fail(
                    f"Markdown link escapes repository: {relative(root, source)} -> {destination}"
                )
                continue
            if not resolved.exists():
                result.fail(
                    f"missing Markdown link: {relative(root, source)} -> {destination}"
                )
            elif is_ignored(root, resolved):
                result.fail(
                    f"ignored Markdown link target: {relative(root, source)} -> {destination}"
                )


def check_text_hygiene(root: Path, files: list[Path], result: Result) -> None:
    for path in files:
        text = read_utf8(path)
        if text is None:
            continue
        result.bump("utf8_text_files")
        name = relative(root, path)
        is_verbatim = bool(set(path.relative_to(root).parts) & VERBATIM_PATH_PARTS)
        if path.suffix.lower() in FORMATTED_TEXT_SUFFIXES and not is_verbatim:
            if text and not text.endswith("\n"):
                result.fail(f"missing final newline: {name}")
            for line_number, line in enumerate(text.splitlines(), start=1):
                trailing = len(line) - len(line.rstrip(" \t"))
                # Two terminal spaces are a deliberate CommonMark hard break.
                allowed_markdown_break = path.suffix.lower() == ".md" and trailing == 2
                if trailing and not allowed_markdown_break:
                    result.fail(f"trailing whitespace: {name}:{line_number}")
                    break
        for personal in PERSONAL_PATH.finditer(text):
            # `/Users/runner` is upstream CI source-location metadata embedded
            # in third-party diagnostics, not a workstation identity.
            if personal.group(0).removeprefix("file://") != "/Users/runner":
                result.fail(f"personal path: {name}: {personal.group(0)}")
                break
        for pattern_name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                result.fail(f"{pattern_name}: {name}")


def check_apk_sidecars(root: Path, files: list[Path], result: Result) -> None:
    for sidecar in files:
        if not sidecar.name.endswith(".apk.sha256"):
            continue
        result.bump("apk_sidecars")
        apk = sidecar.with_suffix("")
        if not apk.is_file():
            result.fail(f"APK sidecar has no binary: {relative(root, sidecar)}")
            continue
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            result.fail(f"invalid APK sidecar: {relative(root, sidecar)}")
            continue
        digest = hashlib.sha256(apk.read_bytes()).hexdigest()
        if digest.lower() != fields[0].lower():
            result.fail(f"APK checksum mismatch: {relative(root, apk)}")


def check_git_diff(root: Path, result: Result) -> None:
    for arguments, label in (
        (["git", "diff", "--check"], "git diff --check"),
        (["git", "diff", "--cached", "--check"], "git diff --cached --check"),
    ):
        completed = run(arguments, cwd=root, check=False)
        result.bump("git_diff_checks")
        if completed.returncode:
            detail = completed.stdout.decode("utf-8", errors="replace").strip()
            result.fail(f"{label}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the summary as JSON instead of human-readable text",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repository_root()
    files = candidate_files(root)
    result = Result(counts={"candidate_files": len(files)})

    check_structured_files(root, files, result)
    check_markdown_links(root, files, result)
    check_text_hygiene(root, files, result)
    check_apk_sidecars(root, files, result)
    check_git_diff(root, result)

    payload = {
        "status": "pass" if not result.errors else "fail",
        "counts": dict(sorted(result.counts.items())),
        "errors": result.errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"publication validation: {payload['status'].upper()}")
        for name, count in payload["counts"].items():
            print(f"  {name}: {count}")
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
