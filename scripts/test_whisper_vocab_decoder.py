#!/usr/bin/env python3
"""Compile and run the dependency-free strict Whisper vocab decoder test."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("ai/liquid/lfmdemo/whisperprobe")
MAIN = ROOT / "android-app/whisper-probe/src/main/java" / PACKAGE
TEST = ROOT / "android-app/whisper-probe/src/test/java" / PACKAGE
PINNED_JDK_17 = Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")


def main() -> None:
    sources = [MAIN / "VocabDecoder.java", TEST / "VocabDecoderHostTest.java"]
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"missing Java source: {source}")

    java = PINNED_JDK_17 / "bin/java"
    javac = PINNED_JDK_17 / "bin/javac"
    if not java.is_file() or not javac.is_file():
        raise SystemExit(f"missing pinned JDK 17: {PINNED_JDK_17}")

    with tempfile.TemporaryDirectory(prefix="whisper-vocab-decoder-") as temporary:
        classes = Path(temporary) / "classes"
        classes.mkdir()
        subprocess.run(
            [
                str(javac),
                "--release",
                "17",
                "-encoding",
                "UTF-8",
                "-d",
                str(classes),
                *map(str, sources),
            ],
            check=True,
            cwd=ROOT,
        )
        subprocess.run(
            [
                str(java),
                "-ea",
                "-cp",
                str(classes),
                "ai.liquid.lfmdemo.whisperprobe.VocabDecoderHostTest",
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
