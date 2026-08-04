#!/usr/bin/env python3
"""Host-side protocol tests for the mksh Android ASR companion handler."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDLER = REPO_ROOT / "scripts/android_asr_companion/handler.sh"


def wav_bytes(samples: int = 1600, sample_rate: int = 16000) -> bytes:
    pcm = b"\0\0" * samples
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def request(method: str, target: str, body: bytes = b"", content_type: str | None = None) -> bytes:
    headers = [f"{method} {target} HTTP/1.1", "Host: 127.0.0.1"]
    if content_type is not None:
        headers.extend((f"Content-Type: {content_type}", f"Content-Length: {len(body)}"))
    headers.append("Connection: close")
    return ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body


def decode_response(raw: bytes) -> tuple[int, dict[str, str], dict | None]:
    head, body = raw.split(b"\r\n\r\n", 1)
    lines = head.decode("ascii").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        name, value = line.split(":", 1)
        headers[name.lower()] = value.strip()
    assert int(headers["content-length"]) == len(body)
    payload = json.loads(body) if body else None
    return status, headers, payload


class HandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="asr-companion-test-")
        self.root = Path(self.temporary.name)
        (self.root / "bin").mkdir()
        (self.root / "qnn").mkdir()
        (self.root / "sensevoice").mkdir()
        (self.root / "russian").mkdir()
        (self.root / "tmp").mkdir()

        toybox = self.root / "bin/toybox"
        toybox.write_text("#!/bin/sh\napplet=$1\nshift\nexec \"$applet\" \"$@\"\n")
        runner = self.root / "bin/sherpa-onnx-offline"
        runner.write_text(
            """#!/bin/sh
language=
backend=qnn
wav_count=0
for arg in "$@"; do
  case "$arg" in
    --sense-voice-language=*) language=${arg#*=} ;;
    --encoder=*) backend=ru ;;
    --*) ;;
    *) wav_count=$((wav_count + 1)) ;;
  esac
done
printf '%s\\n' '0.0ms [WARN] provider diagnostic on stdout'
if [ "$backend" = ru ]; then
  if [ "$wav_count" -lt 2 ]; then
    printf '%s\\n' '{"text":"","tokens":[]}'
  else
    printf '%s\\n' '{"text":"Привет, мир","tokens":["П"]}'
    printf '%s\\n' '{"text":"Привет, мир","tokens":["П"]}'
  fi
elif [ "$language" = en ]; then
  printf '%s\\n' '{"lang":"<|en|>","text":"hello world"}'
else
  printf '%s\\n' '{"lang":"<|zh|>","text":"你好，世界"}'
fi
printf '%s\\n' '0.0ms [WARN] trailing provider diagnostic'
"""
        )
        toybox.chmod(0o755)
        runner.chmod(0o755)
        for name in ("libonnxruntime.so",):
            (self.root / "bin" / name).touch()
        for name in (
            "libQnnHtp.so",
            "libQnnSystem.so",
            "libQnnHtpV81Stub.so",
            "libQnnHtpV81Skel.so",
        ):
            (self.root / "qnn" / name).touch()
        for name in ("model.bin", "tokens.txt"):
            (self.root / "sensevoice" / name).touch()
        for name in ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"):
            (self.root / "russian" / name).touch()

        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{self.root / 'bin'}:{self.environment['PATH']}",
                "ASR_COMPANION_ROOT": str(self.root),
                "ASR_RUNNER_ROOT": str(self.root / "bin"),
                "ASR_RUNNER": str(runner),
                "QNN_RUNTIME_ROOT": str(self.root / "qnn"),
                "SENSEVOICE_MODEL_ROOT": str(self.root / "sensevoice"),
                "RUSSIAN_MODEL_ROOT": str(self.root / "russian"),
                "ASR_TMP_ROOT": str(self.root / "tmp"),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self, raw_request: bytes) -> tuple[int, dict[str, str], dict | None]:
        process = subprocess.run(
            ["bash", str(HANDLER)],
            input=raw_request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode("utf-8", "replace"))
        return decode_response(process.stdout)

    def test_health_and_cors(self) -> None:
        status, headers, payload = self.invoke(request("GET", "/health"))
        self.assertEqual(status, 200)
        self.assertEqual(headers["access-control-allow-origin"], "*")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["backends"]["sensevoice_qnn"]["ready"])
        self.assertTrue(payload["backends"]["zipformer_ru_cpu"]["ready"])

    def test_options(self) -> None:
        status, headers, payload = self.invoke(request("OPTIONS", "/asr"))
        self.assertEqual(status, 204)
        self.assertEqual(headers["access-control-allow-methods"], "GET, POST, OPTIONS")
        self.assertEqual(headers["access-control-allow-private-network"], "true")
        self.assertIsNone(payload)

    def test_explicit_qnn_and_russian(self) -> None:
        audio = wav_bytes()
        status, _, zh = self.invoke(request("POST", "/asr?language=zh", audio, "audio/wav"))
        self.assertEqual(status, 200)
        self.assertEqual(zh["candidates"][0]["backend"], "sensevoice_qnn")
        self.assertEqual(zh["candidates"][0]["result"]["text"], "你好，世界")

        status, _, ru = self.invoke(request("POST", "/asr?language=ru", audio, "audio/wav"))
        self.assertEqual(status, 200)
        self.assertEqual(ru["candidates"][0]["backend"], "zipformer_ru_cpu")
        self.assertEqual(ru["candidates"][0]["result"]["text"], "Привет, мир")

        status, _, qnn_auto = self.invoke(
            request("POST", "/asr?language=qnn-auto", audio, "audio/wav")
        )
        self.assertEqual(status, 200)
        self.assertEqual(qnn_auto["requested_language"], "qnn-auto")
        self.assertEqual(qnn_auto["candidates"][0]["backend"], "sensevoice_qnn")

    def test_auto_returns_both_parallel_candidates(self) -> None:
        audio = wav_bytes()
        status, _, payload = self.invoke(
            request("POST", "/asr?language=auto", audio, "application/octet-stream")
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["complete"])
        self.assertTrue(payload["selection_required"])
        self.assertEqual(
            [candidate["backend"] for candidate in payload["candidates"]],
            ["sensevoice_qnn", "zipformer_ru_cpu"],
        )

    def test_rejects_wrong_shape_and_oversized_length(self) -> None:
        wrong_rate = wav_bytes(sample_rate=8000)
        status, _, payload = self.invoke(
            request("POST", "/asr?language=en", wrong_rate, "audio/wav")
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_wav")

        raw = (
            b"POST /asr?language=en HTTP/1.1\r\n"
            b"Content-Type: audio/wav\r\n"
            b"Content-Length: 320046\r\n\r\n"
        )
        status, _, payload = self.invoke(raw)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "body_too_large")


if __name__ == "__main__":
    unittest.main()
