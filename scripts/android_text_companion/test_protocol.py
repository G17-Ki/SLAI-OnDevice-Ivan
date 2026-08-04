#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_model  # noqa: E402
import prepare_runtime  # noqa: E402
import protocol  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_payload_disables_thinking_and_bounds_output(self) -> None:
        payload = protocol.build_chat_payload(
            "中国的首都是哪里？", language="zh", max_tokens=768, stream=True
        )
        self.assertEqual(payload["model"], "Qwen3-1.7B-Q4_0")
        self.assertEqual(payload["max_tokens"], 768)
        self.assertEqual(payload["repeat_penalty"], 1.08)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("简体中文", payload["messages"][0]["content"])

    def test_payload_rejects_empty_and_unbounded_requests(self) -> None:
        for prompt, maximum in (("", 32), ("ok", 0), ("ok", 769)):
            with self.subTest(prompt=prompt, maximum=maximum):
                with self.assertRaises(ValueError):
                    protocol.build_chat_payload(prompt, max_tokens=maximum)

    def test_extract_completion_and_reasoning_detection(self) -> None:
        clean = {"choices": [{"message": {"content": "Москва."}}]}
        exposed = {
            "choices": [
                {"message": {"content": "Ответ", "reasoning_content": "hidden"}}
            ]
        }
        self.assertEqual(protocol.extract_completion(clean), "Москва.")
        self.assertFalse(protocol.response_exposes_reasoning(clean))
        self.assertTrue(protocol.response_exposes_reasoning(exposed))

    def test_sse_parser_handles_comments_empty_lines_and_done(self) -> None:
        lines = [
            b": keepalive\n",
            b"\n",
            b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}\n',
            'data: {"choices":[{"delta":{"content":"好"}}]}\n',
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n',
        ]
        self.assertEqual("".join(protocol.iter_sse_text(lines)), "你好")

    def test_invalid_sse_json_is_not_silently_ignored(self) -> None:
        with self.assertRaises(ValueError):
            list(protocol.iter_sse_json(["data: not-json\n"]))


class AssetAndScriptTests(unittest.TestCase):
    def test_manifest_pins_expected_model(self) -> None:
        manifest = json.loads((SCRIPT_DIR / "assets.json").read_text(encoding="utf-8"))
        model = manifest["model"]
        self.assertEqual(model["bytes"], 1056782912)
        self.assertEqual(
            model["sha256"],
            "c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5",
        )
        self.assertEqual(manifest["target"]["listen"], "127.0.0.1:6020")
        self.assertEqual(manifest["runtime"]["context_tokens"], 2048)
        self.assertEqual(manifest["runtime"]["max_completion_tokens"], 768)
        self.assertEqual(manifest["runtime"]["generation_threads"], 4)
        self.assertEqual(manifest["runtime"]["batch_threads"], 4)
        self.assertEqual(manifest["runtime"]["reasoning_budget"], 0)
        runner = manifest["runner"]
        self.assertEqual(runner["version"], "b10083")
        self.assertEqual(runner["archive_bytes"], 74822792)
        self.assertEqual(
            runner["archive_sha256"],
            "c53e7989142bb1c56e29f1df780451810363b2cfae829827d4e2d79bb21c7189",
        )
        self.assertEqual(
            runner["files"]["llama-server"]["sha256"],
            "db82377125f69e87c06b402bba1af072e2a4c4cc9945946e4d6378cc213a84d7",
        )

    def test_verify_file_checks_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture.gguf"
            fixture.write_bytes(b"pinned fixture")
            spec = {
                "bytes": fixture.stat().st_size,
                "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            }
            prepare_model.verify_file(fixture, spec)
            fixture.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                prepare_model.verify_file(fixture, spec)

    def test_runtime_extraction_is_allowlisted_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "runtime.tar.gz"
            payload = b"synthetic runtime"
            spec = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "executable": True,
            }
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("bundle/llama-server")
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))
                ignored = tarfile.TarInfo("bundle/unrelated")
                ignored.size = 4
                bundle.addfile(ignored, io.BytesIO(b"skip"))
            destination = root / "out"
            prepare_runtime.extract_verified(
                archive,
                destination,
                extracted_directory="bundle",
                files={"llama-server": spec},
            )
            prepare_model.verify_file(destination / "llama-server", spec)
            self.assertFalse((destination / "unrelated").exists())

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT_DIR / "deploy.sh")], check=True)
        subprocess.run(["sh", "-n", str(SCRIPT_DIR / "start.sh")], check=True)
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "deploy.sh"), "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def test_runtime_command_is_loopback_cpu_and_reasoning_off(self) -> None:
        source = (SCRIPT_DIR / "start.sh").read_text(encoding="utf-8")
        for default in (
            "TEXT_CTX_SIZE=${TEXT_CTX_SIZE:-2048}",
            "TEXT_THREADS=${TEXT_THREADS:-4}",
            "TEXT_BATCH_THREADS=${TEXT_BATCH_THREADS:-4}",
        ):
            self.assertIn(default, source)
        for required in (
            "--host 127.0.0.1",
            "--port 6020",
            "--alias Qwen3-1.7B-Q4_0",
            "--n-gpu-layers 0",
            "--reasoning off",
            "--reasoning-budget 0",
            "--reasoning-format deepseek",
            "--threads-http 1",
            "--parallel 1",
        ):
            self.assertIn(required, source)
        self.assertNotIn("toybox awk", source)
        deploy = (SCRIPT_DIR / "deploy.sh").read_text(encoding="utf-8")
        for default in ("ctx_size=2048", "threads=4", "batch_threads=4"):
            self.assertIn(default, deploy)
        self.assertIn("prepare_runtime.py", deploy)
        self.assertIn("runtime-b10083", deploy)
        self.assertNotIn("nativeLibraryDir", deploy)

    def test_no_model_weight_is_stored_with_scripts(self) -> None:
        self.assertEqual(list(SCRIPT_DIR.glob("*.gguf")), [])
        self.assertEqual(list(SCRIPT_DIR.glob("*.tar.gz")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
