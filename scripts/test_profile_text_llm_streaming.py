#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("profile_text_llm_streaming.py")
SPEC = importlib.util.spec_from_file_location("profile_text_llm_streaming", SCRIPT)
assert SPEC and SPEC.loader
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class ModelCapabilityTests(unittest.TestCase):
    def args(self, model: str, thinking: str = "off") -> types.SimpleNamespace:
        return types.SimpleNamespace(
            model=model,
            thinking=thinking,
            max_tokens=64,
            system_prompt="Answer directly.",
        )

    def test_qwen25_omits_qwen3_thinking_protocol(self) -> None:
        body = profile.build_request_body(
            self.args("Qwen2.5-1.5B-Instruct-Q4_K_M"),
            {"prompt": "中国的首都是哪里？"},
        )
        self.assertNotIn("chat_template_kwargs", body)
        self.assertNotIn("/no_think", body["messages"][1]["content"])
        self.assertFalse(
            profile.model_capabilities("Qwen2.5-1.5B-Instruct-Q4_K_M")[
                "qwen3_thinking_controls"
            ]
        )

    def test_qwen3_retains_explicit_thinking_controls(self) -> None:
        body = profile.build_request_body(
            self.args("Qwen3-1.7B-Q4_0"),
            {"prompt": "What is 2 + 2?"},
        )
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(body["messages"][1]["content"].endswith(" /no_think"))

    def test_unsupported_thinking_on_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support"):
            profile.build_request_body(
                self.args("Qwen2.5-1.5B-Instruct-Q4_K_M", thinking="on"),
                {"prompt": "Explain briefly."},
            )

    def test_fuji_scorer_accepts_common_thousands_formats(self) -> None:
        suite = json.loads(
            Path(__file__).with_name("multilingual_voice_quality_suite.json").read_text(
                encoding="utf-8"
            )
        )
        case = next(item for item in suite if item["id"] == "en_fuji")
        for response in ("3,776 meters", "3, 776 meters", "3 776 meters", "3776 meters"):
            with self.subTest(response=response):
                self.assertTrue(profile.score(case, response))


if __name__ == "__main__":
    unittest.main(verbosity=2)
