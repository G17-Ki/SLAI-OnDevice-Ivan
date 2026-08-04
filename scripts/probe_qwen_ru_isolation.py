#!/usr/bin/env python3
"""Run a small Russian prompt-isolation grid against the phone Qwen server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "Qwen3-1.7B-Q4_0"
SYSTEM = (
    "Be concise and accurate. Answer in one or two short sentences unless asked "
    "for another length. Use the requested language. Never output hidden reasoning "
    "or <think> tags."
)
RUSSIAN_POLICY = "Ответь на запрос пользователя только по-русски, кириллицей."


def ui_user(text: str) -> str:
    return f"{RUSSIAN_POLICY}\n{text}\n/no_think"


CASES = [
    {
        "name": "exact_ui_temperature_0_2",
        "temperature": 0.2,
        "top_p": 0.8,
        "user": ui_user("Как называется столица России? Ответь одним словом."),
        "expected_any": ["Москва"],
    },
    {
        "name": "exact_ui_temperature_0",
        "temperature": 0.0,
        "top_p": 0.8,
        "user": ui_user("Как называется столица России? Ответь одним словом."),
        "expected_any": ["Москва"],
    },
    {
        "name": "analyze_fact_then_russian",
        "temperature": 0.0,
        "top_p": 0.8,
        "user": ui_user(
            "Сначала проанализируй факт, затем ответь только по-русски: "
            "как называется столица России?"
        ),
        "expected_any": ["Москва"],
    },
    {
        "name": "bilingual_internal_english",
        "temperature": 0.0,
        "top_p": 0.8,
        "user": ui_user(
            "Understand the question in English internally, answer only in Russian: "
            "как называется столица России?"
        ),
        "expected_any": ["Москва"],
    },
    {
        "name": "english_question_russian_output",
        "temperature": 0.0,
        "top_p": 0.8,
        "user": ui_user("What is the capital of Russia? Answer with the city name only."),
        "expected_any": ["Москва"],
    },
    {
        "name": "russian_capital_of_france_control",
        "temperature": 0.0,
        "top_p": 0.8,
        "user": ui_user("Как называется столица Франции? Ответь одним словом."),
        "expected_any": ["Париж"],
    },
    {
        "name": "russian_arithmetic_control",
        "temperature": 0.0,
        "top_p": 0.8,
        "user": ui_user("Сколько будет два плюс два? Ответь одним числом."),
        "expected_any": ["4", "четыре"],
    },
]


def request_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any], float]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
        status = response.status
    return status, body, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:6020")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    output: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": MODEL,
        "cases": [],
    }
    for case in CASES:
        payload = {
            "model": MODEL,
            "stream": False,
            "temperature": case["temperature"],
            "top_p": case["top_p"],
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": case["user"]},
            ],
        }
        status, response, elapsed_ms = request_json(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            payload,
            args.timeout,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = message.get("content") if isinstance(message.get("content"), str) else ""
        reasoning = message.get("reasoning_content")
        item = {
            "name": case["name"],
            "request": payload,
            "http_status": status,
            "elapsed_ms": round(elapsed_ms, 3),
            "expected_any": case["expected_any"],
            "expected_fact_present": any(
                expected.casefold() in text.casefold() for expected in case["expected_any"]
            ),
            "finish_reason": choice.get("finish_reason"),
            "text": text,
            "reasoning_content": reasoning,
            "think_tag_seen": "<think" in text.casefold(),
            "usage": response.get("usage"),
            "timings": response.get("timings"),
            "response": response,
        }
        output["cases"].append(item)
        print(
            f"{item['name']}: status={status} elapsed_ms={item['elapsed_ms']} "
            f"fact_ok={item['expected_fact_present']} text={text!r}"
        )
    output["finished_at"] = datetime.now(timezone.utc).isoformat()
    output["passed_expected_facts"] = sum(
        bool(item["expected_fact_present"]) for item in output["cases"]
    )
    output["case_count"] = len(output["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
