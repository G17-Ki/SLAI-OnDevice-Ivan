#!/usr/bin/env python3
"""Compare the current and hardened Russian UI policies on phone Qwen."""

from __future__ import annotations

import argparse
import json
import re
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
POLICIES = {
    "current": "Ответь на запрос пользователя только по-русски, кириллицей.",
    "hardened": (
        "Сначала проверь основной факт. Ответь точно и кратко только по-русски. "
        "Если не уверен, скажи «Не знаю»; не выдумывай."
    ),
}
CASES = [
    {
        "name": "capital_russia",
        "prompt": "Как называется столица России? Ответь только названием города.",
        "expected_regex": r"москв",
    },
    {
        "name": "capital_japan",
        "prompt": "Как называется столица Японии? Ответь только названием города.",
        "expected_regex": r"токио",
    },
    {
        "name": "multiply_11_12",
        "prompt": "Сколько будет 11 умножить на 12? Ответь только числом.",
        "expected_regex": r"\b132\b",
    },
    {
        "name": "water_freezing_point",
        "prompt": "При какой температуре по Цельсию замерзает чистая вода при нормальном давлении? Ответь кратко.",
        "expected_regex": r"(?:\b0\b|нол)",
    },
    {
        "name": "largest_planet",
        "prompt": "Какая планета Солнечной системы самая большая? Ответь только названием.",
        "expected_regex": r"юпитер",
    },
    {
        "name": "instruction_exact_word",
        "prompt": "Напиши ровно одно слово: СИНИЙ",
        "expected_regex": r"^\s*синий[.!]?\s*$",
    },
    {
        "name": "unknown_fictional_mayor",
        "prompt": "Как зовут нынешнего мэра вымышленного города Нордоград на Марсе? Если достоверного факта нет, ответь «Не знаю».",
        "expected_regex": r"не знаю",
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
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 5:
        raise SystemExit("--repeats must be between 1 and 5")

    results: list[dict[str, Any]] = []
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    for policy_name, policy in POLICIES.items():
        for case in CASES:
            for repeat in range(1, args.repeats + 1):
                payload = {
                    "model": MODEL,
                    "stream": False,
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "max_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": f"{policy}\n{case['prompt']}\n/no_think",
                        },
                    ],
                }
                status, response, elapsed_ms = request_json(endpoint, payload, args.timeout)
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                text = message.get("content") if isinstance(message.get("content"), str) else ""
                reasoning = message.get("reasoning_content")
                passed = re.search(case["expected_regex"], text, flags=re.IGNORECASE) is not None
                item = {
                    "policy": policy_name,
                    "case": case["name"],
                    "repeat": repeat,
                    "expected_regex": case["expected_regex"],
                    "passed": passed,
                    "http_status": status,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "finish_reason": choice.get("finish_reason"),
                    "text": text,
                    "reasoning_content": reasoning,
                    "think_tag_seen": "<think" in text.casefold(),
                    "usage": response.get("usage"),
                    "timings": response.get("timings"),
                    "request": payload,
                    "response": response,
                }
                results.append(item)
                print(
                    f"{policy_name}/{case['name']}/r{repeat}: "
                    f"pass={passed} elapsed_ms={item['elapsed_ms']} text={text!r}"
                )

    summaries = {}
    for policy_name in POLICIES:
        subset = [item for item in results if item["policy"] == policy_name]
        summaries[policy_name] = {
            "passed": sum(bool(item["passed"]) for item in subset),
            "total": len(subset),
            "all_finish_reason_stop": all(item["finish_reason"] == "stop" for item in subset),
            "reasoning_exposed": any(
                item["think_tag_seen"]
                or (isinstance(item["reasoning_content"], str) and item["reasoning_content"].strip())
                for item in subset
            ),
        }
    output = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": MODEL,
        "repeats": args.repeats,
        "policies": POLICIES,
        "case_definitions": CASES,
        "summaries": summaries,
        "results": results,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
