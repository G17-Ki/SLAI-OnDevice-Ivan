#!/usr/bin/env python3
"""Profile a local OpenAI-compatible text LLM with a tiny multilingual diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


SUITE = (
    {
        "id": "en_fuji",
        "language": "en",
        "prompt": "What is the height of Mount Fuji? Answer with the number in meters.",
        "patterns": (r"\b3[\s,]?776\b",),
    },
    {
        "id": "en_multiply",
        "language": "en",
        "prompt": "What is 17 multiplied by 23? Answer with just the number.",
        "patterns": (r"\b391\b",),
    },
    {
        "id": "en_freezing",
        "language": "en",
        "prompt": "At standard pressure, at what temperature in Celsius does pure water freeze?",
        "patterns": (r"(^|\D)0(?:\s*°?\s*c)?(\D|$)", r"zero degrees"),
    },
    {
        "id": "es_capital",
        "language": "es",
        "prompt": "¿Cuál es la capital de Francia? Responde en una frase corta en español.",
        "patterns": (r"par[ií]s",),
    },
    {
        "id": "de_capital",
        "language": "de",
        "prompt": "Was ist die Hauptstadt von Japan? Antworte in einem kurzen deutschen Satz.",
        "patterns": (r"tokio", r"tokyo"),
    },
    {
        "id": "it_capital",
        "language": "it",
        "prompt": "Qual è la capitale della Germania? Rispondi con una breve frase in italiano.",
        "patterns": (r"berlino", r"berlin"),
    },
    {
        "id": "zh_capital",
        "language": "zh",
        "prompt": "日本的首都是哪里？请用简短的中文句子回答。",
        "patterns": (r"东京", r"東京"),
    },
    {
        "id": "ja_capital",
        "language": "ja",
        "prompt": "フランスの首都はどこですか。日本語の短い一文で答えてください。",
        "patterns": (r"パリ",),
    },
    {
        "id": "exact_instruction",
        "language": "en",
        "prompt": "Reply with exactly READY and nothing else.",
        "exact": "READY",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3-0.6B-Q4_0")
    parser.add_argument("--runs", type=int, default=3, help="Measured repetitions per prompt")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--suite",
        type=Path,
        help="Optional JSON array of cases with id, language, prompt, and patterns or exact.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "Answer accurately and concisely. Use the same language as the user. "
            "Do not show reasoning. If unsure, say that you are unsure rather than inventing a fact."
        ),
    )
    parser.add_argument("--thinking", choices=("off", "on"), default="off")
    parser.add_argument(
        "--flush-chars",
        type=int,
        default=64,
        help="Mark text speakable at terminal punctuation or after this many visible characters.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def score(case: dict[str, Any], response: str) -> bool:
    normalized = response.strip()
    if "exact" in case:
        return normalized == case["exact"]
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in case["patterns"])


def visible_response(response: str) -> str:
    """Remove Qwen-style reasoning wrappers before user-visible scoring or TTS."""
    if "<think>" in response and "</think>" not in response:
        return ""
    visible = re.sub(r"<think>[\s\S]*?</think>", "", response)
    # A finite llama.cpp reasoning budget can inject the closing tag while the
    # model also emits its own, leaving one harmless orphan before the answer.
    return re.sub(r"</?think>", "", visible).strip()


def load_suite(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return SUITE
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit("--suite must contain a non-empty JSON array")
    required = {"id", "language", "prompt"}
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not required.issubset(case):
            raise SystemExit("Every suite case needs id, language, and prompt")
        if case["id"] in seen:
            raise SystemExit(f"Duplicate suite case id: {case['id']}")
        seen.add(case["id"])
        if "exact" not in case and not case.get("patterns"):
            raise SystemExit(f"Suite case {case['id']} needs exact or patterns")
    return tuple(cases)


def is_speakable(text: str, flush_chars: int) -> bool:
    visible = text.strip()
    return bool(visible) and (
        len(visible) >= flush_chars or bool(re.search(r"[.!?。！？:]\s*$", visible))
    )


def model_capabilities(model: str) -> dict[str, bool]:
    normalized = model.casefold().replace("-", "").replace("_", "")
    return {
        # Qwen2.5 has no thinking-mode chat-template switch. Sending Qwen3's
        # template kwargs or slash directives changes the tested prompt and can
        # make a model sweep compare protocols rather than models.
        "qwen3_thinking_controls": "qwen3" in normalized,
    }


def build_request_body(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    capabilities = model_capabilities(args.model)
    if args.thinking == "on" and not capabilities["qwen3_thinking_controls"]:
        raise ValueError(f"{args.model} does not support Qwen3 thinking controls")
    user_content = case["prompt"]
    if capabilities["qwen3_thinking_controls"]:
        user_content += " /think" if args.thinking == "on" else " /no_think"
    body = {
        "model": args.model,
        "stream": True,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [
            {
                "role": "system",
                "content": args.system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }
    if capabilities["qwen3_thinking_controls"]:
        body["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    return body


def request_once(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    body = build_request_body(args, case)
    request = urllib.request.Request(
        args.url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    first_visible_token_at: float | None = None
    first_speakable_at: float | None = None
    last_content_at: float | None = None
    chunks: list[str] = []
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                last_content_at = time.perf_counter()
                if first_token_at is None:
                    first_token_at = last_content_at
                chunks.append(content)
                visible = visible_response("".join(chunks))
                if first_visible_token_at is None and visible:
                    first_visible_token_at = last_content_at
                if first_speakable_at is None and is_speakable(visible, args.flush_chars):
                    first_speakable_at = last_content_at
            if event.get("usage"):
                usage = event["usage"]
            if event.get("timings"):
                timings = event["timings"]
    ended = time.perf_counter()
    response_text = "".join(chunks).strip()
    visible_text = visible_response(response_text)
    if first_speakable_at is None and visible_text:
        first_speakable_at = last_content_at or ended
    return {
        "first_token_ms": None if first_token_at is None else (first_token_at - started) * 1000,
        "first_visible_token_ms": (
            None
            if first_visible_token_at is None
            else (first_visible_token_at - started) * 1000
        ),
        "first_speakable_ms": (
            None if first_speakable_at is None else (first_speakable_at - started) * 1000
        ),
        "total_ms": (ended - started) * 1000,
        "response": response_text,
        "response_visible": visible_text,
        "pass": score(case, visible_text),
        "usage": usage,
        "timings": timings,
    }


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be positive and --warmups cannot be negative")
    results: list[dict[str, Any]] = []
    suite = load_suite(args.suite)
    total_repetitions = args.warmups + args.runs
    for case in suite:
        for repetition in range(total_repetitions):
            measured = repetition >= args.warmups
            result = request_once(args, case)
            record = {
                "case_id": case["id"],
                "language": case["language"],
                "prompt": case["prompt"],
                "repetition": repetition,
                "measured": measured,
                **result,
            }
            results.append(record)
            phase = "measure" if measured else "warmup"
            print(
                f"{case['id']} {phase} {repetition + 1}/{total_repetitions}: "
                f"first={result['first_visible_token_ms']!s} ms total={result['total_ms']:.1f} ms "
                f"pass={result['pass']} response={result['response_visible']!r}",
                flush=True,
            )

    measured = [record for record in results if record["measured"]]
    first_token = [float(record["first_token_ms"]) for record in measured if record["first_token_ms"]]
    first_visible_token = [
        float(record["first_visible_token_ms"])
        for record in measured
        if record["first_visible_token_ms"]
    ]
    total = [float(record["total_ms"]) for record in measured]
    prompt_rates = [
        float(record["timings"]["prompt_per_second"])
        for record in measured
        if isinstance(record.get("timings"), dict)
        and record["timings"].get("prompt_per_second") is not None
    ]
    generation_rates = [
        float(record["timings"]["predicted_per_second"])
        for record in measured
        if isinstance(record.get("timings"), dict)
        and record["timings"].get("predicted_per_second") is not None
    ]
    per_case = {}
    for case in suite:
        rows = [record for record in measured if record["case_id"] == case["id"]]
        per_case[case["id"]] = {
            "language": case["language"],
            "category": case.get("category"),
            "passes": sum(record["pass"] for record in rows),
            "runs": len(rows),
            "responses": [record["response"] for record in rows],
            "visible_responses": [record["response_visible"] for record in rows],
        }
    document = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warning": (
            "This is a small deterministic diagnostic, not a statistically meaningful language or "
            "knowledge benchmark. Repeated temperature-zero responses measure timing stability, "
            "not independent quality samples."
        ),
        "configuration": {
            "url": args.url,
            "model": args.model,
            "thinking": args.thinking,
            "model_capabilities": model_capabilities(args.model),
            "system_prompt": args.system_prompt,
            "max_tokens": args.max_tokens,
            "flush_chars": args.flush_chars,
            "case_count": len(suite),
            "measured_repetitions_per_case": args.runs,
            "warmups_per_case": args.warmups,
            "suite": str(args.suite) if args.suite else "built-in tiny suite",
            "suite_sha256": (
                hashlib.sha256(args.suite.read_bytes()).hexdigest() if args.suite else None
            ),
        },
        "summary": {
            "passes": sum(record["pass"] for record in measured),
            "runs": len(measured),
            "accuracy": sum(record["pass"] for record in measured) / len(measured),
            "first_token_ms": describe(first_token),
            "first_visible_token_ms": describe(first_visible_token),
            "first_speakable_ms": describe(
                [
                    float(record["first_speakable_ms"])
                    for record in measured
                    if record["first_speakable_ms"] is not None
                ]
            ),
            "total_ms": describe(total),
            "prompt_tokens_per_second": describe(prompt_rates) if prompt_rates else None,
            "generation_tokens_per_second": (
                describe(generation_rates) if generation_rates else None
            ),
            "per_case": per_case,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps(document["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
