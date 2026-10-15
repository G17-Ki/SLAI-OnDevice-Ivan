#!/usr/bin/env python3
"""Sweep the 28-case multilingual suite against the app-owned raw-prompt LLM path.

The production app talks to the runtime companion over
``POST /v1/completions`` with a single raw ``prompt`` string built by
``buildQwen3AsrPrompt`` (dialogue template). This driver sends the identical
prompt through an ``adb forward``ed port so the sweep measures the exact
on-phone protocol, not the chat-completions protocol the Qwen2.5 baseline used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import profile_text_llm_streaming as base

INDEX_HTML = (
    Path(__file__).resolve().parent.parent
    / "android-app/app/src/main/assets/demo-ui/index.html"
)

# Byte-exact copy of the app template (android-app/app/src/main/assets/demo-ui/index.html),
# kept as the individual JS source fragments. verify_template() re-checks every
# fragment against the live file on each run, so any drift fails loudly.
QWEN3ASR_SYSTEM = (
    "你是一个有帮助的中文语音助手。请根据对话内容直接、简洁、准确地回答用户的问题。"
)
BASE_FRAGMENTS = (
    "下面是一段用户和助手之间的真实对话。\n",
    "你现在扮演“助手”。\n",
    "请站在助手的角色回答最后一个用户问题。\n",
    "注意区分用户说的“我”和对助手说的“你”。\n",
    "只回答问题，不要继续编写新的用户对话。\n\n",
    "系统：",
    "\n",
)
USER_PREFIX = "\n用户："
TURN_SUFFIX = "\n助手回答："


def verify_template() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    for fragment in (QWEN3ASR_SYSTEM, *BASE_FRAGMENTS, USER_PREFIX, TURN_SUFFIX):
        literal = json.dumps(fragment, ensure_ascii=False)
        if literal not in html:
            raise SystemExit(
                "app prompt template drifted; update the fragments in this script:\n"
                + fragment[:60]
            )


def build_prompt(user_text: str) -> str:
    base = "".join(BASE_FRAGMENTS[:6]) + QWEN3ASR_SYSTEM + BASE_FRAGMENTS[6]
    return base + USER_PREFIX + user_text + TURN_SUFFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:16020/v1/completions")
    parser.add_argument("--model", default="Qwen3-ASR-0.6B-Text-Q4_K_M")
    parser.add_argument("--runs", type=int, default=1, help="Measured repetitions per prompt")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=768,
                        help="Matches RESPONSE_TOKEN_LIMITS in the app.")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).resolve().parent / "multilingual_voice_quality_suite.json",
    )
    parser.add_argument(
        "--flush-chars",
        type=int,
        default=64,
        help="Mark text speakable at terminal punctuation or after this many visible characters.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-prompt",
        type=int,
        choices=(0, 1),
        default=1,
        help="Send cache_prompt (1 = app-faithful, 0 = A/B stability test).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Pause between requests in seconds (app turns are seconds apart).",
    )
    parser.add_argument(
        "--app-warmup",
        action="store_true",
        help="Before each case, send the app-style warmup (same prompt, stream=false, max_tokens=1).",
    )
    return parser.parse_args()


def health_base(url: str) -> str:
    return url.rsplit("/", 1)[0]


def server_alive(args: argparse.Namespace) -> bool:
    try:
        with urllib.request.urlopen(health_base(args.url) + "/health", timeout=5) as probe:
            return probe.status == 200
    except Exception:
        return False


def app_warmup(args: argparse.Namespace, case: dict) -> None:
    """Mimic the app's per-case warmTextModelForLanguage: same prompt, no stream, 1 token."""
    body = {
        "model": args.model,
        "stream": False,
        "cache_prompt": bool(args.cache_prompt),
        "prompt": build_prompt(case["prompt"]),
        "temperature": 0,
        "top_k": 1,
        "max_tokens": 1,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()
    except Exception as exc:
        print(f"  warmup error: {type(exc).__name__}: {exc}", flush=True)


def request_once(args: argparse.Namespace, case: dict) -> dict:
    prompt = build_prompt(case["prompt"])
    body = {
        "model": args.model,
        "stream": True,
        "cache_prompt": bool(args.cache_prompt),
        "prompt": prompt,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    first_visible_token_at: float | None = None
    first_speakable_at: float | None = None
    last_content_at: float | None = None
    chunks: list[str] = []
    usage: dict | None = None
    timings: dict | None = None
    finish_reason: str | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choice = (event.get("choices") or [{}])[0]
                # /v1/chat/completions streams choices[0].delta.content;
                # /v1/completions streams choices[0].text (object text_completion).
                content = choice.get("text") or choice.get("delta", {}).get("content")
                if content:
                    last_content_at = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = last_content_at
                    chunks.append(content)
                    visible = base.visible_response("".join(chunks))
                    if first_visible_token_at is None and visible:
                        first_visible_token_at = last_content_at
                    if first_speakable_at is None and base.is_speakable(visible, args.flush_chars):
                        first_speakable_at = last_content_at
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("timings"):
                    timings = event["timings"]
    except Exception as exc:  # server hang / reset: record, don't kill the sweep
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    response_text = "".join(chunks).strip()
    visible_text = base.visible_response(response_text)
    if first_speakable_at is None and visible_text:
        first_speakable_at = last_content_at or ended
    return {
        "first_token_ms": None if first_token_at is None else (first_token_at - started) * 1000,
        "first_visible_token_ms": (
            None if first_visible_token_at is None else (first_visible_token_at - started) * 1000
        ),
        "first_speakable_ms": (
            None if first_speakable_at is None else (first_speakable_at - started) * 1000
        ),
        "total_ms": (ended - started) * 1000,
        "finish_reason": finish_reason,
        "error": error,
        "response": response_text,
        "response_visible": visible_text,
        "pass": base.score(case, visible_text) if error is None else False,
        "usage": usage,
        "timings": timings,
    }


def main() -> int:
    args = parse_args()
    verify_template()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be positive and --warmups cannot be negative")
    suite = base.load_suite(args.suite)
    results: list[dict] = []
    total_repetitions = args.warmups + args.runs
    aborted = False
    for case in suite:
        for repetition in range(total_repetitions):
            measured = repetition >= args.warmups
            if args.app_warmup and measured:
                app_warmup(args, case)
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
                f"pass={result['pass']} response={result['response_visible']!r} "
                f"error={result['error']!r}",
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "incomplete": True,
                        "configuration": {"url": args.url, "model": args.model},
                        "results": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            if result["error"] and not server_alive(args):
                print("server unhealthy after error; aborting remaining cases", flush=True)
                aborted = True
                break
            if args.sleep > 0:
                time.sleep(args.sleep)
        if aborted:
            break

    measured = [record for record in results if record["measured"]]
    first_token = [float(r["first_token_ms"]) for r in measured if r["first_token_ms"]]
    first_visible = [float(r["first_visible_token_ms"]) for r in measured if r["first_visible_token_ms"]]
    first_speakable = [float(r["first_speakable_ms"]) for r in measured if r["first_speakable_ms"]]
    total = [float(r["total_ms"]) for r in measured]
    prompt_rates = [
        float(r["timings"]["prompt_per_second"])
        for r in measured
        if isinstance(r.get("timings"), dict) and r["timings"].get("prompt_per_second") is not None
    ]
    generation_rates = [
        float(r["timings"]["predicted_per_second"])
        for r in measured
        if isinstance(r.get("timings"), dict) and r["timings"].get("predicted_per_second") is not None
    ]
    per_case: dict[str, dict] = {}
    per_language: dict[str, dict] = {}
    for case in suite:
        rows = [r for r in measured if r["case_id"] == case["id"]]
        per_case[case["id"]] = {
            "language": case["language"],
            "category": case.get("category"),
            "passes": sum(r["pass"] for r in rows),
            "runs": len(rows),
            "responses": [r["response"] for r in rows],
        }
        lang = per_language.setdefault(case["language"], {"passes": 0, "runs": 0})
        lang["passes"] += sum(r["pass"] for r in rows)
        lang["runs"] += len(rows)
    document = {
        "schema_version": 1,
        "incomplete": False,
        "aborted": aborted,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warning": (
            "Raw-prompt sweep of the app-owned Qwen3-ASR-0.6B text decoder. Protocol differs from "
            "the Qwen2.5 chat-completions baseline (dialogue template, no system role); per-case "
            "pass counts remain comparable, timings are not."
        ),
        "configuration": {
            "url": args.url,
            "model": args.model,
            "protocol": "raw_prompt /v1/completions (buildQwen3AsrPrompt, temperature=0 top_k=1 cache_prompt=true)",
            "max_tokens": args.max_tokens,
            "cache_prompt": bool(args.cache_prompt),
            "flush_chars": args.flush_chars,
            "case_count": len(suite),
            "measured_repetitions_per_case": args.runs,
            "warmups_per_case": args.warmups,
            "suite": str(args.suite),
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "index_html_sha256": hashlib.sha256(INDEX_HTML.read_bytes()).hexdigest(),
        },
        "summary": {
            "passes": sum(r["pass"] for r in measured),
            "runs": len(measured),
            "accuracy": sum(r["pass"] for r in measured) / len(measured),
            "per_language": per_language,
            "first_token_ms": base.describe(first_token) if first_token else None,
            "first_visible_token_ms": base.describe(first_visible) if first_visible else None,
            "first_speakable_ms": base.describe(first_speakable) if first_speakable else None,
            "total_ms": base.describe(total) if total else None,
            "prompt_tokens_per_second": base.describe(prompt_rates) if prompt_rates else None,
            "generation_tokens_per_second": (
                base.describe(generation_rates) if generation_rates else None
            ),
            "per_case": per_case,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    print(json.dumps(document["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
