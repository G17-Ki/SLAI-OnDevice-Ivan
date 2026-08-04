#!/usr/bin/env python3
"""Probe a deployed Qwen Android text companion without third-party packages."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import (  # noqa: E402
    build_chat_payload,
    extract_completion,
    iter_sse_text,
    response_exposes_reasoning,
)


def request(url: str, *, data: bytes | None, timeout: float, accept: str) -> urllib.response.addinfourl:
    headers = {"Accept": accept, "Connection": "close"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=timeout
    )


def health(base_url: str, timeout: float) -> int:
    started = time.perf_counter()
    with request(
        base_url.rstrip("/") + "/health",
        data=None,
        timeout=timeout,
        accept="application/json",
    ) as response:
        health_body = response.read().decode("utf-8")
        health_status = response.status
    with request(
        base_url.rstrip("/") + "/v1/models",
        data=None,
        timeout=timeout,
        accept="application/json",
    ) as response:
        models_body = response.read().decode("utf-8")
        models_status = response.status
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        health_payload = json.loads(health_body)
    except json.JSONDecodeError:
        health_payload = {"raw": health_body}
    try:
        models_payload = json.loads(models_body)
    except json.JSONDecodeError:
        models_payload = {"raw": models_body}
    model_ids = [
        item.get("id")
        for item in models_payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ] if isinstance(models_payload, dict) else []
    identity_ok = "Qwen3-1.7B-Q4_0" in model_ids
    print(
        json.dumps(
            {
                "health_http_status": health_status,
                "models_http_status": models_status,
                "elapsed_ms": round(elapsed_ms, 1),
                "model_identity_ok": identity_ok,
                "expected_model": "Qwen3-1.7B-Q4_0",
                "model_ids": model_ids,
                "health": health_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if health_status == 200 and models_status == 200 and identity_ok else 1


def chat(args: argparse.Namespace, *, stream: bool) -> int:
    payload = build_chat_payload(
        args.prompt,
        language=args.language,
        max_tokens=args.max_tokens,
        stream=stream,
    )
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    with request(
        url,
        data=encoded,
        timeout=args.timeout,
        accept="text/event-stream" if stream else "application/json",
    ) as response:
        if stream:
            first_text_ms: float | None = None
            for text in iter_sse_text(response):
                if first_text_ms is None:
                    first_text_ms = (time.perf_counter() - started) * 1000
                print(text, end="", flush=True)
            total_ms = (time.perf_counter() - started) * 1000
            print()
            print(
                json.dumps(
                    {
                        "http_status": response.status,
                        "first_text_ms": None if first_text_ms is None else round(first_text_ms, 1),
                        "total_ms": round(total_ms, 1),
                    }
                ),
                file=sys.stderr,
            )
            return 0 if response.status == 200 and first_text_ms is not None else 1
        body = json.load(response)
    elapsed_ms = (time.perf_counter() - started) * 1000
    text = extract_completion(body)
    print(text)
    print(
        json.dumps(
            {
                "http_status": response.status,
                "elapsed_ms": round(elapsed_ms, 1),
                "reasoning_exposed": response_exposes_reasoning(body),
                "usage": body.get("usage"),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0 if response.status == 200 and not response_exposes_reasoning(body) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:6020")
    parser.add_argument("--timeout", type=float, default=90.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")
    for command in ("chat", "stream"):
        child = subparsers.add_parser(command)
        child.add_argument("--prompt", required=True)
        child.add_argument("--language", choices=("auto", "en", "zh", "ru"), default="auto")
        child.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "health":
            return health(args.base_url, args.timeout)
        return chat(args, stream=args.command == "stream")
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        print(f"probe.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
