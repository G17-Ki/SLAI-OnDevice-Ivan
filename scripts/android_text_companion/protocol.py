"""Pure OpenAI-compatible protocol helpers for the text companion."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


MODEL_NAME = "Qwen3-1.7B-Q4_0"
DEFAULT_SYSTEM_PROMPT = (
    "Answer accurately and concisely in the same language as the user. "
    "Give the answer before optional detail. Do not show reasoning."
)
LANGUAGE_PROMPTS = {
    "auto": DEFAULT_SYSTEM_PROMPT,
    "en": "Answer accurately and concisely in English. Do not show reasoning.",
    "zh": "请用简体中文准确、简洁地回答。先给出答案，再给必要说明。不要展示思考过程。",
    "ru": "Отвечай точно и кратко по-русски. Сначала дай ответ, затем пояснение. Не показывай ход рассуждений.",
}


def build_chat_payload(
    prompt: str,
    *,
    language: str = "auto",
    max_tokens: int = 128,
    stream: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    if language not in LANGUAGE_PROMPTS:
        raise ValueError(f"unsupported language: {language}")
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 768:
        raise ValueError("max_tokens must be between 1 and 768")
    system = system_prompt if system_prompt is not None else LANGUAGE_PROMPTS[language]
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system prompt must be non-empty text")
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.08,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract_completion(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response lacks choices[0].message.content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("completion content is empty or non-text")
    return content


def response_exposes_reasoning(response: dict[str, Any]) -> bool:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False
    if not isinstance(message, dict):
        return False
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    return bool(
        (isinstance(reasoning, str) and reasoning.strip())
        or (isinstance(content, str) and "<think>" in content.lower())
    )


def iter_sse_json(lines: Iterable[bytes | str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from OpenAI-style SSE lines until ``[DONE]``."""

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            value = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid SSE JSON: {data[:120]}") from error
        if not isinstance(value, dict):
            raise ValueError("SSE data must decode to an object")
        yield value


def iter_sse_text(lines: Iterable[bytes | str]) -> Iterator[str]:
    for event in iter_sse_json(lines):
        try:
            content = event["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(content, str) and content:
            yield content
