"""The same token segments are used by training and the persistent KV server."""

import json
from pathlib import Path

SYSTEM = "你是一个有帮助的中文语音助手。请根据对话内容直接、简洁、准确地回答用户的问题。"
PROMPT_VERSION = "teacher_explicit_roles_v1"


def base_prompt(system=SYSTEM):
    return (
        "下面是一段用户和助手之间的真实对话。\n"
        "你现在扮演“助手”。\n"
        "请站在助手的角色回答最后一个用户问题。\n"
        "注意区分用户说的“我”和对助手说的“你”。\n"
        "只回答问题，不要继续编写新的用户对话。\n\n"
        f"系统：{system}\n"
    )


def user_segment(text):
    return "\n用户：" + text + "\n助手回答："


def assistant_action(message):
    if message.get("tool_calls"):
        return json.dumps(
            {"tool_calls": [call["function"] for call in message["tool_calls"]]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    return message["content"]


def prompt_segments(messages, tools=None):
    """History excludes the current assistant target and contains no history EOS."""
    instructions = [SYSTEM] + [m["content"] for m in messages if m["role"] == "system"]
    if tools:
        instructions.append(
            "需要调用工具时，只输出 JSON，格式为 {\"tool_calls\":[{\"name\":工具名,"
            "\"arguments\":参数对象}]}。收到工具结果后再简洁播报，失败时说明失败。"
            "可用工具：" + json.dumps(tools, ensure_ascii=False, sort_keys=True)
        )
    segments = [base_prompt("\n".join(instructions))]
    for message in messages:
        role = message["role"]
        if role == "user":
            segments.append(user_segment(message["content"]))
        elif role == "assistant":
            segments.append(assistant_action(message))
        elif role == "tool":
            segments.append("\n工具结果（" + message["name"] + "）：" + message["content"] + "\n助手回答：")
        elif role != "system":
            raise ValueError(f"Unsupported role: {role}")
    return segments


def encode_segments(tokenizer, segments):
    return [token for segment in segments for token in tokenizer.encode(segment, add_special_tokens=False)]


def serving_base_prompt(system_file=None):
    """Load the optional system message using the exact training serializer."""
    if not system_file:
        return base_prompt()
    system = Path(system_file).read_text().strip()
    if not system:
        raise ValueError('Serving system prompt is empty')
    return prompt_segments([dict(role='system', content=system)])[0]
