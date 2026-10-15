#!/usr/bin/env python3
"""Probe a llama.cpp server (/v1/completions) the way the phone main app talks
to it: raw prompt strings, phone sampling params (temperature 0.2, top_p 0.8,
repeat_penalty 1.08), cache_prompt on.

Purpose: quantify, for the Qwen3-ASR text-decoder checkpoints, how often the
dialogue branch stays clean vs falls into the ASR target format
('language X<asr_text>...') under the phone runtime, and whether a prefix
strip recovers usable text.

Usage:
    python llama_server_probe.py --server http://127.0.0.1:18999
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dialogue_prompt import prompt_segments  # noqa: E402

D = lambda r, c: {"role": r, "content": c}

SINGLE_TURN = [
    "17乘以23等于多少？",
    "珠穆朗玛峰有多高？",
    "用一句话解释什么是通货膨胀",
    "今天适合洗车吗？",
    "中国的首都哪里？",
    "苹果是什么颜色的？",
    "把“你好”翻译成英文",
    "1加1等于几？",
    "地球是圆的吗？",
    "用三个词形容大海",
]

MULTI_TURN = [
    ([D("user", "中国的首都哪里？"), D("assistant", "中国的首都是北京。")], "那上海呢？"),
    ([D("user", "我给你说三个数字"), D("assistant", "好的，请说。"),
     D("user", "一、三、五"), D("assistant", "收到，是一三五。"),
     D("user", "它们加起来是多少？"), D("assistant", "一共是九。")], "第三个数字呢？"),
    ([D("user", "帮我记一下，明天下午三点开会"), D("assistant", "好的，明天下午三点开会。"),
     D("user", "对了，参会人员有张三和李四"), D("assistant", "记下了，张三和李四。"),
     D("user", "还有王五"), D("assistant", "收到，加上王五。"),
     D("user", "一共几个人？"), D("assistant", "一共三个人。")], "明天几点开会？"),
]

ASR_PREFIX = re.compile(r"^language\s+(?:Chinese|English|None)?\s*<asr_text>\s*")
REPEAT_MARKERS = ("用户：", "助手回答：", "工具结果")


def post(server, payload):
    req = urllib.request.Request(
        server + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def gen(server, prompt, **sampling):
    payload = {
        "model": "qwen3-asr-text",
        "prompt": prompt,
        "max_tokens": 128,
        "stream": False,
        "cache_prompt": True,
        **sampling,
    }
    t0 = time.time()
    resp = post(server, payload)
    text = resp["choices"][0]["text"].strip()
    return text, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    args = ap.parse_args()

    phone = {"temperature": 0.2, "top_p": 0.8, "repeat_penalty": 1.08}
    greedy = {"temperature": 0, "top_k": 1}

    for label, sampling in [("phone", phone), ("greedy", greedy)]:
        print(f"===== sampling: {label} ({sampling}) =====")
        clean = asr = bad = 0
        rows = []
        for q in SINGLE_TURN:
            prompt = "".join(prompt_segments([D("user", q)]))
            text, dt = gen(args.server, prompt, **sampling)
            stripped = ASR_PREFIX.sub("", text)
            is_asr = text != stripped
            repeats = any(m in stripped for m in REPEAT_MARKERS)
            ok = (not is_asr) or (stripped and not repeats)
            clean += not is_asr
            asr += is_asr
            bad += is_asr and not ok
            rows.append((q, text, ok, dt))
        for history, q in MULTI_TURN:
            prompt = "".join(prompt_segments(history + [D("user", q)]))
            text, dt = gen(args.server, prompt, **sampling)
            stripped = ASR_PREFIX.sub("", text)
            is_asr = text != stripped
            repeats = any(m in stripped for m in REPEAT_MARKERS)
            ok = (not is_asr) or (stripped and not repeats)
            clean += not is_asr
            asr += is_asr
            bad += is_asr and not ok
            rows.append((q, text, ok, dt))
        n = len(SINGLE_TURN) + len(MULTI_TURN)
        print(f"  raw clean: {clean}/{n}   ASR-format: {asr}/{n}   "
              f"ASR+repeat: {bad}/{n}")
        print(f"  usable after prefix strip: {sum(1 for (_, _, ok, _) in rows if ok)}/{n}")
        for q, text, ok, dt in rows:
            print(f"  [{'ok ' if ok else 'BAD'} {dt:4.1f}s] {q[:14]:<16} -> {text[:58]!r}")
        print()


if __name__ == "__main__":
    main()
