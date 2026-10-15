#!/usr/bin/env python3
"""Tokenization-parity check for the Qwen3-0.6B text-decoder swap.

A) Segment-wise encoding (training distribution) vs whole-string encoding
   (phone path), pure tokenizer, no model, no torch.
   Verified result (2026-10-15): 1-5 structural-boundary merges per prompt
   (base_prompt|user '\\n\\n' + one per extra history turn), 1-3% of prompt,
   text identical. HF outputs identical under both encodings -> whole-string
   raw prompt is accepted for the phone path.

NOTE (do not resurrect): a llama-cli-based GGUF parity section existed here.
b10083 llama-cli defaults to interactive conversation mode, which mangles the
prompt and makes the dialogue branch fall into the ASR target format
('language X<asr_text>...') 10/10 - a harness artifact. The deployment path
(llama-server /v1/completions, exactly what the phone JS calls) is clean
13/13. Use llama_server_probe.py for model-level checks instead.

Usage:
    python parity_check.py --workdir ~/.cache/lmf-demo/llm-swap
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dialogue_prompt import prompt_segments  # noqa: E402

D = lambda r, c: {"role": r, "content": c}
PROMPTS = [
    ("arith-0", [], "17乘以23等于多少？"),
    ("fact-0", [], "珠穆朗玛峰有多高？"),
    ("followup-1", [D("user", "中国的首都哪里？"), D("assistant", "中国的首都是北京。")], "那上海呢？"),
    ("memory-3", [
        D("user", "我给你说三个数字"), D("assistant", "好的，请说。"),
        D("user", "一、三、五"), D("assistant", "收到，是一三五。"),
        D("user", "它们加起来是多少？"), D("assistant", "一共是九。"),
    ], "第三个数字呢？"),
    ("memory-5", [
        D("user", "帮我记一下，明天下午三点开会"), D("assistant", "好的，明天下午三点开会。"),
        D("user", "对了，参会人员有张三和李四"), D("assistant", "记下了，张三和李四。"),
        D("user", "还有王五"), D("assistant", "收到，加上王五。"),
        D("user", "一共几个人？"), D("assistant", "一共三个人。"),
    ], "明天几点开会？"),
    ("en-0", [], "What is the capital of France?"),
    ("concise-0", [], "用一句话解释什么是通货膨胀"),
    ("nohallucinate-0", [], "我上次告诉你我的生日是哪天？"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    workdir = Path(args.workdir)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(workdir / "out/qwen3_text/tokenizer.json"))
    print("=== A: segment-wise vs whole-string tokenization (no model) ===")
    total = 0
    for name, history, user in PROMPTS:
        segs = prompt_segments(history + [D("user", user)])
        seg_ids = [t for s in segs for t in tok.encode(s, add_special_tokens=False).ids]
        full_ids = tok.encode("".join(segs), add_special_tokens=False).ids
        delta = len(seg_ids) - len(full_ids)
        total += delta
        print(f"  {name:<16} seg={len(seg_ids):<5} full={len(full_ids):<5} "
              f"merges={delta}  user_turns={len(history)//2 + 1}")
    print(f"  total merges: {total} across {len(PROMPTS)} prompts")
    print("  Gate: merges <= number of user turns, structural boundaries only.")
    print("  Model-level checks: see llama_server_probe.py (deployment path).")


if __name__ == "__main__":
    main()
