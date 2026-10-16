#!/usr/bin/env python3
"""Extract the text decoder of a Qwen3-ASR-0.6B export into a standard Qwen3 HF
directory, ready for convert_hf_to_gguf.py.

v2 vs v1 (extract_text_decoder.py):
  - Accepts mixed-precision checkpoints (F32 master + BF16 tensors, as emitted
    by the 2026-10-16 "edge" training runs) and casts every kept tensor to
    BF16. v1 required all-BF16 input and copied raw bytes.
  - Requires torch + safetensors (run with the llm-swap venv python).
  - Same structure assertions and config.json output as v1.

Source layout (verified 2026-10-15 / 2026-10-16):
    model.safetensors: 611 tensors
      thinker.audio_tower.*  301 tensors (dropped)
      thinker.model.*        310 tensors = embed_tokens(1) + 28 layers x 11 + norm(1)
    No separate lm_head tensor (tied embeddings).

Usage:
    LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/torch/lib \
        $VENV/bin/python extract_text_decoder_v2.py \
        --model-dir /path/to/checkpoint \
        --out-dir /path/to/extracted_text_source
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "thinker."
EXPECTED_LAYERS = 28
EXPECTED_PER_LAYER = 11
LAYER_KEYS = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        sys.exit(f"refusing to overwrite existing out dir: {out_dir}")

    src = model_dir / "model.safetensors"

    # ---- pass 1: header scan + assertions ----
    src_cfg = json.loads((model_dir / "config.json").read_text())
    src_sha = sha256_file(src)
    with safe_open(src, framework="pt") as f:
        keys = list(f.keys())
        src_dtypes = Counter(f.get_slice(k).get_dtype() for k in keys
                             if k.startswith(PREFIX + "model."))

    text_keys = [k for k in keys if k.startswith(PREFIX + "model.")]
    picked = {k[len(PREFIX):] for k in text_keys}
    assert len(picked) == 310, f"expected 310 tensors, got {len(picked)}"
    assert "model.embed_tokens.weight" in picked
    assert "model.norm.weight" in picked
    assert not any("audio" in k for k in picked), "audio tensors leaked into extraction"
    assert not any("lm_head" in k for k in picked), "unexpected lm_head (expected tied embeddings)"

    layers = {}
    for k in picked:
        parts = k.split(".")
        if len(parts) >= 3 and parts[1] == "layers":
            idx = int(parts[2])
            layers.setdefault(idx, []).append(k)
    assert len(layers) == EXPECTED_LAYERS, f"expected {EXPECTED_LAYERS} layers, got {len(layers)}"
    for idx in range(EXPECTED_LAYERS):
        got = sorted(k.split("layers.%d." % idx)[1] for k in layers[idx])
        assert got == sorted(LAYER_KEYS), f"layer {idx} keys mismatch: {got}"

    vocab, hidden = None, None
    inter = q_dim = k_dim = head_dim = None
    out_sd = {}
    with safe_open(src, framework="pt") as f:
        for k in text_keys:
            t = f.get_tensor(k).to(torch.bfloat16).contiguous()
            out_sd[k[len(PREFIX):]] = t
        st = f.get_slice("thinker.model.embed_tokens.weight").get_shape()
        vocab, hidden = st[0], st[1]
        inter = f.get_slice("thinker.model.layers.0.mlp.gate_proj.weight").get_shape()[0]
        q_dim = f.get_slice("thinker.model.layers.0.self_attn.q_proj.weight").get_shape()[0]
        k_dim = f.get_slice("thinker.model.layers.0.self_attn.k_proj.weight").get_shape()[0]
        head_dim = f.get_slice("thinker.model.layers.0.self_attn.q_norm.weight").get_shape()[0]
    n_heads, n_kv = q_dim // head_dim, k_dim // head_dim

    out_dir.mkdir(parents=True)
    save_file(out_sd, out_dir / "model.safetensors")
    del out_sd

    # ---- config.json (Qwen3-0.6B text-only; identical to v1 output) ----
    def find_key(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                r = find_key(v, key)
                if r is not None:
                    return r
        return None

    rope_theta = find_key(src_cfg, "rope_theta") or 1000000.0
    rms_eps = find_key(src_cfg, "rms_norm_eps") or 1e-06
    max_pos = 40960  # official Qwen3-0.6B text value (see v1 comment)

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": find_key(src_cfg, "bos_token_id") or 151643,
        "eos_token_id": [151643, 151645],
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "initializer_range": 0.02,
        "intermediate_size": inter,
        "max_position_embeddings": max_pos,
        "model_type": "qwen3",
        "num_attention_heads": n_heads,
        "num_hidden_layers": EXPECTED_LAYERS,
        "num_key_value_heads": n_kv,
        "rms_norm_eps": rms_eps,
        "rope_scaling": None,
        "rope_theta": rope_theta,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.57.0",
        "use_cache": True,
        "vocab_size": vocab,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # ---- tokenizer: copy as-is, drop chat_template ----
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.json", "merges.txt"):
        s = model_dir / fn
        if s.exists():
            shutil.copy2(s, out_dir / fn)
    tcfg_path = out_dir / "tokenizer_config.json"
    if tcfg_path.exists():
        tcfg = json.loads(tcfg_path.read_text())
        dropped = tcfg.pop("chat_template", None)
        tcfg_path.write_text(json.dumps(tcfg, indent=2, ensure_ascii=False) + "\n")
        print(f"tokenizer_config: chat_template {'dropped' if dropped else 'absent (ok)'}")

    (out_dir / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": config["bos_token_id"], "eos_token_id": [151643, 151645],
         "transformers_version": "4.57.0"}, indent=2) + "\n")

    # ---- self-check: re-read written header ----
    with safe_open(out_dir / "model.safetensors", framework="pt") as f:
        h2 = list(f.keys())
    assert len(h2) == 310 and set(h2) == picked, "round-trip header mismatch"
    for name in ("model.embed_tokens.weight", "model.norm.weight"):
        t = None
        with safe_open(out_dir / "model.safetensors", framework="pt") as f:
            t = f.get_tensor(name)
        assert t.dtype == torch.bfloat16, f"{name} not bf16 after write"
        del t

    print(f"source sha256: {src_sha}")
    print(f"source text dtypes: {dict(src_dtypes)}")
    print(f"extracted {len(picked)} tensors -> {out_dir / 'model.safetensors'} "
          f"({(out_dir / 'model.safetensors').stat().st_size} bytes, all bf16)")
    print(f"hidden={hidden} inter={inter} heads={n_heads} kv={n_kv} "
          f"head_dim={head_dim} vocab={vocab} rope_theta={rope_theta}")


if __name__ == "__main__":
    main()
