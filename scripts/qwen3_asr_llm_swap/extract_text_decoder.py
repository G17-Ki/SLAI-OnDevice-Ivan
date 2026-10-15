#!/usr/bin/env python3
"""Extract the text decoder of the full Qwen3-ASR-0.6B export into a standard
Qwen3 HF directory, ready for convert_hf_to_gguf.py.

Source layout (verified 2026-10-15):
    model.safetensors: 611 tensors
      thinker.audio_tower.*  301 tensors (dropped)
      thinker.model.*        310 tensors = embed_tokens(1) + 28 layers x 11 + norm(1)
    No separate lm_head tensor (tied embeddings).

This script is stdlib-only (struct+json) so it runs on any Python, including
hosts without torch. It copies raw tensor bytes and rewrites the safetensors
header with the `thinker.` prefix stripped, yielding exact Qwen3 naming
(model.embed_tokens.weight, model.layers.N.*, model.norm.weight).

Usage:
    python3 extract_text_decoder.py \
        --model-dir /path/to/qwen3_asr_0p6b_spoken_multiturn_90h_20261014 \
        --out-dir /path/to/qwen3_asr_0p6b_text_q4_source
"""

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

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


def read_safetensors(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data = f.read()
    return header, data, len(data)


def write_safetensors(path, tensors, blob, metadata=None):
    """tensors: name -> (dtype, shape). blob: concatenated raw bytes, in order."""
    itemsize = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "F64": 8}
    sizes = {}
    total = 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        sizes[name] = n * itemsize[dtype]
        total += n * itemsize[dtype]
    assert total == len(blob), f"data size mismatch: {total} vs {len(blob)}"
    header_out = {}
    if metadata:
        header_out["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape) in tensors.items():
        header_out[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + sizes[name]],
        }
        offset += sizes[name]
    header_bytes = json.dumps(header_out, separators=(",", ":")).encode("utf-8")
    while len(header_bytes) % 8:
        header_bytes += b" "
    out = struct.pack("<Q", len(header_bytes)) + header_bytes + blob
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)
    return len(header_out)


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
    header, data, _ = read_safetensors(src)
    tensors_all = {k: v for k, v in header.items() if k != "__metadata__"}

    picked = {}
    for name, meta in tensors_all.items():
        if name.startswith(PREFIX + "model."):
            new_name = name[len(PREFIX):]
            picked[new_name] = (meta["dtype"], meta["shape"], meta["data_offsets"])

    # ---- assertions on the extracted set ----
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
    assert all(t[0] == "BF16" for t in picked.values()), "unexpected dtype"

    vocab, hidden = picked["model.embed_tokens.weight"][1]
    inter = picked["model.layers.0.mlp.gate_proj.weight"][1][0]
    q_dim = picked["model.layers.0.self_attn.q_proj.weight"][1][0]
    k_dim = picked["model.layers.0.self_attn.k_proj.weight"][1][0]
    head_dim = picked["model.layers.0.self_attn.q_norm.weight"][1][0]
    n_heads, n_kv = q_dim // head_dim, k_dim // head_dim

    # ---- write model.safetensors (raw byte copy of kept tensors, remapped header) ----
    blob_parts = []
    meta_only = {}
    for name, (dtype, shape, (s, e)) in picked.items():
        blob_parts.append(data[s:e])
        meta_only[name] = (dtype, shape)
    blob = b"".join(blob_parts)
    n_written = write_safetensors(out_dir / "model.safetensors", meta_only, blob)
    assert n_written == 310

    # ---- config.json (Qwen3-0.6B text-only) ----
    src_cfg = json.loads((model_dir / "config.json").read_text())

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
    # Official Qwen3-0.6B value. The training export carries 65536 from the
    # multimodal container; using it as-is let llama.cpp's default context
    # allocate ~7.5GB of KV and OOM-killed the WSL2 host on 2026-10-15.
    # Training collator max_length was 2048; the phone passes --ctx-size 2048.
    max_pos = 40960

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

    # ---- tokenizer: copy as-is, drop chat_template (text path never uses it;
    #      removing it makes any accidental template use fail loudly) ----
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
    h2, d2, _ = read_safetensors(out_dir / "model.safetensors")
    t2 = {k: v for k, v in h2.items() if k != "__metadata__"}
    assert len(t2) == 310 and set(t2) == set(meta_only), "round-trip header mismatch"
    off = 0
    for name in meta_only:
        s, e = t2[name]["data_offsets"]
        assert (s, e) == (off, off + (e - s)), f"offset drift at {name}"
        off = e

    print("=== extraction summary ===")
    print(f"out_dir:            {out_dir}")
    print(f"tensors:            {n_written} (all BF16)")
    print(f"arch:               Qwen3ForCausalLM text-only (tied embeddings)")
    print(f"layers/hidden:      {EXPECTED_LAYERS} / {hidden}")
    print(f"heads/kv/head_dim:  {n_heads} / {n_kv} / {head_dim}")
    print(f"intermediate:       {inter}")
    print(f"vocab:              {vocab}")
    print(f"rope_theta:         {rope_theta}")
    print(f"rms_norm_eps:       {rms_eps}")
    print(f"max_position_embeddings: {max_pos}")
    print(f"model.safetensors sha256: {sha256_file(out_dir / 'model.safetensors')}")


if __name__ == "__main__":
    main()
