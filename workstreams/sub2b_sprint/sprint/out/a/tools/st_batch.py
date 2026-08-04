"""safetensors header parse + tie detection for every transformer component.

Tie detection: if lm_head.weight and *.embed_tokens.weight have identical shape
and dtype, byte-compare N blocks spread across both data regions. Identical at
every probe => TIED => count once. This is what HF's safetensors.total does NOT
do, and it is worth 155,582,464 params on a Qwen-class vocab.
"""
import hashlib
import json
import sys

import net

DT = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2,
      "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
      "I64": 8, "U64": 8, "F64": 8}
NON_LEARNED = {"I64", "U64", "I32", "U32", "I16", "U16", "I8", "U8", "BOOL"}

MODELS = [
    ("qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B",
     "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", ["model.safetensors"]),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "main", ["model.safetensors"]),
    ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "main",
     ["model.safetensors-00001-of-00001.safetensors"]),
    ("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct",
     "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", ["model.safetensors"]),
    ("falcon-h1-1.5b-instruct", "tiiuae/Falcon-H1-1.5B-Instruct",
     "80ebc50d7799a440b96c93bb6686a3924a09b0cb", ["model.safetensors"]),
    ("falcon-h1-1.5b-deep-instruct", "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
     "main", ["model.safetensors"]),
]


def prod(s):
    p = 1
    for x in s:
        p *= x
    return p


def probe_tie(url, hdr, hlen, a, b, nblocks=17, blk=32768):
    base = 8 + hlen
    (sa, ea), (sb, eb) = hdr[a]["data_offsets"], hdr[b]["data_offsets"]
    L = ea - sa
    if L != eb - sb:
        return {"identical": False, "reason": "different region length"}
    ha, hb = hashlib.sha256(), hashlib.sha256()
    eq = 0
    diff = 0
    for i in range(nblocks):
        off = (L - blk) * i // max(1, nblocks - 1)
        x = net.rng(url, base + sa + off, base + sa + off + blk - 1)
        y = net.rng(url, base + sb + off, base + sb + off + blk - 1)
        ha.update(x)
        hb.update(y)
        if x == y:
            eq += 1
        else:
            diff += 1
    return {"identical": diff == 0, "blocks_equal": eq, "blocks_differ": diff,
            "blocks": nblocks, "block_bytes": blk,
            "bytes_compared": nblocks * blk * 2,
            "sampled_sha_a": ha.hexdigest(), "sampled_sha_b": hb.hexdigest(),
            "region_bytes": L, "tensor_a": a, "tensor_b": b,
            "shape": hdr[a]["shape"], "dtype": hdr[a]["dtype"],
            "params_at_stake": prod(hdr[a]["shape"])}


def one(key, repo, rev, files):
    merged = {}
    hlen0 = None
    url0 = None
    for f in files:
        h, meta, n, u = net.st(repo, f, rev)
        if hlen0 is None:
            hlen0, url0 = n, u
        for k, v in h.items():
            merged[k] = dict(v, _file=f)
    learned = nonlearned = 0
    by_dt = {}
    for k, v in merged.items():
        p = prod(v["shape"])
        by_dt[v["dtype"]] = by_dt.get(v["dtype"], 0) + p
        if v["dtype"] in NON_LEARNED:
            nonlearned += p
        else:
            learned += p
    rec = {"repo": repo, "revision": rev, "files": files,
           "n_tensors": len(merged), "serialized_total": learned + nonlearned,
           "learned": learned, "non_learned": nonlearned, "by_dtype": by_dt,
           "header_len": hlen0}
    lm = [k for k in merged if k.endswith("lm_head.weight")]
    emb = [k for k in merged if k.endswith("embed_tokens.weight")]
    rec["lm_head_tensors"] = lm
    rec["embed_tensors"] = emb
    if lm and emb and merged[lm[0]]["shape"] == merged[emb[0]]["shape"] \
            and len(files) == 1:
        rec["tie_probe"] = probe_tie(url0, merged, hlen0, lm[0], emb[0])
        if rec["tie_probe"]["identical"]:
            rec["tie_status"] = "TIED"
            rec["unique_total"] = rec["serialized_total"] - prod(merged[lm[0]]["shape"])
        else:
            rec["tie_status"] = "UNTIED"
            rec["unique_total"] = rec["serialized_total"]
    elif lm and not emb:
        rec["tie_status"] = "LM_HEAD_ONLY"
        rec["unique_total"] = rec["serialized_total"]
    elif emb and not lm:
        rec["tie_status"] = "TIED_NOT_MATERIALISED (no lm_head tensor on disk)"
        rec["unique_total"] = rec["serialized_total"]
    else:
        rec["tie_status"] = "N/A"
        rec["unique_total"] = rec["serialized_total"]
    try:
        cfg = net.get(net.resolve(repo, "config.json", rev)).json()
        rec["config_tie_word_embeddings"] = cfg.get("tie_word_embeddings")
        rec["config_vocab_size"] = cfg.get("vocab_size")
        rec["config_hidden_size"] = cfg.get("hidden_size")
        rec["config_num_hidden_layers"] = cfg.get("num_hidden_layers")
        rec["config_model_type"] = cfg.get("model_type")
    except Exception as e:
        rec["config_error"] = repr(e)
    return rec, merged


def main(out):
    res = {}
    for key, repo, rev, files in MODELS:
        try:
            rec, merged = one(key, repo, rev, files)
            res[key] = rec
            with open("../artifacts/hdr_%s.json" % key, "w") as f:
                json.dump(merged, f)
            print("%-30s ser=%-15s uniq=%-15s tie=%-8s cfg_tie=%s" % (
                key, "{:,}".format(rec["serialized_total"]),
                "{:,}".format(rec["unique_total"]), rec["tie_status"],
                rec.get("config_tie_word_embeddings")), flush=True)
        except Exception as e:
            import traceback
            res[key] = {"error": repr(e), "trace": traceback.format_exc()}
            print("%-30s FAIL %s" % (key, repr(e)[:140]), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
