"""Repo inventory sweep: full recursive file tree + size, for every component
repo in the six stacks. This is the A3 'diff the HF API total against the full
file tree' step -- subdirectory weights are invisible to the API safetensors
total and this is where hidden dependencies show up."""
import json
import sys

import net

REPOS = [
    # (key, repo, revision, kind)
    ("parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", "main", "models"),
    ("qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B",
     "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", "models"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "main", "models"),
    ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "main", "models"),
    ("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct",
     "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "models"),
    ("falcon-h1-1.5b-instruct", "tiiuae/Falcon-H1-1.5B-Instruct",
     "80ebc50d7799a440b96c93bb6686a3924a09b0cb", "models"),
    ("falcon-h1-1.5b-deep-instruct", "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
     "main", "models"),
    ("omniasr-ctc-300m", "facebook/omniASR-CTC-300M",
     "8e35f0cc28fa6099e0c14d56db85ce0423baa691", "models"),
    ("canary-180m-flash", "nvidia/canary-180m-flash", "main", "models"),
    ("gigaam-multilingual", "ai-sage/GigaAM-Multilingual", "main", "models"),
    ("magpie-tts-multilingual", "nvidia/magpie_tts_multilingual_357m", "main",
     "models"),
    ("sensevoice-small", "FunAudioLLM/SenseVoiceSmall", "main", "models"),
    ("k2fsa-zipformer-zh-en", "pfluo/k2fsa-zipformer-chinese-english-mixed",
     "main", "models"),
    ("charsiu-byt5-g2p", "charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
     "main", "models"),
    ("g2pm", "kakaobrain/g2pM", "main", "models"),
    ("silero-vad-onnx", "onnx-community/silero-vad", "main", "models"),
    ("moss-tts-nano-100m", "OpenMOSS-Team/MOSS-TTS-Nano-100M",
     "44502f80dbf9743528fa921cc544d662c685ebec", "models"),
    ("moss-audio-tokenizer-nano", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
     "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68", "models"),
    ("supertonic-3", "Supertone/supertonic-3",
     "3cadd1ee6394adea1bd021217a0e650ede09a323", "models"),
    ("matcha-icefall-zh-baker", "csukuangfj/matcha-icefall-zh-baker",
     "75e64a57a80fb370abb18a4e4d86bb090ed46e93", "models"),
    ("piper-voices", "rhasspy/piper-voices",
     "9f967d15e9ccdf43078586d1476ee70f314401bd", "models"),
    ("qwen3.5-0.8b-gguf", "unsloth/Qwen3.5-0.8B-GGUF", "main", "models"),
    ("qwen3.5-0.8b-mtp-gguf", "unsloth/Qwen3.5-0.8B-MTP-GGUF", "main", "models"),
]

WEIGHTY = (".safetensors", ".bin", ".onnx", ".onnx_data", ".pt", ".pth", ".nemo",
           ".gguf", ".ckpt", ".msgpack", ".h5", ".tflite", ".so", ".mlmodel",
           ".pte", ".engine", ".plan", ".bin.gz", ".npz", ".jit", ".ts",
           ".onnx.data", ".data")


def run(only=None):
    out = {}
    for key, repo, rev, kind in REPOS:
        if only and key not in only:
            continue
        rec = {"repo": repo, "revision_requested": rev, "kind": kind}
        try:
            t = net.tree(repo, rev, kind)
            files = [{"path": e["path"], "size": net.sizeof(e)}
                     for e in t if e.get("type") == "file"]
            files.sort(key=lambda x: -x["size"])
            rec["ok"] = True
            rec["n_files"] = len(files)
            rec["total_bytes"] = sum(f["size"] for f in files)
            rec["files"] = files
            rec["weighty"] = [f for f in files
                              if f["path"].lower().endswith(WEIGHTY)]
            rec["weighty_bytes"] = sum(f["size"] for f in rec["weighty"])
            rec["subdirs"] = sorted({f["path"].split("/")[0]
                                     for f in files if "/" in f["path"]})
            try:
                info = net.api("/api/%s/%s/revision/%s" % (kind, repo, rev))
                rec["sha"] = info.get("sha")
                rec["api_safetensors"] = info.get("safetensors")
                rec["gated"] = info.get("gated")
                rec["license"] = (info.get("cardData") or {}).get("license")
                rec["license_name"] = (info.get("cardData") or {}).get("license_name")
                rec["license_link"] = (info.get("cardData") or {}).get("license_link")
                rec["tags"] = info.get("tags")
            except Exception as e:
                rec["api_error"] = repr(e)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = repr(e)
        out[key] = rec
        st = rec.get("api_safetensors")
        print("%-30s %-6s files=%-5s bytes=%-16s api_st_total=%s sha=%s" % (
            key, "OK" if rec.get("ok") else "FAIL", rec.get("n_files", "-"),
            "{:,}".format(rec.get("total_bytes", 0)),
            "{:,}".format(st["total"]) if st and "total" in st else "-",
            (rec.get("sha") or "")[:12]), flush=True)
        if not rec.get("ok"):
            print("    ", rec.get("error")[:200], flush=True)
    return out


if __name__ == "__main__":
    only = set(sys.argv[2:]) or None
    res = run(only)
    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)
