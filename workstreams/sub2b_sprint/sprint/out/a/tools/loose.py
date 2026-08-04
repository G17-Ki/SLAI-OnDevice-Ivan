"""Loose ends: hidden dependencies found by the A3 sweep + components whose
artefact lives outside HuggingFace."""
import json
import sys
import traceback

import container as C
import net
import onnxr

REPOS = [
    ("nemo-nano-codec-22khz", "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps", "main"),
    ("moss-audio-tokenizer-nano", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
     "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"),
    ("silero-models", "snakers4/silero-models", "main"),
    ("g2pm-kakaobrain", "kakaobrain/g2pM", "main"),
    ("g2pw", "nomnomnonono/g2pW", "main"),
    ("ten-vad", "TEN-framework/ten-vad", "main"),
    ("moss-tts-nano-onnx", "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "main"),
    ("ruaccent", "ruaccent/ruaccent_big_poetry", "main"),
    ("sensevoice-onnx", "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
     "main"),
]


def show(key, repo, rev):
    rec = {"repo": repo, "revision": rev}
    try:
        t = net.tree(repo, rev)
        files = sorted([{"path": e["path"], "size": net.sizeof(e)}
                        for e in t if e.get("type") == "file"],
                       key=lambda x: -x["size"])
        rec["ok"] = True
        rec["files"] = files
        rec["total_bytes"] = sum(f["size"] for f in files)
        try:
            info = net.api("/api/models/%s/revision/%s" % (repo, rev))
            rec["sha"] = info.get("sha")
            rec["api_safetensors"] = info.get("safetensors")
            rec["license"] = (info.get("cardData") or {}).get("license")
            rec["license_name"] = (info.get("cardData") or {}).get("license_name")
        except Exception as e:
            rec["api_error"] = repr(e)
        print("== %-26s %-55s bytes=%s sha=%s lic=%s" % (
            key, repo, "{:,}".format(rec["total_bytes"]),
            (rec.get("sha") or "")[:10], rec.get("license")))
        for f in files[:10]:
            print("      %14s  %s" % ("{:,}".format(f["size"]), f["path"]))
        st = rec.get("api_safetensors")
        if st:
            print("      api_safetensors:", json.dumps(st))
    except Exception as e:
        rec["ok"] = False
        rec["error"] = repr(e)
        print("== %-26s %-55s FAIL %s" % (key, repo, repr(e)[:110]))
    return rec


if __name__ == "__main__":
    out = {}
    for k, r, rv in REPOS:
        out[k] = show(k, r, rv)
        print()
    json.dump(out, open(sys.argv[1], "w"), indent=1)
