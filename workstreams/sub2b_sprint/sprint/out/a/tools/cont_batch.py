"""Container batch: torch .bin/.pt and NeMo .nemo (TAR -> model_weights.ckpt)."""
import json
import sys
import traceback

import container as C
import net

TORCH = [
    ("gigaam-multilingual", net.resolve("ai-sage/GigaAM-Multilingual", "pytorch_model.bin")),
    ("omniasr-ctc-300m", net.resolve("facebook/omniASR-CTC-300M", "omniASR-CTC-300M.pt",
                                     "8e35f0cc28fa6099e0c14d56db85ce0423baa691")),
    ("sensevoice-small", net.resolve("FunAudioLLM/SenseVoiceSmall", "model.pt")),
    ("moss-tts-nano-100m", net.resolve("OpenMOSS-Team/MOSS-TTS-Nano-100M", "pytorch_model.bin",
                                       "44502f80dbf9743528fa921cc544d662c685ebec")),
    ("charsiu-byt5-g2p", net.resolve("charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
                                     "pytorch_model.bin")),
    ("zipformer-zh-en.pretrained_pt",
     net.resolve("pfluo/k2fsa-zipformer-chinese-english-mixed", "exp/pretrained.pt")),
    ("zipformer-zh-en.rnnlm",
     net.resolve("pfluo/k2fsa-zipformer-chinese-english-mixed",
                 "exp/rnn-lm-small-text-19.pt")),
]

NEMO = [
    ("canary-180m-flash", net.resolve("nvidia/canary-180m-flash", "canary-180m-flash.nemo")),
    ("magpie-tts-multilingual", net.resolve("nvidia/magpie_tts_multilingual_357m",
                                            "magpie_tts_multilingual_357m.nemo")),
    ("parakeet-tdt-0.6b-v3.nemo", net.resolve("nvidia/parakeet-tdt-0.6b-v3",
                                              "parakeet-tdt-0.6b-v3.nemo")),
]


def do_torch(key, url, sub=None):
    r = C.torch_zip_tensors(sub if sub is not None else url)
    s = C.summarise_torch(r)
    s["n_members"] = len(r["members"])
    s["top"] = sorted(
        [{"name": t["name"], "shape": t["shape"], "dtype": t["dtype"],
          "params": C.prod(t["shape"])} for t in r["tensors"] if t["shape"]],
        key=lambda x: -x["params"])[:15]
    return s, r


def main(out):
    res = {}
    for key, url in TORCH:
        try:
            s, r = do_torch(key, url)
            res[key] = s
            json.dump([{k: t[k] for k in ("name", "shape", "dtype", "storage_key")}
                       for t in r["tensors"]],
                      open("../artifacts/tensors_%s.json" % key, "w"))
            print("%-32s learned=%-14s int=%-9s total=%-14s dtypes=%s" % (
                key, "{:,}".format(s["learned"]), "{:,}".format(s["non_learned"]),
                "{:,}".format(s["total"]), s["by_dtype"]), flush=True)
        except Exception as e:
            res[key] = {"error": repr(e), "trace": traceback.format_exc()}
            print("%-32s FAIL %s" % (key, repr(e)[:160]), flush=True)

    for key, url in NEMO:
        try:
            ms, size = C.tar_members(url)
            ck = [m for m in ms if m["name"].endswith(".ckpt")]
            rec = {"nemo_bytes": size,
                   "members": [{"name": m["name"], "size": m["size"]} for m in ms
                               if m["type"] == "0"]}
            if ck:
                sub = C.SubFile(url, ck[0]["data_offset"], ck[0]["size"])
                s, r = do_torch(key, url, sub)
                rec.update(s)
                rec["ckpt_member"] = ck[0]["name"]
                rec["ckpt_bytes"] = ck[0]["size"]
                json.dump([{k: t[k] for k in ("name", "shape", "dtype", "storage_key")}
                           for t in r["tensors"]],
                          open("../artifacts/tensors_%s.json" % key, "w"))
                print("%-32s learned=%-14s int=%-9s total=%-14s dtypes=%s" % (
                    key, "{:,}".format(s["learned"]),
                    "{:,}".format(s["non_learned"]),
                    "{:,}".format(s["total"]), s["by_dtype"]), flush=True)
            else:
                print("%-32s no .ckpt member: %s" % (key, [m['name'] for m in ms][-4:]),
                      flush=True)
            res[key] = rec
        except Exception as e:
            res[key] = {"error": repr(e), "trace": traceback.format_exc()}
            print("%-32s FAIL %s" % (key, repr(e)[:160]), flush=True)

    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
