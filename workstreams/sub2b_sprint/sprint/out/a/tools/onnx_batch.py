"""Run remote ONNX initializer enumeration across every ONNX component."""
import json
import sys
import traceback

import net
import onnxr

PV = "rhasspy/piper-voices"
PVREV = "9f967d15e9ccdf43078586d1476ee70f314401bd"

TARGETS = [
    # key, url
    ("silero-vad.onnx", net.resolve("onnx-community/silero-vad", "onnx/model.onnx")),
    ("piper.en_US-lessac-medium", net.resolve(PV, "en/en_US/lessac/medium/en_US-lessac-medium.onnx", PVREV)),
    ("piper.es_ES-davefx-medium", net.resolve(PV, "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx", PVREV)),
    ("piper.fr_FR-siwis-medium", net.resolve(PV, "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx", PVREV)),
    ("piper.ru_RU-dmitri-medium", net.resolve(PV, "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx", PVREV)),
    ("piper.en_US-kusal-medium", net.resolve(PV, "en/en_US/kusal/medium/en_US-kusal-medium.onnx", PVREV)),
    ("piper.zh_CN-huayan-medium", net.resolve(PV, "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx", PVREV)),
    ("matcha-icefall-zh-baker.model-steps-3",
     net.resolve("csukuangfj/matcha-icefall-zh-baker", "model-steps-3.onnx",
                 "75e64a57a80fb370abb18a4e4d86bb090ed46e93")),
    ("supertonic3.vector_estimator", net.resolve("Supertone/supertonic-3", "onnx/vector_estimator.onnx", "3cadd1ee6394adea1bd021217a0e650ede09a323")),
    ("supertonic3.vocoder", net.resolve("Supertone/supertonic-3", "onnx/vocoder.onnx", "3cadd1ee6394adea1bd021217a0e650ede09a323")),
    ("supertonic3.text_encoder", net.resolve("Supertone/supertonic-3", "onnx/text_encoder.onnx", "3cadd1ee6394adea1bd021217a0e650ede09a323")),
    ("supertonic3.duration_predictor", net.resolve("Supertone/supertonic-3", "onnx/duration_predictor.onnx", "3cadd1ee6394adea1bd021217a0e650ede09a323")),
    ("zipformer-zh-en.encoder", net.resolve("pfluo/k2fsa-zipformer-chinese-english-mixed", "exp/encoder-epoch-99-avg-1.onnx")),
    ("zipformer-zh-en.decoder", net.resolve("pfluo/k2fsa-zipformer-chinese-english-mixed", "exp/decoder-epoch-99-avg-1.onnx")),
    ("zipformer-zh-en.joiner", net.resolve("pfluo/k2fsa-zipformer-chinese-english-mixed", "exp/joiner-epoch-99-avg-1.onnx")),
    ("vocos-22khz-univ",
     "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx"),
]


def main(out):
    res = {}
    for key, url in TARGETS:
        try:
            s, ts = onnxr.report(key, url=url)
            big = sorted(ts, key=lambda t: -onnxr.prod(t["dims"]))[:12]
            s["largest"] = [{"name": t["name"], "dims": t["dims"],
                             "dtype": onnxr.ONNX_DT.get(t["data_type"], ("?",))[0],
                             "params": onnxr.prod(t["dims"]),
                             "source": t["source"]} for t in big]
            s["has_external_data"] = any(t.get("external") for t in ts)
            s["url"] = url
            res[key] = s
            print("%-40s learned=%-14s int=%-9s total=%-14s file=%-14s fetched=%-12s cov=%.3f" % (
                key, "{:,}".format(s["learned"]), "{:,}".format(s["non_learned"]),
                "{:,}".format(s["total"]), "{:,}".format(s["file_bytes"]),
                "{:,}".format(s["bytes_fetched"]), s["coverage"]), flush=True)
        except Exception as e:
            res[key] = {"error": repr(e), "trace": traceback.format_exc(), "url": url}
            print("%-40s FAIL %s" % (key, repr(e)[:120]), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
