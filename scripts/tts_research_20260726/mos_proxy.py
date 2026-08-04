#!/usr/bin/env python3
"""Reference-free speech-quality proxy over a blinded listening bundle.

This runs DNSMOS (ITU-T P.835 SIG/BAK/OVRL plus P.808) on the loudness-equalized
clips and reports per-engine means.

**What this is not.** DNSMOS was trained to score noise-suppression quality on
recorded speech, not to predict how human a synthesized voice sounds. It
reliably penalizes buzzy, muffled, or artifacted output, which is a real part of
"robotic", but a perfectly clean voice with wrong prosody or wrong tones scores
well. It is a ranking aid and an artifact detector. It cannot promote a
candidate and it is not a substitute for the blinded human score sheet.

Scoring the blinded clips (not the raw bench WAVs) keeps loudness out of the
result, since the bundle is already normalized to a common RMS.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, action="append", required=True,
                    help="listening bundle directory containing key.json and samples/")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import librosa
    from speechmos import dnsmos

    payload = {
        "schema_version": 1,
        "metric": "DNSMOS P.835 (sig/bak/ovrl) + P.808",
        "caveat": (
            "trained for noise-suppression quality on recorded speech, not TTS naturalness; "
            "detects artifacts, not prosody or tone errors; ranking aid only"
        ),
        "bundles": {},
    }

    for bundle in args.bundle_dir:
        key = json.loads((bundle / "key.json").read_text(encoding="utf-8"))
        per_engine = defaultdict(list)
        clips = []
        for entry in key["entries"]:
            wav = bundle / "samples" / f"{entry['blind_id']}.wav"
            samples, rate = sf.read(wav, dtype="float32", always_2d=False)
            if rate != 16000:
                samples = librosa.resample(samples, orig_sr=rate, target_sr=16000)
            peak = float(np.max(np.abs(samples)))
            if peak > 0:
                # DNSMOS expects roughly speech-level input; the bundle is already
                # RMS-equalized, so this only guards against inter-sample clipping.
                samples = samples / max(peak, 0.7) * min(peak, 0.7)
            scores = dnsmos.run(np.asarray(samples, dtype=np.float32), sr=16000)
            record = {
                "blind_id": entry["blind_id"],
                "engine_label": entry["engine_label"],
                "case_id": entry["case_id"],
                "sample_rate_hz": entry["sample_rate_hz"],
                **{k: float(v) for k, v in scores.items()},
            }
            clips.append(record)
            per_engine[entry["engine_label"]].append(record)
            print(f"  {entry['blind_id']} {entry['engine_label']:<18} "
                  f"ovrl={record['ovrl_mos']:.3f} p808={record['p808_mos']:.3f}", flush=True)

        summary = {}
        for label, records in per_engine.items():
            summary[label] = {
                "n": len(records),
                "ovrl_mos_mean": statistics.mean(r["ovrl_mos"] for r in records),
                "sig_mos_mean": statistics.mean(r["sig_mos"] for r in records),
                "bak_mos_mean": statistics.mean(r["bak_mos"] for r in records),
                "p808_mos_mean": statistics.mean(r["p808_mos"] for r in records),
            }
        payload["bundles"][key["language"]] = {"clips": clips, "per_engine": summary}

        print(f"\n=== {key['language']} per-engine means")
        for label, s in sorted(summary.items(), key=lambda kv: -kv[1]["ovrl_mos_mean"]):
            print(f"  {label:<20} ovrl={s['ovrl_mos_mean']:.3f} sig={s['sig_mos_mean']:.3f} "
                  f"bak={s['bak_mos_mean']:.3f} p808={s['p808_mos_mean']:.3f}  (n={s['n']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
