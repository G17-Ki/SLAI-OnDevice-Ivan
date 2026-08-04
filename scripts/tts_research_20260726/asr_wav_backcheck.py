#!/usr/bin/env python3
"""Fixed-locale ASR back-check for arbitrary WAV files.

Used for the MeloTTS exact-QRD chain, where the comparison is between a local
golden waveform and the waveform the SM8850 QRD produced from the same fixture.
As everywhere else in this run, a matching transcript is an intelligibility
screen, not evidence of naturalness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asr_judge import dominant_script, error_rates, EXPECTED_SCRIPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", required=True, help="label::path.wav")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--locale", required=True, choices=sorted(EXPECTED_SCRIPT))
    ap.add_argument("--model", default="small")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type, cpu_threads=2)
    results = {"schema_version": 1, "locale": args.locale, "reference": args.reference,
               "judge": {"model": Path(args.model).name, "compute_type": args.compute_type,
                         "language_forced": args.locale},
               "items": []}

    for spec in args.wav:
        label, _, path = spec.partition("::")
        wav = Path(path)
        if not wav.is_file():
            results["items"].append({"label": label, "error": "missing", "path": wav.name})
            continue
        segments, _info = model.transcribe(
            str(wav), language=args.locale, beam_size=5,
            condition_on_previous_text=False, vad_filter=False,
        )
        hypothesis = "".join(seg.text for seg in segments)
        script = dominant_script(hypothesis)
        item = {"label": label, "wav": wav.name, "hypothesis": hypothesis,
                "dominant_script": script, "script_ok": script == EXPECTED_SCRIPT[args.locale]}
        item.update(error_rates(args.reference, hypothesis, args.locale))
        results["items"].append(item)
        print(f"{label}: cer={item['cer']:.4f} script={script} hyp={hypothesis!r}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
