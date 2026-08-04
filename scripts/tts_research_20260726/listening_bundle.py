#!/usr/bin/env python3
"""Build a blinded listening bundle and score sheet.

ASR error rates and automatic MOS predictors rank intelligibility and
"averageness"; neither is evidence that a voice sounds human. This bundle is
the only artifact in the run that can settle naturalness, and it is
deliberately blinded so the listener cannot see which engine produced a clip.

Loudness is equalized before blinding. Without it the test would mostly measure
output gain: the candidates in this run differ by more than 20 dB in peak
level, and the louder clip nearly always wins an unequalized preference test.
Sample rate is deliberately *not* equalized, because it is a real property of
what would ship; the key records it so the confound stays visible.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


# -24 dBFS rather than -20: at -20 several high-crest-factor Russian clips hit
# the peak ceiling before reaching the target, leaving up to 3.3 dB of residual
# level spread in exactly the comparison the bundle exists to make fair. -24
# clears the ceiling for every candidate measured here, so equalization is exact.
TARGET_RMS_DBFS = -24.0
PEAK_CEILING = 0.95


def normalize(samples: np.ndarray) -> tuple[np.ndarray, dict]:
    arr = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float64)))))
    if rms <= 0:
        return arr, {"applied_gain_db": 0.0, "reason": "silent"}
    target = 10 ** (TARGET_RMS_DBFS / 20.0)
    gain = target / rms
    peak = float(np.max(np.abs(arr)))
    limited = False
    if peak * gain > PEAK_CEILING:
        gain = PEAK_CEILING / peak
        limited = True
    return (arr * gain).astype(np.float32), {
        "source_rms": rms,
        "source_peak": peak,
        "applied_gain_db": float(20 * np.log10(gain)),
        "peak_limited": limited,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, required=True,
                    help="directory containing one subdirectory per bench run")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--case", action="append", required=True,
                    help="case ids to include; repeat for several")
    ap.add_argument("--engine", action="append", default=[],
                    help="restrict to these bench directory names")
    ap.add_argument("--extra", action="append", default=[],
                    help="label::path.wav for samples produced outside bench.py")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for bench_dir in sorted(args.bench_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        if args.engine and bench_dir.name not in args.engine:
            continue
        bench_files = list(bench_dir.glob("bench_*.json"))
        if not bench_files:
            continue
        bench = json.loads(bench_files[0].read_text(encoding="utf-8"))
        if bench["language"] != args.language:
            continue
        for case_id in args.case:
            summary = bench["summary"].get(case_id)
            if not summary or not summary.get("wav_path"):
                continue
            entries.append({
                "engine_label": f"{bench['engine_id']}{bench.get('tag', '')}",
                "bench_dir": bench_dir.name,
                "family": bench["family"],
                "case_id": case_id,
                "text": summary["text"],
                "source_wav": bench_dir / "audio" / summary["wav_path"],
                "source_sha256": summary.get("wav_sha256"),
            })

    for spec in args.extra:
        label, _, path = spec.partition("::")
        case_id, _, wav = label.partition("@")
        p = Path(wav or path)
        if p.exists():
            entries.append({
                "engine_label": case_id or p.stem,
                "bench_dir": "external",
                "family": "external",
                "case_id": case_id or p.stem,
                "text": "",
                "source_wav": p,
                "source_sha256": None,
            })

    if not entries:
        raise SystemExit("no samples matched the requested language/cases")

    rng = random.Random(args.seed)
    rng.shuffle(entries)

    key = []
    for index, entry in enumerate(entries, 1):
        data, rate = sf.read(entry["source_wav"], dtype="float32", always_2d=False)
        normalized, gain_info = normalize(data)
        blind_id = f"{args.language}_{index:03d}"
        target = samples_dir / f"{blind_id}.wav"
        sf.write(target, normalized, rate, subtype="PCM_16")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        key.append({
            "blind_id": blind_id,
            "engine_label": entry["engine_label"],
            "family": entry["family"],
            "case_id": entry["case_id"],
            "text": entry["text"],
            "sample_rate_hz": int(rate),
            "duration_s": float(len(data) / rate),
            "loudness_normalization": gain_info,
            "source_wav_sha256": entry["source_sha256"],
            "blinded_wav_sha256": digest,
        })

    (out / "key.json").write_text(
        json.dumps({"schema_version": 1, "language": args.language, "seed": args.seed,
                    "target_rms_dbfs": TARGET_RMS_DBFS, "entries": key},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Score sheet: no engine names, no ordering hints, one row per clip.
    rows = "\n".join(
        f"""      <tr>
        <td class="id">{html.escape(e['blind_id'])}</td>
        <td><audio controls preload="none" src="samples/{html.escape(e['blind_id'])}.wav"></audio></td>
        <td class="txt">{html.escape(e['text'])}</td>
        <td>{"".join(f'<label><input type="radio" name="nat_{e["blind_id"]}" value="{v}">{v}</label>' for v in range(1, 6))}</td>
        <td>{"".join(f'<label><input type="radio" name="int_{e["blind_id"]}" value="{v}">{v}</label>' for v in range(1, 6))}</td>
        <td><input type="text" name="note_{html.escape(e['blind_id'])}" placeholder="artifacts, tone errors, prosody"></td>
      </tr>"""
        for e in key
    )

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Blinded TTS listening test - {html.escape(args.language)}</title>
<style>
 body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .5rem .4rem; vertical-align: top; }}
 th {{ text-align: left; background: #f6f6f6; position: sticky; top: 0; }}
 .id {{ font-family: ui-monospace, monospace; white-space: nowrap; }}
 .txt {{ max-width: 26rem; }}
 label {{ margin-right: .4rem; white-space: nowrap; }}
 audio {{ width: 240px; }}
 .note {{ background: #fffbe6; border: 1px solid #f0e0a0; padding: .8rem; margin-bottom: 1rem; }}
</style>
<h1>Blinded listening test - {html.escape(args.language)}</h1>
<div class="note">
 <p><strong>Instructions.</strong> Play each clip once, then score it. Do not
 look up which engine produced a clip; the mapping is withheld until every row
 is scored.</p>
 <ul>
  <li><strong>Naturalness (1-5)</strong>: 1 = obviously robotic, 3 = clearly synthetic but pleasant, 5 = could be a human recording.</li>
  <li><strong>Intelligibility (1-5)</strong>: 1 = cannot follow, 5 = effortless.</li>
 </ul>
 <p>All clips were loudness-normalized to {TARGET_RMS_DBFS:g} dBFS RMS, so do not
 score loudness. Sample rates differ on purpose and are part of what would
 ship.</p>
</div>
<table>
 <thead><tr><th>Clip</th><th>Audio</th><th>Text</th><th>Naturalness</th><th>Intelligibility</th><th>Notes</th></tr></thead>
 <tbody>
{rows}
 </tbody>
</table>
<p>Total clips: {len(key)}. Unblinding key: <code>key.json</code> (do not open before scoring).</p>
"""
    (out / "score_sheet.html").write_text(page, encoding="utf-8")

    lines = ["# Blinded listening score sheet - " + args.language, "",
             "Score naturalness and intelligibility 1-5. Do not open `key.json` first.", "",
             "| Clip | Text | Naturalness | Intelligibility | Notes |", "|---|---|---|---|---|"]
    for e in key:
        lines.append(f"| `{e['blind_id']}` | {e['text']} |  |  |  |")
    (out / "score_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(key)} blinded clips to {samples_dir}")
    print(f"score sheet: {out / 'score_sheet.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
