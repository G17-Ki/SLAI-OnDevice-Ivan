#!/usr/bin/env python3
"""Summarize on-device sweep results and compare them with the host run.

The host study repeatedly hedged its conclusions with "the phone will be
slower" and modelled that as a 1.5x/2.0x slowdown. This computes the *measured*
ratio per candidate instead, so that assumption is replaced by a number.

Reads the JSONL the bench app writes (one record per measurement, flushed as it
goes) and the host bench JSONs, and emits a joined table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Phrase ids in the device bundle mapped to the host fixture case they mirror,
# so the two runs are compared on the same text rather than on whatever each
# happened to call "short".
PHRASE_TO_HOST_CASE = {
    "micro": {"zh": "zh_micro_ack", "ru": "ru_micro_ack"},
    "clause": {"zh": "zh_short_clause", "ru": "ru_short_clause"},
    "answer": {"zh": "zh_answer_short", "ru": "ru_answer_short"},
    "weather": {"zh": "zh_answer_medium", "ru": "ru_answer_medium"},
    "question": {"zh": "zh_question", "ru": "ru_question"},
    "long": {"zh": "zh_answer_long", "ru": "ru_answer_long"},
}

# Device candidate id -> host bench directory.
DEVICE_TO_HOST_DIR = {
    "matcha_zh_baker": "matcha_zh_baker",
    "melotts_zh": "melotts_zh",
    "melotts_zh_int8": "melotts_zh_int8",
    "piper_zh_huayan": "piper_zh_huayan",
    "kokoro_zh_fp32": "kokoro_zh_fp32",
    "kokoro_zh_int8": "kokoro_zh",
    "zipvoice_zh": "zipvoice_zh",
    "supertonic3_s4": "supertonic_ru_s4",
    "supertonic3_s5": "supertonic_ru_s5",
    "supertonic3_s6": "supertonic_ru_s6",
    "supertonic3_s8": "supertonic_ru_s8",
    "piper_ru_dmitri": "piper_ru_dmitri",
    "piper_ru_irina": "piper_ru_irina",
    "piper_ru_ruslan": "piper_ru_ruslan",
}


def load_jsonl(paths):
    header = None
    runs, failures = [], []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A run cut off mid-write leaves one truncated line; the rest of
                # the file is still good evidence.
                failures.append({"record": "truncated_line", "path": Path(path).name})
                continue
            kind = rec.get("record")
            if kind == "header":
                header = rec
            elif kind == "run":
                runs.append(rec)
            else:
                failures.append(rec)
    return header, runs, failures


def dist(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "p95": sorted(clean)[min(len(clean) - 1, int(round(0.95 * (len(clean) - 1))))],
        "min": min(clean),
        "max": max(clean),
    }


def host_case_stats(host_root: Path, candidate: str, case_id: str):
    d = DEVICE_TO_HOST_DIR.get(candidate)
    if not d:
        return None
    files = list((host_root / d).glob("bench_*.json")) if (host_root / d).is_dir() else []
    if not files:
        return None
    bench = json.loads(files[0].read_text(encoding="utf-8"))
    return bench["summary"].get(case_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, action="append", required=True)
    ap.add_argument("--host-bench-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    header, runs, failures = load_jsonl(args.jsonl)
    if not runs:
        print("no run records found")
        return 1

    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["candidate_id"], r.get("threads", 0), r["phrase_id"])].append(r)

    rows = []
    for (cand, threads, phrase), group in sorted(by_key.items()):
        lang = group[0].get("language", "zh")
        host_case = PHRASE_TO_HOST_CASE.get(phrase, {}).get(lang)
        host = host_case_stats(args.host_bench_root, cand, host_case) if (
            args.host_bench_root and host_case) else None

        first = dist([g["first_chunk_ms"] for g in group])
        rtf = dist([g["rtf"] for g in group])
        gate_pass = sum(1 for g in group if g["gate_passed"])
        dropped = sum(g.get("dropped_chunks_failed_gate", 0) for g in group)

        speedup = None
        if host and rtf and host["rtf"]["median"]:
            speedup = host["rtf"]["median"] / rtf["median"] if rtf["median"] else None

        rows.append({
            "candidate_id": cand,
            "language": lang,
            "threads": threads,
            "phrase_id": phrase,
            "host_case_id": host_case,
            "device": {
                "first_chunk_ms": first,
                "rtf": rtf,
                "audio_s": dist([g["audio_duration_s"] for g in group]),
                "peak_pss_mib": dist([g["peak_total_pss_kb"] / 1024 for g in group]),
                "load_ms": dist([g["load_ms"] for g in group]),
                "chunk_count": dist([g["chunk_count"] for g in group]),
                "first_audible_ms": dist(
                    [g["first_audible_ms"] for g in group if g.get("first_audible_ms", -1) >= 0]
                ),
                "underruns": dist(
                    [g["audiotrack_underruns"] for g in group if g.get("audiotrack_underruns", -1) >= 0]
                ),
                "gate_pass": gate_pass,
                "gate_runs": len(group),
                "gate_failures": sorted({f for g in group for f in g.get("gate_failures", [])}),
                "gate_warnings": sorted({w for g in group for w in g.get("gate_warnings", [])}),
                "dropped_chunks": dropped,
            },
            "host": {
                "first_chunk_s_p95": host["first_chunk_s"]["p95"] if host else None,
                "rtf_median": host["rtf"]["median"] if host else None,
            } if host else None,
            "device_speedup_vs_host_rtf": speedup,
        })

    speedups = [r["device_speedup_vs_host_rtf"] for r in rows if r["device_speedup_vs_host_rtf"]]
    payload = {
        "schema_version": 1,
        "device": header.get("device") if header else None,
        "sweep": {k: header.get(k) for k in ("tag", "repeats", "play", "thread_sweep")} if header else None,
        "run_count": len(runs),
        "failure_records": failures,
        "device_vs_host_rtf_speedup": {
            "note": "host RTF median / device RTF median; >1 means the phone is faster",
            **(dist(speedups) or {}),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.markdown:
        lines = [
            "| Candidate | Lang | Phrase | thr | first chunk ms med/p95 | RTF med | host RTF | speed-up | peak PSS MiB | gate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in sorted(rows, key=lambda x: (x["language"], x["candidate_id"], x["phrase_id"])):
            d = r["device"]
            f = d["first_chunk_ms"]
            host_rtf = r["host"]["rtf_median"] if r["host"] else None
            speedup = r["device_speedup_vs_host_rtf"]
            gate = "pass" if d["gate_pass"] == d["gate_runs"] else "FAIL " + ",".join(d["gate_failures"])
            lines.append(
                "| `{cid}` | {lang} | {phrase} | {thr} | {fmed:.0f}/{fp95:.0f} | {rtf:.3f} | "
                "{hrtf} | {sp} | {pss:.0f} | {gate} |".format(
                    cid=r["candidate_id"], lang=r["language"], phrase=r["phrase_id"],
                    thr=r["threads"], fmed=f["median"], fp95=f["p95"],
                    rtf=d["rtf"]["median"],
                    hrtf="-" if host_rtf is None else "{:.3f}".format(host_rtf),
                    sp="-" if not speedup else "{:.1f}x".format(speedup),
                    pss=d["peak_pss_mib"]["max"], gate=gate,
                )
            )
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"runs: {len(runs)}   failures: {len(failures)}")
    if speedups:
        s = dist(speedups)
        print(f"device vs host RTF speed-up: median {s['median']:.2f}x  range {s['min']:.2f}-{s['max']:.2f}x")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
