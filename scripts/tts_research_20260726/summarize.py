#!/usr/bin/env python3
"""Merge bench and ASR-judge outputs into one comparable table.

Gate thresholds are defined below and applied uniformly. The verdict this
produces is a *screen*: it can disqualify a candidate on measured grounds, but
it cannot promote one, because naturalness is decided by the blinded listening
bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATES = {
    "first_chunk_p95_s": 0.9,
    "rtf": 0.7,
    "zh_micro_cer": 0.20,
    "zh_micro_cer_hard": 0.40,
    "ru_micro_wer": 0.30,
    "ru_micro_wer_hard": 0.50,
    "ru_micro_cer": 0.20,
    "weights_preferred_mib": 250,
    "weights_conditional_mib": 500,
}


def fmt(value, spec="{:.3f}") -> str:
    return "-" if value is None else spec.format(value)


def pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


# The <0.9 s first-chunk gate describes the clause the orchestrator flushes
# first, which the streaming design bounds to a sentence or ~50-70 characters.
# Scoring it against a whole four-sentence paragraph submitted as one blob would
# measure a case the phone never produces, so the gate uses the streaming cases
# and the paragraph case is reported separately as the worst case.
STREAMING_KINDS = {"streaming_floor", "streaming_first_chunk"}


def p95_of(summary: dict, field: str, kinds: set | None = None) -> float | None:
    values = [
        c[field]["p95"]
        for c in summary.values()
        if c.get(field, {}).get("p95") is not None and (kinds is None or c.get("kind") in kinds)
    ]
    return max(values) if values else None


def median_of(summary: dict, field: str) -> float | None:
    values = [c[field]["median"] for c in summary.values() if c.get(field, {}).get("median") is not None]
    return max(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, required=True)
    ap.add_argument("--asr", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    # Keyed by (locale, engine) because the same engine id appears in several
    # language suites - eSpeak runs in all six - and a locale-blind key would
    # silently attribute one language's error rates to another.
    asr_by_engine: dict[tuple, dict] = {}
    asr_calibration: dict[str, list] = {}
    asr_excluded_cases: dict[str, list] = {}
    for path in args.asr:
        data = json.loads(path.read_text(encoding="utf-8"))
        asr_calibration[data["locale"]] = data.get("calibration", [])
        asr_excluded_cases[data["locale"]] = data.get("digit_rendering_cases_excluded", [])
        for key, val in data["engines"].items():
            asr_by_engine[(data["locale"], key)] = {**val, "locale": data["locale"]}

    rows = []
    for bench_dir in sorted(args.bench_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        files = list(bench_dir.glob("bench_*.json"))
        if not files:
            continue
        bench = json.loads(files[0].read_text(encoding="utf-8"))
        # Flush-profile runs carry only a sweep and no scored suite cases; they
        # belong in the flush report, not in the candidate screen.
        if not any(c["gate_run_count"] for c in bench["summary"].values()):
            continue
        key = f"{bench['engine_id']}{bench.get('tag', '')}"
        summary = bench["summary"]
        asr = asr_by_engine.get((bench["language"], key), {})

        first_p95 = p95_of(summary, "first_chunk_s", STREAMING_KINDS)
        first_p95_worst = p95_of(summary, "first_chunk_s")
        rtf_med = median_of(summary, "rtf")
        # A streaming budget is set by the tail, not the middle: an engine whose
        # median RTF clears the gate but whose P95 reaches real time will still
        # starve the playback queue on the slow runs.
        rtf_p95 = p95_of(summary, "rtf")
        gate_fail_cases = [cid for cid, c in summary.items() if c["gate_pass_count"] < c["gate_run_count"]]
        all_hard = sorted({f for c in summary.values() for f in c["hard_failures"]})
        footprint_mib = bench["footprint"]["total_mib"]
        peak_rss = max(
            (c["peak_rss_bytes"]["max"] for c in summary.values() if c["peak_rss_bytes"]["max"]), default=None
        )
        gap_free = all(c["gap_free_1_5x"] for c in summary.values())

        blocking = []
        if gate_fail_cases:
            blocking.append(f"waveform gate failed on {len(gate_fail_cases)} case(s): {','.join(all_hard)}")
        if bench["language"] == "zh" and asr.get("micro_cer") is not None:
            if asr["micro_cer"] > GATES["zh_micro_cer_hard"]:
                blocking.append(f"Mandarin micro-CER {asr['micro_cer']:.1%} above the 40% hard limit")
        if bench["language"] == "ru" and asr.get("micro_wer") is not None:
            if asr["micro_wer"] > GATES["ru_micro_wer_hard"]:
                blocking.append(f"Russian micro-WER {asr['micro_wer']:.1%} above the 50% hard limit")
        if asr.get("hard_failures"):
            blocking.append("ASR hard failure: " + ",".join(asr["hard_failures"]))
        if footprint_mib > GATES["weights_conditional_mib"]:
            blocking.append(f"footprint {footprint_mib:.0f} MiB above the 500 MiB conditional ceiling")
        if bench["license_risk"] == "unclear":
            blocking.append("weights license unclear")

        missed_targets = []
        if first_p95 is not None and first_p95 >= GATES["first_chunk_p95_s"]:
            missed_targets.append(f"first playable chunk P95 {first_p95:.2f}s >= 0.90s")
        if first_p95_worst is not None and first_p95_worst >= GATES["first_chunk_p95_s"]:
            missed_targets.append(
                f"worst single-case first chunk P95 {first_p95_worst:.2f}s >= 0.90s "
                "(orchestrator must bound the first flush length)"
            )
        if rtf_med is not None and rtf_med >= GATES["rtf"]:
            missed_targets.append(f"RTF {rtf_med:.2f} >= 0.70")
        if rtf_p95 is not None and rtf_p95 >= GATES["rtf"]:
            missed_targets.append(f"RTF P95 {rtf_p95:.2f} >= 0.70")
        if bench["language"] == "zh" and asr.get("micro_cer") is not None and asr["micro_cer"] > GATES["zh_micro_cer"]:
            missed_targets.append(f"micro-CER {asr['micro_cer']:.1%} > 20% target")
        if bench["language"] == "ru":
            if asr.get("micro_wer") is not None and asr["micro_wer"] > GATES["ru_micro_wer"]:
                missed_targets.append(f"micro-WER {asr['micro_wer']:.1%} > 30% target")
            if asr.get("micro_cer") is not None and asr["micro_cer"] > GATES["ru_micro_cer"]:
                missed_targets.append(f"micro-CER {asr['micro_cer']:.1%} > 20% target")
        if not gap_free:
            missed_targets.append("simulated playback gap at 1.5x host slowdown")
        if footprint_mib > GATES["weights_preferred_mib"]:
            missed_targets.append(f"footprint {footprint_mib:.0f} MiB above the 250 MiB preference")
        if bench["license_risk"] == "restricted":
            missed_targets.append("weights license restricted")

        rows.append({
            "key": key,
            "engine_id": bench["engine_id"],
            "tag": bench.get("tag", ""),
            "family": bench["family"],
            "language": bench["language"],
            "bench_dir": bench_dir.name,
            "first_chunk_p95_s": first_p95,
            "first_chunk_p95_worst_case_s": first_p95_worst,
            "rtf_median_worst_case": rtf_med,
            "rtf_p95_worst_case": rtf_p95,
            "audio_gate_pass": not gate_fail_cases,
            "audio_gate_failures": all_hard,
            "gate_warnings": sorted({
                w for r in bench["runs"] for w in r["gate"]["warnings"]
            }),
            "gap_free_1_5x": gap_free,
            "micro_cer": asr.get("micro_cer"),
            "micro_wer": asr.get("micro_wer"),
            "asr_hard_failures": asr.get("hard_failures", []),
            "footprint_mib": footprint_mib,
            "load_s": bench["load"]["load_s"],
            "incremental_rss_mib": round(bench["load"]["incremental_rss_bytes"] / 1048576, 1),
            "peak_rss_mib": round(peak_rss / 1048576, 1) if peak_rss else None,
            "rss_cap_exceeded": bench["rss_cap_exceeded_any_run"],
            "license_weights": bench["license_weights"],
            "license_risk": bench["license_risk"],
            "chunk_floor_summary": bench.get("chunk_floor_summary", {}),
            "blocking_failures": blocking,
            "missed_targets": missed_targets,
            "screen_verdict": "blocked" if blocking else ("conditional" if missed_targets else "clear"),
        })

    payload = {
        "schema_version": 1,
        "gates": GATES,
        "asr_judge_calibration": asr_calibration,
        "asr_digit_rendering_cases_excluded": asr_excluded_cases,
        "note": "screen_only_naturalness_requires_blinded_listening",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.markdown:
        lines = [
            "| Candidate | Lang | First chunk P95 (s) | RTF median | RTF P95 | micro-CER | micro-WER | Footprint (MiB) | Peak RSS (MiB) | Waveform gate | Screen |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for r in sorted(rows, key=lambda x: (x["language"], x["key"])):
            lines.append(
                f"| `{r['key']}` | {r['language']} | {fmt(r['first_chunk_p95_s'], '{:.2f}')} | "
                f"{fmt(r['rtf_median_worst_case'])} | {fmt(r['rtf_p95_worst_case'])} | "
                f"{pct(r['micro_cer'])} | {pct(r['micro_wer'])} | "
                f"{r['footprint_mib']:.0f} | {fmt(r['peak_rss_mib'], '{:.0f}')} | "
                f"{'pass' if r['audio_gate_pass'] else 'FAIL'} | {r['screen_verdict']} |"
            )
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in sorted(rows, key=lambda x: (x["language"], x["key"])):
        print(
            f"{r['key']:<22} {r['language']} first_p95={fmt(r['first_chunk_p95_s'], '{:.2f}')} "
            f"rtf={fmt(r['rtf_median_worst_case'])} cer={pct(r['micro_cer'])} "
            f"wer={pct(r['micro_wer'])} -> {r['screen_verdict']}"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
