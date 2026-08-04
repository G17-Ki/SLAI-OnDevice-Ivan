#!/usr/bin/env python3
"""What a relaxed first-audio budget actually buys.

The 0.9 s gate treats time-to-first-audio as the whole story. It is not. Two
quantities are coupled:

* **first audible time** = when the first chunk lands, plus the startup cushion;
* **queue safety** = whether later chunks arrive before the player drains.

A slow synthesizer starves the queue *only if playback starts too early*. Give
it a bigger cushion and the starvation disappears — at the cost of a later first
word. So under a 1–2 s cap the question is not "which engines are under 0.9 s"
but **"what is the smallest cushion that makes each engine gap-free, and does
the resulting first-audible time still fit the budget?"**

For chunks produced at times p_k (scaled by a host-slowdown factor) with
durations d_k, playback starting at p_0 + C stays gap-free exactly when

    C >= max_k ( p_k - p_0 - sum_{j<k} d_j )

so the minimum safe cushion is that maximum floored at zero, and the earliest
safe first-audible time is p_0 + C_min.

Results are split by case kind because they answer different questions. The
short streaming cases model an orchestrator that bounds its first flush; the
long-response case is where a slow engine's *later* chunks starve the queue, and
no first-audio budget can fix that.

This reads bench JSONs already produced; it re-runs nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GROUPS = {
    "bounded_first_flush": {"streaming_floor", "streaming_first_chunk", "answer", "prosody"},
    "long_response": {"long_response"},
}


def min_cushion_seconds(chunk_times, chunk_sizes, sample_rate: int, slowdown: float):
    """Smallest startup cushion that keeps the playback queue fed."""
    if not chunk_times or not chunk_sizes:
        return None
    produced = [t * slowdown for t in chunk_times]
    durations = [n / float(sample_rate) for n in chunk_sizes]
    deficit = 0.0
    cumulative = 0.0
    for index, p in enumerate(produced):
        deficit = max(deficit, p - produced[0] - cumulative)
        cumulative += durations[index]
    return max(0.0, deficit)


def stats(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {"n": len(clean), "median": float(statistics.median(clean)), "max": float(max(clean))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, required=True)
    ap.add_argument("--budget-s", type=float, default=2.0)
    ap.add_argument("--slowdown", type=float, action="append", default=None)
    ap.add_argument("--language", action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    slowdowns = args.slowdown or [1.0, 1.5, 2.0]
    rows = []

    for bench_dir in sorted(args.bench_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        files = list(bench_dir.glob("bench_*.json"))
        if not files:
            continue
        bench = json.loads(files[0].read_text(encoding="utf-8"))
        if args.language and bench["language"] not in args.language:
            continue
        kind_of = {cid: c.get("kind") for cid, c in bench["summary"].items()}

        groups = {}
        for group, kinds in GROUPS.items():
            per_slowdown = {}
            for slowdown in slowdowns:
                cushions, first_audible = [], []
                for run in bench["runs"]:
                    if kind_of.get(run["case_id"]) not in kinds:
                        continue
                    rate = run["gate"]["metrics"].get("sample_rate_hz")
                    if not rate:
                        continue
                    cushion = min_cushion_seconds(
                        run["chunk_times_s"], run["chunk_sizes"], rate, slowdown
                    )
                    if cushion is None:
                        continue
                    cushions.append(cushion)
                    first_audible.append(run["chunk_times_s"][0] * slowdown + cushion)
                per_slowdown[f"{slowdown:.1f}x"] = {
                    "min_cushion_s": stats(cushions),
                    "earliest_safe_first_audible_s": stats(first_audible),
                    "worst_case_fits_budget": max(first_audible) <= args.budget_s if first_audible else None,
                    "median_fits_budget": (
                        statistics.median(first_audible) <= args.budget_s if first_audible else None
                    ),
                }
            groups[group] = per_slowdown

        rows.append({
            "key": f"{bench['engine_id']}{bench.get('tag', '')}",
            "language": bench["language"],
            "family": bench["family"],
            "license_risk": bench["license_risk"],
            "groups": groups,
        })

    payload = {
        "schema_version": 1,
        "budget_s": args.budget_s,
        "case_groups": {k: sorted(v) for k, v in GROUPS.items()},
        "method": (
            "minimum startup cushion that keeps the clause-streamed playback queue fed, derived "
            "from recorded chunk callback times and sizes; earliest safe first audible = "
            "first chunk time x slowdown + that cushion"
        ),
        "caveat": "host measurement; slowdown factors are a proxy for phone CPU, not a phone measurement",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def cell(entry):
        d = entry["earliest_safe_first_audible_s"]
        if not d:
            return "-"
        return f"{d['median']:.2f} / {d['max']:.2f}" + ("" if entry["worst_case_fits_budget"] else " !")

    lines = []
    for group in GROUPS:
        lines += [
            f"### {group.replace('_', ' ')}",
            "",
            f"Earliest safe first-audible time, median / worst (s), against a {args.budget_s:g} s budget. "
            "`!` marks a worst case over budget.",
            "",
            "| Candidate | Lang | " + " | ".join(f"{s:.1f}x" for s in slowdowns)
            + " | cushion needed @1.5x (s) |",
            "|---|---|" + "---:|" * (len(slowdowns) + 1),
        ]
        for r in sorted(rows, key=lambda x: (x["language"], x["key"])):
            per = r["groups"][group]
            cushion = per.get("1.5x", {}).get("min_cushion_s")
            lines.append(
                f"| `{r['key']}` | {r['language']} | "
                + " | ".join(cell(per[f"{s:.1f}x"]) for s in slowdowns)
                + (f" | {cushion['max']:.2f} |" if cushion else " | - |")
            )
        lines.append("")

    text = "\n".join(lines)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
