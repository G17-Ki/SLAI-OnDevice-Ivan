#!/usr/bin/env python3
"""How long a first flush of N characters costs, and how long it may be.

`bench.py --flush-sweep` synthesizes single segments of graded length. First
chunk latency turns out to be close to affine in the flush length, so a
least-squares fit over the sweep gives two directly useful numbers per
candidate: the fixed per-call cost, and the marginal cost per character.

Inverting the fit answers the question an integrator actually has — **"how many
characters may the first flush contain and still hit the budget?"** — which is
the one knob the orchestrator controls.

The fit is reported with its residual so a candidate that is *not* affine is
visible rather than silently extrapolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit(chars, seconds):
    """Least-squares affine fit with the residual that justifies trusting it."""
    x = np.asarray(chars, dtype=np.float64)
    y = np.asarray(seconds, dtype=np.float64)
    if x.size < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "fixed_cost_s": float(intercept),
        "seconds_per_char": float(slope),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "max_abs_residual_s": float(np.max(np.abs(y - predicted))),
    }


def max_flush_chars(model, budget_s: float, slowdown: float):
    """Largest first flush whose first chunk still lands inside the budget."""
    if not model or model["seconds_per_char"] <= 0:
        return None
    host_budget = budget_s / slowdown
    chars = (host_budget - model["fixed_cost_s"]) / model["seconds_per_char"]
    return int(chars) if chars > 0 else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, required=True)
    ap.add_argument("--budget-s", type=float, action="append", default=None)
    ap.add_argument("--slowdown", type=float, action="append", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    budgets = args.budget_s or [1.0, 2.0]
    slowdowns = args.slowdown or [1.0, 1.5]

    rows = []
    for bench_dir in sorted(args.bench_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        files = list(bench_dir.glob("bench_*.json"))
        if not files:
            continue
        bench = json.loads(files[0].read_text(encoding="utf-8"))
        sweep = bench.get("flush_sweep_summary") or {}
        if not sweep:
            continue
        points = sorted(
            (v["chars"], v["first_chunk"]["p95"], v["first_chunk"]["median"])
            for v in sweep.values()
            if v["first_chunk"]["p95"] is not None
        )
        model = fit([p[0] for p in points], [p[1] for p in points])
        rows.append({
            "key": f"{bench['engine_id']}{bench.get('tag', '')}",
            "language": bench["language"],
            "points": [{"chars": c, "first_chunk_p95_s": p, "first_chunk_median_s": m}
                       for c, p, m in points],
            "fit": model,
            "max_flush_chars": {
                f"{b:g}s@{s:g}x": max_flush_chars(model, b, s)
                for b in budgets for s in slowdowns
            },
        })

    payload = {
        "schema_version": 1,
        "method": "affine least-squares fit of first-chunk P95 against flush length in characters",
        "caveat": "host measurement; slowdown is a proxy for phone CPU, not a phone measurement",
        "budgets_s": budgets,
        "slowdowns": slowdowns,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    keys = [f"{b:g}s@{s:g}x" for b in budgets for s in slowdowns]
    lines = [
        "Largest first flush (characters) whose first chunk still lands inside the budget, "
        "from an affine fit of the flush sweep.",
        "",
        "| Candidate | Lang | fixed cost (s) | s/char | R² | " + " | ".join(keys) + " |",
        "|---|---|---:|---:|---:|" + "---:|" * len(keys),
    ]
    for r in sorted(rows, key=lambda x: (x["language"], x["key"])):
        f = r["fit"]
        lines.append(
            f"| `{r['key']}` | {r['language']} | {f['fixed_cost_s']:.3f} | {f['seconds_per_char']:.4f} | "
            f"{f['r_squared']:.4f} | "
            + " | ".join(str(r["max_flush_chars"][k]) for k in keys) + " |"
        )
    text = "\n".join(lines)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
