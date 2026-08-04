#!/usr/bin/env python3
"""Convert the initial Kokoro evidence file to strict JSON without rerunning TTS.

The first benchmark serializer emitted bare NaN for undefined statistics on an
all-non-finite Mandarin warm-up. This mechanical repair preserves every
measurement, replaces only undefined scalar statistics with null, adds an
explicit status/failure record, and writes with allow_nan=False.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def repair_record(record: dict[str, object]) -> None:
    nonfinite_count = int(record.get("nonfinite_count", 0))
    num_samples = int(record.get("num_samples", 0))
    record["audio_status"] = (
        "ok"
        if nonfinite_count == 0
        else "all_nonfinite"
        if nonfinite_count == num_samples
        else "contains_nonfinite"
    )
    for key in ("peak_abs", "rms", "dc_mean"):
        value = record.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            record[key] = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_json", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.benchmark_json.read_text())
    for record in [*payload.get("warmups", []), *payload.get("runs", [])]:
        repair_record(record)

    failures = []
    for warmup in payload.get("warmups", []):
        if int(warmup.get("nonfinite_count", 0)) == 0:
            continue
        failures.append(
            {
                key: warmup[key]
                for key in (
                    "case_id",
                    "lang",
                    "text",
                    "audio_status",
                    "num_samples",
                    "nonfinite_count",
                    "pcm_f32_sha256",
                )
            }
        )
    payload["warmup_failures"] = failures
    payload["measured_output_gate"] = {
        "all_measured_case_outputs_finite": all(
            int(run.get("nonfinite_count", 0)) == 0 for run in payload.get("runs", [])
        ),
        "all_measured_case_outputs_unclipped": all(
            int(run.get("clip_count", 0)) == 0 for run in payload.get("runs", [])
        ),
    }
    args.benchmark_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
