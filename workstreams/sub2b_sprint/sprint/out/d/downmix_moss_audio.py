#!/usr/bin/env python3
"""Convert retained Lane G MOSS WAVs to disclosed mono PCM16 and refresh receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

import run_moss_g


OUT_D = Path(__file__).resolve().parent
CHECKPOINT = OUT_D / "artifacts" / "moss" / "synthesis_records.jsonl"


def key(row: dict) -> tuple[str, str, int, int]:
    return row["suite"], row["case_id"], int(row["repeat"]), int(row["seed"])


def main() -> int:
    rows = [
        json.loads(line)
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latest = {key(row): row for row in rows}
    converted = 0
    for row in latest.values():
        output = row.get("output")
        if not output:
            continue
        path = OUT_D / output["path"]
        values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if values.shape[1] > 1:
            mono = values.mean(axis=1, dtype=np.float32)
            sf.write(path, mono, sample_rate, subtype="PCM_16")
            converted += 1
        refreshed = run_moss_g.file_waveform_record(path)
        if not refreshed["passed"]:
            raise SystemExit(f"downmixed waveform failed gate: {path}")
        row["output"] = refreshed
        row["serialization"] = {
            "source": "raw 48 kHz stereo decoder tensor",
            "operation": "arithmetic mean across two channels",
            "retained_format": "mono PCM16 WAV",
        }

    temp = CHECKPOINT.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row_key in sorted(latest):
            stream.write(
                json.dumps(latest[row_key], ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, CHECKPOINT)
    print(json.dumps({"records": len(latest), "converted": converted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
