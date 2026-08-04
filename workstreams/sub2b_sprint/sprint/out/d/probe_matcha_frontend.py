#!/usr/bin/env python3
"""Capture Matcha's mixed-script OOV behavior without writing audio."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path


OUT_D = Path(__file__).resolve().parent
SYNTH_SCRIPT = OUT_D / "artifacts" / "matcha" / "synthesize_matcha_sets.py"
OUTPUT = OUT_D / "artifacts" / "matcha" / "frontend_oov_probe.json"
D5_RISK_INDICES = (2, 9, 11, 12, 13, 14)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def child() -> int:
    spec = importlib.util.spec_from_file_location("matcha_synth", SYNTH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    corpus = load_json(OUT_D / "D1_corpus.json")
    prompts = load_json(OUT_D / "D5_prompts.json")
    cases = []
    for item in corpus["items"]:
        text = item["texts"]["zh"]
        if re.search(r"[A-Za-z]", text):
            cases.append({"suite": "D1", "id": item["item_id"], "text": text})
    for prompt in prompts["languages"]["zh"]:
        if int(prompt["listen_index"]) in D5_RISK_INDICES:
            cases.append(
                {
                    "suite": "D5",
                    "id": f"zh_{int(prompt['listen_index']):02d}",
                    "text": prompt["text"],
                }
            )
    tts, load_s = module.build_tts()
    rows = []
    for case in cases:
        record, samples, sample_rate = module.synthesize(tts, case["text"])
        rows.append(
            {
                **case,
                "sample_rate_hz": sample_rate,
                "sample_count": int(samples.size),
                "nonfinite_count": record["nonfinite_count"],
                "passed_waveform_assertions": record["passed"],
            }
        )
    print(json.dumps({"model_load_s": load_s, "cases": rows}, ensure_ascii=False))
    return 0


def parent() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Matcha OOV child failed ({result.returncode}):\n{result.stderr}"
        )
    child_payload = json.loads(result.stdout)
    stderr_lines = [line for line in result.stderr.splitlines() if line.strip()]
    ignored = []
    for line in stderr_lines:
        match = re.search(r"Ignore OOV '([^']*)'", line)
        if match and match.group(1) not in ignored:
            ignored.append(match.group(1))
    unknown_tokens = []
    for line in stderr_lines:
        match = re.search(r"Unknown token: (\S+)", line)
        if match and match.group(1) not in unknown_tokens:
            unknown_tokens.append(match.group(1))
    payload = {
        "schema_version": 1,
        "evidence_class": "HOST",
        "run_started_utc": started,
        "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Capture sherpa-onnx Matcha frontend OOV diagnostics on frozen D1/D5 mixed-script text without writing WAVs.",
        "runtime": {
            "python": sys.version.split()[0],
            "child_returncode": result.returncode,
            "model_load_s": child_payload["model_load_s"],
        },
        "ignored_oov_tokens_in_first_seen_order": ignored,
        "lexicon_unknown_tokens": unknown_tokens,
        "stderr_lines": stderr_lines,
        "cases": child_payload["cases"],
        "conclusion": (
            "The frontend explicitly ignores unsupported tokens while still returning "
            "finite PCM; waveform success therefore does not prove mixed-script coverage."
        ),
    }
    assert all(row["passed_waveform_assertions"] for row in payload["cases"])
    assert {"Wild", "Card", "SANParks", "CPU", "RAM", "GB", "API", "JSON", "VPN"}.issubset(
        set(ignored)
    )
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")
    print("ignored OOV:", ", ".join(ignored))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    return child() if args.child else parent()


if __name__ == "__main__":
    raise SystemExit(main())
