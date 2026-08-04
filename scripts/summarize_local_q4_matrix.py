#!/usr/bin/env python3
"""Summarize repeated local LFM Q4 inference-technique runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


REFERENCES = {
    "question": "Can you help me come up with a slogan for my woodworking site business?",
    "asr": (
        "The stale smell of old beer lingers. It takes heat to bring out the odor. "
        "A cold dip restores health and zest. A salt pickle tastes fine with ham. "
        "Tacos al pastor are my favorite. A zestful food is the hot cross bun."
    ),
}


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for index, left in enumerate(a, 1):
        current = [index]
        for jndex, right in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[jndex] + 1,
                    previous[jndex - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def match_float(pattern: str, text: str) -> float | None:
    found = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return float(found.group(1)) if found else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem
    config, audio, run_text = stem.split("__")
    generated_match = re.search(
        r"=== GENERATED TEXT ===\s*(.*?)\s*(?:ggml_metal_free:|$)",
        text,
        re.DOTALL,
    )
    generated = generated_match.group(1).strip() if generated_match else ""
    reference = REFERENCES[audio]
    ref_words = normalize(reference)
    hyp_words = normalize(generated)
    wer = 100.0 * edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    kv_matches = [float(value) for value in re.findall(r"KV buffer size\s*=\s*([0-9.]+) MiB", text)]
    total_matches = [float(value) for value in re.findall(r"total time\s*=\s*([0-9.]+) ms", text)]
    unsupported = sorted(set(re.findall(r"^warmup:\s+([A-Z0-9_]+): type", text, re.MULTILINE)))
    cache_match = re.search(r"K \(([^)]+)\):\s*[0-9.]+ MiB, V \(([^)]+)\)", text)
    return {
        "config": config,
        "audio": audio,
        "run": int(run_text.removeprefix("run")),
        "status": "passed" if generated else "failed",
        "generated_text": generated,
        "reference_text": reference,
        "exact_match_normalized": hyp_words == ref_words,
        "wer_percent": wer,
        "cache_k": cache_match.group(1) if cache_match else None,
        "cache_v": cache_match.group(2) if cache_match else None,
        "kv_buffer_mib": kv_matches[0] if kv_matches else None,
        "audio_encode_ms": match_float(r"audio slice encoded in\s+([0-9.]+) ms", text),
        "audio_feature_decode_ms": match_float(r"audio decoded .*? in\s+([0-9.]+) ms", text),
        "prompt_tokens_per_second": match_float(
            r"^llama_perf_context_print:\s+prompt eval time.*?([0-9.]+) tokens per second", text
        ),
        "generation_tokens_per_second": match_float(
            r"^llama_perf_context_print:\s+eval time.*?([0-9.]+) tokens per second", text
        ),
        "total_model_ms": total_matches[-1] if total_matches else None,
        "wall_ms": (
            value * 1000.0
            if (value := match_float(r"^real\s+([0-9.]+)$", text)) is not None
            else None
        ),
        "projected_device_memory_mib": match_float(
            r"projected device memory[^0-9]*([0-9.]+)\s*(?:MiB|MB)", text
        ),
        "reported_text_tokens_per_second": match_float(r"text\s+tokens\s+per second:\s+([0-9.]+)", text),
        "unsupported_metal_ops": unsupported,
        "log": str(path.relative_to(path.parents[2])),
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    runs = [parse_log(path) for path in sorted(args.input_dir.glob("*.log"))]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run in runs:
        groups.setdefault((str(run["config"]), str(run["audio"])), []).append(run)

    summary = []
    metrics = (
        "wer_percent",
        "kv_buffer_mib",
        "audio_encode_ms",
        "audio_feature_decode_ms",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "total_model_ms",
        "wall_ms",
        "projected_device_memory_mib",
        "reported_text_tokens_per_second",
    )
    for (config, audio), values in sorted(groups.items()):
        row: dict[str, object] = {
            "config": config,
            "audio": audio,
            "runs": len(values),
            "passed_runs": sum(value["status"] == "passed" for value in values),
            "exact_match_runs": sum(bool(value["exact_match_normalized"]) for value in values),
            "transcripts": sorted(set(str(value["generated_text"]) for value in values if value["generated_text"])),
            "cache_k": next((value["cache_k"] for value in values if value["cache_k"]), None),
            "cache_v": next((value["cache_v"] for value in values if value["cache_v"]), None),
            "unsupported_metal_ops": sorted({op for value in values for op in value["unsupported_metal_ops"]}),
        }
        for metric in metrics:
            present = [float(value[metric]) for value in values if value[metric] is not None]
            row[f"{metric}_median"] = median(present)
            row[f"{metric}_p95"] = percentile(present, 0.95)
        summary.append(row)

    payload = {
        "scope": "Local Apple M2 diagnostic using the official Q4_0 GGUF runner; not Qualcomm or AR1 evidence.",
        "model": "LiquidAI/LFM2.5-Audio-1.5B-GGUF",
        "quantization": "Q4_0 weights",
        "context_positions": 4096,
        "runs": runs,
        "summary": summary,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    flat_rows = [
        {key: value for key, value in row.items() if key not in {"transcripts", "unsupported_metal_ops"}}
        | {
            "transcripts": " || ".join(row["transcripts"]),
            "unsupported_metal_ops": ";".join(row["unsupported_metal_ops"]),
        }
        for row in summary
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
    print(json.dumps({"runs": len(runs), "groups": len(summary), "json": str(args.json)}, indent=2))


if __name__ == "__main__":
    main()
