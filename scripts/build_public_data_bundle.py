#!/usr/bin/env python3
"""Create a credential-free, path-free data appendix for the public report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    components = load_json(project / "reports/aihub_component_results.json")
    component_rows = []
    for row in components["results"]:
        comparisons = row.get("comparisons", {})
        component_rows.append(
            {
                "component": row["component"],
                "runtime": row["runtime"],
                "status": row["status"],
                "device": "QCS8550 (Proxy), Android 12, Hexagon v73",
                "latency_ms": row.get("latency_ms"),
                "peak_memory_mib": row.get("peak_memory_mb"),
                "npu_runtime_layers": row.get("npu_runtime_layers"),
                "cpu_fallback_runtime_layers": row.get("cpu_fallback_runtime_layers"),
                "other_runtime_layers": row.get("other_runtime_layers"),
                "output_checks_passed": all(value.get("passed", False) for value in comparisons.values()),
            }
        )
    write_csv(args.output_dir / "lfm_component_results.csv", component_rows)

    quality = load_json(project / "reports/colab_candidate_quality/summary.json")
    moonshine = load_json(project / "reports/moonshine_quality/summary.json")
    quality_rows = []
    for row in [*quality["summary"], *moonshine["summary"]]:
        quality_rows.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "samples": row["samples"],
                "audio_seconds": row["audio_seconds"],
                "wer_percent": row["wer_percent"],
                "measurement_hardware": "Apple M2 MPS" if row["model"].startswith("UsefulSensors/") else "NVIDIA L4",
            }
        )
    write_csv(args.output_dir / "asr_quality_diagnostic.csv", quality_rows)

    memory = load_json(project / "reports/memory/memory_ledger.json")
    memory_rows = []
    for row in memory["rows"]:
        memory_rows.append(
            {
                "category": row["category"],
                "item": row["item"],
                "bytes": row["bytes"],
                "mib": row["mib"],
                "source_class": "measured local file size" if row["category"] == "measured file" else row["source"],
            }
        )
    write_csv(args.output_dir / "memory_ledger.csv", memory_rows)

    comparator_rows = []
    with (project / "reports/comparators/qai_comparator_matrix.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            comparator_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"job_id"}
                }
            )
    write_csv(args.output_dir / "qualcomm_comparator_components.csv", comparator_rows)

    quant = load_json(project / "reports/local_q4_matrix/summary.json")
    quant_run_rows = []
    for row in quant["runs"]:
        cache, flash = {
            "f16_fa_on": ("f16", "on"),
            "q8_fa_on": ("q8_0", "on"),
            "q4_fa_on": ("q4_0", "on"),
            "f16_fa_off": ("f16", "off"),
        }.get(row["config"], (row.get("cache_k"), "unknown"))
        quant_run_rows.append(
            {
                "config_id": row["config"],
                "sample_id": row["audio"],
                "repeat": row["run"],
                "weight_quant": "Q4_0",
                "cache_type_k": row.get("cache_k") or cache,
                "cache_type_v": row.get("cache_v") or cache,
                "flash_attention": flash,
                "backend": "Apple M2 Metal with documented encoder fallbacks",
                "status": row["status"],
                "generated_text": row["generated_text"],
                "reference_text": row["reference_text"],
                "exact_match_normalized": row["exact_match_normalized"],
                "wer_percent": row["wer_percent"],
                "projected_device_memory_mib": row.get("projected_device_memory_mib"),
                "kv_buffer_mib": row.get("kv_buffer_mib"),
                "audio_encode_ms": row.get("audio_encode_ms"),
                "audio_decode_ms": row.get("audio_feature_decode_ms"),
                "prompt_tokens_per_second": row.get("prompt_tokens_per_second"),
                "generation_tokens_per_second": row.get("generation_tokens_per_second"),
                "total_model_ms": row.get("total_model_ms"),
                "wall_ms": row.get("wall_ms"),
                "fallback_ops": ";".join(row.get("unsupported_metal_ops", [])),
            }
        )
    write_csv(args.output_dir / "q4_inference_runs.csv", quant_run_rows)

    quant_summary_rows = []
    for row in quant["summary"]:
        quant_summary_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"transcripts", "unsupported_metal_ops"}
            }
            | {
                "fallback_ops": ";".join(row.get("unsupported_metal_ops", [])),
            }
        )
    write_csv(args.output_dir / "q4_inference_summary.csv", quant_summary_rows)

    report_md = project / "reports/public/LFM2.5_Audio_Glasses_Feasibility_Report.md"
    report_pdf = project / "output/pdf/LFM2.5_Audio_Glasses_Feasibility_Report.pdf"
    expected_files = {
        "lfm_component_results.csv",
        "asr_quality_diagnostic.csv",
        "memory_ledger.csv",
        "qualcomm_comparator_components.csv",
        "q4_inference_runs.csv",
        "q4_inference_summary.csv",
    }
    manifest = {
        "scope": "Credential-free public report data. No model weights, OAuth files, API tokens, private paths, or raw datasets are included.",
        "model_revision": "c362a0625dfe45aa588dce5f0ada28a7e5707628",
        "liquid_audio_commit": "19e65845923a7f136442c95137884ec61eb386aa",
        "qai_hub_models_version": "0.57.3",
        "report_sha256": {
            "markdown": sha256(report_md),
            "pdf": sha256(report_pdf),
        },
        "evidence_sha256": {
            name: sha256(args.output_dir / name)
            for name in sorted(expected_files)
        },
        "files": sorted(expected_files | {"manifest.json"}),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
