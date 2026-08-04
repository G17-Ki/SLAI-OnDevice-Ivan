#!/usr/bin/env python3
"""Extract comparable QCS8550 speech metrics from an official QAI Hub Models wheel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import yaml


MODELS = (
    "zipformer",
    "whisper_tiny",
    "whisper_base",
    "whisper_small",
    "whisper_small_quantized",
    "pipertts_en",
    "melotts_en",
    "llama_v3_2_1b_instruct",
    "qwen3_0_6b",
    "qwen3_5_0_8b",
)
RUNTIME_PREFERENCE = (
    "qnn_context_binary",
    "voice_ai",
    "precompiled_qnn_onnx",
    "genie",
    "geniex_qairt",
    "geniex_llamacpp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(wheel: Path) -> str:
    match = re.search(r"qai_hub_models-([0-9][^-]*)-", wheel.name)
    return match.group(1) if match else "unknown"


def choose_runtime(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for runtime in RUNTIME_PREFERENCE:
        if runtime in metrics:
            return runtime, metrics[runtime]
    return next(iter(metrics.items()), None)


def extract_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    models_root = root / "qai_hub_models" / "models"
    for model_id in MODELS:
        model_root = models_root / model_id
        if not model_root.is_dir():
            continue
        info = yaml.safe_load((model_root / "info.yaml").read_text(encoding="utf-8"))
        perf = yaml.safe_load((model_root / "perf.yaml").read_text(encoding="utf-8"))
        metadata[model_id] = {
            "name": info.get("name"),
            "description": info.get("description"),
            "technical_details": info.get("technical_details", {}),
            "license_type": info.get("license_type"),
            "source_repo": info.get("source_repo"),
            "research_paper": info.get("research_paper"),
            "supported_devices": perf.get("supported_devices", []),
        }
        for precision, precision_data in perf.get("precisions", {}).items():
            for component, component_data in precision_data.get("components", {}).items():
                device_metrics = component_data.get("performance_metrics", {}).get(
                    "QCS8550 (Proxy)", {}
                )
                selected = choose_runtime(device_metrics)
                if selected is None:
                    continue
                runtime, values = selected
                layers = values.get("layer_counts", {})
                memory = values.get("estimated_peak_memory_range_mb", {})
                rows.append(
                    {
                        "model_id": model_id,
                        "model_name": info.get("name"),
                        "component": component,
                        "precision": precision,
                        "runtime": runtime,
                        "device": "QCS8550 (Proxy)",
                        "latency_ms": values.get("inference_time_milliseconds"),
                        "peak_memory_min_mb": memory.get("min"),
                        "peak_memory_max_mb": memory.get("max"),
                        "ops_total": layers.get("total"),
                        "ops_npu": layers.get("npu"),
                        "job_id": values.get("job_id"),
                        "job_status": values.get("job_status"),
                        "qairt_version": values.get("tool_versions", {}).get("qairt"),
                    }
                )
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.wheel.is_file():
        raise FileNotFoundError(args.wheel)

    with tempfile.TemporaryDirectory(prefix="qai-comparators-") as temp_dir:
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(temp_dir)
        rows, models = extract_rows(Path(temp_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "qai_comparator_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "source": {
            "package": "qai-hub-models",
            "version": package_version(args.wheel),
            "wheel_sha256": sha256(args.wheel),
            "wheel_filename": args.wheel.name,
            "package_url": "https://pypi.org/project/qai-hub-models/",
            "catalog_url": "https://aihub.qualcomm.com/models/",
            "extracted_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        },
        "interpretation": {
            "device": "QCS8550 (Proxy), not AR1/AR1+",
            "latency": "Per fixed-shape component invocation, not end-to-end pipeline latency.",
            "memory": "AI Hub estimated peak runtime range; do not treat as resident weight or total app memory.",
            "placement": "ops_npu == ops_total is strict full-NPU placement evidence for that component.",
        },
        "models": models,
        "rows": rows,
    }
    (args.output_dir / "qai_comparator_matrix.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
