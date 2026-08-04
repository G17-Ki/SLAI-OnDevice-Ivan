#!/usr/bin/env python3
"""Extract latency/placement evidence for a low-latency multilingual voice stack."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml


MODELS = (
    "whisper_tiny",
    "whisper_base",
    "whisper_small_quantized",
    "zipformer",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "pipertts_en",
    "pipertts_de",
    "pipertts_it",
    "melotts_en",
    "melotts_es",
    "melotts_zh",
)
DEFAULT_TARGET = "Snapdragon 8 Elite Gen 5 QRD"
DEFAULT_SAME_GENERATION_FALLBACK = "Samsung Galaxy S26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="qai-hub-models wheel to inspect")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--same-generation-fallback", default=DEFAULT_SAME_GENERATION_FALLBACK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_yaml(archive: zipfile.ZipFile, model: str, filename: str) -> dict[str, Any]:
    path = f"qai_hub_models/models/{model}/{filename}"
    return yaml.safe_load(archive.read(path))


def select_device(
    metrics: dict[str, Any], target: str, fallback: str
) -> tuple[str | None, str | None]:
    if target in metrics:
        return target, "exact_target"
    if fallback in metrics:
        return fallback, "same_generation_retail_proxy"
    return None, None


def summarize_component_metrics(
    perf: dict[str, Any], target: str, fallback: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for precision, precision_data in (perf.get("precisions") or {}).items():
        for component, component_data in (precision_data.get("components") or {}).items():
            metrics = component_data.get("performance_metrics") or {}
            device, evidence_class = select_device(metrics, target, fallback)
            if device is None:
                continue
            for runtime, runtime_data in (metrics.get(device) or {}).items():
                if not isinstance(runtime_data, dict):
                    continue
                llm_metrics = runtime_data.get("llm_metrics")
                if llm_metrics:
                    for llm_metric in llm_metrics:
                        rows.append(
                            {
                                "precision": precision,
                                "component": component,
                                "device": device,
                                "evidence_class": evidence_class,
                                "runtime": runtime,
                                "kind": "llm",
                                **llm_metric,
                            }
                        )
                    continue
                rows.append(
                    {
                        "precision": precision,
                        "component": component,
                        "device": device,
                        "evidence_class": evidence_class,
                        "runtime": runtime,
                        "kind": "component",
                        "job_id": runtime_data.get("job_id"),
                        "job_status": runtime_data.get("job_status"),
                        "latency_ms": runtime_data.get("inference_time_milliseconds"),
                        "estimated_peak_memory_range_mb": runtime_data.get(
                            "estimated_peak_memory_range_mb"
                        ),
                        "primary_compute_unit": runtime_data.get("primary_compute_unit"),
                        "layer_counts": runtime_data.get("layer_counts"),
                        "tool_versions": runtime_data.get("tool_versions"),
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    if not args.wheel.is_file():
        raise SystemExit(f"Wheel does not exist: {args.wheel}")

    models: dict[str, Any] = {}
    with zipfile.ZipFile(args.wheel) as archive:
        for model in MODELS:
            info = read_yaml(archive, model, "info.yaml")
            perf = read_yaml(archive, model, "perf.yaml")
            models[model] = {
                "name": info.get("name"),
                "status": info.get("status"),
                "headline": info.get("headline"),
                "description": info.get("description"),
                "license_type": info.get("license_type"),
                "license": info.get("license"),
                "source_repo": info.get("source_repo"),
                "technical_details": info.get("technical_details"),
                "runtime_technical_details": info.get("runtime_technical_details"),
                "supported_devices": perf.get("supported_devices"),
                "supported_chipsets": perf.get("supported_chipsets"),
                "metrics": summarize_component_metrics(
                    perf, args.target, args.same_generation_fallback
                ),
            }

    result = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "wheel": args.wheel.name,
            "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
            "target": args.target,
            "same_generation_fallback": args.same_generation_fallback,
        },
        "interpretation": {
            "exact_target": "Published metric is for the requested QRD.",
            "same_generation_retail_proxy": (
                "The requested QRD is listed as supported, but the wheel publishes the metric under "
                "a same-generation retail device. It is not an exact QRD measurement."
            ),
            "component_latency": (
                "Fixed-shape neural invocation latency, not application end-to-end latency. Front-end, "
                "tokenization, orchestration, host transfers, audio I/O, and repeated decoding are excluded."
            ),
        },
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
