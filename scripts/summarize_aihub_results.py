#!/usr/bin/env python3
"""Create compact JSON/Markdown summaries from all downloaded AI Hub runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from path_config import REPO_ROOT, WORK_ROOT

AIHUB_ROOT = WORK_ROOT / "lfm-feasibility/aihub"
REPORT_ROOT = REPO_ROOT / "reports"


def compact_comparisons(comparisons: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "comparison",
        "passed",
        "max_abs_error",
        "mean_abs_error",
        "p95_abs_error",
        "p99_abs_error",
        "normalized_rmse",
        "cosine_similarity",
    )
    return {
        name: {field: value.get(field) for field in fields if field in value}
        for name, value in comparisons.items()
    }


def compact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    placement = data.get("placement") or {}
    metrics = data.get("metrics") or {}
    return {
        "component": path.parent.name,
        "runtime": path.stem.removeprefix("summary-"),
        "status": data.get("status"),
        "source": str(path),
        "jobs": data.get("jobs", {}),
        "compile_message": (data.get("jobs", {}).get("compile") or {}).get("message", ""),
        "strict_npu_pass": data.get("strict_npu_pass"),
        "truncate_64bit_io": data.get("truncate_64bit_io"),
        "remote_input_dtypes": data.get("remote_input_dtypes"),
        "latency_ms": metrics.get("latency_ms"),
        "peak_memory_mb": metrics.get("peak_memory_mb"),
        "npu_runtime_layers": placement.get("npu_runtime_layers"),
        "cpu_fallback_runtime_layers": placement.get("cpu_fallback_runtime_layers"),
        "other_runtime_layers": placement.get("other_runtime_layers"),
        "comparisons": compact_comparisons(data.get("comparisons") or {}),
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def comparison_note(comparisons: dict[str, Any]) -> str:
    notes = []
    for name, value in comparisons.items():
        if value.get("comparison") == "exact":
            notes.append(f"{name}: exact={value.get('passed')}")
        else:
            note = f"{name}: pass={value.get('passed')}"
            if value.get("max_abs_error") is not None:
                note += f", max={value['max_abs_error']:.4g}"
            if value.get("cosine_similarity") is not None:
                note += f", cos={value['cosine_similarity']:.6f}"
            notes.append(note)
    return "; ".join(notes) or "-"


def main() -> None:
    latest: dict[tuple[str, str], Path] = {}
    for path in AIHUB_ROOT.glob("**/summary-*.json"):
        key = (path.parent.name, path.stem.removeprefix("summary-"))
        if key not in latest or path.stat().st_mtime_ns > latest[key].stat().st_mtime_ns:
            latest[key] = path

    rows = [compact(path) for path in latest.values()]
    rows.sort(key=lambda row: (row["component"], row["runtime"]))
    payload = {
        "target_note": (
            "Runs use AI Hub QCS8550 (Proxy), Android 12, Hexagon v73. "
            "Placement is transfer evidence; latency and memory are not AR1 measurements."
        ),
        "results": rows,
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "aihub_component_results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    lines = [
        "# Qualcomm AI Hub component results",
        "",
        payload["target_note"],
        "",
        "| Component | Runtime | Status | NPU | CPU | Other | Latency ms | Peak MB | Golden comparison |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["component"],
                    row["runtime"],
                    str(row["status"]),
                    format_number(row["npu_runtime_layers"], 0),
                    format_number(row["cpu_fallback_runtime_layers"], 0),
                    format_number(row["other_runtime_layers"], 0),
                    format_number(row["latency_ms"]),
                    format_number(row["peak_memory_mb"]),
                    comparison_note(row["comparisons"]),
                ]
            )
            + " |"
        )
    (REPORT_ROOT / "aihub_component_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
