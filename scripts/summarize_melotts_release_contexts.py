#!/usr/bin/env python3
"""Merge per-component MeloTTS release-context profile states safely."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


COMPONENTS = ("encoder", "flow", "decoder", "bert_wrapper")
NEURAL_AUDIO_COMPONENTS = ("encoder", "flow", "decoder")


def parse_state_argument(value: str) -> tuple[str, Path]:
    component, separator, raw_path = value.partition("=")
    if not separator or component not in COMPONENTS or not raw_path:
        raise argparse.ArgumentTypeError(
            "state must be COMPONENT=PATH where COMPONENT is one of "
            + ", ".join(COMPONENTS)
        )
    return component, Path(raw_path)


def load_component(component: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state = json.loads(path.read_text(encoding="utf-8"))
    item = state.get("components", {}).get(component)
    if not isinstance(item, dict):
        raise ValueError(f"{path} does not contain component {component!r}")
    return state, item


def summarize(states: dict[str, Path]) -> dict[str, Any]:
    if set(states) != set(COMPONENTS):
        missing = sorted(set(COMPONENTS) - set(states))
        extra = sorted(set(states) - set(COMPONENTS))
        raise ValueError(f"expected exactly {COMPONENTS}; missing={missing}, extra={extra}")

    target = None
    release = None
    components: dict[str, Any] = {}
    for component in COMPONENTS:
        state, item = load_component(component, states[component])
        if target is None:
            target = state.get("target")
            release = state.get("release")
        elif state.get("target") != target or state.get("release") != release:
            raise ValueError("component states disagree on target or release provenance")

        metrics = item.get("metrics") or {}
        placement = item.get("placement") or {}
        components[component] = {
            "status": item.get("status"),
            "artifact": item.get("artifact"),
            "model_id": item.get("model_id"),
            "profile_job_id": item.get("profile_job_id"),
            "latency_ms": metrics.get("latency_ms"),
            "estimated_inference_peak_mib": metrics.get("peak_memory_mib"),
            "operation_count": placement.get("operation_count"),
            "placement_counts": placement.get("counts"),
            "strict_npu_pass": placement.get("strict_npu_pass"),
        }

    all_pass = all(
        components[name]["status"] == "profile_passed_strict_npu"
        and components[name]["strict_npu_pass"] is True
        for name in COMPONENTS
    )
    audio_pass = all(
        components[name]["status"] == "profile_passed_strict_npu"
        and components[name]["strict_npu_pass"] is True
        for name in NEURAL_AUDIO_COMPONENTS
    )

    def sum_field(names: tuple[str, ...], field: str) -> float | int | None:
        values = [components[name][field] for name in names]
        if any(value is None for value in values):
            return None
        return sum(values)

    return {
        "schema_version": 1,
        "release": release,
        "target": target,
        "evidence_scope": {
            "proves": [
                "context portability from sm8850-ad to exact sm8850 QRD",
                "profile latency and estimated peak memory",
                "operator placement",
            ],
            "does_not_prove": [
                "numerical equivalence",
                "Mandarin intelligibility or naturalness",
                "Nubia or glasses runtime behavior",
                "end-to-end first-audio latency",
            ],
        },
        "components": components,
        "summary": {
            "all_components_strict_npu_pass": all_pass,
            "audio_components_strict_npu_pass": audio_pass,
            "audio_component_latency_sum_ms": sum_field(
                NEURAL_AUDIO_COMPONENTS, "latency_ms"
            ),
            "all_component_latency_sum_ms": sum_field(COMPONENTS, "latency_ms"),
            "audio_component_operation_count": sum_field(
                NEURAL_AUDIO_COMPONENTS, "operation_count"
            ),
            "all_component_operation_count": sum_field(COMPONENTS, "operation_count"),
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=parse_state_argument,
        action="append",
        required=True,
        help="Repeat as COMPONENT=PATH for encoder, flow, decoder, bert_wrapper.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state_map = dict(args.state)
    if len(state_map) != len(args.state):
        raise SystemExit("duplicate --state component")
    payload = summarize(state_map)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
