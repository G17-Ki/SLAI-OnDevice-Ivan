#!/usr/bin/env python3
"""Derive Voice AI TTS chunk-streaming bounds from pinned public evidence.

This is a source/scorecard audit, not an end-to-end benchmark. It reads the
installed qai-hub-models constants without importing model modules or loading
weights, then combines them with the repository's captured v81 scorecard data.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
from typing import Any


MODEL_CONFIG = {
    "pipertts_en": {
        "family": "PiperTTS",
        "language": "en",
        "constants_module": "models/_shared/pipertts/model.py",
        "app_module": "models/_shared/pipertts/app.py",
        "core_components": ("encoder", "sdp", "flow", "decoder"),
        "preprocess_components": ("charsiu_encoder", "charsiu_decoder"),
        "preprocess_mode": "optional autoregressive CharSiU G2P; the Python reference app instead uses host piper_phonemize",
    },
    "pipertts_de": {
        "family": "PiperTTS",
        "language": "de",
        "constants_module": "models/_shared/pipertts/model.py",
        "app_module": "models/_shared/pipertts/app.py",
        "core_components": ("encoder", "sdp", "flow", "decoder"),
        "preprocess_components": ("charsiu_encoder", "charsiu_decoder"),
        "preprocess_mode": "optional autoregressive CharSiU G2P; the Python reference app instead uses host piper_phonemize",
    },
    "pipertts_it": {
        "family": "PiperTTS",
        "language": "it",
        "constants_module": "models/_shared/pipertts/model.py",
        "app_module": "models/_shared/pipertts/app.py",
        "core_components": ("encoder", "sdp", "flow", "decoder"),
        "preprocess_components": ("charsiu_encoder", "charsiu_decoder"),
        "preprocess_mode": "optional autoregressive CharSiU G2P; the Python reference app instead uses host piper_phonemize",
    },
    "melotts_es": {
        "family": "MeloTTS",
        "language": "es",
        "constants_module": "models/_shared/melotts/model.py",
        "app_module": "models/_shared/melotts/app.py",
        "core_components": ("encoder", "flow", "decoder"),
        "preprocess_components": ("t5_encoder", "t5_decoder"),
        "preprocess_mode": "autoregressive T5 G2P; decoder call count depends on the text",
    },
    "melotts_zh": {
        "family": "MeloTTS",
        "language": "zh",
        "constants_module": "models/_shared/melotts/model.py",
        "app_module": "models/_shared/melotts/app.py",
        "core_components": ("encoder", "flow", "decoder"),
        "preprocess_components": ("bert_wrapper",),
        "preprocess_mode": "one fixed-shape BERT wrapper call in the published component inventory",
    },
}

REQUIRED_CONSTANTS = (
    "SAMPLE_RATE",
    "MAX_DEC_SEQ_LEN",
    "DEC_SEQ_OVERLAP",
    "UPSAMPLE_FACTOR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "reports/android_phone/streaming_response/"
            "qai_hub_models_0.58.0_voice_candidates.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-qaihm-version",
        default="0.58.0",
        help="Fail if the installed qai-hub-models package differs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_root() -> Path:
    spec = util.find_spec("qai_hub_models")
    if spec is None or spec.origin is None:
        raise RuntimeError("qai-hub-models is not installed")
    return Path(spec.origin).resolve().parent


def literal_constants(path: Path, names: Iterable[str]) -> dict[str, int]:
    wanted = set(names)
    values: dict[str, int] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, int):
            raise TypeError(f"{target.id} in {path} is not an integer literal")
        values[target.id] = value
    missing = sorted(wanted - values.keys())
    if missing:
        raise ValueError(f"Missing constants in {path}: {missing}")
    return values


def component_map(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = model.get("metrics")
    if not isinstance(metrics, list):
        raise TypeError(f"Invalid metrics for {model.get('name')}")
    result = {}
    for metric in metrics:
        component = metric.get("component")
        if isinstance(component, str):
            result[component] = metric
    return result


def sum_latency(components: dict[str, dict[str, Any]], names: Iterable[str]) -> float:
    return sum(float(components[name]["latency_ms"]) for name in names)


def strict_npu(components: dict[str, dict[str, Any]], names: Iterable[str]) -> bool:
    for name in names:
        metric = components[name]
        layers = metric.get("layer_counts", {})
        if metric.get("primary_compute_unit") != "NPU" or layers.get(
            "total"
        ) != layers.get("npu"):
            return False
    return True


def main() -> None:
    args = parse_args()
    installed_version = metadata.version("qai-hub-models")
    if installed_version != args.expected_qaihm_version:
        raise RuntimeError(
            f"Expected qai-hub-models {args.expected_qaihm_version}, "
            f"found {installed_version}"
        )

    inventory_path = args.inventory.expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    models = inventory.get("models")
    if not isinstance(models, dict):
        raise TypeError("Inventory has no model mapping")

    root = package_root()
    families: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}

    for key, config in MODEL_CONFIG.items():
        constants_relative = str(config["constants_module"])
        app_relative = str(config["app_module"])
        constants_path = root / constants_relative
        app_path = root / app_relative
        family = str(config["family"])
        if family not in families:
            constants = literal_constants(constants_path, REQUIRED_CONSTANTS)
            chunk_samples = constants["MAX_DEC_SEQ_LEN"] * constants["UPSAMPLE_FACTOR"]
            chunk_ms = chunk_samples / constants["SAMPLE_RATE"] * 1000.0
            families[family] = {
                "constants": constants,
                "decoder_input_frames": constants["MAX_DEC_SEQ_LEN"]
                + 2 * constants["DEC_SEQ_OVERLAP"],
                "emitted_frames_per_call": constants["MAX_DEC_SEQ_LEN"],
                "emitted_samples_per_call": chunk_samples,
                "emitted_audio_ms_per_call": chunk_ms,
                "source": {
                    "constants_module": constants_relative,
                    "constants_sha256": sha256(constants_path),
                    "app_module": app_relative,
                    "app_sha256": sha256(app_path),
                },
            }

        model = models.get(key)
        if not isinstance(model, dict):
            raise KeyError(f"Inventory is missing {key}")
        components = component_map(model)
        core_names = tuple(config["core_components"])
        preprocess_names = tuple(config["preprocess_components"])
        missing = [
            name for name in (*core_names, *preprocess_names) if name not in components
        ]
        if missing:
            raise KeyError(f"{key} is missing components: {missing}")

        family_record = families[family]
        chunk_ms = float(family_record["emitted_audio_ms_per_call"])
        decoder_ms = float(components["decoder"]["latency_ms"])
        core_ms = sum_latency(components, core_names)
        preprocess_fixed_ms = 0.0
        preprocess_per_token_ms = None
        if preprocess_names:
            preprocess_fixed_ms = float(components[preprocess_names[0]]["latency_ms"])
            if len(preprocess_names) == 2:
                preprocess_per_token_ms = float(
                    components[preprocess_names[1]]["latency_ms"]
                )

        results[key] = {
            "name": model.get("name"),
            "family": family,
            "language": config["language"],
            "evidence_class": components["decoder"].get("evidence_class"),
            "device": components["decoder"].get("device"),
            "precision": components["decoder"].get("precision"),
            "core_components": list(core_names),
            "core_to_first_decoder_output_ms": core_ms,
            "preprocess_components": list(preprocess_names),
            "preprocess_mode": config["preprocess_mode"],
            "preprocess_fixed_call_ms": preprocess_fixed_ms,
            "preprocess_decoder_per_token_ms": preprocess_per_token_ms,
            "one_preprocess_step_plus_core_ms": core_ms
            + preprocess_fixed_ms
            + (preprocess_per_token_ms or 0.0),
            "decoder_call_ms": decoder_ms,
            "decoder_audio_payload_ms": chunk_ms,
            "decoder_compute_to_audio_ratio": decoder_ms / chunk_ms,
            "decoder_audio_realtime_multiple": chunk_ms / decoder_ms,
            "all_listed_components_strict_npu": strict_npu(
                components, (*core_names, *preprocess_names)
            ),
            "component_job_ids": {
                name: components[name].get("job_id")
                for name in (*core_names, *preprocess_names)
            },
        }

    payload = {
        "schema_version": 1,
        "analysis_type": "source_and_published_scorecard_derived_bounds",
        "qai_hub_models_version": installed_version,
        "inventory_sha256": sha256(inventory_path),
        "families": families,
        "models": results,
        "interpretation": {
            "proves": [
                "The pinned reference decoder boundary yields one fixed audio payload per call.",
                "Published same-generation v81 decoder compute is faster than the audio payload duration.",
                "The reference app can be refactored to emit each decoder result instead of concatenating all chunks before writing a WAV.",
            ],
            "does_not_prove": [
                "Nubia or glasses text-to-first-audible latency",
                "Android runtime/package compatibility",
                "host preprocessing, model loading, transfer, scheduling, or speaker startup latency",
                "audio quality or numerical equivalence on the exact QRD",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
