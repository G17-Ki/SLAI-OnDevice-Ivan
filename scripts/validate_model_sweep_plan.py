#!/usr/bin/env python3
"""Validate the durable multilingual model-sweep inventory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "reports/android_phone/model_sweep_20260729/model_sweep_plan.json"
REPORT = ROOT / "reports/android_phone/model_sweep_20260729/PRIORITIZED_MODEL_SWEEP.md"
STAGES = ("asr", "backbone", "tts")
REQUIRED_ROUTES = {"en", "zh", "ja", "ko", "ru", "fallback"}
EVIDENCE_CLASSES = {
    "phone_measured",
    "phone_compatibility",
    "exact_target_hub",
    "exact_target_published",
    "same_generation_proxy",
    "host_measured",
    "projected",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_evidence(paths: object, owner: str) -> None:
    require(isinstance(paths, list) and paths, f"{owner}: evidence must be non-empty")
    for raw in paths:
        require(isinstance(raw, str) and raw, f"{owner}: invalid evidence path")
        path = ROOT / raw
        require(path.is_file(), f"{owner}: evidence path does not exist: {raw}")


def validate_priorities(items: list[dict[str, object]], owner: str) -> None:
    priorities = [item.get("priority") for item in items]
    require(
        priorities == list(range(1, len(items) + 1)),
        f"{owner}: priorities must be contiguous and ordered, got {priorities}",
    )


def load_json(relative: str) -> dict[str, object]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"source JSON must be an object: {relative}")
    return value


def by_id(items: list[dict[str, object]], item_id: str) -> dict[str, object]:
    matches = [item for item in items if item.get("id") == item_id]
    require(len(matches) == 1, f"expected one item named {item_id}, got {len(matches)}")
    return matches[0]


def main() -> None:
    require(PLAN.is_file(), f"missing plan: {PLAN}")
    require(REPORT.is_file(), f"missing report: {REPORT}")
    data = json.loads(PLAN.read_text(encoding="utf-8"))

    require(data.get("schema_version") == 1, "unsupported schema_version")
    require(set(data.get("evidence_classes", {})) == EVIDENCE_CLASSES, "evidence classes drifted")
    require(
        set(data.get("target", {}).get("required_languages", [])) == REQUIRED_ROUTES,
        "required language set drifted",
    )
    require(set(data.get("routing", {})) == REQUIRED_ROUTES, "routing matrix is incomplete")

    current = data.get("current_stack")
    sweep = data.get("sweep")
    require(isinstance(current, dict), "current_stack must be an object")
    require(isinstance(sweep, dict), "sweep must be an object")
    require(set(current) == set(STAGES), "current_stack stages are incomplete")
    require(set(sweep) == set(STAGES), "sweep stages are incomplete")

    ids: set[str] = set()
    for stage in STAGES:
        current_items = current[stage]
        require(isinstance(current_items, list) and current_items, f"{stage}: empty baseline")
        for item in current_items:
            item_id = item.get("id")
            require(isinstance(item_id, str) and item_id, f"{stage}: baseline id missing")
            require(item_id not in ids, f"duplicate model id: {item_id}")
            ids.add(item_id)
            require(item.get("evidence_class") in EVIDENCE_CLASSES, f"{item_id}: bad evidence class")
            validate_evidence(item.get("evidence"), item_id)

        candidates = sweep[stage]
        require(isinstance(candidates, list) and len(candidates) >= 2, f"{stage}: need >=2 candidates")
        validate_priorities(candidates, stage)
        for candidate in candidates:
            candidate_id = candidate.get("id")
            require(isinstance(candidate_id, str) and candidate_id, f"{stage}: candidate id missing")
            require(candidate_id not in ids, f"duplicate model id: {candidate_id}")
            ids.add(candidate_id)
            require(
                candidate.get("evidence_class") in EVIDENCE_CLASSES,
                f"{candidate_id}: bad evidence class",
            )
            require(isinstance(candidate.get("measured"), dict), f"{candidate_id}: measured missing")
            require(isinstance(candidate.get("projected"), dict), f"{candidate_id}: projected missing")
            gaps = candidate.get("missing_assets_or_jobs")
            require(isinstance(gaps, list) and gaps, f"{candidate_id}: missing gap inventory")
            require(
                isinstance(candidate.get("next_action"), str) and candidate["next_action"],
                f"{candidate_id}: next_action missing",
            )
            gates = candidate.get("acceptance_gates")
            require(isinstance(gates, list) and gates, f"{candidate_id}: acceptance gates missing")
            validate_evidence(candidate.get("evidence"), candidate_id)

    jobs = data.get("job_queue")
    require(isinstance(jobs, list) and jobs, "job_queue missing")
    validate_priorities(jobs, "job_queue")
    for job in jobs:
        require(job.get("stage") in STAGES, f"job has invalid stage: {job.get('stage')}")
        require(job.get("candidate") in ids, f"job references unknown candidate: {job.get('candidate')}")
        require(job.get("heavy_mac_work") is False, "job queue must not schedule heavy Mac work")

    markdown = REPORT.read_text(encoding="utf-8")
    for heading in ("## ASR sweep", "## Backbone sweep", "## TTS sweep", "## Promotion gates"):
        require(heading in markdown, f"report is missing heading: {heading}")
    require("**PROJECTED**" in markdown, "report does not distinguish projected evidence")
    require("23/28" in markdown and "24/28" in markdown, "Qwen score discrepancy is not recorded")

    # Check a small set of decision-driving numbers against their source artifacts.
    sense_source = load_json(
        "reports/android_phone/streaming_response/phone_acceptance/npu/"
        "sensevoice_qnn_reviewed_smoke.json"
    )
    sense_plan = by_id(current["asr"], "sensevoice_small_int8_qnn_10s")
    require(
        sense_plan["phone_measurements"]["qnn_decode_ms"]["en"]
        == sense_source["runs"]["en"]["qnn_decode_ms"],
        "SenseVoice EN phone latency drifted",
    )
    require(
        sense_plan["phone_measurements"]["qnn_decode_ms"]["zh"]
        == sense_source["runs"]["zh"]["qnn_decode_ms"],
        "SenseVoice ZH phone latency drifted",
    )

    qwen_source = load_json(
        "reports/android_phone/streaming_response/qwen2_5_1_5b_q4_k_m_app_owned_phone.json"
    )
    qwen_plan = by_id(current["backbone"], "qwen2_5_1_5b_q4_k_m_cpu")
    require(
        qwen_plan["phone_measurements"]["machine_scored_passes"]
        == qwen_source["summary"]["passes"],
        "Qwen phone pass count drifted",
    )
    require(
        qwen_plan["phone_measurements"]["first_speakable_ms_p95"]
        == qwen_source["summary"]["first_speakable_ms"]["p95"],
        "Qwen phone first-speakable P95 drifted",
    )

    whisper_source = load_json(
        "reports/android_phone/streaming_response/whisper_v81/"
        "nubia_whisper_ort_qnn_probe_current.json"
    )
    whisper_plan = by_id(sweep["asr"], "whisper_small_w8a16_ort_qnn")
    require(
        whisper_plan["measured"]["phone"]["encoder_plus_four_decoder_execute_ms"]
        == whisper_source["on_device_execution"]["rerun_timing_ms"]["execute"],
        "Whisper phone execute timing drifted",
    )
    require(
        whisper_plan["artifact"]["isolated_probe_apk_bytes_including_runtime"]
        == whisper_source["apk"]["bytes"],
        "Whisper probe APK size drifted",
    )

    qai_source = load_json(
        "reports/android_phone/streaming_response/qai_hub_models_0.58.0_voice_candidates.json"
    )
    qwen3_metrics = qai_source["models"]["qwen3_1_7b"]["metrics"]
    qwen3_w4a16 = [
        metric
        for metric in qwen3_metrics
        if metric.get("precision") == "w4a16" and metric.get("desired_compute_unit") == "npu"
    ]
    require(len(qwen3_w4a16) == 1, "Qwen3-1.7B W4A16 exact-target metric is ambiguous")
    qwen3_plan = by_id(sweep["backbone"], "qwen3_1_7b_w4a16_genie_npu")
    require(
        qwen3_plan["measured"]["exact_target_published"]["response_tokens_per_second"]
        == qwen3_w4a16[0]["tokens_per_second"],
        "Qwen3-1.7B NPU response rate drifted",
    )

    tts_bundle = load_json(
        "reports/android_phone/streaming_response/tts_research_20260726/evidence/"
        "device_bundle_manifest.json"
    )
    matcha_plan = by_id(current["tts"], "matcha_zh_baker_cpu")
    require(
        matcha_plan["artifact"]["model_mib_including_shared"]
        == tts_bundle["per_candidate_mib_including_shared"]["matcha_zh_baker"]["mib"],
        "Matcha bundle size drifted",
    )

    print(
        "Model sweep plan validated: "
        f"{sum(len(sweep[stage]) for stage in STAGES)} candidates, "
        f"{len(jobs)} queued jobs, {len(ids)} total model/runtime entries"
    )


if __name__ == "__main__":
    main()
