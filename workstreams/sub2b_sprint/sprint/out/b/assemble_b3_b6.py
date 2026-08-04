#!/usr/bin/env python3
"""Assemble Lane B's B3--B6 reports from frozen harness/scorer artifacts.

The script is deliberately an assembler, not an evaluator: llama.cpp raw rows
and receipts must already exist, and native generations must already have been
scored by ``score_native_generations.py`` with strict dependency pins.  It
validates those artifacts against the frozen B1 protocol, combines the 80 MCQ
and 40 native-generation items into the registered per-language score, and
applies the pre-registered backbone rule without a cross-language average.

Expected artifact layout is flexible.  Under ``--artifacts-dir`` the script
discovers the exact harness filenames recursively:

* ``global_mmlu.raw.jsonl`` and its sibling ``.receipt.json``;
* ``native_generation.raw.jsonl`` and its sibling ``.receipt.json``;
* each model's complete rendered directory (both JSONLs, MCQ binary and receipt);
* the copied ``build_receipt.json`` and receipted runtime logs; and
* native score JSON reports with schema ``lane-b-native-score/1.0``.

Score reports may be supplied explicitly with repeated ``--score-report``
arguments.  A report may contain one arm or all four arms.  Production output
always uses the frozen 10,000 bootstrap resamples and seed 20260801.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "lane-b-b3-b6-assembly/1.0"
EXPECTED_PREREG_SHA256 = "6107258882449913b9136ad9eba930e88952192b6df08a301eafc00f03f6255b"
LLAMA_CPP_COMMIT = "11924d4c17abc27383376a1ac6a24fa3e36c1c0c"
EXPECTED_SCORER_SHA256 = "8fe0dff8b3c081d26a55358ad39340700a57876cbeecdf4b7dccc3116b4290ca"
PRE_RESOURCE_CORRECTION_HARNESS_SHA256 = "d7d2077148a11f2b0242fb0889642a5c4c9c31579c638fe13ae7753ccae21b37"
FIRST_RESOURCE_CORRECTION_HARNESS_SHA256 = "82e65de893d047378af9e6c702aaec8d2ca29fe89b1ecef6caba273b204c097c"
FALLBACK_RESOURCE_CORRECTION_HARNESS_SHA256 = "0ddb1e5d24d76abb2c2f42338cb18b311407b9e2bf69b4200ee01ba8d1ef76e2"
PRE_BUILD_PROVENANCE_HARNESS_SHA256 = "9f99a37eeabe6be36f27e8342e98805c4dbd88d0e202e457f291b998a0b726e7"
EXPECTED_HARNESS_SHA256 = "3e9b03db7da7ce3e126a4fed80959a2db2b579a59a87cd859c784b9c496c9c2a"
EXPECTED_SETUP_SHA256 = "0372a681cfc3aaf885586d8b5f64fce1abd7ab7c790f88f45dc80a717bb013ca"
LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH = "tools/perplexity/perplexity.cpp"
LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256 = "380aadd9d8000bf1b95cc3a338708385ac75240a5ba1c1f2bb34b208892a1777"
LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256 = "65051abf6eb3e68d26cb2f24b62e2fa1af1bf7ddea96a2db53f944c3f0232b78"
LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH = "llama_mcq_memory.patch"
LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256 = "2539f2da48852fa44fdacdba26154e144a01b3986ff7c88dfe6974ac3e4f9dd7"
LLAMA_CPP_EXPECTED_SOURCE_STATUS = " M tools/perplexity/perplexity.cpp"
LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH = "bin/libllama-perplexity-impl.so"
LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_SHA256 = "2adce2e89998bbcbf220ee92af3ac5b91d8d080712db7cfc7cb7809dbd33f714"
LLAMA_CPP_PERPLEXITY_LAUNCHER_SHA256 = "3407a0ae6bb48bd300799b965fbe937f4b06a040ff9dcea7917e88cc239dc255"
MODEL_IDS = ("qwen35_08b", "qwen25_15b", "qwen3_06b", "falcon_h1_15b")
B3_MODEL_IDS = ("qwen35_08b", "qwen25_15b", "qwen3_06b")
B4_MODEL_IDS = ("falcon_h1_15b",)
LANGUAGES = ("en", "zh", "es", "fr", "ru")
DECISION_LANGUAGES = ("es", "fr", "ru")
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_RESAMPLES = 10_000
MODE_PIN_VOID_RATE = 0.005
COMPETENCE_THRESHOLD = Fraction(3, 4)
MCQ_EXECUTION_BATCHING = {
    "ctx_size": 4096,
    "batch_size": 1024,
    "ubatch_size": 128,
    "parallel": 4,
}
FALLBACK_MCQ_EXECUTION_BATCHING = {
    "ctx_size": 3072,
    "batch_size": 1024,
    "ubatch_size": 128,
    "parallel": 4,
}
PRE_INFERENCE_EXECUTION_RECORDS = {
    "b1_execution_clarifications.json": {
        "schema_version": "lane-b-execution-clarification/1.0",
        "sha256": "81192513f12a4e0046c38532fea94870bc28661e3fc7e0894408ca7d3a4f0895",
    },
    "b1_execution_clarification_correction.json": {
        "schema_version": "lane-b-execution-clarification-correction/1.0",
        "sha256": "b2b31d80f9a1a3aaf545e310d7fa5300b48eb78a79693f058e0f6622acfa7f64",
    },
    "b1_execution_clarification_addendum.json": {
        "schema_version": "lane-b-execution-clarification-addendum/1.0",
        "sha256": "cba9f25342edf3c9e7260a9e1cf3194e60e221fdfbe09026d5062358acd055fe",
    },
}
RESOURCE_CORRECTION_RECORD = {
    "name": "b1_execution_resource_correction.json",
    "schema_version": "lane-b-execution-resource-correction/1.0",
    "sha256": "c2528fb2e7595de5dab2ba8389b18ac9f917882fc9cc3fea31d70894043c70e1",
}
RESOURCE_CORRECTION_FALLBACK_RECORD = {
    "name": "b1_execution_resource_correction_fallback.json",
    "schema_version": "lane-b-execution-resource-correction-fallback/1.0",
    "sha256": "a4c69997baf1c999275f4ca358aafff98e4e9c311afe44a96b0208d5433a27b0",
}
TOOL_MEMORY_PATCH_RECORD = {
    "name": "b1_execution_tool_memory_patch.json",
    "schema_version": "lane-b-execution-tool-memory-patch/1.0",
    "sha256": "3a2dcd9aa0ef39f400202ed97f5110bfec3823d58d18eceb5b5e37c621a92192",
}
BUILD_PROVENANCE_ADDENDUM_RECORD = {
    "name": "b1_execution_build_provenance_addendum.json",
    "schema_version": "lane-b-execution-build-provenance-addendum/1.0",
    "sha256": "343e5f6e3000e42dcb6752f4c40f35aa1d45fd166e2a14ae20e8f08fe6ea659e",
}
EXECUTION_RECORD_ORDER = (
    "b1_execution_clarifications.json",
    "b1_execution_clarification_correction.json",
    "b1_execution_clarification_addendum.json",
    RESOURCE_CORRECTION_RECORD["name"],
    RESOURCE_CORRECTION_FALLBACK_RECORD["name"],
    TOOL_MEMORY_PATCH_RECORD["name"],
    BUILD_PROVENANCE_ADDENDUM_RECORD["name"],
)
SCRIPT_DIR = Path(__file__).resolve().parent
_PINNED_SCORER: Any = None
_PINNED_HARNESS: Any = None


class AssemblyError(RuntimeError):
    """A frozen input, receipt, score report, or decision invariant failed."""


@dataclass(frozen=True)
class FrozenInputs:
    lane_dir: Path
    prereg: dict[str, Any]
    prereg_sha256: str
    model_configs: dict[str, dict[str, Any]]
    mmlu_rows: list[dict[str, Any]]
    native_rows: list[dict[str, Any]]
    mmlu_by_id: dict[str, dict[str, Any]]
    native_by_id: dict[str, dict[str, Any]]
    mmlu_sha256: str
    native_sha256: str
    execution_records: list[dict[str, Any]]


@dataclass(frozen=True)
class BundlePaths:
    model_id: str
    mcq_jsonl: Path
    mcq_receipt: Path
    generation_jsonl: Path
    generation_receipt: Path
    render_receipt: Path
    mmlu_rendered_jsonl: Path
    generation_rendered_jsonl: Path
    multiple_choice_binary: Path
    build_receipt: Path
    score_report: Path


@dataclass
class ModelEvidence:
    model_id: str
    mcq_rows: list[dict[str, Any]]
    generation_rows: list[dict[str, Any]]
    score_rows: list[dict[str, Any]]
    source_hashes: dict[str, str]
    render_receipt: dict[str, Any]
    mcq_receipt: dict[str, Any]
    generation_receipt: dict[str, Any]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssemblyError(f"cannot import pinned implementation {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as exc:
        raise AssemblyError(f"cannot import pinned implementation {path}: {exc}") from exc
    return module


def load_pinned_scorer(lane_dir: Path) -> Any:
    global _PINNED_SCORER
    path = lane_dir / "score_native_generations.py"
    _require(path.is_file(), f"missing committed scorer implementation {path}")
    actual_sha = sha256_file(path)
    _require(actual_sha == EXPECTED_SCORER_SHA256, f"scorer implementation hash mismatch: expected {EXPECTED_SCORER_SHA256}, got {actual_sha}")
    if _PINNED_SCORER is None:
        module = _load_module_from_path("lane_b_pinned_native_scorer", path)
        _require(module.SCHEMA_VERSION == "lane-b-native-score/1.0", "pinned scorer schema changed")
        try:
            module._dependency_versions(True)
        except Exception as exc:
            raise AssemblyError(f"pinned scorer dependency/Python validation failed: {exc}") from exc
        _PINNED_SCORER = module
    return _PINNED_SCORER


def load_pinned_harness(lane_dir: Path) -> Any:
    global _PINNED_HARNESS
    path = lane_dir / "b34_harness.py"
    _require(path.is_file(), f"missing committed harness implementation {path}")
    actual_sha = sha256_file(path)
    _require(actual_sha == EXPECTED_HARNESS_SHA256, f"harness implementation hash mismatch: expected {EXPECTED_HARNESS_SHA256}, got {actual_sha}")
    if _PINNED_HARNESS is None:
        module = _load_module_from_path("lane_b_pinned_b34_harness", path)
        _require(module.LLAMA_CPP_COMMIT == LLAMA_CPP_COMMIT, "pinned harness llama.cpp commit changed")
        _require(module.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH, "pinned harness MCQ source path changed")
        _require(module.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256 == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256, "pinned harness original MCQ source hash changed")
        _require(module.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256 == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256, "pinned harness patched MCQ source hash changed")
        _require(module.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH, "pinned harness MCQ patch path changed")
        _require(module.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256 == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256, "pinned harness MCQ patch hash changed")
        _require(module.LLAMA_CPP_EXPECTED_SOURCE_STATUS == LLAMA_CPP_EXPECTED_SOURCE_STATUS, "pinned harness patched source status changed")
        _require(
            {
                "ctx_size": module.MCQ_CONTEXT_TOKENS,
                "batch_size": module.MCQ_BATCH_SIZE,
                "ubatch_size": module.MCQ_UBATCH_SIZE,
                "parallel": module.MCQ_PARALLEL,
            }
            == MCQ_EXECUTION_BATCHING,
            "pinned harness final MCQ batching changed",
        )
        _PINNED_HARNESS = module
    return _PINNED_HARNESS


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"expected a JSON object in {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise AssemblyError(f"cannot open JSONL {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssemblyError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise AssemblyError(f"expected object at {path}:{line_number}")
            yield row


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def _require_number(value: Any, label: str, *, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AssemblyError(f"{label} must be a finite number")
    result = float(value)
    if low is not None and result < low:
        raise AssemblyError(f"{label}={result} is below {low}")
    if high is not None and result > high:
        raise AssemblyError(f"{label}={result} is above {high}")
    return result


def _manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(iter_jsonl(path))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in by_id:
            raise AssemblyError(f"{path} has missing or duplicate item_id {item_id!r}")
        by_id[item_id] = row
    return rows, by_id


def _validate_execution_record(path: Path, prereg_sha256: str) -> dict[str, Any]:
    record = load_json(path)
    name = path.name
    resource_name = RESOURCE_CORRECTION_RECORD["name"]
    fallback_name = RESOURCE_CORRECTION_FALLBACK_RECORD["name"]
    tool_patch_name = TOOL_MEMORY_PATCH_RECORD["name"]
    build_addendum_name = BUILD_PROVENANCE_ADDENDUM_RECORD["name"]
    allowed_names = set(PRE_INFERENCE_EXECUTION_RECORDS) | {resource_name, fallback_name, tool_patch_name, build_addendum_name}
    _require(name in allowed_names, f"unexpected append-only execution record: {name}")
    if name == resource_name:
        expected = RESOURCE_CORRECTION_RECORD
    elif name == fallback_name:
        expected = RESOURCE_CORRECTION_FALLBACK_RECORD
    elif name == tool_patch_name:
        expected = TOOL_MEMORY_PATCH_RECORD
    elif name == build_addendum_name:
        expected = BUILD_PROVENANCE_ADDENDUM_RECORD
    else:
        expected = PRE_INFERENCE_EXECUTION_RECORDS[name]
    _require(record.get("schema_version") == expected["schema_version"], f"{name} schema changed")
    _require(record.get("sprint_id") == "B1", f"{name} sprint id changed")

    parent = record.get("parent_preregistration")
    if isinstance(parent, dict):
        _require(Path(str(parent.get("path", ""))).name == "b1_prereg.json", f"{name} preregistration path changed")
        _require(parent.get("sha256") == prereg_sha256, f"{name} does not bind to the frozen preregistration")

    correction = record.get("corrects")
    if isinstance(correction, dict) and isinstance(correction.get("path"), str) and isinstance(correction.get("sha256"), str):
        corrected_name = Path(correction["path"]).name
        corrected_path = path.parent / corrected_name
        _require(corrected_path.is_file(), f"{name} corrects missing record {corrected_name}")
        _require(sha256_file(corrected_path) == correction["sha256"], f"{name} correction parent hash mismatch")

    parent_records = record.get("parent_records", [])
    _require(isinstance(parent_records, list), f"{name} parent_records must be a list")
    for parent_record in parent_records:
        _require(isinstance(parent_record, dict), f"{name} has malformed parent record")
        parent_name = Path(str(parent_record.get("path", ""))).name
        parent_sha = parent_record.get("sha256")
        parent_path = path.parent / parent_name
        _require(parent_name and parent_path.is_file(), f"{name} references missing parent record {parent_name!r}")
        _require(sha256_file(parent_path) == parent_sha, f"{name} parent record hash mismatch for {parent_name}")

    post_attempt_names = {resource_name, fallback_name, tool_patch_name}
    if name in post_attempt_names:
        _require(record.get("recorded_before_candidate_inference") is False, f"{name} must disclose that it followed a failed candidate attempt")
        _require(record.get("recorded_after_failed_unscored_candidate_attempt") is True, f"{name} failed-attempt timing disclosure changed")
        _require(record.get("candidate_result_rows_observed") == 0, f"{name} is inadmissible after any candidate result row")
        _require(parent is not None, f"{name} must bind the frozen preregistration")

    if name == resource_name:
        expected_parents = [
            {"path": parent_name, "sha256": details["sha256"]}
            for parent_name, details in PRE_INFERENCE_EXECUTION_RECORDS.items()
        ]
        _require(parent_records == expected_parents, f"{name} parent record chain changed")
        failed = record.get("failed_attempt")
        _require(isinstance(failed, dict), f"{name} failed-attempt evidence missing")
        _require(failed.get("model_id") == "qwen35_08b" and failed.get("run_kind") == "global_mmlu_log_likelihood", f"{name} failed-attempt identity changed")
        _require(failed.get("previous_harness_source_sha256") == PRE_RESOURCE_CORRECTION_HARNESS_SHA256, f"{name} previous harness hash changed")
        _require(failed.get("returncode") == -9, f"{name} failed-attempt return code changed")
        _require(failed.get("raw_jsonl_created") is False, f"{name} may not follow a created raw result JSONL")
        _require(failed.get("cumulative_accuracy_rows_observed") == 0, f"{name} may not follow any cumulative result row")
        _require(failed.get("log_artifact") == "failures/qwen35_08b.mcq.batch4096.parallel4.sigkill.log", f"{name} failure-log path changed")
        _require(failed.get("log_sha256") == "d63d8bf77af8c0353236e3c3abca580736daf034fad5eeb7367ba6864b1e150e", f"{name} failure-log hash changed")
        _require(failed.get("log_bytes") == 854, f"{name} failure-log size changed")
        corrected = record.get("corrected_mcq_execution")
        _require(isinstance(corrected, dict), f"{name} corrected execution missing")
        _require(corrected.get("applies_identically_to_models") == list(MODEL_IDS), f"{name} does not apply identically to all four arms")
        _require(
            {
                "ctx_size": corrected.get("context_tokens"),
                "batch_size": corrected.get("batch_size"),
                "ubatch_size": corrected.get("ubatch_size"),
                "parallel": corrected.get("parallel"),
            }
            == MCQ_EXECUTION_BATCHING,
            f"{name} corrected batching changed",
        )
        _require(corrected.get("gpu_layers") == 999 and corrected.get("seed") == 1729, f"{name} invariant MCQ settings changed")
        _require(corrected.get("largest_observed_qwen35_task_sequence_tokens") == 577, f"{name} task-token audit changed")
        _require(corrected.get("largest_observed_qwen35_required_batch_tokens") == 580, f"{name} required-batch-token audit changed")
        _require(580 <= MCQ_EXECUTION_BATCHING["batch_size"], f"{name} corrected batch no longer contains the largest task in one decode")
    elif name == fallback_name:
        _require(parent_records == [{"path": resource_name, "sha256": RESOURCE_CORRECTION_RECORD["sha256"]}], f"{name} correction-chain order changed")
        failed = record.get("failed_attempt")
        _require(isinstance(failed, dict), f"{name} failed-attempt evidence missing")
        _require(failed.get("model_id") == "qwen35_08b" and failed.get("run_kind") == "global_mmlu_log_likelihood", f"{name} failed-attempt identity changed")
        _require(failed.get("previous_harness_source_sha256") == FIRST_RESOURCE_CORRECTION_HARNESS_SHA256, f"{name} previous harness hash changed")
        _require(failed.get("returncode") == -9, f"{name} failed-attempt return code changed")
        _require(failed.get("raw_jsonl_created") is False, f"{name} may not follow a created raw result JSONL")
        _require(failed.get("cumulative_accuracy_rows_observed") == 0, f"{name} may not follow any cumulative result row")
        _require(
            {
                "ctx_size": failed.get("command", {}).get("context_tokens"),
                "batch_size": failed.get("command", {}).get("batch_size"),
                "ubatch_size": failed.get("command", {}).get("ubatch_size"),
                "parallel": failed.get("command", {}).get("parallel"),
            }
            == MCQ_EXECUTION_BATCHING,
            f"{name} failed command differs from the first correction",
        )
        _require(failed.get("log_artifact") == "failures/qwen35_08b.mcq.ctx4096.batch1024.ubatch128.parallel4.sigkill.log", f"{name} failure-log path changed")
        _require(failed.get("log_sha256") == "c6a4e4eac96c54be575f2dcfe0236210c2c15d53233884b6e6bbcdc7c22f5f51", f"{name} failure-log hash changed")
        _require(failed.get("log_bytes") == 854, f"{name} failure-log size changed")
        corrected = record.get("corrected_mcq_execution")
        _require(isinstance(corrected, dict), f"{name} corrected execution missing")
        _require(corrected.get("applies_identically_to_models") == list(MODEL_IDS), f"{name} does not apply identically to all four arms")
        _require(
            {
                "ctx_size": corrected.get("context_tokens"),
                "batch_size": corrected.get("batch_size"),
                "ubatch_size": corrected.get("ubatch_size"),
                "parallel": corrected.get("parallel"),
            }
            == FALLBACK_MCQ_EXECUTION_BATCHING,
            f"{name} corrected batching changed",
        )
        _require(corrected.get("gpu_layers") == 999 and corrected.get("seed") == 1729, f"{name} invariant MCQ settings changed")
        _require(corrected.get("largest_observed_qwen35_task_sequence_tokens") == 577, f"{name} task-token audit changed")
        _require(corrected.get("per_sequence_context_tokens") == 768, f"{name} per-sequence context audit changed")
        _require(corrected.get("fixed_qwen35_logits_allocation_bytes") == 3_051_279_360, f"{name} logits-allocation audit changed")
        _require(FALLBACK_MCQ_EXECUTION_BATCHING["ctx_size"] // FALLBACK_MCQ_EXECUTION_BATCHING["parallel"] == 768 > 577, f"{name} fallback context no longer contains the largest sequence")
    elif name == tool_patch_name:
        _require(parent_records == [{"path": fallback_name, "sha256": RESOURCE_CORRECTION_FALLBACK_RECORD["sha256"]}], f"{name} correction-chain order changed")
        failed = record.get("failed_attempt")
        _require(isinstance(failed, dict), f"{name} failed-attempt evidence missing")
        _require(failed.get("model_id") == "qwen35_08b" and failed.get("run_kind") == "global_mmlu_log_likelihood", f"{name} failed-attempt identity changed")
        _require(failed.get("previous_harness_source_sha256") == FALLBACK_RESOURCE_CORRECTION_HARNESS_SHA256, f"{name} previous harness hash changed")
        _require(failed.get("returncode") == -9, f"{name} failed-attempt return code changed")
        _require(failed.get("raw_jsonl_created") is False, f"{name} may not follow a created raw result JSONL")
        _require(failed.get("cumulative_accuracy_rows_observed") == 0, f"{name} may not follow any cumulative result row")
        _require(
            {
                "ctx_size": failed.get("command", {}).get("context_tokens"),
                "batch_size": failed.get("command", {}).get("batch_size"),
                "ubatch_size": failed.get("command", {}).get("ubatch_size"),
                "parallel": failed.get("command", {}).get("parallel"),
            }
            == FALLBACK_MCQ_EXECUTION_BATCHING,
            f"{name} failed command differs from the fallback correction",
        )
        _require(failed.get("log_artifact") == "failures/qwen35_08b.mcq.ctx3072.batch1024.ubatch128.parallel4.sigkill.log", f"{name} failure-log path changed")
        _require(failed.get("log_sha256") == "10338b94d465016699295e5031a2671cea2661434ab40ca0b7303a385e16d2fd", f"{name} failure-log hash changed")
        _require(failed.get("log_bytes") == 854, f"{name} failure-log size changed")
        audited = record.get("audited_tool_patch")
        _require(isinstance(audited, dict), f"{name} audited tool patch missing")
        _require(audited.get("applies_identically_to_models") == list(MODEL_IDS), f"{name} patch does not apply identically to all four arms")
        _require(audited.get("llama_cpp_commit") == LLAMA_CPP_COMMIT, f"{name} patch commit changed")
        _require(audited.get("source_relative_path") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH, f"{name} patch source path changed")
        _require(audited.get("source_original_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256, f"{name} original source hash changed")
        _require(audited.get("patch_path") == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH, f"{name} patch path changed")
        _require(audited.get("patch_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256, f"{name} patch hash changed")
        _require(audited.get("source_patched_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256, f"{name} patched source hash changed")
        final = record.get("final_mcq_execution")
        _require(isinstance(final, dict), f"{name} final execution missing")
        _require(
            {
                "ctx_size": final.get("context_tokens"),
                "batch_size": final.get("batch_size"),
                "ubatch_size": final.get("ubatch_size"),
                "parallel": final.get("parallel"),
            }
            == MCQ_EXECUTION_BATCHING,
            f"{name} final batching changed",
        )
        _require(final.get("gpu_layers") == 999 and final.get("seed") == 1729, f"{name} invariant final MCQ settings changed")
        _require(final.get("harness_source_sha256") == PRE_BUILD_PROVENANCE_HARNESS_SHA256, f"{name} historical final harness hash changed")
        _require(final.get("setup_script_sha256") == EXPECTED_SETUP_SHA256, f"{name} setup-script hash changed")
        impl = final.get("patched_impl_library_receipt")
        _require(isinstance(impl, dict), f"{name} patched implementation-library receipt missing")
        _require(impl.get("relative_path") == LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH, f"{name} implementation-library path changed")
        _require(impl.get("sha256") == LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_SHA256, f"{name} implementation-library hash changed")
        _require(impl.get("launcher_sha256") == LLAMA_CPP_PERPLEXITY_LAUNCHER_SHA256, f"{name} llama-perplexity launcher hash changed")
    elif name == build_addendum_name:
        _require(record.get("recorded_before_candidate_inference") is False, f"{name} must disclose its post-run timing")
        _require(record.get("recorded_after_candidate_inference") is True, f"{name} post-run timing disclosure changed")
        _require(record.get("candidate_result_rows_observed") == 600, f"{name} must bind the complete 400+200 superseded arm")
        _require(parent is not None, f"{name} must bind the frozen preregistration")
        _require(parent_records == [{"path": tool_patch_name, "sha256": TOOL_MEMORY_PATCH_RECORD["sha256"]}], f"{name} correction-chain order changed")
        triggering = record.get("triggering_audit")
        _require(isinstance(triggering, dict), f"{name} triggering audit is missing")
        _require(triggering.get("risk_class") == "build-provenance ambiguity, not an observed inference or scoring defect", f"{name} audit risk classification changed")
        superseded = record.get("superseded_candidate_run")
        _require(isinstance(superseded, dict), f"{name} superseded run is missing")
        _require(superseded.get("model_id") == "qwen35_08b", f"{name} superseded model changed")
        _require(superseded.get("harness_source_sha256") == PRE_BUILD_PROVENANCE_HARNESS_SHA256, f"{name} superseded harness hash changed")
        _require(superseded.get("build_receipt_sha256") == "43dbaec478a92ae39b5436bde34d2f81813470e2815757e0903d4b16b2ae2ceb", f"{name} superseded build receipt hash changed")
        old_mcq = superseded.get("global_mmlu")
        _require(isinstance(old_mcq, dict), f"{name} superseded MCQ evidence is missing")
        _require(old_mcq.get("n") == 400 and old_mcq.get("correct") == 150, f"{name} superseded MCQ row/count declaration changed")
        _require(old_mcq.get("raw_jsonl_sha256") == "9d33083195a9e97a3351b0be36c0dfb5c7e00495d1901bed3f8a46d412e8c80d", f"{name} superseded MCQ raw hash changed")
        _require(old_mcq.get("receipt_sha256") == "f6b15576c69ffe88f2504f38c942db2e9b117fd6d3f0166944a67eda1fd01f33", f"{name} superseded MCQ receipt hash changed")
        old_generation = superseded.get("native_generation")
        _require(isinstance(old_generation, dict), f"{name} superseded generation evidence is missing")
        _require(old_generation.get("n") == 200, f"{name} superseded generation row count changed")
        _require(old_generation.get("raw_jsonl_sha256") == "9adccd075fbd1851712c2980a88d87fe624bc841ebf568b9cc08442788f7a19e", f"{name} superseded generation raw hash changed")
        _require(old_generation.get("receipt_sha256") == "69fcf5078e6bc4b13589774461203e62a3f47e63865a36c79c25a21846423ff1", f"{name} superseded generation receipt hash changed")
        archive = superseded.get("preserved_archive")
        _require(isinstance(archive, dict), f"{name} superseded archive declaration is missing")
        _require(archive.get("path") == "superseded/9f99a37e/archives/qwen35_08b.evidence.tar.gz", f"{name} superseded archive path changed")
        _require(archive.get("sha256") == "14c49cafc005206763efe9c1300cacd55056a529b2ed431b58b7b30cab3b265d", f"{name} superseded archive hash changed")
        _require(superseded.get("excluded_from_final_analysis") is True, f"{name} does not exclude the superseded run")
        hardened = record.get("hardened_execution")
        _require(isinstance(hardened, dict), f"{name} hardened execution is missing")
        _require(hardened.get("applies_identically_to_models") == list(MODEL_IDS), f"{name} hardened harness does not apply to all four arms")
        _require(hardened.get("harness_source_sha256") == EXPECTED_HARNESS_SHA256, f"{name} hardened harness hash changed")
        _require(hardened.get("setup_script_sha256") == EXPECTED_SETUP_SHA256, f"{name} hardened setup hash changed")
        _require(hardened.get("llama_cpp_commit") == LLAMA_CPP_COMMIT, f"{name} hardened llama.cpp commit changed")
        _require(hardened.get("patch_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256, f"{name} hardened patch hash changed")
        _require(hardened.get("patched_source_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256, f"{name} hardened source hash changed")
        required_cmake = hardened.get("required_cmake_configuration")
        _require(isinstance(required_cmake, dict), f"{name} required CMake configuration is missing")
        _require(required_cmake.get("CMAKE_BUILD_TYPE") == "Release" and required_cmake.get("GGML_CUDA") == "ON" and required_cmake.get("BUILD_SHARED_LIBS") == "ON", f"{name} required CMake settings changed")
        hardened_impl = hardened.get("implementation_library")
        _require(isinstance(hardened_impl, dict), f"{name} hardened implementation-library receipt is missing")
        _require(hardened_impl.get("relative_path") == LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH, f"{name} hardened implementation-library path changed")
        _require(hardened_impl.get("sha256") == LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_SHA256, f"{name} hardened implementation-library hash changed")
        final = hardened.get("final_mcq_execution")
        _require(isinstance(final, dict), f"{name} hardened final MCQ execution is missing")
        _require(
            {
                "ctx_size": final.get("context_tokens"),
                "batch_size": final.get("batch_size"),
                "ubatch_size": final.get("ubatch_size"),
                "parallel": final.get("parallel"),
            }
            == MCQ_EXECUTION_BATCHING,
            f"{name} hardened final batching changed",
        )
        _require(final.get("gpu_layers") == 999 and final.get("seed") == 1729, f"{name} hardened invariant MCQ settings changed")
    else:
        _require(record.get("recorded_before_candidate_inference") is True, f"execution record was not frozen before inference: {name}")

    digest = sha256_file(path)
    _require(digest == expected["sha256"], f"{name} hash changed: expected {expected['sha256']}, got {digest}")
    return {
        "path": name,
        "sha256": digest,
        "schema_version": record.get("schema_version"),
        "recorded_at_utc": record.get("recorded_at_utc"),
        "recorded_before_candidate_inference": record.get("recorded_before_candidate_inference"),
        "recorded_after_failed_unscored_candidate_attempt": record.get("recorded_after_failed_unscored_candidate_attempt", False),
        "recorded_after_candidate_inference": record.get("recorded_after_candidate_inference", False),
        "candidate_result_rows_observed": record.get("candidate_result_rows_observed"),
        "parent_records": parent_records,
    }


def _validate_archived_execution_evidence(
    lane_dir: Path,
    record_paths: Sequence[Path],
    execution_records: list[dict[str, Any]],
) -> None:
    metadata_by_name = {row["path"]: row for row in execution_records}
    for specification in (
        RESOURCE_CORRECTION_RECORD,
        RESOURCE_CORRECTION_FALLBACK_RECORD,
        TOOL_MEMORY_PATCH_RECORD,
    ):
        record_path = next(path for path in record_paths if path.name == specification["name"])
        record = load_json(record_path)
        failed = record["failed_attempt"]
        recorded_log = Path(str(failed["log_artifact"]))
        local_log = lane_dir / "artifacts" / "failures" / recorded_log.name
        _require(local_log.is_file(), f"missing archived failed-attempt log {local_log}")
        _require(local_log.stat().st_size == failed["log_bytes"], f"failed-attempt log byte count changed: {local_log.name}")
        actual_sha = sha256_file(local_log)
        _require(actual_sha == failed["log_sha256"], f"failed-attempt log hash changed: {local_log.name}")
        text = local_log.read_text(encoding="utf-8", errors="replace")
        cumulative_rows = 0
        for line in text.splitlines():
            clean_line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            if re.fullmatch(r"\s*\d+\s+[0-9]+(?:\.[0-9]+)?\s*", clean_line):
                cumulative_rows += 1
        _require(cumulative_rows == 0, f"failed-attempt log contains {cumulative_rows} cumulative candidate-result rows: {local_log.name}")
        metadata_by_name[specification["name"]]["failed_attempt_log_evidence"] = {
            "path": str(local_log.relative_to(lane_dir)),
            "sha256": actual_sha,
            "bytes": local_log.stat().st_size,
            "cumulative_candidate_result_rows": cumulative_rows,
        }

    addendum_path = next(path for path in record_paths if path.name == BUILD_PROVENANCE_ADDENDUM_RECORD["name"])
    addendum = load_json(addendum_path)
    archive = addendum["superseded_candidate_run"]["preserved_archive"]
    local_archive = lane_dir / "artifacts" / str(archive["path"])
    _require(local_archive.is_file(), f"missing superseded Qwen3.5 evidence archive {local_archive}")
    archive_sha = sha256_file(local_archive)
    _require(archive_sha == archive["sha256"], "superseded Qwen3.5 evidence archive hash changed")
    metadata_by_name[BUILD_PROVENANCE_ADDENDUM_RECORD["name"]]["superseded_evidence_archive"] = {
        "path": str(local_archive.relative_to(lane_dir)),
        "sha256": archive_sha,
        "excluded_from_final_analysis": True,
    }


def load_frozen_inputs(lane_dir: Path) -> FrozenInputs:
    load_pinned_scorer(lane_dir)
    load_pinned_harness(lane_dir)
    prereg_path = lane_dir / "b1_prereg.json"
    prereg_sha_path = lane_dir / "b1_prereg.sha256"
    mmlu_path = lane_dir / "protocol" / "global_mmlu_80x5.jsonl"
    native_path = lane_dir / "protocol" / "native_generation_40x5.jsonl"
    setup_path = lane_dir / "b34_colab_setup.sh"
    patch_path = lane_dir / LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH
    for path in (prereg_path, prereg_sha_path, mmlu_path, native_path, setup_path, patch_path):
        _require(path.is_file(), f"missing frozen input {path}")
    _require(sha256_file(setup_path) == EXPECTED_SETUP_SHA256, "Colab setup script hash changed")
    _require(sha256_file(patch_path) == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256, "audited llama.cpp MCQ patch hash changed")

    registered_sha = prereg_sha_path.read_text(encoding="utf-8").split()[0]
    actual_sha = sha256_file(prereg_path)
    _require(registered_sha == EXPECTED_PREREG_SHA256, "b1_prereg.sha256 is not the frozen Lane B hash")
    _require(actual_sha == registered_sha, f"b1_prereg.json hash mismatch: expected {registered_sha}, got {actual_sha}")
    prereg = load_json(prereg_path)
    _require(prereg.get("schema_version") == "lane-b-prereg/1.0", "unexpected preregistration schema")
    _require(tuple(prereg.get("languages", [])) == LANGUAGES, "frozen language order changed")
    _require(prereg.get("harness", {}).get("llama_cpp_commit") == LLAMA_CPP_COMMIT, "llama.cpp pin changed")
    _require(prereg.get("decision_rule", {}).get("threshold") == 0.75, "decision threshold changed")
    _require(prereg.get("suites", {}).get("global_mmlu", {}).get("items_per_language") == 80, "MCQ n changed")
    _require(prereg.get("suites", {}).get("native_generation", {}).get("items_per_language") == 40, "generation n changed")

    model_configs: dict[str, dict[str, Any]] = {}
    for config in prereg.get("models", []):
        model_id = config.get("model_id")
        _require(model_id in MODEL_IDS and model_id not in model_configs, f"unexpected or duplicate model {model_id!r}")
        model_configs[model_id] = config
    _require(tuple(model_configs) == MODEL_IDS, "frozen model order or inventory changed")

    mmlu_rows, mmlu_by_id = _manifest(mmlu_path)
    native_rows, native_by_id = _manifest(native_path)
    mmlu_sha = sha256_file(mmlu_path)
    native_sha = sha256_file(native_path)
    _require(mmlu_sha == prereg["suites"]["global_mmlu"]["manifest_sha256"], "Global-MMLU manifest hash mismatch")
    _require(native_sha == prereg["suites"]["native_generation"]["manifest_sha256"], "native manifest hash mismatch")
    mmlu_counts = Counter(row.get("language") for row in mmlu_rows)
    native_counts = Counter(row.get("language") for row in native_rows)
    _require(len(mmlu_rows) == 400 and mmlu_counts == Counter({lang: 80 for lang in LANGUAGES}), "Global-MMLU is not frozen 80x5")
    _require(len(native_rows) == 200 and native_counts == Counter({lang: 40 for lang in LANGUAGES}), "native suite is not frozen 40x5")

    discovered_execution_paths = list(lane_dir.glob("b1_execution_*.json"))
    expected_execution_names = set(EXECUTION_RECORD_ORDER)
    actual_execution_names = {path.name for path in discovered_execution_paths}
    _require(actual_execution_names == expected_execution_names, f"append-only execution record inventory changed: expected {sorted(expected_execution_names)}, got {sorted(actual_execution_names)}")
    execution_paths = [lane_dir / name for name in EXECUTION_RECORD_ORDER]
    execution_records = [_validate_execution_record(path, actual_sha) for path in execution_paths]
    _validate_archived_execution_evidence(lane_dir, execution_paths, execution_records)

    return FrozenInputs(
        lane_dir=lane_dir,
        prereg=prereg,
        prereg_sha256=actual_sha,
        model_configs=model_configs,
        mmlu_rows=mmlu_rows,
        native_rows=native_rows,
        mmlu_by_id=mmlu_by_id,
        native_by_id=native_by_id,
        mmlu_sha256=mmlu_sha,
        native_sha256=native_sha,
        execution_records=execution_records,
    )


def _first_jsonl_model(path: Path) -> str:
    first = next(iter_jsonl(path), None)
    if first is None:
        raise AssemblyError(f"empty artifact {path}")
    model_id = first.get("model_id")
    if model_id not in MODEL_IDS:
        raise AssemblyError(f"{path} has unknown model_id {model_id!r}")
    return str(model_id)


def _discover_exact_jsonl(root: Path, filename: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(root.rglob(filename)):
        model_id = _first_jsonl_model(path)
        if model_id in found:
            raise AssemblyError(f"multiple {filename} artifacts for {model_id}: {found[model_id]} and {path}")
        found[model_id] = path
    return found


def _score_models(report: dict[str, Any]) -> set[str]:
    return {
        str(row.get("model_id"))
        for row in report.get("results", [])
        if isinstance(row, dict) and row.get("model_id") in MODEL_IDS
    }


def discover_bundles(
    artifacts_dir: Path,
    score_report_paths: Sequence[Path] = (),
    *,
    allow_incomplete: bool = False,
) -> tuple[dict[str, BundlePaths], dict[str, str]]:
    _require(artifacts_dir.is_dir(), f"artifact directory does not exist: {artifacts_dir}")
    mcq = _discover_exact_jsonl(artifacts_dir, "global_mmlu.raw.jsonl")
    generation = _discover_exact_jsonl(artifacts_dir, "native_generation.raw.jsonl")

    render: dict[str, Path] = {}
    for path in sorted(artifacts_dir.rglob("render_receipt.json")):
        receipt = load_json(path)
        model_id = receipt.get("model_id")
        if model_id not in MODEL_IDS:
            continue
        if model_id in render:
            raise AssemblyError(f"multiple render receipts for {model_id}: {render[model_id]} and {path}")
        render[str(model_id)] = path

    build_candidates = sorted(artifacts_dir.rglob("build_receipt.json"))
    build_receipt: Path | None = None
    if build_candidates:
        build_hashes = {sha256_file(path) for path in build_candidates}
        _require(len(build_hashes) == 1, "multiple non-identical build_receipt.json artifacts were discovered")
        build_receipt = build_candidates[0]

    candidate_score_paths: list[Path]
    if score_report_paths:
        candidate_score_paths = [path.resolve() for path in score_report_paths]
    else:
        candidate_score_paths = sorted(artifacts_dir.rglob("*.json"))
    scores: dict[str, Path] = {}
    for path in candidate_score_paths:
        try:
            report = load_json(path)
        except AssemblyError:
            if score_report_paths:
                raise
            continue
        if report.get("schema_version") != "lane-b-native-score/1.0":
            if score_report_paths:
                raise AssemblyError(f"explicit score report has wrong schema: {path}")
            continue
        models = _score_models(report)
        _require(models, f"score report has no frozen model rows: {path}")
        for model_id in models:
            if model_id in scores:
                raise AssemblyError(f"multiple native score reports for {model_id}: {scores[model_id]} and {path}")
            scores[model_id] = path

    bundles: dict[str, BundlePaths] = {}
    unavailable: dict[str, str] = {}
    for model_id in MODEL_IDS:
        missing: list[str] = []
        mcq_path = mcq.get(model_id)
        generation_path = generation.get(model_id)
        render_path = render.get(model_id)
        score_path = scores.get(model_id)
        if mcq_path is None:
            missing.append("global_mmlu.raw.jsonl")
        if generation_path is None:
            missing.append("native_generation.raw.jsonl")
        if render_path is None:
            missing.append("render_receipt.json")
        if build_receipt is None:
            missing.append("build_receipt.json")
        if score_path is None:
            missing.append("native score report")
        mmlu_rendered = render_path.parent / "global_mmlu.rendered.jsonl" if render_path else None
        generation_rendered = render_path.parent / "native_generation.rendered.jsonl" if render_path else None
        multiple_choice_binary = render_path.parent / "global_mmlu.multiple_choice.bin" if render_path else None
        if mmlu_rendered is not None and not mmlu_rendered.is_file():
            missing.append(mmlu_rendered.name)
        if generation_rendered is not None and not generation_rendered.is_file():
            missing.append(generation_rendered.name)
        if multiple_choice_binary is not None and not multiple_choice_binary.is_file():
            missing.append(multiple_choice_binary.name)
        mcq_receipt = Path(str(mcq_path) + ".receipt.json") if mcq_path else None
        generation_receipt = Path(str(generation_path) + ".receipt.json") if generation_path else None
        if mcq_receipt is not None and not mcq_receipt.is_file():
            missing.append(mcq_receipt.name)
        if generation_receipt is not None and not generation_receipt.is_file():
            missing.append(generation_receipt.name)
        if missing:
            unavailable[model_id] = "missing: " + ", ".join(missing)
            continue
        assert mcq_path and generation_path and render_path and score_path and mcq_receipt and generation_receipt
        assert mmlu_rendered and generation_rendered and multiple_choice_binary and build_receipt
        bundles[model_id] = BundlePaths(
            model_id=model_id,
            mcq_jsonl=mcq_path,
            mcq_receipt=mcq_receipt,
            generation_jsonl=generation_path,
            generation_receipt=generation_receipt,
            render_receipt=render_path,
            mmlu_rendered_jsonl=mmlu_rendered,
            generation_rendered_jsonl=generation_rendered,
            multiple_choice_binary=multiple_choice_binary,
            build_receipt=build_receipt,
            score_report=score_path,
        )
    if unavailable and not allow_incomplete:
        rendered = "; ".join(f"{model}: {reason}" for model, reason in unavailable.items())
        raise AssemblyError(f"incomplete B3/B4 artifact set: {rendered}")
    return bundles, unavailable


def _validate_render_receipt(
    bundle: BundlePaths,
    model_id: str,
    frozen: FrozenInputs,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    receipt = load_json(bundle.render_receipt)
    config = frozen.model_configs[model_id]
    _require(receipt.get("schema_version") == "lane-b-b34-render-receipt/1.0", f"{model_id}: wrong render receipt schema")
    _require(receipt.get("status") == "PASS", f"{model_id}: render receipt is not PASS")
    _require(receipt.get("harness_source_sha256") == EXPECTED_HARNESS_SHA256, f"{model_id}: render harness source hash mismatch")
    _require(receipt.get("model_id") == model_id, f"{model_id}: render receipt model mismatch")
    _require(receipt.get("b1_prereg_sha256") == frozen.prereg_sha256, f"{model_id}: render prereg hash mismatch")
    _require(receipt.get("tokenizer_repo") == config["tokenizer_repo"], f"{model_id}: tokenizer repo mismatch")
    _require(receipt.get("tokenizer_revision") == config["tokenizer_revision"], f"{model_id}: tokenizer revision mismatch")
    resolved = receipt.get("resolved_tokenizer_revision")
    _require(resolved in {None, config["tokenizer_revision"]}, f"{model_id}: resolved tokenizer revision mismatch")
    internal_commit = receipt.get("tokenizer_internal_commit_hash")
    if internal_commit is None:
        _require(
            resolved == config["tokenizer_revision"],
            f"{model_id}: null tokenizer internal commit requires exact resolved tokenizer revision",
        )
    else:
        _require(internal_commit == config["tokenizer_revision"], f"{model_id}: tokenizer internal commit mismatch")
    _require(isinstance(receipt.get("chat_template_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", receipt["chat_template_sha256"]) is not None, f"{model_id}: invalid chat-template hash")
    _require(receipt.get("transformers_version") == frozen.prereg["harness"]["transformers_version"], f"{model_id}: transformers render version mismatch")
    _require(receipt.get("huggingface_hub_version") == frozen.prereg["harness"]["huggingface_hub_version"], f"{model_id}: huggingface-hub render version mismatch")
    _require(receipt.get("global_mmlu_source_sha256") == frozen.mmlu_sha256, f"{model_id}: render MCQ source hash mismatch")
    _require(receipt.get("native_generation_source_sha256") == frozen.native_sha256, f"{model_id}: render generation source hash mismatch")
    _require(receipt.get("global_mmlu_count") == 400, f"{model_id}: render MCQ count is not 400")
    _require(receipt.get("native_generation_count") == 200, f"{model_id}: render generation count is not 200")
    _require(receipt.get("global_mmlu_rendered_sha256") == sha256_file(bundle.mmlu_rendered_jsonl), f"{model_id}: rendered MCQ JSONL hash mismatch")
    _require(receipt.get("native_generation_rendered_sha256") == sha256_file(bundle.generation_rendered_jsonl), f"{model_id}: rendered generation JSONL hash mismatch")
    _require(receipt.get("multiple_choice_binary_sha256") == sha256_file(bundle.multiple_choice_binary), f"{model_id}: multiple-choice binary hash mismatch")
    should_disable = model_id in {"qwen35_08b", "qwen3_06b"}
    _require(receipt.get("thinking_disabled_at_template") is should_disable, f"{model_id}: thinking-mode template pin mismatch")

    mmlu_rendered = list(iter_jsonl(bundle.mmlu_rendered_jsonl))
    generation_rendered = list(iter_jsonl(bundle.generation_rendered_jsonl))
    _require(len(mmlu_rendered) == 400, f"{model_id}: rendered MCQ JSONL is not 400 rows")
    _require(len(generation_rendered) == 200, f"{model_id}: rendered generation JSONL is not 200 rows")
    for ordinal, (row, source) in enumerate(zip(mmlu_rendered, frozen.mmlu_rows, strict=True)):
        context = f"{model_id} rendered MCQ {ordinal}"
        _require(row.get("schema_version") == "lane-b-b34-rendered/1.0", f"{context}: wrong schema")
        _require(row.get("suite") == "global_mmlu" and row.get("ordinal") == ordinal, f"{context}: suite/order mismatch")
        _require(row.get("model_id") == model_id and row.get("item_id") == source["item_id"], f"{context}: model/item mismatch")
        _require(row.get("sample_id") == source["sample_id"] and row.get("language") == source["language"], f"{context}: sample/language mismatch")
        _require(row.get("gold_answer") == source["answer"], f"{context}: gold answer mismatch")
        _require(row.get("candidate_answers") == ["A", "B", "C", "D"], f"{context}: candidate answers changed")
        _require(row.get("candidate_continuations") == [" A", " B", " C", " D"], f"{context}: candidate continuations changed")
        _require(row.get("source_item") == source, f"{context}: source item mismatch")
        prompts = frozen.prereg["prompts"]
        expected_user = prompts["mcq_user_template"].format(
            question=source["question"],
            option_a=source["option_a"],
            option_b=source["option_b"],
            option_c=source["option_c"],
            option_d=source["option_d"],
            localized_instruction=prompts["mcq_instruction_by_language"][source["language"]],
        )
        _require(row.get("messages") == [
            {"role": "system", "content": prompts["system_by_language"][source["language"]]},
            {"role": "user", "content": expected_user},
        ], f"{context}: rendered messages differ from frozen prompt")
        _require(type(row.get("hf_prompt_token_count")) is int and row["hf_prompt_token_count"] > 0, f"{context}: invalid tokenizer prompt count")
        prompt = row.get("rendered_prompt")
        _require(isinstance(prompt, str) and row.get("prompt_sha256") == sha256_text(prompt), f"{context}: prompt hash mismatch")
        if should_disable:
            _require(re.search(r"<think>\s*</think>", prompt) is not None, f"{context}: non-thinking block missing")
    for ordinal, (row, source) in enumerate(zip(generation_rendered, frozen.native_rows, strict=True)):
        context = f"{model_id} rendered generation {ordinal}"
        _require(row.get("schema_version") == "lane-b-b34-rendered/1.0", f"{context}: wrong schema")
        _require(row.get("suite") == "native_generation" and row.get("ordinal") == ordinal, f"{context}: suite/order mismatch")
        _require(row.get("model_id") == model_id and row.get("item_id") == source["item_id"], f"{context}: model/item mismatch")
        _require(row.get("language") == source["language"] and row.get("category") == source["category"], f"{context}: language/category mismatch")
        _require(row.get("source_item") == source, f"{context}: source item mismatch")
        prompts = frozen.prereg["prompts"]
        expected_user = prompts["native_user_template"].format(prompt=source["prompt"])
        _require(row.get("messages") == [
            {"role": "system", "content": prompts["system_by_language"][source["language"]]},
            {"role": "user", "content": expected_user},
        ], f"{context}: rendered messages differ from frozen prompt")
        _require(type(row.get("hf_prompt_token_count")) is int and row["hf_prompt_token_count"] > 0, f"{context}: invalid tokenizer prompt count")
        prompt = row.get("rendered_prompt")
        _require(isinstance(prompt, str) and row.get("prompt_sha256") == sha256_text(prompt), f"{context}: prompt hash mismatch")
        if should_disable:
            _require(re.search(r"<think>\s*</think>", prompt) is not None, f"{context}: non-thinking block missing")
    first_mmlu = receipt.get("first_prompt_sha256", {}).get("global_mmlu", {})
    first_generation = receipt.get("first_prompt_sha256", {}).get("native_generation", {})
    for language in LANGUAGES:
        _require(first_mmlu.get(language) == next(row["prompt_sha256"] for row in mmlu_rendered if row["language"] == language), f"{model_id}/{language}: first rendered MCQ hash mismatch")
        _require(first_generation.get(language) == next(row["prompt_sha256"] for row in generation_rendered if row["language"] == language), f"{model_id}/{language}: first rendered generation hash mismatch")
    harness = load_pinned_harness(frozen.lane_dir)
    try:
        reconstructed_binary = harness.build_multiple_choice_binary(mmlu_rendered)
    except Exception as exc:
        raise AssemblyError(f"{model_id}: pinned harness could not reconstruct multiple-choice binary: {exc}") from exc
    _require(bundle.multiple_choice_binary.read_bytes() == reconstructed_binary, f"{model_id}: multiple-choice binary differs from pinned-harness reconstruction")
    return receipt, mmlu_rendered, generation_rendered


def _validate_build_receipt(bundle: BundlePaths) -> tuple[dict[str, Any], str]:
    receipt = load_json(bundle.build_receipt)
    digest = sha256_file(bundle.build_receipt)
    _require(receipt.get("schema_version") == "lane-b-b34-build-receipt/1.0", "wrong build receipt schema")
    _require(receipt.get("status") == "PASS", "build receipt is not PASS")
    _require(receipt.get("harness_source_sha256") == EXPECTED_HARNESS_SHA256, "build receipt harness source hash mismatch")
    _require(receipt.get("llama_cpp_commit") == LLAMA_CPP_COMMIT, "build receipt llama.cpp commit mismatch")
    _require(receipt.get("source_tree_clean") is False, "build receipt falsely claims a clean llama.cpp tree")
    _require(receipt.get("source_tree_patch_exact") is True, "build receipt does not attest the exact one-file patch")
    _require(receipt.get("source_status") == LLAMA_CPP_EXPECTED_SOURCE_STATUS, "build receipt patched source status changed")
    _require(receipt.get("multiple_choice_source_relative_path") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH, "build receipt MCQ source path changed")
    _require(receipt.get("multiple_choice_source_original_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256, "build receipt original MCQ source hash changed")
    _require(receipt.get("multiple_choice_source_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256, "build receipt patched MCQ source hash changed")
    _require(receipt.get("multiple_choice_patch_relative_path") == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH, "build receipt MCQ patch path changed")
    _require(receipt.get("multiple_choice_patch_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256, "build receipt MCQ patch hash changed")
    _require(receipt.get("ggml_cuda") is True, "build receipt is not a CUDA build")
    cmake = receipt.get("cmake_configuration")
    _require(isinstance(cmake, dict), "build receipt CMake configuration is missing")
    cache_path_value = cmake.get("cache_path")
    source_dir_value = cmake.get("CMAKE_HOME_DIRECTORY")
    build_dir_value = cmake.get("CMAKE_CACHEFILE_DIR")
    bin_dir_value = cmake.get("llama_bin_dir")
    _require(all(isinstance(value, str) and Path(value).is_absolute() for value in (cache_path_value, source_dir_value, build_dir_value, bin_dir_value)), "build receipt CMake paths are not absolute")
    cache_path = Path(str(cache_path_value))
    source_dir = Path(str(source_dir_value))
    build_dir = Path(str(build_dir_value))
    bin_dir = Path(str(bin_dir_value))
    _require(cache_path == build_dir / "CMakeCache.txt", "build receipt CMake cache/build paths disagree")
    _require(bin_dir == build_dir / "bin", "build receipt CMake bin/build paths disagree")
    _require(receipt.get("llama_cpp_source") == str(source_dir), "build receipt patched source and CMake source directories disagree")
    _require(cmake.get("CMAKE_BUILD_TYPE") == "Release", "build receipt CMake build type changed")
    _require(cmake.get("GGML_CUDA") == "ON", "build receipt CMake CUDA setting changed")
    _require(cmake.get("BUILD_SHARED_LIBS") == "ON", "build receipt CMake shared-library setting changed")
    cmake_sha = cmake.get("cache_sha256")
    _require(isinstance(cmake_sha, str) and re.fullmatch(r"[0-9a-f]{64}", cmake_sha) is not None, "build receipt CMake cache hash is invalid")
    _require(receipt.get("cmake_cache_sha256") == cmake_sha, "build receipt CMake cache hashes disagree")
    impl = receipt.get("multiple_choice_impl_library")
    _require(isinstance(impl, dict), "build receipt MCQ implementation-library entry is missing")
    impl_path = impl.get("path")
    _require(isinstance(impl_path, str) and Path(impl_path).is_absolute(), "build receipt MCQ implementation-library path is not absolute")
    _require(Path(impl_path).name == Path(LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH).name, "build receipt MCQ implementation-library filename changed")
    _require(impl.get("relative_path") == LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH, "build receipt MCQ implementation-library relative path changed")
    _require(Path(impl_path) == bin_dir / Path(LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH).name, "build receipt MCQ implementation-library path is outside the receipted build")
    _require(impl.get("sha256") == LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_SHA256, "build receipt MCQ implementation-library hash changed")
    binaries = receipt.get("binaries")
    _require(isinstance(binaries, dict) and set(binaries) == {"llama-perplexity", "llama-server"}, "build receipt binary inventory mismatch")
    for name, binary in binaries.items():
        _require(isinstance(binary, dict), f"build receipt {name} entry is malformed")
        binary_path = binary.get("path")
        _require(isinstance(binary_path, str) and Path(binary_path) == bin_dir / name, f"build receipt {name} path is outside the receipted build")
        _require(isinstance(binary.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", binary["sha256"]) is not None, f"build receipt {name} hash is invalid")
    _require(binaries["llama-perplexity"]["sha256"] == LLAMA_CPP_PERPLEXITY_LAUNCHER_SHA256, "build receipt llama-perplexity launcher hash changed")
    return receipt, digest


def _validate_raw_common(row: dict[str, Any], model_id: str, config: Mapping[str, Any], item_id: str, language: str) -> None:
    _require(row.get("schema_version") == "lane-b-b34-raw/1.0", f"{model_id}/{item_id}: wrong raw schema")
    _require(row.get("evidence_class") == "HOST", f"{model_id}/{item_id}: evidence is not HOST")
    _require(row.get("model_id") == model_id, f"{model_id}/{item_id}: model mismatch")
    _require(row.get("item_id") == item_id, f"{model_id}/{item_id}: item mismatch")
    _require(row.get("language") == language, f"{model_id}/{item_id}: language mismatch")
    _require(row.get("quant_format") == config["quant_format"], f"{model_id}/{item_id}: quant format mismatch")
    _require(row.get("quant_file_sha256") == config["quant_file_sha256"], f"{model_id}/{item_id}: GGUF hash mismatch")
    _require(row.get("llama_cpp_commit") == LLAMA_CPP_COMMIT, f"{model_id}/{item_id}: llama.cpp commit mismatch")
    prompt = row.get("rendered_prompt")
    _require(isinstance(prompt, str) and sha256_text(prompt) == row.get("prompt_sha256"), f"{model_id}/{item_id}: prompt hash mismatch")
    if model_id in {"qwen35_08b", "qwen3_06b"}:
        _require(re.search(r"<think>\s*</think>", prompt) is not None, f"{model_id}/{item_id}: non-thinking empty-think block missing")


def _validate_copied_log(raw_jsonl: Path, receipt: Mapping[str, Any], path_key: str, hash_key: str, context: str) -> dict[str, str]:
    recorded_path = receipt.get(path_key)
    expected_sha = receipt.get(hash_key)
    _require(isinstance(recorded_path, str) and recorded_path, f"{context}: receipt is missing {path_key}")
    _require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None, f"{context}: invalid {hash_key}")
    local_path = raw_jsonl.parent / Path(recorded_path).name
    _require(local_path.is_file(), f"{context}: copied raw directory is missing {local_path.name}")
    actual_sha = sha256_file(local_path)
    _require(actual_sha == expected_sha, f"{context}: copied log hash mismatch for {local_path.name}")
    return {"file": local_path.name, "sha256": actual_sha}


def _validate_mcq(
    bundle: BundlePaths,
    frozen: FrozenInputs,
    render_receipt: dict[str, Any],
    rendered_rows: Sequence[dict[str, Any]],
    build_receipt: dict[str, Any],
    build_receipt_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_id = bundle.model_id
    config = frozen.model_configs[model_id]
    rows = list(iter_jsonl(bundle.mcq_jsonl))
    _require(len(rows) == 400, f"{model_id}: MCQ raw row count is {len(rows)}, expected 400")
    by_id: dict[str, dict[str, Any]] = {}
    previous_correct = 0
    for ordinal, row in enumerate(rows):
        item_id = row.get("item_id")
        _require(isinstance(item_id, str) and item_id in frozen.mmlu_by_id, f"{model_id}: unknown MCQ item {item_id!r}")
        _require(item_id == frozen.mmlu_rows[ordinal]["item_id"], f"{model_id}: MCQ manifest order changed at ordinal {ordinal}")
        _require(item_id not in by_id, f"{model_id}: duplicate MCQ item {item_id}")
        source = frozen.mmlu_by_id[item_id]
        rendered = rendered_rows[ordinal]
        _validate_raw_common(row, model_id, config, item_id, source["language"])
        _require(row.get("prompt_sha256") == rendered.get("prompt_sha256") and row.get("rendered_prompt") == rendered.get("rendered_prompt"), f"{model_id}/{item_id}: raw prompt differs from rendered JSONL")
        _require(row.get("run_kind") == "global_mmlu_log_likelihood", f"{model_id}/{item_id}: wrong MCQ run kind")
        _require(row.get("ordinal") == ordinal, f"{model_id}/{item_id}: MCQ order changed")
        _require(row.get("sample_id") == source["sample_id"], f"{model_id}/{item_id}: sample id mismatch")
        _require(row.get("gold_answer") == source["answer"], f"{model_id}/{item_id}: gold answer mismatch")
        _require(row.get("gold_index") == ["A", "B", "C", "D"].index(source["answer"]), f"{model_id}/{item_id}: gold index mismatch")
        _require(row.get("source_item") == source, f"{model_id}/{item_id}: embedded source item differs from frozen manifest")
        _require(row.get("candidate_answers") == ["A", "B", "C", "D"], f"{model_id}/{item_id}: candidate answers changed")
        _require(row.get("candidate_continuations") == [" A", " B", " C", " D"], f"{model_id}/{item_id}: candidate continuations changed")
        _require(type(row.get("correct")) is bool, f"{model_id}/{item_id}: correct must be boolean")
        cumulative = row.get("cumulative_correct")
        _require(type(cumulative) is int and cumulative == previous_correct + int(row["correct"]), f"{model_id}/{item_id}: invalid cumulative correctness")
        previous_correct = cumulative
        cumulative_pct = _require_number(row.get("cumulative_accuracy_pct"), f"{model_id}/{item_id} cumulative accuracy", low=0.0, high=100.0)
        _require(math.isclose(cumulative_pct, 100.0 * cumulative / (ordinal + 1), rel_tol=0.0, abs_tol=5e-7), f"{model_id}/{item_id}: cumulative accuracy cannot be reconstructed")
        unavailable_fields = (
            "choice_logprob_sums",
            "choice_token_counts",
            "choice_mean_logprobs",
            "predicted_index",
            "predicted_answer",
        )
        _require(all(row.get(field) is None for field in unavailable_fields), f"{model_id}/{item_id}: unavailable MCQ fields were fabricated")
        by_id[item_id] = row
    _require(set(by_id) == set(frozen.mmlu_by_id), f"{model_id}: MCQ item set differs from frozen manifest")

    first_hashes = render_receipt.get("first_prompt_sha256", {}).get("global_mmlu", {})
    for language in LANGUAGES:
        first = next(row for row in rows if row["language"] == language)
        _require(first_hashes.get(language) == first["prompt_sha256"], f"{model_id}/{language}: first MCQ prompt hash mismatch")

    receipt = load_json(bundle.mcq_receipt)
    raw_sha = sha256_file(bundle.mcq_jsonl)
    _require(receipt.get("schema_version") == "lane-b-b34-run-receipt/1.0", f"{model_id}: wrong MCQ receipt schema")
    _require(receipt.get("status") == "PASS" and receipt.get("run_kind") == "global_mmlu_log_likelihood", f"{model_id}: MCQ receipt is not PASS")
    _require(receipt.get("model_id") == model_id, f"{model_id}: MCQ receipt model mismatch")
    _require(receipt.get("harness_source_sha256") == EXPECTED_HARNESS_SHA256, f"{model_id}: MCQ harness source hash mismatch")
    _require(receipt.get("b1_prereg_sha256") == frozen.prereg_sha256, f"{model_id}: MCQ receipt prereg mismatch")
    _require(receipt.get("render_receipt_sha256") == sha256_file(bundle.render_receipt), f"{model_id}: MCQ render receipt hash mismatch")
    _require(receipt.get("build_receipt_sha256") == build_receipt_sha256, f"{model_id}: MCQ build receipt hash mismatch")
    _require(receipt.get("n") == 400 and receipt.get("correct") == previous_correct, f"{model_id}: MCQ receipt count mismatch")
    _require(receipt.get("raw_jsonl_sha256") == raw_sha, f"{model_id}: MCQ raw hash differs from receipt")
    _require(receipt.get("stock_multiple_choice_binary_unmodified") is False, f"{model_id}: MCQ executable patch declaration changed")
    _require(receipt.get("stock_multiple_choice_scoring_unmodified") is True, f"{model_id}: stock MCQ scoring was not preserved")
    _require(receipt.get("multiple_choice_memory_patch_exact") is True, f"{model_id}: exact MCQ memory patch was not receipted")
    _require(receipt.get("multiple_choice_patch_sha256") == LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256, f"{model_id}: MCQ memory patch hash mismatch")
    _require(receipt.get("per_choice_logprob_vectors_available") is False, f"{model_id}: MCQ availability declaration changed")
    model_receipt = receipt.get("model_receipt", {})
    _require(model_receipt.get("status") == "PASS" and model_receipt.get("model_id") == model_id, f"{model_id}: MCQ nested model receipt is not PASS")
    _require(model_receipt.get("b1_prereg_sha256") == frozen.prereg_sha256, f"{model_id}: MCQ nested model receipt prereg mismatch")
    _require(model_receipt.get("quant_file_sha256") == config["quant_file_sha256"], f"{model_id}: MCQ model receipt GGUF mismatch")
    build = receipt.get("build_receipt", {})
    _require(build == build_receipt, f"{model_id}: MCQ embedded build receipt differs from copied build receipt")
    command = receipt.get("command")
    expected_tail = [
        "--multiple-choice",
        "--ctx-size", str(MCQ_EXECUTION_BATCHING["ctx_size"]),
        "--batch-size", str(MCQ_EXECUTION_BATCHING["batch_size"]),
        "--ubatch-size", str(MCQ_EXECUTION_BATCHING["ubatch_size"]),
        "--parallel", str(MCQ_EXECUTION_BATCHING["parallel"]),
        "--n-gpu-layers", "999",
        "--seed", str(frozen.prereg["decoding"]["global_mmlu"]["seed"]),
    ]
    _require(isinstance(command, list) and len(command) == 5 + len(expected_tail), f"{model_id}: MCQ command shape changed")
    _require(Path(str(command[0])).name == "llama-perplexity", f"{model_id}: MCQ executable changed")
    _require(command[1] == "--model" and Path(str(command[2])).name == config["quant_file"], f"{model_id}: MCQ model command changed")
    _require(command[3] == "--binary-file" and Path(str(command[4])).name == bundle.multiple_choice_binary.name, f"{model_id}: MCQ binary command changed")
    _require(command[5:] == expected_tail, f"{model_id}: MCQ command settings differ from frozen protocol")
    _require(receipt.get("execution_batching") == MCQ_EXECUTION_BATCHING, f"{model_id}: MCQ batching receipt changed")
    _require(receipt.get("candidate_continuations") == [" A", " B", " C", " D"], f"{model_id}: MCQ receipt continuations changed")
    receipt["copied_llama_log"] = _validate_copied_log(bundle.mcq_jsonl, receipt, "llama_log", "llama_log_sha256", f"{model_id} MCQ")
    log_path = bundle.mcq_jsonl.parent / Path(str(receipt["llama_log"])).name
    harness = load_pinned_harness(frozen.lane_dir)
    try:
        trace = harness.parse_cumulative_accuracy(log_path.read_text(encoding="utf-8"), 400)
        reconstructed = harness.reconstruct_item_correctness(trace, 400)
    except Exception as exc:
        raise AssemblyError(f"{model_id}: copied llama log cannot reproduce MCQ rows: {exc}") from exc
    for ordinal, (row, recovered) in enumerate(zip(rows, reconstructed, strict=True)):
        _require(row["correct"] is recovered["correct"], f"{model_id}: llama log correctness mismatch at ordinal {ordinal}")
        _require(row["cumulative_correct"] == recovered["cumulative_correct"], f"{model_id}: llama log cumulative count mismatch at ordinal {ordinal}")
        _require(math.isclose(float(row["cumulative_accuracy_pct"]), float(recovered["cumulative_accuracy_pct"]), rel_tol=0.0, abs_tol=1e-12), f"{model_id}: llama log cumulative accuracy mismatch at ordinal {ordinal}")
        _require(row.get("raw_llama_line") == recovered["raw_line"], f"{model_id}: raw llama trace line mismatch at ordinal {ordinal}")
    return rows, receipt


def _validate_generation(
    bundle: BundlePaths,
    frozen: FrozenInputs,
    render_receipt: dict[str, Any],
    rendered_rows: Sequence[dict[str, Any]],
    build_receipt: dict[str, Any],
    build_receipt_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_id = bundle.model_id
    config = frozen.model_configs[model_id]
    rows = list(iter_jsonl(bundle.generation_jsonl))
    _require(len(rows) == 200, f"{model_id}: generation raw row count is {len(rows)}, expected 200")
    by_id: dict[str, dict[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        item_id = row.get("item_id")
        _require(isinstance(item_id, str) and item_id in frozen.native_by_id, f"{model_id}: unknown generation item {item_id!r}")
        _require(item_id == frozen.native_rows[ordinal]["item_id"], f"{model_id}: generation manifest order changed at ordinal {ordinal}")
        _require(item_id not in by_id, f"{model_id}: duplicate generation item {item_id}")
        source = frozen.native_by_id[item_id]
        rendered = rendered_rows[ordinal]
        _validate_raw_common(row, model_id, config, item_id, source["language"])
        _require(row.get("prompt_sha256") == rendered.get("prompt_sha256") and row.get("rendered_prompt") == rendered.get("rendered_prompt"), f"{model_id}/{item_id}: raw prompt differs from rendered JSONL")
        _require(row.get("run_kind") == "native_generation_greedy", f"{model_id}/{item_id}: wrong generation run kind")
        _require(row.get("ordinal") == ordinal, f"{model_id}/{item_id}: generation ordinal changed")
        _require(row.get("category") == source["category"], f"{model_id}/{item_id}: generation category mismatch")
        _require(row.get("source_item") == source, f"{model_id}/{item_id}: embedded source item differs from frozen manifest")
        completion = row.get("raw_completion")
        _require(isinstance(completion, str), f"{model_id}/{item_id}: raw completion is not text")
        _require(row.get("raw_completion_sha256") == sha256_text(completion), f"{model_id}/{item_id}: completion hash mismatch")
        request = row.get("request")
        expected_generation = frozen.prereg["decoding"]["native_generation"]
        expected_request = {
            "n_predict": expected_generation["max_new_tokens"],
            "temperature": expected_generation["temperature"],
            "top_p": expected_generation["top_p"],
            "top_k": expected_generation["top_k"],
            "min_p": expected_generation["min_p"],
            "repeat_penalty": expected_generation["repeat_penalty"],
            "seed": expected_generation["seed"],
            "stream": expected_generation["stream"],
            "cache_prompt": expected_generation["cache_prompt"],
            "stop": expected_generation["stop_strings"],
            "prompt": row["rendered_prompt"],
        }
        _require(request == expected_request, f"{model_id}/{item_id}: generation request differs from frozen decoding")
        server_response = row.get("server_response")
        _require(isinstance(server_response, dict) and server_response.get("content") == completion, f"{model_id}/{item_id}: server response/completion mismatch")
        by_id[item_id] = row
    _require(set(by_id) == set(frozen.native_by_id), f"{model_id}: generation item set differs from frozen manifest")

    first_hashes = render_receipt.get("first_prompt_sha256", {}).get("native_generation", {})
    for language in LANGUAGES:
        first = next(row for row in rows if row["language"] == language)
        _require(first_hashes.get(language) == first["prompt_sha256"], f"{model_id}/{language}: first generation prompt hash mismatch")

    receipt = load_json(bundle.generation_receipt)
    raw_sha = sha256_file(bundle.generation_jsonl)
    _require(receipt.get("schema_version") == "lane-b-b34-run-receipt/1.0", f"{model_id}: wrong generation receipt schema")
    _require(receipt.get("status") == "PASS" and receipt.get("run_kind") == "native_generation_greedy", f"{model_id}: generation receipt is not PASS")
    _require(receipt.get("model_id") == model_id, f"{model_id}: generation receipt model mismatch")
    _require(receipt.get("harness_source_sha256") == EXPECTED_HARNESS_SHA256, f"{model_id}: generation harness source hash mismatch")
    _require(receipt.get("b1_prereg_sha256") == frozen.prereg_sha256, f"{model_id}: generation receipt prereg mismatch")
    _require(receipt.get("render_receipt_sha256") == sha256_file(bundle.render_receipt), f"{model_id}: generation render receipt hash mismatch")
    _require(receipt.get("build_receipt_sha256") == build_receipt_sha256, f"{model_id}: generation build receipt hash mismatch")
    _require(receipt.get("build_receipt") == build_receipt, f"{model_id}: generation embedded build receipt differs from copied build receipt")
    nested_model = receipt.get("model_receipt")
    _require(isinstance(nested_model, dict), f"{model_id}: generation nested model receipt missing")
    _require(nested_model.get("status") == "PASS" and nested_model.get("model_id") == model_id, f"{model_id}: generation nested model receipt is not PASS")
    _require(nested_model.get("b1_prereg_sha256") == frozen.prereg_sha256 and nested_model.get("quant_file_sha256") == config["quant_file_sha256"], f"{model_id}: generation nested model receipt mismatch")
    _require(receipt.get("n_expected") == 200 and receipt.get("n_present") == 200 and receipt.get("is_complete") is True, f"{model_id}: generation receipt is incomplete")
    _require(receipt.get("smoke_or_partial_limit") is None, f"{model_id}: smoke/partial generation is inadmissible")
    _require(receipt.get("raw_jsonl_sha256") == raw_sha, f"{model_id}: generation raw hash differs from receipt")
    expected_template = {key: value for key, value in expected_request.items() if key != "prompt"}
    _require(receipt.get("request_template") == expected_template, f"{model_id}: generation receipt request template differs from frozen decoding")
    server_command = receipt.get("server_command")
    _require(isinstance(server_command, list) and len(server_command) == 17, f"{model_id}: generation server command shape changed")
    _require(Path(str(server_command[0])).name == "llama-server", f"{model_id}: generation executable changed")
    _require(server_command[1] == "--model" and Path(str(server_command[2])).name == config["quant_file"], f"{model_id}: generation model command changed")
    _require(server_command[3:16] == [
        "--ctx-size", "4096",
        "--batch-size", "4096",
        "--ubatch-size", "512",
        "--parallel", "1",
        "--n-gpu-layers", "999",
        "--host", "127.0.0.1",
        "--port",
    ], f"{model_id}: generation server settings differ from frozen protocol")
    _require(str(server_command[16]).isdigit(), f"{model_id}: generation server port is invalid")
    receipt["copied_server_log"] = _validate_copied_log(bundle.generation_jsonl, receipt, "server_log", "server_log_sha256", f"{model_id} generation")
    return rows, receipt


def _score_recomputation_mismatch(
    recorded: Any,
    recomputed: Any,
    path: tuple[str, ...] = (),
) -> str | None:
    """Return the first material scorer mismatch, tolerating Lingua roundoff only.

    ``lingua-language-detector`` can vary its normalized confidence values by a
    few IEEE-754 ulps across calls because its internal parallel reduction order
    is not stable.  Those diagnostic probabilities do not drive the frozen
    score once the recomputed categorical fields are also required to match.
    Every other scorer value remains byte-semantically exact.
    """
    rendered_path = ".".join(path) or "<root>"
    if isinstance(recorded, dict) or isinstance(recomputed, dict):
        if not isinstance(recorded, dict) or not isinstance(recomputed, dict):
            return rendered_path
        if set(recorded) != set(recomputed):
            return rendered_path
        for key in recorded:
            mismatch = _score_recomputation_mismatch(recorded[key], recomputed[key], (*path, str(key)))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(recorded, list) or isinstance(recomputed, list):
        if not isinstance(recorded, list) or not isinstance(recomputed, list) or len(recorded) != len(recomputed):
            return rendered_path
        for index, (recorded_item, recomputed_item) in enumerate(zip(recorded, recomputed, strict=True)):
            mismatch = _score_recomputation_mismatch(recorded_item, recomputed_item, (*path, str(index)))
            if mismatch is not None:
                return mismatch
        return None
    lingua_probability = len(path) >= 3 and path[:2] == ("surface", "language_detection")
    if lingua_probability and type(recorded) is float and type(recomputed) is float:
        if math.isfinite(recorded) and math.isfinite(recomputed) and math.isclose(recorded, recomputed, rel_tol=0.0, abs_tol=1e-12):
            return None
        return rendered_path
    return None if type(recorded) is type(recomputed) and recorded == recomputed else rendered_path


def _validate_score_report(
    bundle: BundlePaths,
    frozen: FrozenInputs,
    generation_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_id = bundle.model_id
    report = load_json(bundle.score_report)
    _require(report.get("schema_version") == "lane-b-native-score/1.0", f"{model_id}: wrong scorer schema")
    _require(report.get("scorer_source_sha256") == EXPECTED_SCORER_SHA256, f"{model_id}: score report scorer-source hash mismatch")
    scorer = load_pinned_scorer(frozen.lane_dir)
    dependencies = report.get("dependencies", {})
    _require(dependencies.get("strict_pins_enforced") is True, f"{model_id}: scorer dependency pins were not enforced")
    _require(dependencies.get("mismatches") == [], f"{model_id}: scorer dependency mismatch")
    _require(dependencies.get("versions") == scorer.PINNED_DEPENDENCIES, f"{model_id}: scorer dependency versions differ from frozen pins")
    scored_python = dependencies.get("python")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", str(scored_python))
    _require(match is not None and (3, 11) <= (int(match.group(1)), int(match.group(2))) < (3, 14), f"{model_id}: scorer Python is outside >=3.11,<3.14")
    _require(report.get("allow_partial") is False, f"{model_id}: partial scorer output is inadmissible")
    manifest = report.get("manifest", {})
    _require(manifest.get("sha256") == frozen.native_sha256 and manifest.get("items") == 200, f"{model_id}: scorer manifest mismatch")

    all_results = report.get("results")
    _require(isinstance(all_results, list), f"{model_id}: scorer results missing")
    unknown_score_models = {
        row.get("model_id")
        for row in all_results
        if isinstance(row, dict) and row.get("model_id") not in MODEL_IDS
    }
    _require(not unknown_score_models, f"{model_id}: scorer report contains unknown model ids {sorted(map(str, unknown_score_models))}")
    score_models = _score_models(report)
    results = [row for row in all_results if isinstance(row, dict) and row.get("model_id") == model_id]
    _require(len(results) == 200, f"{model_id}: scorer has {len(results)} rows, expected 200")
    summary = report.get("summary", {}).get("models", {}).get(model_id, {})
    _require(summary.get("n") == 200 and summary.get("b5_generation_denominator_at_least_200") is True, f"{model_id}: scorer summary denominator mismatch")
    if score_models == {model_id}:
        generations = report.get("generations", {})
        _require(generations.get("rows") == 200, f"{model_id}: single-arm scorer row count mismatch")
        _require(generations.get("sha256") == sha256_file(bundle.generation_jsonl), f"{model_id}: scorer did not score the receipted generation JSONL")

    raw_by_id = {row["item_id"]: row for row in generation_rows}
    seen: set[str] = set()
    for result in results:
        item_id = result.get("item_id")
        _require(isinstance(item_id, str) and item_id in raw_by_id, f"{model_id}: scorer has unknown item {item_id!r}")
        _require(item_id not in seen, f"{model_id}: duplicate scored item {item_id}")
        seen.add(item_id)
        source = frozen.native_by_id[item_id]
        _require(result.get("language") == source["language"], f"{model_id}/{item_id}: scored language mismatch")
        raw = raw_by_id[item_id]["raw_completion"]
        _require(result.get("raw_output") == raw, f"{model_id}/{item_id}: scorer raw text differs from harness raw text")
        _require(result.get("raw_output_sha256") == sha256_text(raw), f"{model_id}/{item_id}: scorer raw hash mismatch")
        try:
            recomputed = scorer.score_item(source, raw, model_id)
        except Exception as exc:
            raise AssemblyError(f"{model_id}/{item_id}: committed scorer could not recompute raw output: {exc}") from exc
        mismatch = _score_recomputation_mismatch(result, recomputed)
        _require(mismatch is None, f"{model_id}/{item_id}: score report differs from deterministic committed-scorer recomputation at {mismatch}")
        traditional = result.get("surface", {}).get("traditional_han")
        _require(isinstance(traditional, dict), f"{model_id}/{item_id}: Traditional-Han analysis is missing")
        _require(type(traditional.get("eligible")) is bool, f"{model_id}/{item_id}: invalid Traditional-Han eligibility")
        leakage = traditional.get("leakage")
        _require(leakage is None or type(leakage) is bool, f"{model_id}/{item_id}: invalid Traditional-Han leakage flag")
        han_count = traditional.get("han_count")
        _require(type(han_count) is int and han_count >= 0, f"{model_id}/{item_id}: invalid Traditional-Han count")
        changed_fraction = traditional.get("changed_han_fraction")
        _require(changed_fraction is None or (not isinstance(changed_fraction, bool) and isinstance(changed_fraction, (int, float)) and math.isfinite(float(changed_fraction)) and 0.0 <= float(changed_fraction) <= 1.0), f"{model_id}/{item_id}: invalid changed-Han fraction")
        if traditional["eligible"]:
            _require(type(leakage) is bool and changed_fraction is not None and han_count > 0, f"{model_id}/{item_id}: eligible Traditional-Han structure is incomplete")
        else:
            _require(changed_fraction is None, f"{model_id}/{item_id}: ineligible Traditional-Han row has a fraction")
        correctness = result.get("correctness")
        _require(type(correctness) is int and correctness in {0, 1}, f"{model_id}/{item_id}: invalid correctness")
        rubric_passed = result.get("rubric", {}).get("passed")
        _require(type(rubric_passed) is bool and int(rubric_passed) == correctness, f"{model_id}/{item_id}: rubric/correctness mismatch")
        score = _require_number(result.get("generation_item_score"), f"{model_id}/{item_id} generation score", low=0.0, high=1.0)
        surface = result.get("surface", {})
        surface_score = _require_number(surface.get("surface_fluency_proxy"), f"{model_id}/{item_id} surface proxy", low=0.0, high=1.0)
        language_component = _require_number(surface.get("language_component"), f"{model_id}/{item_id} language component", low=0.0, high=1.0)
        script_component = _require_number(surface.get("script_component"), f"{model_id}/{item_id} script component", low=0.0, high=1.0)
        nondegenerate_component = _require_number(surface.get("nondegenerate_component"), f"{model_id}/{item_id} nondegenerate component", low=0.0, high=1.0)
        speakability_component = _require_number(surface.get("speakability_component"), f"{model_id}/{item_id} speakability component", low=0.0, high=1.0)
        _require(language_component in {0.0, 0.5, 1.0}, f"{model_id}/{item_id}: language component is outside frozen values")
        for name, value in (
            ("script", script_component),
            ("nondegenerate", nondegenerate_component),
            ("speakability", speakability_component),
        ):
            _require(value in {0.0, 1.0}, f"{model_id}/{item_id}: {name} component is not binary")
        expected_surface = (
            0.50 * language_component
            + 0.25 * script_component
            + 0.15 * nondegenerate_component
            + 0.10 * speakability_component
        )
        _require(math.isclose(surface_score, expected_surface, rel_tol=0.0, abs_tol=1e-12), f"{model_id}/{item_id}: surface proxy does not match the frozen formula")
        concision_passed = result.get("concision", {}).get("passed")
        _require(type(concision_passed) is bool, f"{model_id}/{item_id}: invalid concision flag")
        expected_score = (0.70 + 0.20 * surface_score + 0.10 * int(concision_passed)) if correctness else 0.0
        _require(math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12), f"{model_id}/{item_id}: generation item score does not match the frozen formula")
        strict_pass = result.get("strict_generation_pass")
        _require(type(strict_pass) is bool, f"{model_id}/{item_id}: invalid strict-pass flag")
        expected_strict = bool(correctness and surface_score == 1.0 and concision_passed)
        _require(strict_pass is expected_strict, f"{model_id}/{item_id}: strict-pass flag does not match the frozen formula")
        language_detection = surface.get("language_detection", {})
        _require(type(language_detection.get("eligible")) is bool, f"{model_id}/{item_id}: invalid language eligibility")
        wrong_language = language_detection.get("wrong_language")
        _require(wrong_language is None or type(wrong_language) is bool, f"{model_id}/{item_id}: invalid wrong-language flag")
        if not language_detection["eligible"]:
            _require(wrong_language is None, f"{model_id}/{item_id}: ineligible output was counted as language evidence")
        script = surface.get("script", {})
        _require(type(script.get("wrong_script_eligible")) is bool, f"{model_id}/{item_id}: invalid script eligibility")
        wrong_script = script.get("wrong_script")
        _require(wrong_script is None or type(wrong_script) is bool, f"{model_id}/{item_id}: invalid wrong-script flag")
        if not script["wrong_script_eligible"]:
            _require(wrong_script is None, f"{model_id}/{item_id}: ineligible output was counted as script evidence")
        _require(type(surface.get("reasoning_leakage", {}).get("leakage")) is bool, f"{model_id}/{item_id}: invalid reasoning-leakage flag")
    _require(seen == set(frozen.native_by_id), f"{model_id}: scorer item set differs from frozen manifest")
    order = {row["item_id"]: index for index, row in enumerate(frozen.native_rows)}
    return sorted(results, key=lambda row: order[row["item_id"]])


def validate_bundle(bundle: BundlePaths, frozen: FrozenInputs) -> ModelEvidence:
    render_receipt, mmlu_rendered, generation_rendered = _validate_render_receipt(bundle, bundle.model_id, frozen)
    build_receipt, build_receipt_sha256 = _validate_build_receipt(bundle)
    mcq_rows, mcq_receipt = _validate_mcq(
        bundle,
        frozen,
        render_receipt,
        mmlu_rendered,
        build_receipt,
        build_receipt_sha256,
    )
    generation_rows, generation_receipt = _validate_generation(
        bundle,
        frozen,
        render_receipt,
        generation_rendered,
        build_receipt,
        build_receipt_sha256,
    )
    score_rows = _validate_score_report(bundle, frozen, generation_rows)
    return ModelEvidence(
        model_id=bundle.model_id,
        mcq_rows=mcq_rows,
        generation_rows=generation_rows,
        score_rows=score_rows,
        source_hashes={
            "global_mmlu_raw_sha256": sha256_file(bundle.mcq_jsonl),
            "global_mmlu_receipt_sha256": sha256_file(bundle.mcq_receipt),
            "native_generation_raw_sha256": sha256_file(bundle.generation_jsonl),
            "native_generation_receipt_sha256": sha256_file(bundle.generation_receipt),
            "native_score_report_sha256": sha256_file(bundle.score_report),
            "render_receipt_sha256": sha256_file(bundle.render_receipt),
            "global_mmlu_rendered_sha256": sha256_file(bundle.mmlu_rendered_jsonl),
            "native_generation_rendered_sha256": sha256_file(bundle.generation_rendered_jsonl),
            "multiple_choice_binary_sha256": sha256_file(bundle.multiple_choice_binary),
            "build_receipt_sha256": sha256_file(bundle.build_receipt),
            "scorer_source_sha256": EXPECTED_SCORER_SHA256,
            "harness_source_sha256": EXPECTED_HARNESS_SHA256,
            "multiple_choice_patch_sha256": LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
            "multiple_choice_patched_source_sha256": LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256,
            "multiple_choice_impl_library_sha256": build_receipt["multiple_choice_impl_library"]["sha256"],
            "llama_perplexity_launcher_sha256": build_receipt["binaries"]["llama-perplexity"]["sha256"],
        },
        render_receipt=render_receipt,
        mcq_receipt=mcq_receipt,
        generation_receipt=generation_receipt,
    )


def wilson_95(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def binary_summary(values: Iterable[bool]) -> dict[str, Any]:
    materialized = list(values)
    successes = sum(materialized)
    total = len(materialized)
    return {
        "count": successes,
        "n": total,
        "rate": successes / total if total else None,
        "wilson_95": wilson_95(successes, total),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    _require(bool(sorted_values), "cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def stratified_bootstrap_95(
    mcq_scores: Sequence[float],
    generation_scores: Sequence[float],
    *,
    rng: random.Random,
    resamples: int,
) -> list[float]:
    _require(len(mcq_scores) == 80, f"bootstrap requires 80 MCQ scores, got {len(mcq_scores)}")
    _require(len(generation_scores) == 40, f"bootstrap requires 40 generation scores, got {len(generation_scores)}")
    _require(resamples > 0, "bootstrap resamples must be positive")
    draws: list[float] = []
    for _ in range(resamples):
        mcq_total = math.fsum(rng.choices(mcq_scores, k=80))
        generation_total = math.fsum(rng.choices(generation_scores, k=40))
        draws.append(100.0 * (mcq_total + generation_total) / 120.0)
    draws.sort()
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def advance_missing_model_bootstrap_stream(*, rng: random.Random, resamples: int) -> None:
    """Preserve the frozen global traversal state when a partial pack lacks an arm."""
    draws_per_cell = 80 + 40
    for _language in LANGUAGES:
        for _ in range(resamples * draws_per_cell):
            rng.random()


def _ratio_summary(ratios: Sequence[float]) -> dict[str, Any]:
    if not ratios:
        return {"n": 0, "mean": None, "minimum": None}
    return {"n": len(ratios), "mean": math.fsum(ratios) / len(ratios), "minimum": min(ratios)}


def _exact_generation_item_score(row: Mapping[str, Any]) -> Fraction:
    correctness = int(row["correctness"])
    if not correctness:
        return Fraction(0)
    surface = row["surface"]
    language_component = Fraction(str(surface["language_component"]))
    script_component = Fraction(str(surface["script_component"]))
    nondegenerate_component = Fraction(str(surface["nondegenerate_component"]))
    speakability_component = Fraction(str(surface["speakability_component"]))
    fluency = (
        Fraction(1, 2) * language_component
        + Fraction(1, 4) * script_component
        + Fraction(3, 20) * nondegenerate_component
        + Fraction(1, 10) * speakability_component
    )
    return Fraction(7, 10) + Fraction(1, 5) * fluency + Fraction(1, 10) * int(row["concision"]["passed"])


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _score_fraction(model: Mapping[str, Any], language: str) -> Fraction:
    cell = model["languages"][language]
    exact = cell.get("headline_score_exact_fraction")
    if isinstance(exact, str):
        return Fraction(exact)
    return Fraction(str(cell["headline_score_0_100"]))


def _ratio_fraction(model: Mapping[str, Any], language: str) -> Fraction | None:
    cell = model["languages"][language]
    exact = cell.get("competence_ratio_exact_fraction")
    if isinstance(exact, str):
        return Fraction(exact)
    ratio = cell.get("competence_ratio_to_en")
    return None if ratio is None else Fraction(str(ratio))


def _changed_han_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row["surface"]["traditional_han"] for row in rows if row["surface"]["traditional_han"]["eligible"]]
    fractions = [float(row["changed_han_fraction"]) for row in eligible]
    leakage = binary_summary(bool(row["leakage"]) for row in eligible)
    return {
        **leakage,
        "mean_changed_han_fraction": math.fsum(fractions) / len(fractions) if fractions else None,
        "maximum_changed_han_fraction": max(fractions) if fractions else None,
    }


def summarize_model(
    evidence: ModelEvidence,
    frozen: FrozenInputs,
    *,
    rng: random.Random,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    model_id = evidence.model_id
    config = frozen.model_configs[model_id]
    languages: dict[str, Any] = {}
    b5_languages: dict[str, Any] = {}
    exact_language_scores: dict[str, Fraction] = {}
    for language in LANGUAGES:
        mcq = [row for row in evidence.mcq_rows if row["language"] == language]
        native = [row for row in evidence.score_rows if row["language"] == language]
        _require(len(mcq) == 80 and len(native) == 40, f"{model_id}/{language}: wrong suite counts")
        mcq_values = [float(int(row["correct"])) for row in mcq]
        generation_values = [float(row["generation_item_score"]) for row in native]
        mcq_correct = sum(int(value) for value in mcq_values)
        exact_generation_values = [_exact_generation_item_score(row) for row in native]
        exact_headline = Fraction(100, 120) * (Fraction(mcq_correct) + sum(exact_generation_values, Fraction()))
        exact_language_scores[language] = exact_headline
        headline = float(exact_headline)
        bootstrap = stratified_bootstrap_95(
            mcq_values,
            generation_values,
            rng=rng,
            resamples=bootstrap_resamples,
        )
        correctness = binary_summary(bool(row["correctness"]) for row in native)
        concision = binary_summary(bool(row["concision"]["passed"]) for row in native)
        strict_pass = binary_summary(bool(row["strict_generation_pass"]) for row in native)
        binary_surface_components = {
            "script_S": binary_summary(float(row["surface"]["script_component"]) == 1.0 for row in native),
            "nondegenerate_D": binary_summary(float(row["surface"]["nondegenerate_component"]) == 1.0 for row in native),
            "speakability_P": binary_summary(float(row["surface"]["speakability_component"]) == 1.0 for row in native),
        }
        surface_mean = math.fsum(float(row["surface"]["surface_fluency_proxy"]) for row in native) / 40
        exact_generation_score = Fraction(100, 40) * sum(exact_generation_values, Fraction())
        generation_score = float(exact_generation_score)
        languages[language] = {
            "evidence_class": "HOST",
            "n": 120,
            "headline_score_0_100": headline,
            "headline_score_exact_fraction": _fraction_text(exact_headline),
            "bootstrap_95": bootstrap,
            "bootstrap_method": (
                f"{bootstrap_resamples:,}-resample stratified percentile bootstrap using one assembly-wide "
                f"Python random.Random MT19937 stream seeded once at {BOOTSTRAP_SEED}; model traversal "
                f"{list(MODEL_IDS)}, language traversal {list(LANGUAGES)}; within every replicate draw 80 "
                "equal-weight MCQ items with replacement before 40 equal-weight generation items; "
                "type-7 linear interpolation at 2.5/97.5 percentiles"
            ),
            "global_mmlu": {
                "evidence_class": "HOST",
                "n": 80,
                "correct": mcq_correct,
                "accuracy": mcq_correct / 80.0,
                "wilson_95": wilson_95(mcq_correct, 80),
            },
            "native_generation": {
                "evidence_class": "HOST",
                "n": 40,
                "correctness": correctness,
                "mean_surface_fluency_proxy": surface_mean,
                "surface_proxy_warning": "Mechanical surface language/script/degeneracy/speakability proxy; not native fluency or naturalness.",
                "binary_surface_component_rates": binary_surface_components,
                "concision": concision,
                "strict_generation_pass": strict_pass,
                "native_generation_score_0_100": generation_score,
                "native_generation_score_exact_fraction": _fraction_text(exact_generation_score),
            },
            "competence_ratio_to_en": None,
            "competence_ratio_exact_fraction": None,
        }

        wrong_language_values = [row["surface"]["language_detection"]["wrong_language"] for row in native]
        wrong_script_values = [row["surface"]["script"]["wrong_script"] for row in native]
        reasoning_values = [bool(row["surface"]["reasoning_leakage"]["leakage"]) for row in native]
        script_ratios: list[float] = []
        ratio_name: str | None = None
        if language == "ru":
            ratio_name = "cyrillic_ratio"
        elif language == "zh":
            ratio_name = "han_ratio"
        if ratio_name:
            script_ratios = [
                float(row["surface"]["script"][ratio_name])
                for row in native
                if row["surface"]["script"].get(ratio_name) is not None
            ]
        b5_languages[language] = {
            "n_native_generations": 40,
            "wrong_language": binary_summary(bool(value) for value in wrong_language_values if value is not None),
            "wrong_language_eligibility_rule": ">=8 Unicode letters; ineligible short answers are excluded",
            "wrong_script": binary_summary(bool(value) for value in wrong_script_values if value is not None),
            "wrong_script_eligibility_rule": ">=4 script-assigned characters",
            "script_ratio": {"name": ratio_name, **_ratio_summary(script_ratios)} if ratio_name else None,
            "traditional_han_leakage": _changed_han_summary(native) if language == "zh" else None,
            "reasoning_leakage": binary_summary(reasoning_values),
        }

    english_score = exact_language_scores["en"]
    if english_score > 0:
        for language in LANGUAGES:
            ratio = exact_language_scores[language] / english_score
            languages[language]["competence_ratio_to_en"] = float(ratio)
            languages[language]["competence_ratio_exact_fraction"] = _fraction_text(ratio)

    reasoning_all = [bool(row["surface"]["reasoning_leakage"]["leakage"]) for row in evidence.score_rows]
    mode_pin_count = sum(reasoning_all)
    mode_pin_rate = mode_pin_count / 200.0
    mode_pin = {
        "n": 200,
        "prompt_template_violations": 0,
        "output_marker_or_initial_deliberation_violations": mode_pin_count,
        "total_violations": mode_pin_count,
        "rate": mode_pin_rate,
        "void_threshold": ">0.5%",
        "ranking_void": mode_pin_rate > MODE_PIN_VOID_RATE,
    }
    return {
        "model_id": model_id,
        "base_repo": config["base_repo"],
        "base_revision": config["base_revision"],
        "quant_repo": config["quant_repo"],
        "quant_revision": config["quant_revision"],
        "quant_file": config["quant_file"],
        "quant_format": config["quant_format"],
        "quant_file_bytes": config["quant_file_bytes"],
        "quant_file_sha256": config["quant_file_sha256"],
        "evidence_class": "HOST",
        "n_per_language": 120,
        "n_global_mmlu": 400,
        "n_native_generations": 200,
        "languages": languages,
        "b5": {
            "denominator_interpretation": "n=40 per language and n=200 per model across five languages; weaker than the parent plan's per-language requirement",
            "languages": b5_languages,
            "all_language_reasoning_leakage": binary_summary(reasoning_all),
        },
        "mode_pin": mode_pin,
        "ranking_admissible": not mode_pin["ranking_void"],
        "source_hashes": evidence.source_hashes,
    }


def compare_q_vs_f(models: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    q = models.get("qwen25_15b")
    f = models.get("falcon_h1_15b")
    if q is None or f is None:
        return {"status": "UNDECIDABLE", "reason": "Qwen2.5 and Falcon complete HOST arms are both required"}
    if not q.get("ranking_admissible") or not f.get("ranking_admissible"):
        return {"status": "UNDECIDABLE", "reason": "Qwen2.5 or Falcon is void for ranking under the mode-pin rule"}

    language_winners: dict[str, str] = {}
    wins = {"qwen25_15b": 0, "falcon_h1_15b": 0, "ties": 0}
    for language in LANGUAGES:
        q_score = _score_fraction(q, language)
        f_score = _score_fraction(f, language)
        if q_score > f_score:
            winner = "qwen25_15b"
        elif f_score > q_score:
            winner = "falcon_h1_15b"
        else:
            winner = "tie"
        language_winners[language] = winner
        if winner == "tie":
            wins["ties"] += 1
        else:
            wins[winner] += 1

    trace: list[dict[str, Any]] = [{"step": "language_wins", "counts": wins, "per_language": language_winners}]
    if wins["qwen25_15b"] != wins["falcon_h1_15b"]:
        leader = "qwen25_15b" if wins["qwen25_15b"] > wins["falcon_h1_15b"] else "falcon_h1_15b"
        return {"status": "DECIDED", "leader_model_id": leader, "deciding_step": "language_wins", "trace": trace}

    minima: dict[str, Fraction] = {}
    for model_id, model in (("qwen25_15b", q), ("falcon_h1_15b", f)):
        ratios = [_ratio_fraction(model, language) for language in DECISION_LANGUAGES]
        if any(ratio is None for ratio in ratios):
            return {"status": "UNDECIDABLE", "reason": "Q/F language wins tie and an English score is zero", "trace": trace}
        minima[model_id] = min(ratio for ratio in ratios if ratio is not None)
    trace.append({
        "step": "higher_minimum_es_fr_ru_ratio",
        "values": {model_id: float(value) for model_id, value in minima.items()},
        "exact_fractions": {model_id: _fraction_text(value) for model_id, value in minima.items()},
    })
    if minima["qwen25_15b"] != minima["falcon_h1_15b"]:
        leader = "qwen25_15b" if minima["qwen25_15b"] > minima["falcon_h1_15b"] else "falcon_h1_15b"
        return {"status": "DECIDED", "leader_model_id": leader, "deciding_step": "higher_minimum_es_fr_ru_ratio", "trace": trace}

    ru_scores = {
        "qwen25_15b": _score_fraction(q, "ru"),
        "falcon_h1_15b": _score_fraction(f, "ru"),
    }
    trace.append({
        "step": "higher_ru_score",
        "values": {model_id: float(value) for model_id, value in ru_scores.items()},
        "exact_fractions": {model_id: _fraction_text(value) for model_id, value in ru_scores.items()},
    })
    if ru_scores["qwen25_15b"] != ru_scores["falcon_h1_15b"]:
        leader = "qwen25_15b" if ru_scores["qwen25_15b"] > ru_scores["falcon_h1_15b"] else "falcon_h1_15b"
        return {"status": "DECIDED", "leader_model_id": leader, "deciding_step": "higher_ru_score", "trace": trace}

    trace.append({"step": "incumbent_headroom_tie_break", "winner": "qwen25_15b"})
    return {
        "status": "DECIDED",
        "leader_model_id": "qwen25_15b",
        "deciding_step": "incumbent_headroom_tie_break",
        "trace": trace,
    }


def apply_decision_rule(models: Mapping[str, dict[str, Any]], unavailable: Mapping[str, str]) -> dict[str, Any]:
    qwen35 = models.get("qwen35_08b")
    if qwen35 is None:
        return {
            "verdict_code": "UNDECIDABLE",
            "qwen35_outcome": "UNKNOWN",
            "reason": "Qwen3.5-0.8B complete HOST arm is missing",
            "unavailable_models": dict(unavailable),
        }
    if not qwen35.get("ranking_admissible"):
        return {
            "verdict_code": "UNDECIDABLE",
            "qwen35_outcome": "VOID",
            "reason": "Qwen3.5-0.8B exceeds the frozen >0.5% mode-pin violation threshold",
            "mode_pin": qwen35["mode_pin"],
        }
    english_score = _score_fraction(qwen35, "en")
    if english_score == 0:
        return {
            "verdict_code": "UNDECIDABLE",
            "qwen35_outcome": "UNDECIDABLE",
            "reason": "Qwen3.5-0.8B English score is zero, so competence ratios are undefined",
        }

    exact_ratios = {language: _ratio_fraction(qwen35, language) for language in DECISION_LANGUAGES}
    _require(all(value is not None for value in exact_ratios.values()), "Qwen3.5 ratios are missing despite nonzero English score")
    ratios = {language: float(value) for language, value in exact_ratios.items() if value is not None}
    ratio_fractions = {language: _fraction_text(value) for language, value in exact_ratios.items() if value is not None}
    failures = [language for language in DECISION_LANGUAGES if exact_ratios[language] < COMPETENCE_THRESHOLD]
    base = {
        "threshold": 0.75,
        "threshold_exact_fraction": "3/4",
        "equality_passes": True,
        "qwen35_english_score_0_100": float(english_score),
        "qwen35_english_score_exact_fraction": _fraction_text(english_score),
        "qwen35_ratios": ratios,
        "qwen35_ratio_exact_fractions": ratio_fractions,
        "failed_languages": failures,
    }
    if not failures:
        return {
            **base,
            "verdict_code": "SUB1B_HOLDS",
            "qwen35_outcome": "SUB1B_HOLDS",
            "selected_model_id": "qwen35_08b",
            "selected_backbone_class": "SUB_1B",
            "proceeding_stacks": ["L", "D", "M"],
        }
    if failures == ["ru"]:
        return {
            **base,
            "verdict_code": "SUB1B_HOLDS_RU_REDUCED_SCOPE",
            "qwen35_outcome": "SUB1B_HOLDS_RU_REDUCED_SCOPE",
            "selected_model_id": "qwen35_08b",
            "selected_backbone_class": "SUB_1B",
            "proceeding_stacks": ["L", "D", "M"],
            "disclosure": "Russian is reduced-scope under the frozen RU-only exception.",
        }

    qf = compare_q_vs_f(models)
    if qf.get("status") != "DECIDED":
        return {
            **base,
            "verdict_code": "UNDECIDABLE",
            "qwen35_outcome": "MOVE_TO_Q_OR_F",
            "selected_backbone_class": "UNKNOWN",
            "reason": qf.get("reason", "Q/F comparison is undecidable"),
            "q_vs_f": qf,
            "unavailable_models": dict(unavailable),
        }
    leader = str(qf["leader_model_id"])
    return {
        **base,
        "verdict_code": "MOVE_TO_Q" if leader == "qwen25_15b" else "MOVE_TO_F",
        "qwen35_outcome": "MOVE_TO_Q_OR_F",
        "selected_model_id": leader,
        "selected_backbone_class": "Q_OR_F_1_5B",
        "proceeding_stacks": ["Q"] if leader == "qwen25_15b" else ["F"],
        "q_vs_f": qf,
    }


def _protocol_block(frozen: FrozenInputs, bootstrap_resamples: int) -> dict[str, Any]:
    return {
        "b1_prereg_sha256": frozen.prereg_sha256,
        "global_mmlu_manifest_sha256": frozen.mmlu_sha256,
        "native_generation_manifest_sha256": frozen.native_sha256,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "committed_harness_source_sha256": EXPECTED_HARNESS_SHA256,
        "committed_native_scorer_source_sha256": EXPECTED_SCORER_SHA256,
        "colab_setup_script_sha256": EXPECTED_SETUP_SHA256,
        "mcq_execution_batching": dict(MCQ_EXECUTION_BATCHING),
        "llama_cpp_tool_patch": {
            "source_relative_path": LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH,
            "source_original_sha256": LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256,
            "patch_relative_path": LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH,
            "patch_sha256": LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
            "source_patched_sha256": LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256,
            "source_status": LLAMA_CPP_EXPECTED_SOURCE_STATUS,
            "implementation_library_relative_path": LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH,
            "implementation_library_sha256": LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_SHA256,
            "launcher_sha256": LLAMA_CPP_PERPLEXITY_LAUNCHER_SHA256,
            "stock_scoring_unmodified": True,
        },
        "evidence_class": "HOST",
        "n_per_language": 120,
        "global_mmlu_n_per_language": 80,
        "native_generation_n_per_language": 40,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_implementation": {
            "rng": "Python stdlib random.Random (MT19937/Mersenne Twister)",
            "seed_scope": "Initialized once per complete assembly at seed 20260801; never reset per model-language cell.",
            "model_traversal_order": list(MODEL_IDS),
            "language_traversal_order": list(LANGUAGES),
            "partial_pack_state_rule": "Unavailable cells consume the same 120 random() values per replicate so every later available cell retains its frozen global-stream position.",
            "within_replicate_resampling_order": "80 equal-weight MCQ draws with replacement, then 40 equal-weight native-generation draws with replacement",
            "score_formula": "100 * (sum(MCQ draws) + sum(generation draws)) / 120",
            "percentile_interpolation": "Hyndman-Fan type 7: h=(N-1)*p, linear interpolation between floor(h) and ceil(h), p=0.025/0.975",
        },
        "assembly_environment": {
            "sys.version": sys.version,
            "python_version": sys.version.split()[0],
        },
        "append_only_execution_records": frozen.execution_records,
        "screen_warning": "Reduced-n screen, not n>=1,200 non-inferiority evidence and not native-speaker validation.",
    }


def _task_status(required: Sequence[str], models: Mapping[str, Any], unavailable: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    missing = {model_id: unavailable.get(model_id, "validated bundle unavailable") for model_id in required if model_id not in models}
    return ("COMPLETE" if not missing else "PARTIAL", missing)


def build_reports(
    frozen: FrozenInputs,
    models: Mapping[str, dict[str, Any]],
    unavailable: Mapping[str, str],
    *,
    completed_at_utc: str,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    protocol = _protocol_block(frozen, bootstrap_resamples)
    b3_status, b3_missing = _task_status(B3_MODEL_IDS, models, unavailable)
    b4_status, b4_missing = _task_status(B4_MODEL_IDS, models, unavailable)
    b5_status, b5_missing = _task_status(MODEL_IDS, models, unavailable)
    provisional_decision = apply_decision_rule(models, unavailable)
    all_four_arms_valid = set(models) == set(MODEL_IDS)
    if all_four_arms_valid:
        decision = provisional_decision
    else:
        decision = {
            "verdict_code": "UNDECIDABLE",
            "qwen35_outcome": provisional_decision.get("qwen35_outcome", "UNKNOWN"),
            "selected_backbone_class": "UNKNOWN",
            "reason": "A definitive Lane B programme verdict requires all four frozen arms to validate.",
            "unavailable_models": dict(unavailable),
            "provisional_complete-cell_rule_result": provisional_decision,
        }

    b3_models = {model_id: models[model_id] for model_id in B3_MODEL_IDS if model_id in models}
    b4_models = {model_id: models[model_id] for model_id in B4_MODEL_IDS if model_id in models}
    b3 = {
        "schema_version": SCHEMA_VERSION,
        "sprint_id": "B3",
        "status": b3_status,
        "completed_at_utc": completed_at_utc,
        "protocol": protocol,
        "models": b3_models,
        "missing_or_invalid_models": b3_missing,
        "interpretation": {
            "headline": "Five-language reduced-n HOST screen at shipping quantization.",
            "no_cross_language_average": True,
            "surface_proxy_is_not_native_fluency": True,
            "qwen35_decision_diagnostic": {
                "threshold": 0.75,
                "ratios": provisional_decision.get("qwen35_ratios"),
                "failed_languages": provisional_decision.get("failed_languages"),
                "outcome": provisional_decision.get("qwen35_outcome"),
                "provisional_only_until_all_four_arms_validate": not all_four_arms_valid,
            },
        },
    }
    b4 = {
        "schema_version": SCHEMA_VERSION,
        "sprint_id": "B4",
        "status": b4_status,
        "completed_at_utc": completed_at_utc,
        "protocol": protocol,
        "protocol_identity": "Identical frozen prompts, manifests, item counts, decoding, scorer, shipping-quantization venue and language-score formula as B3.",
        "models": b4_models,
        "missing_or_invalid_models": b4_missing,
        "scope_warning": "HOST response-quality screen only; no phone, AI Hub, accelerator, or Falcon portability evidence is inferred.",
    }
    b5 = {
        "schema_version": SCHEMA_VERSION,
        "sprint_id": "B5",
        "status": b5_status,
        "completed_at_utc": completed_at_utc,
        "evidence_class": "HOST",
        "denominator": "Exactly 40 native generations per language and 200 per complete model. The >=200 sprint requirement is met across all five languages per model, not per model-language.",
        "prior_context": {
            "evidence_class": "PRIOR",
            "reasoning_leakage_count": 11,
            "n": 63,
            "rate": 11 / 63,
            "not_substituted_for_current_measurement": True,
        },
        "models": {
            model_id: {
                "n": model["n_native_generations"],
                "denominator_at_least_200": model["n_native_generations"] >= 200,
                "languages": model["b5"]["languages"],
                "all_language_reasoning_leakage": model["b5"]["all_language_reasoning_leakage"],
                "mode_pin": model["mode_pin"],
                "source_hashes": model["source_hashes"],
            }
            for model_id, model in models.items()
        },
        "missing_or_invalid_models": b5_missing,
        "detector_limit": "Mechanical marker detector is a reproducible lower bound; marker-free prose reasoning can be missed.",
    }

    selected_id = decision.get("selected_model_id")
    selected = models.get(str(selected_id)) if selected_id else None
    per_language = {
        language: {
            "model_id": selected_id,
            "evidence_class": "HOST",
            "n": selected["languages"][language]["n"],
            "score_0_100": selected["languages"][language]["headline_score_0_100"],
            "score_exact_fraction": selected["languages"][language]["headline_score_exact_fraction"],
            "bootstrap_95": selected["languages"][language]["bootstrap_95"],
            "ratio_to_own_english": selected["languages"][language]["competence_ratio_to_en"],
            "ratio_to_own_english_exact_fraction": selected["languages"][language]["competence_ratio_exact_fraction"],
        }
        for language in LANGUAGES
    } if selected else {}
    qwen35_per_language = {
        language: {
            "evidence_class": "HOST",
            "n": models["qwen35_08b"]["languages"][language]["n"],
            "score_0_100": models["qwen35_08b"]["languages"][language]["headline_score_0_100"],
            "score_exact_fraction": models["qwen35_08b"]["languages"][language]["headline_score_exact_fraction"],
            "bootstrap_95": models["qwen35_08b"]["languages"][language]["bootstrap_95"],
            "ratio_to_en": models["qwen35_08b"]["languages"][language]["competence_ratio_to_en"],
            "ratio_to_en_exact_fraction": models["qwen35_08b"]["languages"][language]["competence_ratio_exact_fraction"],
        }
        for language in LANGUAGES
    } if "qwen35_08b" in models else {}

    verdict_code = decision["verdict_code"]
    if verdict_code == "SUB1B_HOLDS":
        verdict_text = "SUB-1B HOLDS — Qwen3.5-0.8B clears the 0.75x English floor in ES, FR and RU."
    elif verdict_code == "SUB1B_HOLDS_RU_REDUCED_SCOPE":
        verdict_text = "SUB-1B HOLDS WITH RUSSIAN REDUCED-SCOPE — RU alone misses the 0.75x floor."
    elif verdict_code == "MOVE_TO_Q":
        verdict_text = "MOVE TO Q — the sub-1B branch fails and Qwen2.5-1.5B leads Falcon-H1 under the frozen tie-break."
    elif verdict_code == "MOVE_TO_F":
        verdict_text = "MOVE TO F — the sub-1B branch fails and Falcon-H1-1.5B leads Qwen2.5 under the frozen tie-break."
    else:
        verdict_text = "UNDECIDABLE — the frozen rule cannot select an admissible backbone from the available HOST evidence."

    confidence = (
        "LIMITED: deterministic HOST screen at n=120/language with only 40 machine-scored native prompts and no native-speaker validation."
        if selected
        else "UNKNOWN: no definitive selected-model HOST cells are available; the frozen programme verdict remains undecidable."
    )
    backbone_verdict = {
        "schema_version": "lane-b-backbone-verdict/1.0",
        "sprint_id": "B6",
        "completed_at_utc": completed_at_utc,
        "verdict": verdict_text,
        "verdict_code": verdict_code,
        "selected_backbone_class": decision.get("selected_backbone_class", "UNKNOWN"),
        "selected_model_id": selected_id,
        "proceeding_stacks": decision.get("proceeding_stacks", []),
        "evidence_class": "HOST" if selected else "UNKNOWN",
        "per_language": per_language,
        "qwen35_decision_per_language": qwen35_per_language,
        "decision_trace": decision,
        "protocol": protocol,
        "all_model_language_scores": {
            model_id: {
                language: {
                    "score_0_100": model["languages"][language]["headline_score_0_100"],
                    "score_exact_fraction": model["languages"][language]["headline_score_exact_fraction"],
                    "n": model["languages"][language]["n"],
                    "evidence_class": "HOST",
                }
                for language in LANGUAGES
            }
            for model_id, model in models.items()
        },
        "confidence": confidence,
        "unknowns": [
            "Native grammaticality, cultural appropriateness and naturalness remain UNKNOWN.",
            "Phone latency, memory, thermals and accelerator portability remain UNKNOWN in Lane B.",
        ],
    }
    b6 = {
        "schema_version": SCHEMA_VERSION,
        "sprint_id": "B6",
        "status": ("COMPLETE" if verdict_code != "UNDECIDABLE" else "UNDECIDABLE") if all_four_arms_valid else "PARTIAL",
        "completed_at_utc": completed_at_utc,
        "mechanical_rule_applied": all_four_arms_valid,
        "provisional_rule_applied_to_available_complete_cells": True,
        "decision_trace": decision,
        "backbone_verdict": backbone_verdict,
        "task_completeness": {"B3": b3_status, "B4": b4_status, "B5": b5_status},
    }
    return {"B3": b3, "B4": b4, "B5": b5, "B6": b6, "backbone_verdict": backbone_verdict}


def _fmt_score(value: float | None) -> str:
    return "UNKNOWN" if value is None else f"{value:.2f}"


def _fmt_ratio(value: float | None) -> str:
    return "UNKNOWN" if value is None else f"{value:.3f}x"


def _fmt_count(summary: Mapping[str, Any]) -> str:
    n = summary.get("n", 0)
    count = summary.get("count", 0)
    return f"{count}/{n}" if n else "0/0 (ineligible)"


def _score_table(models: Mapping[str, dict[str, Any]]) -> str:
    lines = [
        "| Model | Lang | HOST score /100 (n=120) | Bootstrap 95% | MMLU | Native correctness | Surface proxy | Concision | Ratio to EN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, model in models.items():
        for language in LANGUAGES:
            cell = model["languages"][language]
            ci = cell["bootstrap_95"]
            native = cell["native_generation"]
            lines.append(
                f"| {model_id} | {language.upper()} | {_fmt_score(cell['headline_score_0_100'])} | "
                f"[{ci[0]:.2f}, {ci[1]:.2f}] | {cell['global_mmlu']['correct']}/80 | "
                f"{native['correctness']['count']}/40 | {100*native['mean_surface_fluency_proxy']:.1f}% | "
                f"{native['concision']['count']}/40 | {_fmt_ratio(cell['competence_ratio_to_en'])} |"
            )
    return "\n".join(lines)


def render_b3_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# B3 — five-language backbone screen",
        "",
        f"**Status:** {report['status']}. All numeric result cells are `HOST`; n=80 Global-MMLU + 40 native generations per language.",
        "",
        _score_table(report["models"]),
        "",
    ]
    diagnostic = report["interpretation"]["qwen35_decision_diagnostic"]
    lines.append(
        "Qwen3.5 diagnostic: "
        + (f"{diagnostic['outcome']}; ratios " + ", ".join(f"{lang.upper()}={_fmt_ratio(value)}" for lang, value in diagnostic["ratios"].items()) if diagnostic.get("ratios") else "UNKNOWN")
        + "."
    )
    lines.extend([
        "",
        "The surface-fluency quantity is a mechanical language/script/degeneracy/speakability proxy, not a native-speaker fluency or naturalness score. This is a reduced-n screen, not the parent plan's n>=1,200 non-inferiority study; no cross-language average is reported.",
    ])
    if report["missing_or_invalid_models"]:
        lines.extend(["", "Missing or invalid arms: " + json.dumps(report["missing_or_invalid_models"], ensure_ascii=False) + "."])
    return "\n".join(lines) + "\n"


def render_b4_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# B4 — Falcon-H1 identical-protocol arm",
        "",
        f"**Status:** {report['status']}. The Falcon arm uses the byte-identical frozen manifests, scoring rule, decoding settings and HOST venue class used by B3.",
        "",
    ]
    if report["models"]:
        lines.extend([_score_table(report["models"]), ""])
    lines.append("This result is a response-quality screen only. It supplies no phone, AI Hub, memory, latency, or Falcon accelerator-portability evidence.")
    if report["missing_or_invalid_models"]:
        lines.extend(["", "Missing or invalid arm: " + json.dumps(report["missing_or_invalid_models"], ensure_ascii=False) + "."])
    return "\n".join(lines) + "\n"


def render_b5_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# B5 — failure-mode counters",
        "",
        f"**Status:** {report['status']}. Counts are mechanical `HOST` measurements over 40 generations per language and 200 per complete model.",
        "",
        "| Model | Lang | Wrong language / eligible | Wrong script / eligible | Traditional Han / eligible | Reasoning leakage / 40 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, model in report["models"].items():
        for language in LANGUAGES:
            cell = model["languages"][language]
            traditional = cell["traditional_han_leakage"]
            lines.append(
                f"| {model_id} | {language.upper()} | {_fmt_count(cell['wrong_language'])} | "
                f"{_fmt_count(cell['wrong_script'])} | {_fmt_count(traditional) if traditional else 'n/a'} | "
                f"{_fmt_count(cell['reasoning_leakage'])} |"
            )
    lines.extend(["", "Model-wide reasoning leakage (the registered >=200 denominator):"])
    for model_id, model in report["models"].items():
        leakage = model["all_language_reasoning_leakage"]
        void = model["mode_pin"]["ranking_void"]
        lines.append(f"- {model_id}: {_fmt_count(leakage)}; ranking void under >0.5% mode-pin rule: {'yes' if void else 'no'}.")
    lines.extend([
        "",
        f"Prior context is 11/63 = {100*report['prior_context']['rate']:.1f}% (`PRIOR`), shown only as context and never substituted for this run. The regex detector is a lower bound: marker-free prose reasoning can be missed. Ineligible short numeric/symbol answers are excluded from wrong-language rates rather than counted as language evidence.",
    ])
    if report["missing_or_invalid_models"]:
        lines.extend(["", "Missing or invalid arms: " + json.dumps(report["missing_or_invalid_models"], ensure_ascii=False) + "."])
    return "\n".join(lines) + "\n"


def render_b6_md(report: Mapping[str, Any]) -> str:
    verdict = report["backbone_verdict"]
    decision = report["decision_trace"]
    evidence_line = (
        f"Evidence: `{verdict['evidence_class']}` reduced-n screen, n=120 per language for every selected-model cell."
        if verdict["per_language"]
        else "Evidence: `UNKNOWN`; no definitive selected-model HOST cells are available."
    )
    lines = [
        "# B6 — mechanical backbone verdict",
        "",
        f"**{verdict['verdict']}**",
        "",
        evidence_line,
        "",
    ]
    if decision.get("qwen35_ratios"):
        lines.append("Qwen3.5-0.8B rule inputs: " + ", ".join(f"{lang.upper()}={_fmt_ratio(value)}" for lang, value in decision["qwen35_ratios"].items()) + "; threshold = 0.750x (equality passes).")
        lines.append("")
    if decision.get("q_vs_f"):
        qf = decision["q_vs_f"]
        if qf.get("status") == "DECIDED":
            lines.append(f"Q/F leader: `{qf['leader_model_id']}` at tie-break step `{qf['deciding_step']}`.")
        else:
            lines.append(f"Q/F comparison: UNDECIDABLE — {qf.get('reason')}.")
        lines.append("")
    if verdict["per_language"]:
        lines.extend([
            "| Lang | Selected model | HOST score /100 | n | Ratio to own EN |",
            "|---|---|---:|---:|---:|",
        ])
        for language, cell in verdict["per_language"].items():
            lines.append(f"| {language.upper()} | {cell['model_id']} | {_fmt_score(cell['score_0_100'])} | {cell['n']} | {_fmt_ratio(cell['ratio_to_own_english'])} |")
        lines.append("")
    lines.append(verdict["confidence"])
    lines.append("Native naturalness/cultural validity and all phone/device properties remain UNKNOWN; the screen cannot promote those evidence classes.")
    return "\n".join(lines) + "\n"


def render_lane_summary(reports: Mapping[str, Any]) -> str:
    verdict = reports["backbone_verdict"]
    selected_evidence_summary = (
        "The decision is mechanically faithful to the frozen B1 point-estimate rule and every selected-model score is `HOST` at n=120/language. Confidence is limited because only 40 prompts/language exercise native generation, their rubrics are mechanical, and no native speaker validated fluency, grammar or cultural appropriateness."
        if verdict["per_language"]
        else "No definitive selected-model HOST cells are available, so the programme verdict remains undecidable and evidence confidence is `UNKNOWN`."
    )
    task_statuses = {
        "B1": "COMPLETE",
        "B2": "COMPLETE_WITH_LIVE_CELL_DEPENDENCY_UNMET",
        "B3": reports["B3"]["status"],
        "B4": reports["B4"]["status"],
        "B5": reports["B5"]["status"],
        "B6": reports["B6"]["status"],
    }
    lines = [
        "# Lane B summary — backbone decision",
        "",
        f"## Headline",
        "",
        f"**{verdict['verdict']}**",
        "",
        "## Confidence",
        "",
        verdict["confidence"],
        "",
        selected_evidence_summary,
        "",
        "## Decision cells",
        "",
    ]
    if verdict["qwen35_decision_per_language"]:
        lines.extend([
            "| Lang | Qwen3.5 HOST score /100 | n | Ratio to EN |",
            "|---|---:|---:|---:|",
        ])
        for language, cell in verdict["qwen35_decision_per_language"].items():
            lines.append(f"| {language.upper()} | {_fmt_score(cell['score_0_100'])} | {cell['n']} | {_fmt_ratio(cell['ratio_to_en'])} |")
    lines.extend(["", "## Failure modes", ""])
    for model_id, model in reports["B5"]["models"].items():
        leakage = model["all_language_reasoning_leakage"]
        lines.append(f"- {model_id}: reasoning leakage {_fmt_count(leakage)} across five languages; ranking void = {'yes' if model['mode_pin']['ranking_void'] else 'no'}.")
    lines.extend([
        "",
        "Wrong-language denominators exclude outputs with fewer than eight Unicode letters; B5 therefore reports eligible n explicitly. Traditional-Han leakage is measured on raw ZH output before normalization.",
        "",
        "## Completeness and limits",
        "",
    ])
    for sprint_id, status in task_statuses.items():
        lines.append(f"- {sprint_id}: {status}")
    lines.extend([
        "",
        "No training, fine-tuning, conversion or local >1B inference was performed. Lane B does not establish phone latency, PSS, thermal behaviour, accelerator compatibility, or native-speaker acceptance; those remain `UNKNOWN` here.",
    ])
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise AssemblyError(f"refusing to overwrite existing report without --replace: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_reports(output_dir: Path, reports: Mapping[str, Any], *, replace: bool) -> list[Path]:
    payloads: list[tuple[str, bytes]] = [
        ("B3.json", json_bytes(reports["B3"])),
        ("B3.md", render_b3_md(reports["B3"]).encode("utf-8")),
        ("B4.json", json_bytes(reports["B4"])),
        ("B4.md", render_b4_md(reports["B4"]).encode("utf-8")),
        ("B5.json", json_bytes(reports["B5"])),
        ("B5.md", render_b5_md(reports["B5"]).encode("utf-8")),
        ("B6.json", json_bytes(reports["B6"])),
        ("B6.md", render_b6_md(reports["B6"]).encode("utf-8")),
        ("backbone_verdict.json", json_bytes(reports["backbone_verdict"])),
        # LANE_SUMMARY.md is intentionally last, per the lane output contract.
        ("LANE_SUMMARY.md", render_lane_summary(reports).encode("utf-8")),
    ]
    if not replace:
        conflicts = [str(output_dir / name) for name, _ in payloads if (output_dir / name).exists()]
        _require(not conflicts, "refusing partial write because report files already exist: " + ", ".join(conflicts))
    written: list[Path] = []
    for name, data in payloads:
        path = output_dir / name
        _atomic_write(path, data, replace=replace)
        written.append(path)
    return written


def assemble(
    lane_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    score_report_paths: Sequence[Path] = (),
    allow_incomplete: bool = False,
    replace: bool = False,
    completed_at_utc: str | None = None,
    _test_bootstrap_resamples: int | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    frozen = load_frozen_inputs(lane_dir)
    bundles, unavailable = discover_bundles(
        artifacts_dir,
        score_report_paths,
        allow_incomplete=allow_incomplete,
    )
    evidence: dict[str, ModelEvidence] = {}
    for model_id in MODEL_IDS:
        bundle = bundles.get(model_id)
        if bundle is None:
            continue
        try:
            evidence[model_id] = validate_bundle(bundle, frozen)
        except AssemblyError as exc:
            if not allow_incomplete:
                raise
            unavailable[model_id] = f"invalid artifact: {exc}"

    resamples = BOOTSTRAP_RESAMPLES if _test_bootstrap_resamples is None else _test_bootstrap_resamples
    rng = random.Random(BOOTSTRAP_SEED)
    models: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        if model_id in evidence:
            models[model_id] = summarize_model(
                evidence[model_id],
                frozen,
                rng=rng,
                bootstrap_resamples=resamples,
            )
        else:
            advance_missing_model_bootstrap_stream(rng=rng, resamples=resamples)
    reports = build_reports(
        frozen,
        models,
        unavailable,
        completed_at_utc=completed_at_utc or utc_now(),
        bootstrap_resamples=resamples,
    )
    written = write_reports(output_dir, reports, replace=replace)
    return reports, written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-dir", type=Path, default=SCRIPT_DIR, help="Lane B output directory containing frozen B1 inputs")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Root containing copied harness raw artifacts, receipts and scorer reports")
    parser.add_argument("--score-report", type=Path, action="append", default=[], help="Explicit native scorer JSON; repeat for per-model reports")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Directory for B3--B6 reports")
    parser.add_argument("--allow-incomplete", action="store_true", help="Emit honest PARTIAL/UNDECIDABLE reports instead of failing on missing/invalid arms")
    parser.add_argument("--replace", action="store_true", help="Replace existing B3--B6 reports atomically")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reports, written = assemble(
            args.lane_dir.resolve(),
            args.artifacts_dir.resolve(),
            args.output_dir.resolve(),
            score_report_paths=[path.resolve() for path in args.score_report],
            allow_incomplete=args.allow_incomplete,
            replace=args.replace,
        )
    except AssemblyError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "PASS",
        "verdict_code": reports["backbone_verdict"]["verdict_code"],
        "written": [str(path) for path in written],
        "lane_summary_written_last": bool(written and written[-1].name == "LANE_SUMMARY.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
