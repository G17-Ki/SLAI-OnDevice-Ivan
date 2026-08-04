#!/usr/bin/env python3
"""Focused tests for the Lane B B3--B6 assembler."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import assemble_b3_b6 as assembler


LANE_DIR = Path(__file__).resolve().parent


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_summary(scores: dict[str, float], *, admissible: bool = True) -> dict[str, object]:
    exact_scores = {language: Fraction(str(score)) for language, score in scores.items()}
    en = exact_scores["en"]
    return {
        "ranking_admissible": admissible,
        "mode_pin": {"ranking_void": not admissible},
        "languages": {
            language: {
                "headline_score_0_100": score,
                "headline_score_exact_fraction": str(exact_scores[language]),
                "competence_ratio_to_en": float(exact_scores[language] / en) if en else None,
                "competence_ratio_exact_fraction": str(exact_scores[language] / en) if en else None,
            }
            for language, score in scores.items()
        },
    }


class DecisionRuleTests(unittest.TestCase):
    def test_all_three_ratios_pass_including_exact_threshold(self) -> None:
        models = {
            "qwen35_08b": _model_summary({"en": 80.0, "zh": 70.0, "es": 60.0, "fr": 64.0, "ru": 60.0}),
        }
        result = assembler.apply_decision_rule(models, {})
        self.assertEqual(result["verdict_code"], "SUB1B_HOLDS")
        self.assertEqual(result["failed_languages"], [])

    def test_ru_only_failure_uses_reduced_scope_exception(self) -> None:
        models = {
            "qwen35_08b": _model_summary({"en": 80.0, "zh": 70.0, "es": 65.0, "fr": 64.0, "ru": 59.9}),
        }
        result = assembler.apply_decision_rule(models, {})
        self.assertEqual(result["verdict_code"], "SUB1B_HOLDS_RU_REDUCED_SCOPE")
        self.assertEqual(result["failed_languages"], ["ru"])

    def test_es_only_failure_moves_to_language_win_leader(self) -> None:
        models = {
            "qwen35_08b": _model_summary({"en": 80.0, "zh": 70.0, "es": 59.0, "fr": 64.0, "ru": 62.0}),
            "qwen25_15b": _model_summary({"en": 70.0, "zh": 70.0, "es": 70.0, "fr": 70.0, "ru": 70.0}),
            "falcon_h1_15b": _model_summary({"en": 69.0, "zh": 69.0, "es": 69.0, "fr": 69.0, "ru": 69.0}),
        }
        result = assembler.apply_decision_rule(models, {})
        self.assertEqual(result["qwen35_outcome"], "MOVE_TO_Q_OR_F")
        self.assertEqual(result["verdict_code"], "MOVE_TO_Q")
        self.assertEqual(result["q_vs_f"]["deciding_step"], "language_wins")

    def test_qf_tie_break_uses_minimum_ratio_then_incumbent(self) -> None:
        q = _model_summary({"en": 100.0, "zh": 50.0, "es": 90.0, "fr": 80.0, "ru": 70.0})
        f = _model_summary({"en": 99.0, "zh": 51.0, "es": 89.0, "fr": 81.0, "ru": 70.0})
        result = assembler.compare_q_vs_f({"qwen25_15b": q, "falcon_h1_15b": f})
        self.assertEqual(result["leader_model_id"], "falcon_h1_15b")
        self.assertEqual(result["deciding_step"], "higher_minimum_es_fr_ru_ratio")

        identical = _model_summary({"en": 50.0, "zh": 50.0, "es": 50.0, "fr": 50.0, "ru": 50.0})
        final = assembler.compare_q_vs_f({"qwen25_15b": identical, "falcon_h1_15b": identical})
        self.assertEqual(final["leader_model_id"], "qwen25_15b")
        self.assertEqual(final["deciding_step"], "incumbent_headroom_tie_break")

    def test_void_or_zero_english_is_undecidable(self) -> None:
        void = _model_summary({"en": 80.0, "zh": 70.0, "es": 70.0, "fr": 70.0, "ru": 70.0}, admissible=False)
        self.assertEqual(assembler.apply_decision_rule({"qwen35_08b": void}, {})["verdict_code"], "UNDECIDABLE")
        zero = _model_summary({"en": 0.0, "zh": 0.0, "es": 0.0, "fr": 0.0, "ru": 0.0})
        self.assertEqual(assembler.apply_decision_rule({"qwen35_08b": zero}, {})["verdict_code"], "UNDECIDABLE")


class StatisticsTests(unittest.TestCase):
    def test_wilson_and_bootstrap_are_deterministic(self) -> None:
        self.assertEqual(assembler.wilson_95(0, 0), None)
        interval = assembler.wilson_95(80, 80)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLess(interval[0], 1.0)
        self.assertAlmostEqual(interval[1], 1.0)

        mcq = [1.0] * 40 + [0.0] * 40
        native = [1.0] * 20 + [0.0] * 20
        first = assembler.stratified_bootstrap_95(mcq, native, rng=random.Random(20260801), resamples=200)
        second = assembler.stratified_bootstrap_95(mcq, native, rng=random.Random(20260801), resamples=200)
        self.assertEqual(first, second)
        self.assertLess(first[0], 50.0)
        self.assertGreater(first[1], 50.0)

    def test_missing_arm_advances_the_frozen_global_rng_position(self) -> None:
        advanced = random.Random(assembler.BOOTSTRAP_SEED)
        assembler.advance_missing_model_bootstrap_stream(rng=advanced, resamples=2)
        explicit = random.Random(assembler.BOOTSTRAP_SEED)
        for _language in assembler.LANGUAGES:
            for _ in range(2):
                explicit.choices([0.0] * 80, k=80)
                explicit.choices([0.0] * 40, k=40)
        self.assertEqual(advanced.getstate(), explicit.getstate())


class ExecutionRecordTests(unittest.TestCase):
    def test_zero_result_correction_chain_is_the_only_admitted_post_attempt_history(self) -> None:
        prereg_sha = _sha(LANE_DIR / "b1_prereg.json")
        for specification in (
            assembler.RESOURCE_CORRECTION_RECORD,
            assembler.RESOURCE_CORRECTION_FALLBACK_RECORD,
            assembler.TOOL_MEMORY_PATCH_RECORD,
        ):
            with self.subTest(record=specification["name"]):
                metadata = assembler._validate_execution_record(LANE_DIR / specification["name"], prereg_sha)
                self.assertFalse(metadata["recorded_before_candidate_inference"])
                self.assertTrue(metadata["recorded_after_failed_unscored_candidate_attempt"])
                self.assertEqual(metadata["candidate_result_rows_observed"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            arbitrary = Path(temporary) / "b1_execution_response_conditioned_retune.json"
            _write_json(arbitrary, {
                "schema_version": "lane-b-execution-resource-correction/1.0",
                "sprint_id": "B1",
                "recorded_before_candidate_inference": False,
                "recorded_after_failed_unscored_candidate_attempt": True,
                "candidate_result_rows_observed": 0,
            })
            with self.assertRaisesRegex(assembler.AssemblyError, "unexpected append-only execution record"):
                assembler._validate_execution_record(arbitrary, prereg_sha)

    def test_resource_correction_after_any_result_row_is_rejected(self) -> None:
        prereg_sha = _sha(LANE_DIR / "b1_prereg.json")
        for specification in (
            assembler.RESOURCE_CORRECTION_RECORD,
            assembler.RESOURCE_CORRECTION_FALLBACK_RECORD,
            assembler.TOOL_MEMORY_PATCH_RECORD,
        ):
            with self.subTest(record=specification["name"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in assembler.EXECUTION_RECORD_ORDER:
                    (root / name).write_bytes((LANE_DIR / name).read_bytes())
                source = LANE_DIR / specification["name"]
                record = json.loads(source.read_text())
                record["candidate_result_rows_observed"] = 1
                resource = root / source.name
                _write_json(resource, record)
                with self.assertRaisesRegex(assembler.AssemblyError, "inadmissible after any candidate result row"):
                    assembler._validate_execution_record(resource, prereg_sha)

    def test_exact_post_run_build_provenance_addendum_is_admitted(self) -> None:
        prereg_sha = _sha(LANE_DIR / "b1_prereg.json")
        addendum_path = LANE_DIR / assembler.BUILD_PROVENANCE_ADDENDUM_RECORD["name"]
        metadata = assembler._validate_execution_record(addendum_path, prereg_sha)
        self.assertFalse(metadata["recorded_before_candidate_inference"])
        self.assertTrue(metadata["recorded_after_candidate_inference"])
        self.assertEqual(metadata["candidate_result_rows_observed"], 600)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in assembler.EXECUTION_RECORD_ORDER:
                (root / name).write_bytes((LANE_DIR / name).read_bytes())
            record = json.loads(addendum_path.read_text())
            record["candidate_result_rows_observed"] = 599
            tampered = root / addendum_path.name
            _write_json(tampered, record)
            with self.assertRaisesRegex(assembler.AssemblyError, r"complete 400\+200 superseded arm"):
                assembler._validate_execution_record(tampered, prereg_sha)


class ArchivedExecutionEvidenceTests(unittest.TestCase):
    def _write_fixture(self, root: Path, *, first_log: bytes = b"header only\n") -> tuple[list[Path], list[dict[str, object]]]:
        failure_dir = root / "artifacts" / "failures"
        failure_dir.mkdir(parents=True)
        log_payloads = {
            assembler.RESOURCE_CORRECTION_RECORD["name"]: (
                "qwen35_08b.mcq.batch4096.parallel4.sigkill.log",
                first_log,
            ),
            assembler.RESOURCE_CORRECTION_FALLBACK_RECORD["name"]: (
                "qwen35_08b.mcq.ctx4096.batch1024.ubatch128.parallel4.sigkill.log",
                b"second header only\n",
            ),
            assembler.TOOL_MEMORY_PATCH_RECORD["name"]: (
                "qwen35_08b.mcq.ctx3072.batch1024.ubatch128.parallel4.sigkill.log",
                b"third header only\n",
            ),
        }
        record_paths: list[Path] = []
        metadata: list[dict[str, object]] = []
        for record_name, (log_name, payload) in log_payloads.items():
            log_path = failure_dir / log_name
            log_path.write_bytes(payload)
            record_path = root / record_name
            _write_json(record_path, {
                "failed_attempt": {
                    "log_artifact": f"failures/{log_name}",
                    "log_bytes": len(payload),
                    "log_sha256": hashlib.sha256(payload).hexdigest(),
                },
            })
            record_paths.append(record_path)
            metadata.append({"path": record_name})

        archive_payload = b"preserved superseded evidence"
        archive_relative = "superseded/test/archives/qwen35_08b.evidence.tar.gz"
        archive_path = root / "artifacts" / archive_relative
        archive_path.parent.mkdir(parents=True)
        archive_path.write_bytes(archive_payload)
        addendum_path = root / assembler.BUILD_PROVENANCE_ADDENDUM_RECORD["name"]
        _write_json(addendum_path, {
            "superseded_candidate_run": {
                "preserved_archive": {
                    "path": archive_relative,
                    "sha256": hashlib.sha256(archive_payload).hexdigest(),
                },
            },
        })
        record_paths.append(addendum_path)
        metadata.append({"path": assembler.BUILD_PROVENANCE_ADDENDUM_RECORD["name"]})
        return record_paths, metadata

    def test_archived_evidence_is_bound_into_execution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_paths, metadata = self._write_fixture(root)
            assembler._validate_archived_execution_evidence(root, record_paths, metadata)
            first = metadata[0]["failed_attempt_log_evidence"]
            self.assertEqual(first["cumulative_candidate_result_rows"], 0)
            self.assertTrue(first["path"].startswith("artifacts/failures/"))
            addendum = metadata[-1]["superseded_evidence_archive"]
            self.assertTrue(addendum["excluded_from_final_analysis"])

    def test_hash_valid_log_with_candidate_result_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_paths, metadata = self._write_fixture(root, first_log=b"task acc_norm\n1 0.50000000\n")
            with self.assertRaisesRegex(assembler.AssemblyError, "contains 1 cumulative candidate-result rows"):
                assembler._validate_archived_execution_evidence(root, record_paths, metadata)


class ScorerRecomputationComparisonTests(unittest.TestCase):
    def test_lingua_probability_roundoff_is_tolerated(self) -> None:
        recorded = {
            "surface": {
                "language_detection": {
                    "confidences": {"en": 0.9911291426655663},
                    "expected_confidence": 0.9911291426655663,
                    "wrong_language": False,
                },
            },
        }
        recomputed = {
            "surface": {
                "language_detection": {
                    "confidences": {"en": 0.9911291426655662},
                    "expected_confidence": 0.9911291426655662,
                    "wrong_language": False,
                },
            },
        }
        self.assertIsNone(assembler._score_recomputation_mismatch(recorded, recomputed))

    def test_material_or_categorical_scorer_changes_are_rejected(self) -> None:
        recorded = {
            "generation_item_score": 0.9,
            "surface": {
                "language_detection": {
                    "confidences": {"en": 0.9},
                    "wrong_language": False,
                },
            },
        }
        material_probability = json.loads(json.dumps(recorded))
        material_probability["surface"]["language_detection"]["confidences"]["en"] = 0.8
        self.assertEqual(
            assembler._score_recomputation_mismatch(recorded, material_probability),
            "surface.language_detection.confidences.en",
        )
        changed_decision = json.loads(json.dumps(recorded))
        changed_decision["surface"]["language_detection"]["wrong_language"] = True
        self.assertEqual(
            assembler._score_recomputation_mismatch(recorded, changed_decision),
            "surface.language_detection.wrong_language",
        )
        changed_score = json.loads(json.dumps(recorded))
        changed_score["generation_item_score"] = 0.9000000000000001
        self.assertEqual(
            assembler._score_recomputation_mismatch(recorded, changed_score),
            "generation_item_score",
        )


class FullArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = assembler.load_frozen_inputs(LANE_DIR)
        cls.scorer = assembler.load_pinned_scorer(LANE_DIR)
        cls.harness = assembler.load_pinned_harness(LANE_DIR)

    def _make_model_artifacts(self, root: Path, model_id: str) -> None:
        frozen = self.frozen
        config = frozen.model_configs[model_id]
        raw_dir = root / "raw" / model_id
        render_dir = root / "rendered" / model_id
        qwen_nonthinking = model_id in {"qwen35_08b", "qwen3_06b"}
        build_receipt = {
            "schema_version": "lane-b-b34-build-receipt/1.0",
            "status": "PASS",
            "harness_source_sha256": assembler.EXPECTED_HARNESS_SHA256,
            "llama_cpp_commit": assembler.LLAMA_CPP_COMMIT,
            "llama_cpp_source": "/content/lane_b_b34/llama.cpp",
            "source_tree_clean": False,
            "source_tree_patch_exact": True,
            "source_status": assembler.LLAMA_CPP_EXPECTED_SOURCE_STATUS,
            "multiple_choice_source_relative_path": assembler.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH,
            "multiple_choice_source_original_sha256": assembler.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256,
            "multiple_choice_source_sha256": assembler.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256,
            "multiple_choice_patch_relative_path": assembler.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH,
            "multiple_choice_patch_sha256": assembler.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
            "ggml_cuda": True,
            "cmake_cache_sha256": "4" * 64,
            "cmake_configuration": {
                "cache_path": "/content/lane_b_b34/llama.cpp/build-b34/CMakeCache.txt",
                "cache_sha256": "4" * 64,
                "CMAKE_HOME_DIRECTORY": "/content/lane_b_b34/llama.cpp",
                "CMAKE_CACHEFILE_DIR": "/content/lane_b_b34/llama.cpp/build-b34",
                "CMAKE_BUILD_TYPE": "Release",
                "GGML_CUDA": "ON",
                "BUILD_SHARED_LIBS": "ON",
                "llama_bin_dir": "/content/lane_b_b34/llama.cpp/build-b34/bin",
            },
            "multiple_choice_impl_library": {
                "path": "/content/lane_b_b34/llama.cpp/build-b34/bin/libllama-perplexity-impl.so",
                "relative_path": assembler.LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_RELATIVE_PATH,
                "sha256": assembler.LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_SHA256,
            },
            "binaries": {
                "llama-perplexity": {
                    "path": "/content/lane_b_b34/llama.cpp/build-b34/bin/llama-perplexity",
                    "sha256": assembler.LLAMA_CPP_PERPLEXITY_LAUNCHER_SHA256,
                },
                "llama-server": {
                    "path": "/content/lane_b_b34/llama.cpp/build-b34/bin/llama-server",
                    "sha256": "2" * 64,
                },
            },
        }
        build_path = root / "build_receipt.json"
        _write_json(build_path, build_receipt)

        first_mcq: dict[str, str] = {}
        cumulative = 0
        mcq_rows: list[dict[str, object]] = []
        mmlu_rendered_rows: list[dict[str, object]] = []
        llama_lines: list[str] = []
        for ordinal, source in enumerate(frozen.mmlu_rows):
            prompt = ("<think></think> " if qwen_nonthinking else "") + f"mcq {model_id} {source['item_id']}"
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            first_mcq.setdefault(source["language"], prompt_sha)
            correct = ordinal % 2 == 0
            cumulative += int(correct)
            llama_line = f"{ordinal + 1} {100.0 * cumulative / (ordinal + 1):.8f}"
            llama_lines.append(llama_line)
            mmlu_rendered_rows.append({
                "schema_version": "lane-b-b34-rendered/1.0",
                "suite": "global_mmlu",
                "ordinal": ordinal,
                "model_id": model_id,
                "item_id": source["item_id"],
                "sample_id": source["sample_id"],
                "language": source["language"],
                "gold_answer": source["answer"],
                "candidate_answers": ["A", "B", "C", "D"],
                "candidate_continuations": [" A", " B", " C", " D"],
                "messages": [
                    {"role": "system", "content": frozen.prereg["prompts"]["system_by_language"][source["language"]]},
                    {"role": "user", "content": frozen.prereg["prompts"]["mcq_user_template"].format(
                        question=source["question"],
                        option_a=source["option_a"],
                        option_b=source["option_b"],
                        option_c=source["option_c"],
                        option_d=source["option_d"],
                        localized_instruction=frozen.prereg["prompts"]["mcq_instruction_by_language"][source["language"]],
                    )},
                ],
                "rendered_prompt": prompt,
                "prompt_sha256": prompt_sha,
                "hf_prompt_token_count": 1,
                "source_item": source,
            })
            mcq_rows.append({
                "schema_version": "lane-b-b34-raw/1.0",
                "run_kind": "global_mmlu_log_likelihood",
                "evidence_class": "HOST",
                "model_id": model_id,
                "quant_format": config["quant_format"],
                "quant_file_sha256": config["quant_file_sha256"],
                "llama_cpp_commit": assembler.LLAMA_CPP_COMMIT,
                "ordinal": ordinal,
                "item_id": source["item_id"],
                "sample_id": source["sample_id"],
                "language": source["language"],
                "prompt_sha256": prompt_sha,
                "rendered_prompt": prompt,
                "candidate_answers": ["A", "B", "C", "D"],
                "candidate_continuations": [" A", " B", " C", " D"],
                "choice_mean_logprobs": None,
                "predicted_answer": None,
                "gold_index": ["A", "B", "C", "D"].index(source["answer"]),
                "gold_answer": source["answer"],
                "correct": correct,
                "cumulative_correct": cumulative,
                "cumulative_accuracy_pct": float(f"{100.0 * cumulative / (ordinal + 1):.8f}"),
                "raw_llama_line": llama_line,
                "source_item": source,
            })
        mcq_path = raw_dir / "global_mmlu.raw.jsonl"
        _write_jsonl(mcq_path, mcq_rows)
        mcq_log = raw_dir / "global_mmlu.raw.jsonl.llama.log"
        mcq_log.write_text("\n".join(llama_lines) + "\n", encoding="utf-8")
        _write_json(Path(str(mcq_path) + ".receipt.json"), {
            "schema_version": "lane-b-b34-run-receipt/1.0",
            "run_kind": "global_mmlu_log_likelihood",
            "status": "PASS",
            "harness_source_sha256": assembler.EXPECTED_HARNESS_SHA256,
            "model_id": model_id,
            "b1_prereg_sha256": frozen.prereg_sha256,
            "render_receipt_sha256": None,
            "build_receipt_sha256": _sha(build_path),
            "command": [
                "/content/bin/llama-perplexity",
                "--model", "/content/current-model/" + config["quant_file"],
                "--binary-file", "/content/rendered/" + model_id + "/global_mmlu.multiple_choice.bin",
                "--multiple-choice",
                "--ctx-size", "4096",
                "--batch-size", "1024",
                "--ubatch-size", "128",
                "--parallel", "4",
                "--n-gpu-layers", "999",
                "--seed", "1729",
            ],
            "execution_batching": dict(assembler.MCQ_EXECUTION_BATCHING),
            "candidate_continuations": [" A", " B", " C", " D"],
            "n": 400,
            "correct": cumulative,
            "raw_jsonl_sha256": _sha(mcq_path),
            "llama_log": "/content/lane_b/raw/" + model_id + "/" + mcq_log.name,
            "llama_log_sha256": _sha(mcq_log),
            "stock_multiple_choice_binary_unmodified": False,
            "stock_multiple_choice_scoring_unmodified": True,
            "multiple_choice_memory_patch_exact": True,
            "multiple_choice_patch_sha256": assembler.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
            "per_choice_logprob_vectors_available": False,
            "model_receipt": {
                "status": "PASS",
                "model_id": model_id,
                "b1_prereg_sha256": frozen.prereg_sha256,
                "quant_file_sha256": config["quant_file_sha256"],
            },
            "build_receipt": build_receipt,
        })

        first_native: dict[str, str] = {}
        generation_rows: list[dict[str, object]] = []
        generation_rendered_rows: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        generation_config = frozen.prereg["decoding"]["native_generation"]
        request_template = {
            "n_predict": generation_config["max_new_tokens"],
            "temperature": generation_config["temperature"],
            "top_p": generation_config["top_p"],
            "top_k": generation_config["top_k"],
            "min_p": generation_config["min_p"],
            "repeat_penalty": generation_config["repeat_penalty"],
            "seed": generation_config["seed"],
            "stream": generation_config["stream"],
            "cache_prompt": generation_config["cache_prompt"],
            "stop": generation_config["stop_strings"],
        }
        for ordinal, source in enumerate(frozen.native_rows):
            prompt = ("<think></think> " if qwen_nonthinking else "") + f"native {model_id} {source['item_id']}"
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            first_native.setdefault(source["language"], prompt_sha)
            raw = ""
            raw_sha = hashlib.sha256(raw.encode()).hexdigest()
            generation_rendered_rows.append({
                "schema_version": "lane-b-b34-rendered/1.0",
                "suite": "native_generation",
                "ordinal": ordinal,
                "model_id": model_id,
                "item_id": source["item_id"],
                "language": source["language"],
                "category": source["category"],
                "messages": [
                    {"role": "system", "content": frozen.prereg["prompts"]["system_by_language"][source["language"]]},
                    {"role": "user", "content": frozen.prereg["prompts"]["native_user_template"].format(prompt=source["prompt"])},
                ],
                "rendered_prompt": prompt,
                "prompt_sha256": prompt_sha,
                "hf_prompt_token_count": 1,
                "source_item": source,
            })
            generation_rows.append({
                "schema_version": "lane-b-b34-raw/1.0",
                "run_kind": "native_generation_greedy",
                "evidence_class": "HOST",
                "model_id": model_id,
                "quant_format": config["quant_format"],
                "quant_file_sha256": config["quant_file_sha256"],
                "llama_cpp_commit": assembler.LLAMA_CPP_COMMIT,
                "ordinal": ordinal,
                "item_id": source["item_id"],
                "language": source["language"],
                "category": source["category"],
                "prompt_sha256": prompt_sha,
                "rendered_prompt": prompt,
                "request": {**request_template, "prompt": prompt},
                "raw_completion": raw,
                "raw_completion_sha256": raw_sha,
                "server_response": {"content": raw},
                "source_item": source,
            })
            score_rows.append(self.scorer.score_item(source, raw, model_id))
        generation_path = raw_dir / "native_generation.raw.jsonl"
        _write_jsonl(generation_path, generation_rows)
        server_log = raw_dir / "native_generation.raw.jsonl.server.log"
        server_log.write_text("synthetic server log\n", encoding="utf-8")

        mmlu_rendered_path = render_dir / "global_mmlu.rendered.jsonl"
        generation_rendered_path = render_dir / "native_generation.rendered.jsonl"
        binary_path = render_dir / "global_mmlu.multiple_choice.bin"
        _write_jsonl(mmlu_rendered_path, mmlu_rendered_rows)
        _write_jsonl(generation_rendered_path, generation_rendered_rows)
        binary_path.write_bytes(self.harness.build_multiple_choice_binary(mmlu_rendered_rows))
        render_receipt_path = render_dir / "render_receipt.json"
        _write_json(render_receipt_path, {
            "schema_version": "lane-b-b34-render-receipt/1.0",
            "status": "PASS",
            "harness_source_sha256": assembler.EXPECTED_HARNESS_SHA256,
            "model_id": model_id,
            "b1_prereg_sha256": frozen.prereg_sha256,
            "tokenizer_repo": config["tokenizer_repo"],
            "tokenizer_revision": config["tokenizer_revision"],
            "resolved_tokenizer_revision": config["tokenizer_revision"],
            # transformers 5.14.1 leaves this null for these tokenizers even
            # though huggingface_hub resolves the exact frozen revision.
            "tokenizer_internal_commit_hash": None,
            "chat_template_sha256": "3" * 64,
            "transformers_version": frozen.prereg["harness"]["transformers_version"],
            "huggingface_hub_version": frozen.prereg["harness"]["huggingface_hub_version"],
            "global_mmlu_source_sha256": frozen.mmlu_sha256,
            "native_generation_source_sha256": frozen.native_sha256,
            "global_mmlu_rendered_sha256": _sha(mmlu_rendered_path),
            "native_generation_rendered_sha256": _sha(generation_rendered_path),
            "multiple_choice_binary_sha256": _sha(binary_path),
            "global_mmlu_count": 400,
            "native_generation_count": 200,
            "thinking_disabled_at_template": qwen_nonthinking,
            "first_prompt_sha256": {"global_mmlu": first_mcq, "native_generation": first_native},
        })

        mcq_receipt_path = Path(str(mcq_path) + ".receipt.json")
        mcq_receipt = json.loads(mcq_receipt_path.read_text())
        mcq_receipt["render_receipt_sha256"] = _sha(render_receipt_path)
        _write_json(mcq_receipt_path, mcq_receipt)

        model_receipt = {
            "status": "PASS",
            "model_id": model_id,
            "b1_prereg_sha256": frozen.prereg_sha256,
            "quant_file_sha256": config["quant_file_sha256"],
        }
        _write_json(Path(str(generation_path) + ".receipt.json"), {
            "schema_version": "lane-b-b34-run-receipt/1.0",
            "run_kind": "native_generation_greedy",
            "status": "PASS",
            "harness_source_sha256": assembler.EXPECTED_HARNESS_SHA256,
            "model_id": model_id,
            "b1_prereg_sha256": frozen.prereg_sha256,
            "model_receipt": model_receipt,
            "render_receipt_sha256": _sha(render_receipt_path),
            "build_receipt_sha256": _sha(build_path),
            "build_receipt": build_receipt,
            "server_command": [
                "/content/bin/llama-server",
                "--model", "/content/current-model/" + config["quant_file"],
                "--ctx-size", "4096",
                "--batch-size", "4096",
                "--ubatch-size", "512",
                "--parallel", "1",
                "--n-gpu-layers", "999",
                "--host", "127.0.0.1",
                "--port", "8080",
            ],
            "n_expected": 200,
            "n_present": 200,
            "is_complete": True,
            "smoke_or_partial_limit": None,
            "raw_jsonl_sha256": _sha(generation_path),
            "request_template": request_template,
            "server_log": "/content/lane_b/raw/" + model_id + "/" + server_log.name,
            "server_log_sha256": _sha(server_log),
        })
        _write_json(raw_dir / "native_score.json", {
            "schema_version": "lane-b-native-score/1.0",
            "scorer_source_sha256": assembler.EXPECTED_SCORER_SHA256,
            "manifest": {"sha256": frozen.native_sha256, "items": 200},
            "generations": {"sha256": _sha(generation_path), "rows": 200},
            "dependencies": self.scorer._dependency_versions(True),
            "allow_partial": False,
            "summary": self.scorer._summarise(score_rows),
            "results": score_rows,
        })

    def test_full_assembly_validates_and_writes_summary_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            output = root / "reports"
            for model_id in assembler.MODEL_IDS:
                self._make_model_artifacts(artifacts, model_id)
            reports, written = assembler.assemble(
                LANE_DIR,
                artifacts,
                output,
                completed_at_utc="2026-08-02T00:00:00Z",
                _test_bootstrap_resamples=20,
            )
            self.assertEqual(reports["backbone_verdict"]["verdict_code"], "SUB1B_HOLDS")
            self.assertEqual(len(written), 10)
            self.assertEqual(written[-1].name, "LANE_SUMMARY.md")
            b3_json = json.loads((output / "B3.json").read_text())
            self.assertEqual(b3_json["protocol"]["bootstrap_resamples"], 20)
            self.assertEqual(b3_json["protocol"]["mcq_execution_batching"], assembler.MCQ_EXECUTION_BATCHING)
            self.assertEqual(
                b3_json["protocol"]["llama_cpp_tool_patch"]["patch_sha256"],
                assembler.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
            )
            self.assertIn("sys.version", b3_json["protocol"]["assembly_environment"])
            self.assertIn("never reset", b3_json["protocol"]["bootstrap_implementation"]["seed_scope"])
            native = b3_json["models"]["qwen35_08b"]["languages"]["en"]["native_generation"]
            for component in ("script_S", "nondegenerate_D", "speakability_P"):
                component_rate = native["binary_surface_component_rates"][component]
                self.assertEqual(component_rate["n"], 40)
                self.assertEqual(len(component_rate["wilson_95"]), 2)
            record_names = {row["path"] for row in b3_json["protocol"]["append_only_execution_records"]}
            self.assertIn("b1_execution_clarification_addendum.json", record_names)
            execution_records = b3_json["protocol"]["append_only_execution_records"]
            self.assertEqual([row["path"] for row in execution_records], list(assembler.EXECUTION_RECORD_ORDER))
            for record_name in (
                assembler.RESOURCE_CORRECTION_RECORD["name"],
                assembler.RESOURCE_CORRECTION_FALLBACK_RECORD["name"],
                assembler.TOOL_MEMORY_PATCH_RECORD["name"],
            ):
                post_attempt = next(row for row in execution_records if row["path"] == record_name)
                self.assertFalse(post_attempt["recorded_before_candidate_inference"])
                self.assertTrue(post_attempt["recorded_after_failed_unscored_candidate_attempt"])
                self.assertEqual(post_attempt["candidate_result_rows_observed"], 0)
            build_addendum = next(
                row
                for row in execution_records
                if row["path"] == assembler.BUILD_PROVENANCE_ADDENDUM_RECORD["name"]
            )
            self.assertTrue(build_addendum["recorded_after_candidate_inference"])
            self.assertEqual(build_addendum["candidate_result_rows_observed"], 600)
            self.assertIn("SUB-1B HOLDS", (output / "LANE_SUMMARY.md").read_text())
            with self.assertRaises(assembler.AssemblyError):
                assembler.assemble(
                    LANE_DIR,
                    artifacts,
                    output,
                    completed_at_utc="2026-08-02T00:00:00Z",
                    _test_bootstrap_resamples=1,
                )

    def test_tampered_raw_generation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            for model_id in assembler.MODEL_IDS:
                self._make_model_artifacts(artifacts, model_id)
            path = artifacts / "raw" / "qwen35_08b" / "native_generation.raw.jsonl"
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(assembler.AssemblyError, "generation raw hash differs from receipt"):
                assembler.assemble(
                    LANE_DIR,
                    artifacts,
                    root / "reports",
                    _test_bootstrap_resamples=1,
                )

    def test_pre_correction_mcq_batching_is_rejected_in_command_and_receipt(self) -> None:
        for target, error in (
            ("command", "MCQ command settings differ from frozen protocol"),
            ("execution_batching", "MCQ batching receipt changed"),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifacts = root / "artifacts"
                self._make_model_artifacts(artifacts, "qwen35_08b")
                receipt_path = artifacts / "raw" / "qwen35_08b" / "global_mmlu.raw.jsonl.receipt.json"
                receipt = json.loads(receipt_path.read_text())
                if target == "command":
                    receipt["command"][receipt["command"].index("--batch-size") + 1] = "4096"
                    receipt["command"][receipt["command"].index("--ubatch-size") + 1] = "512"
                else:
                    receipt["execution_batching"] = {"ctx_size": 4096, "batch_size": 4096, "ubatch_size": 512, "parallel": 4}
                _write_json(receipt_path, receipt)
                bundles, _ = assembler.discover_bundles(artifacts, (), allow_incomplete=True)
                with self.assertRaisesRegex(assembler.AssemblyError, error):
                    assembler.validate_bundle(bundles["qwen35_08b"], self.frozen)

    def test_tool_memory_patch_receipts_fail_closed(self) -> None:
        for target, error in (
            ("build_patch", "build receipt MCQ patch hash changed"),
            ("cmake_source", "patched source and CMake source directories disagree"),
            ("impl_path", "implementation-library path is outside the receipted build"),
            ("impl_hash", "build receipt MCQ implementation-library hash changed"),
            ("launcher_hash", "build receipt llama-perplexity launcher hash changed"),
            ("mcq_scoring", "stock MCQ scoring was not preserved"),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifacts = root / "artifacts"
                self._make_model_artifacts(artifacts, "qwen35_08b")
                if target in {"build_patch", "cmake_source", "impl_path", "impl_hash", "launcher_hash"}:
                    path = artifacts / "build_receipt.json"
                    receipt = json.loads(path.read_text())
                    if target == "build_patch":
                        receipt["multiple_choice_patch_sha256"] = "0" * 64
                    elif target == "cmake_source":
                        receipt["cmake_configuration"]["CMAKE_HOME_DIRECTORY"] = "/content/unrelated-llama.cpp"
                    elif target == "impl_path":
                        receipt["multiple_choice_impl_library"]["path"] = "/content/libllama-perplexity-impl.so"
                    elif target == "impl_hash":
                        receipt["multiple_choice_impl_library"]["sha256"] = "0" * 64
                    else:
                        receipt["binaries"]["llama-perplexity"]["sha256"] = "0" * 64
                else:
                    path = artifacts / "raw" / "qwen35_08b" / "global_mmlu.raw.jsonl.receipt.json"
                    receipt = json.loads(path.read_text())
                    receipt["stock_multiple_choice_scoring_unmodified"] = False
                _write_json(path, receipt)
                bundles, _ = assembler.discover_bundles(artifacts, (), allow_incomplete=True)
                with self.assertRaisesRegex(assembler.AssemblyError, error):
                    assembler.validate_bundle(bundles["qwen35_08b"], self.frozen)

    def test_superseded_harness_receipts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            self._make_model_artifacts(artifacts, "qwen35_08b")
            receipt_path = artifacts / "rendered" / "qwen35_08b" / "render_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["harness_source_sha256"] = assembler.PRE_BUILD_PROVENANCE_HARNESS_SHA256
            _write_json(receipt_path, receipt)
            bundles, _ = assembler.discover_bundles(artifacts, (), allow_incomplete=True)
            with self.assertRaisesRegex(assembler.AssemblyError, "render harness source hash mismatch"):
                assembler.validate_bundle(bundles["qwen35_08b"], self.frozen)

    def test_null_internal_tokenizer_hash_requires_exact_resolved_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            self._make_model_artifacts(artifacts, "qwen35_08b")
            receipt_path = artifacts / "rendered" / "qwen35_08b" / "render_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            self.assertIsNone(receipt["tokenizer_internal_commit_hash"])
            bundles, _ = assembler.discover_bundles(
                artifacts,
                (),
                allow_incomplete=True,
            )
            evidence = assembler.validate_bundle(bundles["qwen35_08b"], self.frozen)
            self.assertEqual(evidence.model_id, "qwen35_08b")

            receipt["resolved_tokenizer_revision"] = "0" * 40
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(assembler.AssemblyError, "resolved tokenizer revision mismatch"):
                assembler.validate_bundle(bundles["qwen35_08b"], self.frozen)

    def test_non_null_internal_tokenizer_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            self._make_model_artifacts(artifacts, "qwen35_08b")
            receipt_path = artifacts / "rendered" / "qwen35_08b" / "render_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["tokenizer_internal_commit_hash"] = "0" * 40
            _write_json(receipt_path, receipt)
            bundles, _ = assembler.discover_bundles(artifacts, (), allow_incomplete=True)
            with self.assertRaisesRegex(assembler.AssemblyError, "tokenizer internal commit mismatch"):
                assembler.validate_bundle(bundles["qwen35_08b"], self.frozen)

    def test_fabricated_self_consistent_score_is_rejected_by_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            for model_id in assembler.MODEL_IDS:
                self._make_model_artifacts(artifacts, model_id)
            path = artifacts / "raw" / "qwen35_08b" / "native_score.json"
            report = json.loads(path.read_text())
            row = report["results"][0]
            row["correctness"] = 1
            row["rubric"]["passed"] = True
            fluency = row["surface"]["surface_fluency_proxy"]
            row["generation_item_score"] = 0.70 + 0.20 * fluency + 0.10 * int(row["concision"]["passed"])
            row["strict_generation_pass"] = bool(fluency == 1.0 and row["concision"]["passed"])
            _write_json(path, report)
            with self.assertRaisesRegex(assembler.AssemblyError, "differs from deterministic committed-scorer recomputation"):
                assembler.assemble(
                    LANE_DIR,
                    artifacts,
                    root / "reports",
                    _test_bootstrap_resamples=1,
                )

    def test_malformed_traditional_han_row_raises_assembly_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            for model_id in assembler.MODEL_IDS:
                self._make_model_artifacts(artifacts, model_id)
            path = artifacts / "raw" / "qwen35_08b" / "native_score.json"
            report = json.loads(path.read_text())
            del report["results"][0]["surface"]["traditional_han"]["han_count"]
            _write_json(path, report)
            with self.assertRaises(assembler.AssemblyError):
                assembler.assemble(
                    LANE_DIR,
                    artifacts,
                    root / "reports",
                    _test_bootstrap_resamples=1,
                )

    def test_non_first_raw_prompt_cannot_diverge_from_rendered_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            for model_id in assembler.MODEL_IDS:
                self._make_model_artifacts(artifacts, model_id)
            raw_path = artifacts / "raw" / "qwen35_08b" / "global_mmlu.raw.jsonl"
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            rows[17]["rendered_prompt"] += " tampered-non-first-prompt"
            rows[17]["prompt_sha256"] = hashlib.sha256(rows[17]["rendered_prompt"].encode()).hexdigest()
            _write_jsonl(raw_path, rows)
            receipt_path = Path(str(raw_path) + ".receipt.json")
            receipt = json.loads(receipt_path.read_text())
            receipt["raw_jsonl_sha256"] = _sha(raw_path)
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(assembler.AssemblyError, "raw prompt differs from rendered JSONL"):
                assembler.assemble(
                    LANE_DIR,
                    artifacts,
                    root / "reports",
                    _test_bootstrap_resamples=1,
                )

    def test_copied_llama_log_is_reparsed_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            for model_id in assembler.MODEL_IDS:
                self._make_model_artifacts(artifacts, model_id)
            raw_path = artifacts / "raw" / "qwen35_08b" / "global_mmlu.raw.jsonl"
            log_path = artifacts / "raw" / "qwen35_08b" / "global_mmlu.raw.jsonl.llama.log"
            lines = log_path.read_text().splitlines()
            lines[1] = "2 100.00000000"
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            receipt_path = Path(str(raw_path) + ".receipt.json")
            receipt = json.loads(receipt_path.read_text())
            receipt["llama_log_sha256"] = _sha(log_path)
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(assembler.AssemblyError, "llama log correctness mismatch"):
                assembler.assemble(
                    LANE_DIR,
                    artifacts,
                    root / "reports",
                    _test_bootstrap_resamples=1,
                )

    def test_allow_incomplete_emits_partial_undecidable_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            reports, written = assembler.assemble(
                LANE_DIR,
                artifacts,
                root / "reports",
                allow_incomplete=True,
                completed_at_utc="2026-08-02T00:00:00Z",
                _test_bootstrap_resamples=1,
            )
            self.assertEqual(reports["B3"]["status"], "PARTIAL")
            self.assertEqual(reports["backbone_verdict"]["verdict_code"], "UNDECIDABLE")
            self.assertTrue(reports["backbone_verdict"]["confidence"].startswith("UNKNOWN:"))
            self.assertEqual(written[-1].name, "LANE_SUMMARY.md")
            summary = written[-1].read_text(encoding="utf-8")
            self.assertIn("No definitive selected-model HOST cells are available", summary)
            self.assertNotIn("every selected-model score is `HOST`", summary)

    def test_one_valid_arm_cannot_emit_definitive_programme_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            self._make_model_artifacts(artifacts, "qwen35_08b")
            reports, _ = assembler.assemble(
                LANE_DIR,
                artifacts,
                root / "reports",
                allow_incomplete=True,
                completed_at_utc="2026-08-02T00:00:00Z",
                _test_bootstrap_resamples=1,
            )
            self.assertEqual(reports["B6"]["status"], "PARTIAL")
            verdict = reports["backbone_verdict"]
            self.assertEqual(verdict["verdict_code"], "UNDECIDABLE")
            self.assertIsNone(verdict["selected_model_id"])
            self.assertIn("provisional_complete-cell_rule_result", verdict["decision_trace"])


if __name__ == "__main__":
    unittest.main()
