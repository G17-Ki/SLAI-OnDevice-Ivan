#!/usr/bin/env python3
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import b34_harness as h


LANE_DIR = Path(__file__).resolve().parent


def read_u32(blob: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_i32(blob: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<i", blob, offset)[0], offset + 4


def read_string(blob: bytes, offset: int) -> tuple[str, int]:
    size, offset = read_u32(blob, offset)
    return blob[offset : offset + size].decode("utf-8"), offset + size


def read_answers(blob: bytes, offset: int) -> tuple[list[str], list[int], int]:
    count, offset = read_u32(blob, offset)
    answers = []
    for _ in range(count):
        answer, offset = read_string(blob, offset)
        answers.append(answer)
    labels = []
    for _ in range(count):
        label, offset = read_i32(blob, offset)
        labels.append(label)
    return answers, labels, offset


class FakeTokenizer:
    def __init__(self, rendered: str) -> None:
        self.rendered = rendered

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return self.rendered


class B34HarnessTests(unittest.TestCase):
    def test_frozen_protocol_validates(self) -> None:
        receipt = h.validate_protocol(LANE_DIR)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["global_mmlu_rows"], 400)
        self.assertEqual(receipt["native_generation_rows"], 200)
        self.assertEqual(receipt["llama_cpp_commit"], h.LLAMA_CPP_COMMIT)

    def test_multiple_choice_binary_matches_llama_wire_format(self) -> None:
        rows = [
            {"rendered_prompt": "prompt one", "gold_answer": "B"},
            {"rendered_prompt": "提示二", "gold_answer": "D"},
        ]
        blob = h.build_multiple_choice_binary(rows)
        task_count, offset = read_u32(blob, 0)
        self.assertEqual(task_count, 2)
        positions = []
        for _ in range(task_count):
            position, offset = read_u32(blob, offset)
            positions.append(position)
        self.assertEqual(positions[0], offset)

        for task_index, expected in enumerate(rows):
            offset = positions[task_index]
            prompt, offset = read_string(blob, offset)
            answers, labels, offset = read_answers(blob, offset)
            unused_answers, unused_labels, offset = read_answers(blob, offset)
            self.assertEqual(prompt, expected["rendered_prompt"])
            self.assertEqual(answers, list(h.CHOICES))
            self.assertEqual(labels, [int(choice == expected["gold_answer"]) for choice in h.CHOICES])
            self.assertEqual(unused_answers, [])
            self.assertEqual(unused_labels, [])
            expected_end = positions[task_index + 1] if task_index + 1 < len(positions) else len(blob)
            self.assertEqual(offset, expected_end)

    def test_stock_accuracy_trace_reconstructs_item_correctness(self) -> None:
        output = "task\tacc_norm\n1\t100.00000000\n2\t50.00000000\n3\t66.66666667\n"
        trace = h.parse_cumulative_accuracy(output, 3)
        reconstructed = h.reconstruct_item_correctness(trace, 3)
        self.assertEqual([row["correct"] for row in reconstructed], [True, False, True])
        self.assertEqual([row["cumulative_correct"] for row in reconstructed], [1, 1, 2])

    def test_accuracy_trace_rejects_missing_rows(self) -> None:
        with self.assertRaises(h.HarnessError):
            h.parse_cumulative_accuracy("1\t100.00000000\n3\t66.66666667\n", 3)

    def test_thinking_mode_pin_is_required(self) -> None:
        prompt = h._render_prompt(
            FakeTokenizer("<|assistant|><think>\n\n</think>\n"),
            [{"role": "user", "content": "x"}],
            disable_thinking=True,
        )
        self.assertIn("</think>", prompt)
        with self.assertRaises(h.HarnessError):
            h._render_prompt(
                FakeTokenizer("<|assistant|>"),
                [{"role": "user", "content": "x"}],
                disable_thinking=True,
            )

    def test_model_rows_are_complete(self) -> None:
        prereg, _ = h.read_prereg(LANE_DIR)
        for model_id in h.MODEL_IDS:
            row = h.model_config(prereg, model_id)
            self.assertEqual(len(row["quant_file_sha256"]), 64)
            self.assertGreater(row["quant_file_bytes"], 0)

    def test_mcq_execution_settings_are_frozen(self) -> None:
        self.assertEqual(h.MCQ_CONTEXT_TOKENS, 4096)
        self.assertEqual(h.MCQ_BATCH_SIZE, 1024)
        self.assertEqual(h.MCQ_UBATCH_SIZE, 128)
        self.assertEqual(h.MCQ_PARALLEL, 4)

    def test_llama_mcq_memory_patch_is_exact_and_one_file(self) -> None:
        patch_path = LANE_DIR / h.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH
        self.assertEqual(h.sha256_file(patch_path), h.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256)
        patch = patch_path.read_text(encoding="utf-8")
        expected_path = h.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH
        self.assertEqual(patch.count("diff --git "), 1)
        self.assertIn(f"diff --git a/{expected_path} b/{expected_path}", patch)
        self.assertIn("-    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);", patch)
        self.assertIn("+    std::vector<float> batch_logits;", patch)
        self.assertIn("+        batch_logits.resize(i_logits*n_vocab);", patch)
        self.assertEqual(patch.count("batch_logits.resize"), 1)

    def test_validate_build_for_run_requires_exact_patch_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_dir = root / "work"
            bin_dir = root / "bin"
            work_dir.mkdir()
            bin_dir.mkdir()
            llama_src = root / "llama.cpp"
            llama_src.mkdir()
            cmake_cache = root / "CMakeCache.txt"
            cmake_cache.write_text(
                "\n".join(
                    (
                        "BUILD_SHARED_LIBS:BOOL=ON",
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        f"CMAKE_CACHEFILE_DIR:INTERNAL={root.resolve()}",
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={llama_src.resolve()}",
                        "GGML_CUDA:BOOL=ON",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            cmake_receipt = h.cmake_build_configuration_receipt(llama_src, bin_dir)
            wrong_bin_dir = root / "not-bin"
            wrong_bin_dir.mkdir()
            with self.assertRaises(h.HarnessError):
                h.cmake_build_configuration_receipt(llama_src, wrong_bin_dir)
            binary_receipts = {}
            for name in ("llama-perplexity", "llama-server"):
                path = bin_dir / name
                path.write_bytes(name.encode("ascii"))
                binary_receipts[name] = {"sha256": h.sha256_file(path)}
            impl_path = bin_dir / h.LLAMA_CPP_PERPLEXITY_IMPL_LIBRARY_NAME
            impl_path.write_bytes(b"patched implementation library")
            impl_receipt = h.multiple_choice_impl_library_receipt(bin_dir)

            source_receipt = {
                "llama_cpp_commit": h.LLAMA_CPP_COMMIT,
                "llama_cpp_source": str(llama_src),
                "source_tree_clean": False,
                "source_tree_patch_exact": True,
                "source_status": h.LLAMA_CPP_EXPECTED_SOURCE_STATUS,
                "multiple_choice_source_relative_path": h.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_RELATIVE_PATH,
                "multiple_choice_source_original_sha256": h.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_ORIGINAL_SHA256,
                "multiple_choice_source_sha256": h.LLAMA_CPP_MULTIPLE_CHOICE_SOURCE_SHA256,
                "multiple_choice_patch_relative_path": h.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_RELATIVE_PATH,
                "multiple_choice_patch_sha256": h.LLAMA_CPP_MULTIPLE_CHOICE_PATCH_SHA256,
            }
            receipt = {
                **source_receipt,
                "status": "PASS",
                "harness_source_sha256": h.sha256_file(h.SCRIPT_PATH),
                "ggml_cuda": True,
                "cmake_cache_sha256": cmake_receipt["cache_sha256"],
                "cmake_configuration": cmake_receipt,
                "multiple_choice_impl_library": impl_receipt,
                "binaries": binary_receipts,
            }
            h.write_json(work_dir / "build_receipt.json", receipt)
            with mock.patch.object(h, "validate_llama_source_patch", return_value=source_receipt):
                validated = h.validate_build_for_run(LANE_DIR, work_dir, bin_dir)
            self.assertTrue(validated["source_tree_patch_exact"])

            receipt["source_status"] = ""
            h.write_json(work_dir / "build_receipt.json", receipt, replace=True)
            with mock.patch.object(h, "validate_llama_source_patch", return_value=source_receipt):
                with self.assertRaises(h.HarnessError):
                    h.validate_build_for_run(LANE_DIR, work_dir, bin_dir)

            impl_path.write_bytes(b"patched implementation library")
            cmake_cache.write_text(
                cmake_cache.read_text(encoding="utf-8").replace(
                    "CMAKE_BUILD_TYPE:STRING=Release", "CMAKE_BUILD_TYPE:STRING=Debug"
                ),
                encoding="utf-8",
            )
            with mock.patch.object(h, "validate_llama_source_patch", return_value=source_receipt):
                with self.assertRaises(h.HarnessError):
                    h.validate_build_for_run(LANE_DIR, work_dir, bin_dir)

            receipt["source_status"] = h.LLAMA_CPP_EXPECTED_SOURCE_STATUS
            h.write_json(work_dir / "build_receipt.json", receipt, replace=True)
            impl_path.write_bytes(b"stale pristine implementation library")
            with mock.patch.object(h, "validate_llama_source_patch", return_value=source_receipt):
                with self.assertRaises(h.HarnessError):
                    h.validate_build_for_run(LANE_DIR, work_dir, bin_dir)

    def test_colab_guide_preserves_complete_rendered_evidence(self) -> None:
        guide = (LANE_DIR / "B34_COLAB_README.md").read_text(encoding="utf-8")
        self.assertIn("$WORK/rendered/<model_id>/", guide)
        for filename in (
            "global_mmlu.rendered.jsonl",
            "native_generation.rendered.jsonl",
            "global_mmlu.multiple_choice.bin",
            "render_receipt.json",
        ):
            self.assertIn(filename, guide)


if __name__ == "__main__":
    unittest.main(verbosity=2)
