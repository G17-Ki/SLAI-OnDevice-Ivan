#!/usr/bin/env python3
"""Regression tests for Piper AI Hub result bookkeeping."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_aihub_piper_flow_valid_fixture import (  # noqa: E402
    monotonic_attention,
    parse_phoneme_ids,
)
from run_aihub_piper_accumulated_chain import (  # noqa: E402
    comparison_rows,
    comparisons_pass,
    decoder_inputs,
    stitch_decoder_outputs,
)
from piper_v81.analyze_chain import decoder_stage_names  # noqa: E402
from run_aihub_tts_v81 import compare_outputs  # noqa: E402


class CompareOutputsTest(unittest.TestCase):
    def test_exact_remote_names_win_over_compiled_output_order(self) -> None:
        goldens = OrderedDict(
            (
                ("x_encoded", np.full((1, 2, 3), 1.0, dtype=np.float32)),
                ("m_p", np.full((1, 2, 3), 2.0, dtype=np.float32)),
                ("logs_p", np.full((1, 2, 3), 3.0, dtype=np.float32)),
                ("x_mask", np.ones((1, 1, 3), dtype=np.float32)),
            )
        )
        remote = {name: [value.copy()] for name, value in goldens.items()}
        target_model = SimpleNamespace(
            output_spec={
                "graph": [
                    SimpleNamespace(name="m_p"),
                    SimpleNamespace(name="logs_p"),
                    SimpleNamespace(name="x_encoded"),
                    SimpleNamespace(name="x_mask"),
                ]
            }
        )

        comparisons, saved = compare_outputs(
            remote,
            goldens,
            target_model,
            atol=0.03,
            rtol=0.03,
        )

        self.assertEqual(
            {name: result["remote_name"] for name, result in comparisons.items()},
            {name: name for name in goldens},
        )
        self.assertTrue(all(result["passed"] for result in comparisons.values()))
        for name, golden in goldens.items():
            np.testing.assert_array_equal(saved[f"actual__{name}"], golden)


class PiperFlowFixtureTest(unittest.TestCase):
    def test_parse_phoneme_ids_accepts_boundary_values(self) -> None:
        self.assertEqual(parse_phoneme_ids("0, 1, 255"), [0, 1, 255])

    def test_parse_phoneme_ids_rejects_empty_and_out_of_range(self) -> None:
        for value in ("", "-1", "256"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_phoneme_ids(value)

    def test_monotonic_attention_is_one_hot_on_valid_rows(self) -> None:
        duration = torch.tensor([[[2.0, 1.0, 0.0]]], dtype=torch.float32)
        mask = torch.zeros((1, 1, 5, 3), dtype=torch.float32)
        mask[:, :, :3, :] = 1.0

        attention = monotonic_attention(duration, mask)

        expected = torch.tensor(
            [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(attention, expected)
        torch.testing.assert_close(
            attention[0, 0, :3].sum(dim=-1), torch.ones(3)
        )

    def test_decoder_overlap_rule_produces_two_chunks_for_71_frames(self) -> None:
        z = np.zeros((1, 192, 1536), dtype=np.float32)
        inputs = decoder_inputs(z, y_length=71)
        outputs = [
            np.arange(16384, dtype=np.float32).reshape(1, 1, -1)
            for _ in inputs
        ]

        audio = stitch_decoder_outputs(outputs, y_length=71)

        self.assertEqual([value.shape for value in inputs], [(1, 192, 64)] * 2)
        self.assertEqual(audio.shape, (71 * 256,))
        self.assertEqual(float(audio[40 * 256]), float(12 * 256))


class AccumulatedChainClassificationTest(unittest.TestCase):
    def test_comparisons_require_nonempty_all_pass(self) -> None:
        passing = {
            "encoder": {
                "x_encoded": {"passed": True},
                "x_mask": {"passed": True},
            },
            "audio": {"passed": True},
        }
        warning = {
            **passing,
            "flow": {"passed": False},
        }

        self.assertEqual(len(comparison_rows(passing)), 3)
        self.assertTrue(comparisons_pass(passing))
        self.assertFalse(comparisons_pass(warning))
        self.assertFalse(comparisons_pass({}))

    def test_decoder_stages_are_discovered_dynamically(self) -> None:
        state = {
            "stages": {
                "encoder": {},
                "decoder_3": {},
                "decoder_0": {},
                "sdp": {},
                "decoder_2": {},
                "decoder_1": {},
                "flow": {},
            }
        }
        self.assertEqual(
            decoder_stage_names(state),
            ["decoder_0", "decoder_1", "decoder_2", "decoder_3"],
        )

    def test_decoder_stage_gap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            decoder_stage_names(
                {"stages": {"decoder_0": {}, "decoder_2": {}}}
            )


if __name__ == "__main__":
    unittest.main()
