#!/usr/bin/env python3
"""Dependency-free tests for MeloTTS release-context aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from summarize_melotts_release_contexts import COMPONENTS, summarize


class MeloSummaryTest(unittest.TestCase):
    def write_state(self, root: Path, component: str, latency: float, ops: int) -> Path:
        path = root / f"{component}.json"
        path.write_text(
            json.dumps(
                {
                    "release": {"generated_for_chipset": "sm8850-ad"},
                    "target": {"chipset": "sm8850"},
                    "components": {
                        component: {
                            "status": "profile_passed_strict_npu",
                            "artifact": {"sha256": component},
                            "model_id": f"model-{component}",
                            "profile_job_id": f"job-{component}",
                            "metrics": {"latency_ms": latency, "peak_memory_mib": 1.0},
                            "placement": {
                                "operation_count": ops,
                                "counts": {
                                    "hexagon_npu": ops,
                                    "cpu_fallback": 0,
                                    "unsupported_or_export_failure": 0,
                                },
                                "strict_npu_pass": True,
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_complete_strict_npu_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = {
                "encoder": (23.0, 830),
                "flow": (74.0, 763),
                "decoder": (44.0, 447),
                "bert_wrapper": (3.0, 332),
            }
            states = {
                component: self.write_state(root, component, *values[component])
                for component in COMPONENTS
            }
            result = summarize(states)
            self.assertTrue(result["summary"]["all_components_strict_npu_pass"])
            self.assertEqual(result["summary"]["audio_component_latency_sum_ms"], 141.0)
            self.assertEqual(result["summary"]["all_component_operation_count"], 2372)

    def test_requires_every_component(self) -> None:
        with self.assertRaises(ValueError):
            summarize({})


if __name__ == "__main__":
    unittest.main()
