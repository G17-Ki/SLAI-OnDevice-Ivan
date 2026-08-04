#!/usr/bin/env python3
"""Deterministic tests for the 2026-07-26 TTS research harness.

These cover the parts of the harness that decide verdicts: the waveform gates,
the playback-supply simulation, and the ASR scoring normalization. Each test
encodes a defect actually observed during the run, so a regression here would
silently change a recommendation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gates  # noqa: E402
from asr_judge import (  # noqa: E402
    digit_rendering_mismatch,
    dominant_script,
    error_rates,
    normalize,
    strip_trailing_hallucination,
)
from bench import simulate_playback  # noqa: E402
from latency_budget import min_cushion_seconds  # noqa: E402


def tone(seconds: float, rate: int = 22050, freq: float = 180.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    # A slow amplitude sweep keeps consecutive windows from correlating at
    # 0.995, which is what the collapse detector looks for.
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    return (amp * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class GateTest(unittest.TestCase):
    def test_healthy_waveform_passes(self) -> None:
        result = gates.evaluate(tone(2.0), 22050)
        self.assertTrue(result.passed, result.hard_failures)
        self.assertEqual(result.metrics["nonfinite_count"], 0)

    def test_empty_waveform_fails(self) -> None:
        result = gates.evaluate(np.array([], dtype=np.float32), 22050)
        self.assertFalse(result.passed)
        self.assertIn("empty_waveform", result.hard_failures)

    def test_all_non_finite_fails_without_raising(self) -> None:
        # Kokoro-ZH returned entirely non-finite PCM for short Mandarin input.
        samples = np.full(4096, np.nan, dtype=np.float32)
        result = gates.evaluate(samples, 24000)
        self.assertFalse(result.passed)
        self.assertIn("all_samples_non_finite", result.hard_failures)

    def test_partially_non_finite_still_reports_finite_statistics(self) -> None:
        samples = tone(1.0).copy()
        samples[100:110] = np.inf
        result = gates.evaluate(samples, 22050)
        self.assertFalse(result.passed)
        self.assertIn("non_finite_pcm:10", result.hard_failures)
        # A single infinity must not poison the remaining metrics.
        self.assertTrue(np.isfinite(result.metrics["rms"]))
        self.assertGreater(result.metrics["rms"], 0.0)

    def test_near_silence_fails(self) -> None:
        # MeloTTS-ZH produced RMS ~0.003 for single-character Mandarin input.
        result = gates.evaluate(tone(1.0, amp=0.004), 22050)
        self.assertFalse(result.passed)
        self.assertTrue(any(f.startswith("near_silent_rms") for f in result.hard_failures))

    def test_dc_offset_warns_but_does_not_fail(self) -> None:
        # Kokoro carries ~2.5% full-scale DC: a real defect, removable with a
        # DC blocker, so it must warn rather than reject the waveform.
        result = gates.evaluate(tone(2.0) + 0.025, 22050)
        self.assertTrue(result.passed, result.hard_failures)
        self.assertTrue(any(w.startswith("dc_offset_needs_blocker") for w in result.warnings))

    def test_severe_dc_offset_fails(self) -> None:
        result = gates.evaluate(tone(2.0) + 0.2, 22050)
        self.assertFalse(result.passed)
        self.assertTrue(any(f.startswith("severe_dc_offset") for f in result.hard_failures))

    def test_repeated_window_collapse_fails(self) -> None:
        window = np.sin(2 * np.pi * 200 * np.arange(int(0.25 * 22050)) / 22050) * 0.3
        collapsed = np.tile(window, 12).astype(np.float32)
        result = gates.evaluate(collapsed, 22050)
        self.assertFalse(result.passed)
        self.assertTrue(any(f.startswith("repeated_pattern_run") for f in result.hard_failures))

    def test_long_internal_silence_fails(self) -> None:
        speech = tone(1.0)
        silence = np.zeros(int(2.0 * 22050), dtype=np.float32)
        result = gates.evaluate(np.concatenate([speech, silence, speech]), 22050)
        self.assertFalse(result.passed)
        self.assertTrue(any(f.startswith("internal_silence_s") for f in result.hard_failures))

    def test_clipping_fails(self) -> None:
        samples = tone(1.0).copy()
        samples[:2000] = 1.0
        result = gates.evaluate(samples, 22050)
        self.assertFalse(result.passed)
        self.assertTrue(any(f.startswith("clipping_fraction") for f in result.hard_failures))


class PlaybackSimulationTest(unittest.TestCase):
    def test_fast_engine_is_gap_free(self) -> None:
        rate = 22050
        # Three one-second chunks produced far ahead of playback.
        result = simulate_playback([0.1, 0.2, 0.3], [rate] * 3, rate, 1.0)
        self.assertTrue(result["gap_free"])
        self.assertEqual(result["gap_count"], 0)
        self.assertAlmostEqual(result["first_audible_s"], 0.24, places=6)

    def test_slow_engine_starves_the_queue(self) -> None:
        rate = 22050
        # Chunk 1 lands 2.0 s after chunk 0, but chunk 0 only holds 1.0 s.
        result = simulate_playback([0.5, 2.5], [rate, rate], rate, 1.0)
        self.assertFalse(result["gap_free"])
        self.assertEqual(result["gap_count"], 1)
        # playback starts 0.64, drains at 1.64, next chunk at 2.5 -> 0.86 s gap.
        self.assertAlmostEqual(result["max_gap_s"], 0.86, places=6)

    def test_slowdown_multiplies_production_times_only(self) -> None:
        rate = 22050
        fast = simulate_playback([0.5, 1.2], [rate, rate], rate, 1.0)
        slow = simulate_playback([0.5, 1.2], [rate, rate], rate, 2.0)
        self.assertTrue(fast["gap_free"])
        self.assertFalse(slow["gap_free"])

    def test_no_callbacks_is_reported_not_assumed_gap_free(self) -> None:
        result = simulate_playback([], [], 22050, 1.0)
        self.assertFalse(result["simulated"])
        self.assertNotIn("gap_free", result)


class MinCushionTest(unittest.TestCase):
    """The cushion metric decides whether a relaxed budget rescues an engine."""

    def test_engine_that_stays_ahead_needs_no_cushion(self) -> None:
        rate = 22050
        self.assertAlmostEqual(
            min_cushion_seconds([0.1, 0.2, 0.3], [rate] * 3, rate, 1.0), 0.0, places=9
        )

    def test_cushion_covers_the_worst_deficit_not_the_last_one(self) -> None:
        rate = 22050
        # Chunk 1 is 1.5 s late against 1.0 s of queued audio (deficit 0.5 s);
        # chunk 2 arrives immediately after, so the worst deficit is chunk 1's.
        cushion = min_cushion_seconds([0.0, 1.5, 1.6], [rate, rate, rate], rate, 1.0)
        self.assertAlmostEqual(cushion, 0.5, places=6)

    def test_cushion_grows_with_slowdown(self) -> None:
        rate = 22050
        # One-second chunks: at 1.0x the second arrives at 0.8 s, comfortably
        # inside the first second of audio; at 2.0x it arrives at 1.6 s, 0.6 s
        # after the queue would have drained.
        fast = min_cushion_seconds([0.0, 0.8], [rate, rate], rate, 1.0)
        slow = min_cushion_seconds([0.0, 0.8], [rate, rate], rate, 2.0)
        self.assertAlmostEqual(fast, 0.0, places=9)
        self.assertAlmostEqual(slow, 0.6, places=6)

    def test_applying_the_cushion_removes_every_gap(self) -> None:
        rate = 22050
        times, sizes = [0.0, 1.5, 1.6], [rate, rate, rate]
        cushion = min_cushion_seconds(times, sizes, rate, 1.0)
        # Re-simulate with that cushion and confirm the queue never starves.
        cursor = times[0] + cushion
        for produced, size in zip(times, sizes):
            self.assertLessEqual(produced, cursor + 1e-9)
            cursor += size / rate

    def test_no_callbacks_returns_none(self) -> None:
        self.assertIsNone(min_cushion_seconds([], [], 22050, 1.0))


class AsrScoringTest(unittest.TestCase):
    def test_traditional_han_folds_to_simplified(self) -> None:
        self.assertEqual(normalize("會議安排", "zh"), normalize("会议安排", "zh"))

    def test_mandarin_punctuation_is_removed(self) -> None:
        self.assertEqual(normalize("北京是首都。", "zh"), "北京是首都")

    def test_russian_yo_folds_and_case_is_dropped(self) -> None:
        self.assertEqual(normalize("Ещё раз!", "ru"), "еще раз")

    def test_exact_match_scores_zero(self) -> None:
        rates = error_rates("Москва — столица России.", "москва столица россии", "ru")
        self.assertEqual(rates["cer"], 0.0)
        self.assertEqual(rates["wer"], 0.0)

    def test_mandarin_has_no_word_error_rate(self) -> None:
        rates = error_rates("北京是首都", "北京是首都", "zh")
        self.assertEqual(rates["cer"], 0.0)
        self.assertNotIn("wer", rates)

    def test_digit_rendering_mismatch_is_detected(self) -> None:
        # The judge writes "24" where the fixture spells the numeral out; both
        # are correct renderings of the same audio.
        self.assertTrue(digit_rendering_mismatch("около двадцати четырех градусов",
                                                 "около 24 градусов"))
        self.assertFalse(digit_rendering_mismatch("около двадцати четырех градусов",
                                                  "около двадцати четырех градусов"))
        # A reference that already contains digits is not a mismatch.
        self.assertFalse(digit_rendering_mismatch("rooms 24 and 25", "rooms 24 and 25"))

    def test_trailing_subtitle_hallucination_is_stripped(self) -> None:
        # Whisper appended this credit line to well-synthesized Russian
        # paragraphs, inflating CER from ~0.02 to ~0.18 and making higher
        # Supertonic step counts look like a regression.
        real = "фотосинтез это процесс в котором растения превращают энергию света"
        polluted = real + " редактор субтитров н закомолдина корректор а егорова"
        kept, removed = strip_trailing_hallucination(polluted)
        self.assertEqual(kept, real)
        self.assertIn("редактор субтитров", removed)

    def test_marker_inside_real_content_is_not_stripped(self) -> None:
        # A marker word early in the utterance is real speech, not an artifact.
        text = "корректор работает в редакции и проверяет каждый текст очень внимательно каждый день"
        kept, removed = strip_trailing_hallucination(text)
        self.assertEqual(kept, text)
        self.assertIsNone(removed)

    def test_clean_hypothesis_is_untouched(self) -> None:
        text = "москва столица россии"
        kept, removed = strip_trailing_hallucination(text)
        self.assertEqual(kept, text)
        self.assertIsNone(removed)

    def test_error_rates_records_what_was_stripped(self) -> None:
        rates = error_rates(
            "Москва — столица России.",
            "москва столица россии. Редактор субтитров Н. Закомолдина",
            "ru",
        )
        self.assertEqual(rates["cer"], 0.0)
        self.assertIn("stripped_trailing_hallucination", rates)
        self.assertIn("hypothesis_before_hallucination_strip", rates)

    def test_dominant_script_detection(self) -> None:
        self.assertEqual(dominant_script("北京是中国的首都"), "han")
        self.assertEqual(dominant_script("Москва столица"), "cyrillic")
        self.assertEqual(dominant_script("Moskva stolica"), "latin")
        self.assertEqual(dominant_script("123 ..."), "empty")


if __name__ == "__main__":
    unittest.main()
