#!/usr/bin/env python3
"""Dependency-free orchestration test for the chunked host TTS queue."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace


# The production profiler imports optional numerical/audio packages at module
# load. This test replaces them before import and stubs the actual synthesis,
# so no model or third-party runtime is loaded.
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

import profile_cascade_voice_pipeline as pipeline


class StreamingTtsQueueTest(unittest.TestCase):
    def test_first_chunk_overlaps_stream_and_queue_is_ordered_and_bounded(self) -> None:
        original_transcribe = pipeline.transcribe
        original_events = pipeline.sse_events
        original_synthesize = pipeline.synthesize
        started_chunks: list[str] = []
        started_lock = threading.Lock()
        first_started = threading.Event()
        stream_finished = threading.Event()

        def fake_transcribe(_args, _local_asr):
            return {
                "transcript": "Where is Paris?",
                "first_text_ms": 1.0,
                "complete_ms": 2.0,
                "detected_language": "en",
                "language_probability": 1.0,
            }

        def fake_events(_url, _payload, _timeout):
            yield {"choices": [{"delta": {"content": "Paris is in France."}}]}
            # The next visible token supplies the deliberate ASCII punctuation
            # lookahead and commits the first sentence.
            yield {"choices": [{"delta": {"content": " It"}}]}
            self.assertTrue(first_started.wait(1.0), "TTS did not overlap the open LLM stream")
            yield {"choices": [{"delta": {"content": " is the capital."}}]}
            yield {"choices": [{"delta": {"content": " Short tail"}}]}
            stream_finished.set()

        def fake_synthesize(_tts, _args, text, _capture_audio):
            started = time.perf_counter()
            with started_lock:
                started_chunks.append(text)
                ordinal = len(started_chunks)
            if ordinal == 1:
                first_started.set()
                # With a one-slot queue, submitting the second chunk must wait
                # here instead of creating an unbounded executor backlog.
                time.sleep(0.03)
            ended = time.perf_counter()
            return {
                "started_at": started,
                "first_at": started,
                "ended_at": ended,
                "first_callback_ms": 0.0,
                "complete_ms": (ended - started) * 1000,
                "samples": None,
                "first_chunk": None,
                "sample_rate": 24_000,
                "audio_seconds": 0.1,
            }

        args = SimpleNamespace(
            llm_model="test-model",
            max_tokens=64,
            system_prompt="Answer directly.",
            llm_url="http://test.invalid/v1/chat/completions",
            timeout=2.0,
            flush_chars=48,
            max_queued_tts_chunks=1,
            playback_cushion_ms=140.0,
            sample_dir=None,
            warmups=0,
        )

        pipeline.transcribe = fake_transcribe
        pipeline.sse_events = fake_events
        pipeline.synthesize = fake_synthesize
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = pipeline.run_once(args, object(), executor, None, 0, True)
        finally:
            pipeline.transcribe = original_transcribe
            pipeline.sse_events = original_events
            pipeline.synthesize = original_synthesize

        self.assertTrue(stream_finished.is_set())
        expected = ["Paris is in France.", "It is the capital.", "Short tail"]
        self.assertEqual(started_chunks, expected)
        self.assertEqual([chunk["text"] for chunk in result["tts_chunks"]], expected)
        self.assertEqual(result["first_tts_chunk"], expected[0])
        self.assertEqual(result["tts_chunk_count"], 3)
        self.assertGreater(result["timing_ms"]["tts_queue_backpressure_total"], 10.0)


if __name__ == "__main__":
    unittest.main()
