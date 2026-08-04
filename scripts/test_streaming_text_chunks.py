#!/usr/bin/env python3
"""Deterministic tests for the streamed standalone-TTS text boundary."""

from __future__ import annotations

import unittest

from streaming_text_chunks import ChunkPolicy, IncrementalTextChunker, chunk_text


class IncrementalTextChunkerTest(unittest.TestCase):
    def test_waits_for_ascii_sentence_lookahead(self) -> None:
        chunker = IncrementalTextChunker(ChunkPolicy(min_chars=8, clause_chars=16, max_chars=32))
        self.assertEqual(chunker.push("Paris is in France."), [])
        self.assertEqual(chunker.push(" It is the capital."), ["Paris is in France."])
        self.assertEqual(chunker.finish(), ["It is the capital."])

    def test_keeps_decimal_together_across_fragments(self) -> None:
        chunks = chunk_text(
            ["The measured value is 3.", "14 volts. ", "That is within range."],
            ChunkPolicy(min_chars=8, clause_chars=24, max_chars=48),
        )
        self.assertEqual(chunks, ["The measured value is 3.14 volts.", "That is within range."])

    def test_keeps_common_abbreviation_inside_sentence(self) -> None:
        chunks = chunk_text(
            ["Ask Dr. Smith for the result. ", "Then report it."],
            ChunkPolicy(min_chars=8, clause_chars=24, max_chars=48),
        )
        self.assertEqual(chunks, ["Ask Dr. Smith for the result.", "Then report it."])

    def test_includes_closing_quote(self) -> None:
        chunker = IncrementalTextChunker(ChunkPolicy(min_chars=8, clause_chars=20, max_chars=48))
        self.assertEqual(chunker.push('Try "Built to last."'), [])
        self.assertEqual(chunker.push(" Another option follows."), ['Try "Built to last."'])

    def test_cjk_sentence_does_not_require_space(self) -> None:
        chunker = IncrementalTextChunker(ChunkPolicy(min_chars=4, clause_chars=8, max_chars=20))
        self.assertEqual(
            chunker.push("パリはフランスです。次の質問です"),
            ["パリはフランスです。"],
        )
        self.assertEqual(chunker.finish(), ["次の質問です"])

    def test_clause_policy_flushes_long_clause(self) -> None:
        policy = ChunkPolicy(min_chars=8, clause_chars=20, max_chars=40, boundary_policy="clause")
        chunker = IncrementalTextChunker(policy)
        self.assertEqual(chunker.push("Start with the direct answer,"), [])
        self.assertEqual(chunker.push(" then add one detail"), ["Start with the direct answer,"])
        self.assertEqual(chunker.finish(), ["then add one detail"])

    def test_bounded_policy_flushes_at_word_boundary(self) -> None:
        policy = ChunkPolicy(min_chars=8, clause_chars=16, max_chars=24, boundary_policy="bounded")
        chunker = IncrementalTextChunker(policy)
        emitted = chunker.push("one two three four five six seven")
        self.assertEqual(emitted, ["one two three four five"])
        self.assertEqual(chunker.finish(), ["six seven"])

    def test_sentence_policy_does_not_hard_flush(self) -> None:
        policy = ChunkPolicy(min_chars=8, clause_chars=16, max_chars=24, boundary_policy="sentence")
        chunker = IncrementalTextChunker(policy)
        self.assertEqual(chunker.push("one two three four five six seven"), [])
        self.assertEqual(chunker.finish(), ["one two three four five six seven"])

    def test_multiple_ready_sentences_are_drained_in_order(self) -> None:
        chunks = chunk_text(
            ["First answer. Second answer! Third answer? "],
            ChunkPolicy(min_chars=5, clause_chars=12, max_chars=32),
        )
        self.assertEqual(chunks, ["First answer.", "Second answer!", "Third answer?"])

    def test_result_is_independent_of_sse_fragmentation(self) -> None:
        text = "Paris is in France. The measured value is 3.14 volts, which is valid."
        policy = ChunkPolicy(min_chars=8, clause_chars=24, max_chars=48)
        expected = chunk_text([text], policy)
        for first in range(1, len(text), 7):
            for second in range(first + 1, len(text), 11):
                self.assertEqual(
                    chunk_text([text[:first], text[first:second], text[second:]], policy),
                    expected,
                )

    def test_finish_is_idempotent_and_push_after_finish_fails(self) -> None:
        chunker = IncrementalTextChunker()
        chunker.push("short tail")
        self.assertEqual(chunker.finish(), ["short tail"])
        self.assertEqual(chunker.finish(), [])
        with self.assertRaises(RuntimeError):
            chunker.push("late")

    def test_policy_validation(self) -> None:
        with self.assertRaises(ValueError):
            ChunkPolicy(min_chars=0)
        with self.assertRaises(ValueError):
            ChunkPolicy(min_chars=12, clause_chars=8)
        with self.assertRaises(ValueError):
            ChunkPolicy(min_chars=8, clause_chars=16, max_chars=12)
        with self.assertRaises(ValueError):
            ChunkPolicy(boundary_policy="invalid")


if __name__ == "__main__":
    unittest.main()
