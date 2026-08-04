#!/usr/bin/env python3
"""Deterministic sentence/clause chunking for streamed text-to-speech.

The chunker waits for a stable boundary instead of treating the current end of
an in-progress token stream as a sentence boundary.  This matters for decimal
numbers, URLs, abbreviations, and punctuation that arrives in a later SSE
fragment.  If punctuation is sparse, the bounded policy flushes at a word
boundary so a TTS queue cannot grow without limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_ASCII_SENTENCE = frozenset(".!?")
_CJK_SENTENCE = frozenset("。！？")
_ASCII_CLAUSE = frozenset(",;:")
_CJK_CLAUSE = frozenset("，、；：")
_CLOSERS = frozenset("\"'”’）)]】》」』")
_ABBREVIATIONS = frozenset(
    {
        "approx",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)


@dataclass(frozen=True)
class ChunkPolicy:
    """Controls when an incremental TTS chunk becomes safe to enqueue."""

    min_chars: int = 12
    clause_chars: int = 32
    max_chars: int = 64
    boundary_policy: str = "bounded"

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be positive")
        if self.clause_chars < self.min_chars:
            raise ValueError("clause_chars must be at least min_chars")
        if self.max_chars < self.clause_chars:
            raise ValueError("max_chars must be at least clause_chars")
        if self.boundary_policy not in {"sentence", "clause", "bounded"}:
            raise ValueError("boundary_policy must be sentence, clause, or bounded")


class IncrementalTextChunker:
    """Turn arbitrary streamed fragments into ordered, bounded TTS chunks.

    ``push()`` returns only boundaries that are stable with the characters
    observed so far. ``finish()`` flushes the final tail after the upstream LLM
    has completed. Whitespace is canonicalized because the chunks are intended
    for speech synthesis rather than byte-for-byte text reconstruction.
    """

    def __init__(self, policy: ChunkPolicy | None = None) -> None:
        self.policy = policy or ChunkPolicy()
        self._buffer = ""
        self._closed = False

    @property
    def pending_text(self) -> str:
        return self._buffer

    def push(self, fragment: str) -> list[str]:
        if self._closed:
            raise RuntimeError("cannot push after finish")
        if not isinstance(fragment, str):
            raise TypeError("fragment must be a string")
        if fragment:
            self._buffer = re.sub(r"\s+", " ", self._buffer + fragment).lstrip()
        return self._drain(final=False)

    def finish(self) -> list[str]:
        if self._closed:
            return []
        self._closed = True
        chunks = self._drain(final=True)
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            chunks.append(tail)
        return chunks

    def _drain(self, final: bool) -> list[str]:
        chunks: list[str] = []
        while self._buffer:
            end = self._sentence_boundary(final)
            if end is None and self.policy.boundary_policy != "sentence":
                end = self._clause_boundary(final)
            if end is None and self.policy.boundary_policy == "bounded":
                end = self._hard_boundary()
            if end is None:
                break
            chunk = self._buffer[:end].strip()
            self._buffer = self._buffer[end:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _sentence_boundary(self, final: bool) -> int | None:
        text = self._buffer
        for index, character in enumerate(text):
            if character not in _ASCII_SENTENCE and character not in _CJK_SENTENCE:
                continue
            if character == "." and self._period_is_internal(index):
                continue

            end = index + 1
            while end < len(text) and text[end] in _ASCII_SENTENCE | _CJK_SENTENCE:
                end += 1
            while end < len(text) and text[end] in _CLOSERS:
                end += 1

            if character in _ASCII_SENTENCE:
                if end == len(text) and not final:
                    # Wait for one character of lookahead. For example, "3."
                    # is not safe until the next fragment proves it is not 3.14.
                    continue
                if end < len(text) and not text[end].isspace():
                    continue

            if self._visible_chars(text[:end]) >= self.policy.min_chars:
                return end
        return None

    def _period_is_internal(self, index: int) -> bool:
        text = self._buffer
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isdigit() and following.isdigit():
            return True
        if following == "." or (index > 0 and previous == "."):
            return True

        prefix = text[:index]
        match = re.search(r"([\w.]+)$", prefix, flags=re.UNICODE)
        token = match.group(1).casefold().strip(".") if match else ""
        if len(token) == 1 or token in _ABBREVIATIONS:
            return True
        return False

    def _clause_boundary(self, final: bool) -> int | None:
        text = self._buffer
        for index, character in enumerate(text):
            if character not in _ASCII_CLAUSE and character not in _CJK_CLAUSE:
                continue
            end = index + 1
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            if character in _ASCII_CLAUSE:
                if end == len(text) and not final:
                    continue
                if end < len(text) and not text[end].isspace():
                    continue
            if self._visible_chars(text[:end]) >= self.policy.clause_chars:
                return end
        return None

    def _hard_boundary(self) -> int | None:
        text = self._buffer
        if len(text) < self.policy.max_chars:
            return None
        boundary = text.rfind(" ", self.policy.min_chars, self.policy.max_chars + 1)
        if boundary >= self.policy.min_chars:
            return boundary
        return self.policy.max_chars

    @staticmethod
    def _visible_chars(text: str) -> int:
        return sum(character.isalnum() for character in text)


def chunk_text(
    fragments: list[str],
    policy: ChunkPolicy | None = None,
) -> list[str]:
    """Convenience helper used by deterministic tests and offline adapters."""

    chunker = IncrementalTextChunker(policy)
    chunks: list[str] = []
    for fragment in fragments:
        chunks.extend(chunker.push(fragment))
    chunks.extend(chunker.finish())
    return chunks
