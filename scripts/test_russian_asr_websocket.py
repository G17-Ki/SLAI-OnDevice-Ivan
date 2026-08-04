#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from russian_asr_websocket.persistent_client import (
    frame_payload,
    send_utterance,
    transcribe_on_connection,
)


class FakeWebSocket:
    def __init__(self, results: list[str]) -> None:
        self.results = list(results)
        self.sent: list[bytes | str] = []

    async def send(self, message: bytes | str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if not self.results:
            raise AssertionError("No fake result remains")
        return self.results.pop(0)


class RussianAsrWebSocketTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.wav = Path(self.tempdir.name) / "fixture.wav"
        with wave.open(str(self.wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<hhhh", -32768, -16384, 0, 32767))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_frame_matches_upstream_protocol(self) -> None:
        payload = frame_payload(self.wav)
        sample_rate, byte_count = struct.unpack("<ii", payload[:8])
        samples = struct.unpack("<ffff", payload[8:])
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(byte_count, 16)
        self.assertAlmostEqual(samples[0], -1.0)
        self.assertAlmostEqual(samples[1], -0.5)
        self.assertAlmostEqual(samples[2], 0.0)
        self.assertAlmostEqual(samples[3], 32767 / 32768.0)

    def test_one_utterance_is_chunked_and_validated(self) -> None:
        websocket = FakeWebSocket([json.dumps({"text": "я тебя люблю"})])
        result = asyncio.run(
            send_utterance(
                websocket,
                self.wav,
                chunk_bytes=8,
                require_nonempty=True,
            )
        )
        self.assertEqual(result["result"]["text"], "я тебя люблю")
        self.assertTrue(all(isinstance(item, bytes) for item in websocket.sent))
        self.assertEqual(b"".join(websocket.sent), frame_payload(self.wav))

    def test_multiple_utterances_share_one_connection(self) -> None:
        websocket = FakeWebSocket(
            [json.dumps({"text": "один"}), json.dumps({"text": "два"})]
        )
        outputs = asyncio.run(
            transcribe_on_connection(
                websocket,
                [self.wav, self.wav],
                require_nonempty=True,
            )
        )
        self.assertEqual([item["result"]["text"] for item in outputs], ["один", "два"])
        self.assertEqual(websocket.sent[-1], "Done")
        self.assertEqual(sum(item == "Done" for item in websocket.sent), 1)

    def test_empty_transcript_fails_closed(self) -> None:
        websocket = FakeWebSocket([json.dumps({"text": ""})])
        with self.assertRaisesRegex(RuntimeError, "empty transcript"):
            asyncio.run(
                send_utterance(websocket, self.wav, require_nonempty=True)
            )


if __name__ == "__main__":
    unittest.main()
