#!/usr/bin/env python3
"""Deterministic source and PCM-ring checks for NativeAudioBridge VAD mode."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java"


def block_starting_at(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated Java block: {marker}")


def source_invariants(source: str) -> None:
    for method, return_type in (
        ("startVadConversation", "boolean"),
        ("resumeVadConversation", "boolean"),
        ("stopVadConversation", "void"),
    ):
        assert re.search(
            rf"@JavascriptInterface\s+public\s+{return_type}\s+{method}\s*\(\s*\)",
            source,
        ), f"missing JavaScript VAD method: {method}"

    assert "private static final int SAMPLE_RATE = 16_000;" in source
    assert "private static final int VAD_FRAME_SAMPLES = 320;" in source
    assert "private static final int VAD_MAX_UTTERANCE_MS = 15_000;" in source
    assert "private static final int VAD_PRE_ROLL_MS = 240;" in source
    assert "private static final int VAD_ENDPOINT_HANGOVER_MS = 420;" in source
    assert "+ VAD_FRAME_SAMPLES;" in source, "ring needs a complete-frame margin"
    assert "AudioRecord.READ_BLOCKING" in source
    assert "filled < VAD_FRAME_SAMPLES" in source, "short reads must be accumulated"
    assert "filled != VAD_FRAME_SAMPLES" in source, "partial frames must not reach VAD"
    assert source.index("ring.append(frameStartSample") < source.index(
        "detector.processFrame("
    ), "endpoint frame must be retained before its sample decision"

    endpoint = block_starting_at(source, "private void deliverVadEndpoint(")
    assert "endpoint.getRecommendedAudioStartSample()" in endpoint
    assert "endpoint.getDecisionSample()" in endpoint
    assert "ring.slice(audioStartSample, decisionSample)" in endpoint
    assert 'putJson(callbackDetails, "wav_base64"' in endpoint
    assert "Base64.NO_WRAP" in endpoint
    assert 'putJson(callbackDetails, "paused", true)' in endpoint
    assert 'putJson(callbackDetails, "requires_resume", true)' in endpoint

    capture = block_starting_at(source, "private void captureVad(")
    assert capture.index("pauseVadAtEndpoint(") < capture.index("deliverVadEndpoint(")
    assert "thread.join" not in capture, "capture thread must never join itself"
    joiner = block_starting_at(source, "private static void joinCaptureThread(")
    assert "thread == Thread.currentThread()" in joiner

    assert "vadSessionActive || captureMode == CaptureMode.VAD" in source
    assert "recording || captureMode != CaptureMode.NONE" in source
    assert "AdaptiveEnergyVad.BargeInPolicy.DISABLED" in source
    assert "detector.setAssistantOutputActive(active)" in source
    assert 'postJavascriptStringCallback("onNativeVadEvent", event.toString())' in source
    assert "JSONObject.quote(value" in source, "callback strings must be JavaScript escaped"
    assert 'stopVadConversationInternal("activity_destroyed", false)' in source

    for event in (
        "native_vad_session_started",
        "native_vad_listening",
        "native_vad_speech_confirmed",
        "native_vad_hangover",
        "native_vad_endpoint",
        "native_vad_burst_rejected",
        "native_vad_resumed",
        "native_vad_stopped",
        "native_vad_error",
    ):
        assert event in source, f"missing VAD telemetry event: {event}"


def run_extracted_ring_test(source: str) -> None:
    converter = block_starting_at(source, "static byte[] pcm16ToLittleEndian(")
    ring = block_starting_at(source, "static final class PcmRingBuffer")
    harness = f"""
package ai.liquid.lfmdemo;

import java.util.Arrays;

final class NativeVadBridgeExtractedTest {{
    {converter}

    {ring}

    public static void main(String[] args) {{
        PcmRingBuffer buffer = new PcmRingBuffer(8);
        buffer.reset(100);
        short[] first = new short[] {{0, 1, 2, 3, 4}};
        buffer.append(100, first, 0, first.length);
        check(Arrays.equals(buffer.slice(101, 104), new short[] {{1, 2, 3}}),
                "simple absolute slice failed");

        short[] second = new short[] {{5, 6, 7, 8, 9, 10}};
        buffer.append(105, second, 0, second.length);
        check(buffer.capacitySamples() == 8, "capacity changed");
        check(buffer.getFirstSampleOffset() == 103, "wrong retained start after wrap");
        check(buffer.getNextSampleOffset() == 111, "wrong next offset after wrap");
        check(Arrays.equals(buffer.slice(104, 110), new short[] {{4, 5, 6, 7, 8, 9}}),
                "wrapped absolute slice failed");

        expectFailure(() -> buffer.append(109, new short[] {{11}}, 0, 1),
                "non-contiguous append was accepted");
        expectFailure(() -> buffer.slice(102, 104),
                "evicted prefix was returned");

        byte[] encoded = pcm16ToLittleEndian(
                new short[] {{0, 1, -1, Short.MAX_VALUE, Short.MIN_VALUE}}
        );
        byte[] expected = new byte[] {{
                0, 0, 1, 0, -1, -1, -1, 127, 0, -128
        }};
        check(Arrays.equals(encoded, expected), "PCM16 little-endian conversion failed");
        System.out.println("native VAD bridge extracted ring tests passed");
    }}

    private static void expectFailure(Runnable action, String message) {{
        try {{
            action.run();
        }} catch (IllegalArgumentException expected) {{
            return;
        }}
        throw new AssertionError(message);
    }}

    private static void check(boolean condition, String message) {{
        if (!condition) {{
            throw new AssertionError(message);
        }}
    }}
}}
"""
    with tempfile.TemporaryDirectory(prefix="lfm-native-vad-") as temporary:
        directory = Path(temporary)
        java_file = directory / "NativeVadBridgeExtractedTest.java"
        java_file.write_text(harness, encoding="utf-8")
        subprocess.run(
            ["javac", "-encoding", "UTF-8", "-d", str(directory), str(java_file)],
            check=True,
        )
        completed = subprocess.run(
            ["java", "-cp", str(directory), "ai.liquid.lfmdemo.NativeVadBridgeExtractedTest"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "extracted ring tests passed" in completed.stdout


def main() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    source_invariants(source)
    run_extracted_ring_test(source)
    print("native VAD bridge source tests passed")


if __name__ == "__main__":
    main()
