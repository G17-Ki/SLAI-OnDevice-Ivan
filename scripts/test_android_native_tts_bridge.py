#!/usr/bin/env python3
"""Lightweight source invariants for the optional Android system-TTS bridge."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java"
MAIN_ACTIVITY = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/MainActivity.java"
MANIFEST = ROOT / "android-app/app/src/main/AndroidManifest.xml"
UI = ROOT / "android-app/app/src/main/assets/demo-ui/index.html"
SERVICE = ROOT / "android-app/app/src/main/java/ai/liquid/lfmdemo/LfmService.java"
INSTALLER = ROOT / "scripts/install_espeak_phone.sh"


def main() -> None:
    bridge = BRIDGE.read_text(encoding="utf-8")
    main_activity = MAIN_ACTIVITY.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    ui = UI.read_text(encoding="utf-8")
    service = SERVICE.read_text(encoding="utf-8")
    installer = INSTALLER.read_text(encoding="utf-8")

    assert "new Intent(this, LfmService.class)" not in main_activity, (
        "the deployed app must not start the legacy in-process LFM runtime"
    )
    assert "private static final boolean LFM_RUNTIME_ENABLED = false;" in bridge
    assert re.search(
        r"@JavascriptInterface\s+public\s+boolean\s+lfmRuntimeEnabled\s*\(\s*\)",
        bridge,
    ), "the Web UI needs the native no-LFM deployment flag"

    for method in (
        "defaultTtsLocale",
        "hasTtsEngine",
        "hasTtsEngineForLocale",
        "preloadTts",
        "initializeTts",
        "queueTtsChunk",
        "finishTtsResponse",
        "cancelTts",
    ):
        assert re.search(
            rf"@JavascriptInterface\s+public\s+\S+\s+{method}\s*\(", bridge
        ), f"missing JavaScript bridge method: {method}"

    for event in (
        "native_tts_init_requested",
        "native_tts_init_ready",
        "native_tts_init_error",
        "native_tts_preload_requested",
        "native_tts_preload_ready",
        "native_tts_preload_skipped",
        "native_tts_preload_error",
        "native_tts_preload_stale",
        "native_tts_chunk_queued",
        "native_tts_queue_flush",
        "native_tts_queue_cancel",
        "native_tts_utterance_start",
        "native_tts_utterance_done",
        "native_tts_utterance_error",
        "native_tts_first_audio_start_proxy",
        "native_tts_queue_closed",
        "native_tts_playback_terminal",
        "native_tts_stale_command_ignored",
    ):
        assert event in bridge, f"missing native telemetry event: {event}"

    assert "JSONObject.quote(value" in bridge, "native callback payloads must be escaped"
    assert "if (!released)" in bridge, "callbacks must stop after lifecycle release"
    assert "textToSpeech.shutdown()" in bridge, "TTS engine must be released"
    assert "UtteranceProgressListener.onStart" in bridge, "start proxy must name its source"
    assert "long generation" in bridge, "native commands must carry turn generation"
    assert "if (currentMatch && textToSpeech != null)" in bridge, (
        "stale cancellation must not stop the active engine"
    )
    assert "generation < latestTtsGeneration" in bridge, (
        "late initialization must be rejected"
    )
    assert "pendingTtsTurnId = null" in bridge, "cancelled pending init must be invalidated"
    assert 'details.put("native_reason", nativeReason)' in bridge, (
        "neural playback failures must preserve the low-level AudioTrack reason"
    )
    assert "handleErrorOnUi(utteranceId, errorCode, null)" in bridge, (
        "system TTS errors must continue through the shared error path"
    )
    sherpa = (
        ROOT
        / "android-app/app/src/main/java/ai/liquid/lfmdemo/tts/SherpaTtsEngine.kt"
    ).read_text(encoding="utf-8")
    registry = (
        ROOT
        / "android-app/app/src/main/java/ai/liquid/lfmdemo/tts/CandidateRegistry.kt"
    ).read_text(encoding="utf-8")
    assert "WRITE_QUANTUM_MS" in sherpa and "requestSize" in sherpa, (
        "neural PCM writes must be bounded before AudioTrack playback starts"
    )
    assert "if (count < 0)" in sherpa and "if (count == 0)" in sherpa, (
        "AudioTrack zero-progress writes must be retried instead of treated as errors"
    )
    assert "startPlayback(session)" in sherpa, (
        "the writer must start playback as soon as the startup cushion is primed"
    )
    assert '"generation_language" to c.generationLanguage' in sherpa, (
        "TTS metrics must expose the validated per-request generation language"
    )
    assert '"engine_generation" to task.generation' in sherpa
    assert '"generation" to task.generation' not in sherpa, (
        "engine playback epochs must not masquerade as WebView turn generations"
    )
    assert 'details.put("generation", utterance.generation)' in bridge, (
        "engine metrics must be correlated to the owning WebView turn generation"
    )
    assert "val candidate: Candidate" in sherpa and "task.candidate" in sherpa, (
        "shared graphs must retain per-request generation configuration"
    )
    assert "fun shutdownAndAwaitRelease()" in sherpa, (
        "model replacement must wait for native graph release"
    )
    assert "newSingleThreadExecutor" in bridge and "sherpa-tts-model-owner" in bridge, (
        "neural model load and release operations must be serialized"
    )
    assert "previousEngine.shutdownAndAwaitRelease()" in bridge
    assert "handoff.await()" in bridge, (
        "a stale loaded graph must be released before the next queued load starts"
    )
    assert "postToWebView(() -> applyTtsPolicyOnUi(requested))" in bridge, (
        "native TTS policy changes must be owned by the UI state thread"
    )
    preload = re.search(
        r"private void preloadTtsOnUi\([\s\S]*?\n    private void recordAutomaticTtsFallback\(",
        bridge,
    )
    assert preload, "selected-locale preload implementation is missing"
    preload_source = preload.group(0)
    assert "currentTtsTurnId != null || pendingTtsTurnId != null" in preload_source, (
        "preload must refuse to disturb a pending or active speech turn"
    )
    assert "previousEngine.shutdownAndAwaitRelease()" in preload_source
    assert preload_source.index("previousEngine.shutdownAndAwaitRelease()") < preload_source.index(
        "engine.load(candidate)"
    ), "preload must release the old native graph before allocating the replacement"
    assert "sherpaModelExecutor.execute(loadTask)" in preload_source, (
        "preload must use the same serialized model-owner executor as turn loading"
    )
    assert "loadToken == sherpaLoadToken" in preload_source
    assert "preloadToken == ttsPreloadToken" in preload_source
    assert "requestId.equals(pendingTtsPreloadRequestId)" in preload_source
    assert "policy.equals(selectedTtsEngineId)" in preload_source
    assert "onNativeTtsPreloadEvent" in preload_source, (
        "the UI needs an asynchronous preload status callback"
    )
    assert "accepted_tts_turn_started" in bridge, (
        "an accepted real turn must invalidate speculative idle warming"
    )
    assert bridge.index("recordStaleTtsCommand(\"initialize\"") < bridge.index(
        "invalidatePendingTtsPreloadOnUi(\"accepted_tts_turn_started\")"
    ), "a stale WebView turn must not cancel a newer preload"
    release = re.search(r"void release\(\) \{([\s\S]*?)\n    \}", bridge)
    assert release and "clearPendingTtsPreloadOnUi()" in release.group(1), (
        "activity destruction must clear preload ownership"
    )
    assert "ttsPreloadToken += 1" in release.group(1), (
        "activity destruction must invalidate in-flight preload handoffs"
    )
    assert "canonicalLanguageCode.matches(language)" in registry
    assert "supertonicRequiredFiles" in registry
    assert 'extra["lang"] == null || extra["lang"] == language' in registry
    assert "must declare extra.lang" in registry
    assert "Duplicate TTS candidate id" in registry
    assert "android.intent.action.TTS_SERVICE" in manifest, "TTS engine query is required"
    assert '<package android:name="com.reecedunn.espeak" />' in manifest, (
        "the pinned offline eSpeak package must be visible"
    )
    assert "queryIntentServices" in bridge, "the UI needs a synchronous TTS availability gate"
    assert "warmSystemTtsCapabilitiesOnUi" in bridge
    assert "refreshSystemTtsCapabilitiesOnUi" in bridge
    assert "systemTtsOfflineLanguages.contains(language)" in bridge, (
        "package installation alone must not claim locale support"
    )
    assert "candidateSupportsLocale(candidate, localeTag)" in bridge
    assert "candidateSupportsLocale(candidate, resolvedLocaleTag)" in bridge, (
        "a pinned neural voice must not silently generate in a different language"
    )
    assert "canInitializeSystemTtsForLocale(resolvedLocaleTag)" in bridge
    assert "canInitializeSystemTtsForLocale(localeTag)" in bridge, (
        "neural load failure must not route to an unsupported eSpeak locale"
    )
    assert 'systemFallbackAvailable ? "system" : "none"' in bridge
    assert "!systemTtsCapabilitiesKnown" in bridge
    assert "isUsableOfflineVoice(voice)" in bridge
    assert "KEY_FEATURE_NOT_INSTALLED" in bridge, (
        "network-only or not-installed voices must not satisfy an unplugged locale gate"
    )
    assert "autoCandidateForLocale(localeTag) != null || hasSystemTtsEngine()" not in bridge
    assert 'TTS_ENGINE_PACKAGE = "com.reecedunn.espeak"' in bridge, (
        "the offline baseline must pin the audited eSpeak engine"
    )
    assert re.search(
        r"new\s+TextToSpeech\s*\(\s*context,[\s\S]*?TTS_ENGINE_PACKAGE\s*\)", bridge
    ), "TTS initialization must not silently use the system-default engine"
    assert 'details.put("engine_package", TTS_ENGINE_PACKAGE)' in bridge, (
        "TTS telemetry must identify the selected engine"
    )
    assert 'VERSION_CODE="22"' in installer
    assert 'APK_BYTES="10446659"' in installer
    assert 'APK_SHA256="0a7822fec54d7f7ae759ffedc9afeaa77f809587b171fb2101998787821ec790"' in installer
    assert "f-droid.org/repo/com.reecedunn.espeak_22.apk" in installer
    assert "install -r" in installer

    language_selector = re.search(
        r'<select id="language-select"[^>]*>(.*?)</select>', ui, re.DOTALL
    )
    assert language_selector, "global language selector is required"
    # Order matters only for presentation: es and fr are adjacent because they
    # are the two CPU-only Parakeet routes and their labels say so, and it is
    # last because it is the one language with no companion route at all.
    assert re.findall(r'<option value="([^"]+)">', language_selector.group(1)) == [
        "auto",
        "en",
        "zh",
        "ja",
        "ko",
        "ru",
        "es",
        "fr",
        "it",
    ], "language selector must contain exactly the supported public languages"
    assert re.search(r'<option value="es">Spanish \(CPU\)</option>', language_selector.group(1)), \
        "the Spanish option must disclose that its recognizer runs on the CPU"
    assert re.search(r'<option value="fr">French \(CPU\)</option>', language_selector.group(1)), \
        "the French option must disclose that its recognizer runs on the CPU"
    voice_selector = re.search(
        r'<select id="lfm-voice-select"[^>]*>(.*?)</select>', ui, re.DOTALL
    )
    assert voice_selector, "English natural LFM voice selector is required"
    assert re.findall(r'<option value="([^"]+)">', voice_selector.group(1)) == [
        "uk_female",
        "us_female",
        "us_male",
    ], "only physically accepted LFM voices should be exposed"
    assert 'const TEXT_API_URL = "http://127.0.0.1:6020/v1/completions"' in ui
    assert (
        "const LFM_RUNTIME_ENABLED = Boolean(window.NativeAudio?.lfmRuntimeEnabled?.());"
        in ui
    )
    assert "if (LFM_RUNTIME_ENABLED) {\n        readinessChecks.push(fetch(LFM_HEALTH_URL" in ui, (
        "deployed readiness must not contact port 8080 when LFM is disabled"
    )
    assert "modelReady = asrModelReady && textModelReady" in ui
    assert "&& (!LFM_RUNTIME_ENABLED || lfmModelReady);" in ui
    assert "&& LFM_RUNTIME_ENABLED\n        && LFM_NATURAL_TTS_LANGUAGES" in ui, (
        "LFM voice routing must be disabled by the native deployment flag"
    )
    assert "if (!LFM_RUNTIME_ENABLED || languageSelection !== \"en\") throw error;" in ui
    assert "LFM ASR is disabled in this deployment" in ui, (
        "companion ASR failure must fail closed instead of contacting port 8080"
    )
    assert 'const QWEN_TEXT_MODEL = "Qwen3-ASR-0.6B-Text-Q4_K_M"' in ui
    assert 'const DEFAULT_LANGUAGE_SELECTION = "en"' in ui
    assert 'const LANGUAGE_SELECTION_STORAGE_KEY = "lfm.language.selection.v1"' in ui
    assert 'asr_resident_route_invalidated' in ui
    assert 'residentAsr?.route !== expectedAsrRoute' in ui
    assert 'routeWarmState.ready' in ui
    assert "chat_template_kwargs: { enable_thinking: false }" not in ui
    assert "/no_think" not in ui
    assert "createReasoningStreamFilter" in ui
    assert 'voiceBackend: "android_native_tts"' in ui
    assert 'const TTS_POLICY_STORAGE_KEY = "lfm.tts.policy.v2"' in ui
    assert 'const DEFAULT_TTS_POLICY = "auto"' in ui
    assert 'legacy === "system" ? DEFAULT_TTS_POLICY : legacy' in ui
    assert "nativeTtsEngineAvailable(languageRoute.localeTag)" in ui
    assert 'event.type === "ready"' in ui and '"tts_engine_resolved"' in ui
    assert "details.tts_engine_id" in ui
    assert "tts_policy: metrics.selectedPolicy" in ui
    assert "resolved_tts_engine: metrics.resolvedEngine" in ui
    assert ">Offline voice</button>" in ui
    assert ">System voice</button>" not in ui
    assert "modalities: requestMode" not in ui
    assert "extractSafeTtsChunks" in ui
    assert "extractNaturalTtsChunks" in ui
    assert '"--threads", "4"' in service
    assert '"--threads-batch", "4"' in service
    assert '"--poll", "0"' in service
    assert '"--poll-batch", "0"' in service
    assert "signal: turn.controller.signal" in ui, "fetch must be abortable per turn"
    assert "await queueNativeTtsChunk(turn, chunk, chunkId)" in ui, (
        "queue closure must wait for native chunk acknowledgement"
    )
    assert '"tts_text_sse_complete"' in ui, "text completion needs a distinct event"

    print("android native TTS bridge source tests passed")


if __name__ == "__main__":
    main()
