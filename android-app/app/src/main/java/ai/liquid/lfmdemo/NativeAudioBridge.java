package ai.liquid.lfmdemo;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.SystemClock;
import ai.liquid.lfmdemo.tts.Candidate;
import ai.liquid.lfmdemo.tts.CandidateRegistry;
import ai.liquid.lfmdemo.tts.SherpaTtsEngine;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.speech.tts.Voice;
import android.util.Base64;
import android.util.Log;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

final class NativeAudioBridge {
    private static final String TAG = "LfmNativeAudio";
    private static final boolean LFM_RUNTIME_ENABLED = false;
    private static final String TELEMETRY_FILE = "streaming-telemetry.jsonl";
    private static final String TTS_ENGINE_PACKAGE = "com.reecedunn.espeak";
    private static final int SAMPLE_RATE = 16_000;
    private static final int MAX_SECONDS = 30;
    private static final int VAD_FRAME_SAMPLES = 320;
    private static final int VAD_MAX_UTTERANCE_MS = 15_000;
    private static final int VAD_PRE_ROLL_MS = 240;
    private static final int VAD_ENDPOINT_HANGOVER_MS = 420;
    private static final int VAD_RING_CAPACITY_SAMPLES =
            SAMPLE_RATE
                    * (VAD_MAX_UTTERANCE_MS + VAD_PRE_ROLL_MS + VAD_ENDPOINT_HANGOVER_MS)
                    / 1_000
                    + VAD_FRAME_SAMPLES;

    private enum CaptureMode {
        NONE,
        MANUAL,
        VAD
    }

    private final Context context;
    private final WebView webView;
    private final Object lock = new Object();
    private final Object telemetryLock = new Object();
    private AudioRecord recorder;
    private Thread recordingThread;
    private volatile boolean recording;
    private volatile boolean released;
    private CaptureMode captureMode = CaptureMode.NONE;
    private ByteArrayOutputStream pcm;
    private AdaptiveEnergyVad vad;
    private PcmRingBuffer vadPcmRing;
    private AdaptiveEnergyVad.Event pendingVadEndpoint;
    private String vadSessionId;
    private long nextVadSessionId;
    private boolean vadSessionActive;
    private boolean vadPaused;
    private boolean vadStopRequested;
    // Optional offline neural TTS. The selected value is a routing policy
    // (auto, system, or a pinned candidate); the resolved value is the actual
    // backend for the pending/current turn.
    private static final String TTS_BUNDLE_DIR = "tts-eval";
    private SherpaTtsEngine sherpaTts;
    private volatile String selectedTtsEngineId = "auto";
    private String resolvedTtsEngineId;
    private volatile long sherpaLoadToken;
    // Idle selected-locale preload state. This is deliberately separate from
    // pending/current turn ownership: warming a graph must never advance a Web
    // turn generation or make a synthetic turn visible to queue/cancel logic.
    private long ttsPreloadToken;
    private String pendingTtsPreloadRequestId;
    private String pendingTtsPreloadLocaleTag;
    private String pendingTtsPreloadPolicyId;
    private String pendingTtsPreloadEngineId;
    private final java.util.concurrent.ExecutorService sherpaModelExecutor =
            java.util.concurrent.Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "sherpa-tts-model-owner");
                thread.setDaemon(true);
                return thread;
            });
    private java.util.List<Candidate> sherpaCandidates = java.util.Collections.emptyList();
    private String sherpaCandidateError;
    private final java.util.Map<String, org.json.JSONObject> lastTtsMetrics =
            new ConcurrentHashMap<>();

    private final TtsProgressListener ttsListener = new TtsProgressListener();
    private TextToSpeech textToSpeech;
    private boolean textToSpeechReady;
    private volatile boolean systemTtsCapabilitiesKnown;
    private volatile Set<String> systemTtsOfflineLanguages =
            java.util.Collections.emptySet();
    private String pendingTtsTurnId;
    private String pendingTtsLocaleTag;
    private long pendingTtsGeneration = -1;
    private String pendingTtsPolicyId;
    private String pendingTtsEngineId;
    private String currentTtsTurnId;
    private String currentTtsLocaleTag;
    private long currentTtsGeneration = -1;
    private long latestTtsGeneration = -1;
    private int nextTtsUtterance;
    private boolean firstTtsChunk;
    private final ConcurrentHashMap<String, TtsUtterance> ttsUtterances =
            new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, TtsTurnState> ttsTurns =
            new ConcurrentHashMap<>();

    NativeAudioBridge(Context context, WebView webView) {
        this.context = context.getApplicationContext();
        this.webView = webView;
        // Prime the pinned engine once so the synchronous JavaScript locale
        // gate can answer from real offline Voice records instead of treating
        // package installation as proof that every requested locale works.
        this.webView.post(this::warmSystemTtsCapabilitiesOnUi);
    }

    @JavascriptInterface
    public boolean lfmRuntimeEnabled() {
        return LFM_RUNTIME_ENABLED;
    }

    @JavascriptInterface
    public boolean glassesBridgePermissionGranted() {
        return android.os.Build.VERSION.SDK_INT < 31
                || context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                        == PackageManager.PERMISSION_GRANTED;
    }

    @JavascriptInterface
    public boolean startGlassesBridge() {
        if (android.os.Build.VERSION.SDK_INT >= 31
                && context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                        != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "glasses bridge: BLUETOOTH_CONNECT not granted");
            return false;
        }
        try {
            context.startForegroundService(
                    new Intent(context, ai.liquid.lfmdemo.glasses.GlassesBridgeService.class));
        } catch (Exception error) {
            Log.e(TAG, "glasses bridge: start failed", error);
            return false;
        }
        // The service can exit within seconds (e.g. SPP listen failure), so
        // verify it actually stayed up before reporting success to the UI.
        try {
            Thread.sleep(1500);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        android.app.ActivityManager manager =
                (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        if (manager == null) {
            return true;
        }
        String target = ai.liquid.lfmdemo.glasses.GlassesBridgeService.class.getName();
        for (android.app.ActivityManager.RunningServiceInfo info
                : manager.getRunningServices(Integer.MAX_VALUE)) {
            if (target.equals(info.service.getClassName())) {
                return true;
            }
        }
        Log.w(TAG, "glasses bridge: service did not stay up");
        return false;
    }

    @JavascriptInterface
    public void stopGlassesBridge() {
        context.stopService(
                new Intent(context, ai.liquid.lfmdemo.glasses.GlassesBridgeService.class));
    }

    @JavascriptInterface
    public double monotonicNowMs() {
        return SystemClock.elapsedRealtimeNanos() / 1_000_000.0;
    }

    @JavascriptInterface
    public boolean resetTelemetry() {
        synchronized (telemetryLock) {
            try (FileOutputStream ignored = context.openFileOutput(
                    TELEMETRY_FILE,
                    Context.MODE_PRIVATE
            )) {
                return true;
            } catch (IOException error) {
                Log.e(TAG, "Could not reset telemetry", error);
                return false;
            }
        }
    }

    @JavascriptInterface
    public boolean recordTelemetry(String payload) {
        try {
            JSONObject event = new JSONObject(payload);
            event.put("native_received_monotonic_ms", monotonicNowMs());
            return appendTelemetry(event);
        } catch (Exception error) {
            Log.e(TAG, "Could not record telemetry", error);
            return false;
        }
    }

    @JavascriptInterface
    public String defaultTtsLocale() {
        return Locale.getDefault().toLanguageTag();
    }

    @JavascriptInterface
    public boolean hasTtsEngine() {
        if (hasSystemTtsEngine()) return true;
        loadSherpaCandidatesIfNeeded();
        for (Candidate candidate : sherpaCandidates) {
            if (candidate.hasAllFiles(ttsBundleRoot()).isEmpty()) return true;
        }
        return false;
    }

    @JavascriptInterface
    public boolean hasTtsEngineForLocale(String localeTag) {
        loadSherpaCandidatesIfNeeded();
        if ("system".equals(selectedTtsEngineId)) {
            return hasSystemTtsEngineForLocale(localeTag);
        }
        if ("auto".equals(selectedTtsEngineId)) {
            return autoCandidateForLocale(localeTag) != null
                    || hasSystemTtsEngineForLocale(localeTag);
        }
        Candidate candidate = candidateById(selectedTtsEngineId);
        return candidate != null
                && candidateSupportsLocale(candidate, localeTag)
                && candidate.hasAllFiles(ttsBundleRoot()).isEmpty();
    }

    private boolean hasSystemTtsEngine() {
        Intent serviceIntent = new Intent(TextToSpeech.Engine.INTENT_ACTION_TTS_SERVICE);
        for (ResolveInfo service : context.getPackageManager()
                .queryIntentServices(serviceIntent, PackageManager.MATCH_DEFAULT_ONLY)) {
            if (service.serviceInfo != null
                    && TTS_ENGINE_PACKAGE.equals(service.serviceInfo.packageName)) {
                return true;
            }
        }
        return false;
    }

    private boolean hasSystemTtsEngineForLocale(String localeTag) {
        if (!hasSystemTtsEngine() || !systemTtsCapabilitiesKnown) return false;
        String language = Locale.forLanguageTag(
                normalizedTtsLocaleTag(localeTag)
        ).getLanguage();
        return !language.isEmpty() && systemTtsOfflineLanguages.contains(language);
    }

    private boolean canInitializeSystemTtsForLocale(String localeTag) {
        if (!hasSystemTtsEngine()) return false;
        // A warm-up still in flight is allowed to finish; configureTtsLocaleOnUi
        // performs the authoritative voice check before reporting ready.
        return !systemTtsCapabilitiesKnown || hasSystemTtsEngineForLocale(localeTag);
    }

    private void warmSystemTtsCapabilitiesOnUi() {
        if (released || textToSpeech != null || !hasSystemTtsEngine()) return;
        createSystemTtsOnUi();
    }

    private void createSystemTtsOnUi() {
        if (released || textToSpeech != null) return;
        systemTtsCapabilitiesKnown = false;
        systemTtsOfflineLanguages = java.util.Collections.emptySet();
        textToSpeech = new TextToSpeech(
                context,
                status -> postToWebView(() -> onTtsEngineInitializedOnUi(status)),
                TTS_ENGINE_PACKAGE
        );
    }

    private void refreshSystemTtsCapabilitiesOnUi() {
        java.util.HashSet<String> languages = new java.util.HashSet<>();
        Set<Voice> voices = textToSpeech == null ? null : textToSpeech.getVoices();
        if (voices != null) {
            for (Voice voice : voices) {
                Locale locale = voice.getLocale();
                if (locale == null || !isUsableOfflineVoice(voice)) continue;
                String language = locale.getLanguage();
                if (!language.isEmpty()) languages.add(language);
            }
        }
        systemTtsOfflineLanguages = java.util.Collections.unmodifiableSet(languages);
        systemTtsCapabilitiesKnown = true;
    }

    /** Voices available on this device, as JSON, for the engine picker. */
    @JavascriptInterface
    public String listTtsEngines() {
        JSONObject payload = new JSONObject();
        try {
            loadSherpaCandidatesIfNeeded();
            JSONArray engines = new JSONArray();

            JSONObject automatic = new JSONObject();
            automatic.put("id", "auto");
            automatic.put("label", "Automatic by language");
            automatic.put("language", "any");
            automatic.put("family", "language-aware routing");
            automatic.put("available", hasTtsEngine());
            automatic.put("notes", "English uses Supertonic 3 voice 4; Mandarin uses "
                    + "Matcha Baker; Russian uses Piper Ruslan; other languages use "
                    + "their highest-priority installed offline voice, then eSpeak.");
            engines.put(automatic);

            JSONObject system = new JSONObject();
            system.put("id", "system");
            system.put("label", "Offline eSpeak NG (control)");
            system.put("language", "any");
            system.put("family", "formant synthesis");
            system.put("available", hasSystemTtsEngine());
            system.put("notes", "Offline and instant, but measured at 55% CER in Mandarin "
                    + "and 62% WER in Russian. Retained as the control.");
            engines.put(system);

            for (Candidate candidate : sherpaCandidates) {
                JSONObject item = new JSONObject();
                item.put("id", candidate.getId());
                item.put("label", candidate.getLabel());
                item.put("language", candidate.getLanguage());
                item.put("family", candidate.getEngine());
                item.put("threads", candidate.getNumThreads());
                item.put("steps", candidate.getNumSteps());
                item.put("available", candidate.hasAllFiles(ttsBundleRoot()).isEmpty());
                item.put("notes", candidate.getNotes());
                if (candidate.getKnown() != null) {
                    item.put("known_issue", candidate.getKnown());
                }
                engines.put(item);
            }
            payload.put("engines", engines);
            payload.put("selected", selectedTtsEngineId);
            payload.put("resolved", resolvedTtsEngineId == null
                    ? JSONObject.NULL : resolvedTtsEngineId);
            payload.put("bundle_root", ttsBundleRoot().getAbsolutePath());
            if (sherpaCandidateError != null) {
                payload.put("error", sherpaCandidateError);
            }
        } catch (Exception error) {
            Log.e(TAG, "Could not list TTS engines", error);
        }
        return payload.toString();
    }

    /**
     * Choose the routing policy for the standalone-TTS path. Loading happens
     * on the next turn so switching never blocks the UI.
     */
    @JavascriptInterface
    public boolean setTtsEngine(String engineId) {
        String requested = engineId == null || engineId.trim().isEmpty()
                ? "auto" : engineId.trim();
        loadSherpaCandidatesIfNeeded();
        if (!"auto".equals(requested) && !"system".equals(requested)) {
            Candidate candidate = candidateById(requested);
            if (candidate == null) {
                Log.w(TAG, "Unknown TTS engine requested: " + requested);
                return false;
            }
            java.util.List<String> missing = candidate.hasAllFiles(ttsBundleRoot());
            if (!missing.isEmpty()) {
                Log.w(TAG, "Unavailable TTS engine requested: " + requested
                        + " (missing " + missing.get(0) + ")");
                return false;
            }
        }
        postToWebView(() -> applyTtsPolicyOnUi(requested));
        return true;
    }

    /**
     * Warm the selected offline voice for one explicit locale while no speech
     * turn owns TTS. Completion is asynchronous and reported through
     * {@code window.onNativeTtsPreloadEvent(json)}.
     *
     * This API intentionally takes a caller-generated request id rather than a
     * turn generation. Preload requests are settings work, not synthetic turns.
     */
    @JavascriptInterface
    public boolean preloadTts(String localeTag, String requestId) {
        if (released) return false;
        String safeLocaleTag = cleanIdentifier(localeTag, "auto");
        String safeRequestId = cleanIdentifier(requestId, "tts-preload");
        if ("auto".equalsIgnoreCase(safeLocaleTag)) {
            return false;
        }
        postToWebView(() -> preloadTtsOnUi(safeLocaleTag, safeRequestId));
        return true;
    }

    private void applyTtsPolicyOnUi(String requested) {
        if (released || requested.equals(selectedTtsEngineId)) return;
        invalidatePendingTtsPreloadOnUi("tts_policy_changed");
        String previous = selectedTtsEngineId;
        selectedTtsEngineId = requested;
        resolvedTtsEngineId = null;
        JSONObject details = new JSONObject();
        try {
            details.put("tts_policy", requested);
            details.put("previous_tts_policy", previous);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_engine_selected", "settings", details);
        // Engine changes invalidate pending loads and active playback before
        // releasing the old model. Otherwise a stale 0.5-1.2 GiB load can
        // replace a newer selection or strand a turn with pending chunks.
        switchSherpaEngineOnUi();
    }

    /** Measurements from the most recent spoken chunk, for the UI panel. */
    @JavascriptInterface
    public String lastTtsMetrics(String turnId) {
        JSONObject metrics = lastTtsMetrics.get(turnId);
        return metrics == null ? "{}" : metrics.toString();
    }

    private File ttsBundleRoot() {
        return new File(context.getExternalFilesDir(null), TTS_BUNDLE_DIR);
    }

    private void loadSherpaCandidatesIfNeeded() {
        if (!sherpaCandidates.isEmpty() || sherpaCandidateError != null) {
            return;
        }
        kotlin.Pair<java.util.List<Candidate>, String> loaded =
                CandidateRegistry.INSTANCE.load(ttsBundleRoot());
        sherpaCandidates = loaded.getFirst();
        sherpaCandidateError = loaded.getSecond();
    }

    private Candidate candidateById(String id) {
        if (id == null) return null;
        for (Candidate candidate : sherpaCandidates) {
            if (id.equals(candidate.getId())) return candidate;
        }
        return null;
    }

    private Candidate autoCandidateForLocale(String localeTag) {
        return firstAvailableAutoCandidate(autoCandidatesForLocale(localeTag));
    }

    private boolean candidateSupportsLocale(Candidate candidate, String localeTag) {
        String language = Locale.forLanguageTag(
                normalizedTtsLocaleTag(localeTag)
        ).getLanguage();
        return !language.isEmpty() && language.equals(candidate.getLanguage());
    }

    private java.util.List<Candidate> autoCandidatesForLocale(String localeTag) {
        String language = Locale.forLanguageTag(
                normalizedTtsLocaleTag(localeTag)
        ).getLanguage();
        java.util.ArrayList<Candidate> candidates = new java.util.ArrayList<>();
        for (Candidate candidate : sherpaCandidates) {
            if (!language.equals(candidate.getLanguage())
                    || candidate.getAutoPriority() <= 0) {
                continue;
            }
            candidates.add(candidate);
        }
        // Higher positive priorities win; zero/negative values opt out of auto.
        candidates.sort((left, right) -> {
            int byPriority = Integer.compare(right.getAutoPriority(), left.getAutoPriority());
            return byPriority != 0 ? byPriority : left.getId().compareTo(right.getId());
        });
        return candidates;
    }

    private Candidate firstAvailableAutoCandidate(java.util.List<Candidate> ranked) {
        for (Candidate candidate : ranked) {
            if (candidate.hasAllFiles(ttsBundleRoot()).isEmpty()) return candidate;
        }
        return null;
    }

    private Candidate nextAvailableAutoCandidateForLocale(
            String localeTag,
            Candidate failedCandidate
    ) {
        boolean afterFailed = false;
        for (Candidate candidate : autoCandidatesForLocale(localeTag)) {
            if (!afterFailed) {
                afterFailed = candidate.getId().equals(failedCandidate.getId());
                continue;
            }
            if (candidate.hasAllFiles(ttsBundleRoot()).isEmpty()) return candidate;
        }
        return null;
    }

    private void switchSherpaEngineOnUi() {
        sherpaLoadToken += 1;
        if (currentTtsTurnId != null) {
            cancelTtsOnUi(currentTtsTurnId, currentTtsGeneration, "engine_changed");
        }
        if (pendingTtsTurnId != null) {
            cancelTtsOnUi(pendingTtsTurnId, pendingTtsGeneration, "engine_changed");
        }
        releaseSherpaEngineOnUi();
    }

    private void releaseSherpaEngineOnUi() {
        SherpaTtsEngine engine = sherpaTts;
        sherpaTts = null;
        if (engine != null) shutdownSherpaAsync(engine);
    }

    private void shutdownSherpaAsync(SherpaTtsEngine engine) {
        try {
            sherpaModelExecutor.execute(engine::shutdownAndAwaitRelease);
        } catch (java.util.concurrent.RejectedExecutionException rejected) {
            Thread fallback = new Thread(
                    engine::shutdownAndAwaitRelease,
                    "sherpa-tts-shutdown-fallback"
            );
            fallback.setDaemon(true);
            fallback.start();
        }
    }

    private boolean usingSherpaTts() {
        return resolvedTtsEngineId != null && !"system".equals(resolvedTtsEngineId);
    }

    private String normalizedTtsLocaleTag(String localeTag) {
        if (localeTag == null || localeTag.trim().isEmpty()
                || "auto".equalsIgnoreCase(localeTag.trim())) {
            return Locale.getDefault().toLanguageTag();
        }
        return localeTag.trim();
    }

    private void preloadTtsOnUi(String requestedLocaleTag, String requestId) {
        if (released) return;
        String localeTag = normalizedTtsLocaleTag(requestedLocaleTag);
        String policy = selectedTtsEngineId;
        if (currentTtsTurnId != null || pendingTtsTurnId != null) {
            JSONObject details = ttsPreloadDetails(
                    requestId,
                    requestedLocaleTag,
                    localeTag,
                    policy,
                    null
            );
            putJson(details, "reason", "tts_turn_active");
            reportTtsPreloadStatus(
                    "busy",
                    "native_tts_preload_skipped",
                    requestId,
                    details
            );
            return;
        }

        invalidatePendingTtsPreloadOnUi("superseded_by_new_preload");
        long preloadToken = ++ttsPreloadToken;
        pendingTtsPreloadRequestId = requestId;
        pendingTtsPreloadLocaleTag = localeTag;
        pendingTtsPreloadPolicyId = policy;
        pendingTtsPreloadEngineId = null;

        JSONObject requestedDetails = ttsPreloadDetails(
                requestId,
                requestedLocaleTag,
                localeTag,
                policy,
                null
        );
        recordNativeTelemetry("native_tts_preload_requested", requestedDetails);

        loadSherpaCandidatesIfNeeded();
        Candidate candidate = null;
        if ("auto".equals(policy)) {
            candidate = autoCandidateForLocale(localeTag);
        } else if (!"system".equals(policy)) {
            candidate = candidateById(policy);
            if (candidate == null
                    || !candidateSupportsLocale(candidate, localeTag)
                    || !candidate.hasAllFiles(ttsBundleRoot()).isEmpty()) {
                JSONObject details = ttsPreloadDetails(
                        requestId,
                        requestedLocaleTag,
                        localeTag,
                        policy,
                        candidate
                );
                putJson(details, "reason", "selected_voice_unavailable_for_locale");
                completePendingTtsPreloadOnUi(
                        preloadToken,
                        requestId,
                        "error",
                        "native_tts_preload_error",
                        details
                );
                return;
            }
        }

        if (candidate == null) {
            if (!canInitializeSystemTtsForLocale(localeTag)) {
                JSONObject details = ttsPreloadDetails(
                        requestId,
                        requestedLocaleTag,
                        localeTag,
                        policy,
                        null
                );
                putJson(details, "reason", "no_offline_voice_for_locale");
                completePendingTtsPreloadOnUi(
                        preloadToken,
                        requestId,
                        "error",
                        "native_tts_preload_error",
                        details
                );
                return;
            }
            // The pinned Android engine is already created during bridge startup.
            // There is no large neural graph to allocate for this route, but a
            // prior neural graph must not remain resident after selecting system.
            sherpaLoadToken += 1;
            releaseSherpaEngineOnUi();
            resolvedTtsEngineId = "system";
            warmSystemTtsCapabilitiesOnUi();
            JSONObject details = ttsPreloadDetails(
                    requestId,
                    requestedLocaleTag,
                    localeTag,
                    policy,
                    null
            );
            putJson(details, "tts_engine_id", "system");
            putJson(details, "resolved_engine", "system");
            putJson(details, "reason", "system_voice_has_no_neural_graph_to_preload");
            completePendingTtsPreloadOnUi(
                    preloadToken,
                    requestId,
                    "skipped",
                    "native_tts_preload_skipped",
                    details
            );
            return;
        }

        pendingTtsPreloadEngineId = candidate.getId();
        if (sherpaTts != null && sherpaTts.canReuseModel(candidate)) {
            sherpaTts.selectCandidate(candidate);
            resolvedTtsEngineId = candidate.getId();
            JSONObject details = ttsPreloadDetails(
                    requestId,
                    requestedLocaleTag,
                    localeTag,
                    policy,
                    candidate
            );
            putJson(details, "reused_graph", true);
            putJson(details, "load_elapsed_ms", 0);
            completePendingTtsPreloadOnUi(
                    preloadToken,
                    requestId,
                    "ready",
                    "native_tts_preload_ready",
                    details
            );
            return;
        }

        preloadSherpaCandidateOnUi(
                requestedLocaleTag,
                localeTag,
                requestId,
                preloadToken,
                policy,
                candidate
        );
    }

    private void preloadSherpaCandidateOnUi(
            String requestedLocaleTag,
            String localeTag,
            String requestId,
            long preloadToken,
            String policy,
            Candidate candidate
    ) {
        // As in the turn-owned loader, release the previous graph on the model
        // owner before allocating the replacement. Selected-locale warming must
        // never create a transient dual-model residency spike.
        final SherpaTtsEngine previousEngine = sherpaTts;
        sherpaTts = null;
        final long loadToken = ++sherpaLoadToken;
        final long startedAt = SystemClock.elapsedRealtime();
        final String expectedEngineId = candidate.getId();
        final SherpaTtsEngine engine = new SherpaTtsEngine(
                ttsBundleRoot(),
                new SherpaTtsListener()
        );
        Runnable loadTask = () -> {
            try {
                if (previousEngine != null) previousEngine.shutdownAndAwaitRelease();
                if (released || loadToken != sherpaLoadToken) return;
                engine.load(candidate);
                if (released || loadToken != sherpaLoadToken) {
                    engine.shutdownAndAwaitRelease();
                    return;
                }

                java.util.concurrent.CountDownLatch handoff =
                        new java.util.concurrent.CountDownLatch(1);
                AtomicBoolean handedOff = new AtomicBoolean(false);
                boolean posted = webView.post(() -> {
                    try {
                        boolean stillCurrent = !released
                                && loadToken == sherpaLoadToken
                                && isPendingTtsPreload(
                                        preloadToken,
                                        requestId,
                                        localeTag,
                                        policy,
                                        expectedEngineId
                                )
                                && currentTtsTurnId == null
                                && pendingTtsTurnId == null
                                && policy.equals(selectedTtsEngineId);
                        if (!stillCurrent) return;
                        sherpaTts = engine;
                        resolvedTtsEngineId = expectedEngineId;
                        JSONObject details = ttsPreloadDetails(
                                requestId,
                                requestedLocaleTag,
                                localeTag,
                                policy,
                                candidate
                        );
                        putJson(details, "reused_graph", false);
                        putJson(
                                details,
                                "load_elapsed_ms",
                                SystemClock.elapsedRealtime() - startedAt
                        );
                        completePendingTtsPreloadOnUi(
                                preloadToken,
                                requestId,
                                "ready",
                                "native_tts_preload_ready",
                                details
                        );
                        handedOff.set(true);
                    } finally {
                        handoff.countDown();
                    }
                });
                if (posted) {
                    boolean interrupted = false;
                    while (true) {
                        try {
                            handoff.await();
                            break;
                        } catch (InterruptedException ignored) {
                            interrupted = true;
                        }
                    }
                    if (interrupted) Thread.currentThread().interrupt();
                }
                if (!posted || !handedOff.get()) engine.shutdownAndAwaitRelease();
            } catch (Throwable error) {
                Log.e(TAG, "Offline voice preload failed", error);
                engine.shutdownAndAwaitRelease();
                postSherpaLoadFailure(() -> {
                    if (!isPendingTtsPreload(
                            preloadToken,
                            requestId,
                            localeTag,
                            policy,
                            expectedEngineId
                    )) {
                        return;
                    }
                    resolvedTtsEngineId = null;
                    JSONObject details = ttsPreloadDetails(
                            requestId,
                            requestedLocaleTag,
                            localeTag,
                            policy,
                            candidate
                    );
                    putJson(details, "reason", "model_load_failed");
                    putJson(details, "message", String.valueOf(error.getMessage()));
                    putJson(
                            details,
                            "load_elapsed_ms",
                            SystemClock.elapsedRealtime() - startedAt
                    );
                    completePendingTtsPreloadOnUi(
                            preloadToken,
                            requestId,
                            "error",
                            "native_tts_preload_error",
                            details
                    );
                });
            }
        };
        try {
            sherpaModelExecutor.execute(loadTask);
        } catch (java.util.concurrent.RejectedExecutionException rejected) {
            if (previousEngine != null) shutdownSherpaAsync(previousEngine);
            engine.shutdownAndAwaitRelease();
            JSONObject details = ttsPreloadDetails(
                    requestId,
                    requestedLocaleTag,
                    localeTag,
                    policy,
                    candidate
            );
            putJson(details, "reason", "model_executor_unavailable");
            completePendingTtsPreloadOnUi(
                    preloadToken,
                    requestId,
                    "error",
                    "native_tts_preload_error",
                    details
            );
        }
    }

    private boolean isPendingTtsPreload(
            long preloadToken,
            String requestId,
            String localeTag,
            String policy,
            String engineId
    ) {
        return preloadToken == ttsPreloadToken
                && requestId.equals(pendingTtsPreloadRequestId)
                && localeTag.equals(pendingTtsPreloadLocaleTag)
                && policy.equals(pendingTtsPreloadPolicyId)
                && engineId.equals(pendingTtsPreloadEngineId);
    }

    private void invalidatePendingTtsPreloadOnUi(String reason) {
        if (pendingTtsPreloadRequestId == null) return;
        String requestId = pendingTtsPreloadRequestId;
        JSONObject details = ttsPreloadDetails(
                requestId,
                pendingTtsPreloadLocaleTag,
                pendingTtsPreloadLocaleTag,
                pendingTtsPreloadPolicyId,
                candidateById(pendingTtsPreloadEngineId)
        );
        putJson(details, "reason", reason);
        clearPendingTtsPreloadOnUi();
        ttsPreloadToken += 1;
        sherpaLoadToken += 1;
        reportTtsPreloadStatus(
                "stale",
                "native_tts_preload_stale",
                requestId,
                details
        );
    }

    private void completePendingTtsPreloadOnUi(
            long preloadToken,
            String requestId,
            String status,
            String telemetryEvent,
            JSONObject details
    ) {
        if (preloadToken != ttsPreloadToken
                || !requestId.equals(pendingTtsPreloadRequestId)) {
            return;
        }
        clearPendingTtsPreloadOnUi();
        reportTtsPreloadStatus(status, telemetryEvent, requestId, details);
    }

    private void clearPendingTtsPreloadOnUi() {
        pendingTtsPreloadRequestId = null;
        pendingTtsPreloadLocaleTag = null;
        pendingTtsPreloadPolicyId = null;
        pendingTtsPreloadEngineId = null;
    }

    private JSONObject ttsPreloadDetails(
            String requestId,
            String requestedLocaleTag,
            String localeTag,
            String policy,
            Candidate candidate
    ) {
        JSONObject details = new JSONObject();
        putJson(details, "request_id", requestId);
        putJson(details, "requested_locale", requestedLocaleTag);
        putJson(details, "resolved_locale", localeTag);
        putJson(details, "tts_policy", policy);
        if (candidate != null) {
            putJson(details, "tts_engine_id", candidate.getId());
            putJson(details, "engine_label", candidate.getLabel());
            putJson(details, "engine_family", candidate.getEngine());
            putJson(details, "threads", candidate.getNumThreads());
        }
        return details;
    }

    private void reportTtsPreloadStatus(
            String status,
            String telemetryEvent,
            String requestId,
            JSONObject details
    ) {
        putJson(details, "status", status);
        recordNativeTelemetry(telemetryEvent, details);
        JSONObject event = new JSONObject();
        putJson(event, "schema_version", 1);
        putJson(event, "type", status);
        putJson(event, "request_id", requestId);
        putJson(event, "monotonic_ms", monotonicNowMs());
        putJson(event, "details", details);
        postJavascriptStringCallback("onNativeTtsPreloadEvent", event.toString());
    }

    private void recordAutomaticTtsFallback(
            String turnId,
            long generation,
            Candidate candidate,
            String fallbackEngineId,
            String reason,
            String message
    ) {
        JSONObject details = new JSONObject();
        try {
            details.put("tts_policy", "auto");
            details.put("tts_engine_id", candidate == null
                    ? JSONObject.NULL : candidate.getId());
            details.put("tts_route_fallback", fallbackEngineId);
            details.put("tts_route_fallback_reason", reason);
            details.put("generation", generation);
            if (message != null) details.put("message", message);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_route_fallback", turnId, details);
        emitTtsCallback("route_fallback", turnId, details);
    }

    @JavascriptInterface
    public void initializeTts(String localeTag, String turnId, long generation) {
        String safeTurnId = cleanIdentifier(turnId, "unknown-turn");
        String safeLocaleTag = cleanIdentifier(localeTag, "auto");
        JSONObject details = new JSONObject();
        try {
            details.put("requested_locale", safeLocaleTag);
            details.put("engine_package", TTS_ENGINE_PACKAGE);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        try {
            details.put("generation", generation);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_init_requested", safeTurnId, details);
        postToWebView(() -> initializeTtsOnUi(safeLocaleTag, safeTurnId, generation));
    }

    @JavascriptInterface
    public void queueTtsChunk(String text, String turnId, String chunkId, long generation) {
        String safeText = text == null ? "" : text.trim();
        String safeTurnId = cleanIdentifier(turnId, "unknown-turn");
        String safeChunkId = cleanIdentifier(chunkId, "chunk");
        postToWebView(() -> queueTtsChunkOnUi(safeText, safeTurnId, safeChunkId, generation));
    }

    @JavascriptInterface
    public void finishTtsResponse(String turnId, long generation) {
        String safeTurnId = cleanIdentifier(turnId, "unknown-turn");
        postToWebView(() -> finishTtsResponseOnUi(safeTurnId, generation));
    }

    @JavascriptInterface
    public void cancelTts(String turnId, long generation, String reason) {
        String safeTurnId = cleanIdentifier(turnId, "unknown-turn");
        String safeReason = cleanIdentifier(reason, "user_cancel");
        postToWebView(() -> cancelTtsOnUi(safeTurnId, generation, safeReason));
    }

    @JavascriptInterface
    public boolean startVadConversation() {
        String sessionId;
        synchronized (lock) {
            sessionId = newVadSessionIdLocked();
            if (released) {
                emitVadError(sessionId, "start", "The native audio bridge has been released.");
                return false;
            }
            if (vadSessionActive) {
                emitVadError(
                        vadSessionId,
                        "start",
                        vadPaused
                                ? "A VAD conversation is paused; resume or stop it first."
                                : "A VAD conversation is already listening."
                );
                return false;
            }
            if (recording || captureMode != CaptureMode.NONE) {
                emitVadError(
                        sessionId,
                        "capture_conflict",
                        "Stop the manual microphone recorder before starting VAD mode."
                );
                return false;
            }
            if (!hasMicrophonePermission()) {
                emitVadError(sessionId, "permission", "Android microphone permission is not granted.");
                return false;
            }

            AudioRecord candidate = null;
            try {
                int bufferSize = microphoneBufferSize();
                candidate = createAudioRecorder(bufferSize);
                AdaptiveEnergyVad.Config config = createVadConfig();
                PcmRingBuffer ring = new PcmRingBuffer(VAD_RING_CAPACITY_SAMPLES);
                AdaptiveEnergyVad detector = new AdaptiveEnergyVad(
                        config,
                        event -> onVadEvent(sessionId, event)
                );
                detector.setAssistantOutputActive(false);

                candidate.startRecording();
                recorder = candidate;
                vad = detector;
                vadPcmRing = ring;
                pendingVadEndpoint = null;
                vadSessionId = sessionId;
                vadSessionActive = true;
                vadPaused = false;
                vadStopRequested = false;
                captureMode = CaptureMode.VAD;
                recording = true;

                JSONObject details = vadConfigDetails(config);
                recordVadTelemetry("native_vad_session_started", sessionId, details);
                emitVadCallback("session_started", sessionId, details);
                detector.start();

                AudioRecord sessionRecorder = candidate;
                recordingThread = new Thread(
                        () -> captureVad(sessionId, sessionRecorder, detector, ring),
                        "lfm-vad-microphone"
                );
                recordingThread.start();
                return true;
            } catch (RuntimeException error) {
                Log.e(TAG, "Could not start VAD conversation", error);
                if (candidate != null) {
                    stopAndReleaseRecorder(candidate);
                }
                clearVadSessionLocked();
                emitVadError(sessionId, "start", errorMessage(error, "VAD capture could not start."));
                return false;
            }
        }
    }

    @JavascriptInterface
    public boolean resumeVadConversation() {
        synchronized (lock) {
            String sessionId = vadSessionId == null ? "unassigned" : vadSessionId;
            if (released) {
                emitVadError(sessionId, "resume", "The native audio bridge has been released.");
                return false;
            }
            if (!vadSessionActive || vad == null || vadPcmRing == null) {
                emitVadError(sessionId, "resume", "No VAD conversation is available to resume.");
                return false;
            }
            if (recording || !vadPaused || vad.getState() != AdaptiveEnergyVad.State.ENDPOINT) {
                emitVadError(
                        sessionId,
                        "resume",
                        "VAD resume is valid only after an endpoint has paused capture."
                );
                return false;
            }
            if (captureMode != CaptureMode.VAD) {
                emitVadError(sessionId, "capture_conflict", "Another microphone mode owns capture.");
                return false;
            }
            if (!hasMicrophonePermission()) {
                emitVadError(sessionId, "permission", "Android microphone permission is not granted.");
                return false;
            }

            AudioRecord candidate = null;
            try {
                candidate = createAudioRecorder(microphoneBufferSize());
                candidate.startRecording();
                long nextSample = vad.snapshotMetrics().getSamplesProcessed();
                vadPcmRing.reset(nextSample);
                pendingVadEndpoint = null;
                vadStopRequested = false;
                vadPaused = false;
                recording = true;
                recorder = candidate;

                JSONObject details = new JSONObject();
                putJson(details, "next_sample_offset", nextSample);
                putJson(details, "frame_samples", VAD_FRAME_SAMPLES);
                recordVadTelemetry("native_vad_resumed", sessionId, details);
                emitVadCallback("resumed", sessionId, details);
                vad.setAssistantOutputActive(false);
                vad.rearm();

                AudioRecord sessionRecorder = candidate;
                AdaptiveEnergyVad sessionVad = vad;
                PcmRingBuffer sessionRing = vadPcmRing;
                recordingThread = new Thread(
                        () -> captureVad(sessionId, sessionRecorder, sessionVad, sessionRing),
                        "lfm-vad-microphone"
                );
                recordingThread.start();
                return true;
            } catch (RuntimeException error) {
                Log.e(TAG, "Could not resume VAD conversation", error);
                if (candidate != null) {
                    stopAndReleaseRecorder(candidate);
                }
                AdaptiveEnergyVad failedDetector = vad;
                clearVadSessionLocked();
                if (failedDetector != null
                        && failedDetector.getState() != AdaptiveEnergyVad.State.IDLE) {
                    failedDetector.cancel();
                }
                emitVadError(sessionId, "resume", errorMessage(error, "VAD capture could not resume."));
                return false;
            }
        }
    }

    @JavascriptInterface
    public void stopVadConversation() {
        stopVadConversationInternal("user_stop", true);
    }

    @JavascriptInterface
    public void setVadAssistantOutputActive(boolean active) {
        AdaptiveEnergyVad detector;
        String sessionId;
        synchronized (lock) {
            if (!vadSessionActive || vad == null) {
                return;
            }
            detector = vad;
            sessionId = vadSessionId;
        }
        detector.setAssistantOutputActive(active);
        JSONObject details = new JSONObject();
        putJson(details, "active", active);
        putJson(details, "barge_in_policy", "disabled");
        recordVadTelemetry("native_vad_output_gate", sessionId, details);
        emitVadCallback("output_gate", sessionId, details);
    }

    @JavascriptInterface
    public boolean startRecording() {
        synchronized (lock) {
            if (vadSessionActive || captureMode == CaptureMode.VAD) {
                sendError("Stop VAD conversation mode before using the manual recorder.");
                return false;
            }
            if (recording && captureMode == CaptureMode.MANUAL) {
                return true;
            }
            if (recording || captureMode != CaptureMode.NONE) {
                sendError("Another microphone capture mode is active.");
                return false;
            }
            if (!hasMicrophonePermission()) {
                sendError("Android microphone permission is not granted.");
                return false;
            }
            int bufferSize = microphoneBufferSize();
            try {
                recorder = createAudioRecorder(bufferSize);
                pcm = new ByteArrayOutputStream(SAMPLE_RATE * 2 * 8);
                captureMode = CaptureMode.MANUAL;
                recording = true;
                recorder.startRecording();
                recordNativeTelemetry("native_recording_started", null);
                recordingThread = new Thread(() -> captureManual(bufferSize), "lfm-microphone");
                recordingThread.start();
                return true;
            } catch (RuntimeException error) {
                Log.e(TAG, "Could not start recording", error);
                cleanupRecorder();
                sendError(error.getMessage());
                return false;
            }
        }
    }

    @JavascriptInterface
    public void stopRecording() {
        recordNativeTelemetry("native_recording_stop_requested", null);
        AudioRecord active;
        synchronized (lock) {
            if (vadSessionActive || captureMode == CaptureMode.VAD) {
                sendError("Use stopVadConversation() to stop VAD conversation mode.");
                return;
            }
            if (!recording) {
                return;
            }
            if (captureMode != CaptureMode.MANUAL) {
                sendError("The manual recorder does not own microphone capture.");
                return;
            }
            recording = false;
            active = recorder;
        }
        try {
            active.stop();
        } catch (IllegalStateException ignored) {
        }
        Thread thread = recordingThread;
        joinCaptureThread(thread);
        byte[] captured;
        synchronized (lock) {
            captured = pcm == null ? new byte[0] : pcm.toByteArray();
            cleanupRecorder();
        }
        if (captured.length == 0) {
            sendError("No microphone samples were captured.");
            return;
        }
        byte[] wav = makeWav(captured);
        JSONObject details = new JSONObject();
        try {
            details.put("wav_bytes", wav.length);
            details.put("pcm_bytes", captured.length);
            details.put("sample_rate", SAMPLE_RATE);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_wav_ready", details);
        String encoded = Base64.encodeToString(wav, Base64.NO_WRAP);
        postJavascriptStringCallback("onNativeRecording", encoded);
    }

    void release() {
        sherpaLoadToken += 1;
        ttsPreloadToken += 1;
        clearPendingTtsPreloadOnUi();
        if (currentTtsTurnId != null) {
            cancelTtsOnUi(currentTtsTurnId, currentTtsGeneration, "activity_destroyed");
        }
        if (pendingTtsTurnId != null) {
            cancelTtsOnUi(pendingTtsTurnId, pendingTtsGeneration, "activity_destroyed");
        }
        released = true;
        stopVadConversationInternal("activity_destroyed", false);

        AudioRecord activeRecorder;
        synchronized (lock) {
            recording = false;
            activeRecorder = recorder;
        }
        if (activeRecorder != null) {
            try {
                activeRecorder.stop();
            } catch (IllegalStateException ignored) {
            }
        }
        Thread thread = recordingThread;
        joinCaptureThread(thread);
        synchronized (lock) {
            cleanupRecorder();
        }

        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
            textToSpeech = null;
        }
        textToSpeechReady = false;
        releaseSherpaEngineOnUi();
        sherpaModelExecutor.shutdown();
        ttsUtterances.clear();
        ttsTurns.clear();
        currentTtsTurnId = null;
        currentTtsLocaleTag = null;
        currentTtsGeneration = -1;
        pendingTtsTurnId = null;
        pendingTtsLocaleTag = null;
        pendingTtsGeneration = -1;
        pendingTtsPolicyId = null;
        pendingTtsEngineId = null;
        resolvedTtsEngineId = null;
    }

    private void captureManual(int bufferSize) {
        byte[] buffer = new byte[bufferSize];
        int maximum = SAMPLE_RATE * 2 * MAX_SECONDS;
        while (recording && pcm.size() < maximum) {
            int read = recorder.read(buffer, 0, buffer.length);
            if (read > 0) {
                pcm.write(buffer, 0, read);
            } else if (read < 0) {
                Log.e(TAG, "AudioRecord read error " + read);
                break;
            }
        }
        if (recording) {
            postJavascriptCallback("window.stopNativeRecordingFromLimit()");
        }
    }

    private void captureVad(
            String sessionId,
            AudioRecord sessionRecorder,
            AdaptiveEnergyVad detector,
            PcmRingBuffer ring
    ) {
        short[] frame = new short[VAD_FRAME_SAMPLES];
        String failure = null;
        boolean endpointReached = false;
        AdaptiveEnergyVad.Event completedEndpoint = null;
        try {
            while (isCurrentVadCapture(sessionId, sessionRecorder, detector, ring)) {
                int filled = 0;
                while (filled < VAD_FRAME_SAMPLES
                        && isCurrentVadCapture(sessionId, sessionRecorder, detector, ring)) {
                    int read = sessionRecorder.read(
                            frame,
                            filled,
                            VAD_FRAME_SAMPLES - filled,
                            AudioRecord.READ_BLOCKING
                    );
                    if (read > 0) {
                        filled += read;
                    } else if (read < 0) {
                        if (isCurrentVadCapture(sessionId, sessionRecorder, detector, ring)) {
                            failure = "AudioRecord read error " + read;
                        }
                        break;
                    }
                }
                if (filled != VAD_FRAME_SAMPLES) {
                    break;
                }
                if (!isCurrentVadCapture(sessionId, sessionRecorder, detector, ring)) {
                    break;
                }

                long frameStartSample = detector.snapshotMetrics().getSamplesProcessed();
                ring.append(frameStartSample, frame, 0, VAD_FRAME_SAMPLES);
                AdaptiveEnergyVad.FrameResult result = detector.processFrame(
                        frame,
                        0,
                        VAD_FRAME_SAMPLES
                );
                if (result.getState() == AdaptiveEnergyVad.State.ENDPOINT) {
                    AdaptiveEnergyVad.Event endpoint = takePendingVadEndpoint(sessionId);
                    if (endpoint == null) {
                        throw new IllegalStateException("VAD reached ENDPOINT without an endpoint event");
                    }
                    if (!pauseVadAtEndpoint(sessionId, sessionRecorder, detector, ring)) {
                        break;
                    }
                    endpointReached = true;
                    try {
                        sessionRecorder.stop();
                    } catch (IllegalStateException ignored) {
                    }
                    completedEndpoint = endpoint;
                    break;
                }
            }
        } catch (RuntimeException error) {
            Log.e(TAG, "VAD capture failed", error);
            failure = errorMessage(error, "VAD capture failed.");
        } finally {
            boolean ownsRecorder;
            boolean currentSession;
            boolean stopRequested;
            synchronized (lock) {
                ownsRecorder = recorder == sessionRecorder;
                if (ownsRecorder) {
                    recorder = null;
                }
                if (recordingThread == Thread.currentThread()) {
                    recordingThread = null;
                }
                currentSession = vadSessionActive
                        && sessionId.equals(vadSessionId)
                        && detector == vad
                        && ring == vadPcmRing;
                stopRequested = vadStopRequested;
                if (currentSession) {
                    recording = false;
                    if (endpointReached) {
                        vadPaused = true;
                    }
                }
            }
            if (ownsRecorder) {
                stopAndReleaseRecorder(sessionRecorder);
            }

            if (completedEndpoint != null && currentSession && !stopRequested) {
                try {
                    deliverVadEndpoint(sessionId, completedEndpoint, ring);
                } catch (RuntimeException error) {
                    Log.e(TAG, "Could not publish VAD endpoint", error);
                    failure = errorMessage(error, "VAD endpoint could not be published.");
                }
            }

            if (failure != null && currentSession && !stopRequested) {
                if (!endpointReached) {
                    terminateVadSessionAfterCaptureError(sessionId, detector, ring);
                }
                emitVadError(sessionId, "capture", failure);
            }
        }
    }

    private boolean isCurrentVadCapture(
            String sessionId,
            AudioRecord sessionRecorder,
            AdaptiveEnergyVad detector,
            PcmRingBuffer ring
    ) {
        synchronized (lock) {
            return !released
                    && vadSessionActive
                    && !vadPaused
                    && !vadStopRequested
                    && recording
                    && captureMode == CaptureMode.VAD
                    && sessionId.equals(vadSessionId)
                    && recorder == sessionRecorder
                    && vad == detector
                    && vadPcmRing == ring;
        }
    }

    private boolean pauseVadAtEndpoint(
            String sessionId,
            AudioRecord sessionRecorder,
            AdaptiveEnergyVad detector,
            PcmRingBuffer ring
    ) {
        synchronized (lock) {
            if (vadSessionActive
                    && !vadStopRequested
                    && sessionId.equals(vadSessionId)
                    && recorder == sessionRecorder
                    && vad == detector
                    && vadPcmRing == ring) {
                recording = false;
                vadPaused = true;
                pendingVadEndpoint = null;
                return true;
            }
            return false;
        }
    }

    private AdaptiveEnergyVad.Event takePendingVadEndpoint(String sessionId) {
        synchronized (lock) {
            if (!vadSessionActive || !sessionId.equals(vadSessionId)) {
                return null;
            }
            AdaptiveEnergyVad.Event endpoint = pendingVadEndpoint;
            pendingVadEndpoint = null;
            return endpoint;
        }
    }

    private void deliverVadEndpoint(
            String sessionId,
            AdaptiveEnergyVad.Event endpoint,
            PcmRingBuffer ring
    ) {
        long audioStartSample = endpoint.getRecommendedAudioStartSample();
        long decisionSample = endpoint.getDecisionSample();
        short[] selectedSamples = ring.slice(audioStartSample, decisionSample);
        byte[] selectedPcm = pcm16ToLittleEndian(selectedSamples);
        byte[] wav = makeWav(selectedPcm);

        JSONObject telemetryDetails = vadEventDetails(endpoint);
        putJson(telemetryDetails, "sample_rate", SAMPLE_RATE);
        putJson(telemetryDetails, "frame_samples", VAD_FRAME_SAMPLES);
        putJson(telemetryDetails, "pcm_bytes", selectedPcm.length);
        putJson(telemetryDetails, "wav_bytes", wav.length);
        putJson(
                telemetryDetails,
                "audio_duration_ms",
                selectedSamples.length * 1_000.0 / SAMPLE_RATE
        );
        putJson(telemetryDetails, "paused", true);
        putJson(telemetryDetails, "requires_resume", true);
        putJson(telemetryDetails, "ring_first_sample", ring.getFirstSampleOffset());
        putJson(telemetryDetails, "ring_next_sample", ring.getNextSampleOffset());
        recordVadTelemetry("native_vad_endpoint", sessionId, telemetryDetails);

        JSONObject callbackDetails = vadEventDetails(endpoint);
        putJson(callbackDetails, "sample_rate", SAMPLE_RATE);
        putJson(callbackDetails, "frame_samples", VAD_FRAME_SAMPLES);
        putJson(callbackDetails, "pcm_bytes", selectedPcm.length);
        putJson(callbackDetails, "wav_bytes", wav.length);
        putJson(
                callbackDetails,
                "audio_duration_ms",
                selectedSamples.length * 1_000.0 / SAMPLE_RATE
        );
        putJson(callbackDetails, "paused", true);
        putJson(callbackDetails, "requires_resume", true);
        putJson(callbackDetails, "wav_base64", Base64.encodeToString(wav, Base64.NO_WRAP));
        emitVadCallback("endpoint", sessionId, callbackDetails);
    }

    private void terminateVadSessionAfterCaptureError(
            String sessionId,
            AdaptiveEnergyVad detector,
            PcmRingBuffer ring
    ) {
        synchronized (lock) {
            if (!vadSessionActive
                    || !sessionId.equals(vadSessionId)
                    || detector != vad
                    || ring != vadPcmRing) {
                return;
            }
            vadSessionActive = false;
            vadPaused = false;
            vadStopRequested = false;
            recording = false;
            captureMode = CaptureMode.NONE;
            vad = null;
            vadPcmRing = null;
            pendingVadEndpoint = null;
            vadSessionId = null;
        }
        if (detector.getState() != AdaptiveEnergyVad.State.IDLE) {
            detector.cancel();
        }
    }

    private void onVadEvent(String sessionId, AdaptiveEnergyVad.Event event) {
        synchronized (lock) {
            if (!vadSessionActive || !sessionId.equals(vadSessionId)) {
                return;
            }
            if (event.getType() == AdaptiveEnergyVad.EventType.ENDPOINT) {
                pendingVadEndpoint = event;
                return;
            }
        }

        JSONObject details = vadEventDetails(event);
        if (event.getType() == AdaptiveEnergyVad.EventType.SPEECH_CONFIRMED) {
            recordVadTelemetry("native_vad_speech_confirmed", sessionId, details);
            emitVadCallback("speech_confirmed", sessionId, details);
        } else if (event.getType() == AdaptiveEnergyVad.EventType.SHORT_BURST_REJECTED) {
            recordVadTelemetry("native_vad_burst_rejected", sessionId, details);
            emitVadCallback("burst_rejected", sessionId, details);
        } else if (event.getType() == AdaptiveEnergyVad.EventType.STATE_CHANGED) {
            if (event.getState() == AdaptiveEnergyVad.State.LISTENING) {
                recordVadTelemetry("native_vad_listening", sessionId, details);
                emitVadCallback("listening", sessionId, details);
            } else if (event.getState() == AdaptiveEnergyVad.State.HANGOVER) {
                recordVadTelemetry("native_vad_hangover", sessionId, details);
                emitVadCallback("hangover", sessionId, details);
            }
        }
    }

    private void stopVadConversationInternal(String reason, boolean notifyJavascript) {
        AudioRecord activeRecorder;
        Thread activeThread;
        AdaptiveEnergyVad detector;
        String sessionId;
        synchronized (lock) {
            if (!vadSessionActive) {
                return;
            }
            sessionId = vadSessionId;
            detector = vad;
            vadStopRequested = true;
            recording = false;
            activeRecorder = recorder;
            activeThread = recordingThread;
        }

        if (activeRecorder != null) {
            try {
                activeRecorder.stop();
            } catch (IllegalStateException ignored) {
            }
        }
        joinCaptureThread(activeThread);

        AudioRecord recorderToRelease;
        synchronized (lock) {
            recorderToRelease = recorder;
            recorder = null;
            recordingThread = null;
            vadSessionActive = false;
            vadPaused = false;
            vadStopRequested = false;
            recording = false;
            captureMode = CaptureMode.NONE;
            vad = null;
            vadPcmRing = null;
            pendingVadEndpoint = null;
            vadSessionId = null;
        }
        if (recorderToRelease != null) {
            stopAndReleaseRecorder(recorderToRelease);
        }
        if (detector != null && detector.getState() != AdaptiveEnergyVad.State.IDLE) {
            detector.cancel();
        }

        JSONObject details = detector == null
                ? new JSONObject() : vadMetricsDetails(detector.snapshotMetrics());
        putJson(details, "reason", reason);
        recordVadTelemetry("native_vad_stopped", sessionId, details);
        if (notifyJavascript) {
            emitVadCallback("stopped", sessionId, details);
        }
    }

    private void cleanupRecorder() {
        recording = false;
        if (recorder != null) {
            recorder.release();
            recorder = null;
        }
        recordingThread = null;
        if (captureMode == CaptureMode.MANUAL) {
            captureMode = CaptureMode.NONE;
        }
    }

    private void sendError(String message) {
        String safe = message == null ? "Unknown microphone error" : message;
        postJavascriptStringCallback("onNativeRecordingError", safe);
    }

    private boolean hasMicrophonePermission() {
        return context.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    private static int microphoneBufferSize() {
        int minimum = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        if (minimum <= 0) {
            throw new IllegalStateException("AudioRecord reported an invalid minimum buffer size");
        }
        return Math.max(4096, minimum * 2);
    }

    private AudioRecord createAudioRecorder(int bufferSize) {
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            throw new SecurityException("Android microphone permission is not granted");
        }
        AudioRecord candidate = new AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
        );
        if (candidate.getState() != AudioRecord.STATE_INITIALIZED) {
            candidate.release();
            throw new IllegalStateException("AudioRecord initialization failed");
        }
        return candidate;
    }

    private static AdaptiveEnergyVad.Config createVadConfig() {
        return AdaptiveEnergyVad.Config.builder()
                .setPreRollMs(VAD_PRE_ROLL_MS)
                .setSilenceEndpointMs(VAD_ENDPOINT_HANGOVER_MS)
                .setMaxUtteranceMs(VAD_MAX_UTTERANCE_MS)
                .setBargeInPolicy(AdaptiveEnergyVad.BargeInPolicy.DISABLED)
                .build();
    }

    private String newVadSessionIdLocked() {
        nextVadSessionId += 1;
        return "vad-"
                + Long.toString(SystemClock.elapsedRealtimeNanos(), 36)
                + "-"
                + Long.toString(nextVadSessionId, 36);
    }

    private void clearVadSessionLocked() {
        recorder = null;
        recordingThread = null;
        recording = false;
        if (captureMode == CaptureMode.VAD) {
            captureMode = CaptureMode.NONE;
        }
        vad = null;
        vadPcmRing = null;
        pendingVadEndpoint = null;
        vadSessionId = null;
        vadSessionActive = false;
        vadPaused = false;
        vadStopRequested = false;
    }

    private static void stopAndReleaseRecorder(AudioRecord activeRecorder) {
        try {
            activeRecorder.stop();
        } catch (IllegalStateException ignored) {
        }
        activeRecorder.release();
    }

    private static void joinCaptureThread(Thread thread) {
        if (thread == null || thread == Thread.currentThread()) {
            return;
        }
        try {
            thread.join(1500);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        }
    }

    private static String errorMessage(RuntimeException error, String fallback) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? fallback : message;
    }

    private static JSONObject vadConfigDetails(AdaptiveEnergyVad.Config config) {
        JSONObject details = new JSONObject();
        putJson(details, "sample_rate", config.getSampleRateHz());
        putJson(details, "frame_samples", VAD_FRAME_SAMPLES);
        putJson(details, "pre_roll_samples", config.getPreRollSamples());
        putJson(details, "silence_endpoint_ms", config.getSilenceEndpointMs());
        putJson(details, "max_utterance_ms", config.getMaxUtteranceMs());
        putJson(details, "ring_capacity_samples", VAD_RING_CAPACITY_SAMPLES);
        putJson(details, "barge_in_policy", "disabled");
        putJson(details, "half_duplex", true);
        return details;
    }

    private static JSONObject vadEventDetails(AdaptiveEnergyVad.Event event) {
        JSONObject details = new JSONObject();
        putJson(details, "sequence", event.getSequence());
        putJson(details, "previous_state", enumName(event.getPreviousState()));
        putJson(details, "state", enumName(event.getState()));
        putJson(details, "endpoint_reason", enumName(event.getEndpointReason()));
        putJson(details, "decision_sample", event.getDecisionSample());
        putJson(details, "speech_start_sample", event.getSpeechStartSample());
        putJson(details, "speech_end_sample", event.getSpeechEndSample());
        putJson(details, "audio_start_sample", event.getRecommendedAudioStartSample());
        putJson(details, "rms_dbfs", event.getRmsDbfs());
        putJson(details, "noise_floor_dbfs", event.getNoiseFloorDbfs());
        putJson(details, "threshold_dbfs", event.getThresholdDbfs());
        putJson(details, "barge_in", event.isBargeIn());
        putJson(details, "cancel_output_recommended", event.isCancelOutputRecommended());
        return details;
    }

    private static JSONObject vadMetricsDetails(AdaptiveEnergyVad.Metrics metrics) {
        JSONObject details = new JSONObject();
        putJson(details, "frames_processed", metrics.getFramesProcessed());
        putJson(details, "samples_processed", metrics.getSamplesProcessed());
        putJson(details, "confirmed_utterances", metrics.getConfirmedUtterances());
        putJson(details, "endpoints", metrics.getEndpoints());
        putJson(details, "trailing_silence_endpoints", metrics.getTrailingSilenceEndpoints());
        putJson(details, "maximum_duration_endpoints", metrics.getMaximumDurationEndpoints());
        putJson(details, "rejected_short_bursts", metrics.getRejectedShortBursts());
        putJson(details, "rejected_impulse_frames", metrics.getRejectedImpulseFrames());
        putJson(details, "output_blocked_frames", metrics.getOutputBlockedFrames());
        putJson(details, "noise_floor_dbfs", metrics.getNoiseFloorDbfs());
        return details;
    }

    private static String enumName(Enum<?> value) {
        return value == null ? "none" : value.name().toLowerCase(Locale.US);
    }

    private static void putJson(JSONObject object, String name, Object value) {
        try {
            object.put(name, value);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
    }

    private void recordVadTelemetry(String name, String sessionId, JSONObject details) {
        JSONObject event = new JSONObject();
        putJson(event, "schema_version", 1);
        putJson(event, "event", name);
        putJson(event, "source", "native");
        putJson(event, "mode", "vad_conversation");
        putJson(event, "session_id", sessionId == null ? "unassigned" : sessionId);
        putJson(event, "monotonic_ms", monotonicNowMs());
        if (details != null) {
            putJson(event, "details", details);
        }
        appendTelemetry(event);
    }

    private void emitVadCallback(String type, String sessionId, JSONObject details) {
        JSONObject event = new JSONObject();
        putJson(event, "schema_version", 1);
        putJson(event, "type", type);
        putJson(event, "session_id", sessionId == null ? "unassigned" : sessionId);
        putJson(event, "monotonic_ms", monotonicNowMs());
        if (details != null) {
            putJson(event, "details", details);
        }
        postJavascriptStringCallback("onNativeVadEvent", event.toString());
    }

    private void emitVadError(String sessionId, String stage, String message) {
        JSONObject details = new JSONObject();
        putJson(details, "stage", stage);
        putJson(details, "message", message == null ? "Unknown VAD error" : message);
        recordVadTelemetry("native_vad_error", sessionId, details);
        emitVadCallback("error", sessionId, details);
    }

    private void recordNativeTelemetry(String name, JSONObject details) {
        recordNativeTelemetry(name, null, details);
    }

    private void recordNativeTelemetry(String name, String turnId, JSONObject details) {
        JSONObject event = new JSONObject();
        try {
            event.put("schema_version", 1);
            event.put("event", name);
            event.put("source", "native");
            event.put("monotonic_ms", monotonicNowMs());
            if (turnId != null) {
                event.put("turn_id", turnId);
                event.put("mode", "native_tts");
            }
            if (details != null) {
                event.put("details", details);
            }
            appendTelemetry(event);
        } catch (Exception error) {
            Log.e(TAG, "Could not build native telemetry", error);
        }
    }

    private void initializeTtsOnUi(String localeTag, String turnId, long generation) {
        if (released) {
            return;
        }
        if (generation < latestTtsGeneration
                || (generation == latestTtsGeneration
                && !turnId.equals(pendingTtsTurnId)
                && !turnId.equals(currentTtsTurnId))) {
            recordStaleTtsCommand("initialize", turnId, generation);
            return;
        }
        // A real accepted turn always wins over speculative idle warming.
        // Invalidating here (after the stale-turn gate) prevents an obsolete
        // WebView command from cancelling a newer preload request.
        invalidatePendingTtsPreloadOnUi("accepted_tts_turn_started");
        if (generation > latestTtsGeneration) {
            if (currentTtsTurnId != null) {
                cancelTtsOnUi(currentTtsTurnId, currentTtsGeneration, "superseded");
            }
            if (pendingTtsTurnId != null && "system".equals(pendingTtsEngineId)) {
                cancelTtsOnUi(pendingTtsTurnId, pendingTtsGeneration, "superseded");
            }
            latestTtsGeneration = generation;
        }
        String requestedLocaleTag = localeTag;
        String resolvedLocaleTag = normalizedTtsLocaleTag(localeTag);
        String policy = selectedTtsEngineId;
        pendingTtsLocaleTag = resolvedLocaleTag;
        pendingTtsTurnId = turnId;
        pendingTtsGeneration = generation;
        pendingTtsPolicyId = policy;
        pendingTtsEngineId = null;

        Candidate candidate = null;
        if ("auto".equals(policy)) {
            java.util.List<Candidate> ranked = autoCandidatesForLocale(resolvedLocaleTag);
            Candidate preferred = ranked.isEmpty() ? null : ranked.get(0);
            candidate = firstAvailableAutoCandidate(ranked);
            if (preferred != null && candidate != preferred) {
                java.util.List<String> missing = preferred.hasAllFiles(ttsBundleRoot());
                boolean systemFallbackAvailable = candidate == null
                        && canInitializeSystemTtsForLocale(resolvedLocaleTag);
                recordAutomaticTtsFallback(
                        turnId,
                        generation,
                        preferred,
                        candidate == null
                                ? (systemFallbackAvailable ? "system" : "none")
                                : candidate.getId(),
                        "missing_files",
                        "Missing " + missing.size() + " required voice file(s)."
                );
            }
        } else if (!"system".equals(policy)) {
            candidate = candidateById(policy);
            if (candidate == null
                    || !candidateSupportsLocale(candidate, resolvedLocaleTag)
                    || !candidate.hasAllFiles(ttsBundleRoot()).isEmpty()) {
                resolvedTtsEngineId = null;
                reportTtsError(
                        turnId,
                        generation,
                        "init",
                        "The selected offline voice is unavailable for "
                                + resolvedLocaleTag + ". Reselect a voice."
                );
                return;
            }
        }

        if (candidate != null) {
            resolvedTtsEngineId = candidate.getId();
            pendingTtsEngineId = candidate.getId();
            initializeSherpaTtsOnUi(
                    resolvedLocaleTag,
                    requestedLocaleTag,
                    turnId,
                    generation,
                    policy,
                    candidate
            );
            return;
        }
        if (!canInitializeSystemTtsForLocale(resolvedLocaleTag)) {
            resolvedTtsEngineId = null;
            pendingTtsEngineId = null;
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "No installed offline TTS voice supports " + resolvedLocaleTag + "."
            );
            return;
        }
        resolvedTtsEngineId = "system";
        pendingTtsEngineId = "system";
        initializeSystemTtsOnUi(resolvedLocaleTag, turnId, generation);
    }

    private void initializeSystemTtsOnUi(String localeTag, String turnId, long generation) {
        // A system/eSpeak route must not leave a large neural model or an
        // in-flight neural load resident after a language switch.
        sherpaLoadToken += 1;
        releaseSherpaEngineOnUi();
        if (textToSpeechReady && textToSpeech != null) {
            configureTtsLocaleOnUi(localeTag, turnId, generation);
            return;
        }
        if (textToSpeech != null) {
            return;
        }
        if (!hasSystemTtsEngine()) {
            reportTtsError(
                    turnId,
                    generation,
                    "init",
                    "Offline eSpeak is not installed or enabled: " + TTS_ENGINE_PACKAGE + "."
            );
            return;
        }
        createSystemTtsOnUi();
    }

    private void onTtsEngineInitializedOnUi(int status) {
        if (released) {
            return;
        }
        if (status != TextToSpeech.SUCCESS || textToSpeech == null) {
            textToSpeechReady = false;
            systemTtsOfflineLanguages = java.util.Collections.emptySet();
            systemTtsCapabilitiesKnown = true;
            if (textToSpeech != null) {
                textToSpeech.shutdown();
            }
            textToSpeech = null;
            if (pendingTtsTurnId != null) {
                reportTtsError(
                        pendingTtsTurnId,
                        pendingTtsGeneration,
                        "init",
                        "Offline eSpeak Text-to-speech could not start. Install or enable "
                                + TTS_ENGINE_PACKAGE + "."
                );
            }
            return;
        }
        textToSpeechReady = true;
        refreshSystemTtsCapabilitiesOnUi();
        textToSpeech.setOnUtteranceProgressListener(ttsListener);
        if (pendingTtsTurnId != null && "system".equals(pendingTtsEngineId)) {
            configureTtsLocaleOnUi(
                    pendingTtsLocaleTag,
                    pendingTtsTurnId,
                    pendingTtsGeneration
            );
        }
    }

    private void configureTtsLocaleOnUi(String requestedTag, String turnId, long generation) {
        if (released || textToSpeech == null || !textToSpeechReady) {
            return;
        }
        if (!turnId.equals(pendingTtsTurnId)
                || generation != pendingTtsGeneration
                || generation != latestTtsGeneration
                || !"system".equals(pendingTtsEngineId)) {
            recordStaleTtsCommand("configure", turnId, generation);
            return;
        }
        String resolvedTag = normalizedTtsLocaleTag(requestedTag);
        Locale locale = Locale.forLanguageTag(resolvedTag);
        if (locale.getLanguage().trim().isEmpty()) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "Invalid Android TTS locale: " + resolvedTag
            );
            return;
        }

        int availability = textToSpeech.isLanguageAvailable(locale);
        if (availability == TextToSpeech.LANG_MISSING_DATA) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "Voice data for " + resolvedTag
                            + " is missing. Install it in Android Text-to-speech settings."
            );
            return;
        }
        if (availability == TextToSpeech.LANG_NOT_SUPPORTED) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "The active Android TTS engine does not support " + resolvedTag + "."
            );
            return;
        }
        if (textToSpeech.setLanguage(locale) == TextToSpeech.ERROR) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "Android TTS rejected locale " + resolvedTag + "."
            );
            return;
        }

        Voice preferred = selectPreferredVoice(locale, textToSpeech.getVoices());
        if (preferred == null) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "The active Android TTS engine has no offline voice for " + resolvedTag + "."
            );
            return;
        }
        boolean preferredVoiceAccepted = textToSpeech.setVoice(preferred) != TextToSpeech.ERROR;
        if (!preferredVoiceAccepted && textToSpeech.setLanguage(locale) == TextToSpeech.ERROR) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "Android TTS rejected the selected voice for " + resolvedTag + "."
            );
            return;
        }
        Voice activeVoice = textToSpeech.getVoice();
        if (activeVoice == null
                || activeVoice.getLocale() == null
                || !locale.getLanguage().equals(activeVoice.getLocale().getLanguage())
                || !isUsableOfflineVoice(activeVoice)) {
            reportTtsError(
                    turnId,
                    generation,
                    "locale",
                    "Android TTS did not activate a matching offline voice for "
                            + resolvedTag + "."
            );
            return;
        }
        String actualLocale = activeVoice.getLocale().toLanguageTag();

        if (currentTtsTurnId != null && !currentTtsTurnId.equals(turnId)) {
            cancelTtsOnUi(currentTtsTurnId, currentTtsGeneration, "new_turn");
        }
        currentTtsTurnId = turnId;
        currentTtsLocaleTag = actualLocale;
        currentTtsGeneration = generation;
        String policy = pendingTtsPolicyId == null ? selectedTtsEngineId : pendingTtsPolicyId;
        resolvedTtsEngineId = "system";
        pendingTtsTurnId = null;
        pendingTtsLocaleTag = null;
        pendingTtsGeneration = -1;
        pendingTtsPolicyId = null;
        pendingTtsEngineId = null;
        nextTtsUtterance = 0;
        firstTtsChunk = true;
        TtsTurnState turn = new TtsTurnState(
                turnId,
                actualLocale,
                generation,
                policy,
                "system"
        );
        ttsTurns.put(turnId, turn);

        JSONObject details = new JSONObject();
        try {
            details.put("requested_locale", requestedTag);
            details.put("resolved_locale", actualLocale);
            details.put("tts_policy", policy);
            details.put("tts_engine_id", "system");
            details.put("engine_package", TTS_ENGINE_PACKAGE);
            details.put("availability", availability);
            details.put("generation", generation);
            details.put("preferred_voice_accepted", preferredVoiceAccepted);
            details.put("voice_name", activeVoice.getName());
            details.put(
                    "network_connection_required",
                    activeVoice.isNetworkConnectionRequired()
            );
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_init_ready", turnId, details);
        emitTtsCallback("ready", turnId, details);
    }

    private void queueTtsChunkOnUi(
            String text,
            String turnId,
            String chunkId,
            long generation
    ) {
        if (released) {
            return;
        }
        if (!isCurrentTtsOwner(turnId, generation)) {
            recordStaleTtsCommand("queue_chunk", turnId, generation);
            return;
        }
        if (text.trim().isEmpty()) {
            reportTtsError(
                    turnId,
                    generation,
                    "queue",
                    "Refusing to queue an empty TTS chunk."
            );
            return;
        }
        if (text.length() > 1000) {
            reportTtsError(
                    turnId,
                    generation,
                    "queue",
                    "A TTS chunk exceeded the 1000-character safety limit."
            );
            return;
        }
        TtsTurnState turn = ttsTurns.get(turnId);
        if (turn == null || turn.generation != generation || turn.terminal.get()) {
            reportTtsError(
                    turnId,
                    generation,
                    "queue",
                    "Android TTS turn state is unavailable."
            );
            return;
        }
        if (turn.queueClosed.get()) {
            recordStaleTtsCommand("queue_after_close", turnId, generation);
            return;
        }
        if (!turn.usesSherpaTts() && (textToSpeech == null || !textToSpeechReady)) {
            reportTtsError(
                    turnId,
                    generation,
                    "queue",
                    "Offline eSpeak is not ready for this response."
            );
            return;
        }
        if (turn.usesSherpaTts()
                && (sherpaTts == null
                || !turn.engineId.equals(sherpaTts.getLoadedCandidateId()))) {
            reportTtsError(
                    turnId,
                    generation,
                    "queue",
                    "The resolved offline voice is not loaded for this response."
            );
            return;
        }
        int chunkIndex = nextTtsUtterance++;
        String utteranceId = "lfm-tts-" + turnId + "-g" + generation + "-" + chunkIndex;
        TtsUtterance utterance = new TtsUtterance(
                utteranceId,
                turnId,
                chunkId,
                chunkIndex,
                currentTtsLocaleTag,
                generation,
                text.codePointCount(0, text.length()),
                turn.policyId,
                turn.engineId
        );
        ttsUtterances.put(utteranceId, utterance);
        turn.pending.incrementAndGet();

        int queueMode = firstTtsChunk ? TextToSpeech.QUEUE_FLUSH : TextToSpeech.QUEUE_ADD;
        if (firstTtsChunk) {
            JSONObject flushDetails = new JSONObject();
            try {
                flushDetails.put("reason", "first_chunk");
            } catch (Exception impossible) {
                throw new AssertionError(impossible);
            }
            recordNativeTelemetry("native_tts_queue_flush", turnId, flushDetails);
            firstTtsChunk = false;
        }
        if (turn.usesSherpaTts()) {
            sherpaTts.speak(text, utteranceId, queueMode == TextToSpeech.QUEUE_FLUSH);
            turn.queued.incrementAndGet();
            JSONObject sherpaDetails = utterance.details();
            try {
                sherpaDetails.put("queue_mode",
                        queueMode == TextToSpeech.QUEUE_FLUSH ? "flush" : "add");
                sherpaDetails.put("pending_utterances", turn.pending.get());
                sherpaDetails.put("tts_policy", turn.policyId);
                sherpaDetails.put("tts_engine_id", turn.engineId);
            } catch (Exception impossible) {
                throw new AssertionError(impossible);
            }
            recordNativeTelemetry("native_tts_chunk_queued", turnId, sherpaDetails);
            emitTtsCallback("chunk_queued", turnId, sherpaDetails);
            return;
        }
        int result = textToSpeech.speak(text, queueMode, new Bundle(), utteranceId);
        if (result == TextToSpeech.ERROR) {
            ttsUtterances.remove(utteranceId);
            turn.pending.decrementAndGet();
            reportTtsUtteranceError(utterance, TextToSpeech.ERROR, "speak_rejected");
            failTtsTurnOnUi(
                    turnId,
                    generation,
                    "speak_rejected",
                    "Android TTS rejected a queued speech chunk."
            );
            return;
        }
        turn.queued.incrementAndGet();

        JSONObject details = utterance.details();
        try {
            details.put("queue_mode", queueMode == TextToSpeech.QUEUE_FLUSH ? "flush" : "add");
            details.put("pending_utterances", turn.pending.get());
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_chunk_queued", turnId, details);
        emitTtsCallback("chunk_queued", turnId, details);
    }

    private void finishTtsResponseOnUi(String turnId, long generation) {
        if (!isCurrentTtsOwner(turnId, generation)) {
            recordStaleTtsCommand("finish_response", turnId, generation);
            return;
        }
        TtsTurnState turn = ttsTurns.get(turnId);
        if (turn == null || turn.generation != generation || turn.terminal.get()) {
            recordStaleTtsCommand("finish_response", turnId, generation);
            return;
        }
        int pending = turn.pending.get();
        turn.queueClosed.set(true);
        JSONObject details = new JSONObject();
        try {
            details.put("pending_utterances", pending);
            details.put("chunks_queued", turn.queued.get());
            details.put("generation", generation);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_queue_closed", turnId, details);
        emitTtsCallback("queue_closed", turnId, details);
        completeTtsTurnIfDrainedOnUi(turnId, generation);
    }

    private void cancelTtsOnUi(String turnId, long generation, String reason) {
        if (generation > latestTtsGeneration) {
            latestTtsGeneration = generation;
        }
        boolean currentMatch = isCurrentTtsOwner(turnId, generation);
        boolean pendingMatch = turnId.equals(pendingTtsTurnId)
                && generation == pendingTtsGeneration;
        if (!currentMatch && !pendingMatch) {
            recordStaleTtsCommand("cancel", turnId, generation);
            return;
        }
        if (currentMatch && textToSpeech != null) {
            textToSpeech.stop();
        }
        if (currentMatch && sherpaTts != null) {
            sherpaTts.stop();
        }
        int removed = removeTtsUtterances(turnId, generation);
        TtsTurnState turn = ttsTurns.get(turnId);
        int chunksQueued = turn == null ? 0 : turn.queued.get();
        if (turn != null && turn.generation == generation) {
            turn.pending.set(0);
        }
        if (pendingMatch) {
            pendingTtsTurnId = null;
            pendingTtsLocaleTag = null;
            pendingTtsGeneration = -1;
            pendingTtsPolicyId = null;
            pendingTtsEngineId = null;
        }
        JSONObject details = new JSONObject();
        try {
            details.put("reason", reason);
            details.put("removed_utterances", removed);
            details.put("generation", generation);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_queue_cancel", turnId, details);
        emitTtsCallback("queue_cancelled", turnId, details);
        emitTtsTerminalOnUi(
                turnId,
                generation,
                "cancelled",
                0,
                chunksQueued,
                reason
        );
    }

    private void reportTtsError(String turnId, long generation, String stage, String message) {
        JSONObject details = new JSONObject();
        try {
            details.put("stage", stage);
            details.put("message", message);
            details.put("generation", generation);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        String eventName = "init".equals(stage) || "locale".equals(stage)
                ? "native_tts_init_error"
                : "native_tts_error";
        recordNativeTelemetry(eventName, turnId, details);
        emitTtsCallback("error", turnId, details);
        failTtsTurnOnUi(turnId, generation, stage, message);
    }

    private void reportTtsUtteranceError(TtsUtterance utterance, int errorCode, String stage) {
        JSONObject details = utterance.details();
        try {
            details.put("stage", stage);
            details.put("error_code", errorCode);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_utterance_error", utterance.turnId, details);
        emitTtsCallback("utterance_error", utterance.turnId, details);
    }

    private void completeTtsTurnIfDrainedOnUi(String turnId, long generation) {
        TtsTurnState turn = ttsTurns.get(turnId);
        if (turn == null
                || turn.generation != generation
                || !turn.queueClosed.get()
                || turn.pending.get() != 0
                || turn.terminal.get()) {
            return;
        }
        emitTtsTerminalOnUi(
                turnId,
                generation,
                "completed",
                0,
                turn.queued.get(),
                null
        );
    }

    private void failTtsTurnOnUi(
            String turnId,
            long generation,
            String stage,
            String message
    ) {
        boolean currentMatch = isCurrentTtsOwner(turnId, generation);
        boolean pendingMatch = turnId.equals(pendingTtsTurnId)
                && generation == pendingTtsGeneration;
        if (!currentMatch && !pendingMatch) {
            recordStaleTtsCommand("fail", turnId, generation);
            return;
        }
        if (currentMatch && textToSpeech != null) {
            textToSpeech.stop();
        }
        if (currentMatch && sherpaTts != null) {
            sherpaTts.stop();
        }
        removeTtsUtterances(turnId, generation);
        TtsTurnState turn = ttsTurns.get(turnId);
        int chunksQueued = turn == null ? 0 : turn.queued.get();
        if (turn != null && turn.generation == generation) {
            turn.pending.set(0);
        }
        if (pendingMatch) {
            pendingTtsTurnId = null;
            pendingTtsLocaleTag = null;
            pendingTtsGeneration = -1;
            pendingTtsPolicyId = null;
            pendingTtsEngineId = null;
        }
        emitTtsTerminalOnUi(
                turnId,
                generation,
                "failed",
                0,
                chunksQueued,
                stage + ": " + message
        );
    }

    private void emitTtsTerminalOnUi(
            String turnId,
            long generation,
            String status,
            int pending,
            int chunksQueued,
            String message
    ) {
        TtsTurnState turn = ttsTurns.get(turnId);
        if (turn != null && turn.generation == generation
                && !turn.terminal.compareAndSet(false, true)) {
            return;
        }
        JSONObject details = new JSONObject();
        try {
            details.put("status", status);
            details.put("pending_utterances", pending);
            details.put("chunks_queued", chunksQueued);
            details.put("generation", generation);
            if (turn != null && turn.generation == generation) {
                details.put("tts_policy", turn.policyId);
                details.put("tts_engine_id", turn.engineId);
            }
            if (message != null) {
                details.put("message", message);
            }
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_playback_terminal", turnId, details);
        emitTtsCallback("terminal", turnId, details);
        if (turn != null && turn.generation == generation) {
            ttsTurns.remove(turnId, turn);
        }
        if (isCurrentTtsOwner(turnId, generation)) {
            currentTtsTurnId = null;
            currentTtsLocaleTag = null;
            currentTtsGeneration = -1;
            firstTtsChunk = true;
        }
        if (turnId.equals(pendingTtsTurnId) && generation == pendingTtsGeneration) {
            pendingTtsTurnId = null;
            pendingTtsLocaleTag = null;
            pendingTtsGeneration = -1;
            pendingTtsPolicyId = null;
            pendingTtsEngineId = null;
        }
    }

    private int removeTtsUtterances(String turnId, long generation) {
        int removed = 0;
        for (TtsUtterance utterance : ttsUtterances.values()) {
            if (turnId.equals(utterance.turnId)
                    && generation == utterance.generation
                    && ttsUtterances.remove(utterance.utteranceId) != null) {
                removed += 1;
            }
        }
        return removed;
    }

    private boolean isCurrentTtsOwner(String turnId, long generation) {
        return turnId != null
                && turnId.equals(currentTtsTurnId)
                && generation == currentTtsGeneration;
    }

    private void recordStaleTtsCommand(String command, String turnId, long generation) {
        JSONObject details = new JSONObject();
        try {
            details.put("command", command);
            details.put("generation", generation);
            details.put("latest_generation", latestTtsGeneration);
            details.put("current_turn_id", currentTtsTurnId);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_stale_command_ignored", turnId, details);
        emitTtsCallback("stale_ignored", turnId, details);
    }

    private void emitTtsCallback(String type, String turnId, JSONObject details) {
        JSONObject event = new JSONObject();
        try {
            event.put("schema_version", 1);
            event.put("type", type);
            event.put("turn_id", turnId);
            event.put("monotonic_ms", monotonicNowMs());
            if (details != null) {
                event.put("details", details);
            }
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        postJavascriptStringCallback("onNativeTtsEvent", event.toString());
    }

    private void postJavascriptStringCallback(String functionName, String value) {
        postJavascriptCallback(
                "window." + functionName + "(" + JSONObject.quote(value == null ? "" : value) + ")"
        );
    }

    private void postJavascriptCallback(String script) {
        postToWebView(() -> webView.evaluateJavascript(script, null));
    }

    private void postToWebView(Runnable action) {
        if (released) {
            return;
        }
        webView.post(() -> {
            if (!released) {
                action.run();
            }
        });
    }

    private static String cleanIdentifier(String value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String cleaned = value.trim();
        if (cleaned.isEmpty()) {
            return fallback;
        }
        return cleaned.length() > 200 ? cleaned.substring(0, 200) : cleaned;
    }

    private static Voice selectPreferredVoice(Locale requested, Set<Voice> voices) {
        if (voices == null || voices.isEmpty()) {
            return null;
        }
        Voice best = null;
        int bestScore = Integer.MIN_VALUE;
        for (Voice voice : voices) {
            Locale candidate = voice.getLocale();
            if (candidate == null
                    || !isUsableOfflineVoice(voice)
                    || !requested.getLanguage().equals(candidate.getLanguage())) {
                continue;
            }
            int score = 100;
            if (!requested.getCountry().trim().isEmpty()
                    && requested.getCountry().equals(candidate.getCountry())) {
                score += 20;
            }
            score += voice.getQuality();
            if (score > bestScore) {
                best = voice;
                bestScore = score;
            }
        }
        return best;
    }

    private static boolean isUsableOfflineVoice(Voice voice) {
        Set<String> features = voice.getFeatures();
        return !voice.isNetworkConnectionRequired()
                && (features == null
                || !features.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED));
    }

    /**
     * Load the selected offline voice and hand the turn over to it.
     *
     * Loading is done on a worker thread: these models take 0.6-2.5 s to load
     * on this phone and blocking the UI thread for that would stall the whole
     * conversation. The turn is only marked ready once the voice is resident.
     */
    private void initializeSherpaTtsOnUi(
            String localeTag,
            String requestedLocaleTag,
            String turnId,
            long generation,
            String policy,
            Candidate candidate
    ) {
        if (!turnId.equals(pendingTtsTurnId)
                || generation != pendingTtsGeneration
                || !policy.equals(pendingTtsPolicyId)
                || !candidate.getId().equals(pendingTtsEngineId)) {
            recordStaleTtsCommand("initialize_sherpa", turnId, generation);
            return;
        }
        if (sherpaTts != null && sherpaTts.canReuseModel(candidate)) {
            sherpaTts.selectCandidate(candidate);
            markSherpaTtsReadyOnUi(
                    localeTag,
                    requestedLocaleTag,
                    turnId,
                    generation,
                    policy,
                    candidate
            );
            return;
        }

        // Detach the old model now, then release it on the same worker before
        // allocating the replacement. This avoids a 0.5-1.2 GiB dual-model
        // residency spike during automatic neural-language switching.
        final SherpaTtsEngine previousEngine = sherpaTts;
        sherpaTts = null;
        final String expectedPolicy = policy;
        final String expectedEngineId = candidate.getId();
        final long loadToken = ++sherpaLoadToken;
        final SherpaTtsEngine engine = new SherpaTtsEngine(ttsBundleRoot(), new SherpaTtsListener());
        Runnable loadTask = () -> {
            try {
                if (previousEngine != null) previousEngine.shutdownAndAwaitRelease();
                if (released || loadToken != sherpaLoadToken) return;
                engine.load(candidate);
                if (released || loadToken != sherpaLoadToken) {
                    engine.shutdownAndAwaitRelease();
                    return;
                }

                java.util.concurrent.CountDownLatch handoff =
                        new java.util.concurrent.CountDownLatch(1);
                AtomicBoolean handedOff = new AtomicBoolean(false);
                boolean posted = webView.post(() -> {
                    try {
                        boolean stillCurrent = !released
                                && loadToken == sherpaLoadToken
                                && candidate.getId().equals(expectedEngineId)
                                && expectedPolicy.equals(selectedTtsEngineId)
                                && turnId.equals(pendingTtsTurnId)
                                && generation == pendingTtsGeneration
                                && expectedPolicy.equals(pendingTtsPolicyId)
                                && expectedEngineId.equals(pendingTtsEngineId)
                                && expectedEngineId.equals(resolvedTtsEngineId);
                        if (!stillCurrent) return;
                        sherpaTts = engine;
                        markSherpaTtsReadyOnUi(
                                localeTag,
                                requestedLocaleTag,
                                turnId,
                                generation,
                                expectedPolicy,
                                candidate
                        );
                        handedOff.set(true);
                    } finally {
                        handoff.countDown();
                    }
                });
                if (posted) {
                    boolean interrupted = false;
                    while (true) {
                        try {
                            handoff.await();
                            break;
                        } catch (InterruptedException ignored) {
                            interrupted = true;
                        }
                    }
                    if (interrupted) Thread.currentThread().interrupt();
                }
                if (!posted || !handedOff.get()) engine.shutdownAndAwaitRelease();
            } catch (Throwable error) {
                Log.e(TAG, "Offline voice failed to load", error);
                engine.shutdownAndAwaitRelease();
                postSherpaLoadFailure(() -> {
                    if (loadToken == sherpaLoadToken
                            && turnId.equals(pendingTtsTurnId)
                            && generation == pendingTtsGeneration
                            && expectedPolicy.equals(pendingTtsPolicyId)
                            && expectedEngineId.equals(pendingTtsEngineId)) {
                        String message = "Offline voice \"" + candidate.getLabel()
                                + "\" could not load: " + error.getMessage();
                        if ("auto".equals(expectedPolicy)) {
                            Candidate fallbackCandidate = nextAvailableAutoCandidateForLocale(
                                    localeTag,
                                    candidate
                            );
                            boolean systemFallbackAvailable = fallbackCandidate == null
                                    && canInitializeSystemTtsForLocale(localeTag);
                            recordAutomaticTtsFallback(
                                    turnId,
                                    generation,
                                    candidate,
                                    fallbackCandidate == null
                                            ? (systemFallbackAvailable ? "system" : "none")
                                            : fallbackCandidate.getId(),
                                    "load_failure",
                                    message
                            );
                            if (fallbackCandidate != null) {
                                resolvedTtsEngineId = fallbackCandidate.getId();
                                pendingTtsEngineId = fallbackCandidate.getId();
                                initializeSherpaTtsOnUi(
                                        localeTag,
                                        requestedLocaleTag,
                                        turnId,
                                        generation,
                                        expectedPolicy,
                                        fallbackCandidate
                                );
                            } else if (systemFallbackAvailable) {
                                resolvedTtsEngineId = "system";
                                pendingTtsEngineId = "system";
                                initializeSystemTtsOnUi(localeTag, turnId, generation);
                            } else {
                                resolvedTtsEngineId = null;
                                pendingTtsEngineId = null;
                                reportTtsError(
                                        turnId,
                                        generation,
                                        "init",
                                        message + " No installed offline system voice supports "
                                                + localeTag + "."
                                );
                            }
                        } else {
                            resolvedTtsEngineId = null;
                            reportTtsError(turnId, generation, "init", message);
                        }
                    }
                });
            }
        };
        try {
            sherpaModelExecutor.execute(loadTask);
        } catch (java.util.concurrent.RejectedExecutionException rejected) {
            engine.shutdownAndAwaitRelease();
        }
    }

    private void postSherpaLoadFailure(Runnable action) {
        if (released) return;
        webView.post(() -> {
            if (!released) action.run();
        });
    }

    private void markSherpaTtsReadyOnUi(
            String localeTag,
            String requestedLocaleTag,
            String turnId,
            long generation,
            String policy,
            Candidate candidate
    ) {
        if (released
                || !turnId.equals(pendingTtsTurnId)
                || generation != pendingTtsGeneration
                || !policy.equals(pendingTtsPolicyId)
                || !candidate.getId().equals(pendingTtsEngineId)) {
            return;
        }
        currentTtsTurnId = turnId;
        currentTtsGeneration = generation;
        currentTtsLocaleTag = localeTag;
        resolvedTtsEngineId = candidate.getId();
        pendingTtsTurnId = null;
        pendingTtsLocaleTag = null;
        pendingTtsGeneration = -1;
        pendingTtsPolicyId = null;
        pendingTtsEngineId = null;
        nextTtsUtterance = 0;
        firstTtsChunk = true;
        ttsTurns.put(turnId, new TtsTurnState(
                turnId,
                localeTag,
                generation,
                policy,
                candidate.getId()
        ));

        JSONObject details = new JSONObject();
        try {
            details.put("engine", candidate.getId());
            details.put("tts_policy", policy);
            details.put("tts_engine_id", candidate.getId());
            details.put("engine_label", candidate.getLabel());
            details.put("engine_family", candidate.getEngine());
            details.put("threads", candidate.getNumThreads());
            details.put("requested_locale", requestedLocaleTag);
            details.put("resolved_locale", candidate.getLanguage());
            details.put("sample_rate_hz", sherpaTts == null ? 0 : sherpaTts.getSampleRate());
            details.put("generation", generation);
        } catch (Exception impossible) {
            throw new AssertionError(impossible);
        }
        recordNativeTelemetry("native_tts_init_ready", turnId, details);
        emitTtsCallback("ready", turnId, details);
    }

    /**
     * Bridges the offline engine's progress onto exactly the same turn
     * bookkeeping the eSpeak path uses, so barge-in, queue draining and
     * telemetry behave identically whichever voice is selected.
     */
    private final class SherpaTtsListener implements SherpaTtsEngine.Listener {
        @Override
        public void onUtteranceStart(String utteranceId) {
            postToWebView(() -> ttsListener.handleStartOnUi(utteranceId));
        }

        @Override
        public void onUtteranceDone(String utteranceId) {
            postToWebView(() -> ttsListener.handleDoneOnUi(utteranceId));
        }

        @Override
        public void onUtteranceError(String utteranceId, String reason) {
            Log.w(TAG, "offline voice utterance failed: " + reason);
            postToWebView(() -> ttsListener.handleErrorOnUi(
                    utteranceId,
                    TextToSpeech.ERROR,
                    reason
            ));
        }

        @Override
        public void onUtteranceMetrics(
                String utteranceId,
                java.util.Map<String, ? extends Object> metrics
        ) {
            postToWebView(() -> handleMetricsOnUi(utteranceId, metrics));
        }

        private void handleMetricsOnUi(
                String utteranceId,
                java.util.Map<String, ? extends Object> metrics
        ) {
            TtsUtterance utterance = ttsUtterances.get(utteranceId);
            String turnId = utterance == null ? "unknown-turn" : utterance.turnId;
            JSONObject details = new JSONObject();
            try {
                for (java.util.Map.Entry<String, ? extends Object> entry : metrics.entrySet()) {
                    details.put(entry.getKey(), entry.getValue());
                }
                if (utterance != null) {
                    details.put("generation", utterance.generation);
                }
            } catch (Exception error) {
                Log.e(TAG, "Could not encode TTS metrics", error);
                return;
            }
            lastTtsMetrics.put(turnId, details);
            recordNativeTelemetry("native_tts_engine_metrics", turnId, details);
            emitTtsCallback("engine_metrics", turnId, details);
        }
    }

    private final class TtsProgressListener extends UtteranceProgressListener {
        @Override
        public void onStart(String utteranceId) {
            postToWebView(() -> handleStartOnUi(utteranceId));
        }

        private void handleStartOnUi(String utteranceId) {
            TtsUtterance utterance = ttsUtterances.get(utteranceId);
            if (utterance == null || released) {
                return;
            }
            TtsTurnState turn = ttsTurns.get(utterance.turnId);
            if (turn == null
                    || turn.generation != utterance.generation
                    || turn.terminal.get()) {
                return;
            }
            JSONObject details = utterance.details();
            recordNativeTelemetry("native_tts_utterance_start", utterance.turnId, details);
            emitTtsCallback("utterance_start", utterance.turnId, details);
            if (turn.firstStart.compareAndSet(false, true)) {
                JSONObject proxyDetails = utterance.details();
                try {
                    proxyDetails.put("proxy", "UtteranceProgressListener.onStart");
                } catch (Exception impossible) {
                    throw new AssertionError(impossible);
                }
                recordNativeTelemetry(
                        "native_tts_first_audio_start_proxy",
                        utterance.turnId,
                        proxyDetails
                );
                emitTtsCallback("first_audio_start_proxy", utterance.turnId, proxyDetails);
            }
        }

        @Override
        public void onDone(String utteranceId) {
            postToWebView(() -> handleDoneOnUi(utteranceId));
        }

        private void handleDoneOnUi(String utteranceId) {
            TtsUtterance utterance = ttsUtterances.remove(utteranceId);
            if (utterance == null || released) {
                return;
            }
            TtsTurnState turn = ttsTurns.get(utterance.turnId);
            if (turn == null
                    || turn.generation != utterance.generation
                    || turn.terminal.get()) {
                return;
            }
            int pending = Math.max(0, turn.pending.decrementAndGet());
            JSONObject details = utterance.details();
            try {
                details.put("pending_utterances", pending);
                details.put("queue_closed", turn.queueClosed.get());
            } catch (Exception impossible) {
                throw new AssertionError(impossible);
            }
            recordNativeTelemetry("native_tts_utterance_done", utterance.turnId, details);
            emitTtsCallback("utterance_done", utterance.turnId, details);
            if (pending == 0 && turn.queueClosed.get()) {
                completeTtsTurnIfDrainedOnUi(utterance.turnId, utterance.generation);
            }
        }

        @Override
        public void onError(String utteranceId) {
            onError(utteranceId, TextToSpeech.ERROR);
        }

        @Override
        public void onError(String utteranceId, int errorCode) {
            postToWebView(() -> handleErrorOnUi(utteranceId, errorCode, null));
        }

        private void handleErrorOnUi(
                String utteranceId,
                int errorCode,
                String nativeReason
        ) {
            TtsUtterance utterance = ttsUtterances.remove(utteranceId);
            if (utterance == null || released) {
                return;
            }
            TtsTurnState turn = ttsTurns.get(utterance.turnId);
            int pending = 0;
            if (turn != null && turn.generation == utterance.generation) {
                pending = turn.pending.updateAndGet(value -> Math.max(0, value - 1));
            }
            JSONObject details = utterance.details();
            try {
                details.put("pending_utterances", pending);
                details.put("terminal", true);
                details.put("stage", "playback");
                details.put("error_code", errorCode);
                if (nativeReason != null) {
                    details.put("native_reason", nativeReason);
                }
            } catch (Exception impossible) {
                throw new AssertionError(impossible);
            }
            recordNativeTelemetry("native_tts_utterance_error", utterance.turnId, details);
            emitTtsCallback("utterance_error", utterance.turnId, details);
            failTtsTurnOnUi(
                    utterance.turnId,
                    utterance.generation,
                    "playback",
                    nativeReason == null
                            ? "Android TTS failed while playing a speech chunk."
                            : "Offline voice failed while playing a speech chunk: "
                            + nativeReason
            );
        }
    }

    private static final class TtsTurnState {
        final String turnId;
        final String localeTag;
        final long generation;
        final String policyId;
        final String engineId;
        final AtomicBoolean firstStart = new AtomicBoolean();
        final AtomicBoolean queueClosed = new AtomicBoolean();
        final AtomicBoolean terminal = new AtomicBoolean();
        final AtomicInteger pending = new AtomicInteger();
        final AtomicInteger queued = new AtomicInteger();

        TtsTurnState(
                String turnId,
                String localeTag,
                long generation,
                String policyId,
                String engineId
        ) {
            this.turnId = turnId;
            this.localeTag = localeTag;
            this.generation = generation;
            this.policyId = policyId;
            this.engineId = engineId;
        }

        boolean usesSherpaTts() {
            return engineId != null && !"system".equals(engineId);
        }
    }

    private static final class TtsUtterance {
        final String utteranceId;
        final String turnId;
        final String chunkId;
        final int chunkIndex;
        final String localeTag;
        final long generation;
        final int characterCount;
        final String policyId;
        final String engineId;

        TtsUtterance(
                String utteranceId,
                String turnId,
                String chunkId,
                int chunkIndex,
                String localeTag,
                long generation,
                int characterCount,
                String policyId,
                String engineId
        ) {
            this.utteranceId = utteranceId;
            this.turnId = turnId;
            this.chunkId = chunkId;
            this.chunkIndex = chunkIndex;
            this.localeTag = localeTag;
            this.generation = generation;
            this.characterCount = characterCount;
            this.policyId = policyId;
            this.engineId = engineId;
        }

        JSONObject details() {
            JSONObject details = new JSONObject();
            try {
                details.put("chunk_id", chunkId);
                details.put("chunk_index", chunkIndex);
                details.put("locale", localeTag);
                details.put("generation", generation);
                details.put("character_count", characterCount);
                details.put("tts_policy", policyId);
                details.put("tts_engine_id", engineId);
            } catch (Exception impossible) {
                throw new AssertionError(impossible);
            }
            return details;
        }
    }

    private boolean appendTelemetry(JSONObject event) {
        byte[] line = (event.toString() + "\n").getBytes(StandardCharsets.UTF_8);
        synchronized (telemetryLock) {
            try (FileOutputStream output = context.openFileOutput(
                    TELEMETRY_FILE,
                    Context.MODE_APPEND
            )) {
                output.write(line);
                output.flush();
                return true;
            } catch (IOException error) {
                Log.e(TAG, "Could not append telemetry", error);
                return false;
            }
        }
    }

    static byte[] pcm16ToLittleEndian(short[] samples) {
        byte[] bytes = new byte[samples.length * 2];
        for (int index = 0; index < samples.length; index++) {
            int sample = samples[index];
            bytes[index * 2] = (byte) (sample & 0xff);
            bytes[index * 2 + 1] = (byte) ((sample >>> 8) & 0xff);
        }
        return bytes;
    }

    static final class PcmRingBuffer {
        private final short[] samples;
        private int firstIndex;
        private int size;
        private long firstSampleOffset;
        private long nextSampleOffset;

        PcmRingBuffer(int capacitySamples) {
            if (capacitySamples <= 0) {
                throw new IllegalArgumentException("capacitySamples must be positive");
            }
            samples = new short[capacitySamples];
        }

        void reset(long nextOffset) {
            if (nextOffset < 0) {
                throw new IllegalArgumentException("nextOffset must be non-negative");
            }
            firstIndex = 0;
            size = 0;
            firstSampleOffset = nextOffset;
            nextSampleOffset = nextOffset;
        }

        void append(long sampleOffset, short[] source, int offset, int length) {
            if (source == null || offset < 0 || length < 0 || offset > source.length - length) {
                throw new IllegalArgumentException("invalid PCM source range");
            }
            if (sampleOffset != nextSampleOffset) {
                throw new IllegalArgumentException(
                        "non-contiguous PCM: expected "
                                + nextSampleOffset
                                + " but received "
                                + sampleOffset
                );
            }
            for (int index = 0; index < length; index++) {
                if (size < samples.length) {
                    int writeIndex = (firstIndex + size) % samples.length;
                    samples[writeIndex] = source[offset + index];
                    size += 1;
                } else {
                    samples[firstIndex] = source[offset + index];
                    firstIndex = (firstIndex + 1) % samples.length;
                    firstSampleOffset += 1;
                }
                nextSampleOffset += 1;
            }
        }

        short[] slice(long startSample, long endSample) {
            if (startSample < firstSampleOffset
                    || endSample < startSample
                    || endSample > nextSampleOffset) {
                throw new IllegalArgumentException(
                        "PCM slice ["
                                + startSample
                                + ", "
                                + endSample
                                + ") is outside retained ["
                                + firstSampleOffset
                                + ", "
                                + nextSampleOffset
                                + ")"
                );
            }
            long requested = endSample - startSample;
            if (requested > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("PCM slice is too large");
            }
            short[] result = new short[(int) requested];
            long relative = startSample - firstSampleOffset;
            int readIndex = (int) ((firstIndex + relative) % samples.length);
            for (int index = 0; index < result.length; index++) {
                result[index] = samples[readIndex];
                readIndex = (readIndex + 1) % samples.length;
            }
            return result;
        }

        long getFirstSampleOffset() {
            return firstSampleOffset;
        }

        long getNextSampleOffset() {
            return nextSampleOffset;
        }

        int capacitySamples() {
            return samples.length;
        }
    }

    private static byte[] makeWav(byte[] pcm) {
        ByteArrayOutputStream output = new ByteArrayOutputStream(44 + pcm.length);
        try {
            writeAscii(output, "RIFF");
            writeLe32(output, 36 + pcm.length);
            writeAscii(output, "WAVEfmt ");
            writeLe32(output, 16);
            writeLe16(output, 1);
            writeLe16(output, 1);
            writeLe32(output, SAMPLE_RATE);
            writeLe32(output, SAMPLE_RATE * 2);
            writeLe16(output, 2);
            writeLe16(output, 16);
            writeAscii(output, "data");
            writeLe32(output, pcm.length);
            output.write(pcm);
        } catch (IOException impossible) {
            throw new AssertionError(impossible);
        }
        return output.toByteArray();
    }

    private static void writeAscii(ByteArrayOutputStream output, String text) throws IOException {
        output.write(text.getBytes(StandardCharsets.US_ASCII));
    }

    private static void writeLe16(ByteArrayOutputStream output, int value) {
        output.write(value & 0xff);
        output.write((value >>> 8) & 0xff);
    }

    private static void writeLe32(ByteArrayOutputStream output, int value) {
        output.write(value & 0xff);
        output.write((value >>> 8) & 0xff);
        output.write((value >>> 16) & 0xff);
        output.write((value >>> 24) & 0xff);
    }
}
