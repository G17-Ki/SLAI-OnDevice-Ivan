#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const html = fs.readFileSync(
  new URL("../android-app/app/src/main/assets/demo-ui/index.html", import.meta.url),
  "utf8"
);
const serviceWorker = fs.readFileSync(
  new URL("../android-app/app/src/main/assets/demo-ui/service-worker.js", import.meta.url),
  "utf8"
);
const match = html.match(/<script>([\s\S]*)<\/script>/);
assert(match, "embedded UI script was not found");
assert.match(serviceWorker, /url\.origin !== self\.location\.origin/);
assert.match(serviceWorker, /if \(isNavigation\) return caches\.match\("\/index\.html"\)/);
assert.doesNotMatch(serviceWorker, /response \|\| caches\.match\("\/index\.html"\)/);

const scheduledStarts = [];
const fakeAudioSources = [];
class FakeAudioContext {
  constructor() {
    this.currentTime = 1;
    this.destination = {};
    this.state = "running";
  }

  async resume() {
    this.state = "running";
  }

  createBuffer(_channels, length, sampleRate) {
    return {
      duration: length / sampleRate,
      copyToChannel(samples) {
        assert.equal(samples.length, length);
      }
    };
  }

  createBufferSource() {
    const listeners = new Map();
    const source = {
      buffer: null,
      stopped: false,
      connect() {},
      start(at) { scheduledStarts.push(at); },
      stop() { this.stopped = true; },
      addEventListener(type, callback) { listeners.set(type, callback); },
      dispatch(type) { listeners.get(type)?.(); }
    };
    fakeAudioSources.push(source);
    return source;
  }
}

function makeElement(id) {
  const listeners = new Map();
  const attributes = new Map();
  const classes = new Set();
  const element = {
    id,
    textContent: "",
    disabled: false,
    hidden: false,
    value: "",
    src: "",
    paused: true,
    ended: false,
    clientWidth: 320,
    clientHeight: 180,
    children: [],
    classList: {
      add(...names) { for (const name of names) classes.add(name); },
      remove(...names) { for (const name of names) classes.delete(name); },
      contains(name) { return classes.has(name); }
    },
    setAttribute(name, value) { attributes.set(name, String(value)); },
    getAttribute(name) { return attributes.get(name) ?? null; },
    removeAttribute(name) { attributes.delete(name); },
    addEventListener(type, callback) {
      const callbacks = listeners.get(type) || [];
      callbacks.push(callback);
      listeners.set(type, callbacks);
    },
    removeEventListener(type, callback) {
      const callbacks = listeners.get(type) || [];
      listeners.set(type, callbacks.filter(candidate => candidate !== callback));
    },
    dispatchEvent(event) {
      for (const callback of listeners.get(event.type) || []) callback.call(this, event);
      return true;
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    click() { this.dispatchEvent({ type: "click" }); },
    pause() { this.paused = true; },
    play: async function () { this.paused = false; this.ended = false; },
    getBoundingClientRect: () => ({ width: 320, height: 180 })
  };
  if (id === "waveform") {
    element.getContext = () => ({
      setTransform() {},
      clearRect() {},
      beginPath() {},
      moveTo() {},
      lineTo() {},
      stroke() {},
      get strokeStyle() { return ""; },
      set strokeStyle(_) {},
      get lineWidth() { return 1; },
      set lineWidth(_) {}
    });
  }
  return element;
}

const elements = new Map();
const telemetryPayloads = [];
const nativeTtsCalls = [];
const nativeTtsPolicyCalls = [];
const nativeTtsLocaleChecks = [];
const nativeTtsPreloadCalls = [];
const nativeVadCalls = [];
let autoCompleteNativeTts = false;
let nativeTtsPolicy = "auto";
const nativeTtsEngines = [
  {
    id: "auto",
    label: "Automatic by language",
    language: "any",
    family: "language-aware routing",
    available: true
  },
  {
    id: "system",
    label: "Offline eSpeak NG (control)",
    language: "any",
    family: "formant synthesis",
    available: true
  },
  {
    id: "matcha_zh_baker",
    label: "Matcha Baker",
    language: "zh",
    family: "matcha",
    available: true
  },
  {
    id: "supertonic3_s5",
    label: "Supertonic 3 s5",
    language: "ru",
    family: "supertonic",
    available: true
  }
];
const storageValues = new Map();
globalThis.localStorage = {
  getItem(key) { return storageValues.has(key) ? storageValues.get(key) : null; },
  setItem(key, value) { storageValues.set(String(key), String(value)); },
  removeItem(key) { storageValues.delete(String(key)); },
  clear() { storageValues.clear(); }
};
globalThis.document = {
  documentElement: {},
  createElement(tagName) {
    return makeElement(String(tagName));
  },
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  }
};
globalThis.window = {
  devicePixelRatio: 1,
  addEventListener() {},
  NativeAudio: {
    lfmRuntimeEnabled: () => true,
    monotonicNowMs: () => 1234.5,
    defaultTtsLocale: () => "en-US",
    hasTtsEngine: () => true,
    hasTtsEngineForLocale(localeTag) {
      nativeTtsLocaleChecks.push(localeTag);
      return true;
    },
    listTtsEngines() {
      return JSON.stringify({ selected: nativeTtsPolicy, engines: nativeTtsEngines });
    },
    setTtsEngine(policy) {
      nativeTtsPolicyCalls.push(policy);
      if (!nativeTtsEngines.some(engine => engine.id === policy && engine.available !== false)) return false;
      nativeTtsPolicy = policy;
      return true;
    },
    preloadTts(locale, requestId) {
      nativeTtsPreloadCalls.push({ locale, requestId, policy: nativeTtsPolicy });
      queueMicrotask(() => window.onNativeTtsPreloadEvent(JSON.stringify({
        schema_version: 1,
        type: "ready",
        request_id: requestId,
        monotonic_ms: 1238,
        details: {
          requested_locale: locale,
          resolved_locale: locale,
          tts_policy: nativeTtsPolicy,
          tts_engine_id: nativeTtsPolicy
        }
      })));
      return true;
    },
    recordTelemetry(payload) {
      telemetryPayloads.push(JSON.parse(payload));
      return true;
    },
    startRecording() {
      return true;
    },
    stopRecording() {},
    startVadConversation() {
      nativeVadCalls.push({ method: "start" });
      return true;
    },
    resumeVadConversation() {
      nativeVadCalls.push({ method: "resume" });
      return true;
    },
    stopVadConversation() {
      nativeVadCalls.push({ method: "stop" });
    },
    setVadAssistantOutputActive(active) {
      nativeVadCalls.push({ method: "output_gate", active });
    },
    initializeTts(locale, turnId, generation) {
      nativeTtsCalls.push({ method: "initialize", locale, turnId, generation });
      if (autoCompleteNativeTts) queueMicrotask(() => window.onNativeTtsEvent(JSON.stringify({
        type: "ready",
        turn_id: turnId,
        monotonic_ms: 1240,
        details: {
          generation,
          resolved_locale: locale,
          network_connection_required: false,
          tts_policy: nativeTtsPolicy,
          tts_engine_id: nativeTtsPolicy === "auto"
            ? (locale === "zh-CN" ? "matcha_zh_baker"
              : (locale === "ru-RU" ? "supertonic3_s5" : "system"))
            : nativeTtsPolicy
        }
      })));
    },
    queueTtsChunk(text, turnId, chunkId, generation) {
      nativeTtsCalls.push({ method: "queue", text, turnId, chunkId, generation });
      if (autoCompleteNativeTts) queueMicrotask(() => {
        window.onNativeTtsEvent(JSON.stringify({
          type: "chunk_queued",
          turn_id: turnId,
          monotonic_ms: 1250,
          details: { generation, chunk_id: chunkId, chunk_index: 0 }
        }));
        window.onNativeTtsEvent(JSON.stringify({
          type: "first_audio_start_proxy",
          turn_id: turnId,
          monotonic_ms: 1260,
          details: { generation }
        }));
      });
    },
    finishTtsResponse(turnId, generation) {
      nativeTtsCalls.push({ method: "finish", turnId, generation });
      if (autoCompleteNativeTts) queueMicrotask(() => {
        window.onNativeTtsEvent(JSON.stringify({
          type: "queue_closed",
          turn_id: turnId,
          monotonic_ms: 1270,
          details: { generation }
        }));
        window.onNativeTtsEvent(JSON.stringify({
          type: "terminal",
          turn_id: turnId,
          monotonic_ms: 1280,
          details: { generation, status: "completed" }
        }));
      });
    },
    cancelTts(turnId, generation, reason) {
      nativeTtsCalls.push({ method: "cancel", turnId, generation, reason });
    }
  }
};
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {}
});
globalThis.AudioContext = FakeAudioContext;
globalThis.OfflineAudioContext = FakeAudioContext;
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#000" });
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.setTimeout = () => 0;
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
globalThis.fetch = async () => { throw new Error("test offline"); };
globalThis.URL.createObjectURL = () => "blob:test";
globalThis.URL.revokeObjectURL = () => {};

const exposed = `
globalThis.__lfmUiTest = {
  base64ToFloat32,
  base64Int16ToFloat32,
  decodeAudioDelta,
  queueAudioChunk,
  stopStreamingPlayback,
  streamingPlaybackDrainTimeoutMs,
  waitForStreamingPlaybackDrain,
  waitForAudioElementDrain,
  waitForVadAssistantOutput,
  scheduleManualTurnTerminal,
  getPlaybackContext,
  mapContextTimeToPerformanceMs,
  getOutputHeadCrossing,
  observeOutputHeadCrossing,
  newTurnId,
  nativeMonotonicMs,
  emitTelemetry,
  decodedBase64ByteLength,
  base64ToBytes,
  inspectCanonicalPcm16Wav,
  encodeCanonicalPcm16Wav,
  segmentCanonicalWav,
  mergeSegmentTranscripts,
  SUPPORTED_LANGUAGES,
  HISTORY_LIMITS,
  approximateTextTokens,
  classifyLanguageEvidence,
  resolveLanguageSelection,
  detectTtsLocale,
  resolveTtsLocale,
  nativeTtsEngineAvailable,
  currentTtsPolicy,
  loadStoredLanguageSelection,
  storeLanguageSelection,
  loadStoredTtsPolicy,
  refreshTtsEngines,
  applyTtsEngine,
  preloadTtsModelForLanguage,
  prepareAsrModelForLanguage,
  warmTextModelForLanguage,
  warmSelectedRoute,
  asrRouteForSelection,
  selectedRouteWarmState,
  updateTtsPolicyControlState,
  createConversationHistory,
  conversationHistory,
  conversationContextDecision,
  setHistoryEnabled,
  isHistoryEnabled,
  asrSystemPrompt,
  conversationSystemPrompt,
  buildConversationUserText,
  buildAsrPayload,
  buildConversationPayload,
  buildLfmTtsPayload,
  buildDirectLfmConversationPayload,
  resolveEnglishResponseStyle,
  resolveLfmTtsVoice,
  naturalTtsAudioCheck,
  firstVoicedSampleOffset,
  streamLfmTtsChunk,
  createReasoningStreamFilter,
  parseSseEvent,
  normalizeAsrTranscript,
  scriptEvidence,
  senseVoiceLanguageHint,
  meanTokenLogProbability,
  normalizeCompanionCandidate,
  selectCompanionAsr,
  mergeCompanionPayloads,
  plannedAsrFacts,
  measureTextRepetition,
  responseQualityWarnings,
  ttsChunkScriptCheck,
  preflightTtsChunks,
  streamingResponseCollapse,
  phoneticTranscriptBlockReason,
  commitConversationTurn,
  languageTelemetryDetails,
  resolveVoiceDelivery,
  extractSafeTtsChunks,
  extractNaturalTtsChunks,
  packNaturalTtsTailChunks,
  parseNativeTtsEvent,
  parseNativeVadEvent,
  createVadTtsTerminalPromise,
  settleVadTtsTerminal,
  ttsSystemPrompt,
  abortError,
  isAbortError,
  createTurnCoordinator,
  createTimedWaiter,
  initializeNativeTts,
  queueNativeTtsChunk,
  cancelNativeTts,
  createTtsClientState,
  ttsClientTurns,
  turnCoordinator,
  VAD_BACKEND_FACTS,
  vadState,
  npuFailClosedFacts,
  vadTelemetryDetails,
  setInputBusy,
  setMode,
  setInputMode,
  startVadSession,
  stopVadSession,
  processVadEndpoint,
  startRecording,
  stopRecording
};`;
vm.runInThisContext(match[1] + exposed, { filename: "demo-ui/index.html" });

const api = globalThis.__lfmUiTest;

const firstTurnId = api.newTurnId();
const secondTurnId = api.newTurnId();
assert.notEqual(firstTurnId, secondTurnId);
assert.equal(api.nativeMonotonicMs(), 1234.5);
const telemetry = api.emitTelemetry(firstTurnId, "unit_test", { answer: 42 });
assert.equal(telemetry.turn_id, firstTurnId);
assert.equal(telemetry.event, "unit_test");
assert.equal(telemetry.monotonic_ms, 1234.5);
assert.equal(telemetry.input_mode, "push");
assert.deepEqual(telemetry.details, { answer: 42 });
assert.deepEqual(telemetryPayloads.at(-1), telemetry);

const ttsEngineSelectElement = elements.get("tts-engine-select");
const initialLanguageSelectElement = elements.get("language-select");
assert.equal(initialLanguageSelectElement.value, "en", "fresh installs warm an explicit route");
assert.equal(localStorage.getItem("lfm.language.selection.v1"), "en");
assert.equal(api.asrRouteForSelection("en"), "sensevoice_qnn:en");
assert.equal(api.asrRouteForSelection("ru"), "zipformer_ru_cpu");
assert.equal(api.asrRouteForSelection("fr"), "parakeet_v3_cpu");
assert.equal(api.currentTtsPolicy(), "auto", "locale-aware offline routing is the default policy");
assert.equal(ttsEngineSelectElement.value, "auto");
assert.equal(localStorage.getItem("lfm.tts.policy.v2"), "auto");
assert(nativeTtsPolicyCalls.includes("auto"));
assert.match(elements.get("tts-engine-status").textContent, /Automatic.*best installed offline voice.*eSpeak fallback/i);
assert.equal(await api.preloadTtsModelForLanguage("auto"), true);
assert.equal(nativeTtsPreloadCalls.length, 0, "automatic language cannot safely choose one TTS graph early");
assert.equal(await api.preloadTtsModelForLanguage("ru"), true);
assert.equal(nativeTtsPreloadCalls.length, 1);
assert.equal(nativeTtsPreloadCalls[0].locale, "ru-RU");
assert.equal(nativeTtsPreloadCalls[0].policy, "auto");
assert.equal(await api.preloadTtsModelForLanguage("ru"), true);
assert.equal(nativeTtsPreloadCalls.length, 1, "an already resident policy/locale graph is reused");
const asrPrepareCalls = [];
const offlineFetch = globalThis.fetch;
globalThis.fetch = async (url, options = {}) => {
  asrPrepareCalls.push({ url: String(url), method: options.method });
  return {
    ok: true,
    status: 200,
    async json() {
      return {
        ok: true,
        resident: true,
        route: String(url).includes("language=es")
          ? "parakeet_v3_cpu" : "sensevoice_qnn:en",
        load_ms: 12,
        reused: false,
        runner_pid: 42
      };
    }
  };
};
assert.equal(await api.prepareAsrModelForLanguage("es"), true);
assert.match(asrPrepareCalls[0].url, /\/prepare\?language=es$/);
assert.equal(asrPrepareCalls[0].method, "POST");
assert.equal(await api.prepareAsrModelForLanguage("es"), true);
assert.equal(asrPrepareCalls.length, 1, "the selected resident ASR route is not reloaded");
globalThis.fetch = offlineFetch;
assert.match(html, /id="mode-tts"[^>]*>Offline voice</);
assert.doesNotMatch(html, />System voice</i);

localStorage.clear();
assert.equal(api.loadStoredTtsPolicy("system"), "auto", "a fresh install defaults to auto");
localStorage.setItem("lfm.tts.engine", "system");
api.refreshTtsEngines();
assert.equal(api.currentTtsPolicy(), "auto", "the legacy implicit system default migrates to auto");
assert.equal(localStorage.getItem("lfm.tts.policy.v2"), "auto");

localStorage.clear();
localStorage.setItem("lfm.tts.engine", "matcha_zh_baker");
api.refreshTtsEngines();
assert.equal(api.currentTtsPolicy(), "matcha_zh_baker", "a legacy concrete voice remains pinned");
assert.equal(localStorage.getItem("lfm.tts.policy.v2"), "matcha_zh_baker");

ttsEngineSelectElement.value = "supertonic3_s5";
ttsEngineSelectElement.dispatchEvent({ type: "change" });
assert.equal(api.currentTtsPolicy(), "supertonic3_s5");
assert.equal(localStorage.getItem("lfm.tts.policy.v2"), "supertonic3_s5");
assert.equal(telemetryPayloads.at(-1).event, "tts_policy_selection_changed");
assert.equal(telemetryPayloads.at(-1).details.tts_policy, "supertonic3_s5");
assert.equal(telemetryPayloads.at(-1).details.resolved_tts_engine, null);
assert.equal(await api.preloadTtsModelForLanguage("ru"), true);
assert.equal(nativeTtsPreloadCalls.length, 2, "a policy change invalidates the prior preload key");
assert.equal(nativeTtsPreloadCalls.at(-1).policy, "supertonic3_s5");

ttsEngineSelectElement.value = "auto";
ttsEngineSelectElement.dispatchEvent({ type: "change" });
assert.equal(api.currentTtsPolicy(), "auto");
api.setInputBusy(true);
assert.equal(ttsEngineSelectElement.disabled, true, "voice policy is locked while input is busy");
api.setInputBusy(false);
assert.equal(ttsEngineSelectElement.disabled, false);
const playbackLockTurn = api.turnCoordinator.begin("chat", "file");
assert.equal(ttsEngineSelectElement.disabled, true, "voice policy remains locked for the active playback turn");
api.turnCoordinator.markTerminal(playbackLockTurn);
api.turnCoordinator.clearIfCurrent(playbackLockTurn);
assert.equal(ttsEngineSelectElement.disabled, false);
elements.get("audio").paused = false;
elements.get("audio").ended = false;
elements.get("audio").dispatchEvent({ type: "play" });
assert.equal(ttsEngineSelectElement.disabled, true, "voice policy is locked during audio replay");
elements.get("audio").paused = true;
elements.get("audio").ended = true;
elements.get("audio").dispatchEvent({ type: "ended" });
assert.equal(ttsEngineSelectElement.disabled, false);

assert.equal(api.detectTtsLocale("巴黎在法国。"), "zh-CN");
assert.equal(api.detectTtsLocale("日本語で答えてください。"), "ja-JP");
assert.equal(api.detectTtsLocale("한국어로 답하세요."), "ko-KR");
assert.equal(api.detectTtsLocale("Привет, расскажи о погоде."), "ru-RU");
assert.equal(api.detectTtsLocale("Hola, gracias por la ayuda."), "es-ES");
assert.equal(api.detectTtsLocale("Bonjour et merci pour la réponse."), "fr-FR");
assert.equal(api.detectTtsLocale("Ciao, grazie per la risposta."), "it-IT");
assert.equal(api.detectTtsLocale("Hello, please explain the answer."), "en-US");
assert.equal(api.detectTtsLocale("la"), null, "one shared token is deliberately ambiguous");
assert.equal(api.resolveTtsLocale("ru", "ignored", "en-US"), "ru-RU");
assert.equal(api.resolveTtsLocale("auto", "这是中文。", "en-US"), "zh-CN");
assert.equal(api.resolveTtsLocale("auto", "Neutral proper nouns only", "it-IT"), "it-IT");

const explicitFrench = api.resolveLanguageSelection("fr", "hello", "en");
assert.deepEqual(explicitFrench, {
  selection: "fr",
  language: "fr",
  localeTag: "fr-FR",
  confidence: 1,
  reason: "explicit user selection",
  fallbackUsed: false
});
for (const [selection, [localeTag, languageName]] of Object.entries({
  en: ["en-US", "English"],
  zh: ["zh-CN", "Mandarin Chinese"],
  ja: ["ja-JP", "Japanese"],
  ko: ["ko-KR", "Korean"],
  ru: ["ru-RU", "Russian"],
  fr: ["fr-FR", "French"],
  it: ["it-IT", "Italian"],
  es: ["es-ES", "Spanish"]
})) {
  const route = api.resolveLanguageSelection(selection, "contradictory input", "en");
  assert.equal(route.language, selection);
  assert.equal(route.localeTag, localeTag);
  assert.equal(route.confidence, 1);
  assert.equal(api.asrSystemPrompt(selection), "Perform ASR.");
  assert.match(api.ttsSystemPrompt(route), /Answer the current user directly and accurately/);
}
assert.equal(api.nativeTtsEngineAvailable(), true);
assert.equal(api.nativeTtsEngineAvailable("zh-CN"), true);
assert.equal(nativeTtsLocaleChecks.at(-1), "zh-CN", "the route gate asks for the resolved locale");
assert.equal(api.resolveLanguageSelection("auto", "Ni Hao, Qing Gao Zuo Zintian Tianzi.").language, "zh");
assert.equal(api.resolveLanguageSelection("auto", "Priviat pajaosta reskaje apagodi siwodna.").language, "ru");
const autoChinese = api.resolveLanguageSelection("auto", "请告诉我今天的天气。", null);
assert.equal(autoChinese.language, "zh");
assert(autoChinese.confidence >= 0.93);
assert.equal(autoChinese.fallbackUsed, false);
assert.equal(api.resolveLanguageSelection("auto", "日本語で答えてください。", null).language, "ja");
assert.equal(api.resolveLanguageSelection("auto", "한국어로 답하세요.", null).language, "ko");
const priorFallback = api.resolveLanguageSelection("auto", "yes", "it");
assert.equal(priorFallback.language, "it");
assert.equal(priorFallback.fallbackUsed, true);
assert(priorFallback.confidence < 0.5);
assert.match(priorFallback.reason, /continuing prior Italian/);
const coldFallback = api.resolveLanguageSelection("auto", "OK", null);
assert.equal(coldFallback.language, "en");
assert.equal(coldFallback.fallbackUsed, true);
assert.match(coldFallback.reason, /conservative English fallback/);

assert.deepEqual(api.extractSafeTtsChunks("Hello world."), {
  chunks: [],
  remainder: "Hello world."
});
assert.deepEqual(api.extractSafeTtsChunks("Hello world. Next"), {
  chunks: ["Hello world."],
  remainder: "Next"
});
assert.deepEqual(api.extractSafeTtsChunks("第一句。第二句！余り"), {
  chunks: ["第一句。", "第二句！"],
  remainder: "余り"
});
assert.deepEqual(api.extractSafeTtsChunks("A short final clause", true), {
  chunks: ["A short final clause"],
  remainder: ""
});
assert.deepEqual(api.extractSafeTtsChunks('The phrase is "Stay curious."', true), {
  chunks: ['The phrase is "Stay curious."'],
  remainder: ""
});
assert.deepEqual(
  api.extractSafeTtsChunks("This clause is deliberately long enough, and the remainder waits", false, { commaMinChars: 20 }),
  { chunks: ["This clause is deliberately long enough,"], remainder: "and the remainder waits" }
);
const boundedChunk = api.extractSafeTtsChunks(
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron remainder"
);
assert.equal(boundedChunk.chunks.length, 1);
assert(boundedChunk.chunks[0].length >= 50 && boundedChunk.chunks[0].length <= 70);
assert.deepEqual(
  api.extractNaturalTtsChunks(
    "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. Next sentence"
  ),
  {
    chunks: ["Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy."],
    remainder: "Next sentence"
  }
);
assert.deepEqual(
  api.extractNaturalTtsChunks(
    "This clause is deliberately long enough, but it should keep its comma and continue naturally until the complete sentence arrives."
  ),
  {
    chunks: [],
    remainder: "This clause is deliberately long enough, but it should keep its comma and continue naturally until the complete sentence arrives."
  }
);
const boundedNaturalChunk = api.extractNaturalTtsChunks(
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega and more words continue here"
);
assert.equal(boundedNaturalChunk.chunks.length, 1);
assert(boundedNaturalChunk.chunks[0].length >= 100 && boundedNaturalChunk.chunks[0].length <= 140);
assert.deepEqual(
  api.packNaturalTtsTailChunks(
    "Plants use sunlight, water, and carbon dioxide to produce glucose through a series of reactions. The glucose is then used for energy and growth. Finally, plants release oxygen as a byproduct of this process."
  ),
  [
    "Plants use sunlight, water, and carbon dioxide to produce glucose through a series of reactions. The glucose is then used for energy and growth. Finally, plants release oxygen as a byproduct of this process."
  ]
);
const packedLongTail = api.packNaturalTtsTailChunks(
  `${"A bounded natural sentence with enough words to exercise packing. ".repeat(10)}`
);
assert(packedLongTail.length >= 2);
assert(packedLongTail.every(chunk => chunk.length <= 420));
assert.deepEqual(api.parseNativeTtsEvent('{"type":"ready","turn_id":"1"}'), {
  type: "ready",
  turn_id: "1"
});
assert.equal(api.parseNativeTtsEvent("not-json"), null);
assert.deepEqual(api.parseNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "listening",
  session_id: 42,
  monotonic_ms: 1500,
  details: { state: "LISTENING" }
})), {
  schema_version: 1,
  type: "listening",
  session_id: "42",
  monotonic_ms: 1500,
  details: { state: "LISTENING" }
});
assert.equal(api.parseNativeVadEvent('{"schema_version":2,"type":"listening"}'), null);
assert.equal(api.parseNativeVadEvent("not-json"), null);
assert.match(api.ttsSystemPrompt("ru"), /Answer the current user directly and accurately/);
assert.match(api.ttsSystemPrompt("auto", "Bonjour merci"), /requested language/);
assert.equal(api.detectTtsLocale("東京"), "zh-CN", "supported-language routing treats Han as Mandarin");

for (const locale of ["auto", "en", "zh", "ja", "ko", "ru", "fr", "it", "es"]) {
  assert.match(html, new RegExp(`<option value="${locale}">`));
}
for (const unsupported of ["de"]) {
  assert.doesNotMatch(html, new RegExp(`<option value="${unsupported}">`));
}
assert.match(html, /id="input-push"[^>]*>Push-to-talk</);
assert.match(html, /id="input-vad"[^>]*>Hands-free VAD</);
assert.match(html, /id="vad-status"[^>]*>Push-to-talk · Java CPU VAD idle · barge-in off</);
const historySwitchMarkup = html.match(/<button\b[^>]*\bid="history-toggle"[^>]*>/)?.[0];
assert(historySwitchMarkup, "context switch markup was not found");
assert.match(historySwitchMarkup, /\brole="switch"/);
assert.match(historySwitchMarkup, /\baria-checked="false"/);
assert.equal(api.isHistoryEnabled(), false, "conversation memory must be opt-in");
assert.equal(elements.get("history-toggle").getAttribute("aria-checked"), "false");
assert.match(elements.get("history-status").textContent, /Memory off.*no prior turns/i);
assert.match(html, /window\.onNativeVadEvent = raw =>/);
assert.doesNotMatch(html, /onNativeVadRecording/);
assert.equal(api.VAD_BACKEND_FACTS.vad_backend, "adaptive_energy_java_cpu");
assert.equal(api.VAD_BACKEND_FACTS.inference_backend, "cpu_q4");
assert.equal(api.VAD_BACKEND_FACTS.npu_executed, false);
assert.equal(api.VAD_BACKEND_FACTS.barge_in_enabled, false);
assert.equal(
  api.npuFailClosedFacts("auto").npu_fallback_reason,
  "broad_auto_requires_cpu_multilingual_asr"
);
assert.equal(
  api.npuFailClosedFacts("zh", "zh").npu_fallback_reason,
  "app_owned_qnn_runtime_unavailable"
);
assert.equal(
  api.npuFailClosedFacts("ko", "ko").npu_fallback_reason,
  "app_owned_qnn_runtime_unavailable"
);
assert.equal(
  api.npuFailClosedFacts("fr", "fr").npu_fallback_reason,
  "language_not_supported_by_sensevoice_qnn"
);
assert.match(html, /const requestStartEvent = emitTelemetry\(turnId, "request_start"/);
assert.match(html, /requestStartedAt = requestStartEvent\.performance_ms/);
assert.match(html, /requestStartedNativeMs: requestStartEvent\.monotonic_ms/);

const nativeRecordingCallback = html.match(
  /window\.onNativeRecording = async encoded => \{([\s\S]*?)\n    \};/
);
assert(nativeRecordingCallback, "native recording callback was not found");
assert.doesNotMatch(nativeRecordingCallback[1], /\batob\b|\bbytesToBase64\b/);
for (const byteLength of [0, 1, 2, 3, 47, 48, 49]) {
  const encoded = Buffer.alloc(byteLength, 0xa5).toString("base64");
  assert.equal(api.decodedBase64ByteLength(encoded), byteLength);
  assert.deepEqual([...api.base64ToBytes(encoded)], [...Buffer.alloc(byteLength, 0xa5)]);
}

const shortCanonicalWav = api.encodeCanonicalPcm16Wav(new Uint8Array(16000 * 2));
const shortSegmentation = api.segmentCanonicalWav(shortCanonicalWav);
assert.equal(shortSegmentation.segmented, false);
assert.equal(shortSegmentation.segments.length, 1);
assert.equal(shortSegmentation.segments[0].bytes, shortCanonicalWav, "short native WAV handoff remains byte-for-byte");
const longCanonicalWav = api.encodeCanonicalPcm16Wav(new Uint8Array(Math.round(20.4 * 16000) * 2));
const longSegmentation = api.segmentCanonicalWav(longCanonicalWav);
assert.equal(longSegmentation.segmented, true);
assert(longSegmentation.segments.length >= 3);
assert(longSegmentation.segments.every(segment => segment.bytes.length <= 304044));
assert.equal(api.inspectCanonicalPcm16Wav(longSegmentation.segments.at(-1).bytes).sampleRate, 16000);
assert.equal(
  api.mergeSegmentTranscripts([
    "The quick brown fox jumps over",
    "fox jumps over the lazy dog."
  ], "en"),
  "The quick brown fox jumps over the lazy dog."
);
assert.equal(api.mergeSegmentTranscripts(["北京是中国", "中国的首都。"], "zh"), "北京是中国的首都。");
assert.throws(
  () => api.segmentCanonicalWav(api.encodeCanonicalPcm16Wav(new Uint8Array(Math.round(30.1 * 16000) * 2))),
  /up to 30 seconds/
);

const coordinatorCancels = [];
const fakeControllers = [];
const coordinator = api.createTurnCoordinator({
  initialGeneration: 5000,
  makeId: (() => { let value = 0; return () => `test-${++value}`; })(),
  makeController() {
    const controller = { aborted: false, abort() { this.aborted = true; } };
    fakeControllers.push(controller);
    return controller;
  },
  onCancel(turn, reason) { coordinatorCancels.push({ id: turn.id, reason }); }
});
const overlapA = coordinator.begin("tts", "file");
assert.equal(overlapA.generation, 5001, "generation epochs can survive a WebView reload without colliding with native TTS state");
let readerCancelled = false;
overlapA.reader = { cancel() { readerCancelled = true; return Promise.resolve(); } };
const overlapB = coordinator.begin("tts", "microphone");
assert.equal(overlapA.cancelled, true);
assert.equal(fakeControllers[0].aborted, true);
assert.equal(readerCancelled, true);
assert.equal(coordinator.current, overlapB);
assert.deepEqual(coordinatorCancels, [{ id: overlapA.id, reason: "replaced_by_new_input" }]);
assert.equal(coordinator.cancel(overlapA, "stale_cancel"), false);
assert.equal(coordinator.current, overlapB);

const boundedHistory = api.createConversationHistory();
for (let index = 1; index <= 7; index++) {
  boundedHistory.addTurn({
    user: `user turn ${index}`,
    assistant: `assistant turn ${index}`,
    language: index % 2 ? "en" : "fr"
  });
}
assert.equal(api.HISTORY_LIMITS.maxTurns, 2, "only two recent whole turns are retained");
assert.equal(boundedHistory.summary().turns, api.HISTORY_LIMITS.maxTurns);
assert.equal(boundedHistory.snapshot()[0].user, "user turn 6", "only the two newest complete turns are retained");
assert.equal(boundedHistory.latestLanguage(), "en");
assert.equal(boundedHistory.toMessages("en").length, 2);
assert.equal(boundedHistory.toMessages("fr").length, 2);
assert.deepEqual(boundedHistory.toMessages("zh"), [], "history from another language must not leak into a prompt");
assert(boundedHistory.toMessages("en").every(message => typeof message.content === "string"));
assert.throws(
  () => boundedHistory.addTurn({ user: { input_audio: "blob" }, assistant: "unsafe", language: "en" }),
  /must be text/
);

const multilingualHistory = api.createConversationHistory({
  ...api.HISTORY_LIMITS,
  maxTurns: 3
});
multilingualHistory.addTurn({ user: "Hello", assistant: "Hi", language: "en" });
multilingualHistory.addTurn({ user: "你好", assistant: "您好", language: "zh" });
multilingualHistory.addTurn({ user: "How are you?", assistant: "Well", language: "en" });
assert.deepEqual(multilingualHistory.toMessages("zh"), [
  { role: "user", content: "你好" },
  { role: "assistant", content: "您好" }
]);
assert.deepEqual(multilingualHistory.toMessages("ru"), []);

const characterBoundedHistory = api.createConversationHistory({
  maxTurns: 10,
  maxCharacters: 40,
  maxApproxTokens: 20,
  approximateCharactersPerToken: 4
});
const characterSummary = characterBoundedHistory.addTurn({
  user: "u".repeat(80),
  assistant: "a".repeat(80),
  language: "es"
});
assert.equal(characterSummary.retained, false);
assert.equal(characterSummary.reason, "oversize_turn");
assert.equal(characterSummary.characters, 0);
assert.equal(characterBoundedHistory.snapshot().length, 0, "oversized turns are discarded instead of clipped into fake completed messages");
const clearedHistory = boundedHistory.clear();
assert.equal(clearedHistory.removedTurns, api.HISTORY_LIMITS.maxTurns);
assert.deepEqual(boundedHistory.toMessages("en"), []);
const confidenceAwareHistory = api.createConversationHistory();
confidenceAwareHistory.addTurn({
  user: "ambiguous", assistant: "fallback", language: "ru",
  languageConfidence: 0.3, languageFallbackUsed: true
});
assert.equal(confidenceAwareHistory.latestLanguage(), null, "fallback routes do not become confident priors");
confidenceAwareHistory.addTurn({
  user: "bonjour merci", assistant: "avec plaisir", language: "fr",
  languageConfidence: 0.9, languageFallbackUsed: false
});
assert.equal(confidenceAwareHistory.latestLanguage(), "fr");

const englishHistoryMessages = [
  { role: "user", content: "The code phrase is Cobalt Tulip." },
  { role: "assistant", content: "I will remember Cobalt Tulip." }
];
assert.deepEqual(api.conversationContextDecision({
  enabled: false,
  transcript: "What was the code phrase?",
  language: "en",
  historyMessages: englishHistoryMessages
}), { inject: false, reason: "context_disabled", policy: "followup_only" });
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "What was the code phrase?",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, true, "explicit recall language injects retained history");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Do you remember the code phrase?",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, true, "an explicit English memory request injects retained history");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "What secret phrase did I ask you to remember?",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, true, "the physical recall wording injects retained history");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Remember the secret phrase Cobalt Tulip.",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, false, "a storage request must not inject unrelated old history");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "What is the capital of Canada?",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, false, "an unrelated factual question stays context-free");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Why is the sky blue?",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, false, "a standalone English why-question does not inherit unrelated context");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "那它为什么重要？",
  language: "zh",
  historyMessages: [{ role: "user", content: "光合作用是什么？" }, { role: "assistant", content: "植物利用光制造养分。" }]
}).inject, false, "a generic Chinese pronoun cannot silently inject memory");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "你还记得密码吗？",
  language: "zh",
  historyMessages: [{ role: "user", content: "密码是蓝色月亮。" }, { role: "assistant", content: "我记住了。" }]
}).inject, true, "an explicit Chinese memory request injects retained history");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "现在回答另一个问题：澳大利亚的首都是哪里？请简短回答。",
  language: "zh",
  historyMessages: [{ role: "user", content: "澳大利亚人口最多的城市是哪里？" }, { role: "assistant", content: "悉尼。" }]
}).inject, false, "the Chinese distractor control stays context-free");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "为什么天空是蓝色的？",
  language: "zh",
  historyMessages: [{ role: "user", content: "光合作用是什么？" }, { role: "assistant", content: "植物利用光制造养分。" }]
}).inject, false, "a standalone Chinese why-question does not inherit unrelated context");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Какой была кодовая фраза?",
  language: "ru",
  historyMessages: [{ role: "user", content: "Что такое фотосинтез?" }, { role: "assistant", content: "Это процесс у растений." }]
}).inject, true);
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Ты помнишь кодовую фразу?",
  language: "ru",
  historyMessages: [{ role: "user", content: "Кодовая фраза — синий месяц." }, { role: "assistant", content: "Я запомнил." }]
}).inject, true, "an explicit Russian memory request injects retained history");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Теперь другой вопрос: какова столица России? Ответь одним словом.",
  language: "ru",
  historyMessages: [{ role: "user", content: "Какова столица Украины?" }, { role: "assistant", content: "Киев." }]
}).inject, false, "the Russian distractor control stays context-free");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "Почему небо голубое?",
  language: "ru",
  historyMessages: [{ role: "user", content: "Что такое фотосинтез?" }, { role: "assistant", content: "Это процесс у растений." }]
}).inject, false, "a standalone Russian why-question does not inherit unrelated context");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "What about it?",
  language: "en",
  historyMessages: englishHistoryMessages
}).inject, false, "generic English pronouns cannot silently inject memory");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "那这个呢？",
  language: "zh",
  historyMessages: [{ role: "user", content: "密码是蓝色月亮。" }, { role: "assistant", content: "我记住了。" }]
}).inject, false, "generic Chinese connectors cannot silently inject memory");
assert.equal(api.conversationContextDecision({
  enabled: true,
  transcript: "А это?",
  language: "ru",
  historyMessages: [{ role: "user", content: "Кодовая фраза — синий месяц." }, { role: "assistant", content: "Я запомнил." }]
}).inject, false, "generic Russian pronouns cannot silently inject memory");

const englishTranscript = "What should I wear?";
const compactEnglishPrompt = api.buildConversationUserText({
  transcript: englishTranscript,
  languageRoute: api.resolveLanguageSelection("en")
});
assert.match(compactEnglishPrompt, /Answer only in concise, natural English/);
assert.match(compactEnglishPrompt, /What should I wear\?/);
assert.doesNotMatch(compactEnglishPrompt, /\/no_think/);
const chineseTranscript = "今天天气怎么样？";
const compactChinesePrompt = api.buildConversationUserText({
  transcript: chineseTranscript,
  languageRoute: api.resolveLanguageSelection("zh")
});
assert.match(compactChinesePrompt, /简体中文/);
assert.match(compactChinesePrompt, /今天天气怎么样/);
assert(compactChinesePrompt.length <= 100, "a no-memory language hint must remain compact");
assert.doesNotMatch(compactChinesePrompt, /Conversation history|Current user utterance|Never mention|instructions/i);

const payloadHistory = api.createConversationHistory({
  ...api.HISTORY_LIMITS,
  maxTurns: 3
});
payloadHistory.addTurn({ user: "What is the weather?", assistant: "It is sunny.", language: "en" });
payloadHistory.addTurn({ user: "Quel temps fait-il ?", assistant: "Il fait beau.", language: "fr" });
const asrPayload = api.buildAsrPayload({ encodedWav: "BASE64-WAV", languageSelection: "zh" });
assert.equal(asrPayload.messages[0].content, "Perform ASR.");
assert.equal(asrPayload.messages.at(-1).content[0].input_audio.data, "BASE64-WAV");
const conversationPayload = api.buildConversationPayload({
  requestMode: "chat",
  transcript: "Que dois-je porter ?",
  languageRoute: api.resolveLanguageSelection("fr"),
  historyMessages: payloadHistory.toMessages("fr")
});
assert.equal("modalities" in conversationPayload, false, "the text companion must not receive audio modalities");
assert.equal(conversationPayload.model, "Qwen2.5-1.5B-Instruct-Q4_K_M");
assert.equal(conversationPayload.max_tokens, 768);
assert.equal(conversationPayload.repeat_penalty, 1.08);
assert.equal("chat_template_kwargs" in conversationPayload, false);
assert.match(conversationPayload.messages[0].content, /Answer the current user directly and accurately/);
assert.deepEqual(conversationPayload.messages.map(message => message.role), ["system", "user", "assistant", "user"]);
assert.equal(conversationPayload.messages[1].content, "Quel temps fait-il ?");
assert.equal(conversationPayload.messages[2].content, "Il fait beau.");
assert.match(conversationPayload.messages[3].content, /Répondez directement et uniquement en français/);
assert.match(conversationPayload.messages[3].content, /Que dois-je porter \?/);
assert.doesNotMatch(conversationPayload.messages[3].content, /\/no_think/);
assert.doesNotMatch(conversationPayload.messages[1].content, /What is the weather|It is sunny/);
assert.doesNotMatch(JSON.stringify(conversationPayload), /input_audio|CURRENT-AUDIO/);
assert.match(
  api.buildConversationUserText({
    transcript: "13たす19はいくつですか。",
    languageRoute: api.resolveLanguageSelection("ja")
  }),
  /日本語だけで答えてください/
);
assert.match(
  api.buildConversationUserText({
    transcript: "13 더하기 19는 얼마인가요?",
    languageRoute: api.resolveLanguageSelection("ko")
  }),
  /한국어로만 답하세요/
);
assert.throws(
  () => api.buildConversationPayload({
    requestMode: "chat",
    transcript: "x",
    languageRoute: explicitFrench,
    historyMessages: [{ role: "user", content: [{ input_audio: "forbidden" }] }]
  }),
  /only user\/assistant text messages/
);
assert.throws(
  () => api.parseSseEvent('data: {"error":{"message":"Unsupported system prompt."}}'),
  /Unsupported system prompt/
);
assert.equal(api.parseSseEvent("data: [DONE]"), null);
const reasoningFilter = api.createReasoningStreamFilter();
assert.equal(reasoningFilter.push("<thi"), "");
assert.equal(reasoningFilter.push("nk>private chain"), "");
assert.equal(reasoningFilter.push(" of thought</think>\n\nVisible "), "Visible ");
assert.equal(reasoningFilter.push("answer."), "answer.");
assert.equal(reasoningFilter.push("", true), "");
assert.equal(reasoningFilter.stats().reasoning_blocks_removed, 1);
assert.equal(reasoningFilter.stats().visible_characters, "Visible answer.".length);
assert.equal(
  api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("zh"), "Ni Hao"),
  "Mandarin conversation is blocked because this LFM ASR pass returned phonetic Latin text instead of Chinese characters."
);
assert.equal(api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("zh"), "你好"), null);
assert.equal(
  api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("ja"), "Nihongo de kotaete"),
  "Japanese conversation is blocked because this ASR pass returned phonetic Latin text instead of Japanese characters."
);
assert.equal(api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("ja"), "日本語で答えて"), null);
assert.equal(
  api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("ko"), "Hangugeoro daphae"),
  "Korean conversation is blocked because this ASR pass returned phonetic Latin text instead of Hangul."
);
assert.equal(api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("ko"), "한국어로 답해"), null);
assert.equal(
  api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("ru"), "Priviat"),
  "Russian conversation is blocked because this LFM ASR pass returned phonetic Latin text instead of Cyrillic Russian."
);
assert.equal(api.phoneticTranscriptBlockReason(api.resolveLanguageSelection("ru"), "Привет"), null);
assert.equal(api.normalizeAsrTranscript('Transcript: “Bonjour tout le monde”'), "Bonjour tout le monde");

const companionPayload = candidates => ({
  ok: true,
  complete: true,
  schema_version: 1,
  requested_language: "auto",
  selection_required: true,
  candidates
});
const qnnCandidate = (text, lang = "<|en|>", logProbabilities = [-0.05, -0.08]) => ({
  backend: "sensevoice_qnn",
  provider: "qnn",
  ok: true,
  result: { text, lang, ys_log_probs: logProbabilities }
});
const russianCandidate = (text, logProbabilities = [-0.1, -0.15, -0.08]) => ({
  backend: "zipformer_ru_cpu",
  provider: "cpu",
  ok: true,
  result: { text, ys_log_probs: logProbabilities }
});
// Parakeet returns no per-utterance language tag, so the ES/FR route is
// validated by Latin-script evidence rather than by a SenseVoice tag.
const parakeetCandidate = (text, ok = true, logProbabilities = [-0.1, -0.2]) => ({
  backend: "parakeet_v3_cpu",
  provider: "cpu",
  ok,
  ...(ok
    ? { result: { text, ys_log_probs: logProbabilities } }
    : { error: { code: "parakeet_not_ready", message: "assets incomplete" } })
});

const selectedChinese = api.selectCompanionAsr(companionPayload([
  qnnCandidate("你好，请告诉我今天的天气。", "<|zh|>"),
  russianCandidate("")
]), "auto");
assert.equal(selectedChinese.languageRoute.language, "zh");
assert.equal(selectedChinese.transcript, "你好，请告诉我今天的天气。");
assert.equal(selectedChinese.asrFacts.backend, "sensevoice_qnn");
assert.equal(selectedChinese.asrFacts.npu_executed, true);
assert.equal(selectedChinese.asrFacts.npu_selected, true);

const selectedJapanese = api.selectCompanionAsr(companionPayload([
  qnnCandidate("日本語で短く答えてください。", "<|ja|>")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(selectedJapanese.languageRoute.language, "ja");
assert.equal(selectedJapanese.languageRoute.localeTag, "ja-JP");
assert.equal(selectedJapanese.asrFacts.backend, "sensevoice_qnn");

const selectedKorean = api.selectCompanionAsr(companionPayload([
  qnnCandidate("한국어로 짧게 답하세요.", "<|ko|>")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(selectedKorean.languageRoute.language, "ko");
assert.equal(selectedKorean.languageRoute.localeTag, "ko-KR");
assert.equal(selectedKorean.asrFacts.backend, "sensevoice_qnn");
assert.equal(api.senseVoiceLanguageHint({ lang: "<|ja|>" }), "ja");
assert.equal(api.senseVoiceLanguageHint({ lang: "korean" }), "ko");
assert.equal(
  api.senseVoiceLanguageHint({ lang: "english_korean" }),
  null,
  "language hints use exact schema tokens rather than substring matches"
);
const mislabeledChineseAsJapanese = api.selectCompanionAsr(companionPayload([
  qnnCandidate("北京是中国的首都。", "<|ja|>")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(
  mislabeledChineseAsJapanese,
  null,
  "a Japanese tag cannot turn a Han-only Mandarin transcript into Japanese"
);
assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    qnnCandidate("北京是中国的首都。", "<|ja|>")
  ]), "ja"),
  /reliable Japanese characters/
);
assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    qnnCandidate("フランスの首都はどこですか。", "<|zh|>")
  ]), "ja"),
  /mismatching SenseVoice language tag/
);
const mislabeledLatinAsKorean = api.selectCompanionAsr(companionPayload([
  qnnCandidate("This answer is in English.", "<|ko|>")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(mislabeledLatinAsKorean, null);

const selectedEnglish = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Hello, please explain the weather today."),
  russianCandidate("хэлло", [-1.4])
]), "auto");
assert.equal(selectedEnglish.languageRoute.language, "en");
assert.equal(selectedEnglish.asrFacts.backend, "sensevoice_qnn");

const earlyEnglish = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Hello, please explain the weather today.")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(earlyEnglish.languageRoute.language, "en");
const ordinaryEnglish = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Who invented calculus?")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(ordinaryEnglish.languageRoute.language, "en", "SenseVoice EN tag plus a sentence-shaped transcript should avoid needless Russian fallback");
const genericSentenceNeedsRussianCheck = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Alpha beta gamma delta.")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(genericSentenceNeedsRussianCheck, null, "generic EN-tagged Latin output must not bypass the Russian disambiguation stage");
const ambiguousQnnOnly = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Kazhikradka Zvaitsa.")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true });
assert.equal(ambiguousQnnOnly, null, "uncertain QNN Latin text must trigger the Russian second stage");
const russianPriorDefersEnglish = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Who invented calculus?")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true, previousLanguage: "ru" });
assert.equal(russianPriorDefersEnglish, null, "a confident Russian context must force the Russian disambiguation stage");
const englishPriorStillChecksRussian = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Yeah, blue.")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true, previousLanguage: "en" });
assert.equal(englishPriorStillChecksRussian, null, "an English prior must not hide a possible language switch to Russian");
const shortChineseFollowup = api.selectCompanionAsr(companionPayload([
  qnnCandidate("好", "<|zh|>")
]), "auto", { allowUnopposedEnglish: false, ambiguousReturnsNull: true, previousLanguage: "zh" });
assert.equal(shortChineseFollowup.languageRoute.language, "zh");

const mergedAutoPayload = api.mergeCompanionPayloads(
  "auto",
  companionPayload([qnnCandidate("Priviat pajaosta reskaje apagodi siwodna.")]),
  companionPayload([russianCandidate("Привет, пожалуйста, расскажи о погоде сегодня.")])
);
assert.equal(mergedAutoPayload.requested_language, "auto");
assert.deepEqual(
  mergedAutoPayload.candidates.map(candidate => candidate.backend),
  ["sensevoice_qnn", "zipformer_ru_cpu"]
);

const selectedRussian = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Priviat pajaosta reskaje apagodi siwodna."),
  russianCandidate("Привет, пожалуйста, расскажи о погоде сегодня.", [-0.08, -0.12, -0.09, -0.11, -0.07])
]), "auto");
assert.equal(selectedRussian.languageRoute.language, "ru");
assert.equal(selectedRussian.asrFacts.backend, "zipformer_ru_cpu");
assert.equal(selectedRussian.asrFacts.npu_executed, true, "Auto also executed the QNN candidate");
assert.equal(selectedRussian.asrFacts.npu_selected, false);
const russianWinsOverPlausibleEnglishHallucination = api.selectCompanionAsr(companionPayload([
  qnnCandidate("What do you need today?"),
  russianCandidate("Расскажи мне о погоде сегодня.", [-0.08, -0.12, -0.09, -0.11, -0.07])
]), "auto");
assert.equal(russianWinsOverPlausibleEnglishHallucination.languageRoute.language, "ru");
const languageSwitchAfterEnglish = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Yeah, blue."),
  russianCandidate("я тебя люблю", [-0.01, -0.12, -0.10])
]), "auto", { previousLanguage: "en" });
assert.equal(languageSwitchAfterEnglish.languageRoute.language, "ru");
const shortRussianFollowup = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Yeah."),
  russianCandidate("да", [-0.10])
]), "auto", { previousLanguage: "ru" });
assert.equal(shortRussianFollowup.languageRoute.language, "ru");
const englishWinsOverWeakRussianHallucination = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Who invented calculus?"),
  russianCandidate("шум", [-1.2])
]), "auto");
assert.equal(englishWinsOverWeakRussianHallucination.languageRoute.language, "en");
const shortEnglishFollowupAfterCheck = api.selectCompanionAsr(companionPayload([
  qnnCandidate("Yes."),
  russianCandidate("шум", [-1.2])
]), "auto", { previousLanguage: "en" });
assert.equal(shortEnglishFollowupAfterCheck.languageRoute.language, "en");

assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    qnnCandidate("Kazhikradka Zvaitsa."),
    russianCandidate("привет", [-0.05])
  ]), "auto"),
  /Auto language is ambiguous/
);
assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    qnnCandidate("Ni hao", "<|zh|>"),
    russianCandidate("")
  ]), "zh"),
  /reliable Chinese characters/
);
assert.equal(api.plannedAsrFacts("zh").backend, "sensevoice_qnn");
assert.equal(api.plannedAsrFacts("ja").backend, "sensevoice_qnn");
assert.equal(api.plannedAsrFacts("ko").backend, "sensevoice_qnn");
assert.equal(api.plannedAsrFacts("ru").backend, "zipformer_ru_cpu");

// --- Spanish and French: Parakeet-TDT-0.6B-v3 INT8 on the CPU provider -----
// The route must never claim acceleration. `npu_gate` is "blocked" before the
// request and the fallback reason names the exact version conflict.
for (const language of ["es", "fr"]) {
  const planned = api.plannedAsrFacts(language);
  assert.equal(planned.backend, "parakeet_v3_cpu");
  assert.equal(planned.provider, "cpu");
  assert.equal(planned.npu_executed, false);
  assert.equal(planned.npu_selected, false);
  assert.equal(planned.npu_gate, "blocked");
  assert.match(planned.npu_fallback_reason, /qnn_2_40_context_versus_2_45_pin/);
  assert.equal(planned.asr_companion, true);
}
// Italian is deliberately NOT wired: it still falls through to LFM on CPU.
assert.equal(api.plannedAsrFacts("it").backend, "lfm_audio_q4_cpu");

const selectedSpanish = api.selectCompanionAsr(companionPayload([
  parakeetCandidate("No preguntes qué puede hacer tu país por ti.")
]), "es");
assert.equal(selectedSpanish.languageRoute.language, "es");
assert.equal(selectedSpanish.languageRoute.localeTag, "es-ES");
assert.equal(selectedSpanish.asrFacts.backend, "parakeet_v3_cpu");
assert.equal(selectedSpanish.asrFacts.provider, "cpu");
assert.equal(selectedSpanish.asrFacts.npu_executed, false);
assert.equal(selectedSpanish.asrFacts.npu_selected, false);
assert.match(
  selectedSpanish.asrFacts.npu_fallback_reason,
  /qnn_2_40_context_versus_2_45_pin/
);

const selectedFrench = api.selectCompanionAsr(companionPayload([
  parakeetCandidate("Ne vous demandez pas ce que votre pays peut faire pour vous.")
]), "fr");
assert.equal(selectedFrench.languageRoute.language, "fr");
assert.equal(selectedFrench.languageRoute.localeTag, "fr-FR");
assert.equal(selectedFrench.asrFacts.backend, "parakeet_v3_cpu");

// A dead Parakeet backend fails closed; it must not silently borrow SenseVoice.
assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    parakeetCandidate("", false),
    qnnCandidate("Ask not what your country can do for you.")
  ]), "es"),
  /Spanish ASR is unavailable on the phone/
);
// Non-Latin output on an ES/FR route is rejected rather than spoken.
assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    parakeetCandidate("привет")
  ]), "fr"),
  /reliable Latin text/
);
// Auto must never route to ES/FR: the ladder is EN/ZH/JA/KO/RU only.
assert.throws(
  () => api.selectCompanionAsr(companionPayload([
    qnnCandidate("Kazhikradka Zvaitsa."),
    russianCandidate("привет", [-0.05])
  ]), "auto"),
  /Spanish and French are always an explicit choice/
);
const multilingualScriptEvidence = api.scriptEvidence("你好, ひらがな, カナ, 한국어, Привет, hello");
assert.equal(multilingualScriptEvidence.han, 2);
assert.equal(multilingualScriptEvidence.hiragana, 4);
assert.equal(multilingualScriptEvidence.katakana, 2);
assert.equal(multilingualScriptEvidence.kana, 6);
assert.equal(multilingualScriptEvidence.hangul, 3);
assert(api.meanTokenLogProbability({ ys_log_probs: [-0.1, -0.3] }) < -0.19);
const repeatedChinese = api.measureTextRepetition("你好你好你好你好");
assert.equal(repeatedChinese.token_count, 8);
assert(repeatedChinese.duplicate_gram_ratio > 0);
assert(repeatedChinese.max_gram_count >= 2);

const staleHistory = api.createConversationHistory();
const staleCoordinator = api.createTurnCoordinator({
  makeId: (() => { let value = 0; return () => `history-${++value}`; })(),
  makeController: () => ({ abort() {} })
});
const staleHistoryTurn = staleCoordinator.begin("chat", "file");
const currentHistoryTurn = staleCoordinator.begin("chat", "file");
assert.deepEqual(
  api.commitConversationTurn(staleHistory, staleCoordinator, staleHistoryTurn, {
    transcript: "stale user", assistant: "stale assistant", language: "en"
  }, true),
  { committed: false, reason: "stale_turn" }
);
assert.equal(staleHistory.summary().turns, 0, "cancelled/stale turns cannot mutate history");
const disabledHistoryCommit = api.commitConversationTurn(staleHistory, staleCoordinator, currentHistoryTurn, {
  transcript: "private user", assistant: "private assistant", language: "en"
}, false);
assert.equal(disabledHistoryCommit.committed, false);
assert.equal(disabledHistoryCommit.reason, "history_disabled");
assert.equal(staleHistory.summary().turns, 0, "memory-off turns cannot mutate history");
const committedHistory = api.commitConversationTurn(staleHistory, staleCoordinator, currentHistoryTurn, {
  transcript: "real user", assistant: "real assistant", language: "ru"
}, true);
assert.equal(committedHistory.committed, true);
assert.equal(staleHistory.latestLanguage(), "ru");
const qualityRejectedHistory = api.createConversationHistory();
const qualityRejectedCommit = api.commitConversationTurn(
  qualityRejectedHistory,
  staleCoordinator,
  currentHistoryTurn,
  { transcript: "repeat please", assistant: "loop loop loop", language: "en" },
  true,
  { qualityWarnings: ["repeated_four_grams"] }
);
assert.equal(qualityRejectedCommit.committed, false);
assert.equal(qualityRejectedCommit.reason, "quality_rejected");
assert.equal(qualityRejectedHistory.summary().turns, 0, "quality-gated answers never poison later context");
assert.deepEqual(
  api.responseQualityWarnings("Paris is the capital of France.", "stop", api.resolveLanguageSelection("en")).warnings,
  []
);
assert.deepEqual(
  api.responseQualityWarnings("42", "stop", api.resolveLanguageSelection("zh")).warnings,
  [],
  "numeric-only answers are language-neutral rather than wrong-script output"
);
assert.equal(
  api.ttsChunkScriptCheck("42.", api.resolveLanguageSelection("zh")).safe,
  true,
  "numeric-only speech chunks remain language-neutral"
);
assert.equal(
  api.ttsChunkScriptCheck("AI 是人工智能。", api.resolveLanguageSelection("zh")).safe,
  true,
  "mixed technical terms are allowed when the requested script is present"
);
assert.equal(
  api.ttsChunkScriptCheck("OpenAI.", api.resolveLanguageSelection("zh")).safe,
  true,
  "a bounded standalone technical brand remains speakable in Mandarin mode"
);
assert.equal(
  api.ttsChunkScriptCheck("https://openai.com/docs", api.resolveLanguageSelection("ru")).safe,
  true,
  "a standalone URL remains script-neutral in Russian mode"
);
assert.equal(
  api.ttsChunkScriptCheck("答えは三十二です。", api.resolveLanguageSelection("ja")).safe,
  true,
  "Japanese Han and kana are accepted for Japanese speech"
);
assert.equal(
  api.ttsChunkScriptCheck("정답은 32입니다.", api.resolveLanguageSelection("ko")).safe,
  true,
  "Hangul is accepted for Korean speech"
);
assert.equal(
  api.ttsChunkScriptCheck("The answer is 32.", api.resolveLanguageSelection("ja")).safe,
  false,
  "Latin prose is rejected before Japanese TTS"
);
assert.equal(
  api.ttsChunkScriptCheck("The answer is 32.", api.resolveLanguageSelection("ko")).safe,
  false,
  "Latin prose is rejected before Korean TTS"
);
assert.equal(
  api.ttsChunkScriptCheck(
    "This response is mostly English with 日本語.",
    api.resolveLanguageSelection("ja")
  ).safe,
  false,
  "one Japanese token cannot make mostly-English prose safe"
);
assert.equal(
  api.ttsChunkScriptCheck(
    "This response is mostly English with 한국.",
    api.resolveLanguageSelection("ko")
  ).safe,
  false,
  "two Hangul characters cannot make mostly-English prose safe"
);
assert.equal(
  api.ttsChunkScriptCheck("Visit https://openai.com/docs for details.", api.resolveLanguageSelection("zh")).safe,
  false,
  "an English sentence cannot become script-neutral merely by containing a URL"
);
assert.equal(
  api.ttsChunkScriptCheck("This is English.", api.resolveLanguageSelection("zh")).safe,
  false,
  "an all-Latin Mandarin chunk is rejected before TTS"
);
assert.equal(
  api.ttsChunkScriptCheck(
    "This answer is almost entirely English with one token 中.",
    api.resolveLanguageSelection("zh")
  ).safe,
  false,
  "one Han character cannot mask an otherwise Latin Mandarin chunk"
);
assert.equal(
  api.ttsChunkScriptCheck(
    "OpenAI GPT-4 is a model，由美国公司开发。",
    api.resolveLanguageSelection("zh")
  ).safe,
  false,
  "minority Han prose cannot hide a mostly English Mandarin chunk"
);
assert.equal(
  api.responseQualityWarnings(
    "This answer is almost entirely English with one token 中.",
    "stop",
    api.resolveLanguageSelection("zh")
  ).warnings.includes("response_script_mismatch"),
  true,
  "the same ratio gate prevents a disguised wrong-script answer entering history"
);
assert.equal(
  api.ttsChunkScriptCheck("Hello.", api.resolveLanguageSelection("ru")).reason,
  "response_script_mismatch"
);
assert.equal(
  api.preflightTtsChunks(
    ["Привет!", "Чем я могу помочь?"],
    api.resolveLanguageSelection("ru")
  ).ok,
  true
);
const preTtsWrongScript = api.preflightTtsChunks(
  ["Hello. This answer is in the wrong language."],
  api.resolveLanguageSelection("ru")
);
assert.equal(preTtsWrongScript.ok, false);
assert.equal(preTtsWrongScript.issue.reason, "response_script_mismatch");
const preTtsRepetition = api.preflightTtsChunks(
  ["loop one two three.", "loop one two three.", "loop one two three.", "loop one two three."],
  api.resolveLanguageSelection("en")
);
assert.equal(preTtsRepetition.ok, false);
assert.equal(preTtsRepetition.issue.reason, "repeated_four_grams");
assert.equal(api.preflightTtsChunks(
  [
    "For case A, the answer is yes.",
    "For case B, the answer is yes.",
    "For case C, the answer is yes."
  ],
  api.resolveLanguageSelection("en")
).ok, true, "three-item structured output is not mistaken for a repetition collapse");
assert.equal(api.preflightTtsChunks(
  [
    "First scenario A concludes the answer is yes.",
    "A separate scenario B also concludes the answer is yes.",
    "In scenario C, independently, the answer is yes.",
    "Finally, scenario D reports the answer is yes."
  ],
  api.resolveLanguageSelection("en")
).ok, true, "a four-item list with one repeated phrase is not treated as a collapse");
assert.equal(api.preflightTtsChunks(
  ["安全第一。", "安全第一。", "安全第一。"],
  api.resolveLanguageSelection("zh")
).ok, true, "three-fold Chinese emphasis remains speakable");
assert.equal(api.responseQualityWarnings(
  "安全第一。安全第一。安全第一。",
  "stop",
  api.resolveLanguageSelection("zh")
).warnings.includes("repeated_four_grams"), false, "speakable emphasis is also safe for history");
assert(api.responseQualityWarnings(
  "This response ended without a terminal reason.",
  null,
  api.resolveLanguageSelection("en")
).warnings.includes("generation_incomplete"));
assert(api.responseQualityWarnings(
  "This response was filtered.",
  "content_filter",
  api.resolveLanguageSelection("en")
).warnings.includes("generation_incomplete"));
assert(api.responseQualityWarnings(
  "loop one two three loop one two three loop one two three",
  "length",
  api.resolveLanguageSelection("en")
).warnings.includes("generation_length_cap"));
assert.equal(api.streamingResponseCollapse("This is a normal concise answer with no looping words at all.").collapsed, false);
assert.equal(api.streamingResponseCollapse(
  "loop one two three loop one two three loop one two three loop one two three loop one two three loop one two three"
).collapsed, true, "streaming collapse guard stops repeated output before a long spoken loop");

const historyToggleElement = elements.get("history-toggle");
api.setHistoryEnabled(true);
assert.equal(api.isHistoryEnabled(), true);
assert.equal(historyToggleElement.getAttribute("aria-checked"), "true");
assert.match(elements.get("history-status").textContent, /Memory on.*explicit recall only.*0\/2 turns/i);
api.conversationHistory.addTurn({ user: "forget me", assistant: "forgotten", language: "en" });
api.setHistoryEnabled(false);
assert.equal(api.isHistoryEnabled(), false);
assert.equal(historyToggleElement.getAttribute("aria-checked"), "false");
assert.equal(api.conversationHistory.summary().turns, 0, "turning context off clears retained text");
assert.match(elements.get("history-status").textContent, /Memory off.*no prior turns/i);
api.setHistoryEnabled(true);
api.setInputBusy(true);
assert.equal(historyToggleElement.disabled, true, "context cannot be changed during an active turn");
api.setInputBusy(false);
assert.equal(historyToggleElement.disabled, false);
historyToggleElement.click();
assert.equal(api.isHistoryEnabled(), false, "the visible context switch drives the same state API");

api.setHistoryEnabled(true);
api.conversationHistory.addTurn({ user: "clear me", assistant: "cleared", language: "it" });
elements.get("clear-history").click();
assert.equal(api.conversationHistory.summary().turns, 0, "clear-history control resets shared context");
api.setHistoryEnabled(false);
const languageSelectElement = elements.get("language-select");
languageSelectElement.value = "zh";
languageSelectElement.dispatchEvent({ type: "change" });
assert.equal(localStorage.getItem("lfm.language.selection.v1"), "zh");
assert.match(elements.get("language-status").textContent, /Mandarin Chinese.*100%.*explicit user selection/);
languageSelectElement.value = "auto";
languageSelectElement.dispatchEvent({ type: "change" });
assert.equal(localStorage.getItem("lfm.language.selection.v1"), "auto");
assert.match(elements.get("language-status").textContent, /awaiting speech evidence/);
const lfmVoiceSelectElement = elements.get("lfm-voice-select");
const englishResponseStyleElement = elements.get("english-response-style");
assert.equal(englishResponseStyleElement.value, "expressive");
assert.match(elements.get("english-response-status").textContent, /listens and answers directly.*less exact/i);
assert.equal(lfmVoiceSelectElement.value, "uk_female");
assert.equal(lfmVoiceSelectElement.disabled, true, "fixed voices are not used by expressive interleaved generation");
assert.match(elements.get("lfm-voice-status").textContent, /UK female saved.*Exact text/i);
englishResponseStyleElement.value = "exact";
englishResponseStyleElement.dispatchEvent({ type: "change" });
assert.equal(lfmVoiceSelectElement.disabled, false);
assert.match(elements.get("english-response-status").textContent, /Qwen writes the answer/i);
assert.match(elements.get("lfm-voice-status").textContent, /UK female.*sentence-level LFM.*Exact text/i);
assert.equal(telemetryPayloads.at(-1).event, "english_response_style_changed");
assert.equal(telemetryPayloads.at(-1).details.response_style, "exact");
lfmVoiceSelectElement.value = "us_male";
lfmVoiceSelectElement.dispatchEvent({ type: "change" });
assert.match(elements.get("lfm-voice-status").textContent, /US male.*sentence-level LFM/);
assert.equal(telemetryPayloads.at(-1).event, "lfm_voice_selection_changed");
assert.equal(telemetryPayloads.at(-1).details.voice, "us_male");
lfmVoiceSelectElement.value = "uk_female";
lfmVoiceSelectElement.dispatchEvent({ type: "change" });
englishResponseStyleElement.value = "expressive";
englishResponseStyleElement.dispatchEvent({ type: "change" });

const telemetryDetails = api.languageTelemetryDetails("auto", priorFallback, staleHistory.summary());
assert.equal(telemetryDetails.selected_language, "auto");
assert.equal(telemetryDetails.resolved_language, "it");
assert.equal(telemetryDetails.fallback_used, true);
assert.equal(telemetryDetails.resolution_reason, priorFallback.reason);
assert.equal(telemetryDetails.history_turns, 1);
assert.equal(telemetryDetails.backend, "lfm_audio_q4_cpu");
assert.deepEqual(api.resolveVoiceDelivery("chat", autoChinese), {
  responseMode: "tts",
  voiceBackend: "android_native_tts",
  rerouted: true,
  reason: "LFM natural TTS is not validated for Mandarin Chinese; using the explicit multilingual offline voice fallback"
});
assert.equal(
  api.resolveVoiceDelivery("chat", api.resolveLanguageSelection("en")).voiceBackend,
  "lfm_audio_interleaved_streaming"
);
assert.equal(
  api.resolveVoiceDelivery("chat", api.resolveLanguageSelection("en"), "exact").voiceBackend,
  "lfm_audio_natural_tts_streaming"
);
assert.equal(
  api.resolveVoiceDelivery("chat", api.resolveLanguageSelection("en"), "expressive", { requiresExactText: true }).voiceBackend,
  "lfm_audio_natural_tts_streaming"
);
assert.equal(api.resolveEnglishResponseStyle("invalid").key, "expressive");
assert.deepEqual(api.resolveLfmTtsVoice("uk_female"), {
  key: "uk_female",
  label: "UK female",
  prompt: "Perform TTS. Use the UK female voice."
});
assert.equal(api.resolveLfmTtsVoice("not-a-voice").key, "uk_female");
assert.deepEqual(api.buildLfmTtsPayload("  Hello   from the natural voice.  "), {
  model: "LFM2.5-Audio-1.5B",
  stream: true,
  reset_context: true,
  modalities: ["audio"],
  max_tokens: 512,
  messages: [
    { role: "system", content: "Perform TTS. Use the UK female voice." },
    { role: "user", content: "Hello from the natural voice." }
  ]
});
assert.equal(
  api.buildLfmTtsPayload("Hello.", "us_male").messages[0].content,
  "Perform TTS. Use the US male voice."
);
assert.deepEqual(api.buildDirectLfmConversationPayload("  What is 3 plus 2?  "), {
  model: "LFM2.5-Audio-1.5B",
  stream: true,
  reset_context: true,
  modalities: ["text", "audio"],
  temperature: 0,
  max_tokens: 512,
  messages: [
    {
      role: "system",
      content: "Respond with interleaved text and audio."
    },
    { role: "user", content: "What is 3 plus 2?" }
  ]
});
assert.throws(() => api.buildDirectLfmConversationPayload("  "), /non-empty ASR transcript/);
assert.equal(api.resolveVoiceDelivery("tts", explicitFrench).responseMode, "tts");
const chineseVoiceDelivery = api.resolveVoiceDelivery("chat", autoChinese);
const chineseVoicePayload = api.buildConversationPayload({
  requestMode: chineseVoiceDelivery.responseMode,
  transcript: chineseTranscript,
  languageRoute: autoChinese,
  historyMessages: []
});
assert.equal("modalities" in chineseVoicePayload, false);
assert.equal(chineseVoicePayload.max_tokens, 768, "system-TTS text generation gets the extended response budget");
assert.match(chineseVoicePayload.messages[0].content, /Answer the current user directly and accurately/);
assert.match(chineseVoicePayload.messages[1].content, /只用简体中文回答当前用户/);
const noMemoryPayload = api.buildConversationPayload({
  requestMode: "chat",
  transcript: "What secret phrase did I ask you to remember?",
  languageRoute: api.resolveLanguageSelection("en"),
  historyMessages: [],
  memoryUnavailableAction: "recall"
});
assert.match(noMemoryPayload.messages.at(-1).content, /Conversation memory is unavailable/);
assert.match(noMemoryPayload.messages.at(-1).content, /Do not repeat the question or name any phrase/);

const makeMetrics = () => ({
  requestStartedNativeMs: 1000,
  resolvedLocale: null,
  selectedPolicy: api.currentTtsPolicy(),
  resolvedEngine: null,
  chunksSubmitted: 0,
  chunksConfirmed: 0,
  firstAudioStartProxyMs: null,
  queueClosed: false,
  playbackTerminalStatus: null
});

const nativeTurnA = api.turnCoordinator.begin("tts", "file");
api.ttsClientTurns.set(nativeTurnA.id, api.createTtsClientState(nativeTurnA, makeMetrics()));
const initA = api.initializeNativeTts(nativeTurnA, "en-US");
const nativeTurnB = api.turnCoordinator.begin("tts", "file");
api.ttsClientTurns.set(nativeTurnB.id, api.createTtsClientState(nativeTurnB, makeMetrics()));
await assert.rejects(initA, error => api.isAbortError(error));
assert(nativeTtsCalls.some(call => call.method === "cancel"
  && call.turnId === nativeTurnA.id && call.generation === nativeTurnA.generation));

const initB = api.initializeNativeTts(nativeTurnB, "ja-JP");
window.onNativeTtsEvent(JSON.stringify({
  type: "ready",
  turn_id: nativeTurnA.id,
  monotonic_ms: 1010,
  details: { generation: nativeTurnA.generation, resolved_locale: "en-US" }
}));
assert(api.ttsClientTurns.has(nativeTurnB.id), "stale ready callback must not remove the current turn");
window.onNativeTtsEvent(JSON.stringify({
  type: "ready",
  turn_id: nativeTurnB.id,
  monotonic_ms: 1020,
  details: {
    generation: nativeTurnB.generation,
    resolved_locale: "ja-JP",
    network_connection_required: false,
    tts_policy: "auto",
    tts_engine_id: "system"
  }
}));
await initB;
assert.equal(api.ttsClientTurns.get(nativeTurnB.id).metrics.selectedPolicy, "auto");
assert.equal(api.ttsClientTurns.get(nativeTurnB.id).metrics.resolvedEngine, "system");
const engineResolvedTelemetry = telemetryPayloads.find(event =>
  event.event === "tts_engine_resolved" && event.turn_id === nativeTurnB.id
);
assert.equal(engineResolvedTelemetry.details.tts_policy, "auto");
assert.equal(engineResolvedTelemetry.details.resolved_tts_engine, "system");

const acknowledged = api.queueNativeTtsChunk(nativeTurnB, "こんにちは。", "chunk-test");
window.onNativeTtsEvent(JSON.stringify({
  type: "chunk_queued",
  turn_id: nativeTurnB.id,
  monotonic_ms: 1030,
  details: { generation: nativeTurnB.generation, chunk_id: "chunk-test", chunk_index: 0 }
}));
await acknowledged;
assert.equal(api.ttsClientTurns.get(nativeTurnB.id).metrics.chunksConfirmed, 1);

const rejectedChunk = api.queueNativeTtsChunk(nativeTurnB, "失敗。", "chunk-fail");
window.onNativeTtsEvent(JSON.stringify({
  type: "terminal",
  turn_id: nativeTurnB.id,
  monotonic_ms: 1040,
  details: { generation: nativeTurnB.generation, status: "failed", message: "synthetic failure" }
}));
await assert.rejects(rejectedChunk, /synthetic failure/);
assert.equal(api.ttsClientTurns.has(nativeTurnB.id), false, "terminal callback must release per-turn state");

const originalSetTimeout = globalThis.setTimeout;
const originalClearTimeout = globalThis.clearTimeout;
let timeoutCallback = null;
globalThis.setTimeout = callback => { timeoutCallback = callback; return 77; };
globalThis.clearTimeout = () => {};
const timeoutWaiter = api.createTimedWaiter(5, "ack timeout");
timeoutCallback();
await assert.rejects(timeoutWaiter.promise, /ack timeout/);
globalThis.setTimeout = originalSetTimeout;
globalThis.clearTimeout = originalClearTimeout;

const floatInput = new Float32Array([0.25, -0.5]);
const floatData = Buffer.from(floatInput.buffer).toString("base64");
const legacy = api.decodeAudioDelta({
  audio_chunk: { data: floatData, sample_rate: 24000, format: "pcm_f32le" }
});
assert.equal(legacy.sampleRate, 24000);
assert.deepEqual(Array.from(legacy.samples), Array.from(floatInput));

const int16 = Buffer.alloc(6);
int16.writeInt16LE(32767, 0);
int16.writeInt16LE(-32768, 2);
int16.writeInt16LE(0, 4);
const current = api.decodeAudioDelta({
  audio: { data: int16.toString("base64"), sample_rate: 24000 }
});
assert.equal(current.sampleRate, 24000);
assert(Math.abs(current.samples[0] - 32767 / 32768) < 1e-7);
assert.equal(current.samples[1], -1);
assert.equal(current.samples[2], 0);
assert.equal(api.naturalTtsAudioCheck(current).safe, true);
assert.throws(
  () => api.decodeAudioDelta({ audio: { data: "AA==", format: "opus", sample_rate: 24000 } }),
  /Unsupported streamed audio encoding/
);
assert.throws(
  () => api.base64ToFloat32(Buffer.from([0, 1, 2]).toString("base64")),
  /invalid byte length/
);
const voicedProbe = new Float32Array(960);
voicedProbe.fill(0.02, 480);
assert.equal(api.firstVoicedSampleOffset(voicedProbe, 24000), 480);
assert.equal(api.firstVoicedSampleOffset(new Float32Array(960), 24000), null);
assert.equal(api.naturalTtsAudioCheck({ samples: new Float32Array([Number.NaN]), sampleRate: 24000 }).reason, "non_finite_pcm");
assert.equal(api.naturalTtsAudioCheck({ samples: new Float32Array([1.2]), sampleRate: 24000 }).reason, "clipped_pcm");
assert.equal(api.naturalTtsAudioCheck({ samples: new Float32Array([0.1]), sampleRate: 96000 }).reason, "invalid_sample_rate");

assert.equal(
  api.mapContextTimeToPerformanceMs({ contextTime: 2.25, performanceTime: 5000 }, 2.20),
  4950
);
assert.equal(api.mapContextTimeToPerformanceMs({ contextTime: NaN, performanceTime: 5000 }, 2.20), null);
assert.equal(api.mapContextTimeToPerformanceMs({ contextTime: null, performanceTime: 5000 }, 2.20), null);
assert.equal(api.getOutputHeadCrossing({ contextTime: 2.19, performanceTime: 4990 }, 2.20), null);
assert.deepEqual(
  api.getOutputHeadCrossing({ contextTime: 2.25, performanceTime: 5000 }, 2.20),
  { contextTime: 2.20, performanceTime: 4950 }
);

let unsupportedResult = null;
const unsupportedObserver = api.observeOutputHeadCrossing(
  {},
  1.14,
  result => { unsupportedResult = result; }
);
assert.equal(unsupportedObserver.supported, false);
assert.equal(unsupportedObserver.status, "unsupported");
assert.equal(unsupportedResult, null);

const timestampSequence = [
  { contextTime: 1.10, performanceTime: 1000 },
  { contextTime: 1.18, performanceTime: 1080 }
];
const pendingPolls = [];
let observedResult = null;
let observerClock = 0;
const crossingObserver = api.observeOutputHeadCrossing(
  { getOutputTimestamp: () => timestampSequence.shift() },
  1.14,
  result => { observedResult = result; },
  {
    now: () => observerClock,
    schedulePoll: callback => { pendingPolls.push(callback); return callback; },
    cancelPoll: () => {},
    timeoutMs: 500
  }
);
assert.equal(crossingObserver.supported, true);
assert.equal(crossingObserver.status, "measuring");
assert.equal(observedResult, null);
assert.equal(pendingPolls.length, 1);
observerClock = 16;
pendingPolls.shift()();
assert.deepEqual(observedResult, {
  status: "observed",
  contextTime: 1.14,
  performanceTime: 1040
});

const timeoutPolls = [];
let timeoutResult = null;
let timeoutClock = 0;
api.observeOutputHeadCrossing(
  { getOutputTimestamp: () => ({ contextTime: 1.10, performanceTime: 1000 }) },
  1.14,
  result => { timeoutResult = result; },
  {
    now: () => timeoutClock,
    schedulePoll: callback => { timeoutPolls.push(callback); return callback; },
    cancelPoll: () => {},
    timeoutMs: 500
  }
);
assert.equal(timeoutResult, null);
timeoutClock = 500;
timeoutPolls.shift()();
assert.deepEqual(timeoutResult, { status: "unavailable" });

const context = api.getPlaybackContext();
const first = await api.queueAudioChunk(new Float32Array(1440), 24000);
assert(Math.abs(first.scheduledDelayMs - 140) < 0.01);
assert(Math.abs(first.startAtContextTime - 1.14) < 1e-9);
assert.equal(first.underflow, false);
assert(Math.abs(scheduledStarts[0] - 1.14) < 1e-9);

context.currentTime = 1.05;
const second = await api.queueAudioChunk(new Float32Array(1920), 24000);
assert.equal(second.underflow, false);
assert(Math.abs(scheduledStarts[1] - 1.20) < 1e-9);

context.currentTime = 1.30;
const afterGap = await api.queueAudioChunk(new Float32Array(1920), 24000);
assert.equal(afterGap.underflow, true);
assert(Math.abs(scheduledStarts[2] - 1.32) < 1e-9);

api.stopStreamingPlayback();
context.currentTime = 1.50;
const afterReset = await api.queueAudioChunk(new Float32Array(1440), 24000);
assert.equal(afterReset.underflow, false);
assert(Math.abs(afterReset.scheduledDelayMs - 140) < 0.01);
assert(Math.abs(scheduledStarts[3] - 1.64) < 1e-9);
api.stopStreamingPlayback();

context.currentTime = 2.0;
await api.queueAudioChunk(new Float32Array(1200000), 24000);
assert(api.streamingPlaybackDrainTimeoutMs() > 60000, "long queued speech extends the drain watchdog");
assert.equal(api.streamingPlaybackDrainTimeoutMs(1234), 1234, "an explicit bounded test timeout is preserved");
api.stopStreamingPlayback();

const staleAudioTurn = api.turnCoordinator.begin("chat", "file");
staleAudioTurn.voiceBackend = "lfm_audio_natural_tts_streaming";
const staleStartCount = scheduledStarts.length;
const originalResume = context.resume.bind(context);
context.state = "suspended";
context.resume = async () => {
  context.state = "running";
  api.turnCoordinator.cancel(staleAudioTurn, "cancelled_during_audio_resume");
};
await assert.rejects(
  api.queueAudioChunk(new Float32Array(1440), 24000, staleAudioTurn),
  error => api.isAbortError(error)
);
assert.equal(scheduledStarts.length, staleStartCount, "cancelled PCM must not be scheduled after AudioContext resume");
context.resume = originalResume;

context.currentTime = 2.0;
await api.queueAudioChunk(new Float32Array(1440), 24000);
const playbackDrain = api.waitForStreamingPlaybackDrain();
fakeAudioSources.at(-1).dispatch("ended");
assert.deepEqual(await playbackDrain, { status: "drained" });
api.stopStreamingPlayback();

const vadTtsTurn = { id: "vad-tts-unit" };
const vadTtsTerminal = api.createVadTtsTerminalPromise(vadTtsTurn);
assert.equal(api.settleVadTtsTerminal(vadTtsTurn.id, "completed", { monotonic_ms: 2000 }), true);
assert.deepEqual(await vadTtsTerminal, {
  status: "completed",
  monotonic_ms: 2000
});
assert.equal(api.settleVadTtsTerminal(vadTtsTurn.id, "completed"), false);

const nativeWav = api.encodeCanonicalPcm16Wav(new Uint8Array([0, 0]));
const nativeWavBase64 = Buffer.from(nativeWav).toString("base64");
const handoffTelemetryStart = telemetryPayloads.length;
const handoffNativeTtsCallStart = nativeTtsCalls.length;
const handoffAudioSourceStart = fakeAudioSources.length;
const handoffRequests = [];
const originalFetch = globalThis.fetch;
const originalAtob = globalThis.atob;
const originalBtoa = globalThis.btoa;
let handoffAtobCalls = 0;
let handoffWavAtobCalls = 0;
let handoffAudioAtobCalls = 0;
let handoffBtoaCalls = 0;
const handoffSseBody = 'data: {"choices":[{"delta":{"content":"<think>\\n\\n</think>\\n\\nOf course. What do you need?"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
const handoffPcm = new Float32Array(1440);
for (let index = 0; index < handoffPcm.length; index++) {
  handoffPcm[index] = Math.sin(index / 12) * 0.05;
}
const handoffAudioBase64 = Buffer.from(handoffPcm.buffer).toString("base64");
const handoffLfmSseBody = `data: {"choices":[{"delta":{"audio_chunk":{"data":"${handoffAudioBase64}","format":"pcm","sample_rate":24000}},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n`;
try {
  englishResponseStyleElement.value = "exact";
  englishResponseStyleElement.dispatchEvent({ type: "change" });
  autoCompleteNativeTts = false;
  globalThis.atob = data => {
    handoffAtobCalls += 1;
    if (data === nativeWavBase64) handoffWavAtobCalls += 1;
    if (data === handoffAudioBase64) handoffAudioAtobCalls += 1;
    return originalAtob(data);
  };
  globalThis.btoa = data => {
    handoffBtoaCalls += 1;
    return originalBtoa(data);
  };
  globalThis.fetch = async (url, options) => {
    handoffRequests.push({ url, options });
    if (String(url).includes("127.0.0.1:6010/asr")) {
      return {
        ok: true,
        status: 200,
        async json() {
          return companionPayload([
            qnnCandidate("Hello, please help me."),
            russianCandidate("хэлло", [-1.4])
          ]);
        }
      };
    }
    const body = String(url).includes("127.0.0.1:8080/v1/chat/completions")
      ? handoffLfmSseBody
      : handoffSseBody;
    const chunks = [new TextEncoder().encode(body)];
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (chunks.length) return { value: chunks.shift(), done: false };
              return { value: new Uint8Array(), done: true };
            },
            async cancel() {}
          };
        }
      }
    };
  };

  await api.startRecording();
  api.stopRecording();
  await window.onNativeRecording(nativeWavBase64);
  for (const source of fakeAudioSources.slice(handoffAudioSourceStart)) source.dispatch("ended");
  await Promise.resolve();
} finally {
  autoCompleteNativeTts = false;
  englishResponseStyleElement.value = "expressive";
  englishResponseStyleElement.dispatchEvent({ type: "change" });
  globalThis.fetch = originalFetch;
  globalThis.atob = originalAtob;
  globalThis.btoa = originalBtoa;
}

assert.equal(handoffWavAtobCalls, 1, "the companion handoff decodes the input WAV exactly once");
assert.equal(handoffAudioAtobCalls, 2, "each streamed LFM sentence decodes one audio payload");
assert.equal(handoffAtobCalls, 3);
assert.equal(handoffBtoaCalls, 0, "native request handoff must not re-encode the WAV in JavaScript");
assert.equal(handoffRequests.length, 4, "natural voice mode performs ASR, Qwen generation, then sentence-streamed LFM TTS");
assert.match(String(handoffRequests[0].url), /127\.0\.0\.1:6010\/asr\?language=qnn-auto/);
assert.equal(handoffRequests[0].options.headers["Content-Type"], "audio/wav");
assert.deepEqual(
  [...handoffRequests[0].options.body],
  [...nativeWav],
  "the Java-provided WAV must reach the companion byte-for-byte"
);
const responseHandoffPayload = JSON.parse(handoffRequests[1].options.body);
assert.match(String(handoffRequests[1].url), /127\.0\.0\.1:6020\/v1\/chat\/completions/);
assert.equal(responseHandoffPayload.model, "Qwen2.5-1.5B-Instruct-Q4_K_M");
assert.equal("chat_template_kwargs" in responseHandoffPayload, false);
assert.match(responseHandoffPayload.messages[0].content, /Answer the current user directly and accurately/);
assert.equal(typeof responseHandoffPayload.messages[1].content, "string");
assert.match(responseHandoffPayload.messages[1].content, /Hello, please help me\./);
assert.doesNotMatch(responseHandoffPayload.messages[1].content, /\/no_think/);
assert.doesNotMatch(JSON.stringify(responseHandoffPayload), /input_audio|BASE64/);
assert.equal(elements.get("response").textContent, "Of course. What do you need?");
const handoffSpokenText = nativeTtsCalls.slice(handoffNativeTtsCallStart)
  .filter(call => call.method === "queue")
  .map(call => call.text)
  .join(" ");
assert.equal(handoffSpokenText, "", "English natural mode must not invoke Android system TTS");
const lfmTtsPayloads = handoffRequests.slice(2).map(request => JSON.parse(request.options.body));
assert(handoffRequests.slice(2).every(request => /127\.0\.0\.1:8080\/v1\/chat\/completions/.test(String(request.url))));
assert.equal(lfmTtsPayloads.map(payload => payload.messages[1].content).join(" "), "Of course. What do you need?");
assert(lfmTtsPayloads.every(payload => payload.messages[0].content === "Perform TTS. Use the UK female voice."));
assert(lfmTtsPayloads.every(payload => !/<\/?think>/.test(payload.messages[1].content)));
assert(lfmTtsPayloads.every(payload => JSON.stringify(payload.modalities) === '["audio"]'));
assert(lfmTtsPayloads.every(payload => payload.reset_context === true));
const handoffTelemetry = telemetryPayloads.slice(handoffTelemetryStart);
const wavReadyEvent = handoffTelemetry.find(event => event.event === "wav_ready");
const requestStartEvent = handoffTelemetry.find(event => event.event === "request_start");
const languageResolvedEvent = handoffTelemetry.find(event => event.event === "language_resolved");
const completionEvent = handoffTelemetry.find(event => event.event === "lfm_tts_response_complete");
const responseReadyEvent = handoffTelemetry.find(event => event.event === "conversation_request_ready");
assert.equal(wavReadyEvent.details.input, "native_microphone");
assert.equal(wavReadyEvent.details.wav_bytes, nativeWav.length);
assert.equal(requestStartEvent.details.wav_bytes, nativeWav.length);
assert.equal(requestStartEvent.details.lfm_tts_voice, "uk_female");
assert.equal(requestStartEvent.details.backend, "sensevoice_qnn_then_zipformer_ru_cpu_on_ambiguity");
assert.equal(languageResolvedEvent.details.resolved_language, "en");
assert.equal(languageResolvedEvent.details.fallback_used, false);
assert.equal(languageResolvedEvent.details.backend, "sensevoice_qnn");
assert.equal(languageResolvedEvent.details.npu_executed, true);
assert.equal(responseReadyEvent.details.response_input, "asr_text");
assert.equal(responseReadyEvent.details.duplicate_audio_encode, false);
assert.equal(responseReadyEvent.details.response_backend, "qwen2_5_1_5b_q4_k_m_cpu");
assert.equal(completionEvent.details.history_turns, 0);
assert.equal(completionEvent.details.history_mutated, false);
assert.equal(completionEvent.details.voice_backend, "lfm_audio_natural_tts_streaming");
assert.equal(completionEvent.details.response_backend, "qwen2_5_1_5b_q4_k_m_cpu");
assert.equal(completionEvent.details.lfm_tts_chunks_submitted, 2);
assert.equal(completionEvent.details.lfm_tts_chunks_completed, 2);
assert.equal(completionEvent.details.lfm_tts_audio_chunks, 2);
assert.equal(completionEvent.details.lfm_tts_voice, "uk_female");
assert.equal(completionEvent.details.audio_chunks, 2);
assert.equal(completionEvent.details.reasoning_filter.reasoning_blocks_removed, 1);
assert.equal(completionEvent.details.finish_reason, "stop");
assert.deepEqual(api.conversationHistory.snapshot(), []);
assert.equal(api.turnCoordinator.current, null, "a completed manual turn must not leak into the next input");

const directTelemetryStart = telemetryPayloads.length;
const directNativeTtsStart = nativeTtsCalls.length;
const directAudioSourceStart = fakeAudioSources.length;
const directRequests = [];
const directSseBody = `data: {"choices":[{"delta":{"content":"Five.","audio_chunk":{"data":"${handoffAudioBase64}","format":"pcm","sample_rate":24000}},"finish_reason":"stop"}]}

data: [DONE]

`;
try {
  englishResponseStyleElement.value = "expressive";
  englishResponseStyleElement.dispatchEvent({ type: "change" });
  globalThis.fetch = async (url, options) => {
    directRequests.push({ url, options });
    if (String(url).includes("127.0.0.1:6010/asr")) {
      return {
        ok: true,
        status: 200,
        async json() {
          return companionPayload([
            qnnCandidate("What is three plus two?"),
            russianCandidate("", [-2.0])
          ]);
        }
      };
    }
    const chunks = [new TextEncoder().encode(directSseBody)];
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (chunks.length) return { value: chunks.shift(), done: false };
              return { value: new Uint8Array(), done: true };
            },
            async cancel() {}
          };
        }
      }
    };
  };

  await api.startRecording();
  api.stopRecording();
  await window.onNativeRecording(nativeWavBase64);
  for (const source of fakeAudioSources.slice(directAudioSourceStart)) source.dispatch("ended");
  await Promise.resolve();
} finally {
  globalThis.fetch = originalFetch;
}

assert.equal(directRequests.length, 2, "expressive mode performs ASR followed by one interleaved LFM request");
assert.match(String(directRequests[0].url), /127\.0\.0\.1:6010\/asr\?language=qnn-auto/);
assert.match(String(directRequests[1].url), /127\.0\.0\.1:8080\/v1\/chat\/completions/);
const directPayload = JSON.parse(directRequests[1].options.body);
assert.equal(directPayload.model, "LFM2.5-Audio-1.5B");
assert.deepEqual(directPayload.modalities, ["text", "audio"]);
assert.equal(directPayload.reset_context, true);
assert.match(directPayload.messages[0].content, /Respond with interleaved text and audio/);
assert.equal(directPayload.messages[1].content, "What is three plus two?");
assert.doesNotMatch(JSON.stringify(directPayload), /input_audio|Qwen/);
assert.equal(elements.get("response").textContent, "Five.");
assert.equal(
  nativeTtsCalls.slice(directNativeTtsStart).some(call => call.method === "queue"),
  false,
  "expressive mode must not invoke Android or fixed-text TTS"
);
const directTelemetry = telemetryPayloads.slice(directTelemetryStart);
const directReady = directTelemetry.find(event => event.event === "conversation_request_ready");
const directComplete = directTelemetry.find(event => event.event === "lfm_interleaved_response_complete");
assert.equal(directReady.details.response_backend, "lfm2_5_audio_interleaved_q4_cpu");
assert.equal(directReady.details.direct_lfm_interleaved, true);
assert.equal(directReady.details.response_input, "asr_text");
assert.equal(directComplete.details.voice_backend, "lfm_audio_interleaved_streaming");
assert.equal(directComplete.details.response_model, "LFM2.5-Audio-1.5B");
assert.equal(directComplete.details.direct_lfm_interleaved, true);
assert.equal(directComplete.details.audio_chunks, 1);
assert.equal(directComplete.details.finish_reason, "stop");
assert.equal(directComplete.details.first_voiced_pcm_offset_ms, 0);
assert(Number.isFinite(directComplete.details.first_voiced_scheduled_ms));
assert.equal(api.turnCoordinator.current, null, "a completed expressive turn releases ownership after playback drain");

const streamFailureTelemetryStart = telemetryPayloads.length;
const streamFailureAudioStart = fakeAudioSources.length;
try {
  englishResponseStyleElement.value = "expressive";
  englishResponseStyleElement.dispatchEvent({ type: "change" });
  globalThis.fetch = async url => {
    if (String(url).includes("127.0.0.1:6010/asr")) {
      return {
        ok: true,
        status: 200,
        async json() {
          return companionPayload([
            qnnCandidate("What is three plus two?"),
            russianCandidate("", [-2.0])
          ]);
        }
      };
    }
    const chunks = [
      new TextEncoder().encode(`data: {"choices":[{"delta":{"content":"Five.","audio_chunk":{"data":"${handoffAudioBase64}","format":"pcm","sample_rate":24000}}}]}

`),
      new TextEncoder().encode("data: {malformed-json}\n\n")
    ];
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (chunks.length) return { value: chunks.shift(), done: false };
              return { value: new Uint8Array(), done: true };
            },
            async cancel() {}
          };
        }
      }
    };
  };

  await api.startRecording();
  api.stopRecording();
  await window.onNativeRecording(nativeWavBase64);
  await Promise.resolve();
} finally {
  globalThis.fetch = originalFetch;
}
const failedStreamSources = fakeAudioSources.slice(streamFailureAudioStart);
assert(failedStreamSources.length >= 1, "the malformed-stream case queued audio before failing");
assert(failedStreamSources.every(source => source.stopped), "mid-stream failure must stop every queued source");
assert(
  telemetryPayloads.slice(streamFailureTelemetryStart).some(event =>
    event.event === "turn_error" && event.details.stage === "native_recording"
  ),
  "mid-stream failure is surfaced as a turn error"
);

const wrongScriptTelemetryStart = telemetryPayloads.length;
const wrongScriptNativeCallStart = nativeTtsCalls.length;
const wrongScriptLocaleCheckStart = nativeTtsLocaleChecks.length;
let wrongScriptReaderCancelled = false;
const wrongScriptSseBody = 'data: {"choices":[{"delta":{"content":"This answer is in English."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
try {
  api.setMode("chat");
  languageSelectElement.value = "zh";
  languageSelectElement.dispatchEvent({ type: "change" });
  autoCompleteNativeTts = true;
  globalThis.fetch = async url => {
    if (String(url).includes("127.0.0.1:6010/asr")) {
      return {
        ok: true,
        status: 200,
        async json() {
          return companionPayload([
            qnnCandidate("你好，请帮助我。", "<|zh|>"),
            russianCandidate("")
          ]);
        }
      };
    }
    const chunks = [new TextEncoder().encode(wrongScriptSseBody)];
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (chunks.length) return { value: chunks.shift(), done: false };
              return { value: new Uint8Array(), done: true };
            },
            async cancel() { wrongScriptReaderCancelled = true; }
          };
        }
      }
    };
  };

  await api.startRecording();
  api.stopRecording();
  await window.onNativeRecording(nativeWavBase64);
} finally {
  autoCompleteNativeTts = false;
  globalThis.fetch = originalFetch;
  languageSelectElement.value = "auto";
  languageSelectElement.dispatchEvent({ type: "change" });
}
const wrongScriptNativeCalls = nativeTtsCalls.slice(wrongScriptNativeCallStart);
assert(
  nativeTtsLocaleChecks.slice(wrongScriptLocaleCheckStart).includes("zh-CN"),
  "the native voice gate must validate the resolved Mandarin locale"
);
assert.equal(
  wrongScriptNativeCalls.some(call => call.method === "queue"),
  false,
  "an all-Latin Mandarin answer must never reach the native speech queue"
);
assert(
  wrongScriptNativeCalls.some(call => call.method === "cancel" && call.reason === "response_script_mismatch"),
  "the initialized native TTS turn is cancelled on a pre-speech script mismatch"
);
assert.equal(wrongScriptReaderCancelled, true, "the mismatched response stream is cancelled");
const wrongScriptAbort = telemetryPayloads.slice(wrongScriptTelemetryStart)
  .find(event => event.event === "response_quality_abort");
assert.equal(wrongScriptAbort.details.stage, "pre_tts_chunk");
assert.equal(wrongScriptAbort.details.reason, "response_script_mismatch");
const wrongScriptEngine = telemetryPayloads.slice(wrongScriptTelemetryStart)
  .find(event => event.event === "tts_engine_resolved");
assert.equal(wrongScriptEngine.details.tts_policy, "auto");
assert.equal(wrongScriptEngine.details.resolved_tts_engine, "matcha_zh_baker");
assert.equal(api.turnCoordinator.current, null, "a quality-aborted manual turn releases ownership");

const pendingQualityNativeStart = nativeTtsCalls.length;
const splitQualitySse = [
  'data: {"choices":[{"delta":{"content":"这是一个正确的中文句子。"},"finish_reason":null}]}\n\n',
  'data: {"choices":[{"delta":{"content":"This later chunk is wrongly in English."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
].map(value => new TextEncoder().encode(value));
try {
  api.setMode("chat");
  languageSelectElement.value = "zh";
  languageSelectElement.dispatchEvent({ type: "change" });
  autoCompleteNativeTts = false;
  globalThis.fetch = async url => {
    if (String(url).includes("127.0.0.1:6010/asr")) {
      return {
        ok: true,
        status: 200,
        async json() {
          return companionPayload([
            qnnCandidate("你好，请帮助我。", "<|zh|>"),
            russianCandidate("")
          ]);
        }
      };
    }
    const chunks = [...splitQualitySse];
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (chunks.length) return { value: chunks.shift(), done: false };
              return { value: new Uint8Array(), done: true };
            },
            async cancel() {}
          };
        }
      }
    };
  };

  await api.startRecording();
  api.stopRecording();
  await window.onNativeRecording(nativeWavBase64);
  await Promise.resolve();
} finally {
  globalThis.fetch = originalFetch;
  languageSelectElement.value = "auto";
  languageSelectElement.dispatchEvent({ type: "change" });
}
const pendingQualityCalls = nativeTtsCalls.slice(pendingQualityNativeStart);
assert(
  pendingQualityCalls.some(call => call.method === "initialize"),
  "the pending-queue regression must exercise native TTS initialization"
);
assert(
  pendingQualityCalls.some(call => call.method === "cancel" && call.reason === "response_script_mismatch"),
  "the later wrong-script chunk must cancel the native turn"
);
assert.equal(
  pendingQualityCalls.some(call => call.method === "queue"),
  false,
  "safe chunks still waiting for initialization become no-ops after a later quality abort"
);

const incompleteFinishNativeStart = nativeTtsCalls.length;
const incompleteFinishTelemetryStart = telemetryPayloads.length;
const incompleteFinishSse = [
  new TextEncoder().encode('data: {"choices":[{"delta":{"content":"这是一个完整句子。"},"finish_reason":null}]}\n\ndata: [DONE]\n\n')
];
try {
  api.setMode("chat");
  languageSelectElement.value = "zh";
  languageSelectElement.dispatchEvent({ type: "change" });
  autoCompleteNativeTts = false;
  globalThis.fetch = async url => {
    if (String(url).includes("127.0.0.1:6010/asr")) {
      return {
        ok: true,
        status: 200,
        async json() {
          return companionPayload([
            qnnCandidate("你好，请帮助我。", "<|zh|>"),
            russianCandidate("")
          ]);
        }
      };
    }
    const chunks = [...incompleteFinishSse];
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (chunks.length) return { value: chunks.shift(), done: false };
              return { value: new Uint8Array(), done: true };
            },
            async cancel() {}
          };
        }
      }
    };
  };

  await api.startRecording();
  api.stopRecording();
  await window.onNativeRecording(nativeWavBase64);
  await Promise.resolve();
} finally {
  globalThis.fetch = originalFetch;
  languageSelectElement.value = "auto";
  languageSelectElement.dispatchEvent({ type: "change" });
}
const incompleteFinishCalls = nativeTtsCalls.slice(incompleteFinishNativeStart);
assert(
  incompleteFinishCalls.some(call => call.method === "cancel" && call.reason === "generation_incomplete"),
  "a missing terminal stop reason cancels native TTS"
);
assert.equal(
  incompleteFinishCalls.some(call => call.method === "queue"),
  false,
  "a missing terminal stop reason cannot flush or submit a pending speech chunk"
);
const incompleteFinishAbort = telemetryPayloads.slice(incompleteFinishTelemetryStart)
  .find(event => event.event === "response_quality_abort");
assert.equal(incompleteFinishAbort.details.stage, "terminal_finish");
assert.equal(incompleteFinishAbort.details.reason, "generation_incomplete");
assert.equal(api.turnCoordinator.current, null, "an incomplete terminal response releases turn ownership");

assert.equal(api.setMode("asr"), undefined);
assert.equal(api.setInputMode("vad"), true);
assert.equal(elements.get("input-vad").getAttribute("aria-pressed"), "true");
assert.match(elements.get("vad-status").textContent, /Hands-free stopped.*barge-in off/);
const vadCallStart = nativeVadCalls.length;
assert.equal(api.startVadSession(), true);
assert.deepEqual(nativeVadCalls.slice(vadCallStart), [{ method: "start" }]);
assert.equal(api.vadState.requested, true);

const vadSessionId = "vad-session-1";
assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "session_started",
  session_id: vadSessionId,
  monotonic_ms: 3000,
  details: { sample_rate: 16000, frame_samples: 320 }
})), true);
assert.equal(api.vadState.sessionId, vadSessionId);
assert.match(elements.get("vad-status").textContent, /Hands-free active/);
for (const [type, expectedStatus] of [
  ["listening", /Listening/],
  ["speech_confirmed", /Speech detected/],
  ["hangover", /Speech ending/]
]) {
  assert.equal(window.onNativeVadEvent(JSON.stringify({
    schema_version: 1,
    type,
    session_id: vadSessionId,
    monotonic_ms: 3010,
    details: {}
  })), true);
  assert.match(elements.get("vad-status").textContent, expectedStatus);
}

const vadRequests = [];
const vadFetchBefore = globalThis.fetch;
const vadTelemetryStart = telemetryPayloads.length;
try {
  globalThis.fetch = async (url, options) => {
    vadRequests.push({ url, options });
    return {
      ok: true,
      status: 200,
      async json() {
        return companionPayload([
          qnnCandidate("Hello, please help me."),
          russianCandidate("хэлло", [-1.4])
        ]);
      }
    };
  };

  await window.onNativeVadEvent(JSON.stringify({
    schema_version: 1,
    type: "endpoint",
    session_id: vadSessionId,
    monotonic_ms: 3200,
    details: {
      endpoint_reason: "silence_hangover",
      sample_rate: 16000,
      frame_samples: 320,
      wav_bytes: nativeWav.length,
      audio_duration_ms: 620,
      paused: true,
      requires_resume: true,
      wav_base64: nativeWavBase64
    }
  }));
} finally {
  globalThis.fetch = vadFetchBefore;
}

assert.equal(vadRequests.length, 1, "ASR-only hands-free endpoint performs one request");
assert.match(String(vadRequests[0].url), /127\.0\.0\.1:6010\/asr\?language=qnn-auto/);
assert.deepEqual([...vadRequests[0].options.body], [...nativeWav]);
assert.equal(elements.get("response").textContent, "Hello, please help me.");
assert.equal(api.vadState.requested, true, "hands-free session persists after a turn");
assert.equal(nativeVadCalls.at(-1).method, "resume");
const vadTurnTelemetry = telemetryPayloads.slice(vadTelemetryStart);
for (const requiredEvent of [
  "vad_endpoint",
  "wav_ready",
  "vad_output_gate_requested",
  "vad_assistant_output_settled",
  "vad_resume_requested"
]) {
  assert(vadTurnTelemetry.some(event => event.event === requiredEvent), `${requiredEvent} telemetry missing`);
}
const endpointTelemetry = vadTurnTelemetry.find(event => event.event === "vad_endpoint");
assert.equal(endpointTelemetry.input_mode, "vad");
assert.equal(endpointTelemetry.details.vad_backend, "adaptive_energy_java_cpu");
assert.equal(endpointTelemetry.details.inference_backend, "cpu_q4");
assert.equal(endpointTelemetry.details.npu_executed, false);
assert.equal(endpointTelemetry.details.npu_fallback_reason, "broad_auto_requires_cpu_multilingual_asr");
assert.equal(endpointTelemetry.details.barge_in_enabled, false);
assert.equal("wav_base64" in endpointTelemetry.details, false, "telemetry must not duplicate endpoint audio");
assert(nativeVadCalls.some(call => call.method === "output_gate" && call.active === true));
assert(nativeVadCalls.some(call => call.method === "output_gate" && call.active === false));

assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "resumed",
  session_id: vadSessionId,
  monotonic_ms: 3400,
  details: { next_sample_offset: 9920 }
})), true);
assert.match(elements.get("vad-status").textContent, /Listening/);

const cancellableVadTurn = api.turnCoordinator.begin("asr", "vad");
cancellableVadTurn.vadSessionId = vadSessionId;
api.vadState.processingTurn = cancellableVadTurn;
assert.equal(api.stopVadSession("manual_stop_test"), true);
assert.equal(cancellableVadTurn.cancelled, true);
assert.equal(cancellableVadTurn.controller.signal.aborted, true);
assert.equal(api.vadState.requested, false);
assert.equal(api.vadState.stopping, true);
assert.equal(nativeVadCalls.at(-1).method, "stop");
assert.match(elements.get("vad-status").textContent, /Hands-free stopping/);
assert.equal(elements.get("record").disabled, true, "restart stays disabled until native stop acknowledgement");
assert.equal(api.startVadSession(), false, "a new native session cannot race the previous stop callback");

assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "stopped",
  session_id: vadSessionId,
  monotonic_ms: 3500,
  details: { reason: "user_stop" }
})), true);
assert.equal(api.vadState.phase, "stopped");
assert.equal(api.vadState.stopping, false);
assert.equal(elements.get("record").getAttribute("aria-label"), "Start hands-free VAD session");

const postStopCallStart = nativeVadCalls.length;
assert.equal(api.startVadSession(), true);
assert.deepEqual(nativeVadCalls.slice(postStopCallStart), [{ method: "start" }]);
assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "stopped",
  session_id: vadSessionId,
  monotonic_ms: 3600,
  details: { reason: "late_old_session_callback" }
})), false, "a stale stop cannot cancel a newly starting session");
assert.equal(api.vadState.requested, true);
assert.equal(api.stopVadSession("pending_start_cleanup"), true);
assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "stopped",
  session_id: "vad-session-2",
  monotonic_ms: 3700,
  details: { reason: "user_stop" }
})), true, "stop acknowledgement is accepted even when stop preceded session_started");
assert.equal(api.vadState.stopping, false);

assert.equal(api.startVadSession(), true);
assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "session_started",
  session_id: "vad-session-3",
  monotonic_ms: 3800,
  details: { sample_rate: 16000, frame_samples: 320 }
})), true);
assert.equal(window.onNativeVadEvent(JSON.stringify({
  schema_version: 1,
  type: "error",
  session_id: "vad-session-3",
  monotonic_ms: 3810,
  details: { stage: "capture", message: "synthetic native capture failure" }
})), false);
assert.equal(api.vadState.requested, false);
assert.equal(api.vadState.stopping, false, "fatal native errors must not await a stop callback that may never arrive");
assert.equal(api.vadState.sessionId, null);
assert.match(elements.get("vad-status").textContent, /Hands-free error/);
assert.equal(elements.get("record").getAttribute("aria-label"), "Start hands-free VAD session");

console.log("streaming audio UI tests passed");
