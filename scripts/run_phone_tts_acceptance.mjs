#!/usr/bin/env node

/**
 * Physical Android TTS acceptance over an already-exposed WebView CDP endpoint.
 *
 * This script deliberately does not invoke adb. Expose the debuggable app
 * WebView before running it, then point --host/--port at that endpoint.
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const HARNESS_SCHEMA_VERSION = 1;
const ESPEAK_PACKAGE = "com.reecedunn.espeak";

export const DEFAULT_CASES = Object.freeze([
  {
    name: "auto_en_supertonic_voice4",
    policy: "auto",
    locale: "en-US",
    behavior: "complete",
    chunks: ["This English voice is running fully offline on the phone."],
    expectedEngine: "supertonic3_en_v4_s5",
    expectedPolicy: "auto",
    expectedRequestedLocale: "en-US",
    expectedResolvedLocale: "en",
    expectedGenerationLanguage: "en",
    maxFirstChunkMs: 1200,
    maxRtf: 0.8,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  },
  {
    name: "auto_zh_matcha",
    policy: "auto",
    locale: "zh-CN",
    behavior: "complete",
    chunks: ["你好，这是中文离线语音路由测试。"],
    expectedEngine: "matcha_zh_baker",
    expectedPolicy: "auto",
    expectedTerminalStatus: "completed",
  },
  {
    name: "three_chunk_fifo_and_drain",
    policy: "auto",
    locale: "zh-CN",
    behavior: "complete",
    chunks: ["这是第一段。", "这是第二段。", "这是第三段。"],
    expectedEngine: "matcha_zh_baker",
    expectedPolicy: "auto",
    expectedTerminalStatus: "completed",
    validateThreeChunkQueue: true,
  },
  {
    name: "immediate_cancellation",
    policy: "auto",
    locale: "zh-CN",
    behavior: "cancel_immediate",
    chunks: [
      "这是一个用于立即取消测试的较长中文句子，语音不应在取消以后继续完成播放。",
    ],
    expectedEngine: "matcha_zh_baker",
    expectedPolicy: "auto",
    expectedTerminalStatus: "cancelled",
    validateCancellation: true,
  },
  {
    name: "playback_start_cancellation",
    policy: "auto",
    locale: "zh-CN",
    behavior: "cancel_on_start",
    chunks: [
      "这是一个用于播放开始后取消测试的较长中文句子，后续音频必须立即停止并且不能迟到完成。",
    ],
    expectedEngine: "matcha_zh_baker",
    expectedPolicy: "auto",
    expectedTerminalStatus: "cancelled",
    validateCancellation: true,
    requireStartBeforeCancellation: true,
    playbackHoldMs: 500,
  },
  {
    name: "auto_ru_piper_ruslan",
    policy: "auto",
    locale: "ru-RU",
    behavior: "complete",
    chunks: ["Привет. Это проверка русской офлайн речи."],
    expectedEngine: "piper_ru_ruslan",
    expectedPolicy: "auto",
    expectedRequestedLocale: "ru-RU",
    expectedResolvedLocale: "ru",
    expectedGenerationLanguage: "ru",
    maxFirstChunkMs: 1200,
    maxRtf: 0.8,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  },
  {
    name: "audition_en_kokoro_sarah",
    policy: "kokoro_en_sarah_v1",
    locale: "en-US",
    behavior: "complete",
    chunks: ["This is the Kokoro Sarah offline voice audition."],
    expectedEngine: "kokoro_en_sarah_v1",
    expectedPolicy: "kokoro_en_sarah_v1",
    expectedRequestedLocale: "en-US",
    expectedResolvedLocale: "en",
    expectedGenerationLanguage: "en",
    maxFirstChunkMs: 15000,
    maxRtf: 5.0,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  },
  ...[0, 9].map(sid => ({
    name: `audition_en_supertonic_voice_${sid}`,
    policy: `supertonic3_en_v${sid}_s5`,
    locale: "en-US",
    behavior: "complete",
    chunks: [`This is Supertonic English voice ${sid}.`],
    expectedEngine: `supertonic3_en_v${sid}_s5`,
    expectedPolicy: `supertonic3_en_v${sid}_s5`,
    expectedRequestedLocale: "en-US",
    expectedResolvedLocale: "en",
    expectedGenerationLanguage: "en",
    maxFirstChunkMs: 5000,
    maxRtf: 2.0,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  })),
  ...[0, 9].map(sid => ({
    name: `audition_ru_supertonic_voice_${sid}`,
    policy: `supertonic3_ru_v${sid}_s5`,
    locale: "ru-RU",
    behavior: "complete",
    chunks: [`Привет. Это русская офлайн речь, голос ${sid}.`],
    expectedEngine: `supertonic3_ru_v${sid}_s5`,
    expectedPolicy: `supertonic3_ru_v${sid}_s5`,
    expectedRequestedLocale: "ru-RU",
    expectedResolvedLocale: "ru",
    expectedGenerationLanguage: "ru",
    maxFirstChunkMs: 5000,
    maxRtf: 2.0,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  })),
  ...[
    ["irina", "piper_ru_irina"],
    ["ruslan", "piper_ru_ruslan"],
  ].map(([voice, policy]) => ({
    name: `audition_ru_piper_${voice}`,
    policy,
    locale: "ru-RU",
    behavior: "complete",
    chunks: [`Привет. Это голос ${voice}, работающий полностью офлайн.`],
    expectedEngine: policy,
    expectedPolicy: policy,
    expectedRequestedLocale: "ru-RU",
    expectedResolvedLocale: "ru",
    expectedGenerationLanguage: "ru",
    maxFirstChunkMs: 5000,
    maxRtf: 2.0,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  })),
  {
    name: "auto_ja_supertonic",
    policy: "auto",
    locale: "ja-JP",
    behavior: "complete",
    chunks: ["答えは三十二です。"],
    expectedEngine: "supertonic3_ja_s5",
    expectedPolicy: "auto",
    expectedRequestedLocale: "ja-JP",
    expectedResolvedLocale: "ja",
    expectedGenerationLanguage: "ja",
    maxFirstChunkMs: 1200,
    maxRtf: 0.8,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  },
  {
    name: "auto_ko_supertonic",
    policy: "auto",
    locale: "ko-KR",
    behavior: "complete",
    chunks: ["답은 32입니다."],
    expectedEngine: "supertonic3_ko_s5",
    expectedPolicy: "auto",
    expectedRequestedLocale: "ko-KR",
    expectedResolvedLocale: "ko",
    expectedGenerationLanguage: "ko",
    maxFirstChunkMs: 1200,
    maxRtf: 0.8,
    maxDroppedChunks: 0,
    expectedTerminalStatus: "completed",
  },
  {
    name: "auto_fr_espeak_fallback",
    policy: "auto",
    locale: "fr-FR",
    behavior: "complete",
    chunks: ["Bonjour, ceci est un test vocal entièrement hors ligne."],
    expectedEngine: "system",
    expectedEnginePackage: ESPEAK_PACKAGE,
    expectedPolicy: "auto",
    expectedTerminalStatus: "completed",
  },
  {
    name: "manual_matcha_pinned_for_ru_locale",
    policy: "matcha_zh_baker",
    locale: "ru-RU",
    behavior: "complete",
    chunks: ["手动选择的中文语音保持固定。"],
    expectedEngine: "matcha_zh_baker",
    expectedPolicy: "matcha_zh_baker",
    expectedRequestedLocale: "ru-RU",
    expectedResolvedLocale: "zh",
    expectedTerminalStatus: "completed",
  },
]);

function usage() {
  return `Usage: node scripts/run_phone_tts_acceptance.mjs [options]

The app WebView CDP endpoint must already be reachable; this script never runs adb.

Options:
  --host HOST                 CDP HTTP host (default: 127.0.0.1)
  --port PORT                 CDP HTTP port (default: 9224)
  --output PATH               JSON result path (default: timestamped report)
  --timeout-ms MS             Per-case CDP timeout (default: 150000)
  --event-timeout-ms MS       Native ready/playback timeout (default: 120000)
  --late-event-grace-ms MS    Observation window after cancellation (default: 2500)
  --cooldown-ms MS            Pause between physical cases (default: 500)
  --target-url-includes TEXT  Restrict CDP page selection by URL/title substring
  --cases NAME[,NAME...]      Run only named cases
  --list-cases                Print available case names and exit
  --help                      Show this help
`;
}

function timestampForPath(date = new Date()) {
  return date.toISOString().replaceAll(":", "").replaceAll("-", "").replace(/\.\d{3}Z$/u, "Z");
}

export function parseArgs(argv) {
  const args = {
    host: "127.0.0.1",
    port: 9224,
    timeoutMs: 150_000,
    eventTimeoutMs: 120_000,
    lateEventGraceMs: 2_500,
    cooldownMs: 500,
    targetUrlIncludes: null,
    caseNames: null,
    listCases: false,
    help: false,
    output: path.resolve(
      "reports/android_phone/tts_acceptance",
      `phone_tts_acceptance_${timestampForPath()}.json`,
    ),
  };
  const valueOptions = new Set([
    "--host",
    "--port",
    "--output",
    "--timeout-ms",
    "--event-timeout-ms",
    "--late-event-grace-ms",
    "--cooldown-ms",
    "--target-url-includes",
    "--cases",
  ]);
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--help" || key === "-h") {
      args.help = true;
      continue;
    }
    if (key === "--list-cases") {
      args.listCases = true;
      continue;
    }
    if (!valueOptions.has(key)) throw new Error(`Unknown argument: ${key}`);
    if (index + 1 >= argv.length) throw new Error(`Missing value for ${key}`);
    const value = argv[index + 1];
    index += 1;
    if (key === "--host") args.host = value;
    else if (key === "--port") args.port = Number(value);
    else if (key === "--output") args.output = path.resolve(value);
    else if (key === "--timeout-ms") args.timeoutMs = Number(value);
    else if (key === "--event-timeout-ms") args.eventTimeoutMs = Number(value);
    else if (key === "--late-event-grace-ms") args.lateEventGraceMs = Number(value);
    else if (key === "--cooldown-ms") args.cooldownMs = Number(value);
    else if (key === "--target-url-includes") args.targetUrlIncludes = value;
    else if (key === "--cases") {
      args.caseNames = value.split(",").map(item => item.trim()).filter(Boolean);
      if (!args.caseNames.length) throw new Error("--cases must name at least one case");
    }
  }
  for (const [name, value] of [
    ["port", args.port],
    ["timeout-ms", args.timeoutMs],
    ["event-timeout-ms", args.eventTimeoutMs],
    ["late-event-grace-ms", args.lateEventGraceMs],
    ["cooldown-ms", args.cooldownMs],
  ]) {
    if (!Number.isFinite(value) || value < 0 || (name === "port" && value < 1)) {
      throw new Error(`Invalid --${name}: ${value}`);
    }
  }
  return args;
}

function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

function persist(outputPath, payload) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, outputPath);
}

class CdpClient {
  constructor(url, defaultTimeoutMs) {
    this.url = url;
    this.defaultTimeoutMs = defaultTimeoutMs;
    this.nextId = 1;
    this.pending = new Map();
    this.socket = null;
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener("message", event => {
      const message = JSON.parse(String(event.data));
      if (!message.id) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(`CDP ${pending.method}: ${message.error.message}`));
      else pending.resolve(message.result);
    });
    this.socket.addEventListener("close", () => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(new Error("CDP WebSocket closed"));
      }
      this.pending.clear();
    });
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Timed out opening CDP WebSocket")), 10_000);
      this.socket.addEventListener("open", () => {
        clearTimeout(timer);
        resolve();
      }, { once: true });
      this.socket.addEventListener("error", event => {
        clearTimeout(timer);
        reject(new Error(`CDP WebSocket failed: ${event.message || "unknown error"}`));
      }, { once: true });
    });
    await this.call("Runtime.enable");
  }

  call(method, params = {}, timeoutMs = this.defaultTimeoutMs) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer, method });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression, timeoutMs = this.defaultTimeoutMs) {
    const result = await this.call("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
      replMode: false,
    }, timeoutMs);
    if (result.exceptionDetails) {
      throw new Error(
        result.exceptionDetails.exception?.description
          || result.exceptionDetails.text
          || "Runtime.evaluate failed",
      );
    }
    return result.result?.value;
  }

  close() {
    this.socket?.close();
  }
}

async function discoverTargets(host, port) {
  let response = await fetch(`http://${host}:${port}/json/list`, {
    signal: AbortSignal.timeout(10_000),
  });
  if (response.status === 404) {
    response = await fetch(`http://${host}:${port}/json`, {
      signal: AbortSignal.timeout(10_000),
    });
  }
  if (!response.ok) throw new Error(`CDP target discovery failed: HTTP ${response.status}`);
  const targets = await response.json();
  return targets.filter(target => target.type === "page" && target.webSocketDebuggerUrl);
}

async function connectToAppTarget(args) {
  let targets = await discoverTargets(args.host, args.port);
  if (args.targetUrlIncludes) {
    const needle = args.targetUrlIncludes.toLocaleLowerCase();
    targets = targets.filter(target => `${target.url || ""}\n${target.title || ""}`
      .toLocaleLowerCase().includes(needle));
  }
  if (!targets.length) throw new Error("No matching debuggable WebView page target was found");
  const errors = [];
  for (const target of targets) {
    const client = new CdpClient(target.webSocketDebuggerUrl, args.timeoutMs);
    try {
      await client.connect();
      const hasBridge = await client.evaluate(String.raw`(() => ({
        nativeAudio: Boolean(window.NativeAudio),
        initializeTts: typeof window.NativeAudio?.initializeTts === "function",
        coordinator: typeof turnCoordinator !== "undefined"
      }))()`, 15_000);
      if (hasBridge?.nativeAudio && hasBridge?.initializeTts && hasBridge?.coordinator) {
        return { client, target };
      }
      errors.push(`${target.title || target.url}: NativeAudio/turnCoordinator unavailable`);
    } catch (error) {
      errors.push(`${target.title || target.url}: ${error.message || error}`);
    }
    client.close();
  }
  throw new Error(`No LFM demo WebView target was usable: ${errors.join("; ")}`);
}

export const PREFLIGHT_EXPRESSION = String.raw`(() => {
  const parseJson = value => {
    try { return JSON.parse(value || "{}"); } catch (_) { return null; }
  };
  const requiredMethods = [
    "listTtsEngines", "setTtsEngine", "initializeTts", "queueTtsChunk",
    "finishTtsResponse", "cancelTts"
  ];
  const methods = Object.fromEntries(requiredMethods.map(name => [
    name, typeof window.NativeAudio?.[name] === "function"
  ]));
  let busy = null;
  let currentTurn = null;
  let clientTtsTurns = null;
  let nativeCapture = null;
  let mediaRecorderState = null;
  let vadRequested = null;
  let vadStopping = null;
  let browserPlaybackSources = null;
  let htmlAudioPlaying = null;
  const inspection = {
    inputBusy: false,
    currentTurn: false,
    clientTtsTurns: false,
    nativeCapture: false,
    mediaRecorderState: false,
    vadState: false,
    browserPlaybackSources: false,
    htmlAudioPlaying: false
  };
  try {
    busy = Boolean(inputBusy);
    inspection.inputBusy = true;
  } catch (_) {}
  try {
    currentTurn = turnCoordinator.current ? {
      id: turnCoordinator.current.id,
      generation: turnCoordinator.current.generation,
      mode: turnCoordinator.current.mode,
      inputKind: turnCoordinator.current.inputKind
    } : null;
    inspection.currentTurn = true;
  } catch (_) {}
  try {
    clientTtsTurns = ttsClientTurns.size;
    inspection.clientTtsTurns = true;
  } catch (_) {}
  try {
    nativeCapture = Boolean(nativeRecording);
    inspection.nativeCapture = true;
  } catch (_) {}
  try {
    mediaRecorderState = mediaRecorder?.state || null;
    inspection.mediaRecorderState = true;
  } catch (_) {}
  try {
    vadRequested = Boolean(vadState.requested);
    vadStopping = Boolean(vadState.stopping);
    inspection.vadState = true;
  } catch (_) {}
  try {
    browserPlaybackSources = playbackSources.size;
    inspection.browserPlaybackSources = true;
  } catch (_) {}
  try {
    htmlAudioPlaying = Boolean(audioEl && !audioEl.paused && !audioEl.ended);
    inspection.htmlAudioPlaying = true;
  } catch (_) {}
  return {
    title: document.title,
    url: location.href,
    userAgent: navigator.userAgent,
    methods,
    engineCatalog: parseJson(window.NativeAudio?.listTtsEngines?.()),
    appState: {
      inputBusy: busy,
      currentTurn,
      clientTtsTurns,
      nativeCapture,
      mediaRecorderState,
      vadRequested,
      vadStopping,
      browserPlaybackSources,
      htmlAudioPlaying,
      inspection
    }
  };
})()`;

export const INSTALL_HOOK_EXPRESSION = String.raw`(() => {
  const old = window.__lfmPhysicalTtsAcceptance;
  if (old?.handler && window.onNativeTtsEvent === old.handler) {
    window.onNativeTtsEvent = old.originalHandler;
  }
  const state = {
    schemaVersion: 1,
    installedWallTime: new Date().toISOString(),
    installedPerformanceMs: performance.now(),
    originalHandler: window.onNativeTtsEvent,
    originalHandlerErrors: [],
    events: [],
    listeners: new Set(),
    nextSequence: 0,
    activeTurn: null
  };
  state.waitFor = (predicate, fromSequence, timeoutMs, label) => {
    const existing = state.events.find(record =>
      record.sequence >= fromSequence && predicate(record)
    );
    if (existing) return Promise.resolve(existing);
    return new Promise((resolve, reject) => {
      const waiter = record => {
        if (record.sequence < fromSequence || !predicate(record)) return;
        clearTimeout(timer);
        state.listeners.delete(waiter);
        resolve(record);
      };
      const timer = setTimeout(() => {
        state.listeners.delete(waiter);
        reject(new Error("Timed out waiting for " + label + " after " + timeoutMs + " ms"));
      }, timeoutMs);
      state.listeners.add(waiter);
    });
  };
  state.handler = raw => {
    let event = null;
    let parseError = null;
    try { event = typeof raw === "string" ? JSON.parse(raw) : raw; }
    catch (error) { parseError = String(error?.message || error); }
    const record = {
      sequence: state.nextSequence++,
      captured_wall_time: new Date().toISOString(),
      captured_performance_ms: performance.now(),
      event,
      raw: typeof raw === "string" ? raw : JSON.stringify(raw),
      parse_error: parseError
    };
    state.events.push(record);
    for (const listener of [...state.listeners]) {
      try { listener(record); } catch (_) {}
    }
    if (typeof state.originalHandler === "function") {
      try { state.originalHandler.call(window, raw); }
      catch (error) {
        state.originalHandlerErrors.push({
          at: new Date().toISOString(),
          message: String(error?.message || error)
        });
      }
    }
  };
  window.__lfmPhysicalTtsAcceptance = state;
  window.onNativeTtsEvent = state.handler;
  return {
    installed: true,
    replacedExistingHarness: Boolean(old),
    originalHandlerPresent: typeof state.originalHandler === "function",
    installedWallTime: state.installedWallTime
  };
})()`;

export function buildCaseExpression(testCase, options) {
  const config = JSON.stringify({
    ...testCase,
    eventTimeoutMs: options.eventTimeoutMs,
    lateEventGraceMs: options.lateEventGraceMs,
  });
  return String.raw`(async () => {
    const config = ${config};
    const state = window.__lfmPhysicalTtsAcceptance;
    if (!state?.waitFor || window.onNativeTtsEvent !== state.handler) {
      throw new Error("The physical TTS acceptance event hook is not installed");
    }
    const nativeAudio = window.NativeAudio;
    const parseJson = value => {
      try { return JSON.parse(value || "{}"); } catch (_) { return null; }
    };
    const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
    const typeOf = record => record?.event?.type || null;
    const turnMatches = (record, turnId) => record?.event?.turn_id === turnId;
    const startedAt = performance.now();
    const fromSequence = state.nextSequence;
    const result = {
      name: config.name,
      policy: config.policy,
      locale: config.locale,
      behavior: config.behavior,
      startedAt: new Date().toISOString(),
      turnId: null,
      generation: null,
      driver: {
        setPolicyAccepted: false,
        queueDispatchSpanMs: null,
        cancelIssuedSequence: null,
        cancelIssuedPerformanceMs: null,
        cancellationTrigger: null,
        startCapturedPerformanceMs: null,
        postStartHoldMs: null,
        terminalVerified: false,
        cleanupIncomplete: false
      },
      catalogBefore: parseJson(nativeAudio.listTtsEngines()),
      catalogAfterReady: null,
      catalogAfter: null,
      captureStartSequence: fromSequence,
      captureEndSequence: null,
      capturedEvents: [],
      events: [],
      thrown: null
    };
    let turn = null;
    try {
      result.driver.setPolicyAccepted = Boolean(nativeAudio.setTtsEngine(config.policy));
      if (!result.driver.setPolicyAccepted) {
        throw new Error("NativeAudio rejected TTS policy " + config.policy);
      }
      // setTtsEngine posts cancellation/model-release work back to the WebView.
      // Give that work a task boundary before allocating the next generation.
      await delay(100);
      if (turnCoordinator.current) {
        throw new Error("The app became busy before the acceptance turn began");
      }
      turn = turnCoordinator.begin("tts", "cdp_tts_acceptance");
      result.turnId = turn.id;
      result.generation = turn.generation;
      state.activeTurn = { id: turn.id, generation: turn.generation, turn };
      nativeAudio.initializeTts(config.locale, turn.id, turn.generation);
      const initOutcome = await state.waitFor(
        record => turnMatches(record, turn.id)
          && ["ready", "error", "terminal"].includes(typeOf(record)),
        fromSequence,
        config.eventTimeoutMs,
        "TTS ready/error for " + config.name
      );
      if (typeOf(initOutcome) !== "ready") {
        const detail = initOutcome?.event?.details || {};
        throw new Error(detail.message || "TTS initialization ended with " + typeOf(initOutcome));
      }
      if (!turnCoordinator.isCurrent(turn)) {
        throw new Error("The acceptance turn was replaced while TTS initialized");
      }
      result.catalogAfterReady = parseJson(nativeAudio.listTtsEngines());
      const chunks = Array.isArray(config.chunks) ? config.chunks : [];
      const chunkIds = chunks.map((_, index) => "chunk-" + index);
      if (config.behavior === "complete") {
        const queueStarted = performance.now();
        chunks.forEach((text, index) => {
          nativeAudio.queueTtsChunk(text, turn.id, chunkIds[index], turn.generation);
        });
        result.driver.queueDispatchSpanMs = performance.now() - queueStarted;
        nativeAudio.finishTtsResponse(turn.id, turn.generation);
      } else if (config.behavior === "cancel_immediate") {
        const queueStarted = performance.now();
        nativeAudio.queueTtsChunk(chunks[0], turn.id, chunkIds[0], turn.generation);
        result.driver.queueDispatchSpanMs = performance.now() - queueStarted;
        result.driver.cancelIssuedSequence = state.nextSequence;
        result.driver.cancelIssuedPerformanceMs = performance.now();
        result.driver.cancellationTrigger = "immediate_after_queue_dispatch";
        nativeAudio.cancelTts(turn.id, turn.generation, "cdp_immediate_cancel");
      } else if (config.behavior === "cancel_on_start") {
        const queueStarted = performance.now();
        nativeAudio.queueTtsChunk(chunks[0], turn.id, chunkIds[0], turn.generation);
        result.driver.queueDispatchSpanMs = performance.now() - queueStarted;
        nativeAudio.finishTtsResponse(turn.id, turn.generation);
        const startOutcome = await state.waitFor(
          record => turnMatches(record, turn.id)
            && ["utterance_start", "utterance_error", "error", "terminal"].includes(typeOf(record)),
          fromSequence,
          config.eventTimeoutMs,
          "playback start for " + config.name
        );
        if (typeOf(startOutcome) !== "utterance_start") {
          const detail = startOutcome?.event?.details || {};
          throw new Error(
            detail.message || "Playback ended with " + typeOf(startOutcome) + " before start"
          );
        }
        await state.waitFor(
          record => turnMatches(record, turn.id)
            && ["first_audio_start_proxy", "utterance_error", "error", "terminal"]
              .includes(typeOf(record)),
          startOutcome.sequence,
          config.eventTimeoutMs,
          "first-audio proxy for " + config.name
        ).then(proxyOutcome => {
          if (typeOf(proxyOutcome) !== "first_audio_start_proxy") {
            const detail = proxyOutcome?.event?.details || {};
            throw new Error(
              detail.message || "Playback ended with " + typeOf(proxyOutcome)
                + " before the first-audio proxy"
            );
          }
        });
        result.driver.startCapturedPerformanceMs = startOutcome.captured_performance_ms;
        const holdMs = Math.max(0, Number(config.playbackHoldMs || 0));
        await delay(holdMs);
        result.driver.postStartHoldMs = performance.now()
          - result.driver.startCapturedPerformanceMs;
        const endedDuringHold = state.events.find(record =>
          record.sequence > startOutcome.sequence
            && turnMatches(record, turn.id)
            && ["utterance_done", "utterance_error", "error", "terminal"]
              .includes(typeOf(record))
        );
        if (endedDuringHold) {
          throw new Error(
            "TTS ended before the post-start playback window elapsed: "
              + typeOf(endedDuringHold)
          );
        }
        // The bridge exposes first valid PCM, not AudioTrack's playback head.
        // Holding beyond the engine's prebuffer before cancellation exercises
        // the physical playback phase without requiring an app-source hook.
        result.driver.cancelIssuedSequence = state.nextSequence;
        result.driver.cancelIssuedPerformanceMs = performance.now();
        result.driver.cancellationTrigger = "post_start_playback_window";
        nativeAudio.cancelTts(turn.id, turn.generation, "cdp_playback_start_cancel");
      } else {
        throw new Error("Unsupported case behavior: " + config.behavior);
      }
      await state.waitFor(
        record => turnMatches(record, turn.id) && typeOf(record) === "terminal",
        fromSequence,
        config.eventTimeoutMs,
        "terminal event for " + config.name
      );
      result.driver.terminalVerified = true;
      if (config.validateCancellation) await delay(config.lateEventGraceMs);
    } catch (error) {
      result.thrown = {
        name: error?.name || "Error",
        message: String(error?.message || error),
        stack: error?.stack || null
      };
      if (turn) {
        const alreadyTerminal = state.events.some(record =>
          turnMatches(record, turn.id) && typeOf(record) === "terminal"
        );
        if (alreadyTerminal) {
          result.driver.terminalVerified = true;
        } else {
          try {
            const cleanupFrom = state.nextSequence;
            nativeAudio.cancelTts(turn.id, turn.generation, "cdp_case_cleanup");
            await state.waitFor(
              record => turnMatches(record, turn.id) && typeOf(record) === "terminal",
              cleanupFrom,
              Math.min(config.eventTimeoutMs, 10_000),
              "cleanup terminal for " + config.name
            );
            result.driver.terminalVerified = true;
          } catch (_) {}
        }
      }
    } finally {
      if (turn && result.driver.terminalVerified) {
        try { turnCoordinator.markTerminal(turn); } catch (_) {}
        try { turnCoordinator.clearIfCurrent(turn); } catch (_) {}
        state.activeTurn = null;
      } else if (turn) {
        result.driver.cleanupIncomplete = true;
      }
      result.catalogAfter = parseJson(nativeAudio.listTtsEngines());
      result.captureEndSequence = state.nextSequence;
      result.capturedEvents = state.events.filter(record =>
        record.sequence >= fromSequence && record.sequence < result.captureEndSequence
      );
      result.events = result.capturedEvents.filter(record =>
        !turn || turnMatches(record, turn.id)
      );
      result.originalHandlerErrors = [...state.originalHandlerErrors];
      result.elapsedMs = performance.now() - startedAt;
      result.finishedAt = new Date().toISOString();
    }
    return result;
  })()`;
}

function eventType(record) {
  return record?.event?.type || null;
}

function eventDetails(record) {
  return record?.event?.details || {};
}

function resolvedEngine(record) {
  const details = eventDetails(record);
  return details.tts_engine_id || details.resolved_engine_id || details.engine_id
    || details.engine || details.engine_package || null;
}

function orderedChunkIds(records) {
  return records.map(record => eventDetails(record).chunk_id ?? null);
}

function arraysEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

export function validateCase(testCase, result) {
  const records = Array.isArray(result?.events) ? result.events : [];
  const capturedRecords = Array.isArray(result?.capturedEvents)
    ? result.capturedEvents : records;
  const byType = type => records.filter(record => eventType(record) === type);
  const ready = byType("ready")[0] || null;
  const terminals = byType("terminal");
  const terminal = terminals[0] || null;
  const queued = byType("chunk_queued");
  const starts = byType("utterance_start");
  const dones = byType("utterance_done");
  const queueClosed = byType("queue_closed");
  const queueCancelled = byType("queue_cancelled");
  const engineMetrics = byType("engine_metrics");
  const errors = records.filter(record => ["error", "utterance_error"].includes(eventType(record)));
  const expectedChunkIds = (testCase.chunks || []).map((_, index) => `chunk-${index}`);
  const terminalSequence = terminal?.sequence ?? -1;
  const lastDoneSequence = dones.length ? Math.max(...dones.map(record => record.sequence)) : -1;
  const cancelSequence = result?.driver?.cancelIssuedSequence;
  const lateDone = Number.isFinite(cancelSequence)
    ? dones.filter(record => record.sequence >= cancelSequence)
    : [];
  const lateStart = Number.isFinite(cancelSequence)
    ? starts.filter(record => record.sequence >= cancelSequence)
    : [];
  const generationRequiredTypes = new Set([
    "ready", "error", "route_fallback", "chunk_queued", "queue_closed",
    "queue_cancelled", "utterance_start", "first_audio_start_proxy",
    "utterance_done", "utterance_error", "engine_metrics", "terminal",
    "stale_ignored",
  ]);
  const generationRecords = records.filter(record => generationRequiredTypes.has(eventType(record)));
  const generationFieldsPresent = generationRecords.length > 0
    && generationRecords.every(record => Number.isFinite(Number(eventDetails(record).generation)));
  const generationConsistent = generationFieldsPresent
    && generationRecords.every(record =>
      Number(eventDetails(record).generation) === Number(result?.generation)
    );
  const recordForChunk = (items, chunkId) =>
    items.find(record => eventDetails(record).chunk_id === chunkId) || null;
  const maxSequence = items => items.length
    ? Math.max(...items.map(record => record.sequence)) : -1;
  const assertions = {
    driverCompletedWithoutError: !result?.thrown,
    driverVerifiedTerminal: result?.driver?.terminalVerified === true,
    driverCleanupComplete: result?.driver?.cleanupIncomplete !== true,
    policyAccepted: result?.driver?.setPolicyAccepted === true,
    capturedEventsParseClean: capturedRecords.every(record =>
      !record?.parse_error && record?.event && typeof record.event === "object"
    ),
    capturedEventsBelongToCaseTurn: capturedRecords.every(record =>
      record?.event?.turn_id === result?.turnId
    ),
    eventSchemaValid: records.length > 0 && records.every(record =>
      record?.event?.schema_version === 1
        && typeof record.event.type === "string"
        && record.event.type.length > 0
        && record.event.turn_id === result?.turnId
    ),
    originalHandlerStayedHealthy: (result?.originalHandlerErrors || []).length === 0,
    readySeenExactlyOnce: byType("ready").length === 1,
    expectedPolicyResolved: eventDetails(ready).tts_policy === testCase.expectedPolicy,
    expectedEngineResolved: resolvedEngine(ready) === testCase.expectedEngine,
    catalogReportsResolvedEngine: result?.catalogAfterReady?.resolved === testCase.expectedEngine,
    generationFieldsPresent,
    generationConsistent,
    noNativeErrorEvent: errors.length === 0,
    terminalSeenExactlyOnce: terminals.length === 1,
    expectedTerminalStatus: eventDetails(terminal).status === testCase.expectedTerminalStatus,
    terminalAfterReady: Boolean(ready && terminal && terminal.sequence > ready.sequence),
  };
  if (testCase.expectedEnginePackage) {
    assertions.expectedEnginePackage = eventDetails(ready).engine_package
      === testCase.expectedEnginePackage;
  }
  if (testCase.expectedRequestedLocale) {
    assertions.expectedRequestedLocale = eventDetails(ready).requested_locale
      === testCase.expectedRequestedLocale;
  }
  if (testCase.expectedResolvedLocale) {
    assertions.expectedResolvedLocale = eventDetails(ready).resolved_locale
      === testCase.expectedResolvedLocale;
  }
  const neuralMetricsRequired = testCase.behavior === "complete"
    && Boolean(testCase.expectedEngine)
    && testCase.expectedEngine !== "system";
  const engineMetricsRequired = neuralMetricsRequired
    || Boolean(testCase.expectedGenerationLanguage)
    || Number.isFinite(testCase.maxFirstChunkMs)
    || Number.isFinite(testCase.maxRtf)
    || Number.isFinite(testCase.maxDroppedChunks);
  if (engineMetricsRequired) {
    assertions.engineMetricsSeen = engineMetrics.length === expectedChunkIds.length;
  }
  if (neuralMetricsRequired) {
    assertions.engineMetricsEngineMatched = engineMetrics.length === expectedChunkIds.length
      && engineMetrics.every(record =>
        eventDetails(record).engine_id === testCase.expectedEngine
      );
  }
  if (testCase.expectedGenerationLanguage) {
    assertions.expectedGenerationLanguage = engineMetrics.length > 0
      && engineMetrics.every(record => eventDetails(record).generation_language
        === testCase.expectedGenerationLanguage);
  }
  if (engineMetrics.length > 0) {
    assertions.engineMetricsGenerationMatched = engineMetrics.length > 0
      && engineMetrics.every(record =>
        Number.isFinite(Number(eventDetails(record).generation))
        && Number(eventDetails(record).generation) === Number(result?.generation)
      );
  }
  if (Number.isFinite(testCase.maxFirstChunkMs)) {
    assertions.engineMetricsFirstChunkWithinLimit = engineMetrics.length > 0
      && engineMetrics.every(record => {
        const value = Number(eventDetails(record).first_chunk_ms);
        return Number.isFinite(value) && value >= 0 && value <= testCase.maxFirstChunkMs;
      });
  }
  if (Number.isFinite(testCase.maxRtf)) {
    assertions.engineMetricsRtfWithinLimit = engineMetrics.length > 0
      && engineMetrics.every(record => {
        const value = Number(eventDetails(record).rtf);
        return Number.isFinite(value) && value >= 0 && value <= testCase.maxRtf;
      });
  }
  if (Number.isFinite(testCase.maxDroppedChunks)) {
    assertions.engineMetricsDroppedChunksWithinLimit = engineMetrics.length > 0
      && engineMetrics.every(record => {
        const value = Number(eventDetails(record).dropped_chunks);
        return Number.isFinite(value) && value >= 0
          && value <= testCase.maxDroppedChunks;
      });
  }
  if (testCase.behavior === "complete") {
    assertions.allChunksAcknowledged = queued.length === expectedChunkIds.length;
    assertions.allChunksStarted = starts.length === expectedChunkIds.length;
    assertions.allChunksDone = dones.length === expectedChunkIds.length;
    assertions.queuedInOrder = arraysEqual(orderedChunkIds(queued), expectedChunkIds);
    assertions.startedInOrder = arraysEqual(orderedChunkIds(starts), expectedChunkIds);
    assertions.doneInOrder = arraysEqual(orderedChunkIds(dones), expectedChunkIds);
    assertions.queueClosedExactlyOnce = queueClosed.length === 1;
    assertions.perChunkLifecycleOrdered = expectedChunkIds.every(chunkId => {
      const queuedRecord = recordForChunk(queued, chunkId);
      const startRecord = recordForChunk(starts, chunkId);
      const doneRecord = recordForChunk(dones, chunkId);
      return Boolean(queuedRecord && startRecord && doneRecord
        && queuedRecord.sequence < startRecord.sequence
        && startRecord.sequence < doneRecord.sequence);
    });
    assertions.queueClosedAfterAllAcks = queued.length === expectedChunkIds.length
      && queueClosed.length === 1
      && queueClosed[0].sequence > maxSequence(queued);
    assertions.terminalAfterQueueClosedAndAllDone = dones.length === expectedChunkIds.length
      && queueClosed.length === 1
      && terminalSequence > queueClosed[0].sequence
      && terminalSequence > lastDoneSequence;
  }
  if (testCase.validateThreeChunkQueue) {
    assertions.exactlyThreeRapidDispatches = expectedChunkIds.length === 3
      && Number.isFinite(result?.driver?.queueDispatchSpanMs)
      && result.driver.queueDispatchSpanMs <= 500;
    assertions.donePendingCountsAreCumulative = arraysEqual(
      dones.map(record => Number(eventDetails(record).pending_utterances)),
      [2, 1, 0],
    );
  }
  if (testCase.validateCancellation) {
    const expectedReason = testCase.behavior === "cancel_immediate"
      ? "cdp_immediate_cancel" : "cdp_playback_start_cancel";
    const cancelledDetails = eventDetails(queueCancelled[0]);
    assertions.cancelWasIssued = Number.isFinite(cancelSequence);
    assertions.cancelledChunkWasAcknowledged = queued.length === 1
      && orderedChunkIds(queued)[0] === "chunk-0";
    assertions.queueCancelledExactlyOnce = queueCancelled.length === 1;
    assertions.queueCancelledAfterCancelDispatch = queueCancelled.length === 1
      && Number.isFinite(cancelSequence)
      && queueCancelled[0].sequence >= cancelSequence;
    assertions.queueCancelledAfterQueueAck = queued.length === 1
      && queueCancelled.length === 1
      && queueCancelled[0].sequence > queued[0].sequence;
    assertions.queueCancelReasonMatched = cancelledDetails.reason === expectedReason;
    assertions.cancelRemovedQueuedWork = Number(cancelledDetails.removed_utterances) >= 1;
    assertions.cancelledTerminalReportsQueuedChunk = Number(
      eventDetails(terminal).chunks_queued,
    ) === 1;
    assertions.cancelledTerminalAfterCancel = Boolean(
      terminal && Number.isFinite(cancelSequence) && terminal.sequence >= cancelSequence,
    );
    assertions.cancelledTerminalAfterQueueCancelled = Boolean(
      terminal && queueCancelled.length === 1
        && terminal.sequence > queueCancelled[0].sequence,
    );
    assertions.noLateUtteranceStart = lateStart.length === 0;
    assertions.noLateUtteranceDone = lateDone.length === 0;
  }
  if (testCase.behavior === "cancel_immediate") {
    assertions.immediateCancelPreventedStart = starts.length === 0;
    assertions.immediateCancelDidNotCloseQueue = queueClosed.length === 0;
  }
  if (testCase.requireStartBeforeCancellation) {
    assertions.playbackStartSeen = starts.length >= 1;
    assertions.firstAudioStartProxySeen = byType("first_audio_start_proxy").length >= 1;
    assertions.firstAudioProxyBeforeCancel = Boolean(
      byType("first_audio_start_proxy")[0]
        && Number.isFinite(cancelSequence)
        && byType("first_audio_start_proxy")[0].sequence < cancelSequence,
    );
    assertions.cancelIssuedAfterStart = Boolean(
      starts[0] && Number.isFinite(cancelSequence) && cancelSequence > starts[0].sequence,
    );
    assertions.playbackWindowElapsedBeforeCancel = Number.isFinite(
      result?.driver?.postStartHoldMs,
    ) && result.driver.postStartHoldMs >= Number(testCase.playbackHoldMs || 0) - 10;
    assertions.playbackQueueClosedBeforeCancel = queueClosed.length === 1
      && Number.isFinite(cancelSequence)
      && queueClosed[0].sequence < cancelSequence;
  }
  const allPassed = Object.values(assertions).every(value => value === true);
  return {
    assertions,
    allPassed,
    observed: {
      readyEngine: resolvedEngine(ready),
      readyPolicy: eventDetails(ready).tts_policy ?? null,
      readyRequestedLocale: eventDetails(ready).requested_locale ?? null,
      readyResolvedLocale: eventDetails(ready).resolved_locale ?? null,
      generationLanguages: engineMetrics.map(record =>
        eventDetails(record).generation_language ?? null),
      engineMetricEngineIds: engineMetrics.map(record =>
        eventDetails(record).engine_id ?? null),
      terminalStatus: eventDetails(terminal).status ?? null,
      eventTypes: records.map(eventType),
      queuedChunkIds: orderedChunkIds(queued),
      startedChunkIds: orderedChunkIds(starts),
      doneChunkIds: orderedChunkIds(dones),
      lateDoneCount: lateDone.length,
      lateStartCount: lateStart.length,
      captureStartSequence: result?.captureStartSequence ?? null,
      captureEndSequence: result?.captureEndSequence ?? null,
    },
  };
}

export const CLEANUP_EXPRESSION = String.raw`(async () => {
  const state = window.__lfmPhysicalTtsAcceptance;
  const result = {
    activeTurnPresent: Boolean(state?.activeTurn),
    activeTurnCancelled: false,
    activeTurnTerminalVerified: null,
    coordinatorRepaired: null,
    policyRestoreAccepted: null,
    handlerRestored: false,
    originalHandlerErrors: state?.originalHandlerErrors || []
  };
  if (state?.activeTurn) {
    const active = state.activeTurn;
    const terminalSeen = () => state.events.some(record =>
      record?.event?.turn_id === active.id
        && record?.event?.type === "terminal"
        && Number(record?.event?.details?.generation) === Number(active.generation)
    );
    try {
      if (!terminalSeen()) {
        const fromSequence = state.nextSequence;
        window.NativeAudio?.cancelTts?.(
          active.id,
          active.generation,
          "cdp_harness_cleanup"
        );
        result.activeTurnCancelled = true;
        try {
          await state.waitFor(
            record => record?.event?.turn_id === active.id
              && record?.event?.type === "terminal"
              && Number(record?.event?.details?.generation) === Number(active.generation),
            fromSequence,
            10_000,
            "outer cleanup terminal"
          );
        } catch (_) {}
      }
      result.activeTurnTerminalVerified = terminalSeen();
    } catch (_) {}
    try {
      turnCoordinator.markTerminal(active.turn);
      turnCoordinator.clearIfCurrent(active.turn);
      result.coordinatorRepaired = !turnCoordinator.current
        || turnCoordinator.current !== active.turn;
    } catch (_) {
      result.coordinatorRepaired = false;
    }
    state.activeTurn = null;
  }
  if (state) {
    const restorePolicy = state.restorePolicy || "auto";
    try {
      result.policyRestoreAccepted = Boolean(window.NativeAudio?.setTtsEngine?.(restorePolicy));
    } catch (_) {
      result.policyRestoreAccepted = false;
    }
  }
  if (state?.handler && window.onNativeTtsEvent === state.handler) {
    window.onNativeTtsEvent = state.originalHandler;
    result.handlerRestored = true;
  }
  try { delete window.__lfmPhysicalTtsAcceptance; } catch (_) {}
  return result;
})()`;

export function buildEventExportExpression(graceMs) {
  const waitMs = Math.max(0, Number(graceMs || 0));
  return String.raw`(async () => {
    const state = window.__lfmPhysicalTtsAcceptance;
    if (!state) throw new Error("The physical TTS acceptance event hook is unavailable");
    await new Promise(resolve => setTimeout(resolve, ${waitMs}));
    return {
      grace_ms: ${waitMs},
      next_sequence: state.nextSequence,
      events: [...state.events],
      original_handler_errors: [...state.originalHandlerErrors],
      active_turn: state.activeTurn ? {
        id: state.activeTurn.id,
        generation: state.activeTurn.generation
      } : null
    };
  })()`;
}

function preflightFailures(preflight, cases) {
  const failures = [];
  for (const [method, available] of Object.entries(preflight?.methods || {})) {
    if (!available) failures.push(`missing NativeAudio.${method}`);
  }
  for (const [field, inspected] of Object.entries(preflight?.appState?.inspection || {})) {
    if (!inspected) failures.push(`could not inspect app state: ${field}`);
  }
  if (preflight?.appState?.inputBusy) failures.push("app input is busy");
  if (preflight?.appState?.currentTurn) failures.push("app has an active turn");
  if (Number(preflight?.appState?.clientTtsTurns || 0) !== 0) {
    failures.push("app has pending UI-owned TTS turns");
  }
  if (preflight?.appState?.nativeCapture) failures.push("native microphone capture is active");
  if (preflight?.appState?.mediaRecorderState
      && preflight.appState.mediaRecorderState !== "inactive") {
    failures.push(`browser MediaRecorder is ${preflight.appState.mediaRecorderState}`);
  }
  if (preflight?.appState?.vadRequested || preflight?.appState?.vadStopping) {
    failures.push("hands-free VAD is active or stopping");
  }
  if (Number(preflight?.appState?.browserPlaybackSources || 0) !== 0) {
    failures.push("browser audio playback is still active");
  }
  if (preflight?.appState?.htmlAudioPlaying) failures.push("HTML audio playback is active");
  const engines = preflight?.engineCatalog?.engines || [];
  const requiredEngineIds = new Set();
  for (const testCase of cases) {
    requiredEngineIds.add(testCase.policy);
    requiredEngineIds.add(testCase.expectedEngine);
  }
  for (const id of requiredEngineIds) {
    const engine = engines.find(item => item?.id === id);
    if (!engine) failures.push(`TTS engine catalog is missing ${id}`);
    else if (engine.available !== true) failures.push(`TTS engine ${id} is unavailable`);
  }
  return failures;
}

function selectCases(caseNames) {
  if (!caseNames) return DEFAULT_CASES.map(item => ({ ...item, chunks: [...item.chunks] }));
  const requested = new Set(caseNames);
  const selected = DEFAULT_CASES.filter(item => requested.has(item.name))
    .map(item => ({ ...item, chunks: [...item.chunks] }));
  const missing = caseNames.filter(name => !selected.some(item => item.name === name));
  if (missing.length) throw new Error(`Unknown case name(s): ${missing.join(", ")}`);
  return selected;
}

export function auditEventCapture(capture) {
  const events = Array.isArray(capture?.events) ? capture.events : [];
  const sequences = events.map(record => record?.sequence);
  const assertions = {
    captureAvailable: Boolean(capture),
    eventSequencesValid: sequences.every(Number.isSafeInteger),
    eventSequencesStrictlyIncreasing: sequences.every((value, index) =>
      index === 0 || value > sequences[index - 1]
    ),
    allEventsParsed: events.every(record =>
      !record?.parse_error && record?.event && typeof record.event === "object"
    ),
    allEventSchemasValid: events.every(record =>
      record?.event?.schema_version === 1
        && typeof record.event.type === "string"
        && record.event.type.length > 0
        && typeof record.event.turn_id === "string"
        && record.event.turn_id.length > 0
    ),
    originalHandlerStayedHealthy: (capture?.original_handler_errors || []).length === 0,
    noActiveTurnAtFinalAudit: capture?.active_turn == null,
  };
  return {
    assertions,
    allPassed: Object.values(assertions).every(value => value === true),
    eventCount: events.length,
  };
}

export function applyFinalEventAudit(payload) {
  const events = payload.eventCapture?.events || [];
  const handlerErrors = payload.eventCapture?.original_handler_errors || [];
  for (const result of payload.cases) {
    result.events = events.filter(record => record?.event?.turn_id === result.turnId);
    result.capturedEvents = events.filter(record =>
      record.sequence >= result.captureStartSequence
        && record.sequence < result.captureEndSequence
    );
    result.originalHandlerErrors = handlerErrors;
    const testCase = DEFAULT_CASES.find(item => item.name === result.name);
    if (testCase) result.acceptance = validateCase(testCase, result);
  }
  payload.eventCapture.audit = auditEventCapture(payload.eventCapture);
}

function summarizePayload(payload) {
  const passed = payload.cases.filter(item => item.acceptance?.allPassed).length;
  const cleanupPassed = !payload.hook || Boolean(
    payload.cleanup?.handlerRestored === true
      && payload.cleanup?.policyRestoreAccepted === true
      && (
        payload.cleanup?.activeTurnPresent !== true
          || (
            payload.cleanup?.activeTurnTerminalVerified === true
              && payload.cleanup?.coordinatorRepaired === true
          )
      )
  );
  const eventCapturePassed = !payload.hook
    ? false : payload.eventCapture?.audit?.allPassed === true;
  return {
    total: payload.cases.length,
    passed,
    failed: payload.cases.length - passed,
    caseAssertionsPassed: passed === payload.cases.length,
    eventCapturePassed,
    cleanupPassed,
    operationalFailure: Boolean(payload.failure),
    allPassed: passed === payload.cases.length
      && payload.cases.length > 0
      && eventCapturePassed
      && cleanupPassed
      && !payload.failure,
    failedCases: payload.cases
      .filter(item => !item.acceptance?.allPassed)
      .map(item => item.name),
  };
}

export async function main(argv = process.argv) {
  const args = parseArgs(argv);
  if (args.help) {
    process.stdout.write(usage());
    return true;
  }
  if (args.listCases) {
    process.stdout.write(`${DEFAULT_CASES.map(item => item.name).join("\n")}\n`);
    return true;
  }
  const cases = selectCases(args.caseNames);
  const payload = {
    schema_version: HARNESS_SCHEMA_VERSION,
    harness: "physical_android_tts_cdp",
    startedAt: new Date().toISOString(),
    endpoint: { host: args.host, port: args.port },
    options: {
      timeoutMs: args.timeoutMs,
      eventTimeoutMs: args.eventTimeoutMs,
      lateEventGraceMs: args.lateEventGraceMs,
      cooldownMs: args.cooldownMs,
      targetUrlIncludes: args.targetUrlIncludes,
      selectedCases: cases.map(item => item.name),
    },
    output: args.output,
    target: null,
    preflight: null,
    hook: null,
    eventCapture: null,
    cases: [],
    cleanup: null,
    summary: null,
    failure: null,
  };
  persist(args.output, payload);
  let client = null;
  try {
    const connected = await connectToAppTarget(args);
    client = connected.client;
    payload.target = {
      id: connected.target.id || null,
      title: connected.target.title || null,
      url: connected.target.url || null,
      type: connected.target.type || null,
    };
    payload.preflight = await client.evaluate(PREFLIGHT_EXPRESSION, 30_000);
    payload.preflight.failures = preflightFailures(payload.preflight, cases);
    persist(args.output, payload);
    if (payload.preflight.failures.length) {
      throw new Error(`TTS preflight failed: ${payload.preflight.failures.join("; ")}`);
    }
    payload.hook = await client.evaluate(INSTALL_HOOK_EXPRESSION, 30_000);
    const restorePolicy = payload.preflight.engineCatalog?.selected || "auto";
    await client.evaluate(`window.__lfmPhysicalTtsAcceptance.restorePolicy = ${JSON.stringify(restorePolicy)}`);
    persist(args.output, payload);

    for (const testCase of cases) {
      process.stderr.write(`TTS_CASE_START ${testCase.name} ${testCase.policy}/${testCase.locale}\n`);
      const result = await client.evaluate(
        buildCaseExpression(testCase, args),
        Math.max(args.timeoutMs, args.eventTimeoutMs + args.lateEventGraceMs + 15_000),
      );
      result.expectations = {
        policy: testCase.expectedPolicy,
        engine: testCase.expectedEngine,
        enginePackage: testCase.expectedEnginePackage || null,
        requestedLocale: testCase.expectedRequestedLocale || null,
        resolvedLocale: testCase.expectedResolvedLocale || null,
        terminalStatus: testCase.expectedTerminalStatus,
      };
      result.acceptance = validateCase(testCase, result);
      payload.cases.push(result);
      persist(args.output, payload);
      process.stderr.write(
        `TTS_CASE_DONE ${testCase.name} ${Math.round(result.elapsedMs)}ms `
          + `passed=${result.acceptance.allPassed}\n`,
      );
      if (result.driver?.cleanupIncomplete) {
        throw new Error(`Case ${testCase.name} left native TTS work without a verified terminal`);
      }
      if (args.cooldownMs) await sleep(args.cooldownMs);
    }
  } catch (error) {
    payload.failure = {
      at: new Date().toISOString(),
      name: error?.name || "Error",
      message: String(error?.message || error),
      stack: error?.stack || null,
    };
  } finally {
    if (client) {
      if (payload.hook) {
        try {
          payload.eventCapture = await client.evaluate(
            buildEventExportExpression(args.lateEventGraceMs),
            args.lateEventGraceMs + 15_000,
          );
          applyFinalEventAudit(payload);
          persist(args.output, payload);
        } catch (error) {
          payload.eventCapture = {
            error: String(error?.message || error),
            audit: { allPassed: false, assertions: { captureAvailable: false } },
          };
          if (!payload.failure) {
            payload.failure = {
              at: new Date().toISOString(),
              name: "EventCaptureError",
              message: `Final TTS event audit failed: ${payload.eventCapture.error}`,
              stack: null,
            };
          }
        }
      }
      try {
        payload.cleanup = await client.evaluate(CLEANUP_EXPRESSION, 30_000);
      } catch (error) {
        payload.cleanup = { error: String(error?.message || error) };
      }
      if (!payload.failure && payload.hook && (
        payload.cleanup?.handlerRestored !== true
        || payload.cleanup?.policyRestoreAccepted !== true
        || (
          payload.cleanup?.activeTurnPresent === true
            && (
              payload.cleanup?.activeTurnTerminalVerified !== true
                || payload.cleanup?.coordinatorRepaired !== true
            )
        )
      )) {
        payload.failure = {
          at: new Date().toISOString(),
          name: "CleanupError",
          message: `TTS harness cleanup was incomplete: ${JSON.stringify(payload.cleanup)}`,
          stack: null,
        };
      }
      client.close();
    }
    payload.summary = summarizePayload(payload);
    payload.finishedAt = new Date().toISOString();
    persist(args.output, payload);
  }
  process.stdout.write(`${JSON.stringify({
    output: args.output,
    summary: payload.summary,
    failure: payload.failure?.message || null,
  }, null, 2)}\n`);
  return !payload.failure && payload.summary?.allPassed === true;
}

const isMain = process.argv[1]
  && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
  main().then(ok => {
    if (!ok) process.exitCode = 1;
  }).catch(error => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exitCode = 1;
  });
}
