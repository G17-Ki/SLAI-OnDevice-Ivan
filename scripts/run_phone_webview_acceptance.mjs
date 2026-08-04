#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import {
  collectNativeTtsEvidence,
  nativeTtsAcceptanceAssertions,
  requiredScriptMatches,
  responseThinkingDisabled,
  validateAcceptancePlan,
} from "./phone_acceptance_evidence.mjs";

function parseArgs(argv) {
  const args = {
    host: "127.0.0.1",
    port: 9224,
    timeoutMs: 300_000,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--host") args.host = value;
    else if (key === "--port") args.port = Number(value);
    else if (key === "--plan") args.plan = value;
    else if (key === "--output") args.output = value;
    else if (key === "--timeout-ms") args.timeoutMs = Number(value);
    else throw new Error(`Unknown argument: ${key}`);
    index += 1;
  }
  if (!args.plan) throw new Error("--plan is required");
  if (!args.output) throw new Error("--output is required");
  return args;
}

function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function discoverTarget(host, port) {
  const response = await fetch(`http://${host}:${port}/json`);
  if (!response.ok) throw new Error(`CDP target discovery failed: HTTP ${response.status}`);
  const targets = await response.json();
  const target = targets.find(item => item.type === "page" && item.webSocketDebuggerUrl);
  if (!target) throw new Error("No debuggable WebView page target was found");
  return target;
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
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Timed out opening CDP WebSocket")), 10_000);
      this.socket.addEventListener("open", () => {
        clearTimeout(timeout);
        resolve();
      }, { once: true });
      this.socket.addEventListener("error", event => {
        clearTimeout(timeout);
        reject(new Error(`CDP WebSocket failed: ${event.message || "unknown error"}`));
      }, { once: true });
    });
    this.socket.addEventListener("message", event => {
      const message = JSON.parse(String(event.data));
      if (!message.id) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timeout);
      if (message.error) pending.reject(new Error(`CDP ${pending.method}: ${message.error.message}`));
      else pending.resolve(message.result);
    });
    this.socket.addEventListener("close", () => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timeout);
        pending.reject(new Error("CDP WebSocket closed"));
      }
      this.pending.clear();
    });
    await this.call("Runtime.enable", {});
  }

  call(method, params = {}, timeoutMs = this.defaultTimeoutMs) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timeout, method });
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
      const description = result.exceptionDetails.exception?.description
        || result.exceptionDetails.text
        || "Runtime.evaluate failed";
      throw new Error(description);
    }
    return result.result?.value;
  }

  close() {
    this.socket?.close();
  }
}

const STATIC_INSPECTION = String.raw`(async () => {
  const inspectJsonEndpoint = async url => {
    try {
      const response = await fetch(url, { method: "GET", cache: "no-store" });
      const text = await response.text();
      let body = null;
      try { body = text ? JSON.parse(text) : null; } catch (_) {}
      return {
        url,
        httpStatus: response.status,
        httpOk: response.ok,
        body,
        bodyCharacters: text.length,
        error: null
      };
    } catch (error) {
      return {
        url,
        httpStatus: null,
        httpOk: false,
        body: null,
        bodyCharacters: 0,
        error: String(error?.message || error)
      };
    }
  };
  const responseHealthUrl = typeof TEXT_HEALTH_URL === "string"
    ? TEXT_HEALTH_URL
    : "http://127.0.0.1:6020/health";
  const responseModelsUrl = responseHealthUrl.replace(/\/health(?:\?.*)?$/, "/v1/models");
  const lfmHealthUrl = typeof LFM_HEALTH_URL === "string"
    ? LFM_HEALTH_URL
    : "http://127.0.0.1:8080/";
  const [responseHealth, responseModels, directHealth] = await Promise.all([
    inspectJsonEndpoint(responseHealthUrl),
    inspectJsonEndpoint(responseModelsUrl),
    inspectJsonEndpoint(lfmHealthUrl)
  ]);
  const responseModelIds = Array.isArray(responseModels.body?.data)
    ? responseModels.body.data.map(item => item?.id).filter(Boolean)
    : [];
  const sampleRoutes = {
    en: resolveLanguageSelection("auto", "Hello, could you explain the weather today?"),
    zh: resolveLanguageSelection("auto", "请用中文介绍北京。"),
    ja: resolveLanguageSelection("auto", "日本語で短く答えてください。"),
    ko: resolveLanguageSelection("auto", "한국어로 짧게 답하세요."),
    ru: resolveLanguageSelection("auto", "Расскажи коротко о Москве."),
    fr: resolveLanguageSelection("auto", "Bonjour, répondez seulement en français."),
    it: resolveLanguageSelection("auto", "Ciao, rispondi soltanto in italiano."),
    es: resolveLanguageSelection("auto", "Hola, responde solamente en español.")
  };
  const asrPayload = buildAsrPayload({ encodedWav: "UklGRg==", languageSelection: "en" });
  const responsePayload = buildConversationPayload({
    requestMode: "chat",
    transcript: "What did I ask?",
    languageRoute: resolveLanguageSelection("en"),
    historyMessages: [
      { role: "user", content: "Remember turquoise." },
      { role: "assistant", content: "I will remember turquoise." }
    ]
  });
  const countAudioParts = value => (value.messages || []).reduce((count, message) => {
    const content = message?.content;
    if (!Array.isArray(content)) return count;
    return count + content.filter(part => part?.type === "input_audio").length;
  }, 0);
  const countAudioKeys = value => {
    if (!value || typeof value !== "object") return 0;
    if (Array.isArray(value)) return value.reduce((count, item) => count + countAudioKeys(item), 0);
    return Object.entries(value).reduce((count, [key, item]) =>
      count + (key === "input_audio" ? 1 : 0) + countAudioKeys(item), 0);
  };
  return {
    title: document.title,
    url: location.href,
    modelReady,
    componentReadiness: {
      asr: typeof asrModelReady === "boolean" ? asrModelReady : null,
      text: typeof textModelReady === "boolean" ? textModelReady : null,
      lfm: typeof lfmModelReady === "boolean" ? lfmModelReady : null
    },
    deviceStatus: deviceStatusText.textContent,
    mode,
    inputMode,
    runtimeState: {
      inputBusy,
      nativeRecording,
      mediaRecorderState: mediaRecorder?.state || null,
      currentTurn: turnCoordinator.current ? {
        id: turnCoordinator.current.id,
        mode: turnCoordinator.current.mode,
        inputKind: turnCoordinator.current.inputKind,
        cancelled: turnCoordinator.current.cancelled,
        terminal: turnCoordinator.current.terminal
      } : null,
      vad: { ...vadState },
      buttons: {
        chatDisabled: chatButton.disabled,
        asrDisabled: asrButton.disabled,
        ttsDisabled: ttsButton.disabled,
        recordDisabled: recordButton.disabled
      }
    },
    prompts: {
      asr: LFM_ASR_SYSTEM_PROMPT,
      response: typeof QWEN_CHAT_SYSTEM_PROMPT === "string"
        ? QWEN_CHAT_SYSTEM_PROMPT
        : (typeof LFM_CHAT_SYSTEM_PROMPT === "string" ? LFM_CHAT_SYSTEM_PROMPT : null)
    },
    languageOptions: [...languageSelect.options].map(option => ({
      value: option.value,
      label: option.textContent.trim()
    })),
    controls: {
      chat: Boolean(document.getElementById("mode-chat")),
      asr: Boolean(document.getElementById("mode-asr")),
      tts: Boolean(document.getElementById("mode-tts")),
      englishResponseStyle: Boolean(document.getElementById("english-response-style")),
      selectedEnglishResponseStyle: englishResponseStyleSelect?.value || null,
      push: Boolean(document.getElementById("input-push")),
      vad: Boolean(document.getElementById("input-vad")),
      clearHistory: Boolean(document.getElementById("clear-history"))
    },
    nativeBridge: {
      present: Boolean(window.NativeAudio),
      telemetry: typeof window.NativeAudio?.recordTelemetry === "function",
      resetTelemetry: typeof window.NativeAudio?.resetTelemetry === "function",
      tts: typeof window.NativeAudio?.initializeTts === "function",
      vadStart: typeof window.NativeAudio?.startVadConversation === "function",
      vadResume: typeof window.NativeAudio?.resumeVadConversation === "function",
      vadStop: typeof window.NativeAudio?.stopVadConversation === "function",
      ttsEngineAvailable: nativeTtsEngineAvailable(),
      defaultTtsLocale: defaultNativeTtsLocale()
    },
    npuFacts: {
      auto: npuFailClosedFacts("auto"),
      en: npuFailClosedFacts("en", "en"),
      zh: npuFailClosedFacts("zh", "zh"),
      ja: npuFailClosedFacts("ja", "ja"),
      ko: npuFailClosedFacts("ko", "ko"),
      fr: npuFailClosedFacts("fr", "fr")
    },
    history: conversationHistory.summary(),
    historyEnabled: isHistoryEnabled(),
    historySwitch: {
      ariaChecked: historyToggleButton.getAttribute("aria-checked"),
      text: historyToggleButton.textContent
    },
    asrCompanion: {
      url: typeof ASR_COMPANION_URL === "string" ? ASR_COMPANION_URL : null
    },
    responseCompanion: {
      apiUrl: typeof TEXT_API_URL === "string" ? TEXT_API_URL : null,
      healthUrl: responseHealthUrl,
      backend: typeof QWEN_RESPONSE_BACKEND === "string" ? QWEN_RESPONSE_BACKEND : null,
      requestModel: typeof QWEN_TEXT_MODEL === "string" ? QWEN_TEXT_MODEL : null,
      health: responseHealth,
      models: responseModels,
      modelIds: responseModelIds
    },
    directResponse: {
      apiUrl: typeof LFM_API_URL === "string" ? LFM_API_URL : null,
      backend: typeof LFM_DIRECT_RESPONSE_BACKEND === "string"
        ? LFM_DIRECT_RESPONSE_BACKEND
        : null,
      requestModel: typeof LFM_DIRECT_MODEL === "string" ? LFM_DIRECT_MODEL : null,
      modelReady: typeof lfmModelReady === "boolean" ? lfmModelReady : Boolean(modelReady),
      healthUrl: lfmHealthUrl,
      health: directHealth,
      systemPrompt: typeof LFM_DIRECT_SYSTEM_PROMPT === "string"
        ? LFM_DIRECT_SYSTEM_PROMPT
        : null
    },
    autoRoutes: sampleRoutes,
    protocol: {
      asrInputAudioParts: countAudioParts(asrPayload),
      asrInputAudioKeys: countAudioKeys(asrPayload),
      responseInputAudioParts: countAudioParts(responsePayload),
      responseInputAudioKeys: countAudioKeys(responsePayload),
      asrRoles: asrPayload.messages.map(message => message.role),
      responseRoles: responsePayload.messages.map(message => message.role),
      responseUserType: typeof responsePayload.messages[1].content,
      responseUserText: responsePayload.messages[1].content,
      asrModalities: asrPayload.modalities,
      responseModalities: responsePayload.modalities
    },
    serviceWorker: {
      controlled: Boolean(navigator.serviceWorker?.controller),
      scriptURL: navigator.serviceWorker?.controller?.scriptURL || null,
      cacheKeys: null,
      cacheInspection: "omitted because caches.keys() can stall Android WebView CDP"
    }
  };
})()`;

const INSTALL_FETCH_CAPTURE = String.raw`(() => {
  if (window.__lfmAcceptanceOriginalFetch) {
    window.fetch = window.__lfmAcceptanceOriginalFetch;
  }
  const originalFetch = window.fetch.bind(window);
  window.__lfmAcceptanceOriginalFetch = originalFetch;
  window.__lfmFetchCapture = [];
  window.__lfmFetchInspectionPromises = [];
  if (window.__lfmAcceptanceNativeTtsHandler
      && window.onNativeTtsEvent === window.__lfmAcceptanceNativeTtsHandler) {
    window.onNativeTtsEvent = window.__lfmAcceptanceOriginalNativeTtsHandler;
  }
  const originalNativeTtsHandler = typeof window.onNativeTtsEvent === "function"
    ? window.onNativeTtsEvent
    : null;
  window.__lfmAcceptanceOriginalNativeTtsHandler = originalNativeTtsHandler;
  window.__lfmAcceptanceNativeTtsHandler = null;
  window.__lfmNativeTtsCapture = [];
  window.__lfmNativeTtsOriginalHandlerErrors = [];
  if (originalNativeTtsHandler) {
    window.__lfmAcceptanceNativeTtsHandler = raw => {
      let event = null;
      let parseError = null;
      let rawText = null;
      try {
        event = typeof raw === "string" ? JSON.parse(raw) : raw;
        rawText = typeof raw === "string" ? raw : JSON.stringify(raw);
      } catch (error) {
        parseError = String(error?.message || error);
      }
      const record = {
        sequence: window.__lfmNativeTtsCapture.length,
        captured_wall_time: new Date().toISOString(),
        captured_performance_ms: performance.now(),
        event,
        raw: rawText,
        parse_error: parseError,
        original_handler_error: null
      };
      window.__lfmNativeTtsCapture.push(record);
      try {
        originalNativeTtsHandler.call(window, raw);
      } catch (error) {
        record.original_handler_error = String(error?.message || error);
        window.__lfmNativeTtsOriginalHandlerErrors.push({
          at: new Date().toISOString(),
          message: record.original_handler_error
        });
      }
    };
    window.onNativeTtsEvent = window.__lfmAcceptanceNativeTtsHandler;
  }
  if (window.__lfmAcceptanceFileInputBlocker) {
    fileInput.removeEventListener("change", window.__lfmAcceptanceFileInputBlocker, true);
  }
  window.__lfmBlockedFileInputCount = 0;
  window.__lfmAcceptanceFileInputBlocker = event => {
    window.__lfmBlockedFileInputCount += 1;
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  fileInput.addEventListener("change", window.__lfmAcceptanceFileInputBlocker, true);
  const headerValue = (headers, name) => {
    if (typeof headers?.get === "function") return headers.get(name);
    const match = Object.entries(headers || {}).find(
      ([key]) => key.toLocaleLowerCase() === name.toLocaleLowerCase()
    );
    return match ? String(match[1]) : null;
  };
  const bodyByteLength = body => {
    if (body == null) return 0;
    if (typeof body === "string") return new TextEncoder().encode(body).byteLength;
    if (body instanceof ArrayBuffer) return body.byteLength;
    if (ArrayBuffer.isView(body)) return body.byteLength;
    if (typeof Blob !== "undefined" && body instanceof Blob) return body.size;
    return null;
  };
  const modelIdentity = value => String(value || "")
    .split("/")
    .at(-1)
    .replace(/\.gguf$/iu, "");
  const inspectStreamingResponse = async (response, capture) => {
    const text = await response.clone().text();
    capture.responseBodyBytes = new TextEncoder().encode(text).byteLength;
    let responseText = "";
    let reasoningText = "";
    let responseModel = null;
    let finishReason = null;
    let doneSeen = false;
    let parseErrors = 0;
    for (const line of text.split(/\r?\n/u)) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      if (payload === "[DONE]") {
        doneSeen = true;
        continue;
      }
      let event;
      try {
        event = JSON.parse(payload);
      } catch (_) {
        parseErrors += 1;
        continue;
      }
      if (event?.model) responseModel = String(event.model);
      const choice = event?.choices?.[0] || {};
      if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
        finishReason = String(choice.finish_reason);
      }
      const delta = choice.delta || choice.message || {};
      if (typeof delta.content === "string") responseText += delta.content;
      if (typeof delta.reasoning_content === "string") reasoningText += delta.reasoning_content;
      if (typeof choice.reasoning_content === "string") reasoningText += choice.reasoning_content;
    }
    capture.responseModel = responseModel;
    capture.responseModelIdentity = modelIdentity(responseModel || capture.requestModel);
    capture.finishReason = finishReason;
    capture.reasoningContentCharacters = reasoningText.length;
    capture.responseCharacters = responseText.length;
    capture.thinkTagSeen = /<\/?think\b/iu.test(
      String(responseText) + "\n" + String(reasoningText)
    );
    capture.sseDoneSeen = doneSeen;
    capture.sseParseErrors = parseErrors;
  };
  window.fetch = async function acceptanceFetch(input, options = {}) {
    const url = typeof input === "string" ? input : (input?.url || String(input));
    let capture = null;
    const isAsrCompanion = url.includes("127.0.0.1:6010/asr");
    const isResponseCompanion = url.includes("127.0.0.1:6020/v1/chat/completions");
    if (isAsrCompanion) {
      let requestedLanguage = null;
      try {
        requestedLanguage = new URL(url, location.href).searchParams.get("language");
      } catch (_) {}
      capture = {
        url,
        at: new Date().toISOString(),
        stage: "asr_companion",
        method: options?.method || "GET",
        requestedLanguage,
        contentType: headerValue(options?.headers, "Content-Type"),
        bodyBytes: bodyByteLength(options?.body),
        roles: null,
        modalities: null,
        inputAudioParts: 0,
        inputAudioKeys: 0,
        userContentType: null,
        userText: null
      };
    } else if (url.includes("/v1/chat/completions") && typeof options?.body === "string") {
      try {
        const body = JSON.parse(options.body);
        const systemPrompt = body.messages?.[0]?.content ?? null;
        const isDirectResponse = systemPrompt === LFM_DIRECT_SYSTEM_PROMPT
          && Array.isArray(body.modalities)
          && body.modalities.includes("text")
          && body.modalities.includes("audio");
        const userMessage = [...(body.messages || [])].reverse().find(message => message?.role === "user");
        const userContent = userMessage?.content;
        const countInputAudioKeys = value => {
          if (!value || typeof value !== "object") return 0;
          if (Array.isArray(value)) return value.reduce((count, item) => count + countInputAudioKeys(item), 0);
          return Object.entries(value).reduce((count, [key, item]) =>
            count + (key === "input_audio" ? 1 : 0) + countInputAudioKeys(item), 0);
        };
        capture = {
          url,
          at: new Date().toISOString(),
          method: options?.method || "GET",
          stage: isResponseCompanion
            ? "response_companion"
            : (systemPrompt === LFM_ASR_SYSTEM_PROMPT
              ? "asr"
              : (isDirectResponse ? "response" : "unknown")),
          systemPrompt,
          bodyBytes: bodyByteLength(options.body),
          contentType: headerValue(options?.headers, "Content-Type"),
          maxTokens: body.max_tokens ?? null,
          enableThinking: body.chat_template_kwargs?.enable_thinking ?? null,
          stream: body.stream ?? null,
          requestBackend: isResponseCompanion && typeof QWEN_RESPONSE_BACKEND === "string"
            ? QWEN_RESPONSE_BACKEND
            : (isDirectResponse && typeof LFM_DIRECT_RESPONSE_BACKEND === "string"
              ? LFM_DIRECT_RESPONSE_BACKEND
              : null),
          requestModel: body.model ?? null,
          requestModelIdentity: modelIdentity(body.model),
          roles: Array.isArray(body.messages) ? body.messages.map(message => message.role) : [],
          modalities: body.modalities || null,
          inputAudioParts: Array.isArray(userContent)
            ? userContent.filter(part => part?.type === "input_audio").length
            : 0,
          inputAudioKeys: countInputAudioKeys(body),
          userContentType: typeof userContent,
          userText: typeof userContent === "string"
            ? userContent
            : null
        };
      } catch (error) {
        capture = { url, at: new Date().toISOString(), parseError: String(error) };
      }
    }
    if (capture) {
      window.__lfmFetchCapture.push(capture);
    }
    try {
      const response = await originalFetch(input, options);
      if (capture) {
        capture.httpStatus = response.status;
        capture.httpOk = response.ok;
        if (["response", "response_companion"].includes(capture.stage) && response.body) {
          const inspection = inspectStreamingResponse(response, capture).catch(error => {
            capture.responseInspectionError = String(error?.message || error);
          });
          window.__lfmFetchInspectionPromises.push(inspection);
        }
      }
      return response;
    } catch (error) {
      if (capture) capture.fetchError = String(error?.message || error);
      throw error;
    }
  };
  return {
    installed: true,
    captureCount: window.__lfmFetchCapture.length,
    fileInputBlocked: true,
    nativeTtsCaptureSupported: Boolean(window.__lfmAcceptanceNativeTtsHandler),
    nativeTtsOriginalHandlerPresent: Boolean(originalNativeTtsHandler)
  };
})()`;

const REMOVE_ACCEPTANCE_HOOKS = String.raw`(() => {
  if (window.__lfmAcceptanceOriginalFetch) {
    window.fetch = window.__lfmAcceptanceOriginalFetch;
    delete window.__lfmAcceptanceOriginalFetch;
  }
  if (window.__lfmAcceptanceFileInputBlocker) {
    fileInput.removeEventListener("change", window.__lfmAcceptanceFileInputBlocker, true);
    delete window.__lfmAcceptanceFileInputBlocker;
  }
  let nativeTtsHandlerRestored = null;
  if (window.__lfmAcceptanceNativeTtsHandler) {
    if (window.onNativeTtsEvent === window.__lfmAcceptanceNativeTtsHandler) {
      window.onNativeTtsEvent = window.__lfmAcceptanceOriginalNativeTtsHandler;
      nativeTtsHandlerRestored = true;
    } else {
      nativeTtsHandlerRestored = false;
    }
  }
  const nativeTtsCaptureCount = window.__lfmNativeTtsCapture?.length || 0;
  const nativeTtsOriginalHandlerErrors = [
    ...(window.__lfmNativeTtsOriginalHandlerErrors || [])
  ];
  delete window.__lfmAcceptanceNativeTtsHandler;
  delete window.__lfmAcceptanceOriginalNativeTtsHandler;
  delete window.__lfmNativeTtsCapture;
  delete window.__lfmNativeTtsOriginalHandlerErrors;
  return {
    removed: true,
    blockedFileInputEvents: window.__lfmBlockedFileInputCount || 0,
    nativeTtsHandlerRestored,
    nativeTtsCaptureCount,
    nativeTtsOriginalHandlerErrors
  };
})()`;

function clearHistoryExpression(resetTelemetry) {
  return `(() => {
    const cleared = conversationHistory.clear();
    renderHistoryStatus(cleared);
    renderPendingLanguageSelection();
    const telemetryReset = ${resetTelemetry ? "Boolean(window.NativeAudio?.resetTelemetry?.())" : "null"};
    return { cleared, telemetryReset };
  })()`;
}

function setContextExpression(enabled) {
  return `(() => {
    const requested = ${enabled ? "true" : "false"};
    const accepted = setHistoryEnabled(requested);
    return {
      requested,
      accepted,
      historyEnabled: isHistoryEnabled(),
      switchAriaChecked: historyToggleButton.getAttribute("aria-checked"),
      switchText: historyToggleButton.textContent,
      summary: conversationHistory.summary()
    };
  })()`;
}

function buildCaseExpression(testCase, encodedWav) {
  const caseJson = JSON.stringify({
    name: testCase.name,
    mode: testCase.mode,
    language: testCase.language,
    ttsPolicy: testCase.ttsPolicy || null,
    englishResponseStyle: testCase.englishResponseStyle || null,
    serverWarmth: testCase.serverWarmth || null,
    contextEnabled: typeof testCase.contextEnabled === "boolean"
      ? testCase.contextEnabled
      : null,
    expectations: {
      outcome: testCase.expectedOutcome || null,
      resolvedLanguage: testCase.expectedResolvedLanguage || null,
      responseMode: testCase.expectedResponseMode || null,
      voiceBackend: testCase.expectedVoiceBackend || null,
      responseBackend: testCase.expectedResponseBackend || null,
      responseModel: testCase.expectedResponseModel || null,
      finishReason: testCase.expectedFinishReason || null,
      noReasoning: typeof testCase.expectedNoReasoning === "boolean"
        ? testCase.expectedNoReasoning
        : null,
      maxFirstAudioMs: Number.isFinite(testCase.maxFirstAudioMs)
        ? Number(testCase.maxFirstAudioMs)
        : null,
      maxFirstVoicedAudioMs: Number.isFinite(testCase.maxFirstVoicedAudioMs)
        ? Number(testCase.maxFirstVoicedAudioMs)
        : null,
      minPcmDurationMs: Number.isFinite(testCase.minPcmDurationMs)
        ? Number(testCase.minPcmDurationMs)
        : null,
      minNativeTtsPlaybackMs: Number.isFinite(testCase.minNativeTtsPlaybackMs)
        ? Number(testCase.minNativeTtsPlaybackMs)
        : null,
      ttsTerminalStatus: testCase.expectedTtsTerminalStatus || null,
      ttsPolicy: testCase.expectedTtsPolicy || null,
      ttsEngine: testCase.expectedTtsEngine || null,
      ttsRequestedLocale: testCase.expectedTtsRequestedLocale || null,
      ttsResolvedLocale: testCase.expectedTtsResolvedLocale || null,
      ttsGenerationLanguage: testCase.expectedTtsGenerationLanguage || null,
      maxTtsFirstChunkMs: Number.isFinite(testCase.maxTtsFirstChunkMs)
        ? Number(testCase.maxTtsFirstChunkMs)
        : null,
      maxTtsRtf: Number.isFinite(testCase.maxTtsRtf)
        ? Number(testCase.maxTtsRtf)
        : null,
      maxTtsDroppedChunks: Number.isFinite(testCase.maxTtsDroppedChunks)
        ? Number(testCase.maxTtsDroppedChunks)
        : null,
      asrBackend: testCase.expectedAsrBackend || null,
      asrProvider: testCase.expectedAsrProvider || null,
      asrLanguageHint: testCase.expectedAsrLanguageHint || null,
      npuExecuted: typeof testCase.expectedNpuExecuted === "boolean"
        ? testCase.expectedNpuExecuted
        : null,
      npuSelected: typeof testCase.expectedNpuSelected === "boolean"
        ? testCase.expectedNpuSelected
        : null,
      errorIncludes: testCase.expectedErrorIncludes || null,
      historyTurnsAfter: Number.isInteger(testCase.expectedHistoryTurnsAfter)
        ? testCase.expectedHistoryTurnsAfter
        : null,
      responseRequestIncludes: Array.isArray(testCase.expectedResponseRequestIncludes)
        ? testCase.expectedResponseRequestIncludes
        : (typeof testCase.expectedResponseRequestIncludes === "string"
          ? [testCase.expectedResponseRequestIncludes]
          : []),
      responseRequestExcludes: Array.isArray(testCase.expectedResponseRequestExcludes)
        ? testCase.expectedResponseRequestExcludes
        : (typeof testCase.expectedResponseRequestExcludes === "string"
          ? [testCase.expectedResponseRequestExcludes]
          : []),
      companionLanguages: Array.isArray(testCase.expectedCompanionLanguages)
        ? testCase.expectedCompanionLanguages
        : [],
      responseIncludes: Array.isArray(testCase.expectedResponseIncludes)
        ? testCase.expectedResponseIncludes
        : (typeof testCase.expectedResponseIncludes === "string"
          ? [testCase.expectedResponseIncludes]
          : []),
      responseExcludes: Array.isArray(testCase.expectedResponseExcludes)
        ? testCase.expectedResponseExcludes
        : (typeof testCase.expectedResponseExcludes === "string"
          ? [testCase.expectedResponseExcludes]
          : []),
      transcriptIncludes: Array.isArray(testCase.expectedTranscriptIncludes)
        ? testCase.expectedTranscriptIncludes
        : (typeof testCase.expectedTranscriptIncludes === "string"
          ? [testCase.expectedTranscriptIncludes]
          : []),
      requiredTranscriptScript: testCase.expectedTranscriptScript || null,
      requiredResponseScript: testCase.expectedResponseScript || null,
      minResponseCharacters: Number.isFinite(testCase.expectedMinResponseCharacters)
        ? Number(testCase.expectedMinResponseCharacters)
        : null,
      maxDuplicateGramRatio: Number.isFinite(testCase.expectedMaxDuplicateGramRatio)
        ? Number(testCase.expectedMaxDuplicateGramRatio)
        : null
    }
  });
  const audioJson = JSON.stringify(encodedWav);
  const responseThinkingDisabledSource = responseThinkingDisabled.toString();
  const collectNativeTtsEvidenceSource = collectNativeTtsEvidence.toString();
  const nativeTtsAcceptanceAssertionsSource = nativeTtsAcceptanceAssertions.toString();
  const requiredScriptMatchesSource = requiredScriptMatches.toString();
  return `(async () => {
    const testCase = ${caseJson};
    const encodedWav = ${audioJson};
    const responseThinkingDisabled = (${responseThinkingDisabledSource});
    const collectNativeTtsEvidence = (${collectNativeTtsEvidenceSource});
    const nativeTtsAcceptanceAssertions = (${nativeTtsAcceptanceAssertionsSource});
    const requiredScriptMatches = (${requiredScriptMatchesSource});
    const captureStart = window.__lfmFetchCapture?.length || 0;
    const inspectionStart = window.__lfmFetchInspectionPromises?.length || 0;
    const nativeTtsCaptureStart = window.__lfmNativeTtsCapture?.length || 0;
    const nativeTtsCaptureSupportedAtStart = Boolean(
      window.__lfmAcceptanceNativeTtsHandler
      && window.onNativeTtsEvent === window.__lfmAcceptanceNativeTtsHandler
      && Array.isArray(window.__lfmNativeTtsCapture)
    );
    const historyBefore = conversationHistory.summary();
    const preModeState = {
      mode,
      inputBusy,
      historyEnabled: isHistoryEnabled(),
      historySwitchAriaChecked: historyToggleButton.getAttribute("aria-checked"),
      nativeRecording,
      mediaRecorderState: mediaRecorder?.state || null,
      vadRequested: vadState.requested,
      vadStopping: vadState.stopping,
      currentTurnId: turnCoordinator.current?.id || null,
      currentTurnTerminal: turnCoordinator.current?.terminal ?? null
    };
    languageSelect.value = testCase.language;
    languageSelect.dispatchEvent(new Event("change"));
    if (testCase.englishResponseStyle) {
      englishResponseStyleSelect.value = testCase.englishResponseStyle;
      englishResponseStyleSelect.dispatchEvent(new Event("change"));
    }
    setMode(testCase.mode);
    if (mode !== testCase.mode) throw new Error(\`Could not select mode \${testCase.mode}; live mode is \${mode}\`);
    let ttsPolicySetup = null;
    if (testCase.ttsPolicy) {
      const accepted = applyTtsEngine(testCase.ttsPolicy, { silent: true, persist: false });
      ttsPolicySetup = {
        requested: testCase.ttsPolicy,
        accepted,
        live: currentTtsPolicy()
      };
      if (!accepted || currentTtsPolicy() !== testCase.ttsPolicy) {
        throw new Error(\`Could not select TTS policy \${testCase.ttsPolicy}\`);
      }
    }
    const turn = beginInputTurn("cdp_fixture");
    setInputBusy(true);
    const started = performance.now();
    let thrown = null;
    let drain = null;
    try {
      await sendToPhone(encodedWav, turn);
      if (turn.responseMode === "tts") {
        const deadline = performance.now() + 90_000;
        while (ttsClientTurns.has(turn.id) && performance.now() < deadline) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        drain = {
          status: ttsClientTurns.has(turn.id) ? "tts_terminal_timeout" : "tts_terminal"
        };
      } else if (turn.responseMode === "chat" && turn.streamedPlayback) {
        drain = await waitForStreamingPlaybackDrain(turn);
      } else if (turn.responseMode === "chat" && turn.htmlAudioFallback) {
        drain = await waitForAudioElementDrain(turn);
      } else {
        drain = { status: "no_pending_audio" };
      }
    } catch (error) {
      thrown = {
        name: error?.name || "Error",
        message: error?.message || String(error),
        stack: error?.stack || null
      };
      if (!isAbortError(error)) showError(thrown.message);
    } finally {
      setInputBusy(false);
      try {
        await scheduleManualTurnTerminal(turn);
      } catch (_) {}
      if (turnCoordinator.isCurrent(turn)) {
        turnCoordinator.markTerminal(turn);
        turnCoordinator.clearIfCurrent(turn);
      }
    }
    const pendingInspections = (window.__lfmFetchInspectionPromises || []).slice(inspectionStart);
    if (pendingInspections.length) await Promise.allSettled(pendingInspections);
    const fetches = (window.__lfmFetchCapture || []).slice(captureStart);
    const asrFetches = fetches.filter(item => item?.stage === "asr");
    const companionFetches = fetches.filter(item => item?.stage === "asr_companion");
    const responseFetches = fetches.filter(item =>
      item?.stage === "response" || item?.stage === "response_companion"
    );
    const responseCompanionFetches = fetches.filter(item => item?.stage === "response_companion");
    const modelFetches = fetches.filter(item =>
      ["asr", "response", "response_companion", "unknown"].includes(item?.stage)
    );
    const validLfmAsr = asrFetches.length === 1
      && asrFetches.every(item => item.inputAudioParts === 1 && item.inputAudioKeys === 1);
    const validCompanionAsr = companionFetches.length >= 1
      && companionFetches.every(item =>
        item.method === "POST"
        && item.contentType?.toLocaleLowerCase().startsWith("audio/wav")
        && Number(item.bodyBytes) > 44
      );
    const expectsResponseRequest = ["chat", "tts"].includes(testCase.mode) && !thrown;
    const protocolAssertions = {
      asrRequestPlanValid: (validLfmAsr && companionFetches.length === 0)
        || (validCompanionAsr && asrFetches.length === 0),
      asrEachRequestHasOneInput: (validLfmAsr && companionFetches.length === 0)
        || (validCompanionAsr && asrFetches.length === 0),
      companionUsesRawWav: companionFetches.length
        ? validCompanionAsr
        : null,
      responseRequestPresent: expectsResponseRequest ? responseFetches.length === 1 : null,
      responseContainsNoInputAudio: expectsResponseRequest
        ? responseFetches.length === 1 && responseFetches.every(item =>
          item.inputAudioParts === 0 && item.inputAudioKeys === 0
        )
        : null,
      responseUsesTextUserContent: expectsResponseRequest
        ? responseFetches.length === 1
          && responseFetches.every(item => item.userContentType === "string")
        : null,
      asrRolesOnlySystemAndUser: asrFetches.every(item =>
        Array.isArray(item.roles) && item.roles.every(role => role === "system" || role === "user")
      ),
      responseRolesSupported: expectsResponseRequest
        ? responseFetches.length === 1 && responseFetches.every(item =>
          Array.isArray(item.roles)
          && item.roles.length >= 2
          && item.roles[0] === "system"
          && item.roles.at(-1) === "user"
          && item.roles.every(role => ["system", "user", "assistant"].includes(role))
        )
        : null,
      responseThinkingDisabled: responseCompanionFetches.length
        ? responseThinkingDisabled(responseCompanionFetches)
        : null
    };
    const observedOutcome = thrown ? "blocked" : "complete";
    const asrFacts = turn.languageRoute?.asrFacts || null;
    const responseRequestText = responseFetches.at(-1)?.userText || null;
    const responseFetch = responseFetches.at(-1) || null;
    const completedResponseText = responseEl.textContent;
    const completedTranscript = turn.transcript
      || (turn.responseMode === "asr" ? responseEl.textContent : "");
    const responseScriptEvidence = scriptEvidence(completedResponseText);
    const transcriptScriptEvidence = scriptEvidence(completedTranscript);
    const responseRepetition = measureTextRepetition(completedResponseText);
    const responseQuality = {
      characters: completedResponseText.length,
      scriptEvidence: responseScriptEvidence,
      repetition: responseRepetition
    };
    const historyAfter = conversationHistory.summary();
    const expected = testCase.expectations;
    const nativeTtsEvents = Array.isArray(window.__lfmNativeTtsCapture)
      ? window.__lfmNativeTtsCapture.slice(nativeTtsCaptureStart)
      : [];
    const nativeTtsCaptureSupported = nativeTtsCaptureSupportedAtStart
      && window.onNativeTtsEvent === window.__lfmAcceptanceNativeTtsHandler;
    const nativeTtsEvidence = collectNativeTtsEvidence(
      nativeTtsEvents,
      turn.id,
      nativeTtsCaptureSupported
    );
    const expectsNativeTtsMetrics = expected.ttsGenerationLanguage !== null
      || expected.maxTtsFirstChunkMs !== null
      || expected.maxTtsRtf !== null
      || expected.maxTtsDroppedChunks !== null;
    const nativeTtsAssertions = nativeTtsAcceptanceAssertions(
      nativeTtsEvidence,
      expected.ttsTerminalStatus,
      expected.minNativeTtsPlaybackMs,
      {
        policy: expected.ttsPolicy,
        engine: expected.ttsEngine,
        requestedLocale: expected.ttsRequestedLocale,
        resolvedLocale: expected.ttsResolvedLocale,
        generationLanguage: expected.ttsGenerationLanguage,
        generation: expectsNativeTtsMetrics ? turn.generation : null,
        maxFirstChunkMs: expected.maxTtsFirstChunkMs,
        maxRtf: expected.maxTtsRtf,
        maxDroppedChunks: expected.maxTtsDroppedChunks
      }
    );
    const selectedAsrCandidate = Array.isArray(asrFacts?.candidate_summaries)
      ? asrFacts.candidate_summaries.find(candidate =>
        candidate?.backend === asrFacts?.backend
        && candidate?.provider === asrFacts?.provider) || null
      : null;
    const acceptanceAssertions = {
      outcome: expected.outcome
        ? observedOutcome === expected.outcome
        : null,
      resolvedLanguage: expected.resolvedLanguage
        ? turn.languageRoute?.language === expected.resolvedLanguage
        : null,
      responseMode: expected.responseMode
        ? turn.responseMode === expected.responseMode
        : null,
      voiceBackend: expected.voiceBackend
        ? turn.voiceBackend === expected.voiceBackend
        : null,
      responseBackend: expected.responseBackend
        ? responseFetch?.requestBackend === expected.responseBackend
        : null,
      responseModel: expected.responseModel
        ? (responseFetch?.responseModelIdentity || responseFetch?.requestModelIdentity)
          === expected.responseModel
        : null,
      finishReason: expected.finishReason
        ? responseFetch?.finishReason === expected.finishReason
        : null,
      noReasoning: expected.noReasoning === null
        ? null
        : expected.noReasoning === (
          !completedResponseText.toLocaleLowerCase().includes("<think")
          && !responseFetch?.thinkTagSeen
          && Number(responseFetch?.reasoningContentCharacters || 0) === 0
        ),
      maxFirstAudioMs: expected.maxFirstAudioMs === null
        ? null
        : Number.isFinite(turn.firstAudioScheduledMs)
          && turn.firstAudioScheduledMs <= expected.maxFirstAudioMs,
      maxFirstVoicedAudioMs: expected.maxFirstVoicedAudioMs === null
        ? null
        : Number.isFinite(turn.firstVoicedOutputHeadMs ?? turn.firstVoicedScheduledMs)
          && (turn.firstVoicedOutputHeadMs ?? turn.firstVoicedScheduledMs)
            <= expected.maxFirstVoicedAudioMs,
      minPcmDurationMs: expected.minPcmDurationMs === null
        ? null
        : Number.isFinite(turn.pcmDurationSeconds)
          && turn.pcmDurationSeconds * 1000 >= expected.minPcmDurationMs,
      ...nativeTtsAssertions,
      asrBackend: expected.asrBackend
        ? asrFacts?.backend === expected.asrBackend
        : null,
      asrProvider: expected.asrProvider
        ? asrFacts?.provider === expected.asrProvider
        : null,
      asrLanguageHint: expected.asrLanguageHint
        ? selectedAsrCandidate?.language_hint === expected.asrLanguageHint
        : null,
      npuExecuted: expected.npuExecuted === null
        ? null
        : typeof asrFacts?.npu_executed === "boolean"
          && asrFacts.npu_executed === expected.npuExecuted,
      npuSelected: expected.npuSelected === null
        ? null
        : typeof asrFacts?.npu_selected === "boolean"
          && asrFacts.npu_selected === expected.npuSelected,
      errorIncludes: expected.errorIncludes
        ? Boolean(thrown?.message?.includes(expected.errorIncludes))
        : null,
      contextEnabled: testCase.contextEnabled === null
        ? null
        : isHistoryEnabled() === testCase.contextEnabled,
      historyTurnsAfter: expected.historyTurnsAfter === null
        ? null
        : historyAfter.turns === expected.historyTurnsAfter,
      responseRequestIncludes: expected.responseRequestIncludes.length
        ? Boolean(responseRequestText)
          && expected.responseRequestIncludes.every(value =>
            responseRequestText.toLocaleLowerCase().includes(String(value).toLocaleLowerCase())
          )
        : null,
      responseRequestExcludes: expected.responseRequestExcludes.length
        ? expected.responseRequestExcludes.every(value =>
          !String(responseRequestText || "").toLocaleLowerCase().includes(String(value).toLocaleLowerCase())
        )
        : null,
      companionLanguages: expected.companionLanguages.length
        ? JSON.stringify(companionFetches.map(item => item.requestedLanguage))
          === JSON.stringify(expected.companionLanguages)
        : null,
      responseIncludes: expected.responseIncludes.length
        ? expected.responseIncludes.every(value =>
          completedResponseText.toLocaleLowerCase().includes(String(value).toLocaleLowerCase())
        )
        : null,
      responseExcludes: expected.responseExcludes.length
        ? expected.responseExcludes.every(value =>
          !completedResponseText.toLocaleLowerCase().includes(String(value).toLocaleLowerCase())
        )
        : null,
      transcriptIncludes: expected.transcriptIncludes.length
        ? expected.transcriptIncludes.every(value =>
          completedTranscript.toLocaleLowerCase().includes(String(value).toLocaleLowerCase())
        )
        : null,
      transcriptScript: requiredScriptMatches(
        transcriptScriptEvidence,
        expected.requiredTranscriptScript
      ),
      responseScript: requiredScriptMatches(
        responseScriptEvidence,
        expected.requiredResponseScript
      ),
      minResponseCharacters: expected.minResponseCharacters === null
        ? null
        : completedResponseText.length >= expected.minResponseCharacters,
      maxDuplicateGramRatio: expected.maxDuplicateGramRatio === null
        ? null
        : responseRepetition.duplicate_gram_ratio <= expected.maxDuplicateGramRatio,
      protocol: Object.values(protocolAssertions).every(value => value !== false)
    };
    const classifiedAssertions = Object.values(acceptanceAssertions)
      .filter(value => typeof value === "boolean");
    acceptanceAssertions.allPassed = classifiedAssertions.length
      ? classifiedAssertions.every(Boolean)
      : null;
    return {
      name: testCase.name,
      requestedMode: testCase.mode,
      selectedLanguage: testCase.language,
      requestedEnglishResponseStyle: testCase.englishResponseStyle,
      selectedEnglishResponseStyle: turn.englishResponseStyle || null,
      serverWarmth: testCase.serverWarmth,
      contextEnabledRequested: testCase.contextEnabled,
      ttsPolicySetup,
      historyEnabled: isHistoryEnabled(),
      historySwitchAriaChecked: historyToggleButton.getAttribute("aria-checked"),
      preModeState,
      turnId: turn.id,
      generation: turn.generation,
      elapsedMs: performance.now() - started,
      thrown,
      drain,
      responseMode: turn.responseMode || null,
      voiceBackend: turn.voiceBackend || null,
      playback: {
        firstTextMs: turn.firstTextMs ?? null,
        firstPcmMs: turn.firstPcmMs ?? null,
        firstAudioScheduledMs: turn.firstAudioScheduledMs ?? null,
        firstOutputHeadMs: turn.firstOutputHeadMs ?? null,
        firstVoicedPcmOffsetMs: turn.firstVoicedPcmOffsetMs ?? null,
        firstVoicedScheduledMs: turn.firstVoicedScheduledMs ?? null,
        firstVoicedOutputHeadMs: turn.firstVoicedOutputHeadMs ?? null,
        underflows: turn.playbackUnderflows ?? null,
        pcmDurationSeconds: turn.pcmDurationSeconds ?? null,
        audioChunks: turn.audioChunkCount ?? null,
        completedMs: turn.completedMs ?? null
      },
      languageRoute: turn.languageRoute || null,
      asrFacts,
      observedOutcome,
      expectations: expected,
      acceptanceAssertions,
      response: completedResponseText,
      transcript: completedTranscript,
      transcriptScriptEvidence,
      responseQuality,
      responseRequestText,
      languageStatus: languageStatus.textContent,
      historyStatus: historyStatus.textContent,
      historyBefore,
      historyAfter,
      historySnapshot: conversationHistory.snapshot(),
      error: errorEl.textContent,
      timer: timer.textContent,
      recordLabel: recordLabel.textContent,
      fetches,
      nativeTtsEvents,
      nativeTtsEvidence,
      protocolAssertions
    };
  })()`;
}

function persist(outputPath, payload) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

export function prepareAcceptanceAudioFixtures(plan, cwd = process.cwd()) {
  return (plan?.cases || []).map((testCase, index) => {
    const audioPath = path.resolve(cwd, testCase.audio);
    let audioBytes;
    try {
      audioBytes = fs.readFileSync(audioPath);
    } catch (error) {
      throw new Error(
        `cases[${index}].audio could not be read: ${testCase.audio}: ${error.message}`,
      );
    }
    const actualSha256 = createHash("sha256").update(audioBytes).digest("hex");
    const expectedSha256 = typeof testCase.expectedAudioSha256 === "string"
      ? testCase.expectedAudioSha256
      : null;
    if (expectedSha256 !== null && actualSha256 !== expectedSha256) {
      throw new Error(
        `cases[${index}].audio SHA-256 mismatch for ${testCase.audio}: `
          + `expected ${expectedSha256}, actual ${actualSha256}`,
      );
    }
    return {
      path: audioPath,
      bytes: audioBytes.length,
      expectedSha256,
      actualSha256,
      sha256Matched: expectedSha256 === null ? null : true,
      encodedWav: audioBytes.toString("base64"),
    };
  });
}

export function summarizePayload(payload) {
  const passed = payload.cases.filter(item =>
    item?.acceptanceAssertions?.allPassed === true).length;
  const cleanupPassed = payload.cleanup?.removed === true
    && payload.cleanup?.nativeTtsHandlerRestored !== false
    && (payload.cleanup?.nativeTtsOriginalHandlerErrors || []).length === 0;
  return {
    total: payload.cases.length,
    passed,
    failed: payload.cases.length - passed,
    caseAssertionsPassed: payload.cases.length > 0 && passed === payload.cases.length,
    cleanupPassed,
    operationalFailure: Boolean(payload.failure),
    allPassed: payload.cases.length > 0
      && passed === payload.cases.length
      && cleanupPassed
      && !payload.failure,
    failedCases: payload.cases
      .filter(item => item?.acceptanceAssertions?.allPassed !== true)
      .map(item => item?.name || "unnamed"),
  };
}

export async function main(argv = process.argv) {
  const args = parseArgs(argv);
  const plan = JSON.parse(fs.readFileSync(args.plan, "utf8"));
  const planErrors = validateAcceptancePlan(plan);
  for (const [index, testCase] of (plan.cases || []).entries()) {
    if (typeof testCase?.audio === "string" && !fs.existsSync(path.resolve(testCase.audio))) {
      planErrors.push(`cases[${index}].audio does not exist: ${testCase.audio}`);
    }
  }
  if (planErrors.length) {
    throw new Error(`Acceptance plan is unsafe:\n- ${planErrors.join("\n- ")}`);
  }
  // Freeze and authenticate every fixture before target discovery opens CDP.
  // Cases consume these exact verified bytes, so a later filesystem change
  // cannot make the report hash differ from the audio submitted to the app.
  const audioFixtures = prepareAcceptanceAudioFixtures(plan);
  const outputPath = path.resolve(args.output);
  const target = await discoverTarget(args.host, args.port);
  const client = new CdpClient(target.webSocketDebuggerUrl, args.timeoutMs);
  const payload = {
    schemaVersion: 2,
    startedAt: new Date().toISOString(),
    target: {
      id: target.id,
      title: target.title,
      url: target.url,
      description: target.description,
    },
    plan,
    staticInspection: null,
    responseServicePreflight: null,
    setup: null,
    cases: [],
    failure: null,
    summary: null,
  };

  try {
    await client.connect();
    payload.staticInspection = await client.evaluate(STATIC_INSPECTION, 30_000);
    const expectedModels = [...new Set([
      plan.expectedResponseModel,
      ...(plan.cases || []).map(item => item.expectedResponseModel)
    ].filter(Boolean))];
    const expectedBackends = [...new Set([
      plan.expectedResponseBackend,
      ...(plan.cases || []).map(item => item.expectedResponseBackend)
    ].filter(Boolean))];
    if (expectedModels.length > 1) {
      throw new Error(`Acceptance plan contains conflicting response models: ${expectedModels.join(", ")}`);
    }
    if (expectedBackends.length > 1) {
      throw new Error(`Acceptance plan contains conflicting response backends: ${expectedBackends.join(", ")}`);
    }
    const responseService = payload.staticInspection?.responseCompanion || {};
    const directResponse = payload.staticInspection?.directResponse || {};
    const expectedBackend = expectedBackends[0] || null;
    const expectedModel = expectedModels[0] || null;
    const expectsDirectResponse = Boolean(
      expectedBackend && expectedBackend === directResponse.backend
    ) || Boolean(
      expectedModel && expectedModel === directResponse.requestModel
    );
    const modelIds = expectsDirectResponse
      ? [directResponse.requestModel].filter(Boolean)
      : (Array.isArray(responseService.modelIds) ? responseService.modelIds : []);
    payload.responseServicePreflight = {
      route: expectsDirectResponse ? "direct_lfm" : "qwen_companion",
      healthOk: expectsDirectResponse
        ? directResponse.health?.httpStatus !== null
        : responseService.health?.httpOk === true,
      modelsOk: expectsDirectResponse ? null : responseService.models?.httpOk === true,
      expectedModel,
      observedModels: modelIds,
      modelIdentityMatched: expectedModels.length
        ? modelIds.length === 1 && modelIds[0] === expectedModels[0]
        : null,
      expectedBackend,
      observedBackend: expectsDirectResponse
        ? (directResponse.backend || null)
        : (responseService.backend || null),
      backendMatched: expectedBackends.length
        ? (expectsDirectResponse ? directResponse.backend : responseService.backend)
          === expectedBackends[0]
        : null
    };
    const preflightFailures = Object.entries(payload.responseServicePreflight)
      .filter(([key, value]) => ["healthOk", "modelsOk", "modelIdentityMatched", "backendMatched"].includes(key)
        && value === false)
      .map(([key]) => key);
    if (preflightFailures.length) {
      throw new Error(`Response-service preflight failed: ${preflightFailures.join(", ")}`);
    }
    payload.setup = {
      fetchCapture: await client.evaluate(INSTALL_FETCH_CAPTURE, 30_000),
      clear: await client.evaluate(clearHistoryExpression(Boolean(plan.resetTelemetry)), 30_000),
    };
    persist(outputPath, payload);

    for (const [caseIndex, testCase] of (plan.cases || []).entries()) {
      const effectiveCase = {
        ...testCase,
        englishResponseStyle: testCase.englishResponseStyle
          || plan.englishResponseStyle
          || null,
        expectedResponseBackend: testCase.expectedResponseBackend
          || plan.expectedResponseBackend
          || null,
        expectedResponseModel: testCase.expectedResponseModel
          || plan.expectedResponseModel
          || null,
        expectedFinishReason: testCase.expectedFinishReason
          || plan.expectedFinishReason
          || null,
        expectedNoReasoning: typeof testCase.expectedNoReasoning === "boolean"
          ? testCase.expectedNoReasoning
          : (typeof plan.expectedNoReasoning === "boolean" ? plan.expectedNoReasoning : null),
        maxFirstAudioMs: Number.isFinite(testCase.maxFirstAudioMs)
          ? testCase.maxFirstAudioMs
          : (["lfm_audio_natural_tts_streaming", "lfm_audio_interleaved_streaming"]
            .includes(testCase.expectedVoiceBackend)
            && Number.isFinite(plan.maxFirstAudioMs) ? plan.maxFirstAudioMs : null),
        maxFirstVoicedAudioMs: Number.isFinite(testCase.maxFirstVoicedAudioMs)
          ? testCase.maxFirstVoicedAudioMs
          : (["lfm_audio_natural_tts_streaming", "lfm_audio_interleaved_streaming"]
            .includes(testCase.expectedVoiceBackend)
            && Number.isFinite(plan.maxFirstVoicedAudioMs)
            ? plan.maxFirstVoicedAudioMs
            : null),
        expectedTtsTerminalStatus: testCase.expectedTtsTerminalStatus
          || plan.expectedTtsTerminalStatus
          || null
      };
      if (effectiveCase.clearHistoryBefore) {
        await client.evaluate(clearHistoryExpression(false), 30_000);
      }
      let contextSetup = null;
      if (typeof effectiveCase.contextEnabled === "boolean") {
        contextSetup = await client.evaluate(
          setContextExpression(effectiveCase.contextEnabled),
          30_000
        );
      }
      const audioFixture = audioFixtures[caseIndex];
      process.stderr.write(`CASE_START ${effectiveCase.name} ${effectiveCase.mode}/${effectiveCase.language}\n`);
      const result = await client.evaluate(buildCaseExpression(
        effectiveCase,
        audioFixture.encodedWav,
      ),
        Number(effectiveCase.timeoutMs || args.timeoutMs));
      result.audio = {
        path: audioFixture.path,
        bytes: audioFixture.bytes,
        expectedSha256: audioFixture.expectedSha256,
        actualSha256: audioFixture.actualSha256,
        sha256Matched: audioFixture.sha256Matched,
      };
      result.contextSetup = contextSetup;
      payload.cases.push(result);
      persist(outputPath, payload);
      process.stderr.write(`CASE_DONE ${effectiveCase.name} ${Math.round(result.elapsedMs)}ms error=${result.thrown?.message || "none"} acceptance=${result.acceptanceAssertions?.allPassed ?? "unclassified"}\n`);
      await sleep(Number(effectiveCase.cooldownMs || plan.cooldownMs || 500));
    }
  } catch (error) {
    payload.failure = {
      at: new Date().toISOString(),
      name: error?.name || "Error",
      message: error?.message || String(error),
      stack: error?.stack || null,
    };
    persist(outputPath, payload);
    throw error;
  } finally {
    try {
      payload.cleanup = await client.evaluate(REMOVE_ACCEPTANCE_HOOKS, 30_000);
    } catch (cleanupError) {
      payload.cleanup = { removed: false, error: String(cleanupError) };
    }
    payload.finishedAt = new Date().toISOString();
    payload.summary = summarizePayload(payload);
    if (!payload.failure && payload.summary.allPassed !== true) {
      payload.failure = {
        at: payload.finishedAt,
        name: "AcceptanceFailure",
        message: `Full-turn acceptance failed: ${JSON.stringify(payload.summary)}`,
        stack: null,
      };
      payload.summary = summarizePayload(payload);
    }
    persist(outputPath, payload);
    client.close();
  }
  process.stdout.write(`${JSON.stringify({
    output: outputPath,
    summary: payload.summary,
    failure: payload.failure?.message || null,
  }, null, 2)}\n`);
  return payload.summary?.allPassed === true;
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
