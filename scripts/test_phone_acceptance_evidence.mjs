#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  collectNativeTtsEvidence,
  nativeTtsAcceptanceAssertions,
  requiredScriptMatches,
  responseThinkingDisabled,
  validateAcceptancePlan,
} from "./phone_acceptance_evidence.mjs";
import {
  prepareAcceptanceAudioFixtures,
  summarizePayload,
} from "./run_phone_webview_acceptance.mjs";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

const responseFetch = overrides => ({
  requestBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  requestModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  ...overrides,
});

assert.equal(responseThinkingDisabled([]), false);
assert.equal(responseThinkingDisabled([responseFetch({})]), true);
assert.equal(responseThinkingDisabled([responseFetch({ enableThinking: null })]), true);
assert.equal(responseThinkingDisabled([responseFetch({ enableThinking: false })]), true);
assert.equal(responseThinkingDisabled([responseFetch({ enableThinking: true })]), false);
assert.equal(responseThinkingDisabled([{
  requestBackend: null,
  requestModelIdentity: "Qwen2.5-1.5B-Instruct-Q4_K_M",
}]), true);
assert.equal(responseThinkingDisabled([{
  requestBackend: "qwen3_1_7b_q4_cpu",
  requestModel: "Qwen3-1.7B-Q4_0",
  enableThinking: false,
}]), true);
assert.equal(responseThinkingDisabled([{
  requestBackend: "qwen3_1_7b_q4_cpu",
  requestModel: "Qwen3-1.7B-Q4_0",
  enableThinking: null,
}]), false);
assert.equal(responseThinkingDisabled([{
  requestBackend: "qwen3_1_7b_q4_cpu",
  requestModel: "Qwen3-1.7B-Q4_0",
}]), false);
assert.equal(responseThinkingDisabled([{
  requestBackend: "qwen3_1_7b_q4_cpu",
  requestModel: "Qwen3-1.7B-Q4_0",
  enableThinking: true,
}]), false);
assert.equal(responseThinkingDisabled([
  responseFetch({}),
  {
    requestBackend: "qwen3_1_7b_q4_cpu",
    requestModel: "Qwen3-1.7B-Q4_0",
    enableThinking: null,
  },
]), false);

const scriptFacts = overrides => ({
  letters: 10,
  han: 0,
  hiragana: 0,
  katakana: 0,
  kana: 0,
  hangul: 0,
  cyrillic: 0,
  latin: 0,
  hanRatio: 0,
  japaneseRatio: 0,
  hangulRatio: 0,
  cyrillicRatio: 0,
  latinRatio: 0,
  ...overrides,
});
assert.equal(requiredScriptMatches(scriptFacts({ kana: 4, japaneseRatio: 0.8 }), "japanese"), true);
assert.equal(requiredScriptMatches(scriptFacts({ han: 4, japaneseRatio: 0.8 }), "japanese"), false);
assert.equal(requiredScriptMatches(scriptFacts({ hangul: 2, hangulRatio: 0.2 }), "korean"), false);
assert.equal(requiredScriptMatches(scriptFacts({ hangul: 6, hangulRatio: 0.6 }), "korean"), true);
assert.equal(requiredScriptMatches(scriptFacts({ latin: 8, hangul: 2, hangulRatio: 0.2 }), "korean"), false);

const completePlanCase = {
  name: "ja_full_turn",
  mode: "chat",
  language: "ja",
  audio: "question_ja.wav",
  expectedAudioSha256: "a".repeat(64),
  expectedOutcome: "complete",
  expectedResolvedLanguage: "ja",
  expectedResponseMode: "tts",
  expectedVoiceBackend: "android_native_tts",
  expectedAsrBackend: "sensevoice_qnn",
  expectedAsrProvider: "qnn",
  expectedNpuExecuted: true,
  expectedNpuSelected: true,
  expectedAsrLanguageHint: "ja",
  ttsPolicy: "auto",
  expectedTtsPolicy: "auto",
  expectedTtsEngine: "supertonic3_ja_s5",
  expectedTtsRequestedLocale: "ja-JP",
  expectedTtsResolvedLocale: "ja",
  expectedTtsGenerationLanguage: "ja",
  maxTtsFirstChunkMs: 1200,
  maxTtsRtf: 0.8,
  maxTtsDroppedChunks: 0,
  expectedTtsTerminalStatus: "completed",
  minNativeTtsPlaybackMs: 500,
  expectedTranscriptScript: "japanese",
  expectedResponseIncludes: "パリ",
  expectedResponseScript: "japanese",
  expectedMinResponseCharacters: 4,
  expectedMaxDuplicateGramRatio: 0.2,
};
assert.deepEqual(validateAcceptancePlan({
  schema_version: 3,
  expectedResponseBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  expectedResponseModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  expectedFinishReason: "stop",
  expectedNoReasoning: true,
  cases: [completePlanCase],
}), []);
assert.deepEqual(validateAcceptancePlan({
  schema_version: 3,
  expectedResponseBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  expectedResponseModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  expectedFinishReason: "stop",
  expectedNoReasoning: true,
  cases: [{ ...completePlanCase, name: "en_native_tts", mode: "tts" }],
}), []);
const schema3WithoutAudioHash = structuredClone(completePlanCase);
delete schema3WithoutAudioHash.expectedAudioSha256;
const missingAudioHashErrors = validateAcceptancePlan({
  schema_version: 3,
  expectedResponseBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  expectedResponseModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  expectedFinishReason: "stop",
  expectedNoReasoning: true,
  cases: [schema3WithoutAudioHash],
});
assert(missingAudioHashErrors.some(value => value.includes("expectedAudioSha256")));
const uppercaseAudioHash = {
  ...completePlanCase,
  expectedAudioSha256: "A".repeat(64),
};
assert(validateAcceptancePlan({
  schema_version: 3,
  expectedResponseBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  expectedResponseModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  expectedFinishReason: "stop",
  expectedNoReasoning: true,
  cases: [uppercaseAudioHash],
}).some(value => value.includes("expectedAudioSha256")));
const schema3ChatWithoutTranscript = structuredClone(completePlanCase);
delete schema3ChatWithoutTranscript.expectedTranscriptScript;
const missingChatTranscriptErrors = validateAcceptancePlan({
  schema_version: 3,
  expectedResponseBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  expectedResponseModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  expectedFinishReason: "stop",
  expectedNoReasoning: true,
  cases: [schema3ChatWithoutTranscript],
});
assert(missingChatTranscriptErrors.some(value =>
  value.includes("requires a transcript content or script expectation")
));
assert.deepEqual(validateAcceptancePlan({
  schema_version: 2,
  expectedResponseBackend: "qwen2_5_1_5b_q4_k_m_cpu",
  expectedResponseModel: "Qwen2.5-1.5B-Instruct-Q4_K_M",
  expectedFinishReason: "stop",
  expectedNoReasoning: true,
  cases: [schema3ChatWithoutTranscript],
}), []);
const unsafePlanErrors = validateAcceptancePlan({ cases: [{ name: "unsafe" }] });
assert(unsafePlanErrors.some(value => value.includes("expectedOutcome")));
assert(unsafePlanErrors.some(value => value.includes("audio")));

const fixturePath = path.join(SCRIPT_DIR, "phone_acceptance_fixtures/question_ja.wav");
const fixtureBytes = fs.readFileSync(fixturePath);
const fixtureSha256 = createHash("sha256").update(fixtureBytes).digest("hex");
const [preparedFixture] = prepareAcceptanceAudioFixtures({
  cases: [{
    name: "fixture_provenance",
    audio: fixturePath,
    expectedAudioSha256: fixtureSha256,
  }],
});
assert.equal(preparedFixture.path, fixturePath);
assert.equal(preparedFixture.bytes, fixtureBytes.length);
assert.equal(preparedFixture.expectedSha256, fixtureSha256);
assert.equal(preparedFixture.actualSha256, fixtureSha256);
assert.equal(preparedFixture.sha256Matched, true);
assert.equal(Buffer.from(preparedFixture.encodedWav, "base64").length, fixtureBytes.length);
assert.throws(
  () => prepareAcceptanceAudioFixtures({
    cases: [{
      name: "fixture_hash_mismatch",
      audio: fixturePath,
      expectedAudioSha256: "0".repeat(64),
    }],
  }),
  error => error.message.includes("SHA-256 mismatch")
    && error.message.includes(`actual ${fixtureSha256}`),
);

const record = (type, monotonicMs, details = {}, overrides = {}) => ({
  event: {
    schema_version: 1,
    type,
    turn_id: "turn-test",
    monotonic_ms: monotonicMs,
    details,
  },
  parse_error: null,
  original_handler_error: null,
  ...overrides,
});

const evidence = collectNativeTtsEvidence([
  record("ready", 900, {
    generation: 42,
    tts_policy: "auto",
    tts_engine_id: "supertonic3_ja_s5",
    requested_locale: "ja-JP",
    resolved_locale: "ja",
  }),
  record("utterance_start", 1000, { generation: 42 }),
  record("engine_metrics", 1100, {
    engine_id: "supertonic3_ja_s5",
    generation: 42,
    engine_generation: 1,
    generation_language: "ja",
    first_chunk_ms: 260,
    rtf: 0.15,
    dropped_chunks: 0,
  }),
  record("utterance_done", 1500, { generation: 42 }),
  record("terminal", 1650, { generation: 42, status: "completed" }),
  {
    event: {
      type: "terminal",
      turn_id: "another-turn",
      monotonic_ms: 1700,
      details: { status: "failed" },
    },
  },
], "turn-test", true);
assert.equal(evidence.supported, true);
assert.equal(evidence.eventsCaptured, 5);
assert.equal(evidence.readyCount, 1);
assert.equal(evidence.readyEngine, "supertonic3_ja_s5");
assert.equal(evidence.startCount, 1);
assert.equal(evidence.doneCount, 1);
assert.equal(evidence.terminalCount, 1);
assert.equal(evidence.playbackSpanMs, 650);
assert.equal(evidence.terminalStatus, "completed");
assert.deepEqual(evidence.generationLanguages, ["ja"]);
assert.deepEqual(evidence.engineMetricGenerations, [42]);
assert.deepEqual(evidence.engineMetricEngineIds, ["supertonic3_ja_s5"]);
assert.equal(evidence.parseClean, true);
assert.equal(evidence.originalHandlerHealthy, true);
assert.deepEqual(
  nativeTtsAcceptanceAssertions(evidence, "completed", 500, {
    policy: "auto",
    engine: "supertonic3_ja_s5",
    requestedLocale: "ja-JP",
    resolvedLocale: "ja",
    generationLanguage: "ja",
    generation: 42,
    maxFirstChunkMs: 1200,
    maxRtf: 0.8,
    maxDroppedChunks: 0,
  }),
  {
    nativeTtsEvidenceSupported: true,
    nativeTtsEventsParseClean: true,
    nativeTtsOriginalHandlerHealthy: true,
    ttsPolicy: true,
    ttsEngine: true,
    ttsRequestedLocale: true,
    ttsResolvedLocale: true,
    ttsMetricsComplete: true,
    ttsMetricsEngine: true,
    ttsGenerationLanguage: true,
    ttsMetricsGeneration: true,
    maxTtsFirstChunkMs: true,
    maxTtsRtf: true,
    maxTtsDroppedChunks: true,
    ttsTerminalStatus: true,
    minNativeTtsPlaybackMs: true,
  },
);
assert.deepEqual(
  nativeTtsAcceptanceAssertions(evidence, null, null),
  {
    nativeTtsEvidenceSupported: null,
    nativeTtsEventsParseClean: null,
    nativeTtsOriginalHandlerHealthy: null,
    ttsPolicy: null,
    ttsEngine: null,
    ttsRequestedLocale: null,
    ttsResolvedLocale: null,
    ttsMetricsComplete: null,
    ttsMetricsEngine: null,
    ttsGenerationLanguage: null,
    ttsMetricsGeneration: null,
    maxTtsFirstChunkMs: null,
    maxTtsRtf: null,
    maxTtsDroppedChunks: null,
    ttsTerminalStatus: null,
    minNativeTtsPlaybackMs: null,
  },
);

const parseFailure = collectNativeTtsEvidence([
  record("utterance_start", 1000),
  record("terminal", 1500, { status: "completed" }, { parse_error: "bad JSON" }),
], "turn-test", true);
assert.equal(parseFailure.parseClean, false);

const handlerFailure = collectNativeTtsEvidence([
  record("utterance_start", 1000),
  record("terminal", 1500, { status: "completed" }, {
    original_handler_error: "handler failed",
  }),
], "turn-test", true);
assert.equal(handlerFailure.originalHandlerHealthy, false);

const unsupported = collectNativeTtsEvidence([], "turn-test", false);
assert.equal(unsupported.supported, false);
assert.equal(unsupported.unsupportedReason, "native_tts_event_capture_unavailable");
assert.equal(unsupported.playbackSpanMs, null);
assert.equal(unsupported.terminalStatus, null);
assert.deepEqual(
  nativeTtsAcceptanceAssertions(unsupported, "completed", 500),
  {
    nativeTtsEvidenceSupported: false,
    nativeTtsEventsParseClean: true,
    nativeTtsOriginalHandlerHealthy: true,
    ttsPolicy: null,
    ttsEngine: null,
    ttsRequestedLocale: null,
    ttsResolvedLocale: null,
    ttsMetricsComplete: null,
    ttsMetricsEngine: null,
    ttsGenerationLanguage: null,
    ttsMetricsGeneration: null,
    maxTtsFirstChunkMs: null,
    maxTtsRtf: null,
    maxTtsDroppedChunks: null,
    ttsTerminalStatus: false,
    minNativeTtsPlaybackMs: false,
  },
);

const failedTerminal = collectNativeTtsEvidence([
  record("utterance_start", 1000),
  record("terminal", 1700, { status: "failed" }),
], "turn-test", true);
const failedTerminalAssertions = nativeTtsAcceptanceAssertions(
  failedTerminal,
  "completed",
  500,
);
assert.equal(failedTerminalAssertions.ttsTerminalStatus, false);
assert.equal(failedTerminalAssertions.minNativeTtsPlaybackMs, true);

const partialMetrics = collectNativeTtsEvidence([
  record("ready", 900, {
    tts_policy: "auto",
    tts_engine_id: "supertonic3_ja_s5",
  }),
  record("utterance_start", 1000),
  record("utterance_start", 1010),
  record("engine_metrics", 1100, {
    engine_id: "supertonic3_ja_s5",
    generation: 42,
    generation_language: "ja",
    first_chunk_ms: 260,
    rtf: 0.15,
    dropped_chunks: 0,
  }),
  record("utterance_done", 1400),
  record("utterance_done", 1450),
  record("terminal", 1650, { status: "completed" }),
], "turn-test", true);
const partialAssertions = nativeTtsAcceptanceAssertions(
  partialMetrics,
  "completed",
  500,
  {
    engine: "supertonic3_ja_s5",
    generationLanguage: "ja",
    generation: 42,
  },
);
assert.equal(partialAssertions.ttsMetricsComplete, false);

const passingSummary = summarizePayload({
  cases: [{ name: "pass", acceptanceAssertions: { allPassed: true } }],
  cleanup: {
    removed: true,
    nativeTtsHandlerRestored: true,
    nativeTtsOriginalHandlerErrors: [],
  },
  failure: null,
});
assert.equal(passingSummary.allPassed, true);
const failedSummary = summarizePayload({
  cases: [{ name: "fail", acceptanceAssertions: { allPassed: false } }],
  cleanup: {
    removed: true,
    nativeTtsHandlerRestored: true,
    nativeTtsOriginalHandlerErrors: [],
  },
  failure: null,
});
assert.equal(failedSummary.allPassed, false);
assert.deepEqual(failedSummary.failedCases, ["fail"]);

console.log("phone acceptance evidence tests passed");
