#!/usr/bin/env node

import assert from "node:assert/strict";

import {
  CLEANUP_EXPRESSION,
  DEFAULT_CASES,
  INSTALL_HOOK_EXPRESSION,
  PREFLIGHT_EXPRESSION,
  applyFinalEventAudit,
  auditEventCapture,
  buildCaseExpression,
  buildEventExportExpression,
  parseArgs,
  validateCase,
} from "./run_phone_tts_acceptance.mjs";

const byName = name => DEFAULT_CASES.find(item => item.name === name);

function record(sequence, type, details = {}) {
  return {
    sequence,
    captured_performance_ms: sequence * 10,
    event: {
      schema_version: 1,
      type,
      turn_id: "turn-test",
      monotonic_ms: 1000 + sequence * 10,
      details: { generation: 42, ...details },
    },
  };
}

function result(events, overrides = {}) {
  const base = {
    turnId: "turn-test",
    generation: 42,
    driver: {
      setPolicyAccepted: true,
      queueDispatchSpanMs: 4,
      cancelIssuedSequence: null,
      terminalVerified: true,
      cleanupIncomplete: false,
    },
    catalogAfterReady: { resolved: "matcha_zh_baker" },
    capturedEvents: events,
    events,
    originalHandlerErrors: [],
    thrown: null,
  };
  return {
    ...base,
    ...overrides,
    driver: { ...base.driver, ...overrides.driver },
  };
}

{
  const args = parseArgs([
    "node",
    "harness",
    "--host",
    "localhost",
    "--port",
    "9333",
    "--cases",
    "auto_zh_matcha,auto_ru_piper_ruslan",
  ]);
  assert.equal(args.host, "localhost");
  assert.equal(args.port, 9333);
  assert.deepEqual(args.caseNames, ["auto_zh_matcha", "auto_ru_piper_ruslan"]);
}

{
  const testCase = byName("auto_zh_matcha");
  const events = [
    record(0, "ready", {
      tts_policy: "auto",
      tts_engine_id: "matcha_zh_baker",
      requested_locale: "zh-CN",
      resolved_locale: "zh",
    }),
    record(1, "chunk_queued", { chunk_id: "chunk-0" }),
    record(2, "queue_closed", { pending_utterances: 1 }),
    record(3, "utterance_start", { chunk_id: "chunk-0" }),
    record(4, "engine_metrics", {
      engine_id: "matcha_zh_baker",
      generation_language: "ko",
      first_chunk_ms: 480,
      rtf: 0.27,
      dropped_chunks: 0,
    }),
    record(5, "utterance_done", { chunk_id: "chunk-0", pending_utterances: 0 }),
    record(6, "terminal", { status: "completed" }),
  ];
  const acceptance = validateCase(testCase, result(events));
  assert.equal(acceptance.allPassed, true);
  assert.equal(acceptance.observed.readyEngine, "matcha_zh_baker");
  assert.deepEqual(acceptance.observed.engineMetricEngineIds, ["matcha_zh_baker"]);

  const missingMetrics = events.filter(item => item.event.type !== "engine_metrics");
  const missingMetricsFailure = validateCase(testCase, result(missingMetrics));
  assert.equal(missingMetricsFailure.allPassed, false);
  assert.equal(missingMetricsFailure.assertions.engineMetricsSeen, false);
  assert.equal(missingMetricsFailure.assertions.engineMetricsEngineMatched, false);

  const duplicateMetrics = [
    ...events.slice(0, 5),
    record(5, "engine_metrics", { engine_id: "matcha_zh_baker" }),
    ...events.slice(5).map(item => ({
      ...structuredClone(item),
      sequence: item.sequence + 1,
    })),
  ];
  const duplicateMetricsFailure = validateCase(testCase, result(duplicateMetrics));
  assert.equal(duplicateMetricsFailure.allPassed, false);
  assert.equal(duplicateMetricsFailure.assertions.engineMetricsSeen, false);
  assert.equal(duplicateMetricsFailure.assertions.engineMetricsEngineMatched, false);

  const wrongEngineMetrics = events.map(item => structuredClone(item));
  wrongEngineMetrics.find(item => item.event.type === "engine_metrics")
    .event.details.engine_id = "supertonic3_ko_s5";
  const wrongEngineFailure = validateCase(testCase, result(wrongEngineMetrics));
  assert.equal(wrongEngineFailure.allPassed, false);
  assert.equal(wrongEngineFailure.assertions.engineMetricsSeen, true);
  assert.equal(wrongEngineFailure.assertions.engineMetricsEngineMatched, false);

  const missingGeneration = events.map(item => structuredClone(item));
  delete missingGeneration[0].event.details.generation;
  const generationFailure = validateCase(testCase, result(missingGeneration));
  assert.equal(generationFailure.allPassed, false);
  assert.equal(generationFailure.assertions.generationFieldsPresent, false);

  const engineGenerationOnly = events.map(item => structuredClone(item));
  const engineMetric = engineGenerationOnly.find(item =>
    item.event.type === "engine_metrics");
  delete engineMetric.event.details.generation;
  engineMetric.event.details.engine_generation = 1;
  const engineGenerationFailure = validateCase(
    testCase,
    result(engineGenerationOnly),
  );
  assert.equal(engineGenerationFailure.allPassed, false);
  assert.equal(engineGenerationFailure.assertions.generationFieldsPresent, false);
  assert.equal(
    engineGenerationFailure.assertions.engineMetricsGenerationMatched,
    false,
  );
}

{
  const testCase = byName("auto_ko_supertonic");
  const events = [
    record(0, "ready", {
      tts_policy: "auto",
      tts_engine_id: "supertonic3_ko_s5",
      requested_locale: "ko-KR",
      resolved_locale: "ko",
    }),
    record(1, "chunk_queued", { chunk_id: "chunk-0" }),
    record(2, "queue_closed", { pending_utterances: 1 }),
    record(3, "utterance_start", { chunk_id: "chunk-0" }),
    record(4, "engine_metrics", {
      engine_id: "supertonic3_ko_s5",
      generation_language: "ko",
      first_chunk_ms: 480,
      rtf: 0.27,
      dropped_chunks: 0,
    }),
    record(5, "utterance_done", { chunk_id: "chunk-0", pending_utterances: 0 }),
    record(6, "terminal", { status: "completed" }),
  ];
  const acceptance = validateCase(testCase, result(events, {
    catalogAfterReady: { resolved: "supertonic3_ko_s5" },
  }));
  assert.equal(acceptance.allPassed, true);
  assert.equal(acceptance.observed.readyEngine, "supertonic3_ko_s5");
  assert.deepEqual(acceptance.observed.generationLanguages, ["ko"]);

  const wrongLanguage = events.map(item => structuredClone(item));
  wrongLanguage.find(item => item.event.type === "engine_metrics")
    .event.details.generation_language = "ru";
  const wrongLanguageAcceptance = validateCase(testCase, result(wrongLanguage, {
    catalogAfterReady: { resolved: "supertonic3_ko_s5" },
  }));
  assert.equal(wrongLanguageAcceptance.allPassed, false);
  assert.equal(wrongLanguageAcceptance.assertions.expectedGenerationLanguage, false);

  const slowAndDropped = events.map(item => structuredClone(item));
  const slowMetrics = slowAndDropped.find(item => item.event.type === "engine_metrics");
  slowMetrics.event.details.first_chunk_ms = 1201;
  slowMetrics.event.details.rtf = 0.81;
  slowMetrics.event.details.dropped_chunks = 1;
  const slowAcceptance = validateCase(testCase, result(slowAndDropped, {
    catalogAfterReady: { resolved: "supertonic3_ko_s5" },
  }));
  assert.equal(slowAcceptance.allPassed, false);
  assert.equal(
    slowAcceptance.assertions.engineMetricsFirstChunkWithinLimit,
    false,
  );
  assert.equal(slowAcceptance.assertions.engineMetricsRtfWithinLimit, false);
  assert.equal(
    slowAcceptance.assertions.engineMetricsDroppedChunksWithinLimit,
    false,
  );
}

{
  const testCase = byName("three_chunk_fifo_and_drain");
  const events = [
    record(0, "ready", { tts_policy: "auto", tts_engine_id: "matcha_zh_baker" }),
    record(1, "chunk_queued", { chunk_id: "chunk-0" }),
    record(2, "chunk_queued", { chunk_id: "chunk-1" }),
    record(3, "chunk_queued", { chunk_id: "chunk-2" }),
    record(4, "queue_closed", { pending_utterances: 3 }),
    record(5, "utterance_start", { chunk_id: "chunk-0" }),
    record(6, "utterance_start", { chunk_id: "chunk-1" }),
    record(7, "utterance_start", { chunk_id: "chunk-2" }),
    record(8, "engine_metrics", { engine_id: "matcha_zh_baker" }),
    record(9, "engine_metrics", { engine_id: "matcha_zh_baker" }),
    record(10, "engine_metrics", { engine_id: "matcha_zh_baker" }),
    record(11, "utterance_done", { chunk_id: "chunk-0", pending_utterances: 2 }),
    record(12, "utterance_done", { chunk_id: "chunk-1", pending_utterances: 1 }),
    record(13, "utterance_done", { chunk_id: "chunk-2", pending_utterances: 0 }),
    record(14, "terminal", { status: "completed" }),
  ];
  const acceptance = validateCase(testCase, result(events));
  assert.equal(acceptance.allPassed, true);
  assert.equal(acceptance.assertions.donePendingCountsAreCumulative, true);

  const reordered = events.map(item => structuredClone(item));
  reordered[11].event.details.chunk_id = "chunk-1";
  reordered[12].event.details.chunk_id = "chunk-0";
  const failed = validateCase(testCase, result(reordered));
  assert.equal(failed.allPassed, false);
  assert.equal(failed.assertions.doneInOrder, false);

  const impossibleLifecycle = [
    ...events.slice(0, 5),
    record(5, "utterance_done", { chunk_id: "chunk-0", pending_utterances: 2 }),
    record(6, "utterance_done", { chunk_id: "chunk-1", pending_utterances: 1 }),
    record(7, "utterance_done", { chunk_id: "chunk-2", pending_utterances: 0 }),
    record(8, "utterance_start", { chunk_id: "chunk-0" }),
    record(9, "utterance_start", { chunk_id: "chunk-1" }),
    record(10, "utterance_start", { chunk_id: "chunk-2" }),
    record(11, "terminal", { status: "completed" }),
  ];
  const impossible = validateCase(testCase, result(impossibleLifecycle));
  assert.equal(impossible.allPassed, false);
  assert.equal(impossible.assertions.perChunkLifecycleOrdered, false);
}

{
  const testCase = byName("immediate_cancellation");
  const events = [
    record(0, "ready", { tts_policy: "auto", tts_engine_id: "matcha_zh_baker" }),
    record(1, "chunk_queued", { chunk_id: "chunk-0" }),
    record(2, "queue_cancelled", {
      reason: "cdp_immediate_cancel",
      removed_utterances: 1,
    }),
    record(3, "terminal", { status: "cancelled", chunks_queued: 1 }),
  ];
  const acceptance = validateCase(testCase, result(events, {
    driver: { cancelIssuedSequence: 2 },
  }));
  assert.equal(acceptance.allPassed, true);

  const withLateDone = [...events, record(4, "utterance_done", { chunk_id: "chunk-0" })];
  const failed = validateCase(testCase, result(withLateDone, {
    driver: { cancelIssuedSequence: 2 },
  }));
  assert.equal(failed.allPassed, false);
  assert.equal(failed.assertions.noLateUtteranceDone, false);

  const withoutQueuedWork = events.filter(item => item.event.type !== "chunk_queued");
  const falsePositiveGuard = validateCase(testCase, result(withoutQueuedWork, {
    driver: { cancelIssuedSequence: 2 },
  }));
  assert.equal(falsePositiveGuard.allPassed, false);
  assert.equal(falsePositiveGuard.assertions.cancelledChunkWasAcknowledged, false);

  const initiallyPassing = result(events, {
    name: testCase.name,
    captureStartSequence: 0,
    captureEndSequence: 4,
    driver: { cancelIssuedSequence: 2 },
  });
  initiallyPassing.acceptance = validateCase(testCase, initiallyPassing);
  assert.equal(initiallyPassing.acceptance.allPassed, true);
  const lateAcrossBoundary = record(10, "utterance_done", { chunk_id: "chunk-0" });
  const payload = {
    cases: [initiallyPassing],
    eventCapture: {
      events: [...events, lateAcrossBoundary],
      original_handler_errors: [],
      active_turn: null,
    },
  };
  applyFinalEventAudit(payload);
  assert.equal(payload.cases[0].acceptance.allPassed, false);
  assert.equal(payload.cases[0].acceptance.assertions.noLateUtteranceDone, false);
  assert.equal(payload.eventCapture.audit.allPassed, true);

  const malformedAudit = auditEventCapture({
    events: [{ sequence: 0, event: null, parse_error: "bad JSON" }],
    original_handler_errors: [],
    active_turn: null,
  });
  assert.equal(malformedAudit.allPassed, false);
  assert.equal(malformedAudit.assertions.allEventsParsed, false);
}

{
  const expression = buildCaseExpression(byName("playback_start_cancellation"), {
    eventTimeoutMs: 120_000,
    lateEventGraceMs: 2_500,
  });
  for (const marker of [
    "turnCoordinator.begin",
    "nativeAudio.initializeTts",
    "nativeAudio.queueTtsChunk",
    "nativeAudio.finishTtsResponse",
    "nativeAudio.cancelTts",
    "utterance_start",
  ]) {
    assert.ok(expression.includes(marker), `missing physical driver marker: ${marker}`);
  }
  assert.ok(!expression.includes("Date.now() * 1000"), "must not poison native generations");
  for (const [name, source] of [
    ["preflight", PREFLIGHT_EXPRESSION],
    ["event hook", INSTALL_HOOK_EXPRESSION],
    ["case driver", expression],
    ["event export", buildEventExportExpression(10)],
    ["cleanup", CLEANUP_EXPRESSION],
  ]) {
    assert.doesNotThrow(
      () => new Function(`return (${source});`),
      `${name} browser expression must parse`,
    );
  }
}

{
  const fakeWindow = { onNativeTtsEvent: null };
  let selected = "auto";
  let resolved = null;
  let generation = 100;
  const queued = [];
  const fakeCoordinator = {
    current: null,
    begin(mode, inputKind) {
      this.current = {
        id: "mock-turn",
        generation: ++generation,
        mode,
        inputKind,
        terminal: false,
      };
      return this.current;
    },
    markTerminal(turn) {
      turn.terminal = true;
      return true;
    },
    isCurrent(turn) {
      return this.current === turn && !turn.terminal;
    },
    clearIfCurrent(turn) {
      if (this.current === turn) this.current = null;
    },
  };
  const emit = (type, turnId, details) => {
    fakeWindow.onNativeTtsEvent(JSON.stringify({
      schema_version: 1,
      type,
      turn_id: turnId,
      monotonic_ms: performance.now(),
      details,
    }));
  };
  fakeWindow.NativeAudio = {
    listTtsEngines() {
      return JSON.stringify({ selected, resolved });
    },
    setTtsEngine(policy) {
      selected = policy;
      resolved = null;
      return true;
    },
    initializeTts(locale, turnId, turnGeneration) {
      resolved = "matcha_zh_baker";
      setTimeout(() => emit("ready", turnId, {
        generation: turnGeneration,
        tts_policy: selected,
        tts_engine_id: resolved,
        requested_locale: locale,
        resolved_locale: "zh",
      }), 0);
    },
    queueTtsChunk(text, turnId, chunkId, turnGeneration) {
      queued.push({ text, turnId, chunkId, turnGeneration });
    },
    finishTtsResponse(turnId, turnGeneration) {
      setTimeout(() => {
        queued.forEach(item => emit("chunk_queued", turnId, {
          generation: turnGeneration,
          chunk_id: item.chunkId,
        }));
        emit("queue_closed", turnId, {
          generation: turnGeneration,
          pending_utterances: queued.length,
        });
        queued.forEach(item => emit("utterance_start", turnId, {
          generation: turnGeneration,
          chunk_id: item.chunkId,
        }));
        queued.forEach(() => emit("engine_metrics", turnId, {
          generation: turnGeneration,
          engine_id: resolved,
        }));
        queued.forEach((item, index) => emit("utterance_done", turnId, {
          generation: turnGeneration,
          chunk_id: item.chunkId,
          pending_utterances: queued.length - index - 1,
        }));
        emit("terminal", turnId, { generation: turnGeneration, status: "completed" });
      }, 0);
    },
    cancelTts() {},
  };
  const installHook = new Function("window", `return (${INSTALL_HOOK_EXPRESSION});`);
  installHook(fakeWindow);
  fakeWindow.__lfmPhysicalTtsAcceptance.restorePolicy = "auto";
  const testCase = byName("auto_zh_matcha");
  const expression = buildCaseExpression(testCase, {
    eventTimeoutMs: 1_000,
    lateEventGraceMs: 0,
  });
  const driveCase = new Function(
    "window",
    "turnCoordinator",
    `return (${expression});`,
  );
  const driven = await driveCase(fakeWindow, fakeCoordinator);
  const drivenAcceptance = validateCase(testCase, driven);
  assert.equal(
    drivenAcceptance.allPassed,
    true,
    JSON.stringify(drivenAcceptance.assertions, null, 2),
  );
  assert.equal(fakeCoordinator.current, null);

  const cleanup = new Function("window", `return (${CLEANUP_EXPRESSION});`);
  const cleanupResult = await cleanup(fakeWindow);
  assert.equal(cleanupResult.handlerRestored, true);
  assert.equal(cleanupResult.policyRestoreAccepted, true);
}

process.stdout.write("phone TTS CDP acceptance harness self-tests passed\n");
