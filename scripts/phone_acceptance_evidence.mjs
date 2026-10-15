export function responseThinkingDisabled(responseFetches) {
  return Array.isArray(responseFetches) && responseFetches.length > 0
    && responseFetches.every(item => {
    const backend = String(item?.requestBackend || "").toLocaleLowerCase();
    const model = String(
      item?.requestModel || item?.requestModelIdentity || "",
    ).toLocaleLowerCase();
    const isQwen25 = /qwen2(?:[._-]?5)/u.test(`${backend}\n${model}`);
    // The Qwen3-ASR text decoder is served through the raw /v1/completions
    // prompt path (no chat template), so its serving path has no thinking
    // mode at all - the same no-thinking category as Qwen2.5.
    const isRawPromptNoThinking = /qwen3[-_]?asr[-_]?0[._-p]?6b/u.test(`${backend}\n${model}`);
    return (isQwen25 || isRawPromptNoThinking)
      ? item?.enableThinking == null || item.enableThinking === false
      : item?.enableThinking === false;
  });
}

export function requiredScriptMatches(evidence, requiredScript) {
  if (!requiredScript) return null;
  const facts = evidence || {};
  const minimumRatio = 0.50;
  if (requiredScript === "han") {
    return Number(facts.han || 0) >= 2 && Number(facts.hanRatio || 0) >= minimumRatio;
  }
  if (requiredScript === "hiragana") {
    return Number(facts.hiragana || 0) >= 1
      && Number(facts.japaneseRatio || 0) >= minimumRatio;
  }
  if (requiredScript === "katakana") {
    return Number(facts.katakana || 0) >= 1
      && Number(facts.japaneseRatio || 0) >= minimumRatio;
  }
  if (requiredScript === "kana" || requiredScript === "japanese") {
    // Kana is the unambiguous discriminator between Japanese and a Han-only
    // Mandarin transcript. Exact-text assertions remain available for valid
    // short, Han-only utterances such as place names.
    return Number(facts.kana || 0) >= 1
      && Number(facts.japaneseRatio || 0) >= minimumRatio;
  }
  if (requiredScript === "hangul" || requiredScript === "korean") {
    return Number(facts.hangul || 0) >= 2
      && Number(facts.hangulRatio || 0) >= minimumRatio;
  }
  if (requiredScript === "cyrillic") {
    return Number(facts.cyrillic || 0) >= 3
      && Number(facts.cyrillicRatio || 0) >= minimumRatio;
  }
  if (requiredScript === "latin") {
    return Number(facts.latin || 0) >= 3
      && Number(facts.latinRatio || 0) >= minimumRatio;
  }
  return false;
}

function expectationPresent(value) {
  if (Array.isArray(value)) return value.length > 0;
  return typeof value === "string" ? value.trim().length > 0 : value != null;
}

export function validateAcceptancePlan(plan) {
  const errors = [];
  if (!plan || typeof plan !== "object") {
    return ["plan must be a JSON object"];
  }
  if (!Array.isArray(plan.cases) || plan.cases.length === 0) {
    return ["plan.cases must contain at least one case"];
  }
  const seenNames = new Set();
  const strictRouteEvidence = Number(plan.schema_version) >= 3;
  plan.cases.forEach((rawCase, index) => {
    const prefix = `cases[${index}]`;
    if (!rawCase || typeof rawCase !== "object") {
      errors.push(`${prefix} must be an object`);
      return;
    }
    const testCase = {
      ...rawCase,
      expectedResponseBackend: rawCase.expectedResponseBackend
        || plan.expectedResponseBackend || null,
      expectedResponseModel: rawCase.expectedResponseModel
        || plan.expectedResponseModel || null,
      expectedFinishReason: rawCase.expectedFinishReason
        || plan.expectedFinishReason || null,
      expectedNoReasoning: typeof rawCase.expectedNoReasoning === "boolean"
        ? rawCase.expectedNoReasoning
        : (typeof plan.expectedNoReasoning === "boolean" ? plan.expectedNoReasoning : null),
    };
    const name = String(testCase.name || "").trim();
    if (!name) errors.push(`${prefix}.name is required`);
    else if (seenNames.has(name)) errors.push(`${prefix}.name duplicates ${name}`);
    else seenNames.add(name);
    if (!expectationPresent(testCase.audio)) errors.push(`${prefix}.audio is required`);
    if (strictRouteEvidence
        && (typeof testCase.expectedAudioSha256 !== "string"
          || !/^[0-9a-f]{64}$/u.test(testCase.expectedAudioSha256))) {
      errors.push(
        `${prefix}.expectedAudioSha256 must be lowercase 64-hex for schema 3`,
      );
    }
    if (!expectationPresent(testCase.language)) errors.push(`${prefix}.language is required`);
    if (!["chat", "tts", "asr"].includes(testCase.mode)) {
      errors.push(`${prefix}.mode must be chat, tts, or asr`);
    }
    if (!["complete", "blocked"].includes(testCase.expectedOutcome)) {
      errors.push(`${prefix}.expectedOutcome must be complete or blocked`);
      return;
    }
    if (testCase.expectedOutcome === "blocked") {
      if (!expectationPresent(testCase.expectedErrorIncludes)) {
        errors.push(`${prefix}.expectedErrorIncludes is required for a blocked case`);
      }
      return;
    }

    for (const field of [
      "expectedResolvedLanguage",
      "expectedAsrBackend",
      "expectedAsrProvider",
    ]) {
      if (!expectationPresent(testCase[field])) errors.push(`${prefix}.${field} is required`);
    }
    if (typeof testCase.expectedNpuExecuted !== "boolean") {
      errors.push(`${prefix}.expectedNpuExecuted must be boolean`);
    }
    if (strictRouteEvidence && typeof testCase.expectedNpuSelected !== "boolean") {
      errors.push(`${prefix}.expectedNpuSelected must be boolean for schema 3`);
    }
    if (strictRouteEvidence && testCase.expectedAsrBackend === "sensevoice_qnn"
        && !expectationPresent(testCase.expectedAsrLanguageHint)) {
      errors.push(`${prefix}.expectedAsrLanguageHint is required for SenseVoice`);
    }
    const transcriptOracle = expectationPresent(testCase.expectedTranscriptIncludes)
      || expectationPresent(testCase.expectedTranscriptScript);
    const transcriptOracleRequired = testCase.mode === "asr"
      || (strictRouteEvidence && testCase.mode === "chat");
    if (transcriptOracleRequired && !transcriptOracle) {
      errors.push(`${prefix} requires a transcript content or script expectation`);
    }
    if (!["chat", "tts"].includes(testCase.mode)) return;

    for (const field of [
      "expectedResponseMode",
      "expectedVoiceBackend",
      "expectedResponseBackend",
      "expectedResponseModel",
      "expectedFinishReason",
      "expectedResponseScript",
    ]) {
      if (!expectationPresent(testCase[field])) errors.push(`${prefix}.${field} is required`);
    }
    if (typeof testCase.expectedNoReasoning !== "boolean") {
      errors.push(`${prefix}.expectedNoReasoning must be boolean`);
    }
    if (strictRouteEvidence && testCase.expectedVoiceBackend === "android_native_tts") {
      for (const field of [
        "ttsPolicy",
        "expectedTtsPolicy",
        "expectedTtsEngine",
        "expectedTtsRequestedLocale",
        "expectedTtsResolvedLocale",
        "expectedTtsTerminalStatus",
      ]) {
        if (!expectationPresent(testCase[field])) errors.push(`${prefix}.${field} is required`);
      }
      if (expectationPresent(testCase.ttsPolicy)
          && expectationPresent(testCase.expectedTtsPolicy)
          && testCase.ttsPolicy !== testCase.expectedTtsPolicy) {
        errors.push(`${prefix}.ttsPolicy must match expectedTtsPolicy`);
      }
      if (String(testCase.expectedTtsEngine || "").startsWith("supertonic")
          && !expectationPresent(testCase.expectedTtsGenerationLanguage)) {
        errors.push(
          `${prefix}.expectedTtsGenerationLanguage is required for Supertonic`,
        );
      }
      if (String(testCase.expectedTtsEngine || "").startsWith("supertonic")) {
        if (!Number.isFinite(testCase.maxTtsFirstChunkMs)
            || Number(testCase.maxTtsFirstChunkMs) <= 0) {
          errors.push(`${prefix}.maxTtsFirstChunkMs must be positive for Supertonic`);
        }
        if (!Number.isFinite(testCase.maxTtsRtf)
            || Number(testCase.maxTtsRtf) <= 0) {
          errors.push(`${prefix}.maxTtsRtf must be positive for Supertonic`);
        }
        if (!Number.isFinite(testCase.maxTtsDroppedChunks)
            || Number(testCase.maxTtsDroppedChunks) < 0) {
          errors.push(`${prefix}.maxTtsDroppedChunks must be non-negative for Supertonic`);
        }
      }
      if (!Number.isFinite(testCase.minNativeTtsPlaybackMs)
          || Number(testCase.minNativeTtsPlaybackMs) <= 0) {
        errors.push(`${prefix}.minNativeTtsPlaybackMs must be positive`);
      }
    }
    if (!Number.isFinite(testCase.expectedMinResponseCharacters)
        || Number(testCase.expectedMinResponseCharacters) <= 0) {
      errors.push(`${prefix}.expectedMinResponseCharacters must be positive`);
    }
    if (!Number.isFinite(testCase.expectedMaxDuplicateGramRatio)
        || Number(testCase.expectedMaxDuplicateGramRatio) < 0
        || Number(testCase.expectedMaxDuplicateGramRatio) > 1) {
      errors.push(`${prefix}.expectedMaxDuplicateGramRatio must be between 0 and 1`);
    }
    const responseOracle = expectationPresent(testCase.expectedResponseIncludes)
      || expectationPresent(testCase.expectedResponseRequestIncludes);
    if (!responseOracle) {
      errors.push(`${prefix} requires a response or response-request content expectation`);
    }
  });
  return errors;
}

export function collectNativeTtsEvidence(records, turnId, captureSupported = true) {
  const available = captureSupported && Array.isArray(records);
  const captured = available ? records : [];
  const scoped = captured.filter(record => record?.event?.turn_id === turnId);
  const readyEvents = scoped.filter(record => record?.event?.type === "ready");
  const starts = scoped.filter(record => record?.event?.type === "utterance_start");
  const dones = scoped.filter(record => record?.event?.type === "utterance_done");
  const terminals = scoped.filter(record => record?.event?.type === "terminal");
  const engineMetrics = scoped.filter(record => record?.event?.type === "engine_metrics");
  const metricDetails = engineMetrics.map(record => record?.event?.details || {});
  const firstChunkValues = metricDetails.map(details => Number(details.first_chunk_ms));
  const rtfValues = metricDetails.map(details => Number(details.rtf));
  const droppedChunkValues = metricDetails.map(details => Number(details.dropped_chunks));
  const metricGenerations = metricDetails.map(details => Number(details.generation));
  const metricEngineIds = metricDetails.map(details => details.engine_id ?? null);
  const ready = readyEvents[0] || null;
  const firstStart = starts.find(record => Number.isFinite(record?.event?.monotonic_ms));
  const terminal = [...terminals]
    .reverse()
    .find(record => Number.isFinite(record?.event?.monotonic_ms)) || terminals.at(-1) || null;
  const startMonotonicMs = firstStart?.event?.monotonic_ms ?? null;
  const terminalMonotonicMs = terminal?.event?.monotonic_ms ?? null;
  const playbackSpanMs = Number.isFinite(startMonotonicMs)
      && Number.isFinite(terminalMonotonicMs)
      && terminalMonotonicMs >= startMonotonicMs
    ? terminalMonotonicMs - startMonotonicMs
    : null;
  return {
    supported: available,
    unsupportedReason: available ? null : "native_tts_event_capture_unavailable",
    recordsCaptured: captured.length,
    eventsCaptured: scoped.length,
    parseClean: captured.every(record => !record?.parse_error && record?.event),
    originalHandlerHealthy: captured.every(record => !record?.original_handler_error),
    readyCount: readyEvents.length,
    readyPolicy: ready?.event?.details?.tts_policy ?? null,
    readyEngine: ready?.event?.details?.tts_engine_id
      ?? ready?.event?.details?.resolved_engine_id
      ?? ready?.event?.details?.engine_id
      ?? ready?.event?.details?.engine
      ?? null,
    readyRequestedLocale: ready?.event?.details?.requested_locale ?? null,
    readyResolvedLocale: ready?.event?.details?.resolved_locale ?? null,
    engineMetricsCount: engineMetrics.length,
    generationLanguages: engineMetrics.map(record =>
      record?.event?.details?.generation_language ?? null),
    engineMetricGenerations: metricGenerations,
    engineMetricEngineIds: metricEngineIds,
    firstChunkMsValues: firstChunkValues,
    rtfValues,
    droppedChunkValues,
    startCount: starts.length,
    doneCount: dones.length,
    terminalCount: terminals.length,
    firstStartMonotonicMs: startMonotonicMs,
    terminalMonotonicMs,
    playbackSpanMs,
    terminalStatus: terminal?.event?.details?.status ?? null,
    terminalMessage: terminal?.event?.details?.message ?? null,
  };
}

export function nativeTtsAcceptanceAssertions(
  evidence,
  expectedTerminalStatus,
  minPlaybackMs,
  expectedRoute = {},
) {
  const expectedPolicy = expectedRoute?.policy ?? null;
  const expectedEngine = expectedRoute?.engine ?? null;
  const expectedRequestedLocale = expectedRoute?.requestedLocale ?? null;
  const expectedResolvedLocale = expectedRoute?.resolvedLocale ?? null;
  const expectedGenerationLanguage = expectedRoute?.generationLanguage ?? null;
  const expectedGeneration = expectedRoute?.generation ?? null;
  const maxFirstChunkMs = expectedRoute?.maxFirstChunkMs ?? null;
  const maxRtf = expectedRoute?.maxRtf ?? null;
  const maxDroppedChunks = expectedRoute?.maxDroppedChunks ?? null;
  const metricsRequired = (expectedEngine !== null && expectedEngine !== "system")
    || expectedGenerationLanguage !== null || expectedGeneration !== null
    || maxFirstChunkMs !== null || maxRtf !== null || maxDroppedChunks !== null;
  const evidenceRequired = expectedTerminalStatus !== null || minPlaybackMs !== null
    || expectedPolicy !== null || expectedEngine !== null
    || expectedRequestedLocale !== null || expectedResolvedLocale !== null
    || expectedGenerationLanguage !== null || expectedGeneration !== null
    || maxFirstChunkMs !== null
    || maxRtf !== null || maxDroppedChunks !== null;
  return {
    nativeTtsEvidenceSupported: evidenceRequired ? evidence.supported === true : null,
    nativeTtsEventsParseClean: evidenceRequired ? evidence.parseClean === true : null,
    nativeTtsOriginalHandlerHealthy: evidenceRequired
      ? evidence.originalHandlerHealthy === true
      : null,
    ttsPolicy: expectedPolicy === null
      ? null
      : evidence.supported === true
        && evidence.readyCount === 1
        && evidence.readyPolicy === expectedPolicy,
    ttsEngine: expectedEngine === null
      ? null
      : evidence.supported === true
        && evidence.readyCount === 1
        && evidence.readyEngine === expectedEngine,
    ttsRequestedLocale: expectedRequestedLocale === null
      ? null
      : evidence.supported === true
        && evidence.readyCount === 1
        && evidence.readyRequestedLocale === expectedRequestedLocale,
    ttsResolvedLocale: expectedResolvedLocale === null
      ? null
      : evidence.supported === true
        && evidence.readyCount === 1
        && evidence.readyResolvedLocale === expectedResolvedLocale,
    ttsMetricsComplete: metricsRequired
      ? evidence.supported === true
        && evidence.engineMetricsCount > 0
        && evidence.startCount === evidence.engineMetricsCount
        && evidence.doneCount === evidence.engineMetricsCount
      : null,
    ttsMetricsEngine: metricsRequired && expectedEngine !== null
      ? evidence.supported === true
        && evidence.engineMetricEngineIds.length === evidence.engineMetricsCount
        && evidence.engineMetricEngineIds.every(engineId => engineId === expectedEngine)
      : null,
    ttsGenerationLanguage: expectedGenerationLanguage === null
      ? null
      : evidence.supported === true
        && evidence.engineMetricsCount > 0
        && evidence.generationLanguages.every(language =>
          language === expectedGenerationLanguage),
    ttsMetricsGeneration: expectedGeneration === null
      ? null
      : evidence.supported === true
        && evidence.engineMetricsCount > 0
        && evidence.engineMetricGenerations.length === evidence.engineMetricsCount
        && evidence.engineMetricGenerations.every(generation =>
          Number.isFinite(generation) && generation === Number(expectedGeneration)),
    maxTtsFirstChunkMs: maxFirstChunkMs === null
      ? null
      : evidence.supported === true
        && evidence.engineMetricsCount > 0
        && evidence.firstChunkMsValues.length === evidence.engineMetricsCount
        && evidence.firstChunkMsValues.every(value =>
          Number.isFinite(value) && value >= 0 && value <= maxFirstChunkMs),
    maxTtsRtf: maxRtf === null
      ? null
      : evidence.supported === true
        && evidence.engineMetricsCount > 0
        && evidence.rtfValues.length === evidence.engineMetricsCount
        && evidence.rtfValues.every(value =>
          Number.isFinite(value) && value >= 0 && value <= maxRtf),
    maxTtsDroppedChunks: maxDroppedChunks === null
      ? null
      : evidence.supported === true
        && evidence.engineMetricsCount > 0
        && evidence.droppedChunkValues.length === evidence.engineMetricsCount
        && evidence.droppedChunkValues.every(value =>
          Number.isFinite(value) && value >= 0 && value <= maxDroppedChunks),
    ttsTerminalStatus: expectedTerminalStatus === null
      ? null
      : evidence.supported === true
        && evidence.terminalCount === 1
        && evidence.terminalStatus === expectedTerminalStatus,
    minNativeTtsPlaybackMs: minPlaybackMs === null
      ? null
      : evidence.supported === true
        && evidence.startCount >= 1
        && evidence.terminalCount === 1
        && Number.isFinite(evidence.playbackSpanMs)
        && evidence.playbackSpanMs >= minPlaybackMs,
  };
}
