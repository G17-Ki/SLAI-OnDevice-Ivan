# Multilingual conversation UI and bounded history

Date: 2026-07-23

Status: source implementation and deterministic host tests pass. No APK was
built and no physical-phone claim is made in this checkpoint.

## Implemented behavior

The phone UI now has one global language selector shared by integrated voice,
transcription, and Native Android TTS modes. Its choices are exactly:

- Auto
- English
- Mandarin Chinese
- Russian
- French
- Italian
- Spanish

An explicit selection controls both the model system instruction and, whenever
Android TTS is used, the exact locale tag (`en-US`, `zh-CN`, `ru-RU`, `fr-FR`,
`it-IT`, or `es-ES`). The selection is captured at turn start, so a turn cannot
silently change policy while it is running.

Conversation turns now use this sequence:

1. Run a deterministic-temperature ASR request and retain its real transcript.
2. Resolve the selected/Auto language from that transcript.
3. Send the bounded prior text history plus the current audio to the response
   request under a language-specific system policy.
4. Add the transcript and completed assistant response to history only if the
   same turn is still current and succeeds.

ASR-only mode displays the transcript but deliberately does not add an
assistant-less item to conversational history. Raw WAV/Base64 data is used only
for the current inference request and is never stored in history.

## Auto routing policy

Auto is intentionally conservative rather than a general-purpose language-ID
claim:

- two or more Han characters provide high-confidence Mandarin evidence;
- three or more Cyrillic characters provide high-confidence Russian evidence;
- English, French, Italian, and Spanish use weighted distinctive words,
  repeated function-word evidence, orthographic evidence, and a minimum margin
  over the runner-up;
- one shared word, a very short acknowledgement, or tied lexical evidence is
  not reported as a confident detection;
- ambiguous input continues the most recent non-fallback language with at least
  0.6 confidence; without one, it uses an explicitly labelled low-confidence
  English fallback.

The UI shows the resolved language, numeric confidence, evidence/fallback
reason, and fallback status. The same facts are emitted in telemetry. This
policy is deterministic and testable, but it is not intended to replace a
trained language-ID model for broad code-switching or unsupported languages.

## Voice-backend safety policy

The retained integrated LFM PCM evidence is English-only. Therefore:

- resolved English in Voice response mode keeps integrated streamed LFM PCM;
- every resolved non-English Voice response is explicitly rerouted to Native
  Android TTS and emits `voice_backend_routed` with the evidence boundary as
  its reason;
- Native TTS mode always uses Native Android TTS;
- transcription mode has no voice backend.

This avoids silently presenting unvalidated multilingual integrated PCM as a
supported result. Installed voice availability and quality remain physical
phone gates.

## History budget

The shared in-memory history has explicit limits:

- six complete user/assistant turns;
- 6,000 normalized text characters;
- approximately 1,500 tokens using a documented four-characters-per-token
  estimate.

Oldest complete turns are removed first. A single oversized newest turn is
clipped to the effective character budget instead of being discarded. The UI
shows current turns and approximate tokens and provides a Clear history
control. History is session-local and resets when the WebView process/page is
reloaded; persistence is intentionally not claimed yet.

## Telemetry

Every successful turn emits selected/resolved language, confidence, evidence,
fallback status, history turns, history characters, approximate tokens, and
the current `cpu_q4` compute backend. Relevant events include:

- `request_start`
- `asr_request_start`
- `language_resolved`
- `voice_backend_routed` when non-English Voice response is redirected
- `response_complete` or `tts_text_sse_complete`
- `history_cleared`

The response-completion clock starts before the ASR pass, so reported
first-token and first-audio proxies include the added language/transcription
work rather than hiding it.

## Verification

`node scripts/test_streaming_audio_ui.mjs` passes. The deterministic coverage
includes:

- all explicit language mappings and exact selector options;
- high-evidence Chinese, Russian, English, French, Italian, and Spanish cases;
- ambiguous short-input continuation and cold English fallback;
- fixed system instructions and TTS locale mapping;
- turn-count, character, approximate-token, clipping, and oldest-first history
  limits;
- clear/reset behavior;
- text-only history payload construction and rejection of audio/object history;
- stale/cancelled turn protection;
- English integrated-PCM versus non-English Native-TTS routing;
- the existing native WAV zero-copy, streaming audio, playback timing, native
  TTS acknowledgement, timeout, and stale-generation regressions.

## Integration risks and next phone gate

The main cost is the new sequential ASR pass: response generation cannot begin
until transcription completes, and conversational modes process the current
audio twice. This is the simplest honest way to obtain a real user transcript,
resolve Auto before choosing a voice, and construct text-only history, but its
endpoint-to-first-audio cost has not yet been measured. It may need replacement
by server-side transcript metadata, a shared-encoder path, or a dedicated
streaming ASR component.

The next physical-phone run should measure at least ten English and ten Mandarin
turns, plus smaller Russian/French/Italian/Spanish samples. It must verify:

- ASR completion and total endpoint-to-first-audio latency;
- explicit versus Auto routing telemetry and visible reasons;
- installed Native TTS voices and exact locale resolution;
- multi-turn reference retention, history trimming, and Clear history;
- cancellation while ASR, response streaming, and Native TTS are active;
- CPU load, RSS, thermals, and stability caused by the sequential two-request
  conversational path.
