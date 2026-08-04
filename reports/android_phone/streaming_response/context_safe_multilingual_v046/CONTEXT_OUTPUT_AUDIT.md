# Context and long-output audit

Updated: 2026-07-24 (Asia/Shanghai)

## Implemented and verified

- Context is visibly switchable and defaults off on every app start.
- Turning context off clears retained text.
- When enabled, only same-language history can be considered and injection is
  limited to detected follow-up/recall utterances.
- Generic standalone `why` questions in English, Chinese, and Russian no
  longer trigger unrelated history.
- Explicit recall forms such as `remember`, `记得`, and `помнишь` do trigger
  history.
- Numeric-only answers such as `42` are no longer incorrectly marked as a
  wrong-script response.
- The service-worker cache was advanced to
  `v11-strict-followup-context` so an APK update cannot retain the older cue
  rules.
- Severe repeated-four-gram streaming collapse cancels generation and native
  TTS, and rejected/capped/repetitive/wrong-script answers are not retained in
  history.

Host tests passed after the changes:

- `node scripts/test_streaming_audio_ui.mjs`
- `python3 scripts/test_android_native_tts_bridge.py`
- `python3 scripts/test_android_asr_companion.py` (5/5)
- `bash scripts/android_text_companion/run_host_tests.sh` (11/11)
- `bash scripts/test_adaptive_energy_vad.sh` (8/8)
- `python3 scripts/test_phone_acceptance_harness.py` (6/6)

## The ten-second question

There is no ten-second text or spoken-output cap.

- Text generation ceiling: 512 tokens.
- Phone text-server context: 2048 tokens.
- Native-TTS terminal watchdog: 300 seconds.
- The fixed ten-second shape belongs only to one SenseVoice ASR request.
- Push-to-talk/file input accepts up to 30 seconds and is segmented into
  at-most-9.5-second ASR requests with overlap removal.
- Hands-free VAD has a separate 15-second maximum utterance guard in the Java
  bridge; this is an input-turn safety bound, not an answer-audio limit.

## TTS and translation status

- `zh-CN` and `ru-RU` are routed to Android native TTS.
- The bridge explicitly requests the pinned F-Droid eSpeak engine
  `com.reecedunn.espeak` and prefers offline voices.
- `scripts/install_espeak_phone.sh` reproducibly installs the pinned eSpeak
  1.52.0 APK with Mandarin and Russian voice data.
- Audible Mandarin/Russian intelligibility is not yet certified because the
  Nubia was not visible to `adb` during this final audit.
- No translator is integrated. Direct small-model Russian quality failed the
  model gate; a real offline translator is therefore needed for a robust
  Russian claim. Prompt-only English pivoting is not acceptable.

## Remaining quality risk

The severe streaming loop guard is effective, but the final moderate-
repetition, generation-length, and script checks currently run after text has
streamed and after some native-TTS chunks may already have been queued. They
protect conversation history, not every sound already emitted. Before a
public Russian/Chinese demo, add a bounded incremental script gate before the
first TTS chunk and cancel/replace unsafe final output. This was deliberately
left for a separate bounded implementation rather than starting an untested
translator/TTS rewrite in this audit.

Documentation also needs one cleanup pass: older sections of
`scripts/android_text_companion/README.md` and `docs/ANDROID_DEPLOYMENT.md`
still mention 320 output tokens, a 1024-token context, a 180-second TTS wait,
or claim that long-input segmentation is missing.
