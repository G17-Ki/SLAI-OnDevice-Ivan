# Android native-TTS source validation — 2026-07-23

> This records the native-TTS feature checkpoint. Current source is version
> `0.3.4-input-handoff`, which retains these invariants and removes a redundant
> native-microphone Base64 round trip; see
> [`NATIVE_INPUT_HANDOFF.md`](NATIVE_INPUT_HANDOFF.md).

## Scope

This checkpoint adds an optional Android system-`TextToSpeech` control path.
It does not replace the integrated LFM audio path or transcription mode. The
source checks below preceded a later one-worker APK build. No cloud job, phone
run, APK installation, Git commit, or push was performed.

## Implemented

- Third UI mode requesting text-only streamed LFM output.
- Incremental clause extraction and `QUEUE_FLUSH`/`QUEUE_ADD` native TTS
  submission while the SSE stream remains open.
- Locale selector for auto, EN, ZH, ES, JA, KO, DE, IT, and FR.
- Android language-data checks, offline-voice preference, explicit missing or
  unsupported-language errors, and network-requirement reporting.
- Escaped JSON-string callbacks, lifecycle-safe engine shutdown, queue cancel,
  and preservation of the existing microphone and WebAudio paths.
- App-private monotonic events for native TTS init, chunk queued, queue flush,
  queue cancel, utterance start/done/error, first playback-start proxy, and
  distinct text-SSE completion, queue closure, and playback-terminal status.
- Telemetry summarization for native-TTS timing fields and errors.

## Passed lightweight checks

```text
node scripts/test_streaming_audio_ui.mjs
  streaming audio UI tests passed

python3 scripts/test_android_native_tts_bridge.py
  android native TTS bridge source tests passed

python3 scripts/test_android_streaming_telemetry.py
  android streaming telemetry tests passed

python3 -m py_compile \
  scripts/pull_android_streaming_telemetry.py \
  scripts/test_android_native_tts_bridge.py \
  scripts/test_android_streaming_telemetry.py
  passed

JDK 17 javac -source 8 -target 8 against Android API 35 android.jar \
  NativeAudioBridge.java
  passed (only the required deprecated onError(String) override and the
  pre-existing ignored resetTelemetry resource produced warnings)

xmllint --noout android-app/app/src/main/AndroidManifest.xml
  passed

git diff --check -- <Android native-TTS scoped files>
  passed
```

## Build artifact and remaining measurement gate

At this checkpoint, `android-app/app/build.gradle` used version code 6 and
version name `0.3.3-native-tts`. After the memory-gated experiment opened enough host
headroom, one low-priority Gradle worker completed the debug build. The
retained artifact is
`android-app/releases/lfm-audio-demo-0.3.3-native-tts-debug.apk`, 4,674,179
bytes, SHA-256
`f8931486f8f0b749be3ed1174fd219e98929d701808cc9c064050e257b587f9e`.
Package/version/API/ABI/signature and embedded-asset checks pass; see
[`validation_2026-07-23_native_tts_apk.md`](validation_2026-07-23_native_tts_apk.md).
The later `0.3.4` artifact is validated separately in
[`validation_2026-07-23_input_handoff_apk.md`](validation_2026-07-23_input_handoff_apk.md).

The build does not close the physical gate:

- `scripts/install_native_phone_demo.sh` correctly remains pinned to the
  existing `0.3.2-turn-telemetry` APK;
- Android engine discovery, installed-language behavior, actual speech,
  timing, memory, and cancellation behavior remain unverified on hardware;
- `UtteranceProgressListener.onStart` is a software callback proxy, not an
  acoustic-onset measurement.

The next operator should install the versioned candidate explicitly on the
Nubia and test every explicit locale, conservative Auto fallbacks,
unavailable-language failure, interruption, cancellation, lifecycle release,
timing, memory, heat, and acoustic onset before promoting this path.

## Release-blocking state-machine review closure

The follow-up review in `work/NATIVE_TTS_REVIEW.md` found stale-turn ownership,
fire-and-forget queue acceptance, and premature completion accounting. The
source now closes those blockers as follows:

- One monotonically increasing turn generation owns the microphone/file input,
  abort controller, streamed reader, UI writes, TTS initialization, chunk
  queue, and terminal callbacks. Starting a replacement input aborts and
  cancels the prior generation; stale `finally` blocks and callbacks cannot
  mutate the replacement UI.
- File input is disabled during recording/processing, while deliberate
  programmatic replacement still routes through the same abort-and-replace
  coordinator.
- Every native TTS command carries turn ID plus generation. Java cancellation
  calls `TextToSpeech.stop()` only for the exact current owner, ignores older
  commands, and invalidates a cancelled pending initialization so `onInit`
  cannot resurrect it.
- Cancelled JavaScript initialization waiters reject with `AbortError`.
  Every continuation rechecks current ownership before calling the bridge.
- Each chunk has a five-second native-acceptance waiter keyed by chunk ID.
  Queue closure occurs only after all submitted chunks receive
  `chunk_queued`; terminal failure rejects all outstanding acknowledgements.
- Telemetry and summarization now distinguish `tts_text_sse_complete`,
  `native_tts_queue_closed`, and `native_tts_playback_terminal`. A TTS turn is
  counted complete only when the terminal status is `completed`; missing-done,
  failed, and cancelled traces remain incomplete.
- The emitted `request_start` event now defines the shared request clock for
  UI elapsed fields, native-TTS proxy timing, scheduled WebAudio start, and
  output-head timing. The summarizer independently reconstructs scheduled and
  output-head latency from event timestamps, so pre-payload work is no longer
  mislabeled as request latency.
- The earlier 240-character unpunctuated fallback conflicted with the published
  50–70-character first-chunk policy. It is now a whitespace fallback bounded
  to 50–70 characters. Explicit locale remains the measurement policy; Auto is
  documented as heuristic and is tested for ambiguous Latin and Han-only text.

Regression coverage now includes overlapping A/B turns, aborted readers,
cancel-during-init, stale-ready callbacks, exact-generation cancellation,
chunk acknowledgement, acknowledgement timeout, terminal failure cleanup,
queue-closed-without-terminal, missing utterance-done, failed terminal,
cancelled terminal, and normal playback completion.

## Files in this checkpoint

- `android-app/app/build.gradle`
- `android-app/app/src/main/AndroidManifest.xml`
- `android-app/app/src/main/assets/demo-ui/index.html`
- `android-app/app/src/main/assets/demo-ui/service-worker.js`
- `android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java`
- `docs/ANDROID_DEPLOYMENT.md`
- `reports/android_phone/streaming_response/CONTINUATION_LOG.md`
- `reports/android_phone/streaming_response/validation_2026-07-23_native_tts_apk.md`
- `reports/android_phone/streaming_response/validation_2026-07-23_native_tts_source.md`
- `scripts/pull_android_streaming_telemetry.py`
- `scripts/test_android_native_tts_bridge.py`
- `scripts/test_android_streaming_telemetry.py`
- `scripts/test_streaming_audio_ui.mjs`
