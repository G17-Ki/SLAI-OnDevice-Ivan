# Android input-handoff APK validation — 2026-07-23

## Scope

This checkpoint packages the optional Android system-`TextToSpeech` mode plus
the native-microphone input-handoff optimization. The Java bridge already
produces a Base64 WAV; version `0.3.4-input-handoff` now passes that exact string
into the local LFM request instead of decoding and immediately re-encoding the
full buffer in JavaScript.

The build used one low-priority Gradle worker. No phone was connected, so the
APK was not installed or executed on hardware. The default installer remains
pinned to the physically exercised `0.3.2-turn-telemetry` release.

## Retained artifact

- APK: `android-app/releases/lfm-audio-demo-0.3.4-input-handoff-debug.apk`
- Size: **4,674,367 bytes**
- SHA-256:
  `a1cc2a4ec8bc1342bd3ad3b32001b908e8f50677d7ed7f3e96a0d26857d5b145`
- Checksum sidecar:
  `android-app/releases/lfm-audio-demo-0.3.4-input-handoff-debug.apk.sha256`
- Gradle output: `android-app/app/build/outputs/apk/debug/app-debug.apk`
  matched the retained release artifact byte for byte at validation time.

## Package and signature checks

Android build-tools 35 reported:

- application ID `ai.liquid.lfmdemo`;
- version code `7` and version name `0.3.4-input-handoff`;
- minimum API 28, target API 35, and compile API 35;
- native ABI `arm64-v8a` only;
- application label `LFM Audio`;
- one Android debug signer;
- APK Signature Scheme v2 verification passed.

This is a debug artifact, not a production-signed store release.

## Packaged-source consistency

The behavior-bearing WebView assets inside the APK match current source:

| Asset | SHA-256 in source and APK |
|---|---|
| `assets/demo-ui/index.html` | `48bd738c9cc37bb4a6fcd4b5cd09a186415a6dc693b9043d6ac4864e669337ee` |
| `assets/demo-ui/service-worker.js` | `88e78217dc8fcf26074fec4f7ecd77796637385f5541d82fa2b5983c0dca1b68` |

The release sidecar verifies against the retained APK. The focused source and
regression evidence is in
[`NATIVE_INPUT_HANDOFF.md`](NATIVE_INPUT_HANDOFF.md); the native-TTS state and
telemetry invariants remain documented in
[`validation_2026-07-23_native_tts_source.md`](validation_2026-07-23_native_tts_source.md).

## Lightweight validation

The following checks pass:

```text
node scripts/test_streaming_audio_ui.mjs
  streaming audio UI tests passed

python3 scripts/test_android_native_tts_bridge.py
  android native TTS bridge source tests passed

shasum -a 256 -c \
  android-app/releases/lfm-audio-demo-0.3.4-input-handoff-debug.apk.sha256
  lfm-audio-demo-0.3.4-input-handoff-debug.apk: OK
```

The dynamic UI regression observes zero `atob` and zero `btoa` calls during a
native-microphone request, verifies byte-for-byte Base64 payload reuse, and
checks the exact decoded WAV size in both `wav_ready` and `request_start`
telemetry.

## Hardware boundary and promotion gate

This build removes redundant endpoint-side work, but it provides **no new
phone latency number**. It also provides no installed-voice, language-quality,
memory, thermal, battery, output-head, or acoustic-onset evidence.

When the Nubia reconnects, install this candidate explicitly and run ten
fixed-file plus ten microphone turns. Compare `input_stop_requested`,
`native_wav_ready`, `wav_ready`, and `request_start` to measure the handoff;
then retain the existing first-PCM, output-head, gap, RSS, thermal, and external
acoustic-onset gates. Keep `0.3.2-turn-telemetry` as the default until that
physical acceptance passes.
