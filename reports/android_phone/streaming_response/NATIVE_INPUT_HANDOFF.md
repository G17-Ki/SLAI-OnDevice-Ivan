# Native microphone input handoff

Date: 2026-07-23

## Outcome

The Android microphone callback now reuses the Base64 WAV string produced by
the Java bridge directly in the LFM request. The previous JavaScript path
decoded that string with `atob`, copied it into a `Uint8Array`, converted it
back to a binary string, and encoded it again with `btoa` before starting the
request.

The new path removes those redundant full-buffer transforms:

1. Java still captures PCM, creates the WAV, and emits padded, no-wrap Base64.
2. JavaScript derives the decoded byte count from the Base64 length and final
   padding in constant time.
3. The original Base64 string is inserted into `input_audio.data` unchanged.

Web-microphone and file inputs are unchanged: both still produce a
`Uint8Array` locally and use the existing `bytesToBase64` path.

## Correctness and ownership

- `wav_ready.details.wav_bytes` and `request_start.details.wav_bytes` contain
  the exact decoded WAV size. The formula is `(encoded_length / 4) * 3 -
  padding`, where padding is zero, one, or two bytes. This matches the Java
  bridge's `Base64.NO_WRAP` output.
- The callback still captures and clears the generation-owned
  `recordingTurn`, rejects stale turns before doing work, checks ownership
  again before telemetry, and uses the same abortable request path.
- No Java, native service, model, request schema, response streaming, or TTS
  behavior changed.

## Regression coverage

`scripts/test_streaming_audio_ui.mjs` now executes the native microphone
handoff and proves that:

- neither `atob` nor `btoa` is called while preparing the request;
- the request contains the exact Java-provided Base64 string;
- both WAV-size telemetry events report the exact decoded byte count; and
- decoded byte-length accounting is correct across empty, unpadded-block, and
  one- and two-padding-byte fixtures.

The source-level callback check also rejects future direct uses of `atob` or
`bytesToBase64` in `onNativeRecording`.

Validation completed locally:

```text
node scripts/test_streaming_audio_ui.mjs
streaming audio UI tests passed

python3 scripts/test_android_native_tts_bridge.py
android native TTS bridge source tests passed
```

## Expected effect and measurement boundary

This removes two Base64 transforms, two full byte-copy loops, one temporary
`Uint8Array`, and the associated temporary binary strings from the native
input handoff. It should reduce handoff CPU time and peak JavaScript memory,
especially near the 30-second recording limit. It does not remove the initial
Java Base64 encoding, JSON serialization, HTTP transfer, or any model work.

No on-phone latency number is claimed yet. The source change is packaged as
`android-app/releases/lfm-audio-demo-0.3.4-input-handoff-debug.apk`, but the
phone is disconnected and the APK has not been installed. Its build, signature,
checksum, and embedded-asset checks are recorded in
[`validation_2026-07-23_input_handoff_apk.md`](validation_2026-07-23_input_handoff_apk.md).
After installation, the existing `wav_ready` and `request_start` clocks can
quantify the optimized JavaScript handoff on the physical phone as part of the
normal fixed-file and microphone-turn run.
