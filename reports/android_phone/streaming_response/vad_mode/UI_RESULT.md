# Hands-free VAD UI integration

Date: 2026-07-23 22:43 CST
Status: source-integrated and host-tested; physical-phone validation and APK packaging remain

## Result

The Android demo UI now keeps the three existing output modes (`Voice response`,
`Transcription`, and `Native TTS`) and adds an independent microphone input switch:

- `Push-to-talk` retains the existing manual start/stop behavior.
- `Hands-free VAD` turns the main microphone button into a session start/stop control. A native
  endpoint automatically reuses the existing ASR, language routing, bounded history, response,
  and telemetry path. The session remains active across turns.

The UI consumes only the documented native callback:

```javascript
window.onNativeVadEvent(jsonString)
```

It does not introduce a second recording callback or decode/re-encode the native WAV. The
`details.wav_base64` string is handed to the existing request path byte-for-byte.

## Half-duplex sequencing

The initial mode deliberately has barge-in disabled.

1. Native VAD publishes an endpoint only after microphone capture is paused and the old
   `AudioRecord` is released.
2. The UI marks the assistant-output gate active and submits the endpoint.
3. For integrated LFM PCM, the UI waits until all scheduled WebAudio sources report `ended`.
4. For Android TTS, it waits for the native `terminal` playback event, not merely queue close or
   text generation completion.
5. Transcription-only turns rearm after ASR completion. Request/playback errors rearm only after
   cancellation and UI cleanup settle.
6. The UI clears the output gate and calls `resumeVadConversation()` exactly once for the next
   turn.

Manual session stop aborts the active request through its `AbortController`, cancels a native TTS
turn when present, stops queued WebAudio, clears the output gate, stops native VAD capture, and
does not rearm from the cancelled turn's `finally` path. A new session remains disabled until the
native `stopped` acknowledgement arrives, preventing a delayed callback from the previous capture
from cancelling or being mistaken for the next session. A fatal native error is itself treated as
the terminal acknowledgement, because the native bridge may already have torn the session down
and therefore may not emit a second `stopped` event.

## Visible state and telemetry

The screen exposes `starting`, `active`, `listening`, `speech detected`, `hangover`, `endpoint`,
`assistant output`, `resuming`, `stopped`, and `error` states. The status line always reports that
the adaptive-energy VAD is a Java CPU component and that barge-in is off.

JavaScript telemetry covers:

```text
vad_session_start_requested
vad_session_started
vad_listening
vad_speech_confirmed
vad_hangover
vad_endpoint
vad_burst_rejected
vad_output_gate_requested
vad_output_gate
vad_assistant_output_settled
vad_turn_error_settled
vad_resume_requested
vad_resumed
vad_session_stop_requested
vad_stopped
vad_error
vad_stale_event_ignored
```

Every VAD event includes the input mode, session/phase, `android_audiorecord`,
`adaptive_energy_java_cpu`, the current `cpu_q4` inference fact, and `barge_in_enabled=false`.
Endpoint audio itself is excluded from telemetry.

NPU reporting is fail-closed. It identifies `sensevoice_qnn_sm8850` only as a candidate and always
sets `npu_executed=false` in this build:

- Auto: `broad_auto_requires_cpu_multilingual_asr`.
- Explicit English/Chinese: `phone_qnn_smoke_not_passed`.
- Russian/French/Italian/Spanish: `language_not_supported_by_sensevoice_qnn`.

No runtime NPU execution is claimed.

## Reproducible validation

Run:

```sh
node scripts/test_streaming_audio_ui.mjs
```

Observed result:

```text
streaming audio UI tests passed
```

The focused test now covers the native VAD schema parser; the input controls and barge-in-off
label; backend/fallback facts; streamed-audio drain notification; Android-TTS terminal waiting;
session/listening/speech/hangover state transitions; byte-for-byte endpoint handoff; automatic
ASR completion and resume; telemetry coverage/redaction; persistent session state; and manual
in-flight cancellation.

`git diff --check` also passed for the edited UI and test paths.

## Remaining phone gate

This is host-side interaction-state evidence, not an acoustic or Android lifecycle result. The new
APK should be exercised on the Nubia for at least ten consecutive turns in each output mode, plus
explicit stop during ASR/LLM/playback, silence false triggers, short-burst rejection, maximum
utterance, Chinese and English Auto routing, the four secondary languages, speaker echo, screen
rotation/backgrounding, and service errors. Confirm that resume occurs only after audible output
ends and measure last speech to endpoint, endpoint to request start, response acoustic onset,
false triggers, misses, CPU/RSS, thermals, and battery.
