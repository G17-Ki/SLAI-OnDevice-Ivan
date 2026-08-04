# Native VAD capture bridge

Date: 2026-07-23
Status: Android source integrated and host-compiled; deterministic tests pass; physical-phone tuning remains

## Result

`NativeAudioBridge` now provides a separate half-duplex, hands-free capture path around
`AdaptiveEnergyVad`. It does not change the existing manual recorder contract. The VAD remains a
small CPU component; no NPU dispatch is appropriate for this energy gate.

The JavaScript interface is:

```text
NativeAudio.startVadConversation() -> boolean
NativeAudio.resumeVadConversation() -> boolean
NativeAudio.stopVadConversation() -> void
NativeAudio.setVadAssistantOutputActive(boolean) -> void
```

The first call creates one native session and starts listening. An endpoint immediately stops and
releases `AudioRecord`, marks the session paused, publishes one immutable WAV, and waits. JavaScript
must call `resumeVadConversation()` only after the request/response output is finished. `stop` ends
the session from either listening or paused state. Manual and VAD capture reject each other rather
than sharing or silently stealing the microphone.

## Capture and endpoint invariants

- Format: 16 kHz, mono, signed PCM16.
- Detector frame: exactly 320 samples / 20 ms. Short `AudioRecord` reads are accumulated; partial
  frames are never sent to the detector.
- Initial VAD settings: 240 ms pre-roll, 420 ms trailing-silence endpoint, 15 s maximum utterance.
- Retention ring: 250,880 samples / 15.68 s / 501,760 PCM bytes. This is bounded to maximum
  utterance + pre-roll + endpoint hangover + one complete-frame margin.
- The ring uses absolute detector sample offsets and rejects discontinuous appends.
- An endpoint WAV is sliced exactly over
  `[recommendedAudioStartSample, decisionSample)`. The first phone build therefore retains the
  decision hangover instead of guessing a trim point.
- The endpoint callback is queued only after the old `AudioRecord` has stopped, been released, and
  been removed as the active capture thread. Immediate JavaScript handling cannot race a subsequent
  resume against the old microphone owner.
- Capture-thread shutdown never joins the capture thread itself. User stop and lifecycle cleanup
  may join it for at most 1.5 s from a control thread.

## Callback contract

All state is delivered through:

```javascript
window.onNativeVadEvent(jsonString)
```

The argument is escaped with `JSONObject.quote`; JavaScript should parse it once with
`JSON.parse`. Its top-level schema is:

```json
{
  "schema_version": 1,
  "type": "endpoint",
  "session_id": "vad-...",
  "monotonic_ms": 1234.5,
  "details": {}
}
```

Event types are `session_started`, `listening`, `speech_confirmed`, `hangover`, `endpoint`,
`burst_rejected`, `resumed`, `stopped`, `output_gate`, and `error`.

An `endpoint` includes:

```text
endpoint_reason
decision_sample
speech_start_sample
speech_end_sample
audio_start_sample
sample_rate = 16000
frame_samples = 320
pcm_bytes
wav_bytes
audio_duration_ms
paused = true
requires_resume = true
wav_base64
```

`wav_base64` uses Android `Base64.NO_WRAP`. Large audio is excluded from the native telemetry file;
only the callback contains it.

## Telemetry

Native JSONL telemetry uses `mode = vad_conversation`, a stable `session_id`, and these event names:

```text
native_vad_session_started
native_vad_listening
native_vad_speech_confirmed
native_vad_hangover
native_vad_endpoint
native_vad_burst_rejected
native_vad_resumed
native_vad_stopped
native_vad_output_gate
native_vad_error
```

Speech, rejection, hangover, and endpoint events include detector sequence/state, sample offsets,
RMS, learned noise floor, and threshold. Stop includes cumulative frames/samples, confirmed
utterances, endpoint reasons, rejected bursts/impulses, blocked-output frames, and final noise
floor.

## Output gate and barge-in

The detector is constructed with `BargeInPolicy.DISABLED`. Normal conversation is half-duplex:
capture is absent from endpoint through response completion. The optional output-active interface
also freezes/blocks detector input if the UI deliberately keeps capture open around playback. The
UI should set output inactive before resuming; native resume also clears the gate defensively. This
build makes no acoustic-echo-cancellation or speakerphone barge-in claim.

## Reproducible validation

Focused bridge test:

```sh
python3 scripts/test_native_vad_bridge.py
```

This checks the public interfaces, fixed frame/retention policy, exact endpoint slice, callback
escaping/Base64, pause-before-publication, telemetry coverage, lifecycle cleanup, output gate,
manual/VAD conflicts, and no-self-join invariant. It also extracts the production PCM ring and
PCM16 encoder source, compiles them with `javac`, and deterministically tests absolute offsets,
wraparound, eviction, discontinuity rejection, and little-endian encoding.

Observed:

```text
native VAD bridge source tests passed
```

Core detector regression:

```sh
scripts/test_adaptive_energy_vad.sh
# AdaptiveEnergyVad: 8 tests passed
```

Android Java compile, constrained to one worker:

```sh
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
cd android-app
nice -n 10 ./gradlew --no-daemon --max-workers=1 :app:compileDebugJavaWithJavac
# BUILD SUCCESSFUL
```

## Remaining phone gate

This is not yet evidence of acoustic robustness. On the Nubia, exercise at least ten consecutive
endpoint/resume turns plus quiet-room, fan/traffic, Mandarin, accented English, short acknowledgments,
false-trigger silence, maximum-duration, explicit stop, lifecycle destruction, and output-gate cases.
Record clipped starts/ends, false triggers, misses, last speech to endpoint median/P95, endpoint to
request start, CPU time, RSS, temperature, and battery. Tune the energy thresholds only from saved
phone recordings; replace the detector with a neural VAD behind the same bridge if this gate fails.
