# Adaptive energy VAD core

Date: 2026-07-23
Status: host-tested source component; not yet connected to Android microphone capture; no new APK

## Decision

The first hands-free mode can use the new pure-Java
[`AdaptiveEnergyVad`](../../../../android-app/app/src/main/java/ai/liquid/lfmdemo/AdaptiveEnergyVad.java)
as a deterministic endpoint controller. It is deliberately an adaptive energy
VAD, not a claim of neural-VAD quality. The core is suitable for bring-up and
latency instrumentation because it has no model/runtime dependency, uses a
sample-count clock, and gives the capture layer exact pre-roll and endpoint
offsets.

This component should stay on the CPU. At 16 kHz with 20 ms frames it performs
two simple passes over 320 samples, or about 32,000 sample inspections per
second. NPU dispatch and tensor marshaling would cost more than the work. NPU
effort should remain focused on ASR, LLM, and neural TTS graphs. If real-phone
recordings show that an energy gate is not robust enough, a later neural VAD
can replace the decision core behind the same recorder/state-machine boundary.

## Implemented behavior

The public states are:

`IDLE -> LISTENING -> SPEECH -> HANGOVER -> ENDPOINT`

- `LISTENING` contains an internal provisional onset. A click or short burst
  does not become public `SPEECH` and cannot create a request endpoint.
- Speech confirmation requires both a continuous activation run and a minimum
  total amount of active speech. Small gaps are tolerated only after the onset
  activation gate has passed.
- `SPEECH -> HANGOVER` occurs on the first inactive frame. Activity during the
  hangover returns to `SPEECH`; enough trailing silence produces `ENDPOINT`.
- A maximum-utterance endpoint prevents an unbounded capture.
- The learned noise floor updates only on non-impulse, inactive frames while
  listening or during initial calibration. It is frozen throughout confirmed
  `SPEECH`/`HANGOVER` and while assistant playback is blocked, so speech and
  speaker output are not learned as background noise.
- A crest-factor plus energetic-coverage check rejects isolated clicks before
  they can contaminate the noise floor.
- Events and metrics use processed sample counts rather than wall-clock time.
  Callback sequence numbers, onset, speech end, decision point, pre-roll start,
  thresholds, endpoint reason, and barge-in flags are deterministic for a PCM
  sequence.
- `rearm()` preserves the learned noise floor and cumulative metrics for the
  next utterance. `reset()` restores the complete construction state.

Default bring-up settings are:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| Initial calibration | 300 ms | Learn the room before allowing activation |
| Activation run | 60 ms | Reject isolated clicks and tiny spikes |
| Minimum active speech | 160 ms | Reject short non-speech bursts |
| Provisional gap tolerance | 60 ms | Tolerate a small gap during onset |
| Trailing-silence endpoint | 420 ms | Keep the endpoint inside the 300–500 ms latency budget |
| Maximum utterance | 15 s | Bound one request; tune after phone tests |
| Pre-roll | 240 ms / 3,840 samples / 7,680 PCM bytes | Preserve initial consonants |
| Activation threshold | max(-48 dBFS, noise + 10 dB) | Dynamic onset gate |
| Release threshold | max(-54 dBFS, noise + 6 dB) | Hysteresis for speech continuation |

These values are starting settings, not acoustically validated Nubia settings.

## Reproducible tests

Run:

```sh
scripts/test_adaptive_energy_vad.sh
```

The dependency-free harness compiles the production class with `javac` in a
temporary directory and tests:

1. clean speech confirmation and pre-roll offsets;
2. gradual background-noise adaptation without false activation;
3. single-click and short-burst rejection;
4. trailing-silence endpoint timing;
5. maximum-duration endpoint timing;
6. cancellation and complete reset;
7. two consecutive utterances with `rearm()`; and
8. blocked and cancel-on-speech barge-in policies.

Observed result:

```text
AdaptiveEnergyVad: 8 tests passed
```

The same sources compile with the Java 8 language/API gate used for Android.
The test JVM itself ran the small synthetic suite in about 0.03 seconds on the
development Mac; that is only a source sanity observation, not a phone power or
latency measurement.

Test sources:

- [`AdaptiveEnergyVadTest.java`](../../../../scripts/AdaptiveEnergyVadTest.java)
- [`test_adaptive_energy_vad.sh`](../../../../scripts/test_adaptive_energy_vad.sh)

## Exact Android integration boundary

Do not put VAD logic in the HTML layer. Instantiate one detector per native
capture session and feed it from the existing `AudioRecord` thread:

```java
AdaptiveEnergyVad.Config vadConfig = AdaptiveEnergyVad.Config.builder()
        .setSilenceEndpointMs(420)
        .setMaxUtteranceMs(15_000)
        .setBargeInPolicy(AdaptiveEnergyVad.BargeInPolicy.DISABLED)
        .build();

AdaptiveEnergyVad vad = new AdaptiveEnergyVad(vadConfig, event -> {
    // Record telemetry here. Post endpoint/control work off the capture thread.
});
vad.start();

// In the AudioRecord loop, preferably with a 320-sample (20 ms) short[] read:
AdaptiveEnergyVad.FrameResult result = vad.processFrame(samples, 0, samplesRead);
```

Integration rules:

1. Prefer `AudioRecord.read(short[], ..., READ_BLOCKING)` with 320-sample
   frames. If the existing byte reads are retained, preserve an odd-byte carry
   and repack them into complete little-endian PCM16 samples before calling the
   detector. Large 4–8 KiB reads make endpoint timing too coarse.
2. While `LISTENING`, keep a 3,840-sample circular PCM buffer. When
   `SPEECH_CONFIRMED` arrives, pin audio starting at
   `event.getRecommendedAudioStartSample()` and continue appending subsequent
   frames. The VAD intentionally does not copy or retain audio itself.
3. On `ENDPOINT`, publish a capture/control message containing endpoint reason,
   `speechStartSample`, `speechEndSample`, `decisionSample`, and the recommended
   audio start. Do not call the current synchronous `stopRecording()` directly
   inside the listener: it joins the capture thread and would risk joining the
   current thread. Post the stop/finalize action to the UI/control thread.
4. For the first phone build, retain audio through the endpoint decision so
   truncation cannot remove a final consonant. Once recordings are inspected,
   it is safe to test trimming most of the 420 ms trailing silence before WAV
   construction to reduce input-prefill work.
5. After the request owns an immutable PCM/WAV copy, either stop capture for a
   half-duplex turn or `rearm()` for the next utterance. Never reuse the mutable
   ring buffer as an in-flight request payload.
6. Calls that mutate/process the detector are synchronized. Callbacks are
   synchronous and must remain short. Metrics snapshots and state reads can be
   collected from the control thread.

Recommended telemetry names for the integration are
`native_vad_listening`, `native_vad_speech_confirmed`,
`native_vad_hangover`, `native_vad_endpoint`, `native_vad_burst_rejected`,
and `native_vad_cancelled`. The endpoint event must carry the sample offsets
above so acoustic recordings can distinguish last detected speech from the
420 ms decision delay.

## Barge-in policy

The initial speakerphone build should use `DISABLED`: call
`setAssistantOutputActive(true)` while response audio is playing so the VAD
neither triggers on the phone speaker nor adapts its noise floor to playback.
Set it back to `false` when playback terminates.

`DETECT_ONLY` and `CANCEL_OUTPUT_ON_SPEECH` are available for controlled tests.
The latter marks a confirmed speech event with
`isCancelOutputRecommended()`. It does not stop playback itself. Full speaker
barge-in still needs acoustic echo cancellation or headset testing; an energy
gate alone cannot reliably distinguish the user from synthesized output.

## Remaining physical gate

Before calling the mode robust, run quiet room, fan/traffic noise, phone-at-arm's
length, accented English, Mandarin, and speaker-echo cases. Measure false
activations per hour, missed starts, clipped initial/final phonemes, last speech
to endpoint median/P95, microphone endpoint to request start, CPU time, RSS,
temperature, and battery. Compare against a neural VAD only if this gate misses
the product target.
