# Integrated LFM latency-gap audit

Date: 2026-07-23
Scope: current warm, on-phone integrated LFM chat path versus a 2-3 second
first-audio target. No model, cloud, build, or new phone run was used for this
audit.

## Decision

The retained evidence says the model is **not the current first-audio
bottleneck**. On the Nubia, the fixed 4.9-second question reached the first LFM
PCM packet in **799.7 ms median / 837.4 ms P95** from client request start. The
streaming UI then gives the first packet a fixed **140 ms** scheduling lead.
That places the derived request-to-scheduled-start boundary at about **940 ms
median / 977 ms P95**, well inside the target.

What remains unproven is the interval before the request and the interval after
WebAudio scheduling. In particular, there is no measured VAD-close event, no
phone trace from the released telemetry APK, and no acoustic observation of
the built-in speaker. The correct conclusion is therefore **credible with
substantial margin, but not yet accepted on the phone**. The roughly 6.8-second
full-response time is no longer first-audio critical because PCM is consumed as
it arrives.

## Keep the time origins separate

| Boundary | What it means in this project | Current status |
|---|---|---|
| Last speech sample | Acoustic end of the user's utterance. | **Not recorded.** |
| VAD close | Endpoint decision that authorizes inference. | **Not implemented in this phone UI.** The current microphone path uses a manual tap. |
| `input_stop_requested` | JavaScript event immediately before asking the native recorder to stop. | Recorded by `0.3.2-turn-telemetry`; it is a **manual-stop proxy**, not VAD close. |
| `wav_ready` | Complete WAV has returned across the native-to-WebView bridge and has been decoded into bytes. | Instrumented, but not yet measured on the phone release. |
| `request_start` | Client pre-fetch marker after the audio Base64 and payload object exist. Telemetry persistence and `JSON.stringify(payload)` still occur before `fetch` is invoked. | The intended per-turn request origin in the new telemetry, but not an exact network-send or server-receive timestamp. |
| First PCM SSE | First generated audio packet parsed from the local server stream. | Retained direct phone profile: **799.7/837.4 ms median/P95** from request start. |
| First buffer queued | WebAudio buffer creation and scheduling have completed. | Instrumented. The first target time is deliberately `currentTime + 140 ms`. |
| Output-head crossing | `AudioContext.getOutputTimestamp()` says the WebAudio rendering head crossed the scheduled context time. | A useful internal proxy; the API may be unsupported or unavailable. It excludes Android's mixer/HAL and the speaker. |
| Physical audible onset | Sound observed outside the phone, after any leading silence in the generated packet. | **Not measured.** This is the acceptance boundary. |

The older retained profile and the new app telemetry have similar but not
identical request origins. The profiling script builds the complete request
body first and starts its clock immediately before opening the local HTTP
request. The app emits `request_start` after Base64 creation but before its
telemetry write, request JSON serialization, and `fetch` invocation. Those
small client-side steps are therefore included in the app's event-to-event
latency and excluded from the retained direct-server profile.

One interpretation trap applies to the retained `0.3.2` APK. In that build,
the UI variable `requestStartedAt` is captured earlier, before the second WAV
Base64 encoding and JSON construction. Its legacy summary fields can therefore
include pre-request payload work. For a `0.3.2` trace, compute scheduled start
as `request_start -> first_buffer_queued event + scheduled_delay_ms`, and
compute output-head time from its recorded `performance_time` minus
`request_start.performance_ms`. Do not mix either value with the manual-stop
or VAD origin.

### Follow-up source correction

The subsequent `0.3.3-native-tts` source change closed this telemetry-origin
issue. The current `0.3.4-input-handoff` APK retains that correction and also
removes the native microphone's redundant JavaScript Base64 decode/re-encode.
Neither candidate has run on the phone. The UI now adopts the emitted
`request_start` event's own
`performance_ms` and monotonic timestamp as the request clock, and the
summarizer derives scheduled start from the `request_start`/queue event
difference plus `scheduled_delay_ms`. Output-head latency is derived from the
recorded output-head `performance_time` minus `request_start.performance_ms`.
Legacy elapsed-detail fields remain fallback-only for older traces. The
retained `0.3.2` APK was not rebuilt, so its raw trace must still be interpreted
with these event-based formulas. Build-only evidence for the optimized path is
in [`NATIVE_INPUT_HANDOFF.md`](NATIVE_INPUT_HANDOFF.md).

## Current latency budget

| Segment | Best current evidence | Classification |
|---|---:|---|
| Manual/VAD close -> request start | 100-300 ms engineering allowance | Derived; unmeasured |
| Request start -> first PCM | 800/837 ms median/P95 | Measured on Nubia, fixed 4.9-second question |
| First PCM -> scheduled WebAudio start | 140 ms | Implementation constant |
| Scheduled start -> WebAudio output head | Unknown | Awaiting telemetry APK phone run |
| Output head -> acoustic onset | Unknown | Requires external recording |

From a VAD-close origin, the known plus allocated P95 path to the scheduled
WebAudio start is approximately **1.08-1.28 seconds** (`100-300 + 837 + 140`).
That leaves about **0.72 seconds under a 2-second target** and **1.72 seconds
under a 3-second target** for Android output latency and unmodeled jitter. If
the requirement instead starts at the last speech sample, adding the existing
300-500 ms VAD allowance gives **1.38-1.78 seconds** before the still-unknown
hardware/acoustic tail. These are planning sums, not end-to-end measurements.

Queue continuity is encouraging but also not accepted on-device. Deterministic
replay stayed gap-free up to a 1.96x uniform post-first-packet slowdown with the
140 ms cushion; the retained phone's aggregate generation pace was 1.44x the
pooled Mac pace. That supports keeping the cushion for the first physical run
rather than optimizing it prematurely.

## Remaining code-level blockers and risks

1. **The input handoff is fully blocking and has an unmeasured tail.** The
   native bridge stops `AudioRecord`, waits up to 1.5 seconds for its capture
   thread, copies the entire PCM buffer, creates a WAV, Base64-encodes it,
   crosses into JavaScript, serializes the request JSON, and only then starts
   ASR. Version `0.3.4` removes the prior JavaScript decode/re-encode, but every
   remaining step is still synchronous and unmeasured on the phone. Long
   utterances remain the highest-risk endpoint-to-request case.
2. **The phone demo has no VAD event or speech-end marker.** It can validate
   manual-stop-to-audio latency, but it cannot by itself validate the stated
   last-speech/VAD-close requirement or the professor's always-on interaction
   mode.
3. **The corrected telemetry and input handoff are build-only evidence.** The
   `0.3.4` APK contains the shared request origin and direct Base64 path, but no
   phone trace proves their timing or lifecycle behavior yet. Retained `0.3.2`
   traces still require the event-based interpretation above.
4. **Playback activation is primed from the user's recording tap.** That is
   appropriate for this UI, but an automatic VAD turn after a long idle may
   encounter a suspended `AudioContext` and pay an unmeasured resume delay.
   Include an idle turn in the physical check before transferring the result
   to an always-on design.
5. **The UI retains every output packet while also scheduling it.** This does
   not gate first audio, but the full replay copy plus all not-yet-ended
   `AudioBufferSourceNode`s can create memory/GC jitter or stale queued speech
   on a long answer. The current UI also disables the next turn until the HTTP
   response closes, so barge-in/continuous-turn acceptance remains separate.
6. **The local server is serial.** Any future second request sent to the same
   LFM process waits behind the active turn. This does not hurt the present
   single integrated request, but it rules out same-process standalone-TTS
   overlap and must be respected by an always-on controller.

None of these findings justifies changing the LFM model before measuring the
released streaming path. The first-text-to-first-PCM gap is only about 100-140
ms in the retained chat runs, so model/prompt tuning has much less available
gain than validating and, if necessary, shortening the endpoint and playback
tails.

## Smallest next phone acceptance experiment

Install
`android-app/releases/lfm-audio-demo-0.3.4-input-handoff-debug.apk` explicitly
for this acceptance run so the corrected request clock and direct microphone
handoff are measured. Keep the existing `0.3.2-turn-telemetry` APK as the
rollback/control and do not change the default installer until the candidate
passes.

1. Install with replacement, keep the model warm, disable Bluetooth, and use
   the built-in speaker at fixed volume. Clear the app telemetry once.
2. Exclude two warm-up turns, then perform **10 microphone chat turns** using
   the same short script. Stop immediately after the final word. Include one
   turn after at least 60 seconds idle.
3. Film the screen, tapping finger, and speaker with a second device at 120 fps
   or faster while recording audio. Treat the visible tap only as the current
   manual endpoint proxy. Define acoustic onset as at least 10 dB above the
   prior 500 ms noise floor for at least 40 ms.
4. Pull the JSONL trace and report, per turn: manual stop -> request start,
   request -> first PCM, request -> corrected scheduled start, corrected
   output-head result, external tap -> acoustic onset, and every underflow or
   error.
5. Pass this minimal first-audio gate only if all 10 turns are audible,
   manual-stop -> acoustic onset is <=2.0 seconds median and <=3.0 seconds P95,
   request -> first PCM is <=1.2 seconds P95, and there are zero logged
   underflows. Label the result **manual-stop**, not VAD, until a real VAD-close
   marker is added and rerun.

Ten measured turns are the smallest useful set here: nearest-rank P95 becomes
the worst observed turn, so one slow tail cannot be hidden. A longer thermal,
barge-in, true-VAD, and multilingual matrix should follow only after this
single-path acoustic uncertainty is resolved.

## Evidence inspected

- [`native_profile/20260716t120118z/profile.json`](../native_profile/20260716t120118z/profile.json)
- [`FIRST_AUDIO_3S_DECISION_MEMO.md`](FIRST_AUDIO_3S_DECISION_MEMO.md)
- [`queue_robustness.md`](queue_robustness.md)
- [`validation_2026-07-23_turn_telemetry.md`](validation_2026-07-23_turn_telemetry.md)
- [`index.html`](../../../android-app/app/src/main/assets/demo-ui/index.html)
- [`NativeAudioBridge.java`](../../../android-app/app/src/main/java/ai/liquid/lfmdemo/NativeAudioBridge.java)
- [`pull_android_streaming_telemetry.py`](../../../scripts/pull_android_streaming_telemetry.py)
