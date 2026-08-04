# Android 0.4.0 multilingual conversation and hands-free candidate

Date: 2026-07-23 (Asia/Shanghai)
Status: APK built and host-validated; physical-phone acceptance is not run

## Release decision

Version code 8 / version name `0.4.0-multilingual-vad` is a new opt-in
candidate. It combines four previously separate goals:

1. explicit Auto/EN/ZH/RU/FR/IT/ES language selection;
2. conservative visible multilingual Auto routing;
3. bounded text-only conversation history; and
4. a persistent, VAD-driven hands-free input mode.

The candidate is not promoted. The default installer remains pinned to the
accepted `0.3.2-turn-telemetry` APK:

```text
android-app/releases/lfm-audio-demo-0.3.2-turn-telemetry-debug.apk
SHA-256 be6c6fdf355d4800a2c569d0bf3d8df04951a5f70e86ccef098190b18de12b98
```

The one-worker low-priority build produced this opt-in artifact:

```text
android-app/releases/lfm-audio-demo-0.4.0-multilingual-vad-debug.apk
Bytes: 4,762,649
SHA-256: 4943dbd56a5083f0cde2f0c0d6566e59a502495f40d9ae3b644ebe0b1c01c918
```

Build-tools validation confirms:

- application ID `ai.liquid.lfmdemo`;
- version code 8 / version name `0.4.0-multilingual-vad`;
- minimum/target SDK 28/35;
- ARM64-only native payload;
- valid v2 Android Debug signature;
- checksum sidecar matches;
- retained APK is byte-identical to the Gradle output; and
- packaged `index.html`, `service-worker.js`, and `manifest.webmanifest`
  hashes exactly match source.

This is artifact validation, not execution evidence. The phone is disconnected,
so none of the new language, history, VAD, TTS, latency, memory, thermal, or
acoustic behavior has physical-phone evidence.

## What the candidate does

### Language selection and Auto routing

One selector applies to Voice response, Transcription, and Native TTS:

| Selection | Model policy | Android TTS locale | Current voice-output route |
| --- | --- | --- | --- |
| Auto | Resolve after real ASR transcript | Resolved locale | English to integrated PCM; other supported languages to Android TTS |
| English | English | `en-US` | Integrated streamed LFM PCM |
| Mandarin Chinese | Mandarin Chinese | `zh-CN` | Android TTS |
| Russian | Russian | `ru-RU` | Android TTS |
| French | French | `fr-FR` | Android TTS |
| Italian | Italian | `it-IT` | Android TTS |
| Spanish | Spanish | `es-ES` | Android TTS |

The integrated LFM PCM path remains evidence-backed only for English. A
non-English Voice response is therefore explicitly rerouted to Android TTS and
records that decision. Installed voice availability, offline status, and
quality remain phone gates.

Auto is deterministic and deliberately conservative:

- Han and Cyrillic script provide strong Mandarin and Russian evidence;
- English, French, Italian, and Spanish use weighted distinctive words,
  repeated function-word evidence, orthography, and a winning margin;
- short acknowledgements, tied evidence, and a single shared word are not
  treated as confident language identification;
- ambiguous input continues the most recent non-fallback language whose
  confidence was at least 0.6;
- without such history, the route is a visibly labelled low-confidence English
  fallback.

The UI shows the resolved language, confidence, evidence/fallback reason, and
fallback state. This is useful bring-up behavior, not a trained language-ID
model or a broad code-switching claim.

Conversational modes perform a sequential ASR request before response
generation. That provides a real transcript for Auto and history, but it also
processes the current audio twice and prevents response generation from
starting until ASR finishes. The added phone latency and thermal cost are
unknown.

Detailed evidence:
[multilingual conversation result](../multilingual_conversation/RESULT.md).

### Bounded conversation history

History is shared by the conversational output modes and stores only normalized
text from a successful current turn:

- real user transcript;
- completed assistant response;
- resolved language and routing metadata.

It never stores WAV, Base64, binary audio, failed/stale turns, or ASR-only
assistant-less entries. Limits are six complete turns, 6,000 characters, and
an approximately 1,500-token display estimate. Oldest complete turns are
removed first; an oversized newest turn is clipped to the effective character
budget. **Clear history** resets it.

History is memory-only. Page/WebView process reload clears it, and persistence
is not claimed.

### Hands-free VAD mode

The UI keeps the three output modes and adds an independent input selector:

- `Push-to-talk`: existing manual recording flow;
- `Hands-free VAD`: one microphone tap starts or stops a persistent native
  conversation session.

The native bridge captures 16 kHz mono PCM16 and feeds exactly 320-sample /
20 ms frames to a pure-Java adaptive-energy VAD. Bring-up defaults are:

| Parameter | Value |
| --- | ---: |
| Initial room calibration | 300 ms |
| Activation run | 60 ms |
| Minimum active speech | 160 ms |
| Endpoint silence | 420 ms |
| Pre-roll | 240 ms |
| Maximum utterance | 15 s |

At endpoint, native capture is stopped and released before one immutable WAV is
published. The WAV covers the selected pre-roll start through the endpoint
decision and is handed to the existing request path without a JavaScript
decode/re-encode cycle.

The initial interaction is strictly half-duplex:

1. endpoint pauses the microphone;
2. the UI marks assistant output active and submits the turn;
3. integrated PCM waits for all scheduled WebAudio sources to end;
4. Android TTS waits for its native terminal playback event;
5. Transcription waits for ASR completion;
6. only then does the UI clear the gate and resume native listening.

Manual stop aborts the active request, cancels Android TTS when applicable,
stops WebAudio, clears the gate, stops native capture, and prevents the
cancelled turn from rearming. Restart remains disabled until native stop is
acknowledged, so a delayed callback cannot terminate the next session. A fatal
native error is itself accepted as terminal because Android may already have
torn down the session and emit no second `stopped` callback. Barge-in and
acoustic echo cancellation are disabled and visibly reported.

The energy VAD remains on CPU by design: its simple 16 kHz sample work is
smaller than a useful NPU dispatch. Its current thresholds are deterministic
bring-up values, not acoustic robustness evidence.

Detailed evidence:
[VAD core](../vad_mode/CORE_RESULT.md),
[native bridge](../vad_mode/BRIDGE_RESULT.md), and
[UI integration](../vad_mode/UI_RESULT.md).

## Compute-backend truth

The candidate application is CPU-only.

| Component | Backend in 0.4.0 | Evidence boundary |
| --- | --- | --- |
| LFM2.5-Audio Q4 runner | CPU | Existing phone baseline; new 0.4 behavior not phone-profiled |
| Integrated English response PCM | CPU | Streamed implementation; audible-start acceptance pending |
| Multilingual ASR preflight | CPU | Host-tested routing path; phone latency/quality pending |
| Android system TTS | Android TTS engine | Locale/voice/offline status depends on installed engine data |
| Adaptive-energy VAD | Java CPU | Deterministic synthetic tests; acoustic phone tuning pending |
| Routing/history/control | CPU | Deterministic source tests |
| SenseVoice QNN | Not integrated or executed | Candidate only; phone transcript-parity gate pending |

The fail-closed telemetry facts are:

```text
inference_backend=cpu_q4
vad_backend=adaptive_energy_java_cpu
npu_candidate=sensevoice_qnn_sm8850
npu_executed=false
```

Fallback reasons are exact and language-dependent:

- broad Auto: `broad_auto_requires_cpu_multilingual_asr`;
- explicit English/Mandarin: `phone_qnn_smoke_not_passed`;
- Russian/French/Italian/Spanish:
  `language_not_supported_by_sensevoice_qnn`.

The partial-NPU feasibility result remains useful but separate. The exact
SM8850/HTP-v81 SenseVoice ten-second context has existing AI Hub evidence of
10.871 ms estimated neural execution, 126.223 MiB estimated peak memory, and
2,597/2,597 HTP operations with zero fallback. A checksum-pinned external QNN
2.40 bundle and QNN-enabled ARM64 Sherpa runner were also prepared. This does
not prove retail-phone execution, end-to-end latency, transcript parity, or app
integration. Broad six-language Auto cannot use SenseVoice because the model
does not support Russian, French, Italian, or Spanish.

Detailed evidence:
[multilingual NPU integration gate](../npu_multilingual/INTEGRATION_GATE.md).

## Host validation at this checkpoint

| Gate | Result |
| --- | --- |
| Streaming/multilingual/VAD UI regression | Pass |
| Native Android TTS bridge regression | Pass |
| Streaming telemetry regression | Pass |
| Adaptive-energy VAD synthetic suite | Pass, 8/8 cases |
| Native VAD bridge/ring regression | Pass |
| Android API-35 Java compilation, one worker | Pass |
| NPU backend selector suite | Pass, 7 cases |
| Versioned 0.4.0 APK build | Pass, 4,762,649 bytes |
| Package/version/signature/embedded-asset checks | Pass |
| Physical Nubia run | Not run; phone disconnected |

Reproduce the focused source checks with:

```sh
node scripts/test_streaming_audio_ui.mjs
python3 scripts/test_android_native_tts_bridge.py
python3 scripts/test_android_streaming_telemetry.py
scripts/test_adaptive_energy_vad.sh
python3 scripts/test_native_vad_bridge.py
python3 scripts/npu_multilingual/test_backend_gate.py
```

## How to install and operate the candidate

1. Install it explicitly; do not change the default installer target:

   ```sh
   scripts/install_native_phone_demo.sh \
     --model-dir work/models/lfm25-audio-q4 \
     --apk android-app/releases/lfm-audio-demo-0.4.0-multilingual-vad-debug.apk
   ```

2. Wait for the foreground service to report that LFM is running.
3. Select Voice response, Transcription, or Native TTS.
4. Select an explicit language for controlled tests, or Auto for exploratory
   routing while watching its visible evidence and fallback status.
5. For Push-to-talk, tap once to begin and once to finish.
6. For Hands-free VAD, tap once to start the session, allow initial room
   calibration, speak, and wait for the assistant to finish before speaking
   again. Tap again to stop and cancel current work.
7. Watch the history budget and use **Clear history** before an independent
   scenario.

Install and verify the desired offline Android TTS voices before disconnecting
USB/network. A locale offered only by a network voice is not an offline demo
result.

## Physical-phone acceptance gate

The complete operational checklist is in
[the Android deployment guide](../../../../docs/ANDROID_DEPLOYMENT.md#040-candidate-phone-acceptance-checklist).
At minimum, the Nubia run must cover:

- explicit EN/ZH/RU/FR/IT/ES transcription and response behavior;
- a recorded Auto confusion matrix, especially EN/ZH and ambiguous short
  phrases;
- installed locale and offline-voice checks;
- multi-turn references, trim limits, Clear history, and restart reset;
- at least ten consecutive VAD turns in each output mode;
- silence, clicks, noise, short speech, maximum duration, echo, lifecycle, and
  manual cancellation;
- last speech to endpoint, endpoint to request, ASR, first token, first PCM/TTS
  proxy, output head, external acoustic onset, completion, and gap metrics;
- CPU/RSS/thermal stability and an unplugged endurance run;
- exact CPU/NPU fallback telemetry; and
- a separate EN/ZH QNN runtime/transcript-parity smoke before any NPU route is
  considered.

The response target remains at most 2.0 seconds median and 3.0 seconds P95 from
end of speech to first audible output, with no audible gap above 80 ms. Source
tests and WebAudio scheduling timestamps cannot establish this acoustic claim.

## Promotion rule

Keep `0.3.2-turn-telemetry` as the default and describe 0.4.0 only as an
unverified candidate until the versioned artifact, physical-phone language,
history, VAD, latency, continuity, resource, and backend-truth gates all have
recorded evidence. AI Hub placement must never be presented as execution inside
this Android application or as AR1/glasses performance.
