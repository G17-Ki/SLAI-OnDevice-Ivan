# Nubia 0.4.1 physical acceptance

Date: 2026-07-24 (Asia/Shanghai)
Device: Nubia/REDMAGIC NX809J, Snapdragon SM8850, Android 16, arm64-v8a
Application: `ai.liquid.lfmdemo`, version code/name
`9 / 0.4.1-protocol-recovery`
APK SHA-256:
`4f13fb28ef244dd5d2bd90bf7731841a0995fa2e8b4231c4feccf24068942a1d`

## Decision

The candidate passes the physical English streaming, bounded-history,
request-protocol, and hands-free transcription-endpoint gates. It reaches the
observed WebAudio output head in 0.691--1.647 seconds across the five
audio-producing fixed-file turns, with zero reported underflows. This meets the
2--3 second first-audio goal for the tested push-to-talk fixtures.

It does **not** pass robust multilingual Auto. Mandarin and Russian audio was
transcribed as Latin phonetics and then misrouted through a low-confidence
English fallback. Explicit Mandarin and Russian correctly fail closed instead
of generating a misleading answer. French, Italian, and Spanish produce
correct text, but this phone has no registered Android TTS engine, so those
turns are text-only.

SenseVoice QNN feasibility now passes on the same physical SM8850/HTP-v81
phone, including exact English and Mandarin parity against the CPU model under
a mandatory fixed ten-second zero-padding contract. The APK still contains no
Sherpa/QNN JNI or service integration, and every application telemetry record
truthfully reports `npu_executed=false`.

The default installer therefore remains pinned to
`0.3.2-turn-telemetry`. Version 0.4.1 is an opt-in research candidate, not a
robust multilingual release.

## Acceptance matrix

| Capability | Physical result | Verdict |
|---|---|---|
| English streamed voice | Relevant woodworking-slogan response; 0.881 s scheduled audio, 0.963 s output head, 4.632 s response completion | Pass |
| Two-turn history | Second answer was `The passphrase is sapphire.`; history grew from one to two turns | Pass |
| Request protocol | One audio object only in ASR; response receives consolidated text history and no audio; roles are only `system` and `user` | Pass |
| Explicit French | `La capitale de la France est Paris.` | Text-only partial |
| Explicit Italian | `La capitale d'Italia è Roma.` | Text-only partial |
| Explicit Spanish | `La capital de España es Madrid.` | Text-only partial |
| Explicit Mandarin | Blocks when LFM ASR returns Latin phonetics; no response request or history mutation | Safe fail-closed; feature unavailable |
| Explicit Russian | Blocks when LFM ASR returns Latin phonetics; no response request or history mutation | Safe fail-closed; feature unavailable |
| Broad Auto Mandarin | Phonetic Latin ASR is classified as 15% English fallback and spoken back | Fail |
| Broad Auto Russian | Phonetic Latin ASR is classified as 15% English fallback and produces a fabricated explanation | Fail |
| Quiet/rapid VAD lifecycle | One quiet cycle and three rapid start/stop cycles return cleanly to `stopped` | Pass |
| Prerecorded VAD endpoint | One trailing-silence endpoint, correct English transcript, output gate, resume, and clean stop | Pass in Transcription mode |
| Hands-free Voice-response playback | Not acoustically exercised | Open |
| SenseVoice QNN CLI | Physical HTP-v81 execution plus exact fixed-shape EN/ZH CPU parity | Feasibility pass |
| QNN inside APK | No JNI/service integration; application remains CPU Q4 | Not implemented |

The complete public-safe case data is in
[`matrix_results.json`](matrix_results.json). The compact event counts, timing,
VAD sequence, and stability caveat are in
[`telemetry_summary.json`](telemetry_summary.json).

## First-audio measurements

All values start at the application request clock, which begins before the ASR
preflight. `Output head` is the WebAudio output-head observation. `Drain` is
the harness wall time after waiting for all scheduled audio to finish, so it is
not response-generation latency.

| Turn | First text | First PCM | Scheduled audio | Output head | Response complete | Playback drain | Underflows |
|---|---:|---:|---:|---:|---:|---:|---:|
| English woodworking | 0.605 s | 0.695 s | 0.881 s | 0.963 s | 4.632 s | 11.230 s | 0 |
| History turn 1 | 0.780 s | 0.878 s | 1.024 s | 1.061 s | 1.680 s | 3.487 s | 0 |
| History turn 2 | 0.410 s | 0.509 s | 0.653 s | 0.691 s | 2.072 s | 5.672 s | 0 |
| Unsafe Auto Mandarin | 1.034 s | 1.165 s | 1.311 s | 1.350 s | 7.206 s | 14.571 s | 0 |
| Unsafe Auto Russian | 1.312 s | 1.466 s | 1.610 s | 1.647 s | 7.162 s | 11.990 s | 0 |

The last two rows pass only the streaming-latency mechanism. Their language
behavior fails and they must not be used as positive multilingual examples.

## Protocol recovery proof

The exclusive matrix captured every `/v1/chat/completions` request before it
left the WebView.

- All 12 ASR requests contain exactly one `input_audio` part and one
  `input_audio` key.
- All eight response requests contain zero audio parts and zero audio keys.
- Response user content is a string containing the current transcript and,
  when present, the bounded text history.
- Only `system` and `user` roles are sent.
- Explicit Mandarin/Russian guards and ASR-only cases correctly make no
  response-stage request.
- Telemetry reports `response_input="asr_text"` and
  `duplicate_audio_encode=false`.

This closes the earlier invalid-role/duplicated-audio request failure without
depending on server-side tolerance.

## Language and TTS boundary

The application exposes Auto, English, Mandarin, Russian, French, Italian, and
Spanish. Explicit selection correctly controls the response language policy.
The tested phone reports `tts_engine_available=false`; a default locale string
of `en-GB` does not imply that a synthesizer exists.

Consequently, French, Italian, and Spanish correctly recover to
`text_only_no_android_tts`. This proves text generation and safe fallback, not
spoken multilingual output. Before a spoken demo, install and verify an
offline Android TTS engine and the required locales, then rerun the same
matrix and an independent listening/back-transcription check.

Broad Auto must remain disabled or visibly experimental for Mandarin and
Russian until the LFM ASR front end is replaced or supplemented. A safe near-
term route is:

- explicit English and Mandarin: SenseVoice QNN after APK integration;
- Auto explicitly limited to English and Mandarin: SenseVoice QNN Auto;
- broad EN/ZH/RU/FR/IT/ES Auto: multilingual CPU ASR or language-ID-first
  routing;
- explicit RU/FR/IT/ES: multilingual CPU ASR.

## Hands-free VAD result

The Java CPU adaptive-energy VAD passed one quiet session and three rapid
start/stop cycles. The final prerecorded endpoint produced:

- one confirmed utterance and one trailing-silence endpoint;
- 3.420 seconds of 16 kHz PCM, 109,440 PCM bytes and 109,484 WAV bytes;
- 22 rejected short bursts and no extra endpoint;
- transcript `Can you help me come up with a slogan for my work?`;
- 333.5 ms ASR first text and 483.4 ms request completion;
- approximately 512.4 ms from the native endpoint event to native resume;
- output gating before request work, clearing only after
  `vad_assistant_output_settled`;
- native and JavaScript return to listening, followed by a clean user stop.

No raw endpoint audio is retained in the public evidence. This test used
Transcription mode, so it proves endpoint, half-duplex gate, request, resume,
and lifecycle mechanics. It does not yet prove last-speech-to-audible-answer
latency or echo behavior with streamed assistant speech.

## Physical QNN feasibility

The external, checksum-pinned SenseVoice bundle used QNN `2.40.0.251030` on
SM8850/HTP-v81. The physical command-line runner initialized the HTP provider
and returned:

| Fixture | Source duration | Recognizer init | QNN decode | Whole command | Result |
|---|---:|---:|---:|---:|---|
| English | 7.152 s | 319 ms | 41 ms | 585 ms | Correct text, `<|en|>` |
| Mandarin | 5.592 s | 383 ms | 36 ms | 640 ms | Correct text, `<|zh|>` |

These are single runs per language, not median/P95, sustained-runtime, memory,
or thermal evidence. The QNN version attribution comes from the pinned bundle
manifest and verified runtime hashes; the smoke command did not independently
print a version string for every loaded library.

Independent CPU inference matches both outputs exactly only when the input is
zero-padded or truncated to the QNN context's fixed ten-second shape. Without
that padding, English changes `gold` to `code` and Mandarin changes `开饭` to
`开放`. Padding/segmentation is therefore an integration contract, not an
optional preprocessing detail.

The reviewed smoke record activates the fail-closed selector for explicit
English, explicit Mandarin, and Auto limited to EN/ZH. Broad Auto and
RU/FR/IT/ES remain on CPU. These are selector and physical-CLI results only;
the current APK has neither the recognizer nor the QNN runtime path.

See:

- [`npu/sensevoice_cpu_parity.json`](npu/sensevoice_cpu_parity.json)
- [`npu/sensevoice_qnn_reviewed_smoke.json`](npu/sensevoice_qnn_reviewed_smoke.json)
- [`npu/backend_selection_matrix.json`](npu/backend_selection_matrix.json)
- [`npu/device_facts.json`](npu/device_facts.json)

## Stability evidence and limitation

The 12-case harness finished with `failure=null`, 97 matrix events, zero
`turn_error`, zero `turn_cancelled`, and zero blocked file-input events. The
full VAD run contains 331 events with zero `turn_error`, `turn_cancelled`, or
`vad_error`. The final Activity state was foreground, focused, and resumed.

Both saved logcat captures are zero bytes. They cannot independently prove the
absence of a crash or ANR. The defensible claim is that the harness, telemetry,
server tail, and final resumed Activity showed no failure during this run.

## Reproduce the WebView matrix

1. Install the candidate explicitly; the default installer intentionally
   remains on 0.3.2:

   ```bash
   scripts/install_native_phone_demo.sh \
     --model-dir work/models/lfm25-audio-q4 \
     --apk android-app/releases/lfm-audio-demo-0.4.1-protocol-recovery-debug.apk
   ```

2. Wait for the app to report `Model ready`. The local server root returning
   HTTP 404 is the expected readiness response.

3. Copy [`acceptance_plan.template.json`](acceptance_plan.template.json) to a
   work directory. Supply mono PCM16 WAV fixtures for the listed utterances.
   Raw speaker recordings are deliberately not published here.

4. Forward the debuggable WebView socket and run the exact matrix harness:

   ```bash
   APP_PID=$(adb shell pidof ai.liquid.lfmdemo | tr -d '\r')
   adb forward tcp:9224 localabstract:webview_devtools_remote_${APP_PID}
   node scripts/run_phone_webview_acceptance.mjs \
     --port 9224 \
     --plan work/phone_acceptance/acceptance_plan.json \
     --output work/phone_acceptance/raw_matrix.json
   ```

5. Pull the app-private telemetry and create a public-safe summary:

   ```bash
   adb exec-out run-as ai.liquid.lfmdemo \
     cat files/streaming-telemetry.jsonl \
     > work/phone_acceptance/telemetry.jsonl
   python3 scripts/summarize_phone_acceptance.py --help
   ```

The sanitizer requires the matrix/full telemetry plus the VAD result files. It
removes fixture paths, WebView IDs, internal stacks, and full prompt text.
Never publish raw QNN smoke logs: upstream source locations can contain a local
home-directory path. Publish the reviewed smoke record instead.

## Public evidence

The raw acceptance screenshots are intentionally excluded from the public
tree; the current release UI is shown in the
[v0.4.13 result](../../expressive_lfm/REPORT.md).

File hashes and sizes are recorded in
[`artifact_manifest.json`](artifact_manifest.json).
