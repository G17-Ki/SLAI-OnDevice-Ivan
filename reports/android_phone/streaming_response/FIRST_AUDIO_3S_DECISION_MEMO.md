# First-audio ≤3-second decision memo

Date: 2026-07-23
Scope: Nubia phone demo first, with a path toward multilingual SM8850/HTP-v81
deployment

## Decision

The shortest path to a convincing demo is to **stream the current integrated
LFM audio on the phone**. Do not replace the English model stack before testing
the new playback path. Retained Nubia measurements already put the first LFM
PCM at 800 ms median / 837 ms P95 after the request begins; version
`0.3.2-turn-telemetry` now schedules that packet immediately with a 140 ms startup
cushion instead of waiting roughly 6.8 seconds for generation to finish.

Use a **separate, warm standalone TTS runtime** for multilingual expansion.
The modular route is ASR -> compact text LLM -> sentence/clause-chunked TTS,
with exactly one locale stack resident at a time. The standalone TTS request
must not go to the same LFM server: the packaged server has one serial request
worker, so that request would wait behind the active answer.

The evidence makes a sub-3-second first response credible, but it is not yet a
phone result. The updated APK has not been run on the physical phone. Current
phone evidence stops at first received PCM; the WebAudio output head, Android
audio path, acoustic speaker onset, VAD/bridge overhead, queue continuity,
memory, and thermal behavior remain open.

## Evidence labels

- **PHONE-M**: measured on the Nubia. It may still stop before acoustic output.
- **MAC-M**: measured end-to-end or by component on the Apple M2 host.
- **QRD-M**: account-owned AI Hub measurement on Snapdragon 8 Elite Gen 5 QRD,
  Android 16, SM8850, HTP v81. It is not a Nubia measurement.
- **DERIVED**: arithmetic, source inspection, or deterministic replay over
  measured data.
- **UPSTREAM**: a published same-generation result, not measured in this
  project.
- **OPEN**: not yet measured.

## Current latency decomposition

### Integrated English LFM

The existing phone profile starts its clock at request submission after a
complete input WAV is available. The updated UI starts its own clock even
later, inside `sendToPhone`, after the native microphone WAV has crossed the
Java-to-JavaScript bridge. Neither trace measures the complete interval from
the user's last speech sample to the speaker.

| Segment | Current evidence | Class | Interpretation |
|---|---:|---|---|
| Last speech sample -> VAD close | 300-500 ms engineering budget | DERIVED | Professor-confirmed interaction is VAD-gated; this endpoint delay is not present in the retained profile. |
| VAD close -> HTTP request start | 100-300 ms engineering budget | DERIVED | Includes native stop/join, WAV creation, Base64/JavaScript transfer, JSON creation, and hand-off. `0.3.2-turn-telemetry` now records a manual-stop proxy, WAV readiness, and request start, but no physical trace exists yet. |
| Request start -> first text | 677 / 701 ms median/P95 | PHONE-M | Retained Nubia Q4 CPU chat profile. |
| Request start -> first PCM | **800 / 837 ms median/P95** | PHONE-M | The model already emits audio far before completion. |
| First PCM -> scheduled playback | 140 ms by implementation | DERIVED | A fixed initial queue lead, not a measured Android output delay. |
| Scheduled playback -> WebAudio output head | unknown | OPEN | `0.3.2-turn-telemetry` persists the rendering-boundary estimate, but it has not run on the phone. |
| Output head -> acoustic speaker onset | unknown | OPEN | Mixer/HAL, DAC, speaker, and Bluetooth are outside WebAudio telemetry. |
| Request start -> response complete | 6,786 / 6,877 ms median/P95 | PHONE-M | No longer first-audio critical if streaming stays continuous. |

Using the retained first-PCM P95 and the current engineering allocations gives
a planning range of approximately:

`300-500 VAD + 100-300 hand-off + 837 model + 100-200 queue/device = 1.34-1.84 s`

That is a budget calculation, not a completed phone measurement. It leaves
about 1.2 seconds of margin under the 3-second gate for unmodeled scheduling,
but only the acoustic phone run can establish whether that margin is real.

Queue supply is encouraging. The retained phone generated 11.42 seconds of
audio while completing in about 6.8 seconds. A deterministic replay of 20 Mac
packet timelines stayed gap-free at the 140 ms cushion under a uniform 1.96x
post-first-packet slowdown. The retained phone's aggregate generation pace was
1.44x slower than the pooled Mac trace. This is continuity evidence, not a
phone underflow measurement.

### Modular multilingual route

| Path or component | First-audio-related result | Class | What it proves |
|---|---:|---|---|
| LFM ASR -> LFM2.5-1.2B -> Supertonic, English | 1,153 / 1,213 ms scheduled first audio | MAC-M | A complete separate-TTS cascade can fit the budget on the host and returned the intended answer. |
| LFM ASR -> Qwen3-1.7B budgeted route -> Supertonic, English | 1,164 / 1,262 ms scheduled first audio | MAC-M | A somewhat stronger text route still fits on the host when reasoning is bounded. |
| Whisper Tiny -> Qwen3-1.7B -> Supertonic, Spanish | 1,021 / 1,121 ms scheduled first audio | MAC-M | Orchestration works, but Whisper Tiny's broader quality was rejected. |
| SenseVoiceSmall INT8 | 101 / 174 ms final transcript | MAC-M | Leading EN/JA/KO/ZH ASR candidate; offline/final rather than partial streaming. |
| SenseVoice ten-second QNN context | 10.871 ms, 126.223 MiB | QRD-M | QNN-2.40 context accepted under QAIRT 2.45; 2,597/2,597 operations on HTP with zero fallback. Profile-only: no transcript comparison or host/pre/post time. |
| LFM2.5-1.2B Q4 | 111 / 215 ms first speakable text | MAC-M | Default compact multilingual text model. |
| Qwen3-1.7B, reasoning budget 16 | 509 / 707 ms first speakable text | MAC-M | Selective quality route; unconstrained thinking is too slow. |
| Supertonic, worst tested short-language case | 543 / 628 ms first PCM | MAC-M | One shared EN/ES/DE/FR/IT/JA/KO/RU-class TTS can fit after a short text chunk; Mandarin is unsupported. |
| SenseVoice + LFM + worst tested Supertonic + 140 ms | 1,157 ms P95 arithmetic | DERIVED | Independent host-component planning sum; not one phone run. |
| Whisper Small CPU + LFM + same Supertonic + 140 ms | 3,365 ms P95 arithmetic | DERIVED | Latin-language Whisper Small needs NPU acceleration or a different ASR to meet the target. |
| Four-step Supertonic neural graph sum | 118.891 ms | QRD-M | All four fixed-shape neural components run on HTP v81; excludes host work, transfers, dynamic lengths, playback, and quality acceptance. |
| Piper Italian fixed neural component sum | 34.311 ms | QRD-M | Encoder, SDP, flow, and decoder are fully NPU placed. The replacement real flow fixture passes at NRMSE 0.000616 and cosine 0.9999998. |
| Piper/Melo first decoder payload | about 464 ms of new audio per call | DERIVED + UPSTREAM | Their graph boundary can stream decoder chunks; the public app's complete-WAV behavior is not a graph requirement. |

The measured Mac cascades begin after the complete input waveform is
available. They do not include VAD endpointing and are not projections of
Nubia CPU, NPU, memory, or audio-device behavior.

## Prioritized architecture

### Priority 0: validate streamed integrated LFM

Ship no new model for the first English test. Install the telemetry APK,
measure actual output-head and acoustic onset, and verify that the 140 ms queue
does not starve. Full response completion may remain 5-7 seconds; it is not a
failure when speech starts promptly and the queue stays ahead.

Input can remain VAD/end-of-turn gated for this milestone. The current system
buffers one complete input WAV, then streams text and output PCM. Moving ASR to
duplex input streaming is a later optimization unless the measured VAD plus
bridge segment breaks its budget.

### Priority 1: add a separate multilingual TTS path

Use the tested bounded sentence/clause chunker and one dedicated warm TTS
worker. Feed only newly visible, non-reasoning text; synthesize chunks in order;
bound the queue; play every first PCM callback immediately; and invalidate
stale chunks on barge-in.

- Use **Supertonic** as the fastest shared CPU integration prototype for its
  supported languages.
- Use **Piper** as the preferred Qualcomm-native EN/DE/IT route once the
  complete phone chain passes numerics, intelligibility, and playback.
- Use **MeloTTS** as the preferred Qualcomm-native ES/ZH route after a fresh
  exact-target and phone run.
- Do not create a second full LFM process for TTS; retained phone residency is
  already 2,169 MiB (about 2.12 GiB) combined sampled peak.

### Priority 2: move multilingual ASR/TTS to HTP and reduce residency

Select and prewarm one locale at session activation. Keep one ASR, one text
model, and one output engine resident. Do not keep LFM and Qwen, or every TTS
family, loaded merely to switch per turn. NPU work is important for power and
memory as well as latency, but it should not delay the English streaming phone
gate.

## Language routing and present readiness

| Language | Route to validate | Present decision |
|---|---|---|
| **EN** | Integrated LFM streaming. Modular fallback: SenseVoice -> LFM2.5-1.2B -> Piper-EN or Supertonic. | **Next phone gate.** First PCM is already measured on the Nubia; acoustic start is open. |
| **ES** | Whisper Small, fixed locale -> LFM2.5-1.2B -> MeloTTS-ES; Supertonic fallback. | **HTP component gate passed.** Exact-QRD W8A16 encoder plus 20 decoder calls is 264.917 ms of neural arithmetic, but quantized transcript quality, host decode, and phone latency remain open. |
| **DE** | Whisper Small -> LFM2.5-1.2B -> Piper-DE; Supertonic fallback. | Architecture selected, but no end-to-end phone result. |
| **FR** | Whisper Small -> LFM2.5-1.2B -> Supertonic. | Architecture selected, but French TTS quality and phone timing remain open. |
| **IT** | Whisper Small -> Qwen budget 16 when quality is needed, otherwise LFM2.5-1.2B -> Piper-IT. | Best fresh Qualcomm TTS component evidence. All four neural components and the real flow fixture pass. An accumulated run completed only with an invalid hand-built phoneme-ID fixture; the corrected production-phonemizer-ID rerun and phone integration remain open. |
| **ZH** | SenseVoice -> LFM2.5-1.2B -> MeloTTS-ZH. | **No public end-to-end claim yet.** SenseVoice now has a 10.871 ms exact-QRD strict-NPU compatibility profile, but no QRD transcript gate. Supertonic does not support Mandarin; Kokoro remains no-go after invalid PCM on 10/31 short safety calls. |
| **JA** | SenseVoice -> LFM2.5-1.2B -> standalone LFM-JP or Supertonic. | Use a modular route only. Persistent interleaved LFM-JP is release-blocked by corrupted/clipped turn 2. One Japanese Supertonic exact-QRD accumulated phrase now backchecks at 0.0 CER, but it is only a conditional intelligibility pass. |
| **KO** | SenseVoice -> LFM2.5-1.2B -> Supertonic. | Promising host components, no phone end-to-end proof. |
| **RU** | Whisper Small -> Qwen budget 16 when quality is needed, otherwise LFM2.5-1.2B -> Supertonic. | Architecture selected; ASR fixture and phone execution remain open. |

Language should be selected or cached for the session. Do not spend each short
turn on unreliable language detection. Every route must answer and synthesize
in the selected language; ASR support alone is not multilingual operation.

## Exact phone validation protocol

### 1. Artifact and environment

1. Verify
   `android-app/releases/lfm-audio-demo-0.3.2-turn-telemetry-debug.apk` has SHA-256
   `be6c6fdf355d4800a2c569d0bf3d8df04951a5f70e86ccef098190b18de12b98`.
2. Install with `adb install -r` so app-private model files remain intact.
3. Disable Bluetooth and use the built-in speaker. Fix media volume, display
   brightness, power mode, and ambient conditions. Run unplugged after the
   battery is stabilized; record battery level and temperature.
4. Record phone model, Android build, app version, model hashes, runtime
   arguments, and whether the route is CPU or QNN. Do not combine results from
   different configurations.

### 2. One monotonic event trace per turn

Version `0.3.2-turn-telemetry` gives every turn a unique ID and records the
implemented events to app-private JSONL using Android's monotonic clock. Pull
them with `scripts/pull_android_streaming_telemetry.py`. The remaining
speech-end/VAD and barge-in events below are acceptance-protocol requirements,
not yet implemented measurements.

1. `speech_end`: last reference/non-silent input sample;
2. `vad_close`: endpoint accepted, or manual stop for the current APK;
3. `wav_ready`: native WAV and Base64 payload ready;
4. `request_start`: immediately before `fetch`;
5. `first_text_sse`;
6. `first_pcm_sse`, plus every later PCM packet arrival and audio duration;
7. `first_buffer_queued` and its scheduled context time;
8. `output_head_crossing`, or an explicit unsupported/unavailable state;
9. `response_complete`;
10. every queue-underflow event, barge-in/cancel event, and stale-generation
    rejection.

Sample app/model RSS, VmHWM, Android thermal status, and battery temperature at
turn start, first PCM, completion, and five seconds after completion.

### 3. Acoustic ground truth

Internal output-head telemetry is not speaker onset. Record the screen and
speaker with a second device at 120 fps or faster. Flash an unambiguous visual
marker when `vad_close` is logged. Define acoustic onset as signal at least
10 dB above the preceding 500 ms noise floor, sustained for at least 40 ms.
Keep the raw recording and report marker-to-onset latency. Calibrate the visual
marker delay once with an app-generated reference tone; do not infer acoustic
time only from WebAudio.

### 4. Run matrix

For the integrated English route:

1. Record three cold model-ready times separately; cold load is not mixed into
   warm per-turn latency.
2. Run five warm-up turns and exclude them.
3. Run ten deterministic fixed-file chat turns without restarting the model.
4. Run ten microphone turns from a fixed script, also without restarting.
5. Continue for 20 more turns in the same session to expose thermal slowdown,
   state leakage, and queue drift. Trigger barge-in on turns 5, 10, 15, and 20.
6. Report fixed-file, microphone, and sustained-session distributions
   separately: median, nearest-rank P95, maximum, first five versus last five,
   and every failure.

For each modular language route, load only that locale stack, then run five
warm-ups, ten fixed-file turns, ten microphone turns, and a 20-turn sustained
session. Use scripted inputs with reference transcripts and deterministic
expected answer facts. Preserve the full ASR text, LLM text, TTS text chunks,
and generated audio for scoring.

## Explicit go/no-go thresholds

### Integrated English demo

All of these must pass:

- 20/20 primary fixed-file and microphone turns produce finite, audible,
  English output with no crash or missing response.
- Last speech sample -> acoustic onset is **<=2.0 s median and <=3.0 s P95**
  in both fixed-file and microphone sets.
- Speech end -> VAD close is <=500 ms P95; VAD close -> request start is
  <=300 ms P95; request start -> first PCM is <=1.2 s P95; first PCM ->
  acoustic onset is <=250 ms P95.
- Zero audible gaps longer than 80 ms and zero logged queue underflows in the
  primary 20 turns.
- Barge-in stops queued speech and prevents stale audio within 250 ms.
- Android thermal status never reaches `SEVERE`; last-five first-audio P95 is
  no more than 25% slower than first-five P95.
- Combined app plus model peak RSS is <=2.6 GiB and does not grow by more than
  10% between the first and final five turns. This is a phone-demo guard based
  on the retained 2,169 MiB (about 2.12 GiB) peak, not a glasses memory claim.

A failure of only the WebAudio output-head API is not itself a no-go if the
external acoustic measurement passes. A latency miss, audible gap, stale
barge-in audio, crash, non-finite PCM, or reproducible turn-state corruption is
a no-go.

### Modular multilingual route

The same acoustic, continuity, thermal, barge-in, and stability gates apply,
plus:

- fixed-locale ASR final transcript <=700 ms P95 on the phone;
- first speakable text <=750 ms P95 after ASR completes;
- TTS first PCM <=900 ms P95 after its first chunk is ready;
- 100% of primary turns answer and speak in the selected language;
- scripted intent/fact accuracy >=95%; no safety-critical or demo-critical
  answer may be wrong;
- TTS emits finite, non-clipped PCM on every call, stays below 0.7 RTF after
  the first callback, and has no ASR-backcheck regression greater than five
  percentage points versus the local reference;
- at least 20 generated phrases per target language pass human listening for
  intelligibility, pronunciation, and absence of severe artifacts before a
  public multilingual claim.

Additional hard blocks remain in force: do not ship persistent interleaved
LFM-JP until 20 sequential turns show no turn-2 corruption/clipping; do not use
Kokoro as the Mandarin primary until at least 100 short/boundary phrases have
zero non-finite outputs; do not describe the Supertonic QRD chain as quality-
accepted from one English ASR backcheck.

## Active checkpoint interpretation

- The English Supertonic accumulated QRD chain is a conditional one-phrase
  intelligibility pass, not a waveform or human-quality pass.
- The fresh Piper Italian QRD components total 34.311 ms with zero reported CPU
  fallback. The original random flow fixture overflowed locally to all-NaN;
  the deterministic real Italian replacement passes against the same linked
  context at NRMSE 0.000616 and cosine 0.9999998.
- A later two-decoder Piper chain used a 34-ID hand-built sequence that omitted
  Piper's required interspersed pad ID `0`. Its finite 0.824-second waveform
  and 0.464-second first decoder output are low-level propagation evidence
  only, not Italian intelligibility or a production phrase boundary. The exact
  79-ID production fixture is persisted and its inference-only retry is gated
  in detached tmux.
- The Japanese Supertonic accumulated chain completed all seven inference-only
  stages. One fixed-locale SenseVoice pass transcribed dynamic-local,
  fixed-local, and remote audio exactly at 0.0 CER despite large waveform drift.
- The ten-second SenseVoice QNN context passed exact-QRD placement and latency
  under QAIRT 2.45, but no AI Hub transcript-equivalence run is possible from
  the release archive alone.
- The public Galaxy S26 W8A16 Whisper Small contexts also passed unchanged on
  exact SM8850: encoder 186.277 ms and decoder 3.932 ms per call, with
  4,867/4,867 graph operations on HTP across the two components. The
  264.917 ms encoder-plus-20-call sum is component arithmetic, not a phone
  transcript measurement, and the release has no matching numerical golden.
- The phone is disconnected; no `0.3.2-turn-telemetry` physical measurement is
  represented as complete.

## Evidence

- [Consolidated real-time/multilingual report](REPORT.md)
- [Phone LFM data-flow and server-concurrency audit](LFM_PHONE_PIPELINE_AUDIT.md)
- [Current per-language routing](LANGUAGE_ROUTING.md)
- [Android turn-telemetry APK validation](validation_2026-07-23_turn_telemetry.md)
- [Queue robustness replay](queue_robustness.md)
- [Retained Nubia native profile](../native_profile/20260716t120118z/REPORT.md)
- [Supertonic host benchmark](supertonic3_mac_m2/REPORT.md)
- [SenseVoice host and QNN gate](sensevoice_asr_mac_m2/REPORT.md)
- [SenseVoice exact-target AI Hub profile](sensevoice_asr_mac_m2/AIHUB_V81_PROFILE.md)
- [Whisper W8A16 exact-target AI Hub profile](whisper_v81/RESULT.md)
- [Account-owned HTP-v81 voice results](aihub_voice_v81/REPORT.md)
- [Piper exact-target production-chain result](piper_v81/PRODUCTION_CHAIN_RESULT.md)
- [Qualcomm TTS streaming-boundary audit](VOICEAI_TTS_STREAMING_GATE.md)
- [English accumulated Supertonic chain](supertonic_v81/ACCUMULATED_CHAIN_RESULT.md)
- [Japanese Supertonic result](supertonic_v81/ja_paris/RESULT.md)

The numeric results above are copied or arithmetically derived from the linked
persisted evidence. No physical phone mutation or measurement was performed
while synchronizing this memo.
