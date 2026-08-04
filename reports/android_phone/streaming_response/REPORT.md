# Real-time multilingual voice response study

Date: 2026-07-22 to 2026-07-23<br>
Target: first audible response within 2-3 seconds after VAD closes<br>
Phone: Nubia NX809J, SM8850, Android 16<br>
Status: preliminary architecture study; English streaming APK built, while
multilingual integration and physical-phone playback/QNN validation remain pending

## Executive decision

Use two paths, in this order:

1. **Primary English demo: stream the current integrated LFM audio.** The model
   already emits the first PCM far before generation completes. The Android UI
   was buffering those packets into a complete WAV; version
   `0.3.2-turn-telemetry` now schedules them immediately with a 140 ms startup
   cushion, distinguishes the queued start from the WebAudio output head, and
   persists a native-clock JSONL trace for each turn.
2. **Multilingual expansion: fixed-locale ASR -> compact text LLM -> chunked
   standalone TTS.** SenseVoiceSmall is the leading EN/JA/KO/ZH recognizer;
   Whisper Small is the current ES/DE/FR/IT/RU candidate. Its public W8A16
   contexts now pass exact-SM8850 profiling at 186.277 ms for the fixed
   30-second encoder and 3.932 ms per decoder call, entirely on HTP; transcript
   equivalence and Nubia execution remain open. LFM2.5-1.2B remains the default
   smaller/faster text model; Qwen3-1.7B with a 16-token reasoning budget is a
   selective quality route. Piper and MeloTTS are the strongest
   Qualcomm-native output routes for their supported languages, Supertonic 3
   is the broadest shared fallback, and standalone LFM-JP is the current
   Japanese TTS quality-evidence leader. Public MeloTTS-ZH HTP-v81 contexts
   now have an account-owned exact-SM8850 compatibility/profile pass at
   144.839 ms for the four neural components with 2,372/2,372 operations on
   HTP; real Mandarin synthesis and equivalence remain open.

An optional Android system-voice route is also staged as a low-risk integration
control: version `0.3.4-input-handoff` can feed completed text clauses to
Android `TextToSpeech` for nine locale choices. It also removes the redundant
native-microphone Base64 decode/re-encode before request submission. A
versioned debug APK has passed local package/signature, embedded-asset, and
direct-handoff validation, but it has not been installed or measured on a
phone. It is not a replacement for the reproducible Qualcomm-native TTS work
or the default `0.3.2` release APK.

The 2-3 second target is architecturally credible but not yet proven end to end
on the phone. The remaining uncertainty is physical:
whether the phone starts its audio device on time without gaps, whether the
chosen locale stack fits the memory envelope, and whether the exact QNN runtime
works on the Nubia. The phone is currently disconnected, so actual playback-
head timing and QNN deployment remain open gates.

The per-language route is summarized in
[`LANGUAGE_ROUTING.md`](LANGUAGE_ROUTING.md). It deliberately stages one ASR,
one text model, and one TTS engine per active locale instead of keeping every
candidate resident. The current accepted phone boundary is documented in the
[Expressive LFM result](../expressive_lfm/REPORT.md).

## What changed in the Android app

The embedded WebView now:

- accepts both the current PCM16 `delta.audio` schema and the older Float32
  `delta.audio_chunk` schema;
- primes the output device from the user's recording tap;
- schedules the first PCM buffer with 140 ms of lead and later buffers on one
  continuous audio timeline;
- reports queue underflows/gaps;
- stops queued sources on barge-in/new recording;
- writes app-private monotonic events for request start, first text, first PCM,
  first-buffer scheduling, output-head result, underflow, error, and completion;
- retains the complete response WAV for replay and as a playback fallback.

The built but phone-unverified `0.3.4-input-handoff` checkpoint additionally
adds a third mode that requests text-only output and streams safe completed
clauses into the phone's Android `TextToSpeech` queue. Auto plus
EN/ZH/ES/JA/KO/DE/IT/FR locale
selection, offline-voice preference, queue cancellation, lifecycle release,
and monotonic TTS telemetry have lightweight source/test coverage. The
4,674,179-byte APK has SHA-256
`f8931486f8f0b749be3ed1174fd219e98929d701808cc9c064050e257b587f9e`;
package/version/signature and packaged-source checks pass. No installed-voice,
phone timing, language-quality, memory, or acoustic result exists yet; see the
[`source validation`](validation_2026-07-23_native_tts_source.md) and
[`APK validation`](validation_2026-07-23_native_tts_apk.md).

Artifact:

- Default APK: `android-app/releases/lfm-audio-demo-0.3.2-turn-telemetry-debug.apk`
- Default SHA-256: `be6c6fdf355d4800a2c569d0bf3d8df04951a5f70e86ccef098190b18de12b98`
- Validation: [`validation_2026-07-23_turn_telemetry.md`](validation_2026-07-23_turn_telemetry.md)
- Optional unverified APK:
  `android-app/releases/lfm-audio-demo-0.3.4-input-handoff-debug.apk`
- Optional APK SHA-256:
  `a1cc2a4ec8bc1342bd3ad3b32001b908e8f50677d7ed7f3e96a0d26857d5b145`
- Retained predecessor: `lfm-audio-demo-0.3.0-streaming-debug.apk`, SHA-256
  `d9ca8c7235e51d7d5b70b4dcf28186c2065191a24e8fc1e2ba35ada9af291e8f`

The UI displays the scheduled WebAudio start and, where supported, the mapped
output-head crossing. The latter is still a rendering-boundary estimate, not
an acoustic speaker measurement. The phone run must retain an external audible-
start measurement.

## Latency evidence

All values are median / P95. Timings start when the complete input waveform is
available, so microphone capture and VAD endpoint delay are excluded.

| Path | Platform | ASR complete | First useful text/chunk | First PCM | Scheduled first audio | Result |
|---|---|---:|---:|---:|---:|---|
| Integrated LFM, original runner | Nubia CPU Q4 | n/a | first text 677 / 701 ms | 800 / 837 ms | not measured | Existing phone evidence proves early audio emission |
| Integrated LFM, streamed queue | Apple M2, Metal-offloaded Q4 | integrated | first text 230 / 278 ms | 307 / 353 ms | 447 / 493 ms | 10/10 runs had no simulated gap at 140 ms |
| Japanese LFM, first interleaved turn | Apple M2, Metal backbone + GPU/CPU audio decoder, Q4 | integrated | first text 214 / 223 ms | 288 / 307 ms | 428 / 447 ms | Fast and coherent on a fresh server; persistent turn 2 is corrupted |
| LFM ASR -> LFM2.5-1.2B -> Supertonic | Apple M2, mixed Metal LFM/LLM + CPU TTS | 372 / 400 ms | TTS text ready 145 / 157 ms after LLM start | 1,013 / 1,073 ms | 1,153 / 1,213 ms | Correct direct English slogan |
| LFM ASR -> Qwen3-1.7B -> Supertonic | Apple M2, mixed Metal LFM/LLM + CPU TTS | 381 / 454 ms | TTS text ready 223 / 229 ms after LLM start | 1,024 / 1,122 ms | 1,164 / 1,262 ms | Correct direct English slogan after prompt hardening |
| Whisper Tiny -> Qwen3-1.7B -> Supertonic, Spanish | Apple M2, CPU ASR/TTS + Metal LLM | 411 / 455 ms | TTS text ready 113 / 234 ms | 881 / 981 ms | 1,021 / 1,121 ms | Correct answer: `PARIS.` |

The integrated Mac run generated 12.22 seconds of audio while the response
completed in 4.80 / 5.81 seconds. Its stream therefore accumulated audio ahead
of playback rather than starving the queue.

A deterministic replay of 20 captured PCM packet timelines gives the current
140 ms cushion 152.4 ms of worst-case one-packet delay headroom and remains
gap-free under a uniform 1.96x post-first-packet slowdown. The retained Nubia
generation pace is about 1.44x slower than the pooled Mac trace. This supports
continuity under average conditions, but it is not a measurement of the
Android playback head, speaker, or thermal throttling. See
[`queue_robustness.md`](queue_robustness.md).

## Why streaming LFM is the first demo path

- It reuses the already-working phone-only model and voice.
- Retained phone evidence puts the first PCM at about 0.8 seconds. Adding the
  tested 140 ms cushion predicts first playback around 0.94-0.98 seconds before
  VAD overhead.
- It avoids loading separate ASR, text, and TTS models for the first demo.
- Full generation latency no longer blocks speaking. A 5-7 second completion
  time can coexist with sub-second first audio if the queue stays fed.

Physical acceptance remains mandatory because WebAudio scheduling, Android
audio-device startup, and speaker output can differ from the deterministic
queue simulation.

A source-level audit confirms that the current input remains complete-WAV and
VAD-gated, while decoded text pieces and PCM stream on output. It also rules
out sending standalone-TTS requests back to the same LFM server: the packaged
runtime has one serial request worker, so the TTS request would wait behind the
active answer. The host cascade now uses a separate warm TTS worker, a bounded
multilingual sentence/clause chunker, ordered synthesis, and explicit queue
backpressure. See
[`LFM_PHONE_PIPELINE_AUDIT.md`](LFM_PHONE_PIPELINE_AUDIT.md).

## Compact multilingual text models

The diagnostic contains 28 unique cases across nine languages and three simple
categories: factual knowledge, arithmetic, and exact instruction following.
The score is the number of case patterns passed, not a statistically powered
quality benchmark. Temperature-zero repetitions measure timing stability and
repeatability; they are not independent samples.

| Model | Q4 file | Passed case checks | First visible token | First speakable text | Retained steady RSS | Verdict |
|---|---:|---:|---:|---:|---:|---|
| LFM2.5-1.2B-Instruct | 664 MiB | 24/28; 72/84 repeats | 20 / 107 ms | 111 / 215 ms | not remeasured in this round | **Default: smaller and fastest** |
| Qwen3-1.7B, reasoning budget 16 | about 1.0 GiB | 25/28; 75/84 repeats | 339 / 405 ms | 509 / 707 ms | about 1,043 MiB in the retained Qwen run | Selective quality route |
| Qwen3-1.7B, thinking off | about 1.0 GiB | 21/28; 63/84 repeats | 26 / 42 ms | 168 / 332 ms | about 1,043 MiB | Reject as the universal default |
| Qwen3-1.7B, unconstrained thinking | about 1.0 GiB | 26/28; one pass | 1,985 / 3,930 ms | 2,094 / 4,094 ms | about 1,043 MiB | Quality arrives too late for voice |
| Qwen3.5-0.8B, thinking off | 537 MiB | 18/28; 54/84 repeats | 55 / 99 ms | 187 / 273 ms | about 1,346 MiB | Reject: quality and memory |

Reasoning budgets 16, 32, and 64 each passed 25/28 cases in their one-pass
screens. Their first-speakable medians were 521, 771, and 1,273 ms,
respectively, so 16 is the current Pareto point. Its three-repeat failures were
systematic: the Chinese Australia-capital case, Japanese spelling of Ottawa,
and the Korean largest-planet case. LFM remains the default because it is much
faster and smaller; Qwen budget 16 is useful where its measured gain matters,
especially the current Italian and Russian fixture cells.

Historical runs did not use an identical warm-up policy: the LFM, Qwen3-1.7B
thinking-off, and Qwen3.5 runs used different warm-up counts, while the finite-
budget stability run used none. Accuracy checks remain directly comparable on
the same suite, but small latency differences across those historical files
must not be treated as a controlled model-only benchmark.

Identical prompts were also submitted consecutively, and llama.cpp reused the
longest common prefix/KV state. For example, LFM first-speakable median was
154.8 ms on the first request of each case and 86.9 ms on later identical
repeats. The scorer usually checks only an expected substring; except for exact
cases, it does not prove that the whole response is correct, concise, or in the
requested language. The finite-budget series now has a separate machine-
readable server manifest; older result JSON records request settings but not
every server-side argument.

Raw evidence:

- [`lfm2_5_1_2b_q4_multilingual_voice_quality.json`](lfm2_5_1_2b_q4_multilingual_voice_quality.json)
- [`qwen3_1_7b_q4_multilingual_voice_quality.json`](qwen3_1_7b_q4_multilingual_voice_quality.json)
- [`qwen3_1_7b_q4_multilingual_voice_quality_thinking.json`](qwen3_1_7b_q4_multilingual_voice_quality_thinking.json)
- [`qwen3_1_7b_q4_reasoning_budget16_stability.json`](qwen3_1_7b_q4_reasoning_budget16_stability.json)
- [`qwen3_1_7b_q4_reasoning_budget32_screen.json`](qwen3_1_7b_q4_reasoning_budget32_screen.json)
- [`qwen3_1_7b_q4_reasoning_budget64_screen.json`](qwen3_1_7b_q4_reasoning_budget64_screen.json)
- [`qwen3_1_7b_finite_reasoning_manifest.json`](qwen3_1_7b_finite_reasoning_manifest.json)
- [`qwen3_5_0_8b_q4_multilingual_voice_quality.json`](qwen3_5_0_8b_q4_multilingual_voice_quality.json)
- [`qwen3_5_0_8b_memory_snapshot_mac.json`](qwen3_5_0_8b_memory_snapshot_mac.json)

## Multilingual ASR

The pinned CC0 diagnostic contains 21 real Common Voice clips: three each for
English, Spanish, German, French, Japanese, Korean, and Mandarin Chinese.

| Model | Fixed-locale median / P95 | Incremental peak RSS | Latin WER | JA/KO/ZH CER | Decision |
|---|---:|---:|---:|---:|---|
| Whisper Tiny INT8 | 217 / 395 ms | 276 MiB | 91.8% | 48.0% | Reject |
| Whisper Base INT8 | 419 / 507 ms | 315 MiB | 36.1% | 39.2% | Integration fallback |
| Whisper Small INT8 | 1,687 / 2,382 ms | 614 MiB | 18.6% | 29.7% | Quality candidate; CPU too slow |

SenseVoiceSmall was then measured on the matching EN/JA/KO/ZH subset: 12 clips,
three timed repeats, two threads, and a fixed locale.

| Model | Fixed-locale median / P95 | Incremental peak RSS | EN WER | JA/KO/ZH CER | Decision |
|---|---:|---:|---:|---:|---|
| SenseVoiceSmall INT8 | 101 / 174 ms | 621 MiB | 15.8% | 24.3% | **Leading EN/JA/KO/ZH recognizer** |
| Whisper Small INT8, same subset | 1,614 / 2,219 ms | 614 MiB | 21.1% | 29.7% | Slower and less accurate on this fixture |

Two streaming Zipformer variants were also screened on the shared three
English plus three Mandarin clips:

| Model | Final median / P95 | Incremental peak RSS | EN WER | ZH CER | Decision |
|---|---:|---:|---:|---:|---|
| Public Qualcomm-family Zipformer INT8 | 251 / 348 ms | 293 MiB | 52.6% | 75.8% | Reject; four empty transcripts |
| X-ASR 480 ms INT8 | 288 / 374 ms | 428 MiB | **15.8%** | 51.5% | Retain only as an English streaming challenger |

X-ASR tied SenseVoice's English WER on this tiny set and emitted English
partials, but the first partials appeared only after 1.6-3.1 seconds of supplied
audio. It decoded the Mandarin `否` clip empty and materially regressed
Mandarin CER, so it is not the shared bilingual route. The detailed gates are
[`zipformer_v81/QUALITY_GATE.md`](zipformer_v81/QUALITY_GATE.md) and
[`zipformer_v81/XASR_480MS_QUALITY_GATE.md`](zipformer_v81/XASR_480MS_QUALITY_GATE.md).

These cells contain only three clips per language and are screening evidence,
not product-quality claims. SenseVoice does not cover Spanish, German, French,
Italian, or Russian. Use a selected/cached locale instead of detecting the
language on every short turn.

Qualcomm's published HTP-v81 scorecards show full-NPU fixed-shape Whisper
Tiny/Base/Small graphs. The published Small float component estimate is about
205 ms for a 20-token utterance and 426 ms for 50 tokens, excluding host work.
Fresh source export/compilation through `qai-hub-models==0.58.0` remains
blocked because the package exposes no release asset for the exporter, but a
different route now passes: the pinned public Galaxy S26 W8A16 contexts ran
unchanged on the exact Android-16 SM8850 QRD. Encoder job `jp8497lx5` reports
**186.277 ms**, **145.727 MiB**, and **2,014/2,014** HTP operations; decoder
job `jgdz9rnr5` reports **3.932 ms per call**, **146.398 MiB**, and
**2,853/2,853** HTP operations. Both have zero fallback. Encoder plus 20
decoder calls is **264.917 ms** of component arithmetic, not end-to-end
transcript latency. The cross-SKU release archive has no matching inputs or
goldens, so numerical/transcript equivalence and phone runtime remain open;
see [`whisper_v81/RESULT.md`](whisper_v81/RESULT.md).

SenseVoice has a more direct exact-target path. Sherpa publishes a fixed
10-second SM8850/HTP-v81 context generated with QNN `2.40.0.251030`, plus a
self-contained arm64 APK containing the matching QNN runtime, ONNX Runtime,
JNI bridge, model, tokens, and VAD. Qualcomm's public QAIRT 2.40 archive also
contains the exact v81 Stub/Skel/backend/System files. Do not mix this context
with the Nubia OEM QNN 2.37 stack or assume QAIRT 2.48 compatibility without a
separate validation run.

The app-level path can therefore be smoke-tested when the phone reconnects.
Controlled CLI profiling still requires building `sherpa-onnx-offline` with
QNN and binaries enabled plus `libonnxruntime.so`; the generic Sherpa Android
release is not QNN-enabled. Raw Qualcomm runtime libraries should not be
vendored standalone in this public repository.

The checksum-pinned ten-second SenseVoice context also passed an account-owned
profile on the exact Snapdragon 8 Elite Gen 5 QRD. Job `j5w1ov24g` reports
**10.871 ms** estimated inference latency, **126.223 MiB** estimated inference
peak memory, and **2,597/2,597** runtime operations on Hexagon/HTP with zero
fallback. AI Hub rejected a matching QAIRT API 2.40 request before job
submission, then accepted the QNN-2.40-generated context under QAIRT 2.45.
This establishes exact-target runtime compatibility and placement, not
transcript equivalence: the release archive has no matching source golden, and
the profile excludes host/pre/post-processing, VAD, transfers, and playback.

Qualcomm's released catalog-float Zipformer context is independently confirmed
as an exact QAIRT-2.45 SM8850/HTP-v81 binary. It keeps floating I/O at FP32,
uses internal FP16 HTP kernels without an AI Hub QuantizeJob, and is distinct
from the separate W8A16/W16A16 mixed route. Its decoder ran on the Android-16
Snapdragon 8 Elite Gen 5 QRD at 0.053 ms with 8/8 operations on Hexagon; the
joiner reached 0.137 ms with 5/5, and the encoder reached 4.741 ms with
2,541/2,541. All had zero fallback. This establishes exact-target compatibility
and placement for every neural component, but not transcript equivalence: the
release archive contains no matching local source/golden outputs. See
[`zipformer_v81/RELEASE_CONTEXT_PROFILE.md`](zipformer_v81/RELEASE_CONTEXT_PROFILE.md).

Full ASR report:
[`multilingual_asr_mac_m2/README.md`](multilingual_asr_mac_m2/README.md).
SenseVoice report and runtime gate:
[`sensevoice_asr_mac_m2/REPORT.md`](sensevoice_asr_mac_m2/REPORT.md) and
[`sensevoice_asr_mac_m2/QNN_RUNTIME_GATE.md`](sensevoice_asr_mac_m2/QNN_RUNTIME_GATE.md).
Exact-target profile:
[`sensevoice_asr_mac_m2/AIHUB_V81_PROFILE.md`](sensevoice_asr_mac_m2/AIHUB_V81_PROFILE.md).

### Specialized Japanese LFM checkpoint

The official pinned `LFM2.5-Audio-1.5B-JP` Q4 bundle materially changes the
Japanese result:

- On Liquid AI's 14.13-second official Japanese sample it achieved 0.0%
  normalized CER in 3/3 measured runs, with 401 / 411 ms first text and
  909 / 920 ms completion.
- The English-checkpoint control on the same sample produced insertion-heavy
  Latin-script gibberish at 504.9% CER and 2,905 ms median completion.
- A first interleaved turn on each freshly initialized server emitted PCM at
  288 / 307 ms. Adding the UI cushion gives a 428 / 447 ms scheduled-start
  estimate.
- Standalone Japanese TTS emitted PCM at 62 / 62 ms and generated a 2.14-second
  waveform in 720 / 721 ms. Three measured outputs were bit-identical.

The current Japanese runner has a release-blocking persistent-turn defect. The
second identical interleaved request emits mixed Chinese/Japanese nonsense and
near-clipped PCM despite `reset_context=true`. ASR repeats correctly and
standalone TTS repeats bit-identically, so use the modular Japanese path while
the interleaved decoder/vocoder reset is investigated. Do not hide the defect by
restarting the model for every product turn.

Full pinned hashes, prompts, PCM statistics, raw JSON, and the repeated-turn
evidence are in
[`lfm_jp_q4_mac_m2/REPORT.md`](lfm_jp_q4_mac_m2/REPORT.md).

## Standalone multilingual TTS

### Supertonic 3

- 31 languages in one model, including EN/ES/DE/FR/IT/JA/KO/RU; no Mandarin.
- 138.6 MiB unpacked model files.
- 137.8-263.5 MiB incremental construction RSS across the retained two-thread
  host runs; about 300 MiB maximum sampled process RSS in the main matrix.
- Recommended local settings: two threads, four denoising steps, speed 1.05.
- About 384 ms median to the first 64-character chunk on the M2 at those
  settings.
- Callback generation allows later chunks to synthesize while the first audio
  is playing.

Full report: [`supertonic3_mac_m2/REPORT.md`](supertonic3_mac_m2/REPORT.md).

Fresh account-owned AI Hub work completed all four neural graphs on Snapdragon
8 Elite Gen 5 QRD, Android 16, SM8850/HTP v81, QAIRT 2.45:

| Component | Latency | Placement | Numerical result |
|---|---:|---:|---|
| Text encoder | 1.046 ms | 462 NPU / 0 CPU | Pass |
| Duration predictor | 0.408 ms | 277 NPU / 0 CPU | Pass |
| Vector estimator, one call | 28.571 ms | 1,111 NPU / 0 CPU | Warning: 2.253% NRMSE |
| Vocoder | 3.153 ms | 212 NPU / 0 CPU | Warning: 16.110% NRMSE |

The often-quoted **118.891 ms** remains arithmetic over fixed-shape component
profiles: `1.046 + 0.408 + 4 x 28.571 + 3.153`. It is a neural-compute planning
sum, not measured text-to-first-audio latency. The shape is batch 1, text length
32, latent `[1,144,32]`, and vocoder output `[1,98304]`; host preprocessing,
duration-derived lengths, loop orchestration, transfers, runtime load, and
playback are excluded. Per-job 117-118 MiB estimates are not additive app RSS.

A separate real-text chain now executed four accumulated vector calls with
`total_step=4`, feeding every remote output into the next stage. The final
remote/fixed-local waveform still fails the strict comparison (NRMSE 0.467,
cosine 0.893), but its spectral convergence is 0.160. Dynamic-local agreement
is much worse because the 31-to-32-position padding/topology change already
alters the waveform before compilation. Most importantly, one fixed-locale
SenseVoiceSmall pass transcribed dynamic-local, fixed-local, and remote audio
identically as `It is Paris in France.` at 0.0 normalized WER.

An inference-only Japanese extension reused the same four compiles for
`パリはフランスにあります。`. Its numerical drift was larger: final vector
NRMSE/cosine 0.0909/0.9959 and remote/fixed-local waveform NRMSE/cosine
0.751/0.728. Even so, one fixed-locale Japanese SenseVoice pass transcribed the
dynamic-local, fixed-local, and remote WAVs identically at 0.0 CER. This adds a
second-language intelligibility sanity check, not general Japanese quality;
prior broader Japanese checks still contain errors.

These are conditional one-phrase feasibility passes, not general quality
acceptance. Human listening, a broader phrase/language matrix, Nubia runtime
integration, and physical playback remain open. Detailed evidence is in
[`supertonic_v81/ACCUMULATED_CHAIN_RESULT.md`](supertonic_v81/ACCUMULATED_CHAIN_RESULT.md),
[`supertonic_v81/ja_paris/RESULT.md`](supertonic_v81/ja_paris/RESULT.md),
and [`aihub_voice_v81/REPORT.md`](aihub_voice_v81/REPORT.md).

### Kokoro v1.1-zh INT8

- English and Mandarin only; the generic espeak data bundled with the archive
  does not make Spanish or Japanese supported languages.
- Apache-2.0 source model card and top-level archive license; bundled frontend
  data still needs a third-party license audit before whole-archive
  redistribution.
- Two-thread M2 first-PCM median/P95: 1.318/1.355 s for the matching English
  short case and 1.073/1.099 s for the first comma-delimited Mandarin clause.
- A longer first clause took 2.080/2.120 s in English and 2.263/2.270 s in
  Mandarin before any ASR, LLM, or playback delay.
- Callback granularity is a completed sentence/clause, not vocoder-frame
  streaming. Aggressive punctuation/time/length flushing is mandatory.
- Critical gate: the original `你好。` warm-up returned 26,400/26,400
  non-finite samples. A follow-up boundary run reproduced invalid PCM on 10/31
  Mandarin calls, including repeated calls after a discarded warm-up. Every
  tested phrase with at least three Han characters was finite in this small
  sweep, but finite-PCM validation remains authoritative.
- Incremental RSS after construction was 377 MiB; maximum sampled process RSS
  was 572 MiB. This is substantially heavier and slower than Supertonic for
  English.
- SenseVoice retranscription gave 0.0% micro WER over three English samples and
  4.0% micro CER over three Mandarin samples. Audio remains human-unverified.
- Every exact PCM hash differed across five repeats because the ONNX graph
  contains unseeded random nodes; duration varied by at most 3.46 ms and none
  of the 30 main six-case outputs clipped or became non-finite. The separate
  short-Mandarin boundary failures remain release-blocking.

Decision: do not replace Supertonic for English. Retain Kokoro only as an
experimental secondary Mandarin CPU candidate behind finite-PCM rejection and
short-chunk merge/fallback logic, followed by a phone comparison against the
Qualcomm-native MeloTTS Chinese route. Full report:
[`kokoro_int8_mac_m2/REPORT.md`](kokoro_int8_mac_m2/REPORT.md).

### Qualcomm-native TTS

Published SM8850-ad/HTP-v81 component evidence reports full NPU placement:

| Model | Languages | Published one-call component sum | Placement | Interpretation |
|---|---|---:|---:|---|
| Piper | EN/DE/IT | about 35.2 ms | 2,090/2,090 NPU layers | Best first Qualcomm-native fallback |
| MeloTTS | ES | about 138.2 ms | 2,681/2,681 NPU layers | Spanish route |
| MeloTTS | ZH | about 140.0 ms | 2,372/2,372 NPU layers | Mandarin route |

These are repeated fixed-shape neural component calls, not text-to-speaker
latency. Tokenization, host orchestration, runtime startup, overlap/repeated
decoding, and audio I/O are excluded.

Pinned source inspection shows that Piper and MeloTTS can expose each decoder
result as roughly 464 ms of new audio instead of waiting for the complete WAV.
The derived decoder supply margins are encouraging, but they remain
source/scorecard evidence rather than phone latency; see the
[`VOICEAI_TTS_STREAMING_GATE.md`](VOICEAI_TTS_STREAMING_GATE.md) report.

Fresh account-owned portability evidence now covers the full public
Piper-English Voice AI package. Its Galaxy S26 `sm8850-ad` contexts ran
unchanged on exact Android-16 plain SM8850:

| Component | Exact QRD latency | Estimated inference peak | Placement |
|---|---:|---:|---:|
| Encoder | 17.014 ms | 119.203 MiB | 401/401 HTP |
| SDP | 6.577 ms | 118.250 MiB | 620/620 HTP |
| Flow | 8.924 ms | 126.465 MiB | 310/310 HTP |
| Audio decoder, one call | 1.822 ms | 117.699 MiB | 117/117 HTP |
| Charsiu encoder, one call | 0.487 ms | 118.031 MiB | 461/461 HTP |
| Charsiu decoder, one output token | 0.280 ms | 118.922 MiB | 180/180 HTP |

The acoustic core is 34.337 ms and 1,448/1,448 HTP operations. One call each
of all six graphs is 35.104 ms and 2,089/2,089 HTP operations, with zero CPU or
unsupported placement; this is within 0.34% of Qualcomm's published component
sum. One audio-decoder payload exposes about 464.4 ms of new audio after host
overlap slicing. The release contains no matching inputs or goldens, so no
compile, inference, numerical-equivalence, or English-quality claim is made.
See [`piper_en_v81/RESULT.md`](piper_en_v81/RESULT.md).

Fresh exact-QRD evidence now covers all four official Piper Italian neural
components:

| Component | Exact QRD latency | Placement | Numerical result |
|---|---:|---:|---|
| Encoder | 16.958 ms | 401/401 NPU | Pass |
| SDP | 6.602 ms | 621/621 NPU | Pass; both outputs exact |
| Flow | 8.930 ms | 310/310 NPU | Pass on valid pipeline fixture; NRMSE 0.000616, cosine 0.9999998 |
| Decoder | 1.821 ms | 117/117 NPU | Pass; cosine 0.9999959 |

The one-call neural sum is 34.311 ms, with 1,449/1,449 runtime layers on NPU
and zero fallback. The first flow test's generic dense-random fixture produced
an all-NaN local golden because its projected log values overflowed Float32
`exp`; that invalid result is retained. An inference-only recheck using the
official serialized encoder, SDP, monotonic attention, and exact same flow
context passed without recompiling, relinking, or re-profiling.

The 34.311 ms sum excludes phonemization/G2P, host orchestration and transfers,
context loading, repeated decoder calls/overlap stitching, and playback. It is
not a complete Piper latency or audible-start measurement.

The corrected production-phonemizer chain now provides stronger end-to-end
graph evidence without changing that latency boundary. Its pinned 79-ID
Italian fixture reused the four linked contexts for encoder, SDP, Flow, and
four decoder calls, producing 35,584 finite samples / 1.613787 seconds. Flow
retains a strict warning because 1 of 294,912 values fails
`allclose(atol=0.03, rtol=0.03)`, although its NRMSE is 0.001437 and cosine is
0.9999990. The final stitched waveform separately passes the same gate with
NRMSE 0.024622, cosine 0.9998499, max absolute error 0.023516, and zero failing
samples. Faster-Whisper Small recovered the exact normalized Italian sentence
from local and exact-QRD WAVs at 0.0 WER/CER. This is a conditional one-phrase
intelligibility pass, not broad listening or phone acceptance. Four decoder
calls bring the profiled full-phrase neural arithmetic to 39.774 ms; the first
40-frame payload is 0.464399 seconds of audio. Full evidence:
[`piper_v81/PRODUCTION_CHAIN_RESULT.md`](piper_v81/PRODUCTION_CHAIN_RESULT.md).

The earlier 34-ID hand-built chain remains invalid for production claims. It
omitted Piper's interspersed pad IDs, so its 0.824-second waveform and
0.464-second first payload prove finite propagation only, not the expected
Italian sentence or a production phrase boundary.

Fresh account-owned exact-QRD compatibility evidence now also covers the four
public QAIRT-2.45 MeloTTS-ZH contexts generated for SM8850-AD:

| Component | Exact QRD latency | Estimated inference peak | Placement |
|---|---:|---:|---:|
| BERT wrapper | 2.679 ms | 118.078 MiB | 332/332 HTP |
| Encoder | 23.697 ms | 125.617 MiB | 830/830 HTP |
| Flow | 74.563 ms | 122.141 MiB | 763/763 HTP |
| Decoder | 43.900 ms | 82.211 MiB | 447/447 HTP |
| **Sum** | **144.839 ms** | not additive | **2,372/2,372 HTP** |

The contexts ran unchanged on the plain-SM8850 Android-16 QRD under QAIRT
2.45, with zero CPU fallback or unsupported operations. This proves
cross-SKU context portability, placement, and per-component profile timing. It
does not prove source-model numerical equivalence, Mandarin intelligibility,
end-to-end text-to-first-audio latency, or Nubia/glasses execution. MeloTTS-ES
still has published same-generation evidence only, and a fresh source export
for either Melo route still requires the Linux/Colab preparation path.

Detailed MeloTTS-ZH evidence:
[`melotts_zh_v81/RESULT.md`](melotts_zh_v81/RESULT.md).

Canonical current AI Hub result:
[`aihub_voice_v81/REPORT.md`](aihub_voice_v81/REPORT.md). The older inventory
note [`../npu/multilingual_tts_v81.md`](../npu/multilingual_tts_v81.md) is kept
for published package context and must not be read as the live job-status page.
The MeloTTS-ZH result is maintained separately because it profiles public
release contexts rather than the source-exported component batch summarized by
`aihub_voice_v81`.

## Memory implication

The modular cascade is fast on the Mac but cannot be declared glasses-ready as
one CPU-resident stack:

- compact text model: hundreds of MiB to about 1 GiB;
- SenseVoiceSmall and Whisper Small: about +621/+614 MiB maximum incremental
  peak in their separate host tests;
- Whisper Base: about +315 MiB maximum incremental peak;
- Supertonic: +137.8 to +263.5 MiB after construction across retained
  two-thread runs;
- Kokoro v1.1-zh: +377 MiB immediately after construction and 572 MiB maximum
  sampled process RSS;
- the Japanese LFM Q4 bundle is about 1.00 GiB on disk and its ready Mac server
  was observed at about 1.44 GiB RSS;
- application/runtime/audio buffers add more.

Therefore multilingual deployment should keep at least ASR and TTS on HTP/NPU,
or use language-specific assets and staged loading. Raw cascade timing alone is
not a memory acceptance result.

The default session policy is one selected/cached locale, one resident ASR,
one resident text model, and one resident output engine. Do not keep LFM2.5 and
Qwen resident merely to switch per turn. The complete per-language policy and
explicitly non-phone planning arithmetic are in
[`LANGUAGE_ROUTING.md`](LANGUAGE_ROUTING.md).

## VAD and interaction policy

The professor confirmed a VAD-gated trigger followed by continuous interaction.
Use this P95 budget after the user stops speaking:

| Segment | P95 budget |
|---|---:|
| VAD endpoint | 300-500 ms |
| Request hand-off / ASR start | 100-300 ms |
| Model work to first PCM | at most 1,700 ms |
| Audio queue and device start | 100-200 ms |
| End of speech to audible response | at most 3,000 ms |

Support barge-in: stop queued response audio when new user speech is accepted.
Use a shorter endpoint after clear sentence-final prosody and a longer timeout
after conjunctions or obviously unfinished phrases.

## Reproduction

Integrated LFM stream trace:

```bash
python3 scripts/profile_streaming_response.py input.wav \
  --mode chat --runs 10 --warmups 1 \
  --output reports/android_phone/streaming_response/lfm_stream.json
```

Modular cascade:

```bash
python3 scripts/profile_cascade_voice_pipeline.py input.wav \
  --tts-model-dir /path/to/supertonic-int8 \
  --threads 2 --steps 4 --language en --flush-chars 48 \
  --runs 10 --warmups 1 --output cascade.json
```

Text diagnostic:

```bash
python3 scripts/profile_text_llm_streaming.py \
  --url http://127.0.0.1:PORT/v1/chat/completions \
  --model MODEL_NAME \
  --suite scripts/multilingual_voice_quality_suite.json \
  --runs 3 --warmups 0 --output result.json
```

Selected Qwen finite-reasoning server settings:

```bash
REASONING=on REASONING_FORMAT=deepseek REASONING_BUDGET=16 \
  scripts/run_qwen3_1_7b_server.sh \
  /path/to/qwen3-1.7b-q4 /path/to/llama.cpp 18084

python3 scripts/profile_text_llm_streaming.py \
  --url http://127.0.0.1:18084/v1/chat/completions \
  --model Qwen3-1.7B-Q4_0 \
  --suite scripts/multilingual_voice_quality_suite.json \
  --thinking on --max-tokens 192 --runs 3 --warmups 0 \
  --output qwen3_1_7b_q4_reasoning_budget16_stability.json
```

The retained run used llama.cpp build 10083 (`846e991ec`) and Qwen model
SHA-256 `c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5`.

UI and APK:

```bash
node scripts/test_streaming_audio_ui.mjs
ANDROID_HOME=/path/to/android-sdk ANDROID_SDK_ROOT=/path/to/android-sdk \
  scripts/build_android_apk.sh
```

The current `0.3.4-input-handoff` source was built successfully on 2026-07-23
with one low-priority Gradle worker after explicitly selecting the installed
Android SDK. The generated APK matches the retained versioned `0.3.4` artifact
byte for byte. The default installer remains pinned to
`0.3.2-turn-telemetry` pending physical-phone acceptance. See
[`NATIVE_INPUT_HANDOFF.md`](NATIVE_INPUT_HANDOFF.md) and
[`validation_2026-07-23_input_handoff_apk.md`](validation_2026-07-23_input_handoff_apk.md).

AI Hub and Colab jobs must be launched inside detached `tmux` sessions. Colab
was not used in this continuation because the stored OAuth state required an
interactive re-authorization; no GPU runtime was allocated.

## Phone acceptance run

When the phone reconnects:

1. Install with `adb install -r` to preserve app-private model files.
2. Run ten fixed-file and ten microphone chat turns.
3. Record VAD close, first delta, first queued buffer, actual playback-head
   movement, completion, underflows/gaps, model/app RSS, and thermal state.
   Pull the app-private trace with
   `scripts/pull_android_streaming_telemetry.py`; its manual input-stop event is
   not a substitute for an acoustic speech-end timestamp.
4. Pass when end-of-speech to audible response is at most 2.0 seconds median and
   3.0 seconds P95, with no audible gap above 80 ms.
5. Separately smoke-test the upstream SenseVoice SM8850 APK, then build the
   pinned QNN-enabled Sherpa CLI for controlled transcript/latency/RSS logging;
   pair the context only with QNN 2.40.
6. If integrated streaming fails, integrate standalone TTS CPU first for a
   functional fallback, then move supported components to HTP v81.

## Licensing and redistribution boundaries

- LFM model/runtime artifacts use LFM Open License v1.0, not an unrestricted
  OSI-style license. Commercial use by a legal entity with annual revenue of
  USD 10,000,000 or more is not licensed by that agreement. Redistribution
  requires the license, modification notices, and applicable attribution/NOTICE
  material; the upstream audio documentation also identifies CC-BY-4.0-derived
  components whose attribution must be retained.
- SenseVoice weights use the custom FunASR Model Open Source License Agreement
  v1.1. Do not describe them as Apache-2.0 or OSI-approved.
- Supertonic model weights carry BigScience OpenRAIL-M terms; the Sherpa runtime
  and packaging have separate licenses.
- Kokoro's source model card and top-level archive license are Apache-2.0, but
  the complete bundled frontend-data notice inventory is unresolved. This
  repository does not redistribute the model archive.
- Qualcomm QAIRT/QNN object code must not be published as a standalone runtime
  bundle from this repository. The exact acquisition/build path is documented,
  while runtime binaries and SDK state remain outside Git.

## Limitations and provenance

- Mac timing is architecture evidence, not phone, NPU, or glasses performance.
- Retained phone first-PCM timing predates the new streaming UI; actual audible
  output is not inferred as a completed measurement.
- Published Galaxy S26 SM8850-ad scorecards are same-generation references, not
  exact QRD/Nubia measurements.
- The quality suites are small diagnostics and cannot establish product-level
  multilingual reliability.
- AI Hub state, local goldens, remote tensors, and full profile/job logs remain
  in operator-owned external work directories. Public summaries retain job IDs
  and metrics but are not independently replayable from a clean clone without
  AI Hub access and the original source artifacts.
- Measured JSON was produced by scripts. This Markdown report was hand-written
  from those JSON files and persisted AI Hub state; numeric evidence was not
  manually changed.
