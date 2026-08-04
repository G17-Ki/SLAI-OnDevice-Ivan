# Real-time voice response decision

> Condensed current decision. [`REPORT.md`](REPORT.md) is the canonical full
> multilingual comparison and contains the complete evidence boundaries.
>
> **Superseded for Mandarin and Russian by
> [`tts_research_20260726/REPORT.md`](tts_research_20260726/REPORT.md)**
> (2026-07-26). That run disqualified eSpeak by measurement, closed the
> MeloTTS-ZH exact-QRD chain, settled the Supertonic four-versus-five-step
> question in favour of five under the 1–2 second budget (four only under a
> strict <0.9 second gate), and produced a private blinded listening bundle.
> The English Expressive-LFM decision below is unchanged.

Last updated: 2026-07-23<br>
Target: first audible response within 2–3 seconds after VAD closes<br>
Current phone: Nubia `NX809J`, `SM8850`, Android 16

## Decision

Use **streamed LFM audio as the primary demo path**. The phone runner already
emits its first audio chunk far before the full response is complete; the long
delay came from buffering all chunks in the UI before playback. The updated UI
queues each PCM delta immediately with a 140 ms startup cushion and retains the
complete-WAV path only as a fallback.

Keep **ASR → compact text model → standalone TTS** as the multilingual
fallback. The current LFM-ASR → LFM2.5-1.2B → Supertonic cascade reached
scheduled first audio in 1.153 seconds median / 1.213 seconds P95 on the Mac;
the bounded Qwen3-1.7B route reached 1.164 / 1.262 seconds. These prove the
orchestration margin, not phone or end-to-end multilingual acceptance.

Do not wait for the full text response in either architecture. Start playback
from native LFM PCM immediately, or flush a standalone-TTS clause at a sentence
boundary, useful punctuation boundary, or a bounded 50–70 character limit.

## What was actually measured

| Path | Evidence platform | First useful event | First PCM / scheduled audible | Full response | Verdict |
|---|---|---:|---:|---:|---|
| Integrated LFM, existing phone runner | Nubia CPU Q4 | first text 677 ms median / 701 ms P95 | first PCM 800 ms / 837 ms | 6.79 s / 6.88 s | Model already streams early; UI buffering was the main user-visible delay |
| Integrated LFM with playback-queue simulation | Apple M2, Metal-offloaded Q4 | first text 230 ms / 278 ms | first PCM 307 ms / 353 ms; scheduled playback 447 ms / 493 ms | 4.80 s / 5.81 s | Current ten-run trace; no simulated underflow or gap at a 140 ms cushion |
| LFM ASR → LFM2.5-1.2B → Supertonic 3, four steps | Apple M2, mixed Metal + CPU | first speakable text 111 ms / 215 ms after ASR | scheduled first audio 1,153 ms / 1,213 ms | full response continues after playback starts | Historical strict-latency configuration; superseded for the 1–2 s budget |
| LFM ASR → Qwen3-1.7B budgeted route → Supertonic 3 | Apple M2, mixed Metal + CPU | first speakable text 509 ms / 707 ms after ASR | scheduled first audio 1,164 ms / 1,262 ms | full response continues after playback starts | Selective quality route; unconstrained thinking is rejected for latency |

Values are median / P95. All timings begin after the complete input waveform is
available, so microphone capture and VAD endpoint delay are excluded.

## Why streaming LFM is first

- The phone already delivered an audio delta at about 0.8 seconds while total
  generation took about 6.8 seconds. Replacing the model cannot recover as much
  time as simply playing that delta.
- On the Mac, the first LFM PCM packet arrived about 4.4 seconds before the
  response completed. The stream stayed ahead of playback with only a 140 ms
  initial queue.
- It preserves LFM's integrated speech behavior, avoids a second model and its
  memory, and needs no text segmentation or voice hand-off.
- The Android UI accepts both the older `delta.audio_chunk` float32 schema and
  the newer `delta.audio` PCM16 schema used by the current upstream runner.

The remaining proof is physical-phone playback: install the rebuilt APK, make
ten microphone turns, and measure playback-head movement rather than only the
JavaScript scheduling timestamp.

## Multilingual fallback

Supertonic 3 INT8 currently has the best coverage/latency balance among the
standalone models tested locally:

- 31 supported languages, including English, Spanish, German, Italian,
  Japanese, Korean, Russian, French, Portuguese, Hindi, Arabic, and Ukrainian;
- 138 MiB of ONNX weights;
- short-sentence first callback between 0.35 and 0.54 seconds median across six
  tested languages on the M2 at five denoising steps;
- the later 15-repeat study selected five steps: 0.529 s P95 to the first short
  chunk with better measured intelligibility than four; four remains relevant
  only under a strict <0.9 s device gate;
- chunk callbacks let later text synthesize while the first chunk is playing.

Mandarin Chinese is not supported. Keep MeloTTS-ZH or another Chinese-specific
engine for that case. Piper remains a strong Qualcomm-native option for
English, German, and Italian, while MeloTTS covers Spanish and Chinese. Fresh
exact-QRD Piper English release-context profiles place all six QNN graphs on
HTP: 35.104 ms for one call each and 2,089/2,089 operations, including a
34.337 ms acoustic core. The release has no matching inputs or goldens, so this
is placement/portability evidence rather than English audio quality. Fresh
exact-QRD Piper Italian measurements now cover all four neural components:
34.311 ms for one encoder + SDP + flow + decoder call, with all 1,449 reported
runtime layers on NPU and valid component numerical gates. The corrected
79-ID production chain adds a strict final-waveform pass and conditional exact
one-phrase Italian ASR pass. It still has a one-element Flow warning and no
physical-phone audible-onset measurement.

Standalone TTS solves only output speech. A multilingual assistant also needs:

1. multilingual ASR;
2. a text model that answers accurately in the detected language;
3. language propagation into TTS;
4. a policy for mixed-language turns and unsupported languages.

The default compact text route is now LFM2.5-1.2B-Instruct Q4: it scored
72/84 on the deterministic multilingual diagnostic and reached 111/215 ms
median/P95 to first speakable text. Qwen3-1.7B with thinking disabled scored
63/84 and is not the default; a bounded reasoning budget is retained only for
selected Italian/Russian quality cases. Unconstrained thinking is rejected
because its first-text tail already exceeds the complete voice budget.

## Exact HTP v81 work

Fixed-shape Supertonic INT8 bundles were generated for:

| Component | Fixed shape | Source size | Current exact-QRD state |
|---|---|---:|---|
| Duration predictor | text `[1,32]` | 3.53 MiB | **pass**: 0.408 ms, 117.30 MiB, 277/277 NPU; compile `jp2v287x5`, profile `j5qmjoymg`, inference `jp3wy29z5` |
| Text encoder | text `[1,32]` → embedding `[1,256,32]` | 34.73 MiB | **passed**: 1.046 ms, 117.84 MiB peak, 462/462 runtime layers on NPU; compile `j5mdj79d5`, profile `jpxdwv1jg`, inference `jp84lo3z5` |
| Vector estimator | latent `[1,144,32]`, text `[1,256,32]` | 74.77 MiB | **placement pass / numerical warning**: 28.571 ms, 118.05 MiB, 1,111/1,111 NPU; compile `j5w1o2k6g`, profile `jpxdvnq1g`, inference `j5mdrq7w5` |
| Vocoder | latent `[1,144,32]` → 98,304 samples | 24.79 MiB | **placement pass / numerical warning**: 3.153 ms, 117.81 MiB, 212/212 NPU; corrected compile `jgo4ykxx5`, profile `jgjwxz9x5`, inference `jpel9eq1g` |

Target for every job is `Snapdragon 8 Elite Gen 5 QRD`, Android 16, `SM8850`,
HTP v81, QAIRT 2.45. These are synthetic but deterministic component inputs;
the remote output is compared with an ONNX Runtime golden. A successful
component profile is not an end-to-end Supertonic deployment: text processing,
duration-derived dynamic latent lengths, denoising-loop orchestration, context
binary packaging, and the Nubia's older vendor runtime still need integration.

The fixed-shape four-step neural sum is 118.891 ms. The text encoder and
duration predictor pass their saved 5% comparisons. The vector estimator and
vocoder place fully on NPU but retain explicit strict numerical warnings, so
placement/latency is not promoted to audio-quality acceptance.

Official Qualcomm package evidence remains useful in parallel:

- Piper EN/DE/IT published v81 component sum: about 35.2 ms per one-call set;
- MeloTTS ES/ZH published v81 component sum: about 138–140 ms;
- these are repeated fixed-shape neural calls, not text-to-first-audio latency.

The public Piper-English Galaxy S26 `sm8850-ad` contexts now also have an
account-owned exact plain-SM8850 portability profile. Encoder, SDP, flow, and
one audio-decoder call total 34.337 ms and 1,448/1,448 HTP operations. One
Charsiu encoder plus one autoregressive decoder-token call adds 0.767 ms; all
six graphs total 35.104 ms and 2,089/2,089 HTP operations, with zero fallback.
The exact and published one-call sums differ by only 0.34%. The 64-frame audio
decoder exposes about 464.4 ms of new 22.05 kHz audio after host overlap
slicing, supporting decoder-payload streaming. Because the package includes no
matching input/golden, this run did not submit inference and makes no numerical
or intelligibility claim. See [`piper_en_v81/RESULT.md`](piper_en_v81/RESULT.md).

Account-owned exact-target Piper Italian evidence is now stronger than the
published proxy: encoder 16.958 ms, SDP 6.602 ms, flow 8.930 ms, and decoder
1.821 ms on the Android-16 SM8850 QRD. Their 34.311 ms sum excludes
phonemization/G2P, host orchestration and transfers, context loading, repeated
decoder overlap calls, and playback. The flow numerical gate passes on a valid
official-pipeline fixture; the earlier all-NaN dense-random golden is retained
as invalid-fixture provenance.

The corrected production fixture then reused those same contexts for 139
latent frames and four decoder calls. It produced 1.613787 seconds of finite
audio. Flow has one failing value among 294,912 at the strict 3%/3% gate, but
the stitched waveform passes with zero failing samples, NRMSE 0.024622, and
cosine 0.9998499. Faster-Whisper Small recovered the exact normalized Italian
sentence from local and remote WAVs at 0.0 WER/CER. The profiled arithmetic is
34.311 ms through the first decoder call and 39.774 ms through all four; the
first 40-frame payload contains 0.464399 seconds of audio. These remain neural
and payload facts rather than text-to-speaker latency. See
[`piper_v81/PRODUCTION_CHAIN_RESULT.md`](piper_v81/PRODUCTION_CHAIN_RESULT.md).

Inference-only accumulated Supertonic chains now provide conditional
one-phrase intelligibility passes in English and Japanese: a fixed-locale ASR
judge recovered each reference exactly from dynamic-local, fixed-local, and
remote audio. Both chains still have large late-stage waveform drift and are
not human-quality or broad-language acceptance.

A two-decoder Piper chain also completed, but its hand-built 34-ID fixture
omitted Piper's required interspersed pad tokens. Its finite 0.824-second audio
and 0.464-second first decoder output prove low-level propagation only, not
Italian intelligibility or a production phrase boundary. It remains preserved
as invalid-fixture provenance and is not substituted for the corrected 79-ID
result above.

## VAD budget

The professor confirmed a VAD-gated interaction followed by continuous audio
interaction. Keep the response target explicit:

| Segment | P95 budget |
|---|---:|
| VAD endpoint after the user stops | 300–500 ms |
| Request hand-off and first model/ASR work | 100–300 ms |
| Generation to first PCM | at most 1,700 ms |
| Audio queue and playback start | 100–200 ms |
| End of speech to audible response | at most 3,000 ms |

For continuous post-activation interaction, use a short endpoint for a clear
utterance and a slightly longer continuation timeout after conjunctions or
unfinished prosody. Support barge-in by stopping queued response sources as
soon as new user speech is accepted.

## Reproduction

Integrated LFM stream timing:

```bash
python3 scripts/profile_streaming_response.py input.wav \
  --mode chat --runs 10 --warmups 1 \
  --output reports/android_phone/streaming_response/lfm_stream.json
```

Standalone cascade timing:

```bash
python3 scripts/profile_cascade_voice_pipeline.py input.wav \
  --tts-model-dir /path/to/supertonic-int8 \
  --threads 2 --steps 4 --language en --flush-chars 64 \
  --runs 10 --warmups 1 --output cascade.json
```

Supertonic fixed-shape AI Hub run:

```bash
python3 scripts/prepare_supertonic_aihub.py \
  --model-dir /path/to/supertonic-int8 --output-dir /persistent/fixed-io

tmux new-session -d -s aihub-supertonic-v81 \
  "python3 scripts/run_aihub_supertonic_v81.py \
    --model-dir /path/to/supertonic-int8 \
    --io-dir /persistent/fixed-io --output-dir /persistent/v81-run"
```

Streaming UI verification:

```bash
node scripts/test_streaming_audio_ui.mjs
scripts/build_android_apk.sh
```

## Next acceptance run

1. Install `0.3.4-input-handoff` explicitly on the Nubia without clearing app
   data; retain `0.3.2` as the rollback/control.
2. Run ten fixed-file and ten microphone turns with the integrated LFM path.
3. Record native WAV readiness, JavaScript WAV readiness, request start, first
   delta received, first buffer queued, first `AudioTrack`/WebAudio playback-
   head movement, completion, gaps, thermal state, and incremental RSS.
4. Pass if end-of-VAD to audible response is at most 2.0 seconds median and
   3.0 seconds P95, with no audible gap above 80 ms.
5. Only if integrated streaming fails that gate, integrate Supertonic CPU first;
   use two vs four threads and four vs five steps, while LFM remains resident.
6. Add multilingual ASR and a stronger compact text model before calling the
   standalone cascade multilingual end to end.

## Evidence files

- `mac_lfm_streaming_restart_2026-07-22.json`
- `lfm2_5_1_2b_supertonic3_cascade_direct_answer_mac.json`
- `qwen3_1_7b_supertonic3_cascade_answerfirst_mac.json`
- `supertonic3_mac_m2/REPORT.md`
- `supertonic_v81/ACCUMULATED_CHAIN_RESULT.md`
- `supertonic_v81/ja_paris/RESULT.md`
- `piper_en_v81/RESULT.md`
- `piper_v81/PROGRESS.md`
- `NATIVE_INPUT_HANDOFF.md`
- `validation_2026-07-23_input_handoff_apk.md`
- `validation_2026-07-23_turn_telemetry.md`
- `../native_profile/20260716t120118z/REPORT.md`
- `../npu/multilingual_tts_v81.md`

This report was hand-written from script-generated JSON and persisted AI Hub
state. No measured value was manually altered.
