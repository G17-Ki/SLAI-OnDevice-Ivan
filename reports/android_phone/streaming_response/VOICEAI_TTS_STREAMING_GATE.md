# Qualcomm Voice AI TTS streaming gate

Date: 2026-07-23<br>
Evidence: pinned `qai-hub-models==0.58.0` reference source plus its published
Samsung Galaxy S26 / HTP-v81 scorecards

## Decision

Piper and MeloTTS do not need to wait for the complete waveform at the app
boundary. Their fixed decoder call consumes 64 latent frames (40 new frames
plus 12 frames of overlap on each side) and yields **464.4 ms of new audio**.
The current Qualcomm Python reference apps concatenate those decoder results
and write one WAV at the end, but that is an app choice rather than a graph
constraint. A phone integration can enqueue the first decoder result
immediately and continue producing later chunks while it plays.

This makes **Piper the strongest standalone low-risk streaming fallback** for
English, German, and Italian. MeloTTS remains the Qualcomm-native choice for
Spanish and Mandarin. Neither replaces the already-working streamed integrated
LFM path for the first English demo.

## Source-derived bounds

The values below are not text-to-speaker measurements. They combine fixed graph
boundaries from the pinned public source with published same-generation v81
component latencies.

| Route | Core neural work before first decoder output | Decoder compute | New audio per decoder call | Decoder supply rate |
|---|---:|---:|---:|---:|
| Piper EN | 34.455 ms | 1.827 ms | 464.399 ms | 254.2x real time |
| Piper DE | 34.433 ms | 1.822 ms | 464.399 ms | 254.9x real time |
| Piper IT | 34.499 ms | 1.821 ms | 464.399 ms | 255.0x real time |
| MeloTTS ES | 137.458 ms | 42.433 ms | 464.399 ms | 10.94x real time |
| MeloTTS ZH | 137.299 ms | 42.440 ms | 464.399 ms | 10.94x real time |

“Core” is encoder + duration work when present + flow + the first decoder call.
Piper phonemization is outside that sum. The public package also exposes a
CharSiU encoder plus a 0.260-0.281 ms autoregressive decoder call; its call
count depends on the text. Melo Spanish similarly adds a 0.488 ms T5 encoder
and 0.267 ms per generated G2P token. Melo Chinese adds one published 2.689 ms
BERT-wrapper call.

Even Melo's slower decoder creates about 464 ms of audio in about 42 ms of
neural compute, leaving roughly an 11x raw-compute continuity margin. This does
not include transfers or runtime scheduling, but it shows that decoder compute
itself should not cause playback starvation on the v81 generation.

## Fresh exact-QRD Piper result

Account-owned measurements on `Snapdragon 8 Elite Gen 5 QRD`, Android 16,
SM8850/HTP v81, QAIRT 2.45 now cover the complete Italian neural core:

| Component | Latency | Placement | Numerical gate |
|---|---:|---:|---|
| Encoder | 16.958 ms | 401/401 NPU | Pass |
| SDP | 6.602 ms | 621/621 NPU | Pass |
| Flow | 8.930 ms | 310/310 NPU | Pass on valid pipeline fixture |
| Decoder | 1.821 ms | 117/117 NPU | Pass |

The sum is **34.311 ms**, with 1,449/1,449 runtime layers on NPU and zero
fallback. The flow's generic random fixture was invalid because all projected
log values overflowed Float32 `exp`, making the entire local output NaN. A
fully finite inference-only recheck reused the exact linked flow context and
passed at NRMSE 0.000615842, cosine 0.999999812, and max absolute error
0.028513. No graph stage was repeated.

This strengthens the compute-cadence case but remains a one-call neural sum.
It excludes phonemization/G2P, host orchestration and transfers, context load,
repeated decoder calls/overlap slicing, and audio-device scheduling.

## What must change in the app

The reference implementations currently do this:

1. run encoder/duration and flow for the selected text chunk;
2. call the decoder repeatedly with 12-frame overlap;
3. concatenate every returned waveform slice;
4. write the complete WAV.

The streaming integration should instead:

1. keep text chunks short and speakable (sentence boundary first, then a
   bounded fallback);
2. run the first three neural stages once for that text chunk;
3. enqueue the first 464 ms decoder result immediately;
4. continue decoder calls and remove only the overlap already handled by the
   reference slicing rule;
5. start the next text chunk before the current audio queue empties;
6. retain finite-PCM, underflow, barge-in, and complete-WAV fallback checks.

## Acceptance status

- **Source boundary:** pass. The fixed decoder interface is chunkable.
- **Same-generation compute cadence:** pass. Every listed component is
  published as fully NPU placed, and every decoder is substantially faster
  than its audio payload duration.
- **Exact QRD neural core:** pass. Encoder, SDP, flow, and decoder all pass
  strict placement and valid numerical gates; their one-call sum is 34.311 ms.
- **Exact QRD text-to-first-decoder pipeline:** open. Phonemization/G2P, host
  orchestration, context load, and real repeated decoder scheduling are not in
  the component sum.
- **Phone playback:** open. No physical Nubia playback-head measurement exists
  for Piper or Melo.
- **Melo exact-QRD numerics:** open. Its Linux/Colab source export is still
  authorization-gated.

Machine-readable derivation:
[`voiceai_tts_streaming_bounds.json`](voiceai_tts_streaming_bounds.json).

Reproduce in an isolated QAIHM 0.58 environment without loading model weights
(the repository's older main AI Hub extra is intentionally not reused):

```bash
python3.11 -m venv .venv-qaihm058-audit
.venv-qaihm058-audit/bin/python -m pip install 'qai-hub-models==0.58.0'
.venv-qaihm058-audit/bin/python scripts/analyze_voiceai_tts_streaming.py \
  --output reports/android_phone/streaming_response/voiceai_tts_streaming_bounds.json
```

The script fails if the installed `qai-hub-models` version differs from 0.58.0,
parses the decoder constants directly from package source, records source and
inventory hashes, and labels the result as a derived bound rather than a phone
benchmark.
