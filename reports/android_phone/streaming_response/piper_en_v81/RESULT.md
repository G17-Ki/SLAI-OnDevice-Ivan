# Piper English release contexts on SM8850 / HTP v81

## Result

Qualcomm's public Piper-English Voice AI contexts generated for the Samsung
Galaxy S26 `sm8850-ad` variant run unchanged on AI Hub's exact Android-16
plain-SM8850 QRD. Every operation in all six QNN graphs stays on Hexagon/HTP
v81.

| Component | Exact-QRD latency | Estimated inference peak | HTP placement | Galaxy S26 proxy |
|---|---:|---:|---:|---:|
| encoder | 17.014 ms | 119.203 MiB | 401/401 | 17.056 ms |
| SDP | 6.577 ms | 118.250 MiB | 620/620 | 6.599 ms |
| flow | 8.924 ms | 126.465 MiB | 310/310 | 8.973 ms |
| audio decoder, one call | 1.822 ms | 117.699 MiB | 117/117 | 1.827 ms |
| **Acoustic core** | **34.337 ms** | not additive | **1,448/1,448** | **34.455 ms** |
| Charsiu encoder, one call | 0.487 ms | 118.031 MiB | 461/461 | 0.489 ms |
| Charsiu decoder, one output token | 0.280 ms | 118.922 MiB | 180/180 | 0.281 ms |
| **All six, one call each** | **35.104 ms** | not additive | **2,089/2,089** | **35.225 ms** |

There are **zero CPU fallback** and **zero unsupported, export-failed, or
unknown-placement operations**. The exact-target all-six one-call sum is
0.121 ms, or 0.34%, faster than the six published Galaxy S26 component rows.

The archive also contains `bert_normalizer.bin`. Qualcomm's metadata describes
it as a supplementary optimized-Unicode binary, not a QNN graph. It therefore
has no AI Hub operator-placement report and must not be mislabeled as either an
NPU operation or CPU fallback.

Machine-readable summary:
[`aihub_exact_qrd_profiles.json`](aihub_exact_qrd_profiles.json). Raw execution
summaries and per-operation logs are intentionally excluded from the public
tree.

## Latency interpretation

The **34.337 ms acoustic-core sum** is the useful fixed neural boundary once
phoneme IDs and host attention are ready. It includes one encoder, one duration
predictor, one flow, and one audio-decoder call. It is not text-to-audio
latency.

The audio decoder consumes 64 latent frames and emits 16,384 samples. The
existing host slicing retains 40 new frames after 12-frame overlap on both
sides, exposing 10,240 new samples, or about **464.4 ms of 22.05 kHz audio**.
At 1.822 ms of decoder compute per call, that boundary produces new audio about
255 times faster than real time. A phone implementation can enqueue the first
decoder payload immediately instead of waiting for the complete waveform.

English preprocessing has two supported shapes:

- the public Charsiu route adds one 0.487 ms encoder call plus **0.280 ms for
  every generated G2P output token**;
- the Python reference route uses host `piper_phonemize`, bypassing the two
  Charsiu QNN graphs.

Consequently, 35.104 ms is only a one-call arithmetic sanity check. Full
latency also includes Unicode normalization, tokenization, the complete G2P
decode loop or host phonemizer, host attention construction, transfers,
repeated 64-frame audio-decoder calls, overlap slicing, queueing, the Android
audio output head, and acoustic onset.

Per-component estimated inference peaks span 117.699-126.465 MiB and are not
additive. First and warm context-load times span 82.725-100.072 ms. For the
low-latency demo, contexts should be initialized before a turn rather than
loaded after the first text chunk arrives. Aggregate resident-set memory still
requires a real phone measurement.

## Decision

Piper-English is a strong standalone English TTS fallback for the phone demo:
the published HTP-v81 package is exactly-target portable, fully NPU placed, and
its acoustic arithmetic is comfortably below the project's TTS-stage budget.
This result supports implementing decoder-chunk streaming; it does not yet
justify replacing integrated LFM streaming as the primary path because no
English quality or phone acoustic-onset measurement was run here.

The next phone experiment should compare host `piper_phonemize` against the
Charsiu route on fixed English clauses, keep all required contexts resident,
enqueue each 464.4 ms payload as soon as it is available, and measure:

1. text-chunk ready to first PCM queued;
2. first PCM queued to Android output-head callback;
3. text-chunk ready to physical acoustic onset;
4. inter-chunk gaps, process RSS, and thermals;
5. ASR backcheck plus human naturalness/listening acceptance.

## Evidence boundary

This experiment proves:

- portability of the pinned Galaxy S26 `sm8850-ad` contexts to exact plain
  `sm8850`;
- exact-QRD steady-state latency and estimated per-component inference peak;
- strict HTP placement under QAIRT 2.45.

It does not prove:

- numerical equivalence, English intelligibility, or naturalness;
- end-to-end text-to-first-audio or VAD-close-to-audio latency;
- Nubia or glasses deployment/runtime compatibility;
- full autoregressive G2P or repeated audio-decoder cost;
- physical playback behavior, RSS, thermals, or always-on stability.

No AI Hub compile or inference job was submitted. The release contains no
matching sample inputs or golden outputs, so arbitrary-tensor inference would
not provide a valid correctness gate. The catalog labels the release
`precision: float`, while its Voice AI config also says
`is_model_quantized: true`; this run therefore makes no claim about actual
per-operation tensor precision.

## Reproduction

Fetch the pinned public 0.58.0 package from detached `tmux`:

```bash
qai-hub-models fetch pipertts_en \
  --runtime voice_ai \
  --precision float \
  --chipset qualcomm-snapdragon-8-elite-gen5-for-galaxy \
  --sdk-version qairt=2.45 \
  --version 0.58.0 \
  --output-dir /path/to/output
```

Verify the ZIP is 69,259,115 bytes with SHA-256
`950f20398da2d0c3549b8ee5cc73ce53e4e093cbe430015d143ed3988483f1d6`,
then run [`profile_piper_en_release_context.py`](../../../../scripts/profile_piper_en_release_context.py)
inside a detached `tmux` session. The script validates the archive, metadata,
supplementary assets, and every graph hash; persists model and profile IDs
before polling; uploads the existing contexts unchanged; and refuses to
submit compile or inference work.

Cloud jobs:

- encoder model `mqpre7gom`, profile `jgk9nlz25`;
- SDP model `mmdw12doq`, profile `j5qmk784g`;
- flow model `mnw4kdyrq`, profile `jgo401yx5`;
- audio decoder model `mn463pw0q`, profile `jpv9o13j5`;
- Charsiu encoder model `mmxk6wprn`, profile `jg9x131lg`;
- Charsiu decoder model `mnl094wkn`, profile `jp1vldl2p`.
