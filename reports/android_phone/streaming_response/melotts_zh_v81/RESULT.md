# MeloTTS-ZH on SM8850 / HTP v81

## Result

The public QAIRT-2.45 MeloTTS-ZH Voice AI contexts generated for the Samsung
SM8850-AD variant are portable to Qualcomm AI Hub's exact plain-SM8850 target.
Every profiled operation stays on Hexagon/HTP v81.

| Component | Exact-QRD latency | Estimated inference peak | HTP placement |
|---|---:|---:|---:|
| BERT wrapper | 2.679 ms | 118.078 MiB | 332/332 |
| encoder | 23.697 ms | 125.617 MiB | 830/830 |
| flow | 74.563 ms | 122.141 MiB | 763/763 |
| decoder | 43.900 ms | 82.211 MiB | 447/447 |
| **Sum** | **144.839 ms** | not additive | **2,372/2,372** |

There are zero CPU fallback and zero unsupported/unknown operations. The
machine-readable result is in [`aihub_exact_qrd_profiles.json`](aihub_exact_qrd_profiles.json).

## What this changes

MeloTTS-ZH is no longer only a published same-generation candidate. Its
downloadable HTP-v81 contexts have an account-owned exact-QRD compatibility and
placement pass. The fresh 144.839 ms component sum is close to Qualcomm's
published 139.988 ms Samsung Galaxy S26 sum.

For a clause-streamed assistant, the fixed decoder consumes 64 latent frames
with 12-frame overlap and contributes 40 new frames per call. At 512 samples
per frame and 44.1 kHz, each call exposes about 464.4 ms of new audio. The
neural compute therefore leaves substantial room inside the project's
900-ms TTS-stage target after the first speakable text chunk becomes ready.

## Evidence boundary

This result proves:

- SM8850-AD to SM8850 context portability for these pinned files;
- exact-QRD latency and estimated per-component inference peak memory;
- strict Hexagon/HTP placement.

It does not prove:

- numerical equivalence to the source PyTorch model;
- Mandarin intelligibility, naturalness, or speaker quality;
- Nubia or glasses runtime compatibility;
- end-to-end VAD-close to acoustic-onset latency.

The next required experiment is one real Mandarin sentence through
preprocessing, BERT features, encoder, host attention, flow, repeated decoder
calls, overlap slicing, and an ASR plus human-listening backcheck. The phone is
currently disconnected, and Colab requires interactive OAuth reauthorization,
so neither is represented as fresh evidence here.

## Reproduction

1. Fetch the pinned public package described by
   [`release_manifest.json`](release_manifest.json).
2. Run each component through
   `scripts/profile_melotts_release_context.py` from detached tmux.
3. Merge the four saved states with
   `scripts/summarize_melotts_release_contexts.py`.

Cloud jobs:

- BERT wrapper: `jprnqv6k5`;
- encoder: `jprnq3k05`;
- flow: `jpv9okok5`;
- decoder: `jpv9ok475`.
