# Zipformer EN/ZH CPU quality gate

Date: 2026-07-23 CST<br>
Decision: **reject the public Sherpa INT8 route for the phone demo**

This is a bounded host comparison of the public INT8 ONNX export from the same
`pfluo/k2fsa-zipformer-chinese-english-mixed` checkpoint family used by the
official Qualcomm model. It is not an inference run of Qualcomm's distinct
float/FP16 compiled release and does not replace the
[exact-QRD release-context profile](RELEASE_CONTEXT_PROFILE.md).

Machine-readable evidence:
[`zipformer_en_zh_gate_result.json`](zipformer_en_zh_gate_result.json).

## Setup

- Model revision:
  `pfluo/k2fsa-zipformer-chinese-english-mixed@6eb615ae77ecac05c5628d5c8ed7037c14a338d5`.
- Runtime: Sherpa-ONNX 1.13.4, Apple M2 CPU, two threads, greedy streaming
  transducer.
- Model files: INT8 encoder and joiner plus FP32 decoder, about 93.2 MiB total.
- Fixture: the same three English and three Mandarin Common Voice clips already
  used by the SenseVoice and Whisper comparison.
- Protocol: one warm-up, three timed repetitions per clip. MP3 decode occurred
  before timing. The timed region included stream creation, waveform input,
  660 ms zero tail, final decode, and result retrieval. A separate replay fed
  100 ms audio increments to observe first partial output.

Model checksums:

| File | Bytes | SHA-256 |
|---|---:|---|
| INT8 encoder | 80,530,659 | `e92df89d84ae5cb1571a2489c85d5d385c31fb8022892a240bd8c4c301b3ecc8` |
| FP32 decoder | 13,876,468 | `d91b63c84f27f1d822244ced4cc2b9f7fb8a60a328f13b860408b8889dd0a0ed` |
| INT8 joiner | 3,259,918 | `16e901969495d5704a27a929e396a2ff9cc6fbfdeafdc6418e8c88b370113af6` |
| Tokens | 56,317 | `a8e0e4ec53810e433789b54a5c0134a7eaa2ffca595a6334d54c00da858841d3` |

## Result

| Model | EN WER | ZH CER | Same-six final latency median/P95 | Incremental peak RSS |
|---|---:|---:|---:|---:|
| SenseVoiceSmall INT8 | 15.8% | 33.3% | 125/177 ms | 621 MiB |
| Whisper Small | 21.1% | 48.5% | 1,598/2,158 ms | 614 MiB |
| **Zipformer public INT8** | **52.6%** | **75.8%** | **251/348 ms** | **293 MiB** |

Zipformer had attractive compute behavior: median/P95 real-time factor was
0.0427/0.0489, load time was 922 ms, and all three repetitions were
deterministic. Quality failed decisively:

| Clip | Reference | Zipformer output | First nonempty partial |
|---|---|---|---:|
| EN short | `six` | empty | none |
| EN short | `She'll be all right.` | empty | none |
| EN long | Joe Keaton sentence | `KITAN DISAPPROVED OF FILMS AND BUSTER ALSO HAD PRESERVATIONS ABOUT THE MEDIA` | after 2.1 s of supplied audio |
| ZH short | `否` | empty | none |
| ZH medium | `宋朝末年年间定居粉岭围。` | empty | none |
| ZH long | `黑身准裂腹鱼为辐鳍鱼纲鲤形目鲤科的其中一种。` | `准列负于为负几于刚离行木理科的其中一种` | after 2.1 s of supplied audio |

Eleven model-bundled test WAVs produced nonempty transcripts, so the failure is
not a broken model/runtime configuration. It is a domain/quality failure on the
shared demo fixture, amplified by the cheap public INT8 conversion.

## Decision boundary

Do not integrate this CPU INT8 model. A Qualcomm exact-QRD Zipformer route is
worth continuing only if its approved local/remote golden passes all of these
gates:

- no empty transcript on any of the six shared clips;
- EN WER no worse than SenseVoice: at most 3/19 word edits;
- ZH CER no worse than SenseVoice: at most 11/33 character edits;
- identical normalized transcript across three repetitions;
- streaming real-time factor P95 at most 0.10;
- first partial by 1.5 seconds of audio for every utterance longer than two
  seconds;
- Mac incremental RSS at most 350 MiB for a compact fallback;
- exact-phone/QNN transcript equivalence and zero CPU/unknown placement;
- VAD-close-to-final-ASR at most 300 ms median and 700 ms P95 on the phone.

The newer Sherpa X-ASR 480 ms checkpoint was screened next. It tied SenseVoice
on the three English clips but regressed Mandarin to 51.5% CER and decoded the
one-character `否` clip empty. Retain it only as an English streaming
challenger; reject it as the shared EN/ZH route. See
[`XASR_480MS_QUALITY_GATE.md`](XASR_480MS_QUALITY_GATE.md).
