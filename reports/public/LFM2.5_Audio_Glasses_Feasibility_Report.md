# LFM2.5-Audio on AI Glasses: Feasibility, Quality, and Baseline Decision

**Public technical report - July 16, 2026**

## Executive decision

Proceed with LFM2.5-Audio as the integrated research baseline, but do not yet call it glasses-ready. The tested fixed-shape neural partitions compile and execute entirely on the Hexagon NPU of a Qualcomm QCS8550 proxy. The full BF16 model is functionally healthy on an NVIDIA L4, including persistent two-turn speech interaction. The remaining blockers are target-hardware verification, target-QNN resident memory, a production cache interface, and a numerically safe output-audio path.

Run a second, modular baseline in parallel as deployment insurance: Moonshine Base ASR -> Qwen3-0.6B in non-thinking mode -> Pocket TTS. If Moonshine or Pocket export stalls, use the Qualcomm-native Zipformer/Piper components as the conservative fallback. This challenger is not yet a quality winner; it is the best footprint and toolchain control.

## Product mode

The confirmed mode is VAD-gated activation followed by a continuous conversational session. VAD lowers idle compute, but it does not bound memory during an active session. The runtime therefore needs a deliberate context reset, summarization, or eviction policy.

## Recommended LFM partition

```text
Microphone -> VAD -> FastConformer on NPU -> LFM backbone on NPU
           -> depth/RQ decoder on NPU -> FP32 detokenizer outside FP16 HTP path -> speaker
```

The FP32 detokenizer placement is a proposed partition. CPU-host latency and memory have not yet been measured on the glasses.

## What was measured

- Eight official-weight fixed-shape PT2 graphs exported and reloaded locally.
- Strict QNN HTP FP16 placement, component latency, tool-reported peak memory, and numerical comparison on QCS8550 (Proxy), Android 12, Hexagon v73.
- Full official BF16 ASR and two-turn interleaved speech interaction on one NVIDIA L4.
- A small matched ASR diagnostic under clean, 10 dB white noise, and 5 dB competing speech conditions.
- Repeated official Q4_0 GGUF inference with FP16, Q8_0, and Q4_0 KV caches plus flash-attention on/off controls on Apple M2 Metal.
- Static checkpoint and cache accounting.

No exact AR1, AR1+, or unambiguous 5100 target was available in this AI Hub account. Proxy latency is not AR1 latency.

## Qualcomm proxy component results

| Component | NPU / CPU | Latency | Peak | Numerical result |
|---|---:|---:|---:|---|
| FastConformer + adapter | 1215 / 0 | 4.786 ms | 87.0 MiB | mismatch |
| Conv layer prefill | 36 / 0 | 2.693 ms | 92.2 MiB | passed |
| Attention layer prefill | 62 / 0 | 2.444 ms | 91.5 MiB | passed |
| Conv cached decode | 32 / 0 | 2.695 ms | 91.9 MiB | passed |
| Attention cached decode | 61 / 0 | 2.452 ms | 87.0 MiB | passed |
| Depth/RQ decoder | 2953 / 0 | 25.540 ms | 92.0 MiB | passed |
| Detokenizer neural T=4 | 371 / 0 | 1.880 ms | 88.0 MiB | mismatch |
| Detokenizer neural T=8 | 371 / 0 | 2.577 ms | 87.4 MiB | mismatch |

All tested strict graphs placed with zero CPU fallback on the proxy. This is not a claim that the complete model is fully NPU placed. The full 16-layer backbone, orchestration, complete waveform reconstruction, VAD, and audio I/O were not submitted as one graph.

## Quality findings

The FastConformer proxy output has cosine similarity 0.998424 and NRMSE 0.06976 against the local golden. All ten frames preserve their nearest temporal identity. A frozen downstream LFM check preserves 16 of 17 top-1 choices; the only change is the semantically similar choice `seems` -> `looks`. The golden choice remains in the remote top five at every step. This is encouraging but proves that feature cosine alone is not an acceptance test.

The strict-NPU depth decoder returns exact audio code tokens. By contrast, the FP16 NPU detokenizer is numerically unusable despite full placement. A repeat with the first eight real generated audio frames from turn 1 gives waveform cosine 0.0052, NRMSE 1.010, and SI-SDR -45.74 dB. The synthetic-probe failure was therefore not an input artifact. The report keeps the detokenizer outside the first FP16 HTP partition.

### Small matched ASR diagnostic

WER is normalized, lower is better. This 18-utterance subset is a smoke diagnostic, not a publication-scale benchmark.

| Model | Clean | White noise 10 dB | Competing speech 5 dB |
|---|---:|---:|---:|
| LFM2.5-Audio | 2.86% | 6.43% | 55.71% |
| Whisper Tiny | 11.43% | 21.67% | 71.90% |
| Whisper Base | 8.81% | 20.71% | 55.00% |
| Whisper Small | 7.14% | 10.71% | 31.43% |
| Moonshine Base | 8.57% | 14.52% | 64.05% |

On this small test, LFM is strongest on clean speech and 10 dB white noise, while Whisper Small is clearly strongest under the synthetic competing-speaker condition. Moonshine is a footprint challenger, not a demonstrated multi-speaker quality winner.

### Generated-audio intelligibility proxy

A frozen Whisper Small retranscription gives 0.00% WER for turn 1 and 21.43% WER for turn 2 against LFM's own text stream. This measures audio-text consistency, not naturalness or MOS.

## Memory and context

The pinned BF16 runtime checkpoint set occupies 3.390 GiB in static files. The complete local Q4 GGUF bundle occupies 1.001 GiB, a 3.39x smaller package-to-package comparison. The formats and backends differ; neither number is total resident application memory.

The LFM backbone has six attention layers and ten convolution layers. The idealized FP16 KV payload is 48 MiB at 4,096 positions, 96 MiB at 8,192, 192 MiB at 16,384, and 384 MiB at 32,768; idealized INT8 payload halves those values. Actual runtime buffers can differ because of scales and alignment. The convolution cache is small and fixed; the attention cache is the active-session growth risk.

## Local Q4 inference-technique matrix

The official Q4_0 GGUF bundle was repeated on two official audio samples using Apple M2 Metal. This is a two-sample functional sanity check, not Qualcomm QNN/AR1 evidence or a general ASR benchmark. CLI model time excludes model load; wall time includes process startup and file-cache effects. Five-run p95 values are descriptive.

Across 40 runs, 40/40 normalized transcripts match their references exactly.

| KV cache | Flash | Sample | Exact | WER med/p95 | KV MiB | Encode ms med/p95 | Gen tok/s med/p95 | Model ms med/p95 | Wall ms med/p95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | off | asr | 5/5 | 0.00 / 0.00% | 48.00 | 449.0 / 629.8 | 24.3 / 25.6 | 3425.8 / 3555.5 | 4650.0 / 4668.0 |
| FP16 | off | question | 5/5 | 0.00 / 0.00% | 48.00 | 140.0 / 298.4 | 29.1 / 57.4 | 816.9 / 1590.3 | 1620.0 / 2528.0 |
| FP16 | on | asr | 5/5 | 0.00 / 0.00% | 48.00 | 439.0 / 592.4 | 23.2 / 26.4 | 3389.3 / 3855.5 | 4170.0 / 5066.0 |
| FP16 | on | question | 5/5 | 0.00 / 0.00% | 48.00 | 172.0 / 196.8 | 28.6 / 38.8 | 891.4 / 1096.6 | 2110.0 / 2536.0 |
| Q4_0 | on | asr | 5/5 | 0.00 / 0.00% | 13.50 | 413.0 / 476.8 | 25.7 / 26.1 | 3122.0 / 3285.7 | 4150.0 / 4150.0 |
| Q4_0 | on | question | 5/5 | 0.00 / 0.00% | 13.50 | 139.0 / 148.8 | 32.6 / 48.6 | 738.1 / 904.6 | 1630.0 / 2422.0 |
| Q8_0 | on | asr | 5/5 | 0.00 / 0.00% | 25.50 | 486.0 / 780.4 | 26.5 / 28.1 | 3254.8 / 3780.5 | 4160.0 / 4638.0 |
| Q8_0 | on | question | 5/5 | 0.00 / 0.00% | 25.50 | 137.0 / 195.4 | 41.7 / 67.7 | 627.5 / 982.5 | 2100.0 / 2192.0 |

The runner reports Metal fallback hotspots in the audio encoder (`CONV_2D_DW`, `ROLL`, `UNARY`). These are Apple Metal implementation gaps, not QNN CPU-fallback evidence. No post-training-quantized LFM QNN graph has yet been profiled on QCS8550 or the physical glasses.

## Additional Qualcomm speech profiles

Qualcomm's `qai-hub-models==0.57.3` scorecards provide a useful deployment control on the same QCS8550 proxy. Every component below reports full NPU placement.

| Model | Encoder / main stage | Recurrent stage | Interpretation |
|---|---:|---:|---|
| Zipformer | 8.895 ms per 0.71 s chunk | decoder 0.075 ms; joiner 0.187 ms | Strong streaming fallback |
| Whisper Tiny | 25.307 ms | 2.459 ms per decoder call | Smallest Whisper control |
| Whisper Base | 47.978 ms | 4.202 ms per decoder call | Middle footprint/quality control |
| Whisper Small FP16 | 130.318 ms | 12.074 ms per decoder call | Best overlap robustness in the small quality test |
| Whisper Small W8A16 | 376.895 ms | 7.856 ms per decoder call | Lower decoder cost, but 2.9x slower encoder |
| Piper TTS | encoder 30.344 ms; flow 15.189 ms | decoder 3.018 ms per chunk | Conservative output-audio fallback |

These are per-component fixed-shape scorecard values, not end-to-end pipeline latency. Whisper Small W8A16 makes the encoder 2.89x slower while making the decoder about 35% faster, so quantization must be profiled stage by stage.

## Candidate decision

| Candidate | Role | Weight view | Deployment evidence | Decision |
|---|---|---:|---|---|
| LFM2.5-Audio-1.5B | Integrated research baseline | Q4 bundle 1.001 GiB locally | Strongest measured partition evidence; detokenizer blocker | Continue |
| Moonshine Base + Qwen3-0.6B + Pocket TTS | Best next footprint challenger | Estimated 0.55-0.66 GB weights | Qwen has official Qualcomm package; ASR/TTS export pending | Profile next |
| Zipformer + Qwen3-0.6B + Piper TTS | Conservative toolchain fallback | About 0.84 GB from Qualcomm component size cards | Zipformer and Piper are fully NPU placed on QCS8550 cards; Qwen QCS8550 result absent | Demo insurance |
| Mini-Omni | End-to-end scientific control | About 976M total estimated | Custom scheduler and SNAC export risk | Later control |
| LLaMA-Omni2-0.5B | Misleading size label | Public checkpoint alone 3.857 GB BF16 | Multiple external components | Exclude |

## Exact next steps

1. Confirm the physical glasses SoC, available app RAM, QNN/Voice AI SDK version, and deployment interface.
2. Run the modular challenger export in this order: Qwen3-0.6B, Moonshine Base encoder/decoder, then Pocket TTS host path. Keep Zipformer/Piper ready as swaps.
3. Build four-layer LFM backbone shards and a production cached decoder with explicit position/mask handling and a bounded context policy.
4. Convert and profile target-QNN quantized encoder/backbone/depth variants separately. Validate downstream tokens/transcripts after every precision change.
5. Measure the official FP32 detokenizer on the intended host CPU/DSP path. Do not move it to FP16 HTP until real-code audio fidelity passes.
6. Integrate VAD and measure microphone-end to first audible PCM, total process RSS, 30-minute temperature, and battery drop on the real glasses.

## Acceptance gates for a physical-glasses demo

- Provisional model-process peak at or below 1.3 GB, to be revised after confirming app-available RAM and device reserve.
- No unsupported main-backbone operation and no undeclared CPU fallback in NPU shards.
- Quantized ASR WER increase no more than 0.3 percentage points on each chosen evaluation split.
- First audible PCM within 500 ms after VAD close on the actual device.
- Two-turn context retention no more than five points below the BF16 golden test set.
- No NaN/Inf, systematic truncation, or material audio-text consistency regression.

## Limitations

- No exact AR1/AR1+/5100 AI Hub target was available.
- Component latency and memory cannot be summed into an end-to-end estimate.
- Backbone placement covers representative official layers and fixed cache probes, not one complete compiled network.
- The small quality suite uses synthetic noise and a dummy LibriSpeech subset, not glasses microphone recordings.
- Host FP32 detokenizer viability remains proposed, not measured on target.
- The Q4 matrix uses two official samples on Apple M2 Metal; it does not establish general quality, Qualcomm placement, or glasses memory.
- LFM weights use the LFM Open License v1.0, including a commercial-use revenue threshold; downstream components carry their own licenses and attribution duties.

## Reproducibility

- Model: `LiquidAI/LFM2.5-Audio-1.5B` revision `c362a0625dfe45aa588dce5f0ada28a7e5707628`.
- Q4 GGUF: `LiquidAI/LFM2.5-Audio-1.5B-GGUF` revision `7d525f883a077e20afb782f2ff618edcae0e39e4`; runner build 7641 (`68d8edf2`), Apple M2 Metal.
- Liquid Audio source commit: `19e65845923a7f136442c95137884ec61eb386aa`.
- QCS8550 proxy: Android 12, Hexagon v73, QNN HTP FP16.
- Qualcomm comparator catalog: `qai-hub-models==0.57.3`, QAIRT 2.45.0 scorecards.
- Full golden: PyTorch 2.8.0, Transformers 4.56.1, NVIDIA L4 BF16.

## References

1. Liquid AI, LFM2.5-Audio model card: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B
2. Liquid Audio source: https://github.com/Liquid4All/liquid-audio
3. Qualcomm AI Hub model catalog: https://aihub.qualcomm.com/models/
4. Qualcomm Zipformer: https://aihub.qualcomm.com/models/zipformer
5. Qualcomm Whisper Small: https://aihub.qualcomm.com/models/whisper_small
6. Qualcomm PiperTTS English: https://aihub.qualcomm.com/models/pipertts_en
7. Moonshine Base: https://huggingface.co/UsefulSensors/moonshine-base
8. Moonshine paper: https://arxiv.org/abs/2410.15608
9. Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
10. Qualcomm Qwen3-0.6B: https://aihub.qualcomm.com/models/qwen3_0_6b
11. Pocket TTS: https://github.com/kyutai-labs/pocket-tts
12. Mini-Omni: https://arxiv.org/abs/2408.16725
13. LLaMA-Omni2: https://arxiv.org/abs/2505.02625
