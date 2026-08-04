# Nubia deployment experiment batch — memory gate

Date: 2026-07-17

## Decision

The previously unexplained memory is now attributed. CPU weight repacking is
the largest avoidable allocation, followed by fixed compute buffers. Disabling
repacking reduces short-input active RSS by approximately 538 MiB and preserves
the established ASR diagnostic quality, but the longest 29.4-second diagnostic
utterance still raises model-process `VmHWM` to 1365.3 MiB.

No tested configuration satisfies all three requirements simultaneously:

1. model-process peak at or below the stated 1.3 GB gate;
2. WER increase no greater than 0.3 percentage points on every diagnostic split;
3. acceptable latency for the interactive baseline.

The closest quality-preserving configuration is `-c 512 --no-repack`. The
configurations below 1.3 GiB use smaller micro-batches and/or Q4 KV, but every
one exceeds the allowed competing-speech WER delta. Therefore the CPU Q4
deployment is a useful phone demo and research baseline, but the 1.3 GB product
gate is **not accepted as passed**.

The gate must specify units. `1.3 GB` decimal is 1239.8 MiB; `1.3 GiB` is
1331.2 MiB. The best quality-preserving result exceeds both. Some rejected
quality configurations fit under 1.3 GiB but none fits under 1.3 GB decimal.

## Phase 0 — device and accelerator record

- Device: Nubia NX809J, `SM8850`, Android 16 / SDK 36.
- RAM: 15,600,544 KiB, approximately 14.88 GiB.
- ABI: ARM64 only.
- CPU: eight cores in a 6+2 arrangement. CPUs 0–5 report capacity 741 and a
  3.6288 GHz maximum; CPUs 6–7 report capacity 1024 and a 4.7424 GHz maximum.
- HTP generation: v81, positively identified from `libQnnHtpV81.so`, the v81
  skeleton, and the v81 stub present on the phone.
- The APK contains no QNN runtime and its model process uses all CPU cores. The
  current deployment remains CPU-only despite the phone's HTP v81 capability.
- Instrumented APK: version `0.2.0-profile`. It reads optional private
  `runtime.args` for controlled experiments; normal operation is unchanged when
  that file is absent.

Primary record:
`reports/android_phone/device/20260717t052107z/`.

Post-experiment battery/thermal record:
`reports/android_phone/device/20260717t055657z/`.

## Phase 1 — memory attribution

One interleaved chat request at the original 4096 context produced:

| Capture | Model RSS | Anonymous | GGUF-backed | VmHWM |
|---|---:|---:|---:|---:|
| Idle | 1690.8 MiB | 983.0 MiB | 697.6 MiB | 1690.3 MiB |
| Active peak | 1749.2 MiB | 1041.2 MiB | 697.6 MiB | 1748.2 MiB |

The active `smaps` table sums to the reported process residency. File-backed
pages are dominated by:

| GGUF | Active RSS |
|---|---:|
| Main LFM backbone | 652.3 MiB |
| Audio detokenizer/tokenizer model | 45.4 MiB |

The encoder and vocoder loaders do not retain separately identifiable
file-backed GGUF mappings in `smaps`; their runtime allocations are included in
anonymous residency. Exact per-component separation for those two files is not
exposed by this runner.

The retained load log reports these major allocations:

| Runtime allocation | Declared size |
|---|---:|
| Main-model CPU repack | 555.75 MiB |
| Detokenizer CPU repack | 19.27 MiB |
| Main compute buffer | 132.00 MiB |
| Detokenizer compute buffer | 131.50 MiB |
| Audio-encoder compute buffer | 195.19 MiB |
| Main KV at context 4096, FP16 | 48.00 MiB |
| Detokenizer sliding-window KV | 2.25 MiB |

Declared buffers need not all be simultaneously resident, so their sum is not
expected to equal `smaps` RSS exactly. Nevertheless, the log and `smaps` agree
on the conclusion: the former ~800 MiB gap is primarily repacked weights and
compute geometry, not unexplained static model files.

Full evidence:
`reports/android_phone/memory_attribution/20260717t052135z/`.

## Context and audio-length sweeps

All context sizes from 512 through 8192 transcribed both official samples
exactly. On the short 4.9-second sample, active RSS ranged from 1702.1 MiB at
context 512 to 1795.2 MiB at context 8192. Reducing the original context 4096
to 512 saved only 42.9 MiB at active peak.

The 18.4-second sample increased active residency substantially:

| Context | Short active RSS | 18.4 s active RSS | Exact ASR |
|---:|---:|---:|---|
| 512 | 1702.1 MiB | 1827.3 MiB | yes |
| 1024 | 1708.9 MiB | 1834.3 MiB | yes |
| 2048 | 1723.5 MiB | 1847.9 MiB | yes |
| 4096 | 1745.0 MiB | 1870.6 MiB | yes |
| 8192 | 1795.2 MiB | 1920.9 MiB | yes |

This confirms that encoder activation memory scales with audio length and is a
product-relevant peak, even though it is released or partially released after
the request.

Artifacts:

- Short context sweep: `memory_matrix/20260717t052254z_context/`.
- Long context sweep: `memory_matrix/20260717t052359z_context/`.

## Phase 2 — memory reduction results

### Disable CPU repacking

At context 512:

| Workload | Repack active RSS | No-repack active RSS | Repack total | No-repack total |
|---|---:|---:|---:|---:|
| Short ASR | 1702.4 MiB | 1164.2 MiB | 493 ms | 816 ms |
| 18.4 s ASR | 1827.4 MiB | 1279.8 MiB | 1.64 s | 2.52 s |
| Interleaved chat | 1705.8 MiB | 1163.0 MiB | 3.19 s | 5.59 s |

The no-repack chat run produced a valid but different assistant response and
first audio moved from 339 ms to 1218 ms. These are single ordered runs, not a
latency distribution, but the direction and magnitude make the tradeoff clear.

Artifacts:

- Short ASR: `memory_matrix/20260717t052530z_repack/`.
- Long ASR: `memory_matrix/20260717t052621z_repack/`.
- Interleaved chat: `memory_matrix/20260717t052710z_repack/`.

### Batch geometry

On the 18.4-second sample, decreasing micro-batch size reduced active RSS from
1280.0 to 1240.2 MiB, but increased total time from 2.49 to 4.29 seconds. The
audio encoder's 195.19 MiB compute buffer did not change; only the two LFM
compute buffers shrank.

Artifact: `memory_matrix/20260717t052829z_batch/`.

### KV quantization

At context 512, main + detokenizer KV allocation falls from 7.50 MiB in FP16
to 2.11 MiB in Q4_0. This is real but too small to determine the overall gate;
sampled RSS variation was of similar magnitude.

Artifact: `memory_matrix/20260717t052945z_kv/`.

## Full 18-utterance quality gate

The matched phone corpus contains 18 clean utterances and deterministic 10 dB
Gaussian and 5 dB competing-speech variants. There were zero request failures.

Baseline versus quality-preserving no-repack:

| Configuration | VmHWM | Clean WER | Gaussian WER | Competing WER |
|---|---:|---:|---:|---:|
| Repack, context 512 | 1919.3 MiB | 3.57% | 7.14% | 54.29% |
| No-repack, context 512 | 1365.3 MiB | 3.57% | 6.90% | 54.29% |

No-repack WER deltas are 0.00, −0.24, and 0.00 percentage points, passing the
quality rule on every split. Its median total request times are 2.33–2.42 s,
versus 0.80–1.20 s with repacking.

Artifact: `asr_diagnostic/20260717t053446z/`.

Configurations pushed below 1.3 GiB or close to it changed competing-speech
behavior beyond the 0.3 pp allowance:

| Configuration | VmHWM | Clean delta | Gaussian delta | Competing delta | Result |
|---|---:|---:|---:|---:|---|
| ubatch 16, Q4 KV | 1302.9 MiB | −0.71 pp | 0.00 pp | +1.67 pp | reject |
| ubatch 16, FP16 KV | 1307.7 MiB | 0.00 pp | 0.00 pp | +0.48 pp | reject |
| ubatch 32, FP16 KV | 1309.8 MiB | 0.00 pp | 0.00 pp | +1.43 pp | reject |
| ubatch 128, Q4 KV | 1320.2 MiB | −0.48 pp | 0.00 pp | +1.90 pp | reject |

Artifacts:

- ubatch 16 + Q4 KV: `asr_diagnostic/20260717t054128z/`.
- ubatch 16 + FP16 KV: `asr_diagnostic/20260717t054432z/`.
- ubatch 32 + FP16 KV: `asr_diagnostic/20260717t054802z/`.
- ubatch 128 + Q4 KV: `asr_diagnostic/20260717t055156z/`.

## Thermal and power caveat

The phone remained AC powered at 100%, so no battery-drain result is valid.
After the repeated matrices and 324 diagnostic requests, the thermal service
reported status 0 and current CPU readings around 45 °C, but cached CPU values
reached 99.6 °C and CPU/GPU cooling devices had nonzero states. This was not a
controlled 30-minute time series, so it is warning evidence rather than the
Phase 4.3 sustained-throttling result.

## Task status

- Phase 0.1 device identity: complete.
- Phase 0.2 QNN/HTP scan: complete; HTP v81 identified.
- Phase 0.3 load log: complete and retained.
- Phase 1.1 mapping attribution: complete.
- Phase 1.2 per-component residency: main and detokenizer complete; encoder and
  vocoder remain jointly represented in anonymous loader allocations.
- Phase 1.3 context sweep: complete.
- Phase 1.4 audio-length scaling: complete for 4.9, 18.4, and 29.4 seconds.
- Phase 1.5 idle residency: complete.
- Phase 2.1 minimum tested context: complete for single-turn ASR/chat; multi-turn
  retention at context 512 remains untested.
- Phase 2.2 KV quantization: complete for the memory/quality screening pass.
- Phase 2.3 repack comparison: complete. mmap/mlock comparison remains open.
- Phase 2.4 batch geometry: complete.
- Phase 4.7 on-device 18-utterance diagnostic: complete.

## Recommended next work

1. Seek a selective or lower-memory repack strategy. Plain `--no-repack`
   preserves quality but is approximately 34 MiB above 1.3 GiB on the full
   diagnostic. A modest 35–100 MiB saving that does not change evaluation
   geometry would be more valuable than further KV work.
2. Measure staged encoder/detokenizer loading. The encoder has a fixed 195 MiB
   compute buffer and audio-length-dependent peaks; lifecycle changes may save
   memory without changing numerical execution.
3. Complete QAIRT support analysis for the phone's HTP v81 and run AI Hub
   device discovery. Q4 GGUF remains a CPU format; the NPU path is a new export.
4. Run the unplugged 30-minute power/thermal experiment before making any
   sustained-use claim.
5. Continue the modular challenger in parallel. The CPU baseline is now close
   to the memory gate, but latency and thermal behavior remain poor for glasses.
