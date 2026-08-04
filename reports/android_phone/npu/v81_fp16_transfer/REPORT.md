# HTP v81 FP16 transfer matrix

Date: 2026-07-17  
Target: `Snapdragon 8 Elite Gen 5 QRD`, Android 16, SM8850, HTP v81  
v73 baseline: `QCS8550 (Proxy)`, Android 12, Hexagon v73; submitted 2026-07-15  
Runtime: strict QNN FP16 context binary, current Qualcomm AI Hub toolchain

## Verdict

All eight fixed-shape components compile and execute entirely on HTP v81 after
the depth decoder's required int64 boundary correction. There is no CPU or
unspecified runtime fallback in any successful profile. The four backbone
probes pass their existing float tolerances, and the depth/RQ decoder retains
exact token IDs plus a passing next-embedding comparison.

The v73 numerical failure pattern also reproduces: FastConformer remains outside
the tensor tolerance, and both FP16 detokenizer probes remain numerically wrong.
The v81 result therefore supports the encoder/backbone/depth partition, but it
does not change the V1 decision to keep detokenization in FP32 on the host.

The QRD is a Qualcomm reference device. It is not a retail device, it is not the
Nubia phone, and these are not measurements from the physical glasses.

## v73 baseline versus v81

Placement is reported as AI Hub runtime-layer counts in `NPU/CPU/other` order.
Float acceptance uses `atol=1e-3`, `rtol=1e-3`; integer tokens require exact
equality. Peak memory is the AI Hub estimated inference peak for one fixed-shape
component invocation.

| Component | v73 placement | v73 latency | v73 peak | v73 numerics | v81 placement | v81 latency | v81 peak | v81 numerics |
|---|---:|---:|---:|---|---:|---:|---:|---|
| FastConformer + adapter | 1215/0/0 | 4.786 ms | 87.004 MiB | mismatch | 1215/0/0 | 3.185 ms | 117.555 MiB | mismatch; max abs 0.080966, cosine 0.998423 |
| Backbone conv prefill | 36/0/0 | 2.693 ms | 92.242 MiB | pass | 36/0/0 | 1.833 ms | 117.691 MiB | pass |
| Backbone attention prefill | 62/0/0 | 2.444 ms | 91.547 MiB | pass | 62/0/0 | 1.667 ms | 117.723 MiB | pass |
| Backbone conv cached decode | 32/0/0 | 2.695 ms | 91.883 MiB | pass | 32/0/0 | 1.831 ms | 117.520 MiB | pass |
| Backbone attention cached decode | 61/0/0 | 2.452 ms | 87.008 MiB | pass | 61/0/0 | 1.673 ms | 117.836 MiB | pass |
| Depth/RQ decoder | 2953/0/0 | 25.540 ms | 91.980 MiB | tokens exact; embedding pass | 2953/0/0 | 16.778 ms | 117.801 MiB | tokens exact; embedding pass, max abs 0.0000763 |
| Detokenizer T=4 | 371/0/0 | 1.880 ms | 87.984 MiB | mismatch | 371/0/0 | 1.284 ms | 117.855 MiB | mismatch; log-abs cosine 0.751307, angle cosine 0.263420 |
| Detokenizer T=8 | 371/0/0 | 2.577 ms | 87.395 MiB | mismatch | 371/0/0 | 1.372 ms | 117.977 MiB | mismatch; log-abs cosine 0.849379, angle cosine 0.278111 |

**Mandatory comparison caveat:** v73 numbers were compiled by the AI Hub toolchain
current at their submission date; v81 numbers use the current toolchain.
Differences reflect silicon generation AND toolchain version jointly; neither is
isolated.

Within that limitation, the v81 fixed-shape latencies are 1.46x to 1.88x lower
than the v73 values, while the reported peaks are 25.45 to 30.83 MiB higher.
Neither change should be attributed to silicon alone. The stable numerical
classes are more important for this transfer gate: backbone/depth pass, whereas
FastConformer and both detokenizer probes mismatch on both generations.

## Jobs

All primary compiles were submitted before polling. After that compile stage,
all available profile jobs and all available inference jobs were submitted
before their polling stage.

| Component | Compile | Profile | Inference | Result |
|---|---|---|---|---|
| FastConformer + adapter | `jgz4yon4p` | `jp84r0jz5` | `jp84r0rz5` | success |
| Backbone conv prefill | `j5w1z244g` | `jgo4wkd45` | `j5qm1e17g` | success |
| Backbone attention prefill | `jg9x2jdmg` | `j5w1z2yzg` | `jgl1868e5` | success |
| Backbone conv cached decode | `jp1v1y6np` | `j5mdkq9y5` | `j56dmemvp` | success |
| Backbone attention cached decode | `j577n09n5` | `jgn7qlqvp` | `jp3w7v7x5` | success |
| Depth/RQ decoder, initial | `jpxdrnx8g` | - | - | compile failed |
| Depth/RQ decoder, corrected | `j5w1z2n6g` | `j577nl7n5` | `jp494d925` | success |
| Detokenizer T=4 | `j5mdkq875` | `jp2vd0dx5` | `jgo4wkw45` | success, numerical mismatch |
| Detokenizer T=8 | `jgn7qlkjp` | `jp0v9392g` | `jpv9m0m75` | success, numerical mismatch |

The initial depth compile failed with this exact AI Hub message:

```text
Must use --truncate_64bit_io when output tensors have type int64.
```

The corrected retry added `--truncate_64bit_io`. The returned token tensor is
int32 at the QNN boundary, remains inside range, and exactly equals the int64
golden values.

## Method and evidence

The existing remote PT2 source models and already-uploaded input datasets were
reused because direct upload of the local PT2/NPZ artifacts was rejected by the
execution environment; the exact rejection is retained in
`raw/20260717t070923z/direct-upload-blocked.txt`. No graph was re-exported.
`scripts/run_v81_aihub_batch.py` submitted and polled the jobs at five-minute
intervals, downloaded every available server log, classified placement, and
compared remote outputs with the existing local NPZ goldens.

Primary script-generated evidence:

- `raw/20260717t070923z/` - seven successful components plus the preserved
  initial depth failure.
- `raw/20260717t074742z-depth-retry/` - corrected depth compile, profile,
  inference, placement, and numerical comparison.
- v73 source: `reports/aihub_component_results.json`,
  `reports/aihub_component_results.md`, and `reports/operator_matrix.csv`.

This `REPORT.md` was hand-written from those archived raw results. No measured
numeric value was edited. Publication cleanup redacted absolute local path
strings in archived JSON evidence; measured values are unchanged. The original
batch launcher expanded its shell return code variable too early, so
`batch-returncode.txt` is blank; the Python batch
itself completed with `finished_utc=2026-07-17T08:07:19Z` and produced all seven
summaries. The corrected depth launcher recorded return code 0.

## Scope boundary

These are isolated, fixed-shape component calls. They do not include transfers,
cache management across the full backbone, VAD, microphone/audio I/O, host FP32
detokenization, or end-to-end response latency. The result is an NPU transfer
gate, not a complete phone or glasses deployment.
