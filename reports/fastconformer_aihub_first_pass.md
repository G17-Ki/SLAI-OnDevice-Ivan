# FastConformer Qualcomm AI Hub first pass

## Target and interpretation

- AI Hub does not expose Snapdragon AR1, AR1+, or an unambiguous 5100 target.
- This first pass uses `QCS8550 (Proxy)`, Android 12, Hexagon v73, QNN, HTP FP16.
- Placement compatibility is useful evidence for the AR1 investigation. The
  latency and memory figures are proxy measurements, not AR1 predictions.

## Strict NPU result

| Field | Result |
|---|---:|
| Compile job | `jpxdly8jg` - success |
| Profile job | `j577exzq5` - success |
| Inference job | `jp1vn82lp` - success |
| NPU runtime layers | 1,215 |
| CPU fallback runtime layers | 0 |
| Other/unknown runtime layers | 0 |
| Median/estimated inference latency | 4.786 ms |
| Peak target memory | 87.004 MB |
| Output shape | `[1,10,2048]` - correct |
| Output finite | yes |

The fixed FastConformer + adapter graph therefore compiles and executes as a
fully NPU-placed QNN context binary on the v73 proxy.

## Remote output versus local FP32 golden

| Metric | Result |
|---|---:|
| Cosine similarity | 0.99842447 |
| RMSE | 0.01423536 |
| Golden RMS | 0.20406491 |
| Normalized RMSE | 0.06975897 |
| Mean absolute error | 0.01069085 |
| 95th-percentile absolute error | 0.02868927 |
| 99th-percentile absolute error | 0.03971733 |
| Maximum absolute error | 0.08194214 |

The original `atol=1e-3, rtol=1e-3` allclose gate fails. This is not a shape,
NaN, placement, or execution failure; it is FP16 numerical drift accumulated
through the full encoder. Acceptance should be decided through downstream
token/transcript equivalence or a calibrated component tolerance, not by
silently widening the threshold.

## Next checks

1. Complete the ONNX Runtime + QNN diagnostic pass.
2. Run the remaining component graphs with strict-NPU-first classification.
3. Feed the remote encoder output through the local downstream model and
   compare decoded behavior with the local golden.
4. Repeat on the physical glasses or an exact AR1-class target when its chipset
   and deployment interface are confirmed.
