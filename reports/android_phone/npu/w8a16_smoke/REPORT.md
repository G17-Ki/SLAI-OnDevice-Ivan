# W8A16 depth/RQ decoder smoke

Date: 2026-07-17  
Target: `Snapdragon 8 Elite Gen 5 QRD`, Android 16, SM8850, HTP v81  
Requested quantization: INT8 weights, INT16 activations

## Verdict

**Depth/RQ decoder: blocked.** The PT2 source converted to ONNX, but AI Hub's
quantizer rejected seven shared depth-embedding initializers because each is
consumed by both a `Gemm` and a `Gather`. No quantized depth context binary was
created, so exact-token acceptance, placement, latency, peak memory, and a
W8A16-to-FP16 depth binary-size ratio are unavailable.

**Required conv-prefill fallback: full-NPU but numerically rejected.** It
quantized, compiled, profiled, and inferred on HTP v81 with no CPU fallback,
but failed the existing float tolerance. Its high cosine similarity does not
override that acceptance failure.

The QRD is a Qualcomm reference device. It is not a retail device, it is not the
Nubia phone, and these are not measurements from the physical glasses.

## Four requested metrics

| Component | Numerical verdict | Placement NPU/CPU/other | Latency | Peak memory | Binary size versus FP16 |
|---|---|---:|---:|---:|---|
| Depth/RQ decoder W8A16 | blocked before QNN compile; no token output | unavailable | unavailable | unavailable | no W8A16 binary produced |
| Conv-prefill W8A16 fallback | failed tolerance; max abs 0.023046, cosine 0.9999955 | 50/0/0 | 0.977 ms | 118.176 MiB | 67,743,744 B (64.605 MiB), 50.35% of FP16 |

For the fallback's current-toolchain FP16 comparator, the context binary is
134,541,312 bytes (128.309 MiB), latency is 1.833 ms, peak memory is 117.691
MiB, and the numerical check passes. W8A16 is 49.65% smaller and 1.88x faster
for this isolated invocation, but its max error exceeds the acceptance
tolerance and its reported peak is 0.484 MiB higher. It is therefore not an
accepted replacement.

The downloaded binaries exceed the repository's 5 MiB publication limit and
remain ignored and unstaged. Their byte counts and SHA-256 hashes are retained
in the raw summary.

## Calibration recipe and limitation

`scripts/capture_depth_calibration.py` replayed the pinned real BF16 turn on the
Mac and captured 80 depth inputs, each shaped `[1, 2048]`. They come from audio
frame indices 0 through 79 and generation steps 6 through 120 of turn 1. The
manifest records the source model revision, upstream commit, input-audio hash,
golden-token hash, and the exact generation step that produced every sample:

`raw/20260717t074742z/calibration/calibration-manifest.json`.

The capture passed in 91.834 seconds and produced a 619,613-byte NPZ with
SHA-256 `2e3e4e282385339c1f09574107a64cde31457b40c80c55ea064a2911bb8b35e2`.
The first local capture attempt failed because the helper passed a local
snapshot as a string, causing the library to validate it as a Hub repo ID. The
failure JSON and full tool log are retained under `raw/20260717t070923z/`.

Uploading the real 80-sample calibration set was blocked before any AI Hub
dataset was created. The exact rejection is retained in
`raw/20260717t074742z/calibration-upload-blocked.txt`:

```text
This action was rejected due to unacceptable risk.
Reason: This would upload locally generated calibration embeddings from the workspace to Qualcomm AI Hub, an untrusted external destination under this policy, and the user has not explicitly re-approved that specific data export after earlier upload denials.
The agent must not attempt to achieve the same outcome via workaround, indirect execution, or policy circumvention. Proceed only with a materially safer alternative, or if the user explicitly approves the action after being informed of the risk. Otherwise, stop and request user input.
```

The earlier Colab capture/upload path was independently blocked; its exact
rejection is in `raw/20260717t070923z/colab-upload-blocked.txt`.

The policy-safe fallback reused datasets already present in the configured AI
Hub account: depth dataset `d9emwmov2` and conv-prefill dataset `d7lvvjgn2`.
Each contains one matching example. This does **not** satisfy the requested
32-128 real-sample calibration recipe, so even a passing output would have been
provisional rather than publication acceptance. The captured depth embeddings
also cannot calibrate the conv fallback because its input shape is
`[1, 16, 2048]`.

## Job record

| Component/stage | Job ID | State |
|---|---|---|
| Depth PT2 to ONNX | `j5mdk6ly5` | success |
| Depth W8A16 quantize | `jgjwyevx5` | failed |
| Conv fallback PT2 to ONNX | `jg9x2wd8g` | success |
| Conv fallback W8A16 quantize | `jgjwyem85` | success |
| Conv fallback strict v81 QNN compile | `jgo4wvvq5` | success |
| Conv fallback profile | `jp0v9rdng` | success |
| Conv fallback inference | `jp84r76o5` | success |

All cloud stages ran in detached tmux session `npu-m1a-w8a16`. Dependent jobs
were necessarily sequential across ONNX conversion, quantization, and QNN
compilation; profile and inference were submitted together before polling.

## Depth quantization error, verbatim

```text
RuntimeError: Found shared parameter(s) with conflicting consumer types:

  - input name: depth_embeddings.0.to_logits.weight
    - consumer 0: node_linear_31 (Gemm)
    - consumer 1: node_embedding (Gather)
  - input name: depth_embeddings.1.to_logits.weight
    - consumer 0: node_linear_62 (Gemm)
    - consumer 1: node_embedding_1 (Gather)
  - input name: depth_embeddings.2.to_logits.weight
    - consumer 0: node_linear_93 (Gemm)
    - consumer 1: node_embedding_2 (Gather)
  - input name: depth_embeddings.3.to_logits.weight
    - consumer 0: node_linear_124 (Gemm)
    - consumer 1: node_embedding_3 (Gather)
  - input name: depth_embeddings.4.to_logits.weight
    - consumer 0: node_linear_155 (Gemm)
    - consumer 1: node_embedding_4 (Gather)
  - input name: depth_embeddings.5.to_logits.weight
    - consumer 0: node_linear_186 (Gemm)
    - consumer 1: node_embedding_5 (Gather)
  - input name: depth_embeddings.6.to_logits.weight
    - consumer 0: node_linear_217 (Gemm)
    - consumer 1: node_embedding_6 (Gather)

Please call ``aimet_onnx.utils.duplicate_shared_initializers(onnx_model.graph)`` before creating QuantizationSimModel to ensure each consumer takes a unique copy of the initializer as input. - Check the quantize logs for detailed error information.
```

This is a graph-preparation blocker, not an HTP placement result. The next
depth attempt should duplicate the seven shared ONNX initializers as requested
by AIMET, then calibrate with the 80 real captured examples after that specific
data export is approved.

## Provenance

The capture manifest and NPZ were script-generated by
`scripts/capture_depth_calibration.py`. AI Hub state, comparisons, placement,
server logs, and binary hashes under
`raw/20260717t074742z/fallback-remote-calibration/` were script-generated by an
uncommitted temporary workflow driver. The driver was kept outside the repo so
this round added only the one capture helper allowed by the experiment plan;
all effective options, dataset/model IDs, polls, job IDs, errors, and results
are retained in `w8a16-state.json` and the tool log.

This `REPORT.md` was hand-written from those raw artifacts. No measured numeric
value was edited. Publication cleanup redacts local absolute path strings only;
measurements, hashes, job IDs, and error text are otherwise unchanged.
