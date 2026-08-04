# Zipformer exact-SM8850 release-context profile

Date: 2026-07-23 CST<br>
Decision: **placement and latency pass; numerical/transcript acceptance remains
open**

Machine-readable evidence:
[`release_context_sm8850_profile_result.json`](release_context_sm8850_profile_result.json).

## Result

Qualcomm's prebuilt catalog-float QNN contexts were profiled on the exact
account target: `Snapdragon 8 Elite Gen 5 QRD`, Android 16, `sm8850`, HTP v81.
This route has no AI Hub QuantizeJob: floating external tensors remain FP32,
integer IDs/cache lengths remain INT32, and the HTP binary lowers internal
kernels and packed weights to FP16. It is not Qualcomm's separate mixed policy
(encoder W8A16, decoder/joiner W16A16). All cloud activity ran in detached tmux
and every uploaded model and profile job was recorded before waiting. The
reproduction script enforces decoder, then joiner,
then encoder; it will not release a later component until the earlier canaries
pass strict NPU placement in the same state directory.

| Component | Profile job | Latency | Estimated inference peak | Hexagon NPU | CPU fallback | Unsupported/unknown |
|---|---|---:|---:|---:|---:|---:|
| Encoder, one 0.71 s feature chunk | `jpxdvmzjg` | **4.741 ms** | 126.47 MiB | **2,541/2,541** | 0 | 0 |
| Decoder, one call | `jpy7evl4p` | **0.053 ms** | 117.49 MiB | **8/8** | 0 | 0 |
| Joiner, one call | `j5qm8wrng` | **0.137 ms** | 117.75 MiB | **5/5** | 0 | 0 |

The lower-bound sum of one encoder, one decoder, and one joiner call is 4.931
ms per 0.71 seconds of audio, or neural-component RTF 0.00695. This is not an
end-to-end ASR RTF: a transducer can call decoder and joiner repeatedly for one
encoder chunk, and feature extraction, cache orchestration, token
stabilization, VAD, and application work are excluded.

Context first/warm load times were 140.434/137.250 ms for the encoder,
86.158/81.134 ms for the decoder, and 82.447/86.694 ms for the joiner.

## Exact-target provenance

The archive is the `qai-hub-models` 0.55.0 float context release, SHA-256
`58df377fcb1cf9967b3925460146c03d4d6879932791a7d2f5ab4e8f61f0494c`.
Its generic catalog label is "Snapdragon 8 Elite Gen 5 Mobile", but both
metadata and binary payload identify the exact target:

- chipset aliases `qualcomm-snapdragon-8-elite-gen5` and `sm8850`;
- reference device `Snapdragon 8 Elite Gen 5 QRD`;
- HTP version 81 and SoC model 87;
- QAIRT `2.45.0.260326154327`, `min_arch=81`, and `soc_type=SM8850`.

This resolves the earlier catalog-label ambiguity. These are exact-target
release-binary measurements, not `sm8850-ad` cross-SKU results. They are still
not a fresh compile made under this account.

## Comparison with the published retail proxy

The installed scorecard reports 4.832/0.046/0.130 ms for encoder, decoder, and
joiner on the Galaxy S26 `sm8850-ad` proxy. The exact QRD result is close:
encoder is 1.9% faster, decoder differs by 0.007 ms, and joiner differs by
0.007 ms. Layer accounting also differs slightly for the encoder (2,541 exact-
QRD profile operations versus 2,546 published scorecard layers), so do not
treat the counts as identical instrumentation.

## What this does and does not prove

This run passes the requested runtime/placement gate:

- every component loads and profiles on exact SM8850/HTP-v81;
- every reported operation is on Hexagon NPU;
- there is no CPU fallback or unsupported operation;
- component latency and memory are recorded for the catalog-float/internal-FP16
  release before any explicit INT8/INT16 quantization experiment.

It does **not** complete the local-golden gate. The downloaded release archive
contains only compiled contexts and interface metadata, not matching source
graphs or reference outputs. Therefore no numerical output or transcript was
compared in this run. A matching source export must generate deterministic
local goldens, followed by exact-QRD inference and transcript equivalence,
before this model can replace SenseVoice. The source path is documented in the
[`SOURCE_EXPORT_GATE.md`](SOURCE_EXPORT_GATE.md) decision memo and implemented
by the
[`zipformer_source_export`](../../../../scripts/zipformer_source_export/README.md)
harness.

## Reproduce the release-context profile

Use a dedicated environment containing `qai-hub==0.52.0` and
`qai-hub-models==0.58.0`. The newer client can fetch the pinned 0.55 release
asset. Keep both the download and AI Hub submission in detached `tmux`.

First fetch the public archive into a persistent work directory:

```bash
FETCH_DIR="$PWD/work/zipformer-release-v055"
mkdir -p "$FETCH_DIR"
tmux new-session -d -s zipformer-release-fetch \
  "cd '$FETCH_DIR' && \
   qai-hub-models fetch zipformer \
     -v 0.55.0 \
     -r qnn_context_binary \
     -p float \
     -d 'Snapdragon 8 Elite Gen 5 QRD' \
     > fetch.log 2>&1"
```

After the session finishes, verify and extract the exact asset:

```bash
ARCHIVE="$FETCH_DIR/zipformer-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
test "$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')" = \
  58df377fcb1cf9967b3925460146c03d4d6879932791a7d2f5ab4e8f61f0494c
unzip -q "$ARCHIVE" -d "$FETCH_DIR/extracted"
MODEL_DIR="$FETCH_DIR/extracted/zipformer-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5"
```

The extracted directory must contain `decoder.bin`, `joiner.bin`,
`encoder.bin`, and `metadata.json`. Submit all three profiles sequentially in
one detached session so the script's decoder → joiner → encoder state gate is
preserved:

```bash
RUN_DIR="$PWD/work/zipformer-release-v055-profile"
mkdir -p "$RUN_DIR"
tmux new-session -d -s zipformer-release-profile \
  "cd '$PWD' && { \
     python3 scripts/profile_zipformer_release_contexts.py \
       --model-dir '$MODEL_DIR' --output-dir '$RUN_DIR' \
       --component decoder --confirm-cloud && \
     python3 scripts/profile_zipformer_release_contexts.py \
       --model-dir '$MODEL_DIR' --output-dir '$RUN_DIR' \
       --component joiner --confirm-cloud && \
     python3 scripts/profile_zipformer_release_contexts.py \
       --model-dir '$MODEL_DIR' --output-dir '$RUN_DIR' \
       --component encoder --confirm-cloud; \
   } > '$RUN_DIR/run.log' 2>&1"
```

Attach with `tmux attach -t zipformer-release-profile`, or rerun the same
command after interruption; the state file resumes by model and job ID. The
compiled contexts are Qualcomm release assets and remain outside this public
repository.

Reproduction code:
[`profile_zipformer_release_contexts.py`](../../../../scripts/profile_zipformer_release_contexts.py).
