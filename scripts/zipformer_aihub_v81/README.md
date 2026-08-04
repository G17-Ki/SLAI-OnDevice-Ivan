# Zipformer exact-SM8850 / HTP-v81 AI Hub runner

This directory contains the lower-level, resumable experiment runner for the
official Zipformer **catalog-float** encoder, decoder, and joiner graphs. It
targets exactly:

- device: `Snapdragon 8 Elite Gen 5 QRD`;
- OS: Android `16`;
- chipset / accelerator: `sm8850` / Hexagon HTP `v81`;
- QAIRT: `2.45`;
- final runtime: QNN context binary.

The final context binary is produced through AI Hub's normal two-stage AOT
route: compile the source artifact to a QNN DLC, then link that DLC into a device-
specific context binary. Catalog-float means there is no calibration or AI Hub
QuantizeJob, floating-point tensor I/O remains FP32, required token/cache tensor
I/O remains INT32, and QAIRT lowers internal HTP compute and weights to FP16.
This is not Qualcomm's separate mixed policy (encoder W8A16; decoder and joiner
W16A16).

This runner does not invoke a model exporter. Generate the pinned artifacts with
the sibling
[`zipformer_source_export`](../zipformer_source_export/README.md) harness, or
supply three already-exported, fixed-shape float source artifacts from the same
official checkpoint/export revision. TorchScript is the normal
`qai-hub-models` source format:

```text
$MODEL_DIR/
  encoder.pt
  decoder.pt
  joiner.pt
```

`.pt2` and fixed-shape `.onnx` sources are also accepted through the three
`--*-model` arguments. If the official exporter emits identically named files
in separate component directories, pass those relative paths explicitly. Do
not use the public Sherpa INT8 encoder/joiner files for this experiment. The
runner rejects filenames or graph markers that suggest a quantized source. For
ONNX it also parses graph metadata with external tensor loading disabled and
rejects quantization operators or INT4/8/16 initializers. For TorchScript/PT2 it
inspects the loaded graph, module types, parameter dtypes, and low-precision
integer buffers. The result is recorded in the fixture manifest and required
again before any cloud action. Torch source preparation also checks the embedded
forward/input count and the canonical output shapes/dtypes before it labels any
output name.

## Safety and ordering

`preflight` and `prepare` are offline. They never construct an AI Hub client.
`run --dry-run` is also offline. A real cloud run requires both an active tmux
session and `--confirm-cloud`; there is no bypass flag.

The cloud order is deliberately strict:

1. decoder: upload, compile, link, profile, inference, placement, and
   tolerance-based numerical comparison;
2. joiner: the same sequence;
3. evaluate the canary gate;
4. upload or submit the encoder only when both canaries have zero CPU/unknown
   operations and all numerical comparisons pass.

The encoder source is not uploaded before that gate. A canary failure or a
pending job leaves the encoder untouched.

Every source upload ID, input-dataset upload ID, compile job ID and target-model
ID, link job ID and context-model ID, profile job ID, and inference job ID is
written atomically to `state.json` immediately after it is returned. Re-running
the same command in the same output directory retrieves those resources by ID
instead of resubmitting them. Failed attempts are retained as evidence; start a
new output directory for an intentional fresh experiment.
Source artifacts are hashed before and after local golden generation and again
immediately before and after upload. An upload ID is still retained if the file
changes, but that ID is marked invalid and is never compiled. Fixture loading
and cloud-state creation share the same output-directory lock.

## 1. Offline preflight

```bash
$PYTHON scripts/zipformer_aihub_v81/runner.py preflight \
  --model-dir "$MODEL_DIR" \
  --output-dir "$RUN_DIR"
```

The result is written to `$RUN_DIR/preflight.json`. It reports package and file
readiness, checksums, the exact target, and the planned job order. It contains
no credentials or local absolute paths.

## 2. Generate fixed local inputs and goldens

Use an existing environment containing NumPy plus the runtime matching the
source: PyTorch for `.pt`/`.pt2`, or both ONNX and ONNX Runtime for `.onnx`.
ONNX graph inspection calls `onnx.load_model(..., load_external_data=False)`, so
external tensor payloads are never resolved by the precision gate. Loading a
self-contained exported artifact does not require Icefall, k2, a checkpoint
download, or a model export.

```bash
$PYTHON scripts/zipformer_aihub_v81/runner.py prepare \
  --model-dir "$MODEL_DIR" \
  --output-dir "$RUN_DIR" \
  --seed 20260723
```

The runner caps PyTorch or ONNX Runtime and common BLAS/OpenMP thread pools at
two. It loads one component at a time, applies the official fixed input
contract, validates fixed ONNX I/O metadata when applicable, executes one
seeded CPU reference inference, and atomically writes:

```text
$RUN_DIR/fixtures/decoder.npz
$RUN_DIR/fixtures/joiner.npz
$RUN_DIR/fixtures/encoder.npz
$RUN_DIR/fixtures/manifest.json
```

Each NPZ stores arrays as `input__<name>` and `golden__<name>`. Encoder cache
states are deterministic zeros; the acoustic chunk and canary vectors/tokens
come from the recorded seed. `manifest.json` binds every NPZ to the SHA-256 of
its exact source artifact.

To regenerate an existing fixture deliberately, add `--overwrite`. To prepare
one component only, repeat `--component` as needed.

## 3. Cloud dry run

This exercises the full local validation and state compatibility checks without
importing an AI Hub client or contacting the service:

```bash
$PYTHON scripts/zipformer_aihub_v81/runner.py run \
  --model-dir "$MODEL_DIR" \
  --output-dir "$RUN_DIR" \
  --dry-run
```

## 4. Real run, only inside tmux

Keep the model and run directories on persistent storage. The runner itself
enforces tmux for all cloud work.

```bash
tmux new-session -d -s zipformer-v81 \
  "$PYTHON scripts/zipformer_aihub_v81/runner.py run \
    --model-dir '$MODEL_DIR' \
    --output-dir '$RUN_DIR' \
    --confirm-cloud \
    > '$RUN_DIR/run.log' 2>&1"
```

Inspect or attach later:

```bash
tmux attach -t zipformer-v81
```

If the Mac closes, a poll ceiling is reached, or the process exits, launch the
same command again. The state fingerprint prevents accidental resume with a
different graph, fixture, seed, tolerance, target, or profiling configuration.

## Outputs and acceptance

For each completed component:

```text
$RUN_DIR/components/<component>/profile.json
$RUN_DIR/components/<component>/placement.json
$RUN_DIR/components/<component>/comparison.json
$RUN_DIR/components/<component>/remote_outputs.npz
```

`placement.json` classifies every runtime operation into exactly one bucket:
`hexagon_npu`, `cpu_fallback`, or `unsupported_or_export_failure`. Missing or
failed profiles receive the third classification. Acceptance requires at least
one profiled operation, every operation on Hexagon/HTP, and every remote output
to pass the configured local-golden tolerance. Integer cache outputs are
compared exactly. Floating outputs use `allclose` with the recorded `atol` and
`rtol` plus max/mean/P95 error, RMSE, normalized RMSE, and cosine similarity.
This is a tolerance-based FP32-reference versus FP16-HTP check, not exact
floating-point identity.

Generated JSON is sanitized before publication: known local roots, generic home
or temporary paths, credential-like values, and signed URL query strings are
redacted. The runner never writes AI Hub credentials. Job and model/dataset IDs
are experiment provenance, not authentication secrets, but they still identify
account resources and should be published only if that is intentional.
`run.log` is raw operational stdout/stderr from the runner and third-party SDK;
keep it ignored and do not treat it as a publication artifact.

## Offline tests

```bash
cd scripts/zipformer_aihub_v81
$PYTHON -m unittest -v test_runner.py
```

The tests do not load a model or contact AI Hub.
