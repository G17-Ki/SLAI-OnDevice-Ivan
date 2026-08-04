# Zipformer QAIHM 0.58 source export

This directory reproduces the source side of the exact-SM8850 Zipformer
experiment without making the Mac load the 302 MB checkpoint or construct the
model. The heavy work runs in an isolated Colab environment; this repository
contains only the public harness and resulting public-safe provenance.

The pin derivation and local-memory rationale are recorded in the
[source-export gate](../../reports/android_phone/streaming_response/zipformer_v81/SOURCE_EXPORT_GATE.md).

The exporter pins and verifies:

- `qai-hub-models==0.58.0`, including the published wheel SHA-256;
- checkpoint revision `6eb615ae77ecac05c5628d5c8ed7037c14a338d5`;
- `exp/pretrained.pt`, size `302,404,721` bytes and SHA-256
  `9fb1a61c426c3910dc7228aec7c3e5783f268c15a25eb93cf37db9faf4f8e243`;
- Torch `2.9.0+cpu`, k2, ONNX, ONNX Runtime, and the other versions recorded in
  `SOURCE_EXPORT_GATE.md`;
- fixed ONNX opset 17 with the legacy exporter (`dynamo=False`) and no dynamic
  axes.

For each component, QAIHM's own `component.serialize(...)` creates the
authoritative TorchScript file. The saved trace is reloaded, exported to a
fixed-shape ONNX mirror, and executed on the same seed-`20260723` inputs. Every
integer output must match exactly. Floating outputs must have the same dtype
and shape and pass `rtol=1e-4`, `atol=1e-5`. A component is resumable only when
its manifest, checkpoint identity, comparison result, and artifact hashes all
still validate.

## Offline validation

These commands do not import PyTorch, download a model, contact Colab, or
contact AI Hub:

```bash
python3 scripts/zipformer_source_export/export_sources.py --self-check
bash scripts/zipformer_source_export/orchestrate_colab.sh --dry-run
python3 -m unittest -v scripts/zipformer_source_export/test_harness.py
```

## Future Colab launch

The private Colab bundle is deliberately outside this public repository. Point
the harness at its directory; never copy `.colab_env` into the repo.

```bash
bash scripts/zipformer_source_export/orchestrate_colab.sh \
  --launch \
  --confirm-cloud \
  --colab-bundle /absolute/private/path/to/colab-cli-skill
```

The command creates a detached host `tmux` session and returns immediately. A
real allocation is impossible without all of `--launch`, `--confirm-cloud`, a
valid private bundle, and the internally verified tmux worker. T4 is the
default because export is CPU-side; L4 is accepted when T4 is unavailable.

The launch command prints three non-secret paths/names:

- the detached tmux session;
- `orchestrator.log`;
- `latest-checkpoint.tar`.

Attach only when needed:

```bash
tmux attach -t zipformer-source-YYYYMMDDTHHMMSSZ
```

Detached tmux survives closing a terminal or losing an SSH connection. It
cannot keep a sleeping or powered-off laptop alive: the Colab CLI keep-alive
process still needs an awake host. The periodic local checkpoint makes that
failure recoverable. For truly unattended operation, run the same public
orchestrator and private bundle on an always-on host.

## Checkpoint and resume behavior

After each completed component, and again on success or failure, the remote
worker atomically creates `checkpoint.tar`. The host downloads a generation
only after matching both its reported byte size and SHA-256 and rejecting
unsafe tar members. It never downloads caches, the Hugging Face checkpoint,
the virtual environment, OAuth material, signed URLs, or the raw worker log.

If a runtime is reaped, start a new detached run from the latest verified
checkpoint:

```bash
bash scripts/zipformer_source_export/orchestrate_colab.sh \
  --launch \
  --confirm-cloud \
  --colab-bundle /absolute/private/path/to/colab-cli-skill \
  --resume-checkpoint artifacts/zipformer-source-export/OLD_RUN/latest-checkpoint.tar
```

The new runtime safely extracts regular relative files only, rebuilds and
verifies the pinned environment, re-verifies all retained artifact hashes, and
continues with the first incomplete component. The order is decoder, joiner,
then encoder so useful canary artifacts are checkpointed before the
memory-heavier encoder.

## Downloaded artifact layout

```text
sources/
  decoder.pt
  decoder.onnx
  joiner.pt
  joiner.onnx
  encoder.pt
  encoder.onnx
validation/
  *_seed_20260723.npz
manifests/
  decoder.json
  joiner.json
  encoder.json
export_manifest.json
export_status.json
worker_status.json
requirements.lock.txt
package_provenance.json
harness_provenance.json
SHA256SUMS
```

The raw operational log remains only in the temporary Colab run root. Colab
CLI stdout/stderr is captured to mode-`0600` temporary files and deleted, so an
expired login cannot print an authorization URL into a public log. The wrapper
also removes optional Hugging Face, GitHub, and W&B tokens from the environment
before any Colab subprocess is started.

## Optional capped Mac canary

The exporter itself is platform-neutral. In a local environment with the exact
packages from the source-export gate, decoder and joiner can be exported at
background priority with two compute threads:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
/usr/bin/time -l taskpolicy -b nice -n 10 \
  "$PY" scripts/zipformer_source_export/export_sources.py \
  --root artifacts/zipformer-source-local-canary \
  --components decoder,joiner \
  --threads 2
```

This is not a tiny-model-only operation: QAIHM constructs the complete model
and verifies the full 302 MB checkpoint even for one component. Budget roughly
1–2 GiB RSS and do not run it when the Mac is memory-constrained. The encoder
trace and ONNX serialization remain Colab-only for this project. The default
orchestrator never performs this local canary.

This harness exports and locally validates source graphs only. It does not
allocate or submit any Qualcomm AI Hub job. Pass the resulting verified ONNX
files (while retaining the `.pt` records) to the
[exact-SM8850 AI Hub runner](../zipformer_aihub_v81/README.md) in a separate,
explicitly confirmed tmux run.
