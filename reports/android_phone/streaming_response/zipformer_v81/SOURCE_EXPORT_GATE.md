# Zipformer 0.58 source-export gate

Date: 2026-07-23 CST<br>
Scope: offline inspection only; no model export, AI Hub job, or Colab run was
started by this gate.

## Decision

The lightest reproducible route is to export once in a clean, pinned
`qai-hub-models==0.58.0` environment and retain **both** artifacts per
component:

1. `encoder.pt`, `decoder.pt`, and `joiner.pt`: the authoritative fixed-shape
   TorchScript sources produced by the same serialization path that the
   official exporter uploads to AI Hub;
2. matching fixed-shape float ONNX mirrors, exported from those saved traces
   with the legacy TorchScript ONNX exporter, then checked against the
   TorchScript outputs with identical seeded inputs.

Do the decoder/joiner canary locally with two CPU threads if useful. Export the
encoder in Colab: this Mac has 8 GiB RAM, and the official loader constructs the
entire 267 MB float model even when only one component is requested. The full
encoder trace plus ONNX serialization can temporarily duplicate weights and
graph state.

The existing `qai-hub-models` CLI cannot create these files by itself. The
0.58 official export pipeline instantiates the model, traces each component to
TorchScript, uploads it, and performs conversion in AI Hub. It does not expose
a local-ONNX-only switch.

## Exact source recipe

The 0.58 model is constructed from:

| Item | Required pin |
|---|---|
| Python | 3.11 recommended; package supports 3.10 through 3.13 |
| `qai-hub-models` | `0.58.0` |
| 0.58.0 wheel SHA-256 | `2100a95f173c6c7b8f3bf65dcb710586098099522a5a901c6f918f3b981c4ada` |
| PyTorch | `2.9.0` (`2.9.0+cpu` on Linux) |
| k2 | `1.24.4.dev20251029+cpu.torch2.9.0` |
| Icefall | commit `693f069de73fd91d7c2009571245d97221cc3a3f` |
| Qualcomm Icefall patch SHA-256 | `bca5978ea4674c956e133880f3212b5961303023ba3b732eda3fde486a19245d` |
| Checkpoint repository | `pfluo/k2fsa-zipformer-chinese-english-mixed` |
| Checkpoint revision | `6eb615ae77ecac05c5628d5c8ed7037c14a338d5` |
| `exp/pretrained.pt` | 302,404,721 bytes; SHA-256 `9fb1a61c426c3910dc7228aec7c3e5783f268c15a25eb93cf37db9faf4f8e243` |
| ONNX export | opset 17, `dynamo=False`, no dynamic axes |

Qualcomm's model code currently downloads `main` without a revision. On
2026-07-23, `main` resolves to the revision and checkpoint hash above, but the
run must still override that unpinned call as shown below.

### Clean environment

Apple Silicon:

```bash
export RUN_ROOT="$PWD/work/zipformer-source-v058"
python3.11 -m venv "$RUN_ROOT/venv"
export PY="$RUN_ROOT/venv/bin/python"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==2.9.0" "torchvision==0.24.0"
"$PY" -m pip install \
  "qai-hub-models[zipformer]==0.58.0" \
  "qai-hub==0.52.0" \
  "onnx==1.18.0" \
  "onnxruntime==1.22.1" \
  "huggingface-hub==0.34.0" \
  "k2==1.24.4.dev20251029+cpu.torch2.9.0" \
  -f https://k2-fsa.github.io/k2/cpu.html
"$PY" -m pip check
"$PY" -m pip freeze > "$RUN_ROOT/requirements.lock.txt"
```

Linux/Colab uses the same commands except for the first PyTorch install:

```bash
"$PY" -m pip install \
  "torch==2.9.0+cpu" "torchvision==0.24.0+cpu" \
  --index-url https://download.pytorch.org/whl/cpu
```

This deliberately does not reuse the existing export environment: that
environment inherits system packages, has Torch 2.9.1, ONNX Runtime 1.18.0,
Hugging Face Hub 0.32.4, and no k2 or Zipformer extras.

### Pin and verify the checkpoint

```bash
export HF_REV=6eb615ae77ecac05c5628d5c8ed7037c14a338d5
export HF_SHA=9fb1a61c426c3910dc7228aec7c3e5783f268c15a25eb93cf37db9faf4f8e243

RUN_ROOT="$RUN_ROOT" HF_REV="$HF_REV" HF_SHA="$HF_SHA" "$PY" - <<'PY'
import hashlib
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

path = Path(hf_hub_download(
    repo_id="pfluo/k2fsa-zipformer-chinese-english-mixed",
    filename="exp/pretrained.pt",
    revision=os.environ["HF_REV"],
    cache_dir=str(Path(os.environ["RUN_ROOT"]) / "ckpt"),
))
digest = sha256(path)
assert digest == os.environ["HF_SHA"], (path, digest)
print(path)
PY
```

### Export authoritative traces and verified ONNX mirrors

Save the following as `$RUN_ROOT/export_sources.py`. It patches the model's
unpinned checkpoint call to the verified local revision, uses Qualcomm's own
input/output specifications, preserves fixed shapes, and compares ONNX Runtime
against the saved TorchScript trace.

```python
import hashlib
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download as real_hf_download
from qai_hub_models.models.zipformer import model as zipformer_model
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx.helpers import safe_torch_onnx_export

ROOT = Path(os.environ["RUN_ROOT"]).resolve()
REV = "6eb615ae77ecac05c5628d5c8ed7037c14a338d5"
CKPT_SHA = "9fb1a61c426c3910dc7228aec7c3e5783f268c15a25eb93cf37db9faf4f8e243"
COMPONENTS = os.environ.get(
    "ZIPFORMER_COMPONENTS", "decoder,joiner,encoder"
).split(",")
OUT = ROOT / "sources"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
torch.set_num_interop_threads(1)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def pinned_download(repo_id, filename, cache_dir=None, **kwargs):
    assert repo_id == "pfluo/k2fsa-zipformer-chinese-english-mixed"
    assert filename == "exp/pretrained.pt"
    path = Path(real_hf_download(
        repo_id=repo_id,
        filename=filename,
        revision=REV,
        cache_dir=str(ROOT / "ckpt"),
        local_files_only=True,
    ))
    assert sha256(path) == CKPT_SHA
    return str(path)

def flatten(value):
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu().numpy()]
    arrays = []
    for item in value:
        arrays.extend(flatten(item))
    return arrays

zipformer_model.hf_hub_download = pinned_download
collection = zipformer_model.HfZipformer.from_pretrained()

for name in COMPONENTS:
    component = collection.components[name].eval()
    spec = component.get_input_spec()
    input_names = list(spec)
    output_names = list(component.get_output_spec())
    inputs = tuple(make_torch_inputs(spec, seed=20260723))

    traced = component.convert_to_torchscript(spec)
    pt_path = OUT / f"{name}.pt"
    onnx_path = OUT / f"{name}.onnx"
    torch.jit.save(traced, pt_path)
    traced = torch.jit.load(str(pt_path)).eval()
    safe_torch_onnx_export(
        traced,
        inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )

    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    assert [x.name for x in graph.graph.input] == input_names
    assert [x.name for x in graph.graph.output] == output_names
    for value in list(graph.graph.input) + list(graph.graph.output):
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        assert dims and all(d > 0 for d in dims), (value.name, dims)

    reference = flatten(traced(*inputs))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(output_names, {
        key: value.detach().cpu().numpy()
        for key, value in zip(input_names, inputs)
    })
    assert len(reference) == len(actual) == len(output_names)
    for expected, observed in zip(reference, actual):
        if np.issubdtype(expected.dtype, np.integer):
            assert np.array_equal(expected, observed)
        else:
            np.testing.assert_allclose(expected, observed, rtol=1e-4, atol=1e-5)
```

Record provenance immediately after export:

```bash
shasum -a 256 "$RUN_ROOT"/sources/* > "$RUN_ROOT/SHA256SUMS"
```

## Capped local preflight

This runs only the decoder and joiner export, at background priority and with
two compute threads. It still loads the full checkpoint once, so budget roughly
1-2 GiB RSS. The estimate is not a measurement; retain `/usr/bin/time -l`
output when the run is executed.

```bash
export QAIHM_STORE_ROOT="$RUN_ROOT/qaihm-cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ZIPFORMER_COMPONENTS=decoder,joiner

/usr/bin/time -l taskpolicy -b nice -n 10 \
  "$PY" "$RUN_ROOT/export_sources.py" \
  > "$RUN_ROOT/local-canary.log" 2>&1

python scripts/zipformer_aihub_v81/runner.py preflight \
  --model-dir "$RUN_ROOT/sources" \
  --output-dir "$RUN_ROOT/runner-local" \
  --encoder-model encoder.onnx \
  --decoder-model decoder.onnx \
  --joiner-model joiner.onnx

python scripts/zipformer_aihub_v81/runner.py prepare \
  --model-dir "$RUN_ROOT/sources" \
  --output-dir "$RUN_ROOT/runner-local" \
  --encoder-model encoder.onnx \
  --decoder-model decoder.onnx \
  --joiner-model joiner.onnx \
  --component decoder --component joiner
```

The preflight will correctly remain incomplete until `encoder.onnx` exists;
the two requested canary fixtures should still be generated and checked.

## Full Colab export

Use the same checkpoint cache and exporter, set
`ZIPFORMER_COMPONENTS=decoder,joiner,encoder`, and retain at least 6 GiB free
RAM and 2 GiB free disk. A conservative expected peak is 3-5 GiB RSS during
encoder trace/ONNX serialization; this is an engineering bound, not a measured
result. The work is CPU-side, so a T4 runtime is sufficient.

Inside the Colab runtime:

```bash
export RUN_ROOT=/content/zipformer-source-v058
export QAIHM_STORE_ROOT="$RUN_ROOT/qaihm-cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 ZIPFORMER_COMPONENTS=decoder,joiner,encoder
python "$RUN_ROOT/export_sources.py"
python -m pip freeze > "$RUN_ROOT/requirements.lock.txt"
sha256sum "$RUN_ROOT"/sources/* > "$RUN_ROOT/SHA256SUMS"
tar -C "$RUN_ROOT" -czf /content/zipformer-source-v058.tar.gz \
  sources SHA256SUMS requirements.lock.txt
```

Per project policy, drive the persistent Colab session and artifact download
from a detached host `tmux` session. Do not use a one-shot local shell whose
keep-alive process dies when the laptop disconnects.

After download, run the repository `preflight` and `prepare` commands for all
three components, then use the exact-QRD runner in detached `tmux`. Compile the
verified ONNX mirrors for the requested ONNX experiment, while retaining the
`.pt` traces as the authoritative official-export source record.

## Cache result

No matching float sources are already cached:

- no `exp/pretrained.pt` was found;
- no Qualcomm `encoder.pt`, `decoder.pt`, or `joiner.pt` was found;
- no matching float encoder/decoder/joiner ONNX set was found;
- the expected patched Icefall cache
  `$HOME/.qaihm/external_repos/models/zipformer/icefall/cba4f2080b173466`
  is absent.

The cached public Sherpa INT8 graphs and X-ASR graphs are different exports and
must not be substituted. The downloaded 0.55 QNN contexts are deployment
binaries, not recoverable local-golden sources.

## Provenance against the 0.55 context asset

The semantic match is strong:

- 0.55 and 0.58 use the same Icefall commit and byte-identical patch;
- `model_config.yaml` and `model_adaption.py` are byte-identical;
- the component forward methods and checkpoint location are unchanged;
- fixed input/output contracts match the downloaded context metadata exactly.

The match is **not cryptographically complete**:

- the 0.55 archive records QAIRT 2.45 and graph I/O, but not the source-model or
  checkpoint SHA-256;
- 0.55 used Torch `2.8.0+cpu` and
  `k2==1.24.4.dev20250807+cpu.torch2.8.0`, while 0.58 uses Torch 2.9.0 and the
  k2 build pinned above;
- `model.py` changed with the collection/output-spec API refactor;
- both releases leave the checkpoint revision implicit in source code.

Therefore, do not call a new 0.58 ONNX or context binary byte-equivalent to the
0.55 release asset. Call it a semantically matched, pinned 0.58 re-export. The
adoption gate is numerical: TorchScript equals local ONNX on the same fixtures,
then exact-QRD inference equals those local goldens within the recorded
tolerances. The already-passing 0.55 contexts remain independent exact-device
release-binary evidence.
