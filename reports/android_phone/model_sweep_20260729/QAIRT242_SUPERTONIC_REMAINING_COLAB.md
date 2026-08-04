# QAIRT 2.42 Supertonic vector/vocoder Colab lane

Status: prepared and locally validated; remote conversion has not started.
Google Colab is still blocked at interactive Google sign-in. No archive was
downloaded and no conversion, inference, training, or fine-tuning ran on the
Mac.

## Purpose

The raw-ONNX stage diagnostic localized the first phone failure to
vector-estimator session creation under ORT 1.25 / QAIRT 2.42 with CPU fallback
disabled. This lane asks two independent offline questions with the exact
matching QAIRT release:

1. Can QAIRT 2.42 generate an SM8850/HTP-v81 context for the exact vector
   estimator despite its unfavorable operator inventory?
2. Can it independently generate the vocoder context after the one already
   authenticated metadata-only repair required by QAIRT?

A vector failure cannot block the vocoder attempt. Each component has separate
content-addressed state beneath one durable Google Drive workspace.

## Frozen controls

| File | SHA-256 |
|---|---|
| `scripts/qairt242_colab_supertonic_remaining.py` | `9a18437aa5eab04a12632744af401d8cf6e17c6da633fba0dcb1ee354288e2b6` |
| `scripts/qairt242_colab_supertonic_duration_canary.py` | `bc39dc8fe71206177321c5fa9165cc3d7ea7d78ef3fbdb3b7e0add5f4a802ca2` |
| `qairt242_colab_supertonic_remaining_spec.json` | `6a843bcb2eeb7819d0e16b81fab16b7e2ab6093db51d413e7a5e624c9b8dcc40` |
| `qairt242_colab_supertonic_remaining.ipynb` | `5f1ae498b36c00f7b08037077961c2493ff21134e4db3874b6c2cf8301c8cf42` |
| `scripts/qairt242_colab_test_supertonic_remaining.py` | `7d156c3153f45bd953aa4cecfa62026728fab3736b89f25105c688db954f85b2` |

The notebook independently pins the other three uploaded control files before
execution. The remaining controller imports only the exact frozen duration
controller for its already-tested download, SDK, Python-environment, stage,
inventory, and deterministic-bundle helpers.

## Exact source and target contracts

- QAIRT archive: `v2.42.0.251225.zip`, 1,543,955,191 bytes, SHA-256
  `1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321`,
  build `2.42.0.251225135753_193295`.
- Supertonic archive: 128,774,318 bytes, SHA-256
  `82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427`.
- Vector estimator: 78,400,833 bytes, SHA-256
  `20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd`.
- Vocoder: 25,991,073 bytes, SHA-256
  `e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152`.
- Target: `soc_model=87`, `dsp_arch=v81`, unsigned PD, HTP, graph
  optimization 3, 8 MiB VTCM, relaxed FP16 precision.

All graph inputs and outputs are made concrete at latent/text length 32. Nodes,
initializers, and tensor dtypes are immutable. For the vocoder only, the
redundant `wav_tts` output entry is removed from `graph.value_info`; before
fixed-shape mutation that metadata-only identity is 25,991,018 bytes, SHA-256
`db8cd48bf21464fc23114941dfe4d1ed421754920823b5ff02eb0ff1889e20df`.

The vector graph's prior remains explicit: 22 `DynamicQuantizeLinear`, 38
`MatMulInteger`, and one `Softplus`. A remote converter failure is a valid
failed-closed result and must not trigger a QAIRT 2.45 substitution in this
lane.

## Resume and verification behavior

The component stages are source extraction, fixed metadata, QNN conversion,
x86_64 model-library generation, SM8850 context generation, and deterministic
deliverable assembly. A completed stage is reused only when its fingerprint and
every recorded output hash still match. Interrupted directories are
quarantined. Verification requires the exact six-stage set, exact manifest
schema, source identity, metadata repair, target, context hash, and negative
phone/NPU claims; a forged `complete` state, incomplete manifest, or fake source
identity fails closed.

Durable state is under:

```text
MyDrive/lmf_qairt242_colab_supertonic_remaining_v1/workspace/
  components/vector_estimator/
  components/vocoder/
```

Each successful component produces a component-specific result manifest and
deterministic bundle. Runtime binaries are intentionally excluded.

## Claim boundary

`context_generated_unexecuted` proves only that the pinned Linux toolchain
created a parseable offline context. It does not prove phone loading, HTP/NPU
execution, numerical equivalence, PCM, audio quality, latency, language
identity, unplugged durability, or integration. A generated artifact may be
loaded only with the exact QAIRT 2.42 runtime in a separate candidate package,
with CPU fallback disabled and app-UID HTP/FastRPC/DMA evidence required.

The separately isolated QAIRT 2.45 lane is expected to reach a first phone
canary sooner because all four matching contexts already exist. Its contexts
and runtime remain a separate package/UID/native namespace and must never be
mixed with the accepted QAIRT 2.42 runtime.

## Local validation

The local gate is lightweight and performs no archive download or conversion:

```bash
python3 scripts/qairt242_colab_test_supertonic_remaining.py
python3 -m py_compile \
  scripts/qairt242_colab_supertonic_remaining.py \
  scripts/qairt242_colab_test_supertonic_remaining.py
ruff check \
  scripts/qairt242_colab_supertonic_remaining.py \
  scripts/qairt242_colab_test_supertonic_remaining.py
```

Fifteen focused tests pass together with Python compilation, Ruff, both plan
renders, notebook JSON/code-cell validation, and `git diff --check`.
