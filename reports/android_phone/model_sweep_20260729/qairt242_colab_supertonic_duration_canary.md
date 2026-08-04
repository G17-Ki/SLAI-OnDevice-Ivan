# QAIRT 2.42 Colab Supertonic duration canary

Status: **prepared and locally validated; not executed in Colab**.  This is an
optional ahead-of-time context experiment, not a deployment blocker.

The exact raw duration ONNX has already executed on the NX809J through ORT
1.25, the app-owned QAIRT 2.42 QNN backend, and HTP-v81 with CPU fallback
disabled. See
[`SUPERTONIC_DURATION_RAW_ONNX_QNN_RESULT.md`](SUPERTONIC_DURATION_RAW_ONNX_QNN_RESULT.md).
The raw-ONNX route remains the primary path for extending the phone probe to
the text encoder, vector estimator, vocoder, and real audio. The Colab route is
retained to answer a narrower question: does an exact offline QAIRT 2.42
context improve session startup/cache behavior while preserving a reproducible
SM8850 artifact?

## Artifacts and trust roots

Upload these three files to Colab:

| File | SHA-256 |
|---|---|
| `qairt242_colab_supertonic_duration_canary.ipynb` | `e7570e779c103b75bb976472d3b586835b5d2c608221d820a5f7a385b26b3947` |
| `scripts/qairt242_colab_supertonic_duration_canary.py` | `bc39dc8fe71206177321c5fa9165cc3d7ea7d78ef3fbdb3b7e0add5f4a802ca2` |
| `qairt242_colab_supertonic_duration_spec.json` | `08e1425f1daba9626827dfce3b072f1191ff840c1e0f555743f39614546702ec` |

The notebook independently pins the controller and spec hashes before it
runs. The controller then pins:

- QAIRT community archive `v2.42.0.251225.zip`, 1,543,955,191 bytes,
  SHA-256
  `1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321`;
- Supertonic archive
  `sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2`, SHA-256
  `82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427`;
- the sole extracted graph, `duration_predictor.int8.onnx`, 3,700,147
  bytes, SHA-256
  `c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db`.

No QAI Hub key, Qualcomm credential, or local credential file is used or
recorded. Google Drive authentication is handled interactively by Colab. The
portable result bundle intentionally excludes QAIRT runtime binaries.

## Exact graph and target contract

Only the smallest Supertonic graph is permitted:

| Tensor | Source | QNN boundary | Fixed shape |
|---|---|---|---|
| `text_ids` | INT64 | INT32 | `1 x 32` |
| `style_dp` | FP32 | FP32 | `1 x 8 x 16` |
| `text_mask` | FP32 | FP32 | `1 x 1 x 32` |
| `duration` | FP32 | FP32 | `1` |

The INT64-to-INT32 boundary matches the earlier AI Hub
`--truncate_64bit_io` contract and the retained fixed fixture. The derivation
checks that all ONNX nodes and initializers are byte-identical before and
after the I/O-only edit.

Offline preparation is fixed to:

- QAIRT build `2.42.0.251225135753_193295`;
- Ubuntu 22.04 x86_64 and Python 3.10;
- clang 14;
- ONNX 1.16.1, ONNX Runtime 1.18.0, ONNX Simplifier 0.4.36;
- SM8850 `soc_model=87`, `dsp_arch=v81`, unsigned PD;
- HTP graph optimization `O=3`, 8 MiB VTCM, relaxed FP16 precision.

QAIRT's dependency checker carries several older package pins that are not a
reliable Python 3.10 wheel set and conflict with ONNX 1.16.1 metadata. The
controller installs pinned Python-3.10-compatible matplotlib 3.8.4, PyYAML
6.0.2, and protobuf 3.20.3 before invoking the checker, then verifies the full
declared framework package inventory.

## Manual Colab procedure

1. In Google Colab, select a standard x86_64 Ubuntu runtime. A GPU is not
   required. Upload and open
   `qairt242_colab_supertonic_duration_canary.ipynb`.
2. Run the first cells to mount Google Drive. When prompted, upload the exact
   controller and spec from the table above. A pre-existing control file with
   a different hash causes a fail-closed stop; it is never overwritten.
3. Run the host bootstrap cell. It installs Python 3.10, clang 14, CMake,
   curl, unzip, and bzip2 in Colab only.
4. Run the plan cell. Confirm that it names only `duration_predictor`, QAIRT
   2.42, `soc_model=87`, and `dsp_arch=v81`.
5. Run the canary cell. The first run downloads about 1.55 GB of QAIRT plus
   the Supertonic release archive, verifies both hashes, extracts only the
   duration graph from the model archive, and generates the context.
6. Run the verification cell. Save the printed context and bundle SHA-256
   values with the phone experiment that consumes the context.

Durable files live under:

```text
MyDrive/lmf_qairt242_colab_supertonic_duration_v1/workspace/
```

The portable outputs are:

```text
stages/deliverable/result_manifest.json
stages/deliverable/qairt242_colab_result.tar.gz
stages/deliverable/qairt242_colab_result.tar.gz.sha256
```

## Resume and failure behavior

If Colab disconnects, reconnect, remount Drive, rerun the control validation
and host bootstrap, then rerun the canary cell. A completed stage is reused
only if its fingerprint and every output size/hash still match. Interrupted
partial directories are moved to `quarantine/` before a retry. A completed but
modified artifact stops the run rather than being regenerated silently.

Failures write:

```text
qairt242_colab_state.json
qairt242_colab_failure_report.json
```

Preserve those files and stage logs. Do not respond to a converter or context
failure by substituting QAIRT 2.45, another graph, or an unverified archive.
The canary is allowed to end as a documented QAIRT 2.42 incompatibility.

## Claim boundary and next comparison

A successful manifest status is `context_generated_unexecuted`. It proves
that the exact pinned Linux toolchain produced a parseable context under the
SM8850/v81 configuration. It does **not** prove phone loading, HTP execution,
numeric agreement, latency, audio, or accepted-app integration.

If Colab produces a context, load it only in the isolated Supertonic candidate
with the exact app-owned QAIRT 2.42 runtime. Reuse the raw-ONNX duration
fixture and compare:

1. session/context load time and first execution;
2. warm execution latency and output error;
3. process RSS/PSS and resident context bytes;
4. app-UID HTP libraries, FastRPC FD, and DMA buffers;
5. force-stop/relaunch and reboot behavior.

The precompiled context is promoted only if that device comparison passes.
Regardless of the result, the next end-to-end TTS gate remains the other three
raw source graphs followed by finite, non-silent multilingual PCM.

## Lightweight local validation

No archive download or conversion is needed on the Mac:

```bash
python3 scripts/qairt242_colab_test_supertonic_duration_canary.py
python3 -m py_compile \
  scripts/qairt242_colab_supertonic_duration_canary.py \
  scripts/qairt242_colab_test_supertonic_duration_canary.py
ruff check \
  scripts/qairt242_colab_supertonic_duration_canary.py \
  scripts/qairt242_colab_test_supertonic_duration_canary.py
```
