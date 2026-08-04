# QAIRT 2.42 Qwen2.5 Genie/HTP lane

Status: **runnable Colab scaffold; host integrity tests passed; Colab build and phone execution not yet run**.

This lane uses the already locked official `Qwen/Qwen2.5-1.5B-Instruct-GGUF` Q4_K_M artifact as the primary model. The smaller 0.5B Q4_0 entry remains an unapproved footprint follow-up; it cannot run from its template lock.

## Exact primary source

- Revision: `91cad51170dc346986eccefdc2dd33a9da36ead9`
- File: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- Bytes: `1,117,320,736`
- SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
- License: Apache-2.0

The workspace candidate manifest pinned this identity. The cached local file was rehashed to the same SHA-256 without running conversion.

The controller also inspects the complete GGUF header before QAIRT import. The locked file has GGUF v3, 26 metadata entries, 339 tensors, and this exact mixed inventory:

| Tensor type | Count |
|---|---:|
| F32 | 141 |
| Q4_K | 169 |
| Q6_K | 29 |

It validates all 339 tensor names, 28 Qwen2.5 layers, tokenizer cardinalities, architecture dimensions, special-token IDs, RoPE theta, and RMS epsilon. Any unknown or changed GGML type fails closed.

## Why Q4_K_M is accepted by pinned QAIRT 2.42

The exact QAIRT archive is `v2.42.0.251225.zip`, 1,543,955,191 bytes, SHA-256 `1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321`.

The controller verifies these relevant members before importing QAIRT:

- `gen_ai_builder_factory.py` (`35df4aef...aac95`) detects `.gguf`, invokes `GGUFBuilder.build_from_gguf()`, maps `Qwen2ForCausalLM`, and returns `QwenBuilderHTP`.
- `gguf_builder.py` (`2500182e...ffbe`) accepts a Q4_K_M filename through its supported `Q4_K` substring gate.
- `utils.py` (`c65651fd...ac17`) declares both `Q4_K` and `Q6_K` supported.
- `ggml.py` (`e7fc1712...cf251`) implements `dequantize_q4_k`, `dequantize_q6_k`, and dispatches GGML type IDs 12 and 14.
- `qwen/builder.py` (`aaef1d3b...3b6b`) defines `QwenBuilderHTP(GenAIBuilderHTP)`.

Thus the mixed-tensor format is supported in source. Actual conversion/compilation compatibility remains a Colab result, not a pre-run claim.

## Build route and target

The only enabled route is:

```text
locked GGUF
  -> QAIRT 2.42 GGUFBuilder
  -> QwenBuilderHTP
  -> target chipset:SM8850;dsp_arch:v81;soc_model:87
  -> LLMContainer.save()
```

The controller uses a durable QAIRT `cache_root`, preserves GGUF-generated ONNX/encoding artifacts, and reuses a completed stage only when its fingerprint and every recorded file hash still match. It never falls back to QAIRT 2.45 or CPU.

Run the notebook in an Ubuntu 22.04 x86_64 high-RAM Colab runtime with at least 24 GB RAM, 24 GB free in Drive, and 24 GB free in `/content`:

[`qairt242_colab_qwen25_genie.ipynb`](qairt242_colab_qwen25_genie.ipynb)

The notebook uploads and verifies these controls:

| Artifact | SHA-256 |
|---|---|
| `qairt242_colab_qwen25_genie.ipynb` | `65c05731a40a4208661a5f1df3f939ee462b3cf3ad3443832e8fe634ef95b8a7` |
| `scripts/qairt242_colab_qwen25_genie.py` | `f5f602276289ec289d9b90595273739f82448e7fa1b45e6b259dc89963c8c5bd` |
| `qairt242_qwen25_genie_spec.json` | `8b25f22513a78aa9cea7898720ead903e41a77ef4475506450e17fa816c4565a` |
| `qairt242_qwen25_15b_genie_source_lock.json` | `cd8af4ced04ff562fbf9b2916a2d2065dfa092c3b4f8e78693e952eb1aa65e5e` |

Expected durable outputs are:

```text
workspace/stages/container_build/container/
workspace/stages/deliverable/result_manifest.json
workspace/stages/deliverable/container.SHA256SUMS
workspace/qairt_cache/
```

The success status is deliberately `context_generated_unexecuted`. The manifest always records:

```json
{
  "phone_execution_proven": false,
  "npu_execution_proven": false
}
```

## Launch handoff

After interactive Google/Colab CLI authorization is refreshed, the first action is to rerun the read-only `colab-cli sessions` check. Allocate a named high-RAM runtime only after that check succeeds.

The safest prepared build route is still the notebook above: select an Ubuntu 22.04 x86_64 high-RAM runtime, mount Drive, upload the three hash-pinned controls, and run the cells in order. There is no Qwen-specific CLI orchestrator yet. The standalone remote worker keeps workspace and scratch under `/content`, does not mount Drive, and does not package or download the result, so it is less durable than the notebook route.

The separately pinned `qairt242_colab_qwen25_preflight.py` is not currently invoked by the notebook or remote worker before prerequisite installation. The controller's own host preflight still fails closed before model/QAIRT download or conversion. Treat the earlier preflight sequencing as an explicit launch gap, not as evidence that a Colab build has run.

## Validation and remaining gates

Focused host-only validation passed on 2026-07-30:

```text
python3 scripts/qairt242_colab_test_qwen25_genie.py
..............
Ran 14 tests in 0.899s
OK
```

Those tests rehash the local 1.5B GGUF, validate its full header/tensor contract, verify the pinned QAIRT source-member hashes and Q4_K/Q6_K implementations, exercise approved/unapproved lock gates, compile the embedded build helper, reject forged completed states/manifests/source identities/stage fingerprints, and verify the worker/notebook/launch-state/boundary hash chain. Standalone completion verification also rehashes the locked GGUF and QAIRT archive, recomputes both stage fingerprints, checks exact stage/build/deliverable/container inventories, and revalidates the recorded builder/config/context evidence.

Remaining gates:

1. Run the high-RAM Colab build and preserve the result manifest/logs.
2. Stage the exact container plus the pinned QAIRT 2.42 Genie/HTP Android runtime in an isolated app-owned phone directory.
3. Disable CPU fallback and capture app-UID HTP libraries, FastRPC/DMA evidence, first-token and token-rate latency, memory, and reboot persistence.
4. Run native multilingual response-quality acceptance before promotion into the end-to-end speech agent.

No device action, network download, QAIRT conversion, training, or fine-tuning was performed on the Mac while preparing this lane.
