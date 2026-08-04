# Lane B B3/B4 Colab inference harness

This harness executes the frozen B1 protocol without training, conversion, or
local inference. It builds the exact llama.cpp commit plus one audited
memory-allocation patch on a Colab CUDA runtime, renders prompts with each
model's pinned Hugging Face tokenizer revision, keeps only one verified GGUF at
a time, and writes append-only raw JSONL.

## 1. Start a GPU Colab runtime

Make this repository available in Colab, then set the two paths below. The lane
directory must contain the committed `b1_prereg.json` and `protocol/` manifests.

```bash
REPO=/content/LMF_sprint-b
LANE=$REPO/workstreams/sub2b_sprint/sprint/out/b
WORK=/content/lane_b_b34
```

Build and validate the pinned, exactly patched runtime:

```bash
bash "$LANE/b34_colab_setup.sh" "$LANE" "$WORK"
```

The setup performs these binding checks:

- `b1_prereg.json` SHA-256 is
  `6107258882449913b9136ad9eba930e88952192b6df08a301eafc00f03f6255b`.
- The two frozen prompt manifests match their embedded hashes and contain
  exactly 80 MCQs plus 40 native generations per language.
- llama.cpp starts at
  `11924d4c17abc27383376a1ac6a24fa3e36c1c0c`; the original
  `tools/perplexity/perplexity.cpp` SHA-256 is
  `380aadd9d8000bf1b95cc3a338708385ac75240a5ba1c1f2bb34b208892a1777`.
- `llama_mcq_memory.patch` has SHA-256
  `2539f2da48852fa44fdacdba26154e144a01b3986ff7c88dfe6974ac3e4f9dd7`,
  produces source SHA-256
  `65051abf6eb3e68d26cb2f24b62e2fa1af1bf7ddea96a2db53f944c3f0232b78`,
  and leaves exactly that one source path modified before the CUDA build.
- The build receipt hashes both `bin/llama-perplexity` and the Linux shared
  implementation library `bin/libllama-perplexity-impl.so`; every run
  revalidates both files. The launcher alone is insufficient evidence because
  `perplexity.cpp` is compiled into the shared implementation library.
- `CMakeCache.txt` is hashed and parsed fail-closed: `CMAKE_HOME_DIRECTORY`
  must resolve to the pinned llama.cpp source, `CMAKE_CACHEFILE_DIR` must resolve
  to the actual build directory, the supplied binary directory must be exactly
  that build's `bin/`, and `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, and
  `BUILD_SHARED_LIBS=ON` are required again before every inference run.
- `transformers==5.14.1` and `huggingface_hub==1.26.0` are used for rendering
  and downloads.

Receipts land in `$WORK/protocol_validation.json` and
`$WORK/build_receipt.json`.

## 2. Run one arm

Use the same sequence for each model. Rendered prompts are model-specific
because the frozen tokenizer/chat template is model-specific.

```bash
MODEL=qwen35_08b
BIN=$WORK/llama.cpp/build-b34/bin

python "$LANE/b34_harness.py" render "$MODEL" \
  --lane-dir "$LANE" --work-dir "$WORK"

python "$LANE/b34_harness.py" prepare-model "$MODEL" \
  --lane-dir "$LANE" --work-dir "$WORK"

python "$LANE/b34_harness.py" run-mcq "$MODEL" \
  --lane-dir "$LANE" --work-dir "$WORK" --llama-bin-dir "$BIN"

python "$LANE/b34_harness.py" run-generation "$MODEL" \
  --lane-dir "$LANE" --work-dir "$WORK" --llama-bin-dir "$BIN"
```

The official arm is complete only when MCQ has `n=400` and generation has
`n_present=200`, `is_complete=true`. A generation smoke test is available with
`--limit 1`; its receipt is deliberately `PARTIAL` and is not B3/B4 evidence.
Interrupted generation can continue with `--resume`. MCQ output is never
overwritten or resumed; rerun it to a new `--output` path if needed.

After copying the raw outputs and receipts to persistent storage, remove the
current GGUF before the next arm:

```bash
python "$LANE/b34_harness.py" purge-model --work-dir "$WORK"
```

Run arms in this order:

```text
B3: qwen35_08b, qwen25_15b, qwen3_06b
B4: falcon_h1_15b
```

The expected GGUFs are 0.38–1.12 GB. `prepare-model` refuses to download a new
arm while a receipted current model exists, enforcing the one-GGUF-at-a-time
storage rule.

## 3. Frozen execution details

MCQ uses the pinned `llama-perplexity --multiple-choice` path with the exact
one-file memory patch above and passes the serialized task set through its
explicit `--binary-file` option:

```text
context=4096, batch=1024, ubatch=128, parallel=4, GPU layers=999, seed=1729
continuations=" A", " B", " C", " D"
```

The execution history is append-only: `b1_execution_resource_correction.json`
records the original 4096/4096/512 allocation failure,
`b1_execution_resource_correction_fallback.json` records the corrected
4096/1024/128 failure and the attempted 3072-token fallback, and
`b1_execution_tool_memory_patch.json` records that fallback's failure plus the
audited patch and restoration of the frozen 4096-token context. All three
attempts ended before any task-result row or raw result JSONL existed.

A complete Qwen3.5 arm was then produced under harness `9f99a37e...`, but an
independent build-provenance audit found that its receipt did not prove the
caller-supplied binary directory came from the exact patched source checkout.
`b1_execution_build_provenance_addendum.json` preserves and explicitly
supersedes that 400-MCQ/200-generation arm. It is excluded from B3–B6; every
official arm must be regenerated under hardened harness `3e9b03db...`, which
binds the CMake source, build and binary directories as one chain.

Stock llama.cpp allocated a float logits buffer for the full context times the
248,320-token vocabulary. The audited patch changes only that buffer: it starts
empty and is resized to
`i_logits*n_vocab` after the MCQ batch has been assembled and validated, before
memory clear and decode. The frozen 4096-token context is therefore restored.
Task order, batch assembly, decode order, candidates, length-normalized
likelihood calculation, GPU offload, and seed are unchanged.

The patched and pristine `llama-perplexity` launcher may have identical bytes:
the launcher is built from unchanged `main.cpp`. The memory patch must instead
change `libllama-perplexity-impl.so`, so its exact path and SHA-256 are mandatory
in `build_receipt.json` and are checked again immediately before inference.

The otherwise-stock implementation chooses the maximum length-normalized
continuation log-likelihood. It exposes only cumulative accuracy, not the four
per-choice values. The harness reconstructs each item's binary correctness from
successive integer cumulative-correct counts, retains every raw trace line, and
writes per-choice log-probability fields as `null`. It never fabricates a
predicted letter or a log-probability vector.

Generation uses `llama-server /completion` with the frozen raw rendered prompt:

```text
n_predict=192, temperature=0, top_p=1, top_k=0, min_p=0,
repeat_penalty=1, seed=1729, stream=false, cache_prompt=false
```

The request includes the frozen stop strings. The exact server response and
unmodified completion text are retained in JSONL.

For Qwen3.5-0.8B and Qwen3-0.6B, rendering passes
`enable_thinking=false` to the pinned chat template and fails unless the
rendered prompt contains the template's empty `<think></think>` block. Prompting
the model not to reason is not accepted as a substitute.

## 4. Outputs to preserve

For each model, preserve the whole `$WORK/raw/<model_id>/` and
`$WORK/rendered/<model_id>/` directories plus:

```text
$WORK/build_receipt.json
$WORK/protocol_validation.json
```

The rendered directory is evidence, not disposable cache. It must include
`global_mmlu.rendered.jsonl`, `native_generation.rendered.jsonl`,
`global_mmlu.multiple_choice.bin`, and `render_receipt.json`; the final
assembler binds every raw prompt and MCQ trace row back to these files.

Every raw row contains the model/GGUF hash, llama.cpp commit, item and language,
rendered prompt, prompt SHA-256, source manifest row, and raw scoring or server
response. Every receipt contains hashes for its JSONL and runtime log.

Do not call an incomplete or failed arm measured. Record it as `UNKNOWN` or as a
documented Colab-run failure, consistent with B1.
