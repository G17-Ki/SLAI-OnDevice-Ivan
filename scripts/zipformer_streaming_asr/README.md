# Streaming Zipformer EN/ZH reproducibility harness

This directory reproduces the CPU quality and latency gate for the streaming
English/Mandarin Zipformer checkpoint
`pfluo/k2fsa-zipformer-chinese-english-mixed` at the immutable revision
`6eb615ae77ecac05c5628d5c8ed7037c14a338d5`.

The downloader retrieves only the INT8 encoder, FP32 decoder, INT8 joiner, and
token table used by the experiment. It verifies every file against a recorded
SHA-256 digest. The benchmark verifies those digests again before inference, so
silently substituting another checkpoint is an error.

It also recognizes the common flat Sherpa layout
`encoder.int8.onnx`/`decoder.onnx`/`joiner.int8.onnx`/`tokens.txt`. Explicit
`--encoder`, `--decoder`, `--joiner`, and `--tokens` paths can select any other
layout; relative overrides are resolved below `--model-dir`. These layout
options do not weaken the default checkpoint verification.

## Requirements

- Python 3.11 on macOS or Linux;
- FFmpeg available on `PATH`;
- enough memory for the roughly 96 MiB of model files and the Sherpa runtime.

Create a dedicated environment:

```bash
python3.11 -m venv .venv-zipformer
source .venv-zipformer/bin/activate
python -m pip install -r scripts/zipformer_streaming_asr/requirements.txt
```

## Download the exact model

```bash
python scripts/zipformer_streaming_asr/download_model.py \
  --output-dir work/zipformer-en-zh-model
```

The resulting directory retains the upstream paths under `exp/` and
`data/lang_char_bpe/`. It also contains `model_manifest.json` with the pinned
revision, sizes, and hashes.

## Dataset manifest

The benchmark consumes an existing multilingual dataset directory and JSON
manifest. It selects only `en` and `zh`/`zh-*` rows and ignores other locales.
The manifest can either be a list of sample objects or an object with a
top-level `samples` list. Each selected row needs:

- `language`;
- `audio_path`, `audio`, or `path`;
- `reference`, `text`, or `transcript`;
- optionally `sample_id` and a duration field.

Paths may be relative to the dataset directory or the manifest directory. Both
the repository's `dataset_manifest.json` format and the original Common Voice
debug metadata selection are accepted.

Prepare the exact public Common Voice fixture used by the published comparison
without downloading the unrelated Whisper control models:

```bash
DATA_ROOT=work/multilingual-asr-mac-m2
mkdir -p "$DATA_ROOT/downloads/common_voice_17_debug"
hf download argmaxinc/common_voice_17_0-debug-zip \
  debug.zip metadata.json \
  --repo-type dataset \
  --revision e77e410d41a77f733efbaf7307bd0ff7ebd6d0af \
  --local-dir "$DATA_ROOT/downloads/common_voice_17_debug"
python scripts/multilingual_asr_mac_m2/prepare_common_voice_subset.py \
  --archive "$DATA_ROOT/downloads/common_voice_17_debug/debug.zip" \
  --metadata "$DATA_ROOT/downloads/common_voice_17_debug/metadata.json" \
  --output-dir "$DATA_ROOT/dataset"
```

The fixture builder verifies the archive and metadata hashes before extracting
three clips per language. Then run the Zipformer gate against the shared
English/Mandarin subset:

```bash
python scripts/zipformer_streaming_asr/benchmark.py \
  --model-dir work/zipformer-en-zh-model \
  --dataset-dir work/multilingual-asr-mac-m2/dataset \
  --manifest work/multilingual-asr-mac-m2/dataset/dataset_manifest.json \
  --output work/zipformer-en-zh-result.json
```

## Fixed method

The public gate intentionally has no flags that alter its core method:

- English and Mandarin only;
- CPU provider with exactly two Sherpa inference threads;
- greedy-search decoding;
- one untimed warm-up followed by three timed repeats per clip;
- FFmpeg conversion to mono 16 kHz signed PCM16 before model timing, followed
  by deterministic normalization to float32 in `[-1, 1)`;
- 660 ms of zero-valued tail audio before `input_finished()`;
- one additional streaming replay per clip in 100 ms input increments;
- process RSS sampled every 5 ms during model load and inference;
- Unicode NFKC, case-folding, and punctuation/symbol removal, with English WER
  and Mandarin CER aggregated once per clip.

The timed final-pass region includes stream creation, waveform and tail
acceptance, `input_finished()`, every ready decode call, and final transcript
retrieval. The incremental probe does not sleep: its
`first_nonempty_audio_fed_ms` is the amount of source audio accepted when the
first non-empty partial appeared, while
`first_nonempty_accumulated_decode_ms` is accumulated inference compute.

The output is strict UTF-8 JSON. Serialization rejects NaN and Infinity. It
contains all timed repeats, partial-probe observations, exact transcripts,
normalizer inputs, aggregate WER/CER counts, latency/RTF percentiles, model and
audio hashes, package versions, and RSS measurements.

For a deliberate comparison against a different checkpoint, add
`--allow-unpinned-model`. The output then records every actual and expected
hash, sets `checkpoint_matches_pinned_revision` to `false`, and leaves the
model repository/revision fields null rather than falsely assigning the pinned
provenance. Do not use this flag for the canonical reproduction run.

## Licenses

- The pinned Hugging Face model card declares Apache-2.0.
- Sherpa-ONNX is Apache-2.0.
- The repository's Common Voice 17 diagnostic fixture is CC0-1.0. If another
  manifest is supplied, its dataset license remains the operator's
  responsibility.

## Interpretation

This script measures the host CPU/Sherpa route. It does not establish QNN/HTP
placement or on-phone latency. Short clips with a reference shorter than the
model's streaming context can also expose checkpoint-specific blank-output
behavior; the raw per-clip transcript is retained so that such failures cannot
be hidden by aggregate scores.
