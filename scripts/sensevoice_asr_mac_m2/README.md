# SenseVoiceSmall INT8 Mac benchmark

This harness evaluates the canonical Sherpa-ONNX SenseVoiceSmall INT8 package
on the English, Japanese, Korean, and Mandarin subset of the existing pinned
21-clip Common Voice diagnostic. SenseVoice does not support the Spanish,
German, or French cells, so the measured subset contains 12 clips.

The model artifact is pinned by all of the following:

- GitHub release tag: `k2-fsa/sherpa-onnx@asr-models`;
- asset ID: `288366523`;
- archive name: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`;
- archive SHA-256: `7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e`.

The dated conversion archive does not declare the exact upstream
`iic/SenseVoiceSmall` source revision. The archive hash therefore pins the
actual executable model, while the missing source revision is recorded rather
than guessed.

Do not substitute the similarly named `2025-09-09` archive without a separate
quality run. Its bundled README says it comes from
`ASLP-lab/WSYue-ASR/sensevoice_small_yue`, a Cantonese-specialized derivative,
not the canonical checkpoint used here.

## Resource bounds

The default run uses two CPU threads, three warm repeats per clip, and only one
model-bearing worker process at a time. Each language is run in a fresh worker,
which gives a clean incremental peak-RSS baseline and prevents four locale-
specific recognizers from being resident together. No training is performed.

Audio is converted to mono 16 kHz float32 PCM before the timed region. Timing
starts before Sherpa stream creation and includes waveform acceptance, feature
extraction, model decode, and retrieval of the final transcript. SenseVoice is
offline/non-streaming in this path, so `first_stable_ms` equals `final_ms`.

## Reproduce

On Apple Silicon with Python 3.11 and FFmpeg installed:

```bash
python3.11 -m venv .venv-sensevoice
source .venv-sensevoice/bin/activate
python -m pip install -r scripts/sensevoice_asr_mac_m2/requirements.txt
PYTHON_BIN="$PWD/.venv-sensevoice/bin/python" \
  scripts/sensevoice_asr_mac_m2/run_benchmark.sh
```

The default work directory is `work/sensevoice-asr-mac-m2`, outside the report
tree. Override it with the first positional argument. `ASR_CPU_THREADS` and
`ASR_REPEATS` are available for controlled experiments; the published run uses
their defaults of two and three.

The fixture is pinned to
`argmaxinc/common_voice_17_0-debug-zip@e77e410d41a77f733efbaf7307bd0ff7ebd6d0af`
and declares CC0-1.0. The same Unicode NFKC/case-fold/punctuation-removal
normalizer used by the Whisper experiment computes English WER and CJK CER.

## Licenses

- Sherpa-ONNX runtime: Apache-2.0.
- SenseVoice model weights: custom FunASR Model Open Source License Agreement
  v1.1, pinned in the report to FunASR commit
  `58830eca4012644aac0c3218c3ccc7d98f003fda`. It requires attribution and
  retention of the model name. It is not represented as an OSI-approved
  license, so publication and product review should treat it separately from
  the Apache-licensed runtime.
- Evaluation audio: CC0-1.0.

The raw JSON retains every timed repeat, exact transcript, normalizer inputs,
file hashes, package versions, host facts, and per-language memory record.

## Official SM8850 / HTP-v81 context

Sherpa also publishes a pre-generated ten-second QNN context for the exact
canonical model and SM8850/HTP-v81. Download and verify it without placing the
254 MiB extracted context in the repository:

```bash
MODEL_DIR=$(scripts/sensevoice_asr_mac_m2/download_qnn_sm8850.sh)
python scripts/sensevoice_asr_mac_m2/inventory_qnn_sm8850.py \
  --model-dir "$MODEL_DIR" \
  --archive "$(dirname "$MODEL_DIR")/$(basename "$MODEL_DIR").tar.bz2" \
  --output work/sensevoice-qnn-sm8850-manifest.json
```

The pinned archive is GitHub asset `326303090`, 162,023,574 bytes, SHA-256
`ecbc1ffba39f8e23582b79a199d12e8455425a22ef7b6b18c535ce25fcff2d64`.
It contains a fixed ten-second `model.bin`; shorter inputs are padded and longer
inputs are truncated. The context was generated with QNN SDK
`2.40.0.251030`, so use matching 2.40 runtime libraries and the v81 Stub/Skel
files. See the result report for the complete phone deployment command plan.
