# Exact-SM8850 voice experiments

This guide covers the standalone-TTS experiments under
`reports/android_phone/streaming_response/`. They target the Qualcomm AI Hub
device `Snapdragon 8 Elite Gen 5 QRD`, Android 16, SM8850/HTP v81, with QAIRT
2.45. The QRD is a same-chipset reference device; it is not the Nubia phone,
AR1, or the glasses.

## Isolated environment

Do not add the voice-model dependencies to the repository's main Python 3.13
environment. The completed Piper runs recorded this control environment:

| Package | Version |
|---|---|
| Python | 3.11.11 |
| `qai-hub` | 0.52.0 |
| `qai-hub-models` | 0.58.0 |
| `qai-hub-models-cli` | 0.58.0 |
| PyTorch | 2.9.1 |
| NumPy | 1.26.4 |

Create a fresh environment rather than layering these packages over another
ML stack:

```bash
python3.11 -m venv .venv-qaihm058
.venv-qaihm058/bin/python -m pip install --upgrade pip
.venv-qaihm058/bin/python -m pip install \
  'qai-hub==0.52.0' \
  'qai-hub-models==0.58.0' \
  'qai-hub-models-cli==0.58.0' \
  'torch==2.9.1' \
  'numpy==1.26.4'
.venv-qaihm058/bin/python -m pip check
```

The Supertonic fixture is separately self-describing. Its checked manifest
records Python 3.11.11, NumPy 2.4.6, ONNX Runtime 1.27.0, and two local
threads. Do not silently regenerate that fixture in the Piper environment and
compare it as if only the target runtime changed.

## Persistent work directory

Keep model caches, serialized graphs, AI Hub state, downloaded job logs, and
remote tensors outside the public repository. For example:

```bash
export LFM_DEMO_WORK_ROOT="${LFM_DEMO_WORK_ROOT:-$HOME/lfm-demo-work}"
export AIHUB_RUN_ROOT="$LFM_DEMO_WORK_ROOT/aihub-voice-v81"
mkdir -p "$AIHUB_RUN_ROOT"
```

Public reports retain job IDs, hashes, metrics, and credential-free reduced
evidence. They intentionally omit temporary signed upload URLs and local
absolute paths.

## Cloud-session rule

Every AI Hub or Colab operation must run in detached `tmux`. State is written
immediately after each accepted submission, before polling. Never start a new
compile merely because a local poller stopped; resume from the persisted job
ID.

The component runner accepts exactly one language and one component per
persistent output directory:

```bash
tmux new-session -d -s aihub-piper-it-encoder \
  "cd /path/to/LMF_demo && \
   PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
   .venv-qaihm058/bin/python scripts/run_aihub_tts_v81.py \
     --language it \
     --component encoder \
     --device 'Snapdragon 8 Elite Gen 5 QRD' \
     --device-os 16 \
     --qairt-version 2.45 \
     --output-dir '$AIHUB_RUN_ROOT/piper-it-encoder' \
     > '$AIHUB_RUN_ROOT/piper-it-encoder.log' 2>&1"
```

Use a distinct output directory for SDP, flow, and decoder. If a process is
interrupted, resume one saved item instead of overwriting its state:

```bash
.venv-qaihm058/bin/python scripts/resume_aihub_tts_v81.py \
  --run-dir "$AIHUB_RUN_ROOT/piper-it-encoder" \
  --item piper_it_encoder
```

The public Piper-English Voice AI release is handled separately because it
already contains six compiled QNN contexts. The resume-safe
`scripts/profile_piper_en_release_context.py` validates the pinned archive and
graph hashes, uploads those contexts unchanged, and submits profile jobs only.
Its exact-SM8850 result is 34.337 ms for the acoustic core and 35.104 ms for one
call each of all six graphs, with 2,089/2,089 HTP operations and zero fallback.
The package has no matching inputs/goldens, so do not add arbitrary-tensor
inference or describe the result as English quality. Full evidence is in
[`piper_en_v81/RESULT.md`](../reports/android_phone/streaming_response/piper_en_v81/RESULT.md).

## Numerical fixtures and accumulated chains

The generic random Piper flow sample is invalid: dense random attention drives
the projected log scale beyond Float32's exponential range. Preserve that raw
failure as provenance, then use
`scripts/run_aihub_piper_flow_valid_fixture.py` against the existing linked
flow context. It feeds a fixed phoneme sequence through the serialized encoder
and SDP, constructs binary monotonic attention, and submits inference only.

After all four linked contexts exist,
`scripts/run_aihub_piper_accumulated_chain.py` can execute the real Italian
fixture through remote encoder, SDP, flow, and repeated decoder calls. It uses
the official 40-frame decoder payload and 12-frame overlap rule, keeps local
and remote WAVs, and is also inference-only.

The completed corrected 79-ID run, every inference job ID, numerical split,
and fixed-locale ASR sanity check are documented in
[`PRODUCTION_CHAIN_RESULT.md`](../reports/android_phone/streaming_response/piper_v81/PRODUCTION_CHAIN_RESULT.md).
Its final waveform passes the strict 3%/3% gate and Faster-Whisper Small
recovers the exact normalized sentence from local and remote WAVs. The chain
still retains a one-element Flow warning and remains a one-phrase QRD result,
not a phone or broad-quality acceptance.

Supertonic uses the separate fixture and resume-safe chain tools:

- `scripts/prepare_supertonic_chain_fixture.py`
- `scripts/run_aihub_supertonic_chain_v81.py`
- `scripts/supertonic_chain_common.py`

The completed English and Japanese one-phrase reports are
[`ACCUMULATED_CHAIN_RESULT.md`](../reports/android_phone/streaming_response/supertonic_v81/ACCUMULATED_CHAIN_RESULT.md)
and
[`ja_paris/RESULT.md`](../reports/android_phone/streaming_response/supertonic_v81/ja_paris/RESULT.md).

## Interpretation

Component profile sums are neural-compute planning bounds. They exclude text
preprocessing or phonemization, host orchestration, tensor transfers, context
load, repeated decoder calls, audio-device startup, and speaker output. Do not
publish them as phone text-to-first-audio latency. The physical acceptance
protocol and go/no-go thresholds are in
[`FIRST_AUDIO_3S_DECISION_MEMO.md`](../reports/android_phone/streaming_response/FIRST_AUDIO_3S_DECISION_MEMO.md).
