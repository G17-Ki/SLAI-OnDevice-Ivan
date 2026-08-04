# Persistent Russian ASR WebSocket

This directory contains a bounded workaround for the Nubia/Android CPU result
where the Russian INT8 transducer returns an empty transcript at decode batch
one but succeeds when the same utterance is supplied twice.

The stock Sherpa WebSocket server cannot guarantee a two-item batch from a
client: it posts a decode as soon as each complete utterance enters the queue.
The pinned patch therefore adds `--duplicate-each-stream=true`. When enabled,
the server creates an identical shadow stream for every real stream, decodes
both in one call, and returns only the real result. Equal-length pairs keep the
active packed transducer batch at two or more through the final frame.

This is a workaround, not a claim that the model itself requires batch two.
The same Sherpa-ONNX 1.13.4 model and ONNX Runtime 1.27.0 pass batch one on the
Apple M2. The remaining fault boundary is the Android arm64 execution path.

## Build

The build stays outside the repository and uses at most two local jobs:

```bash
LMF_BUILD_JOBS=2 scripts/russian_asr_websocket/build_android_server.sh
```

Default output:

```text
~/.cache/lmf-demo/russian-asr-websocket/output/
  sherpa-onnx-1.13.4-android-arm64/
    sherpa-onnx-offline-websocket-server
    libonnxruntime.so
    manifest.json
```

The script pins:

- Sherpa-ONNX `v1.13.4`, commit
  `142807252687d81b40d6315f23470a1512a00de3`;
- ONNX Runtime Android `1.27.0` from the existing verified NPU build cache;
- Android NDK `27.0.12077973` and API 27;
- the exact patch hash in the output manifest.

Model weights remain external and must not be committed.

## Phone launch command

After staging the two runtime files and the external Russian model to the
phone, launch the server from a phone-side `tmux` session if available, or via
the existing checked-PID/nohup lifecycle wrapper. The essential command is:

```bash
export RUNNER=/data/local/tmp/lmf-ru-ws/runtime
export MODEL=/data/local/tmp/lmf-sherpa-ru-20250420

LD_LIBRARY_PATH="$RUNNER" \
  "$RUNNER/sherpa-onnx-offline-websocket-server" \
  --provider=cpu \
  --tokens="$MODEL/tokens.txt" \
  --encoder="$MODEL/encoder.int8.onnx" \
  --decoder="$MODEL/decoder.onnx" \
  --joiner="$MODEL/joiner.int8.onnx" \
  --num-threads=2 \
  --decoding-method=greedy_search \
  --port=6007 \
  --num-io-threads=1 \
  --num-work-threads=1 \
  --max-batch-size=1 \
  --max-utterance-length=30 \
  --duplicate-each-stream=true \
  --log-file=/data/local/tmp/lmf-ru-ws/server.log
```

Binding remains IPv4 wildcard because that is how the upstream C++ server is
implemented. Android/browser integration should connect only to
`ws://127.0.0.1:6007`; do not expose or forward the port except during a
bounded diagnostic.

## Persistent host diagnostic

Install `websockets` in a host virtual environment, then keep one connection
open across repeated utterances:

```bash
python3 scripts/russian_asr_websocket/persistent_client.py \
  --server-addr 127.0.0.1 \
  --server-port 6007 \
  --repeat 5 \
  --require-nonempty \
  /path/to/russian.wav
```

The client reports one JSON line per turn with end-to-end WebSocket latency and
the unmodified Sherpa result. It sends `Done` only after every turn, which
verifies that model initialization is amortized across the whole session.

## Tests

```bash
python3 scripts/test_russian_asr_websocket.py
bash -n scripts/russian_asr_websocket/build_android_server.sh
git -C "$HOME/.cache/lmf-demo/npu-multilingual/build/source/sherpa-onnx" \
  apply --check \
  "$PWD/scripts/russian_asr_websocket/sherpa_onnx_v1.13.4_duplicate_stream.patch"
```

The protocol tests require no phone or model. A real acceptance run must still
confirm non-empty Cyrillic output, five consecutive turns, steady-state
latency, RSS, and process survival after USB removal.
