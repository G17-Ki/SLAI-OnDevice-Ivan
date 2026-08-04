# Android Qwen text companion

> **Deployment status (2026-07-29): superseded.** This directory remains the
> pinned llama.cpp runtime preparer, protocol test bed, and legacy Qwen3
> shell-owned diagnostic path. The current demo uses app-owned
> `Qwen2.5-1.5B-Instruct-Q4_K_M` inside `ai.liquid.lfmdemo.runtime`; see
> [`../../android-app/runtime-companion/README.md`](../../android-app/runtime-companion/README.md).
> Do not use the lifecycle below for the current standalone deployment.

This directory stages the older phone-local text server on the Nubia SM8850.
It is deliberately separate from the LFM audio server:

- Qwen3-1.7B Q4_0 runs on the phone CPU at `127.0.0.1:6020`;
- the official llama.cpp `b10083` Android arm64 release supplies the server
  and shared libraries used by the physically validated path;
- no executable, shared library, release archive, or model weight is stored in
  this repository;
- the process is started phone-side with redirected descriptors, but remains
  in `adbd`'s cgroup and is terminated when USB removal restarts `adbd`;
- it is not a boot service and must be restarted after a disconnect, `adbd`
  restart, or phone reboot.

This is a CPU fallback, not an NPU claim. The server command explicitly uses
zero GPU layers, four generation threads, four prompt threads, one HTTP thread,
one parallel slot, and a 2048-token context by default. These bounded defaults
match the current application's 768-token response ceiling and two-turn memory
policy; deployment still exposes smaller contexts for resource diagnostics.

## Reproducibility and trust boundary

[`assets.json`](assets.json) pins:

- `unsloth/Qwen3-1.7B-GGUF` revision
  `d7f544eead698dbd1f15126ef60b45a1e1933222`;
- `Qwen3-1.7B-Q4_0.gguf`, 1,056,782,912 bytes, SHA-256
  `c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5`;
- official llama.cpp release `b10083`, commit
  `846e991ec3c7ccec49112ff2c5b00b710e5f551d`;
- `llama-b10083-bin-android-arm64.tar.gz`, 74,822,792 bytes, SHA-256
  `c53e7989142bb1c56e29f1df780451810363b2cfae829827d4e2d79bb21c7189`;
- every required extracted runtime file, including all Android CPU variants
  and the MIT license.

The deployer rejects a different SoC, ABI, model, runtime archive, or extracted
runtime file. It copies only the allowlisted verified files to
`/data/local/tmp/lmf-text-companion/runtime-b10083` and sets
`LD_LIBRARY_PATH` explicitly because the Android release has no effective
runtime search path.

The model cache defaults to
`~/.cache/lmf-demo/android-text-companion/` and must remain outside the
repository. Upstream Apache-2.0 terms continue to govern Qwen and the GGUF.

## Prepare without a phone

Use an existing pinned file (preferred when it is already available):

```bash
python3 scripts/android_text_companion/prepare_model.py \
  --source /path/to/Qwen3-1.7B-Q4_0.gguf \
  --offline

python3 scripts/android_text_companion/prepare_runtime.py \
  --source /path/to/llama-b10083-bin-android-arm64.tar.gz \
  --offline
```

Without `--offline`, the same command can download the exact file from the
pinned Hugging Face revision. A partial or mismatched file is never promoted
into the verified cache.

## Legacy shell deployment

The phone must be visible to adb. From the repository root, the historical
Qwen3 diagnostic path is:

```bash
scripts/android_text_companion/deploy.sh \
  --serial YOUR_ADB_SERIAL \
  --model /path/to/Qwen3-1.7B-Q4_0.gguf \
  --runtime-archive /path/to/llama-b10083-bin-android-arm64.tar.gz \
  --offline
```

The deployer:

1. verifies the local model and official runtime archive;
2. confirms arm64 and SM8850;
3. safely extracts only the pinned runtime allowlist;
4. reuses already-correct phone files or atomically pushes and verifies them;
5. writes fixed model/runtime paths and starts the loopback server.

Use `--no-start` to stage only. Use `--forward` only for host-side debugging;
the phone app itself needs no adb forwarding.

Lifecycle commands:

```bash
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-text-companion/start.sh status
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-text-companion/start.sh restart
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-text-companion/start.sh log 100
adb -s YOUR_ADB_SERIAL shell sh /data/local/tmp/lmf-text-companion/start.sh stop
```

The lifecycle manager records one PID and checks `/proc/PID/cmdline` for the
exact server, model, and port before signalling it. It refuses to take over an
unmanaged listener on port 6020 and never calls `pkill` or `killall`.

## Reasoning and response length

Reasoning is disabled at server startup with `--reasoning off` and
`--reasoning-budget 0`. The response parser uses `--reasoning-format deepseek`
only to remove Qwen's otherwise-empty `<think></think>` wrapper from visible
content; it does not enable reasoning. The explicit mode flag is required:
budget zero alone still emitted the empty wrapper in the host control.
Requests produced by [`protocol.py`](protocol.py) additionally set
`chat_template_kwargs.enable_thinking=false`. This prevents Qwen thinking from
consuming the small context or delaying speech.

There is no ten-second text-output cap. Callers choose `max_tokens`; the helper
bounds it to 1-768, while the server context defaults to 2048 total tokens.
The larger default leaves room for a long response plus the application's
bounded two-turn memory. Deployment can select a smaller bounded context
through `--ctx-size`. Any context or response-budget change must be
re-profiled on the phone.

## API and probes

The companion is the native llama.cpp OpenAI-compatible server. Startup fixes
the REST model alias to `Qwen3-1.7B-Q4_0`, and health gates both readiness and
that exact identity:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

After deployment with `--forward`, probe non-streaming and streaming Chinese:

```bash
python3 scripts/android_text_companion/probe.py health

python3 scripts/android_text_companion/probe.py chat \
  --language zh \
  --prompt '中国的首都是哪里？' \
  --max-tokens 128

python3 scripts/android_text_companion/probe.py stream \
  --language zh \
  --prompt '用两句话介绍北京。' \
  --max-tokens 192
```

The streaming probe reports time to first text and total time. The
non-streaming probe fails if the response exposes `<think>` or a non-empty
`reasoning_content` field.

## Host gates

No phone or model inference is used by these tests:

```bash
scripts/android_text_companion/run_host_tests.sh
```

They compile the Python helpers, syntax-check both shell environments, test
Chinese SSE framing, validate request bounds and reasoning controls, exercise
model/runtime hash and safe-extraction handling, and confirm that no GGUF or
runtime archive is stored beside the scripts.
The host gates do not claim physical latency, memory, or multilingual answer
quality. Cable-removal survival has been tested and fails for this
`adb shell`-launched process. The required app-owned Android service is now the
runtime companion, whose installer uses the llama.cpp runtime prepared here but
stages the selected Qwen2.5 model in package-private storage.
