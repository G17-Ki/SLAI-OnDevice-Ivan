# Backbone / LLM deployment checkpoint — 2026-07-29

## Decision

Use `Qwen2.5-1.5B-Instruct-Q4_K_M` as the default compact direct-response
candidate, while retaining the existing Qwen3-1.7B model as a rollback artifact
until Qwen2.5 passes the physical-phone latency, memory, unplug, and restart
gates. Do not describe the direct small-model route as robust Russian yet.

The authoritative multilingual comparison is
`reports/android_phone/streaming_response/model_candidate_followup/FINAL_DECISION.md`.
Its numbers are same-environment Colab CPU comparisons, not Nubia timings:

| Candidate | Priority quality | Mandarin | Russian | Median / P95 first speakable | Process RSS | Decision |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B Q4_K_M | **50/63** | **32/33** | 18/30 | **1.90 / 4.83 s** | **1,840,856 KiB** | Default candidate; Russian below floor |
| Qwen3-1.7B Q4_0, thinking off | 37/63 | 22/33 | 15/30 | 2.27 / 5.60 s | 2,126,404 KiB | Rollback only |
| Qwen3-1.7B, reasoning budget 16 | 31/63 | 21/33 | 10/30 | 5.19 / 8.93 s | 2,163,676 KiB | Reject; slower and leaked reasoning in 11/63 runs |
| LFM2.5-1.2B Q4_0 | 31/63 | — | — | Earlier broad suite was fast | — | Reject as multilingual quality baseline |
| Qwen3.5-0.8B Q4_0 | 18/28 in the earlier broad suite | — | — | 187 / 273 ms | about 1,346 MiB | Reject for quality and memory |
| Qwen3-0.6B Q4 | Control only | — | — | Fast control | — | Too weak for production speech answers |

The selected model is pinned at 1,117,320,736 bytes with SHA-256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
in `model_candidate_followup/candidate_manifest.json`.

## Untethered runtime state

- LFM is already app-owned: `LfmService` executes the packaged native server
  from the app native-library directory, and its model files live in private
  app storage.
- The old Qwen deployment under `/data/local/tmp/lmf-text-companion` is
  shell-owned and launched by `adb shell`; it is therefore not a valid unplugged
  runtime. It is explicitly CPU-only.
- The active implementation uses a separate same-signature foreground runtime
  APK. This avoids native-library collisions between the current llama.cpp
  b10083 archive and the LFM fork in the main APK while preserving the existing
  loopback API on `127.0.0.1:6020`.
- `RuntimeService.java` now selects Qwen2.5, executes
  `libqwenserver.so` from immutable `nativeLibraryDir`, and expects the GGUF at
  private `files/text/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`.
- The Web UI uses the matching alias and no longer sends Qwen3-only `/no_think`
  or `chat_template_kwargs.enable_thinking` controls to Qwen2.5.

The initial CPU server parameters are context 2048, four generation and batch
threads, batch 128, micro-batch 64, one parallel slot, loopback-only port 6020,
Jinja chat templates, and no GPU layers. CPU is deliberate until an app-UID
Genie/HTP package is proven on this phone.

## Native-payload build gate fixed and passed

The native staging tool initially emitted `.so` files directly under its output
directory. Android `jniLibs.srcDir` requires an ABI directory, so the real
payload build failed at `mergeDebugNativeLibs` with `out ... is not an ABI`.

`scripts/prepare_runtime_companion_jni.py` now stages verified payloads under
`arm64-v8a/` and records their APK paths in its manifest. With the pinned cache:

- all ASR, QNN, and llama.cpp files passed byte-count and SHA-256 verification;
- the extracted native payload is about 684 MiB;
- `:runtime-companion:assembleDebug` passed with the complete payload;
- the resulting compressed debug APK is about 112 MiB;
- APK SHA-256 is
  `327e9068413e002884683e9ce377716445969f489108298ac7b74dc07c03d46c`;
- APK inspection confirmed `libqwenserver.so`, `libllama-server-impl.so`,
  `libasrrunner.so`, `libonnxruntime.so`, and the QNN v81 stub under
  `lib/arm64-v8a/`.

Build command used, with the three local paths supplied by the environment:

```sh
JAVA_HOME="$JDK17_HOME" \
ANDROID_HOME="$ANDROID_SDK_ROOT" \
RUNTIME_COMPANION_JNI_DIR="$RUNTIME_COMPANION_JNI_DIR" \
./gradlew :runtime-companion:assembleDebug
```

The code-only build also passes. JDK 26 is not compatible with the current
Android JDK-image transform; use the installed JDK 17 for reproducible builds.

## Qualcomm NPU status

General compact-Qwen NPU deployment is technically credible but is not yet
integrated in this repository. The published exact-SM8850 Qwen3-1.7B W4A16
Genie/QAIRT entry reports 62.03 generated tokens/s, about 1,249.9 prompt
tokens/s, and TTFT from 25.6 to 819 ms depending on prompt shape. This is a
reference result, not evidence that the current app executes the model on HTP.

Current blockers:

- no Genie runtime or W4A16 context binaries are present in the app;
- no app-UID HTP execution has been demonstrated for Qwen;
- the phone OEM QNN stack is 2.37 while the published Qwen package requires
  QNN 2.45, so the compatible runtime must be bundled and validated;
- the existing LFM NPU work covers isolated fixed-shape probes, not an
  end-to-end cached backbone; its FP16 detokenizer is numerically unusable.

Do not force tokenization, orchestration, sampling, or LFM FP32 detokenization
onto the NPU. GPU/Vulkan is lower priority because current evidence shows no
useful advantage over CPU for this model class and the cached b10083 Android
runtime has no GPU backend.

## Remaining acceptance gates

1. Install the companion APK and atomically copy the verified Qwen2.5 GGUF into
   companion-private storage.
2. Prove the listener and model process run under the companion app UID, not
   `shell`.
3. Verify `/health` and `/v1/models` identify the exact alias and model hash.
4. Run ten Exact turns, then repeat in airplane mode after physically removing
   USB.
5. Repeat after force-stop/reopen and after reboot plus manual reopen.
6. Record cold load, TTFT, generation rate, RSS/PSS, peak memory, battery and
   thermal state on the Nubia.
7. Run the current Mandarin/Russian suite on the phone. If Russian remains
   below its quality floor, keep it explicitly gated and pursue the planned
   offline native Russian route rather than hiding the failure with prompting.
8. In parallel, acquire the official Qwen W4A16 Genie bundle through Qualcomm
   tooling, first prove app-UID HTP execution, then compare it against CPU using
   identical prompts, contexts, memory accounting, and thermal conditions.

No unplugged or NPU completion claim is made until these physical-phone gates
pass.

## 00:53 CST — repeatable private installer and live ownership proof

`scripts/install_app_owned_runtime.sh` now reproduces the app-owned deployment
without downloading or converting anything on the Mac. It:

- rebuilds the companion from the verified `arm64-v8a` native staging tree, or
  verifies a supplied APK contains the required Qwen, llama.cpp, ASR, ONNX, and
  QNN libraries;
- pins Qwen2.5 from the candidate manifest and verifies its byte count and
  SHA-256 before use;
- obtains SenseVoice/QNN and Russian Zipformer paths only through the existing
  offline pinned preparers;
- copies every model/runtime artifact through a temporary phone staging file,
  verifies it there, copies it into the companion UID, verifies it again, and
  atomically renames it inside private storage;
- stops only the two known legacy lifecycle scripts, rather than killing an
  unverified listener or arbitrary process;
- launches the companion Activity, validates `/health` plus `/v1/models`, and
  compares the Qwen process UID against the companion package UID.

The idempotence pass on a Nubia NX809J (ADB serial redacted) verified and skipped all
already-correct private assets:

- Qwen2.5 Q4_K_M GGUF;
- SenseVoice `model.bin` and `tokens.txt`;
- QNN 2.40 HTP v81 skeleton;
- Russian Zipformer encoder, decoder, joiner, and tokens.

Live validation after a force-stop/relaunch returned:

```text
model_ids: [Qwen2.5-1.5B-Instruct-Q4_K_M]
ASR app_owned: true
SenseVoice provider: qnn, processor: htp_v81, ready: true
Russian Zipformer provider: cpu, ready: true
companion UID: 10514
Qwen PID: 28771
Qwen process UID: 10514
```

This proves app ownership, exact model identity, private artifact integrity, and
reopen-to-start behavior while USB is connected. A physical USB removal and
post-unplug inference turn are still required before claiming cable-independent
operation.

The ASR `/health` result above proves that the app-owned server sees a complete
QNN bundle and reports its intended backend. It does **not** by itself prove a
successful HTP transcript. The first app-UID inference attempt subsequently
exposed a `libcdsprpc` linker-namespace dependency and is being repaired through
the companion manifest/runtime integration. Keep the QNN execution gate open
until a real transcript succeeds under UID 10514.

## 01:04 CST — app-owned Qwen2.5 physical multilingual sweep

The capability-aware profiler now sends Qwen3 thinking controls only to Qwen3
models. Qwen2.5 requests omit both `/no_think` and
`chat_template_kwargs.enable_thinking`; three regression tests cover Qwen2.5,
Qwen3, and fail-closed unsupported thinking mode.

The 28-case native-language suite ran through the app-owned Qwen2.5 PID on the
Nubia and is preserved at
`reports/android_phone/streaming_response/qwen2_5_1_5b_q4_k_m_app_owned_phone.json`.

| Metric | Physical-phone result |
|---|---:|
| Automated quality | 23 / 28 |
| Manually corrected quality | 24 / 28 |
| Median / P95 first token | 304 / 475 ms |
| Median / P95 first speakable text | 672 / 1,253 ms |
| Median / P95 completion | 707 / 1,601 ms |
| Qwen process PSS / RSS after sweep | 1,995,935 / 1,998,876 KiB |
| One measured short-turn prompt/decode rate | 84.4 / 33.2 tokens/s |

The automated Fuji failure is scorer-only: the answer gave the correct 3,776 m
as `3, 776`, which the old regex did not accept. Real factual failures were:

- French: Berlin was answered as Berne;
- Japanese: Ottawa was answered as Vancouver;
- Korean: Berlin was answered as Munich;
- Korean: Jupiter was answered as the Sun.

Mandarin passed 3/3 and Russian passed 3/3 in this small screen. This is useful
phone evidence, but it does not override the larger Mandarin/Russian study or
prove robust native quality across all languages. The Japanese/Korean/French
misses mean Qwen2.5 should remain quality-gated rather than advertised as a
universal multilingual factual agent.

Compared with the earlier Qwen3-1.7B phone measurement of 2,368,470 KiB PSS,
Qwen2.5 used 372,535 KiB less PSS, a reduction of about 15.7%, while delivering
sub-second median complete text on this short suite.
