# App-owned ASR and response-model runtime

`runtime-companion` is the second APK in the unplugged phone demo. It moves the
ASR and compact text-model processes out of the Android `shell` UID and into a
foreground service owned by an application UID.

| Field | Value |
|---|---|
| Package | `ai.liquid.lfmdemo.runtime` |
| Version | `0.1.0-unplugged-runtime` (code 1) |
| Launcher label | `LFM Offline Runtime` |
| Service | `ai.liquid.lfmdemo.runtime.RuntimeService` |
| Start action | `ai.liquid.lfmdemo.runtime.START` |
| Caller permission | `ai.liquid.lfmdemo.permission.START_RUNTIME` (signature) |

The main `ai.liquid.lfmdemo` app starts this service explicitly when its
activity opens. The companion also has a small launcher activity for manual
startup and diagnostics. It is not a boot service; after reboot, unlock the
phone and reopen the main app.

## What it runs

| Endpoint | Backend | Processor | Current verified scope |
|---|---|---|---|
| `127.0.0.1:6010` | SenseVoiceSmall INT8 ten-second QNN context | HTP v81 | EN/ZH/JA/KO exact fixtures, tags, and full turns |
| `127.0.0.1:6010` | Russian Zipformer INT8 | CPU | Russian native-script fixture |
| `127.0.0.1:6020` | Qwen2.5-1.5B-Instruct Q4_K_M through llama.cpp | CPU | Exact model identity and physical-phone prompt sweep |

All listeners are loopback-only. The app does not need ADB forwarding or a
host process for inference.

Only SenseVoice is an integrated NPU path. Russian ASR and Qwen are deliberate
CPU fallbacks. The existing AI Hub Qwen and LFM component profiles are not
runtime payloads in this APK and do not establish app-UID HTP execution.

## Why this is a separate APK

The main APK already contains the Liquid LFM llama.cpp fork. The response model
uses the pinned official llama.cpp `b10083` Android runtime, which has colliding
native-library names and incompatible runtime assumptions. Keeping Qwen in a
second, same-signature APK isolates those libraries while preserving the
existing loopback API.

Android also restricts execution from mutable application storage. The
companion therefore packages verified native executables as extracted
`arm64-v8a` native libraries:

- `libqwenserver.so` and its llama.cpp shared libraries;
- `libasrrunner.so` and `libonnxruntime.so`;
- QNN 2.40 HTP libraries;
- `libQnnHtpV81Skel.so` for the SM8850 DSP.

Model weights remain separate and are copied into package-private storage by
the installer.

## QNN requirements resolved by this package

App-owned HTP execution initially failed even though the context and ordinary
QNN libraries were present. Two Android integration requirements are now part
of the companion:

1. the manifest declares the public vendor dependency `libcdsprpc.so` with
   `<uses-native-library>` so QNN can reach FastRPC through the app linker
   namespace;
2. `libQnnHtpV81Skel.so` is an immutable extracted APK native library rather
   than a file in mutable app-private storage.

The SenseVoice context also has an exact ten-second input contract. The HTTP
server validates canonical 16 kHz mono PCM16 WAV and zero-pads shorter requests
to exactly 320,044 bytes before invoking QNN. This restored exact fixture
parity on the physical phone.

## External build inputs

Large native payloads and model files are intentionally absent from Git. The
default cache layout is:

```text
~/.cache/lmf-demo/
├── android-asr-companion/
├── android-text-companion/
│   ├── models/qwen2.5-1.5b-instruct-q4_k_m.gguf
│   └── runtime/llama-b10083/
└── npu-multilingual/
    ├── bundle/
    └── build/output/sherpa-onnx-1.13.4-android-arm64-qnn/
```

The pinned Qwen2.5 metadata is in
`reports/android_phone/streaming_response/model_candidate_followup/candidate_manifest.json`.
The QNN, SenseVoice, Sherpa, ONNX Runtime, and llama.cpp metadata is inherited
from the existing preparation manifests. Every copied file is verified by byte
count and SHA-256.

Prepare the caches as described in `docs/ANDROID_DEPLOYMENT.md`. Heavy model
conversion is not performed by this module; the local build only stages
already-pinned runtime files and compiles ordinary Android Java resources.

## Recommended install path

From the repository root, with the caches prepared:

```bash
scripts/install_app_owned_runtime.sh --serial YOUR_ADB_SERIAL
```

The installer:

- rejects non-arm64 and non-SM8850 devices;
- verifies the Qwen2.5 GGUF and every ASR/QNN/native-runtime input;
- creates the JNI staging tree in a temporary directory outside the checkout;
- builds with JDK 17 and at most two Gradle workers;
- inspects the APK for the required native payload;
- uses `adb install -r` and does not clear package data;
- reuses already-correct private files;
- verifies temporary, private-partial, and final copies before atomic rename;
- launches the foreground service;
- validates Qwen health and exact model identity;
- runs real English, Mandarin, Japanese, Korean, and Russian ASR fixtures; and
- confirms process UID ownership.

The default operation is offline after the caches are seeded. Useful controls:

```text
--skip-build     reuse an existing APK after payload inspection
--skip-install   do not reinstall the APK
--skip-assets    do not update private model files
--no-start       install/stage without runtime health checks
--keep-forwards  keep temporary diagnostic ADB forwards
```

The installer stops only the two known legacy lifecycle roots under
`/data/local/tmp`. It does not use broad `pkill` or `killall` matching and does
not clear either application's data.

## Direct build

Normally the installer creates and supplies the verified JNI staging directory.
For an already-prepared staging tree:

```bash
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME="$HOME/Library/Android/sdk" \
  RUNTIME_COMPANION_JNI_DIR=/absolute/path/to/arm64-v8a-parent \
  android-app/gradlew --max-workers=2 \
  -p android-app :runtime-companion:assembleDebug
```

`RUNTIME_COMPANION_JNI_DIR` must point to a directory containing the
`arm64-v8a/` subdirectory emitted by
`scripts/prepare_runtime_companion_jni.py`. Do not point Gradle directly at the
ABI directory.

The APK output is:

```text
android-app/runtime-companion/build/outputs/apk/debug/runtime-companion-debug.apk
```

## Private phone layout

The installer writes these package-relative destinations:

```text
files/
├── asr/models/
│   ├── sensevoice/
│   │   ├── model.bin
│   │   └── tokens.txt
│   └── russian/
│       ├── encoder.int8.onnx
│       ├── decoder.onnx
│       ├── joiner.int8.onnx
│       └── tokens.txt
└── text/
    ├── models/qwen2.5-1.5b-instruct-q4_k_m.gguf
    ├── home/
    ├── tmp/
    └── qwen-server.log
```

Do not copy production assets to `/data/local/tmp` as their final location.
That path is used only as a short-lived ADB staging area before a verified
`run-as` copy into private storage.

## ASR HTTP contract

Recognition uses the same stable envelope as the older diagnostic companion:

```text
POST /asr?language=en|zh|ja|ko|ru|auto|qnn-auto
Content-Type: audio/wav
Content-Length: BYTES
```

Input must be canonical 16,000 Hz, mono, signed PCM16 WAV with a matching
44-byte RIFF header. The service rejects malformed framing, missing or duplicate
lengths, empty PCM, and inputs beyond the fixed ten-second maximum.

`en`, `zh`, `ja`, and `ko` select `sensevoice_qnn`; `ru` selects
`zipformer_ru_cpu`. `qnn-auto` runs only SenseVoice for the main app's
low-latency first-stage language gate. `auto` can return both independent
candidates and does not invent a calibrated cross-model confidence score.

The stable transcript location is:

```text
candidates[i].result.text
```

### Execution-aware health

`GET /health` distinguishes disk state from executed state:

- `artifacts_ready`: all required files for that backend exist;
- `execution_verified`: the current service process produced a non-empty
  transcript through that backend;
- `ready`: artifacts and current-process execution are both verified;
- `service_uid` and `last_runner_uid`: ownership evidence;
- `runner_owned_by_service`: both UIDs match;
- `input_contract`: records the fixed-ten-second zero-padding rule.

After a service restart, artifacts can remain ready while execution readiness
returns to false. This is deliberate: file presence is not NPU proof. A real
EN/ZH/JA/KO transcript and language-tag parity plus QNN/HTP logs is required
before claiming SenseVoice execution.

## Qwen server contract

The companion starts the official llama.cpp server with:

```text
model alias: Qwen2.5-1.5B-Instruct-Q4_K_M
context: 2048
threads / batch threads: 4 / 4
batch / micro-batch: 128 / 64
parallel slots: 1
GPU layers: 0
host / port: 127.0.0.1 / 6020
```

Supported endpoints used by the app and installer are:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Qwen2.5 does not use Qwen3's `/no_think` suffix or
`chat_template_kwargs.enable_thinking` controls. Client capability detection
must keep those Qwen3-only fields out of Qwen2.5 requests.

## Physical evidence and open gates

The 2026-07-29 phone run verified both ASR backends under companion UID 10514:

- English SenseVoice QNN/HTP exact transcript;
- Mandarin SenseVoice QNN/HTP exact transcript;
- Japanese SenseVoice QNN/HTP exact transcript and `<|ja|>` tag;
- Korean SenseVoice QNN/HTP exact transcript and `<|ko|>` tag;
- Russian Zipformer CPU native-script transcript.

The QNN log contains `Initializing HtpProvider`; the English fixture reported
an inference RTF near 0.009. The app-owned Qwen2.5 sweep measured 24/28 after
manual correction, 304 ms median first token, and 672 ms median first speakable
text. These are physical NX809J results, not AI Hub proxy timings.

The final pair also passed cold reboot, first unlock, main-only launch, and
EN/ZH/RU/JA/KO airplane-mode turns. Still open for the complete release claim:

- physical unplug plus full UI turns;
- final-main-hash host-ADB shutdown repetition;
- captured-PCM or human listening validation of JA/KO audible language quality;
- broad multilingual quality beyond the small phone screen.

The app-owned process architecture removes the known `adbd` cgroup dependency,
but these remaining physical gates must be completed before calling a retained
release cable-independent.

## Legacy tools

`scripts/android_asr_companion/` and `scripts/android_text_companion/` remain
useful for pinned-asset preparation, protocol tests, and historical diagnostics.
Their lifecycle scripts start shell-owned processes under `/data/local/tmp` and
must not be used as the current standalone runtime.

## Security and release boundary

The current APK is a debuggable research build. Its exported service is guarded
by a signature permission and its inference endpoints are loopback-only, but it
has not undergone production hardening. No runtime APK or checksum-pinned
`0.6.0` release pair is retained under `android-app/releases/` yet.
