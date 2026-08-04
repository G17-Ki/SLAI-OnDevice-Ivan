# SenseVoice SM8850 QNN runtime gate

Audit date: 2026-07-23 (Asia/Shanghai)<br>
Scope: read-only host and upstream inspection. No phone files, repository history,
or remote compute jobs were changed.

## Decision

| Question | Decision |
|---|---|
| Is the exact runtime for the published SM8850 `model.bin` obtainable? | **Yes.** Qualcomm's public, unauthenticated QAIRT `2.40.0.251030` archive contains the complete HTP-v81 Android Stub/Skel set. |
| Can the model be smoke-tested as an app without building Sherpa first? | **Yes.** Sherpa upstream publishes a self-contained arm64 APK containing the matching model, QNN 2.40 runtime, ONNX Runtime, JNI bridge, and VAD. Treat it as an upstream smoke-test artifact, not as a file to republish here. |
| Can the model be profiled with `sherpa-onnx-offline` immediately? | **Not yet.** No released generic Android package contains a QNN-enabled CLI. Build Sherpa with QNN and binaries enabled, then stage the resulting executable and `libonnxruntime.so` with the exact QNN 2.40 runtime. |
| Is this blocked by HTP-v81 availability? | **No.** It is a packaging/build gate. The model and exact runtime exist; the missing local artifact is the QNN-enabled Sherpa CLI. |
| Should raw Qualcomm `.so` files be added to this public repository? | **No.** Qualcomm's bundled license permits object-code distribution only when incorporated into the developer's application and excludes standalone distribution. |

**Gate result: PASS for private engineering and phone smoke testing; HOLD for a
publicly vendored runtime bundle.** The safest next step is either the upstream
APK for an immediate app-level smoke test or a locally built QNN-enabled CLI for
controlled transcript, latency, memory, and error logging.

## 1. Exact Qualcomm runtime is publicly obtainable

Official archive:

- version: `2.40.0.251030`;
- URL: [Qualcomm QAIRT 2.40 community archive](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip);
- HTTP size: `1,403,532,983` bytes;
- server `Last-Modified`: 2025-11-03;
- server ETag: `9883387eece747967a4c48059550fd81`.

A one-byte range request succeeds without authentication. Qualcomm does not
expose an explicitly labelled SHA-256 or other cryptographic checksum for this
archive. The ETag is opaque and must **not** be represented as an MD5 digest.
For that reason this audit does not add an automatic QAIRT downloader.

Remote ZIP central-directory inspection, without downloading the 1.4 GB SDK,
confirmed these exact members:

| `$QNN_SDK_ROOT` relative path | Uncompressed bytes | ZIP CRC32 |
|---|---:|---:|
| `lib/aarch64-android/libQnnHtp.so` | 2,516,048 | `2431ac16` |
| `lib/aarch64-android/libQnnHtpPrepare.so` | 74,225,392 | `6c7c62a3` |
| `lib/aarch64-android/libQnnSystem.so` | 2,590,072 | `8da51863` |
| `lib/aarch64-android/libQnnHtpV81Stub.so` | 632,336 | `fa7f391c` |
| `lib/hexagon-v81/unsigned/libQnnHtpV81.so` | 13,213,720 | `0efc7a64` |
| `lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so` | 10,561,352 | `ec50feea` |

CRC32 values are useful for detecting accidental member corruption, but they
are not a substitute for an authoritative cryptographic archive checksum.

The current Mac has no configured exact SDK: `$QNN_SDK_ROOT` is unset. Android
platform/build tools are installed, but `sdkmanager --list_installed` lists no
Android NDK package. Thus no local CLI build was attempted in this read-only
gate.

## 2. Checksum-pinned Sherpa runtime mirrors

The Sherpa `asr-models-qnn` release publishes two small, exact-version support
archives:

| Asset | Bytes | SHA-256 | Contents relevant here |
|---|---:|---|---|
| `qnn-libs-2.40.0.251030.tar.bz2` | 51,539,798 | `1078c0833b1e948518ff433e081eda511475c7fd9a82c7798a862f59b95a7ce5` | QNN backend/System plus Stub/Skel files for v68 through v81 |
| `qnn-include-2.40.0.251030.tar.bz2` | 393,044 | `fafe1c9dc3f60393134f22f2c708a9ede287492fa20dedc22c9fabdcf5ebe70d` | QNN headers |

These assets do **not** provide `sherpa-onnx-offline` or
`libonnxruntime.so`. The runtime archive also does not carry a Qualcomm
license/NOTICE file. Its public availability does not override Qualcomm's
license, so do not vendor or republish it from this repository.

The existing project downloader remains appropriate for the model archive
because that GitHub asset has an authoritative SHA-256. No new runtime
downloader was added.

## 3. The generic Sherpa Android release is not the needed executable

The regular v1.13.4 release contains:

- `sherpa-onnx-v1.13.4-android.tar.bz2`;
- 45,068,327 bytes;
- SHA-256
  `7983fc3de23f6e64148f2fb05fa94a2efaa8c0516cc1573383dc5c7d4d2a43b0`.

It is unsuitable for this CLI test. The tag-pinned upstream release workflow:

1. sets only `SHERPA_ONNX_ENABLE_C_API=ON` before invoking the Android build;
2. leaves `SHERPA_ONNX_ENABLE_QNN` at its build-script default of `OFF`;
3. leaves `SHERPA_ONNX_ENABLE_BINARY` at its default of `OFF`; and
4. packages only `install/lib/*.so` into a multi-ABI `jniLibs` archive.

Therefore the generic archive is a CPU/JNI library package. It neither enables
the QNN code path nor contains `sherpa-onnx-offline`.

The upstream QNN build instructions require:

```bash
export QNN_SDK_ROOT=/private/path/to/qairt/2.40.0.251030
export ANDROID_NDK=/private/path/to/android-ndk
export SHERPA_ONNX_ENABLE_QNN=ON
export SHERPA_ONNX_ENABLE_BINARY=ON
./build-android-arm64-v8a.sh
```

Expected outputs:

```text
build-android-arm64-v8a/install/bin/sherpa-onnx-offline
build-android-arm64-v8a/install/lib/libonnxruntime.so
```

For the CLI path, `libsherpa-onnx-jni.so` is not needed. Sherpa's QNN code
includes QNN headers and loads the runtime dynamically, so the pre-generated
`model.bin` avoids any need to run Qualcomm's Linux-only context generator on
macOS. The local prerequisites still missing are the privately extracted exact
SDK and an Android NDK.

## 4. Upstream self-contained APK

Sherpa's APK catalog publishes this exact arm64 application:

[SenseVoice SM8850 ten-second QNN APK](https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/qnn-vad-asr-simulated-streaming/1.13.4/sherpa-onnx-1.13.4-qnn-arm64-v8a-simulated_streaming_asr-zh_en_ko_ja_yue-SM8850_10-seconds-sense_voice_2024_07_17_int8.apk)

- Sherpa version: `1.13.4`;
- ABI: `arm64-v8a`;
- size: `229,898,342` bytes;
- Hugging Face LFS SHA-256:
  `4c667715666c8c05ebf404fde49bb787af39ff9d8ac359baa28413a41c5ac86e`.

Remote APK inventory confirms it bundles:

- `libonnxruntime.so` and `libsherpa-onnx-jni.so`;
- `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`;
- `libQnnHtpV81Stub.so` and `libQnnHtpV81Skel.so`;
- the exact 10-second SM8850 SenseVoice `model.bin` and `tokens.txt`;
- Silero VAD.

The five Qualcomm library sizes and CRC32 values match the corresponding files
in the official QAIRT 2.40 archive exactly. The APK does not contain
`sherpa-onnx-offline`; it is a JNI application.

The Hugging Face repository declares no package-level license, and the APK has
the model's `LICENSE` but no Qualcomm license/NOTICE. Use it as a convenient
upstream smoke-test artifact. Do not copy the APK or its extracted Qualcomm
libraries into this public repository without a distribution review.

## 5. Context/runtime compatibility policy

The selected model archive is already pinned in
[`qnn_sm8850_manifest.json`](qnn_sm8850_manifest.json):

- archive SHA-256:
  `ecbc1ffba39f8e23582b79a199d12e8455425a22ef7b6b18c535ce25fcff2d64`;
- `model.bin` SHA-256:
  `522763135f536bc43e9d122bac39254f49df4afbb003462d6d8bcdb77051de30`;
- target: SM8850 / HTP v81;
- fixed context: 10 seconds;
- generator/runtime version: QNN `2.40.0.251030`.

Sherpa explicitly documents that `model.bin` is OS-independent but dependent
on the target SoC and the QNN SDK version used to generate it. Its guide warns
that changing the QNN SDK version requires regenerating `model.bin`.

QNN 2.40 provides runtime checks including `QnnContext_validateBinary`,
`QNN_CONTEXT_ERROR_BINARY_VERSION`, and permissive/strict binary compatibility
modes. Those APIs let the runtime reject incompatible or suboptimal caches;
they are not a published guarantee that a 2.40 cache works with arbitrary older
or newer runtimes.

Engineering policy for the first measurement:

- pair this context with the exact 2.40 runtime;
- do not mix it with the Nubia OEM QNN 2.37 libraries;
- do not silently substitute QAIRT 2.48 or another newer runtime;
- if another version is tested, record it as a separate compatibility
  experiment and require successful binary validation plus transcript parity.

The broader v81 toolchain and AI Hub audit is recorded in
[`qairt_v81_support.md`](../../npu/qairt_v81_support.md).

## 6. Reproducible CLI smoke-test layout

After the QNN-enabled Sherpa build exists, stage these files outside the
repository:

```text
stage/
  sherpa-onnx-offline
  libonnxruntime.so
  libQnnHtp.so
  libQnnHtpPrepare.so
  libQnnSystem.so
  libQnnHtpV81Stub.so
  libQnnHtpV81Skel.so
  model/
    model.bin
    tokens.txt
    test_wavs/
```

On the phone, from inside `stage/`:

```bash
export LD_LIBRARY_PATH="$PWD"
export ADSP_LIBRARY_PATH="$PWD;${ADSP_LIBRARY_PATH:-}"

./sherpa-onnx-offline \
  --provider=qnn \
  --tokens=./model/tokens.txt \
  --sense-voice-language=en \
  --sense-voice-use-itn=true \
  --sense-voice.qnn-backend-lib=./libQnnHtp.so \
  --sense-voice.qnn-system-lib=./libQnnSystem.so \
  --sense-voice.qnn-context-binary=./model/model.bin \
  ./model/test_wavs/en.wav
```

The semicolon in `ADSP_LIBRARY_PATH` is required. Valid fixed SenseVoice locale
values are `auto`, `zh`, `en`, `ja`, `ko`, and `yue`.

For the first controlled phone run, record:

1. exact hashes of every staged file;
2. cold recognizer creation time and three warm decode repeats;
3. stdout/stderr, relevant `logcat`, process RSS/VmHWM, and thermal state;
4. transcript equality against the local golden output for EN, JA, KO, and ZH;
5. any QNN binary-version/configuration/suboptimal error separately from an
   application or model error.

## 7. Distribution boundary

The `LICENSE.pdf` embedded in Qualcomm's official QAIRT 2.40 archive grants a
license to distribute the Software solely in object-code form **as incorporated
in the developer's software application** and states that it does not grant a
license to distribute or sublicense the Software on a standalone basis.

Practical repository rule:

- private SDK download/extraction: permitted for development under the license;
- application/APK incorporating required object code: contemplated by the
  license, subject to all its conditions and project legal review;
- raw QAIRT/QNN `.so` files or a standalone runtime tarball in this public
  repository: do not publish.

This is why the reproducible public path should document exact acquisition,
hash validation where an authoritative hash exists, build flags, and staging,
while keeping Qualcomm binaries out of Git.

## Sources

- [Qualcomm QAIRT 2.40 community archive](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip)
- [Sherpa QNN build instructions](https://k2-fsa.github.io/sherpa/onnx/qnn/build.html)
- [Sherpa `model.bin` phone guide](https://k2-fsa.github.io/sherpa/onnx/qnn/run-executables-on-your-phone-binary.html)
- [Sherpa QNN model catalog](https://k2-fsa.github.io/sherpa/onnx/qnn/models.html)
- [Sherpa `asr-models-qnn` assets](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn)
- [Sherpa `asr-models-qnn-binary` assets](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn-binary)
- [Sherpa v1.13.4 release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.4)
- [Tag-pinned generic Android workflow](https://github.com/k2-fsa/sherpa-onnx/blob/142807252687d81b40d6315f23470a1512a00de3/.github/workflows/android.yaml)
- [Tag-pinned Android build script](https://github.com/k2-fsa/sherpa-onnx/blob/142807252687d81b40d6315f23470a1512a00de3/build-android-arm64-v8a.sh)
- [Upstream SenseVoice SM8850 APK](https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/qnn-vad-asr-simulated-streaming/1.13.4/sherpa-onnx-1.13.4-qnn-arm64-v8a-simulated_streaming_asr-zh_en_ko_ja_yue-SM8850_10-seconds-sense_voice_2024_07_17_int8.apk)
