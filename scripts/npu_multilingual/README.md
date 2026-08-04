# SM8850 multilingual NPU tools

This directory prepares and gates the pinned SenseVoiceSmall INT8 ten-second
QNN context used by the current app-owned runtime. Qualcomm runtime and model
payloads are intentionally stored outside the repository.

## Current integration status — 2026-07-29

The production path for this repository is no longer the raw ADB smoke or the
shell-owned ASR companion. `ai.liquid.lfmdemo.runtime` packages the verified
runner, ONNX Runtime, QNN 2.40 libraries, and HTP v81 DSP skeleton in an Android
APK, with the SenseVoice context in companion-private storage.

Physical NX809J acceptance passed for:

- English SenseVoice through QNN/HTP v81 with the exact fixture transcript;
- Mandarin SenseVoice through QNN/HTP v81 with the exact fixture transcript;
- Japanese SenseVoice through QNN/HTP v81 with the exact fixture transcript
  and `<|ja|>` tag;
- Korean SenseVoice through QNN/HTP v81 with the exact fixture transcript and
  `<|ko|>` tag;
- process ownership by the runtime-companion application UID.

The QNN log contains `Initializing HtpProvider`. Runtime health becomes ready
only after a successful non-empty transcript in the current service process;
artifact presence is reported separately.

The deployed SenseVoice route does not cover Russian, French, Italian, Spanish,
or German. Russian
uses the verified CPU Zipformer fallback. No full LFM, Qwen, or TTS NPU route is
integrated, and AI Hub component profiles are not end-to-end application proof.
See `reports/progress_20260729/ASR_QNN_AUDIT.md` and
[`../../android-app/runtime-companion/README.md`](../../android-app/runtime-companion/README.md).

## 1. Prepare the external bundle

```bash
python3 scripts/npu_multilingual/prepare_sensevoice_sm8850_bundle.py
```

The default root is `$XDG_CACHE_HOME/lmf-demo/npu-multilingual/bundle`, or
`$HOME/.cache/lmf-demo/npu-multilingual/bundle` when `XDG_CACHE_HOME` is unset.
The preparer verifies every archive and extracted model hash and refuses a root
inside this repository.

To include the upstream self-contained APK for a separate app-level smoke test:

```bash
python3 scripts/npu_multilingual/prepare_sensevoice_sm8850_bundle.py \
  --include-upstream-apk
```

The APK is an upstream test artifact, not part of the LMF demo and not an asset
to republish from this repository.

## 2. Install the Android NDK and build the QNN runner

Install the pinned NDK outside the repository:

```bash
sdkmanager --sdk_root="$HOME/.cache/lmf-demo/android-sdk" \
  "ndk;27.0.12077973"
```

Then build with at most two local jobs:

```bash
LMF_BUILD_JOBS=2 scripts/npu_multilingual/build_sherpa_qnn_android.sh
```

The build pins Sherpa-ONNX 1.13.4 commit
`142807252687d81b40d6315f23470a1512a00de3`, ONNX Runtime 1.27.0, Android
API 27, arm64-v8a, and `SHERPA_ONNX_ENABLE_QNN=ON`. Its external output has a
manifest but deliberately excludes Qualcomm objects and model data.

## 3. Inspect the historical fail-closed selector

This helper predates the app-owned integration and remains a deterministic
research gate. With no phone or no accepted smoke result, it returns CPU:

```bash
python3 scripts/npu_multilingual/select_android_asr_backend.py \
  --language en \
  --bundle-manifest "$HOME/.cache/lmf-demo/npu-multilingual/bundle/bundle-manifest.json"
```

Broad Auto is always CPU because SenseVoice does not support Russian, French,
Italian, or Spanish:

```bash
python3 scripts/npu_multilingual/select_android_asr_backend.py \
  --language auto --auto-scope all
```

An explicitly limited English/Mandarin Auto request can become QNN-eligible
after every device, artifact, runtime, and phone-smoke gate passes:

```bash
python3 scripts/npu_multilingual/select_android_asr_backend.py \
  --language auto --auto-scope zh_en \
  --bundle-manifest "$HOME/.cache/lmf-demo/npu-multilingual/bundle/bundle-manifest.json" \
  --smoke-result /path/to/reviewed-phone-smoke.json
```

## 4. Run the raw preparation smoke

With exactly one SM8850 phone connected:

```bash
python3 scripts/npu_multilingual/run_phone_qnn_smoke.py \
  --bundle-manifest "$HOME/.cache/lmf-demo/npu-multilingual/bundle/bundle-manifest.json" \
  --runner-dir "$HOME/.cache/lmf-demo/npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn" \
  --output work/sensevoice-qnn-phone-smoke.json
```

This raw harness alone cannot authorize the application's QNN route. It records
`raw_pass` after successful execution, then requires an independent review
against English and Mandarin CPU goldens. The current runtime-companion
installer goes further: it proves exact target/runtime/model identity, executes
both fixtures under the app UID, checks their transcripts, and then exposes
execution-aware health.

## 5. Run deterministic tests

```bash
python3 scripts/npu_multilingual/test_backend_gate.py
```

The tests cover explicit English, limited EN/ZH Auto, broad Auto, unsupported
Russian, missing smoke evidence, wrong SoC, and missing runtime files.
