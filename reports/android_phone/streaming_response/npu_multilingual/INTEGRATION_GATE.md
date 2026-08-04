# Multilingual Android NPU integration gate

Date: 2026-07-24 (Asia/Shanghai)

## Decision

SenseVoiceSmall INT8 is the correct first partial-NPU component for the Nubia
SM8850. Its exact ten-second QNN context already passed Qualcomm AI Hub on an
exact SM8850 / HTP-v81 target with 2,597/2,597 reported operations on HTP, zero
fallback, 10.871 ms estimated execution, and 126.223 MiB estimated peak memory.

Physical command-line feasibility now passes on the Nubia with the pinned QNN
2.40 runtime. English decoded in 41 ms and Mandarin in 36 ms; independent CPU
inference matches both text and language tags exactly under the QNN context's
mandatory fixed ten-second zero-padding contract. The reviewed selector now
chooses QNN for explicit English, explicit Mandarin, and Auto explicitly
limited to English plus Mandarin.

It is **not activated in the phone demo APK yet**. The current application has
no Sherpa/QNN JNI or service integration, and application telemetry correctly
reports `npu_executed=false`. Broad six-language Auto remains on CPU because
SenseVoice does not support Russian, French, Italian, or Spanish.

## What is now reproducible

The checked-in tooling provides five layers:

1. [`assets.json`](../../../../scripts/npu_multilingual/assets.json) pins the
   model, QNN 2.40 headers/runtime, Sherpa 1.13.4 source, ONNX Runtime 1.27.0,
   upstream smoke APK, target ABI, and all known hashes.
2. [`prepare_sensevoice_sm8850_bundle.py`](../../../../scripts/npu_multilingual/prepare_sensevoice_sm8850_bundle.py)
   acquires and verifies the private engineering bundle outside the repository.
   It refuses an output path inside the repository.
3. [`select_android_asr_backend.py`](../../../../scripts/npu_multilingual/select_android_asr_backend.py)
   activates QNN only after exact device, ABI, v81, artifact, runtime, and
   phone-smoke gates pass. Seven deterministic regressions exercise the gate.
4. [`compare_sensevoice_cpu_qnn.py`](../../../../scripts/npu_multilingual/compare_sensevoice_cpu_qnn.py)
   runs the independent CPU model on the exact EN/ZH phone fixtures with both
   variable and fixed-ten-second input shapes.
5. [`review_phone_qnn_smoke.py`](../../../../scripts/npu_multilingual/review_phone_qnn_smoke.py)
   promotes raw execution only after runtime, artifact, language-tag, and exact
   fixed-shape transcript parity checks pass.

An external bundle was assembled successfully during this run. Verified core
members are:

| Member | Bytes | SHA-256 |
|---|---:|---|
| `model.bin` | 254,193,664 | `522763135f536bc43e9d122bac39254f49df4afbb003462d6d8bcdb77051de30` |
| `tokens.txt` | 315,894 | `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc` |
| `libQnnHtp.so` | 2,516,048 | `328cf737ca8942c2dde5c6f3e32a113378df2750afb9af1aafdd1775bef59875` |
| `libQnnHtpPrepare.so` | 74,225,392 | `05cfaff3c2180d7d9006320c16579c5869c2261ba5f891bcaab172ced21c266c` |
| `libQnnSystem.so` | 2,590,072 | `2d42b6bb2710155fa963ee623ce3c320f8a9896b9cfd8dcc25506affb41ab8ab` |
| `libQnnHtpV81Stub.so` | 632,336 | `0fccba2bd6b66d3e5ac32381cfbf5b10b80088055ff6dab712ce99633e3bc8c9` |
| `libQnnHtpV81Skel.so` | 10,561,352 | `2dff3367e70920742e8122cdbebfe4d5f9aa1d88d863a200b663842551d2b6f0` |

The `tokens.txt` hash above is intentionally checked again by the preparer
against the canonical value in `assets.json`; the table must be treated as
informational rather than a substitute for the executable gate.

The bounded Android cross-build also succeeded:

| Build output | Bytes | SHA-256 |
|---|---:|---|
| `sherpa-onnx-offline` | 2,546,312 | `e7ab43565de5fa6d4f7badf1569a38c423f6afe361e5baa0f231ebf7d3a4ef40` |
| `libonnxruntime.so` | 21,688,920 | `994848008526a934dfb579ac773b00e5867929234852b061005d45aacaee9533` |

The runner is a stripped ELF64 AArch64 Android PIE using
`/system/bin/linker64`. Its only non-platform dynamic dependency is
`libonnxruntime.so`; the QNN backend is loaded dynamically from the private
external bundle. String inspection confirms the QNN context/backend/system
configuration surface is present. The same runner subsequently initialized the
HTP provider and completed English and Mandarin inference on the physical
phone. The sanitized build inventory is
[`runner_manifest.json`](runner_manifest.json).

## Language-routing policy

| Request | NPU eligibility after the phone gate | Required route today |
|---|---|---|
| Explicit English | SenseVoice QNN | QNN selector pass; APK integration pending |
| Explicit Mandarin | SenseVoice QNN | QNN selector pass; APK integration pending |
| Auto limited to English + Mandarin | SenseVoice QNN Auto | QNN selector pass; APK integration pending |
| Broad Auto: EN/ZH/RU/FR/IT/ES | Not safe on SenseVoice | Multilingual CPU ASR |
| Explicit Russian/French/Italian/Spanish | Unsupported by SenseVoice | Multilingual CPU ASR |
| Explicit Japanese/Korean/Cantonese | SenseVoice QNN supports it | Optional; requires its own parity/quality gate |

SenseVoice Auto recognizes only Mandarin, English, Japanese, Korean, and
Cantonese. It must not be presented as a robust six-language Auto detector:
Russian, French, Italian, or Spanish speech could be forced into an incorrect
supported class. Broad Auto stays on a multilingual CPU backend until a proven
language-ID-first cascade exists.

## Runtime compatibility boundary

The Nubia evidence confirms:

- `ro.soc.model=SM8850` and ABI `arm64-v8a`;
- HTP-v81 Stub/Skel filenames are present;
- the accessible OEM binaries report QAIRT/QNN 2.37.0.

The selected context was generated with QNN 2.40.0.251030. The integration must
therefore package the exact 2.40 application runtime and v81 Stub/Skel under the
application's private distribution boundary. It must not silently use the
phone's older 2.37 OEM objects. It also must not publish the raw Qualcomm
objects as a standalone repository bundle.

## Android APIs

The exact Sherpa 1.13.4 Kotlin configuration surface is:

```kotlin
OfflineRecognizerConfig(
    modelConfig = OfflineModelConfig(
        provider = "qnn",
        tokens = "sensevoice/tokens.txt",
        senseVoice = OfflineSenseVoiceModelConfig(
            language = selectedLanguage, // auto, zh, en, ja, ko, or yue
            useInverseTextNormalization = true,
            qnnConfig = QnnConfig(
                backendLib = "libQnnHtp.so",
                systemLib = "libQnnSystem.so",
                contextBinary = "sensevoice/model.bin",
            ),
        ),
    ),
)
```

This requires a Sherpa JNI build compiled with `SHERPA_ONNX_ENABLE_QNN=ON`, the
QNN application libraries in `jniLibs/arm64-v8a`, and the v81 Skel discoverable
through the application/DSP library path. The current LMF APK does not contain
that JNI integration, so the API above is a concrete next integration point,
not a claim about the retained APK.

The command-line equivalent used by the phone smoke harness is:

```text
--provider=qnn
--sense-voice-language=<auto|zh|en|ja|ko|yue>
--sense-voice.qnn-backend-lib=./libQnnHtp.so
--sense-voice.qnn-system-lib=./libQnnSystem.so
--sense-voice.qnn-context-binary=./model/model.bin
```

## Component placement

| Component | Initial backend | Reason |
|---|---|---|
| SenseVoice neural context | HTP-v81 after gate | Exact context already has strict AI Hub placement evidence. |
| WAV decode/resample + log-mel/features | CPU | Small host-side work; not part of the QNN context profile. |
| Token decoding and text cleanup | CPU | Small deterministic post-processing. |
| VAD | CPU | Silero/WebRTC-class VAD is small; no accepted v81 gain justifies a second runtime path yet. |
| EN/ZH broad language routing | CPU control logic | Needed to enforce the fail-closed route and unsupported-language fallback. |
| Russian/French/Italian/Spanish ASR | CPU multilingual model | SenseVoice does not cover these languages. |
| Piper-English TTS | Hold as secondary NPU candidate | Exact-SM8850 placement passes, but ASR directly unblocks multilingual Auto first. |

The 10.871 ms AI Hub number is neural-context execution only. It excludes
audio decoding, resampling, feature creation, token decoding, VAD endpointing,
application scheduling, and model/context initialization.

## Phone acceptance gate

QNN can become the default for eligible requests only after all of these pass:

1. **Passed:** exact SM8850, arm64-v8a, and v81 detection.
2. **Passed:** exact QNN 2.40 runtime and model hashes.
3. **Passed once per language:** recognizer initialization with no
   binary-version or DSP-load failure.
4. **Passed:** independent English and Mandarin transcript parity against CPU
   goldens under fixed ten-second zero-padding.
5. **Open:** ten repeated fixed-file turns and ten microphone/VAD turns.
6. **Open:** no crash, context corruption, or thermal fallback over the
   repeated run.
7. **Open:** measured endpoint-to-final latency and resident memory inside the
   integrated application budget.
8. **Open:** Sherpa/QNN JNI or service integration plus fail-closed runtime
   fallback in the APK.

[`run_phone_qnn_smoke.py`](../../../../scripts/npu_multilingual/run_phone_qnn_smoke.py)
stages the external runner, exact runtime, model, and EN/ZH fixtures and records
raw output. It intentionally writes `raw_pass`, not production `pass`. The
separate reviewer now produces the accepted record only after transcript and
runtime checks pass. Public-safe evidence is linked below; the raw smoke log is
not published because upstream source locations contain a local path.

## Current result

The external bundle, physical runtime, and exact EN/ZH parity gates pass. The
selector now returns:

| Request | Selector result |
|---|---|
| Explicit English | `sensevoice_qnn_sm8850` |
| Explicit Mandarin | `sensevoice_qnn_sm8850` |
| Auto limited to EN/ZH | `sensevoice_qnn_sm8850` |
| Broad six-language Auto | `cpu_multilingual` |
| Explicit Russian/French/Italian/Spanish | `cpu_multilingual` |

Machine-readable physical/parity/selector evidence is in the
[phone acceptance bundle](../phone_acceptance/REPORT.md), including the
[reviewed smoke](../phone_acceptance/npu/sensevoice_qnn_reviewed_smoke.json),
[CPU parity](../phone_acceptance/npu/sensevoice_cpu_parity.json), and
[selection matrix](../phone_acceptance/npu/backend_selection_matrix.json).

This authorizes guarded APK integration work, not production activation. Keep
the CPU fallback and report `npu_executed=true` only after the integrated
request actually succeeds on QNN.
