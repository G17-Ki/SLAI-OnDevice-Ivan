# Isolated Whisper Small ORT-QNN probe scaffold

- Date: 2026-07-29
- Package: `ai.liquid.lfmdemo.whisperprobe`
- Device target: Nubia NX809J / SM8850 / HTP v81
- Status: **local packaging pass; isolated device compatibility probe pass**

This is a coexistable feasibility APK. It does not replace or modify either
accepted package, and it does not change the canonical
`ai.liquid.lfmdemo.runtime` gate.

## Implemented milestone

- Exact cached `onnxruntime-android-qnn:1.25.0` AAR, SHA-256 pinned.
- Exact QAIRT `2.42.0.251225135753_193295` host libraries and v81 DSP skeleton.
- External EPContext encoder and decoder wrappers plus context binaries copied
  from APK assets into one app-private model directory.
- `OrtSession.SessionOptions.addQnn` with an absolute app-private
  `libQnnHtp.so` path and `session.disable_cpu_ep_fallback=1`.
- `ADSP_LIBRARY_PATH` prepended with the app native-library directory using the
  required semicolon separator before QNN libraries or sessions are loaded.
- Optional `libcdsprpc.so` linker namespace declaration.
- Exact live-session tensor name, dtype, and shape comparison against release
  `metadata.json`.
- Exact UINT16 inputs. ORT Java 1.25 cannot publicly construct UINT16, so a
  guarded helper calls ORT's own private JNI buffer constructor with ONNX type
  UINT16. It fails closed on API drift; it never substitutes INT16.
- Pinned English QAI-compatible feature tensor:
  `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1`.
- Encoder execution followed by four decoder calls, right-to-left mask updates
  `199, 198, 197, 196`, logical-zero self-KV at UINT8 `128`, full returned
  self-KV reuse, UINT16 logits argmax, and first-token reporting.
- VoiceAI `vocab.bin` decoding that special-cases raw-NUL token 188, treats
  token 50256 as empty, skips special IDs 50257+, concatenates raw bytes, then
  decodes UTF-8 once.
- No `INTERNET` or `RECORD_AUDIO` permission. Real-audio frontend and semantic
  transcript claims remain disabled in both source and payload manifest.

## Deterministic external payload

The stager refuses repository output, verifies every input before and after
copy, pins the corrected English QAI/HF feature hash, and emits no output-path or random-temp
field in the embedded manifest. Two fresh staging roots produced byte-identical
manifests:

`d26436d1bc3ae254622cf4e627017957f88ad3c1d0b005995e0a2803666763df`

Current staged root:

`/private/tmp/lmf-whisper-ort-qnn-probe-payload-en29c-v4`

## Build and local verification

Build environment:

- JDK: `/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`
- Android SDK: `<local-cache>/lmf-demo/android-sdk`
- Gradle workers: maximum 2

Final APK:

- Path: `android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk`
- Bytes: `412376700`
- SHA-256: `3a620198ec050fda6f6c8f8e1e04f1ee0f7764368c141c3aadb799b1a47bb64e`
- Exact embedded members verified: 15/15 (8 assets, 5 QAIRT libraries, 2 ORT libraries)
- `aapt2`: isolated package, arm64-v8a only, optional `libcdsprpc.so`, no
  INTERNET or RECORD_AUDIO permissions.

Validation passed:

- payload stager tests: 6/6;
- final APK verifier tests: 12/12;
- source fail-closed gate;
- Ruff;
- probe-only `assembleDebug` under JDK 17;
- Android `lintDebug`.

## Device compatibility result

The coexistable package was installed and passed twice on Nubia NX809J while
airplane mode remained enabled. The successful package-UID process loaded QNN
HTP, the v81 stub, QNN System, ORT, and vendor `libcdsprpc.so`; it held an open
`/dev/fastrpc-cdsp` descriptor and QCOM DMA buffers. Encoder and decoder
sessions were created with CPU EP fallback disabled, the pinned encoder fixture
executed, and all four decoder calls completed. The rerun finished in
`739.743 ms` total. Full evidence is in `ORT_QNN_PROBE_DEVICE_EVIDENCE.md`.

Both accepted APKs were re-hashed afterward and remain byte-identical to their
accepted hashes.

## Open gates

1. Separately integrate and golden-test a real Whisper frontend before enabling
   audio input or claiming transcription. Cached kaldi-native-fbank arm64
   objects are available, but they are not part of this milestone.
2. A probe-package pass is not a pass for `ai.liquid.lfmdemo.runtime`; that
   final integration remains separately gated.
