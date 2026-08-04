# Whisper Small ORT-QNN compatibility probe

This standalone Android package verifies the cached Whisper Small W8A16
external EPContext release with ORT 1.25 QNN EP and QAIRT 2.42 on SM8850/v81.
It shares no package, service, storage, or lifecycle with the accepted demo
applications.

Stage the exact external payload:

```sh
python3 scripts/stage_whisper_ort_qnn_probe.py \
  --output /private/tmp/lmf-whisper-ort-qnn-probe-payload \
  --feature-fixture ../artifacts/whisper_ort_qnn_frontend_golden_v1/input_features.u16le.bin
```

Stage the verification-only EN/JA frontend bundle from the pinned host audit:

```sh
python3 scripts/compare_whisper_native_frontend.py \
  --require-adapted-bounded-parity
python3 scripts/stage_whisper_android_frontend_assets.py \
  --output /private/tmp/lmf-whisper-frontend-assets-v1
```

The additive five-language profile leaves that v1 bundle unchanged. Generate
and stage it explicitly with:

```sh
python3 scripts/compare_whisper_native_frontend.py \
  --fixture-profile v2 \
  --build-flavors portable \
  --modes kissfft-direct-scalar \
  --require-adapted-bounded-parity
python3 scripts/stage_whisper_android_frontend_assets_multilang.py \
  --output /private/tmp/lmf-whisper-frontend-assets-v2
```

Build from `android-app/` with JDK 17 and at most two workers:

```sh
LMF_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo"
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME="$LMF_CACHE_ROOT/android-sdk" \
ANDROID_SDK_ROOT="$LMF_CACHE_ROOT/android-sdk" \
WHISPER_PROBE_PAYLOAD_DIR=/private/tmp/lmf-whisper-ort-qnn-probe-payload \
WHISPER_FRONTEND_ASSET_DIR=/private/tmp/lmf-whisper-frontend-assets-v1 \
WHISPER_FRONTEND_PROFILE=v1 \
./gradlew :whisper-probe:assembleDebug --max-workers=2 --no-daemon
```

`WHISPER_FRONTEND_PROFILE` defaults to `v1`. Set it to `v2` only with the v2
asset directory. A v2 non-semantic build verifies EN/ZH/RU/KO/JA; a v2 semantic
build authenticates all 12 assets but computes only English plus the selected
semantic language.

Verify the final APK payload:

```sh
python3 scripts/verify_whisper_ort_qnn_probe_apk.py \
  --apk android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk
```

The original schema-2 verifier intentionally rejects any extra archive member.
For a build containing the separately pinned frontend overlay, use the overlay
verifier; it still requires the exact unchanged schema-2 base payload:

```sh
python3 scripts/verify_whisper_android_frontend_apk.py \
  --apk android-app/whisper-probe/build/outputs/apk/debug/whisper-probe-debug.apk
```

The app has no network or microphone permission. Its explicit compatibility
button remains disabled in a stub build and execution fails closed unless all
model/runtime hashes pass after installation into app-private storage. With
the frontend bundle embedded, the app computes `[1,80,3000]` UINT16 tensors
from fixed English and Japanese WAVs through a persistent 400-point KissFFT
plan. It records exact tensor hashes, differing counts, maximum deltas, and
frontend/encoder/per-decoder-step timings. The bounded fixture result gates the
English tensor before the existing fixed four-step encoder/decoder probe; it
does not enable microphone input or claim a semantic transcription.

## Default-off semantic candidate

The pure-Java forced-prompt, cumulative-mask, token-suppression policy has a
dependency-free JDK 17 host regression:

```sh
python3 scripts/test_whisper_semantic_decode_policy.py
```

`WHISPER_SEMANTIC_DECODE_ENABLED` defaults to `false`. Setting it to the exact
string `true` creates a distinct candidate build. `WHISPER_SEMANTIC_LANGUAGE`
defaults to `en`. Profile v1 remains restricted to `en` or `ja`; profile v2
admits `en`, `zh`, `ru`, `ko`, or `ja`. The build still executes and reports
the unchanged English four-step compatibility probe first, then runs the
selected semantic candidate with a pinned language token, decoder positions
`0..198`, cumulative right-aligned attention unmasking, and at most 196
generated tokens.
Generated IDs admit only base vocabulary `0..50256` plus EOT `50257`; every
control, language, task, and timestamp ID `50258..51864` is suppressed.
Detokenization concatenates raw token bytes before strict UTF-8 decoding and
rejects malformed bytes, U+FFFD, NUL, and disallowed control characters.

The result is written under the top-level `semantic_decode` object with
`semantic_transcript_claimed=false` and `real_audio_transcription=false`.
Do not install or promote a semantic build until it has its own whole-APK trust
root, semantic report specification, collector/verifier, and accepted-package
integrity gate.
