# Android 0.4.0 candidate validation

Date: 2026-07-23 23:00 CST
Scope: build, source, package, and responsive-layout validation only

## Retained artifact

```text
android-app/releases/lfm-audio-demo-0.4.0-multilingual-vad-debug.apk
Bytes: 4,762,649
SHA-256: 4943dbd56a5083f0cde2f0c0d6566e59a502495f40d9ae3b644ebe0b1c01c918
```

The checksum sidecar verifies, and the retained APK is byte-identical to
`android-app/app/build/outputs/apk/debug/app-debug.apk` from the final
one-worker low-priority build.

## Package gate

- application ID: `ai.liquid.lfmdemo`;
- version code/name: `8` / `0.4.0-multilingual-vad`;
- minimum/target SDK: 28 / 35;
- native payload: `arm64-v8a` only;
- signature: Android Debug certificate, APK Signature Scheme v2 verifies;
- packaged `index.html`, `service-worker.js`, and `manifest.webmanifest`
  SHA-256 values exactly match source.

## Focused source gate

All commands pass:

```sh
node scripts/test_streaming_audio_ui.mjs
python3 scripts/test_android_native_tts_bridge.py
python3 scripts/test_android_streaming_telemetry.py
python3 scripts/test_native_vad_bridge.py
scripts/test_adaptive_energy_vad.sh
python3 scripts/npu_multilingual/test_backend_gate.py
git diff --check
```

The VAD UI regression includes manual stop/restart acknowledgement, a stale
old-session stop callback, stop before `session_started`, and fatal native-error
recovery when no second `stopped` callback can be expected.

Android compilation and packaging pass with Gradle `--no-daemon`, one worker,
JDK 17, compile/target API 35, and low process priority.

## Responsive UI gate

The packaged source UI was inspected at 390 x 844 and 320 x 720 CSS pixels.
There is no horizontal overflow; the language, history, response-mode, and
input-mode controls remain reachable; Mandarin and Hands-free state changes
render correctly; and the browser records no JavaScript warning/error entries.

## Publication gate

`python3 scripts/validate_publication.py` passes after the final artifact and
report reconciliation: 969 candidate files, 235 strict JSON files, 173 Python
files, 46 shell files, 129 Markdown files / 278 local links, 862 UTF-8 text
files, seven APK sidecars, and two Git diff checks, with zero errors.

## Explicit non-results

No Android device is connected. This validation does not establish ASR or
response quality, Android TTS voice availability, acoustic VAD robustness,
conversation latency, first-audio timing, memory, thermal behavior, battery
life, or QNN execution. The default installer remains pinned to
`0.3.2-turn-telemetry`.
