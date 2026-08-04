# Android playback-telemetry validation — 2026-07-23

## Result

Local validation passed for Android version `0.3.1-telemetry` (version code 4).
The embedded UI still decodes both supported streamed-audio delta formats,
schedules the first PCM chunk with a 140 ms cushion, keeps later chunks on one
timeline, and reports queue underflow. It now also distinguishes the scheduled
first-buffer time from an observed WebAudio output-head crossing, with explicit
`measuring`, `unsupported`, and `unavailable` states.

The release APK was built, the Android output was cleaned, and the build was
repeated. The initial build and clean rebuild were byte-identical. No phone was
connected, so installation, real output-head behavior, acoustic first-audio
timing, gaps, RSS, and thermals remain untested for this version.

## Reproduction

From the repository root, with an Android SDK containing API 35:

```bash
node scripts/test_streaming_audio_ui.mjs

ANDROID_HOME="$ANDROID_SDK_ROOT" \
ANDROID_BUILD_MAX_WORKERS=2 \
scripts/build_android_apk.sh

shasum -a 256 android-app/app/build/outputs/apk/debug/app-debug.apk
```

The build helper defaults to two Gradle workers and disables the persistent
Gradle daemon. The validation runs also limited the JVM to two active
processors.

Observed results:

- Streaming UI test: `streaming audio UI tests passed`
- Android build: `BUILD SUCCESSFUL`
- Manifest: package `ai.liquid.lfmdemo`, version code 4, version name
  `0.3.1-telemetry`, minimum API 28, target API 35
- APK size: 4,643,235 bytes
- APK SHA-256:
  `5e0814a3f0b62da8bd8f2d9e0e7fb47e4fe820a89aad63129ab4c97d76a3935f`
- Release APK:
  `android-app/releases/lfm-audio-demo-0.3.1-telemetry-debug.apk`
- The retained `0.3.0-streaming` APK was not overwritten and still hashes to
  `d9ca8c7235e51d7d5b70b4dcf28186c2065191a24e8fc1e2ba35ada9af291e8f`.

## Remaining physical-phone validation

1. Install the release APK with `adb install -r` so private app data and model
   files are preserved.
2. Run at least ten fixed-file and ten microphone chat turns.
3. Capture VAD close, first PCM, scheduled first buffer, output-head state/time,
   completion, displayed gap count, process RSS, and thermal state.
4. Measure actual speaker onset externally. The output-head estimate does not
   include the full Android mixer/HAL, DAC/speaker, Bluetooth, or acoustic path.
5. Accept only if end-of-speech to audible response is at most 2.0 seconds
   median and 3.0 seconds P95, with no audible gap above 80 ms.
