# Android streaming response validation — 2026-07-22

## Result

Local validation passed. The embedded phone UI can decode both supported streamed-audio delta formats, schedule the first PCM chunk with a 140 ms startup cushion, keep later chunks contiguous, and flag a playback underflow. The Android `0.3.0-streaming` debug APK also builds successfully.

No phone was connected during this validation, so the APK was not installed and audible-start timing remains to be measured on hardware.

## Reproduction

From the repository root:

```bash
node scripts/test_streaming_audio_ui.mjs

ANDROID_HOME=/opt/homebrew/share/android-commandlinetools \
ANDROID_SDK_ROOT=/opt/homebrew/share/android-commandlinetools \
scripts/build_android_apk.sh

shasum -a 256 android-app/app/build/outputs/apk/debug/app-debug.apk
/opt/homebrew/share/android-commandlinetools/platform-tools/adb devices -l
```

Observed results:

- Streaming UI test: `streaming audio UI tests passed`
- Android build: `BUILD SUCCESSFUL`
- APK payload check: packaged `assets/demo-ui/index.html` contains the streaming scheduler, dual-format decoder, and requested `modalities`
- ADB: no connected devices
- Release APK: `android-app/releases/lfm-audio-demo-0.3.0-streaming-debug.apk`
- Size: 4.4 MB
- SHA-256: `d9ca8c7235e51d7d5b70b4dcf28186c2065191a24e8fc1e2ba35ada9af291e8f`

## Remaining physical-phone validation

1. Connect the phone and confirm it is authorized with `adb devices -l`.
2. Install with `adb install -r android-app/releases/lfm-audio-demo-0.3.0-streaming-debug.apk` so existing private app data/model files remain intact.
3. Run fixed-file and microphone chat turns; confirm audio starts while text/audio generation is still in progress.
4. Record at least 10 turns for the default and no-repack configurations: first received PCM, scheduled playback time, actual audible start, completion, and displayed gap count.
5. Verify median end-of-speech-to-audible latency is at most 2.0 seconds, P95 at most 3.0 seconds, and no audible gap exceeds 80 ms.
