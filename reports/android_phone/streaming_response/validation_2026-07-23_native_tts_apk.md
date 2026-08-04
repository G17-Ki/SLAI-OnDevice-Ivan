# Android native-TTS APK validation — 2026-07-23

> Historical checkpoint: these hashes match the `0.3.3-native-tts` source at
> its validation time. Current source and the newest unverified build are
> `0.3.4-input-handoff`; see
> [`validation_2026-07-23_input_handoff_apk.md`](validation_2026-07-23_input_handoff_apk.md).

## Scope

This checkpoint validates the reproducible debug APK for the optional Android
system-`TextToSpeech` control path. The build used one low-priority Gradle
worker. No phone was connected, so the APK was not installed or executed on
hardware. No cloud job, Git commit, or push was performed for this build.

The default installer remains pinned to the physically exercised
`0.3.2-turn-telemetry` release until the new mode passes the phone acceptance
gate.

## Retained artifact

- APK: `android-app/releases/lfm-audio-demo-0.3.3-native-tts-debug.apk`
- Size: **4,674,179 bytes**
- SHA-256:
  `f8931486f8f0b749be3ed1174fd219e98929d701808cc9c064050e257b587f9e`
- Checksum sidecar:
  `android-app/releases/lfm-audio-demo-0.3.3-native-tts-debug.apk.sha256`
- Gradle output: `android-app/app/build/outputs/apk/debug/app-debug.apk`
  matched the retained release artifact byte for byte at validation time.

## Package and signature checks

Android build-tools 35 reported:

- application ID `ai.liquid.lfmdemo`;
- version code `6` and version name `0.3.3-native-tts`;
- minimum API 28, target API 35, and compile API 35;
- native ABI `arm64-v8a` only;
- debuggable application label `LFM Audio`;
- one Android debug signer;
- APK Signature Scheme v2 verification passed.

This is a debug artifact, not a production-signed store release.

## Packaged-source consistency

The two behavior-bearing WebView assets inside the APK matched the source at
this checkpoint byte for byte:

| Asset | SHA-256 in source and APK |
|---|---|
| `assets/demo-ui/index.html` | `445157092eb79abc5b5c32de71d729f6fbef0a4248604209574b0f8a51f06141` |
| `assets/demo-ui/service-worker.js` | `88e78217dc8fcf26074fec4f7ecd77796637385f5541d82fa2b5983c0dca1b68` |

The release checksum sidecar also verifies against the retained APK. The
source-level UI, bridge, telemetry, manifest, and API-35 Java checks are
recorded in
[`validation_2026-07-23_native_tts_source.md`](validation_2026-07-23_native_tts_source.md).

## Reproduction

With JDK 17 and an Android SDK containing API/build-tools 35:

```bash
ANDROID_BUILD_MAX_WORKERS=1 \
ANDROID_HOME=/path/to/android-sdk \
ANDROID_SDK_ROOT=/path/to/android-sdk \
  nice -n 15 scripts/build_android_apk.sh
```

The generated file is
`android-app/app/build/outputs/apk/debug/app-debug.apk`. Copy it to a new
versioned release filename rather than overwriting an older APK, then generate
and verify a SHA-256 sidecar.

## Hardware boundary and promotion gate

Because the phone is disconnected, this checkpoint provides **no** evidence
about:

- Android TTS engine initialization or installed offline voices;
- explicit or auto locale behavior;
- cancellation, interruption, lifecycle release, or queue timing on Android;
- first playback, acoustic onset, audio quality, memory, heat, or battery;
- complete end-to-end response latency.

When the Nubia reconnects, install this candidate explicitly with `--apk` and
test every explicit locale, conservative Auto fallbacks, missing-language
failure, replacement/cancellation, lifecycle release, ten fixed-file turns,
and ten microphone turns. Keep `0.3.2-turn-telemetry` as the default until that
physical acceptance passes.
