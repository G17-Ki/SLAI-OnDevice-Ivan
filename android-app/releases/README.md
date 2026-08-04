# Android release artifacts

## Current source status — 2026-07-29

The working deployment has advanced to a two-APK app-owned stack:

| Package | Version |
|---|---|
| `ai.liquid.lfmdemo` | `0.6.0-unplugged` (code 26) |
| `ai.liquid.lfmdemo.runtime` | `0.1.0-unplugged-runtime` (code 1) |

That pair is not retained in this directory yet. Build it from source and use
the checksum-aware installers documented in `docs/ANDROID_DEPLOYMENT.md`. Do
not describe the artifact below as the current app-owned or cable-independent
demo.

## Historical retained main APK

The retained public artifact is the earlier physically validated debug build:

- package: `ai.liquid.lfmdemo`
- version code: `21`
- version name: `0.4.13-expressive-lfm`
- APK: `lfm-audio-demo-0.4.13-expressive-lfm-debug.apk`
- size: 4,814,778 bytes
- SHA-256: `b178c93f05d614f8156843755ea52697f55b299c5bb0b6812209d9ce5f139baf`

Verify it before installation:

```bash
shasum -a 256 -c android-app/releases/lfm-audio-demo-0.4.13-expressive-lfm-debug.apk.sha256
```

This is a debug-signed historical research build, not a production release. Its
recorded `0.4.13` measurements remain valid for that build, but it predates the
app-owned runtime companion. Model weights, Qualcomm runtime files, and
companion executables are intentionally not stored in this repository.
