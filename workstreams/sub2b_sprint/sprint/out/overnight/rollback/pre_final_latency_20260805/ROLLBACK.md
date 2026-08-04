# Pre-final latency deployment rollback

Captured from the working Nubia NX809J immediately before the final
selected-route persistence and health-recovery build was installed on
2026-08-05.

| Package | Saved APK | SHA-256 |
|---|---|---|
| `ai.liquid.lfmdemo` | `lfm-audio-pre-final-latency.apk` | `86bc6629174ab66223de117f2e9ce320990c34fe5e416566e67746916a704fb6` |
| `ai.liquid.lfmdemo.runtime` | `lfm-runtime-pre-final-latency.apk` | `05f45819abf6688c8183fc5157e6ac08fb65b691c3f8e62b72c8cbd584a13f7f` |

The same files are copied to the phone at:

`Download/on-device-speech-agent-rollback-pre-final-20260805/`

Restore in place so app-private models, voices, settings, and telemetry remain:

```sh
adb install -r -d lfm-runtime-pre-final-latency.apk
adb install -r -d lfm-audio-pre-final-latency.apk
```

Never uninstall either package and never use `pm clear`; both operations would
delete resident private model files. After restoring, force-stop both packages,
open the main app, select an explicit language, and wait for the offline-route
ready message.
