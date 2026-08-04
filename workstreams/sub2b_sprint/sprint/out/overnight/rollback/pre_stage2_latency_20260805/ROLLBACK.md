# Pre-stage-2 latency rollback

Captured directly from the working Nubia NX809J before the next latency
optimization deployment on 2026-08-05. The installed pair matched the local
build outputs from source commit
`e1a2a06ff34fd2a23950b49e45d65d71971d21ce` byte for byte.

| Package | Saved APK | SHA-256 |
|---|---|---|
| `ai.liquid.lfmdemo` | `lfm-audio-pre-stage2-latency.apk` | `91b225aaba45a2c0875d5d4316449ff255306773f6593011f8edaa626f797add` |
| `ai.liquid.lfmdemo.runtime` | `lfm-runtime-pre-stage2-latency.apk` | `081cba99cf8abb151ce258481b83297bf8f2caf0446efea4d03d9b1e34b3e971` |

The same hash-verified files are stored on the phone at:

`/sdcard/Download/on-device-speech-agent-rollback-pre-stage2-20260805/`

Restore in place so app-private models, installed voices, settings, and
telemetry remain intact:

```sh
adb install -r -d lfm-runtime-pre-stage2-latency.apk
adb install -r -d lfm-audio-pre-stage2-latency.apk
```

Never uninstall either package and never use `pm clear`: both operations would
delete the resident private model files. After restoring, force-stop both
packages, open the main app, select an explicit language, and wait for
`Offline ASR · Qwen · voice ready` before speaking.

At capture time the app was foregrounded in TTS mode with English selected,
airplane mode enabled, Wi-Fi disabled, no IP route, no ADB forward or reverse,
and the selected SenseVoice/Qwen route resident. The rollback APK install path
has not been exercised because doing so would unnecessarily disturb that
accepted state.
