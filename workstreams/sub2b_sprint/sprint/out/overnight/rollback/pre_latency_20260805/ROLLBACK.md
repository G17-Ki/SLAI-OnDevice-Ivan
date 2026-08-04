# Pre-latency deployment rollback

Captured from the working Nubia NX809J installation before the selected-route
latency work began on 2026-08-05.

| Package | Installed version | Saved APK | SHA-256 |
|---|---|---|---|
| `ai.liquid.lfmdemo` | `0.6.0-unplugged` (`26`) | `lfm-audio-0.6.0-unplugged-v26.apk` | `712632b7987557f02bd3bf67b77c00b49025207178f1719094bbb6a81cb2f53f` |
| `ai.liquid.lfmdemo.runtime` | `0.1.0-unplugged-runtime` (`1`) | `lfm-runtime-0.1.0-unplugged-v1.apk` | `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244` |

The APK bytes are intentionally ignored by Git. They are retained locally in
this directory and copied to the phone at:

`Download/on-device-speech-agent-rollback-20260805/`

Model files and app settings are not cleared by the latency deployment. Restore
the pair without deleting either package so their app-private models remain in
place:

```sh
adb install -r -d lfm-runtime-0.1.0-unplugged-v1.apk
adb install -r -d lfm-audio-0.6.0-unplugged-v26.apk
```

After restoring, force-stop both packages, open the main app once, and confirm
that the runtime companion starts under its own application UID. Do not use
`pm clear` or uninstall either package: those operations delete the resident
models.
