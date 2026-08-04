# Android turn-telemetry validation — 2026-07-23

## Result

Local validation passed for Android version `0.3.2-turn-telemetry` (version
code 5). The streamed-audio behavior from `0.3.1-telemetry` is unchanged, and
the app now records an app-private JSONL trace on Android's monotonic clock.

For each turn the JavaScript layer records:

- input stop request and WAV readiness;
- request start, first streamed text, and first streamed PCM;
- first WebAudio buffer scheduling and scheduled start;
- output-head result (`observed`, `unsupported`, or `unavailable`);
- every queue underflow, terminal error, and response completion.

The native microphone bridge also records native recording start, stop request,
and WAV readiness. These events have no turn ID and are retained as unscoped
cross-checks. The trace is stored only in the app-private
`files/streaming-telemetry.jsonl` file; it is not sent over the network.

## Artifact

- APK: `android-app/releases/lfm-audio-demo-0.3.2-turn-telemetry-debug.apk`
- Size: 4,658,347 bytes
- SHA-256:
  `be6c6fdf355d4800a2c569d0bf3d8df04951a5f70e86ccef098190b18de12b98`
- Package: `ai.liquid.lfmdemo`
- Minimum / target API: 28 / 35

The dependency-free WebView and telemetry-parser tests passed, and the Android
debug build completed successfully with one Gradle worker. This rebuild also
replaced the stale USB-only connection error with phone-only recovery guidance;
the streamed-audio and telemetry logic is otherwise unchanged. The phone was disconnected, so
no physical output-head, underflow, resource, thermal, or acoustic measurement
is claimed for this artifact.

## Pull and summarize after the next phone run

Install with replacement so the app-private model files remain intact:

```bash
adb install -r android-app/releases/lfm-audio-demo-0.3.2-turn-telemetry-debug.apk
```

After the fixed-file and microphone turn matrix, pull and summarize the private
trace without rooting the phone:

```bash
python3 scripts/pull_android_streaming_telemetry.py \
  --output-dir reports/android_phone/streaming_response/phone_turn_telemetry/$(date -u +%Y%m%dT%H%M%SZ)
```

Add `--clear-after` only after the pull has succeeded and the output has been
checked. The summarizer emits strict `telemetry.jsonl`, `summary.json`, and a
Markdown report grouped by turn.

`input_stop_requested` is a manual-stop proxy in this APK, not an acoustically
measured last-speech sample and not a production VAD-close event. Likewise,
WebAudio output-head crossing excludes Android's downstream mixer/HAL,
DAC/speaker, and acoustic propagation. The final public latency claim still
requires an external recording aligned to the turn marker.
