# Native Android phone-only demo

Date: 2026-07-16

## Outcome

`LFM Audio` now runs as a native Android application on the Nubia NX809J. The
demo does not need a Mac, USB cable, network connection, browser, or ADB shell
after the one-time APK/model installation.

The current execution backend remains the official Android ARM64 Q4 CPU runner.
This is not a QNN/Hexagon result.

## Architecture

- Package: `ai.liquid.lfmdemo`
- Target: Nubia NX809J, Snapdragon SM8850, Android 16
- Android foreground service owns the LFM native child process.
- Native `AudioRecord` captures mono PCM16 at 16 kHz.
- An app-local HTTP UI is served on `127.0.0.1:8765`.
- The app-owned LFM streaming API listens on `127.0.0.1:8080`.
- Model files live under the application's private `files/models` directory.
- Generated text streams into the embedded UI and generated PCM audio is
  converted to a playable WAV.

## Verification

1. The app-owned model server loaded all four private model files and reported
   `Server ready at http://127.0.0.1:8080`.
2. The reference WAV produced the exact expected ASR transcript through the
   app-owned process:

   > Can you help me come up with a slogan for my woodworking site business?

3. Native microphone capture completed an interleaved request and produced
   streamed text plus an 8-second generated-audio response.
4. USB ADB transport was removed and only a temporary Wi-Fi debugging control
   channel remained. From that state the app was force-stopped, cold-launched,
   loaded the private models, captured the phone microphone, and completed a
   new interleaved inference. Wi-Fi was used only to observe and automate the
   test; the app itself made no network request.
5. Android reported `.LfmService` as `isForeground=true`, and both app-local
   ports remained available.

## Artifact

- APK: `android-app/releases/lfm-audio-demo-0.1.0-debug.apk`
- Size: approximately 4.4 MiB, excluding separately installed model weights
- SHA-256: `d9b74cd59c86446c0a536f412e03d39d4a27899063ac39e290548c400f3474c5`

## Limitations

- Debug-signed feasibility build, not a production release.
- CPU Q4 only; QNN/NPU integration remains separate work.
- The four private model files occupy approximately 1.0 GiB in total.
- Nubia/Android battery policy can still reclaim a foreground service under
  exceptional pressure, although it is no longer tied to an ADB shell or USB
  session.
- The app starts the model when opened; after a reboot, open `LFM Audio` and
  allow several seconds for model loading.
