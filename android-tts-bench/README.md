# On-device TTS candidate bench

A standalone debug app for comparing every Mandarin and Russian TTS candidate on
the phone: how each one sounds through the real speaker, and what it actually
costs on the phone's CPU.

**It is a separate application from the LFM demo.** It has its own package
(`ai.liquid.ttsbench`), its own data directory, and shares nothing with
`android-app/`. Installing it cannot affect the validated
`0.4.13-expressive-lfm` route, and uninstalling it removes the 1.5 GiB model
bundle with it.

This app has now been exercised on the Nubia SM8850 phone. The published run
contains 682 device measurements, playback-head timing, memory sweeps, thermal
checks, and device-rendered audio; see
[`DEVICE_RESULTS.md`](../reports/android_phone/streaming_response/tts_research_20260726/DEVICE_RESULTS.md).
Those measurements establish execution and engineering behavior, not
naturalness; the blinded listening set is still unscored.

## What it measures

Per synthesis, on the device:

| Metric | Meaning |
|---|---|
| load ms | time to construct the engine (one model resident at a time) |
| first chunk ms | first callback from the synthesizer |
| total ms, RTF | full synthesis time and real-time factor |
| **first audible ms** | from `AudioTrack.getPlaybackHeadPosition` — when the speaker actually moved, not when a buffer was accepted |
| **underruns** | `AudioTrack.getUnderrunCount`, the device's own count, not a simulation |
| peak PSS | whole-process resident memory during the run |
| gate | the real-time safety subset of the host gates: empty/non-finite PCM, near-silence, clipping, and DC offset |

The host study could only *simulate* playback starvation from callback
timestamps. This app reports the device's real underrun count and real
playback-head onset, which is the gap the host work could not close.

The Android gate intentionally covers checks that are safe per callback before
audio reaches the speaker. The host post-processing suite additionally checks
long internal silence and repeated-window collapse over the complete waveform;
those two checks are not claimed by the app.

## Safety behaviours that are not optional

- **Invalid PCM never reaches the speaker.** Kokoro's INT8 build returns an
  entirely non-finite buffer for Mandarin input shorter than three Han
  characters; written to `AudioTrack` that is full-scale noise into someone's
  ear. Every chunk passes `PcmGate` first and a failing chunk is dropped and
  counted. When testing `kokoro_zh_int8` with `好的` you should hear **silence**,
  and the app should report dropped chunks.
- **Loudness is equalized for A/B by default.** The candidates differ by more
  than 20 dB in output level; without this, an A/B mostly measures gain. Uncheck
  it to hear a candidate at its native level.
- **A failed waveform is never saved as a WAV.**

## Build

```bash
tools/fetch_sherpa_aar.sh      # 48 MiB runtime, kept out of this repository
tools/build_tts_bench.sh
```

Needs JDK 17, an Android SDK with `android-35`, and build-tools 34.0.0. The
sherpa-onnx version is pinned to **1.13.4** — the same build the host
measurements were taken with, so a host/device difference is the device and not
a different engine.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing a
benchmark APK or any candidate model output.

## Install and load the candidates

```bash
export TTS_BENCH_BUNDLE=/path/to/device-bundle     # from stage_device_bundle.py
tools/push_tts_bundle.sh
```

The bundle lands in `/sdcard/Android/data/ai.liquid.ttsbench/files/tts-eval/`,
which needs no storage permission. The push is ~1.5 GiB and takes several
minutes over USB 2; the script checks free space first and verifies the file
count afterwards.

## Candidates are data, not code

The app reads `candidates.json` from the pushed bundle at startup. Adding a
voice, changing a speaker id, or trying a different Supertonic step count means
editing that file and re-pushing it — **no rebuild, no reinstall**, which
matters when someone is sitting with the phone doing listening comparisons.

Candidates carrying a `known_issue` show `[!]` in the picker and print the
warning under the selector, so a known-bad configuration is not mistaken for a
device problem.

## Reading the results

`Export JSON` writes to `/sdcard/Android/data/ai.liquid.ttsbench/files/results/`:

```bash
adb pull /sdcard/Android/data/ai.liquid.ttsbench/files/results ./device-results
```

The schema matches the host bench closely enough to compare directly against
`reports/android_phone/streaming_response/tts_research_20260726/evidence/`.
The host numbers are Apple M2 at two threads; expect the phone to be slower, and
treat the ratio as the real "phone slowdown factor" that the host work could
only assume.
