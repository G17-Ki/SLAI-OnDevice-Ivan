# Streaming queue robustness replay

Date: 2026-07-22

## Decision

Keep the current 140 ms startup cushion for the phone acceptance run. Across the captured Mac packet timelines, it leaves 152.4 ms worst-case extra single-packet delay and remains gap-free up to a 1.96x uniform slowdown.

The retained phone profile's normalized post-first generation pace is 1.44x slower than the pooled Mac capture. That is below the deterministic replay limit, so the existing evidence predicts a continuous queue under average conditions.

This is queue arithmetic over captured packet timestamps, not acoustic phone proof. Thermal throttling, Android scheduling, WebView audio-device startup, and bursty stalls still require the physical playback-head test.

## Current queue

| Metric across measured runs | Minimum | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Uniform post-first slowdown limit | 1.96x | 2.54x | 2.68x | 2.69x |
| Extra one-packet delay headroom | 152.4 ms | 154.3 ms | 156.1 ms | 156.6 ms |

The slowdown transform keeps first-PCM time fixed and multiplies every later packet's time distance from the first packet. The single-packet value is the smallest extra arrival delay that would exhaust the queued audio at the current 20 ms scheduling margin.

## Deterministic sensitivity matrix

Each cell is `runs with a gap / measured runs`.

| Startup cushion | 1.00x | 1.25x | 1.50x | 1.75x | 2.00x | 2.25x |
|---:|---:|---:|---:|---:|---:|---:|
| 0 ms | 0/20 | 0/20 | 5/20 | 20/20 | 20/20 | 20/20 |
| 20 ms | 0/20 | 0/20 | 0/20 | 0/20 | 2/20 | 9/20 |
| 40 ms | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 6/20 |
| 60 ms | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 6/20 |
| 80 ms | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 5/20 |
| 100 ms | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 4/20 |
| 140 ms | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 4/20 |
| 200 ms | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 4/20 |

## Phone interpretation

| Evidence | Mac capture pool | Retained Nubia profile |
|---|---:|---:|
| First PCM, median | 301.9 ms | 799.7 ms |
| Completion, median | 4765.8 ms | 6786.5 ms |
| Generated audio | 12.22 s | 11.42 s |
| Post-first wall time per audio second | 0.365 s | 0.524 s |

The phone profile retained only first-audio and completion timestamps, not every packet arrival. This ratio is an aggregate proxy, not a phone underflow measurement.

## Reproduction

```bash
python3 scripts/analyze_stream_queue_robustness.py \
  reports/android_phone/streaming_response/mac_lfm_streaming_2026-07-22.json \
  reports/android_phone/streaming_response/mac_lfm_streaming_restart_2026-07-22.json \
  --phone-profile reports/android_phone/native_profile/20260716t120118z/profile.json \
  --output reports/android_phone/streaming_response/queue_robustness.json \
  --report reports/android_phone/streaming_response/queue_robustness.md
```

The script produced the JSON metrics and this Markdown report without altering the captured evidence files.
