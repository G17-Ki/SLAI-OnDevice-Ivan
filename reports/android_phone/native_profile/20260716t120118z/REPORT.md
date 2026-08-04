# Native Nubia on-device profile

Date: 2026-07-16T12:02:14.538272+00:00

This profile measures the native `LFM Audio` APK and its app-owned CPU Q4 runner. It is not a QNN/NPU measurement.

## Latency and quality

Median / P95 in milliseconds:

| Workload | Runs | First text | First audio | Total | Exact ASR | Mean WER |
|---|---:|---:|---:|---:|---:|---:|
| question_asr | 10 | 250.9 / 272.4 | — | 442.5 / 470.9 | 10/10 | 0.00% |
| long_asr | 5 | 854.8 / 856.0 | — | 1554.4 / 1583.5 | 5/5 | 0.00% |
| question_chat | 5 | 677.2 / 700.5 | 799.7 / 837.4 | 6786.5 / 6876.9 | — | — |

## Workload definitions

- `question_asr`: the official 4.904-second mono `question.wav` sample (16 kHz). It asks, “Can you help me come up with a slogan for my woodworking site business?” The system prompt is `Perform ASR.`, so the expected output is only a transcript. This is the short-speech latency and transcription-accuracy test.
- `long_asr`: the official 18.356-second mono `asr.wav` sample (44.1 kHz), containing a six-sentence speech-recognition test passage. It uses the same `Perform ASR.` prompt. This checks how latency and accuracy scale with a substantially longer input.
- `question_chat`: reuses `question.wav`, but changes the system prompt to `Respond with interleaved text and audio.` The model answers the woodworking question as an assistant, streaming both response text and generated 24 kHz speech. It is not scored with WER because many different answers can be valid.

`First text` is time to the first streamed text fragment; `First audio` is time to the first generated audio chunk; and `Total` is time until the server's completion event. `Exact ASR` counts normalized transcript matches, while WER is word error rate. A dash means the metric does not apply.

The complete audio file is submitted with each request before inference begins. These are batch/file inference timings after utterance completion, not end-to-end timings that include speaking, microphone capture, or VAD endpoint detection.

## Process restart (warm filesystem cache)

- App launch to ready: median 899.3 ms, P95 906.2 ms across 3 launches.
- These launches restart the process but retain model pages in the Linux page cache; this is not a post-reboot cold-load measurement.

## Memory

- Model process peak RSS: 1873.4 MiB.
- Android UI/service process peak RSS: 297.0 MiB.
- Combined sampled peak RSS: 2169.1 MiB.

## CPU utilization probe

- A separate three-run interleaved-chat probe measured `/proc/<pid>/stat` CPU time around each request.
- Model process: median 7.86 average CPU cores utilized; maximum combined model + app utilization was 7.87 cores.
- Android UI/service process CPU time rounded to 0.00 s at the kernel's 100 Hz accounting resolution; inference compute resides in the model process.
- Probe first audio was 430.4–453.1 ms and total request time was 3.93–4.26 s.
- This probe reused the live persistent chat session, so its response length and total latency are not directly comparable with the fixed-output primary chat benchmark above.
- Raw measurements: `cpu_probe.json`.

## Thermal and battery

- CPU max: 59.0 → 59.4 °C.
- GPU max: 47.8 → 52.9 °C.
- Skin: 28.5 → 35.4 °C.
- Battery: 30.0 → 30.0 °C; level 100% → 100%.
- Charging during profile: True. Battery drain is therefore not reported as a valid power measurement.

## Notes

- `first_text` and `first_audio` are client-observed streaming times from request submission.
- Warm-up requests are excluded from reported distributions.
- Exact transcript checks use normalized word equality against the two official reference samples.
