# Connected EN/ZH/RU resource snapshot — 2026-07-29

This snapshot was taken immediately after the passing three-language run in
`qwen2_5_en_zh_ru_connected_wakelock_rebind_20260729.json`. It includes every
persistent process required by the current demo: the main app, its WebView
renderer, LFM2.5-Audio, the runtime companion, and Qwen2.5. The short-lived ASR
runner had already exited, so its transient peak is not included.

## Functional state

- Three of three full turns passed every raw assertion.
- App-private telemetry reports three complete turns, zero queue underflows,
  two native-TTS playback completions, zero native-TTS failures, and two turns
  whose ASR physically executed through QNN/HTP.
- EN response: `The capital of France is Paris.`
- ZH response: `北京是中国的首都，位于中国北部，是世界上最大的城市之一。`
- RU response: `Москва - столица России.`
- ZH resolved to Matcha Baker: 87 ms first synthesis chunk, 255 ms synthesis,
  7.117 s audio, RTF 0.036, zero dropped chunks/write retries.
- RU resolved to Supertonic s5: 508 ms first synthesis chunk, 512 ms synthesis,
  2.147 s audio, RTF 0.238, zero dropped chunks/write retries.

## Memory after the run

`dumpsys meminfo` values are point-in-time PSS/RSS. `VmHWM` is each process's
individual lifetime high-water RSS; summing those values is an upper bound,
not a simultaneously sampled combined peak.

| PID | Process | PSS KiB | RSS KiB | VmHWM KiB | Threads |
|---:|---|---:|---:|---:|---:|
| 716 | runtime companion | 47,899 | 203,348 | 203,152 | 35 |
| 751 | Qwen2.5 server | 1,985,404 | 1,988,308 | 2,121,876 | 6 |
| 31675 | main app / native TTS | 454,826 | 673,260 | 773,756 | 80 |
| 31724 | main WebView renderer | 85,309 | 255,104 | 256,692 | 31 |
| 31785 | LFM2.5-Audio server | 1,732,281 | 1,735,264 | 1,739,180 | 13 |
| **Total** | persistent current stack | **4,305,719** | **4,855,284** | **5,094,656 upper bound** | **165** |

The current persistent stack therefore occupies about 4,204.8 MiB PSS and
4,741.5 MiB RSS after a full multilingual run. Qwen2.5 and LFM2.5-Audio account
for most of the footprint. Future model selection must avoid accidental dual
backbone residency beyond this intentional Qwen-response/LFM-English-voice
combination.

## Latency telemetry

- Request to first text SSE: median 1,901.9 ms; max/P95 2,022.6 ms.
- EN request to first PCM SSE: 2,177.7 ms.
- EN estimated voiced output head: 2,604.0 ms. This excludes the Android
  mixer/HAL, DAC, speaker, and acoustic propagation.
- Native TTS first-audio-start proxy: median 2,247.4 ms; max 2,708.2 ms.
- Request to complete: median 4,818.4 ms; max/P95 8,852.5 ms.

The native TTS timing is `UtteranceProgressListener.onStart`, not microphone-
measured audible onset. External acoustic recording remains required.

## Thermal, LMK, and freezer state

- Android thermal status: 0; all current cooling-device values: 0.
- Current HAL temperatures after the run: CPU max 30.5 °C, GPU max 30.1 °C,
  NSP max 30.1 °C, skin 27.24 °C, battery 28.0 °C.
- Battery: 100%, 28.0 °C, plugged. Cached thermal entries near 99 °C were stale
  framework cache values and are not used as the current reading.
- Filtered run log: no AndroidRuntime crash, LMK/low-memory kill, Sherpa/bridge
  error, or freezer event.
- Runtime companion remained PID 716, adj 0 / BTOP, `isFrozen=false`, with a
  live important binding. Nubia still reported `isPendingFreeze=true`, so this
  OEM diagnostic remains a risk marker even though the 10-minute health soak
  and all three full turns completed normally.

## Evidence

- Raw run: `qwen2_5_en_zh_ru_connected_wakelock_rebind_20260729.json`
- Telemetry: `qwen2_5_en_zh_ru_connected_wakelock_rebind_20260729_telemetry/`
- Raw-run SHA-256:
  `aa8b5641e12572bed5a60cb83acb25c9dd25a5041eff3f56b792d3fea37849e4`
