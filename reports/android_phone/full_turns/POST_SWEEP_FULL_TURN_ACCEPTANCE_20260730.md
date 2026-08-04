# Post-sweep five-language full-turn acceptance

Date: 2026-07-30

## Outcome

The already deployed, phone-resident production stack passed five scripted
speech turns after the Whisper v2 candidate sweep:

`bundled WAV -> on-device ASR -> Qwen2.5-1.5B -> on-device TTS/playback`

All 5/5 cases passed their predeclared transcript, language, response,
backend, latency, TTS completion, and cleanup assertions. Airplane mode stayed
enabled throughout; the run used no model download, build, install, asset copy,
training, fine-tuning, or conversion.

## Route matrix

| Language | ASR | Backbone | TTS | Turn ms | Result |
|---|---|---|---|---:|---|
| EN | SenseVoice QNN/HTP | Qwen2.5-1.5B CPU | LFM Audio natural streaming CPU | 4,462.7 | Pass |
| ZH | SenseVoice QNN/HTP | Qwen2.5-1.5B CPU | Matcha `matcha_zh_baker` CPU | 14,545.5 | Pass |
| RU | Zipformer CPU | Qwen2.5-1.5B CPU | Supertonic `supertonic3_s5` CPU | 4,918.8 | Pass |
| JA | SenseVoice QNN/HTP | Qwen2.5-1.5B CPU | Supertonic `supertonic3_ja_s5` CPU | 4,162.2 | Pass |
| KO | SenseVoice QNN/HTP | Qwen2.5-1.5B CPU | Supertonic `supertonic3_ko_s5` CPU | 3,085.1 | Pass |

Observed transcripts and responses were native-language correct:

- EN: `What is the capital of France, answer in one short sentence?` ->
  `The capital of France is Paris.`
- ZH: `请用中文简单介绍一下北京。` -> a native Chinese Beijing summary.
- RU: `расскажи коротко о москве` -> `Москва - столица России.`
- JA: `フランス の 首都 は どこ で すか？` -> `フランスの首都はパリです。`
- KO: `프랑스의 수도는 어디인가요?` -> `파리입니다.`

## TTS evidence

- EN LFM Audio scheduled first audio at 2,293.8 ms, produced 27 chunks and
  2.14 seconds of PCM, and recorded zero underflows.
- ZH Matcha completed two sentence chunks with first-chunk values 89/129 ms,
  RTF 0.03334/0.03267, zero dropped chunks, and 12,641.1 ms playback span.
- RU Supertonic completed with first chunk 496 ms, RTF 0.23240, zero dropped
  chunks, and 2,112.9 ms playback span.
- JA Supertonic completed with first chunk 577 ms, RTF 0.25940, zero dropped
  chunks, and 2,202.6 ms playback span.
- KO Supertonic completed with first chunk 395 ms, RTF 0.28051, zero dropped
  chunks, and 1,398.7 ms playback span.

Every native-TTS lane reported one terminal `completed` event, matching engine,
locale, generation, and language evidence, with the original native handler
healthy and event parsing clean.

## Evidence

- EN/ZH/RU success matrix:
  `qwen2_5_en_zh_ru_post_whisper_20260730_r2.json`, SHA-256
  `b733b90c9220b543470c95951b30419d12ffb9bdaf5ef58f3993fa60a106d80f`.
- JA/KO success matrix:
  `qwen2_5_ja_ko_post_whisper_20260730.json`, SHA-256
  `5ced393ec4cecc96be7a520f42039c50823ff73a5c46b67fb88eb46f5ab20283`.
- The first EN/ZH/RU attempt was preserved as an operational failure because
  the app WebView was background-hidden and CDP timed out before any case ran:
  `qwen2_5_en_zh_ru_post_whisper_20260730.json`, SHA-256
  `9dfe131d05ae1c1f113a16aebbe02cc3b6adf5ac826133b36813202739ae360c`.
  Foregrounding the existing main activity made the same WebView target
  responsive; no application or model change was made before the passing r2.

## Final phone state

The temporary localhost WebView forward was removed. The isolated Whisper
probe remained stopped. Main/runtime processes remained alive, airplane mode
was `1`, the IP route and ADB forward list were empty, and protected APK hashes
were unchanged:

- main:
  `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`
- runtime:
  `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

This proves the present scripted five-language stack is phone-resident and
network-independent. Literal cable removal, live microphone/VAD/noise/accent
trials, audible speaker-quality review, and long unplugged battery/thermal
soak remain operator or laboratory gates.
