# Exact-APK airplane-mode acceptance — 2026-07-29

## Offline state

Immediately before this run, the cold-rebooted Nubia NX809J reported:

- `airplane_mode_on=1`;
- `Wifi is disabled`;
- no task-owned ADB forwards before setup; and
- the same post-reboot main/runtime PIDs, 10499 and 10598.

The phone was already in this radio state; the test did not change it. A
post-run `ip route` produced no routes, airplane mode remained 1, Wi-Fi remained
disabled, and the same two application PIDs remained alive. The temporary CDP
forward on host port 9224 was removed after evidence capture and the final ADB
forward list was empty.

USB ADB was used only as a test-control channel. Application inference used the
phone's own loopback endpoints and app-private model files.

## Exact installed pair

- main APK SHA-256:
  `2d9a90ce90824d15f143f7847bcbce21bbb3141de3dd011eadeb24db0935871b`
- runtime APK SHA-256:
  `f2f5e805ca524d2fdde7c62ba2ef3dc1e48a9818f105e447347ac37c5bf2070d`
- Linux boot ID: `2ae1f997-1d0f-47c3-971f-b4883a677f1b`

Both hashes were read from the installed device APKs earlier in the same boot,
and all four main/runtime/LFM/Qwen PIDs remained continuous through this run.

## Result

The exact EN/ZH/RU matrix passed **3/3**:

| Language | ASR | Response | Speech | Result |
|---|---|---|---|---|
| English | SenseVoice QNN/HTP | Qwen2.5-1.5B | LFM2.5 Audio | complete; all assertions passed |
| Mandarin | SenseVoice QNN/HTP | Qwen2.5-1.5B | Matcha Baker | complete; all assertions passed |
| Russian | Zipformer CPU | Qwen2.5-1.5B | Supertonic s5 | complete; all assertions passed |

Representative output:

- EN transcript: `What is the capital of France, answer in one short sentence?`
- EN answer: `The capital of France is Paris.`
- ZH transcript: `请用中文简单介绍一下北京。`
- RU transcript: `расскажи коротко о москве`
- RU answer: `Москва - столица России.`

Telemetry reports three complete turns, two NPU-ASR executions, two completed
native-TTS playbacks, zero TTS failures, zero underflows, zero generation-cap
turns, and zero quality or repetition warnings.

Latency proxies:

- request-to-first-text median: 811.7 ms;
- English first PCM: 996.5 ms;
- English first voiced output-head estimate: 1344.8 ms;
- native-TTS first-start proxy median: 2162.4 ms.

These are software timestamps, not microphone-observed acoustic onset.

## Evidence

- `qwen2_5_en_zh_ru_airplane_mode_exact_apks_20260729.json`
- raw evidence SHA-256:
  `2f106e7189094c7fcaf1bb63b66f123b01c72ca3e9ba87dce274c802746a6920`
- `qwen2_5_en_zh_ru_airplane_mode_exact_apks_20260729_telemetry/`

This establishes no-network operation for the exact accepted installed APK pair
recorded by this run. Literal physical removal of the USB cable remains a
separate operator acceptance gate.
