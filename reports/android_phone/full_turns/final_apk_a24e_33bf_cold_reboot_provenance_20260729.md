# Final rebuilt APK cold-reboot provenance — 2026-07-29

This note binds the final authenticated multilingual evidence to the rebuilt
APK pair installed on a Nubia NX809J (ADB serial redacted). It supersedes the
earlier `a9a3...` Japanese semantic limitation and the duplicated-particle
Korean fixture. Neither oracle was weakened: each canonical fixture was
promoted only after three stable direct QNN decodes and a strict one-case
end-to-end screen.

## Installed artifacts and phone-resident assets

| Package | Version | Installed/local APK SHA-256 |
|---|---|---|
| `ai.liquid.lfmdemo` | code 26, `0.6.0-unplugged` | `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a` |
| `ai.liquid.lfmdemo.runtime` | code 1, `0.1.0-unplugged-runtime` | `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244` |

The installed `/data/app` hashes matched the local build outputs. The main app
was installed with `adb install -r`; the runtime was installed only through
`scripts/install_app_owned_runtime.sh --serial <ADB_SERIAL>`. Both updates
preserved the phone-resident LFM, TTS, Qwen, SenseVoice/QNN, and Russian
Zipformer assets. The runtime installer transferred zero already-correct model
files and passed exact Qwen identity plus all five ASR gates:

- EN QNN: `The tribal chieftain called for the boy and presented him with 50 pieces of gold.`
- ZH QNN: `开饭时间早上9点至下午5点。`
- JA QNN: `フランス の 首都 は どこ で すか？`
- KO QNN: `프랑스의 수도는 어디인가요?`
- RU CPU: `расскажи коротко о москве`

The canonical semantic-oracle fixtures are mono 16-bit 16 kHz PCM. Japanese
uses Kyoko at rate 130, 79,464 bytes, SHA-256
`0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761`;
Korean uses Yuna at rate 130, 66,908 bytes, SHA-256
`0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de`.
Promoting the corrected Korean fixture changed only host-side acceptance bytes;
the installed APK pair and all phone-resident model assets were unchanged.

SenseVoice health reported `processor=htp_v81`, `execution_verified=true`,
service UID 10514, and service-owned execution. Russian reported the expected
CPU Zipformer route.

## Hardened warm acceptance

The direct RU/JA/KO TTS harness now requires one `engine_metrics` event per
utterance and an exact engine ID on every metric. It passed 3/3:

| Language | Engine | First chunk | RTF | Dropped PCM |
|---|---|---:|---:|---:|
| RU | `supertonic3_s5` | 749 ms | 0.239 | 0 |
| JA | `supertonic3_ja_s5` | 474 ms | 0.229 | 0 |
| KO | `supertonic3_ko_s5` | 500 ms | 0.274 | 0 |

Evidence:

- `../tts_acceptance/auto_ru_ja_ko_metrics_engine_gate_a24e_20260729.json`
- SHA-256: `5da9081b53a5ffc37743a59baa4de04173967ad17c1778a96287c445e4181aea`

The rebuilt pair then passed both authenticated warm matrices:

- JA/KO 2/2:
  `qwen2_5_ja_ko_strict_authenticated_a24e_33bf_yuna_20260729.json`, SHA-256
  `2e7ef274f13af2a7a502c47b0d63d56db2da05bc0892b1920bfffe2f9626263a`;
- EN/ZH/RU 3/3:
  `qwen2_5_en_zh_ru_authenticated_a24e_33bf_20260729.json`, SHA-256
  `6199860e94c5d74d912eb4cacd1d2e0f2e9c4297b09f617ca91d4b81ec554416`.

Every schema-3 case authenticated its exact WAV bytes before opening CDP and
recorded `sha256Matched=true`. Japanese now preserves `どこ` and receives the
positive answer `フランスの首都はパリです。`; Korean cleanly preserves
`프랑스의`, `수도`, and `어디` and receives `파리입니다.`. The corrected Korean
one-case screen is
`qwen2_5_ko_yuna130_fixture_screen_a24e_33bf_20260729.json`, SHA-256
`8b57a5e75d0b326e3d795c2392fbbbf6be19486dd765c0b5e1fdcbbcb4db602a`.

## Visibility and host-ADB lifecycle

Before Home, the warm PID set was main/runtime/Qwen/LFM
`16092/15964/16003/16187`. Home kept every PID unchanged, removed the important
binding and active-client wake lock, and left the started foreground runtime at
`hasBound=false`, `doRebind=true`. Returning the same task kept all PIDs and
restored `hasBound=true`, the important `CR IMP` binding, and
`ai.liquid.lfmdemo.runtime:active-client` under runtime PID 15964.

With zero forwards, the host ADB server was stopped for 70 seconds. After the
first reconnect, all four PIDs were unchanged, no forward existed, the runtime
was still foreground and started, its binding remained healthy, and the
active-client wake lock was still held. This closes the host-ADB shutdown gate
for the exact rebuilt hashes. Literal USB-cable removal is still an operator
gate.

## Final cold reboot and main-only launch

- Previous boot ID: `0285b43a-a5a2-42f2-b788-79758faff054`.
- Final boot ID: `474f629b-3756-4f29-89a2-598e7453e3f4`.
- Before launch: no main/runtime/Qwen/LFM PID, no forward, airplane mode `1`,
  and an empty route table.
- Only `ai.liquid.lfmdemo/.MainActivity` was launched.
- Cold activity start completed in 404 ms.
- Resulting PIDs: main 10556, runtime 10663, LFM 10690, Qwen 10718.
- `RuntimeService` was foreground, `startRequested=true`, and importantly bound
  to the main app; the active-client wake lock was held by PID 10663.
- The main-app journal records direct start attempt/result at epoch-ms
  `1785322388572/1785322388577` and Binder connection at `1785322388694`:
  5 ms to return and 122 ms from attempt to connection.
- `bootstrapAttempted=false`; the provider/activity fallbacks were not needed.

The current source also fixes two latent lifecycle failures represented by
these installed hashes: an ASR listener is assigned only after successful
`start()` so a later service self-heal can retry, and every foreground attach
has its own epoch so stale provider/activity callbacks cannot launch duplicate
fallbacks after Home/return.

## Authenticated post-reboot full turns

All five post-reboot warm-inference cases passed in airplane mode:

| Language | ASR / response / speech | First text | First-audio proxy | Native TTS metric |
|---|---|---:|---:|---|
| EN | SenseVoice HTP / Qwen2.5 CPU / LFM Audio CPU | 717 ms | 1126 ms scheduled; 1195 ms output head | first PCM 982 ms; 0 underflows |
| ZH | SenseVoice HTP / Qwen2.5 CPU / Matcha CPU | 809 ms | 1727 ms scheduled | 86/117 ms first chunks, RTF 0.033110/0.032503, 0 drops |
| RU | Zipformer CPU / Qwen2.5 CPU / Supertonic CPU | 2118 ms | 2582 ms scheduled | 290 ms first chunk, RTF 0.136462, 0 drops |
| JA | SenseVoice HTP / Qwen2.5 CPU / Supertonic CPU | 2061 ms | 2796 ms scheduled | 584 ms first chunk, RTF 0.271024, 0 drops |
| KO | SenseVoice HTP / Qwen2.5 CPU / Supertonic CPU | 1401 ms | 1912 ms scheduled | 390 ms first chunk, RTF 0.276976, 0 drops |

Evidence:

- JA/KO 2/2:
  `qwen2_5_ja_ko_final_cold_reboot_a24e_33bf_yuna_20260729.json`, SHA-256
  `bd138214ebf905b70b3a2a4ec2c9bafc6ad0b8d59421cee966ad7858d0525a7b`;
- EN/ZH/RU 3/3:
  `qwen2_5_en_zh_ru_final_cold_reboot_a24e_33bf_yuna_20260729.json`, SHA-256
  `6c2dca9c95d168503927002120d271070872726fc695801fa36180371775896a`.

These values are software scheduling/output-head proxies, not external acoustic
onset. The plans intentionally describe warm inference after phone-resident
services became ready; they do not measure first model load from process start.

The final lightweight host rerun passed fixture integrity, full-turn and direct-
TTS harnesses, evidence helpers, runtime/native-TTS source invariants, bundle
metadata and installer tests, streaming UI, Qwen request shaping, relevant
shell/Node/Python syntax checks, and `git diff --check`. It performed no ADB,
Gradle, Android rebuild, or model workload and did not change the worktree.

## Resource and claim boundary

After the final five turns, the four principal processes had 4,532,472 KiB
summed current `/proc` RSS and 4,828,004 KiB summed lifetime VmHWM. This excludes
separate Android system/WebView helper processes and is not a PSS measurement.
Thermal status was 0; current HAL maxima were about 30.5 °C CPU, 30.5 °C GPU,
29.7 °C NSP, 27.19 °C skin, and 28 °C battery, with cooling devices at 0. Main
and runtime both reported `isPendingFreeze=false`, `isFrozen=false`; the
filtered error log contained no matching demo, AndroidRuntime, LMK, freezer,
QNN/HTP, native TTS, or AudioTrack failure. The four PIDs were unchanged,
airplane mode remained `1`, the route table was empty, and the temporary CDP
forward had been removed.

Only SenseVoice EN/ZH/JA/KO is physically proven on HTP/NPU in the retail-phone
application. Russian Zipformer, Qwen2.5, LFM Audio, Matcha, Supertonic, eSpeak,
VAD, sampling, tokenization, and orchestration remain CPU. Exact-SM8850 AI Hub
profiles for Whisper, MeloTTS, Piper, compact Qwen/LFM, and other component
candidates remain research evidence until they execute under the Nubia app UID.

The final evidence uses one prerecorded, explicitly selected fixture per
language. The corrected fixtures close both the Japanese semantic loss and the
Korean duplicated-particle defect, but do not prove broad microphone/VAD/noise/
accent/automatic-language robustness. Literal cable removal, on-phone
microphone turns, captured-speaker PCM or human listening for audible language
quality, external acoustic-onset measurement, and an unplugged battery/power
run remain open operator or laboratory gates.
