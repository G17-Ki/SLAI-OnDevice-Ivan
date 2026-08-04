# Final APK cold-reboot acceptance provenance — 2026-07-29

> Historical scope: this note binds the earlier `a9a3...` / `258d...` pair and
> its superseded Japanese fixture. The rebuilt final pair closes the Japanese
> semantic and host-ADB gates; see
> `final_apk_a24e_33bf_cold_reboot_provenance_20260729.md`.

This note binds the final multilingual phone evidence to the APK pair installed
on a Nubia NX809J (ADB serial redacted). It also records the exact boundary of the
remaining host-side acceptance gate.

## Installed artifacts

The following hashes were read from the installed `/data/app` APKs immediately
before the final reboot:

| Package | Version | Installed APK SHA-256 |
|---|---|---|
| `ai.liquid.lfmdemo` | code 26, `0.6.0-unplugged` | `a9a3ce3639466bb8a1c51b61c78f9fbd3d88a015313d1d5d79c2feb767a1a628` |
| `ai.liquid.lfmdemo.runtime` | code 1, `0.1.0-unplugged-runtime` | `258d8d5ab26d10a1bb9578549d318d797008c0cb8a088efa0c3a9feb939c2a55` |

The main hash matches
`android-app/app/build/outputs/apk/debug/app-debug.apk`. The runtime companion
and all private Qwen, SenseVoice/QNN, and Russian Zipformer assets had already
passed the required app-owned installer integrity checks; the final main-only
update did not clear either package's data.

## Final reboot and main-only launch

- Previous boot ID: `2ae1f997-1d0f-47c3-971f-b4883a677f1b`.
- Final cold-boot ID: `abb8d464-40e4-4822-9318-b00b8604bfb1`.
- Before launch: no demo/runtime/model PID, no ADB forward, airplane mode `1`,
  and an empty IP route table.
- The device has no secure lock. Android reported user 0 `RUNNING_UNLOCKED`; the
  non-secure swipe keyguard was dismissed before launch.
- Both Nubia Apps AI-control policies survived with
  `selfStartPolicy=1`, `relatedStartPolicy=1`, `selfStartMode=3`, and
  `relatedStartMode=3`; doze and frozen whitelist membership remained true.
- Only `ai.liquid.lfmdemo/.MainActivity` was launched.
- The resulting PID set was `10671 10779 10802 10834`. The main PID is `10671`
  from its WebView socket and the runtime PID is `10779` from its wake-lock
  owner. LFM, Qwen, and ASR listened only on loopback ports `8080`, `6020`, and
  `6010` respectively.
- `RuntimeService` was foreground, `startRequested=true`, and had the important
  main-to-runtime binding. The partial wake lock
  `ai.liquid.lfmdemo.runtime:active-client` was held by runtime PID `10779`.
- The app-private launch journal recorded direct start attempt/result at
  epoch-ms `1785314549406/1785314549411` and Binder connection at
  `1785314549526`: about 5 ms to return and 120 ms from attempt to connection.
  `bootstrapAttempted=false` and connection recoveries remained zero.

## Physical acceptance on the final APK

Direct RU/JA/KO native-TTS routing and playback passed 3/3 before reboot:

- evidence: `../tts_acceptance/auto_ru_ja_ko_metrics_fixed_20260729.json`;
- SHA-256: `b49c2f59ed3332e634a87e81c59ac539365368d9ba463eacc5559599226fe457`;
- first chunk: RU 688 ms, JA 505 ms, KO 479 ms;
- RTF: RU 0.220, JA 0.243, KO 0.263;
- every engine metric carried the owning WebView generation, kept the private
  `engine_generation` separate, reported exact `generation_language`, and had
  zero dropped chunks.

The final APK completed both warm airplane-mode matrices under the embedded
routing/content plan before reboot:

- JA/KO 2/2:
  `qwen2_5_ja_ko_final_apk_a9a3_20260729.json`, SHA-256
  `51d9007cc001c45bc0cba9610aadaa9e32ce64ae126f89d5a770628e6d963d3c`;
- EN/ZH/RU 3/3:
  `qwen2_5_en_zh_ru_final_apk_a9a3_passed_20260729.json`, SHA-256
  `6ddf45c78e4adcf849e22c8298e07108ba07b0e8b6611f74dd97a32299767aa6`.

On the new cold boot, with airplane mode still enabled and the route table
empty, all five full turns completed and passed that same embedded plan again:

- EN/ZH/RU 3/3:
  `qwen2_5_en_zh_ru_final_cold_reboot_a9a3_20260729.json`, SHA-256
  `fdd8a9ffb4e779a91d99181c80afeafb24c728fed5353593533788e382d71a05`;
- JA/KO 2/2:
  `qwen2_5_ja_ko_final_cold_reboot_a9a3_20260729.json`, SHA-256
  `4777755334862aa9504fce131f285b15b8aecba80697c2c06da2a0147f52f546`.

Across the two warm and two post-reboot full-turn matrices, the embedded plan
reports 10/10; including direct RU/JA/KO TTS, it reports 13/13. A stricter final
re-audit preserves every routing, NPU, TTS, and latency result but rejects the
Japanese transcript because it lost the question marker `どこ`. See
`final_apk_a9a3_strict_reaudit_20260729.md`. The non-canonical earlier
false-negative reports are retained for diagnosis but must not be cited as
final verdicts.

Post-reboot route summary:

| Language | ASR | Response | Speech output | Key latency evidence |
|---|---|---|---|---|
| EN | SenseVoice QNN/HTP | Qwen2.5 CPU | LFM Audio CPU | first text 1967 ms, first PCM 2210 ms |
| ZH | SenseVoice QNN/HTP | Qwen2.5 CPU | Matcha CPU | first text 822 ms, TTS first chunk 97 ms, RTF 0.039 |
| RU | Zipformer CPU | Qwen2.5 CPU | Supertonic CPU | first text 2168 ms, TTS first chunk 490 ms, RTF 0.230 |
| JA | SenseVoice QNN/HTP | Qwen2.5 CPU | Supertonic CPU | first text 897 ms, TTS first chunks 599/552 ms |
| KO | SenseVoice QNN/HTP | Qwen2.5 CPU | Supertonic CPU | first text 1195 ms, TTS first chunk 366 ms, RTF 0.261 |

All cold-boot native-TTS metrics reported zero dropped chunks. The JA fixture
still transcribes `フランス の 首都 は 田クスカ。`; Qwen nevertheless answers
that Paris is the capital, but the stricter ASR semantic gate now fails because
`どこ` is absent. The Korean transcript contains the known duplicate particle
`프랑스의의`; it retains `어디` and the positive answer remains correct.

These are post-reboot **warm inference** software timings, not cold model-load
latency: the plans deliberately set `serverWarmth` to `warm`, and acceptance
ran after all loopback services were ready. `firstAudioScheduledMs` and Android
`onStart` are scheduling/playback proxies, not microphone-observed acoustic
onset.

## Lifecycle and remaining host gate

Sending the app Home kept all four PIDs unchanged, removed the important
binding and active-client wake lock, and left the foreground runtime started
with `doRebind=true`. Bringing the existing task back to the foreground
succeeded. A redundant post-return read-only snapshot and the planned
70-second `adb kill-server` repetition were blocked by the execution approval
service returning `503 auth_unavailable`; this is a host-tooling blocker, not a
phone failure. No ADB forward remained at that point.

The earlier accepted APK pair passed the same 70-second host-ADB shutdown gate,
but that prior result must not be silently attributed to final main hash
`a9a3ce...`. The final hash therefore still needs one repeated host-ADB shutdown
comparison when command approval is available. Literal physical USB removal is
also an operator gate.

## Claim boundary

Only SenseVoice EN/ZH/JA/KO has physically proven HTP/NPU execution in this
application. Qwen2.5, LFM Audio, Matcha, Supertonic, Russian Zipformer, VAD,
sampling, tokenization, and orchestration remain CPU. Existing AI Hub work for
Whisper Small, MeloTTS-ZH, Piper, and compact Qwen/LFM components is useful
exact-SM8850 evidence but is not Nubia app-UID execution.

The full-turn JSON's `staticInspection.npuFacts` is stale pre-run UI state and
can contradict the executed request. NPU conclusions here use each case's
`asrFacts` (`provider=qnn`, `npu_executed=true`, `npu_selected=true`) together
with the separate runtime health evidence reporting `processor=htp_v81` and
service-owned execution.

Current JA/KO evidence proves explicit ASR hint/tag, HTP execution, response
script/content, exact TTS generation configuration, PCM lifecycle, and playback
completion. Japanese does not yet pass the strengthened semantic transcript
gate. Neither language has audible-naturalness proof; captured PCM with
fixed-locale ASR or a listening check remains required for that claim.
The full-turn fixtures are prerecorded WAVs injected with explicit language
selection; they do not cover microphone capture, VAD close, noisy rooms,
accents, automatic language routing, or barge-in. Host and WebView/device wall
clocks are also offset, so comparisons use monotonic/elapsed fields rather than
cross-clock timestamps.
