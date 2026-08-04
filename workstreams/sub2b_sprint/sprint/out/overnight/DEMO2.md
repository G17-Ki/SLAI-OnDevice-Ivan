# Demo 2 — “IT'S GOOD”

## Status

**PHONE ACCEPTED; FILM CAPTURED, NORMALIZED, AND MEDIA-VALIDATED.**

Final main APK SHA-256 `712632b7987557f02bd3bf67b77c00b49025207178f1719094bbb6a81cb2f53f` completed one cold offline EN turn at 2,452.154 ms and one ZH turn at 2,446.833 ms on the request-to-audio-start software proxy. Final-build engineering evidence is **indirectly receipt-bound n=1 per language**, not a distribution. Both traces report SenseVoice QNN execution, completed playback, and zero underflows. The accepted 42-row inventory totals 1,974,795,009 learned / 2,208,452,225 conservative and proves the required exclusions in the covered APK/private/TTS model roots. The per-run summaries do not self-authenticate APK, reboot, airplane-mode, or ADB-forward state; that binding comes from separate final inventory/protocol receipts.

The preserved main APK is byte-identical to its corresponding accepted installed main artifact, and the preserved runtime APK is byte-identical to its corresponding accepted installed runtime artifact. Their exact pair is bound by `device_bundles/demo2/accepted_apk_pair.json`.

Film: `films/DEMO2_ITS_GOOD.mp4`, **356.360 s**, 4,807,655 bytes, SHA-256 `68740c0d2d521d01564bebd53003bfd32e4bcef0c46b7f4a1ce732c27369871d`. It preserves the complete EN capture followed by the complete ZH capture and passed automated full-decode, audio-stream, volume, size, and hash checks. Detailed manual film-quality and semantic review was intentionally skipped; final viewing approval belongs to the user.

## Target configuration

| Role | Component | Route |
|---|---|---|
| ASR | SenseVoiceSmall | QNN/HTP, exact context/runtime family 2.40.0.251030, runtime-companion UID |
| Backbone | Qwen2.5-1.5B-Instruct Q4_K_M | CPU text generation |
| EN TTS | Supertonic 3, 5 steps | CPU/native offline bridge |
| ZH TTS | Matcha Baker + Vocos 22 kHz | CPU/native offline bridge |
| VAD | `AdaptiveEnergyVad` | dependency-free Java signal processing; 0 learned parameters |
| Languages | EN, ZH | explicit selector; memory off for acceptance |

The main process does not ship a second QNN provider. The only NPU statement authorized by the committed cold evidence is SenseVoice execution in the companion process with its matched 2.40 runtime.

## Cap ledger

Exclusive cap: **3,000,000,000**.

| Scope | Total | Evidence | Headroom | Physical status |
|---|---:|---|---:|---|
| **Installed phone inventory, learned unique** | **1,974,795,009** | physical `PHONE-MEASURED`; row counts `EXACT` | **1,025,204,991** | **PASS** |
| **Installed phone inventory, conservative resident** | **2,208,452,225** | physical `PHONE-MEASURED`; count basis `UB` | **791,547,775** | **PASS** |
| Intentionally overcounting host target, learned unique | 1,988,326,684 | `EXACT` | 1,011,673,316 | Safe bound; includes absent Vocos 24 kHz |
| Intentionally overcounting host target, conservative resident | 2,221,983,906 | `UB` plus `PHONE-MEASURED` Qwen serialized count | 778,016,094 | Safe bound; includes absent Vocos 24 kHz |
| Remove-LFM-only, full TTS laboratory retained | 2,672,875,114 | `UB` | 327,124,886 | Conditional on LFM and all rollback copies being absent |

The physical total uses installed TTS 131,471,497 learned / 131,755,017 conservative. The complete phone listing proves the staged but unreferenced Vocos 24 kHz file is absent. The earlier host bound keeps that row as a fail-safe overcount. The deployed VAD is `AdaptiveEnergyVad.java` SHA-256 `f43f07819d1bf8983809421bc6e0b436046fd7e7a96d2271a9209caf3db4da00` with zero learned parameters; no Silero artifact exists.

## Historical warm results already committed (older LFM-bearing APK)

All runs used airplane mode and had no ADB forwards. Warm distributions used explicit language selection and completed-only metrics.

| Metric | EN | ZH | Evidence |
|---|---:|---:|---|
| Completed warm turns | 21 | 20 | `PHONE-MEASURED` |
| First-audio proxy median | 1,333.866 ms | 1,031.277 ms | `PHONE-MEASURED` |
| First-audio proxy P95, nearest rank | 1,395.874 ms | 1,221.034 ms | `PHONE-MEASURED` |
| First-audio proxy max | 1,490.414 ms | 1,603.032 ms | `PHONE-MEASURED` |
| First-text median | 550.128 ms | 555.779 ms | `PHONE-MEASURED` |
| TTS first-chunk median | 587 ms | 92 ms | `PHONE-MEASURED` |
| TTS RTF median | 0.23664 | 0.03899 | `PHONE-MEASURED` |
| Playback underflows | 0 | 0 | `PHONE-MEASURED` |
| Dropped chunks | 0 | 0 | `PHONE-MEASURED` |

There was one excluded EN cancellation and two excluded ZH cancellations, each explicitly recorded as `replaced_by_new_input`; they are not silently counted as successful turns.

Cold ZH after reboot:

- Airplane mode on; ADB forward list empty before reboot and after boot.
- Only `ai.liquid.lfmdemo/.MainActivity` was explicitly launched; it bound and started the runtime companion.
- First-text proxy: 1,800.305 ms.
- First-audio start proxy: 2,651.683 ms — under the 3,000 ms gate.
- TTS first chunk: 258 ms; TTS RTF 0.11456.
- Playback completed; zero dropped chunks; zero underflows.
- SenseVoice QNN/HTP V81 execution verified under runtime UID 10514 using QNN 2.40.0.251030.

This cold-ZH record and the warm distributions above used main APK SHA-256 `143b96888005ed42ac97a07d281f76fa831d293ccaf7c0f2f11e19b5af5cb3f0`, which still included the LFM laboratory stack. They are retained as historical performance evidence; the final-build EN/ZH acceptance is recorded below.

## Final-pruned cold development record

Retain the whole sequence; the failed traces explain the accepted mitigation.

| Trace | Build boundary | Startup warmup before input | First text | First chunk queued | First-audio proxy | Result |
|---|---|---:|---:|---:|---:|---|
| `en_cold` | predates Qwen warmup | none recorded | 2,546.143 ms | 2,803.456 ms | 3,873.129 ms | **FAIL** |
| `en_cold_warmed` | generic one-token warmup build | 240.6 ms generic warmup | 2,716.036 ms | 2,972.446 ms | 4,040.407 ms | **FAIL** |
| `en_cold_prefix` | intermediate exact-prefix build, before final APK | 2,043.3 ms production-prefix warmup | 755.940 ms | 1,581.101 ms | 2,694.633 ms | timing **PASS**, not final-build acceptance |
| `en_cold_final_712` | final APK `712632…2f53f` | 2,075.8 ms EN production-prefix warmup | 758.862 ms | 1,368.373 ms | 2,452.154 ms | **PASS** |
| `zh_cold_prefix` | final APK `712632…2f53f`; indirect protocol/APK binding supplied by separate receipts | 2,482.8 ms EN startup + 549.8 ms ZH selection warmup | 617.906 ms | 2,153.444 ms | 2,446.833 ms | **PASS** |

Every completed trace above reports SenseVoice QNN execution, completed native playback, and zero underflows. The folder-level telemetry does not record the APK hash, reboot, airplane-mode, ADB-forward state, or verbatim transcript/response; separate final inventory/protocol receipts supply indirect installed-APK and phone-state binding. The startup and selection warmups happen before input is enabled. The hard gate is request-to-`UtteranceProgressListener.onStart`, not boot-to-audio and not acoustic onset.

## Quality basis

- Backbone screen, `HOST`, n=120/language: Qwen2.5 scored 42.5 EN and 54.875 ZH, the measured EN+ZH choice under Lane B's frozen decision rule.
- ASR screen, `HOST`, n=150/language: SenseVoice 9.170% EN WER and 6.547% ZH CER. OmniASR was rejected at 18.630% EN WER and 15.206% ZH CER.
- TTS screen, `HOST`: Supertonic passed bounded EN short-input and backcheck screens. Matcha is the only usable Mandarin research-demo voice, but silently omits Latin/OOV spans. No native-speaker listening acceptance exists.

## Final film capture

The normalized film combines two separate cold-boot recordings in EN-then-ZH order:

| Capture | Raw duration | Runner result | Request-to-in-app-audio-start proxy | Playback |
|---|---:|---|---:|---|
| EN / Supertonic | 227.106356 s | automated acceptance **PASS** | **1,988.922 ms** | complete |
| ZH / Matcha | 129.237756 s | automated acceptance **PASS** | **1,740.289 ms** | complete |

The ZH recording contains a setup attempt before TTS mode was enabled; it submitted no audio. The retained runner case is the first real request. The final MP4 is H.264 High 5.1, yuv420p, 1216×2688 at 25 fps with AAC LC 48 kHz mono audio. Full decode exit code was 0; mean/max audio volume was -36.6/-0.7 dB. These checks establish media integrity and automated runner acceptance, not manual film quality, native-speaker quality, acoustic onset, or whole-stack NPU execution.

## Final acceptance gates

- [x] Final no-LFM APK `712632…2f53f` and runtime APK `33bf0d…7e244` recorded.
- [x] Main `712632…2f53f` and runtime `33bf0d…7e244` preserved in `device_bundles/demo2/`; each binary is byte-identical to its corresponding accepted installed artifact. `accepted_apk_pair.json` binds the pair, while the main signature and deterministic reconstruction control are separately recorded.
- [x] All four LFM weights, extra GGUFs, stale TTS trees, and Silero weights proven absent.
- [x] Pruned EN+ZH TTS tree verified: 22 referenced files plus 3 metadata files.
- [x] Complete 42-row two-package/private/external inventory captured and cap recomputed.
- [x] Cold reboot, airplane mode, no forwards; final EN and ZH both passed the `<3,000 ms` request-to-first-audio software-proxy gate.
- [x] Untrimmed EN→ZH film captured and normalized; automated decode/audio/volume/hash checks passed.

## What this demo does not prove

- It does not measure acoustic onset; first audio is an in-app callback proxy.
- It does not prove whole-stack NPU execution, QNN 2.45 parity, or on-device NPU numerical parity against CPU.
- It does not establish native-speaker acceptance, MOS, or production quality.
- It does not establish mixed-script Mandarin coverage.
- It does not provide glasses, thermal, battery, or long-session evidence.
- It does not prove the same latency distribution on the final build; the ≥20-turn warm distributions belong to the older LFM-bearing APK, while final-build evidence is one cold accepted run per language.
