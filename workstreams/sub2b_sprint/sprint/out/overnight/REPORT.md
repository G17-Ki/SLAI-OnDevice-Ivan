# Overnight handoff — two offline speech demos

## Read this first

**Wake-up status: both demos are filmed, normalized, and media-validated.** Both configurations remain green against their engineering gates. The full handoff captures are intentionally untrimmed, and final viewing/editorial approval is left to the user.

### Earlier post-capture voice-default checkpoint

English and Russian silence was a routing-plus-volume issue, not missing weights. After repairing that path, the user auditioned the expanded offline catalog and selected **Supertonic EN voice 4** and **Piper RU Ruslan**.

At that checkpoint, the phone had the authenticated **26-candidate audition catalog**: **397 resident files verified across 773.7 MiB**. The final default switch was metadata-only—zero model files were transferred in that step. `Automatic by language` survived a cold app restart and resolved EN→`supertonic3_en_v4_s5`, RU→`piper_ru_ruslan`, and ZH→`matcha_zh_baker`; the existing ES/FR defaults and the prior EN/RU Supertonic routes remained available as fallbacks.

At that earlier checkpoint, automated on-phone checks were green 3/3 with zero dropped chunks and completed playback: EN voice 4 first chunk **844 ms**, RU Ruslan **49 ms**, and ZH Matcha **57 ms**. The existing final Russian film answer remained **“Москва”**, which is factually correct. The phone was foregrounded on **Voice answer · Auto · Automatic by language**, airplane mode was on, Wi-Fi was off, all ADB forwards were removed, and media volume was **80/150**. That earlier checkpoint was safe to unplug. Full receipt: `evidence/tts_voice_defaults/deployment_receipt.json`.

The Demo 1 and Demo 2 film/inventory receipts below predate both the audition-catalog/default checkpoint above and the later latency/preload deployment below. Their accepted parameter totals and filmed routes remain historical acceptance evidence; neither later deployment recalculates or rebinds either parameter cap.

### Current selected-route latency/preload checkpoint

The current post-film build keeps input disabled until the explicitly selected offline route is ready: ASR health must report the expected resident route, the Qwen prompt/cache warm language must match the selection, and the TTS preload key must match `policy|locale`. This is one selected route at a time, not simultaneous residency of every language. Fresh installs default to explicit English, and an explicit language choice is persisted.

In final receipt-bound, explicitly selected, warm n=1 phone turns, request-to-Android-`UtteranceProgressListener.onStart` measured **1,008.482 ms EN**, **597.038 ms ZH**, and **461.273 ms for the longer RU case**. First text was 237.300/192.900/220.300 ms and TTS first chunk was 604/90/66 ms respectively. All three passing cases completed playback with zero dropped chunks. EN/ZH used the matched SenseVoice QNN ASR route; RU used Zipformer CPU. These are software start proxies, not acoustic onset, cold/boot-to-audio, or a distribution, and the evidence does not isolate preload as the cause of the improvement.

| Stage | What the final checkpoint supports | Safe conclusion |
|---|---|---|
| Listening / endpointing | The current fixture metric begins after WAV capture, so user speech duration and microphone/VAD end-of-speech delay are outside it. The selected recognizer is resident before input is enabled. | The build moves selected-recognizer setup before input is enabled; this checkpoint does not quantify a cold-load saving. VAD tightening might save time but is not measured here and could clip speech, so it was not changed. |
| ASR + LLM to first text | Combined first-text time is 193–237 ms across the three accepted routes. The acceptance cases use the exact deployed Qwen2.5 model and are labelled warm; direct warmup events were not captured in these files. | The deployed readiness gate passed these warm n=1 routes. The receipt does not separately apportion ASR versus Qwen time or prove preload causality. |
| Speech generation | First chunk is 604 ms EN, 90 ms ZH, and 66 ms RU with the selected voices. | In the EN trace, Supertonic's 604 ms first-chunk interval is the largest visible post-text interval. ZH Matcha and RU Piper are already much shorter in these n=1 traces. |

With explicit English selected, the resident main, WebView, runtime, Qwen, and one ASR process totaled **2.510 GiB PSS / 2.893 GiB RSS**, while the phone still reported **9.106 GiB MemAvailable**. Therefore Qwen plus exactly one selected ASR runner and one selected TTS graph fits comfortably. Loading every language route simultaneously was neither needed nor tested; keep one selected ASR route and one selected TTS graph resident, and never load a second QNN runtime provider into the same process.

Two retained shortest-RU attempts returned the factually correct answer **“Москва.”** and reached the same start proxy at **340.969 ms** and **248.384 ms**, but both lacked `utterance_done` and terminated failed with `playback_drain_timeout`. Only the longer RU case is a completed-playback acceptance; short-utterance playback stability remains open.

With the app foregrounded, airplane mode enabled, Wi-Fi disabled, the IP route empty, and ADB forward/reverse lists empty, the host ADB server was stopped for 75 seconds and then reconnected. The boot ID and main, runtime, Qwen, ASR, and WebView PIDs were unchanged; the runtime foreground service and its partial wake lock remained active, and a temporary post-check confirmed persisted explicit English, TTS Auto, healthy text service, the resident English ASR route, and full selected-route readiness. That temporary forward was removed and the final forward/reverse lists were empty. This supports only 75-second host-debugger-disconnected survival with the app foregrounded. It is not a physical-unplug, app/process restart, process-death recovery, boot persistence, background/always-on, or long-duration reliability test. Full receipt: `evidence/latency_preload/deployment_receipt.json`.

| Demo | Current handoff | Morning action |
|---|---|---|
| **Demo 2 — “IT'S GOOD”** | Engineering pass: complete 42-row inventory at **1,974,795,009 learned / 2,208,452,225 conservative**, below 3B. Film complete: `films/DEMO2_ITS_GOOD.mp4`, **356.360 s**, 4,807,655 bytes, SHA-256 `68740c0d2d521d01564bebd53003bfd32e4bcef0c46b7f4a1ce732c27369871d`; full decode, audio-stream, and hash/media checks passed. The filmed cold-run receipts passed EN at **1,988.922 ms** and ZH at **1,740.289 ms** on the request-to-in-app-audio-start proxy. | Watch and approve the full EN-then-ZH capture. No manual film-quality or Demo 2 semantic review was performed. |
| **Demo 1 — “IT FITS”** | Engineering pass under **1,820,982,302 conservative elements**, strictly below 2B. Film complete: `films/DEMO1_IT_FITS.mp4`, **5,701.162 s**, 46,810,178 bytes, SHA-256 `cf7ba53b9cf1a387b4c52501161253906e1002284b7cc48bd4907a0e80367958`; full decode, audio-stream, and hash/media checks passed. | Watch and approve the untrimmed film. Manual review was intentionally limited to the final RU visible answer, **“Москва”**, which is factually correct. An earlier failed RU attempt remains visible before the successful retry. |

Open [`DEMO_SHOTLIST.md`](DEMO_SHOTLIST.md) to locate the expected turns in the untrimmed films. Detailed records are [`DEMO2.md`](DEMO2.md) and [`DEMO1.md`](DEMO1.md); cap arithmetic is in [`CAP_LEDGER.md`](CAP_LEDGER.md); adjacent overclaims are in [`CLAIMS_AND_LIMITS.md`](CLAIMS_AND_LIMITS.md). The two engineering acceptance receipts are `evidence/demo2/final_pruned/inventory_receipt.json` and `evidence/demo1/final_inventory/inventory_receipt.json`.

For any later edit or narration, do not say whole-stack NPU, Parakeet NPU, native-speaker approved, acoustic onset, production-ready, or anything about glasses. The automated film checks establish media integrity, not semantic or film-quality approval.

## The three findings that hold

1. **The parameter cap was never the binding constraint.** Lane A's six audited stacks all have known-row totals below 2B. Their measured / worst-case totals are H 1,396,245,218 / 1,396,245,218; L 1,676,733,894 / 1,720,220,241; D 1,831,533,225 / 1,831,533,225; M 1,674,645,423 / 1,880,539,567; Q 1,964,482,923 / 1,964,482,923; and F 1,975,632,107 / 1,975,632,107. Only H has every arithmetic row closed; the other verdicts remain `UNVERIFIED` because of explicit `UNKNOWN` frontend rows. Licences are recorded but do not gate this research demo.

2. **Mandarin has one usable research-demo voice, and it has a real defect.** Kokoro v1.1 ZH emitted non-finite PCM on reproduced host probes. Matcha Baker + Vocos emits finite, fast Mandarin, but its frontend silently ignores unsupported Latin/OOV spans. Keep acronyms and digits out of the successful Mandarin filming line, and include the mixed-script failure shot.

3. **Published acceleration paths exist, but version matching is mandatory.** This account could discover but not compile-select SM8850. The phone evidence does prove SenseVoice execution with one matched QNN context/runtime set at 2.40.0.251030 under the runtime-companion UID. It does not prove QAIRT/QNN 2.45 parity, Parakeet NPU execution, or whole-stack NPU execution. Never load two QNN runtime providers in one process.

## Demo 2 — “IT'S GOOD”

Configuration: SenseVoiceSmall ASR; Qwen2.5-1.5B Q4_K_M text backbone; Supertonic 3 EN; Matcha Baker + Vocos ZH; dependency-free `AdaptiveEnergyVad`; EN+ZH only. ASR uses the version-matched SenseVoice QNN 2.40 route; the text backbone and TTS legs are not claimed as NPU execution.

The accepted physical inventory is 1,974,795,009 learned parameters and 2,208,452,225 conservative resident elements, leaving 791,547,775 conservative headroom below 3B. The 42-row receipt covers both APKs, both private `files/` trees, and all 25 files in external `tts-eval`; it proves one GGUF only, no LFM weights, no relevant/8080 listener, no Silero weights, and no sibling stale TTS tree in the covered model roots. The broader host target bound remains 1,988,326,684 / 2,221,983,906 because it intentionally counts an uninstalled Vocos 24 kHz row.

The accepted pair is preserved in `device_bundles/demo2/`: the main APK `712632…2f53f` at 20,197,812 bytes is byte-identical to its corresponding accepted installed main artifact, and the runtime APK `33bf0d…7e244` at 121,215,600 bytes is byte-identical to its corresponding accepted installed runtime artifact. `accepted_apk_pair.json` binds the pair; the main APK's v2 signature, signer certificate, embedded `de57389` UI asset, and byte-for-byte control reconstruction are recorded in `evidence/demo2/final_pruned/apk_reconstruction_receipt.json`. A clean rebuild with a different hash is explicitly rejected as a substitute.

Phone performance already demonstrated on the older LFM-bearing APK:

| Language | Warm completed turns | First-audio proxy median | P95 | Max | Playback |
|---|---:|---:|---:|---:|---|
| EN | 21 | 1,333.866 ms | 1,395.874 ms | 1,490.414 ms | 21/21 complete; 0 underflows; 0 dropped chunks |
| ZH | 20 | 1,031.277 ms | 1,221.034 ms | 1,603.032 ms | 20/20 complete; 0 underflows; 0 dropped chunks |

The retained older-build cold record contains one cold-reboot, airplane-mode, no-forward Mandarin turn at a 2,651.683 ms first-audio proxy. Only the main app was explicitly launched; it bound and started the runtime companion, playback completed, and SenseVoice QNN/HTP execution was verified under the companion UID. That historical trace used the LFM-bearing APK and is not the final cap-valid acceptance. “First audio” is `UtteranceProgressListener.onStart`, not measured acoustic onset.

The final-pruned cold work retained two English failures rather than hiding them: 3,873.129 ms without a Qwen warmup and 4,040.407 ms after a generic one-token warmup. Warming the exact production prefix reduced an intermediate-build English run to 2,694.633 ms. The immutable final APK then passed one cold English turn at 2,452.154 ms after a visible 2,075.8 ms startup warmup; one cold Mandarin turn passed at 2,446.833 ms after a visible 2,482.8 ms startup warmup and a 549.8 ms selection-triggered ZH warmup. Final-build evidence is therefore **indirectly receipt-bound n=1 per language**, not a distribution: the per-trace summaries do not themselves authenticate the APK, reboot, airplane-mode, or ADB-forward state. Those warmups occur before input is enabled, so the `<3 s` metric remains request-to-in-app-audio-start, not boot-to-audio. The intermediate English pass is retained only as development evidence.

The final untrimmed handoff film concatenates two separately cold-booted captures in EN-then-ZH order. Automated runner receipts passed the filmed English request at a **1,988.922 ms** request-to-in-app-audio-start proxy and the filmed Mandarin request at **1,740.289 ms**; both completed playback. The normalized film is H.264 High 5.1 at 1216×2688 with AAC LC mono audio, **356.360 s**, 4,807,655 bytes, SHA-256 `68740c0d2d521d01564bebd53003bfd32e4bcef0c46b7f4a1ce732c27369871d`. A full decode exited 0 and audio measured -36.6 dB mean / -0.7 dB max. These are automated integrity and runner checks, not a manual semantic, acoustic-onset, or editorial approval.

Quality selection comes from bounded host screens, not native-speaker acceptance: Qwen2.5 scored 42.5 EN and 54.875 ZH at n=120 per language; SenseVoice measured 9.17% EN WER and 6.55% ZH CER at n=150; OmniASR was rejected at 18.63% EN and 15.21% ZH; Supertonic and Matcha were the technically usable voice routes. There was no native-speaker listening panel.

## Demo 1 — “IT FITS”

Configuration: exact text-only Qwen3.5-0.8B Q4_0; SenseVoiceSmall for EN/ZH; Parakeet-TDT-0.6B-v3 INT8 on CPU for ES/FR; RU Zipformer INT8 on CPU; shared Supertonic 3 for EN/ES/FR/RU; Matcha Baker + Vocos for ZH; zero-parameter `AdaptiveEnergyVad`.

The accepted physical inventory is **1,810,531,015 unique learned parameters and 1,820,982,302 conservative resident elements**, leaving **179,017,698 conservative headroom** below the exclusive 2B cap. Its 61 rows cover two APKs, 34 private files, and 25 external TTS files. The receipt proves one exact Qwen3.5 GGUF, Qwen2.5/LFM/Silero/Vocos-24k/stale-TTS/partial files absent, no LFM process, no enumerated port-8080 listener, and no Parakeet QNN context/runtime payload. The unprivileged listener probe could not expose owning PIDs, so this is not a “no network sockets” claim. The older host target remains a safe overcount at 1,824,062,690 / 1,834,513,983 because it retained the absent Vocos 24 kHz export.

The Qwen GGUF itself is pinned at 507,154,688 bytes, SHA-256 `1f2002727f6f9c9ea39bdb0eefb58703d5d2652d55a185902aab5dcf1b4ad701`, and contains exactly 752,393,024 tensor elements with no MTP/draft or vision/mmproj tensors.

At the accepted Demo 1 inventory checkpoint, all intended payloads were provisioned: exact Qwen3.5 `444406…c652c`, Qwen2.5 absent, the then-current five-candidate TTS tree verified 22/22 plus 3 metadata files, and CPU-only Parakeet installed from 16 pinned rows with no QNN/Qualcomm/context payload. The later 26-candidate audition deployment is separately receipt-bound and is not folded into that historical cap inventory.

A separate CPU ASR health check executed Parakeet for Spanish and French under runtime service UID 10514 and reproduced the exact expected transcripts. The measured host round trips through a temporary diagnostic ADB forward were 5,828.660 ms for ES and 5,526.684 ms for FR; the forward was then removed and the final forward list was empty. This proves phone-side CPU backend execution and exact transcript parity only. It is not a no-forward app voice turn, not full ASR→LLM→TTS latency, and not Parakeet NPU evidence.

The first phone gate is closed. In the uninterrupted cold-boot sequence, English completed with the exact SenseVoice QNN route, Qwen3.5-0.8B CPU backbone, and Supertonic 3 EN voice. The verbatim ASR transcript was “What is the capital of France, answer in one short sentence?”; the visible response was “Paris is the capital of France.” First audio began at the 2,339.209 ms in-app callback proxy and playback completed at 4,840.146 ms, with zero underflows or dropped chunks. Qwen emitted one reasoning-tag block spanning 19 filtered raw characters; the UI exposed none of it. This supports **no user-visible reasoning leakage**, not a claim that Qwen emitted no reasoning markers. Evidence is committed at `cb8b306` in `evidence/demo1/five_language_cold/en_checkpoint/provenance.json`.

The historical Russian engineering checkpoint completed in the same boot and the same main/runtime processes through Zipformer CPU, Qwen3.5 CPU, and Supertonic 3 RU. Its prompt was “расскажи коротко о москве”; first audio began at 4,437.486 ms and playback terminated at 15,605.082 ms after two TTS chunks, with zero underflows or dropped chunks. Its answer incorrectly calls Moscow “the largest city in the world,” so it remains route-coverage evidence only. Evidence was accepted at `794871f` and corrected at `471db9c`; the current provenance SHA is `3e3995a4…6aed`.

The final film used the separate fixture `evidence/demo1/fixtures/question_ru_simple.wav`, text **“Какова столица России?”**, SHA-256 `e8d931545ebddf124d63acd5d42bbde10fab7ba2f3f65998ce2ab7237693bc45`. The final visible answer was **“Москва”**, which is factually correct. No latency from the historical Russian checkpoint is attributed to this new fixture. The untrimmed film retains an earlier failed Russian attempt before the successful retry.

Mandarin then completed in that same uninterrupted cold process through SenseVoice QNN/HTP, Qwen3.5 CPU, and Matcha ZH. The verbatim transcript was “请用中文简单介绍一下北京。”; the visible 40-character response was Han-only, contained no Latin letters or digits, and included “北京”. First audio began at 1,549.215 ms and playback completed at 11,291.198 ms after two Matcha chunks, with zero underflows or drops. This avoids the known mixed-script/OOV hazard for the accepted line; it does not prove that Matcha handles Latin spans. Evidence is committed at `a001915`, provenance SHA `1334ca78…5eefb`.

Spanish then completed through Parakeet CPU, Qwen3.5 CPU, and Supertonic ES. It reproduced “Dime brevemente cuál es la capital de España.” exactly and displayed “La capital de España es Madrid.” First audio began at 3,343.357 ms and playback completed at 6,104.679 ms, with one TTS chunk and zero underflows or drops. This is a correct filmable coverage turn and makes no Parakeet NPU claim. Evidence is committed at `1948535`, provenance SHA `c4f5dd90…c680`.

French closed the fifth-language execution gate through Parakeet CPU, Qwen3.5 CPU, and Supertonic FR. It reproduced “Dis-moi brièvement quelle est la capitale de la France.” exactly and displayed “La capitale de la France est Paris.” First audio began at 3,223.415 ms and playback completed at 6,022.732 ms, with one TTS chunk and zero underflows or drops. Evidence is committed at `5f4526d`, provenance SHA `7afac4f8…0f39`.

All five engineering turns share one cold boot, airplane mode, empty forward lists, and the same main/runtime PIDs. The final inventory receipt binds those turns to the accepted physical total below 2B. The separate final Russian film take is factually clean but has no reused engineering-latency claim. Lane F's earlier 7–11 s ES/FR projection is superseded by the measured 6.105 s Spanish and 6.023 s French playback-terminal results.

## The six audited stacks

| Stack | Measured known rows | Worst-case known rows | Lane A verdict |
|---|---:|---:|---|
| H | 1,396,245,218 | 1,396,245,218 | `COMPLIANT` |
| L | 1,676,733,894 | 1,720,220,241 | `UNVERIFIED` |
| D | 1,831,533,225 | 1,831,533,225 | `UNVERIFIED` |
| M | 1,674,645,423 | 1,880,539,567 | `UNVERIFIED` |
| Q | 1,964,482,923 | 1,964,482,923 | `UNVERIFIED` |
| F | 1,975,632,107 | 1,975,632,107 | `UNVERIFIED` |

The table proves that model size did not eliminate a candidate. It does not prove all six complete inventories, quality, rights clearance, or phone execution.

## Corrected record

Lane F's statement that no 0.4.14 APK exists is false in the current repository. `android-app/releases/lfm-audio-demo-0.4.14-expressive-default-guard.apk` is present at 4,816,382 bytes with SHA-256 `47bc3b3938c7f347d0a9cddedea777f18ba766c24da2a29b9f55140acf1bc939`, package `ai.liquid.lfmdemo`, version code 22, version name `0.4.14-expressive-default-guard`. It is a real historical debug APK, not the current two-package app-owned deployment and not evidence for either overnight demo.

## What not to say on camera

- Do not say “acoustic onset”; say “in-app first-audio start proxy.”
- Do not call the 2,221,983,906 host bound the installed total; the accepted physical total is 2,208,452,225 conservative elements.
- Do not call the older 1,834,513,983 host bound the installed Demo 1 total; the accepted physical conservative total is 1,820,982,302.
- Do not call the 5,828.660 ms ES or 5,526.684 ms FR diagnostic round trips app end-to-end latency; they are ASR-only checks through a temporary forward.
- Do not say the whole stack runs on the NPU or that NPU/CPU numerical parity passed. The narrow supported claim is SenseVoice QNN/HTP execution with a matched 2.40 artifact/runtime in the recorded phone run.
- Do not say “native quality,” “native-speaker approved,” or “best voice.” Quality evidence is a small-n host screen without native-speaker acceptance.
- Do not say anything about glasses. There is no glasses evidence of any kind.

## Evidence spine

Machine-readable results are in `demo_results.json`; claim-level provenance is in `evidence_index.json`; exact filming lines and acceptance slates are in `DEMO_SHOTLIST.md`; adjacent overclaims are paired in `CLAIMS_AND_LIMITS.md`. The detailed per-demo records are `DEMO1.md` and `DEMO2.md`.

## Evidence appendix

| Claim group / numbers used above | Evidence class | Producing task and source file |
|---|---|---|
| Demo 1 `<2B`, Demo 2 `<3B`, and Demo 2 `<3,000 ms` gate | `EXACT` requirements | Public demo specifications and cap ledger in this directory; the internal run brief is intentionally omitted from the publication snapshot |
| Six stacks H/L/D/M/Q/F measured and known-row worst-case totals | measured `EXACT`; worst-case `UB` | Lane A A7 — `workstreams/sub2b_sprint/sprint/out/a/ledger.json` |
| Demo 2 installed 1,974,795,009 learned / 2,208,452,225 conservative; 42 inventory rows; required exclusions absent | physical `PHONE-MEASURED`; conservative count `UB` | Final phone inventory — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo2/final_pruned/inventory_receipt.json` and `device_inventory.tsv` |
| Demo 2 host overcount bound 1,988,326,684 learned / 2,221,983,906 conservative; 2,672,875,114 remove-LFM-only bound | `EXACT` learned; conservative `UB` | Overnight cap ledger — `workstreams/sub2b_sprint/sprint/out/overnight/cap_ledger.json` |
| Final cold EN 2,452.154 ms and ZH 2,446.833 ms, indirectly receipt-bound at n=1 per language; summaries do not self-authenticate APK/reboot/airplane/forward state | `PHONE-MEASURED` plus separate receipt binding | Final cold traces — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo2/final_pruned/en_cold_final_712/summary.json`, `zh_cold_prefix/summary.json`, and the final inventory/protocol receipt |
| Warm EN n=21, 1,333.866/1,395.874/1,490.414 ms and zero playback failures | `PHONE-MEASURED` | Overnight warm EN — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo2/warm/explicit_en_21/provenance.json` |
| Warm ZH n=20, 1,031.277/1,221.034/1,603.032 ms and zero playback failures | `PHONE-MEASURED` | Overnight warm ZH — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo2/warm/explicit_zh_20/provenance.json` |
| Cold ZH 1,800.305 ms first text, 2,651.683 ms first audio, 258 ms TTS first chunk, 0.11456 RTF, UID/provider/boot state | `PHONE-MEASURED` | Overnight cold ZH — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo2/cold/reboot_zh_main_only/provenance.json`, `health_http.txt`, `qnn_runner.stdout.txt`, and `qnn_runner.stderr.txt` |
| Qwen2.5 EN 42.5 / ZH 54.875 and Qwen3.5 five-language screen, n=120/language | `HOST` | Lane B — `workstreams/sub2b_sprint/sprint/out/b/backbone_verdict.json` |
| SenseVoice 9.170% EN WER / 6.547% ZH CER; OmniASR 18.630% / 15.206%; Parakeet ES 3.551% / FR 5.345%, n=150/language | `HOST` | Lane C — `workstreams/sub2b_sprint/sprint/out/c/asr_table.json` |
| Matcha Latin/OOV omission; Kokoro “你好” 3/3 non-finite; Supertonic/Matcha route status | `HOST` | Lane D — `workstreams/sub2b_sprint/sprint/out/d/D1.json`, `D3.json`, and `tts_table.json` |
| Demo 1 installed 1,810,531,015 learned / 1,820,982,302 conservative; 61 inventory rows; required exclusions absent; five turns bound to one cold process | physical `PHONE-MEASURED`; conservative count `UB` | Final phone inventory — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/final_inventory/inventory_receipt.json`, `device_inventory.tsv`, and `device_state.txt` |
| Demo 1 overcounting host target 1,824,062,690 learned / 1,834,513,983 conservative; Qwen 507,154,688 bytes, 752,393,024 elements, zero MTP/vision tensors | ledger `EXACT`/`UB`; GGUF `EXACT` | Overnight cap/Qwen receipts — `workstreams/sub2b_sprint/sprint/out/overnight/cap_ledger.json` and `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/qwen35_08b_gguf_inventory.json` |
| Exact Demo 1 Qwen3.5 installed; Qwen2.5 removed; one GGUF; main/runtime APK identities | `PHONE-MEASURED` identity | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/qwen35_runtime_install.json` |
| Five-candidate TTS and 16-row CPU-only Parakeet provisioned; hashes verified; no partial/QNN payload | `PHONE-MEASURED` identity | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/provisioning_receipt.json` |
| ES/FR Parakeet CPU diagnostic: exact expected transcript parity under service UID 10514; 5,828.660/5,526.684 ms host round trips through a removed temporary forward | `PHONE-MEASURED` backend health, not app E2E | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/esfr_verifier/provenance.json` |
| Demo 1 cold EN: exact transcript and visible Paris response; Qwen3.5/SenseVoice QNN/Supertonic route; 2,339.209 ms first-audio proxy; 4,840.146 ms playback terminal; zero underflows/drops | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/five_language_cold/en_checkpoint/provenance.json` |
| Demo 1 cold RU coverage: exact Cyrillic transcript; Zipformer CPU/Qwen3.5/Supertonic route; 4,437.486 ms first-audio proxy; 15,605.082 ms playback terminal; zero underflows/drops; factual-quality warning retained | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/five_language_cold/ru_checkpoint/provenance.json` |
| Demo 1 normalized untrimmed film: 5,701.162 s; final visible RU answer “Москва” manually confirmed factually correct; earlier failed RU attempt retained | film identity/stream/decode `EXACT`; volume and permitted manual RU check `HOST` | `workstreams/sub2b_sprint/sprint/out/overnight/films/DEMO1_IT_FITS.mp4` plus `film_receipt.json` |
| Demo 2 normalized untrimmed EN→ZH film: 356.360 s; H.264 High 5.1 + AAC mono; full decode exit 0; -36.6/-0.7 dB mean/max volume | film identity/stream/decode `EXACT`; volume `HOST` | `workstreams/sub2b_sprint/sprint/out/overnight/films/DEMO2_ITS_GOOD.mp4` plus `film_receipt.json` |
| Filmed cold-run runner receipts: EN 1,988.922 ms and ZH 1,740.289 ms request-to-in-app-audio-start proxies; playback complete; automated assertions passed | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo2/film_capture/demo2_en_film_capture.json` and `demo2_zh_film_capture.json` |
| Post-capture EN/RU TTS repair: five-candidate metadata, zero model-weight transfer, automatic EN/RU/ZH routes, completed playback checks, and unplug-ready final state | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/tts_en_ru_repair/deployment_receipt.json`, `en_auto_end_to_end.json`, and `native_auto_ru_zh.json` |
| Final user-selected defaults: 26-candidate catalog; EN voice 4, RU Ruslan, ZH Matcha; 3/3 automatic playback; cold-restart persistence; offline unplug-ready state | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/tts_voice_defaults/deployment_receipt.json`, `automatic_routes.json`, and `supervisor_ready.png` |
| Post-film selected-route build: installed/build APK hash identity; warm n=1 EN/ZH/RU-long first-audio proxies 1,008.482/597.038/461.273 ms; 2.510 GiB resident PSS with 9.106 GiB available; 75-second host-debugger disconnect survival; two disclosed short-RU drain failures | `PHONE-MEASURED`, warm n=1 plus negative attempts | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/latency_preload/deployment_receipt.json`, `en_acceptance.json`, `zh_acceptance.json`, `ru_long_acceptance.json`, `ru_acceptance.json`, and `ru_acceptance_retry.json` |
| Final RU film fixture: “Какова столица России?”; no historical latency reused | fixture identity `EXACT` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/fixtures/question_ru_simple.wav`, SHA-256 `e8d931545ebddf124d63acd5d42bbde10fab7ba2f3f65998ce2ab7237693bc45` |
| Demo 1 cold ZH: exact Han transcript/response; SenseVoice QNN/Qwen3.5/Matcha route; 1,549.215 ms first-audio proxy; 11,291.198 ms terminal; zero underflows/drops; no Latin/digits in accepted response | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/five_language_cold/zh_checkpoint/provenance.json` |
| Demo 1 cold ES: exact Spanish transcript and Madrid answer; Parakeet CPU/Qwen3.5/Supertonic route; 3,343.357 ms first-audio proxy; 6,104.679 ms terminal; zero underflows/drops | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/five_language_cold/es_checkpoint/provenance.json` |
| Demo 1 cold FR: exact French transcript and Paris answer; Parakeet CPU/Qwen3.5/Supertonic route; 3,223.415 ms first-audio proxy; 6,022.732 ms terminal; zero underflows/drops | `PHONE-MEASURED` | `workstreams/sub2b_sprint/sprint/out/overnight/evidence/demo1/five_language_cold/fr_checkpoint/provenance.json` |
| Deployed VAD is `AdaptiveEnergyVad` Java code with zero learned parameters; accepted phone inventories contain no Silero weight in the covered roots | implementation `EXACT`; absence `PHONE-MEASURED` | App source — `android-app/app/src/main/java/ai/liquid/lfmdemo/AdaptiveEnergyVad.java`; Demo 1 and Demo 2 final inventory receipts |
| Lane F's prior ES/FR 7–11 s estimate, superseded by measured 6.105/6.023 s app playback terminals | estimate `PROJECTED`; results `PHONE-MEASURED` | Lane F F2 plus the overnight ES/FR checkpoint provenances |
| SM8850 account selection limitation and exact Parakeet QNN 2.40 prebuilt identity | selection `HOST`; artifact `EXACT` | Lane E — `workstreams/sub2b_sprint/sprint/out/e/E1.json` and `E3.json` |
| Historical ignored 0.4.14 APK: 4,816,382 bytes, SHA-256, package/version metadata | `EXACT` | Durable parse/hash receipt — `workstreams/sub2b_sprint/sprint/out/overnight/evidence/release_0_4_14_receipt.json` |

`evidence_index.json` supplies the SHA-256 for every indexed source and is the authoritative claim-to-file map.
