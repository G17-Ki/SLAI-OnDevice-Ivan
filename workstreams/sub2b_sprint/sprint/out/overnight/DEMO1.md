# Demo 1 — “IT FITS”

## Status

**PASS AND FILMED: complete physical inventory strictly below 2B, cold EN/ZH/ES/FR/RU execution, and an untrimmed media-validated capture.**

The final 61-row phone inventory totals **1,810,531,015 learned parameters / 1,820,982,302 conservative resident elements**, leaving **179,017,698 conservative headroom** below the exclusive 2B cap. Exact Qwen3.5 is installed under the runtime UID; Qwen2.5 is removed; the five-candidate TTS tree verifies 22/22 referenced files plus 3 metadata files; and the CPU-only Parakeet payload is hash-verified with no QNN context/runtime payload. In one uninterrupted cold-boot, airplane-mode, no-forward engineering sequence, all five languages completed end to end under the same main/runtime processes. The separate final film closes the Russian factual-quality qualification with the visible answer **“Москва”**.

Film: `films/DEMO1_IT_FITS.mp4`, **5,701.162 s**, 46,810,178 bytes, SHA-256 `cf7ba53b9cf1a387b4c52501161253906e1002284b7cc48bd4907a0e80367958`. It is the normalized, untrimmed capture and passed automated decode, audio-stream, and hash/media checks. Detailed manual film-quality and semantic review was intentionally skipped; only the final Russian visible answer was manually checked. An earlier failed Russian attempt remains in the full film before the successful retry.

## Configuration and ledger

Exclusive cap: **2,000,000,000**.

| Role | Component | Languages | Learned unique | Conservative resident | Evidence |
|---|---|---|---:|---:|---|
| Backbone | Qwen3.5-0.8B Q4_0 text-only GGUF | EN/ZH/ES/FR/RU | 752,393,024 | 752,393,024 | `EXACT` |
| ASR | SenseVoiceSmall | EN/ZH | 233,999,167 | 233,999,167 | `EXACT` |
| ASR | Parakeet-TDT-0.6B-v3 INT8 CPU export | ES/FR | 627,057,286 | 637,225,053 | `EXACT` learned / `UB` export |
| ASR | RU Zipformer INT8 | RU | 65,610,041 | 65,610,041 | `EXACT` |
| TTS | Installed shared Supertonic 3 EN/ES/FR/RU + Matcha/Vocos ZH | all five | 131,471,497 | 131,755,017 | physical tree; `EXACT` learned / `UB` export |
| VAD | `AdaptiveEnergyVad` Java signal processing | all five | 0 | 0 | `EXACT` source identity |
| **Installed total** |  |  | **1,810,531,015** | **1,820,982,302** | accepted physical inventory |
| **Headroom** |  |  | **189,468,985** | **179,017,698** | strict `<2B` |

The app does not ship a Silero model. The deployed VAD is dependency-free `AdaptiveEnergyVad.java`, SHA-256 `f43f07819d1bf8983809421bc6e0b436046fd7e7a96d2271a9209caf3db4da00`, and contributes zero learned parameters. This corrects the earlier conservative draft that copied a Silero row from the lane registry despite no corresponding app artifact.

The earlier host target remains a safe but intentionally overcounting bound at 1,824,062,690 learned / 1,834,513,983 conservative because it retained an uninstalled Vocos 24 kHz export. Do not label those older numbers as the installed total.

## Exact backbone receipt

- Repository: `unsloth/Qwen3.5-0.8B-GGUF`
- Revision: `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`
- File: `Qwen3.5-0.8B-Q4_0.gguf`
- Bytes: 507,154,688
- SHA-256: `1f2002727f6f9c9ea39bdb0eefb58703d5d2652d55a185902aab5dcf1b4ad701`
- GGUF v3 header: 320 tensors; exactly 752,393,024 tensor elements.
- MTP/draft tensors: 0.
- Vision/visual/mmproj tensors: 0.

This receipt closes the prior MTP/vision sensitivity. Commit `044b7d8` additionally proves app-private installation identity: main APK `cab69190…a140d`, runtime APK `d43c0423…46ae1`, exact 507,154,688-byte GGUF, Qwen2.5 removed, and no runtime process left running after installation. The later five language checkpoints separately prove end-to-end execution.

## Language routes and filming prompts

| Language | ASR | TTS | Exact input line | Expected visible output cue |
|---|---|---|---|---|
| EN | SenseVoice QNN | Supertonic 3 | “What is the capital of France? Answer in one short sentence.” | Latin response including “Paris” |
| ZH | SenseVoice QNN | Matcha + Vocos | “请用中文简单介绍一下北京。” | Han response including “北京”; no Latin acronyms or digits |
| ES | Parakeet CPU | Supertonic 3 | “Dime brevemente cuál es la capital de España.” | Latin response including “Madrid” |
| FR | Parakeet CPU | Supertonic 3 | “Dis-moi brièvement quelle est la capitale de la France.” | Latin response including “Paris” |
| RU | RU Zipformer CPU | Supertonic 3 | “Какова столица России?” | Cyrillic response including “Москва”; no false superlative |

The final Russian film fixture is `evidence/demo1/fixtures/question_ru_simple.wav`, a 16 kHz mono PCM file with SHA-256 `e8d931545ebddf124d63acd5d42bbde10fab7ba2f3f65998ce2ab7237693bc45`. Its exact text is **“Какова столица России?”**. The older Russian fixture remains historical engineering evidence only.

## Phone execution checkpoints

| Language | Route | First audio proxy | Playback terminal | Disposition |
|---|---|---:|---:|---|
| EN | SenseVoice QNN → Qwen3.5 CPU → Supertonic EN | 2,339.209 ms | 4,840.146 ms | Coverage and visible-answer pass |
| ZH | SenseVoice QNN → Qwen3.5 CPU → Matcha ZH | 1,549.215 ms | 11,291.198 ms | Coverage and Han/no-Latin-or-digits pass |
| ES | Parakeet CPU → Qwen3.5 CPU → Supertonic ES | 3,343.357 ms | 6,104.679 ms | Exact transcript and correct Madrid answer pass |
| FR | Parakeet CPU → Qwen3.5 CPU → Supertonic FR | 3,223.415 ms | 6,022.732 ms | Exact transcript and correct Paris answer pass |
| RU, historical fixture | Zipformer CPU → Qwen3.5 CPU → Supertonic RU | 4,437.486 ms | 15,605.082 ms | Coverage pass only; not the final film fixture |

English reproduced the verbatim ASR transcript “What is the capital of France, answer in one short sentence?” and displayed “Paris is the capital of France.” Playback completed with zero underflows or dropped chunks. Qwen emitted one hidden reasoning-tag block spanning 19 filtered characters; the supported claim is **no user-visible reasoning leakage**, not that the model emitted no reasoning markers.

The historical Russian checkpoint reproduced “расскажи коротко о москве” and completed two Supertonic chunks with zero underflows or drops. Its visible answer is fully Cyrillic and mentions Moscow, but incorrectly calls Moscow the largest city in the world. It remains CPU route coverage evidence only. The final film instead uses `question_ru_simple.wav` and ends with the factually correct visible answer **“Москва”**. The historical 4,437.486/15,605.082 ms timings are not reused for the new fixture.

Mandarin reproduced “请用中文简单介绍一下北京。” and displayed a 40-character Han response containing “北京”, with no Latin letters or digits. Two Matcha chunks completed with zero underflows or drops. As with EN/RU, Qwen emitted one hidden 19-character reasoning block and the UI exposed none of it. Crop the stale `EN / 中文 offline demo` footer from a five-language film frame.

Spanish reproduced “Dime brevemente cuál es la capital de España.” exactly and displayed “La capital de España es Madrid.” One Supertonic chunk completed with zero underflows or drops. The route is explicitly CPU-only because the available Parakeet QNN context/runtime pairing is not programme-version matched; no NPU execution is claimed. Crop both stale English-only labels from the film frame.

French reproduced “Dis-moi brièvement quelle est la capitale de la France.” exactly and displayed “La capitale de la France est Paris.” One Supertonic chunk completed with zero underflows or drops. It shares the same explicit CPU-only boundary and stale UI-copy crop requirement as Spanish.

## Known quality and latency boundaries

- Qwen3.5-0.8B backbone screen, `HOST`, n=120/language: EN 43.3, ZH 47.5, ES 31.467, FR 34.167, RU 34.0. It is the size-driven Demo 1 choice, not the five-language quality winner.
- Parakeet clean ASR screen, `HOST`, n=150: ES 3.551% WER and FR 5.345% WER.
- A phone-side diagnostic executed Parakeet on CPU under runtime service UID 10514 and returned exact expected transcripts for both languages. ES took 5,828.660 ms and FR 5,526.684 ms as host round trips through a temporary ADB forward. Those numbers are ASR-only diagnostic timings, not no-forward app end-to-end latency; the forward was removed and the final list was empty.
- Supertonic passed bounded host short-input safety for EN/ES/FR/RU; no native-speaker acceptance was performed.
- Matcha produces finite Mandarin but silently drops unsupported Latin/OOV spans.
- Spanish and French app playback terminals are 6.105 s and 6.023 s `PHONE-MEASURED`. These supersede Lane F's 7–11 s projection. The slower CPU path is visible and must not be cut out.

Provisioning note: Lane F's installer was stopped during a read-only preflight after its stdin-to-device SHA path consumed the 652 MB encoder without returning EOF. It had made no phone writes. Recovery used the same pinned 16-row payload with host, staging, private `.incoming`, atomic rename, and final private hashes verified for every row. The final Parakeet tree has no partial files and no QNN/Qualcomm/context payload.

## Physical acceptance checklist

- [x] Install the exact Qwen3.5 file and verify its hash under the companion UID.
- [x] Remove Qwen2.5 rather than retaining both backbones.
- [x] Verify all four LFM artifacts, stale TTS trees, and extra GGUFs remain absent in the final inventory.
- [x] Install and verify all three Parakeet INT8 CPU ONNX files; make no Parakeet NPU claim.
- [x] Execute ES and FR Parakeet CPU diagnostics with exact parity and separately complete both full no-forward app turns.
- [x] Replace the complete TTS tree with the five-candidate pruned tree; an additive install is insufficient.
- [x] Verify SenseVoice, RU Zipformer, `AdaptiveEnergyVad` source/APK identity, token/front-end files, both APKs, and every retained model file.
- [x] Recompute the learned and conservative totals from the physical two-package/external-TTS inventory.
- [x] Cold reboot; enable airplane mode; verify no ADB forwards for the uninterrupted five-language sequence.
- [x] Complete one finite, non-silent, correct-language/script turn per language under the app/runtime UIDs.
- [x] Record first-audio proxies and full-turn latencies without hiding the ES/FR wait.
- [x] Capture and normalize the untrimmed film; pass automated decode/audio/hash checks.
- [x] Manually confirm only the final Russian answer: “Москва,” factually correct. Retain the earlier failed RU attempt in the full film.

Final receipt: `evidence/demo1/final_inventory/inventory_receipt.json`, SHA-256 `7da4c77a757c9f1dfab63316470b0986233866427312734a3b75fbf1ab546b4a`, committed at `95bd4fc`. It covers two APKs, 34 private rows, and 25 external TTS rows. The unprivileged listener probe found no port-8080 listener but could not expose owning PIDs; do not generalize this to “no network sockets.” Transient caches are outside shipped model inventory.

## What this demo does not prove

- The inventory receipt proves the captured shipped model roots are below 2B; it does not claim every unrelated external location or transient cache was archived.
- The host target ledger is an overcounting cross-check, not the installed total.
- It does not prove Parakeet NPU execution; the intended ES/FR route is CPU.
- It does not prove native quality, native-speaker acceptance, or commercial readiness.
- It does not make the 5.829/5.527 s diagnostic ASR round trips into app end-to-end timings; the separate no-forward app measurements are 6.105/6.023 s to playback terminal.
- It provides no glasses, thermal, battery, or long-session evidence.
