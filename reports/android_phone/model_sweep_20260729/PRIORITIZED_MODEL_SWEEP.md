# Prioritized offline multilingual model sweep

Date: 2026-07-29

Target: Nubia NX809J, SM8850, Android 16, arm64-v8a, HTP v81

Machine-readable plan: [`model_sweep_plan.json`](model_sweep_plan.json)

## Decision

Keep the accepted stack as the control and change one stage at a time. The
highest-value next sequence is:

1. finish real-audio Whisper Small W8A16 on the already proven phone ORT-QNN
   path;
2. stage Qualcomm's Qwen3-1.7B W4A16 NPU package in an isolated response
   runtime and test quality before routing any user turn to it;
3. move the shared Supertonic graphs to an app-owned QNN runtime for
   Japanese, Korean, Russian, and broad fallback;
4. move the already passing MeloTTS-ZH exact-target chain onto the phone as the
   licence-clean Mandarin candidate;
5. screen LFM2.5-1.2B on the phone CPU before spending Colab or Hub work on an
   NPU export.

This order closes the largest capability gaps with the most reusable work.
Whisper supplies a universal ASR fallback, Qwen3 establishes a real NPU
backbone path, and one Supertonic graph set covers several output languages.

## Evidence legend

- **PHONE-MEASURED**: executed on NX809J with real model input.
- **PHONE-COMPATIBILITY**: executed on NX809J, but only as an isolated or
  synthetic probe.
- **AH-QRD**: account-owned Qualcomm AI Hub result on the exact SM8850 QRD.
- **QCOM-QRD**: Qualcomm-published exact-target result, not this app or phone.
- **PROXY-v81**: another HTP-v81 retail target such as Galaxy S26.
- **HOST**: Apple M2 or Colab measurement.
- **PROJECTED**: planning inference only; not a measurement.

Remote neural-component times are never promoted to phone end-to-end times.

## Current accepted baseline

| Stage | Current route | Languages | Size / residency | Measured phone result | NPU state |
|---|---|---|---|---|---|
| ASR | SenseVoiceSmall INT8, fixed 10 s QNN context | EN/ZH/JA/KO/YUE | 162.0 MB archive; 254.2 MB context | EN/ZH direct QNN decode 41/36 ms; accepted EN/ZH/JA/KO full turns | **PHONE-MEASURED HTP** |
| ASR | Russian Zipformer INT8 | RU | 60.2 MB archive | Accepted Russian fixture; full-turn first text 2,118 ms including ASR + backbone | CPU |
| Backbone | Qwen2.5-1.5B-Instruct Q4_K_M | All target languages | 1,117,320,736-byte GGUF; 1,985,404 KiB process PSS after the multilingual run | 304 ms median / 475 ms P95 first token; 672 / 1,253 ms first speakable | CPU |
| English integrated backbone/voice | LFM2.5-Audio-1.5B Q4 | EN | 1.001 GiB bundle; 1,732,281 KiB process PSS | 982 ms first PCM; 1,195 ms output-head proxy; 0 underflows | CPU; only component shards have remote NPU evidence |
| TTS | Matcha Baker | ZH | 138.7 MiB including shared assets; 493 MiB standalone device RSS | 118 ms standalone end-to-speaker; 86/117 ms chunks in final full turns | CPU; non-commercial corpus |
| TTS | Supertonic 3 INT8, five steps, shared graphs | JA/KO/RU and potential fallback | 138.6 MiB shared model; 660 MiB standalone device RSS | Final full-turn chunks: JA 584 ms, KO 390 ms, RU 290 ms; zero drops | CPU; exact-QRD components exist but are not phone-integrated |

The accepted APK hashes remain:

- main: `a24e203a30dcaaf9215dc3f0d5b99fa4b893a9e0ceca71c99c13e49bdb1b020a`;
- runtime: `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`.

After a multilingual run the persistent stack measured 4,305,719 KiB PSS and
4,855,284 KiB RSS. Qwen and LFM account for most of it. Candidate tests must
swap a stage rather than add a second permanent backbone.

### Qwen2.5 score discrepancy

The machine-readable phone result says **23/28**. The runtime README says
**24/28 after manual correction**, apparently treating the formatted Mount
Fuji answer differently. Until a corrected machine artifact is regenerated,
23/28 is the acceptance floor. Its target-language machine cells are EN 3/4,
ZH 3/3, JA 2/3, KO 1/3, and RU 3/3. Korean backbone quality is therefore the
weakest measured core-language cell.

## ASR sweep

| Priority | Candidate | Measured evidence | Size / runtime | NPU feasibility | Missing work | Decision |
|---:|---|---|---|---|---|---|
| 1 | **Whisper Small W8A16 ORT-QNN** | **PHONE-COMPATIBILITY:** CPU fallback disabled; encoder + 4 decoder executions 295.1 ms. **AH-QRD:** 186.277 ms encoder + 3.932 ms/token, all HTP. **HOST:** 18.6% Latin WER, 29.7% CJK CER | 463.7 MiB host model; isolated APK including runtime 412.4 MB | **High** | Exact native QAI/HF frontend, production decode loop, real EN/ZH/JA/KO/RU transcripts, runtime integration | First universal NPU candidate |
| 2 | **Whisper Base QNN** | **HOST:** 36.1% Latin WER, 39.2% character error, 141 MiB model. **PROXY-v81:** 22.028 ms encoder + 2.657 ms/token | 141 MiB host model; published float components 90.7 + 187 MB | Medium-high | Pin exact-target context, run Hub/phone compatibility, reuse Small frontend | Memory fallback only; quality is materially worse |
| 3 | **X-ASR 480 ms streaming EN** | **HOST:** EN WER 15.8%, 288/374 ms final; 428 MiB peak RSS; first partial after 1.6-3.1 s. ZH CER 51.5% | 133.9 MB archive | Medium-low until export | Colab fixed-graph export, Hub compile/profile/inference, phone integration | English streaming experiment only |

The Whisper Small phone probe is no longer a speculative NPU path: sessions,
execution, HTP libraries, FastRPC FD, DMA buffers, and disabled CPU fallback
were all captured under the isolated app UID. It is still not semantic ASR.
The next gate is exact frontend parity followed by real five-language fixtures.

Whisper Tiny and the public EN/ZH Zipformer INT8 model stay out of the active
sweep. Tiny's retained quality is far below the target; public Zipformer scored
52.6% EN WER and 75.8% ZH CER with several empty outputs.

## Backbone sweep

| Priority | Candidate | Measured evidence | Size / runtime | NPU feasibility | Missing work | Decision |
|---:|---|---|---|---|---|---|
| 1 | **Qwen3-1.7B W4A16 Genie/QAIRT** | **QCOM-QRD:** 62.03 tok/s, 25.6-819.3 ms TTFT, 1,249.9 prefill tok/s at context 4096 on NPU. **HOST Q4:** 63/84 broad suite and 37/63 priority ZH/RU/context suite | Published Q4 size 1.06 GB; QAIRT minimum 2.45 | **High deployment feasibility, high quality risk** | Pin the W4A16 package/runtime, isolate QAIRT 2.45 from SenseVoice 2.40, rerun both quality suites on NPU | First NPU backbone experiment, not an automatic promotion |
| 2 | **LFM2.5-1.2B Q4** | **HOST:** 72/84, 111/215 ms first-speakable; KO 9/9. Separate priority suite only 31/63 | 695,751,488 bytes | Medium-low; no production exact-target package | Phone CPU swap-in first; only then Colab source export/quantization and Hub jobs | Best immediate Korean/footprint challenger |
| 3 | **Qwen3-0.6B Q4 Genie NPU** | **QCOM-QRD ctx512:** 61.62 tok/s, 41.5-166.1 ms TTFT, 3,085 prefill tok/s. **HOST:** 18/27 | 429 MB published package | **High** | Pin Genie package and run the current phone suite | Footprint/latency control; likely quality rejection |

Qwen3-1.7B is first because the W4A16 exact-target runtime already exists and
can prove the missing NPU stage. Its retained Q4 quality is worse than current
Qwen2.5, so performance-card numbers alone are insufficient. Promotion requires
at least the machine-readable 23/28 baseline, no core language below 2/3, no
hidden reasoning, and no wrong-script output.

LFM2.5-1.2B should be screened on the phone CPU before any conversion. It is
smaller, fast on the host, and perfect in the retained Korean cell, but its
Mandarin/Russian priority result is weak. Colab/Hub work is conditional on the
phone quality result.

Qwen3.5-0.8B, Qwen2.5-3B, Gemma-3-1B, and the older Qwen3 reasoning variants
are not active priorities. The first two fail memory/licence/quality tradeoffs;
Gemma and the reasoning variants did not beat current Qwen2.5.

## TTS sweep

| Priority | Candidate | Measured evidence | Size / runtime | NPU feasibility | Missing work | Decision |
|---:|---|---|---|---|---|---|
| 1 | **Supertonic 3 s5 QNN, shared graphs** | **PHONE CPU:** JA/KO/RU final chunks 584/390/290 ms, no drops. **AH-QRD:** duration 0.408 ms, encoder 1.046 ms, estimator 28.571 ms/step, vocoder 3.153 ms; five-step neural arithmetic 147.462 ms | 138.6 MiB for all language entries | Medium-high | App-owned QAIRT 2.45 package; accumulated JA/KO/RU audio; resolve estimator/vocoder numerical warnings; phone QNN execution | Best multiplier: one model covers three core languages and fallback |
| 2 | **MeloTTS-ZH QNN** | **PHONE CPU 8 threads:** 663 ms, RTF 0.440, 637 MiB RSS, 16% CER. **AH-QRD real-text chain:** 496 ms neural compute for 4.075 s audio, 2.94% NRMSE, 0.99958 cosine, 0% ASR CER | 182.2 MiB CPU bundle | High model feasibility, medium phone-runtime feasibility | Stage QAIRT 2.45 contexts; native Mandarin preprocessing/attention/stitching; phone QNN and listening | Licence-clean Mandarin ship path |
| 3 | **Piper RU Dmitri** | **PHONE CPU:** 185 ms end-to-speaker at 2 threads; 127 ms / RTF 0.085 at 8 threads; 5.6% CER | 77.4 MiB including shared data | CPU already sufficient; Russian NPU export absent | Native A/B vs Supertonic; GPL eSpeak-data decision; Colab/Hub export only if selected | Compact, CC0-corpus fallback |

Supertonic is the first TTS NPU target because JA, KO, and RU reuse the same
weights. The fixed graphs all place on HTP, but the vector estimator and
vocoder retain numerical warnings. A phone speedup is **projected**, not
measured, until accumulated language-specific audio passes and the app UID
executes the contexts.

MeloTTS-ZH has the strongest model-level NPU evidence of the TTS candidates:
real Mandarin text, repeated decoder calls, finite stitched audio, close
waveform agreement, and exact ASR recovery on the QRD. Its remaining blocker is
phone runtime integration, not model export. It is the only priority Mandarin
candidate that is MIT and does not need GPL eSpeak data.

Piper Dmitri should be enabled as the second Russian auto candidate before an
NPU export is attempted. CPU latency is already excellent. Spend Colab/Hub
effort only if native listening prefers it or co-residency/power requires NPU.

Kokoro INT8, ZipVoice, and eSpeak are excluded from the natural-voice path.
Kokoro INT8 has a confirmed non-finite short-input defect; ZipVoice measured
3.142 s end-to-speaker and 1.226 GiB RSS; eSpeak measured 55.2% Mandarin CER
and 61.7% Russian WER.

## Language routes

| Language | Accepted route | Next full-stack A/B | Main risk |
|---|---|---|---|
| EN | SenseVoice NPU -> Qwen2.5 CPU -> integrated LFM CPU | Whisper Small NPU -> Qwen3-1.7B NPU -> Supertonic/Piper modular TTS | Preserve integrated audio quality while reducing dual-backbone memory |
| ZH | SenseVoice NPU -> Qwen2.5 CPU -> Matcha CPU | Whisper Small NPU -> Qwen3-1.7B NPU -> MeloTTS-ZH NPU | Matcha is non-commercial; Melo phone runtime is absent |
| JA | SenseVoice NPU -> Qwen2.5 CPU -> Supertonic CPU | Whisper Small NPU -> backbone A/B -> Supertonic NPU | Native listening and backbone factual quality |
| KO | SenseVoice NPU -> Qwen2.5 CPU -> Supertonic CPU | Whisper Small NPU -> LFM2.5-1.2B or Qwen3-1.7B -> Supertonic NPU | Current Qwen2.5 scored only 1/3 in the phone screen |
| RU | Zipformer CPU -> Qwen2.5 CPU -> Supertonic CPU | Whisper Small NPU -> Qwen3-1.7B NPU -> Supertonic NPU; Piper Dmitri CPU fallback | Universal ASR NPU is not production-integrated |
| Other | No accepted universal NPU ASR; eSpeak availability control | Whisper Small fixed-locale NPU + one resident backbone + Supertonic where supported; Piper EN/DE/IT or Melo ES as language packages | Avoid presenting emergency formant TTS as natural speech |

Final post-reboot first-audio software proxies were EN 1,195 ms, ZH 1,727 ms,
JA 2,796 ms, KO 1,912 ms, and RU 2,582 ms. Japanese and Russian are closest to
the 3 s cap; NPU ASR/backbone/TTS work should be evaluated against those full
turns, not only component arithmetic.

## Offloaded work queue

| Order | Work | Venue | Mac load |
|---:|---|---|---|
| 1 | Whisper exact frontend and five-language real transcription | Lightweight local tests, then candidate phone runtime | Light; no conversion |
| 2 | Qwen3-1.7B W4A16 package and isolated NPU quality run | Qualcomm package/Hub + phone | Staging only |
| 3 | Supertonic shared QNN runtime and accumulated JA/KO/RU audio | Existing Hub artifacts + phone | Staging/orchestration only |
| 4 | MeloTTS-ZH QNN phone runtime | Existing Hub artifacts + phone | Staging/orchestration only |
| 5 | LFM2.5-1.2B CPU phone screen | Phone | No heavy Mac work |
| 6 | LFM2.5 export only if step 5 passes | Colab source export/quantization + Qualcomm Hub | No heavy Mac conversion |
| 7 | X-ASR streaming export, conditional | Colab + Qualcomm Hub | No heavy Mac conversion |

## Promotion gates

Every candidate must pass all relevant gates before replacing the accepted
route:

1. **Identity and isolation:** checksum-pinned assets, separate candidate
   package/runtime, accepted APK hashes unchanged.
2. **NPU proof:** provider selected, CPU fallback disabled where supported,
   app/process UID, QNN/HTP maps, FastRPC/DMA evidence, and stage timing.
3. **Quality:** fixed EN/ZH/JA/KO/RU set first; then at least 100 representative
   utterances per language before a robustness claim.
4. **Latency:** ASR P95 <=700 ms, first-speakable P95 <=1.25 s, TTS first chunk
   <=700 ms where feasible, and warm end-of-VAD to first-audio <=2.0 s median /
   <=3.0 s P95.
5. **Memory:** measure the selected ASR, exactly one response backbone, and one
   TTS engine together. Reject accidental dual-backbone residency.
6. **Audio safety:** finite, non-silent, unclipped, correct-language PCM; no
   drops or queue starvation; native-speaker listening for ZH/JA/KO/RU.
7. **Offline durability:** airplane mode, force-stop, reboot, host-ADB shutdown,
   literal unplug, microphone turns, and a battery/power soak.

## Primary evidence

- [`final_apk_a24e_33bf_cold_reboot_provenance_20260729.md`](../full_turns/final_apk_a24e_33bf_cold_reboot_provenance_20260729.md)
- [`qwen2_5_1_5b_q4_k_m_app_owned_phone.json`](../streaming_response/qwen2_5_1_5b_q4_k_m_app_owned_phone.json)
- [`ORT_QNN_PROBE_DEVICE_EVIDENCE.md`](../streaming_response/whisper_v81/ORT_QNN_PROBE_DEVICE_EVIDENCE.md)
- [`qai_hub_models_0.58.0_voice_candidates.json`](../streaming_response/qai_hub_models_0.58.0_voice_candidates.json)
- [`FINAL_DECISION.md`](../streaming_response/model_candidate_followup/FINAL_DECISION.md)
- [`DEVICE_RESULTS.md`](../streaming_response/tts_research_20260726/DEVICE_RESULTS.md)
- [`NPU_STATUS.md`](../streaming_response/tts_research_20260726/NPU_STATUS.md)
- [`multilingual_asr_mac_m2/README.md`](../streaming_response/multilingual_asr_mac_m2/README.md)
- [`sensevoice_asr_mac_m2/REPORT.md`](../streaming_response/sensevoice_asr_mac_m2/REPORT.md)

The JSON plan contains the full per-candidate asset gaps, job venues, evidence
paths, and executable acceptance requirements.

## Runtime-compatibility addendum — 2026-07-29

This audit **supersedes the post-Whisper ordering at the top of this report**.
After the real-audio Whisper gate, the single best NPU experiment is now:

> **Supertonic 3 INT8 shared-graph TTS in an isolated ORT-QNN / QAIRT 2.42
> phone probe, starting with the fixed 32-text / 32-latent graph contract.**

Qwen3-1.7B remains the first *backbone* NPU experiment, but it is no longer the
first experiment overall. Its published W4A16 package requires QAIRT 2.45, is
not present locally, and adds about 1.06 GB before runtime overhead. The
published MeloTTS-ZH contexts are also QAIRT 2.45, are not present locally, and
their eight pinned artifacts total 363.5 MiB before runtime libraries. The
audited local cache contains 2.40 and 2.42 Android QNN runtimes, but no complete
2.45 Android runtime bundle.

Supertonic is the safer first probe because all four source graphs are present
locally, one 138.6 MiB weight set covers 31 languages including EN/JA/KO/RU,
and exact-SM8850 Hub runs already place every reported operation on HTP. The
remaining unknown is specifically whether the same graphs compile and execute
under the phone-proven 2.42 runtime.

### Exact inputs

- Model directory:
  `/private/tmp/lmf-tts-bundle-20260729-1145/supertonic3-int8/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`
- Source archive identity: `sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2`,
  SHA-256 `82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427`.
- Exact model/support members:

  | File | Bytes | SHA-256 |
  |---|---:|---|
  | `duration_predictor.int8.onnx` | 3,700,147 | `c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db` |
  | `text_encoder.int8.onnx` | 36,416,150 | `c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff` |
  | `vector_estimator.int8.onnx` | 78,400,833 | `20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd` |
  | `vocoder.int8.onnx` | 25,991,073 | `e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152` |
  | `voice.bin` | 517,168 | `67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8` |
  | `unicode_indexer.bin` | 262,144 | `8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30` |
  | `tts.json` | 8,253 | `42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09` |

  The complete model/support payload is 145,295,768 bytes.
- ORT-QNN AAR:
  `<local-cache>/lmf-demo/whisper-small-v0.58.0/onnxruntime-android-qnn-1.25.0.aar`,
  7,720,525 bytes, SHA-256
  `397277c3e50326f5972409ca42ab5b4e69172b218557923630d68e55b11d6198`.
- Exact QAIRT archive:
  `<local-cache>/lmf-demo/qairt-v2.42.0.251225.zip`,
  1,543,955,191 bytes, SHA-256
  `1a744ed72a813fc9dd4a29e31b1685f784786102fd45414d74915bb3351ae321`.
  It contains the Linux ONNX converter, model-library generator, context-binary
  generator, HTP v81 libraries, and the Android runtime already proven by the
  Whisper probe. Runtime file hashes are pinned in the locally retained
  `nubia_whisper_ort_qnn_runtime_inventory.json`, which is included in the
  non-public raw-evidence archive.

### Execution recipe

1. In the existing AI Hub environment, build the deterministic fixed-shape
   inputs with `scripts/prepare_supertonic_aihub.py` at text length 32 and latent
   length 32.
2. Submit **only the 3.7 MB duration graph first** with
   `scripts/run_aihub_supertonic_v81.py --component duration_predictor
   --qairt-version 2.42`. This is a cheap account-availability gate. The only
   locally documented selectable Hub version is 2.45; an earlier 2.40 request
   was rejected before job creation, so 2.42 must not be assumed.
3. If Hub accepts 2.42, compile/profile/infer all four graphs at 2.42, require
   zero CPU/unknown placement, download and hash the target artifacts, and use
   those exact contexts on the phone.
4. If Hub rejects 2.42, **do not load a 2.45 context with the 2.42 runtime**.
   Run the raw fixed-shape ONNX graphs through ORT 1.25 QNN on the phone with
   `session.disable_cpu_ep_fallback=1`, or generate exact 2.42 contexts in
   Google Colab from the pinned QAIRT archive using
   `qnn-onnx-converter`, `qnn-model-lib-generator`, and
   `qnn-context-binary-generator` for `soc_model=87` / `dsp_arch=v81`.
   No conversion belongs on this MacBook.
5. Package the result in a new candidate package/process. Do not modify the
   accepted main/runtime APKs or the accepted Whisper probe. Reuse the proven
   2.42 Android libraries and require app-UID QNN maps, FastRPC, DMA buffers,
   disabled CPU fallback, and successful execution of all four stages.

The uncompressed raw-ONNX phone payload floor is 254.4 MiB: 138.6 MiB model and
support files plus 115.8 MiB of QAIRT 2.42 and ORT arm64 libraries. Context
cache and APK duplication are not yet measured; reserve a 650 MiB installed
storage cap for the isolated probe and reject rather than silently exceed it.
Use a 700 MiB candidate-process RSS cap, matching the existing CPU-engine
envelope, and measure with exactly one backbone resident.

Promotion requires finite and non-silent audio, correct engine/language
identity, EN and JA fixed-fixture backchecks first, then JA/KO/RU phrase and
native-listening gates, first chunk <=700 ms, RTF <=0.8, no dropped PCM, and
force-stop/relaunch plus reboot/unplugged operation. Mandarin remains on the
existing route until a separate QAIRT-compatible MeloTTS experiment passes.

### QAIRT 2.42 Hub canary result — 2026-07-30

The smallest-graph account gate was executed with a pinned
`qai-hub==0.52.0` client. It reused the successful duration-predictor source
model `mq92z1y2m`, prior compile `jp2v287x5`, and the exact fixed input shapes;
no model upload or local inference was needed.

AI Hub rejected `--qairt_version 2.42` before job creation with
`QAIRT SDK version 2.42 is not valid.` The account's machine-read framework
inventory contains only API versions 2.45, 2.47, and 2.48. Therefore the
remaining three graphs were not submitted and there was no automatic 2.45
retry. Machine evidence is retained locally as
`supertonic_qairt242_compile_state.json` in the non-public raw-evidence archive,
with the trust root in
[`supertonic_qairt242_compile_spec.json`](supertonic_qairt242_compile_spec.json).

The next safe implementation branch is raw fixed-shape Supertonic ONNX through
ORT-QNN 1.25 on the Nubia, initially with only the duration graph and CPU EP
fallback disabled. Exact 2.42 offline context generation in Colab remains a
parallel fallback. Loading an available 2.45 context in the 2.42 runtime stays
explicitly forbidden.

### Raw ORT-QNN duration result — 2026-07-30

That safe branch has now passed for the first component. The exact unchanged
3,700,147-byte duration ONNX executed in isolated package
`ai.liquid.lfmdemo.supertonicprobe` under ORT 1.25 / QAIRT 2.42 with CPU EP
fallback disabled. Output was 2.12890625 versus 2.127588987, session load was
245.144 ms, and execution was 1.320 ms. App-UID HTP/v81 mappings, vendor
FastRPC, 16 QCOM DMA buffers, 236.934 MiB RSS, 85.517 MiB PSS, exact installed
APK/model hashes, and unchanged protected package hashes were preserved.

This promotes the raw 2.42 path from a compatibility hypothesis to a measured
duration-component NPU result. It does not yet promote Supertonic TTS: the text
encoder, vector estimator, vocoder, accumulated chain, finite PCM, and language
quality gates remain open. Full evidence is in
[`SUPERTONIC_DURATION_RAW_ONNX_QNN_RESULT.md`](SUPERTONIC_DURATION_RAW_ONNX_QNN_RESULT.md).
