# Phone deployment experiment tracker

Legend: `[x]` done, `[~]` partial or in progress, `[ ]` open.
Phone results are not AR1 or glasses results.

Primary CPU evidence:
`reports/android_phone/experiment_batch_20260717/REPORT.md`.

## Current deployment checkpoint — 2026-07-29

- [x] Main `0.6.0-unplugged` APK owns LFM, UI/VAD, playback, and TTS.
- [x] Runtime `0.1.0-unplugged-runtime` APK owns ASR and Qwen2.5; no current
  inference process is launched through `adb shell`.
- [x] English, Mandarin, Japanese, and Korean SenseVoice produced exact fixture
  transcripts and language tags through QNN/HTP v81 under the runtime-companion
  UID.
- [x] Russian Zipformer produced the native-script fixture transcript under the
  runtime-companion UID on CPU.
- [x] App-owned Qwen2.5 completed the 28-case physical-phone screen: 24/28 after
  manual correction, 304 ms median first token, 672 ms median first speakable
  text.
- [x] The neural-TTS manifest authenticates every referenced asset; serialized
  queue/cancellation, language switching, RU/JA/KO generation configuration,
  and full-turn physical regressions pass.
- [~] The exact final pair survived a 70-second host-ADB shutdown with unchanged
  PIDs and zero forwards; literal physical USB removal and an on-phone turn
  remain open.
- [x] Cold reboot, first unlock, main-only launch, and EN/ZH/RU/JA/KO turns pass
  in airplane mode on the final APK pair.
- [~] Final-hash current RSS, thermal, freezer, and filtered error-log checks
  pass. Full PSS attribution and an unplugged battery/power run remain open.

Only SenseVoice EN/ZH/JA/KO is currently integrated on HTP. Russian ASR,
LFM, Qwen2.5, and all current TTS routes are CPU. AI Hub component profiles are
tracked separately and do not close an end-to-end phone NPU gate.

## Phase 0 - device record

- [x] **0.1 Device identity.** RAM, ABI, cores, fingerprint, and OS recorded.
- [x] **0.2 QNN library scan.** SM8850 HTP v81 positively identified.
- [x] **0.3 Runner load log.** Allocation and buffer records retained.

## Phase 1 - memory attribution

- [x] **1.1 Mapping attribution.** RSS split into anonymous, GGUF, libraries, and other mappings.
- [x] **1.2 Per-component residency.** Main and detokenizer mappings identified; encoder and vocoder remain jointly anonymous.
- [x] **1.3 Context sweep.** Context 512 through 8192 measured.
- [x] **1.4 Audio-length sweep.** 4.9, 18.4, and 29.4 second inputs measured.
- [x] **1.5 Idle residency.** Persistent-session idle cost measured.

## Phase 2 - memory reduction

- [x] **2.1 Minimum viable context.** Context 512 works for single-turn ASR/chat; multi-turn retention at context 512 remains open.
- [x] **2.2 KV quantization.** FP16, Q8_0, and Q4_0 screening completed.
- [x] **2.3 Mapping/repack controls.** Repack leg completed; mmap/mlock comparison remains open.
- [x] **2.4 Batch geometry.** Memory, latency, and quality tradeoff measured.
- [ ] **2.5 Staged component loading.** Load encoder and detokenizer only when needed.
- [ ] **2.6 Lower weight quantization.** Try Q3_K or IQ4_XS only if earlier work misses the gate.

Gate: model-process peak <= 1331 MiB (1.3 GiB); the stricter decimal reading
1.3 GB = 1240 MiB is noted for reference. WER increase must remain <= 0.3
percentage points on every diagnostic split.

## Phase 3 - NPU path

- [x] **3.1 Toolchain support.** HTP v81 gate passed; see `reports/android_phone/npu/qairt_v81_support.md`.
- [x] **3.2 AI Hub device discovery.** SM8850 / Snapdragon 8 Elite Gen 5 QRD is exposed.
- [~] **3.3 Quantized component export.** Depth W8A16 is blocked at quantization; the conv fallback ran full-NPU but failed tolerance.
- [ ] **3.4 Full compiled backbone.** Replace the representative layer probes.
- [ ] **3.5 Production cached decoder.** Add explicit position/mask inputs and bounded context.
- [ ] **3.6 Detokenizer decision.** Root-cause FP16, try INT8, or measure host FP32.
- [ ] **3.7 Full ASR → response model → TTS pipeline on the NPU.**
- [ ] **3.8 CPU versus NPU on identical silicon.**
- [x] **3.9 Zipformer exact-SM8850 release contexts.** Encoder, decoder, and joiner profiled on the Android-16 QRD with every reported operation on HTP-v81; see `reports/android_phone/streaming_response/zipformer_v81/RELEASE_CONTEXT_PROFILE.md`.
- [ ] **3.10 Zipformer source-golden equivalence.** Export the matching float source, compare deterministic outputs on the exact QRD, then run transcript-level equivalence before considering integration.
- [x] **3.11 SenseVoice app-UID HTP integration.** EN/ZH/JA/KO exact fixtures and language tags pass under the runtime-companion UID with execution-aware health; see `reports/progress_20260729/ASR_QNN_AUDIT.md`.

## Phase 4 - product evidence

- [ ] **4.1 VAD integration and speech-end responsiveness.**
- [ ] **4.2 Unplugged power run.**
- [ ] **4.3 Sustained thermal profile.**
- [x] **4.4 Post-reboot cold load.** Main-only launch restored all phone-resident services and all five full turns passed.
- [ ] **4.5 Multi-turn retention on device.**
- [ ] **4.6 Audio validity and retranscription checks.**
- [x] **4.7 On-device 18-utterance diagnostic.**
- [x] **4.8 App-owned runtime ownership.** LFM belongs to the main app; ASR and Qwen2.5 belong to the runtime companion rather than `shell`/`adbd`.
- [~] **4.9 Latest-pair full-turn acceptance.** Warm and cold-reboot airplane-mode EN/ZH/RU/JA/KO matrices plus the 70-second host-ADB shutdown pass on the exact final pair; literal physical USB removal remains open.
- [x] **4.10 Latest TTS concurrency acceptance.** Three queued chunks, cumulative speaker drain, cancellation/barge-in, Mandarin/Russian switching, and shared RU/JA/KO graph routing pass.

## Phase 5 - modular challenger

- [~] **5.1 Qwen3-0.6B CPU and NPU paths.** CPU control evidence exists; no accepted NPU integration.
- [ ] **5.2 Moonshine Base encoder and cached decoder.**
- [ ] **5.3 Pocket TTS host path.**
- [ ] **5.4 End-to-end modular pipeline versus LFM.**
- [x] **5.5 Compact text-model sweep.** Qwen2.5-1.5B Q4_K_M selected over the tested Qwen3/LFM compact candidates and physically swept under the app UID; broad multilingual robustness remains gated.
- [x] **5.6 Multilingual TTS device sweep.** Matcha Baker remains the Mandarin demo recommendation; Supertonic 3 five-step is selected for Russian and reused with per-request Japanese/Korean generation configuration. Preserve Matcha's corpus restriction.
- [x] **5.7 Locale-aware TTS integration.** Automatic `zh`/`ru`/`ja`/`ko`/fallback policy, serialized ownership, generation telemetry, and physical regression pass.

## Kill criteria

1. If minimum irreducible memory cannot meet the Phase 2 gate, record the finding and promote Phase 5.
2. If QAIRT cannot target SM8850 HTP, stop phone NPU work and retain proxy-only evidence. This criterion was not triggered.
3. If no viable detokenizer path exists, classify the NPU deployment as ASR-only.
4. If the sustained phone run throttles materially, revisit the integrated-model approach before glasses deployment.

## Reporting rules

- Phone, AR1, and glasses are separate evidence classes.
- At `n <= 10`, p95 is the sample maximum; identify it as such.
- RSS peaks are sampled; quote `VmHWM` alongside them.
- Do not report power while charging.
- Recheck transcripts, tokens, and task behavior after every precision change.
- Tensor distance alone is not an acceptance test.
- Keep server and tool logs, with device serials scrubbed.
- Each report states whether it was script-generated or hand-written and declares post-run edits.
