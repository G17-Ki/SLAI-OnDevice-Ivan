# Lane D — TTS summary

## Headline

**No single tested engine safely and commercially covers EN, ZH, ES, FR, and RU.** The best bounded-sprint routes are Kokoro or Supertonic for EN/ES/FR and Supertonic for RU. Lane D found **no general Mandarin ship route**: Kokoro v1.1 ZH is waveform-unsafe, while Matcha Baker + Vocos silently drops unsupported Latin/OOV spans and is commercially blocked by its corpus/checkpoint provenance.

Confidence is **high** for the reproduced waveform failures, parameter count, timing boundary, file inventory, and licence-text observations; **medium** for the intelligibility shortlist because D1 used `n=40`, one reference ASR per language, and unstable normalization controls outside English. Naturalness remains **user judgment**, not a lane verdict.

## D1 — Five-language backcheck

- `[HOST]` Twelve complete engine-language comparisons were scored at `n=40` using matched-human ΔWER/ΔCER. Every complete point estimate was at or below the +5 pp ceiling, but **none clears TH-13**, which requires `n>=60` and broader reference-ASR stability.
- Kokoro v1.1 ZH produced non-finite PCM on 7/40 items. Only 33 valid outputs existed, so that arm was not scored or imputed.
- Matcha ZH returned finite audio and ΔCER -0.351 pp, but a direct 14-case probe captured explicit `Ignore OOV` diagnostics for Latin/acronym spans. Its mixed-script CER is therefore a coverage blind spot.
- Only English normalizer controls were stable enough for ranking; ES/FR/RU/ZH rows are point estimates with no ranking weight.
- Source-provenance limit: the FLEURS fetch used mutable `refs/convert/parquet`, whose resolved revision and source-Parquet hashes were not retained. The evaluated local corpus manifest and each retained human WAV are SHA-256 pinned.

## D2 — Silero and Russian ladder

- Silero `v5_cis_base` has **20,926,952 learned parameters `[EXACT]`**. Substituting this count gives Stack L **1,611,051,613 parameters `[PROJECTED]`**, with **388,948,387 `[PROJECTED]`** headroom.
- The MIT artifact has no bundled accentor; a production Russian route needs an explicit stress frontend. Silero passed D3 but did not receive a D1 backcheck.
- Tested Piper voices are commercially blocked by Lessac-derived provenance. Dmitri also failed D3; Irina passed D3 but lacks D1/D5 evidence and has an additional unknown immediate-dataset licence.

## D3 — Short-input safety

- `[HOST]` **90/96** attempted outputs passed.
- Unsafe primary routes: Piper DaveFX ES (`A`, 0.081270 s), Piper Dmitri RU (`Да`, 0.116100 s), and Kokoro v1.1 ZH (four all-non-finite outputs).
- Supertonic passed 24/24 across EN/ES/FR/RU. Kokoro v1.0 passed 18/18 across EN/ES/FR. Matcha passed 9/9 Han-only runs; that does not repair its mixed-script omission problem.

## D4 — Matcha ZH timing and replacement

- `[HOST]` On the Apple M2 CPU route, the canonical short clause reached first PCM callback at **181.9 ms median / 296.4 ms p95** (`n=10`). Overall RTF was **0.101 median / 0.189 p95** (`n=70`).
- These software-callback timings are not comparable to the earlier 118 ms end-to-speaker phone result `[PRIOR]`; no delta was calculated.
- Matcha's Baker corpus is non-commercial and the exact checkpoint has no model-licence text. A clean replacement was estimated at **$3,075–15,250 `[PROJECTED]`** in known recording/GPU cash cost, excluding engineering, rights review, and evaluation.

## D5 — Personal listening bundle

- `[HOST]` **278/280** expected WAVs are present and validated, totaling **96,321,818 bytes**. Only `zh_kokoro_09.wav` and `zh_kokoro_14.wav` were withheld after non-finite generation.
- `LISTEN.md` presents each language as rows 01–20 with source text, exact side-by-side filenames, completion checkboxes, safety/licence labels, and explicit Matcha omission checks at ZH indices 02, 09, 11, 12, 13, and 14.
- Audio and model artifacts remain git-ignored under `artifacts/`; the listening bundle is present in this working copy and was not committed, as required.

## Closeout and limits

- Final route table: `tts_table.json`; personal review starts at `LISTEN.md`.
- No training, fine-tuning, local quantization, export, phone work, or listening panel was performed. No supplied cloud credentials were used or stored.
- Lane storage is approximately **2.515 GB**, below the **3 GB** cap.
- Outstanding product gates: `n>=60`, dual-reference ASR stability, listening acceptance, wrong-language/LID and speaker-consistency checks, exact-phone profiling, and commercially cleared Mandarin coverage.

<!-- LANE_G_ADDENDUM_START -->
## Lane G addendum — MOSS-TTS-Nano

**MOSS is not a shippable Mandarin route on current Lane G evidence.** Its ZH route disposition is `REJECT_UNSAFE_DIAGNOSTIC_ONLY`: the technical screen is `UNSAFE_PROBABLE_NONTERMINATION_OR_TRUNCATION_CAP_HIT`, personal naturalness/mixed-script listening is still pending, and redistribution remains blocked because the pinned weight/export snapshots do not contain the root licence file required by their own README.

- **G1 — count and package audit:** MOSS TTS is **117,311,232 learned parameters `[EXACT]`** and its audio tokenizer is **21,969,664 `[EXACT]`**, or **139,280,896 `[PROJECTED]`** together. This exactly matches Lane A's component counts. The official ONNX TTS export was confirmed to materialise tied audio-codebook tables redundantly, with three physical copies total.
- **G2 — Mandarin first:** the full 40-item ZH D1 synthesis arm completed and was scored: `PASS` on the point-estimate screen. Mixed-script coverage is explicitly `UNRESOLVED`; all 12 probe WAVs passed the waveform gate, but **4** hit the generation-frame cap and waveform differences were not promoted to spoken-span evidence.
- **G3 — short inputs:** `FAIL` across the frozen 1–3-character battery. Per-language verdicts: `{"en": "FAIL", "es": "FAIL", "fr": "FAIL", "ru": "UNRESOLVED", "zh": "FAIL"}`.
- **G4 — EN/ES/FR/RU:** reduced-n results remain **exploratory/incomplete**, so `single_engine_safe_all_five` stays false. Coverage verdicts: `{"en": "PASS", "es": "FAIL", "fr": "FAIL", "ru": "FAIL", "zh": "UNRESOLVED"}`.
- **G5 — personal listening:** **20/20** MOSS ZH clips are bound to the G5 synthesis receipt and **20/20** clear both the waveform and generation-cap gates. Retained clips appear beside Matcha in `LISTEN.md`; failed cases are explicit no-WAV cells. Six full/ablated mixed-script pairs follow in a separate yes/no/unclear checklist. Naturalness is `PENDING_USER_LISTENING`.
- **Licence:** HF metadata says Apache-2.0 `[VENDOR]`, but no root licence/notice file exists in any pinned TTS, tokenizer, or ONNX snapshot `[EXACT]`. The TTS README explicitly says to treat redistribution as unlicensed until that file appears; MOSS is therefore excluded from every commercially-unblocked candidate list.
- **Storage and boundaries:** no training, fine-tuning, quantization, export, phone work, or supplied cloud credentials were used. Final `out/d` footprint: **2,841,174,207 bytes**, below the 3,000,000,000-byte cap.
<!-- LANE_G_ADDENDUM_END -->
