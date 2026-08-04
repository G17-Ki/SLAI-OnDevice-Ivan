# Five-language offline speech stack — evidence report

This report assembles the landed evidence for a complete offline English, Mandarin Chinese, Spanish, French, and Russian speech system under a strict 2,000,000,000-parameter cap, with the Nubia NX809J / Snapdragon SM8850 as the immediate device target. It is a decision record and demo guide, not a claim that the complete product already runs on the phone. `[A:A7; D:TTS; E:E6]`

## Source key and evidence rule

Every factual claim below ends with a stable source label. Evidence classes retain the producing lane's meaning: `EXACT`, `UB`, `VENDOR`, `BENCHMARK`, `HOST`, `AIHUB-PROFILED`, `PRIOR`, `PROJECTED`, or `UNKNOWN`; this report does not promote one class into another. `[A:A7; D:TTS; E:E5]`

| Label | Producing task and source file |
|---|---|
| `A:A6` | Lane A / A6 — `workstreams/sub2b_sprint/sprint/out/a/A6.json` |
| `A:A7` | Lane A / A7 — `workstreams/sub2b_sprint/sprint/out/a/ledger.json` |
| `A:SUMMARY` | Lane A — `workstreams/sub2b_sprint/sprint/out/a/LANE_SUMMARY.md` |
| `D:D1` | Lane D / D1 — `workstreams/sub2b_sprint/sprint/out/d/D1.json` |
| `D:D3` | Lane D / D3 — `workstreams/sub2b_sprint/sprint/out/d/D3.json` |
| `D:D4` | Lane D / D4 — `workstreams/sub2b_sprint/sprint/out/d/D4.json` |
| `D:D5` | Lane D / D5 — `workstreams/sub2b_sprint/sprint/out/d/D5.json` |
| `D:TTS` | Lane D final route table — `workstreams/sub2b_sprint/sprint/out/d/tts_table.json` |
| `D:SUMMARY` | Lane D — `workstreams/sub2b_sprint/sprint/out/d/LANE_SUMMARY.md` |
| `E:E1` | Lane E / E1 — `workstreams/sub2b_sprint/sprint/out/e/E1.json` |
| `E:E3` | Lane E / E3 — `workstreams/sub2b_sprint/sprint/out/e/E3.json` |
| `E:E5` | Lane E / E5 — `workstreams/sub2b_sprint/sprint/out/e/E5.json` |
| `E:E6` | Lane E / E6 — `workstreams/sub2b_sprint/sprint/out/e/E6.json` |
| `E:REGISTRY` | Lane E artifact record — `workstreams/sub2b_sprint/sprint/out/e/artifact_registry.json` |
| `H:DEMO` | Lane H / H3-H5 media availability — `workstreams/sub2b_sprint/sprint/out/h/DEMO_SHOTLIST.md` |
| `PENDING-B` | Expected Lane B result — `workstreams/sub2b_sprint/sprint/out/b/backbone_verdict.json` |
| `PENDING-C` | Expected Lane C result — `workstreams/sub2b_sprint/sprint/out/c/asr_table.json` |
| `PENDING-F` | Expected Lane F results — `workstreams/sub2b_sprint/sprint/out/f/` |

## 1. What was asked and what was delivered

The decision problem is not “find a small model.” It is to identify a complete five-language inventory—ASR, language backbone, TTS, vocoder or codec, VAD, and every shipped frontend or fallback—that stays strictly below two billion unique learned parameters and has a credible path to an offline SM8850 product. `[A:A7; D:TTS; E:E6]`

Lane A delivered a conservative six-stack parameter ledger and a licence traffic-light. Lane D recorded 278 validated host-generated WAVs out of 280 expected and delivered the five-language route table and listening guide; the git-ignored WAV bundle is not present in this checkout and requires Lane D's original artifact workspace for playback. Lane E delivered a 36-row artifact registry, a `PROJECTED` exact-version QAIRT/QNN policy, a hash-pinned staging bundle, and a fail-closed phone runbook; it did not deliver a runnable phone stack. `[A:A7; D:D5; D:TTS; E:E5; E:E6; E:REGISTRY; H:DEMO]`

The landed work therefore supports a defensible shortlist, committed host-side failure evidence, conditional playback from Lane D's original artifact workspace, and a version-safe device activation plan. It does not yet support an integrated five-language product demonstration or a final stack selection. `[A:A7; D:D5; D:TTS; E:E6; H:DEMO]`

### Pending integration slots

<!-- H5_SLOT_LANE_B_START -->
**PENDING — Lane B.** The H5 re-scan found no final `backbone_verdict.json`; no backbone verdict is asserted here. Fill this slot only from `workstreams/sub2b_sprint/sprint/out/b/backbone_verdict.json`, preserving its evidence classes and limits. `[PENDING-B]`
<!-- H5_SLOT_LANE_B_END -->

<!-- H5_SLOT_LANE_C_START -->
**PENDING — Lane C.** The H5 re-scan found no final `asr_table.json`; no ASR quality, punctuation, latency, or route verdict is asserted here. Fill this slot only from `workstreams/sub2b_sprint/sprint/out/c/asr_table.json`, preserving its evidence classes and limits. `[PENDING-C]`
<!-- H5_SLOT_LANE_C_END -->

<!-- H5_SLOT_LANE_F_START -->
**PENDING — Lane F.** The H5 re-scan found no Lane F result file; no Spanish/French integrated result is asserted here. Fill this slot only from landed files under `workstreams/sub2b_sprint/sprint/out/f/`, preserving their evidence classes and limits. `[PENDING-F]`
<!-- H5_SLOT_LANE_F_END -->

## 2. The three headline findings

### 2.1 The cap did not eliminate a measured candidate; licensing is the cross-stack discriminator

**Evidence: `EXACT` measured component counts, recorded `UB` fields, and `UNKNOWN` unresolved rows.** All six measured stack subtotals and all six recorded known-row UB subtotals are below the two-billion cap, and no stack is marked `NONCOMPLIANT`. Those UB fields do not resolve the six `UNKNOWN` frontend rows, so only H has a closed complete-stack result and is arithmetically `COMPLIANT`; L, D, M, Q, and F remain `UNVERIFIED`. The commercial separation is sharper: M is the only green licence sheet, L is amber, and H, D, Q, and F are red. The licence colours report documented facts and review status, not legal advice. `[A:A6; A:A7; A:SUMMARY]`

The practical implication is that adding headroom alone does not produce a sellable stack. H has the largest measured headroom but a red licence sheet, while M combines the only green sheet with substantial measured headroom and one unresolved punctuation row. Q and F are especially constrained: Lane A found that the cheapest measured neural punctuation restorer would exceed their remaining headroom, so they remain viable only if punctuation and ITN are delegated to the existing backbone with zero added parameters and recorded as a binding architecture decision. `[A:A7; A:SUMMARY]`

### 2.2 Mandarin has no shippable voice in the tested set

**Evidence: `HOST` technical probes plus `EXACT` pinned source/licence observations; naturalness remains `UNKNOWN`.** Kokoro v1.1 ZH reproduced non-finite PCM and is technically unsafe. Matcha Baker + Vocos produced finite Han speech but silently omitted unsupported Latin/OOV spans, while the Baker corpus is non-commercial and the exact checkpoint has no model-licence text. Lane D therefore found no general Mandarin ship route among the tested engines and no single tested engine that safely and commercially covers all five languages. `[D:D1; D:D3; D:TTS; D:SUMMARY]`

This is a coverage and rights finding, not a claim that every Mandarin voice technology fails. A shippable route can still be created by replacing or clearing the checkpoint and then passing the omitted technical, listening, and device gates. `[D:TTS; D:SUMMARY]`

### 2.3 SM8850 is not compile-selectable for this account, and the verified prebuilt is version-locked

**Evidence: `HOST` service observations plus `EXACT` context-header inspection; the programme pin and verdict are `PROJECTED` policy decisions over those facts.** Qualcomm AI Hub authenticated and exposed SM8850 devices, but three compile-selector forms were refused for this account. The one downloaded SM8850 Parakeet prebuilt verified by size and SHA-256, and each of its encoder, decoder, and joiner headers embeds exact QNN `2.40.0.251030114326_189385`. The programme pin is exact QAIRT/QNN `2.45.0.260326154327`; under that policy, those contexts are blocked from the programme's NPU load and the programme-level Parakeet path is `EXPORT`. At artifact level, `REPACKAGE` is allowed only in an isolated process carrying the exact 2.40 runtime, never against the 2.45 programme runtime. `[E:E1; E:E3; E:E5]`

Lane E staged the exact 2.45 runtime supply and safety controls, but zero staged model contexts match that pin and no app-UID runner exists. The correct current product claim is “hash-verified, fail-closed device staging,” not “working on-device NPU inference.” `[E:E5; E:E6; E:REGISTRY]`

## 3. The six stacks

The table uses Lane A's conservative `params` basis: all tensor elements in the shipped artifact, with known rows summed and unresolved rows left explicitly `UNKNOWN`. The measured columns are `EXACT` arithmetic over counted rows; the recorded bound columns remain `UB`. Both headroom columns are against the exclusive 2,000,000,000 cap before resolving the listed open rows, so neither is a closed complete-stack guarantee for L, D, M, Q, or F. `[A:A7]`

| Stack | Measured subtotal | Cap used | Measured headroom | Recorded known-row UB subtotal | Recorded UB headroom | Arithmetic verdict | Licence | Unresolved row(s) | Source |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| H | 1,396,245,218 | 69.81% | 603,754,782 | 1,396,245,218 | 603,754,782 | **COMPLIANT** | **RED** | none | `[A:A7]` |
| L | 1,676,733,894 | 83.84% | 323,266,106 | 1,720,220,241 | 279,779,759 | `UNVERIFIED` | AMBER | L9 Russian stress; L10 Russian punctuation/ITN | `[A:A7]` |
| D | 1,831,533,225 | 91.58% | 168,466,775 | 1,831,533,225 | 168,466,775 | `UNVERIFIED` | **RED** | D11b Parakeet punctuation for ES/FR/RU | `[A:A7]` |
| M | 1,674,645,423 | 83.73% | 325,354,577 | 1,880,539,567 | 119,460,433 | `UNVERIFIED` | **GREEN** | M7 Qwen3-ASR punctuation | `[A:A7]` |
| Q | 1,964,482,923 | 98.22% | 35,517,077 | 1,964,482,923 | 35,517,077 | `UNVERIFIED` | **RED** | Q9 punctuation/ITN; zero-parameter backbone delegation required to fit | `[A:A7]` |
| F | 1,975,632,107 | 98.78% | 24,367,893 | 1,975,632,107 | 24,367,893 | `UNVERIFIED` | **RED** | F9 punctuation/ITN; zero-parameter backbone delegation required to fit | `[A:A7]` |

The table is not a final ranking. H is the only closed arithmetic row but is licence-red; M is the only licence-green row but remains arithmetically unverified; and the TTS evidence currently supplies no shippable Mandarin voice for any complete five-language product. `[A:A7; D:TTS]`

The eventual selection must also incorporate the still-pending backbone, ASR, and Spanish/French integration evidence. Until those files land, this report makes no comparative quality or end-to-end recommendation among the six stacks. `[PENDING-B; PENDING-C; PENDING-F]`

## 4. What the demo shows and what it does not

### What it can show now

- **A visible, auditable trade space.** The six-stack ledger demonstrates where measured parameter headroom exists, which rows remain unknown, and why a small total does not erase a red or amber licence status. `[A:A6; A:A7]`

- **A conditional host-side five-language voice comparison.** Lane D validated 278 of 280 expected WAVs and supplied a listening guide with source text, paired filenames, safety/licence labels, and explicit omission checks. The WAVs are git-ignored and absent from this checkout, so playback requires Lane D's original artifact workspace; without it, the committed evidence cards remain demonstrable but audio stays `PENDING — LANE D MEDIA`. This supports an informed listening demonstration, not a native-speaker acceptance claim. `[D:D5; D:SUMMARY; H:DEMO]`

- **Failure detection as an evaluation capability.** The demo can present the committed Kokoro ZH non-finite-output evidence and Matcha mixed-script omission evidence. Playback of Lane D audio remains conditional on the original artifact workspace. These cases show that the evaluation rejects unsafe or incomplete routes instead of hiding them behind a clean sample. `[D:D1; D:D3; D:D5; D:TTS; H:DEMO]`

- **A version-safe device handoff.** Lane E's scripts and runbook verify identities and hashes, reject mismatched QNN contexts, and stop before NPU execution while the app runner and exact-pin contexts are missing. The packaging policy forbids multiple QNN runtime providers per APK or process; that provider-count policy was not a separate mocked gate. Seven mocked-adb safety cases passed; real phone/adb calls were zero. `[E:E5; E:E6]`

- **A reproducible ASR reference, not a device result.** The official FP32 Parakeet ONNX equivalent produced host CPU transcripts, tokens, and timestamps for the three bundled test WAVs, giving a reference for a later exact-version phone parity check. `[E:E3]`

### What it cannot show yet

- **No integrated five-language product.** Lane E records no complete CPU stack, no app runner, and zero exact-2.45 staged QNN contexts; B, C, and F results are also still pending in this report. `[E:E5; E:E6; PENDING-B; PENDING-C; PENDING-F]`

- **No on-device NPU parity or performance.** The phone was absent from Lanes D and E, the staged 2.40 Parakeet contexts were never loaded with the 2.45 runtime, and no phone latency, thermals, memory, placement, or end-to-end success was measured. `[D:SUMMARY; E:E3; E:E6]`

- **No native-speaker or naturalness acceptance.** Lane D ran no listening panel; its complete D1 arms use host-side point estimates at `n=40`, below the stated `n>=60` gate, with unstable normalization controls outside English. Naturalness remains user judgment. `[D:D1; D:TTS; D:SUMMARY]`

- **No commercially shippable Mandarin route.** Kokoro ZH is unsafe on the reproduced evidence, and Matcha's tested route is technically limited and commercially blocked. `[D:TTS; D:SUMMARY]`

- **No glasses evidence.** The landed evidence covers host evaluation and a staged Nubia phone path; it contains no glasses execution, fit, latency, power, thermal, microphone, or user-study result. `[A:SUMMARY; D:SUMMARY; E:E6]`

## 5. Open questions and what would close each

| Open question | Current state and class | Closure condition | Source |
|---|---|---|---|
| Which backbone is preferred? | **PENDING — Lane B**; no result asserted | Land `backbone_verdict.json`, index its claims in H1, and copy its verdict without changing evidence classes | `[PENDING-B]` |
| Which ASR route is preferred, and which routes emit punctuation? | **PENDING — Lane C**; no result asserted | Land `asr_table.json`, index its claims in H1, and use its raw-output evidence to close only the rows it explicitly addresses | `[PENDING-C; A:A7]` |
| Does the ES/FR integrated path work? | **PENDING — Lane F**; no result asserted | Land the Lane F integration files, index their claims in H1, and report only tested language/path combinations | `[PENDING-F]` |
| Can L close its two unknown rows? | L9/L10 are `UNKNOWN` | Pin and header-parse a Russian accentor or commit to a zero-parameter rule path; enumerate the Russian punctuation route | `[A:A7]` |
| Can D close its remaining ledger row? | D11b is `UNKNOWN` | Inspect the pinned Parakeet tokenizer for ES/FR/RU punctuation support | `[A:A7]` |
| Can M close its remaining ledger row? | M7 is `UNKNOWN` | Run punctuated fixtures in all five languages and inspect raw Qwen3-ASR output | `[A:A7]` |
| Can Q or F remain under cap with usable punctuation? | Q9/F9 are `UNKNOWN`; measured neural restoration does not fit | Make zero-parameter backbone delegation for punctuation/ITN a binding, testable build constraint; otherwise ledger the added model and accept a cap breach | `[A:A7; A:SUMMARY]` |
| What closes the Mandarin ship gap? | No general route in the tested set; `HOST` technical evidence plus pinned licence facts | Obtain written commercial rights for a technically complete checkpoint or replace it, then pass mixed-script safety, the full quality gate, listening acceptance, and exact-phone profiling | `[D:TTS; D:SUMMARY]` |
| What authorizes SM8850 NPU execution? | Compile selection unavailable `HOST`; staged context headers are exact QNN 2.40, while choosing exact 2.45 as the programme pin is `PROJECTED` | Obtain SM8850 compilation access or independently hashed exact-2.45 contexts, build and hash an app-UID runner with exactly one 2.45 provider, integrate the complete stack, then follow the phone runbook | `[E:E1; E:E3; E:E5; E:E6]` |
| What closes naturalness and native credibility? | `UNKNOWN`; no listening panel or native-speaker acceptance | Run the prescribed listening review and a suitably designed native-speaker acceptance exercise; do not infer it from host ASR backchecks | `[D:TTS; D:SUMMARY]` |

## 6. Evidence appendix

The machine-readable evidence spine is `workstreams/sub2b_sprint/sprint/out/h/evidence_index.json`. It maps each report claim to its value, unchanged evidence class, producing task, source file, and SHA-256 where the producing lane provides one. Any future edit that introduces a number or strengthens a claim must first add or update the corresponding H1 entry. `[A:A7; D:TTS; E:REGISTRY]`

The detailed landed records remain the authority behind the index: Lane A's `ledger.json` and archived licence captures, Lane D's `tts_table.json` plus D1–D5 evidence, and Lane E's E1–E6 files, `artifact_registry.json`, and `PHONE_RUNBOOK.md`. Pending B, C, and F material becomes report evidence only after it lands and is indexed. `[A:A6; A:A7; D:TTS; E:E1; E:E3; E:E5; E:E6; E:REGISTRY; PENDING-B; PENDING-C; PENDING-F]`
