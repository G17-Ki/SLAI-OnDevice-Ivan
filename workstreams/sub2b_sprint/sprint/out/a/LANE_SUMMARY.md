# LANE A — LEDGER & LICENCE — summary

**Status: A1–A7 all complete.** ~6 hours against a ~13 h budget. 0 bytes of model weights
downloaded (one 92 MB exception, deleted); peak storage ~94 MB against a 1 GB cap.

## Headline finding

**The parameter cap is not what constrains this programme. All six stacks are under
2,000,000,000 — on the measured basis and on the worst case — and the prior ledger was
accurate to better than 0.8% everywhere. What constrains the programme is licences, and
a punctuation model nobody budgeted for.**

Confidence: **high** on the arithmetic (52 of 58 rows are tensor-level parses; zero `UB`
rows remain). **High** on the licence facts, which are archived verbatim with sha256s;
**not for me to judge** on what those facts legally imply.

## Verdicts

| Stack | measured | Δ vs prior | % cap | headroom | verdict | licence |
|---|---:|---:|---:|---:|---|---|
| **H** | 1,396,245,218 | −756,502 | 69.81% | 603,754,782 | **COMPLIANT** | 🔴 |
| **L** | 1,676,733,894 | −13,390,767 | 83.84% | 323,266,106 | UNVERIFIED | 🟠 |
| **D** | 1,831,533,225 | −8,152 | 91.58% | 168,466,775 | UNVERIFIED | 🔴 |
| **M** | 1,674,645,423 | −71,805 | 83.73% | 325,354,577 | UNVERIFIED | 🟢 **GREEN** |
| **Q** | 1,964,482,923 | −36,490 | 98.22% | 35,517,077 | UNVERIFIED | 🔴 |
| **F** | 1,975,632,107 | −36,490 | 98.78% | 24,367,893 | UNVERIFIED | 🔴 |

**No stack is NONCOMPLIANT.** The only COMPLIANT stack is H — the disposable rig.

## The five things worth acting on

**1. OmniASR cannot emit punctuation, and Q and F cannot afford a model that can.**
Its sentencepiece tokenizer is character-level with **zero** punctuation tokens (0
occurrences of `.` `,` `?` `!` `;` `:` in the full 87,562 B blob; digits and CJK/Cyrillic
present, so the scan is sound). A punctuation source is *architecturally required*. The
cheapest five-language neural restorer I could measure is **58,199,117** — more than Q's
entire headroom (35,517,077) and more than twice F's (24,367,893). Both stacks stay legal
only by delegating punctuation and ITN to the backbone LLM at zero parameters. **That must
be a build-time assertion, not an assumption.**
The irony: TH-60 gates Q on the *g2pW* hazard (102,882,442, now `EXACT`). g2pW is
avoidable — Q7/F7 already specify rule-based pypinyin. The unavoidable row was the one
sized at zero.

**2. Every Piper voice descends from a research-only corpus.** Piper's own MODEL_CARDs
say `en_US-lessac-medium` was "Trained from scratch" on the Blizzard 2013 Lessac corpus
and that kusal, davefx (es), siwis (fr), dmitri (ru) and huayan (zh) are **each**
"Finetuned from U.S. English lessac voice". That corpus's licence, archived verbatim,
grants use *"exclusively for Research Purposes only"*, *"without the right to
sub-licence"*. This puts H's, Q's and F's entire TTS bank (78.3 M / 62.6 M / 62.6 M)
behind a legal question I am not qualified to answer — but the facts are documented and
the question is now explicit.

**3. Falcon-H1's licence terms are not established.** The repo has **no LICENSE file**.
Its cited terms page contains **zero occurrences of "Falcon H1"** — it covers Falcon 2
11B, 180B, Mamba 7B and Falcon 3 only. The Acceptable Use Policy is defined as "the latest
version from time to time" with an instruction to "monitor the web address" — a
unilaterally mutable, forward-binding obligation that an **offline on-device product
cannot satisfy by construction**. Stack Q's backbone is Apache-2.0, has a LICENSE file,
and costs 11,149,184 *fewer* parameters. On licence grounds Q dominates F outright.

**4. Magpie ships no codec — Stack L was missing a row.** Magpie's `model_config.yaml`
declares `codecmodel_path: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`. The checkpoint
holds 16 codebook embedding tables and emits audio tokens; nothing in it makes a waveform.
The codec is **+62,600,978** (inference path; the full `.nemo` is 106,087,325, of which
43,486,347 is a GAN discriminator that never ships). §2.2 predicted exactly this. L
absorbs it.

**5. Stack M is the only clean licence sheet in the programme.** Apache-2.0 for ASR,
backbone, TTS and neural audio tokenizer; MIT for VAD. No copyleft, no non-commercial
corpus, no research-only chain, no mutable terms, and — because MOSS needs no external
G2P — no eSpeak GPL-3.0 linkage. M also has the second-largest headroom (325,354,577) and
its only open row is punctuation.

## What got settled

**A1 — Qwen3-ASR-0.6B `lm_head` is TIED.** `EXACT`. Reports A and B right, C wrong. Both
tensors exist on disk at distinct offsets, and a **full-region SHA-256 over 100% of both
311,164,928 B regions** returns identical digests (`ab067cf4…`). Config corroborates.
C's figure was the HF API's `safetensors.total`, which does no tensor-identity
de-duplication — on a materialised tie it is the *serialized* basis. **M1 = 782,426,112**;
M's planning headroom is 325,282,772, and §2.4's restriction to the serialized figure is
discharged. Compliance was never at stake — M was legal on both bases.
The same trap caught `Qwen3-0.6B` independently (API 751,632,384 → TIED → 596,049,920).

**A2 — Parakeet's Δ24 is 24 I64 BatchNorm counters.** `encoder.layers.{0..23}.conv.norm.num_batches_tracked`.
Both published figures are correct on their own basis: **627,057,286** learned-F32,
**627,057,310** all tensor elements. Byte reconciliation exact to **0**. §1.2 favours
627,057,286; the ledger carries the conservative figure. Bonus: the `.nemo` and
safetensors packagings of the same checkpoint differ by 4,892,144 in their head modules —
identical encoders — so never mix the two bases.

**A3/A4 — every component header-parsed.** 28 components, 58 ledger rows, **52 `EXACT`,
6 `UNKNOWN`, zero `UB`**. Falcon's revision `80ebc50d…` asserted *before* any count was
recorded; the Deep variant measures 1,554,874,768, exactly **11,280** more, as §2.4b
predicted. **TH-59 and TH-61 both discharged.** Four fp16 doubling hazards are not
bounded but *disproven*, because the parse enumerates dtype. Three "conservative" bounds
turned out to be breached (zipformer +3,873,962, canary +1,743,137, RU Zipformer
+610,041). Supertonic ships **ten** voice styles, not five (+64,640).

**A5 — the front-end hazards are now numbers, not adjectives.** g2pW-class =
**102,882,442** `EXACT` (confirms §3.10 to the parameter). Punctuation restorers measured
from 58,199,117 to 558,847,496. D's zh/en punctuation row **closes at zero** —
SenseVoice does ITN natively via `use_itn`. §2.2's claim that Silero bundles a stress
marker is **refuted** (constructor takes `accentor=None`); L needs an external RU stress
row it thought it had. And the like-for-like question in the brief resolves in D's favour:
D's front end is genuinely *cheaper* than L's, not merely less documented.

**A6 — ten licence texts archived** with `captured_utc`, `source_url`, `http_status`,
sha256 and byte count. Confirmed all five of the plan's named blockers; found three it
did not list (TEN VAD's Agora non-compete rider, charsiu byT5 declaring **no licence at
all** under 20,653,952 of Stack H's front end, and Falcon's terms page not naming
Falcon-H1).

**A7 — `ledger.json`** carries a verdict for all six stacks, and **every non-`EXACT` row
names the experiment that would close it** (verified mechanically: 6 of 6).

## What I could not close

| item | class | closing experiment |
|---|---|---|
| **Q9 / F9** punctuation | `UNKNOWN` | **Not closable by measurement** — nothing affordable exists. Closes by architectural commitment to backbone delegation. |
| **D11b** — parakeet punctuation for es/fr/ru | `UNKNOWN` | inspect `tokenizer.json` token set, ~20 min |
| **M7** — does Qwen3-ASR emit punctuation | `UNKNOWN` | punctuated fixture per language; **Lane C can return this at zero marginal cost** |
| **L9** — RU stress marker | `UNKNOWN` | pin an accentor and header-parse (4 candidate RUAccent repo names all 401), or adopt the rule-based rung at 0 |
| **L10** — RU punctuation (GigaAM is char-CTC) | `UNKNOWN` | as D11b |
| **L5b** — does the shipped codec export prune the discriminator? | `UNKNOWN` | worst case +43,486,347; L absorbs it |
| **D3** — in-project RU Zipformer identity | substitution | I measured the public `sherpa-onnx-zipformer-ru-int8-2025-04-20` (matches on date and 60.2 MB archive size); sha256 the in-project binary against it |
| **RUAccent's real count** | `CLAIMED` | none of four candidate repo names resolves; not upgraded, not interpolated |

## Caveats other lanes should carry

- **COMPLIANT is about arithmetic only.** Four of six stacks carry a 🔴 licence flag. This
  lane can certify a number; it cannot certify shippability.
- **Counting convention:** `params` = all tensor elements (conservative, per §2.0);
  `params_learned` excludes I64/I32 buffers per §1.2. Totals use the conservative figure.
  The two differ by at most a few thousand per stack.
- **Qwen3.5-0.8B's vision tower (100,592,896) sits in the same safetensors file as the
  LM.** Excluding it is legitimate *only* because the shipped artefact is the LM-only
  GGUF. Any build shipping upstream safetensors takes the hit — carried in `ledger.json`
  as a NOT-COUNTED row so it cannot be forgotten. Stack D would feel it immediately.
- **Exports can re-materialise tied weights.** MOSS's ONNX carries its 16384×768 embedding
  **three times**; naive per-graph summation gives 391,092,480 against a true 117,311,232
  — a 3.3× trap. Never take a param count from an ONNX repo byte total.

## Deliverables

`A1`–`A7` (`.json` + `.md`) · **`ledger.json`** (58 rows, six verdicts) ·
`licences/` (10 dated captures with sha256) · `tools/` (the header-parsing toolkit:
safetensors, Range-seeking ONNX enumeration, ZIP/pickle meta-device walk, TAR/NeMo, GGUF)
· `artifacts/` (git-ignored raw parses).

The toolkit is reusable and is the reason this lane cost ~94 MB instead of ~40 GB. Any
lane needing a parameter count can call it.

## Dependencies on other lanes

None unmet — Lane A is upstream of everyone. Two cheap offers in return: **Lane C** can
close **M7** and **D11b** at zero marginal cost while running ASR fixtures, and any lane
touching Falcon should read `A6.md` §R-3 before planning around it.
