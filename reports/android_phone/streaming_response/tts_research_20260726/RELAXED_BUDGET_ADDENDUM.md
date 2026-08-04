# Addendum — best quality under a 1–2 s first-audio cap

Follow-up to [`REPORT.md`](REPORT.md), which was written against a < 0.9 s
first-audio gate. This asks a different question: **if the hard cap is 1–2 s,
what is the best-quality offline TTS?**

Host: Apple M2, two threads. Phone still disconnected; nothing here is a phone
measurement.

## Summary of what changes

1. **Russian: five steps, not four** — and this was already true under the old
   gate. The original "four steps" call rested on a five-repeat P95 that was one
   contention outlier. At 15 repeats, five steps is 0.529 s P95 against four
   steps' 0.437 s, and five/six steps tie for best intelligibility (0.3% CER,
   1.7% WER). Six costs 175 ms more for nothing measurable; eight is worse on
   both axes. **Five steps.**
2. **Mandarin on the two-thread host: unchanged.** A 2 s cap does not rescue
   MeloTTS or Kokoro under this host constraint, because both are limited by
   cost that scales with the work, not by a fixed startup cost a longer wait
   could absorb. `matcha_zh_baker` stays the host pick. The later phone run
   overturns the global NPU-only conclusion: MeloTTS-ZH is CPU-viable at 4–8
   threads; see [`DEVICE_RESULTS.md`](DEVICE_RESULTS.md).
3. **Kokoro's non-finite PCM is an INT8 bug, not a model defect** — float32 is
   clean at every input length *and* faster. The rejection reason changes from
   "emits invalid audio" to "too slow".
4. **INT8 was slower than float32 for every model tested.** MeloTTS INT8 is
   three times slower than its float32 build. On this ARM CPU, ONNX dynamic
   quantization is a pessimization.
5. **ZipVoice-distill, the modern quality ceiling, does not fit** — and its
   cost profile is the wrong shape for streaming, being *worse* on short input
   than long.
6. **A third ASR scoring artifact was found and fixed.** Whisper hallucinated
   subtitle credits onto trailing silence, which would have made higher
   Supertonic step counts look like a regression.

## 1. What a relaxed cap actually buys — and what it does not

Time-to-first-audio is not one number. Two things are coupled:

- **first audible time** = when the first chunk lands, plus the startup cushion;
- **queue safety** = whether later chunks arrive before the player drains.

A slow synthesizer starves the queue *only if playback starts too early*. Give
it a larger cushion and the starvation disappears — at the cost of a later first
word. So the right question under a 1–2 s cap is not "which engines are under
0.9 s" but **"what is the smallest cushion that keeps each engine gap-free, and
does the resulting first-audible time still fit?"**

For chunks produced at p_k with durations d_k, playback starting at p_0 + C is
gap-free exactly when C ≥ max_k (p_k − p_0 − Σ_{j<k} d_j). That minimum cushion,
and the earliest safe first-audible time it implies, is computed for every
candidate from the recorded callback traces by
[`latency_budget.py`](../../../../scripts/tts_research_20260726/latency_budget.py).

**The single most useful number this produces is the cushion, not the latency.**
A candidate needing a 0.00 s cushion is limited only by how much text it is
handed — bounding the first flush fully solves it. A candidate needing seconds of
cushion has a throughput problem no budget can fix.

### Earliest safe first-audible time, median / worst (s), 2 s budget

Bounded first flush (short and single-answer cases — what the orchestrator
should actually produce):

| Candidate | Lang | 1.0× | 1.5× | 2.0× | cushion needed @1.5× |
|---|---|---:|---:|---:|---:|
| `matcha_zh_baker` | zh | 0.15 / 0.27 | 0.23 / 0.40 | 0.31 / 0.53 | 0.00 |
| `piper_zh_huayan` | zh | 0.28 / 0.80 | 0.42 / 1.20 | 0.56 / 1.60 | 0.00 |
| `melotts_zh` | zh | 1.05 / 1.94 | 1.83 / 2.93 ! | 3.19 / 4.43 ! | **1.39** |
| `kokoro_zh` (int8) | zh | 3.68 / 4.40 ! | 5.52 / 8.17 ! | 7.36 / 12.17 ! | **3.73** |
| `espeak_ng` | zh | 0.02 / 0.02 | 0.03 / 0.04 | 0.04 / 0.05 | 0.00 |
| `piper_ru_dmitri` | ru | 0.19 / 0.57 | 0.28 / 0.85 | 0.37 / 1.13 | 0.00 |
| `piper_ru_ruslan` | ru | 0.19 / 0.44 | 0.28 / 0.66 | 0.37 / 0.87 | 0.00 |
| `piper_ru_irina` | ru | 0.24 / 0.50 | 0.36 / 0.75 | 0.47 / 1.00 | 0.00 |
| `supertonic3` @4 | ru | 0.76 / 1.46 | 1.14 / 2.19 ! | 1.52 / 2.92 ! | 0.00 |
| `supertonic3` @5 | ru | 1.02 / 1.83 | 1.53 / 2.74 ! | 2.04 / 3.66 ! | 0.00 |
| `espeak_ng` | ru | 0.02 / 0.03 | 0.03 / 0.04 | 0.05 / 0.05 | 0.00 |

Unbounded four-sentence paragraph (what happens if the orchestrator does *not*
bound the flush):

| Candidate | Lang | 1.0× | 1.5× | cushion needed @1.5× |
|---|---|---:|---:|---:|
| `matcha_zh_baker` | zh | 0.31 / 0.36 | 0.46 / 0.54 | 0.00 |
| `piper_zh_huayan` | zh | 0.34 / 0.52 | 0.50 / 0.79 | 0.00 |
| `melotts_zh` | zh | 2.64 / 3.07 ! | 5.01 / 10.43 ! | **6.68** |
| `kokoro_zh` (int8) | zh | 8.94 / 9.24 ! | 20.26 / 20.71 ! | **13.57** |
| `piper_ru_dmitri` | ru | 0.42 / 0.57 | 0.64 / 0.85 | 0.00 |
| `supertonic3` @4 | ru | 3.45 / 3.64 ! | 5.17 / 5.45 ! | 0.00 |
| `supertonic3` @5 | ru | 4.20 / 4.58 ! | 6.30 / 6.88 ! | 0.00 |

`!` marks a worst case over the 2 s budget.

### Reading these two tables together

**Supertonic's paragraph numbers look terrible and are not a real problem.** Its
cushion requirement is 0.00 s at every slowdown: the queue never starves. The
3.45 s figure is entirely "the first sentence of a four-sentence answer is long",
which the orchestrator fixes by flushing earlier. Same for every Piper voice and
Matcha.

**MeloTTS and Kokoro-INT8 are the opposite case.** They need 1.39 s and 3.73 s of
cushion respectively even on bounded flushes, and 6.68 s / 13.57 s on a
paragraph. That is a throughput deficit, not a scheduling choice, and no
first-audio budget removes it.

So a 1–2 s cap changes the answer for **Russian**, where it admits a
higher-quality Supertonic setting, and largely does **not** change it for
**Mandarin on CPU**, where the slow candidates are slow for a reason the budget
cannot address.

## 2. Kokoro's non-finite PCM is an INT8 artifact, not a model defect

The relaxed cap makes Kokoro worth a second look, so it was re-run in float32.
The result overturns how this defect has been recorded since the first Kokoro
evaluation.

Identical model, identical speaker (`sid 3`, `zf_001`), identical inputs, same
chunk-floor sweep:

| Input | Han chars | INT8 finite PCM | **float32 finite PCM** |
|---|---:|---|---|
| `好` | 1 | **0/5** | **5/5** |
| `好的` | 2 | **0/5** | **5/5** |
| `好的。` | 2 | **0/5** | **5/5** |
| `你好啊` | 3 | 5/5 | 5/5 |
| `这是测试` | 4 | 5/5 | 5/5 |
| `请稍等一下` | 5 | 5/5 | 5/5 |
| `北京今天很晴朗` | 7 | 5/5 | 5/5 |

Every short-input failure disappears in float32, and the suite case
`zh_micro_ack` goes from 0/5 to **5/5**. **Kokoro-82M does not have a
short-Mandarin bug. The sherpa-onnx INT8 build does.** The earlier
characterization — a three-Han-character floor, a chunk-length guard as the
mitigation — was describing a quantization artifact as a model property.

The same comparison also overturns the assumption that INT8 is the fast build:

| | INT8 | float32 |
|---|---:|---:|
| RTF median (worst case) | 1.385 | **0.969** |
| RTF P95 (worst case) | 1.412 | **1.347** |
| First chunk P95, short clause | 2.654 s | **2.069 s** |
| First chunk P95, micro-ack | 1.069 s | **0.875 s** |
| Shipped weights | 199 MiB | 401 MiB |
| Peak RSS | 519 MiB | 662 MiB |

Float32 is **faster than INT8 on every case**. This is the well-known ARM
failure mode for ONNX Runtime dynamic quantization: the per-tensor
quantize/dequantize around each matmul costs more than the cheaper integer
arithmetic saves. It is worth remembering the next time an INT8 build is
assumed to be the cheap option — the same question is asked of MeloTTS in
section 4.

**This does not promote Kokoro.** At 0.969 RTF median it barely synthesizes
faster than it plays, its bounded-flush first chunk is 2.07 s (over even the
relaxed cap), and it doubles to 401 MiB of weights and 662 MiB of peak RSS. What
changes is the *reason* for rejection: Kokoro is rejected on throughput, and the
"invalid PCM" line should be struck from its record and replaced with "the INT8
build is unusable; the float32 build is correct but too slow."

## 3. Russian: how many denoising steps fit

### First, a correction to the original report

The first version of `REPORT.md` concluded "four steps is the correct setting"
because the five-step first-chunk P95 measured 0.93 s, apparently over the 0.9 s
gate. Inspecting the individual runs showed that P95 came from **one** outlier
(0.998 s against a 0.629–0.680 s cluster) in a five-repeat sample — where P95 is
just the maximum. Re-measured at 15 repeats the distribution is tight and much
lower. Both `REPORT.md` and this section now say five steps.

### The step ladder, 15 repeats, streaming and answer cases

| Steps | Short clause P95 | Answer P95 | RTF median | RTF P95 | Max flush @2 s, 1.5× |
|---|---:|---:|---:|---:|---:|
| 4 | **0.437 s** | 0.802 s | 0.182–0.205 | 0.217 | 96 chars |
| 5 | **0.529 s** | 0.962 s | 0.221–0.249 | 0.260 | 74 chars |
| 6 | **0.704 s** | 1.082 s | 0.259–0.293 | 0.328 | — |
| 8 | 1.316 s | **2.506 s** | 0.348–0.552 | 0.679 | 42 chars |

Cost scales close to the step count up to six, then steepens: the 6→8 jump is a
1.33× increase in steps for a 1.87× increase in first-chunk time, and eight steps
brings RTF P95 to 0.679 — against a 0.70 gate, with essentially no margin left.

### What this means at each budget

- **Under the original < 0.9 s gate**, four, five *and* six steps all pass on the
  streaming clause (0.437 / 0.529 / 0.704 s). Only eight fails.
- **Under a 2 s cap**, four, five and six pass everywhere. Eight passes on a
  short clause but its answer-case P95 of 2.506 s is over budget, so eight steps
  is only usable with the first flush bounded to ~42 characters.

So the relaxed cap does not unlock a *new* setting for Russian so much as it
widens the margin on settings that were already reachable. The real change came
from re-measuring: five and six steps were always inside the original gate, and
the first report ruled five out on a measurement artifact.

Whether six steps is worth 0.18 s more than five, and eight worth the flush
restriction, depends on whether the extra denoising is audible — see the
intelligibility and proxy scores below and the blinded bundle.

## 4. Mandarin: is there anything better within the budget?

### ZipVoice-distill — the quality ceiling, and not a phone candidate

ZipVoice-distill (zero-shot flow matching, ZH+EN, Emilia training data) is the
most modern architecture available in the sherpa-onnx catalogue and was run as
the quality ceiling the relaxed cap might unlock. It does not fit, by a wide
margin:

| | ZipVoice-distill INT8 | `matcha_zh_baker` |
|---|---:|---:|
| First chunk P95, short clause | **3.60 s** | 0.16 s |
| First chunk P95, micro-ack | 2.78 s | 0.09 s |
| RTF median (worst case) | **3.36** | 0.146 |
| RTF median (answer cases) | 0.96–1.00 | 0.085–0.089 |
| Peak RSS | **1,008 MiB** | 287 MiB |
| Shipped weights | 195 MiB | 139 MiB |

Every waveform gate passes and the chunk-floor sweep is clean, so this is a
latency and memory rejection, not a quality one. Two things make it decisive:
the short-utterance RTF is *worse* than the long-utterance RTF (3.36 on a
two-character acknowledgement against 0.82 on a paragraph), because the
flow-matching decoder has a large fixed cost per call — exactly the wrong shape
for a clause-streamed assistant that flushes short chunks first. And 1,008 MiB
of peak RSS is the largest of any candidate measured, before LFM is resident.

**A defect worth recording even though it is not why ZipVoice was rejected:**
its float output exceeds full scale on **16 of 35 runs**, peaking at 1.197. The
clipped fraction is tiny (0.000047–0.000128, under the gate threshold) but it is
non-zero, and writing that to PCM16 clips. Any use of ZipVoice — including as a
listening reference — needs a limiter or a couple of dB of pad.

It is retained in the listening bundle as a quality ceiling, the scoped role
for a model too heavy to ship.

### Does INT8 rescue MeloTTS-ZH? No — it makes it three times worse

MeloTTS-ZH is the only Mandarin candidate that is MIT-licensed and free of any
GPL-3.0 eSpeak runtime dependency, and float32 misses the latency budget. If
INT8 closed that gap, the licence-clean route would become CPU-viable.

| | float32 | INT8 |
|---|---:|---:|
| First chunk P95, short clause | 0.993 s | **3.238 s** |
| RTF median (worst case) | 0.715 | **2.288** |
| RTF P95 | 0.942 | 2.835 |
| Flush sweep, 42 characters | 4.671 s | **15.791 s** |
| Shipped weights | 182 MiB | 71 MiB |
| micro-CER | 17.6% | 19.2% |

INT8 is three times slower at every flush length and marginally less
intelligible. Together with the Kokoro result this is now a pattern rather than
an anomaly: **on this ARM CPU, ONNX dynamic INT8 quantization cost time for
every model tested.** The footprint saving is real (182 → 71 MiB) and useless at
these speeds.

MeloTTS-ZH is therefore not CPU-viable in either precision under this
two-thread host constraint. The later phone run reaches 0.44 RTF at eight
threads, so its NPU route is an efficiency/deployment option rather than the
only CPU-feasible path.

**A packaging bug found on the way.** The sherpa-onnx `vits-melo-tts-zh_en`
release archive ships `model.int8.onnx` as a **133-byte git-lfs pointer**, not
the model; loading it fails with "Protobuf parsing failed". The real 51 MiB
weights were fetched from the repository the pointer names and verified against
the SHA-256 and size recorded in the pointer, both of which match. Provenance is
recorded beside the model as `INT8_PROVENANCE.txt`, with the original pointer
retained. Anyone reproducing the INT8 measurement from the release archive alone
will hit this.

## 5. First-flush length profile — the number an integrator can act on

The previous report could only say "bound the first flush". This measures how
long a flush of N characters actually costs, using single-segment texts of
graded length so each is exactly one chunk for every engine
([`bench.py --flush-sweep`](../../../../scripts/tts_research_20260726/bench.py)).

First-chunk latency turns out to be close to affine in flush length — R² ≥ 0.991
for every candidate — so inverting the fit gives the largest first flush that
still lands inside a budget
([`flush_profile.py`](../../../../scripts/tts_research_20260726/flush_profile.py)).

| Candidate | Lang | fixed cost | s/char | R² | 1 s @1× | 1 s @1.5× | 2 s @1× | **2 s @1.5×** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `piper_ru_dmitri` | ru | 0.034 s | 0.0037 | 0.9999 | 260 | 170 | 529 | **349** |
| `supertonic3` @4 | ru | 0.216 s | 0.0116 | 0.9974 | 67 | 38 | 153 | **96** |
| `supertonic3` @5 | ru | 0.287 s | 0.0141 | 0.9913 | 50 | 26 | 121 | **74** |
| `supertonic3` @8 | ru | 0.373 s | 0.0227 | 0.9951 | 27 | 12 | 71 | **42** |
| `matcha_zh_baker` | zh | 0.057 s | 0.0115 | 0.9968 | 82 | 53 | 169 | **111** |
| `piper_zh_huayan` | zh | 0.035 s | 0.0153 | 0.9964 | 62 | 41 | 128 | **84** |
| `melotts_zh` | zh | 0.129 s | **0.1083** | 0.9984 | 8 | 4 | 17 | **11** |
| `kokoro_zh_fp32` | zh | 0.765 s | 0.1038 | 0.9906 | 2 | 0 | 11 | **5** |

Read against the shipped flush policy — a sentence, a punctuation boundary, or
**50–70 characters**, whichever comes first:

- **Supertonic at 5 steps fits a 2 s cap** with 74 characters of headroom at 1.5×
  slowdown, comfortably covering the 50–70 character policy. At 8 steps the
  headroom drops to 42 characters, so the policy would have to be tightened.
  Under a **1 s** cap the picture reverses: 4 steps allows only 38 characters and
  5 steps only 26, so a 1 s cap effectively forces 4 steps *and* a shorter flush.
- **Matcha and Piper-ZH have no practical bound** (111 and 84 characters at
  2 s @1.5×); the flush policy can be chosen for prosody rather than latency.
- **MeloTTS-ZH allows 11 characters** at 2 s @1.5×. That is below one Mandarin
  clause, and it is the marginal cost that kills it — 0.108 s/char, nine times
  Matcha's — not a fixed startup cost that a bigger budget could absorb.
- **Kokoro float32 has the opposite shape**: a 0.765 s fixed cost per call, by
  far the largest measured, so even a five-character flush spends most of the
  budget before synthesizing anything.

The two profiles that matter most are `melotts_zh` and `kokoro_zh_fp32`, because
they show *why* a relaxed budget does not rescue them. Both are dominated by a
term that scales with the work, not by a constant that a longer wait amortizes.

## 6. A third ASR scoring artifact — and why it matters

Whisper is trained on subtitle corpora and hallucinates their credit lines onto
trailing silence. On two Russian paragraph cases it appended *"редактор
субтитров н закомолдина корректор а егорова"* — text that is simply not in the
audio.

The effect on the conclusion was direct. Scored raw, six steps looked like a
regression:

| Steps | micro-CER raw | micro-CER after stripping | micro-WER after |
|---|---:|---:|---:|
| 4 | 2.49% | 2.49% | 8.3% |
| 5 | 0.28% | **0.28%** | **1.7%** |
| 6 | **12.98%** | **0.28%** | **1.7%** |
| 8 | 9.12% | 1.10% | 5.0% |

Without the fix the table said "more steps degrade intelligibility"; with it,
five and six steps tie for best. The judge now strips a trailing credit block
only when it begins at or beyond 80% of the reference length, keeps the raw
hypothesis alongside, and records exactly what was removed — so the edit is
auditable and cannot silently delete real recognized speech.

That is the third scoring artifact in this study, after digit rendering and
traditional Han. All three inflated error rates for *well-behaved* engines. The
lesson worth carrying: when an intelligibility number moves a decision, read the
actual transcript before believing it.

## 7. Revised recommendation

### Russian — Supertonic 3 at **five** steps

| Steps | First chunk P95 | RTF median | micro-CER | micro-WER | DNSMOS OVRL | Max flush @2 s, 1.5× |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 0.437 s | 0.203 | 2.49% | 8.3% | 3.188 | 96 |
| **5** | **0.529 s** | **0.246** | **0.28%** | **1.7%** | 3.305 | **74** |
| 6 | 0.704 s | 0.289 | 0.28% | 1.7% | 3.350 | — |
| 8 | 1.316 s | 0.552 | 1.10% | 5.0% | 3.350 | 42 |

Five steps is the knee. It ties six for the best intelligibility measured, costs
175 ms less to first chunk, and leaves 74 characters of flush headroom — enough
for the 50–70 character policy. Six is defensible if listening prefers it (the
proxy marginally does, 3.350 vs 3.305, which is inside the noise). Eight costs
double the latency for worse intelligibility.

Fallback remains Piper `ru_RU-dmitri` (CC0, 0.13 s, 349 characters of headroom),
with the caveat that it scores lowest of the neural Russian voices on the proxy.

### Mandarin — unchanged, and the relaxed cap does not help

`matcha_zh_baker` remains the host pick (0.16 s, RTF 0.146, 4.0% CER, 111-char
flush headroom), still blocked commercially by a non-commercial corpus.
MeloTTS-ZH remains the only licence-clean route and is now conclusively
CPU-unviable in both precisions, so its NPU path is the only one. Nothing in the
relaxed-cap candidate set changes this:

| Candidate | First chunk P95 | RTF median | micro-CER | DNSMOS OVRL | Verdict at 2 s |
|---|---:|---:|---:|---:|---|
| `matcha_zh_baker` | 0.16 s | 0.146 | **4.0%** | 3.182 | fits with margin |
| `piper_zh_huayan` | 0.15 s | 0.091 | 9.6% | 3.305 | fits; corpus licence unknown |
| `melotts_zh` | 0.99 s | 0.715 | 17.6% | 3.226 | flush ≤ 11 chars; queue starves |
| `kokoro_zh_fp32` | 2.07 s | 1.204 | 7.2% | 3.172 | over budget |
| `zipvoice_zh` | 3.59 s | 3.355 | 4.8% | **3.314** | far over; ceiling reference |
| `melotts_zh_int8` | 3.24 s | 2.288 | 19.2% | 3.006 | far over |
| `kokoro_zh` (int8) | 2.65 s | 1.385 | 8.9% | 3.114 | over; invalid PCH under 3 chars |
| `espeak_ng` | 0.02 s | 0.025 | **51.2%** | 2.894 | above the 40% hard limit |

ZipVoice has the best Mandarin proxy score and the second-best CER, which is
consistent with it being the most modern architecture here — and it is four
seconds to first audio with 1 GiB of RSS. That is the honest shape of the
trade-off: the quality ceiling is real and completely out of reach on this CPU.

### What is still not decided

Naturalness. Every ranking above is intelligibility, latency, footprint, and an
artifact-detecting proxy. The Mandarin proxy spread is 3.01–3.31 across engines
that differ enormously in measured intelligibility, and it places eSpeak above
three neural engines on P.808 while ASR puts eSpeak at 51.2% CER. **It cannot
choose between two clean voices.** That is what the on-device bench and a human
ear are for.
