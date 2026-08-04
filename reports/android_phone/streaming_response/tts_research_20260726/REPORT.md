# Natural multilingual TTS research — Mandarin and Russian

Date: 2026-07-26<br>
Repository commit at start: `a0604527ff2a2a4abef4376a41023c4127a4c07d` (`main`, clean)<br>
Host: Apple M2, 8 logical cores, 8 GiB RAM, two inference threads<br>
Phone: **disconnected**. No ADB was run and nothing here is a phone measurement.

> **Follow-up:** this report is written against a **< 0.9 s** first-audio gate.
> [`RELAXED_BUDGET_ADDENDUM.md`](RELAXED_BUDGET_ADDENDUM.md) re-profiles the same
> candidates against a **1–2 s** cap and adds several that only make sense there.
> It also corrects one finding below: Kokoro's non-finite short-input PCM is an
> INT8 quantization artifact, not a model defect — the float32 build is clean.
>
> **Physical-phone correction:** [`DEVICE_RESULTS.md`](DEVICE_RESULTS.md)
> later showed that MeloTTS-ZH is CPU-viable at 4–8 threads. That device result
> supersedes this report's two-thread host conclusion that it was NPU-only.

## What this run set out to do

Replace the robotic Android system-TTS fallback with a natural offline voice,
Mandarin first and Russian second, without disturbing the validated English
Expressive-LFM route. The deliverable is a defensible recommendation plus the
evidence behind it — not a claim that any new voice has been validated on
hardware.

## Decisions

1. **Mandarin: recommend `matcha-icefall-zh-baker` for the demo.** It is the
   only Mandarin candidate that passes every measured gate: 0.16 s first
   playable chunk P95, 0.146 RTF (flat across every case and repeat), 4.0%
   micro-CER, zero waveform-gate failures, no simulated playback gap even at
   2.0× host slowdown, 139 MiB. Its blocker is legal, not technical: the Baker
   corpus it was trained on is **non-commercial only**. Note what this does
   *not* say — Matcha is not a shippable answer, and it is the best available
   one under this two-thread host constraint.
2. **Under the two-thread host constraint, MeloTTS-ZH misses the CPU gate; its
   NPU route passes.** On this host MeloTTS runs at 0.715 RTF (the gate is 0.70) and
   **produces playback gaps at 1.0× host speed** — 0.429 s of gap on a
   four-sentence answer before any mobile slowdown is applied. On the NPU it now
   has a complete exact-QRD pass: a real Mandarin sentence through BERT,
   encoder, flow, and **nine repeated decoder calls**, producing finite audio at
   2.94% NRMSE / 0.99958 cosine against the host golden and **0.0000 CER** on
   the ASR back-check — and the harder accumulated variant, where each stage is
   fed the previous stage's remote output, still passes every waveform gate at
   **0.0000 CER**. It is also the only Mandarin candidate that is MIT and needs
   no GPL-3.0 eSpeak data. See [`NPU_STATUS.md`](NPU_STATUS.md).
3. **Russian: recommend Supertonic 3 at five steps as primary and Piper
   `ru_RU-dmitri` as the licence-clean fallback.** *(Corrected — the first
   version of this report said four steps, on a five-repeat P95 that turned out
   to be a single outlier; see the section below.)* At 15 repeats, five steps is
   0.529 s P95 to first chunk against four steps' 0.437 s, clearing the 0.9 s
   gate with room, and it is better on every quality signal measured. Piper
   `dmitri` is the only candidate in the entire run with no blocking failure and
   no missed target, and the only compact Russian voice found with a
   public-domain (CC0) training corpus.
4. **Do not touch the English Expressive-LFM route.** Nothing measured here
   competes with it; Supertonic English is retained as the existing
   multilingual fallback.
5. **eSpeak is now disqualified by measurement, not by description.** Mandarin
   micro-CER 55.2% exceeds the 40% hard limit; Russian micro-WER 61.7% exceeds
   the 50% hard limit, and one Russian case was transcribed in Latin script.

**The naturalness question is deliberately still open.** No human has scored the
blinded bundle. Everything above is a *screen*: it disqualifies candidates on
measured grounds and orders the survivors, but the final Mandarin choice between
Matcha and MeloTTS, and the Russian choice between Supertonic and Piper, should
be confirmed with the private blinded listening bundle before anything ships.

## Evidence boundaries

- Every latency, RTF, and memory number is an **Apple M2 host** measurement at
  two threads. A phone will be slower; the playback simulation at 1.5× and 2.0×
  host slowdown is the guard against that, not a substitute for it.
- ASR back-check is an **intelligibility screen**. A low CER proves the words
  survived synthesis. It says nothing about whether a voice sounds human.
- DNSMOS is a **speech-quality proxy trained for noise suppression**, not a TTS
  naturalness predictor. Its limits are visible in the data itself: it ranks
  eSpeak Mandarin *above* Kokoro on P.808 while ASR puts eSpeak at 55.2% CER and
  Kokoro at 8.9%. It detects artifacts; it cannot hear wrong tones.
- AI Hub results are **Snapdragon 8 Elite Gen 5 QRD** data. They are not Nubia,
  AR1, or glasses measurements, and placement is not audio quality.

## Method

Harness: [`scripts/tts_research_20260726/`](../../../../scripts/tts_research_20260726/).
Every candidate runs through the same code path — one engine per process, one
resident model, two threads, the whole process tree sampled against a 2.5 GiB
cap. Per case (5 repeats): load time and incremental RSS, first playable chunk,
total synthesis, RTF, waveform gates, per-phrase WAV SHA-256, and a
clause-streamed playback supply simulation using the shipped 140 ms startup
cushion at 1.0×/1.5×/2.0× host slowdown. Short-chunk floor sweeps probe the
minimum first-flush length.

Waveform gates fail closed on: empty output, non-finite PCM, near-silence,
clipping, severe DC offset, long internal silence, and repeated-window
collapse. **A waveform that fails a hard gate is never written out as a WAV**, so
no later stage can mistake invalid PCM for audio.

The gates, the playback simulation, and the ASR normalization each have
deterministic tests (`scripts/tts_research_20260726/test_harness.py`, 21 cases,
all passing) built from the defects actually observed in this run, so a
regression in the scoring path cannot silently change a recommendation.

### Two scoring corrections that changed the conclusions

**The ASR judge's inverse text normalization was being counted as synthesis
error.** faster-whisper writes spoken numerals as digits and freely emits
traditional Han for simplified input. The Mandarin numbers case scored an
identical 0.609 CER for MeloTTS, Matcha, Piper, Kokoro *and* eSpeak — the
signature of a scoring artifact, not a synthesis defect. The judge now folds
traditional to simplified Han, and cases where it rendered digits are excluded
from the headline micro rate as a **union across all engines**, so every
candidate is still compared on identical material. Raw transcripts are kept and
rescoring is a separate pass, so a normalization change can never silently alter
recognition.

**The 0.9 s first-chunk gate was being scored against a four-sentence
paragraph** submitted as a single blob — a case the phone never produces, since
the orchestrator flushes at a sentence, punctuation boundary, or ~50–70
characters. The gate now uses the streaming-kind cases; the paragraph case is
reported separately as `first_chunk_p95_worst_case_s`, which is the number that
tells the integrator to bound the first flush.

**Judge calibration.** Against human-quality macOS-TTS reference audio the judge
scores 0.000 CER (Mandarin) and 0.045 CER / 0.250 WER (Russian). The Russian
floor is not zero, so Russian WER differences under about 5 points are noise.

## Mandarin

| Candidate | First chunk P95 | Worst-case first chunk | RTF median | RTF P95 | micro-CER | Waveform gate | Gaps @1.5× | Footprint | Peak RSS | Licence | Screen |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| `matcha_zh_baker` | **0.16 s** | 0.39 s | **0.146** | **0.146** | **4.0%** | pass | none | 139 MiB | 287 MiB | Baker corpus, **non-commercial** | conditional |
| `piper_zh_huayan` | 0.15 s | 0.75 s | 0.091 | 0.158 | 9.6% | pass | none | 77 MiB | 323 MiB | corpus **unknown** | **blocked** |
| `melotts_zh` | 0.99 s | 3.74 s | 0.715 | **0.942** | 17.6% | pass | **5.06 s gap** | 182 MiB | 482 MiB | **MIT** | conditional |
| `kokoro_zh` | 2.65 s | 5.08 s | 1.385 | 1.412 | 8.9% | **FAIL** | **6.53 s gap** | 199 MiB | 519 MiB | Apache-2.0 | **blocked** |
| `espeak_ng` | 0.02 s | 0.04 s | 0.025 | 0.028 | **55.2%** | pass | none | 0 | 76 MiB | GPL-3.0 | **blocked** |

RTF P95 is the number a streaming budget is actually set by, and it is where
MeloTTS separates from the field: its median 0.715 is marginally over the 0.70
gate, but its P95 reaches **0.942** and its worst observed run 0.982 — that is,
on slow runs it synthesises at essentially real time and cannot get ahead of
playback at all. Matcha is flat at 0.146 across every case and repeat.

Full data: [`evidence/screen_summary.md`](evidence/screen_summary.md),
[`evidence/bench_summaries.json`](evidence/bench_summaries.json),
[`evidence/asr_backcheck_zh.json`](evidence/asr_backcheck_zh.json).

### Kokoro-ZH: the short-input defect, now deterministic

The previous report recorded "10/31 Mandarin calls" producing invalid PCM with a
provisional three-Han-character floor. At speaker id 3 (`zf_001`, the Chinese
voice) the floor sweep makes it exact:

| Input | Han chars | Finite PCM |
|---|---:|---|
| `好` | 1 | **0/5** |
| `好的` | 2 | **0/5** |
| `好的。` | 2 | **0/5** |
| `你好啊` | 3 | 5/5 |
| `这是测试` | 4 | 5/5 |
| `请稍等一下` | 5 | 5/5 |
| `北京今天很晴朗` | 7 | 5/5 |

Fewer than three Han characters produced **all-sample** non-finite PCM in 15/15
calls; every input with three or more was finite in 20/20. The suite case
`好的。` failed 5/5 identically. Kokoro also carries a sustained +0.021 to
+0.029 DC offset (2.1–2.9% of full scale, 0.39–0.45× its own RMS) where every
other candidate measured between 0.00006 and 0.0024.

Kokoro stays rejected — and would stay rejected even with a chunk-length guard
and a DC blocker, because 1.385 RTF and a 2.65 s first chunk miss the latency
gates outright, and its playback queue starves by 3.6 s at *1.0×* host speed.

### MeloTTS-ZH: MIT, NPU-proven, and too slow on CPU

MeloTTS is the strategically valuable Mandarin candidate — MIT-licensed, the
only priority-1 Mandarin option that needs no GPL-3.0 eSpeak data at runtime,
and now the only one with a complete exact-SM8850/HTP-v81 chain: real Mandarin
text through BERT, encoder, flow, and nine repeated decoder calls, finite audio
at 2.94% NRMSE / 0.99958 cosine against the host golden, exact sentence recovery
at 0.0000 CER. Full detail in [`NPU_STATUS.md`](NPU_STATUS.md).

But on CPU it fails the streaming requirement, and not marginally:

- 0.715 RTF against a 0.70 gate;
- 0.99 s first chunk P95 on streaming cases, 3.74 s worst case;
- the playback simulation produces **two gaps totalling 0.429 s at 1.0× host
  speed**, before any mobile slowdown, growing to 5.06 s at 1.5×.

It also has its own short-input floor: single-character input (`好`) produced
near-silent output (RMS 0.0029–0.0047) in 5/5 calls. And its output level is low
across the board — peak 0.05–0.41, RMS 0.010–0.071, roughly 10–20 dB below Piper
and Matcha — so any integration needs makeup gain plus a two-character minimum
first chunk.

### Why the recommended Mandarin voice has a licence problem

The two technically strongest Mandarin candidates both have corpus licence
issues, and this is not incidental:

- `matcha_zh_baker` — the data-baker corpus is explicitly non-commercial.
- `piper_zh_huayan` — the Piper model card records the HuaYan_TTS corpus licence
  as **Unknown**, which the acceptance gates treat as a hard failure.

There is also a packaging distinction worth recording. Piper voices and Kokoro
require `espeak-ng-data` at runtime for phonemisation, which is **GPL-3.0**.
MeloTTS-ZH, Matcha-ZH, and Supertonic do not — they phonemise from a bundled
lexicon/dictionary or their own unicode indexer. For a Mandarin route this makes
MeloTTS-ZH the only priority-1 candidate that is simultaneously MIT and
eSpeak-free.

**The resulting recommendation is therefore conditional on intent.** For a
research demo, Matcha-ZH is the right voice today. For anything that ships
commercially, neither Matcha nor Piper-ZH is usable as-is, and MeloTTS-ZH on the
NPU is the only path that is both licence-clean and fast enough — which is
exactly why the NPU work below matters more for Mandarin than for Russian.

## Russian

| Candidate | First chunk P95 | Worst-case first chunk | RTF median | RTF P95 | micro-CER | micro-WER | Gaps @2.0× | Footprint | DNSMOS OVRL | Licence | Screen |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `piper_ru_dmitri` | **0.13 s** | 0.55 s | 0.130 | 0.142 | 4.7% | 13.3% | none | **77 MiB** | 2.981 | **CC0 corpus** | **clear** |
| `piper_ru_irina` | 0.23 s | 0.52 s | 0.109 | 0.112 | **1.4%** | **5.0%** | none | 77 MiB | 3.283 | corpus **unknown** | **blocked** |
| `piper_ru_ruslan` | 0.14 s | 0.63 s | 0.093 | 0.151 | 5.0% | 18.3% | none | 77 MiB | **3.318** | CC BY-NC-SA 4.0 | conditional |
| `supertonic3` @4 steps | 0.47 s | **3.62 s** | 0.286 | 0.289 | 2.5% | 8.3% | none | 139 MiB | 3.188 | OpenRAIL-M | conditional |
| `supertonic3` @5 steps | 0.93 s | **4.52 s** | 0.427 | 0.489 | 0.3% | 1.7% | none | 139 MiB | 3.305 | OpenRAIL-M | conditional |
| `espeak_ng` | 0.02 s | 0.04 s | 0.028 | 0.032 | 30.1% | **61.7%** | none | 0 | **2.106** | GPL-3.0 | **blocked** |

The worst-case column matters more for Russian than the headline. Supertonic
chunks a four-sentence answer into **2** callbacks where Piper produces **4**,
so its first chunk on an unbounded paragraph is 3.62 s (four steps) or 4.52 s
(five steps) against roughly 0.55 s for any Piper voice. Neither is a defect —
both keep the playback queue gap-free at 2.0× host slowdown once playing — but
it means **the Supertonic route depends on the orchestrator bounding the first
flush**, while the Piper route degrades gracefully if it does not.

### Supertonic 4 versus 5 steps — corrected to five steps

> **This section was wrong in the first version of this report and is
> corrected here.** It concluded "four steps is the correct setting" because the
> five-step first-chunk P95 measured 0.93 s, apparently breaching the 0.9 s gate.
> That P95 came from a single outlier run (0.998 s against a 0.629–0.680 s
> cluster) in a five-repeat sample, where P95 is essentially the maximum. A
> 15-repeat re-measurement is tight and much lower.

| Setting | First chunk, short clause (n=15) | micro-CER | micro-WER | RTF median | DNSMOS OVRL |
|---|---|---:|---:|---:|---:|
| 4 steps | median 0.436 s, **P95 0.437 s** | 2.5% | 8.3% | 0.182–0.287 | 3.188 |
| 5 steps | median 0.528 s, **P95 0.529 s** | **0.3%** | **1.7%** | 0.221–0.350 | **3.305** |

Five steps clears the 0.9 s gate with a 0.37 s margin, costs about 90 ms more to
first chunk, and is better on **every** quality signal available: five times
lower WER, eight times lower CER, and a higher proxy score. **Five steps is the
correct setting**, under the original < 0.9 s gate as well as under a relaxed
one.

The general lesson is worth keeping: with n=5, P95 is the maximum, and one
scheduling hiccup on a shared host can move a recommendation. Latency claims
that decide a setting should be measured with enough repeats that the P95 is
stable — here the 15-repeat spread is 0.436–0.438 s.

See [`RELAXED_BUDGET_ADDENDUM.md`](RELAXED_BUDGET_ADDENDUM.md) for the 6- and
8-step extension and the flush-length budget that goes with each.

### Russian licensing, and why it drives the recommendation

The Russian ranking by measurement alone would put `piper_ru_irina` first — best
CER, best WER, second-best proxy score. Its corpus licence is **Unknown**, which
is a hard failure, so it cannot be recommended.

- `piper_ru_dmitri` — Nabu Casa voice-datasets, **CC0**. The cleanest licence
  found for any compact Russian voice, and the only "clear" screen verdict in the
  whole run. Its weakness is that it scores lowest of the neural Russian voices
  on the DNSMOS proxy (2.981), which is a hint — not proof — that it may be the
  least pleasant of them.
- `piper_ru_ruslan` — RUSLAN corpus, CC BY-NC-SA 4.0. Non-commercial.
- `supertonic3` — OpenRAIL-M. Use restrictions apply but redistribution is
  permitted, and it is already the validated multilingual fallback in this
  project, covering French, Italian, Spanish, and English from the same 139 MiB.

**Silero was excluded on licence, not on quality.** The upstream repository
states its models are published under CC-NC-BY except the `base` cis-tts models,
which are MIT — so `v4_ru`/`v5_ru` are non-commercial. This study required
well-licensed alternatives, so no Silero weights were downloaded.

**RHVoice is covered indirectly and deliberately.** Piper `ru_RU-irina` is
finetuned on the RHVoice corpus, so it carries that voice data into the
comparison under a neural synthesiser that should sound strictly better than
RHVoice's own engine. Building GPL-3.0 RHVoice on macOS to evaluate a weaker
synthesis of the same corpus was judged a poor use of the run.

**Why Supertonic is primary despite Piper being faster and smaller.** Supertonic
already covers Russian, English, French, Italian, and Spanish from one 139 MiB
model that is already integrated and already has exact-QRD component profiles.
Piper needs one 77 MiB voice *per language*, plus GPL-3.0 eSpeak data. For a
single language Piper wins on every axis; for the five-language fallback the
comparison reverses. `piper_ru_dmitri` is recommended as the fallback precisely
because it is the licence-safe escape hatch if OpenRAIL-M becomes a problem.

## English, French, Italian, Spanish

No change is recommended. The validated English Expressive-LFM route stays the
default and was not re-benchmarked here. Supertonic at four steps was measured
for coverage: RTF 0.228 (en), 0.181 (fr), 0.189 (it), 0.180 (es), all
gate-passing with no simulated playback gap at 2.0× slowdown. eSpeak remains the
emergency control only.

## Footprint, memory, and licence table

| Candidate | Shipped weights | Peak RSS | Load | Sample rate | Runtime deps | Weights licence | Risk |
|---|---:|---:|---:|---:|---|---|---|
| `matcha_zh_baker` | 87.5 MiB + 51.4 MiB shared vocoder | 287 MiB | 1.44 s | 22.05 kHz | lexicon + jieba dict | Baker corpus, non-commercial | restricted |
| `melotts_zh` | 182.4 MiB | 482 MiB | 2.52 s | 44.1 kHz | lexicon + jieba dict | MIT | clear |
| `piper_zh_huayan` | 77.4 MiB | 323 MiB | 1.01 s | 22.05 kHz | **espeak-ng-data (GPL-3.0)** | corpus unknown | unclear |
| `kokoro_zh` | 199.3 MiB | 519 MiB | 1.43 s | 24 kHz | **espeak-ng-data (GPL-3.0)** | Apache-2.0 | clear |
| `supertonic3` | 138.6 MiB | 472 MiB | 0.74 s | 44.1 kHz | own unicode indexer | OpenRAIL-M | restricted |
| `piper_ru_dmitri` | 77.4 MiB | 305 MiB | 0.75 s | 22.05 kHz | **espeak-ng-data (GPL-3.0)** | CC0 corpus | clear |
| `piper_ru_ruslan` | 77.4 MiB | 406 MiB | 0.71 s | 22.05 kHz | **espeak-ng-data (GPL-3.0)** | CC BY-NC-SA 4.0 | restricted |
| `piper_ru_irina` | 77.4 MiB | 416 MiB | 0.70 s | 22.05 kHz | **espeak-ng-data (GPL-3.0)** | corpus unknown | unclear |
| `espeak_ng` | system engine | 76 MiB | 0.01 s | 22.05 kHz | — | GPL-3.0 | restricted |

Peak RSS is the whole process tree during synthesis, including the ONNX runtime.
Every run stayed inside the 2.5 GiB cap; the highest value observed anywhere was
519 MiB. **This is a standalone-process figure.** It does not include LFM, Qwen,
or SenseVoice being resident at the same time, which is an open phone gate.

## Blinded listening bundle

The private short-form bundle contains 10 Mandarin clips (5 engines × 2 cases)
and 12 Russian clips (6 engines × 2 cases).

A larger bundle including the four-sentence long-form cases (15 Mandarin and 18
Russian clips) is kept in the external run root. Neither bundle is committed:
some candidate voices have restricted or unclear output-redistribution terms,
and publishing the clip mapping would also defeat the pending blind test.

Clips are loudness-equalized to −24 dBFS RMS with a 0.95 peak ceiling, shuffled
with a fixed seed, and renamed to opaque ids. `key.json` holds the unblinding map
and the exact gain applied to each clip.

Loudness equalization is not cosmetic. The candidates differ by more than 20 dB
in output level — MeloTTS peaks at 0.05–0.41 where Piper peaks near 0.5 — and an
unequalized preference test would mostly measure gain. The target was moved from
−20 to −24 dBFS after verification: at −20 dBFS several high-crest-factor Russian
clips hit the peak ceiling before reaching the target, leaving up to 3.3 dB of
residual spread (Supertonic averaged −21.65 dBFS against Piper-irina at −20.00)
in exactly the comparison the bundle exists to make fair. At −24 dBFS every clip
in both languages lands at −24.00 dBFS with peak ≤ 0.877.

Sample rate is deliberately **not** equalized, because it is a real property of
what would ship; the key records it as a known confound (MeloTTS and Supertonic
at 44.1 kHz will sound fuller than the 22.05 kHz voices for reasons unrelated to
model quality).

### Automatic proxy scores (ranking aid only)

DNSMOS P.835/P.808 over the blinded, loudness-equalized clips. Aggregate means
are reproduced below; the clip-level file is kept private because it maps blind
identifiers back to engines.

Means over the 15 Mandarin and 18 Russian clips of the full external bundle
(3 cases per engine):

| Mandarin | OVRL | P.808 | | Russian | OVRL | P.808 |
|---|---:|---:|---|---|---:|---:|
| `piper_zh_huayan` | 3.305 | 3.953 | | `piper_ru_ruslan` | 3.318 | 3.974 |
| `melotts_zh` | 3.226 | 3.686 | | `supertonic3_s5` | 3.305 | 3.684 |
| `matcha_zh_baker` | 3.182 | 3.856 | | `piper_ru_irina` | 3.283 | 3.792 |
| `kokoro_zh` | 3.114 | 3.332 | | `supertonic3_s4` | 3.188 | 3.591 |
| `espeak_ng` | 2.894 | 3.693 | | `piper_ru_dmitri` | 2.981 | 3.725 |
| | | | | `espeak_ng` | **2.106** | **2.855** |

Read these carefully. In Russian the proxy separates eSpeak decisively (2.106 vs
2.98–3.32) and agrees with every other signal; it also puts Supertonic at five
steps above four (3.305 vs 3.188), which is the one piece of evidence favouring
five steps on grounds other than intelligibility. In Mandarin the neural spread
is only 3.11–3.31 — inside the noise — and the proxy ranks eSpeak *above* Kokoro
on P.808 while ASR puts eSpeak at 55.2% CER and Kokoro at 8.9%. DNSMOS hears
artifacts, not wrong tones. Use it to flag a buzzy candidate, never to choose
between two clean ones.

The one place this proxy is uncomfortable for the recommendation: `piper_ru_dmitri`,
the only licence-clean "clear" candidate, scores lowest of the neural Russian
voices (2.981 OVRL). That is a hint, not a verdict — but it is the specific thing
a human listener should check first.

## Reproduction

```bash
export TTS_RESEARCH_MODEL_ROOT=/external/models RUN_ROOT=/external/run-root
export VENV_PY="$RUN_ROOT/venv-tts/bin/python"
tmux new-session -d -s tts-matrix "bash scripts/tts_research_20260726/run_matrix.sh > $RUN_ROOT/logs/matrix.log 2>&1"
```

Then the ASR back-check, screen table, blinded bundle, and proxy scores as
documented in
[`scripts/tts_research_20260726/README.md`](../../../../scripts/tts_research_20260726/README.md).

Model archives are pinned sherpa-onnx `tts-models` releases. SHA-256 digests
and archive path/link audits are generated during preflight; the
machine-specific preflight manifest remains outside the public tree. No
weights, raw logs, signed URLs, or credentials are stored in this repository.
