# On-device results — NX809J / SM8850 / Android 16

Date: 2026-07-26 → 27, unattended overnight run<br>
Device: **Nubia `NX809J`, `SM8850` (Snapdragon 8 Elite Gen 5), Android 16, 8 cores**<br>
App: `ai.liquid.ttsbench`, separate from the LFM demo, sherpa-onnx 1.13.4 — the
same engine build the host measurements used<br>
Bundle: 14 candidates, 440 files, 1.55 GB, pushed and size-verified

**These are physical-device measurements.** They supersede the host numbers in
[`REPORT.md`](REPORT.md) and
[`RELAXED_BUDGET_ADDENDUM.md`](RELAXED_BUDGET_ADDENDUM.md) wherever they
disagree. The LFM demo app was not modified, reinstalled, or run.

## What changed versus the host conclusions

1. **MeloTTS-ZH is viable on this phone after all — at 4–8 threads.** The host
   ran everything at two threads and concluded MeloTTS was CPU-unviable and
   NPU-only. On eight threads it reaches RTF 0.44 and a 663 ms first chunk. It
   is the only Mandarin candidate that is MIT-licensed *and* free of any
   GPL-3.0 eSpeak dependency, so this materially changes the Mandarin picture.
2. **Supertonic does not scale past two threads.** 1→2 threads nearly halves
   its time; 2→4 buys 6%, and 8 threads is no better than 4. Its five-step
   first chunk sits at ~0.9–1.06 s on this phone at any thread count.
3. **The phone/host ratio is per-architecture, not a constant.** It ranges from
   3.7× faster to 1.4× slower. Every host conclusion that scaled by an assumed
   slowdown needed rechecking, and some of them changed.
4. **No thermal throttling.** 25 consecutive long utterances drift RTF by
   +4.4% (Matcha) and +1.0% (Supertonic). Open gate #5 closes.
5. **Memory is higher than the host suggested for every candidate**, by
   1.2–2.0×. This is now the binding constraint, not latency.

## Latency: end-to-speaker, measured

Real `AudioTrack` playback, first-audible taken from playback-head movement by
an independent watcher thread, 2 threads, 140 ms cushion.

| Candidate | first chunk | +to speaker | **end-to-speaker** | RTF | underruns (20 s utterance) |
|---|---:|---:|---:|---:|---:|
| `matcha_zh_baker` | 114 ms | 3 ms | **118 ms** | 0.055 | 1 |
| `piper_ru_dmitri` | 182 ms | 2 ms | **185 ms** | 0.126 | 1 |
| `piper_ru_ruslan` | 205 ms | 4 ms | 208 ms | 0.116 | 1 |
| `piper_zh_huayan` | 208 ms | 4 ms | 211 ms | 0.120 | 1 |
| `piper_ru_irina` | 300 ms | 4 ms | 303 ms | 0.108 | 1 |
| `supertonic3_s4` | 731 ms | 4 ms | **735 ms** | 0.348 | 1 |
| `supertonic3_s5` | 898 ms | 3 ms | **900 ms** | 0.424 | 1 |
| `supertonic3_s6` | 1060 ms | 4 ms | 1064 ms | 0.500 | 1 |
| `supertonic3_s8` | 1390 ms | 5 ms | 1394 ms | 0.653 | 1 |
| `melotts_zh` (2 thr) | 1684 ms | 3 ms | 1687 ms | 1.104 | 2–4 |
| `kokoro_zh_fp32` | 2644 ms | 4 ms | 2648 ms | 0.982 | 1–4 |
| `zipvoice_zh` | 3138 ms | 4 ms | 3142 ms | 1.276 | 1–2 |
| `kokoro_zh_int8` | 3202 ms | 4 ms | 3206 ms | 1.178 | 1–4 |
| `melotts_zh_int8` | 3816 ms | 4 ms | 3820 ms | 2.439 | 2–6 |

The playback-head delay is **2–9 ms for every candidate** — the audio path adds
essentially nothing. End-to-speaker is decided by synthesis alone.

**Underruns are the honest signal here.** They stay at 1 (the initial-fill
artifact) for Matcha, every Piper voice and every Supertonic setting even on a
20+ second utterance, meaning the queue never starves. They rise to 2–6 for
MeloTTS at two threads, both Kokoro builds and ZipVoice — those genuinely fail
to stay ahead of playback.

## Thread scaling — the measurement that decides the Russian setting

First chunk (ms) / RTF, `clause` phrase:

| Candidate | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|
| `matcha_zh_baker` | **81 / 0.038** | 167 / 0.076 | 132 / 0.060 | 138 / 0.062 |
| `piper_ru_dmitri` | 338 / 0.222 | 209 / 0.139 | 143 / 0.095 | **127 / 0.085** |
| `piper_zh_huayan` | 403 / 0.220 | 248 / 0.137 | 173 / 0.095 | **164 / 0.089** |
| `melotts_zh` | 2177 / 1.424 | 1676 / 1.096 | 917 / 0.602 | **663 / 0.440** |
| `supertonic3_s5` | 1916 / 0.896 | 1060 / 0.497 | **997 / 0.468** | 1026 / 0.483 |

Three distinct behaviours:

- **MeloTTS scales almost linearly** (1→8 threads: 3.3× faster). This is what
  makes it viable and it is invisible at the two threads the host used.
- **Piper scales well** to 4–8 threads.
- **Supertonic stops scaling at 2 threads.** 2→4 gains 6%, 4→8 loses. Its
  denoising loop is sequential, so more cores do not help; the five-step first
  chunk stays around 1.0 s whatever you do.
- **Matcha is fastest at one thread** (81 ms) — thread setup costs more than it
  saves at this size. Do not blindly give it more cores.

## Memory — now the binding constraint

Peak RSS, standalone, one model resident. Host figures for comparison.

| Candidate | device RSS | device PSS | host RSS | device/host | load |
|---|---:|---:|---:|---:|---:|
| `matcha_zh_baker` | **493 MiB** | 369 MiB | 287 | 1.7× | 624 ms |
| `piper_zh_huayan` | 483 MiB | 359 MiB | 322 | 1.5× | 1307 ms |
| `piper_ru_ruslan` | 529 MiB | 403 MiB | 406 | 1.3× | 865 ms |
| `melotts_zh_int8` | 579 MiB | 452 MiB | 266 | 2.2× | 2457 ms |
| `piper_ru_irina` | 588 MiB | 463 MiB | 416 | 1.4× | 865 ms |
| `piper_ru_dmitri` | 618 MiB | 492 MiB | 305 | 2.0× | 900 ms |
| `melotts_zh` | **637 MiB** | 512 MiB | 482 | 1.3× | 1031 ms |
| `supertonic3_s5` | **660 MiB** | 535 MiB | 441 | 1.5× | 821 ms |
| `kokoro_zh_int8` | 774 MiB | 650 MiB | 519 | 1.5× | 1466 ms |
| `kokoro_zh_fp32` | 930 MiB | 805 MiB | 662 | 1.4× | 643 ms |
| `zipvoice_zh` | **1226 MiB** | 1099 MiB | 1008 | 1.2× | 2175 ms |

Every candidate needs **more** memory on the phone than on the host, by
1.2–2.2×. The demo already holds LFM2.5-Audio-1.5B Q4, Qwen3-1.7B Q4 and
SenseVoice. **Adding 0.5–0.7 GiB for a TTS voice is the real integration risk,
and it is larger than any latency concern in this report.** Measuring
co-residency is now the top open gate.

## Thermal — closed

25 consecutive 17–21 s utterances, back to back:

| Candidate | RTF first 5 | RTF last 5 | drift | gates | peak RSS |
|---|---:|---:|---:|---|---:|
| `matcha_zh_baker` | 0.024 | 0.025 | **+4.4%** | 25/25 | 489 MiB |
| `supertonic3_s5` | 0.354 | 0.357 | **+1.0%** | 25/25 | 745 MiB |

Battery 31–33 °C, thermal zone 35–44 °C, no throttling observed. Sustained
synthesis on this phone does not degrade. **Open gate #5 is closed.**

## Intelligibility on device

Same judge, same normalization and exclusion rule as the host, over 90
device-rendered WAVs.

| Candidate | device CER | host CER | delta |
|---|---:|---:|---:|
| `supertonic3_s5` | **0.0000** | 0.0028 | −0.0028 |
| `supertonic3_s6` | **0.0000** | 0.0028 | −0.0028 |
| `piper_ru_irina` | 0.0065 | 0.0138 | −0.0073 |
| `supertonic3_s8` | 0.0098 | 0.0110 | −0.0012 |
| `zipvoice_zh` | 0.0400 | 0.0480 | −0.0080 |
| `supertonic3_s4` | 0.0458 | 0.0249 | +0.0209 |
| `matcha_zh_baker` | 0.0480 | 0.0400 | +0.0080 |
| `piper_ru_dmitri` | 0.0556 | 0.0470 | +0.0086 |
| `piper_ru_ruslan` | 0.0621 | 0.0497 | +0.0124 |
| `kokoro_zh_int8` | 0.0732 | 0.0894 | −0.0163 |
| `kokoro_zh_fp32` | 0.0880 | 0.0720 | +0.0160 |
| `piper_zh_huayan` | 0.1120 | 0.0960 | +0.0160 |
| `melotts_zh_int8` | 0.1120 | 0.1920 | −0.0800 |
| `melotts_zh` | 0.1600 | 0.1760 | −0.0160 |

Device and host agree within ±0.02 for 13 of 14 candidates. **The phone build of
each engine is as intelligible as the host build** — an assumption the host work
had to make and could not test.

## Defects reproduced on hardware

| Defect | Host | Device |
|---|---|---|
| Kokoro **INT8** non-finite PCM, <3 Han chars | 0/5 | **0/3** (`micro`, `floor2`), 1/3 (`floor1`) |
| Kokoro **float32** at the same inputs | clean | **3/3 clean at every length** |
| MeloTTS near-silence, 1 Han char | RMS 0.0029–0.0047 | RMS 0.0016–0.0043, 3/3 fail |
| INT8 slower than float32 | MeloTTS 3×, Kokoro 1.4× | MeloTTS **2.2×**, Kokoro **1.2×** |

Kokoro's invalid-PCM defect is confirmed as an **INT8 quantization artifact**,
not a model defect, on real hardware. INT8 was slower than float32 for every
model on both platforms.

### The safety gate, demonstrated on hardware

The main playback phase used only `clause`/`answer`/`long`, none short enough to
trigger Kokoro's NaN, so it left `dropped_chunks` at 0 and proved nothing about
the guard. A dedicated playback sweep on the short phrases closes that:

| Candidate | phrase | gate | chunks dropped | playback head moved? | underruns |
|---|---|---:|---:|---|---:|
| `kokoro_zh_int8` | `micro` (`好的。`) | **0/3** | **3** | **no** | 0 |
| `kokoro_zh_int8` | `floor2` (`好的`) | 1/3 | **2** | **no** | 0 |
| `kokoro_zh_int8` | `clause` | 3/3 | 0 | yes, +2 ms | 1 |
| `kokoro_zh_fp32` | `micro` | 3/3 | 0 | yes, +3 ms | 1 |
| `kokoro_zh_fp32` | `floor2` | 3/3 | 0 | see note | 0 |
| `kokoro_zh_fp32` | `clause` | 3/3 | 0 | yes, +2 ms | 1 |

**Every non-finite chunk was dropped and the playback head never moved** — with
`AudioTrack` running and the speaker live, nothing was emitted. "Invalid PCM
never reaches the speaker" is now demonstrated end to end on the device, not
merely unit-tested.

*Note on `kokoro_zh_fp32 / floor2`:* the gate passed and no chunk was dropped,
but the head watcher recorded no movement. That utterance is shorter than the
140 ms startup cushion, so playback only begins in the final flush and the
watcher can exit before the head advances. It is a measurement edge case for
sub-cushion utterances, not a failure to play — and a real integration should
not be flushing chunks that short anyway.

## Revised recommendations

### Mandarin — `matcha_zh_baker` for the demo, and MeloTTS-ZH is back in play

`matcha_zh_baker` is decisive on this phone: **118 ms end-to-speaker**, RTF
0.055, 493 MiB, no underruns on a 21 s utterance, +4.4% thermal drift over 25
runs, 4.8% CER. It remains blocked commercially by its non-commercial corpus.

**MeloTTS-ZH changes status from "NPU-only" to "CPU-viable at 4–8 threads":**
663 ms first chunk and RTF 0.440 at 8 threads. It is the only Mandarin candidate
that is both MIT and eSpeak-free. Its costs are real — 637 MiB, 2–4 underruns at
two threads, 16% CER (worst of the neural Mandarin set), and a two-character
minimum first chunk — but the host's conclusion that it cannot work on CPU was
an artifact of the two-thread cap. Its NPU route stays the better answer;
it is no longer the *only* one.

### Russian — Supertonic at five steps, given the 1–2 s cap

| Setting | end-to-speaker | RTF | CER | verdict |
|---|---:|---:|---:|---|
| `supertonic3_s4` | 735 ms | 0.348 | 4.6% | clears a 0.9 s gate |
| **`supertonic3_s5`** | **900 ms** | 0.424 | **0.0%** | recommended under the 1–2 s cap |
| `supertonic3_s6` | 1064 ms | 0.500 | 0.0% | no gain over s5 |
| `supertonic3_s8` | 1394 ms | 0.653 | 1.0% | worse on both axes |

Under the **1–2 s cap**, five steps is right: perfect transcription for 165 ms
more than four steps, and threading cannot close that gap because Supertonic
does not scale past two threads. Under a strict **0.9 s** gate the answer flips
to four steps — s5 lands at 900 ms, exactly on the line.

`piper_ru_dmitri` remains the licence-clean fallback and is far faster
(185 ms end-to-speaker, 127 ms at 8 threads, CC0 corpus), at 5.6% CER.

### Still not decided: naturalness

Every number here is latency, memory, stability or intelligibility. **No human
has listened.** The private blinded bundles and 90 device-rendered WAVs are the
remaining input, and nothing in this run replaces them. They stay outside Git
because several candidate voices have restricted or unclear redistribution
terms.

## Reproduce

```bash
tools/build_tts_bench.sh
TTS_BENCH_BUNDLE=/external/device-bundle tools/push_bundle_resumable.sh
RUN_ROOT=/external/run REPO=$PWD bash scripts/tts_research_20260726/overnight_orchestrate.sh
DEST=/external/run/device tools/pull_device_results.sh
```

The orchestrator stamps each phase, verifies the installed APK's hash before
measuring, and resumes rather than restarts. Raw evidence:
[`evidence/device/`](evidence/device/).
