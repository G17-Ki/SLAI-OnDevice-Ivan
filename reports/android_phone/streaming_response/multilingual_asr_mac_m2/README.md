# Multilingual ASR front end: Whisper Tiny vs Base vs Small

Date: 2026-07-22<br>
Host: Apple M2 MacBook Air, 8 GB RAM, macOS 15.4.1<br>
Scope: Mac host evidence plus published Qualcomm HTP-v81 scorecards; this is not a phone measurement.

## Decision

Use **Whisper Small with a fixed or cached user locale** for the first
quality-oriented multilingual phone demo. Keep Whisper Base as the smaller
plumbing and memory fallback. Reject Whisper Tiny despite its lower latency.

This is a two-level decision:

- Base passes the feasibility role: it is fast enough, has an official all-NPU
  Qualcomm export, and is useful for end-to-end integration.
- Small is the first tested checkpoint to pass the exploratory Latin-language
  quality gate. It cuts Latin WER from 36.1% to 18.6% and the seven-language
  macro primary error from 38.0% to 24.4%.
- Small still does **not** pass the all-language gate: JA/KO/ZH micro CER is
  29.7%, driven by Japanese at 37.5% and Mandarin at 48.5%. A stronger or
  language-specialized recognizer is still needed before claiming reliable
  open-ended Japanese and Mandarin support.

The practical implication for the 2–3 second speech-to-speech target is good:
ASR does not need to be the bottleneck on HTP v81. Small took 1.69 seconds
median / 2.38 seconds P95 on the Mac CPU, but Qualcomm's published v81 component
times imply about 205 ms of float-Small graph compute for a 20-token utterance
and 426 ms for 50 tokens. W8A16 Small is about 254 ms and 373 ms respectively.
Those figures exclude host-side preprocessing and runtime overhead, so the phone
profile is decisive, but they leave a credible path to the 2–3 second total
speech-to-speech target.

## What was measured

The fixture is the public
[`argmaxinc/common_voice_17_0-debug-zip`](https://huggingface.co/datasets/argmaxinc/common_voice_17_0-debug-zip/tree/e77e410d41a77f733efbaf7307bd0ff7ebd6d0af)
repository pinned at commit
`e77e410d41a77f733efbaf7307bd0ff7ebd6d0af`. Its metadata declares
CC0-1.0. It contains exactly three Common Voice 17 clips per language. This run
used all clips for:

- English, Spanish, German, and French;
- Japanese, Korean, and Mandarin Chinese.

That is 21 clips and 102.78 seconds of real speech. Exact IDs, references,
durations, and audio SHA-256 values are in
[`dataset_manifest.json`](dataset_manifest.json).

The main latency profile was:

- Faster-Whisper 1.2.1 / CTranslate2 4.8.1;
- INT8 CPU inference, four threads, one worker;
- audio pre-decoded to 16 kHz PCM before timing, matching microphone input;
- model kept warm and resident;
- greedy decoding, no temperature fallback, no timestamps;
- three timed repeats per clip and mode;
- fixed-language and automatic-language-detection modes for Tiny/Base;
- fixed-language mode for the bounded Small extension.

`first_segment_ms` measures the first stable segment exposed by
Faster-Whisper. On these short utterances it is effectively the final
transcript; it is not internal decoder first-token latency.

## Main results

| Model | Mode | Stable transcript median / P95 | Median RTF | Model files | Incremental peak RSS | Latin micro WER | JA/KO/ZH micro CER | Language ID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Whisper Tiny | fixed locale | 217 / 395 ms | 0.054 | 74.6 MiB | 276.3 MiB | 91.8% | 48.0% | n/a |
| Whisper Tiny | auto | 419 / 839 ms | 0.106 | 74.6 MiB | 276.3 MiB | 91.8% | 62.8% | 85.7% |
| Whisper Base | fixed locale | 419 / 507 ms | 0.097 | 141.0 MiB | 315.2 MiB | 36.1% | 39.2% | n/a |
| Whisper Base | auto | 765 / 836 ms | 0.178 | 141.0 MiB | 315.2 MiB | 36.1% | 42.6% | 95.2% |
| Whisper Small | fixed locale | 1,687 / 2,382 ms | 0.384 | 463.7 MiB | 614.0 MiB | 18.6% | 29.7% | n/a |

Lower is better for every error and latency column. RTF below 1.0 means faster
than real time.

Tiny is not a safe baseline. Its French output entered a long repetition loop,
German was frequently malformed, and its auto language detector selected the
wrong language for one Japanese, one Korean, and one Chinese clip. Base removes
the catastrophic repetition and improves the aggregate primary error rate from
65.0% to 38.0% in fixed-locale mode. Small improves it again to 24.4%, at a
large host CPU and memory cost.

### Fixed-locale quality by language

| Language | Tiny primary error | Base primary error | Small primary error | Primary metric |
|---|---:|---:|---:|---:|
| English | 26.3% | 36.8% | 21.1% | WER |
| Spanish | 23.8% | 19.0% | 19.0% | WER |
| German | 61.9% | 38.1% | 4.8% | WER |
| French | 183.3% | 44.4% | 25.0% | WER |
| Japanese | 58.3% | 47.9% | 37.5% | CER |
| Korean | 28.4% | 25.4% | 14.9% | CER |
| Mandarin Chinese | 72.7% | 54.5% | 48.5% | CER |

The three-sample language cells are diagnostic, not statistically powered.
The transparent normalizer applies Unicode NFKC, case folding, punctuation
removal, and whitespace normalization. It does not expand numbers, so `six`
and `6` count as different. Full references and hypotheses are in
[`transcripts.csv`](transcripts.csv).

### Exploratory acceptance gates

The decision uses explicit gates so “acceptable” does not simply mean “better
than Tiny”:

- Latin-language micro WER: at most 25%;
- Japanese/Korean/Mandarin micro CER: at most 25%, with no individual language
  above 40%;
- eventual phone VAD-close → final transcript: at most 700 ms P95;
- no decoder repetition or unbounded output on any acceptance utterance.

Tiny fails quality. Base fails both quality aggregates. Small passes the Latin
gate and the 40% cap for Japanese and Korean, but fails the combined character
gate and Mandarin cap. Therefore Small is the smallest **quality-oriented
candidate**, while no tested checkpoint is yet accepted for all seven
languages. Phone latency remains unclassified until the QRD artifacts run on
the Nubia.

### Beam search did not solve the quality gap

A second fixed-locale run used beam size five, timestamp termination, and the
standard Whisper fallback thresholds.

| Model | Median / P95 | Latin WER | JA/KO/ZH CER | Incremental peak RSS |
|---|---:|---:|---:|---:|
| Tiny beam-5 | 291 / 1,167 ms | 69.1% | 45.3% | 257.5 MiB |
| Base beam-5 | 555 / 828 ms | 34.0% | 48.0% | 423.5 MiB |

Beam search helped Tiny's Latin aggregate but did not make it acceptable. Base
gained only two WER points and became worse on the character-scored languages.
One very short Mandarin clip took 1.72 seconds and hallucinated unrelated text.
The next quality experiment should therefore increase model capacity rather
than spending more latency on beam search.

## Language mode recommendation

Prefer a selected locale in settings and cache it across turns. This has three
benefits:

1. It nearly halves measured host latency.
2. It prevents short or ambiguous commands from switching decoder language.
3. It makes per-language prompt rules and downstream TTS voice selection
   deterministic.

If automatic detection is required, run it once at session start or only after
a confidence failure, rather than on every utterance. The Mac auto-mode overhead
is partly implementation-specific: Faster-Whisper encodes once for language
detection and then encodes again for transcription. A Qualcomm integration can
reuse the encoder cross-attention cache and obtain the language token from the
same decoder sequence, so do not directly project the Mac auto-mode latency onto
HTP v81. The observed short-utterance classification failures remain relevant.

## Qualcomm HTP-v81 mapping

`qai-hub-models==0.58.0` publishes Tiny, Base, Small, and W8A16 Small as
fixed-shape, file-based models with language detection. Their capability metadata explicitly says
`streaming: false`. The helper named `stream()` merely buffers audio chunks and
runs the file model repeatedly; it is not an incremental streaming encoder.

Both exports use:

- 16 kHz audio and a fixed 30-second `1 x 80 x 3000` mel input;
- an encoder that emits cross-attention caches;
- a one-token decoder with fixed self-attention caches for up to 200 tokens;
- QNN float compilation that converts the graph and I/O to FP16;
- separate encoder and decoder QNN context binaries.

Published Samsung Galaxy S26 (`sm8850-ad`, Hexagon v81) QNN-context results:

| Model/component | Job | Latency | Placement |
|---|---:|---:|---:|
| Tiny encoder | `j5qmrqdog` | 12.510 ms | 294 / 294 NPU layers |
| Tiny decoder, per token | `jgl12m9m5` | 1.498 ms | 509 / 509 NPU layers |
| Base encoder | `jp840o085` | 22.028 ms | 556 / 556 NPU layers |
| Base decoder, per token | `jgk97z7o5` | 2.657 ms | 975 / 975 NPU layers |
| Small float encoder | `jpynlm7lg` | 58.128 ms | 1,582 / 1,582 NPU layers |
| Small float decoder, per token | `jp0kw6vn5` | 7.365 ms | 2,277 / 2,277 NPU layers |
| Small W8A16 encoder | `j5770qyr5` | 175.083 ms | 2,014 / 2,014 NPU layers |
| Small W8A16 decoder, per token | `jp49kzll5` | 3.961 ms | 2,853 / 2,853 NPU layers |

All eight jobs used QAIRT `2.45.0.260326154327`. The package also lists
`Snapdragon 8 Elite Gen 5 QRD` (`sm8850`, Hexagon v81) as supported. Do not put
the published S26 binary directly on a non-Samsung SM8850 phone: submit a fresh
QRD compile and then test runtime compatibility on the Nubia device.

Exact Tiny/Base tensor shapes, Small job data, and derived token-count estimates are
in [`qai_hub_v81_reference.json`](qai_hub_v81_reference.json).

## What to deploy next

1. Compile **Whisper Small W8A16** for `Snapdragon 8 Elite Gen 5 QRD` as the
   quality candidate, and compile **Whisper Base float/FP16** as the lower-memory
   fallback. Both should use QNN context binaries.
2. Keep the ASR model resident. Measured cold load was 845 ms for Small and
   306 ms for Base; paying that on each turn is unnecessary.
3. Use fixed locale, greedy decode, an explicit repetition/EOS guard, and a
   short maximum command length. The Tiny French failure demonstrates why the
   guard is mandatory for every checkpoint.
4. Measure phone VAD-close → final transcript and full-process RSS while LFM and
   TTS remain loaded. Use a P95 ASR gate of 700 ms for the first demo.
5. Compare QRD/phone Small outputs against the local Small transcript goldens.
   The graph rewrite and W8A16 quantization need an explicit numerical and text
   regression check; the Mac CTranslate2 result does not prove QNN quality.
6. If the NPU port is blocked, use Base CPU as the temporary demo path. Small's
   Mac CPU P95 was 2.38 seconds, which consumes almost the whole response budget
   before the LLM or TTS begins.
7. Consider Large-v3-Turbo or language-specialized Japanese/Mandarin ASR if
   Small still misses the quality gate; Large-v3-Turbo's
   published v81 estimate is about 393 ms for 20 tokens before host overhead and
   its resident footprint will compete directly with LFM and TTS.

For a larger acceptance run, use at least 100 utterances per supported language,
include the professor's command vocabulary and room noise, and require roughly
20–25% or lower primary error before presenting open-ended multilingual ASR as
reliable. The 21-clip evidence here is sufficient to choose the first engineering
target, not to make a product-quality claim.

## Reproduction and raw artifacts

- Reproduction harness:
  [`scripts/multilingual_asr_mac_m2`](../../../../scripts/multilingual_asr_mac_m2/README.md)
- Aggregate table: [`summary.csv`](summary.csv)
- Per-language latency: [`latency_by_language.csv`](latency_by_language.csv)
- References and hypotheses: [`transcripts.csv`](transcripts.csv)
- Tiny raw result: [`whisper_tiny.json`](whisper_tiny.json)
- Base raw result: [`whisper_base.json`](whisper_base.json)
- Small raw result: [`whisper_small.json`](whisper_small.json)
- Beam-5 raw results:
  [`whisper_tiny_quality_beam5.json`](whisper_tiny_quality_beam5.json),
  [`whisper_base_quality_beam5.json`](whisper_base_quality_beam5.json)
- Failed data/download routes: [`failures.md`](failures.md)
