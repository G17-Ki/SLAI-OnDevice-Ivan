# Kokoro v1.1-zh INT8: bounded standalone TTS comparison

Date: 2026-07-23<br>
Evidence class: Apple Silicon host CPU benchmark only; not phone, Qualcomm NPU,
AR1, or glasses evidence.

## Decision

Do **not** replace Supertonic with Kokoro for English, and do not treat this
checkpoint as a general multilingual model. Keep it only as an experimental
Mandarin-capable CPU fallback candidate behind a mandatory finite-PCM guard,
then compare it on the phone against the existing MeloTTS Chinese route.

It is not demo-ready yet. The original very short Mandarin warm-up `你好。`
returned a 1.10-second buffer in which all 26,400 samples were non-finite. A
follow-up boundary sweep reproduced invalid PCM on 10 of 31 Mandarin calls,
including repeated calls after a discarded warm-up. All tested phrases with at
least three Han characters were finite in this small sweep, but the sample is
not broad enough to declare a universal safe length. An integration must
validate every chunk and merge or retry invalid or very short text.

Kokoro can fit the 2–3 second end-to-first-audio goal only if the text stream
is flushed into short clauses. With two CPU threads, short/factual utterances
reached first playable PCM in 1.07–1.64 seconds p95. A longer first clause
already took 2.12 seconds in English and 2.27 seconds in Mandarin before ASR,
LLM, playback cushion, or phone slowdown. The runtime callback is
sentence/chunk-level, not frame-level vocoder streaming.

For English, Supertonic remains decisively better on this host: its matching
short case reached 0.348 seconds median versus Kokoro's 1.318 seconds, while
using substantially less memory. Kokoro's advantages are Apache-2.0 source
model weights and native Mandarin support; Supertonic 3 does not support
Mandarin. Whole-archive redistribution still needs a third-party frontend-data
license audit.

## Scope and settings

- Model: Sherpa-ONNX dynamic-INT8 conversion of
  `hexgrad/Kokoro-82M-v1.1-zh`.
- Runtime: `sherpa-onnx==1.13.4`, CPU provider.
- Host: Apple M2, 8 GB RAM, macOS 15.4.1, Python 3.11.11.
- Two inference threads; one resident model; speed 1.0.
- One English and one Mandarin warm-up, then five measured repetitions per
  case.
- English speaker `af_maple` (`sid=0`); Mandarin speaker `zf_001` (`sid=3`).
- `max_num_sentences=1`, so punctuation yields independently playable
  callbacks.
- Output: mono 24 kHz PCM WAV.

Spanish and Japanese are not supported by this checkpoint. They were not
synthesized and were not silently counted as multilingual successes.

## Latency and real-time factor

| Case | First PCM median | First PCM p95 | Total median | Total p95 | Audio duration | Median RTF |
|---|---:|---:|---:|---:|---:|---:|
| English short | 1.318 s | 1.355 s | 1.318 s | 1.355 s | 1.519 s | 0.868 |
| Mandarin short | 1.073 s | 1.099 s | 2.282 s | 2.309 s | 2.466 s | 0.925 |
| English factual | 1.612 s | 1.640 s | 1.612 s | 1.640 s | 1.926 s | 0.837 |
| Mandarin factual | 1.546 s | 1.581 s | 1.546 s | 1.581 s | 1.862 s | 0.831 |
| English, three sentences | 2.080 s | 2.120 s | 5.231 s | 5.309 s | 6.485 s | 0.807 |
| Mandarin, three sentences | 2.263 s | 2.270 s | 5.799 s | 5.921 s | 7.594 s | 0.763 |

The Mandarin short case contains a comma, so its first callback carries only
the first short clause. This is the desired interaction pattern: play that
chunk immediately while the rest continues. The complete utterance still
takes roughly 2.3 seconds to synthesize.

The long cases remain faster than real time, but only narrowly enough that
concurrent LLM work, phone thermals, or slower mobile CPU execution could erase
the playback lead. Phone measurement is required before promising gap-free
speech.

## Critical Mandarin warm-up numerical failure

The English warm-up `Hello.` was valid. The immediately following Mandarin
warm-up `你好。` emitted its callback at 628.6 ms, but all 26,400 float samples
were non-finite. Peak, RMS, and DC statistics are therefore recorded as `null`,
not hidden or coerced to zero. Its exact float-buffer SHA-256 is
`d2e2237c131b9da23336193cd9b445b05c1ba1c80648f8899be0dd810f668cb1`.

This was not included in the six-case latency table and was not fed to ASR.
The strict JSON now exposes it under `warmup_failures`. The original serializer
had emitted bare `NaN` for undefined statistics; a recorded mechanical repair
changed only those scalars to `null`, added explicit status/gate fields, and
enabled `allow_nan=False` without rerunning synthesis.

### Follow-up initialization and boundary sweep

A sequential two-thread follow-up used a fresh process for five initialization
scenarios, then tested seven short Mandarin strings three times each after a
discarded `你好。` warm-up. It kept only one Kokoro model resident at a time.

- 2 of 5 first Mandarin calls after model construction were all-non-finite.
- 8 of 21 calls after the discarded warm-up were all-non-finite.
- `你。` failed in all five observed calls across the fresh and post-warm-up
  scenarios.
- `你好。` was finite once, then all-non-finite in six later calls. Therefore a
  warm-up or retry alone is not a reliable repair.
- `欢迎。` was finite once and all-non-finite twice.
- `你好呀。`, `请稍等。`, `我来帮你。`, and `请稍等一下。` were finite in all
  three repetitions each; the longer factual and assistance phrases were also
  finite in both fresh-process repetitions.
- No finite output in the sweep clipped.

The defect is therefore associated with particular very short Mandarin inputs,
not solely with first-call initialization. The unseeded random nodes may help
explain why the same `你好。` text produced both finite and invalid outputs, but
the causal mechanism is not proven.

Treat this as a release blocker for aggressive tiny-chunk flushing. A safe
implementation needs all of the following:

1. Reject any callback containing a non-finite sample before PCM conversion or
   playback.
2. Use at least three Han characters as a provisional chunk floor, not as a
   guarantee; the finite-PCM check remains authoritative.
3. Merge the failed text with the next clause or retry through a known-safe
   fallback voice.
4. Expand the phrase/phoneme boundary set and run ten repeated turns on the
   phone before enabling the fallback.

## Memory and artifact footprint

- Archive: 147,031,220 bytes (140.22 MiB).
- Extracted regular files: 215,321,602 bytes (205.35 MiB).
- INT8 ONNX: 109.00 MiB.
- `voices.bin`: 51.30 MiB for 103 speaker/style entries.
- Model construction: 1.061 seconds.
- Incremental RSS immediately after model construction: 377.08 MiB.
- Process RSS after both warm-ups: 432.69 MiB.
- Maximum observed process RSS: 572.30 MiB on the long Mandarin case.

This is materially heavier than the existing Supertonic host result, whose
main run peaked near 300 MiB. Do not add independent process peaks to estimate
an integrated app, but assume memory pressure is a serious phone risk when an
LLM and ASR are also resident. Stage models or obtain NPU placement if the
phone cannot keep all components resident safely.

## Intelligibility sanity

SenseVoiceSmall INT8, fixed locale and two threads, retranscribed the first
saved output for each measured case once:

- English: 0.0% micro WER across 33 reference words.
- Mandarin: 4.0% micro CER across 50 reference characters.
- The short and factual Mandarin samples were exact. The long sample had one
  lexical substitution (`还` → `海`) and ITN rendered `两` as `2`.

This is encouraging intelligibility evidence, not MOS, naturalness, prosody,
or speaker-similarity evidence. The included WAVs remain human-unverified.

As a separate sanity-only batch, not part of Kokoro metrics, the same judge
transcribed:

- the existing LFM Japanese standalone TTS sample exactly;
- the existing Supertonic Korean request exactly;
- the existing Supertonic Japanese request at 30.8% CER.

The Japanese discrepancy may reflect the TTS sample, the judge, or both; it
does not establish a human-perceived quality ranking.

## Determinism

All five exact float-PCM hashes differed for every measured case. The ONNX graph
contains one unseeded `RandomUniformLike` and one unseeded `RandomNormalLike`
node, and the Sherpa Kokoro generation path exposes no seed control.

The nondeterminism is small in duration—0 to 3.46 ms range across five runs per
case—but it is real and bit-level reproducibility should not be claimed. No
output in the six measured cases contained non-finite or clipped samples;
maximum absolute amplitude was 0.421. The separate all-non-finite Mandarin
warm-up remains the more important numerical-safety finding.

## Language and licensing facts

The archive filename says `multi-lang`, but both the pinned model card and the
embedded ONNX metadata say English and Chinese. The archive's broad
`espeak-ng-data` directory is frontend support data, not proof that the model
was trained for Spanish, Japanese, or other languages.

The source model card declares Apache-2.0, and the archive includes an exact
top-level Apache-2.0 `LICENSE`. This is a model-weight advantage over
restriction-bearing weight licenses. However, the same archive bundles
`espeak-ng-data` and CppJieba-style dictionaries without a complete
third-party notice inventory. Do not claim that every extracted file is
Apache-2.0 or redistribute the whole archive publicly until that dependency
audit is complete. This repository keeps the model outside Git and pins the
artifact, top-level license, README, model, and voice hashes in
`model_manifest.json`.

The release asset itself is exactly identified, but it contains no build
manifest tying the ONNX conversion to a precise Sherpa source commit. The
report preserves that provenance gap rather than inventing a conversion
revision.

## Integration recommendation

1. Keep streamed integrated LFM audio as the fastest English demo path.
2. Keep Supertonic as the leading broad-language standalone TTS candidate.
3. Add a mandatory finite-PCM check and invalid-chunk merge/fallback before
   any playback call.
4. Compare Kokoro and MeloTTS Chinese on the phone with identical Mandarin
   texts, two CPU threads, and one model resident at a time.
5. Flush the first speakable clause at punctuation or a bounded character/time
   threshold. For Mandarin, merge chunks shorter than three Han characters and
   still reject any non-finite callback.
6. Measure Android `AudioTrack` playback-head start, underflows, incremental
   RSS with the LLM resident, and thermal drift over ten turns.
7. Accept only if end-to-audible is at most 2.0 seconds median / 3.0 seconds
   p95 and no audible gap exceeds 80 ms. The host benchmark alone does not
   pass that phone-level gate.

## Reproduction and evidence

- Reproduction scripts: [`scripts/kokoro_int8_mac_m2`](../../../../scripts/kokoro_int8_mac_m2)
- Full five-run measurements: [`benchmark.json`](benchmark.json)
- Mandarin initialization/boundary evidence:
  [`mandarin_initialization_gate.json`](mandarin_initialization_gate.json)
- SenseVoice transcripts: [`asr_sanity.json`](asr_sanity.json)
- Raw per-locale ASR records are intentionally excluded from the public tree.
- Fixed texts/locales: [`asr_manifest.json`](asr_manifest.json)
- Artifact/provenance hashes: [`model_manifest.json`](model_manifest.json)
- Commands: [`COMMANDS.md`](COMMANDS.md)
- Warnings/failures: [`FAILURES.md`](FAILURES.md)
- Human-review WAVs are intentionally excluded from the public tree.

Upstream references:

- [Pinned Kokoro v1.1-zh model card](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/tree/01e7505bd6a7a2ac4975463114c3a7650a9f7218)
- [Pinned Sherpa release asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2)
- [Sherpa TTS release page](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)

All measured values in this report are derived from the JSON evidence. No
model server, TTS process, or ASR process was left running.
