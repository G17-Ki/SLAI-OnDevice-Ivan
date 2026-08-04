# Supertonic 3 INT8: standalone multilingual TTS feasibility

Date: 2026-07-22<br>
Evidence class: Apple Silicon host benchmark only; this is not a phone, AR1,
or glasses result.

## Decision

Promote Supertonic 3 INT8 to the phone CPU benchmark as the leading
standalone multilingual TTS challenger.

It is fast enough to fit comfortably inside the 2–3 second speech-response
budget once a short speakable text chunk is available. On the M2, a 64-character
first chunk produced 3.26 seconds of playable audio in 0.568 seconds median
(0.621 seconds p95). A short answer took 0.35–0.54 seconds median across six
tested languages at the five-step quality setting.

This does **not** supersede the simpler fix for the current integrated LFM
path: LFM already emits PCM early, and the Android UI had been buffering it.
Use streamed LFM audio for the fastest English demo. Supertonic is most useful
when we want a decoupled text-LLM + TTS architecture or multilingual output.

For the first phone build, use two inference threads and four denoising steps.
Two steps were faster but caused clear retranscription errors. Five steps are
the conservative default; eight added latency without improving the ASR smoke
check.

## What was tested

- Model: `Supertone/supertonic-3`, revision
  `3cadd1ee6394adea1bd021217a0e650ede09a323`.
- Conversion: Sherpa-ONNX Supertonic 3 INT8 release dated 2026-05-11,
  revision `cca5a0e6c96e1d2c720986bf7e75fcc81dee3ae4`.
- Runtime: `sherpa-onnx==1.13.4`, CPU provider.
- Host: Apple M2, 8 GB RAM, macOS 15.4.1, Python 3.11.11.
- Main setting: two threads, five denoising steps, speed 1.05, speaker 6,
  deterministic seed 0, one warm-up, five measured runs per case.
- Languages exercised: English, Spanish, German, Japanese, Korean, Russian.
- Output: mono 44.1 kHz PCM WAV.

The INT8 conversion uses dynamic INT8 for the duration predictor, text encoder,
and vector estimator, and static INT8 for the vocoder, according to the Sherpa
conversion documentation.

## Main latency results

The first callback is the first complete, playable PCM text chunk. For a short
utterance it is also the full synthesis time.

| Case | First PCM median | First PCM p95 | Audio duration | Median RTF |
|---|---:|---:|---:|---:|
| English short | 0.348 s | 0.386 s | 1.994 s | 0.175 |
| Spanish short | 0.433 s | 0.443 s | 2.332 s | 0.186 |
| German short | 0.451 s | 0.484 s | 2.596 s | 0.174 |
| Japanese short | 0.385 s | 0.467 s | 2.189 s | 0.176 |
| Korean short | 0.543 s | 0.628 s | 2.490 s | 0.218 |
| Russian short | 0.466 s | 0.535 s | 2.392 s | 0.195 |
| English sentence | 0.851 s | 0.979 s | 5.417 s | 0.157 |
| English, 64-character chunks | 0.568 s | 0.621 s | 10.573 s total | 0.175 |

The chunked long response generated three callbacks. In the first measured run:

- callback 1 at 0.534 s carried 3.262 s of audio;
- callback 2 at 1.312 s carried another 4.466 s;
- callback 3 at 1.801 s carried another 2.245 s.

Playback can therefore start after callback 1 while synthesis remains far
ahead of the playback head. This is chunk streaming, not frame-level vocoder
streaming.

## Denoising-step trade-off

Three warmed English runs per cell, two threads. The ASR column uses
Faster-Whisper Tiny as an intelligibility smoke check on the long response.

| Steps | First 64-char chunk | Long-response RTF | Long-response WER | Decision |
|---:|---:|---:|---:|---|
| 2 | 0.221 s | 0.071 | 0.423 | Reject: obvious word substitutions/omissions |
| 4 | 0.384 s | 0.125 | 0.038 | Recommended phone starting point |
| 5 | 0.461 s | 0.154 | 0.038 | Conservative quality default |
| 8 | 0.731 s | 0.232 | 0.038 | No ASR benefit over 4–5 steps |

The two-step long response was retranscribed as substantially different text,
while the four-, five-, and eight-step outputs were recovered nearly verbatim.
ASR does not measure naturalness, so a human A/B test is still required before
locking four steps.

## Thread-count trade-off

Three warmed English runs per cell, five denoising steps.

| Threads | First 64-char chunk | Long-response RTF |
|---:|---:|---:|
| 1 | 0.883 s | 0.335 |
| 2 | 0.456 s | 0.154 |
| 4 | 0.456 s | 0.171 |
| 8 | 1.127 s | 0.363 |

Two threads were the best overall M2 setting. Eight threads were strongly
counterproductive. The phone needs its own 2-vs-4-thread sweep because its CPU
topology and thermal behavior differ.

## Footprint

- Verified download archive: 128,774,318 bytes (122.81 MiB).
- Unpacked files: 145,316,356 bytes (138.58 MiB).
- ONNX files alone: 144,508,203 bytes (137.81 MiB).
- Model construction: 0.559 seconds in the main run.
- Incremental RSS after construction: 137.8–263.5 MiB across the retained
  two-thread runs. Keep this construction delta separate from maximum process
  RSS.
- Maximum sampled process RSS during the main matrix: 300.22 MiB.
- No non-finite samples or clipped samples were observed in the five-step
  measured outputs.

RSS is host evidence only. On the phone, measure incremental RSS while LFM
remains loaded; do not add separate process peaks from unrelated runs.

## Multilingual quality sanity

Faster-Whisper Tiny recovered the English, Spanish, German, Korean, and Russian
short sentences exactly after punctuation normalization. It also recovered the
English sentence exactly and the chunked English response with one accent-only
difference (`cafés` vs `cafes`). Japanese had 0.231 CER under this small judge,
which is not enough evidence to distinguish a TTS issue from an ASR-judge
issue. Treat the included WAVs as samples for human review.

This is an intelligibility check, not a MOS, speaker-similarity, prosody, or
naturalness evaluation.

## Language coverage and limitations

The single model supports 31 languages: Arabic, Bulgarian, Croatian, Czech,
Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hindi,
Hungarian, Indonesian, Italian, Japanese, Korean, Latvian, Lithuanian, Polish,
Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Turkish,
Ukrainian, and Vietnamese.

Mandarin Chinese is **not supported**. Keep the Qualcomm MeloTTS path for
Chinese, or evaluate another Chinese-capable model. Also, standalone TTS solves
only the output side: a multilingual voice assistant still needs multilingual
ASR and a text model that answers in the selected language.

## Important integration trap

In Sherpa's Python binding, this does not persist:

```python
config.extra["lang"] = "es"
```

`extra` is returned by value. Assign the complete map instead:

```python
config.extra = {"lang": "es", "seed": "0", "max_len": "64"}
```

Without this change, non-English benchmark cases silently run with the default
English setting.

## Expected end-to-first-audio budget

For a text-streaming pipeline, the relevant approximation is:

`VAD close → first speakable clause + 0.27–0.38 s TTS + playback cushion`

At the observed 0.8-second first LLM token, a clause that becomes available by
roughly 1.3–1.8 seconds should yield first audible output around 1.7–2.3
seconds. Do not wait for the full answer. Flush on punctuation, or on a bounded
50–70 character / short-timeout rule, enqueue TTS immediately, and play each
callback through `AudioTrack` while later chunks synthesize.

This estimate must be replaced by an actual end-to-audible phone trace.

## Phone experiment to run next

1. Integrate the Sherpa-ONNX arm64 Android runtime and keep the 139 MiB model
   outside the APK or as separately installed private app data.
2. Start with two threads, four steps, speed 1.05, and one explicit language
   selector.
3. Feed clauses incrementally from the LLM; do not pass only the final text.
4. Play callback PCM immediately with a small `AudioTrack` cushion.
5. Run at least ten fixed-text turns per setting for two and four threads and
   four and five steps.
6. Record text-chunk-ready, TTS call, first callback, first `AudioTrack` write,
   playback-head movement, completion, underflows, and incremental RSS with LFM
   resident.
7. Repeat English, Spanish, Japanese, Korean, and Russian; add human listening
   notes. Run a separate Chinese-capable TTS path.

The host acceptance gate is passed for short or explicitly chunked responses:
TTS first PCM is below 0.9 seconds p95 and RTF is well below 0.7. The unchunked
5.4-second sentence reached 0.979 seconds p95, which reinforces the need to
flush short clauses. Phone and thermal gates remain open.

## Licensing

- Supertone sample/runtime code: MIT.
- Sherpa-ONNX runtime: Apache-2.0.
- Supertonic 3 model weights and derivatives: BigScience OpenRAIL-M, including
  use-based restrictions and redistribution obligations.

The Sherpa archive includes an MIT `LICENSE` for code, but its README identifies
the accompanying model as OpenRAIL-M. The repository therefore does not store
the model weights. The pinned downloader retrieves and preserves the upstream
OpenRAIL-M model license alongside the downloaded assets.

## Reproduction and evidence

- Reproduction scripts: [`scripts/supertonic3`](../../../../scripts/supertonic3)
- Full five-run benchmark: [`benchmark.json`](benchmark.json)
- Step matrix: [`step_matrix_summary.json`](step_matrix_summary.json)
- Thread matrix: [`thread_matrix_summary.json`](thread_matrix_summary.json)
- ASR smoke check: [`asr_sanity.json`](asr_sanity.json)
- Model manifest and hashes: [`model_manifest.json`](model_manifest.json)
- Human-review WAVs are intentionally excluded from the public tree.

Upstream references:

- [Supertonic 3 model card](https://huggingface.co/Supertone/supertonic-3/tree/3cadd1ee6394adea1bd021217a0e650ede09a323)
- [Supertonic source](https://github.com/supertone-inc/supertonic/tree/dff55dc00064c398736080c78195f577527832ae)
- [Sherpa Supertonic documentation](https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html)
- [Pinned INT8 model release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
- [Supertonic OpenRAIL-M license](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/LICENSE)

This report was hand-written from script-generated JSON. No measured value was
manually altered.
