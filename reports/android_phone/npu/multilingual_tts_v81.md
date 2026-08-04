# Multilingual standalone TTS on SM8850 / HTP v81

Date: 2026-07-22; updated 2026-07-23<br>
**Update 2026-07-26:** the MeloTTS-ZH real-text repeated-decoder chain called for
below is now complete and passing on the exact QRD target. See
[`../streaming_response/tts_research_20260726/NPU_STATUS.md`](../streaming_response/tts_research_20260726/NPU_STATUS.md).<br>
Primary target: Snapdragon 8 Elite Gen 5 QRD, Android 16, `SM8850`, HTP v81<br>
Package audited: `qai-hub-models==0.58.0`<br>
Comparable published runtime: Voice AI, QAIRT `2.45.0.260326154327`

## Decision summary

1. **Keep streamed LFM audio as the first demo path.** The measured LFM server
   already emits PCM in well under one second; standalone TTS is deployment
   insurance and a multilingual expansion path, not a prerequisite for the
   2–3 second response target.
2. **Piper is the safest Qualcomm-native standalone TTS path for English,
   German, and Italian.** All published neural components are fully NPU placed
   on HTP v81. The one-call component latency sum on Galaxy S26 is about 35.2
   ms for every language.
3. **MeloTTS is the official Qualcomm path for Spanish and Chinese.** Its
   one-call component sum is about 138–140 ms on the same v81 generation, also
   with full NPU placement. It is materially heavier than Piper but still
   leaves substantial room inside a 500 ms TTS-first-PCM budget.
4. **Do not call the public package an exact Nubia result.** The downloadable
   v0.58.0 package resolves to `sm8850-ad` (Galaxy) rather than generic
   `sm8850`, and its scorecards contain Galaxy S26 rows rather than QRD rows.
   Fresh QRD compile/profile jobs are required for the closest available proxy
   to the Nubia.
5. **Voice AI packaging is an integration dependency.** These exports are QNN
   context binaries packaged for Qualcomm Voice AI, not a drop-in replacement
   for the current CPU runner. The Nubia vendor QAIRT runtime previously
   reported 2.37 while the public packages use 2.45, so direct sideloading is
   not yet justified.

## Published HTP v81 evidence

The table below is extracted from the official 0.58.0 wheel's `perf.yaml`
files. Every listed layer is reported on NPU; there is no published CPU layer
for these rows.

| Model | Languages | Galaxy S26 component latencies | One-call sum | Placement |
|---|---|---|---:|---|
| PiperTTS | EN | encoder 17.056 ms; SDP 6.599 ms; flow 8.973 ms; decoder 1.827 ms; G2P encoder 0.489 ms; G2P decoder 0.281 ms | 35.225 ms | 2,090 / 2,090 NPU layers |
| PiperTTS | DE | encoder 17.098 ms; SDP 6.603 ms; flow 8.910 ms; decoder 1.822 ms; G2P encoder 0.488 ms; G2P decoder 0.260 ms | 35.181 ms | 2,090 / 2,090 NPU layers |
| PiperTTS | IT | encoder 17.080 ms; SDP 6.618 ms; flow 8.980 ms; decoder 1.821 ms; G2P encoder 0.488 ms; G2P decoder 0.266 ms | 35.253 ms | 2,090 / 2,090 NPU layers |
| MeloTTS | ES | encoder 23.899 ms; flow 71.126 ms; decoder 42.433 ms; T5 encoder 0.488 ms; T5 decoder 0.267 ms | 138.213 ms | 2,681 / 2,681 NPU layers |
| MeloTTS | ZH | encoder 23.806 ms; flow 71.053 ms; decoder 42.440 ms; BERT wrapper 2.689 ms | 139.988 ms | 2,372 / 2,372 NPU layers |

### How to interpret those numbers

- They are fixed-shape component calls, **not** end-to-end text-to-first-audio
  measurements.
- Piper and Melo decode audio in 40-frame payloads with 12-frame overlap. One
  decoder call corresponds to roughly 464 ms of 22.05/44.1 kHz output audio,
  so the decoder row repeats for longer responses.
- The G2P/T5 decoder is autoregressive and also repeats per generated token.
  Therefore the simple sum is a lower-bound orchestration estimate, not a full
  utterance latency claim.
- Even after accounting for repetition, the published neural compute is much
  smaller than the current multi-second wait. Text availability, model
  orchestration, audio buffering, and playback scheduling are the likely
  first-audio limiters.

## Exact-target audit

- Live AI Hub target: `Snapdragon 8 Elite Gen 5 QRD`, Android 16,
  `chipset:sm8850`, `hexagon:v81`, `framework:qnn`.
- Public model metadata lists the QRD as supported.
- The public fetch CLI does **not** return an asset for `-c sm8850` or
  `--device "Snapdragon 8 Elite Gen 5 QRD"`.
- Selecting the human display name `Snapdragon 8 Elite Gen 5 Mobile` resolves
  to a package whose filename ends in
  `qualcomm_snapdragon_8_elite_gen5_for_galaxy.zip`, i.e. `sm8850-ad`.
- Public scorecard job IDs are not visible to this AI Hub account, so their
  `perf.yaml` records are supplementary published evidence only.

## Fresh QRD experiment

The repository runner [run_aihub_tts_v81.py](../../../scripts/run_aihub_tts_v81.py)
loads the official fixed-shape Piper neural components without invoking the
Linux-only phonemizer, records local goldens, and performs the official Voice
AI sequence:

1. serialize and upload TorchScript;
2. compile to QNN DLC;
3. link to an SM8850 context binary;
4. profile on the QRD;
5. run inference and compare against the local golden;
6. classify runtime layers as NPU, CPU fallback, or other.

All cloud submissions are launched from detached tmux. State is written after
every submission so job IDs survive a stopped local poller.

Current account-owned result table:

| Item | State | Compile | Link | Profile | Inference | Result |
|---|---|---|---|---|---|---|
| Piper IT decoder | complete | `jg9x6q1mg` | `jp1vrkn8p` | `jpy7er97p` | `jp0vy3n6g` | **Pass:** 1.821 ms, 117/117 NPU layers, close local-golden match |
| Piper IT SDP | complete | `jp3wko1z5` | `jp84ojzx5` | `j5mdr9ww5` | `jpy7e847p` | **Pass:** 6.602 ms, 621/621 NPU layers, both outputs exact |
| Piper IT encoder | complete | `jg9xvoq8g` | `jprnkxrk5` | `j5qm810mg` | `jgl1n84l5` | **Pass:** 16.958 ms, 401/401 NPU layers, all four outputs pass after semantic-name reconciliation |
| Piper IT flow | complete | `jp3wkrkm5` | `jpy7ek88p` | `jgz4eq0xp` | raw `jp0vy8xng`; valid fixture `j5qm8nv4g` | **Pass:** 8.930 ms, 310/310 NPU layers; valid pipeline fixture NRMSE 0.000616, cosine 0.9999998 |
| Piper IT four-neural-component core | complete | reused rows above | reused rows above | reused rows above | reused rows above | **Pass:** 34.311 ms fixed-shape call sum, 1,449/1,449 NPU runtime layers, zero fallback |
| Supertonic text encoder | complete | `j5mdj79d5` | n/a | `jpxdwv1jg` | `jp84lo3z5` | **Pass:** 1.046 ms, 462/462 NPU layers |
| Supertonic duration predictor | complete | `jp2v287x5` | n/a | `j5qmjoymg` | `jp3wy29z5` | **Pass:** 0.408 ms, 277/277 NPU layers |
| Supertonic vector estimator | complete | `j5w1o2k6g` | n/a | `jpxdvnq1g` | `j5mdrq7w5` | Placement pass; numerical warning, 2.253% NRMSE |
| Supertonic vocoder | complete after metadata-only ONNX repair | `jgo4ykxx5` | n/a | `jgjwxz9x5` | `jpel9eq1g` | Placement pass; numerical warning, 16.110% NRMSE |
| Piper EN/DE full chains | not submitted | — | — | — | — | Published same-generation evidence retained; fresh IT neural core is the account-owned exact-target proof |
| MeloTTS ES/ZH | Linux/Colab export required | blocked before submission | — | — | — | Colab OAuth currently requires re-authorization |

The complete current job evidence, shapes, placement, numerical comparisons,
and interpretation is in
[`../streaming_response/aihub_voice_v81/REPORT.md`](../streaming_response/aihub_voice_v81/REPORT.md).
The first flow comparison used the framework's generic dense-random fixture and
produced an all-NaN local golden because every projected `logs_p` value exceeded
Float32's exponential range. That raw failure is retained as provenance. An
inference-only recheck using the official serialized encoder, SDP, monotonic
attention construction, and exact same flow context passes; no graph was
recompiled or reprofiled.

The 34.311 ms Piper number is the sum of one encoder, one SDP, one flow, and one
decoder call. It excludes phonemization/G2P, host orchestration and transfers,
context loading, repeated decoder calls with overlap stitching, and audio
playback. It is therefore a neural-compute feasibility result, not
text-to-first-audible latency.
The Supertonic 118.891 ms figure remains fixed-shape component-profile
arithmetic, not text-to-first-audio latency. A separate accumulated real-text
chain now completed with four vector calls and a remote vocoder waveform. Its
strict waveform comparison fails, but dynamic-local, fixed-local, and remote
audio all backcheck to the exact five reference words at 0.0 normalized WER.
Treat that as a conditional one-fixture feasibility pass; see
[`../streaming_response/supertonic_v81/ACCUMULATED_CHAIN_RESULT.md`](../streaming_response/supertonic_v81/ACCUMULATED_CHAIN_RESULT.md).

Example reproducible launch:

```bash
tmux new-session -d -s aihub-piper-v81 \
  "cd /path/to/LMF_demo && \
   PYTHONUNBUFFERED=1 \
   /path/to/python3.11 scripts/run_aihub_tts_v81.py \
     --language it \
     --component flow \
     --device 'Snapdragon 8 Elite Gen 5 QRD' \
     --device-os 16 \
     --qairt-version 2.45 \
     --output-dir /persistent/path/piper-it-flow-run \
     > /persistent/path/piper-it-flow-run.log 2>&1"
```

The runner intentionally accepts exactly one language and one component per
state directory so every accepted cloud job has an unambiguous resume path.

## Practical recommendation

For the phone demo, use this order:

1. ship and validate the existing streamed-LFM playback fix;
2. integrate Piper EN first if a standalone TTS fallback is still needed;
3. expose DE/IT through the same Piper runtime and assets;
4. add Melo ES/ZH only after the Voice AI runtime is integrated and its
   incremental RSS is measured with LFM still resident;
5. keep the acceptance gates at TTS first PCM <=500 ms median / <=900 ms P95,
   end-to-first-audible <=2.0 s median / <=3.0 s P95, and no playback gap above
   80 ms.

The public v81 results make NPU feasibility highly likely. The remaining risks
are runtime/package compatibility on the Nubia, app memory while LFM remains
loaded, and end-to-end audio scheduling rather than raw TTS neural latency.
