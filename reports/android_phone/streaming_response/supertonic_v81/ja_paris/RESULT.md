# Japanese Supertonic accumulated SM8850 chain

Date: 2026-07-23<br>
Target: Snapdragon 8 Elite Gen 5 QRD, Android 16, SM8850, HTP v81,
QAIRT 2.45

## Decision

This one-phrase Japanese run is a **conditional intelligibility pass**. The
seven-stage accumulated AI Hub chain completed with finite output, and one
fixed-locale Japanese SenseVoiceSmall INT8 pass transcribed the dynamic-local,
fixed-local, and remote WAVs identically as `パリはフランスにあります。` with
0.0 normalized CER.

This is not general Japanese TTS acceptance. Numerical drift is materially
larger than in the English fixture, later vector stages and the vocoder fail
the configured elementwise comparison, and no human listening, naturalness,
speaker-similarity, phone playback, or glasses test has been run. The result
strengthens multilingual feasibility evidence without displacing the existing
LFM-JP quality route or its broader evaluation requirement.

## Fixture

- Text: `パリはフランスにあります。`
- Preprocessed text: `<ja>パリはフランスにあります。</ja>`
- Text positions: 23 real / 32 fixed; no shortening was required
- Seed: 42
- Speaker: 6
- Denoising calls: 4, with `total_step=4`
- Latent positions: 31 real / 32 fixed, with one zero-masked padding position
- Output length: 95,025 samples at 44.1 kHz, 2.1548 seconds
- Fixture SHA-256:
  `5b24e3c8c25305e82aa5b46bb56a8f384b2f03f2dba8511d422c319ef1f60a82`

The experiment reused the existing component compiles and submitted inference
jobs only. No graph was compiled, linked, profiled, quantized, or changed.

## Sequential remote results

| Stage | Existing compile | Inference | NRMSE vs fixed-local | Cosine | `allclose(0.05, 0.05)` gate |
|---|---|---|---:|---:|---|
| Duration predictor | `jp2v287x5` | `jgl1n9n25` | 0.000222 | 1.000000 | Pass |
| Text encoder | `j5mdj79d5` | `jgdzwknk5` | 0.004686 | 0.999989 | Pass |
| Vector step 0 | `j5w1o2k6g` | `jgl1ndo85` | 0.005840 | 0.999983 | Pass |
| Vector step 1 | `j5w1o2k6g` | `j5mdr7kd5` | 0.020288 | 0.999794 | Fail |
| Vector step 2 | `j5w1o2k6g` | `jpxdvwo8g` | 0.050370 | 0.998740 | Fail |
| Vector step 3 | `j5w1o2k6g` | `jgjwxjox5` | 0.090938 | 0.995871 | Fail |
| Vocoder | `jgo4ykxx5` | `jprnkzr75` | 0.751043 | 0.727700 | Fail |

The remote text embedding and every masked remote vector output were fed into
the next stage. The duration output was checked independently; it did not
change the fixture's precomputed fixed latent length. The vector rows therefore
measure accumulated chain drift rather than isolated component probes.

## Audio comparison

The metrics below use saved float32 tensors before PCM16 WAV serialization.
Recomputing waveform metrics from the linked PCM16 files will differ slightly.

| Remote waveform compared with | Waveform NRMSE | Waveform cosine | Spectral convergence | Log-magnitude RMSE |
|---|---:|---:|---:|---:|
| Fixed-shape local, 32 positions | 0.751074 | 0.727672 | 0.251543 | 0.081089 |
| Dynamic local, 31 positions | 1.518921 | 0.019742 | 1.139351 | 0.311165 |

Fixed-32 versus dynamic-31 local inference already differs strongly before
Qualcomm compilation: waveform NRMSE 1.488393, cosine 0.022698, and spectral
convergence 1.112356. The remote result is much closer to the fixed-shape local
graph than to the dynamic graph, but its fixed-local difference is also larger
than in the completed English fixture. Padding/topology impact and remote
lowering drift therefore remain separate quality risks.

All three listening files are mono PCM16 WAVs at 44.1 kHz with 95,025 frames:

- [dynamic local WAV](local_dynamic_trimmed.wav), SHA-256
  `f9c157d4b495942932921262c2d04f157cd07362b148ec89bca75705368e8a89`
- [fixed-shape local WAV](local_fixed_trimmed.wav), SHA-256
  `06dc870d6cadd49728f5904bf3c42643c939385742562e8e32d358bb16596253`
- [accumulated AI Hub WAV](remote_chain/remote_fixed_trimmed.wav), SHA-256
  `1c265ec46103df0352c642e85596de7f4427b0d8d964b8bced45810fbd2bb027`

## Japanese intelligibility backcheck

One fixed-locale Japanese SenseVoiceSmall INT8 worker used two CPU threads and
processed the three files sequentially with one resident recognizer:

| WAV | Transcript | Normalized CER |
|---|---|---:|
| Dynamic local | `パリはフランスにあります。` | 0.0 |
| Fixed-shape local | `パリはフランスにあります。` | 0.0 |
| Accumulated AI Hub | `パリはフランスにあります。` | 0.0 |

Model loading took 0.810 seconds. The three recognizer calls took 0.057-0.062
seconds each; the first audio decode/resample took 0.512 seconds and the next
two took 0.060-0.082 seconds. Incremental recognizer RSS after load was 393.3
MiB, and observed process peak RSS was 439.4 MiB. The complete machine-readable
record is [asr_sanity.json](asr_sanity.json), SHA-256
`a65d6b4a9985ac9644df1a273e112b79d2b2d07294215100c5863093d55e8400`.

This is an intelligibility sanity check only. It does not measure naturalness,
prosody, accent preference, voice identity, audible artifacts, or human
quality. Using one ASR model on one phrase also cannot establish open-ended
Japanese accuracy.

## Interpretation for language routing

- The same fixed-shape Supertonic graphs now preserve the complete normalized
  reference content in one English and one Japanese accumulated-chain fixture.
- The shared exact-SM8850 component profiles remain fully NPU placed and sum to
  118.891 ms for one duration call, one text-encoder call, four vector calls,
  and one vocoder call. That is a neural-component lower bound, not measured
  text-to-speaker latency for this Japanese run.
- Supertonic remains the lighter Japanese challenger. The larger numerical
  drift and prior broader Japanese ASR errors mean this one exact phrase does
  not replace the LFM-JP standalone quality-evidence leader.
- The next gate is human A/B listening over a representative Japanese phrase
  set, followed by the same fixed graph on the Nubia with actual playback-head,
  RSS, thermal, and repeated-turn measurements.

## Reproduction and durable state

- Fixture builder: `scripts/prepare_supertonic_chain_fixture.py`
- Resume-safe runner: `scripts/run_aihub_supertonic_chain_v81.py`
- ASR worker: `scripts/kokoro_int8_mac_m2/asr_sanity.py`
- Fixture manifest: [fixture_manifest.json](fixture_manifest.json)
- ASR manifest: [asr_manifest.json](asr_manifest.json)
- Resume state is intentionally excluded from the public tree.
- Remote result: [remote_chain/result.json](remote_chain/result.json)

The cloud runner was launched in detached `tmux`; it records a job ID before
polling, checksum-saves each output, and resumes by saved ID/output hash. The
local fixture and ASR passes each used two threads and ran only at macOS memory
pressure level 1 with no competing model process.

Model weights are not stored in the repository. Supertonic model weights and
derivatives carry BigScience OpenRAIL-M terms; the Sherpa-ONNX runtime and
SenseVoice judge have separate licenses. This report contains no Nubia,
physical-glasses, acoustic-start, power, or human-listening measurement.
