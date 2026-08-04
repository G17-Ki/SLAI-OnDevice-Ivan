# Qualcomm AI Hub voice components on SM8850 / HTP v81

Last updated: 2026-07-23 09:07 CST

Target: `Snapdragon 8 Elite Gen 5 QRD`, Android 16, SM8850, HTP v81,
QAIRT 2.45. These are account-owned AI Hub measurements on the QRD, not
measurements from the disconnected Nubia phone or the glasses.

[`results.json`](results.json) preserves the original component snapshot at
its recorded cutoff and now includes a current evidence index for the later
Supertonic accumulated chains, Piper Italian fixture correction and production
chain, Piper English release-context profile, SenseVoice profile, Zipformer
profile, Whisper Small W8A16 profile, and MeloTTS-ZH context-portability
profile. The linked task-specific evidence remains authoritative for each
later result.

## Supertonic 3

The fixed shapes use a 32-symbol text window. All four components compile as QNN
context binaries and place every reported runtime layer on the Hexagon NPU.
The text encoder and duration predictor pass the saved local-golden comparison;
the vector estimator and vocoder retain explicit numerical warnings.

| Component | Compile | Profile | Inference | Latency | Peak memory | Placement | Numerical result |
|---|---|---|---|---:|---:|---|---|
| Text encoder | `j5mdj79d5` | `jpxdwv1jg` | `jp84lo3z5` | 1.046 ms | 117.84 MiB | 462 NPU, 0 CPU, 0 other | Pass; cosine 0.9999875, NRMSE 0.005214, max abs 0.09396 |
| Duration predictor | `jp2v287x5` | `j5qmjoymg` | `jp3wy29z5` | 0.408 ms | 117.30 MiB | 277 NPU, 0 CPU, 0 other | Pass; cosine 1.0, NRMSE 0.000994, max abs 0.003176 |
| Vector estimator (one denoising step) | `j5w1o2k6g` | `jpxdvnq1g` | `j5mdrq7w5` | 28.571 ms | 118.05 MiB | 1,111 NPU, 0 CPU, 0 other | Strict placement pass; strict allclose warning: cosine 0.9997461, NRMSE 0.02253, max abs 0.20518 |
| Vocoder | `jgjwj4q85` failed; corrected `jgo4ykxx5` | `jgjwxz9x5` | `jpel9eq1g` | 3.153 ms | 117.81 MiB | 212 NPU, 0 CPU, 0 other | Strict placement pass; strict allclose warning: cosine 0.9871662, NRMSE 0.16110, max abs 0.13517 |

At the recommended four denoising steps, the measured vector-estimator lower
bound is `4 × 28.571 = 114.284 ms`. The complete measured neural component sum
is therefore `1.046 + 0.408 + 114.284 + 3.153 = 118.891 ms`. This is not an
end-to-end latency claim: orchestration, device transfers, text preprocessing,
and audio scheduling remain outside the sum.

The vector estimator's warning is numerical rather than placement-related. Its
shape and finite-value checks pass and its output direction is very close to
the local golden, but elementwise `allclose(atol=rtol=0.05)` fails. This needs
an audio-level A/B test after the complete chain is runnable; it must not be
silently relabeled as a numerical pass.

The vocoder also has perfect NPU placement but a materially larger numerical
warning. Shape/finite checks pass, yet waveform cosine is 0.98717 and NRMSE is
16.1%. The next gate must compare decoded audio perceptually and by ASR rather
than accepting this graph from placement/latency alone.

The vocoder repair removes only the redundant `wav_tts` entry from
`graph.value_info`; graph nodes, weights, inputs, and outputs are untouched.
The original SHA-256 is
`e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152` and
the checked repaired copy is
`db8cd48bf21464fc23114941dfe4d1ed421754920823b5ff02eb0ff1889e20df`.

### Accumulated real-text chain

The component warnings were followed by a resume-safe, inference-only chain on
the real phrase `It is Paris, in France.`. It reused the four compile jobs
above and used four vector calls with `total_step=4`. The remote text embedding
and each remote vector output were fed forward; the duration result was checked
independently because the fixed latent length was precomputed in the fixture.

| Stage | Inference | NRMSE vs fixed-local | Cosine | `allclose(0.05, 0.05)` gate |
|---|---|---:|---:|---|
| Duration | `jgjwx3d85` | 0.000619 | 1.000000 | Pass |
| Text encoder | `j56d6oz7p` | 0.006385 | 0.999980 | Pass |
| Vector 0 | `jgn7219rp` | 0.003914 | 0.999992 | Pass |
| Vector 1 | `j5qm84zng` | 0.026508 | 0.999655 | Fail |
| Vector 2 | `jp0vyoong` | 0.042234 | 0.999108 | Fail |
| Vector 3 | `j5qm84neg` | 0.065366 | 0.997868 | Fail |
| Vocoder | `jp1v014lp` | 0.467072 | 0.893110 | Fail |

The trimmed remote waveform is 2.1276 seconds and finite. Against fixed-local,
spectral convergence is 0.160421; against dynamic-local it is 0.825993. A
single fixed-locale English SenseVoiceSmall INT8 check transcribed all three
WAVs as `It is Paris in France.` with 0.0 normalized WER. This clears the
one-fixture intelligibility gate but does not replace human listening or a
broader multilingual quality matrix. Full hashes, audio links, and caveats are
in
[`../supertonic_v81/ACCUMULATED_CHAIN_RESULT.md`](../supertonic_v81/ACCUMULATED_CHAIN_RESULT.md).

### Japanese accumulated fixture

The same four compiles were then reused without modification for
`パリはフランスにあります。`: 23/32 text positions, 31/32 latent positions,
speaker 6, seed 42, and four vector calls. Every created job was inference-only.

| Stage | Inference | NRMSE vs fixed-local | Cosine | `allclose(0.05, 0.05)` gate |
|---|---|---:|---:|---|
| Duration | `jgl1n9n25` | 0.000222 | 1.000000 | Pass |
| Text encoder | `jgdzwknk5` | 0.004686 | 0.999989 | Pass |
| Vector 0 | `jgl1ndo85` | 0.005840 | 0.999983 | Pass |
| Vector 1 | `j5mdr7kd5` | 0.020288 | 0.999794 | Fail |
| Vector 2 | `jpxdvwo8g` | 0.050370 | 0.998740 | Fail |
| Vector 3 | `jgjwxjox5` | 0.090938 | 0.995871 | Fail |
| Vocoder | `jprnkzr75` | 0.751043 | 0.727700 | Fail |

Remote versus fixed-local spectral convergence is 0.251543; remote versus
dynamic-local is 1.139351. A single two-thread, fixed-locale Japanese
SenseVoiceSmall INT8 pass nevertheless transcribed all three WAVs exactly as
`パリはフランスにあります。` with 0.0 CER over 12 normalized characters.

This is a conditional one-phrase Japanese intelligibility pass, not a general
quality acceptance. Its numerical drift is larger than the English fixture,
and prior broader Japanese checks still contain errors. Full WAV hashes,
resource data, methodology, and caveats are in
[`../supertonic_v81/ja_paris/RESULT.md`](../supertonic_v81/ja_paris/RESULT.md).

## Piper English release-context portability

The public `qai-hub-models 0.58.0` Piper-English Voice AI package was generated
for Galaxy S26 `sm8850-ad`, HTP v81, QAIRT 2.45. All six compiled QNN contexts
ran unchanged on the exact Android-16 plain-SM8850 QRD:

| Component | Profile | Latency | Peak memory | Placement |
|---|---|---:|---:|---|
| Encoder | `jgk9nlz25` | 17.014 ms | 119.203 MiB | 401/401 HTP |
| SDP | `j5qmk784g` | 6.577 ms | 118.250 MiB | 620/620 HTP |
| Flow | `jgo401yx5` | 8.924 ms | 126.465 MiB | 310/310 HTP |
| Audio decoder | `jpv9o13j5` | 1.822 ms/call | 117.699 MiB | 117/117 HTP |
| Charsiu encoder | `jg9x131lg` | 0.487 ms/call | 118.031 MiB | 461/461 HTP |
| Charsiu decoder | `jp1vldl2p` | 0.280 ms/output token | 118.922 MiB | 180/180 HTP |

The acoustic core is **34.337 ms** and **1,448/1,448 HTP operations**. One
call each of all six graphs is **35.104 ms** and **2,089/2,089 HTP operations**,
with zero CPU fallback or unsupported placement. That all-six sum is within
0.34% of Qualcomm's Galaxy S26 component rows. One audio-decoder call exposes
about 464.4 ms of new audio after overlap slicing, supporting decoder-payload
streaming.

The release contains no matching inputs or goldens. No compile or inference
job was submitted, and the profile makes no numerical-equivalence or English-
quality claim. Full evidence is in
[`../piper_en_v81/RESULT.md`](../piper_en_v81/RESULT.md).

## Piper Italian proof

All four official Italian neural components now have account-owned exact-target
compile, context-link, profile, and inference evidence.

| Component | Compile | Link | Profile | Inference | Latency | Peak memory | Placement | Numerical result |
|---|---|---|---|---|---:|---:|---|---|
| Encoder | `jg9xvoq8g` | `jprnkxrk5` | `j5qm810mg` | `jgl1n84l5` | 16.958 ms | 83.32 MiB | 401 NPU, 0 CPU, 0 other | Pass; four outputs pass after semantic-name reconciliation |
| SDP | `jp3wko1z5` | `jp84ojzx5` | `j5mdr9ww5` | `jpy7e847p` | 6.602 ms | 118.29 MiB | 621 NPU, 0 CPU, 0 other | Pass; `y_lengths` and `w_ceil` exact |
| Flow | `jp3wkrkm5` | `jpy7ek88p` | `jgz4eq0xp` | valid fixture `j5qm8nv4g` | 8.930 ms | 126.14 MiB | 310 NPU, 0 CPU, 0 other | Pass; cosine 0.9999998, NRMSE 0.0006158, max abs 0.028513 |
| Decoder | `jg9x6q1mg` | `jp1vrkn8p` | `jpy7er97p` | `jp0vy3n6g` | 1.821 ms | 117.78 MiB | 117 NPU, 0 CPU, 0 other | Pass; cosine 0.9999959, NRMSE 0.03738, max abs 0.001891 |

One call to each neural component sums to **34.311 ms**, with 1,449/1,449
reported runtime layers on NPU and zero fallback. Peak-memory rows are per
component and must not be added together.

The flow required a fixture correction, not a graph correction. The first
generic dense-random sample made all 294,912 local `z` values NaN: every
projected `logs_p` value was between 105.253 and 151.686, above Float32's
approximately 88.72 exponential-overflow threshold. The mask had no zero
positions, so no mask-aware comparison was justified. That raw inference
`jp0vy8xng` and failure remain preserved.

Inference-only job `j5qm8nv4g` reused the exact linked flow context. Its local
fixture was produced by the official serialized encoder, official serialized
SDP, binary monotonic attention, and the exact serialized flow. It contains 71
valid frames, is fully finite, and has an exactly zero padded tail. The fixture
SHA-256 is
`becc8b2dfad79afa24ebb885458c3849af52b0322913e4c8d4298201f74c1e43`.
No compile, link, or profile stage was repeated.

The 34.311 ms figure is not end-to-end text-to-first-audio latency. It excludes
phonemization/G2P, host orchestration and transfers, context loading, repeated
decoder calls with overlap stitching, and playback.

### Accumulated-chain fixture correction

An inference-only chain subsequently reused all four linked contexts and
completed encoder, SDP, flow, and two decoder calls with finite local and
remote outputs. Its 34-ID hand-built “Italian-like” sequence omitted the
`piper-phonemize==1.1.0` policy of inserting pad ID `0` after BOS and after
every phoneme. Faster-Whisper Base decoded both resulting WAVs as
`partire qui sono`, not the claimed sentence.

The resulting 18,176-sample / 0.824308-second waveform and 10,240-sample /
0.464399-second first decoder output are therefore retained only as low-level
accumulated propagation evidence. They do not establish Italian
intelligibility, a production phrase duration, or a production phrase/chunk
boundary. This does not alter the valid 34.311 ms component sum or the
1,449/1,449 NPU placement result above.

### Corrected production chain

The corrected production fixture contains 79 IDs generated by the pinned
phonemizer, including the exact interspersed-pad policy and voice-config
provenance. Its inference-only run reused all four links above and completed
encoder `jgdz9v065`, SDP `jpy7wdk4p`, Flow `jpy7wdw7p`, and four decoder jobs
`jgn7y96kp`, `jgjwm0ee5`, `jpxdj71lg`, and `jgdz9rrz5`. No compile, link, or
profile stage was repeated.

The remote result contains 139 latent frames and 35,584 finite samples /
1.613787 seconds. The component-chain verdict is a strict numerical warning:
Flow NRMSE is 0.001437 and cosine is 0.9999990, but one of 294,912 values fails
`allclose(atol=0.03, rtol=0.03)`. The padded tail remains exactly zero. The
final stitched waveform separately **passes** the same gate: NRMSE 0.024622,
cosine 0.9998499, max absolute error 0.023516, and 0 of 35,584 samples fail.

A two-thread, fixed-locale Italian Faster-Whisper Small INT8 backcheck recovered
`mi fa piacere che lei sia venuto` exactly from both local and remote WAVs,
with 0.0 WER/CER and language probability 1. The first Base attempt failed
before inference because its local cache lacked `model.bin`; no Base quality
result is claimed. This is a conditional one-phrase intelligibility pass, not
human listening or broad Italian acceptance.

The existing component profiles imply 34.311 ms of neural arithmetic through
the first decoder call and 39.774 ms through all four calls. The first official
40-frame payload is 10,240 samples / 0.464399 seconds of playable audio. These
are compute and payload facts, not measured phone text-to-first-audio latency.
Full comparison detail, every inference ID, and caveats are in
[`../piper_v81/PRODUCTION_CHAIN_RESULT.md`](../piper_v81/PRODUCTION_CHAIN_RESULT.md).

## Failure provenance

Relevant verbatim service/tool messages:

- Supertonic vocoder first compile, `jgjwj4q85`: `Tensors {'wav_tts'} occur in value_info but also in model IO. See https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs.`
- Earlier Supertonic vector upload attempts: `Connection was closed before we received a valid response from endpoint URL` and `EOF occurred in violation of protocol (_ssl.c:2427)`.
- Earlier Piper link poll interruption: `Failed to resolve 'workbench.aihub.qualcomm.com' ([Errno 8] nodename nor servname provided, or not known)` followed by `Could not connect to Qualcomm AI Hub.` The already-created link job later succeeded.
- Piper flow generic fixture: all 294,912 local outputs were NaN because the
  dense-random attention drove every projected log above Float32's exponential
  range. This invalid fixture is retained alongside the passing structured
  inference-only recheck; it is not relabeled as a model mismatch.
- Piper production ASR Base attempt, verbatim except for local-path redaction:
  `RuntimeError: Unable to open file 'model.bin' in model '<redacted local
  faster-whisper-base snapshot>'`. This happened before transcription; the
  successful Small-only backcheck is reported separately.

Full exception tracebacks, job URLs, downloaded logs, local goldens, and remote
outputs are retained in the external experiment-state directories. The public
report omits host-specific absolute paths and temporary presigned upload URLs
while preserving every reproducibility-relevant job ID and failure message.

## Decision

- HTP v81 compute speed is comfortably inside a 2–3 second first-audio budget:
  the fixed-shape four-step Supertonic neural sum is about 119 ms.
- Supertonic now has conditional one-phrase intelligibility passes in English
  and Japanese: both accumulated chains preserve the complete reference under
  fixed-locale ASR despite strict numerical failures. Keep it as a
  fallback/challenger until broader phrase/language listening and phone
  execution pass.
- Piper is the strongest Qualcomm-native fallback for EN/DE/IT. The English
  release contexts are exact-target portable with a 34.337 ms acoustic-core
  sum and zero fallback, but remain profile-only. All four Italian neural
  components pass exact-QRD placement and valid numerical gates, with a 34.311
  ms one-call neural sum. The corrected 79-ID production chain adds a strict
  final-audio pass and conditional exact one-phrase Italian ASR pass, while
  retaining a one-element Flow warning. Broader listening, phone runtime
  integration, warm repeated scheduling, and physical audible-start
  measurement remain open.

## Whisper Small fresh-QRD follow-up

A final high-value ASR attempt targeted the official
`openai/whisper-small` encoder. This is the multilingual checkpoint, not an
English-only `.en` model. The installed Qualcomm metadata explicitly describes
multilingual transcription/translation; its capabilities are file-based input,
language detection, and no streaming. The fixed encoder input is
`[1,80,3000]` log-mel features, representing at most 30 seconds of 16 kHz
audio. The benchmark decoder limit is 200 tokens. Qualcomm lists 102M encoder
parameters (391 MB float) and 139M decoder parameters (533 MB float).

No fresh account-owned job could be submitted cleanly:

1. The official release fetcher returned, verbatim,
   `No release assets match the given filters.` for a float QNN context binary
   targeting `Snapdragon 8 Elite Gen 5 QRD`, QAIRT 2.45. There is also no float
   ONNX release asset exposed by this installed release.
2. The published Samsung S26 reference job `jpynlm7lg` is not usable as an
   account-owned QRD run. The API returned, verbatim,
   `ID 'jpynlm7lg' could not be found. It may not exist or you may not have permission to view it.`
3. Fresh official exporters for both `whisper_small` and fallback
   `whisper_base` terminate before model load or cloud submission under
   `qai-hub-models==0.58.0`, `qai-hub-models-cli==0.58.0`. The exact validation
   failure is:
   `Value error, Model cannot be published: no release assets available`.

That original result remains the authoritative blocker for a fresh float
source export/compile: no custom exporter workaround was started, and the
published Samsung S26 float encoder remains reference-only.

A later release-context route changed the W8A16 result without rewriting that
history. The obtainable public Galaxy S26 `sm8850-ad` W8A16 package was pinned
and its two QAIRT-2.45 HTP-v81 contexts profiled unchanged on exact plain
SM8850. Encoder model `mq3lzg43q`, job `jp8497lx5`, reached **186.277 ms** and
**145.727 MiB** with **2,014/2,014** HTP operations. Decoder model `mqpre1ljm`,
job `jgdz9rnr5`, reached **3.932 ms per call** and **146.398 MiB** with
**2,853/2,853** HTP operations. Both report zero fallback. Encoder plus 20
decoder calls is **264.917 ms** of component arithmetic, not end-to-end
transcript latency. The package has no matching sample input/golden, so this
is cross-SKU context portability and placement evidence, not a fresh compile,
numerical/transcript equivalence, phone runtime, or streaming-ASR result. See
[`../whisper_v81/RESULT.md`](../whisper_v81/RESULT.md).

## Reproducibility entry points

- `scripts/prepare_supertonic_aihub.py`: deterministic fixed-shape inputs and
  local goldens.
- `scripts/run_aihub_supertonic_v81.py` and
  `scripts/resume_aihub_supertonic_v81.py`: initial submission and state/job-ID
  reuse, including the narrow ONNX metadata repair for a failed compile.
- `scripts/run_aihub_tts_v81.py` and `scripts/resume_aihub_tts_v81.py`: official
  Piper component compile, context link, profile, inference, and comparison.
- `scripts/profile_piper_en_release_context.py`: pinned, profile-only reuse of
  the six public Piper-English contexts on exact SM8850; it refuses compile or
  inference work when the release has no matching inputs/goldens.
- `scripts/run_aihub_piper_flow_valid_fixture.py`: inference-only flow recheck
  using official serialized upstream components and a valid monotonic fixture.
- `scripts/profile_whisper_release_context.py`: pinned, resume-safe W8A16
  release-context profile on exact SM8850; it submits no inference without
  matching sample inputs.

The resume commands require their persistent `state.json` run directory and
must be launched inside detached `tmux`. They never resubmit a successful
compile/profile/inference stage whose job ID is already present.
