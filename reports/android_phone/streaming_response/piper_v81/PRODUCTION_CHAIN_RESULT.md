# Piper Italian production-fixture chain on SM8850 / HTP v81

Date: 2026-07-23 08:19 CST<br>
Target: Snapdragon 8 Elite Gen 5 QRD, Android 16, SM8850, HTP v81,
QAIRT 2.45

## Verdict

The corrected 79-ID Italian fixture completed the full inference-only Piper
chain with finite outputs and a **strict final-waveform pass**. The overall
chain retains a **component numerical warning** because one Flow element fails
`allclose(atol=0.03, rtol=0.03)`. This warning does not propagate into a final
audio failure: all 35,584 waveform samples pass the same gate.

A fixed-locale Faster-Whisper Small INT8 backcheck transcribed both the local
reference and exact-QRD remote WAV as `mi fa piacere che lei sia venuto`, with
0.0 WER, 0.0 CER, and Italian language probability 1. This is a conditional
one-phrase intelligibility pass. It is not a human-listening, naturalness,
speaker-similarity, broad-text, phone-runtime, or audible-start acceptance.

## Fixture and execution

The phrase is `Mi fa piacere che lei sia venuto.`. Pinned
`piper-phonemize==1.1.0` generated 79 IDs, including pad ID `0` after BOS and
after every phoneme. The exact IDs, reverse-decoded symbols, voice-config
revision, and hashes are in [`production_fixture.json`](production_fixture.json).

The run reused the four previously linked contexts. It did not compile, link,
profile, or quantize any graph.

| Stage | Reused link | Inference job | Result |
|---|---|---|---|
| Encoder | `jprnkxrk5` | `jgdz9v065` | Success |
| SDP | `jp84ojzx5` | `jpy7wdk4p` | Success |
| Flow | `jpy7ek88p` | `jpy7wdw7p` | Success with comparison warning |
| Decoder 0 | `jp1vrkn8p` | `jgn7y96kp` | Success |
| Decoder 1 | `jp1vrkn8p` | `jgjwm0ee5` | Success |
| Decoder 2 | `jp1vrkn8p` | `jpxdj71lg` | Success |
| Decoder 3 | `jp1vrkn8p` | `jgdz9rrz5` | Success |

The local preparation and cloud-result poller peaked at 0.373 GiB process-tree
RSS under a 2.5 GiB fail-closed limit. The terminal memory-pressure level was
1. The run produced 139 latent frames, four decoder calls, and 35,584 samples
at 22.05 kHz: 1.613787 seconds of audio.

Retained artifact SHA-256 values are:

- local chain NPZ: `47bea642a79debe259eaa366f6d80d89ccbd54e9760601ff6a91cc41efcb0fff`;
- remote chain NPZ: `13ca9e5b9f73d017111f23c7c647597f2791deefd0627b0a051671e5bd5c4f72`;
- local reference WAV: `2089b39a2307682d227d474a723bbded4cc8cf5f751728ac88e8587c9a3650a4`;
- remote chain WAV: `4ac888f09580adab95a483717db66a0915f9253fb15379f02c21f28b895285a9`.

## Numerical comparison

Every retained local and remote array is finite. Encoder and SDP pass their
saved comparisons. The exact SDP duration is 139 frames.

| Output | Gate | NRMSE | Cosine | Max absolute error | Element failures |
|---|---|---:|---:|---:|---:|
| Encoder `x_encoded` | Pass | 0.000663 | 0.9999998 | 0.002424 | 0 |
| Encoder `m_p` | Pass | 0.000656 | 0.9999998 | 0.006736 | 0 |
| Encoder `logs_p` | Pass | 0.000267 | 1.0000000 | 0.001082 | 0 |
| Encoder `x_mask` | Pass | 0 | 1.0000000 | 0 | 0 |
| SDP `y_lengths` | Pass | 0 | 1.0000000 | 0 | 0 |
| SDP `w_ceil` | Pass | 0 | 1.0000000 | 0 | 0 |
| Flow `z` | **Warning** | 0.001437 | 0.9999990 | 0.144056 | **1 / 294,912** |
| Final audio | **Pass** | 0.024622 | 0.9998499 | 0.023516 | **0 / 35,584** |

The Flow warning is extremely sparse. Only one element fails the combined
absolute-plus-relative tolerance: 0.000339% of the fixed tensor, or 0.003747%
of the 26,688 active-frame values. The padded tail is exactly zero locally and
remotely. The reported maximum absolute error is not itself the failing
element because its large golden magnitude gives it a wider relative
tolerance. The sole violation is `z[0,148,22]`: local `0.797302`, remote
`0.853516`, absolute error `0.056214`, permitted error `0.053919`.

All four raw decoder-call outputs independently pass the same 3%/3% gate.
Their waveform NRMSE values are 0.022918, 0.021301, 0.026435, and 0.034540;
cosine similarities are 0.9998413, 0.9998972, 0.9998574, and 0.9999718.
After official overlap slicing and stitching, final-audio mean absolute error
is 0.002027, RMSE is 0.002786, and 0 of 35,584 samples violate the gate.

The verdict is therefore deliberately split:

- chain execution: pass;
- all arrays finite: pass;
- strict all-component numerical gate: warning from one Flow value;
- strict final-audio gate: pass;
- one-phrase ASR intelligibility sanity check: conditional pass.

## ASR backcheck

The originally queued two-model backcheck did not produce a Base transcript.
Faster-Whisper Base failed before inference because its local cache snapshot
was incomplete. The terminal error, verbatim except for the required public
redaction of the local path, was: `RuntimeError: Unable to open file
'model.bin' in model '<redacted local faster-whisper-base snapshot>'`. No Base
quality result is claimed.

A separate already-cached Small-only run then completed with two CPU threads,
one worker, INT8 compute, beam size 5, fixed locale `it`, VAD disabled, and no
cross-segment text conditioning.

| WAV | Transcript | WER | CER | Italian probability | ASR elapsed |
|---|---|---:|---:|---:|---:|
| Local fixed-shape reference | `mi fa piacere che lei sia venuto` | 0.0 | 0.0 | 1.0 | 15.558 s |
| Exact-QRD remote chain | `mi fa piacere che lei sia venuto` | 0.0 | 0.0 | 1.0 | 10.531 s |

ASR elapsed time is a low-priority Mac CPU validation cost, not part of Piper
synthesis latency and not a phone-ASR benchmark.

## Latency and streaming interpretation

The existing exact-QRD component profiles are encoder 16.958 ms, SDP 6.602 ms,
Flow 8.930 ms, and decoder 1.821 ms, with all 1,449 reported runtime layers on
NPU and zero fallback. One encoder + SDP + Flow + first decoder call is
34.311 ms of profiled neural arithmetic. This production phrase needs four
decoder calls, so the corresponding full-phrase arithmetic is
`16.958 + 6.602 + 8.930 + 4 × 1.821 = 39.774 ms`.

The first official 40-frame payload contains 10,240 samples, or 0.464399
seconds of playable audio. That is a valid production-fixture payload size,
not a measured text-to-first-audio latency. The 34.311/39.774 ms arithmetic
excludes phonemization, context loading, host orchestration, tensor transfers,
runtime scheduling, Android integration, queueing, audio-device startup, and
speaker output. It supports a strong feasibility case for streaming Piper but
does not yet prove the 2–3 second physical-phone target.

## Earlier invalid 34-ID fixture

The earlier accumulated chain is retained only as low-level propagation
evidence. Its hand-built 34-ID sequence omitted Piper's required interspersed
pad IDs and produced an over-compressed 0.824308-second waveform. Faster-
Whisper Base decoded both its local and remote WAV as `partire qui sono`.
Its 0.464399-second first decoder payload must not be presented as proof of the
expected Italian sentence or as a production phrase boundary. Nothing in the
corrected result rewrites or hides that invalid-fixture provenance.

## Decision

Piper remains the strongest Qualcomm-native standalone TTS candidate for the
Italian phone route. The corrected chain now adds a final-waveform numerical
pass and a conditional exact one-phrase ASR pass to the existing full-NPU
component evidence. The next acceptance steps are representative Italian
sentences, human listening, direct Nubia execution, repeated warm-turn
scheduling, and physical audible-onset measurement.
