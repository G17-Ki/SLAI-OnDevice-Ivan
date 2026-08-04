# Supertonic accumulated SM8850 chain result

Date: 2026-07-23<br>
Target: Snapdragon 8 Elite Gen 5 QRD, Android 16, SM8850, HTP v81,<br>
QAIRT 2.45

## Decision

The real-text, four-step fixed-shape chain is a **conditional feasibility
pass**. Every remote stage completed, the final waveform is finite, and a
fixed-locale SenseVoiceSmall INT8 backcheck transcribed the dynamic-local,
fixed-local, and accumulated AI Hub WAVs identically with 0.0 normalized WER.

This is not a general audio-quality acceptance. Later denoising steps and the
vocoder fail the configured elementwise `allclose(atol=0.05, rtol=0.05)`
comparison, the test covers only one English phrase/seed/speaker, and no human
MOS, speaker-similarity, or phone playback test has been run. The next quality
gate is a broader multilingual phrase matrix plus human listening, not another
synthetic tensor probe.

## Fixture

- Text: `It is Paris, in France.`
- Preprocessed text: `<en>It is Paris, in France.</en>`
- Seed: 42
- Speaker: 6
- Denoising calls: 4, with `total_step=4`
- Real latent positions: 31
- Fixed latent positions: 32, with one zero-masked padding position
- Output: 93,826 trimmed samples at 44.1 kHz, 2.1276 seconds
- Fixture SHA-256:
  `58b5b7a7a5af083b526029282a35401c8887add5e847ff631bc32457940507d3`

The experiment reused the existing component compiles and submitted inference
jobs only. No graph was recompiled.

## Sequential remote results

| Stage | Existing compile | Inference | NRMSE vs fixed-local | Cosine | `atol=0.05`, `rtol=0.05` gate |
|---|---|---|---:|---:|---|
| Duration predictor | `jp2v287x5` | `jgjwx3d85` | 0.000619 | 1.000000 | Pass |
| Text encoder | `j5mdj79d5` | `j56d6oz7p` | 0.006385 | 0.999980 | Pass |
| Vector step 0 | `j5w1o2k6g` | `jgn7219rp` | 0.003914 | 0.999992 | Pass |
| Vector step 1 | `j5w1o2k6g` | `j5qm84zng` | 0.026508 | 0.999655 | Fail |
| Vector step 2 | `j5w1o2k6g` | `jp0vyoong` | 0.042234 | 0.999108 | Fail |
| Vector step 3 | `j5w1o2k6g` | `j5qm84neg` | 0.065366 | 0.997868 | Fail |
| Vocoder | `jgo4ykxx5` | `jp1v014lp` | 0.467072 | 0.893110 | Fail |

The remote text embedding and each remote vector output were fed forward, and
the final remote latent was sent to the vocoder. The duration output was
checked independently; it did not change the fixture's precomputed fixed
latent length. The vector rows therefore measure accumulated chain drift
rather than independent synthetic component inputs.

## Audio comparison

These metrics were computed from the saved float32 tensors before PCM16 WAV
serialization. The links below identify the serialized listening and ASR
artifacts; recomputing waveform metrics from their quantized PCM samples will
produce slightly different values.

| Remote waveform compared with | Waveform NRMSE | Waveform cosine | Spectral convergence | Log-magnitude RMSE |
|---|---:|---:|---:|---:|
| Fixed-shape local, 32 positions | 0.467038 | 0.893123 | 0.160421 | 0.060112 |
| Dynamic local, 31 positions | 1.150234 | 0.434973 | 0.825993 | 0.308507 |

The remote result is substantially closer to the fixed-shape local graph than
to the dynamic graph. The large dynamic-versus-fixed change therefore cannot
be attributed to AI Hub lowering: padding/topology already changes sample
alignment and waveform shape before compilation.

All three serialized files are valid mono PCM16 WAVs at 44.1 kHz and have the
same 93,826-frame trimmed duration:

- [dynamic local WAV](local_dynamic_trimmed.wav), SHA-256
  `d316f3da32a9543b85f23d3eba283ea49741da1b59bff00081fe10dab2f89f81`
- [fixed-shape local WAV](local_fixed_trimmed.wav), SHA-256
  `5de398beb8ccdc221b3b62dae7e93f08394a18d54d6d57bfed218149f54e60d3`
- [accumulated AI Hub WAV](remote_chain/remote_fixed_trimmed.wav), SHA-256
  `638148f2c416cb0d9b56ddd30f79207caf1171d4a4dbda4d35f1cec80a1e4b3b`

## Intelligibility backcheck

One fixed-locale English SenseVoiceSmall INT8 pass produced:

| WAV | Transcript | Normalized WER |
|---|---|---:|
| Dynamic local | `It is Paris in France.` | 0.0 |
| Fixed-shape local | `It is Paris in France.` | 0.0 |
| Accumulated AI Hub | `It is Paris in France.` | 0.0 |

The judge used two CPU threads. Model loading took 0.788 seconds and the three
recognizer inference calls took 0.059-0.087 seconds. Audio decoding and 16 kHz
resampling took 0.530 seconds on the first call and 0.065-0.067 seconds on the
next two. This is an intelligibility sanity check only; it does not measure
naturalness, pronunciation preference, voice identity, or human-perceived
artifacts.

## Interpretation for the demo

- Prior exact-SM8850 component profiles placed every reported neural layer on
  NPU and measured a 118.891 ms fixed-shape neural component sum, including
  four vector-estimator calls. This remains a lower-bound component sum, not
  end-to-end latency; see the [component report](../aihub_voice_v81/REPORT.md).
- For this fixture, the accumulated chain shows that the ASR judge still
  recovers every reference word despite the numerical drift.
- Elementwise agreement still fails even at `atol=0.05`, `rtol=0.05`, and the
  vocoder-level drift is large, so Supertonic should remain a
  fallback/challenger rather than replacing the already-streaming integrated
  English LFM path.
- The highest-value next run is a small phrase matrix spanning short/long
  utterances and multiple supported languages, followed by blinded listening
  and execution through the Nubia audio path.

## Reproducibility evidence

- [fixture manifest](fixture_manifest.json)
- Resume state is intentionally excluded from the public tree.
- [audio comparison result](remote_chain/result.json)
- [ASR backcheck](asr_sanity.json)
- [SenseVoice judge artifact manifest](../sensevoice_asr_mac_m2/model_manifest.json)
- Runner: `scripts/run_aihub_supertonic_chain_v81.py`
- Fixture builder: `scripts/prepare_supertonic_chain_fixture.py`

The task-local ASR JSON is the one-language worker record and does not repeat
the judge artifact/version fields; the linked model manifest pins the canonical
SenseVoiceSmall INT8 archive and extracted model files used by this project.

Cloud work ran in detached `tmux`. Job IDs and downloaded outputs were written
before advancing to the next stage, so an interrupted poller can resume without
submitting duplicate jobs.
