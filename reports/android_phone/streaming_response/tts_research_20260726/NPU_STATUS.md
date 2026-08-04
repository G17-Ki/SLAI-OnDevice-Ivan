# NPU feasibility status — MeloTTS-ZH on SM8850 / HTP v81

Target for every job: `Snapdragon 8 Elite Gen 5 QRD`, Android 16,
`chipset:sm8850`, `hexagon:v81`, `framework:qnn`, QAIRT 2.45.<br>
Phone: **disconnected**. Nothing here is a Nubia, AR1, or glasses measurement.

## What changed in this run

Before this run, MeloTTS-ZH had an exact-QRD **placement** pass: the pinned
public Voice AI contexts loaded on the QRD with 2,372/2,372 operations on
Hexagon and a 144.839 ms one-call component sum, but no real sentence had ever
been through the chain and there was no numerical or intelligibility claim.

That gap is now closed. A real Mandarin sentence was taken through
preprocessing, BERT features, encoder, host attention construction, flow, **nine
repeated decoder calls with 40-new/12-overlap stitching**, and an ASR
back-check, all on the exact QRD target.

## Fixture

| Property | Value |
|---|---|
| Text | 北京是中国的首都，也是全国的政治和文化中心。 |
| Normalized | 北京是中国的首都,也是全国的政治和文化中心. |
| Phones | 89 |
| Valid BERT tokens | 24 |
| Latent frames (`y_length`) | 351 |
| Decoder calls | **9** |
| Local golden | 179,712 samples @ 44.1 kHz = 4.075 s |
| Host peak RSS during fixture build | 1.40 GiB (cap 2.5 GiB) |

Sources are pinned and verified before use: `qai-hub-models` 0.58.0 wheel and
five of its source members by SHA-256, the MeloTTS checkout at commit
`2091453…`, and all eight Voice AI artifacts by size and SHA-256. The generator
refuses to run if any digest disagrees.

## Result: fixed-local variant — **pass**

Every remote stage received the local golden's inputs, which isolates each
graph's own numerical error.

| Stage | Job | Finite | NRMSE | Cosine |
|---|---|---|---:|---:|
| BERT wrapper `hidden_states` | `jpv7q197p` | yes | 0.85% | 0.9999649 |
| Encoder `m_p` | `jpeyorl75` | yes | 1.22% | 0.9999257 |
| Encoder `logs_p` | `jpeyorl75` | yes | 0.04% | 0.9999999 |
| Encoder `g` | `jpeyorl75` | yes | 0.02% | 1.0000000 |
| Encoder `x_mask` | `jpeyorl75` | yes | 0.00% | exact |
| Encoder `w_ceil` | `jpeyorl75` | yes | **8.32%** | 0.9968528 |
| Encoder `y_lengths` | `jpeyorl75` | yes | **3.13%** | 1.0000000 |
| Flow `z` | `jp162dvk5` | yes | 0.57% | 0.9999837 |
| **Decoder, stitched waveform** | `jp43nx9q5` | **yes** | **2.94%** | **0.9995769** |

Remote waveform: 179,712 samples, 4.075 s, all finite, peak 0.423, maximum
absolute sample deviation from the golden 0.032.

Hub model ids (uploaded previously, reused unchanged): `mngxvev5q`
(bert_wrapper), `mqk5d8d7q` (encoder), `mmxkydy2n` (flow), `mmr8ygo6n`
(decoder).

### Intelligibility back-check

A fixed-locale (`zh`) faster-whisper judge recovered the exact sentence from
**both** waveforms:

| Waveform | Transcript | CER |
|---|---|---:|
| Local golden | 北京是中國的首都,也是全國的政治和文化中心 | **0.0000** |
| Exact-QRD remote | 北京是中國的首都 也是全國的政治和文化中心 | **0.0000** |

(The judge emits traditional Han; the scorer folds it to simplified before
comparison. That folding is applied identically to both sides.)

### The one number worth arguing about

`w_ceil` — the per-phone integer duration — has 8.32% NRMSE with a maximum
absolute deviation of exactly **1.0**, and `y_lengths` differs by exactly **11
frames** out of 351. That is not noise in the usual sense: it is the duration
predictor's rounding landing on the other side of an integer boundary for a
handful of phones after quantization.

It matters because `y_length` decides **how many decoder calls the chain
issues**. A host that computes 351 frames schedules 9 decoder calls; a device
that computes 340 schedules 9 as well, but 11 frames of audio (about 128 ms)
would be missing or duplicated at the tail if the two ever disagreed about the
schedule. Any on-device implementation must derive the decoder schedule from the
**device's own** `y_lengths`, never from a host-computed constant, and must
tolerate the last chunk being short.

This is also the reason the two variants below are run separately.

## Result: accumulated variant — **conditional pass, with a caveat worth stating
carefully**

Run from `aihub-melotts-acc-20260726T053609Z`, feeding each stage the
*previous stage's remote output*: remote BERT hidden states re-expanded to phone
level via the persisted `word2ph`, then remote `m_p`/`logs_p`/`g` into flow, then
remote `z` into the decoder windows. This is the error-propagation case a phone
would actually experience.

| Stage | Job | Fixed-local NRMSE | Accumulated NRMSE | Accumulated cosine |
|---|---|---:|---:|---:|
| BERT `hidden_states` | `jprw64wv5` | 0.85% | 0.85% | 0.9999649 |
| Encoder `m_p` | `jp064162p` | 1.22% | 1.22% | 0.9999256 |
| Flow `z` | `jpv7q177p` | 0.57% | **0.87%** | 0.9999620 |
| **Decoder, stitched waveform** | `jgd2nr2k5` | 2.94% | **35.96%** | 0.9343972 |

Error is essentially flat through BERT and the encoder — the remote BERT hidden
states barely move the encoder output — grows modestly through flow, then jumps
by more than an order of magnitude at the vocoder.

**That 35.96% number should not be read as "the audio broke."** The accumulated
waveform is 179,712 samples, all finite, peak 0.417, and it passes every gate in
this run's suite with statistics almost identical to the golden (RMS 0.0617 vs
0.0628, DC −0.000148 vs −0.000160, longest internal silence 0.046 s vs 0.054 s,
no repeated-window collapse). And the fixed-locale ASR judge recovered the exact
sentence from it at **0.0000 CER**, the same as the golden.

| Waveform | CER | Waveform gates |
|---|---:|---|
| Local golden | **0.0000** | pass |
| Exact-QRD, fixed-local inputs | **0.0000** | pass |
| Exact-QRD, accumulated inputs | **0.0000** | pass |

A sample-wise NRMSE on a vocoder output is dominated by phase and
sample-alignment drift, which destroys the metric long before it destroys the
speech. The honest reading is: **strict waveform NRMSE is the wrong instrument
for judging an accumulated TTS chain**, and the right ones — finiteness, the
waveform gate suite, and intelligibility — all pass. It is equally true that
none of those three is evidence of *naturalness*, which is why the audio is
published for listening rather than declared acceptable.

Audio for an A/B/C listening check is in
[`melotts_chain_audio/`](melotts_chain_audio/) (golden, fixed-local remote,
accumulated remote — same sentence, same length, labelled, not blinded).

Machine-readable results:
[`evidence/melotts_chain_fixed_local.json`](evidence/melotts_chain_fixed_local.json),
[`evidence/melotts_chain_accumulated.json`](evidence/melotts_chain_accumulated.json),
[`evidence/melotts_chain_asr_backcheck.json`](evidence/melotts_chain_asr_backcheck.json).

## What this does and does not prove

**Proven now, on the exact QRD target:**

- the full MeloTTS-ZH graph set runs end to end on real Mandarin text;
- the repeated-decoder schedule with overlap stitching reconstructs a
  continuous, finite 4.075 s waveform;
- the remote waveform is numerically close to the host golden (2.94% NRMSE,
  0.99958 cosine, no non-finite samples);
- the sentence is exactly recoverable by a fixed-locale ASR judge from the
  remote audio.

**Still not proven:**

- **naturalness.** 0.0% CER on one sentence is intelligibility, not voice
  quality, and one sentence is not a voice evaluation.
- **on-device latency.** The 144.839 ms figure is a sum of fixed-shape one-call
  component profiles. This chain issues 1 BERT + 1 encoder + 1 flow + **9**
  decoder calls, so the arithmetic for this sentence is 2.679 + 23.697 + 74.563
  + 9 × 43.900 = **496.0 ms of neural compute**, excluding G2P, host attention
  construction, transfers, context loading, overlap stitching, and playback.
  That is still comfortably inside the 900 ms TTS-stage budget for a 4.075 s
  utterance, but it is neural arithmetic, not text-to-first-audio.
- **runtime compatibility on the Nubia.** These are Qualcomm Voice AI context
  binaries packaged for QAIRT 2.45. The Nubia previously reported vendor QAIRT
  2.37. Sideloading is not justified by this result.
- **memory alongside LFM.** Estimated inference peak per component is 82–126 MiB
  on device, but nothing here measures that with LFM, Qwen, and SenseVoice
  resident.
- **anything about AR1 or glasses.**

## Why this matters for the Mandarin recommendation

MeloTTS-ZH is the only Mandarin candidate that is MIT-licensed, needs no
GPL-3.0 eSpeak data at runtime, and now has an end-to-end exact-QRD chain with a
numerical and intelligibility pass. It is also the Mandarin candidate that
**fails on CPU** — 0.715 RTF and playback gaps at 1.0× host speed.

Those two facts point the same way: MeloTTS-ZH is a viable Mandarin route
*only* on the NPU, and it is the only licence-clean route there is. The
remaining blocker is not the model — it is Voice AI runtime integration and the
QAIRT 2.37/2.45 version gap on the target phone.

## Other candidates' NPU status — unchanged

No new NPU work was done for Supertonic, Piper, Matcha, or Kokoro this run.

- **Supertonic 3**: four fixed-shape components placed on HTP v81; text encoder
  and duration predictor pass their numerical gates, vector estimator (2.253%
  NRMSE) and vocoder (16.110% NRMSE) retain explicit numerical warnings. Not
  promoted to an audio-quality pass.
- **Piper**: exact-QRD Italian four-component core passes at 34.311 ms with
  1,449/1,449 NPU layers and a corrected production chain producing 1.61 s of
  finite audio at 0.0 WER/CER. No Russian or Chinese Piper NPU export exists.
- **Matcha-ZH**: no QNN export attempted. It is the host-quality winner for
  Mandarin but has no NPU path, and its non-commercial corpus makes investing in
  one questionable.
- **Kokoro-ZH**: rejected on host quality; NPU work is not justified.

Under the study rule that NPU work follows host/Colab quality, the only NPU
work in scope for this run was the MeloTTS-ZH chain.

## Reproduce

```bash
tmux new-session -d -s aihub-melotts-chain \
  "$VENV_PY scripts/tts_research_20260726/aihub_melotts_chain.py \
     --fixture-dir /external/melotts-fixture --output-dir /external/run/melotts_fixed_local \
     --variant fixed_local --confirm-cloud"
```

The runner refuses to submit without `--confirm-cloud` and without a `TMUX`
environment, and writes each job id to `state.json` immediately after
submission and before any polling, so an interrupted poller never orphans a
server-side job.
