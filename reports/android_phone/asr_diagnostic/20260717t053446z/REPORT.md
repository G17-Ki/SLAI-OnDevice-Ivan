# Native Android 18-utterance ASR diagnostic

This is the same dummy LibriSpeech subset, deterministic noise seed, competing-speech pairing, and text normalizer used by the earlier Colab diagnostic. WAV serialization is PCM16 for the Android HTTP interface.

| Runtime | Condition | WER | Mean first text | Median total | VmHWM | Failures |
|---|---|---:|---:|---:|---:|---:|
| repack_ctx512 | clean | 3.57% | 560.1 ms | 802.3 ms | 1919.3 MiB | 0 |
| repack_ctx512 | gaussian_10db | 7.14% | 837.3 ms | 1142.5 ms | 1919.3 MiB | 0 |
| repack_ctx512 | competing_speech_5db | 54.29% | 836.9 ms | 1199.6 ms | 1919.3 MiB | 0 |
| no_repack_ctx512 | clean | 3.57% | 1885.5 ms | 2325.6 ms | 1365.3 MiB | 0 |
| no_repack_ctx512 | gaussian_10db | 6.90% | 1913.9 ms | 2344.1 ms | 1365.3 MiB | 0 |
| no_repack_ctx512 | competing_speech_5db | 54.29% | 1917.6 ms | 2422.1 ms | 1365.3 MiB | 0 |

## No-repack quality delta

| Condition | Baseline WER | No-repack WER | Delta | 0.3 pp gate |
|---|---:|---:|---:|---|
| clean | 3.57% | 3.57% | +0.00 pp | PASS |
| gaussian_10db | 7.14% | 6.90% | -0.24 pp | PASS |
| competing_speech_5db | 54.29% | 54.29% | +0.00 pp | PASS |

This 18-utterance suite is a controlled diagnostic, not a publication-scale benchmark. Phone results are not AR1 or glasses results.
