# Native Android 18-utterance ASR diagnostic

This is the same dummy LibriSpeech subset, deterministic noise seed, competing-speech pairing, and text normalizer used by the earlier Colab diagnostic. WAV serialization is PCM16 for the Android HTTP interface.

| Runtime | Condition | WER | Mean first text | Median total | VmHWM | Failures |
|---|---|---:|---:|---:|---:|---:|
| memory_gate_candidate | clean | 2.86% | 1739.6 ms | 2113.4 ms | 1302.9 MiB | 0 |
| memory_gate_candidate | gaussian_10db | 7.14% | 1780.8 ms | 2106.1 ms | 1302.9 MiB | 0 |
| memory_gate_candidate | competing_speech_5db | 55.95% | 1936.4 ms | 2231.8 ms | 1302.9 MiB | 0 |

## No-repack quality delta

| Condition | Baseline WER | No-repack WER | Delta | 0.3 pp gate |
|---|---:|---:|---:|---|

This 18-utterance suite is a controlled diagnostic, not a publication-scale benchmark. Phone results are not AR1 or glasses results.
