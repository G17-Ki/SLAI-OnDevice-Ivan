# Native Android 18-utterance ASR diagnostic

This is the same dummy LibriSpeech subset, deterministic noise seed, competing-speech pairing, and text normalizer used by the earlier Colab diagnostic. WAV serialization is PCM16 for the Android HTTP interface.

| Runtime | Condition | WER | Mean first text | Median total | VmHWM | Failures |
|---|---|---:|---:|---:|---:|---:|
| memory_gate_ub32 | clean | 3.57% | 1838.0 ms | 2097.0 ms | 1309.8 MiB | 0 |
| memory_gate_ub32 | gaussian_10db | 7.14% | 2148.1 ms | 2541.9 ms | 1309.8 MiB | 0 |
| memory_gate_ub32 | competing_speech_5db | 55.71% | 2162.3 ms | 2613.5 ms | 1309.8 MiB | 0 |

## No-repack quality delta

| Condition | Baseline WER | No-repack WER | Delta | 0.3 pp gate |
|---|---:|---:|---:|---|

This 18-utterance suite is a controlled diagnostic, not a publication-scale benchmark. Phone results are not AR1 or glasses results.
