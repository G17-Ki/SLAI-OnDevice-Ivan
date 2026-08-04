# Native Android 18-utterance ASR diagnostic

This is the same dummy LibriSpeech subset, deterministic noise seed, competing-speech pairing, and text normalizer used by the earlier Colab diagnostic. WAV serialization is PCM16 for the Android HTTP interface.

| Runtime | Condition | WER | Mean first text | Median total | VmHWM | Failures |
|---|---|---:|---:|---:|---:|---:|
| memory_gate_ub128_q4kv | clean | 3.10% | 2007.5 ms | 2379.3 ms | 1320.2 MiB | 0 |
| memory_gate_ub128_q4kv | gaussian_10db | 7.14% | 2303.3 ms | 2887.9 ms | 1320.2 MiB | 0 |
| memory_gate_ub128_q4kv | competing_speech_5db | 56.19% | 2347.5 ms | 3029.0 ms | 1320.2 MiB | 0 |

## No-repack quality delta

| Condition | Baseline WER | No-repack WER | Delta | 0.3 pp gate |
|---|---:|---:|---:|---|

This 18-utterance suite is a controlled diagnostic, not a publication-scale benchmark. Phone results are not AR1 or glasses results.
