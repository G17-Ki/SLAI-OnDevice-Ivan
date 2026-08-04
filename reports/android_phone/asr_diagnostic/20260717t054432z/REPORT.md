# Native Android 18-utterance ASR diagnostic

This is the same dummy LibriSpeech subset, deterministic noise seed, competing-speech pairing, and text normalizer used by the earlier Colab diagnostic. WAV serialization is PCM16 for the Android HTTP interface.

| Runtime | Condition | WER | Mean first text | Median total | VmHWM | Failures |
|---|---|---:|---:|---:|---:|---:|
| memory_gate_f16kv | clean | 3.57% | 1736.9 ms | 2089.0 ms | 1307.7 MiB | 0 |
| memory_gate_f16kv | gaussian_10db | 7.14% | 2099.3 ms | 2526.3 ms | 1307.7 MiB | 0 |
| memory_gate_f16kv | competing_speech_5db | 54.76% | 2151.4 ms | 2630.5 ms | 1307.7 MiB | 0 |

## No-repack quality delta

| Condition | Baseline WER | No-repack WER | Delta | 0.3 pp gate |
|---|---:|---:|---:|---|

This 18-utterance suite is a controlled diagnostic, not a publication-scale benchmark. Phone results are not AR1 or glasses results.
