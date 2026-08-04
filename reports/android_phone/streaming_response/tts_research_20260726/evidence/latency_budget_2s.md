### bounded first flush

Earliest safe first-audible time, median / worst (s), against a 2 s budget. `!` marks a worst case over budget.

| Candidate | Lang | 1.0x | 1.5x | 2.0x | cushion needed @1.5x (s) |
|---|---|---:|---:|---:|---:|
| `espeak_ng` | ru | 0.02 / 0.03 | 0.03 / 0.04 | 0.05 / 0.05 | 0.00 |
| `piper_ru_dmitri` | ru | - | - | - | - |
| `piper_ru_dmitri` | ru | 0.19 / 0.57 | 0.28 / 0.85 | 0.37 / 1.13 | 0.00 |
| `piper_ru_irina` | ru | 0.24 / 0.50 | 0.36 / 0.75 | 0.47 / 1.00 | 0.00 |
| `piper_ru_ruslan` | ru | 0.19 / 0.44 | 0.28 / 0.66 | 0.37 / 0.87 | 0.00 |
| `supertonic3_s4` | ru | - | - | - | - |
| `supertonic3_s4` | ru | 0.74 / 1.24 | 1.11 / 1.86 | 1.48 / 2.48 ! | 0.00 |
| `supertonic3_s4` | ru | 0.76 / 1.46 | 1.14 / 2.19 ! | 1.52 / 2.92 ! | 0.00 |
| `supertonic3_s5` | ru | - | - | - | - |
| `supertonic3_s5` | ru | 0.90 / 1.43 | 1.35 / 2.15 ! | 1.80 / 2.87 ! | 0.00 |
| `supertonic3_s5` | ru | 1.02 / 1.83 | 1.53 / 2.74 ! | 2.04 / 3.66 ! | 0.00 |
| `supertonic3_s6` | ru | 1.08 / 2.24 ! | 1.62 / 3.36 ! | 2.16 / 4.48 ! | 0.00 |
| `supertonic3_s6` | ru | 1.06 / 1.74 | 1.59 / 2.62 ! | 2.12 / 3.49 ! | 0.00 |
| `supertonic3_s8` | ru | - | - | - | - |
| `supertonic3_s8` | ru | 1.40 / 3.98 ! | 2.09 / 5.97 ! | 2.79 / 7.96 ! | 0.00 |
| `supertonic3_s8` | ru | 1.38 / 2.15 ! | 2.07 / 3.22 ! | 2.77 / 4.30 ! | 0.00 |
| `espeak_ng` | zh | 0.02 / 0.02 | 0.03 / 0.04 | 0.04 / 0.05 | 0.00 |
| `kokoro_zh` | zh | 3.68 / 4.40 ! | 5.52 / 8.17 ! | 7.36 / 12.17 ! | 3.73 |
| `kokoro_zh_fp32` | zh | - | - | - | - |
| `kokoro_zh_fp32` | zh | 2.05 / 2.98 ! | 3.30 / 5.43 ! | 4.40 / 7.97 ! | 2.58 |
| `matcha_zh_baker` | zh | - | - | - | - |
| `matcha_zh_baker` | zh | 0.15 / 0.27 | 0.23 / 0.40 | 0.31 / 0.53 | 0.00 |
| `melotts_zh` | zh | - | - | - | - |
| `melotts_zh` | zh | 1.05 / 1.94 | 1.83 / 2.93 ! | 3.19 / 4.43 ! | 1.39 |
| `melotts_zh_int8` | zh | 5.66 / 10.18 ! | 8.49 / 16.06 ! | 11.32 / 21.94 ! | 8.49 |
| `piper_zh_huayan` | zh | - | - | - | - |
| `piper_zh_huayan` | zh | 0.28 / 0.80 | 0.42 / 1.20 | 0.56 / 1.60 | 0.00 |
| `zipvoice_zh` | zh | 4.43 / 6.35 ! | 6.65 / 9.53 ! | 8.86 / 12.70 ! | 0.00 |

### long response

Earliest safe first-audible time, median / worst (s), against a 2 s budget. `!` marks a worst case over budget.

| Candidate | Lang | 1.0x | 1.5x | 2.0x | cushion needed @1.5x (s) |
|---|---|---:|---:|---:|---:|
| `espeak_ng` | ru | 0.04 / 0.04 | 0.05 / 0.06 | 0.07 / 0.07 | 0.00 |
| `piper_ru_dmitri` | ru | - | - | - | - |
| `piper_ru_dmitri` | ru | 0.42 / 0.57 | 0.64 / 0.85 | 0.85 / 1.13 | 0.00 |
| `piper_ru_irina` | ru | 0.50 / 0.52 | 0.76 / 0.78 | 1.01 / 1.04 | 0.00 |
| `piper_ru_ruslan` | ru | 0.43 / 0.50 | 0.65 / 0.74 | 0.86 / 0.99 | 0.00 |
| `supertonic3_s4` | ru | - | - | - | - |
| `supertonic3_s4` | ru | 3.00 / 3.28 ! | 4.49 / 4.93 ! | 5.99 / 6.57 ! | 0.00 |
| `supertonic3_s4` | ru | 3.45 / 3.64 ! | 5.17 / 5.45 ! | 6.89 / 7.27 ! | 0.00 |
| `supertonic3_s5` | ru | - | - | - | - |
| `supertonic3_s5` | ru | 3.61 / 4.80 ! | 5.42 / 7.20 ! | 7.23 / 9.61 ! | 0.00 |
| `supertonic3_s5` | ru | 4.20 / 4.58 ! | 6.30 / 6.88 ! | 8.40 / 9.17 ! | 0.00 |
| `supertonic3_s6` | ru | 6.11 / 12.11 ! | 9.16 / 18.16 ! | 12.21 / 24.21 ! | 0.00 |
| `supertonic3_s6` | ru | 4.24 / 4.30 ! | 6.36 / 6.46 ! | 8.48 / 8.61 ! | 0.00 |
| `supertonic3_s8` | ru | - | - | - | - |
| `supertonic3_s8` | ru | 6.64 / 10.16 ! | 9.95 / 15.24 ! | 13.27 / 20.31 ! | 0.00 |
| `supertonic3_s8` | ru | 5.50 / 5.62 ! | 8.24 / 8.43 ! | 10.99 / 11.24 ! | 0.00 |
| `espeak_ng` | zh | 0.04 / 0.04 | 0.06 / 0.07 | 0.08 / 0.09 | 0.00 |
| `kokoro_zh` | zh | 8.94 / 9.24 ! | 20.26 / 20.71 ! | 31.58 / 32.19 ! | 13.57 |
| `kokoro_zh_fp32` | zh | - | - | - | - |
| `kokoro_zh_fp32` | zh | 2.99 / 3.38 ! | 7.19 / 11.76 ! | 14.11 / 20.21 ! | 6.88 |
| `matcha_zh_baker` | zh | - | - | - | - |
| `matcha_zh_baker` | zh | 0.31 / 0.36 | 0.46 / 0.54 | 0.62 / 0.71 | 0.00 |
| `melotts_zh` | zh | - | - | - | - |
| `melotts_zh` | zh | 2.64 / 3.07 ! | 5.01 / 10.43 ! | 10.56 / 17.79 ! | 6.68 |
| `melotts_zh_int8` | zh | 20.41 / 20.69 ! | 36.47 / 36.88 ! | 52.52 / 53.07 ! | 25.80 |
| `piper_zh_huayan` | zh | - | - | - | - |
| `piper_zh_huayan` | zh | 0.34 / 0.52 | 0.50 / 0.79 | 0.67 / 1.05 | 0.00 |
| `zipvoice_zh` | zh | 8.70 / 9.02 ! | 15.04 / 15.57 ! | 24.02 / 24.74 ! | 2.78 |
