# Whisper native frontend audit

## Verdict

Stock kaldi-native-fbank v1.22.3 must not feed the pinned QAI Whisper Small
W8A16 encoder. Its framing, reflection, generated Hann table, generated mel
matrix, recursive FFT, and resulting normalization are not the Hugging Face
frontend contract.

An adapted path with exact Torch framing, the pinned Torch Hann table, the
pinned Hugging Face mel matrix, and cached KissFFT is numerically suitable but
is not a strict byte replacement for PyTorch's FFT on this host:

- English: 99–105 of 240,000 uint16 values differ, depending on compiler and
  arithmetic variant;
- Japanese: 70–75 of 240,000 values differ;
- every adapted difference is exactly one uint16 unit;
- maximum normalized-feature error is `2.3245811462402344e-05`, below one
  half of the release quantization step
  (`4.677007018472068e-05`).

The executable strict gate therefore fails closed. A bounded host gate passes
all adapted variants with maximum uint16 delta `1` and differing fraction at
most `0.001`. That bounded policy is host evidence, not yet a phone acceptance
threshold: the Android arm64 binary must be measured before enabling live-audio
frontend capability.

The practical low-latency route is:

1. keep the current prequantized probe fixture on a strict SHA-256 gate;
2. implement the live frontend with exact framing plus pinned tables and one
   persistent KissFFT plan;
3. run the same EN/JA stage comparator inside the app UID on the SM8850;
4. either accept a target-measured bounded numerical gate plus encoder/output
   parity, or replace only the FFT kernel if strict PyTorch bytes are mandatory.

Do not spend NPU setup/transfer time on this small FFT/mel frontend. The host
implementation processes the full 30-second tensor in about `0.03–0.08 s` in
the target-like fast-math builds. Qualcomm NPU effort should remain on the
Whisper encoder/decoder; the frontend is a CPU/NEON candidate unless phone
profiling proves otherwise.

## Compared implementations

The audit pins two compiler flavors:

- `portable`: `-O2 -fno-fast-math -ffp-contract=off`;
- `android_fast_math`: `-O3 -ffast-math -DNDEBUG`, approximating the cached
  Android KissFFT build flags but still producing a macOS arm64 binary.

Three adapted arithmetic variants were swept:

- `kissfft-direct-scalar`: `real*real + imag*imag`, scalar mel accumulation;
- `kissfft-hypot-scalar`: `hypot(real,imag)^2`, scalar mel accumulation;
- `kissfft-hypot-fma`: `hypot(real,imag)^2`, explicit FMA mel accumulation.

The `hypot-fma` variant most faithfully spells the QAI/HF reference operations.
The direct variant happens to minimize total host uint16 differences, but that
is a numerical coincidence and not a reason to change the reference semantics.

| Route | EN differing uint16 | JA differing uint16 | Maximum delta | Native time per 30 s tensor |
|---|---:|---:|---:|---:|
| fast-math KissFFT direct/scalar | 99 (`0.04125%`) | 71 (`0.02958%`) | 1 | `0.053 s` EN / `0.031 s` JA |
| fast-math KissFFT hypot/FMA | 100 (`0.04167%`) | 74 (`0.03083%`) | 1 | `0.049 s` EN / `0.051 s` JA |
| portable KissFFT hypot/FMA | 105 (`0.04375%`) | 70 (`0.02917%`) | 1 | `0.083 s` EN / `0.075 s` JA |
| portable stock kaldi primitive | 239,995 (`99.9979%`) | 239,994 (`99.9975%`) | 20,113 EN / 13,497 JA | `0.116 s` EN / `0.110 s` JA |

The reference hashes remain:

- EN float32: `8864a2c385f2f768e7719297735cc857a2d3c938c9f775f9e706506f516c6f41`;
- EN uint16: `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1`;
- JA float32: `629ce935d0adb27ca28392f3999d0886353608e2acca0802daca4e466cda0767`;
- corrected JA uint16:
  `8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80`.

## Stage attribution

The adapted path's frame zero is bit-identical to Torch for both fixtures. It
uses frame start `frame*160 - 200`, Torch reflection (`-1 -> 1`), and the
pinned periodic Hann table:

- Hann SHA-256:
  `c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e`;
- mel matrix SHA-256:
  `d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9`.

For portable KissFFT hypot/FMA:

| Stage | EN | JA |
|---|---:|---:|
| Frame-zero differing float elements | 0 / 400 | 0 / 400 |
| Power differing float elements | 58,864 / 603,000 | 43,924 / 603,000 |
| Maximum power error | `3.662109375e-04` | `3.814697265625e-05` |
| Final differing uint16 elements | 105 / 240,000 | 70 / 240,000 |
| Maximum final uint16 delta | 1 | 1 |

Passing the native KissFFT power arrays back through the exact Torch
mel/log/normalization/quantization path produces the same 105 EN and 70 JA
uint16 differences as the fully native hypot/FMA path. The remaining
quantized disagreement in this variant is therefore attributable to the FFT
and power stage, not the pinned mel projection or native postprocessing.

This isolates the minimum strict-parity replacement: the FFT kernel. A
target-verified pocketfft/PyTorch-compatible 400-point real FFT is the next
candidate only if strict live-audio hashes are required. Replacing the whole
frontend library is unnecessary.

## Why stock kaldi-native-fbank fails

The controlled stock mode feeds the same right-padded 480,000 samples through
kaldi-native-fbank's stock frame extraction and `WhisperFeatureComputer` core.
This is already a generous comparison; Sherpa's higher-level feature padding
can introduce additional differences.

Observed stock incompatibilities:

- frame zero starts at `-120`, not `-200`;
- left reflection is `-1 -> 0`, not Torch's `-1 -> 1`;
- frame zero differs in 398/400 EN values and 399/400 JA values;
- generated Hann SHA-256 is
  `2b66ad855bb20e73e657a4df6d8f4a4c840a18431c35cf323f29143f04796c7e`,
  with 318/400 values different from the pinned Torch table;
- generated mel weights differ in 387/16,080 float values from the pinned
  matrix;
- `WhisperFeatureComputer` uses a recursive whisper.cpp-style FFT, not the
  cached KissFFT `Rfft` wrapper;
- `whisper-feature.cc:151` references nonexistent `frame_opts_`; checks-off
  builds hide the bad macro argument, while checks-on builds expose it.

The global max-minus-8 normalization amplifies the early feature mismatch:
all 240,000 normalized EN float values differ, and 239,995 EN uint16 values
differ. Stock kaldi-native-fbank is therefore rejected rather than patched in
place for the QAI encoder.

## Production implementation contract

The Android live frontend should use the following fail-closed contract:

1. normalize PCM as exact float32 `int16 / 32768.0` and right-pad to 480,000;
2. extract 3,000 frames at `frame*160 - 200` with Torch-compatible reflection;
3. multiply by the pinned 400-value Hann table, never regenerate it with
   platform `cos`;
4. allocate one 400-point real KissFFT configuration per frontend instance and
   reuse it for all frames;
5. compute squared complex magnitude and project with the pinned 201x80 mel
   matrix in the documented order;
6. clamp to `1e-10`, apply `log10`, global max-minus-8, then `(x+4)/4`;
7. cast each float32 feature to double before division, round ties to even, add
   zero point `32072`, clamp, and store little-endian uint16;
8. compile the quantizer without fast-math or use an explicitly verified
   ties-to-even implementation;
9. compare live EN and JA target outputs stage by stage before allowing
   `real_audio_transcription=true`.

If the target reproduces the host bound, a practical numerical gate is maximum
uint16 delta `1` and differing fraction below `0.001`, followed by encoder and
decoded-token parity. Do not promote those limits to a release gate until the
SM8850 result exists.

## Artifacts and reproduction

New tracked audit sources:

- `scripts/whisper_native_frontend_audit.cc`;
- `scripts/compare_whisper_native_frontend.py`;
- `scripts/test_compare_whisper_native_frontend.py`.

Generated binaries, reference arrays, native arrays, and the detailed result
JSON are under ignored `artifacts/whisper_native_frontend_audit/`. Current
result JSON SHA-256:
`5dd0ddfe2b3346f19c9b8feab90ad6afb15ea67cdbb137dda506117a33adae9c`.

Run the full EN/JA sweep:

```bash
python3 scripts/compare_whisper_native_frontend.py
```

Run the bounded host gate:

```bash
python3 scripts/compare_whisper_native_frontend.py \
  --require-adapted-bounded-parity
```

Run the strict gate; it currently exits nonzero by design:

```bash
python3 scripts/compare_whisper_native_frontend.py \
  --require-adapted-u16-parity
```

Run focused tests:

```bash
python3 scripts/test_compare_whisper_native_frontend.py
```

Current focused result: 4 tests passed in 3.27 seconds.

## Evidence boundary

This audit proves pinned-source host compilation, exact framing/table input,
stage-level numeric comparison, stock-path rejection, bounded KissFFT
equivalence on macOS arm64, and fail-closed strict-gate behavior.

It does not prove Android ABI behavior, phone latency, QNN delegation, encoder
or decoder output parity, transcript quality, or offline-after-reboot behavior.
No network access, Android build, ADB command, APK mutation, phone mutation,
training, fine-tuning, or host ASR model inference occurred.
