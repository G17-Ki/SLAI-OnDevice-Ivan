# Whisper ORT-QNN frontend fixture discrepancy audit

## Verdict

The two hashes are not competing encodings of the same audio:

- `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1`
  is the QAI-compatible frontend for the tracked English fixture;
- `05ec7e5f57f2584ae6b5e6aef729ac27ad6c4e37683f6079cde7fb173821ac18`
  was generated from the tracked Japanese fixture.

There is also a real defect in `05ec7e...`: its float features are correct,
but its affine quantization used a float32 division intermediate. QAI Hub
Models 0.55 explicitly casts the float input to float64 before division and
`np.rint`. The corrected QAI-compatible Japanese uint16 hash is
`8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80`.

Therefore the current device probe must use the existing committed English
golden `29c478d4...`. The Japanese `05ec7e...` file is rejected. If Japanese
encoder coverage is required later, regenerate and commit the corrected
`8cf37ba9...` fixture with full source/recipe provenance.

The current host stager and APK verifier still pin `05ec7e...`; no probe APK
should be built from them until those pins are replaced. No APK or phone was
mutated during this audit.

## Source WAV difference

| Property | English golden | Japanese staged file |
|---|---|---|
| Tracked WAV | `question_en.wav` | `question_ja.wav` |
| WAV SHA-256 | `b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69` | `0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761` |
| PCM SHA-256 | `e46de49ab58c33d12ffa8502f8a12d9ed0810bb38ece8f7522be57ef236cff4f` | `439ae3efa582f5da8a277c8e3d1130d3c545f7b9cd89c84d50a695ccb4b773c2` |
| PCM frames | 52,819 | 39,710 |
| Float-feature SHA-256 | `8864a2c385f2f768e7719297735cc857a2d3c938c9f775f9e706506f516c6f41` | `629ce935d0adb27ca28392f3999d0886353608e2acca0802daca4e466cda0767` |
| QAI float64-quantized SHA-256 | `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1` | `8cf37ba9a60f5cd91453dae2d721e9fd7fec69919582223bf02a3f76f8421a80` |

Both WAVs are PCM S16LE, mono, and 16 kHz. Both use exact
`float32(int16) / 32768.0` normalization and right-pad to 480,000 samples;
neither is resampled or truncated.

The English and Japanese float tensors differ in all 240,000 elements because
they are different utterances. Their QAI-compatible uint16 tensors have one
equal element and 239,999 different elements. The maximum uint16 difference is
47,943 and the mean absolute difference is 6,525.3466. This is expected for
different source audio and is not evidence of a frontend disagreement.

## Frontend equivalence

The following are identical for both fixtures:

- tensor shape and layout: C-order `[1,80,3000]`;
- serialized type: little-endian uint16;
- periodic Hann window, `n_fft=400`, hop `160`;
- centered STFT with reflective padding and final-frame removal;
- Hugging Face Slaney mel matrix `[201,80]`, SHA-256
  `d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9`;
- `1e-10` clamp, `log10`, max-minus-8 clamp, and `(x + 4) / 4`;
- input-feature scale `4.677007018472068e-05` and zero point `32072`.

The explicit frontend is byte-identical to Transformers 4.53.3 for both
`return_tensors="np"` and `return_tensors="pt"`:

- English: max absolute difference `0.0`, matching float hash `8864a2c3...`;
- Japanese: max absolute difference `0.0`, matching float hash `629ce935...`.

This matches the QAI Hub Models 0.55 route. Its Whisper app obtains
`WhisperFeatureExtractor` from Hugging Face, and the pinned source member
`qai_hub_models/utils/runtime_torch_wrapper.py` performs quantization as:

1. `input_val.astype(np.float64)`;
2. divide by the release scale;
3. `np.rint`;
4. add the zero point, clamp, and cast to the model dtype.

That source member is 15,775 bytes with SHA-256
`7023b4278eb345f3ba4f055a2f1a767b99b05879eba5bd2d4a28b6e1adb4ee4c`.

## Five-element Japanese quantization defect

NumPy keeps `float32_array / Python_float` in float32 in this environment.
The staged Japanese file used that path and is exactly equal to explicit
float32-intermediate quantization. Five values land on an apparent `.5` in
float32 but lie slightly above or below the half in QAI's float64 arithmetic:

| `[batch,mel,frame]` | Float feature | QAI float64 result | Staged float32 result |
|---|---:|---:|---:|
| `[0,34,143]` | `-0.8696660995` | 13,477 | 13,478 |
| `[0,35,141]` | `-0.7913730145` | 15,151 | 15,152 |
| `[0,57,220]` | `-0.5079931021` | 21,211 | 21,210 |
| `[0,60,218]` | `-0.4513077736` | 22,423 | 22,422 |
| `[0,79,29]` | `-0.2462677956` | 26,807 | 26,806 |

This explains the difference between rejected `05ec7e...` and corrected
`8cf37b...`. The English fixture has no values affected by this precision
boundary, so its float32- and float64-intermediate byte hashes happen to be
the same `29c478d4...`.

## Original OpenAI mel asset is not the QAI byte golden

The original OpenAI-style mel asset bundled with MLX Whisper is numerically
near-identical but not byte-identical to the Hugging Face matrix used by QAI:

- OpenAI/MLX matrix SHA-256:
  `69cc90c55dee453f36546168ab156f7a7ea87681c423f029ad5eb7d8776f21f9`;
- 118 of 16,080 float32 coefficients differ;
- maximum coefficient difference: `1.862645149230957e-09`.

Those tiny differences change one English and two Japanese quantized feature
elements. The corresponding English hash is `91757e...`, not `29c478d4...`.
The QAI release contract therefore requires the Hugging Face Slaney matrix,
not the OpenAI/MLX cached matrix, for exact fixture verification.

## Required fail-closed action

1. Reject `/private/tmp/lmf-question-ja-whisper-small-features.u16le` and its
   `05ec7e...` pin.
2. Use
   `artifacts/whisper_ort_qnn_frontend_golden_v1/input_features.u16le.bin`
   with hash `29c478d4...` for the current probe build.
3. Bind the probe payload manifest to the English WAV SHA, PCM SHA, mel hash,
   float-feature hash, float64 quantization rule, tensor layout, and committed
   golden-manifest hash—not just the final binary hash.
4. If a Japanese probe is needed, generate corrected hash `8cf37ba9...` through
   the same committed generator and add it as a separate named fixture.
5. Keep real-audio transcription and frontend capabilities false until the
   phone actually performs those operations; a prequantized tensor proves
   encoder-input execution only.

All comparisons were lightweight local preprocessing and array analysis. No
model inference, network access, Android build, ADB command, package install,
or phone mutation occurred.
