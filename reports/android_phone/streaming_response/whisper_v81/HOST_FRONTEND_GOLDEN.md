# Whisper Small ORT-QNN host frontend golden

## Result

The host-side preprocessing and decoder-initialization contract is now an
exact, reproducible byte fixture for the pinned Whisper Small W8A16 ORT-QNN
route. The generator validates all source containers before producing six
ignored binary artifacts and one committed manifest.

The committed manifest is
[`whisper_ort_qnn_frontend_golden_v1.json`](whisper_ort_qnn_frontend_golden_v1.json),
SHA-256
`38a4ee71438c052dd3a5e69c8a607049097d59fd8a8d3a002537784aa07e8cce`.
Bulky generated files remain under the ignored directory
`artifacts/whisper_ort_qnn_frontend_golden_v1/`.

| Artifact | Shape / type | Bytes | SHA-256 |
|---|---|---:|---|
| `mel_filters.f32le.bin` | `[201,80]` float32 LE | 64,320 | `d18b48f33345d958ea71098bb990b5e32cebc78d9a2cf65ca0d7302f34ebe8a9` |
| `input_features.u16le.bin` | `[1,80,3000]` uint16 LE | 480,000 | `29c478d40bb98b7782280dcfbc98662cee514df7e62c180c5ba849db24a2f6a1` |
| `attention_mask_step0.u16le.bin` | `[1,1,1,200]` uint16 LE | 400 | `116af0fc853dad9b457ed09d126242b536b4f9258a0b500d22d69cb42a1de71f` |
| `input_ids_step0.i32le.bin` | `[1,1]` int32 LE | 4 | `eea21505fadd32ed72c78271dfbdf37c5bcb00f41b51e44e0e8a8d8eb9dbdfe7` |
| `position_ids_step0.i32le.bin` | `[1]` int32 LE | 4 | `df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119` |
| `self_kv_zero_template.u8.bin` | `[12,1,64,199]` uint8 | 152,832 | `df75a967b954ebddbc0176f41a5564065e193b165586c071481e6cc1bf71aedb` |

## Canonical frontend

The tracked source is `scripts/phone_acceptance_fixtures/question_en.wav`:

- WAV SHA-256:
  `b5fa81a74e608624b52fb3e4cf895d6982a455022597a0da0ad3dc52866f8b69`;
- PCM payload SHA-256:
  `e46de49ab58c33d12ffa8502f8a12d9ed0810bb38ece8f7522be57ef236cff4f`;
- PCM S16LE, mono, 16 kHz, 52,819 frames;
- normalization is exact `float32(int16) / 32768.0`, followed by right
  padding to 480,000 samples.

The explicit frontend uses a periodic 400-sample Hann window, hop 160,
centered reflective-padding `torch.stft`, drops the last STFT frame, squares
magnitudes, applies the Hugging Face Slaney mel matrix, clamps to `1e-10`,
uses `log10`, clamps to global max minus 8, then applies `(x + 4) / 4`.

The resulting float32 tensor is `[1,80,3000]`, range
`-0.6529088020324707..1.3470911979675293`, SHA-256
`8864a2c385f2f768e7719297735cc857a2d3c938c9f775f9e706506f516c6f41`.
Release quantization casts the float32 feature tensor to float64, uses scale
`4.677007018472068e-05`, zero point `32072`, `np.rint` ties-to-even rounding,
and uint16 clamping. The float64 intermediate matches QAI Hub Models 0.55's
runtime wrapper. The final range is `18112..60874`.

The mel matrix is generated from an explicit canonical algorithm instead of
depending on whichever Transformers package happens to be installed. QAI Hub
Models 0.55 requests Transformers 4.56.2 for this model, but that exact wheel
is not cached. The available 4.56.1 wheel and its relevant source members are
hash-pinned in the manifest; their Whisper frontend differs from installed
4.53.3 only in documentation/deprecation-warning behavior for this path.

## Decoder step zero

The release metadata is validated before any bytes are written:

- decode length is 200;
- start token is multilingual SOT `50258`;
- position ID is `0`;
- logical mask value `-100.0` quantizes to `0`;
- logical mask value `0.0` quantizes to `65535`;
- step zero unmasks only index `199`;
- all 12 layers' self-K and self-V logical zeros use release zero point `128`.

One self-K byte template is sufficient because the corresponding self-V
shape has the same element count and all bytes are `128`. The runner reuses
the template for 12 K tensors and 12 V tensors.

## Tokenizer and vocabulary

The exact multilingual tiktoken asset has 50,257 mergeable entries and
SHA-256
`b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126`.
Replaying QAI's `vocab.bin` writer produces 357,313 bytes and exactly matches
the public VoiceAI release hash
`0ba87984671b92e03b56b84ce9b217020663f6a269b5a9800901391430b79c4b`.

The runner must preserve two edge cases:

- token ID `188` is the raw byte `0x00`, even though NUL splitting presents
  it as an empty field;
- token ID `50256` is a true empty token;
- IDs `>=50257` are special/model-control tokens and are skipped during raw
  byte detokenization;
- token bytes are concatenated first and decoded as UTF-8 once, never decoded
  token-by-token.

The manifest includes the mergeable-token reference for the fixture text,
but it intentionally does not claim those are Whisper inference results.

## Evidence boundary

This result proves deterministic frontend bytes, release input quantization,
decoder step-zero bytes, tiktoken-to-VoiceAI-vocabulary equivalence, and safe
UTF-8 detokenization semantics.

It does not prove encoder or decoder execution, QNN/NPU delegation, transcript
quality, multilingual WER/CER, or phone latency. Full expected output tokens
cannot be generated offline because host Whisper-small weights and matching
host encoder/decoder outputs are not cached; no tokens were invented.

The remaining ORT-QNN readiness blockers are unchanged:

1. a runner manifest binding these exact artifacts and release assets;
2. app-UID evidence from the SM8850 phone proving model/session load, encoder
   and decoder execution, QNN backend plus HTP v81 load, frontend-golden and
   vocabulary checks, native multilingual prompting, private-storage loading,
   and offline-after-reboot behavior.

No Android module or accepted APK was edited. No ADB command, network access,
phone mutation, training, fine-tuning, or host model inference occurred.

## Reproduction

Generate or verify the known files:

```bash
python3 scripts/prepare_whisper_ort_qnn_golden.py
```

Run the focused tests:

```bash
python3 scripts/test_prepare_whisper_ort_qnn_golden.py
```

Current focused result: 13 tests passed. The next integration action is to
stage `input_features.u16le.bin` and the decoder step-zero tensors into the
isolated Whisper probe payload, then have the app-owned runner emit the two
missing fail-closed gate records.
