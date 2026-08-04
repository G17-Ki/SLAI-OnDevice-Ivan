# Supertonic 3 INT8 benchmark

This directory benchmarks the Sherpa-ONNX INT8 conversion of Supertonic 3 as
a standalone multilingual TTS stage. It does not modify the LFM Android UI.

## Reproduce the Apple Silicon run

```bash
scripts/supertonic3/run_mac_benchmark.sh
```

The wrapper creates an isolated virtual environment, downloads and verifies
the pinned model archive, and runs five repetitions per case. Override paths
with `PYTHON_BIN`, `SUPERTONIC_VENV`, or `SUPERTONIC_MODEL_ROOT`.

To run the optional retranscription smoke check:

```bash
.venv-supertonic3/bin/python -m pip install \
  -r scripts/supertonic3/requirements-asr.txt

.venv-supertonic3/bin/python scripts/supertonic3/asr_sanity.py \
  --benchmark-json reports/android_phone/streaming_response/supertonic3_local/benchmark.json \
  --output-json reports/android_phone/streaming_response/supertonic3_local/asr_sanity.json
```

## Important semantics

- Sherpa emits a callback after a completed text chunk, not on each vocoder
  frame. Start playing that callback immediately and synthesize later chunks
  while it plays.
- Assign `GenerationConfig.extra` as a complete dictionary. Mutating
  `config.extra["lang"]` in place is silently lost by the Python binding.
- The model supports 31 languages but not Mandarin Chinese. Use another TTS
  route, such as MeloTTS, for `zh`.
- The runtime/source code is permissively licensed, but the model weights are
  OpenRAIL-M. The downloader preserves a pinned copy of the model license.
- Model weights are downloaded to `models/` and are intentionally not stored
  in this repository.
