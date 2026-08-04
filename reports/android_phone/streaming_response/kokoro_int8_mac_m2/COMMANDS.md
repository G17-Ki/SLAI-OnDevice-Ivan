# Reproduction commands

The public wrapper creates an isolated Python 3.11 environment, downloads and
verifies the pinned archive, safely extracts it, and runs the two-thread host
benchmark:

```bash
scripts/kokoro_int8_mac_m2/run_mac_benchmark.sh \
  reports/android_phone/streaming_response/kokoro_int8_local
```

The recorded run reused an already downloaded, hash-verified private model
directory and executed the same benchmark payload:

```bash
OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=2 \
python scripts/kokoro_int8_mac_m2/benchmark.py \
  --model-dir /path/to/kokoro-int8-multi-lang-v1_1 \
  --output-dir reports/android_phone/streaming_response/kokoro_int8_mac_m2 \
  --threads 2 \
  --repeats 5 \
  --warmup
```

The one-pass fixed-locale ASR sanity check was run only after TTS latency and
RSS measurement was complete:

```bash
OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=2 \
python scripts/kokoro_int8_mac_m2/asr_sanity.py \
  --repo-root "$PWD" \
  --model-dir /path/to/sherpa-onnx-sense-voice-int8 \
  --output-dir reports/android_phone/streaming_response/kokoro_int8_mac_m2
```

Each ASR locale runs in its own sequential worker process, so only one
SenseVoice model is resident at a time. All TTS and ASR processes had exited at
the end of the run.

The initial evidence serializer wrote undefined warm-up statistics as bare
`NaN`. It was converted to strict JSON without rerunning synthesis:

```bash
python scripts/kokoro_int8_mac_m2/repair_benchmark_json.py \
  reports/android_phone/streaming_response/kokoro_int8_mac_m2/benchmark.json
```
