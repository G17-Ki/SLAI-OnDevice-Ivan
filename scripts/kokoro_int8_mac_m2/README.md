# Kokoro v1.1-zh INT8 benchmark

This bounded harness measures the Sherpa-ONNX dynamic-INT8 conversion of
Kokoro v1.1-zh with one resident model and two CPU threads.

```bash
scripts/kokoro_int8_mac_m2/run_mac_benchmark.sh
```

The pinned model supports English and Mandarin Chinese. Spanish and Japanese
are deliberately not synthesized as quality cases because neither the model
card nor the embedded ONNX metadata declares them supported. The generic
`espeak-ng-data` directory is not evidence that the trained checkpoint can
speak every espeak language.

The callback is sentence/chunk level, not frame-level vocoder streaming. For
low time-to-first-audio, feed a short first clause and play the first callback
immediately while later clauses synthesize.

Validate every returned PCM chunk before playback. In the main run, the very
short Mandarin warm-up `你好。` returned 26,400/26,400 non-finite samples. A
follow-up sweep reproduced invalid PCM on 10/31 Mandarin calls, including
repeated calls after a discarded warm-up. All tested phrases with at least
three Han characters were finite in that small sweep, but this is only a
provisional chunk floor, not a guarantee. Merge or retry an invalid or shorter
chunk; never enqueue it directly to audio output.

Run the bounded safety diagnostic with:

```bash
python scripts/kokoro_int8_mac_m2/mandarin_initialization_gate.py \
  --model-dir /path/to/kokoro-int8-multi-lang-v1_1 \
  --output reports/android_phone/streaming_response/kokoro_int8_local/mandarin_initialization_gate.json
```

Model weights are downloaded to the ignored `models/` directory and are not
stored in this repository. The downloader verifies the GitHub release digest,
audits archive paths/links before extraction, and verifies the model, voices,
and bundled Apache-2.0 license hashes.

For the optional one-pass intelligibility sanity check, provide the canonical
SenseVoiceSmall INT8 directory and ensure `ffmpeg` is installed:

```bash
.venv-kokoro-int8/bin/python scripts/kokoro_int8_mac_m2/asr_sanity.py \
  --repo-root "$PWD" \
  --model-dir /path/to/sherpa-onnx-sense-voice-int8 \
  --output-dir reports/android_phone/streaming_response/kokoro_int8_local
```

This judge is a transcript sanity check only, not a human naturalness or MOS
evaluation. It uses sequential fixed-locale workers so one ASR model is
resident at a time.
