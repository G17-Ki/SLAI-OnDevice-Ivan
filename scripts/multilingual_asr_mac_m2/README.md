# Multilingual ASR Mac M2 benchmark

This isolated harness compares multilingual Whisper Tiny, Base, and Small with the same
21 real Common Voice 17.0 clips: three each in English, Spanish, German, French,
Japanese, Korean, and Mandarin Chinese.

The fixture is pinned to
`argmaxinc/common_voice_17_0-debug-zip@e77e410d41a77f733efbaf7307bd0ff7ebd6d0af`.
Its repository metadata declares CC0-1.0. The model conversions and revisions are
also pinned in `download_assets.py`.

## Run

Use Python 3.11 on Apple Silicon:

```bash
python3.11 -m venv .venv-asr
source .venv-asr/bin/activate
python -m pip install -r scripts/multilingual_asr_mac_m2/requirements.txt
PYTHON_BIN="$PWD/.venv-asr/bin/python" \
  scripts/multilingual_asr_mac_m2/run_mac_benchmark.sh
```

Override `ASR_REPEATS` or `ASR_CPU_THREADS` if needed. The default profile uses
three warm repeats, four CPU threads, INT8 CTranslate2, greedy decoding, and both
fixed-language and automatic-language-detection modes.

The default run limits Small to fixed-language mode, matching the recommended
phone configuration and bounding the larger model's benchmark time.

`benchmark_faster_whisper.py` also supports `--decoder-profile quality_beam5`.
That profile uses beam size five plus standard Whisper fallback and timestamp
termination. It is useful for checking whether a poor greedy result reflects the
checkpoint or merely the cheapest decoding policy.

Audio decoding is performed before the timed region because a phone microphone
already supplies PCM. `first_segment_ms` is time to the first stable public
Faster-Whisper segment; it is not internal decoder first-token latency.
