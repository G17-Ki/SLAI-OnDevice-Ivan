# Natural multilingual TTS research harness (2026-07-26)

Reproducible harness for the Mandarin/Russian offline-voice study that replaces
the robotic Android system-TTS fallback. It measures every candidate the same
way so the numbers are comparable, and it keeps naturalness out of the
automated verdict on purpose.

Model weights, raw outputs, and audio batches are **not** stored in this
repository. Point `TTS_RESEARCH_MODEL_ROOT` at an external directory and write
results to an external run root.

## Environment

Create a dedicated environment instead of installing the research stack into
the system Python:

```bash
python3.12 -m venv .venv-tts-research
source .venv-tts-research/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/tts_research_20260726/requirements.txt
```

The pinned requirements reproduce the local harness, fixed-locale ASR judge,
and DNSMOS proxy environment. The Qualcomm AI Hub chain additionally needs the
repository's `aihub` optional dependencies and valid AI Hub credentials.

## What each part does

| File | Role |
|---|---|
| `fixtures.json` | Fixed assistant-style text suites for zh/ru/en/fr/it/es, plus per-language short-chunk floor sweeps |
| `gates.py` | Fail-closed waveform gates: emptiness, non-finite PCM, near-silence, clipping, DC offset, long internal silence, repeated-window collapse |
| `engines.py` | One adapter per candidate over sherpa-onnx plus an eSpeak NG control, with static licence/footprint metadata |
| `bench.py` | Runs one engine over one suite: latency, RTF, RSS, gates, WAV hashes, playback-supply simulation |
| `asr_judge.py` | Fixed-locale faster-whisper backcheck producing CER (and WER outside Mandarin) |
| `summarize.py` | Merges bench + ASR into one screen table against the fixed study gates |
| `listening_bundle.py` | Loudness-equalized, shuffled, blinded clips plus an HTML/Markdown score sheet |
| `aihub_melotts_chain.py` | Real-Mandarin MeloTTS-ZH chain on the exact SM8850 / HTP v81 QRD |
| `latency_budget.py` | Minimum startup cushion for gap-free playback, and the earliest safe first-audible time it implies, per candidate and slowdown |
| `mos_proxy.py` | DNSMOS speech-quality proxy over the blinded clips (artifact detector, not a naturalness predictor) |
| `asr_wav_backcheck.py` | Fixed-locale ASR back-check for arbitrary WAVs, used for the exact-QRD chain |
| `run_matrix.sh` | Sequential driver for the host matrix, one resident model at a time |
| `run_matrix_relaxed.sh` | Second matrix: candidates that only make sense under a 1–2 s first-audio cap (Kokoro float32, ZipVoice, Supertonic 6/8 steps) |
| `run_matrix_flush.sh` | Third matrix: first-flush length profiles, plus MeloTTS INT8 |
| `test_harness.py` | Deterministic tests for the gates, playback simulation, and ASR normalization |

## Deliberate limits

- **ASR is a screen, not a verdict.** A low CER proves the words survived
  synthesis. It cannot distinguish a natural voice from a robotic one, so
  `summarize.py` can only mark a candidate `blocked` or `conditional`; nothing
  in the automated path promotes a candidate.
- **One model resident at a time.** `bench.py` loads exactly one engine per
  process and samples the whole process tree against a 2.5 GiB cap.
- **Non-finite PCM is never serialized.** A run that fails a hard gate records
  its statistics and withholds the WAV rather than writing audio that a later
  step could mistake for valid output.
- **Loudness is equalized only in the listening bundle**, where an unequalized
  test would measure output gain instead of naturalness. Sample rate is left
  native, because it is a real property of what would ship.
- **Latency is reported as two separate things.** `--flush-sweep` measures how
  long a first flush of N characters costs, which the orchestrator controls;
  `latency_budget.py` measures the cushion an engine needs to keep the playback
  queue fed, which it does not. A candidate needing a 0 s cushion is limited only
  by flush length; one needing seconds of cushion has a throughput deficit that
  no first-audio budget can fix.

## Tests

```bash
python -m unittest discover -s scripts/tts_research_20260726 -p "test_*.py"
```

Each test encodes a defect actually observed in the run — Kokoro's all-non-finite
short-input PCM, MeloTTS's near-silent single character, Kokoro's DC offset, a
starved playback queue — so a regression would silently change a recommendation.

## Reproduce

```bash
export TTS_RESEARCH_MODEL_ROOT=/external/models
export RUN_ROOT=/external/run-root
export VENV_PY="$PWD/.venv-tts-research/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
```

Single candidate:

```bash
"$VENV_PY" scripts/tts_research_20260726/bench.py --engine melotts_zh --suite zh --repeats 5 --chunk-floor-sweep --output-dir "$RUN_ROOT/bench/melotts_zh"
```

Whole host matrix (long; run it detached):

```bash
tmux new-session -d -s tts-matrix "bash scripts/tts_research_20260726/run_matrix.sh > $RUN_ROOT/logs/matrix.log 2>&1"
```

Intelligibility backcheck (Mandarin shown; use `--locale ru` for Russian):

```bash
"$VENV_PY" scripts/tts_research_20260726/asr_judge.py --locale zh --model /path/to/faster-whisper-small --bench-json "$RUN_ROOT/bench/melotts_zh/bench_melotts_zh_zh.json" --output "$RUN_ROOT/asr/zh.json"
```

Screen table and blinded bundle:

```bash
"$VENV_PY" scripts/tts_research_20260726/summarize.py --bench-root "$RUN_ROOT/bench" --asr "$RUN_ROOT/asr/zh.json" --asr "$RUN_ROOT/asr/ru.json" --output "$RUN_ROOT/summary.json" --markdown "$RUN_ROOT/summary.md"

"$VENV_PY" scripts/tts_research_20260726/listening_bundle.py --bench-root "$RUN_ROOT/bench" --language zh --case zh_answer_short --case zh_answer_medium --output-dir "$RUN_ROOT/listening/zh"
```

Exact-QRD MeloTTS-ZH chain (cloud; must be detached and confirmed):

```bash
tmux new-session -d -s aihub-melotts-chain \
  "$VENV_PY scripts/tts_research_20260726/aihub_melotts_chain.py --fixture-dir /external/melotts-fixture --output-dir $RUN_ROOT/aihub/melotts_fixed_local --variant fixed_local --confirm-cloud"
```
