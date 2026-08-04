#!/bin/bash
# Full host quality/latency matrix for the 2026-07-26 multilingual TTS run.
#
# One engine per process, sequentially, so exactly one model is resident at a
# time. Requires TTS_RESEARCH_MODEL_ROOT, RUN_ROOT, and VENV_PY in the
# environment.
set -uo pipefail

: "${TTS_RESEARCH_MODEL_ROOT:?set TTS_RESEARCH_MODEL_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${VENV_PY:?set VENV_PY to the harness python}"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$HERE/bench.py"
OUT="$RUN_ROOT/bench"
REPEATS="${REPEATS:-5}"

run() {
  local name="$1"; shift
  echo "=================== $name"
  "$VENV_PY" "$BENCH" --repeats "$REPEATS" --warmup 1 "$@" \
    2>&1 | tee "$RUN_ROOT/logs/bench_${name}.log" | tail -14
  echo "--- rc=${PIPESTATUS[0]} for $name"
}

# ---- Mandarin -------------------------------------------------------------
run melotts_zh        --engine melotts_zh      --suite zh --sid 0 --chunk-floor-sweep --output-dir "$OUT/melotts_zh"
run matcha_zh_baker   --engine matcha_zh_baker --suite zh --sid 0 --chunk-floor-sweep --output-dir "$OUT/matcha_zh_baker"
run piper_zh_huayan   --engine piper_zh_huayan --suite zh --sid 0 --chunk-floor-sweep --output-dir "$OUT/piper_zh_huayan"
run kokoro_zh         --engine kokoro_zh       --suite zh --sid 3 --chunk-floor-sweep --output-dir "$OUT/kokoro_zh"
run espeak_zh         --engine espeak_ng       --suite zh --chunk-floor-sweep          --output-dir "$OUT/espeak_zh"

# ---- Russian --------------------------------------------------------------
run supertonic_ru_s4  --engine supertonic3     --suite ru --sid 6 --num-steps 4 --tag _s4 --chunk-floor-sweep --output-dir "$OUT/supertonic_ru_s4"
run supertonic_ru_s5  --engine supertonic3     --suite ru --sid 6 --num-steps 5 --tag _s5 --chunk-floor-sweep --output-dir "$OUT/supertonic_ru_s5"
run piper_ru_dmitri   --engine piper_ru_dmitri --suite ru --chunk-floor-sweep --output-dir "$OUT/piper_ru_dmitri"
run piper_ru_ruslan   --engine piper_ru_ruslan --suite ru --chunk-floor-sweep --output-dir "$OUT/piper_ru_ruslan"
run piper_ru_irina    --engine piper_ru_irina  --suite ru --chunk-floor-sweep --output-dir "$OUT/piper_ru_irina"
run espeak_ru         --engine espeak_ng       --suite ru --chunk-floor-sweep --output-dir "$OUT/espeak_ru"

# ---- English / FR / IT / ES coverage --------------------------------------
for lang in en fr it es; do
  run "supertonic_${lang}" --engine supertonic3 --suite "$lang" --sid 6 --num-steps 4 --tag _s4 --output-dir "$OUT/supertonic_${lang}"
  run "espeak_${lang}"     --engine espeak_ng   --suite "$lang" --output-dir "$OUT/espeak_${lang}"
done

echo "MATRIX_COMPLETE"
