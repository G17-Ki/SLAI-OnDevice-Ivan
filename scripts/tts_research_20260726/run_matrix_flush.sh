#!/bin/bash
# Third matrix: first-flush latency profile plus the MeloTTS INT8 question.
#
# The flush sweep synthesizes single segments of graded length so the
# orchestrator can be told "flush at N characters" for a given budget, instead
# of inferring it from mixed-length fixture cases.
set -uo pipefail
: "${TTS_RESEARCH_MODEL_ROOT:?}"; : "${RUN_ROOT:?}"; : "${VENV_PY:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
HERE="$(cd "$(dirname "$0")" && pwd)"; BENCH="$HERE/bench.py"; OUT="$RUN_ROOT/bench"

run() {
  local name="$1"; shift
  echo "=================== $name"
  "$VENV_PY" "$BENCH" --warmup 1 "$@" 2>&1 | tee "$RUN_ROOT/logs/bench_${name}.log" | tail -8
}

# Does INT8 make the only MIT, eSpeak-free Mandarin voice CPU-viable?
run melotts_zh_int8 --engine melotts_zh_int8 --suite zh --repeats 5 --sid 0 \
    --chunk-floor-sweep --flush-sweep --output-dir "$OUT/melotts_zh_int8"

# Flush-length profiles for the candidates still in contention.
run flush_matcha    --engine matcha_zh_baker --suite zh --repeats 0 --flush-sweep --no-wav --output-dir "$OUT/flush_matcha_zh"
run flush_piper_zh  --engine piper_zh_huayan --suite zh --repeats 0 --flush-sweep --no-wav --output-dir "$OUT/flush_piper_zh"
run flush_melotts   --engine melotts_zh      --suite zh --repeats 0 --flush-sweep --no-wav --output-dir "$OUT/flush_melotts_zh"
run flush_kokoro32  --engine kokoro_zh_fp32  --suite zh --repeats 0 --sid 3 --flush-sweep --no-wav --output-dir "$OUT/flush_kokoro_fp32"
run flush_st_s4     --engine supertonic3     --suite ru --repeats 0 --sid 6 --num-steps 4 --tag _s4 --flush-sweep --no-wav --output-dir "$OUT/flush_supertonic_s4"
run flush_st_s5     --engine supertonic3     --suite ru --repeats 0 --sid 6 --num-steps 5 --tag _s5 --flush-sweep --no-wav --output-dir "$OUT/flush_supertonic_s5"
run flush_st_s8     --engine supertonic3     --suite ru --repeats 0 --sid 6 --num-steps 8 --tag _s8 --flush-sweep --no-wav --output-dir "$OUT/flush_supertonic_s8"
run flush_piper_ru  --engine piper_ru_dmitri --suite ru --repeats 0 --flush-sweep --no-wav --output-dir "$OUT/flush_piper_ru_dmitri"

echo "FLUSH_MATRIX_COMPLETE"
