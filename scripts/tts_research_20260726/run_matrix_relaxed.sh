#!/bin/bash
# Second matrix: candidates unlocked by a relaxed 1-2 s first-audio budget.
#
# Requires TTS_RESEARCH_MODEL_ROOT, RUN_ROOT, VENV_PY. Same one-model-per-process
# discipline as run_matrix.sh.
set -uo pipefail
: "${TTS_RESEARCH_MODEL_ROOT:?}"; : "${RUN_ROOT:?}"; : "${VENV_PY:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$HERE/bench.py"; OUT="$RUN_ROOT/bench"; REPEATS="${REPEATS:-5}"

run() {
  local name="$1"; shift
  echo "=================== $name"
  "$VENV_PY" "$BENCH" --repeats "$REPEATS" --warmup 1 "$@" \
    2>&1 | tee "$RUN_ROOT/logs/bench_${name}.log" | tail -12
}

# Mandarin: is Kokoro's cost the model or the INT8 overhead? And how good is a
# modern zero-shot flow-matching model when latency is allowed to be 1-2 s?
run kokoro_zh_fp32 --engine kokoro_zh_fp32 --suite zh --sid 3 --chunk-floor-sweep --output-dir "$OUT/kokoro_zh_fp32"
run zipvoice_zh    --engine zipvoice_zh    --suite zh --chunk-floor-sweep --output-dir "$OUT/zipvoice_zh"

# Russian: more denoising steps buy intelligibility and (per the proxy) quality.
run supertonic_ru_s6 --engine supertonic3 --suite ru --sid 6 --num-steps 6 --tag _s6 --chunk-floor-sweep --output-dir "$OUT/supertonic_ru_s6"
run supertonic_ru_s8 --engine supertonic3 --suite ru --sid 6 --num-steps 8 --tag _s8 --chunk-floor-sweep --output-dir "$OUT/supertonic_ru_s8"

echo "RELAXED_MATRIX_COMPLETE"
