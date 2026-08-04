#!/bin/bash
# Fourth matrix: a stable latency distribution for the Supertonic step sweep.
#
# The 4-vs-5-step recommendation initially rested on a P95 over five runs, and
# the five-step short-clause P95 was driven by a single outlier (0.998 s against
# a 0.629-0.680 s cluster). With n=5 the P95 is essentially the max, which is
# too fragile for a headline decision, so this re-measures latency only - the
# WAVs, ASR and listening clips already exist from the five-repeat runs.
set -uo pipefail
: "${TTS_RESEARCH_MODEL_ROOT:?}"; : "${RUN_ROOT:?}"; : "${VENV_PY:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
HERE="$(cd "$(dirname "$0")" && pwd)"; BENCH="$HERE/bench.py"; OUT="$RUN_ROOT/bench"
REPEATS="${STEP_REPEATS:-15}"

for steps in 4 5 6 8; do
  echo "=================== supertonic latency steps=$steps repeats=$REPEATS"
  "$VENV_PY" "$BENCH" --engine supertonic3 --suite ru --sid 6 \
    --num-steps "$steps" --tag "_s${steps}" --repeats "$REPEATS" --warmup 2 --no-wav \
    --output-dir "$OUT/latency_supertonic_s${steps}" \
    2>&1 | tee "$RUN_ROOT/logs/latency_supertonic_s${steps}.log" | tail -4
done
echo "STEP_MATRIX_COMPLETE"
