#!/bin/zsh

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
if [[ -n ${PYTHON_BIN:-} ]]; then
  python_bin=$PYTHON_BIN
elif command -v python3.11 >/dev/null 2>&1; then
  python_bin=$(command -v python3.11)
elif command -v python3 >/dev/null 2>&1; then
  python_bin=$(command -v python3)
else
  echo "Set PYTHON_BIN to a Python 3.11+ interpreter" >&2
  exit 2
fi
venv=${KOKORO_VENV:-"$repo_root/.venv-kokoro-int8"}
model_root=${KOKORO_MODEL_ROOT:-"$repo_root/models/kokoro-int8-v1_1-zh"}
output_dir=${1:-"$repo_root/reports/android_phone/streaming_response/kokoro_int8_local"}

if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
fi

"$venv/bin/python" -m pip install \
  --disable-pip-version-check \
  --requirement "$script_dir/requirements.txt"

model_dir=$("$script_dir/download_model.sh" "$model_root")

OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=2 \
"$venv/bin/python" "$script_dir/benchmark.py" \
  --model-dir "$model_dir" \
  --output-dir "$output_dir" \
  --threads 2 \
  --repeats 5 \
  --warmup
