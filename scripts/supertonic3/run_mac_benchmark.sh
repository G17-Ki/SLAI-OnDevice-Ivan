#!/bin/zsh

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
python_bin=${PYTHON_BIN:-python3}
venv=${SUPERTONIC_VENV:-"$repo_root/.venv-supertonic3"}
model_root=${SUPERTONIC_MODEL_ROOT:-"$repo_root/models/supertonic3-int8"}
output_dir=${1:-"$repo_root/reports/android_phone/streaming_response/supertonic3_local"}

if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
fi

"$venv/bin/python" -m pip install \
  --requirement "$script_dir/requirements.txt"

model_dir=$("$script_dir/download_model.sh" "$model_root")

"$venv/bin/python" "$script_dir/benchmark.py" \
  --model-dir "$model_dir" \
  --output-dir "$output_dir" \
  --threads 2 \
  --repeats 5 \
  --steps 5 \
  --speed 1.05 \
  --sid 6 \
  --warmup
