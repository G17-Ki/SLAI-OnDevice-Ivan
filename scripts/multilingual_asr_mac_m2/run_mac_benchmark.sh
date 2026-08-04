#!/bin/zsh

set -euo pipefail

script_dir=${0:a:h}
repo_root=${script_dir:h:h}
work_root=${LFM_DEMO_WORK_ROOT:-$repo_root/work}
work_dir=${1:-$work_root/asr-multilingual-mac-m2}
python_bin=${PYTHON_BIN:-python3}
repeats=${ASR_REPEATS:-3}
cpu_threads=${ASR_CPU_THREADS:-4}

"$python_bin" "$script_dir/download_assets.py" --work-dir "$work_dir"

"$python_bin" "$script_dir/prepare_common_voice_subset.py" \
  --archive "$work_dir/downloads/common_voice_17_debug/debug.zip" \
  --metadata "$work_dir/downloads/common_voice_17_debug/metadata.json" \
  --output-dir "$work_dir/dataset"

for model in tiny base small; do
  if [[ "$model" == tiny ]]; then
    repo=Systran/faster-whisper-tiny
    revision=d90ca5fe260221311c53c58e660288d3deb8d356
  elif [[ "$model" == base ]]; then
    repo=Systran/faster-whisper-base
    revision=ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
  else
    repo=Systran/faster-whisper-small
    revision=536b0662742c02347bc0e980a01041f333bce120
  fi
  mode_args=(--modes fixed auto)
  if [[ "$model" == small ]]; then
    mode_args=(--modes fixed)
  fi
  "$python_bin" "$script_dir/benchmark_faster_whisper.py" \
    --manifest "$work_dir/dataset/dataset_manifest.json" \
    --model-dir "$work_dir/models/faster-whisper-$model" \
    --model-name "Whisper-$model" \
    --model-repo "$repo" \
    --model-revision "$revision" \
    --repeats "$repeats" \
    --cpu-threads "$cpu_threads" \
    "${mode_args[@]}" \
    --output "$work_dir/results/whisper_$model.json"
done

"$python_bin" "$script_dir/summarize.py" \
  "$work_dir/results/whisper_tiny.json" \
  "$work_dir/results/whisper_base.json" \
  "$work_dir/results/whisper_small.json" \
  --output-dir "$work_dir/results"
