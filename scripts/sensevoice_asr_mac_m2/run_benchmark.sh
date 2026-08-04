#!/bin/zsh

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
work_root=${LFM_DEMO_WORK_ROOT:-$repo_root/work}
work_dir=${1:-$work_root/sensevoice-asr-mac-m2}
python_bin=${PYTHON_BIN:-python3.11}
repeats=${ASR_REPEATS:-3}
cpu_threads=${ASR_CPU_THREADS:-2}

export OMP_NUM_THREADS=$cpu_threads
export OPENBLAS_NUM_THREADS=$cpu_threads
export VECLIB_MAXIMUM_THREADS=$cpu_threads
export NUMEXPR_NUM_THREADS=$cpu_threads

mkdir -p $work_dir/results

$python_bin $script_dir/download_fixture.py --work-dir $work_dir
model_dir=$($script_dir/download_model.sh $work_dir/models)
archive=$work_dir/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
raw_result=$work_dir/results/sensevoice_small_int8.json

$python_bin $script_dir/benchmark_sensevoice.py \
  --manifest $work_dir/dataset/dataset_manifest.json \
  --model-dir $model_dir \
  --archive $archive \
  --output $raw_result \
  --repeats $repeats \
  --cpu-threads $cpu_threads

summary_args=(
  --sensevoice $raw_result
  --output-dir $work_dir/results
)
whisper_dir=$repo_root/reports/android_phone/streaming_response/multilingual_asr_mac_m2
if [[ -f $whisper_dir/whisper_base.json ]]; then
  summary_args+=(--whisper-base $whisper_dir/whisper_base.json)
fi
if [[ -f $whisper_dir/whisper_small.json ]]; then
  summary_args+=(--whisper-small $whisper_dir/whisper_small.json)
fi
$python_bin $script_dir/summarize.py $summary_args

print $work_dir/results
