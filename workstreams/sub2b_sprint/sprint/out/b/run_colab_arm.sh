#!/usr/bin/env bash
set -euo pipefail

model_id="${1:?usage: run_colab_arm.sh MODEL_ID}"
case "$model_id" in
  qwen35_08b) generation_port=18080 ;;
  qwen25_15b) generation_port=18081 ;;
  qwen3_06b) generation_port=18082 ;;
  falcon_h1_15b) generation_port=18083 ;;
  *)
    echo "unsupported Lane B model id: $model_id" >&2
    exit 2
    ;;
esac

lane_dir="${LANE_B_DIR:-/content/lane_b}"
work_dir="${LANE_B_WORK:-/content/lane_b_b34}"
bin_dir="$work_dir/llama.cpp/build-b34/bin"
python_bin="${LANE_B_PYTHON:-python}"
log_dir="$work_dir/driver_logs"
archive_dir="$work_dir/archives"

mkdir -p "$log_dir" "$archive_dir"
sha256sum "$0" > "$log_dir/$model_id.driver.sha256"

"$python_bin" "$lane_dir/b34_harness.py" render "$model_id" \
  --lane-dir "$lane_dir" \
  --work-dir "$work_dir" \
  --replace-rendered \
  > "$log_dir/$model_id.render.json"

"$python_bin" "$lane_dir/b34_harness.py" prepare-model "$model_id" \
  --lane-dir "$lane_dir" \
  --work-dir "$work_dir" \
  --replace-current \
  > "$log_dir/$model_id.prepare-model.json"

"$python_bin" "$lane_dir/b34_harness.py" run-mcq "$model_id" \
  --lane-dir "$lane_dir" \
  --work-dir "$work_dir" \
  --llama-bin-dir "$bin_dir" \
  > "$log_dir/$model_id.run-mcq.json"

"$python_bin" "$lane_dir/b34_harness.py" run-generation "$model_id" \
  --lane-dir "$lane_dir" \
  --work-dir "$work_dir" \
  --llama-bin-dir "$bin_dir" \
  --port "$generation_port" \
  > "$log_dir/$model_id.run-generation.json"

archive_path="$archive_dir/$model_id.evidence.tar.gz"
rm -f "$archive_path"
tar -C "$work_dir" -czf "$archive_path" \
  build_receipt.json \
  protocol_validation.json \
  "raw/$model_id" \
  "rendered/$model_id" \
  "driver_logs/$model_id.render.json" \
  "driver_logs/$model_id.prepare-model.json" \
  "driver_logs/$model_id.run-mcq.json" \
  "driver_logs/$model_id.run-generation.json" \
  "driver_logs/$model_id.driver.sha256"

sha256sum "$archive_path" > "$archive_path.sha256"

"$python_bin" "$lane_dir/b34_harness.py" purge-model \
  --work-dir "$work_dir" \
  > "$log_dir/$model_id.purge-model.json"

echo "LANE_B_ARM_COMPLETE $model_id $(cat "$archive_path.sha256")"
