#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
cache_root=${XDG_CACHE_HOME:-$HOME/.cache}

adb_bin=${ADB:-adb}
serial=
remote_root=/data/local/tmp/lmf-asr-companion
external_root=$cache_root/lmf-demo/android-asr-companion
sensevoice_bundle_root=$cache_root/lmf-demo/npu-multilingual/bundle
runner_root=$cache_root/lmf-demo/npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn
offline=0
build_runner=0
start_after_deploy=1
forward_port=0
seed_args=()
seed_arg_count=0

usage() {
  cat <<'EOF'
Usage: deploy.sh [options]

Options:
  --serial SERIAL                 Select an adb device (required if >1 connected)
  --adb PATH                      adb executable
  --remote-root PATH              Phone root (default /data/local/tmp/lmf-asr-companion)
  --external-root PATH            External Russian model cache
  --sensevoice-bundle-root PATH   Existing/prepared SenseVoice QNN bundle
  --runner-root PATH              Sherpa Android QNN runner directory
  --seed-dir PATH                 Reuse pinned downloads (repeatable)
  --offline                       Refuse all downloads
  --build-runner                  Build the pinned CLI runner if it is absent
  --no-start                      Stage only
  --forward                       adb-forward host tcp:6010 after deployment
  -h, --help                      Show this help

Models, Qualcomm libraries, and executables remain external to the repository.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) serial=$2; shift 2 ;;
    --adb) adb_bin=$2; shift 2 ;;
    --remote-root) remote_root=$2; shift 2 ;;
    --external-root) external_root=$2; shift 2 ;;
    --sensevoice-bundle-root) sensevoice_bundle_root=$2; shift 2 ;;
    --runner-root) runner_root=$2; shift 2 ;;
    --seed-dir)
      seed_args+=(--seed-dir "$2")
      seed_arg_count=$((seed_arg_count + 1))
      shift 2
      ;;
    --offline) offline=1; shift ;;
    --build-runner) build_runner=1; shift ;;
    --no-start) start_after_deploy=0; shift ;;
    --forward) forward_port=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$remote_root" =~ ^/data/local/tmp/[A-Za-z0-9._/-]+$ ]]; then
  echo "Remote root must be a simple path below /data/local/tmp" >&2
  exit 2
fi
case "$remote_root" in *".."*) echo "Remote root may not contain '..'" >&2; exit 2 ;; esac

command -v "$adb_bin" >/dev/null || { echo "adb not found: $adb_bin" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }

if [[ -z "$serial" ]]; then
  devices=$("$adb_bin" devices | awk 'NR > 1 && $2 == "device" {print $1}')
  device_count=$(printf '%s\n' "$devices" | awk 'NF {count++} END {print count + 0}')
  if [[ $device_count -ne 1 ]]; then
    echo "Expected exactly one connected adb device, found $device_count; use --serial" >&2
    exit 2
  fi
  serial=$(printf '%s\n' "$devices" | awk 'NF {print; exit}')
fi
adb_cmd=("$adb_bin" -s "$serial")

abi=$("${adb_cmd[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')
soc=$("${adb_cmd[@]}" shell getprop ro.soc.model | tr -d '\r')
board=$("${adb_cmd[@]}" shell getprop ro.product.board | tr -d '\r')
if [[ "$abi" != arm64-v8a ]]; then
  echo "Refusing non-arm64 phone: ABI is $abi" >&2
  exit 3
fi
case "$soc $board" in
  *SM8850*|*sm8850*) ;;
  *)
    echo "Refusing QNN context on a non-SM8850 phone (soc=$soc board=$board)" >&2
    exit 3
    ;;
esac

if [[ $build_runner -eq 1 && ! -f "$runner_root/runner-manifest.json" ]]; then
  if [[ $offline -eq 1 ]]; then
    echo "Cannot build a missing runner in --offline mode unless dependencies are already prepared" >&2
  fi
  prepare_command=(python3 "$repo_root/scripts/npu_multilingual/prepare_sensevoice_sm8850_bundle.py" --root "$sensevoice_bundle_root")
  if [[ $seed_arg_count -gt 0 ]]; then
    prepare_command+=("${seed_args[@]}")
  fi
  "${prepare_command[@]}"
  expected_default=$cache_root/lmf-demo/npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn
  if [[ "$runner_root" != "$expected_default" ]]; then
    echo "--build-runner currently requires the default runner root: $expected_default" >&2
    exit 2
  fi
  LMF_NPU_BUNDLE_ROOT="$sensevoice_bundle_root" \
    "$repo_root/scripts/npu_multilingual/build_sherpa_qnn_android.sh"
fi

prepared_json=$(mktemp "${TMPDIR:-/tmp}/lmf-asr-assets.XXXXXX")
env_file=$(mktemp "${TMPDIR:-/tmp}/lmf-asr-env.XXXXXX")
cleanup() {
  rm -f "$prepared_json" "$env_file"
}
trap cleanup EXIT

prepare_command=(
  python3 "$script_dir/prepare_assets.py"
  --external-root "$external_root"
  --sensevoice-bundle-root "$sensevoice_bundle_root"
  --runner-root "$runner_root"
  --output "$prepared_json"
)
if [[ $seed_arg_count -gt 0 ]]; then
  prepare_command+=("${seed_args[@]}")
fi
if [[ $offline -eq 1 ]]; then prepare_command+=(--offline); fi
"${prepare_command[@]}" >/dev/null

json_path() {
  python3 - "$prepared_json" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

qnn_runtime=$(json_path sensevoice.runtime_root)
sensevoice_model=$(json_path sensevoice.model_root)
russian_model=$(json_path russian.model_root)
verified_runner=$(json_path runner.root)

if "${adb_cmd[@]}" shell test -f "$remote_root/start.sh"; then
  # The lifecycle manager verifies its own PID before signalling anything.
  "${adb_cmd[@]}" shell sh "$remote_root/start.sh" stop
fi
"${adb_cmd[@]}" shell rm -rf "$remote_root"
"${adb_cmd[@]}" shell mkdir -p \
  "$remote_root/bin" \
  "$remote_root/qnn-runtime" \
  "$remote_root/models/sensevoice" \
  "$remote_root/models/russian" \
  "$remote_root/tmp"

push_file() {
  local source=$1
  local destination=$2
  [[ -f "$source" ]] || { echo "Missing staged source: $source" >&2; exit 4; }
  "${adb_cmd[@]}" push "$source" "$destination" >/dev/null
}

push_file "$script_dir/handler.sh" "$remote_root/handler.sh"
push_file "$script_dir/start.sh" "$remote_root/start.sh"
push_file "$script_dir/assets.json" "$remote_root/assets.json"
push_file "$verified_runner/sherpa-onnx-offline" "$remote_root/bin/sherpa-onnx-offline"
push_file "$verified_runner/libonnxruntime.so" "$remote_root/bin/libonnxruntime.so"

for filename in \
  libQnnHtp.so libQnnHtpPrepare.so libQnnSystem.so \
  libQnnHtpV81Stub.so libQnnHtpV81Skel.so; do
  push_file "$qnn_runtime/$filename" "$remote_root/qnn-runtime/$filename"
done
for filename in model.bin tokens.txt; do
  push_file "$sensevoice_model/$filename" "$remote_root/models/sensevoice/$filename"
done
for filename in LICENSE README.md info.txt; do
  if [[ -f "$sensevoice_model/$filename" ]]; then
    push_file "$sensevoice_model/$filename" "$remote_root/models/sensevoice/$filename"
  fi
done
for filename in encoder.int8.onnx decoder.onnx joiner.int8.onnx tokens.txt bpe.model README.md; do
  push_file "$russian_model/$filename" "$remote_root/models/russian/$filename"
done

cat > "$env_file" <<EOF
export ASR_COMPANION_ROOT=$remote_root
export ASR_RUNNER_ROOT=$remote_root/bin
export ASR_RUNNER=$remote_root/bin/sherpa-onnx-offline
export QNN_RUNTIME_ROOT=$remote_root/qnn-runtime
export SENSEVOICE_MODEL_ROOT=$remote_root/models/sensevoice
export RUSSIAN_MODEL_ROOT=$remote_root/models/russian
export ASR_TMP_ROOT=$remote_root/tmp
export ASR_CPU_THREADS=2
export ASR_MAX_WAV_BYTES=320044
EOF
push_file "$env_file" "$remote_root/env.sh"

"${adb_cmd[@]}" shell chmod 700 \
  "$remote_root/handler.sh" "$remote_root/start.sh" "$remote_root/bin/sherpa-onnx-offline"
"${adb_cmd[@]}" shell chmod 600 "$remote_root/env.sh"

if [[ $start_after_deploy -eq 1 ]]; then
  "${adb_cmd[@]}" shell sh "$remote_root/start.sh" start
  "${adb_cmd[@]}" shell sh "$remote_root/start.sh" health
fi
if [[ $forward_port -eq 1 ]]; then
  "${adb_cmd[@]}" forward tcp:6010 tcp:6010
  echo "Host forwarding active: http://127.0.0.1:6010/health"
fi

echo "Android ASR companion staged on $serial at $remote_root"
echo "Lifecycle: adb -s $serial shell sh $remote_root/start.sh status"
echo "The service is phone-local and binds only 127.0.0.1:6010."
