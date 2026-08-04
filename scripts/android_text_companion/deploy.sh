#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
cache_root=${XDG_CACHE_HOME:-$HOME/.cache}

adb_bin=${ADB:-adb}
serial=
remote_root=/data/local/tmp/lmf-text-companion
external_root=$cache_root/lmf-demo/android-text-companion
model_source=
runtime_source=
offline=0
start_after_deploy=1
forward_port=0
ctx_size=2048
threads=4
batch_threads=4
seed_dirs=()
seed_dir_count=0
runtime_seed_dirs=()
runtime_seed_dir_count=0

usage() {
  cat <<'EOF'
Usage: deploy.sh [options]

Options:
  --serial SERIAL        Select an adb device (required if >1 connected)
  --adb PATH             adb executable
  --model PATH           Existing pinned Qwen3-1.7B-Q4_0 GGUF
  --seed-dir PATH        Directory containing the pinned model (repeatable)
  --runtime-archive PATH Existing pinned llama.cpp b10083 Android archive
  --runtime-seed-dir DIR Directory containing the pinned runtime archive (repeatable)
  --external-root PATH   External verified model/runtime cache
  --remote-root PATH     Phone root (default /data/local/tmp/lmf-text-companion)
  --ctx-size TOKENS      256, 384, 512, 768, 1024, 1536, or 2048 (default 2048)
  --threads N            Generation threads, 1-4 (default 4)
  --batch-threads N      Prompt threads, 1-4 (default 4)
  --offline              Refuse model downloads
  --no-start             Stage and verify without starting
  --forward              adb-forward host tcp:6020 for host-side probes
  -h, --help             Show this help

The GGUF and official llama.cpp Android release remain external to the
repository. Both are pinned by size and SHA-256 before phone transfer.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) serial=$2; shift 2 ;;
    --adb) adb_bin=$2; shift 2 ;;
    --model) model_source=$2; shift 2 ;;
    --runtime-archive) runtime_source=$2; shift 2 ;;
    --seed-dir)
      seed_dirs+=("$2")
      seed_dir_count=$((seed_dir_count + 1))
      shift 2
      ;;
    --runtime-seed-dir)
      runtime_seed_dirs+=("$2")
      runtime_seed_dir_count=$((runtime_seed_dir_count + 1))
      shift 2
      ;;
    --external-root) external_root=$2; shift 2 ;;
    --remote-root) remote_root=$2; shift 2 ;;
    --ctx-size) ctx_size=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --batch-threads) batch_threads=$2; shift 2 ;;
    --offline) offline=1; shift ;;
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
case "$ctx_size" in 256|384|512|768|1024|1536|2048) ;;
  *) echo "Unsupported bounded context size: $ctx_size" >&2; exit 2 ;;
esac
case "$threads" in 1|2|3|4) ;; *) echo "--threads must be 1-4" >&2; exit 2 ;; esac
case "$batch_threads" in 1|2|3|4) ;; *) echo "--batch-threads must be 1-4" >&2; exit 2 ;; esac

command -v "$adb_bin" >/dev/null || { echo "adb not found: $adb_bin" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }

prepared_json=$(mktemp "${TMPDIR:-/tmp}/lmf-text-model.XXXXXX")
prepared_runtime_json=$(mktemp "${TMPDIR:-/tmp}/lmf-text-runtime.XXXXXX")
env_file=$(mktemp "${TMPDIR:-/tmp}/lmf-text-env.XXXXXX")
cleanup() { rm -f "$prepared_json" "$prepared_runtime_json" "$env_file"; }
trap cleanup EXIT

prepare_command=(python3 "$script_dir/prepare_model.py" --external-root "$external_root" --output "$prepared_json")
if [[ -n "$model_source" ]]; then prepare_command+=(--source "$model_source"); fi
if [[ $seed_dir_count -gt 0 ]]; then
  for seed_dir in "${seed_dirs[@]}"; do prepare_command+=(--seed-dir "$seed_dir"); done
fi
if [[ $offline -eq 1 ]]; then prepare_command+=(--offline); fi
"${prepare_command[@]}" >/dev/null

runtime_command=(python3 "$script_dir/prepare_runtime.py" --external-root "$external_root" --output "$prepared_runtime_json")
if [[ -n "$runtime_source" ]]; then runtime_command+=(--source "$runtime_source"); fi
if [[ $runtime_seed_dir_count -gt 0 ]]; then
  for seed_dir in "${runtime_seed_dirs[@]}"; do runtime_command+=(--seed-dir "$seed_dir"); done
fi
if [[ $offline -eq 1 ]]; then runtime_command+=(--offline); fi
"${runtime_command[@]}" >/dev/null

model_path=$(python3 - "$prepared_json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model"]["path"])
PY
)
model_bytes=$(python3 - "$script_dir/assets.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model"]["bytes"])
PY
)
model_hash=$(python3 - "$script_dir/assets.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model"]["sha256"])
PY
)
runtime_path=$(python3 - "$prepared_runtime_json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["path"])
PY
)

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
case "$soc $board" in *SM8850*|*sm8850*) ;;
  *) echo "Refusing non-SM8850 phone (soc=$soc board=$board)" >&2; exit 3 ;;
esac

phone_sha256() {
  "${adb_cmd[@]}" shell toybox sha256sum "$1" 2>/dev/null | tr -d '\r' | awk 'NR == 1 {print $1}'
}

if "${adb_cmd[@]}" shell test -f "$remote_root/start.sh"; then
  "${adb_cmd[@]}" shell sh "$remote_root/start.sh" stop
fi
remote_runtime=$remote_root/runtime-b10083
"${adb_cmd[@]}" shell mkdir -p "$remote_root/models" "$remote_root/tmp" "$remote_root/home" "$remote_runtime"

push_file() {
  local source=$1
  local destination=$2
  [[ -f "$source" ]] || { echo "Missing staged source: $source" >&2; exit 4; }
  "${adb_cmd[@]}" push "$source" "$destination" >/dev/null
}

push_file "$script_dir/start.sh" "$remote_root/start.sh"
push_file "$script_dir/assets.json" "$remote_root/assets.json"

runner_lines=$(python3 - "$script_dir/assets.json" <<'PY'
import json, sys
files = json.load(open(sys.argv[1], encoding="utf-8"))["runner"]["files"]
for name, spec in sorted(files.items()):
    print(f"{name}\t{spec['sha256']}\t{1 if spec.get('executable') else 0}")
PY
)
# Read the manifest through fd 3 so adb cannot consume the remaining entries
# from stdin while the loop performs phone-side checks and transfers.
while IFS=$'\t' read -r filename expected_hash executable <&3; do
  [[ -n "$filename" ]] || continue
  local_file=$runtime_path/$filename
  remote_file=$remote_runtime/$filename
  remote_hash=$(phone_sha256 "$remote_file" || true)
  if [[ "$remote_hash" != "$expected_hash" ]]; then
    remote_partial=$remote_file.partial
    "${adb_cmd[@]}" shell rm -f "$remote_partial"
    push_file "$local_file" "$remote_partial"
    pushed_hash=$(phone_sha256 "$remote_partial")
    if [[ "$pushed_hash" != "$expected_hash" ]]; then
      "${adb_cmd[@]}" shell rm -f "$remote_partial"
      echo "Phone-side runtime hash mismatch for $filename" >&2
      exit 4
    fi
    "${adb_cmd[@]}" shell mv "$remote_partial" "$remote_file"
  fi
  if [[ "$executable" == 1 ]]; then
    "${adb_cmd[@]}" shell chmod 700 "$remote_file"
  else
    "${adb_cmd[@]}" shell chmod 600 "$remote_file"
  fi
done 3<<< "$runner_lines"

remote_model=$remote_root/models/Qwen3-1.7B-Q4_0.gguf
remote_model_hash=$(phone_sha256 "$remote_model" || true)
if [[ "$remote_model_hash" != "$model_hash" ]]; then
  remote_partial=$remote_model.partial
  "${adb_cmd[@]}" shell rm -f "$remote_partial"
  push_file "$model_path" "$remote_partial"
  pushed_hash=$(phone_sha256 "$remote_partial")
  if [[ "$pushed_hash" != "$model_hash" ]]; then
    "${adb_cmd[@]}" shell rm -f "$remote_partial"
    echo "Phone-side model hash mismatch after adb push" >&2
    exit 4
  fi
  "${adb_cmd[@]}" shell mv "$remote_partial" "$remote_model"
else
  echo "Pinned model already present on phone; skipping 1.06 GB transfer"
fi

cat > "$env_file" <<EOF
export TEXT_COMPANION_ROOT=$remote_root
export TEXT_SERVER=$remote_runtime/llama-server
export TEXT_NATIVE_LIB_DIR=$remote_runtime
export TEXT_MODEL=$remote_model
export TEXT_MODEL_BYTES=$model_bytes
export TEXT_MODEL_SHA256=$model_hash
export TEXT_PORT=6020
export TEXT_CTX_SIZE=$ctx_size
export TEXT_THREADS=$threads
export TEXT_BATCH_THREADS=$batch_threads
export TEXT_BATCH_SIZE=128
export TEXT_UBATCH_SIZE=64
EOF
push_file "$env_file" "$remote_root/env.sh"

"${adb_cmd[@]}" shell chmod 700 "$remote_root/start.sh"
"${adb_cmd[@]}" shell chmod 600 "$remote_root/env.sh" "$remote_model"
"${adb_cmd[@]}" shell sh "$remote_root/start.sh" verify

if [[ $start_after_deploy -eq 1 ]]; then
  "${adb_cmd[@]}" shell sh "$remote_root/start.sh" start
  "${adb_cmd[@]}" shell sh "$remote_root/start.sh" health
fi
if [[ $forward_port -eq 1 ]]; then
  "${adb_cmd[@]}" forward tcp:6020 tcp:6020
  echo "Host forwarding active: http://127.0.0.1:6020/health"
fi

echo "Android text companion staged on $serial at $remote_root"
echo "Lifecycle: adb -s $serial shell sh $remote_root/start.sh status"
echo "The service is phone-local and binds only 127.0.0.1:6020."
