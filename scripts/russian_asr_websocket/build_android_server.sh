#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
patch_file="$script_dir/sherpa_onnx_v1.13.4_duplicate_stream.patch"

base_build_root=${LMF_NPU_BUILD_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/npu-multilingual/build}
work_root=${LMF_RU_WS_BUILD_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/russian-asr-websocket}
base_source=${SHERPA_ONNX_SOURCE_ROOT:-$base_build_root/source/sherpa-onnx}
ort_root=${SHERPA_ONNXRUNTIME_ROOT:-$base_build_root/deps/onnxruntime-1.27.0}
ndk_root=${ANDROID_NDK:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/android-sdk/ndk/27.0.12077973}
jobs=${LMF_BUILD_JOBS:-2}
expected_commit=142807252687d81b40d6315f23470a1512a00de3

case "$work_root" in
  "$repo_root"|"$repo_root"/*)
    echo "Refusing to build external Android dependencies inside the repository" >&2
    exit 2
    ;;
esac
if [ "$jobs" -lt 1 ] || [ "$jobs" -gt 2 ]; then
  echo "LMF_BUILD_JOBS must be 1 or 2 on the local Mac" >&2
  exit 2
fi
if [ ! -f "$patch_file" ]; then
  echo "Patch missing: $patch_file" >&2
  exit 2
fi
if [ ! -d "$base_source/.git" ]; then
  echo "Pinned Sherpa source is missing: $base_source" >&2
  exit 2
fi
if [ "$(git -C "$base_source" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "Base Sherpa source is not pinned to $expected_commit" >&2
  exit 3
fi
if [ ! -f "$ort_root/jni/arm64-v8a/libonnxruntime.so" ] ||
   [ ! -f "$ort_root/headers/onnxruntime_cxx_api.h" ]; then
  echo "Pinned ONNX Runtime Android 1.27.0 tree is incomplete: $ort_root" >&2
  exit 2
fi
if [ ! -f "$ndk_root/build/cmake/android.toolchain.cmake" ]; then
  echo "Android NDK 27.0.12077973 is missing: $ndk_root" >&2
  exit 2
fi

mkdir -p "$work_root/source" "$work_root/output"
patched_source="$work_root/source/sherpa-onnx"
if [ ! -d "$patched_source/.git" ]; then
  git clone --no-hardlinks "$base_source" "$patched_source"
fi
if [ "$(git -C "$patched_source" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "Patched source checkout is not pinned to $expected_commit" >&2
  exit 3
fi

if git -C "$patched_source" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  : # The exact patch is already present.
elif git -C "$patched_source" apply --check "$patch_file" >/dev/null 2>&1; then
  git -C "$patched_source" apply "$patch_file"
else
  echo "Patched source is dirty or incompatible with the pinned patch" >&2
  git -C "$patched_source" status --short >&2
  exit 3
fi

build_dir="$work_root/build-arm64"
export SHERPA_ONNXRUNTIME_LIB_DIR="$ort_root/jni/arm64-v8a"
export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ort_root/headers"
cmake \
  -S "$patched_source" \
  -B "$build_dir" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ndk_root/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-27 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON \
  -DSHERPA_ONNX_ENABLE_QNN=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF

cmake --build "$build_dir" \
  --target sherpa-onnx-offline-websocket-server \
  --parallel "$jobs"

binary=$(find "$build_dir" -type f -name sherpa-onnx-offline-websocket-server -perm -111 | head -n 1)
if [ -z "$binary" ]; then
  echo "Built WebSocket server was not found" >&2
  exit 4
fi

output_root="$work_root/output/sherpa-onnx-1.13.4-android-arm64"
mkdir -p "$output_root"
cp "$binary" "$output_root/"
cp "$ort_root/jni/arm64-v8a/libonnxruntime.so" "$output_root/"

toolchain_prebuilt=$(find "$ndk_root/toolchains/llvm/prebuilt" -mindepth 1 -maxdepth 1 -type d | head -n 1)
strip_tool="$toolchain_prebuilt/bin/llvm-strip"
if [ ! -x "$strip_tool" ]; then
  echo "Android llvm-strip was not found: $strip_tool" >&2
  exit 4
fi
"$strip_tool" "$output_root/sherpa-onnx-offline-websocket-server"

python3 - "$output_root" "$expected_commit" "$patch_file" "$ndk_root" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
patch = pathlib.Path(sys.argv[3])
files = {}
for path in sorted(root.iterdir()):
    if path.is_file() and path.name != "manifest.json":
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
manifest = {
    "schema_version": 1,
    "target": {"abi": "arm64-v8a", "android_api": 27, "provider": "cpu"},
    "sherpa_onnx_commit": sys.argv[2],
    "sherpa_onnx_patch": {
        "name": patch.name,
        "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    },
    "onnxruntime": "1.27.0",
    "android_ndk": pathlib.Path(sys.argv[4]).name,
    "required_server_flag": "--duplicate-each-stream=true",
    "files": files,
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "Built persistent Russian ASR server: $output_root"
