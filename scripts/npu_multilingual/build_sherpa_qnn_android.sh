#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
work_root=${LMF_NPU_BUILD_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/npu-multilingual/build}
bundle_root=${LMF_NPU_BUNDLE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/npu-multilingual/bundle}
ndk_root=${ANDROID_NDK:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/android-sdk/ndk/27.0.12077973}
jobs=${LMF_BUILD_JOBS:-2}
patch_file="$script_dir/sherpa_onnx_v1.13.4_persistent_offline.patch"
patch_sha=d048024b9ff24d044e6780400f61a1cba805e09c33d6b2925b5d530c6a71442c
patched_runner_source_sha=e2ba4608199103815bd98f9ff354ded2988d2ef7ec0e039a9dac7a82f9600208
protocol_version=1

case "$work_root" in
  "$repo_root"|"$repo_root"/*)
    echo "Refusing to build or stage external QNN dependencies inside the repository" >&2
    exit 2
    ;;
esac

if [ "$jobs" -lt 1 ] || [ "$jobs" -gt 2 ]; then
  echo "LMF_BUILD_JOBS must be 1 or 2 on the local Mac" >&2
  exit 2
fi
if [ ! -f "$bundle_root/bundle-manifest.json" ]; then
  echo "Prepare the external bundle first with prepare_sensevoice_sm8850_bundle.py" >&2
  exit 2
fi
if [ ! -f "$ndk_root/build/cmake/android.toolchain.cmake" ]; then
  echo "Android NDK 27.0.12077973 not found at $ndk_root" >&2
  exit 2
fi
if [ ! -f "$patch_file" ]; then
  echo "Persistent ASR runner patch is missing: $patch_file" >&2
  exit 2
fi
actual_patch_sha=$(shasum -a 256 "$patch_file" | awk '{print $1}')
if [ "$actual_patch_sha" != "$patch_sha" ]; then
  echo "Persistent ASR runner patch checksum mismatch" >&2
  exit 3
fi

qnn_sdk_root=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["qnn_sdk_root"])' "$bundle_root/bundle-manifest.json")
if [ ! -f "$qnn_sdk_root/include/QNN/QnnInterface.h" ]; then
  echo "QNN 2.40 headers are incomplete: $qnn_sdk_root" >&2
  exit 2
fi

mkdir -p "$work_root/downloads" "$work_root/source" "$work_root/deps" "$work_root/output"

ort_archive="$work_root/downloads/onnxruntime-android-1.27.0.zip"
ort_url=https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.0/onnxruntime-android-1.27.0.zip
ort_sha=a78f303a26b5e75c84c8b2a97fa2ddb400b2d1b5e069bec19aa229ccd3597fdb
if [ ! -f "$ort_archive" ]; then
  curl -fL --retry 3 --output "$ort_archive.partial" "$ort_url"
  mv "$ort_archive.partial" "$ort_archive"
fi
actual_ort_sha=$(shasum -a 256 "$ort_archive" | awk '{print $1}')
if [ "$actual_ort_sha" != "$ort_sha" ]; then
  echo "ONNX Runtime archive checksum mismatch" >&2
  exit 3
fi

ort_root="$work_root/deps/onnxruntime-1.27.0"
if [ ! -f "$ort_root/jni/arm64-v8a/libonnxruntime.so" ]; then
  rm -rf "$ort_root"
  mkdir -p "$ort_root"
  unzip -q "$ort_archive" -d "$ort_root"
fi

base_sherpa_source="$work_root/source/sherpa-onnx"
if [ ! -d "$base_sherpa_source/.git" ]; then
  git clone --depth 1 --branch v1.13.4 https://github.com/k2-fsa/sherpa-onnx.git "$base_sherpa_source"
fi
expected_commit=142807252687d81b40d6315f23470a1512a00de3
actual_commit=$(git -C "$base_sherpa_source" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
  echo "Sherpa source mismatch: $actual_commit != $expected_commit" >&2
  exit 3
fi

patch_id=${patch_sha:0:12}
sherpa_source="$work_root/source/sherpa-onnx-persistent-$patch_id"
if [ ! -d "$sherpa_source/.git" ]; then
  git clone --no-hardlinks "$base_sherpa_source" "$sherpa_source"
fi
actual_commit=$(git -C "$sherpa_source" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
  echo "Patched Sherpa source mismatch: $actual_commit != $expected_commit" >&2
  exit 3
fi
if git -C "$sherpa_source" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  : # The exact pinned patch is already applied.
elif git -C "$sherpa_source" apply --check "$patch_file" >/dev/null 2>&1; then
  git -C "$sherpa_source" apply "$patch_file"
else
  echo "Persistent Sherpa source is dirty or incompatible with the pinned patch" >&2
  git -C "$sherpa_source" status --short >&2
  exit 3
fi
if [ -n "$(git -C "$sherpa_source" ls-files --others --exclude-standard)" ]; then
  echo "Persistent Sherpa source contains unexpected untracked files" >&2
  git -C "$sherpa_source" status --short >&2
  exit 3
fi
if [ "$(git -C "$sherpa_source" diff --name-only)" != "sherpa-onnx/csrc/sherpa-onnx-offline.cc" ]; then
  echo "Persistent Sherpa source contains changes outside the pinned runner patch" >&2
  git -C "$sherpa_source" status --short >&2
  exit 3
fi
actual_patched_source_sha=$(shasum -a 256 \
  "$sherpa_source/sherpa-onnx/csrc/sherpa-onnx-offline.cc" | awk '{print $1}')
if [ "$actual_patched_source_sha" != "$patched_runner_source_sha" ]; then
  echo "Persistent Sherpa runner source checksum mismatch" >&2
  exit 3
fi

build_dir="$work_root/build-arm64-qnn-persistent-$patch_id"
export QNN_SDK_ROOT="$qnn_sdk_root"
export SHERPA_ONNXRUNTIME_LIB_DIR="$ort_root/jni/arm64-v8a"
export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ort_root/headers"
cmake \
  -S "$sherpa_source" \
  -B "$build_dir" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ndk_root/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-27 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_QNN=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF

cmake --build "$build_dir" --target sherpa-onnx-offline --parallel "$jobs"

runner_root="$work_root/output/sherpa-onnx-1.13.4-android-arm64-qnn"
rm -rf "$runner_root"
mkdir -p "$runner_root"
runner=$(find "$build_dir" -type f -name sherpa-onnx-offline -perm -111 | head -n 1)
if [ -z "$runner" ]; then
  echo "Built sherpa-onnx-offline was not found" >&2
  exit 4
fi
cp "$runner" "$runner_root/"
cp "$ort_root/jni/arm64-v8a/libonnxruntime.so" "$runner_root/"
find "$build_dir" -type f -name '*.so' -exec cp {} "$runner_root/" \;
toolchain_prebuilt=$(find "$ndk_root/toolchains/llvm/prebuilt" -mindepth 1 -maxdepth 1 -type d | head -n 1)
strip_tool="$toolchain_prebuilt/bin/llvm-strip"
if [ ! -x "$strip_tool" ]; then
  echo "Android llvm-strip not found: $strip_tool" >&2
  exit 4
fi
"$strip_tool" "$runner_root/sherpa-onnx-offline"

python3 - \
  "$runner_root" "$expected_commit" "$ndk_root" \
  "$patch_file" "$patch_sha" "$patched_runner_source_sha" \
  "$protocol_version" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = {}
for path in sorted(root.iterdir()):
    if path.is_file():
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
manifest = {
    "schema_version": 1,
    "target": {"abi": "arm64-v8a", "provider": "qnn", "android_api": 27},
    "sherpa_onnx_commit": sys.argv[2],
    "sherpa_onnx_patch": {
        "name": pathlib.Path(sys.argv[4]).name,
        "sha256": sys.argv[5],
        "patched_source_sha256": sys.argv[6],
    },
    "persistent_protocol": {
        "name": "stdin_wav_path_jsonl",
        "version": int(sys.argv[7]),
        "server_flag": "--persistent-server=true",
        "copies_flag": "--persistent-copies=",
        "request_framing": "utf8_wav_path_line",
        "response_framing": "utf8_json_line",
        "ready_payload": {
            "persistent_ready": True,
            "protocol_version": int(sys.argv[7]),
        },
        "quit_command": "__SHERPA_ONNX_PERSISTENT_QUIT__",
    },
    "android_ndk": pathlib.Path(sys.argv[3]).name,
    "qnn_runtime_included": False,
    "files": files,
}
(root / "runner-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "Built runner: $runner_root"
echo "Qualcomm runtime and SenseVoice model remain in the external bundle and are not copied here."
