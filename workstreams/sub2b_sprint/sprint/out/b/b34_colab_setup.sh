#!/usr/bin/env bash
set -euo pipefail

lane_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
work_dir="${2:-/content/lane_b_b34}"
python_bin="${PYTHON_BIN:-python3}"
llama_commit="11924d4c17abc27383376a1ac6a24fa3e36c1c0c"
llama_src="$work_dir/llama.cpp"
llama_build="$llama_src/build-b34"
multiple_choice_source_rel="tools/perplexity/perplexity.cpp"
multiple_choice_source_original_sha256="380aadd9d8000bf1b95cc3a338708385ac75240a5ba1c1f2bb34b208892a1777"
multiple_choice_source_sha256="65051abf6eb3e68d26cb2f24b62e2fa1af1bf7ddea96a2db53f944c3f0232b78"
multiple_choice_patch="$lane_dir/llama_mcq_memory.patch"
multiple_choice_patch_sha256="2539f2da48852fa44fdacdba26154e144a01b3986ff7c88dfe6974ac3e4f9dd7"
expected_source_status=" M $multiple_choice_source_rel"

mkdir -p "$work_dir"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
  echo "A Colab GPU runtime with CUDA/nvcc is required." >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "nvidia-smi found no usable Colab GPU." >&2
  exit 2
fi

"$python_bin" -m pip install --quiet --upgrade \
  "transformers==5.14.1" \
  "huggingface_hub==1.26.0" \
  sentencepiece \
  jinja2

if ! command -v ninja >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y ninja-build
fi

if [[ ! -d "$llama_src/.git" ]]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$llama_src"
fi

if [[ -n "$(git -C "$llama_src" status --porcelain)" ]]; then
  echo "Refusing to build a dirty llama.cpp tree: $llama_src" >&2
  exit 2
fi

git -C "$llama_src" fetch --depth 1 origin "$llama_commit"
git -C "$llama_src" checkout --detach "$llama_commit"

actual_commit="$(git -C "$llama_src" rev-parse HEAD)"
if [[ "$actual_commit" != "$llama_commit" ]]; then
  echo "llama.cpp commit mismatch: $actual_commit" >&2
  exit 2
fi

if [[ ! -f "$multiple_choice_patch" ]]; then
  echo "Missing audited llama.cpp MCQ patch: $multiple_choice_patch" >&2
  exit 2
fi

actual_original_sha256="$(sha256sum "$llama_src/$multiple_choice_source_rel" | awk '{print $1}')"
if [[ "$actual_original_sha256" != "$multiple_choice_source_original_sha256" ]]; then
  echo "Original llama.cpp MCQ source hash mismatch: $actual_original_sha256" >&2
  exit 2
fi

actual_patch_sha256="$(sha256sum "$multiple_choice_patch" | awk '{print $1}')"
if [[ "$actual_patch_sha256" != "$multiple_choice_patch_sha256" ]]; then
  echo "llama.cpp MCQ patch hash mismatch: $actual_patch_sha256" >&2
  exit 2
fi

git -C "$llama_src" apply --check "$multiple_choice_patch"
git -C "$llama_src" apply "$multiple_choice_patch"

actual_patched_sha256="$(sha256sum "$llama_src/$multiple_choice_source_rel" | awk '{print $1}')"
if [[ "$actual_patched_sha256" != "$multiple_choice_source_sha256" ]]; then
  echo "Patched llama.cpp MCQ source hash mismatch: $actual_patched_sha256" >&2
  exit 2
fi

source_status="$(git -C "$llama_src" status --porcelain=v1 --untracked-files=all)"
if [[ "$source_status" != "$expected_source_status" ]]; then
  echo "llama.cpp tree does not contain exactly the audited one-file patch." >&2
  printf 'Expected status: %q\nActual status:   %q\n' "$expected_source_status" "$source_status" >&2
  exit 2
fi
git -C "$llama_src" diff --check
git -C "$llama_src" apply --reverse --check "$multiple_choice_patch"

cmake -S "$llama_src" -B "$llama_build" -G Ninja \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$llama_build" --target llama-perplexity llama-server -j 2

"$python_bin" "$lane_dir/b34_harness.py" validate \
  --lane-dir "$lane_dir" \
  --output "$work_dir/protocol_validation.json"

"$python_bin" "$lane_dir/b34_harness.py" record-build \
  --lane-dir "$lane_dir" \
  --work-dir "$work_dir" \
  --llama-src "$llama_src" \
  --llama-bin-dir "$llama_build/bin"

echo "Lane B B3/B4 Colab setup complete."
echo "Work directory: $work_dir"
echo "llama.cpp binaries: $llama_build/bin"
