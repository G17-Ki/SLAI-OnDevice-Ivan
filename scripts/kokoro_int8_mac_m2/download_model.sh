#!/bin/zsh

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
destination=${1:-"$repo_root/models/kokoro-int8-v1_1-zh"}
archive="$destination/kokoro-int8-multi-lang-v1_1.tar.bz2"
model_dir="$destination/kokoro-int8-multi-lang-v1_1"

archive_url='https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2'
archive_sha256='a1e94694776049035c4f2c6529f003aaece993c76aae9a78995831c3c4dcafc6'

mkdir -p "$destination"

if [[ ! -f "$archive" ]]; then
  curl --location --fail --retry 5 --output "$archive" "$archive_url"
fi

printf '%s  %s\n' "$archive_sha256" "$archive" | shasum -a 256 -c - >&2
bzip2 -tv "$archive" >&2

# Audit before extraction. The pinned archive has 417 entries, no absolute or
# parent-traversal paths, and no symbolic or hard links.
python3 - "$archive" <<'PY'
import sys
import tarfile

archive = sys.argv[1]
with tarfile.open(archive, "r:bz2") as handle:
    for member in handle.getmembers():
        parts = member.name.split("/")
        if member.name.startswith("/") or ".." in parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise SystemExit(
                f"archive links are not accepted: {member.name} -> {member.linkname}"
            )
PY

if [[ ! -d "$model_dir" ]]; then
  tar -xjf "$archive" -C "$destination"
fi

printf '%s  %s\n' \
  'bda15858163726a492d02a9a727bc263551b86ac77f90812c4b30ff41d380e26' \
  "$model_dir/model.int8.onnx" | shasum -a 256 -c - >&2
printf '%s  %s\n' \
  'e64a5a581d8c2a350d848f51c3121657cd83aa07ed6109172177345874a7244c' \
  "$model_dir/voices.bin" | shasum -a 256 -c - >&2
printf '%s  %s\n' \
  'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30' \
  "$model_dir/LICENSE" | shasum -a 256 -c - >&2

echo "$model_dir"
