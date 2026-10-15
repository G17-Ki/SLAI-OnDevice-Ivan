#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/.." && pwd)
cache_root=${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo

package=ai.liquid.lfmdemo.dev.runtime
main_package=ai.liquid.lfmdemo.dev
activity=ai.liquid.lfmdemo.dev.runtime/ai.liquid.lfmdemo.runtime.RuntimeActivity
model_alias=Qwen3-ASR-0.6B-Text-Q4_K_M
model_manifest=$repo_root/reports/android_phone/streaming_response/model_candidate_followup/candidate_manifest.json
model_cache=$cache_root/android-text-companion/models
asr_external_root=$cache_root/android-asr-companion
sensevoice_bundle=$cache_root/npu-multilingual/bundle
runner_root=$cache_root/npu-multilingual/build/output/sherpa-onnx-1.13.4-android-arm64-qnn
apk=$repo_root/android-app/runtime-companion/build/outputs/apk/debug/runtime-companion-debug.apk
adb_bin=${ADB:-adb}
serial=
skip_build=0
skip_install=0
skip_assets=0
start_after_install=1
keep_forwards=0

usage() {
  cat <<'EOF'
Usage: scripts/install_app_owned_runtime.sh [options]

Build and install the app-owned ASR/Qwen runtime companion, then copy verified
models into its private storage. The default path is fully offline and consumes
only already-pinned host caches.

Options:
  --serial SERIAL        Select an adb device (required if more than one exists)
  --adb PATH             adb executable
  --apk PATH             Companion APK output/input path
  --skip-build           Reuse --apk after verifying its required native payload
  --skip-install         Do not reinstall the companion APK
  --skip-assets          Do not copy private ASR/Qwen assets
  --no-start             Do not launch or health-check the companion
  --keep-forwards        Keep temporary adb forwards after a successful run
  -h, --help             Show this help

Required cached inputs:
  ~/.cache/lmf-demo/android-text-companion/models/
  ~/.cache/lmf-demo/android-text-companion/runtime/llama-b10083/
  ~/.cache/lmf-demo/android-asr-companion/
  ~/.cache/lmf-demo/npu-multilingual/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) serial=$2; shift 2 ;;
    --adb) adb_bin=$2; shift 2 ;;
    --apk) apk=$2; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --skip-install) skip_install=1; shift ;;
    --skip-assets) skip_assets=1; shift ;;
    --no-start) start_after_install=0; shift ;;
    --keep-forwards) keep_forwards=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v "$adb_bin" >/dev/null || { echo "adb not found: $adb_bin" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lmf-app-runtime.XXXXXX")
remote_stage=
text_forward=
asr_forward=
cleanup() {
  if [[ -n "$remote_stage" && -n "$serial" ]]; then
    "$adb_bin" -s "$serial" shell rm -rf "$remote_stage" >/dev/null 2>&1 || true
  fi
  if [[ $keep_forwards -eq 0 && -n "$serial" ]]; then
    if [[ -n "$text_forward" ]]; then
      "$adb_bin" -s "$serial" forward --remove "tcp:$text_forward" >/dev/null 2>&1 || true
    fi
    if [[ -n "$asr_forward" ]]; then
      "$adb_bin" -s "$serial" forward --remove "tcp:$asr_forward" >/dev/null 2>&1 || true
    fi
  fi
  rm -rf "$scratch"
}
trap cleanup EXIT

model_fields=$(python3 - "$model_manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for item in manifest["models"]:
    if item.get("id") == "qwen3_asr_0p6b_text_q4_k_m":
        print("\t".join((item["filename"], str(item["bytes"]), item["sha256"])))
        break
else:
    raise SystemExit("Qwen3-ASR-0.6B text Q4_K_M is absent from candidate_manifest.json")
PY
)
IFS=$'\t' read -r model_filename model_bytes model_sha256 <<< "$model_fields"
model_path=$model_cache/$model_filename

verify_file() {
  local path=$1
  local expected_bytes=$2
  local expected_sha256=$3
  python3 - "$path" "$expected_bytes" "$expected_sha256" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_bytes = int(sys.argv[2])
expected_hash = sys.argv[3]
if not path.is_file():
    raise SystemExit(f"missing file: {path}")
if path.stat().st_size != expected_bytes:
    raise SystemExit(f"size mismatch for {path}: {path.stat().st_size} != {expected_bytes}")
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
actual_hash = digest.hexdigest()
if actual_hash != expected_hash:
    raise SystemExit(f"SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}")
PY
}

verify_file "$model_path" "$model_bytes" "$model_sha256"

asr_json=$scratch/asr-assets.json
python3 "$repo_root/scripts/android_asr_companion/prepare_assets.py" \
  --external-root "$asr_external_root" \
  --sensevoice-bundle-root "$sensevoice_bundle" \
  --runner-root "$runner_root" \
  --offline \
  --output "$asr_json" >/dev/null

json_path() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

sensevoice_model=$(json_path "$asr_json" sensevoice.model_root)
russian_model=$(json_path "$asr_json" russian.model_root)
sensevoice_en_fixture=$sensevoice_model/test_wavs/en.wav
sensevoice_zh_fixture=$sensevoice_model/test_wavs/zh.wav
sensevoice_ja_fixture=$repo_root/scripts/phone_acceptance_fixtures/question_ja.wav
sensevoice_ko_fixture=$repo_root/scripts/phone_acceptance_fixtures/question_ko.wav
russian_fixture=$repo_root/scripts/phone_acceptance_fixtures/question_ru.wav
bundle_manifest=$sensevoice_bundle/bundle-manifest.json

verify_file \
  "$sensevoice_ja_fixture" 79464 \
  0da6ee3b9514a96597bf9225a1bcdf7cae86608e095e157490899a443948b761
verify_file \
  "$sensevoice_ko_fixture" 66908 \
  0ce8353dc9faa844cc8214c613f182a127338b71b139955ccb903ca806a7b3de

bundle_file_fields() {
  python3 - "$bundle_manifest" "$1" <<'PY'
import json
import sys

spec = json.load(open(sys.argv[1], encoding="utf-8"))["files"][sys.argv[2]]
print(f"{spec['bytes']}\t{spec['sha256']}")
PY
}

if [[ $skip_build -eq 0 ]]; then
  jni_root=$scratch/jni
  python3 "$repo_root/scripts/prepare_runtime_companion_jni.py" --output "$jni_root" >/dev/null

  android_home=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}
  if [[ ! -d "$android_home" ]]; then
    echo "Android SDK not found: $android_home" >&2
    exit 2
  fi
  java_home=${JAVA_HOME:-}
  if [[ -z "$java_home" && -d /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ]]; then
    java_home=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
  fi
  if [[ -z "$java_home" ]]; then
    java_home=$(/usr/libexec/java_home -v 17 2>/dev/null || true)
  fi
  if [[ -z "$java_home" || ! -x "$java_home/bin/java" ]]; then
    echo "JDK 17 is required for the Android build" >&2
    exit 2
  fi

  (
    cd "$repo_root/android-app"
    JAVA_HOME=$java_home \
      ANDROID_HOME=$android_home \
      RUNTIME_COMPANION_JNI_DIR=$jni_root \
      ./gradlew --max-workers=2 :runtime-companion:assembleDebug
  )
fi

python3 - "$apk" <<'PY'
import pathlib
import sys
import zipfile

apk = pathlib.Path(sys.argv[1])
required = {
    "lib/arm64-v8a/libqwenserver.so",
    "lib/arm64-v8a/libllama-server-impl.so",
    "lib/arm64-v8a/libasrrunner.so",
    "lib/arm64-v8a/libonnxruntime.so",
    "lib/arm64-v8a/libQnnHtp.so",
    "lib/arm64-v8a/libQnnHtpPrepare.so",
    "lib/arm64-v8a/libQnnHtpV81Skel.so",
    "lib/arm64-v8a/libQnnHtpV81Stub.so",
    "lib/arm64-v8a/libQnnSystem.so",
}
if not apk.is_file():
    raise SystemExit(f"missing companion APK: {apk}")
with zipfile.ZipFile(apk) as bundle:
    names = set(bundle.namelist())
missing = sorted(required - names)
if missing:
    raise SystemExit(f"companion APK lacks required arm64 payload: {missing}")
PY

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
  *) echo "Refusing non-SM8850 phone (soc=$soc board=$board)" >&2; exit 3 ;;
esac

if [[ $skip_install -eq 0 ]]; then
  "${adb_cmd[@]}" install -r "$apk"
fi
if ! "${adb_cmd[@]}" shell run-as "$package" true >/dev/null 2>&1; then
  echo "run-as failed for $package; install the same-signature debuggable companion APK" >&2
  exit 4
fi

stop_known_legacy() {
  local root=$1
  if "${adb_cmd[@]}" shell test -f "$root/start.sh"; then
    echo "Stopping verified legacy lifecycle at $root"
    if ! "${adb_cmd[@]}" shell toybox timeout -k 2 15 sh "$root/start.sh" stop; then
      echo "Legacy stop at $root did not finish cleanly within 15 seconds" >&2
    fi
  fi
}
stop_known_legacy /data/local/tmp/lmf-asr-companion
stop_known_legacy /data/local/tmp/lmf-text-companion
"${adb_cmd[@]}" shell am force-stop "$package"

remote_stage=/data/local/tmp/lmf-runtime-install-$$
"${adb_cmd[@]}" shell mkdir -p "$remote_stage"
"${adb_cmd[@]}" shell chmod 711 "$remote_stage"

private_hash() {
  "${adb_cmd[@]}" shell run-as "$package" toybox sha256sum "$1" 2>/dev/null \
    | tr -d '\r' | awk 'NR == 1 {print $1}'
}

private_size() {
  "${adb_cmd[@]}" shell run-as "$package" toybox stat -c %s "$1" 2>/dev/null \
    | tr -d '\r' | awk 'NR == 1 {print $1}'
}

install_private_file() {
  local source=$1
  local destination=$2
  local expected_bytes=$3
  local expected_sha256=$4
  local label=$5

  verify_file "$source" "$expected_bytes" "$expected_sha256"
  local installed_hash installed_bytes
  installed_hash=$(private_hash "$destination" || true)
  installed_bytes=$(private_size "$destination" || true)
  if [[ "$installed_hash" == "$expected_sha256" && "$installed_bytes" == "$expected_bytes" ]]; then
    echo "$label already verified in app-private storage"
    return
  fi

  local staged=$remote_stage/$label.partial
  "${adb_cmd[@]}" push "$source" "$staged" >/dev/null
  "${adb_cmd[@]}" shell chmod 644 "$staged"
  local staged_hash staged_bytes
  staged_hash=$("${adb_cmd[@]}" shell toybox sha256sum "$staged" | tr -d '\r' | awk 'NR == 1 {print $1}')
  staged_bytes=$("${adb_cmd[@]}" shell toybox stat -c %s "$staged" | tr -d '\r' | awk 'NR == 1 {print $1}')
  if [[ "$staged_hash" != "$expected_sha256" || "$staged_bytes" != "$expected_bytes" ]]; then
    echo "adb staging verification failed for $label" >&2
    exit 5
  fi

  local parent=${destination%/*}
  local partial=$destination.partial
  "${adb_cmd[@]}" shell run-as "$package" mkdir -p "$parent"
  "${adb_cmd[@]}" shell run-as "$package" cp "$staged" "$partial"
  "${adb_cmd[@]}" shell run-as "$package" chmod 600 "$partial"
  installed_hash=$(private_hash "$partial")
  installed_bytes=$(private_size "$partial")
  if [[ "$installed_hash" != "$expected_sha256" || "$installed_bytes" != "$expected_bytes" ]]; then
    "${adb_cmd[@]}" shell run-as "$package" rm -f "$partial"
    echo "app-private verification failed for $label" >&2
    exit 5
  fi
  "${adb_cmd[@]}" shell run-as "$package" mv "$partial" "$destination"
  "${adb_cmd[@]}" shell rm -f "$staged"
  echo "$label installed and verified"
}

if [[ $skip_assets -eq 0 ]]; then
  install_private_file \
    "$model_path" \
    "files/text/models/$model_filename" \
    "$model_bytes" "$model_sha256" qwen2.5-model

  IFS=$'\t' read -r sensevoice_bytes sensevoice_sha256 <<< "$(bundle_file_fields model.bin)"
  install_private_file \
    "$sensevoice_model/model.bin" files/asr/models/sensevoice/model.bin \
    "$sensevoice_bytes" "$sensevoice_sha256" sensevoice-model

  IFS=$'\t' read -r tokens_bytes tokens_sha256 <<< "$(bundle_file_fields tokens.txt)"
  install_private_file \
    "$sensevoice_model/tokens.txt" files/asr/models/sensevoice/tokens.txt \
    "$tokens_bytes" "$tokens_sha256" sensevoice-tokens

  for filename in encoder.int8.onnx decoder.onnx joiner.int8.onnx tokens.txt; do
    russian_path=$russian_model/$filename
    russian_fields=$(python3 - "$russian_path" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(f"{path.stat().st_size}\t{digest.hexdigest()}")
PY
)
    IFS=$'\t' read -r russian_bytes russian_sha256 <<< "$russian_fields"
    install_private_file \
      "$russian_path" "files/asr/models/russian/$filename" \
      "$russian_bytes" "$russian_sha256" "russian-$filename"
  done
fi

if [[ $start_after_install -eq 0 ]]; then
  echo "App-owned runtime installed and staged on $serial; launch was skipped"
  exit 0
fi

# Nubia's OEM freezer can suspend a background package even while Android
# still reports its foreground service as active. The visible main app binds
# the runtime companion, and these one-time provisioning exemptions keep both
# packages out of idle/freezer policy when the user returns to the demo.
for power_package in "$package" "$main_package"; do
  if ! "${adb_cmd[@]}" shell dumpsys deviceidle whitelist "+$power_package" >/dev/null; then
    echo "Warning: could not exempt $power_package from device idle; set battery use to Unrestricted manually" >&2
  fi
  if ! "${adb_cmd[@]}" shell am set-standby-bucket "$power_package" active >/dev/null; then
    echo "Warning: could not set $power_package to the active standby bucket" >&2
  fi
done

"${adb_cmd[@]}" shell am start -W -n "$activity" >/dev/null
text_forward=$("${adb_cmd[@]}" forward tcp:0 tcp:6020 | tr -d '\r')
asr_forward=$("${adb_cmd[@]}" forward tcp:0 tcp:6010 | tr -d '\r')

python3 - \
  "$text_forward" "$asr_forward" "$model_alias" \
  "$sensevoice_en_fixture" "$sensevoice_zh_fixture" \
  "$sensevoice_ja_fixture" "$sensevoice_ko_fixture" "$russian_fixture" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

text_port = int(sys.argv[1])
asr_port = int(sys.argv[2])
expected_model = sys.argv[3]
fixtures = {
    "en": pathlib.Path(sys.argv[4]),
    "zh": pathlib.Path(sys.argv[5]),
    "ja": pathlib.Path(sys.argv[6]),
    "ko": pathlib.Path(sys.argv[7]),
    "ru": pathlib.Path(sys.argv[8]),
}
expected = {
    "en": {
        "backend": "sensevoice_qnn",
        "provider": "qnn",
        "text": "The tribal chieftain called for the boy and presented him with 50 pieces of gold.",
    },
    "zh": {
        "backend": "sensevoice_qnn",
        "provider": "qnn",
        "text": "开饭时间早上9点至下午5点。",
    },
    "ja": {
        "backend": "sensevoice_qnn",
        "provider": "qnn",
        "language_tag": "<|ja|>",
        "text": "フランス の 首都 は どこ で すか？",
    },
    "ko": {
        "backend": "sensevoice_qnn",
        "provider": "qnn",
        "language_tag": "<|ko|>",
        "text": "프랑스의 수도는 어디인가요?",
    },
    "ru": {
        "backend": "zipformer_ru_cpu",
        "provider": "cpu",
        "text": "расскажи коротко о москве",
    },
}


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def run_asr(language):
    fixture = fixtures[language]
    if not fixture.is_file():
        raise RuntimeError(f"missing ASR acceptance fixture: {fixture}")
    request = urllib.request.Request(
        f"http://127.0.0.1:{asr_port}/asr?language={language}",
        data=fixture.read_bytes(),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    spec = expected[language]
    candidates = payload.get("candidates", [])
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("backend") == spec["backend"]
        ),
        None,
    )
    result = candidate.get("result", {}) if isinstance(candidate, dict) else {}
    transcript = result.get("text")
    if (
        payload.get("ok") is not True
        or candidate is None
        or candidate.get("ok") is not True
        or candidate.get("provider") != spec["provider"]
        or transcript != spec["text"]
        or (
            "language_tag" in spec
            and result.get("lang") != spec["language_tag"]
        )
    ):
        raise RuntimeError(
            f"{language} ASR execution/parity failed: {payload!r}"
        )
    return {
        "backend": candidate["backend"],
        "provider": candidate["provider"],
        "transcript": transcript,
        "language_tag": result.get("lang"),
    }


deadline = time.monotonic() + 150
last_error = None
while time.monotonic() < deadline:
    try:
        text_health = get_json(f"http://127.0.0.1:{text_port}/health", timeout=3)
        models = get_json(f"http://127.0.0.1:{text_port}/v1/models", timeout=3)
        ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
        asr_health = get_json(f"http://127.0.0.1:{asr_port}/health", timeout=3)
        backends = asr_health.get("backends", {})
        if (
            expected_model in ids
            and asr_health.get("app_owned") is True
            and asr_health.get("artifacts_complete") is True
            and backends.get("sensevoice_qnn", {}).get("artifacts_ready") is True
            and backends.get("zipformer_ru_cpu", {}).get("artifacts_ready") is True
        ):
            break
        last_error = RuntimeError(
            f"identity/artifact mismatch: ids={ids!r}, asr={asr_health!r}"
        )
    except Exception as error:
        last_error = error
    time.sleep(1)
else:
    raise SystemExit(f"runtime artifacts did not become ready: {last_error}")

asr_results = {
    language: run_asr(language)
    for language in ("en", "zh", "ja", "ko", "ru")
}
verified_health = get_json(f"http://127.0.0.1:{asr_port}/health", timeout=5)
verified_backends = verified_health.get("backends", {})
for backend in ("sensevoice_qnn", "zipformer_ru_cpu"):
    state = verified_backends.get(backend, {})
    if (
        state.get("ready") is not True
        or state.get("execution_verified") is not True
        or state.get("runner_owned_by_service") is not True
    ):
        raise SystemExit(f"ASR backend is not execution-verified: {backend}={state!r}")
if verified_health.get("complete") is not True:
    raise SystemExit(f"ASR execution health is incomplete: {verified_health!r}")

print(json.dumps({
    "text_health": text_health,
    "model_ids": ids,
    "expected_model": expected_model,
    "asr_results": asr_results,
    "asr_health": verified_health,
}, ensure_ascii=False, indent=2))
PY

app_uid=$("${adb_cmd[@]}" shell run-as "$package" toybox stat -c %u . | tr -d '\r')
qwen_pids=$("${adb_cmd[@]}" shell ps -A -o PID,ARGS | tr -d '\r' \
  | awk '/libqwenserver[.]so/ {print $1}')
if [[ -z "$qwen_pids" ]]; then
  echo "Could not find libqwenserver.so after successful model health" >&2
  exit 6
fi
for pid in $qwen_pids; do
  process_uid=$("${adb_cmd[@]}" shell cat "/proc/$pid/status" | tr -d '\r' \
    | awk '/^Uid:/ {print $2; exit}')
  if [[ "$process_uid" != "$app_uid" ]]; then
    echo "Qwen PID $pid belongs to UID $process_uid, expected companion UID $app_uid" >&2
    exit 6
  fi
done

echo "App-owned runtime ready on $serial"
echo "Companion UID: $app_uid"
echo "Qwen PID(s): $qwen_pids"
echo "Qwen model: $model_alias"
echo "Qwen SHA-256: $model_sha256"
if [[ $keep_forwards -eq 1 ]]; then
  echo "Text forward: http://127.0.0.1:$text_forward"
  echo "ASR forward: http://127.0.0.1:$asr_forward"
fi
