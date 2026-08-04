#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the five Lane C inference engines in isolated Google Colab runtimes.
# The script intentionally returns only compact JSON/JSONL receipts.  FLEURS
# audio, MUSAN audio, and checkpoint weights remain inside each Colab VM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB_BUNDLE="${LANE_C_COLAB_BUNDLE:-<USER_HOME>/Desktop/projects/handy_art/colab-cli-skill}"
RESULTS_ROOT="$SCRIPT_DIR/artifacts/results"
ENGINE_LIST=(sensevoice nemo zipformer gigaam omni)
HOST_PYTHON="${LANE_C_HOST_PYTHON:-python3}"

usage() {
  printf '%s\n' \
    "usage: $0 [--auth-probe | sensevoice|nemo|zipformer|gigaam|omni|all] [...]"
}

if [[ ! -r "$COLAB_BUNDLE/.colab_env" ]]; then
  printf 'Colab bundle is unavailable: %s\n' "$COLAB_BUNDLE" >&2
  exit 3
fi

# The bundle's env file derives several absolute paths from $PWD while it is
# sourced.  Source it from the bundle directory, then return to the caller's
# directory so this launcher works from any checkout location.
LANE_C_CALLER_DIR="$PWD"
cd "$COLAB_BUNDLE"
set -a
source .colab_env
set +a
cd "$LANE_C_CALLER_DIR"
unset LANE_C_CALLER_DIR

# A task-specific OAuth home can be supplied without overwriting the portable
# bundle's existing credentials.  Lane C uses this during an interactive
# browser handoff, then keeps all subsequent calls non-interactive.
COLAB_HOME="${LANE_C_COLAB_HOME_OVERRIDE:-$COLAB_HOME}"
COLAB_CLIENT_OAUTH_CONFIG="${LANE_C_CLIENT_OAUTH_CONFIG_OVERRIDE:-$COLAB_CLIENT_OAUTH_CONFIG}"
COLAB_AUTH="${LANE_C_COLAB_AUTH_OVERRIDE:-$COLAB_AUTH}"
COLAB_SESSION_CONFIG="${LANE_C_COLAB_SESSION_CONFIG_OVERRIDE:-$COLAB_HOME/.config/colab-cli/sessions.json}"
LANE_C_SESSION_PREFIX="${LANE_C_COLAB_SESSION_PREFIX_OVERRIDE:-lane-c}"
COLAB_PYTHON="${LANE_C_COLAB_PYTHON_OVERRIDE:-$(dirname "$COLAB_CLI")/python}"

if [[ ! -x "$COLAB_CLI" ]]; then
  printf '%s\n' 'Colab CLI is not installed in the configured bundle.' >&2
  exit 3
fi
if [[ ! -x "$COLAB_PYTHON" ]]; then
  printf '%s\n' 'The configured Colab CLI Python runtime is unavailable.' >&2
  exit 3
fi
if ! command -v "$HOST_PYTHON" >/dev/null 2>&1; then
  printf 'Host Python is unavailable: %s\n' "$HOST_PYTHON" >&2
  exit 3
fi
if [[ "$COLAB_AUTH" != "oauth2" && "$COLAB_AUTH" != "adc" ]]; then
  printf 'Unsupported Colab auth provider: %s\n' "$COLAB_AUTH" >&2
  exit 3
fi

CLI_VERSION_OUTPUT="$("$COLAB_CLI" version 2>/dev/null || true)"
if [[ "$CLI_VERSION_OUTPUT" != *"Version: 0.6.0"* ]]; then
  printf '%s\n' 'Lane C requires the official Colab CLI at exactly version 0.6.0.' >&2
  exit 3
fi
unset CLI_VERSION_OUTPUT

# google-colab-cli 0.6.0 imports KernelClient from the 0.15.x API.  Version
# 1.0.0 installs successfully but removed that symbol, so reject the broken
# combination before allocating a runtime.
if ! "$COLAB_PYTHON" - <<'PY'
from importlib.metadata import version

if version("jupyter-kernel-client") != "0.15.0":
    raise SystemExit(1)
from jupyter_kernel_client import KernelClient

if KernelClient is None:
    raise SystemExit(1)
PY
then
  printf '%s\n' \
    'Lane C requires jupyter-kernel-client==0.15.0 with KernelClient available.' >&2
  exit 3
fi

umask 077
mkdir -p "$(dirname "$COLAB_SESSION_CONFIG")"
if [[ ! -f "$COLAB_SESSION_CONFIG" ]]; then
  printf '{}\n' >"$COLAB_SESSION_CONFIG"
fi
chmod 600 "$COLAB_SESSION_CONFIG"

if [[ "$COLAB_AUTH" == "oauth2" ]]; then
  if [[ ! -r "$COLAB_CLIENT_OAUTH_CONFIG" ]]; then
    printf '%s\n' 'Task OAuth client configuration is not readable.' >&2
    exit 3
  fi
  TOKEN_PATH="$COLAB_HOME/.config/colab-cli/token.json"
  if [[ ! -r "$TOKEN_PATH" ]] || ! "$HOST_PYTHON" - "$TOKEN_PATH" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if not isinstance(payload, dict) or not (
    payload.get("token") or payload.get("refresh_token")
):
    raise SystemExit(1)
PY
  then
    printf '%s\n' \
      'Task OAuth token is absent or invalid. Complete the browser handoff before launching.' >&2
    exit 3
  fi
  unset TOKEN_PATH
fi

COLAB_ARGS=(
  --client-oauth-config "$COLAB_CLIENT_OAUTH_CONFIG"
  --config "$COLAB_SESSION_CONFIG"
  --auth "$COLAB_AUTH"
)

sanitize_colab_stderr() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      *"To authorize colab-cli"*|*"accounts.google.com/o/oauth"*|*"Enter the authorization code"*)
        printf '%s\n' '[colab] Interactive OAuth was suppressed; refresh the task credentials.' >&2
        ;;
      *)
        printf '%s\n' "$line" >&2
        ;;
    esac
  done
}

colab_cli() {
  HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" "$@" </dev/null \
    2> >(sanitize_colab_stderr)
}

colab_cli_captured() {
  local timeout_seconds="$1"
  local log_path="$2"
  shift 2
  HOME="$COLAB_HOME" "$HOST_PYTHON" - \
    "$timeout_seconds" "$log_path" "$COLAB_CLI" "${COLAB_ARGS[@]}" "$@" <<'PY'
import os
import subprocess
import sys

timeout_seconds = float(sys.argv[1])
log_path = sys.argv[2]
command = sys.argv[3:]
with open(log_path, "ab", buffering=0) as log:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            timeout=timeout_seconds,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
}

requested=("${@:-all}")
probe_only=0
engines=()
for value in "${requested[@]}"; do
  if [[ "$value" == "--auth-probe" ]]; then
    if [[ "${#requested[@]}" -ne 1 ]]; then
      usage >&2
      exit 2
    fi
    probe_only=1
    continue
  fi
  if [[ "$value" == "all" ]]; then
    engines=("${ENGINE_LIST[@]}")
    break
  fi
  valid=0
  for candidate in "${ENGINE_LIST[@]}"; do
    if [[ "$value" == "$candidate" ]]; then
      valid=1
      break
    fi
  done
  if [[ "$valid" -ne 1 ]]; then
    usage >&2
    exit 2
  fi
  engines+=("$value")
done

AUTH_CHECK_LOG="$(mktemp -t lane-c-colab-auth.XXXXXX)"
chmod 600 "$AUTH_CHECK_LOG"
trap 'rm -f "$AUTH_CHECK_LOG"' EXIT
if ! colab_cli_captured 60 "$AUTH_CHECK_LOG" sessions; then
  printf '%s\n' 'Colab authorization is not ready. Complete the Google sign-in handoff first.' >&2
  exit 3
fi

if [[ "$probe_only" -eq 1 ]]; then
  printf '%s\n' 'Colab CLI 0.6.0 authorization probe succeeded; no session was created.'
  exit 0
fi

mkdir -p "$RESULTS_ROOT"

for required_file in normalisers.py colab_common.py colab_engines.py colab_remote_runner.py; do
  if [[ ! -r "$SCRIPT_DIR/$required_file" ]]; then
    printf 'Required Colab payload is missing: %s\n' "$SCRIPT_DIR/$required_file" >&2
    exit 3
  fi
done
unset required_file

gpu_for_engine() {
  case "$1" in
    sensevoice) printf '%s\n' "${LANE_C_GPU_SENSEVOICE:-L4}" ;;
    nemo) printf '%s\n' "${LANE_C_GPU_NEMO:-L4}" ;;
    zipformer) printf '%s\n' "${LANE_C_GPU_ZIPFORMER:-T4}" ;;
    gigaam) printf '%s\n' "${LANE_C_GPU_GIGAAM:-L4}" ;;
    omni) printf '%s\n' "${LANE_C_GPU_OMNI:-T4}" ;;
  esac
}

timeout_for_engine() {
  case "$1" in
    sensevoice)
      printf '%s\n' "${LANE_C_TIMEOUT_SENSEVOICE_SECONDS:-${LANE_C_EXEC_TIMEOUT_SECONDS:-14400}}"
      ;;
    nemo)
      printf '%s\n' "${LANE_C_TIMEOUT_NEMO_SECONDS:-${LANE_C_EXEC_TIMEOUT_SECONDS:-18000}}"
      ;;
    zipformer)
      printf '%s\n' "${LANE_C_TIMEOUT_ZIPFORMER_SECONDS:-${LANE_C_EXEC_TIMEOUT_SECONDS:-10800}}"
      ;;
    gigaam)
      printf '%s\n' "${LANE_C_TIMEOUT_GIGAAM_SECONDS:-${LANE_C_EXEC_TIMEOUT_SECONDS:-10800}}"
      ;;
    omni)
      printf '%s\n' "${LANE_C_TIMEOUT_OMNI_SECONDS:-${LANE_C_EXEC_TIMEOUT_SECONDS:-14400}}"
      ;;
  esac
}

validate_result_archive() {
  local archive_path="$1"
  local expected_engine="$2"
  "$HOST_PYTHON" - "$SCRIPT_DIR" "$archive_path" "$expected_engine" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1]).resolve()
archive_path = Path(sys.argv[2]).resolve()
expected_engine = sys.argv[3]
sys.path.insert(0, str(script_dir))
import aggregate_results

expected_path = archive_path.parent.parent / expected_engine / "result.zip"
if archive_path != expected_path.resolve():
    print("result archive path does not match launcher engine", file=sys.stderr)
    raise SystemExit(1)
result = aggregate_results.read_engine_zip(archive_path.parent.parent, expected_engine)
if not isinstance(result, dict) or result.get("status") != "COMPLETE":
    error = str((result or {}).get("error") or "remote run did not complete")
    print(error.replace("\n", " ")[:2000], file=sys.stderr)
    raise SystemExit(1)
validation = result.get("_archive_validation")
if not isinstance(validation, dict) or validation.get("valid") is not True:
    print("result archive failed host evidence validation", file=sys.stderr)
    raise SystemExit(1)
PY
}

owned_sessions_for_engine() {
  local engine="$1"
  "$HOST_PYTHON" - "$COLAB_SESSION_CONFIG" "$LANE_C_SESSION_PREFIX-asr-$engine-" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prefix = sys.argv[2]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if isinstance(payload, dict):
    for name in sorted(payload):
        if isinstance(name, str) and name.startswith(prefix):
            print(name)
PY
}

cleanup_existing_engine_sessions() {
  local engine="$1"
  local existing cleanup_log
  while IFS= read -r existing; do
    [[ -n "$existing" ]] || continue
    cleanup_log="$(mktemp -t lane-c-colab-stop.XXXXXX)"
    chmod 600 "$cleanup_log"
    printf 'Stopping prior Lane C session before relaunch: %s\n' "$existing"
    if ! colab_cli_captured 120 "$cleanup_log" stop --session "$existing"; then
      rm -f "$cleanup_log"
      printf 'Could not stop prior Lane C session: %s\n' "$existing" >&2
      return 1
    fi
    rm -f "$cleanup_log"
  done < <(owned_sessions_for_engine "$engine")
}

run_engine() (
  local engine="$1"
  local stamp gpu session local_dir local_zip config_file session_created
  local exec_timeout remote_rc download_rc stop_rc stop_log
  local download_attempts download_interval attempt download_log partial_zip
  stamp="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
  gpu="$(gpu_for_engine "$engine")"
  exec_timeout="$(timeout_for_engine "$engine")"
  case "$gpu" in
    T4|L4|A100|H100|G4) ;;
    *)
      printf 'Unsupported Colab GPU for %s: %s\n' "$engine" "$gpu" >&2
      exit 2
      ;;
  esac
  if [[ ! "$exec_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$exec_timeout" == "0" ]]; then
    printf 'Invalid execution timeout for %s: %s\n' "$engine" "$exec_timeout" >&2
    exit 2
  fi
  session="$LANE_C_SESSION_PREFIX-asr-${engine}-${stamp}"
  local_dir="$RESULTS_ROOT/$engine"
  local_zip="$local_dir/result.zip"
  config_file="$local_dir/engine_config.json"
  session_created=0
  remote_rc=1
  download_rc=1
  stop_rc=1

  mkdir -p "$local_dir"
  if [[ -f "$local_zip" ]]; then
    mv "$local_zip" "$local_dir/result.previous.$stamp.zip"
  fi
  "$HOST_PYTHON" - "$engine" "$config_file" <<'PY'
import json
import sys
from pathlib import Path

engine, destination = sys.argv[1:]
Path(destination).write_text(
    json.dumps(
        {
            "engine": engine,
            "output_zip": f"/content/lane_c_{engine}_result.zip",
            "remote_root": f"/content/lane_c_{engine}",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

  cleanup_engine_session() {
    local rc=$?
    local cleanup_log
    trap - EXIT
    if [[ "$session_created" -eq 1 ]]; then
      session_created=0
      cleanup_log="$(mktemp -t lane-c-colab-stop.XXXXXX)"
      chmod 600 "$cleanup_log"
      if ! colab_cli_captured 120 "$cleanup_log" stop --session "$session"; then
        printf 'WARNING: automatic Colab cleanup failed for %s\n' "$session" >&2
      fi
      rm -f "$cleanup_log"
    fi
    return "$rc"
  }
  trap cleanup_engine_session EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP

  cleanup_existing_engine_sessions "$engine"

  printf 'Lane C Colab engine=%s gpu=%s timeout=%ss session=%s\n' \
    "$engine" "$gpu" "$exec_timeout" "$session"
  session_created=1
  colab_cli new --gpu "$gpu" --session "$session"
  colab_cli status --session "$session"

  colab_cli upload --session "$session" \
    "$SCRIPT_DIR/normalisers.py" /content/normalisers.py
  colab_cli upload --session "$session" \
    "$SCRIPT_DIR/colab_common.py" /content/colab_common.py
  colab_cli upload --session "$session" \
    "$SCRIPT_DIR/colab_engines.py" /content/colab_engines.py
  colab_cli upload --session "$session" \
    "$SCRIPT_DIR/colab_remote_runner.py" /content/lane_c_remote_runner.py
  colab_cli upload --session "$session" \
    "$config_file" /content/lane_c_engine_config.json

  set +e
  colab_cli exec --session "$session" \
    --file "$SCRIPT_DIR/colab_remote_runner.py" --timeout "$exec_timeout"
  remote_rc=$?
  set -e

  download_attempts="${LANE_C_DOWNLOAD_ATTEMPTS:-12}"
  download_interval="${LANE_C_DOWNLOAD_INTERVAL_SECONDS:-15}"
  if [[ ! "$download_attempts" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "$download_interval" =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'Invalid result-download retry configuration.' >&2
    exit 2
  fi
  for ((attempt = 1; attempt <= download_attempts; attempt++)); do
    download_log="$(mktemp -t lane-c-colab-download.XXXXXX)"
    chmod 600 "$download_log"
    partial_zip="$local_dir/.result.$stamp.partial.zip"
    rm -f "$partial_zip"
    if colab_cli_captured 300 "$download_log" download --session "$session" \
      "/content/lane_c_${engine}_result.zip" "$partial_zip"; then
      mv "$partial_zip" "$local_zip"
      download_rc=0
      rm -f "$download_log"
      break
    fi
    rm -f "$partial_zip" "$download_log"
    if ((attempt < download_attempts)); then
      printf 'Result archive not ready for %s (attempt %d/%d); retrying in %ss.\n' \
        "$engine" "$attempt" "$download_attempts" "$download_interval" >&2
      sleep "$download_interval"
    fi
  done

  stop_log="$(mktemp -t lane-c-colab-stop.XXXXXX)"
  chmod 600 "$stop_log"
  if colab_cli_captured 120 "$stop_log" stop --session "$session"; then
    session_created=0
    stop_rc=0
  else
    printf 'WARNING: explicit Colab stop failed for %s; EXIT cleanup will retry.\n' \
      "$session" >&2
  fi
  rm -f "$stop_log"
  trap - INT TERM HUP

  if [[ "$download_rc" -eq 0 ]]; then
    printf 'Lane C result: %s\n' "$local_zip"
    if ! validate_result_archive "$local_zip" "$engine"; then
      printf 'Lane C result receipt is not COMPLETE: %s\n' "$engine" >&2
      exit 6
    fi
  else
    printf 'Lane C result download failed after execution: %s\n' "$engine" >&2
  fi
  if [[ "$remote_rc" -ne 0 ]]; then
    exit "$remote_rc"
  fi
  if [[ "$download_rc" -ne 0 ]]; then
    exit 4
  fi
  if [[ "$stop_rc" -ne 0 ]]; then
    exit 5
  fi
  exit 0
)

overall_rc=0
for engine in "${engines[@]}"; do
  if ! run_engine "$engine"; then
    overall_rc=1
    printf 'Engine failed: %s (continuing so partial evidence is preserved)\n' "$engine" >&2
  fi
done

exit "$overall_rc"
