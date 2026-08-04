#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
lfm_require_colab_bundle
RUNNER="$SCRIPT_DIR/colab_lfm_bf16_runner.py"
LAUNCHER="$SCRIPT_DIR/colab_lfm_bf16_launcher.py"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
CONFIG="${1:-$SCRIPT_DIR/colab_lfm_bf16_config.json}"
ASR_AUDIO="$LFM_DEMO_VENDOR_ROOT/assets/asr.wav"
CHAT_AUDIO="$LFM_DEMO_VENDOR_ROOT/assets/question.wav"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Runner: %s\nConfig: %s\nASR input: %s\nChat input: %s\nColab bundle: %s\nWork root: %s\n' \
    "$RUNNER" "$CONFIG" "$ASR_AUDIO" "$CHAT_AUDIO" \
    "$COLAB_BUNDLE" "$LFM_DEMO_WORK_ROOT"
  exit 0
fi

for required in "$RUNNER" "$LAUNCHER" "$CONFIG" "$ASR_AUDIO" "$CHAT_AUDIO" "$COLAB_BUNDLE/.colab_env"; do
  if [[ ! -f "$required" ]]; then
    printf 'Missing required file: %s\n' "$required" >&2
    exit 2
  fi
done

# The bundle explicitly uses BASH_SOURCE and must be sourced from Bash.
set -a
source "$COLAB_BUNDLE/.colab_env"
set +a

colab_env_bootstrap_cli

# Preserve a token refreshed interactively in the bundle-local private home.
# Re-materializing here would overwrite it with the older base64 snapshot.
if [[ ! -f "$COLAB_HOME/.config/colab-cli/token.json" ]]; then
  printf '%s\n' 'Bundle-local Colab token is missing; run the auth-refresh session first.' >&2
  exit 3
fi
lfm_find_colab_oauth_config
COLAB_ARGS=(--client-oauth-config "$OFFICIAL_OAUTH_CONFIG" --auth "$COLAB_AUTH")

# Verify OAuth without ever echoing a possible interactive auth URL.
AUTH_CHECK_LOG="$(mktemp -t lfm-colab-auth.XXXXXX)"
chmod 600 "$AUTH_CHECK_LOG"
if ! HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" sessions >"$AUTH_CHECK_LOG" 2>&1; then
  rm -f "$AUTH_CHECK_LOG"
  printf '%s\n' 'Colab authorization is not ready. Refresh it interactively, then rerun this wrapper.' >&2
  exit 3
fi
rm -f "$AUTH_CHECK_LOG"

GPU="${REQUESTED_COLAB_GPU:-L4}"
if [[ "$GPU" != "L4" ]]; then
  printf 'This golden run is validated for L4; refusing requested GPU %s.\n' "$GPU" >&2
  exit 4
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
SESSION="${COLAB_SESSION_PREFIX}-lfm-bf16-$STAMP"
LOCAL_OUTPUT_DIR="${LFM_COLAB_ARTIFACT_DIR:-$LFM_DEMO_WORK_ROOT/lfm-feasibility/artifacts/colab-bf16-$STAMP}"
LOCAL_ZIP="$LOCAL_OUTPUT_DIR/lfm_bf16_result.zip"
SESSION_CREATED=0

stop_session() {
  local rc=$?
  if [[ "$SESSION_CREATED" -eq 1 ]]; then
    printf 'Stopping Colab session %s...\n' "$SESSION" >&2
    if HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" stop --session "$SESSION" >/dev/null 2>&1; then
      SESSION_CREATED=0
    else
      printf 'Automatic cleanup failed. Stop this session manually: %s\n' "$SESSION" >&2
    fi
  fi
  exit "$rc"
}
trap stop_session EXIT INT TERM HUP

mkdir -p "$LOCAL_OUTPUT_DIR"
printf 'Creating L4 session %s...\n' "$SESSION"
SESSION_CREATED=1
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" new --gpu "$GPU" --session "$SESSION"

HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" status --session "$SESSION"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload \
  --session "$SESSION" "$CONFIG" /content/colab_lfm_bf16_config.json
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload \
  --session "$SESSION" "$ASR_AUDIO" /content/asr.wav
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload \
  --session "$SESSION" "$CHAT_AUDIO" /content/question.wav
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload \
  --session "$SESSION" "$RUNNER" /content/colab_lfm_bf16_runner.py

set +e
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" exec \
  --session "$SESSION" --file "$LAUNCHER" \
  --timeout "${LFM_COLAB_TIMEOUT_SECONDS:-7200}"
REMOTE_RC=$?
set -e

# The remote runner packages either successful goldens or a compact failure report.
if ! HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" download \
  --session "$SESSION" /content/lfm_bf16_result.zip "$LOCAL_ZIP"; then
  printf 'The remote result archive could not be downloaded.\n' >&2
  exit 5
fi

HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" stop --session "$SESSION"
SESSION_CREATED=0
trap - EXIT INT TERM HUP

if [[ "$REMOTE_RC" -ne 0 ]]; then
  printf 'Remote inference failed; the failure archive is at %s\n' "$LOCAL_ZIP" >&2
  exit "$REMOTE_RC"
fi

printf 'BF16 golden archive: %s\n' "$LOCAL_ZIP"
