#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
lfm_require_colab_bundle
SMOKE_SCRIPT="$SCRIPT_DIR/colab_smoke.py"
STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/artifacts/colab-smoke-$STAMP"
RESULT="$OUTPUT_DIR/colab_smoke_result.json"
LOG="$OUTPUT_DIR/run.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'Smoke script: %s\nColab bundle: %s\nOutput directory: %s\n' \
    "$SMOKE_SCRIPT" "$COLAB_BUNDLE" "$OUTPUT_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec >"$LOG" 2>&1

set -a
source "$COLAB_BUNDLE/.colab_env"
set +a
colab_env_bootstrap_cli

# Preserve a token refreshed interactively in the bundle-local private home.
# Re-materializing here would overwrite it with the older base64 snapshot.
if [[ ! -f "$COLAB_HOME/.config/colab-cli/token.json" ]]; then
  printf '%s\n' 'Bundle-local Colab token is missing; run the auth-refresh session first.'
  exit 3
fi
lfm_find_colab_oauth_config
COLAB_ARGS=(--client-oauth-config "$OFFICIAL_OAUTH_CONFIG" --auth "$COLAB_AUTH")

AUTH_LOG="$(mktemp -t lfm-colab-auth.XXXXXX)"
chmod 600 "$AUTH_LOG"
if ! HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" sessions >"$AUTH_LOG" 2>&1; then
  rm -f "$AUTH_LOG"
  printf '%s\n' 'Colab authorization check failed.'
  exit 3
fi
rm -f "$AUTH_LOG"

GPU="${REQUESTED_COLAB_GPU:-L4}"
if [[ "$GPU" != "L4" ]]; then
  printf 'Expected L4, got %s\n' "$GPU"
  exit 4
fi

SESSION="${COLAB_SESSION_PREFIX}-lfm-smoke-$STAMP"
SESSION_CREATED=0

stop_session() {
  local rc=$?
  if [[ "$SESSION_CREATED" -eq 1 ]]; then
    HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" stop --session "$SESSION" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap stop_session EXIT INT TERM HUP

SESSION_CREATED=1
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" new --gpu "$GPU" --session "$SESSION"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" status --session "$SESSION"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" exec --session "$SESSION" --file "$SMOKE_SCRIPT"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" download \
  --session "$SESSION" /content/colab_smoke_result.json "$RESULT"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" stop --session "$SESSION"
SESSION_CREATED=0
trap - EXIT INT TERM HUP

python3 - "$RESULT" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if not result.get("cuda_available"):
    raise SystemExit("CUDA is not available in the Colab smoke result")
print(json.dumps({"status": "passed", "cuda_device": result.get("cuda_device")}, indent=2))
PY
