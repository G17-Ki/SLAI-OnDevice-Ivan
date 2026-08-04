#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
PYTHON="${LFM_AIHUB_PYTHON:-$LFM_DEMO_WORK_ROOT/venvs/aihub/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/aihub/device-discovery-$STAMP"
LOG="$OUTPUT_DIR/run.log"
RESULT="$OUTPUT_DIR/devices.json"

mkdir -p "$OUTPUT_DIR"
exec >"$LOG" 2>&1

"$PYTHON" \
  "$SCRIPT_DIR/list_aihub_devices.py" --all >"$RESULT"

python3 - "$RESULT" <<'PY'
import json
import re
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
pattern = re.compile(r"AR1|5100|wear|glasses|QCS", re.I)
matches = [row for row in rows if pattern.search(" ".join([row.get("name", ""), row.get("os", ""), *row.get("attributes", [])]))]
print(json.dumps({"status": "passed", "device_count": len(rows), "relevant": matches}, indent=2))
PY
