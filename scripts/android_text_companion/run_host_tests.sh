#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

python3 - \
  "$script_dir/prepare_model.py" \
  "$script_dir/prepare_runtime.py" \
  "$script_dir/protocol.py" \
  "$script_dir/probe.py" \
  "$script_dir/test_protocol.py" <<'PY'
import pathlib
import sys

for argument in sys.argv[1:]:
    path = pathlib.Path(argument)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY
bash -n "$script_dir/deploy.sh"
sh -n "$script_dir/start.sh"
python3 "$script_dir/test_protocol.py"
python3 "$script_dir/prepare_model.py" --help >/dev/null
python3 "$script_dir/prepare_runtime.py" --help >/dev/null
python3 "$script_dir/probe.py" --help >/dev/null

echo "Android text companion host gates passed"
