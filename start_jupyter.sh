#!/usr/bin/env bash
# Start JupyterLab with Isaac Lab's Python environment, serving this folder.
# Open the http://localhost:8888/... link it prints in your browser (on WSL: your Windows browser). Stop with Ctrl+C.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
if [ ! -d "$ISAACLAB_DIR/.venv" ]; then
  echo "Isaac Lab is not installed at $ISAACLAB_DIR. Run ./install.sh first (or set ISAACLAB_DIR)." >&2
  exit 1
fi
cd "$REPO"
exec uv run --project "$ISAACLAB_DIR" --with-editable "$REPO/parkour" --with jupyterlab --with wurlitzer \
  jupyter lab --no-browser --notebook-dir "$REPO"
