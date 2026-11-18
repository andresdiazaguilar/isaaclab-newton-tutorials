#!/usr/bin/env bash
# Run any command with Isaac Lab's Python environment, from this folder. Examples:
#   ./run.sh isaaclab list_envs
#   ./run.sh isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2
#   ./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint checkpoints/go2_flat_model_999.pt
#   ./run.sh tensorboard --logdir logs
# Training logs are written to ./logs (relative to where you run the command).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
if [ ! -d "$ISAACLAB_DIR/.venv" ]; then
  echo "Isaac Lab is not installed at $ISAACLAB_DIR. Run ./install.sh first (or set ISAACLAB_DIR)." >&2
  exit 1
fi
exec uv run --project "$ISAACLAB_DIR" "$@"
