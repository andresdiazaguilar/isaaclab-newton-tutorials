#!/usr/bin/env bash
# Run any command with Isaac Lab's Python environment, from this folder. Examples:
#   ./run.sh isaaclab list_envs
#   ./run.sh isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2
#   ./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint checkpoints/go2_flat_model_999.pt
#   ./run.sh tensorboard --logdir logs
#   ./run.sh isaaclab train --rl_library rsl_rl --task Go2-Parkour        # the task from notebook 02
# Training logs are written to ./logs (relative to where you run the command).
# The parkour task package (./parkour) is added as an editable install, so `isaaclab` finds its tasks
# and your edits to its files take effect immediately.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
if [ ! -d "$ISAACLAB_DIR/.venv" ]; then
  echo "Isaac Lab is not installed at $ISAACLAB_DIR. Run ./install.sh first (or set ISAACLAB_DIR)." >&2
  exit 1
fi
exec uv run --project "$ISAACLAB_DIR" --with-editable "$REPO/parkour" "$@"
