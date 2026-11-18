#!/usr/bin/env bash
# Install everything this tutorial needs. Safe to re-run.
#   1. checks: Linux, NVIDIA GPU visible, git and curl
#   2. uv (Python package manager), into ~/.local/bin
#   3. Isaac Lab, cloned at the commit this tutorial was tested with, into ./IsaacLab
#      (or $ISAACLAB_DIR; an existing ../IsaacLab is reused)
#   4. Isaac Lab's Python environment (Newton physics, RSL-RL, video tools), via `uv sync`
#   5. a quick check that the GPU and the tasks are available
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

step "Checking the system"
[ "$(uname -s)" = "Linux" ] || fail "Linux (or WSL2 on Windows) is required. See docs/windows_wsl_setup.md."
command -v git >/dev/null || fail "git is missing: sudo apt install git"
command -v curl >/dev/null || fail "curl is missing: sudo apt install curl"
NVSMI="$(command -v nvidia-smi || true)"
[ -z "$NVSMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && NVSMI=/usr/lib/wsl/lib/nvidia-smi
[ -n "$NVSMI" ] || fail "nvidia-smi not found. Install the NVIDIA driver (on WSL: the normal Windows driver, nothing inside Ubuntu)."
"$NVSMI" --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi cannot see a GPU."

step "Installing uv (if needed)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi
uv --version

step "Getting Isaac Lab into $ISAACLAB_DIR"
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  git -C "$ISAACLAB_DIR" checkout --quiet "$ISAACLAB_COMMIT"
  echo "checked out the tested commit ${ISAACLAB_COMMIT:0:7}"
else
  HEAD="$(git -C "$ISAACLAB_DIR" rev-parse HEAD)"
  if [ "$HEAD" != "$ISAACLAB_COMMIT" ]; then
    echo "NOTE: existing checkout is at ${HEAD:0:7}, this tutorial was tested with ${ISAACLAB_COMMIT:0:7}."
    echo "      Leaving it as it is. To switch: git -C \"$ISAACLAB_DIR\" checkout $ISAACLAB_COMMIT"
  else
    echo "existing checkout is at the tested commit ${ISAACLAB_COMMIT:0:7}"
  fi
fi

step "Installing Isaac Lab's Python environment (first time: several GB, ~5-15 min)"
(cd "$ISAACLAB_DIR" && uv sync --extra video)

step "Checking the installation"
uv run --project "$ISAACLAB_DIR" python -c "
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'
print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')
"
if uv run --project "$ISAACLAB_DIR" isaaclab list_envs 2>/dev/null | grep -q "Isaac-Velocity-Flat-UnitreeGo2"; then
  echo "Isaac Lab tasks are registered (e.g. Isaac-Velocity-Flat-UnitreeGo2)"
else
  fail "'isaaclab list_envs' did not list the Go2 task. Run it yourself to see the error: ./run.sh isaaclab list_envs"
fi

step "Done"
cat <<EOF
Next steps:
  ./start_jupyter.sh      then open 01_first_policy.ipynb (the link it prints)
  ./run.sh python scripts/drive_policy.py --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint checkpoints/go2_flat_model_999.pt
EOF
